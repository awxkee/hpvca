/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::{
    cabac::{
        CabacEncoder, CabacEstimator, CabacWriter, ContextSet, IntraModeContexts,
        advance_residual_contexts, encode_cbf_chroma, encode_cbf_luma, encode_residual,
        estimate_residual_bits,
    },
    dct,
    error::EncodeError,
    intra,
    yuv::Yuv,
};

#[derive(Clone, Debug)]
pub(crate) struct Nalu {
    pub(crate) _nal_type: u8,
    pub(crate) data: Vec<u8>,
}

pub(crate) struct NaluStream {
    pub(crate) nalus: Vec<Nalu>,
}

impl NaluStream {
    /// Length-prefixed format for the HEIF mdat image item, containing ONLY the
    /// VCL slice NALUs (not VPS/SPS/PPS). Parameter sets live exclusively in the
    /// hvcC configuration box; libheif and Apple put only the coded slice in mdat.
    /// Including parameter sets here is what some strict decoders (VideoToolbox)
    /// reject. Emulation prevention is applied just like to_length_prefixed().
    pub(crate) fn to_length_prefixed_slices(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for nalu in &self.nalus {
            let nal_type = (nalu.data[0] >> 1) & 0x3f;
            // Skip VPS(32), SPS(33), PPS(34) — they belong only in hvcC.
            if matches!(nal_type, 32..=34) {
                continue;
            }
            let mut escaped: Vec<u8> = Vec::with_capacity(nalu.data.len() + 8);
            let mut prev = [0xffu8; 2];
            for &b in &nalu.data {
                if prev[0] == 0 && prev[1] == 0 && b <= 3 {
                    escaped.push(0x03);
                    prev = [prev[1], 0x03];
                }
                escaped.push(b);
                prev = [prev[1], b];
            }
            out.extend_from_slice(&(escaped.len() as u32).to_be_bytes());
            out.extend_from_slice(&escaped);
        }
        out
    }
}

pub(crate) struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32,
    cur_byte: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            bit_pos: 0,
            cur_byte: 0,
        }
    }

    pub(crate) fn write_bits(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 1) as u8;
            self.cur_byte = (self.cur_byte << 1) | bit;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.buf.push(self.cur_byte);
                self.cur_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    pub(crate) fn write_bit(&mut self, v: bool) {
        self.write_bits(v as u32, 1);
    }

    /// Unsigned Exp-Golomb.
    pub(crate) fn write_ue(&mut self, mut v: u32) {
        v += 1;
        let bits = 32 - v.leading_zeros();
        self.write_bits(0, bits - 1);
        self.write_bits(v, bits);
    }

    /// Signed Exp-Golomb.
    pub(crate) fn write_se(&mut self, v: i32) {
        let u = if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        };
        self.write_ue(u);
    }

    pub(crate) fn rbsp_trailing_bits(&mut self) {
        self.write_bit(true);
        while self.bit_pos != 0 {
            self.write_bit(false);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buf.push(self.cur_byte << (8 - self.bit_pos));
        }
        self.buf
    }
}

fn nalu_header(bw: &mut BitWriter, nal_type: u8) {
    bw.write_bit(false); // forbidden_zero_bit
    bw.write_bits(nal_type as u32, 6); // nal_unit_type
    bw.write_bits(0, 6); // nuh_layer_id = 0
    bw.write_bits(1, 3); // nuh_temporal_id_plus1 = 1
}

/// Write the 88-bit decode_profile_tier_level() block (HEVC spec 7.3.3),
/// then general_level_idc (8 bits).
pub(crate) fn level_idc_for(w: u32, h: u32) -> u8 {
    let ps = (w as u64) * (h as u64);
    // (MaxLumaPs, level_idc)
    static TABLE: &[(u64, u8)] = &[
        (36864, 30),
        (122880, 60),
        (245760, 63),
        (552960, 90),
        (983040, 93),
        (2228224, 120),
        (8912896, 150),
        (35651584, 180),
    ];
    for &(maxps, lvl) in TABLE {
        if ps <= maxps {
            return lvl;
        }
    }
    186 // Level 6.2 — effectively unlimited for still images
}

#[inline]
fn uses_rext_profile(
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
    lossless: bool,
) -> bool {
    let is_420 = matches!(
        chroma,
        crate::fmt::ChromaFormat::Yuv420 | crate::fmt::ChromaFormat::Monochrome
    );
    lossless || !is_420 || bit_depth.bits() > 10
}

fn write_profile_tier_level(
    bw: &mut BitWriter,
    level_idc: u8,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
    lossless: bool,
) {
    // Select profile based on chroma and bit depth (matching Apple / x265 behavior):
    //   4:2:0 / mono 8-bit  → profile 3 (Main Still Picture), compat 0x70000000
    //   4:2:0 / mono 10-bit → profile 2 (Main10),            compat 0x20000000
    //   4:2:2 / 4:4:4 / 12-bit → profile 4 (RExt),          compat 0x08000000
    let is_420 = matches!(
        chroma,
        crate::fmt::ChromaFormat::Yuv420 | crate::fmt::ChromaFormat::Monochrome
    );
    let bits = bit_depth.bits();
    // Implicit residual DPCM is an HEVC Range Extensions tool. Lossless streams
    // use it for horizontal/vertical intra modes, so even 8/10-bit 4:2:0 must
    // advertise an RExt profile rather than Main/Main10/Main Still Picture.
    let is_rext = uses_rext_profile(chroma, bit_depth, lossless);

    let (profile_idc, compat): (u32, u32) = if is_rext {
        (4, 0x0800_0000) // RExt
    } else if bits <= 8 {
        (3, 0x7000_0000) // Main Still Picture (compatible w/ Main + Main10 + MSP)
    } else {
        (2, 0x2000_0000) // Main10
    };

    bw.write_bits(0, 2); // general_profile_space = 0
    bw.write_bit(false); // general_tier_flag = 0 (Main tier)
    bw.write_bits(profile_idc, 5); // general_profile_idc
    bw.write_bits(compat, 32); // general_profile_compatibility_flags

    // Source constraint flags — common to all profiles.
    // non_packed_constraint = 1 signals no frame-packing arrangement (correct for
    // all still images and matches Apple's encoder output).
    bw.write_bit(true); // general_progressive_source_flag    = 1
    bw.write_bit(false); // general_interlaced_source_flag     = 0
    bw.write_bit(true); // general_non_packed_constraint_flag = 1
    bw.write_bit(true); // general_frame_only_constraint_flag = 1

    if is_rext {
        // RExt extended constraint block (44 bits = 10 named flags + 34 zeros)
        let is_444 = matches!(chroma, crate::fmt::ChromaFormat::Yuv444);
        let is_mono = matches!(chroma, crate::fmt::ChromaFormat::Monochrome);
        // HM promotes any stream using the general RExt tool set (including
        // implicit RDPCM) to a 4:4:4 constraint profile, even when the coded
        // picture itself is 4:2:0/4:2:2. Narrower RExt profiles do not permit
        // this tool. Keep the actual bit-depth constraint, but deliberately do
        // not claim max-4:2:2/max-4:2:0/monochrome for lossless RDPCM streams.
        let constraint_444 = lossless || is_444;
        bw.write_bit(bits <= 12); // max_12bit_constraint_flag
        bw.write_bit(bits <= 10); // max_10bit_constraint_flag
        bw.write_bit(bits <= 8); // max_8bit_constraint_flag
        bw.write_bit(!constraint_444 && (!is_444 || is_mono)); // max_422chroma_constraint_flag
        bw.write_bit(!constraint_444 && (is_420 || is_mono)); // max_420chroma_constraint_flag
        bw.write_bit(!constraint_444 && is_mono); // max_monochrome_constraint_flag
        bw.write_bit(true); // intra_constraint_flag = 1
        bw.write_bit(false); // one_picture_only_constraint_flag = 0
        bw.write_bit(true); // lower_bit_rate_constraint_flag = 1
        bw.write_bit(bits <= 14); // max_14bit_constraint_flag
        bw.write_bits(0, 32);
        bw.write_bits(0, 2); // 34 reserved zeros
    } else {
        // Non-RExt (Main / Main10 / Main Still Picture): 44 reserved zeros
        bw.write_bits(0, 32);
        bw.write_bits(0, 12);
    }

    bw.write_bits(level_idc as u32, 8);
}

pub(crate) fn build_vps(
    width: u32,
    height: u32,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
    lossless: bool,
) -> Nalu {
    let coded_w = (width + 63) & !63;
    let coded_h = (height + 63) & !63;
    let level = level_idc_for(coded_w, coded_h);
    let mut bw = BitWriter::new();
    nalu_header(&mut bw, 32);

    bw.write_bits(0, 4); // vps_video_parameter_set_id = 0
    bw.write_bit(true); // vps_base_layer_internal_flag
    bw.write_bit(true); // vps_base_layer_available_flag
    bw.write_bits(0, 6); // vps_max_layers_minus1 = 0  (1 layer)
    bw.write_bits(0, 3); // vps_max_sub_layers_minus1 = 0  (1 temporal layer)
    bw.write_bit(true); // vps_temporal_id_nesting_flag
    bw.write_bits(0xFFFF, 16); // vps_reserved_0xffff_16bits

    write_profile_tier_level(&mut bw, level, chroma, bit_depth, lossless);

    // vps_sub_layer_ordering_info_present_flag = false → only [0] entry
    bw.write_bit(false);
    bw.write_ue(0); // vps_max_dec_pic_buffering_minus1[0] = 0 → DPB 1 (matches SPS)
    bw.write_ue(0); // vps_max_num_reorder_pics[0] = 0
    bw.write_ue(0); // vps_max_latency_increase_plus1[0] = 0

    bw.write_bits(0, 6); // vps_max_layer_id = 0
    // vps_num_layer_sets_minus1 = 0  (base layer set only)
    bw.write_ue(0);
    // layer_id_included_flag[i][j] loop: spec says i=0..nls_m1, j=0..max_layer_id
    // BUT ffmpeg's parser iterates i=1..num_layer_sets (skips i=0 as implicit).
    // With nls_m1=0 → num_layer_sets=1, ffmpeg loops i=1..1 → 0 iterations.
    // Writing the spec-correct flag[0][0] would be mis-parsed as the next field.
    // We match what every real encoder does: write NO flags for the base layer set.

    bw.write_bit(false); // vps_timing_info_present_flag
    bw.write_bit(false); // vps_extension_flag

    bw.rbsp_trailing_bits();
    Nalu {
        _nal_type: 32,
        data: bw.finish(),
    }
}

fn write_sps_range_extension(bw: &mut BitWriter, lossless: bool) {
    bw.write_bit(false); // transform_skip_rotation_enabled_flag
    bw.write_bit(false); // transform_skip_context_enabled_flag
    // In transquant-bypass intra TUs this is inferred for final horizontal
    // and vertical prediction modes; no CU/TU syntax element is required.
    bw.write_bit(lossless); // implicit_rdpcm_enabled_flag
    bw.write_bit(false); // explicit_rdpcm_enabled_flag
    bw.write_bit(false); // extended_precision_processing_flag
    bw.write_bit(false); // intra_smoothing_disabled_flag
    bw.write_bit(false); // high_precision_offsets_enabled_flag
    bw.write_bit(false); // persistent_rice_adaptation_enabled_flag
    bw.write_bit(false); // cabac_bypass_alignment_enabled_flag
}

pub(crate) fn build_sps(
    width: u32,
    height: u32,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
    lossless: bool,
    color: Option<&crate::color::Cicp>,
) -> Nalu {
    let mut bw = BitWriter::new();
    nalu_header(&mut bw, 33);

    bw.write_bits(0, 4); // sps_video_parameter_set_id = 0
    bw.write_bits(0, 3); // sps_max_sub_layers_minus1 = 0
    bw.write_bit(true); // sps_temporal_id_nesting_flag

    let sps_level = level_idc_for((width + 63) & !63, (height + 63) & !63);
    write_profile_tier_level(&mut bw, sps_level, chroma, bit_depth, lossless);

    bw.write_ue(0); // sps_seq_parameter_set_id = 0

    bw.write_ue(chroma.idc()); // chroma_format_idc (1=4:2:0, 2=4:2:2, 3=4:4:4)
    // separate_color_plane_flag is present only when chroma_format_idc == 3. We use
    // packed 4:4:4 (the three components share one coding tree), so the flag is 0.
    if chroma.idc() == 3 {
        bw.write_bit(false); // separate_color_plane_flag = 0
    }

    // Picture dimensions = multiple of the CTB size (64). This declares full CTBs
    // with no partial boundary CTBs. Empirically Apple's hardware decoder accepts a
    // LARGER range of sizes with full-CTB declaration than with multiple-of-8 +
    // partial CTBs, so we round to 64 and let the conformance window crop.
    let coded_w = (width + 63) & !63;
    let coded_h = (height + 63) & !63;
    bw.write_ue(coded_w);
    bw.write_ue(coded_h);

    // Conformance window crops the 64-multiple coded size to the visible size.
    // Offsets are in chroma units: SubWidthC (=2 for both 4:2:0/4:2:2) horizontally,
    // SubHeightC (=2 for 4:2:0, =1 for 4:2:2) vertically.
    let sub_w = chroma.sub_w() as u32;
    let sub_h = chroma.sub_h() as u32;
    let crop_right = (coded_w - width) / sub_w;
    let crop_bottom = (coded_h - height) / sub_h;
    let need_window = crop_right > 0 || crop_bottom > 0;
    bw.write_bit(need_window);
    if need_window {
        bw.write_ue(0); // conf_win_left_offset
        bw.write_ue(crop_right); // conf_win_right_offset
        bw.write_ue(0); // conf_win_top_offset
        bw.write_ue(crop_bottom); // conf_win_bottom_offset
    }

    bw.write_ue(bit_depth.minus8() as u32); // bit_depth_luma_minus8
    bw.write_ue(bit_depth.minus8() as u32); // bit_depth_chroma_minus8

    bw.write_ue(4); // log2_max_pic_order_cnt_lsb_minus4 = 4 → max POC = 256

    // sps_sub_layer_ordering_info_present_flag = false
    bw.write_bit(false);
    bw.write_ue(0); // sps_max_dec_pic_buffering_minus1[0] = 0 → DPB 1 (intra-only
    // still image; Apple's encoder also uses 0. VideoToolbox may reject
    // tiles with dpb > 0 in grid mode due to resource constraints).
    bw.write_ue(0); // sps_max_num_reorder_pics[0]
    bw.write_ue(0); // sps_max_latency_increase_plus1[0]

    // Coding-tree unit (CTU) size hierarchy.
    // Apple VideoToolbox's hardware HEVC decoder requires CTB size 64 (the size
    // Apple's own encoder uses). 16 and 32 only decode via software fallback.
    // log2_min_luma_coding_block_size_minus3 = 0  → min CB = 8×8
    bw.write_ue(0);
    // log2_diff_max_min_luma_coding_block_size = 3 → max CB = CTB = 64×64
    bw.write_ue(3);
    // log2_min_luma_transform_block_size_minus2 = 0 → min TB = 4×4
    bw.write_ue(0);
    // log2_diff_max_min_luma_transform_block_size = 3 → max TB = 32×32 (matches x265).
    bw.write_ue(3);
    // HEVC §7.3.2.2.1 orders the transform-hierarchy depths inter-before-intra.
    // max_transform_hierarchy_depth_inter = 0 (matches x265).
    bw.write_ue(0);
    // max_transform_hierarchy_depth_intra = 3 permits recursive 32→16→8→4
    // transform trees. Smaller CUs naturally stop at the 4×4 minimum TB size.
    bw.write_ue(3);

    bw.write_bit(false); // scaling_list_enabled_flag
    bw.write_bit(false); // amp_enabled_flag
    bw.write_bit(true); // sample_adaptive_offset_enabled_flag = 1 (matches x265 &
    // Kvazaar; Apple's decoder expects per-CTB SAO syntax in
    // the slice. We signal SAO "off" for every CTB, so the
    // reconstruction is identical, but the syntax is present.)
    bw.write_bit(false); // pcm_enabled_flag

    bw.write_ue(0); // num_short_term_ref_pic_sets = 0
    bw.write_bit(false); // long_term_ref_pics_present_flag
    bw.write_bit(true); // sps_temporal_mvp_enabled_flag = 1 (matches x265; no effect
    // on I-slice parsing but kept identical to x265's SPS)
    // Keep strong intra smoothing disabled: the encoder's 32×32 predictor uses
    // the normative regular [1 2 1] reference filter, so the decoder must do
    // the same. Enabling this flag without the strong-smoothing eligibility test
    // would make encoder and decoder predictions diverge.
    bw.write_bit(false); // strong_intra_smoothing_enabled_flag

    // VUI parameters: color info so decoders display correctly
    bw.write_bit(true); // vui_parameters_present_flag
    write_vui(&mut bw, color);

    // sps_extension: RExt profiles require sps_range_extension to be present
    // even when all flags within it are 0 (x265 always writes it for profile_idc=4).
    // Apple's decoder rejects 12-bit streams whose SPS lacks the range extension.
    let need_range_ext = uses_rext_profile(chroma, bit_depth, lossless);
    bw.write_bit(need_range_ext); // sps_extension_present_flag
    if need_range_ext {
        bw.write_bit(true); // sps_range_extension_flag = 1
        bw.write_bit(false); // sps_multilayer_extension_flag = 0
        bw.write_bit(false); // sps_3d_extension_flag = 0
        bw.write_bit(false); // sps_scc_extension_flag = 0
        bw.write_bits(0, 4); // sps_extension_4bits = 0
        write_sps_range_extension(&mut bw, lossless);
    }

    bw.rbsp_trailing_bits();
    Nalu {
        _nal_type: 33,
        data: bw.finish(),
    }
}

/// Write minimal VUI (Annex E §E.2.1). When a [`ColorEncoding`] is supplied its
/// primaries / transfer / matrix_coefficients and full-range flag are signalled
/// so the in-stream VUI matches the `colr`/nclx box (a fixed BT.709 matrix here
/// would silently contradict a non-709 `colr` such as YCgCo, making decoders
/// apply the wrong inverse matrix).
///
/// When `color` is `None` the colorimetry is left **unspecified**:
/// `color_description_present_flag = 0`. The `video_signal_type` is still
/// signalled with `video_full_range_flag` so the sample range is unambiguous —
/// the encoder always converts in full range, so that flag defaults to set.
fn write_vui(bw: &mut BitWriter, color: Option<&crate::color::Cicp>) {
    bw.write_bit(false); // aspect_ratio_info_present_flag
    bw.write_bit(false); // overscan_info_present_flag

    // video_signal_type_present_flag = true
    bw.write_bit(true);
    bw.write_bits(5, 3); // video_format = 5 (unspecified)
    bw.write_bit(color.map(|c| c.full_range).unwrap_or(true)); // video_full_range_flag
    match color {
        Some(c) => {
            bw.write_bit(true); // color_description_present_flag
            bw.write_bits(c.primaries as u32, 8); // color_primaries
            bw.write_bits(c.transfer as u32, 8); // transfer_characteristics
            bw.write_bits(c.matrix as u32, 8); // matrix_coefficients (e.g. 8 = YCgCo)
        }
        None => {
            bw.write_bit(false); // color_description_present_flag = 0 (unspecified)
        }
    }

    bw.write_bit(false); // chroma_loc_info_present_flag
    bw.write_bit(false); // neutral_chroma_indication_flag
    bw.write_bit(false); // field_seq_flag
    bw.write_bit(false); // frame_field_info_present_flag
    bw.write_bit(false); // default_display_window_flag
    bw.write_bit(false); // vui_timing_info_present_flag
    bw.write_bit(false); // bitstream_restriction_flag
}

#[cfg(test)]
pub(crate) fn build_pps(qp: u8, lossless: bool, wpp: bool) -> Nalu {
    build_pps_tiled(qp, lossless, wpp, 1, 1)
}

/// PPS with an optional uniform tile grid. `tile_cols`/`tile_rows` == 1 means no
/// tiles (`tiles_enabled_flag = 0`), preserving the exact non-tiled bitstream.
pub(crate) fn build_pps_tiled(
    qp: u8,
    lossless: bool,
    wpp: bool,
    tile_cols: usize,
    tile_rows: usize,
) -> Nalu {
    let mut bw = BitWriter::new();
    nalu_header(&mut bw, 34);

    bw.write_ue(0); // pps_pic_parameter_set_id = 0
    bw.write_ue(0); // pps_seq_parameter_set_id = 0
    bw.write_bit(false); // dependent_slice_segments_enabled_flag
    bw.write_bit(false); // output_flag_present_flag
    bw.write_bits(0, 3); // num_extra_slice_header_bits
    // Sign-data hiding saves one bypass-coded sign in each eligible 4×4
    // coefficient group. It is disabled for transquant-bypass/lossless CUs.
    bw.write_bit(!lossless); // sign_data_hiding_enabled_flag
    bw.write_bit(false); // cabac_init_present_flag
    bw.write_ue(0); // num_ref_idx_l0_default_active_minus1
    bw.write_ue(0); // num_ref_idx_l1_default_active_minus1
    bw.write_se(qp as i32 - 26); // init_qp_minus26: carry the full slice QP here
    bw.write_bit(false); // constrained_intra_pred_flag
    bw.write_bit(false); // transform_skip_enabled_flag

    // Activity AQ uses one quantization group per 64x64 CTU.  A CTU-level QG
    // keeps the syntax and QP predictor cheap while still allowing the encoder
    // to move bits between flat and structurally busy regions.
    let aq = activity_aq_enabled(qp, lossless);
    bw.write_bit(aq); // cu_qp_delta_enabled_flag
    if aq {
        bw.write_ue(0); // diff_cu_qp_delta_depth: QG size = CTB size (64x64)
    }

    // pps_cb_qp_offset and pps_cr_qp_offset: ALWAYS present (HEVC spec §7.3.2.3)
    bw.write_se(0); // pps_cb_qp_offset = 0
    bw.write_se(0); // pps_cr_qp_offset = 0

    bw.write_bit(false); // pps_slice_chroma_qp_offsets_present_flag
    bw.write_bit(false); // weighted_pred_flag
    bw.write_bit(false); // weighted_bipred_flag
    // transquant_bypass_enabled_flag: when set, CUs may carry
    // cu_transquant_bypass_flag to skip transform+quantization (lossless coding).
    bw.write_bit(lossless); // transquant_bypass_enabled_flag
    let tiled = tile_cols > 1 || tile_rows > 1;
    bw.write_bit(tiled); // tiles_enabled_flag
    bw.write_bit(wpp); // entropy_coding_sync_enabled_flag (WPP)
    if tiled {
        // Uniform tile grid: the decoder derives per-column/row CTB spans itself,
        // so only the counts and the uniform flag are signalled (HEVC §7.3.2.3).
        bw.write_ue(tile_cols as u32 - 1); // num_tile_columns_minus1
        bw.write_ue(tile_rows as u32 - 1); // num_tile_rows_minus1
        bw.write_bit(true); // uniform_spacing_flag
        // Prediction and in-loop filtering never cross a tile edge (each tile is
        // coded as an independent sub-image), matching HEIC-grid seam behavior.
        bw.write_bit(false); // loop_filter_across_tiles_enabled_flag
    }
    // seq_loop_filter_across_slices_enabled_flag: ALWAYS present per HEVC spec and
    // ffmpeg decode_pps() unconditionally reads it after tiles/ecs flags.
    bw.write_bit(true); // pps_loop_filter_across_slices_enabled_flag = 1 (matches
    // x265). With a single slice it has no visible effect, but
    // x265 sets it and we keep the PPS identical.

    // Deblocking is enabled by inference with beta/tc offsets 0. Avoid emitting
    // an otherwise redundant control block: it produces the same decoded image
    // while saving its PPS bits.
    bw.write_bit(false); // deblocking_filter_control_present_flag
    bw.write_bit(false); // pps_scaling_list_data_present_flag
    bw.write_bit(false); // lists_modification_present_flag
    bw.write_ue(0); // log2_parallel_merge_level_minus2
    bw.write_bit(false); // slice_segment_header_extension_present_flag
    bw.write_bit(false); // pps_extension_present_flag

    bw.rbsp_trailing_bits();
    Nalu {
        _nal_type: 34,
        data: bw.finish(),
    }
}

/// Encode a still image as a single HEVC IDR picture.
pub(crate) fn encode_intra(
    yuv: &Yuv,
    width: u32,
    height: u32,
    quality: u8,
    lossless: bool,
    color: Option<crate::color::Cicp>,
    sao: bool,
    variance_boost: crate::VarianceBoost,
) -> Result<NaluStream, EncodeError> {
    encode_intra_opts(
        yuv,
        width,
        height,
        quality,
        lossless,
        color,
        false,
        false,
        1,
        sao,
        variance_boost,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_intra_opts(
    yuv: &Yuv,
    width: u32,
    height: u32,
    quality: u8,
    lossless: bool,
    color: Option<crate::color::Cicp>,
    wpp: bool,
    tiles: bool,
    threads: usize,
    sao: bool,
    variance_boost: crate::VarianceBoost,
) -> Result<NaluStream, EncodeError> {
    // Transquant-bypass samples are not modified by in-loop filters. Disable
    // the slice-level tool as well so no redundant SAO syntax is emitted.
    let sao = sao && !lossless;
    // WPP needs at least two CTU columns to form a wavefront; otherwise fall back
    // to an ordinary single-substream slice.
    let ctus_x = (((width + 63) & !63) / 64) as usize;
    let ctus_y = (((height + 63) & !63) / 64) as usize;
    // `wpp` enables Wavefront Parallel Processing (`entropy_coding_sync`). `tiles`
    // additionally splits the picture into a modest grid of HEVC tiles, each
    // internally WPP'd — the two HEVC parallel tools combined (spec-legal, §7.4.3.3):
    // tiles give independent wavefronts (no cross-column ramp coupling, independent
    // CABAC) while WPP supplies the bulk of the parallelism per tile with no
    // prediction seam. Tile count near sqrt(threads) keeps tile seams small. NOTE:
    // the tiles+WPP combination is decoded correctly only by conformant decoders
    // (Apple/hpvcd/HM), not ffmpeg/libheif — callers gate it via `TilesWpp`. The
    // offsets counting emulation-prevention bytes (`epb_substream_lengths`) are what
    // make the many-substream stream decodable.
    let want_tiles = tiles && wpp && threads > 1;
    let tile_target = (threads as f64).sqrt().ceil() as usize;
    let (tile_cols, tile_rows) = if want_tiles {
        choose_tile_grid(ctus_x, ctus_y, tile_target)
    } else {
        (1, 1)
    };
    // WPP needs ≥2 CTU columns (tiles are kept ≥4 CTB wide by `choose_tile_grid`).
    let wpp = wpp && ctus_x >= 2;
    let vps = build_vps(width, height, yuv.chroma, yuv.bit_depth, lossless);
    let sps = build_sps(
        width,
        height,
        yuv.chroma,
        yuv.bit_depth,
        lossless,
        color.as_ref(),
    );
    let qp_val: u8 = ((100 - quality.clamp(1, 100) as u32) * 41 / 99 + 10).min(51) as u8;
    let pps = build_pps_tiled(qp_val, lossless, wpp, tile_cols, tile_rows);
    // Context-local worker pool for this encode's tile/WPP fan-out. `threads-1`
    // background workers plus the calling thread give `threads`-way concurrency; the
    // pool (and its threads) is dropped when this function returns.
    let pool = crate::pool::ThreadPool::new(threads.saturating_sub(1));
    let (idr, _ry, _rcb, _rcr) = build_idr_slice(
        yuv,
        width,
        height,
        quality,
        lossless,
        ParallelPlan {
            wpp,
            threads,
            tile_cols,
            tile_rows,
            pool: &pool,
        },
        sao,
        variance_boost,
    )?;
    Ok(NaluStream {
        nalus: vec![vps, sps, pps, idr],
    })
}

/// A raw-pointer view over a slice that is shareable across scoped threads for
/// WPP wavefront encoding. Soundness rests on two structural invariants enforced
/// by [`encode_wpp_parallel`]: (1) each CTU row is coded by exactly one thread and
/// writes only the reconstruction/map cells of its own 64-pixel CTB-row band, so
/// writes never overlap; (2) a thread reads cells of the rows above only after the
/// `row_progress` atomics mark those CTUs complete, so reads never race a write.
/// Concurrent element accesses are therefore always disjoint. Debug builds
/// additionally assert the wavefront read condition at every CTU.
struct SyncSlice<T> {
    ptr: *mut T,
    len: usize,
}

// Copy is unconditional in `T` (a SyncSlice is just a raw pointer + length), unlike
// the derive, which would demand `T: Copy`. Copying is needed to hand the same view
// to several tile/row tasks.
impl<T> Clone for SyncSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SyncSlice<T> {}

// SAFETY: the pointer targets a slice that outlives the scope in which the
// SyncSlice is shared, and the invariants above guarantee data-race freedom.
unsafe impl<T: Send> Send for SyncSlice<T> {}
unsafe impl<T: Send> Sync for SyncSlice<T> {}

impl<T> SyncSlice<T> {
    fn new(s: &mut [T]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }
    /// SAFETY: the caller must access only elements no other thread accesses
    /// concurrently (guaranteed by the WPP wavefront ordering).
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Wavefront-parallel WPP encode: one CABAC substream per CTU row, coded on a
/// pool of `threads` workers. A worker claims the next row, waits for the row
/// above to be two CTUs ahead (reconstruction references + the WPP context sync
/// point), then codes the row into its own substream. Returns the per-row
/// substream bytes in row order.
#[allow(clippy::too_many_arguments)]
fn encode_wpp_parallel(
    yuv: &Yuv,
    rec: CtuRecState<'_>,
    strides: PlaneStrides,
    coding: SliceCoding,
    ctus_x: usize,
    ctus_y: usize,
    threads: usize,
    is_last_region: bool,
    pool: &crate::pool::ThreadPool,
    sao_params: Option<&[crate::sao::SaoParam]>,
    aq_offsets: &[i8],
    sao_enabled: bool,
) -> Vec<Vec<u8>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    let CtuRecState {
        rec_y,
        rec_cb,
        rec_cr,
        mode_map,
        cu_depth,
        edge_v,
        edge_h,
        qp_map,
        cu_stride,
        mode_stride,
    } = rec;
    let SliceCoding {
        qp,
        lambda,
        lossless,
    } = coding;

    let progress: Vec<AtomicUsize> = (0..ctus_y).map(|_| AtomicUsize::new(0)).collect();
    // Debug guard: each row band must have exactly one writer, so the SyncSlice
    // writes never overlap. Set when a worker claims a row; a double-claim trips.
    #[cfg(debug_assertions)]
    let band_claimed: Vec<std::sync::atomic::AtomicBool> = (0..ctus_y)
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();
    // Context models saved after the 2nd CTU of each row, consumed by the next.
    let sync_ctx: Vec<OnceLock<(ContextSet, IntraModeContexts)>> =
        (0..ctus_y).map(|_| OnceLock::new()).collect();
    let substreams: Vec<Mutex<Vec<u8>>> = (0..ctus_y).map(|_| Mutex::new(Vec::new())).collect();
    let next_row = AtomicUsize::new(0);

    let rec_y = SyncSlice::new(rec_y);
    let rec_cb = SyncSlice::new(rec_cb);
    let rec_cr = SyncSlice::new(rec_cr);
    let mode_map = SyncSlice::new(mode_map);
    let cu_depth = SyncSlice::new(cu_depth);
    let edge_v = SyncSlice::new(edge_v);
    let edge_h = SyncSlice::new(edge_h);
    let qp_map = SyncSlice::new(qp_map);

    let worker = || {
        // Each worker owns one reusable work area; whichever row it codes.
        let mut scratch = Box::new(CompressionContext::new());
        loop {
            let r = next_row.fetch_add(1, Ordering::Relaxed);
            if r >= ctus_y {
                break;
            }
            // Debug guard: assert this band has no other writer (disjoint writes).
            #[cfg(debug_assertions)]
            assert!(
                !band_claimed[r].swap(true, Ordering::Relaxed),
                "WPP row {r} claimed by two workers — writes would alias"
            );
            // Seed the row's context from the row above's WPP sync point.
            let (mut ctx, mut ictx) = if r == 0 {
                (
                    ContextSet::init_islice(qp),
                    IntraModeContexts::init_islice(qp),
                )
            } else {
                loop {
                    if let Some(c) = sync_ctx[r - 1].get() {
                        break c.clone();
                    }
                    std::thread::yield_now();
                }
            };
            let mut cab = CabacEncoder::new();
            // WPP resets qPY_PRED to SliceQpY at the first QG of every CTU row.
            let mut aq_predictor = qp;
            for col in 0..ctus_x {
                if r > 0 {
                    // Wavefront: the CTU above-right must be reconstructed first.
                    let need = (col + 2).min(ctus_x);
                    while progress[r - 1].load(Ordering::Acquire) < need {
                        std::thread::yield_now();
                    }
                    // Debug guard: reads of the row above only touch completed CTUs.
                    debug_assert!(
                        progress[r - 1].load(Ordering::Acquire) >= need,
                        "WPP wavefront violated: row {} not far enough ahead of row {r}",
                        r - 1
                    );
                }
                // SAFETY: this row is the sole writer of its CTB-row band, and the
                // wait above guarantees every cell it reads from rows < r is done.
                let resolved_qp = code_one_ctu(
                    Entropy {
                        enc: &mut cab,
                        ctx: &mut ctx,
                        ictx: &mut ictx,
                    },
                    yuv,
                    CtuRecState {
                        rec_y: unsafe { rec_y.get() },
                        rec_cb: unsafe { rec_cb.get() },
                        rec_cr: unsafe { rec_cr.get() },
                        mode_map: unsafe { mode_map.get() },
                        cu_depth: unsafe { cu_depth.get() },
                        edge_v: unsafe { edge_v.get() },
                        edge_h: unsafe { edge_h.get() },
                        qp_map: unsafe { qp_map.get() },
                        cu_stride,
                        mode_stride,
                    },
                    &mut scratch,
                    strides,
                    SliceCoding {
                        qp,
                        lambda,
                        lossless,
                    },
                    r,
                    col,
                    aq_predictor,
                    aq_offsets.get(r * ctus_x + col).copied().unwrap_or(0),
                    sao_params,
                    sao_enabled,
                );
                aq_predictor = resolved_qp;
                if col == 1 {
                    let _ = sync_ctx[r].set((ctx.clone(), ictx.clone()));
                }
                // end_of_slice_segment_flag: 1 only on the whole slice's final CTU
                // (last row of the last region).
                let is_last = is_last_region && r == ctus_y - 1 && col == ctus_x - 1;
                cab.encode_terminate(is_last as u8);
                progress[r].store(col + 1, Ordering::Release);
            }
            // Close the substream unless its last CTU already ended the slice.
            if !(is_last_region && r == ctus_y - 1) {
                cab.encode_terminate(1); // end_of_subset_one_bit
            }
            *substreams[r].lock().unwrap() = cab.finish();
        }
    };

    // Run the wavefront on the caller's pool: `n` worker loops pull rows in order
    // and stream along the wavefront. The calling thread helps too, so `n` is
    // capped at the pool's execution slots (and never exceeds the row count).
    let n = threads.min(pool.parallelism()).min(ctus_y).max(1);
    pool.scoped(|scope| {
        for _ in 0..n {
            scope.spawn(worker);
        }
    });

    substreams
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Code one 64×64 CTU into `cab`: the per-CTU SAO-disable flags, the forced root
/// split, and the four 32×32 quadtrees via RD search + commit. Shared by the
/// sequential and WPP encoders.
#[allow(clippy::too_many_arguments)]
fn code_one_ctu(
    ent: Entropy<'_, CabacEncoder>,
    yuv: &Yuv,
    rec: CtuRecState<'_>,
    scratch: &mut CompressionContext,
    strides: PlaneStrides,
    coding: SliceCoding,
    ctu_row: usize,
    ctu_col: usize,
    aq_predictor: u8,
    aq_offset: i8,
    sao_params: Option<&[crate::sao::SaoParam]>,
    sao_enabled: bool,
) -> u8 {
    let Entropy {
        enc: cab,
        ctx,
        ictx,
    } = ent;
    let CtuRecState {
        rec_y,
        rec_cb,
        rec_cr,
        mode_map,
        cu_depth,
        edge_v,
        edge_h,
        qp_map,
        cu_stride,
        mode_stride,
    } = rec;
    let SliceCoding {
        qp,
        lambda,
        lossless,
    } = coding;
    let lu_row0 = ctu_row * 64;
    let lu_col0 = ctu_col * 64;
    let aq_enabled = activity_aq_enabled(qp, lossless);
    let target_qp = if aq_enabled {
        (i16::from(qp) + i16::from(aq_offset)).clamp(0, 51) as u8
    } else {
        qp
    };
    let local_lambda = if aq_enabled {
        // QP offsets are bounded to ±3, so use exact 2^(delta/3) constants
        // instead of evaluating powf for every CTU in both SAO passes.
        const SCALE: [f32; 7] = [
            0.5,
            0.629_960_54,
            0.793_700_5,
            1.0,
            1.259_921_1,
            1.587_401,
            2.0,
        ];
        let delta = i16::from(target_qp) - i16::from(qp);
        lambda * SCALE[(delta + 3) as usize]
    } else {
        lambda
    };
    let aq = AqCtuState {
        enabled: aq_enabled,
        predictor: aq_predictor,
        target: target_qp,
        coded: false,
    };

    if sao_enabled {
        let sao_cols = strides.w / 64;
        let sao = sao_params
            .and_then(|params| params.get(ctu_row * sao_cols + ctu_col))
            .copied()
            .unwrap_or_default();
        crate::sao::encode_luma(
            cab,
            ctx,
            sao,
            ctu_col > 0,
            ctu_row > 0,
            yuv.bit_depth.bits(),
        );
    }

    // The 64×64 root cannot be a leaf until the encoder has a 64-CU / four-32-TU
    // transform-tree path, so signal the root split and code four 32×32 quadtrees.
    let root_ctx = (ctu_col > 0) as usize + (ctu_row > 0) as usize;
    cab.encode_bin(1, &mut ctx.split_cu_flag[root_ctx]);

    let mut tree = CuTreeState {
        yuv,
        rec_y,
        rec_cb,
        rec_cr,
        strides,
        qp: target_qp,
        lambda: local_lambda,
        mode_map,
        cu_depth,
        edge_v,
        edge_h,
        qp_map,
        cu_stride,
        mode_stride,
        lossless,
        aq,
        scratch,
    };
    for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
        let row = lu_row0 + dy * 32;
        let col = lu_col0 + dx * 32;
        let plan = rdo_cu32_plan(&mut tree, row, col, ctx, ictx);
        commit_cu32_plan(cab, ctx, ictx, &mut tree, row, col, 1, plan);
    }
    tree.aq.resolved_qp()
}

struct RegionOutput {
    substreams: Vec<Vec<u8>>,
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
    #[cfg(test)]
    unfiltered_y: Vec<u16>,
}

#[derive(Clone, Copy)]
struct CtuActivity {
    mean_log_variance: f32,
    low_contrast_log_variance: f32,
}

#[inline]
fn variance_boost_qp(low_contrast_log_variance: f32, qp: u8, config: crate::VarianceBoost) -> f32 {
    // The gentle curve treats 8-bit variance 256 as the transition out of
    // low-contrast detail. Start above QP 32: at finer quantizers this tool is
    // visually redundant and can perturb an already good bit allocation.
    const LOW_VARIANCE_LOG_LIMIT: f32 = 5.549_076; // ln(1 + 256)
    let low_contrast = ((LOW_VARIANCE_LOG_LIMIT - low_contrast_log_variance)
        / LOW_VARIANCE_LOG_LIMIT)
        .clamp(0.0, 1.0);
    let strength = ((f32::from(qp) - 32.0) / 7.0).clamp(0.0, 1.0);
    low_contrast * strength * config.strength * 0.575
}

/// Log-variance statistics of the 8x8 luma cells inside one CTU. The mean is
/// the ordinary masking signal. The weighted sixth octile detects CTUs that
/// contain a substantial amount of delicate, low-contrast detail even when a
/// few hard edges would otherwise classify the whole CTU as highly active.
fn ctu_activity(yuv: &Yuv, ctu_row: usize, ctu_col: usize, octile: u8) -> CtuActivity {
    let width = yuv.width as usize;
    let height = yuv.height as usize;
    let row0 = ctu_row * 64;
    let col0 = ctu_col * 64;
    if row0 >= height || col0 >= width {
        return CtuActivity {
            mean_log_variance: 0.0,
            low_contrast_log_variance: 0.0,
        };
    }
    let shift = yuv.bit_depth.bits().saturating_sub(8);
    let mut log_sum = 0.0f32;
    let mut log_variances = [0.0f32; 64];
    let mut variance_slots = log_variances.iter_mut();
    let row_end = (row0 + 64).min(height);
    let col_end = (col0 + 64).min(width);
    let ctu_rows = &yuv.y[row0 * width..row_end * width];
    for row_band in ctu_rows.chunks_exact(width * 8) {
        let mut sums = [0.0f32; 8];
        let mut sums_sq = [0.0f32; 8];
        let mut counts = [0.0f32; 8];
        for row in row_band.chunks_exact(width) {
            let blocks = row[col0..col_end].as_chunks::<8>().0.iter();
            for (((block, sum), sum_sq), count) in blocks
                .zip(sums.iter_mut())
                .zip(sums_sq.iter_mut())
                .zip(counts.iter_mut())
            {
                for &sample in block {
                    let sample = f32::from(sample >> shift);
                    *sum += sample;
                    *sum_sq += sample * sample;
                    *count += 1.0;
                }
            }
        }
        for ((sum, sum_sq), count) in sums
            .into_iter()
            .zip(sums_sq)
            .zip(counts)
            .take((col_end - col0).div_ceil(8))
        {
            let mean = sum / count;
            let variance = (sum_sq / count - mean * mean).max(0.0);
            let log_variance = variance.ln_1p();
            log_sum += log_variance;
            *variance_slots
                .next()
                .expect("one variance slot per 8x8 CTU block") = log_variance;
        }
    }
    let blocks = 64 - variance_slots.len();
    if blocks == 0 {
        CtuActivity {
            mean_log_variance: 0.0,
            low_contrast_log_variance: 0.0,
        }
    } else {
        let values = &mut log_variances[..blocks];
        values.sort_unstable_by(f32::total_cmp);
        let ranked = |n: usize| values[(blocks * n).div_ceil(8).saturating_sub(1)];
        // Smooth adjacent octile boundaries so a single 8x8 block cannot make
        // the CTU jump by a complete QP step.
        let center = usize::from(octile);
        let low_contrast = (ranked(center.saturating_sub(1).max(1))
            + 2.0 * ranked(center)
            + ranked((center + 1).min(8)))
            * 0.25;
        CtuActivity {
            mean_log_variance: log_sum / blocks as f32,
            low_contrast_log_variance: low_contrast,
        }
    }
}

/// Build one signed QP offset per CTU. Activity masking lets textured CTUs use
/// a slightly higher QP while spending the saved bits on flat/low-contrast CTUs,
/// where quantization errors are much more visible. The strength ramps in above
/// QP 24 and is bounded to ±3 so adjacent-QG discontinuities stay small.
fn activity_qp_offsets(
    yuv: &Yuv,
    ctus_x: usize,
    ctus_y: usize,
    qp: u8,
    lossless: bool,
    variance_boost: crate::VarianceBoost,
) -> Vec<i8> {
    if !activity_aq_enabled(qp, lossless) {
        return Vec::new();
    }
    let mut activity = Vec::with_capacity(ctus_x * ctus_y);
    for row in 0..ctus_y {
        for col in 0..ctus_x {
            activity.push(ctu_activity(yuv, row, col, variance_boost.octile));
        }
    }
    let mean = activity
        .iter()
        .map(|value| value.mean_log_variance)
        .sum::<f32>()
        / activity.len().max(1) as f32;
    let strength = ((f32::from(qp) - 24.0) / 14.0).clamp(0.25, 1.0) * 1.25;
    let mut offsets: Vec<i8> = activity
        .iter()
        .map(|value| {
            let masking = if variance_boost.boost_only {
                0.0
            } else {
                (value.mean_log_variance - mean) * strength
            };
            let boost = variance_boost_qp(value.low_contrast_log_variance, qp, variance_boost);
            (masking - boost).round().clamp(-3.0, 3.0) as i8
        })
        .collect();

    // Rounding can bias a small image by one QP. Remove that common-mode shift
    // so AQ redistributes the bit budget instead of silently changing quality.
    if !variance_boost.boost_only {
        let rounded_mean = (offsets.iter().map(|&v| i32::from(v)).sum::<i32>() as f32
            / offsets.len().max(1) as f32)
            .round() as i8;
        for offset in &mut offsets {
            *offset = (*offset - rounded_mean).clamp(-3, 3);
        }
    }
    offsets
}

#[inline]
fn fill_cu_qp(map: &mut [u8], stride: usize, row: usize, col: usize, size: usize, qp: u8) {
    if map.is_empty() {
        return;
    }
    let row0 = row / 4;
    let col0 = col / 4;
    let side = size / 4;
    for grid_row in row0..row0 + side {
        map[grid_row * stride + col0..grid_row * stride + col0 + side].fill(qp);
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_region_pass(
    yuv: &Yuv,
    width: u32,
    height: u32,
    qp: u8,
    lossless: bool,
    wpp: bool,
    threads: usize,
    is_last_region: bool,
    pool: &crate::pool::ThreadPool,
    sao_params: Option<&[crate::sao::SaoParam]>,
    aq_offsets: &[i8],
    sao_enabled: bool,
) -> RegionOutput {
    let sub_w = yuv.chroma.sub_w();
    let sub_h = yuv.chroma.sub_h();
    let w = ((width + 63) & !63) as usize;
    let h = ((height + 63) & !63) as usize;
    let cw = w / sub_w;
    let ch = h / sub_h;
    let src_yw = yuv.width as usize;
    let src_yh = yuv.height as usize;
    let src_cw = (yuv.width as usize).div_ceil(sub_w);
    let src_ch = (yuv.height as usize).div_ceil(sub_h);
    let lambda = 0.57_f32 * 2f32.powf((qp as f32 - 12.0) / 3.0);
    let cu_stride = w / 8;
    let mode_stride = w / 4;
    let mut cu_depth = vec![0u8; (w / 8) * (h / 8)];
    let mut mode_map = vec![0u8; (w / 4) * (h / 4)];
    let mut edge_v = vec![false; (w / 4) * (h / 4)];
    let mut edge_h = vec![false; (w / 4) * (h / 4)];
    let mut scratch = Box::new(CompressionContext::new());
    let mut rec_y = pad_plane(&yuv.y, src_yw, src_yh, w, h);
    let (mut rec_cb, mut rec_cr) = if yuv.chroma.is_monochrome() {
        (Vec::new(), Vec::new())
    } else {
        (
            pad_plane(&yuv.cb, src_cw, src_ch, cw, ch),
            pad_plane(&yuv.cr, src_cw, src_ch, cw, ch),
        )
    };
    let ctus_x = w / 64;
    let ctus_y = h / 64;
    // QpY on the 4x4 grid, used by the normative deblocking thresholds after
    // every CTU's delta has either been coded or inferred from its predictor.
    let mut qp_map = if aq_offsets.is_empty() {
        Vec::new()
    } else {
        vec![qp; mode_stride * (h / 4)]
    };
    let strides = PlaneStrides {
        w,
        src_yw,
        src_yh,
        cw,
        src_cw,
        src_ch,
        sub_w,
        sub_h,
    };

    let substreams = if !wpp {
        let mut cab = CabacEncoder::new();
        let mut ctx = ContextSet::init_islice(qp);
        let mut ictx = IntraModeContexts::init_islice(qp);
        let mut aq_predictor = qp;
        for ctu_row in 0..ctus_y {
            for ctu_col in 0..ctus_x {
                let resolved_qp = code_one_ctu(
                    Entropy {
                        enc: &mut cab,
                        ctx: &mut ctx,
                        ictx: &mut ictx,
                    },
                    yuv,
                    CtuRecState {
                        rec_y: &mut rec_y,
                        rec_cb: &mut rec_cb,
                        rec_cr: &mut rec_cr,
                        mode_map: &mut mode_map,
                        cu_depth: &mut cu_depth,
                        edge_v: &mut edge_v,
                        edge_h: &mut edge_h,
                        qp_map: &mut qp_map,
                        cu_stride,
                        mode_stride,
                    },
                    &mut scratch,
                    strides,
                    SliceCoding {
                        qp,
                        lambda,
                        lossless,
                    },
                    ctu_row,
                    ctu_col,
                    aq_predictor,
                    aq_offsets
                        .get(ctu_row * ctus_x + ctu_col)
                        .copied()
                        .unwrap_or(0),
                    sao_params,
                    sao_enabled,
                );
                aq_predictor = resolved_qp;
                let last = is_last_region && ctu_row == ctus_y - 1 && ctu_col == ctus_x - 1;
                cab.encode_terminate(last as u8); // end_of_slice_segment_flag
            }
        }
        if !is_last_region {
            cab.encode_terminate(1); // end_of_subset_one_bit closing this tile
        }
        vec![cab.finish()]
    } else if threads > 1 && ctus_y > 1 {
        encode_wpp_parallel(
            yuv,
            CtuRecState {
                rec_y: &mut rec_y,
                rec_cb: &mut rec_cb,
                rec_cr: &mut rec_cr,
                mode_map: &mut mode_map,
                cu_depth: &mut cu_depth,
                edge_v: &mut edge_v,
                edge_h: &mut edge_h,
                qp_map: &mut qp_map,
                cu_stride,
                mode_stride,
            },
            strides,
            SliceCoding {
                qp,
                lambda,
                lossless,
            },
            ctus_x,
            ctus_y,
            threads,
            is_last_region,
            pool,
            sao_params,
            aq_offsets,
            sao_enabled,
        )
    } else {
        let mut substreams: Vec<Vec<u8>> = Vec::with_capacity(ctus_y);
        let mut prev_sync: Option<(ContextSet, IntraModeContexts)> = None;
        for ctu_row in 0..ctus_y {
            let mut cab = CabacEncoder::new();
            let (mut ctx, mut ictx) = prev_sync.take().unwrap_or_else(|| {
                (
                    ContextSet::init_islice(qp),
                    IntraModeContexts::init_islice(qp),
                )
            });
            let mut this_sync = None;
            // The sequential WPP fallback obeys the same row-start QP reset as
            // the parallel path.
            let mut aq_predictor = qp;
            for ctu_col in 0..ctus_x {
                let resolved_qp = code_one_ctu(
                    Entropy {
                        enc: &mut cab,
                        ctx: &mut ctx,
                        ictx: &mut ictx,
                    },
                    yuv,
                    CtuRecState {
                        rec_y: &mut rec_y,
                        rec_cb: &mut rec_cb,
                        rec_cr: &mut rec_cr,
                        mode_map: &mut mode_map,
                        cu_depth: &mut cu_depth,
                        edge_v: &mut edge_v,
                        edge_h: &mut edge_h,
                        qp_map: &mut qp_map,
                        cu_stride,
                        mode_stride,
                    },
                    &mut scratch,
                    strides,
                    SliceCoding {
                        qp,
                        lambda,
                        lossless,
                    },
                    ctu_row,
                    ctu_col,
                    aq_predictor,
                    aq_offsets
                        .get(ctu_row * ctus_x + ctu_col)
                        .copied()
                        .unwrap_or(0),
                    sao_params,
                    sao_enabled,
                );
                aq_predictor = resolved_qp;
                if ctu_col == 1 {
                    this_sync = Some((ctx.clone(), ictx.clone()));
                }
                let last = is_last_region && ctu_row == ctus_y - 1 && ctu_col == ctus_x - 1;
                cab.encode_terminate(last as u8); // end_of_slice_segment_flag
            }
            if ctu_row < ctus_y - 1 {
                cab.encode_terminate(1); // end_of_subset → carry WPP sync to next row
                prev_sync = this_sync;
            } else if !is_last_region {
                cab.encode_terminate(1); // end_of_subset closing this tile
            }
            substreams.push(cab.finish());
        }
        substreams
    };

    #[cfg(test)]
    let unfiltered_y = rec_y.clone();

    // In-loop filters run after the complete region has been reconstructed.
    // Using filtered pixels for intra prediction would be non-conformant. Every
    // lossless CU is transquant-bypass, whose samples are exempt from filtering.
    if !lossless {
        if qp_map.is_empty() {
            crate::deblock::deblock(
                &mut rec_y,
                w,
                h,
                &mut rec_cb,
                &mut rec_cr,
                cw,
                ch,
                qp,
                yuv.bit_depth,
                yuv.chroma,
                &edge_v,
                &edge_h,
            );
        } else {
            crate::deblock::deblock_with_qp_map(
                &mut rec_y,
                w,
                h,
                &mut rec_cb,
                &mut rec_cr,
                cw,
                ch,
                qp,
                yuv.bit_depth,
                yuv.chroma,
                &edge_v,
                &edge_h,
                &qp_map,
                mode_stride,
            );
        }
    }
    RegionOutput {
        substreams,
        y: rec_y,
        cb: rec_cb,
        cr: rec_cr,
        #[cfg(test)]
        unfiltered_y,
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_region_substreams(
    yuv: &Yuv,
    width: u32,
    height: u32,
    qp: u8,
    lossless: bool,
    wpp: bool,
    threads: usize,
    is_last_region: bool,
    pool: &crate::pool::ThreadPool,
    sao_enabled: bool,
    variance_boost: crate::VarianceBoost,
) -> RegionOutput {
    // AQ analysis is source-only and identical for the SAO analysis/commit
    // passes. Compute it once here and share the compact CTU-offset table.
    let ctus_x = (((width + 63) & !63) / 64) as usize;
    let ctus_y = (((height + 63) & !63) / 64) as usize;
    let aq_offsets = if activity_aq_enabled(qp, lossless) {
        activity_qp_offsets(yuv, ctus_x, ctus_y, qp, false, variance_boost)
    } else {
        Vec::new()
    };
    if lossless || !sao_enabled {
        return encode_region_pass(
            yuv,
            width,
            height,
            qp,
            lossless,
            wpp,
            threads,
            is_last_region,
            pool,
            None,
            &aq_offsets,
            false,
        );
    }

    // SAO syntax precedes coding_tree_unit(), while its statistics are defined
    // on the fully reconstructed, deblocked picture. A deterministic analysis
    // pass resolves that ordering without feeding SAO pixels into intra
    // prediction. The real pass then writes the selected CTU parameters.
    let analysis = encode_region_pass(
        yuv,
        width,
        height,
        qp,
        false,
        wpp,
        threads,
        is_last_region,
        pool,
        None,
        &aq_offsets,
        true,
    );
    let stride = ((width + 63) & !63) as usize;
    let coded_h = ((height + 63) & !63) as usize;
    let original = pad_plane(
        &yuv.y,
        yuv.width as usize,
        yuv.height as usize,
        stride,
        coded_h,
    );
    let lambda = 0.57_f32 * 2f32.powf((qp as f32 - 12.0) / 3.0);
    let params = crate::sao::analyze_luma(
        &original,
        &analysis.y,
        stride,
        width as usize,
        height as usize,
        yuv.bit_depth.bits(),
        lambda,
    );
    let mut output = encode_region_pass(
        yuv,
        width,
        height,
        qp,
        false,
        wpp,
        threads,
        is_last_region,
        pool,
        Some(&params),
        &aq_offsets,
        true,
    );
    crate::sao::apply_luma(
        &mut output.y,
        stride,
        width as usize,
        height as usize,
        yuv.bit_depth.bits(),
        &params,
    );
    output
}

/// Post-emulation-prevention byte length of each substream as it will appear in the
/// concatenated slice segment data. HEVC §7.4.7.1 NOTE 3 requires entry-point offsets
/// to COUNT the emulation-prevention (0x03) bytes, so the sizes must be measured after
/// escaping — not on the raw RBSP. The escape state carries across substream
/// boundaries exactly as the global escaper in `to_length_prefixed_slices` will apply
/// it; the slice header ends byte-aligned on a non-zero (trailing-bit) byte, so the
/// header→data boundary never triggers an escape and starting from `0xff,0xff` is
/// faithful. A 0x03 triggered by a substream's first byte is attributed to that
/// substream (it is emitted while consuming that byte), which is where the decoder's
/// firstByte[] accounting places it.
fn epb_substream_lengths(substreams: &[Vec<u8>]) -> Vec<usize> {
    let mut lens = Vec::with_capacity(substreams.len());
    let mut prev = [0xffu8, 0xff];
    for s in substreams {
        let mut out = 0usize;
        for &b in s {
            if prev[0] == 0 && prev[1] == 0 && b <= 3 {
                out += 1; // emulation_prevention_three_byte (0x03)
                prev = [prev[1], 0x03];
            }
            out += 1;
            prev = [prev[1], b];
        }
        lens.push(out);
    }
    lens
}

/// Assemble slice-header entry points for the concatenated substreams (all but the
/// last) and return the concatenated slice data. Offsets are the POST-escape sizes.
fn assemble_substreams(hdr: &mut BitWriter, substreams: &[Vec<u8>]) -> Vec<u8> {
    let sizes = epb_substream_lengths(substreams);
    let num_entry = sizes.len().saturating_sub(1);
    hdr.write_ue(num_entry as u32);
    if num_entry > 0 {
        let max_off = sizes[..num_entry].iter().map(|&s| s - 1).max().unwrap_or(0);
        let offset_len = if max_off == 0 {
            1
        } else {
            usize::BITS - max_off.leading_zeros()
        };
        hdr.write_ue(offset_len - 1);
        for &s in &sizes[..num_entry] {
            hdr.write_bits((s - 1) as u32, offset_len);
        }
    }
    let mut data = Vec::with_capacity(substreams.iter().map(|s| s.len()).sum());
    for s in substreams {
        data.extend_from_slice(s);
    }
    data
}

#[allow(clippy::type_complexity)]
fn build_idr_slice(
    yuv: &Yuv,
    width: u32,
    height: u32,
    quality: u8,
    lossless: bool,
    plan: ParallelPlan<'_>,
    sao: bool,
    variance_boost: crate::VarianceBoost,
) -> Result<(Nalu, Vec<u16>, Vec<u16>, Vec<u16>), EncodeError> {
    let ParallelPlan {
        wpp,
        threads,
        tile_cols,
        tile_rows,
        pool,
    } = plan;
    // Map quality (1-100) to HEVC QP (0-51): quality=100→QP~10, quality=1→QP=51
    let qp_val: u8 = ((100 - quality.clamp(1, 100) as u32) * 41 / 99 + 10).min(51) as u8;
    let _ = quality; // used above

    // ── Slice header ────────────────────────────────────────────────────────
    let mut hdr = BitWriter::new();
    nalu_header(&mut hdr, 20); // IDR_N_LP (no leading pictures — correct for a
    // single still image; x265 and Apple use this).

    hdr.write_bit(true); // first_slice_segment_in_pic_flag
    // IRAP pictures (types 16-23, incl. IDR_W_RADL=19) must write no_output_of_prior_pics_flag
    hdr.write_bit(false); // no_output_of_prior_pics_flag = 0
    hdr.write_ue(0); // slice_pic_parameter_set_id = 0
    hdr.write_ue(2); // slice_type = I (ue(v): 2)
    // slice_sao_luma_flag / slice_sao_chroma_flag — present because the SPS enables
    // SAO. slice_sao_chroma_flag is only present when ChromaArrayType != 0 (HEVC
    // §7.3.6.1), so it is omitted for monochrome.
    hdr.write_bit(sao); // slice_sao_luma_flag
    if !yuv.chroma.is_monochrome() {
        hdr.write_bit(false); // slice_sao_chroma_flag = 0 (luma SAO search only)
    }
    // QP is carried fully in the PPS init_qp_minus26, so slice_qp_delta = 0.
    hdr.write_se(0); // slice_qp_delta
    // slice_loop_filter_across_slices_enabled_flag — REQUIRED here by HEVC §7.3.6.1:
    // it is present whenever pps_loop_filter_across_slices_enabled_flag (set in our
    // PPS) is 1 and (slice_sao_luma_flag || slice_sao_chroma_flag || deblocking not
    // disabled). Deblocking remains enabled, so it is present even without SAO.
    // Omitting it leaves the slice header one bit short: a strict decoder consumes
    // the byte_alignment() '1' bit as this flag, then reads the following padding
    // '0' as alignment_bit_equal_to_one and rejects the slice (the
    // "alignment_bit_equal_to_one=0 / undecodable NALU 20" seen in recent ffmpeg).
    // Value 1 matches x265 and is a no-op for a single-slice picture.
    hdr.write_bit(true); // slice_loop_filter_across_slices_enabled_flag = 1
    // For WPP the slice header also carries per-substream entry points; those need
    // the encoded substream sizes, so `hdr` is finalized after coding (below).

    // Encode the slice data. Without tiles the whole picture is one region emitting
    // a single substream (non-WPP) or one per CTU row (WPP). With tiles, each tile is
    // an independent region coded in parallel on the pool and its substreams are
    // concatenated in tile-scan order.
    let num_tiles = tile_cols * tile_rows;
    let mut final_y = Vec::new();
    let mut final_cb = Vec::new();
    let mut final_cr = Vec::new();
    let slice_data = if num_tiles <= 1 {
        let output = encode_region_substreams(
            yuv,
            width,
            height,
            qp_val,
            lossless,
            wpp,
            threads,
            true,
            pool,
            sao,
            variance_boost,
        );
        final_y = output.y;
        final_cb = output.cb;
        final_cr = output.cr;
        let substreams = output.substreams;
        if wpp {
            assemble_substreams(&mut hdr, &substreams)
        } else {
            // No tiles and no WPP: one substream, and no entry-point block.
            substreams.into_iter().next().unwrap_or_default()
        }
    } else {
        let w = ((width + 63) & !63) as usize;
        let h = ((height + 63) & !63) as usize;
        let ctus_x = w / 64;
        let ctus_y = h / 64;
        // Uniform-spacing CTB boundaries (must match the decoder's derivation from
        // num_tile_columns_minus1/num_tile_rows_minus1 in the PPS).
        let col_bd: Vec<usize> = (0..=tile_cols).map(|i| i * ctus_x / tile_cols).collect();
        let row_bd: Vec<usize> = (0..=tile_rows).map(|j| j * ctus_y / tile_rows).collect();
        let mut bounds = Vec::with_capacity(num_tiles);
        for tj in 0..tile_rows {
            for ti in 0..tile_cols {
                let x0 = col_bd[ti] * 64;
                let y0 = row_bd[tj] * 64;
                let tw = (col_bd[ti + 1] - col_bd[ti]) * 64;
                let th = (row_bd[tj + 1] - row_bd[tj]) * 64;
                bounds.push((x0, y0, tw, th));
            }
        }
        // Tiles run concurrently AND each WPP-parallelizes its rows internally (nested
        // on the shared pool). Give every tile the full thread count and let the pool's
        // single work queue balance the combined tile×row tasks across the cores.
        let tile_threads = threads;
        let mut tile_outputs: Vec<Option<RegionOutput>> =
            std::iter::repeat_with(|| None).take(num_tiles).collect();
        {
            let slot = SyncSlice::new(&mut tile_outputs);
            pool.scoped(|s| {
                for (idx, &(x0, y0, tw, th)) in bounds.iter().enumerate() {
                    let is_last = idx == num_tiles - 1;
                    s.spawn(move || {
                        let tile_yuv = extract_tile_yuv(yuv, x0, y0, tw, th);
                        let output = encode_region_substreams(
                            &tile_yuv,
                            tw as u32,
                            th as u32,
                            qp_val,
                            lossless,
                            wpp,
                            tile_threads,
                            is_last,
                            pool,
                            sao,
                            variance_boost,
                        );
                        // SAFETY: each task writes a distinct tile-output slot.
                        unsafe {
                            slot.get()[idx] = Some(output);
                        }
                    });
                }
            });
        }
        final_y.resize(w * h, 0);
        let sub_w = yuv.chroma.sub_w();
        let sub_h = yuv.chroma.sub_h();
        let cw = w / sub_w;
        let ch = h / sub_h;
        if !yuv.chroma.is_monochrome() {
            final_cb.resize(cw * ch, 0);
            final_cr.resize(cw * ch, 0);
        }
        let mut all = Vec::new();
        for (idx, output) in tile_outputs.into_iter().enumerate() {
            let output = output.expect("tile worker did not produce output");
            let (x0, y0, tw, th) = bounds[idx];
            for r in 0..th {
                final_y[(y0 + r) * w + x0..(y0 + r) * w + x0 + tw]
                    .copy_from_slice(&output.y[r * tw..r * tw + tw]);
            }
            if !yuv.chroma.is_monochrome() {
                let cx0 = x0 / sub_w;
                let cy0 = y0 / sub_h;
                let tcw = tw / sub_w;
                let tch = th / sub_h;
                for r in 0..tch {
                    final_cb[(cy0 + r) * cw + cx0..(cy0 + r) * cw + cx0 + tcw]
                        .copy_from_slice(&output.cb[r * tcw..r * tcw + tcw]);
                    final_cr[(cy0 + r) * cw + cx0..(cy0 + r) * cw + cx0 + tcw]
                        .copy_from_slice(&output.cr[r * tcw..r * tcw + tcw]);
                }
            }
            all.extend(output.substreams);
        }
        assemble_substreams(&mut hdr, &all)
    };

    hdr.rbsp_trailing_bits(); // byte_alignment() ending the slice header
    let mut nalu_data = hdr.finish();
    nalu_data.extend_from_slice(&slice_data);

    Ok((
        Nalu {
            _nal_type: 20,
            data: nalu_data,
        },
        final_y,
        final_cb,
        final_cr,
    ))
}

/// Pad a plane to (dst_w × dst_h) by edge-replication.
fn pad_plane(src: &[u16], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u16> {
    let mut out = vec![128u16; dst_w * dst_h];
    for r in 0..dst_h {
        let sr = r.min(src_h - 1);
        let src_row = &src[sr * src_w..sr * src_w + src_w];
        let dst_row = &mut out[r * dst_w..r * dst_w + dst_w];

        dst_row[..src_w].copy_from_slice(src_row);

        let edge = src_row[src_w - 1];
        dst_row[src_w..].fill(edge);
    }

    out
}

/// Copy a `tw × th` window starting at (x0, y0) out of a plane, replicating the
/// source edge for any part of the window past the plane bounds.
fn extract_plane(
    src: &[u16],
    sw: usize,
    sh: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; tw * th];
    for r in 0..th {
        let sr = (y0 + r).min(sh - 1);
        let base = sr * sw;
        let dst = &mut out[r * tw..r * tw + tw];
        // Interior columns copy directly; the tail (past sw) replicates the edge.
        let copy = tw.min(sw.saturating_sub(x0));
        for (c, d) in dst.iter_mut().enumerate().take(copy) {
            *d = src[base + x0 + c];
        }
        if copy < tw {
            let edge = src[base + sw - 1];
            dst[copy..].fill(edge);
        }
    }
    out
}

/// Extract one HEVC tile as a standalone sub-picture whose coded size is the tile's
/// (CTB-aligned) pixel span. Encoding this sub-picture independently reproduces exact
/// tile semantics: the tile's edge CTBs see "picture edge" (unavailable) neighbors,
/// which is precisely how a decoder treats cross-tile neighbors — so no cross-tile
/// prediction ever occurs, and no availability refactor is needed.
fn extract_tile_yuv(yuv: &Yuv, x0: usize, y0: usize, tw: usize, th: usize) -> Yuv {
    let sw = yuv.width as usize;
    let sh = yuv.height as usize;
    let y = extract_plane(&yuv.y, sw, sh, x0, y0, tw, th);
    let (cb, cr) = if yuv.chroma.is_monochrome() {
        (Vec::new(), Vec::new())
    } else {
        let subw = yuv.chroma.sub_w();
        let subh = yuv.chroma.sub_h();
        let scw = sw.div_ceil(subw);
        let sch = sh.div_ceil(subh);
        let (cx, cy) = (x0 / subw, y0 / subh);
        let (ctw, cth) = (tw / subw, th / subh);
        (
            extract_plane(&yuv.cb, scw, sch, cx, cy, ctw, cth),
            extract_plane(&yuv.cr, scw, sch, cx, cy, ctw, cth),
        )
    };
    Yuv {
        y,
        cb,
        cr,
        width: tw as u32,
        height: th as u32,
        display_w: tw as u32,
        display_h: th as u32,
        chroma: yuv.chroma,
        bit_depth: yuv.bit_depth,
    }
}

/// Choose a uniform tile grid (columns, rows) targeting `par`-way tile parallelism
/// while keeping tiles reasonably large (≥ `MIN_CTB` per side) and roughly square, so
/// the seam/entry-point overhead of tiling stays small. Returns (1, 1) — no tiles —
/// when there is no parallelism to exploit or the picture is too small to split.
fn choose_tile_grid(ctus_x: usize, ctus_y: usize, par: usize) -> (usize, usize) {
    if par <= 1 {
        return (1, 1);
    }
    const MIN_CTB: usize = 4; // ≥256-px tiles keep prediction seams and overhead low
    let max_cols = (ctus_x / MIN_CTB).max(1);
    let max_rows = (ctus_y / MIN_CTB).max(1);
    let mut best = (1usize, 1usize);
    let mut best_score = i64::MAX;
    for rows in 1..=max_rows {
        for cols in 1..=max_cols {
            let n = cols * rows;
            if n > par {
                break;
            }
            let tw = ctus_x / cols;
            let th = ctus_y / rows;
            // Prefer counts near `par`, breaking ties toward square tiles.
            let score = (par as i64 - n as i64) * 1000 + (tw as i64 - th as i64).abs();
            if score < best_score {
                best_score = score;
                best = (cols, rows);
            }
        }
    }
    best
}

#[derive(Clone, Copy)]
struct IntraModeCandidate {
    mode: u8,
    cost: f32,
}

/// Insert one RMD result into a fixed-size ascending candidate list.
///
/// HM retains eight modes for an 8×8 PU and three for a 16×16 PU when the MPM
/// fast path is enabled. The list is tiny, so a branch-light insertion is both
/// cheaper and more predictable than sorting all 35 modes.
#[inline]
fn update_intra_candidate(candidates: &mut [IntraModeCandidate], mode: u8, cost: f32) {
    let Some(pos) = candidates
        .iter()
        .position(|candidate| cost < candidate.cost)
    else {
        return;
    };
    for index in (pos + 1..candidates.len()).rev() {
        candidates[index] = candidates[index - 1];
    }
    candidates[pos] = IntraModeCandidate { mode, cost };
}

#[inline]
fn estimated_luma_mode_bins(mode: u8, mpm: &[u8; 3]) -> u32 {
    match mpm.iter().position(|&candidate| candidate == mode) {
        Some(0) => 2,           // prev_intra_luma_pred_flag + mpm_idx "0"
        Some(1) | Some(2) => 3, // prev flag + two bypass bins
        None => 6,              // prev flag + rem_intra_luma_pred_mode[5]
        Some(_) => unreachable!(),
    }
}

#[inline]
fn estimate_luma_mode_bits(ictx: &mut IntraModeContexts, mode: u8, mpm: &[u8; 3]) -> f32 {
    if let Some(idx) = mpm.iter().position(|&candidate| candidate == mode) {
        ictx.prev_intra_luma_pred_flag.estimate_and_update(1) + if idx == 0 { 1.0 } else { 2.0 }
    } else {
        ictx.prev_intra_luma_pred_flag.estimate_and_update(0) + 5.0
    }
}

#[inline]
fn push_sorted_unique_candidate(
    candidates: &mut [IntraModeCandidate],
    len: &mut usize,
    candidate: IntraModeCandidate,
) {
    debug_assert!(*len < candidates.len());
    if candidates[..*len]
        .iter()
        .any(|entry| entry.mode == candidate.mode)
    {
        return;
    }
    let pos = candidates[..*len]
        .iter()
        .position(|entry| candidate.cost < entry.cost)
        .unwrap_or(*len);
    for index in (pos..*len).rev() {
        candidates[index + 1] = candidates[index];
    }
    candidates[pos] = candidate;
    *len += 1;
}

/// Bound the expensive reconstruction pass. Three 8×8 candidates and two 16×16
/// candidates recover nearly all of the full shortlist gain in practice; a
/// relative SATD gate usually reduces this to two candidates on easy blocks.
#[inline]
fn full_rdo_candidate_count(candidates: &[IntraModeCandidate], lu: usize) -> usize {
    let min_count = 2.min(candidates.len());
    let max_count = (if lu == 8 { 3 } else { 2 }).min(candidates.len());
    if min_count == max_count {
        return min_count;
    }
    let limit = candidates[0].cost * 1.20;
    let mut count = min_count;
    while count < max_count && candidates[count].cost <= limit {
        count += 1;
    }
    count
}

fn encode_luma_mode<W: CabacWriter>(
    enc: &mut W,
    ictx: &mut IntraModeContexts,
    mode: u8,
    mpm: &[u8; 3],
) {
    if let Some(idx) = mpm.iter().position(|&candidate| candidate == mode) {
        enc.encode_bin(1, &mut ictx.prev_intra_luma_pred_flag);
        // mpm_idx TR(cMax=2) bypass: 0→"0", 1→"10", 2→"11".
        match idx {
            0 => enc.encode_bypass(0),
            1 => {
                enc.encode_bypass(1);
                enc.encode_bypass(0);
            }
            _ => {
                enc.encode_bypass(1);
                enc.encode_bypass(1);
            }
        }
    } else {
        enc.encode_bin(0, &mut ictx.prev_intra_luma_pred_flag);
        // Inverse of the decoder's sorted-MPM insertion process.
        let mut rem = mode as i32;
        for &candidate in mpm {
            if candidate < mode {
                rem -= 1;
            }
        }
        for bit in (0..5).rev() {
            enc.encode_bypass(((rem >> bit) & 1) as u8);
        }
    }
}

/// Code only `prev_intra_luma_pred_flag` (the context-coded bin) for one PU.
/// HEVC PART_NxN codes all four flags before any `mpm_idx`/`rem_intra` bins, so
/// the flag and remainder are split from `encode_luma_mode`.
fn encode_luma_mode_flag<W: CabacWriter>(
    enc: &mut W,
    ictx: &mut IntraModeContexts,
    mode: u8,
    mpm: &[u8; 3],
) {
    let in_mpm = mpm.contains(&mode);
    enc.encode_bin(in_mpm as u8, &mut ictx.prev_intra_luma_pred_flag);
}

/// Code the `mpm_idx` or `rem_intra_luma_pred_mode` bypass bins for one PU, the
/// `prev_intra_luma_pred_flag` having already been coded by
/// [`encode_luma_mode_flag`].
fn encode_luma_mode_rem<W: CabacWriter>(enc: &mut W, mode: u8, mpm: &[u8; 3]) {
    if let Some(idx) = mpm.iter().position(|&candidate| candidate == mode) {
        match idx {
            0 => enc.encode_bypass(0),
            1 => {
                enc.encode_bypass(1);
                enc.encode_bypass(0);
            }
            _ => {
                enc.encode_bypass(1);
                enc.encode_bypass(1);
            }
        }
    } else {
        let mut rem = mode as i32;
        for &candidate in mpm {
            if candidate < mode {
                rem -= 1;
            }
        }
        for bit in (0..5).rev() {
            enc.encode_bypass(((rem >> bit) & 1) as u8);
        }
    }
}

#[inline]
fn block_sse(orig: &[u16], rec: &[u16], n: usize) -> f32 {
    orig[..n * n]
        .iter()
        .zip(&rec[..n * n])
        .map(|(&a, &b)| {
            let d = a as i64 - b as i64;
            d * d
        })
        .sum::<i64>() as f32
}

/// SSE between a contiguous `n×n` original block and an `n×n` region of a strided
/// reconstruction plane at (row, col).
#[inline]
fn sse_plane(orig: &[u16], plane: &[u16], row: usize, col: usize, stride: usize, n: usize) -> f32 {
    let mut acc = 0i64;
    for r in 0..n {
        let orig_row = &orig[r * n..r * n + n];
        let base = (row + r) * stride + col;
        for (&a, &b) in orig_row.iter().zip(&plane[base..base + n]) {
            let d = a as i64 - b as i64;
            acc += d * d;
        }
    }
    acc as f32
}

/// HEVC RExt implicit residual-DPCM direction for an intra prediction mode.
/// The tool is inferred—there is no CU/TU syntax element—when the SPS enables
/// `implicit_rdpcm_enabled_flag` and the TU is transform-skipped or transquant
/// bypassed. HM applies vertical RDPCM to mode 26 and horizontal RDPCM to mode
/// 10, after the 4:2:2 chroma mode remapping has already been performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImplicitRdpcm {
    Off,
    Horizontal,
    Vertical,
}

#[inline]
fn implicit_rdpcm_mode(intra_mode: u8) -> ImplicitRdpcm {
    match intra_mode {
        10 => ImplicitRdpcm::Horizontal,
        26 => ImplicitRdpcm::Vertical,
        _ => ImplicitRdpcm::Off,
    }
}

/// Convert a raster-order, unquantized lossless prediction residual into the
/// coefficient samples decoded by HEVC implicit RDPCM. This is the lossless
/// branch of HM's `applyForwardRDPCM`: the first sample on each prediction line
/// is unchanged and every following sample is differenced from its predecessor.
#[inline]
fn forward_lossless_rdpcm_into(
    residual: &[i32],
    n: usize,
    mode: ImplicitRdpcm,
    levels: &mut [i16],
) {
    debug_assert!(residual.len() >= n * n);
    debug_assert!(levels.len() >= n * n);

    match mode {
        ImplicitRdpcm::Off => {
            for (dst, &src) in levels[..n * n].iter_mut().zip(&residual[..n * n]) {
                *dst = src as i16;
            }
        }
        ImplicitRdpcm::Horizontal => {
            for (src_row, dst_row) in residual[..n * n]
                .chunks_exact(n)
                .zip(levels[..n * n].chunks_exact_mut(n))
            {
                let mut previous = 0i32;
                for (&sample, dst) in src_row.iter().zip(dst_row) {
                    *dst = (sample - previous) as i16;
                    previous = sample;
                }
            }
        }
        ImplicitRdpcm::Vertical => {
            let (src_first, src_rest) = residual[..n * n].split_at(n);
            let (dst_first, dst_rest) = levels[..n * n].split_at_mut(n);
            for (dst, &sample) in dst_first.iter_mut().zip(src_first) {
                *dst = sample as i16;
            }
            for (current, (previous, dst)) in src_rest.chunks_exact(n).zip(
                residual[..n * (n - 1)]
                    .chunks_exact(n)
                    .zip(dst_rest.chunks_exact_mut(n)),
            ) {
                for ((&sample, &above), out) in current.iter().zip(previous).zip(dst) {
                    *out = (sample - above) as i16;
                }
            }
        }
    }
}

/// Reference inverse used by tests and by future decoder-side reuse. Keeping it
/// next to the forward path makes the exact cumulative-sum semantics explicit.
#[cfg(test)]
fn inverse_lossless_rdpcm_into(
    levels: &[i16],
    n: usize,
    mode: ImplicitRdpcm,
    residual: &mut [i32],
) {
    debug_assert!(levels.len() >= n * n);
    debug_assert!(residual.len() >= n * n);

    match mode {
        ImplicitRdpcm::Off => {
            for (dst, &src) in residual[..n * n].iter_mut().zip(&levels[..n * n]) {
                *dst = src as i32;
            }
        }
        ImplicitRdpcm::Horizontal => {
            for (src_row, dst_row) in levels[..n * n]
                .chunks_exact(n)
                .zip(residual[..n * n].chunks_exact_mut(n))
            {
                let mut accumulator = 0i32;
                for (&delta, dst) in src_row.iter().zip(dst_row) {
                    accumulator += delta as i32;
                    *dst = accumulator;
                }
            }
        }
        ImplicitRdpcm::Vertical => {
            for col in 0..n {
                let mut accumulator = 0i32;
                for (src_row, dst_row) in levels[..n * n]
                    .chunks_exact(n)
                    .zip(residual[..n * n].chunks_exact_mut(n))
                {
                    accumulator += src_row[col] as i32;
                    dst_row[col] = accumulator;
                }
            }
        }
    }
}

const CHROMA_DM_SYNTAX_IDX: u8 = 4;

#[derive(Clone, Copy, Debug)]
struct ChromaModeCandidate {
    /// Actual intra prediction mode used by §8.4.4.2 prediction.
    pred_mode: u8,
    /// `intra_chroma_pred_mode`: 0..=3 are explicit, 4 is DM_CHROMA.
    syntax_idx: u8,
    /// First-pass SATD + fractional mode-rate cost.
    cost: f32,
}

/// HEVC's five chroma candidates: planar, vertical, horizontal, DC and
/// DM_CHROMA. When DM resolves to one of the four explicit modes, angular mode
/// 34 replaces that explicit entry so all five candidates remain distinct.
#[inline]
fn chroma_mode_candidates(
    luma_mode: u8,
    chroma: crate::fmt::ChromaFormat,
) -> [ChromaModeCandidate; 5] {
    // Candidate substitution is defined in the nominal 0..=34 mode space. The
    // 4:2:2 directional remap is applied afterwards to the actual prediction
    // mode, including the replacement mode 34 (which maps to 31).
    let mut explicit = [0u8, 26, 10, 1];
    for mode in &mut explicit {
        if *mode == luma_mode {
            *mode = 34;
        }
    }
    let map_mode = |mode: u8| {
        if matches!(chroma, crate::fmt::ChromaFormat::Yuv422) {
            MODE_422_MAP[mode as usize]
        } else {
            mode
        }
    };
    let dm_mode = map_mode(luma_mode);
    let explicit = explicit.map(map_mode);
    [
        ChromaModeCandidate {
            pred_mode: explicit[0],
            syntax_idx: 0,
            cost: f32::MAX,
        },
        ChromaModeCandidate {
            pred_mode: explicit[1],
            syntax_idx: 1,
            cost: f32::MAX,
        },
        ChromaModeCandidate {
            pred_mode: explicit[2],
            syntax_idx: 2,
            cost: f32::MAX,
        },
        ChromaModeCandidate {
            pred_mode: explicit[3],
            syntax_idx: 3,
            cost: f32::MAX,
        },
        ChromaModeCandidate {
            pred_mode: dm_mode,
            syntax_idx: CHROMA_DM_SYNTAX_IDX,
            cost: f32::MAX,
        },
    ]
}

#[inline]
fn estimated_chroma_mode_bins(syntax_idx: u8) -> u32 {
    if syntax_idx == CHROMA_DM_SYNTAX_IDX {
        1
    } else {
        3
    }
}

#[inline]
fn estimate_chroma_mode_bits(ictx: &mut IntraModeContexts, syntax_idx: u8) -> f32 {
    if syntax_idx == CHROMA_DM_SYNTAX_IDX {
        ictx.intra_chroma_pred_mode.estimate_and_update(0)
    } else {
        ictx.intra_chroma_pred_mode.estimate_and_update(1) + 2.0
    }
}

fn encode_chroma_mode<W: CabacWriter>(enc: &mut W, ictx: &mut IntraModeContexts, syntax_idx: u8) {
    if syntax_idx == CHROMA_DM_SYNTAX_IDX {
        enc.encode_bin(0, &mut ictx.intra_chroma_pred_mode);
    } else {
        debug_assert!(syntax_idx < CHROMA_DM_SYNTAX_IDX);
        enc.encode_bin(1, &mut ictx.intra_chroma_pred_mode);
        enc.encode_bypass((syntax_idx >> 1) & 1);
        enc.encode_bypass(syntax_idx & 1);
    }
}

#[inline]
fn update_chroma_candidate(
    candidates: &mut [ChromaModeCandidate; 5],
    mut candidate: ChromaModeCandidate,
) {
    let Some(pos) = candidates
        .iter()
        .position(|entry| candidate.cost < entry.cost)
    else {
        return;
    };
    for index in (pos + 1..candidates.len()).rev() {
        candidates[index] = candidates[index - 1];
    }
    core::mem::swap(&mut candidates[pos], &mut candidate);
}

/// Decide whether the SATD winner is clear enough to commit directly or needs
/// one exact reconstruction-RDO challenger. Chroma has only five legal modes,
/// so all modes still participate in the proxy ranking; the expensive transform,
/// inverse transform and residual-rate walk are reserved for genuinely ambiguous
/// blocks. 4:4:4 gets the widest window because independent chroma directions are
/// most valuable there, while 4:2:0 uses the tightest window.
#[inline]
fn full_rdo_chroma_count(
    candidates: &[ChromaModeCandidate; 5],
    chroma: crate::fmt::ChromaFormat,
) -> usize {
    let threshold = match chroma {
        crate::fmt::ChromaFormat::Yuv444 => 1.08,
        crate::fmt::ChromaFormat::Yuv422 => 1.05,
        crate::fmt::ChromaFormat::Yuv420 => 1.03,
        crate::fmt::ChromaFormat::Monochrome => return 1,
    };
    if candidates[1].cost <= candidates[0].cost * threshold {
        2
    } else {
        1
    }
}

/// HEVC Table 8-3: luma→chroma intra mode mapping for 4:2:2 (DM_CHROMA).
static MODE_422_MAP: [u8; 35] = [
    0, 1, 2, 2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25,
    26, 27, 27, 28, 28, 29, 29, 30, 31,
];

fn mpm_list(cand_a: u8, cand_b: u8) -> [u8; 3] {
    const PLANAR: u8 = 0;
    const DC: u8 = 1;
    const ANG26: u8 = 26;
    if cand_a == cand_b {
        if cand_a < 2 {
            [PLANAR, DC, ANG26]
        } else {
            let m1 = 2 + ((cand_a as i32 - 2 - 1 + 32) % 32) as u8;
            let m2 = 2 + ((cand_a as i32 - 2 + 1) % 32) as u8;
            [cand_a, m1, m2]
        }
    } else {
        let third = if cand_a != PLANAR && cand_b != PLANAR {
            PLANAR
        } else if cand_a != DC && cand_b != DC {
            DC
        } else {
            ANG26
        };
        [cand_a, cand_b, third]
    }
}

/// Decode-order availability for the block containing neighbor pixel (nr,nc),
/// relative to the current block at (cur_r,cur_c). CTUs raster, sub-blocks Z-scan.
fn is_block_decoded(
    nr: usize,
    nc: usize,
    cur_r: usize,
    cur_c: usize,
    ctb: usize,
    width: usize,
) -> bool {
    if nc >= width {
        return false;
    }
    let blk = 8usize;
    let ctus_x = width / ctb;
    let grid = ctb / blk; // sub-blocks per side
    let order = |r: usize, c: usize| -> i64 {
        let ci = (r / ctb) * ctus_x + (c / ctb);
        // Hierarchical Z-scan (Morton) of the sub-block within the CTB.
        let mut sr = ((r % ctb) / blk) as u64;
        let mut sc = ((c % ctb) / blk) as u64;
        let mut z: u64 = 0;
        let mut bit = 0;
        let mut g = grid;
        while g > 1 {
            z |= (sc & 1) << (2 * bit);
            z |= (sr & 1) << (2 * bit + 1);
            sr >>= 1;
            sc >>= 1;
            bit += 1;
            g >>= 1;
        }
        let cells = (grid * grid) as i64;
        ci as i64 * cells + z as i64
    };
    order(nr, nc) < order(cur_r, cur_c)
}

/// Chroma QP derivation from luma QP (HEVC §8.6.1). The mapping table from qPi to
/// QpC applies only to 4:2:0 (ChromaArrayType 1); for 4:2:2 (and 4:4:4) QpC = qPi
/// clamped to 51.
fn chroma_qp_for(qp: u8, chroma: crate::fmt::ChromaFormat) -> u8 {
    let qpi = (qp as i32).clamp(0, 57);
    match chroma {
        crate::fmt::ChromaFormat::Yuv420 => {
            static QP_C: [u8; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
            if qpi < 30 {
                qpi as u8
            } else if qpi > 43 {
                (qpi - 6) as u8
            } else {
                QP_C[(qpi - 30) as usize]
            }
        }
        // Monochrome has no chroma; value is unused. Return luma QP for definiteness.
        crate::fmt::ChromaFormat::Monochrome => qpi.min(51) as u8,
        crate::fmt::ChromaFormat::Yuv422 | crate::fmt::ChromaFormat::Yuv444 => qpi.min(51) as u8,
    }
}

/// Effective chroma rate multiplier relative to the luma lambda. HM applies the
/// inverse relation as a chroma distortion weight; multiplying lambda by
/// `2^((QpC-QpY)/3)` is algebraically identical and avoids touching distortion.
/// Only 4:2:0 has a non-linear QpC mapping in this encoder.
#[inline]
fn chroma_lambda_scale(qp_y: u8, chroma: crate::fmt::ChromaFormat) -> f32 {
    if !matches!(chroma, crate::fmt::ChromaFormat::Yuv420) {
        return 1.0;
    }
    const DELTA_SCALE: [f32; 7] = [
        1.0,
        0.793_700_5,
        0.629_960_54,
        0.5,
        0.396_850_26,
        0.314_980_27,
        0.25,
    ];
    let qp_c = chroma_qp_for(qp_y, chroma);
    let delta = qp_y.saturating_sub(qp_c).min(6) as usize;
    DELTA_SCALE[delta]
}

struct CuGeometry {
    lu_row: usize,
    lu_col: usize,
    ch_row: usize,
    ch_col: usize,
    yw_stride: usize,
    src_yh: usize,
    cw_stride: usize,
    src_cw: usize,
    src_ch: usize,
    /// Stride of the 4×4 luma prediction-mode map used for MPM derivation.
    mode_stride: usize,
}

/// The three entropy-coding state objects, threaded together so the per-CU
/// coding functions take one argument instead of three. Holds mutable borrows
/// so callers keep ownership (the RD trials clone the underlying objects and
/// build a fresh bundle per trial).
struct Entropy<'a, W: CabacWriter> {
    enc: &'a mut W,
    ctx: &'a mut ContextSet,
    ictx: &'a mut IntraModeContexts,
}

struct CuSrcPlanes<'a> {
    y: &'a [u16],
    cb: &'a [u16],
    cr: &'a [u16],
    src_yw: usize,
}

struct CuRecPlanes<'a> {
    y: &'a mut [u16],
    cb: &'a mut [u16],
    cr: &'a mut [u16],
}

struct CuParams {
    qp: u8,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
    /// Luma CU/prediction side: 8, 16, or 32. The root TU may either match
    /// this size or split once into four child TUs.
    lu: usize,
    /// Frame-constant intra RD multiplier, precomputed once from the slice QP.
    lambda: f32,
    /// Lossless (transquant-bypass) coding: code `cu_transquant_bypass_flag = 1`,
    /// skip transform/quantization, and apply inferred RDPCM for pure H/V modes.
    lossless: bool,
}

/// CU-QP-delta state for one 64x64 quantization group. The state is copied for
/// speculative RDO branches and committed only by the selected coding path.
#[derive(Clone, Copy, Debug)]
struct AqCtuState {
    enabled: bool,
    predictor: u8,
    target: u8,
    coded: bool,
}

impl AqCtuState {
    #[inline]
    fn resolved_qp(self) -> u8 {
        if !self.enabled || self.coded {
            self.target
        } else {
            self.predictor
        }
    }
}

#[inline]
fn activity_aq_enabled(qp: u8, lossless: bool) -> bool {
    !lossless && qp >= 24
}

/// Encode HEVC `cu_qp_delta_abs` and `cu_qp_delta_sign_flag`. The absolute
/// value uses truncated unary with cMax=5, followed by bypass EG0 for larger
/// values (HEVC §9.3.3.5). AQ clamps offsets to ±3, but the complete binarizer
/// avoids baking that policy into the syntax writer.
fn encode_cu_qp_delta<W: CabacWriter>(enc: &mut W, ctx: &mut ContextSet, aq: &mut AqCtuState) {
    if !aq.enabled || aq.coded {
        return;
    }
    let delta = i32::from(aq.target) - i32::from(aq.predictor);
    let abs = delta.unsigned_abs();
    let prefix = abs.min(5);
    for bin in 0..prefix {
        enc.encode_bin(1, &mut ctx.cu_qp_delta_abs[usize::from(bin != 0)]);
    }
    if prefix < 5 {
        enc.encode_bin(0, &mut ctx.cu_qp_delta_abs[usize::from(prefix != 0)]);
    } else {
        let code_num = abs - 5;
        let value = code_num + 1;
        let order = 31 - value.leading_zeros();
        for _ in 0..order {
            enc.encode_bypass(1);
        }
        enc.encode_bypass(0);
        for bit in (0..order).rev() {
            enc.encode_bypass(((value >> bit) & 1) as u8);
        }
    }
    if abs != 0 {
        enc.encode_bypass(u8::from(delta < 0));
    }
    aq.coded = true;
}

#[inline]
fn encode_cu_qp_delta_if_needed<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    aq: &mut AqCtuState,
    need_qp: bool,
) {
    if need_qp {
        encode_cu_qp_delta(enc, ctx, aq);
    }
}

/// Frame-constant coding parameters threaded through the CTU/CU intra RD search.
#[derive(Clone, Copy)]
struct SliceCoding {
    qp: u8,
    lambda: f32,
    lossless: bool,
}

/// Mutable reconstruction and neighbor-map state for coding one CTU (or a WPP row
/// band): the padded reconstruction planes plus the per-8×8 CU-depth and per-4×4
/// intra-mode maps and their strides. Passed by value so a per-CTU reborrow yields
/// plain `&mut [_]` fields (the body uses them directly).
struct CtuRecState<'a> {
    rec_y: &'a mut [u16],
    rec_cb: &'a mut [u16],
    rec_cr: &'a mut [u16],
    mode_map: &'a mut [u8],
    cu_depth: &'a mut [u8],
    edge_v: &'a mut [bool],
    edge_h: &'a mut [bool],
    qp_map: &'a mut [u8],
    cu_stride: usize,
    mode_stride: usize,
}

/// How one region (whole picture or one HEVC tile) is parallelized and terminated.
#[derive(Clone, Copy)]
struct ParallelPlan<'a> {
    wpp: bool,
    threads: usize,
    tile_cols: usize,
    tile_rows: usize,
    pool: &'a crate::pool::ThreadPool,
}

/// Luma position/stride geometry for committing one split (NxN child) TB.
#[derive(Clone, Copy)]
struct LumaSplitGeom {
    stride: usize,
    coded_yh: usize,
    block_row: usize,
    block_col: usize,
    parent: usize,
}

/// Per-TB luma coding parameters and pixel-format constants.
#[derive(Clone, Copy)]
struct LumaTbCoding {
    mode: u8,
    qp: u8,
    bit_depth: u8,
    max_val: u16,
    neutral: u16,
    lambda: f32,
    lossless: bool,
}

/// The four chroma planes involved in a chroma-TB decision: the two source planes
/// and the two mutable reconstruction planes.
struct ChromaPlanes<'a> {
    src_cb: &'a [u16],
    src_cr: &'a [u16],
    rec_cb: &'a mut [u16],
    rec_cr: &'a mut [u16],
}

/// Chroma/luma stride + position geometry shared by chroma-TB commit and search.
#[derive(Clone, Copy)]
struct ChromaSplitGeom {
    src_cw: usize,
    src_ch: usize,
    cw_stride: usize,
    coded_ch_h: usize,
    lu_row: usize,
    lu_col: usize,
    parent_luma: usize,
    yw_stride: usize,
    coded_yh: usize,
}

/// Chroma coding parameters and pixel-format constants (no per-candidate mode).
#[derive(Clone, Copy)]
struct ChromaCoding {
    chroma: crate::fmt::ChromaFormat,
    chroma_qp: u8,
    bit_depth: u8,
    max_val: u16,
    lambda: f32,
}

/// Options for one chroma-mode RD evaluation.
struct ChromaEvalOpts {
    candidate: ChromaModeCandidate,
    estimate_rate: bool,
    winner_rdoq: bool,
    cost_limit: f32,
}

/// Stride/position geometry for the PART_NxN intra CU coder.
#[derive(Clone, Copy)]
struct NxnGeom {
    lu_row: usize,
    lu_col: usize,
    yw_stride: usize,
    src_yh: usize,
    cw_stride: usize,
    src_cw: usize,
    src_ch: usize,
    coded_yh: usize,
    coded_ch_h: usize,
    mode_stride: usize,
}

/// Coding parameters for the PART_NxN intra CU coder.
#[derive(Clone, Copy)]
struct NxnParams {
    qp_slice: u8,
    qp: u8,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
    lambda: f32,
}

/// Luma position of one CU: min-CU grid row/col and the CU side (8/16/32).
#[derive(Clone, Copy)]
struct CuPos {
    lu_row: usize,
    lu_col: usize,
    lu: usize,
}

struct ChromaTb {
    cb_zz: [i16; 1024],
    cb_nz: bool,
    cr_zz: [i16; 1024],
    cr_nz: bool,
}

impl ChromaTb {
    const fn new() -> Self {
        Self {
            cb_zz: [0; 1024],
            cb_nz: false,
            cr_zz: [0; 1024],
            cr_nz: false,
        }
    }
}

/// Packed coefficient storage for both the normal one-level transform search and
/// the complete 32→16→8→4 lossless quadtree. Chroma coefficients follow child
/// Z-order and, for 4:2:2, upper/lower TB order inside each child. Keeping this in
/// the persistent compression context avoids per-CU allocation.
struct TuTreeScratch {
    y_zz: [i16; 1024],
    cb_zz: [i16; 1024],
    cr_zz: [i16; 1024],
    y_nz: [bool; 4],
    cb_nz: [bool; 8],
    cr_nz: [bool; 8],
    y_scan_idx: [u8; 4],
    chroma_scan_idx: [u8; 8],
    split: [bool; 85],
    node_y_nz: [bool; 85],
    node_y_scan_idx: [u8; 85],
    node_cb_nz: [[bool; 2]; 85],
    node_cr_nz: [[bool; 2]; 85],
    node_chroma_offset: [[u16; 2]; 85],
    node_chroma_side: [u8; 85],
    node_chroma_scan_idx: [u8; 85],
}

impl TuTreeScratch {
    const fn new() -> Self {
        Self {
            y_zz: [0; 1024],
            cb_zz: [0; 1024],
            cr_zz: [0; 1024],
            y_nz: [false; 4],
            cb_nz: [false; 8],
            cr_nz: [false; 8],
            y_scan_idx: [0; 4],
            chroma_scan_idx: [0; 8],
            split: [false; 85],
            node_y_nz: [false; 85],
            node_y_scan_idx: [0; 85],
            node_cb_nz: [[false; 2]; 85],
            node_cr_nz: [[false; 2]; 85],
            node_chroma_offset: [[0; 2]; 85],
            node_chroma_side: [0; 85],
            node_chroma_scan_idx: [0; 85],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TuLayout {
    Unsplit,
    Split,
    Recursive,
}

/// Per-slice reusable working set for CU prediction, transform and RDO. Keeping
/// the largest 32×32 buffers in one heap allocation removes repeated stack
/// zeroing and return-value memmoves from every mode/CU while remaining safe and
/// naturally private to each parallel tile encoder.
#[repr(align(64))]
struct CompressionContext {
    satd: crate::cost::SatdFn,
    fwd_transform: crate::hevc_transform::FwdTransformFn,
    inv_transform: crate::hevc_transform::InvTransformFn,
    dequantize: crate::hevc_transform::DequantizeFn,
    last_tu_layout: TuLayout,
    orig: [u16; 1024],
    chroma_orig_cb: [u16; 1024],
    chroma_orig_cr: [u16; 1024],
    pred: [u16; 1024],
    best_pred: [u16; 1024],
    reconstructed: [u16; 1024],
    residual: [i32; 1024],
    best_residual: [i32; 1024],
    coeff: [i32; 1024],
    best_coeff: [i32; 1024],
    dequant: [i32; 1024],
    inverse: [i32; 1024],
    transform_tmp: [i32; 1024],
    levels: [i16; 1024],
    scanned: [i16; 1024],
    angular: intra::AngularScratch,
    chroma_tbs: [ChromaTb; 2],
    tu_tree: TuTreeScratch,
    rdoq: crate::hevc_transform::RdoqScratch,
}

impl CompressionContext {
    fn new() -> Self {
        Self {
            satd: crate::cost::resolve_satd(),
            fwd_transform: crate::hevc_transform::resolve_fwd_transform(),
            inv_transform: crate::hevc_transform::resolve_inv_transform(),
            dequantize: crate::hevc_transform::resolve_dequantize(),
            last_tu_layout: TuLayout::Unsplit,
            orig: [0; 1024],
            chroma_orig_cb: [0; 1024],
            chroma_orig_cr: [0; 1024],
            pred: [0; 1024],
            best_pred: [0; 1024],
            reconstructed: [0; 1024],
            residual: [0; 1024],
            best_residual: [0; 1024],
            coeff: [0; 1024],
            best_coeff: [0; 1024],
            dequant: [0; 1024],
            inverse: [0; 1024],
            transform_tmp: [0; 1024],
            levels: [0; 1024],
            scanned: [0; 1024],
            angular: intra::AngularScratch::new(),
            chroma_tbs: [ChromaTb::new(), ChromaTb::new()],
            tu_tree: TuTreeScratch::new(),
            rdoq: crate::hevc_transform::RdoqScratch::new(),
        }
    }

    #[inline]
    fn satd(&self, orig: &[u16], pred: &[u16], n: usize) -> u32 {
        // SAFETY: every SATD implementation has the same slice-based contract;
        // the selected implementation validates the block shape and lengths.
        unsafe { (self.satd)(orig, pred, n) }
    }
}

/// Coded and source plane dimensions for a frame, shared by the per-CU coding
/// and RD-distortion routines. `w`/`cw` are the 64-aligned coded luma/chroma
/// strides; `src_*` are the true (pre-padding) source plane extents used for
/// edge clamping; `sub_w`/`sub_h` are the chroma subsampling factors.
#[derive(Clone, Copy)]
struct PlaneStrides {
    w: usize,
    src_yw: usize,
    src_yh: usize,
    cw: usize,
    src_cw: usize,
    src_ch: usize,
    sub_w: usize,
    sub_h: usize,
}

/// Mutable picture state shared by recursive CU decisions. All maps are indexed
/// on the SPS minimum-CU (8×8 luma) grid.
struct CuTreeState<'a> {
    yuv: &'a Yuv,
    rec_y: &'a mut [u16],
    rec_cb: &'a mut [u16],
    rec_cr: &'a mut [u16],
    strides: PlaneStrides,
    qp: u8,
    lambda: f32,
    mode_map: &'a mut [u8],
    cu_depth: &'a mut [u8],
    edge_v: &'a mut [bool],
    edge_h: &'a mut [bool],
    qp_map: &'a mut [u8],
    cu_stride: usize,
    mode_stride: usize,
    lossless: bool,
    aq: AqCtuState,
    scratch: &'a mut CompressionContext,
}

#[inline]
fn split_cu_context(depths: &[u8], row: usize, col: usize, depth: u8, stride: usize) -> usize {
    let br = row / 8;
    let bc = col / 8;
    let left_deeper = bc > 0 && depths[br * stride + bc - 1] > depth;
    let above_deeper = br > 0 && depths[(br - 1) * stride + bc] > depth;
    left_deeper as usize + above_deeper as usize
}

#[inline]
fn fill_cu_depth(depths: &mut [u8], row: usize, col: usize, size: usize, depth: u8, stride: usize) {
    let side = size / 8;
    let br0 = row / 8;
    let bc0 = col / 8;
    for r in 0..side {
        depths[(br0 + r) * stride + bc0..(br0 + r) * stride + bc0 + side].fill(depth);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn encode_cu_leaf(
    cab: &mut CabacEncoder,
    ctx: &mut ContextSet,
    ictx: &mut IntraModeContexts,
    state: &mut CuTreeState<'_>,
    row: usize,
    col: usize,
    size: usize,
    depth: u8,
) {
    code_one_cu(
        Entropy {
            enc: cab,
            ctx,
            ictx,
        },
        state.yuv,
        CuRecPlanes {
            y: &mut *state.rec_y,
            cb: &mut *state.rec_cb,
            cr: &mut *state.rec_cr,
        },
        CuPos {
            lu_row: row,
            lu_col: col,
            lu: size,
        },
        state.strides,
        SliceCoding {
            qp: state.qp,
            lambda: state.lambda,
            lossless: state.lossless,
        },
        &mut *state.mode_map,
        state.mode_stride,
        &mut state.aq,
        &mut *state.scratch,
    );
    fill_cu_qp(
        &mut *state.qp_map,
        state.mode_stride,
        row,
        col,
        size,
        state.aq.resolved_qp(),
    );
    mark_deblock_edges(
        &mut *state.edge_v,
        &mut *state.edge_h,
        state.mode_stride,
        row,
        col,
        size,
        state.scratch.last_tu_layout,
        &state.scratch.tu_tree.split,
    );
    fill_cu_depth(&mut *state.cu_depth, row, col, size, depth, state.cu_stride);
}

#[allow(clippy::too_many_arguments)]
fn mark_deblock_edges(
    edge_v: &mut [bool],
    edge_h: &mut [bool],
    stride: usize,
    row: usize,
    col: usize,
    size: usize,
    layout: TuLayout,
    split: &[bool; 85],
) {
    let mut mark_box = |r: usize, c: usize, side: usize| {
        if c != 0 && c.is_multiple_of(8) {
            let gx = c / 4;
            for gy in r / 4..(r + side) / 4 {
                edge_v[gy * stride + gx] = true;
            }
        }
        if r != 0 && r.is_multiple_of(8) {
            let gy = r / 4;
            for gx in c / 4..(c + side) / 4 {
                edge_h[gy * stride + gx] = true;
            }
        }
    };
    mark_box(row, col, size);
    match layout {
        TuLayout::Unsplit => {}
        TuLayout::Split => {
            let child = size / 2;
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                mark_box(row + dy * child, col + dx * child, child);
            }
        }
        TuLayout::Recursive => {
            fn walk(
                mark: &mut impl FnMut(usize, usize, usize),
                split: &[bool; 85],
                node: usize,
                row: usize,
                col: usize,
                size: usize,
            ) {
                mark(row, col, size);
                if size > 4 && split[node] {
                    let child = size / 2;
                    for (q, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                        walk(
                            mark,
                            split,
                            node * 4 + 1 + q,
                            row + dy * child,
                            col + dx * child,
                            child,
                        );
                    }
                }
            }
            walk(&mut mark_box, split, 0, row, col, size);
        }
    }
}

/// Source-only split proxy. `score >= 1` means the expected prediction
/// improvement is large enough to pay for the extra CU syntax at the current QP.
/// It is deliberately not a second encoder: no prediction buffers, transforms,
/// quantization, reconstruction, CABAC contexts, or rollback are touched here.
///
/// Currently superseded by the full `rdo_cu32_plan` search; retained as the cheap
/// pruner a production SATD-gated hybrid would run before paying for full RDO.
#[allow(dead_code)]
#[inline]
fn fast_cu_split_score(state: &CuTreeState<'_>, row: usize, col: usize, size: usize) -> f32 {
    let shift = state.yuv.bit_depth.bits().saturating_sub(8) as u32;
    let half = size / 2;
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let mut q_sum = [0u64; 4];
    let mut q_sum_sq = [0u64; 4];
    let mut min_sample = u16::MAX;
    let mut max_sample = 0u16;
    let mut grad_sum = 0u64;
    let mut signed_grad_x = 0i64;
    let mut signed_grad_y = 0i64;
    let mut midline_sse = 0u64;

    for r in 0..size {
        let sy = (row + r).min(state.strides.src_yh - 1);
        for c in 0..size {
            let sx = (col + c).min(state.strides.src_yw - 1);
            let sample = state.yuv.y[sy * state.strides.src_yw + sx] >> shift;
            let v = sample as u64;
            sum += v;
            sum_sq += v * v;
            min_sample = min_sample.min(sample);
            max_sample = max_sample.max(sample);
            let q = (r >= half) as usize * 2 + (c >= half) as usize;
            q_sum[q] += v;
            q_sum_sq[q] += v * v;

            if c > 0 {
                let left_x = (col + c - 1).min(state.strides.src_yw - 1);
                let left = state.yuv.y[sy * state.strides.src_yw + left_x] >> shift;
                let diff = sample as i32 - left as i32;
                grad_sum += diff.unsigned_abs() as u64;
                signed_grad_x += diff as i64;
                if c == half {
                    midline_sse += (diff * diff) as u64;
                }
            }
            if r > 0 {
                let above_y = (row + r - 1).min(state.strides.src_yh - 1);
                let above = state.yuv.y[above_y * state.strides.src_yw + sx] >> shift;
                let diff = sample as i32 - above as i32;
                grad_sum += diff.unsigned_abs() as u64;
                signed_grad_y += diff as i64;
                if r == half {
                    midline_sse += (diff * diff) as u64;
                }
            }
        }
    }

    let pixels = (size * size) as f32;
    let mean = sum as f32 / pixels;
    let variance = (sum_sq as f32 / pixels - mean * mean).max(0.0);
    let q_pixels = (half * half) as f32;
    let mut within_variance = 0.0f32;
    for q in 0..4 {
        let q_mean = q_sum[q] as f32 / q_pixels;
        within_variance += (q_sum_sq[q] as f32 / q_pixels - q_mean * q_mean).max(0.0);
    }
    within_variance *= 0.25;

    let range = max_sample - min_sample;
    let qp = state.qp as f32;
    let flat_range = 2 + (state.qp / 18) as u16;
    let flat_variance = 1.5 + qp * 0.12;
    if range <= flat_range && variance <= flat_variance {
        return 0.0;
    }

    // Splitting can mainly recover the energy between quadrant means. Uniform
    // ramps also have large between-quadrant energy, but a single angular mode
    // predicts them well; gradient coherence discounts that false split signal.
    // A sharp edge crossing the split boundary is handled separately by the
    // midline term, so coherent step edges are not incorrectly suppressed.
    let between_energy = (variance - within_variance).max(0.0) * pixels;
    let coherence = if grad_sum == 0 {
        1.0
    } else {
        ((signed_grad_x.unsigned_abs() + signed_grad_y.unsigned_abs()) as f32 / grad_sum as f32)
            .min(1.0)
    };
    let incoherent = 1.0 - coherence;
    let predicted_gain = between_energy * incoherent * incoherent
        + midline_sse as f32 * if size == 32 { 0.35 } else { 0.50 };

    // Four children add three extra CU headers/modes plus more CBF/residual
    // signaling. Lambda already carries the QP dependence, while the larger
    // penalty at 16×16 avoids exploding into 8×8 CUs for mild texture.
    let extra_bits = if size == 32 { 52.0 } else { 76.0 };
    // Measurements above are normalized to 8-bit. Scale the rate term into
    // that same domain; the committed encoder compares raw-domain SSE, whose
    // magnitude grows by 4^shift for every extra source bit.
    let distortion_scale = (1u32 << (2 * shift)) as f32;
    let lambda = (if state.lossless {
        state.lambda.max(1.0)
    } else {
        state.lambda
    }) / distortion_scale;
    let rate_penalty = (lambda * extra_bits).max(1.0 / distortion_scale);
    let mut score = predicted_gain / rate_penalty;

    // Strong incoherent local texture is a reliable split signal even when its
    // four quadrant means happen to be similar (between_energy is then small).
    let avg_gradient = grad_sum as f32 / (2 * size * (size - 1)).max(1) as f32;
    let texture_limit = if size == 32 {
        48.0 + qp * 2.0
    } else {
        96.0 + qp * 2.8
    };
    let gradient_limit = if size == 32 {
        6.0 + qp * 0.10
    } else {
        9.0 + qp * 0.12
    };
    if within_variance >= texture_limit && avg_gradient >= gradient_limit && coherence < 0.55 {
        score = score.max(1.25);
    }
    score
}

/// Build the complete representable 32→16→8 CU plan without coding either
/// branch. Five source scans (one 32×32 and four 16×16) replace up to 21
/// transform/quantize/reconstruct leaf trials from the previous implementation.
#[allow(dead_code)]
fn fast_cu32_plan(state: &CuTreeState<'_>, row: usize, col: usize) -> Cu32Plan {
    let parent_score = fast_cu_split_score(state, row, col, 32);
    // Flat/coherent 32×32 nodes dominate natural images. Avoid even the four
    // child scans when the parent is nowhere near the split threshold.
    if parent_score < 0.20 {
        return Cu32Plan::default();
    }

    let mut child_scores = [0.0f32; 4];
    for (index, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        child_scores[index] = fast_cu_split_score(state, row + dy * 16, col + dx * 16, 16);
    }

    let split_children = child_scores.iter().filter(|&&score| score >= 1.0).count();
    let strongest_child = child_scores.iter().copied().fold(0.0f32, f32::max);
    let split_32 = parent_score >= 1.0
        || split_children >= 2
        || (strongest_child >= 2.0 && parent_score >= 0.45);
    if !split_32 {
        return Cu32Plan::default();
    }

    Cu32Plan {
        split_32: true,
        split_16: child_scores.map(|score| score >= 1.0),
    }
}

/// SSE (luma + chroma) between the reconstruction and the source over the
/// `size×size` luma region at (row, col). Source reads are clamped to the true
/// picture extent so partial edge CUs compare against replicated borders.
fn cu_region_sse(tree: &CuTreeState<'_>, row: usize, col: usize, size: usize) -> f32 {
    let s = tree.strides;
    let mut sse = 0.0f64;
    for r in 0..size {
        let sy = (row + r).min(s.src_yh - 1);
        for c in 0..size {
            let sx = (col + c).min(s.src_yw - 1);
            let src = tree.yuv.y[sy * s.src_yw + sx] as f64;
            let rec = tree.rec_y[(row + r) * s.w + (col + c)] as f64;
            let d = src - rec;
            sse += d * d;
        }
    }
    if !tree.rec_cb.is_empty() {
        let cw_s = size / s.sub_w;
        let ch_s = size / s.sub_h;
        let cr0 = row / s.sub_h;
        let cc0 = col / s.sub_w;
        let planes: [(&[u16], &[u16]); 2] = [
            (&tree.rec_cb[..], tree.yuv.cb.as_slice()),
            (&tree.rec_cr[..], tree.yuv.cr.as_slice()),
        ];
        for (rec_plane, src_plane) in planes {
            for r in 0..ch_s {
                let sy = (cr0 + r).min(s.src_ch - 1);
                for c in 0..cw_s {
                    let sx = (cc0 + c).min(s.src_cw - 1);
                    let src = src_plane[sy * s.src_cw + sx] as f64;
                    let rec = rec_plane[(cr0 + r) * s.cw + (cc0 + c)] as f64;
                    let d = src - rec;
                    sse += d * d;
                }
            }
        }
    }
    sse as f32
}

/// Trial-encode one CU into a fractional-bit estimator (real reconstruction into
/// rec, but no bitstream/context commit) and return its RD cost J = SSE + λ·bits.
/// Contexts are cloned so the caller's real state is untouched; the reconstruction
/// left in rec is overwritten later when the chosen plan is committed for real.
fn cost_leaf(
    tree: &mut CuTreeState<'_>,
    row: usize,
    col: usize,
    size: usize,
    base_ctx: &ContextSet,
    base_ictx: &IntraModeContexts,
) -> f32 {
    let mut est = CabacEstimator::default();
    let mut ctx = base_ctx.clone();
    let mut ictx = base_ictx.clone();
    let mut aq = tree.aq;
    code_one_cu(
        Entropy {
            enc: &mut est,
            ctx: &mut ctx,
            ictx: &mut ictx,
        },
        tree.yuv,
        CuRecPlanes {
            y: &mut *tree.rec_y,
            cb: &mut *tree.rec_cb,
            cr: &mut *tree.rec_cr,
        },
        CuPos {
            lu_row: row,
            lu_col: col,
            lu: size,
        },
        tree.strides,
        SliceCoding {
            qp: tree.qp,
            lambda: tree.lambda,
            lossless: tree.lossless,
        },
        &mut *tree.mode_map,
        tree.mode_stride,
        &mut aq,
        &mut *tree.scratch,
    );
    cu_region_sse(tree, row, col, size) + tree.lambda * est.bits()
}

/// Source-proxy confidence band for the hybrid CU-quadtree search. Full RDO is
/// paid only where the cheap `fast_cu_split_score` is *ambiguous* about the 32×32
/// node (score near the proxy's own split point of 1.0). Below the band the node
/// is confidently a single 32 (no trials); above it the proxy's structural plan is
/// trusted. This confines the expensive real trials to the small uncertain
/// minority, keeping runtime close to the pure proxy while recovering most of the
/// achievable RD gain.
///
/// (A cheaper SATD-domain estimator was tried in place of the real trials but did
/// not help: with an uncoded region of `rec_y` holding the source, every block
/// size predicts from near-perfect source neighbours, so SATD cannot see the
/// split benefit — that signal only appears once real reconstruction + RDOQ +
/// CABAC rate are measured, i.e. `cost_leaf`.)
const CU_RDO_BAND_LOW: f32 = 0.25;
const CU_RDO_BAND_HIGH: f32 = 3.0;
/// Within an RDO'd 32, a 16 quadrant only trials its four-8 split when the proxy
/// itself sees sub-block texture there.
const CU_RDO_SPLIT8_GATE: f32 = 0.30;

/// Hybrid rate–distortion CU-quadtree decision for one 32×32 region.
/// - proxy score < LOW  → confidently flat: commit a single 32, no trials.
/// - proxy score ≥ HIGH → confidently textured: trust the proxy's structural plan.
/// - otherwise (ambiguous) → measure real J of {32} vs {four 16s, each the cheaper
///   of 16 / four-8} by actually encoding each surviving candidate, and keep it.
///
/// The winning [`Cu32Plan`] is committed once by [`commit_cu32_plan`].
fn rdo_cu32_plan(
    tree: &mut CuTreeState<'_>,
    row: usize,
    col: usize,
    ctx: &ContextSet,
    ictx: &IntraModeContexts,
) -> Cu32Plan {
    let score = fast_cu_split_score(tree, row, col, 32);
    if score < CU_RDO_BAND_LOW {
        return Cu32Plan::default();
    }
    if score >= CU_RDO_BAND_HIGH {
        return fast_cu32_plan(tree, row, col);
    }

    // A coded split_cu_flag is a single context bin ≈ 1 bit; charge λ for each.
    let flag = tree.lambda;

    let cost_32 = cost_leaf(tree, row, col, 32, ctx, ictx);

    let mut cost_split = 0.0f32;
    let mut split_16 = [false; 4];
    for (index, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let r = row + dy * 16;
        let c = col + dx * 16;
        let cost_16 = cost_leaf(tree, r, c, 16, ctx, ictx);
        // Only trial the four-8 split where the proxy sees real sub-block texture.
        let cost_8 = if fast_cu_split_score(tree, r, c, 16) < CU_RDO_SPLIT8_GATE {
            f32::INFINITY
        } else {
            let mut sum = flag;
            for (ey, ex) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                sum += cost_leaf(tree, r + ey * 8, c + ex * 8, 8, ctx, ictx);
            }
            sum
        };
        // Both sub-options pay this 16-level split_cu_flag (0 or 1).
        split_16[index] = cost_8 < cost_16;
        cost_split += flag + cost_16.min(cost_8);
    }

    if cost_32 <= cost_split {
        Cu32Plan::default()
    } else {
        Cu32Plan {
            split_32: true,
            split_16,
        }
    }
}

/// Compact plan for one representable 32×32 subtree. `split_32 == false`
/// ignores `split_16`; otherwise each bit selects four 8×8 children for the
/// corresponding 16×16 quadrant in Z order.
#[derive(Clone, Copy, Default)]
struct Cu32Plan {
    split_32: bool,
    split_16: [bool; 4],
}

/// Encode a preselected 32×32 subtree with no speculative branches. Every leaf
/// gets the normal winner-only RDOQ exactly once.
#[allow(clippy::too_many_arguments)]
fn commit_cu32_plan(
    cab: &mut CabacEncoder,
    ctx: &mut ContextSet,
    ictx: &mut IntraModeContexts,
    state: &mut CuTreeState<'_>,
    row: usize,
    col: usize,
    depth: u8,
    plan: Cu32Plan,
) {
    let split_ctx = split_cu_context(state.cu_depth, row, col, depth, state.cu_stride);
    if !plan.split_32 {
        cab.encode_bin(0, &mut ctx.split_cu_flag[split_ctx]);
        encode_cu_leaf(cab, ctx, ictx, state, row, col, 32, depth);
        return;
    }

    cab.encode_bin(1, &mut ctx.split_cu_flag[split_ctx]);
    for (index, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let child_row = row + dy * 16;
        let child_col = col + dx * 16;
        let child_depth = depth + 1;
        let child_ctx = split_cu_context(
            state.cu_depth,
            child_row,
            child_col,
            child_depth,
            state.cu_stride,
        );
        if plan.split_16[index] {
            cab.encode_bin(1, &mut ctx.split_cu_flag[child_ctx]);
            for (cy, cx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                encode_cu_leaf(
                    cab,
                    ctx,
                    ictx,
                    state,
                    child_row + cy * 8,
                    child_col + cx * 8,
                    8,
                    child_depth + 1,
                );
            }
        } else {
            cab.encode_bin(0, &mut ctx.split_cu_flag[child_ctx]);
            encode_cu_leaf(cab, ctx, ictx, state, child_row, child_col, 16, child_depth);
        }
    }
}

#[inline]
fn split_transform_context(size: usize) -> usize {
    debug_assert!(matches!(size, 8 | 16 | 32));
    (5usize - size.trailing_zeros() as usize).min(2)
}

#[allow(clippy::too_many_arguments)]
fn commit_lossless_luma_leaf(
    scratch: &mut CompressionContext,
    rec_y: &mut [u16],
    stride: usize,
    coded_yh: usize,
    root_row: usize,
    root_col: usize,
    root_size: usize,
    rel_row: usize,
    rel_col: usize,
    size: usize,
    mode: u8,
    neutral: u16,
    max_val: u16,
    coeff_offset: usize,
    node: usize,
    reconstruct: bool,
) -> bool {
    let len = size * size;
    let row = root_row + rel_row;
    let col = root_col + rel_col;
    let ctus_x = stride / 64;
    let (corner0, above0, left0) = intra::get_reference_samples(
        rec_y,
        intra::LumaRefGeometry {
            stride,
            block_row: row,
            block_col: col,
            height: coded_yh,
            n: size,
            ctu: 64,
            ctus_x,
            min_pu: size,
            neutral,
        },
    );
    let (corner, above, left) = if intra::should_filter_refs(mode, size) {
        let (above, left) = intra::filter_references(corner0, &above0, &left0, size);
        let corner = ((above0[0] as i32 + 2 * corner0 as i32 + left0[0] as i32 + 2) >> 2) as u16;
        (corner, above, left)
    } else {
        (corner0, above0, left0)
    };
    match mode {
        0 => intra::predict_planar_into(&above, &left, size, &mut scratch.pred),
        1 => intra::predict_dc_into(&above, &left, size, true, &mut scratch.pred),
        _ => intra::predict_angular_into(
            corner,
            &above,
            &left,
            size,
            mode,
            true,
            max_val as i32,
            &mut scratch.pred,
            &mut scratch.angular,
        ),
    }
    for (r, residual) in scratch.residual[..len].chunks_exact_mut(size).enumerate() {
        let source = (rel_row + r) * root_size + rel_col;
        let prediction = r * size;
        for (c, dst) in residual.iter_mut().enumerate() {
            *dst = scratch.orig[source + c] as i32 - scratch.pred[prediction + c] as i32;
        }
    }
    forward_lossless_rdpcm_into(
        &scratch.residual[..len],
        size,
        implicit_rdpcm_mode(mode),
        &mut scratch.levels,
    );
    let log2 = size.trailing_zeros();
    let scan_idx = dct::scan_idx_for(mode, log2, true, false);
    let scan = dct::coeff_scan(log2, scan_idx);
    let packed = &mut scratch.tu_tree.y_zz[coeff_offset..coeff_offset + len];
    let mut nonzero = false;
    for (dst, &(r, c)) in packed.iter_mut().zip(scan) {
        *dst = scratch.levels[r * size + c];
        nonzero |= *dst != 0;
    }
    scratch.tu_tree.node_y_nz[node] = nonzero;
    scratch.tu_tree.node_y_scan_idx[node] = scan_idx;
    if reconstruct {
        for r in 0..size {
            let source = (rel_row + r) * root_size + rel_col;
            let destination = (row + r) * stride + col;
            rec_y[destination..destination + size]
                .copy_from_slice(&scratch.orig[source..source + size]);
        }
    }
    nonzero
}

#[allow(clippy::too_many_arguments)]
fn choose_lossless_luma_tree(
    scratch: &mut CompressionContext,
    rec_y: &mut [u16],
    stride: usize,
    coded_yh: usize,
    root_row: usize,
    root_col: usize,
    root_size: usize,
    rel_row: usize,
    rel_col: usize,
    size: usize,
    mode: u8,
    neutral: u16,
    max_val: u16,
    coeff_offset: usize,
    node: usize,
    depth: usize,
    base_ctx: &ContextSet,
) -> (f32, ContextSet, bool) {
    let nonzero = commit_lossless_luma_leaf(
        scratch,
        rec_y,
        stride,
        coded_yh,
        root_row,
        root_col,
        root_size,
        rel_row,
        rel_col,
        size,
        mode,
        neutral,
        max_val,
        coeff_offset,
        node,
        false,
    );
    let mut unsplit_ctx = base_ctx.clone();
    let mut unsplit_rate = 0.0;
    if size > 4 {
        unsplit_rate +=
            unsplit_ctx.split_transform_flag[split_transform_context(size)].estimate_and_update(0);
    }
    unsplit_rate +=
        unsplit_ctx.cbf_luma[if depth == 0 { 1 } else { 0 }].estimate_and_update(nonzero as u8);
    if nonzero {
        unsplit_rate += estimate_residual_bits(
            &mut unsplit_ctx,
            &scratch.tu_tree.y_zz[coeff_offset..coeff_offset + size * size],
            size.trailing_zeros(),
            true,
            scratch.tu_tree.node_y_scan_idx[node],
            false,
        );
    }
    if size == 4 {
        let _ = commit_lossless_luma_leaf(
            scratch,
            rec_y,
            stride,
            coded_yh,
            root_row,
            root_col,
            root_size,
            rel_row,
            rel_col,
            size,
            mode,
            neutral,
            max_val,
            coeff_offset,
            node,
            true,
        );
        scratch.tu_tree.split[node] = false;
        return (unsplit_rate, unsplit_ctx, nonzero);
    }

    let mut split_ctx = base_ctx.clone();
    let mut split_rate =
        split_ctx.split_transform_flag[split_transform_context(size)].estimate_and_update(1);
    let child = size / 2;
    let child_len = child * child;
    let mut split_nonzero = false;
    for (quadrant, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let (rate, next_ctx, nz) = choose_lossless_luma_tree(
            scratch,
            rec_y,
            stride,
            coded_yh,
            root_row,
            root_col,
            root_size,
            rel_row + dy * child,
            rel_col + dx * child,
            child,
            mode,
            neutral,
            max_val,
            coeff_offset + quadrant * child_len,
            node * 4 + 1 + quadrant,
            depth + 1,
            &split_ctx,
        );
        split_rate += rate;
        split_ctx = next_ctx;
        split_nonzero |= nz;
    }
    if split_rate < unsplit_rate {
        scratch.tu_tree.split[node] = true;
        scratch.tu_tree.node_y_nz[node] = split_nonzero;
        (split_rate, split_ctx, split_nonzero)
    } else {
        let nonzero = commit_lossless_luma_leaf(
            scratch,
            rec_y,
            stride,
            coded_yh,
            root_row,
            root_col,
            root_size,
            rel_row,
            rel_col,
            size,
            mode,
            neutral,
            max_val,
            coeff_offset,
            node,
            true,
        );
        scratch.tu_tree.split[node] = false;
        (unsplit_rate, unsplit_ctx, nonzero)
    }
}

#[allow(clippy::too_many_arguments)]
fn commit_lossless_chroma_leaf(
    scratch: &mut CompressionContext,
    src_cb: &[u16],
    src_cr: &[u16],
    rec_cb: &mut [u16],
    rec_cr: &mut [u16],
    src_cw: usize,
    src_ch: usize,
    cw_stride: usize,
    coded_ch_h: usize,
    yw_stride: usize,
    coded_yh: usize,
    luma_row: usize,
    luma_col: usize,
    luma_size: usize,
    chroma: crate::fmt::ChromaFormat,
    mode: u8,
    bit_depth: u8,
    max_val: u16,
    node: usize,
    cursor: &mut usize,
) {
    let sub_w = chroma.sub_w();
    let sub_h = chroma.sub_h();
    let side = luma_size / sub_w;
    let len = side * side;
    let stacked = chroma.chroma_tbs_per_cu();
    let ch_row = luma_row / sub_h;
    let ch_col = luma_col / sub_w;
    let scan_idx = dct::scan_idx_for(
        mode,
        side.trailing_zeros(),
        false,
        matches!(chroma, crate::fmt::ChromaFormat::Yuv444),
    );
    let scan = dct::coeff_scan(side.trailing_zeros(), scan_idx);
    scratch.tu_tree.node_chroma_side[node] = side as u8;
    scratch.tu_tree.node_chroma_scan_idx[node] = scan_idx;
    for t in 0..stacked {
        scratch.tu_tree.node_chroma_offset[node][t] = (*cursor + t * len) as u16;
    }

    for component in 0..2 {
        for t in 0..stacked {
            let row = ch_row + t * side;
            let source = if component == 0 { src_cb } else { src_cr };
            let orig = if component == 0 {
                &mut scratch.chroma_orig_cb[..len]
            } else {
                &mut scratch.chroma_orig_cr[..len]
            };
            extract_block_dyn_into(source, src_cw, src_ch, row, ch_col, side, orig);
            let ((bc0, ba, bl), (rc0, ra, rl)) = intra::get_reference_samples_chroma_pair(
                rec_cb,
                rec_cr,
                intra::ChromaRefGeometry {
                    stride: cw_stride,
                    block_row: row,
                    block_col: ch_col,
                    chroma_h: coded_ch_h,
                    n: side,
                    sub_w,
                    sub_h,
                    luma_w: yw_stride,
                    luma_h: coded_yh,
                    luma_ctus_x: yw_stride / 64,
                    min_luma_pu: luma_size,
                    cur_luma_row: luma_row + t * side * sub_h,
                    cur_luma_col: luma_col,
                    neutral: 1u16 << (bit_depth - 1),
                },
            );
            let (corner, above, left) = if component == 0 {
                (bc0, ba, bl)
            } else {
                (rc0, ra, rl)
            };
            intra::predict_chroma_tb_into(
                mode,
                corner,
                &above,
                &left,
                side,
                max_val as i32,
                &mut scratch.pred,
                &mut scratch.angular,
            );
            let orig = if component == 0 {
                &scratch.chroma_orig_cb[..len]
            } else {
                &scratch.chroma_orig_cr[..len]
            };
            intra::compute_residual_i32_into(
                orig,
                &scratch.pred[..len],
                side,
                &mut scratch.residual,
            );
            forward_lossless_rdpcm_into(
                &scratch.residual[..len],
                side,
                implicit_rdpcm_mode(mode),
                &mut scratch.levels,
            );
            let offset = *cursor + t * len;
            let packed = if component == 0 {
                &mut scratch.tu_tree.cb_zz[offset..offset + len]
            } else {
                &mut scratch.tu_tree.cr_zz[offset..offset + len]
            };
            let mut nonzero = false;
            for (dst, &(r, c)) in packed.iter_mut().zip(scan) {
                *dst = scratch.levels[r * side + c];
                nonzero |= *dst != 0;
            }
            if component == 0 {
                scratch.tu_tree.node_cb_nz[node][t] = nonzero;
            } else {
                scratch.tu_tree.node_cr_nz[node][t] = nonzero;
            }
            let rec = if component == 0 {
                &mut *rec_cb
            } else {
                &mut *rec_cr
            };
            for (src_row, dst_row) in orig
                .chunks_exact(side)
                .zip(rec[row * cw_stride + ch_col..].chunks_mut(cw_stride))
            {
                dst_row[..side].copy_from_slice(src_row);
            }
        }
    }
    *cursor += stacked * len;
}

#[allow(clippy::too_many_arguments)]
fn commit_lossless_chroma_tree(
    scratch: &mut CompressionContext,
    src_cb: &[u16],
    src_cr: &[u16],
    rec_cb: &mut [u16],
    rec_cr: &mut [u16],
    src_cw: usize,
    src_ch: usize,
    cw_stride: usize,
    coded_ch_h: usize,
    yw_stride: usize,
    coded_yh: usize,
    luma_row: usize,
    luma_col: usize,
    size: usize,
    chroma: crate::fmt::ChromaFormat,
    mode: u8,
    bit_depth: u8,
    max_val: u16,
    node: usize,
    cursor: &mut usize,
) {
    if scratch.tu_tree.split[node] {
        let child = size / 2;
        for (quadrant, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
            .into_iter()
            .enumerate()
        {
            commit_lossless_chroma_tree(
                scratch,
                src_cb,
                src_cr,
                rec_cb,
                rec_cr,
                src_cw,
                src_ch,
                cw_stride,
                coded_ch_h,
                yw_stride,
                coded_yh,
                luma_row + dy * child,
                luma_col + dx * child,
                child,
                chroma,
                mode,
                bit_depth,
                max_val,
                node * 4 + 1 + quadrant,
                cursor,
            );
        }
        if size == 8 && !matches!(chroma, crate::fmt::ChromaFormat::Yuv444) {
            commit_lossless_chroma_leaf(
                scratch, src_cb, src_cr, rec_cb, rec_cr, src_cw, src_ch, cw_stride, coded_ch_h,
                yw_stride, coded_yh, luma_row, luma_col, size, chroma, mode, bit_depth, max_val,
                node, cursor,
            );
        }
    } else if size > 4 || matches!(chroma, crate::fmt::ChromaFormat::Yuv444) {
        commit_lossless_chroma_leaf(
            scratch, src_cb, src_cr, rec_cb, rec_cr, src_cw, src_ch, cw_stride, coded_ch_h,
            yw_stride, coded_yh, luma_row, luma_col, size, chroma, mode, bit_depth, max_val, node,
            cursor,
        );
    }
}

fn aggregate_lossless_chroma(
    tree: &mut TuTreeScratch,
    node: usize,
    size: usize,
    chroma: crate::fmt::ChromaFormat,
) -> ([bool; 2], [bool; 2]) {
    if !tree.split[node] || (size == 8 && !matches!(chroma, crate::fmt::ChromaFormat::Yuv444)) {
        return (tree.node_cb_nz[node], tree.node_cr_nz[node]);
    }
    let mut cb = [false; 2];
    let mut cr = [false; 2];
    for quadrant in 0..4 {
        let (child_cb, child_cr) =
            aggregate_lossless_chroma(tree, node * 4 + 1 + quadrant, size / 2, chroma);
        for t in 0..2 {
            cb[t] |= child_cb[t];
            cr[t] |= child_cr[t];
        }
    }
    if matches!(chroma, crate::fmt::ChromaFormat::Yuv422) && size > 8 {
        cb = [cb[0] || cb[1]; 2];
        cr = [cr[0] || cr[1]; 2];
    }
    tree.node_cb_nz[node] = cb;
    tree.node_cr_nz[node] = cr;
    (cb, cr)
}

fn advance_lossless_luma_tree_context(
    tree: &TuTreeScratch,
    node: usize,
    size: usize,
    depth: usize,
    coeff_offset: usize,
    ctx: &mut ContextSet,
) {
    if size > 4 && tree.split[node] {
        let child = size / 2;
        let child_len = child * child;
        for quadrant in 0..4 {
            advance_lossless_luma_tree_context(
                tree,
                node * 4 + 1 + quadrant,
                child,
                depth + 1,
                coeff_offset + quadrant * child_len,
                ctx,
            );
        }
        return;
    }
    let nonzero = tree.node_y_nz[node];
    let _ = ctx.cbf_luma[if depth == 0 { 1 } else { 0 }].estimate_and_update(nonzero as u8);
    if nonzero {
        advance_residual_contexts(
            ctx,
            &tree.y_zz[coeff_offset..coeff_offset + size * size],
            size.trailing_zeros(),
            true,
            tree.node_y_scan_idx[node],
            false,
        );
    }
}

fn commit_split_luma(
    scratch: &mut CompressionContext,
    rec_y: &mut [u16],
    geom: LumaSplitGeom,
    coding: LumaTbCoding,
    ctx: &ContextSet,
) -> bool {
    let LumaSplitGeom {
        stride,
        coded_yh,
        block_row,
        block_col,
        parent,
    } = geom;
    let LumaTbCoding {
        mode,
        qp,
        bit_depth,
        max_val,
        neutral,
        lambda,
        lossless,
    } = coding;
    let child = parent / 2;
    let child_len = child * child;
    let log2_child = child.trailing_zeros();
    let scan_idx = dct::scan_idx_for(mode, log2_child, true, false);
    let scan = dct::coeff_scan(log2_child, scan_idx);
    let ctus_x = stride / 64;
    let mut residual_ctx = ctx.clone();
    let mut any_nonzero = false;

    // HEVC performs intra prediction per transform block, not once per CU: each
    // child TB is predicted from the reconstructed samples of the TBs decoded
    // before it (in Z-order, including its siblings inside this CU). Reconstruct
    // straight into rec_y so the next child sees the updated neighbours, exactly
    // as the decoder does.
    for (index, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let row_offset = dy * child;
        let col_offset = dx * child;
        let row = block_row + row_offset;
        let col = block_col + col_offset;

        let (corner0, above0, left0) = intra::get_reference_samples(
            rec_y,
            intra::LumaRefGeometry {
                stride,
                block_row: row,
                block_col: col,
                height: coded_yh,
                n: child,
                ctu: 64,
                ctus_x,
                min_pu: child,
                neutral,
            },
        );
        let (corner, above, left) = if intra::should_filter_refs(mode, child) {
            let (fa, fl) = intra::filter_references(corner0, &above0, &left0, child);
            let cf = ((above0[0] as i32 + 2 * corner0 as i32 + left0[0] as i32 + 2) >> 2) as u16;
            (cf, fa, fl)
        } else {
            (corner0, above0, left0)
        };
        match mode {
            0 => intra::predict_planar_into(&above, &left, child, &mut scratch.pred),
            1 => intra::predict_dc_into(&above, &left, child, true, &mut scratch.pred),
            _ => intra::predict_angular_into(
                corner,
                &above,
                &left,
                child,
                mode,
                true,
                max_val as i32,
                &mut scratch.pred,
                &mut scratch.angular,
            ),
        }

        // Residual = child original (a quadrant of the parent orig block) − pred.
        for (r, res_row) in scratch.residual[..child_len]
            .chunks_exact_mut(child)
            .enumerate()
        {
            let orig_base = (row_offset + r) * parent + col_offset;
            let pred_base = r * child;
            for (c, dst) in res_row.iter_mut().enumerate() {
                *dst = scratch.orig[orig_base + c] as i32 - scratch.pred[pred_base + c] as i32;
            }
        }

        if lossless {
            forward_lossless_rdpcm_into(
                &scratch.residual[..child_len],
                child,
                implicit_rdpcm_mode(mode),
                &mut scratch.levels,
            );
        } else {
            crate::hevc_transform::run_fwd_transform(
                scratch.fwd_transform,
                &scratch.residual[..child_len],
                child,
                bit_depth,
                &mut scratch.coeff,
                &mut scratch.transform_tmp,
                true,
            );
            let tb = crate::hevc_transform::RdoqTb {
                coeff: &scratch.coeff,
                n: child,
                qp,
                bit_depth,
                scan,
                scan_idx,
                lambda,
            };
            crate::hevc_transform::rdoq_luma_at_depth_with_sign_hiding_into(
                &tb,
                1,
                &residual_ctx,
                &mut scratch.levels,
                &mut scratch.rdoq,
            );
        }

        let packed_offset = index * child_len;
        let packed = &mut scratch.tu_tree.y_zz[packed_offset..packed_offset + child_len];
        let mut nonzero = false;
        for (dst, &(scan_row, scan_col)) in packed.iter_mut().zip(scan) {
            let level = scratch.levels[scan_row * child + scan_col];
            *dst = level;
            nonzero |= level != 0;
        }
        scratch.tu_tree.y_nz[index] = nonzero;
        scratch.tu_tree.y_scan_idx[index] = scan_idx;
        any_nonzero |= nonzero;
        let _ = residual_ctx.cbf_luma[0].estimate_and_update(nonzero as u8);
        if nonzero {
            advance_residual_contexts(
                &mut residual_ctx,
                packed,
                log2_child,
                true,
                scan_idx,
                !lossless,
            );
        }

        // Reconstruct pred + residual directly into the picture so subsequent
        // sibling TBs predict from it.
        let dst_start = row * stride + col;
        if lossless {
            for r in 0..child {
                let orig_base = (row_offset + r) * parent + col_offset;
                let dst_base = dst_start + r * stride;
                rec_y[dst_base..dst_base + child]
                    .copy_from_slice(&scratch.orig[orig_base..orig_base + child]);
            }
        } else {
            crate::hevc_transform::run_dequantize(
                scratch.dequantize,
                &scratch.levels,
                child,
                qp,
                bit_depth,
                &mut scratch.dequant,
            );
            crate::hevc_transform::run_inv_transform(
                scratch.inv_transform,
                &scratch.dequant,
                child,
                bit_depth,
                &mut scratch.inverse,
                &mut scratch.transform_tmp,
                true,
            );
            for (r, (inv_row, pred_row)) in scratch.inverse[..child_len]
                .chunks_exact(child)
                .zip(scratch.pred[..child_len].chunks_exact(child))
                .enumerate()
            {
                let base = dst_start + r * stride;
                for (dst, (&prediction, &residual)) in rec_y[base..base + child]
                    .iter_mut()
                    .zip(pred_row.iter().zip(inv_row))
                {
                    *dst = (prediction as i32 + residual).clamp(0, max_val as i32) as u16;
                }
            }
        }
    }

    any_nonzero
}

#[inline]
fn split_chroma_is_shared(parent_luma: usize, chroma: crate::fmt::ChromaFormat) -> bool {
    let child_luma = parent_luma / 2;
    child_luma / chroma.sub_w() < 4 || child_luma / chroma.sub_h() < 4
}

fn commit_split_chroma(
    scratch: &mut CompressionContext,
    planes: ChromaPlanes<'_>,
    geom: ChromaSplitGeom,
    coding: ChromaCoding,
    mode: u8,
    ctx_after_luma: &ContextSet,
    lossless: bool,
) {
    let ChromaPlanes {
        src_cb,
        src_cr,
        rec_cb,
        rec_cr,
    } = planes;
    let ChromaSplitGeom {
        src_cw,
        src_ch,
        cw_stride,
        coded_ch_h,
        lu_row,
        lu_col,
        parent_luma,
        yw_stride,
        coded_yh,
    } = geom;
    let ChromaCoding {
        chroma,
        chroma_qp,
        bit_depth,
        max_val,
        lambda,
    } = coding;
    debug_assert!(!chroma.is_monochrome());
    debug_assert!(!split_chroma_is_shared(parent_luma, chroma));

    let sub_w = chroma.sub_w();
    let sub_h = chroma.sub_h();
    let parent_side = parent_luma / sub_w;
    let child_side = parent_side / 2;
    let child_len = child_side * child_side;
    let child_log2 = child_side.trailing_zeros();
    let stacked = chroma.chroma_tbs_per_cu();
    let is_444 = matches!(chroma, crate::fmt::ChromaFormat::Yuv444);
    let scan_idx = dct::scan_idx_for(mode, child_log2, false, is_444);
    let scan = dct::coeff_scan(child_log2, scan_idx);
    let luma_ctus_x = yw_stride / 64;
    let ch_row = lu_row / sub_h;
    let ch_col = lu_col / sub_w;
    let mut residual_ctx = ctx_after_luma.clone();

    // HEVC predicts each chroma transform block from its own reconstructed
    // neighbours, exactly like luma. Walk every child TB in decode order and
    // reconstruct straight into rec_cb/rec_cr so later siblings see it. Cb and Cr
    // form independent prediction chains, so each component is processed in full.
    for component in 0..2 {
        let source_plane = if component == 0 { src_cb } else { src_cr };
        for child_index in 0..4 {
            let dy = child_index / 2;
            let dx = child_index % 2;
            for stack_index in 0..stacked {
                // One luma child covers `child_side` chroma rows in 4:2:0/4:4:4
                // and two stacked square TBs in 4:2:2. Advancing by parent_side
                // unconditionally skipped a row band for 4:2:0/4:4:4 and made
                // the lower children overrun the reconstruction at picture bottom.
                let child_height = child_side * stacked;
                let c_row = ch_row + dy * child_height + stack_index * child_side;
                let c_col = ch_col + dx * child_side;

                if component == 0 {
                    extract_block_dyn_into(
                        source_plane,
                        src_cw,
                        src_ch,
                        c_row,
                        c_col,
                        child_side,
                        &mut scratch.chroma_orig_cb[..child_len],
                    );
                } else {
                    extract_block_dyn_into(
                        source_plane,
                        src_cw,
                        src_ch,
                        c_row,
                        c_col,
                        child_side,
                        &mut scratch.chroma_orig_cr[..child_len],
                    );
                }

                let ((bc0, ba, bl), (rc0, ra, rl)) = intra::get_reference_samples_chroma_pair(
                    rec_cb,
                    rec_cr,
                    intra::ChromaRefGeometry {
                        stride: cw_stride,
                        block_row: c_row,
                        block_col: c_col,
                        chroma_h: coded_ch_h,
                        n: child_side,
                        sub_w,
                        sub_h,
                        luma_w: yw_stride,
                        luma_h: coded_yh,
                        luma_ctus_x,
                        min_luma_pu: 4,
                        cur_luma_row: c_row * sub_h,
                        cur_luma_col: c_col * sub_w,
                        neutral: 1u16 << (bit_depth - 1),
                    },
                );
                let filter = child_side > 4 && intra::should_filter_refs(mode, child_side);
                let (corner, above, left) = if component == 0 {
                    if filter {
                        let (a, l) = intra::filter_references(bc0, &ba, &bl, child_side);
                        let cf = ((ba[0] as i32 + 2 * bc0 as i32 + bl[0] as i32 + 2) >> 2) as u16;
                        (cf, a, l)
                    } else {
                        (bc0, ba, bl)
                    }
                } else if filter {
                    let (a, l) = intra::filter_references(rc0, &ra, &rl, child_side);
                    let cf = ((ra[0] as i32 + 2 * rc0 as i32 + rl[0] as i32 + 2) >> 2) as u16;
                    (cf, a, l)
                } else {
                    (rc0, ra, rl)
                };

                intra::predict_chroma_tb_into(
                    mode,
                    corner,
                    &above,
                    &left,
                    child_side,
                    max_val as i32,
                    &mut scratch.pred,
                    &mut scratch.angular,
                );
                if component == 0 {
                    intra::compute_residual_i32_into(
                        &scratch.chroma_orig_cb[..child_len],
                        &scratch.pred[..child_len],
                        child_side,
                        &mut scratch.residual,
                    );
                } else {
                    intra::compute_residual_i32_into(
                        &scratch.chroma_orig_cr[..child_len],
                        &scratch.pred[..child_len],
                        child_side,
                        &mut scratch.residual,
                    );
                }
                if lossless {
                    forward_lossless_rdpcm_into(
                        &scratch.residual[..child_len],
                        child_side,
                        implicit_rdpcm_mode(mode),
                        &mut scratch.levels,
                    );
                } else {
                    crate::hevc_transform::run_fwd_transform(
                        scratch.fwd_transform,
                        &scratch.residual[..child_len],
                        child_side,
                        bit_depth,
                        &mut scratch.coeff,
                        &mut scratch.transform_tmp,
                        false,
                    );
                    let tb = crate::hevc_transform::RdoqTb {
                        coeff: &scratch.coeff,
                        n: child_side,
                        qp: chroma_qp,
                        bit_depth,
                        scan,
                        scan_idx,
                        lambda,
                    };
                    crate::hevc_transform::rdoq_chroma_at_depth_with_sign_hiding_into(
                        &tb,
                        1,
                        &residual_ctx,
                        &mut scratch.levels,
                        &mut scratch.rdoq,
                    );
                }

                let block_index = child_index * stacked + stack_index;
                let packed_offset = block_index * child_len;
                let mut nonzero = false;
                if component == 0 {
                    let packed =
                        &mut scratch.tu_tree.cb_zz[packed_offset..packed_offset + child_len];
                    for (dst, &(sr, sc)) in packed.iter_mut().zip(scan) {
                        let level = scratch.levels[sr * child_side + sc];
                        *dst = level;
                        nonzero |= level != 0;
                    }
                    scratch.tu_tree.cb_nz[block_index] = nonzero;
                } else {
                    let packed =
                        &mut scratch.tu_tree.cr_zz[packed_offset..packed_offset + child_len];
                    for (dst, &(sr, sc)) in packed.iter_mut().zip(scan) {
                        let level = scratch.levels[sr * child_side + sc];
                        *dst = level;
                        nonzero |= level != 0;
                    }
                    scratch.tu_tree.cr_nz[block_index] = nonzero;
                }
                scratch.tu_tree.chroma_scan_idx[block_index] = scan_idx;
                let _ = residual_ctx.cbf_chroma[1].estimate_and_update(nonzero as u8);
                if nonzero {
                    let packed = if component == 0 {
                        &scratch.tu_tree.cb_zz[packed_offset..packed_offset + child_len]
                    } else {
                        &scratch.tu_tree.cr_zz[packed_offset..packed_offset + child_len]
                    };
                    advance_residual_contexts(
                        &mut residual_ctx,
                        packed,
                        child_log2,
                        false,
                        scan_idx,
                        !lossless,
                    );
                }

                let dst_start = c_row * cw_stride + c_col;
                let rec_plane = if component == 0 {
                    &mut rec_cb[..]
                } else {
                    &mut rec_cr[..]
                };
                if lossless {
                    let orig = if component == 0 {
                        &scratch.chroma_orig_cb[..child_len]
                    } else {
                        &scratch.chroma_orig_cr[..child_len]
                    };
                    for (src_row, dst_row) in orig
                        .chunks_exact(child_side)
                        .zip(rec_plane[dst_start..].chunks_mut(cw_stride))
                    {
                        dst_row[..child_side].copy_from_slice(src_row);
                    }
                } else {
                    crate::hevc_transform::run_dequantize(
                        scratch.dequantize,
                        &scratch.levels,
                        child_side,
                        chroma_qp,
                        bit_depth,
                        &mut scratch.dequant,
                    );
                    crate::hevc_transform::run_inv_transform(
                        scratch.inv_transform,
                        &scratch.dequant,
                        child_side,
                        bit_depth,
                        &mut scratch.inverse,
                        &mut scratch.transform_tmp,
                        false,
                    );
                    for (r, (inv_row, pred_row)) in scratch.inverse[..child_len]
                        .chunks_exact(child_side)
                        .zip(scratch.pred[..child_len].chunks_exact(child_side))
                        .enumerate()
                    {
                        let base = dst_start + r * cw_stride;
                        for (dst, (&prediction, &residual)) in rec_plane[base..base + child_side]
                            .iter_mut()
                            .zip(pred_row.iter().zip(inv_row))
                        {
                            *dst = (prediction as i32 + residual).clamp(0, max_val as i32) as u16;
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_chroma_mode(
    opts: ChromaEvalOpts,
    scratch: &mut CompressionContext,
    planes: ChromaPlanes<'_>,
    geom: ChromaSplitGeom,
    coding: ChromaCoding,
    trafo_depth: usize,
    residual_ctx_after_luma: &ContextSet,
    ictx: &IntraModeContexts,
) -> f32 {
    let ChromaEvalOpts {
        candidate,
        estimate_rate,
        winner_rdoq,
        cost_limit,
    } = opts;
    let ChromaPlanes {
        src_cb,
        src_cr,
        rec_cb,
        rec_cr,
    } = planes;
    let ChromaSplitGeom {
        src_cw,
        src_ch,
        cw_stride,
        coded_ch_h,
        lu_row,
        lu_col,
        parent_luma,
        yw_stride,
        coded_yh,
    } = geom;
    let ChromaCoding {
        chroma,
        chroma_qp,
        bit_depth,
        max_val,
        lambda,
    } = coding;
    let sub_w = chroma.sub_w();
    let sub_h = chroma.sub_h();
    let side = parent_luma / sub_w;
    let stacked = chroma.chroma_tbs_per_cu();
    let block_len = side * side;
    let log2_side = side.trailing_zeros();
    let is_444 = matches!(chroma, crate::fmt::ChromaFormat::Yuv444);
    let scan_idx = dct::scan_idx_for(candidate.pred_mode, log2_side, false, is_444);
    let scan = dct::coeff_scan(log2_side, scan_idx);
    let luma_ctus_x = yw_stride / 64;
    let ch_row = lu_row / sub_h;
    let ch_col = lu_col / sub_w;
    let mut distortion = 0.0f32;
    let mut rdoq_ctx = residual_ctx_after_luma.clone();

    for component in 0..2 {
        for stack_index in 0..stacked {
            let sub_ch_row = ch_row + stack_index * side;
            if component == 0 {
                extract_block_dyn_into(
                    src_cb,
                    src_cw,
                    src_ch,
                    sub_ch_row,
                    ch_col,
                    side,
                    &mut scratch.chroma_orig_cb[..block_len],
                );
            } else {
                extract_block_dyn_into(
                    src_cr,
                    src_cw,
                    src_ch,
                    sub_ch_row,
                    ch_col,
                    side,
                    &mut scratch.chroma_orig_cr[..block_len],
                );
            }

            let ((bc0, ba, bl), (rc0, ra, rl)) = intra::get_reference_samples_chroma_pair(
                rec_cb,
                rec_cr,
                intra::ChromaRefGeometry {
                    stride: cw_stride,
                    block_row: sub_ch_row,
                    block_col: ch_col,
                    chroma_h: coded_ch_h,
                    n: side,
                    sub_w,
                    sub_h,
                    luma_w: yw_stride,
                    luma_h: coded_yh,
                    luma_ctus_x,
                    // 4×4 PART_NxN chroma PUs (parent_luma == 4) need 4-sample
                    // decode-order granularity so a PU sees its reconstructed
                    // siblings; larger 2Nx2N chroma stays at the min-CU 8.
                    min_luma_pu: parent_luma.min(8),
                    // 4:2:2 stacks two square TBs; anchor each at its own luma row.
                    cur_luma_row: lu_row + stack_index * side * sub_h,
                    cur_luma_col: lu_col,
                    neutral: 1u16 << (bit_depth - 1),
                },
            );
            let filter = side > 4 && intra::should_filter_refs(candidate.pred_mode, side);
            let (baf, blf, bcf) = if filter {
                let (above, left) = intra::filter_references(bc0, &ba, &bl, side);
                let corner = ((ba[0] as i32 + 2 * bc0 as i32 + bl[0] as i32 + 2) >> 2) as u16;
                (above, left, corner)
            } else {
                (ba, bl, bc0)
            };
            let (raf, rlf, rcf) = if filter {
                let (above, left) = intra::filter_references(rc0, &ra, &rl, side);
                let corner = ((ra[0] as i32 + 2 * rc0 as i32 + rl[0] as i32 + 2) >> 2) as u16;
                (above, left, corner)
            } else {
                (ra, rl, rc0)
            };
            let (corner, above, left) = if component == 0 {
                (bcf, &baf[..], &blf[..])
            } else {
                (rcf, &raf[..], &rlf[..])
            };
            intra::predict_chroma_tb_into(
                candidate.pred_mode,
                corner,
                above,
                left,
                side,
                max_val as i32,
                &mut scratch.pred,
                &mut scratch.angular,
            );
            let orig = if component == 0 {
                &scratch.chroma_orig_cb[..block_len]
            } else {
                &scratch.chroma_orig_cr[..block_len]
            };
            intra::compute_residual_i32_into(
                orig,
                &scratch.pred[..block_len],
                side,
                &mut scratch.residual,
            );
            crate::hevc_transform::run_fwd_transform(
                scratch.fwd_transform,
                &scratch.residual[..block_len],
                side,
                bit_depth,
                &mut scratch.coeff,
                &mut scratch.transform_tmp,
                false,
            );
            if winner_rdoq {
                let tb = crate::hevc_transform::RdoqTb {
                    coeff: &scratch.coeff,
                    n: side,
                    qp: chroma_qp,
                    bit_depth,
                    scan,
                    scan_idx,
                    lambda,
                };
                crate::hevc_transform::rdoq_chroma_at_depth_with_sign_hiding_into(
                    &tb,
                    trafo_depth,
                    &rdoq_ctx,
                    &mut scratch.levels,
                    &mut scratch.rdoq,
                );
            } else {
                crate::hevc_transform::quantize_with_sign_hiding_into(
                    &scratch.coeff,
                    side,
                    chroma_qp,
                    bit_depth,
                    scan,
                    &mut scratch.levels,
                );
            }

            let tb = &mut scratch.chroma_tbs[stack_index];
            let (packed, nonzero) = if component == 0 {
                (&mut tb.cb_zz, &mut tb.cb_nz)
            } else {
                (&mut tb.cr_zz, &mut tb.cr_nz)
            };
            *nonzero = false;
            for (dst, &(scan_row, scan_col)) in packed[..block_len].iter_mut().zip(scan) {
                let level = scratch.levels[scan_row * side + scan_col];
                *dst = level;
                *nonzero |= level != 0;
            }
            if winner_rdoq {
                let _ = rdoq_ctx.cbf_chroma[trafo_depth.min(4)].estimate_and_update(*nonzero as u8);
                if *nonzero {
                    advance_residual_contexts(
                        &mut rdoq_ctx,
                        &packed[..block_len],
                        log2_side,
                        false,
                        scan_idx,
                        true,
                    );
                }
            }

            crate::hevc_transform::run_dequantize(
                scratch.dequantize,
                &scratch.levels,
                side,
                chroma_qp,
                bit_depth,
                &mut scratch.dequant,
            );
            crate::hevc_transform::run_inv_transform(
                scratch.inv_transform,
                &scratch.dequant,
                side,
                bit_depth,
                &mut scratch.inverse,
                &mut scratch.transform_tmp,
                false,
            );
            intra::reconstruct_into(
                &scratch.pred[..block_len],
                &scratch.inverse[..block_len],
                side,
                max_val,
                &mut scratch.reconstructed,
            );
            distortion += block_sse(orig, &scratch.reconstructed[..block_len], side);
            if estimate_rate && distortion >= cost_limit {
                return distortion;
            }
            let rec_plane = if component == 0 {
                &mut rec_cb[..]
            } else {
                &mut rec_cr[..]
            };
            let dst_start = sub_ch_row * cw_stride + ch_col;
            for (src_row, dst_row) in scratch.reconstructed[..block_len]
                .chunks_exact(side)
                .zip(rec_plane[dst_start..].chunks_mut(cw_stride))
            {
                dst_row[..side].copy_from_slice(src_row);
            }
        }
    }

    if !estimate_rate || distortion >= cost_limit {
        return distortion;
    }
    let mut trial_ctx = residual_ctx_after_luma.clone();
    let mut trial_ictx = ictx.clone();
    let mut rate = estimate_chroma_mode_bits(&mut trial_ictx, candidate.syntax_idx);
    for tb in &scratch.chroma_tbs[..stacked] {
        rate += trial_ctx.cbf_chroma[trafo_depth.min(4)].estimate_and_update(tb.cb_nz as u8);
    }
    for tb in &scratch.chroma_tbs[..stacked] {
        rate += trial_ctx.cbf_chroma[trafo_depth.min(4)].estimate_and_update(tb.cr_nz as u8);
    }
    for tb in &scratch.chroma_tbs[..stacked] {
        if tb.cb_nz {
            rate += estimate_residual_bits(
                &mut trial_ctx,
                &tb.cb_zz[..block_len],
                log2_side,
                false,
                scan_idx,
                true,
            );
        }
    }
    for tb in &scratch.chroma_tbs[..stacked] {
        if tb.cr_nz {
            rate += estimate_residual_bits(
                &mut trial_ctx,
                &tb.cr_zz[..block_len],
                log2_side,
                false,
                scan_idx,
                true,
            );
        }
    }
    distortion + lambda * rate
}

#[inline]
fn choose_nxn_proxy(
    satd: crate::cost::SatdFn,
    orig: &[u16],
    parent_pred: &[u16],
    lambda: f32,
    bit_depth: u8,
) -> bool {
    debug_assert!(orig.len() >= 64 && parent_pred.len() >= 64);

    // PART_NxN is expensive only after it has been selected: four independent
    // luma mode searches, DSTs and residuals. Keep the gate itself to two tiny
    // source/residual passes. Per-quadrant residual means estimate the gain from
    // independent predictors, while gradient-orientation spread identifies edges
    // that one 8×8 direction cannot represent well.
    let depth_scale = 1u64 << bit_depth.saturating_sub(8);
    // SAFETY: the two slices were checked above and 8 is a supported block size.
    let parent_satd = unsafe { satd(&orig[..64], &parent_pred[..64], 8) } as f32;
    let satd_floor = 48.0 * depth_scale as f32 + lambda.sqrt() * 8.0;
    if parent_satd <= satd_floor {
        return false;
    }

    let mut residual_sum = [0i64; 4];
    let mut residual_abs = [0u64; 4];
    let mut gradient_x = [0u64; 4];
    let mut gradient_y = [0u64; 4];

    for row in 0..8 {
        let quadrant_row = (row >= 4) as usize * 2;
        let orig_row = &orig[row * 8..row * 8 + 8];
        let pred_row = &parent_pred[row * 8..row * 8 + 8];
        for col in 0..8 {
            let quadrant = quadrant_row + (col >= 4) as usize;
            let residual = orig_row[col] as i32 - pred_row[col] as i32;
            residual_sum[quadrant] += residual as i64;
            residual_abs[quadrant] += residual.unsigned_abs() as u64;

            // Do not cross a 4×4 quadrant boundary: each statistic describes the
            // direction preferred by one prospective child PU.
            if col & 3 != 0 {
                gradient_x[quadrant] += orig_row[col].abs_diff(orig_row[col - 1]) as u64;
            }
            if row & 3 != 0 {
                gradient_y[quadrant] += orig_row[col].abs_diff(orig[(row - 1) * 8 + col]) as u64;
            }
        }
    }

    let total_abs: u64 = residual_abs.into_iter().sum();
    if total_abs == 0 {
        return false;
    }

    let means = residual_sum.map(|sum| {
        if sum >= 0 {
            ((sum + 8) / 16) as i32
        } else {
            ((sum - 8) / 16) as i32
        }
    });
    let mut adjusted_abs = 0u64;
    for row in 0..8 {
        let quadrant_row = (row >= 4) as usize * 2;
        let orig_row = &orig[row * 8..row * 8 + 8];
        let pred_row = &parent_pred[row * 8..row * 8 + 8];
        for col in 0..8 {
            let quadrant = quadrant_row + (col >= 4) as usize;
            let residual = orig_row[col] as i32 - pred_row[col] as i32;
            adjusted_abs += residual.abs_diff(means[quadrant]) as u64;
        }
    }
    let dc_gain = total_abs.saturating_sub(adjusted_abs);

    let mut min_orientation = 256u64;
    let mut max_orientation = 0u64;
    let mut active_quadrants = 0usize;
    let mut total_gradient = 0u64;
    let activity_floor = 12 * depth_scale;
    for (&gx, &gy) in gradient_x.iter().zip(&gradient_y) {
        let activity = gx + gy;
        total_gradient += activity;
        if activity < activity_floor {
            continue;
        }
        active_quadrants += 1;
        let orientation = (gx * 256 + activity / 2) / activity;
        min_orientation = min_orientation.min(orientation);
        max_orientation = max_orientation.max(orientation);
    }

    let min_mean = means.into_iter().min().unwrap_or(0);
    let max_mean = means.into_iter().max().unwrap_or(0);
    let mean_span = max_mean.abs_diff(min_mean) as u64;
    let gain_floor = (lambda.sqrt() * 18.0) as u64 * depth_scale;
    let useful_dc_gain =
        dc_gain > gain_floor.max(8 * depth_scale) && dc_gain * 100 >= total_abs * 15;
    let mixed_direction = active_quadrants >= 2
        && max_orientation.saturating_sub(min_orientation) >= 88
        && total_gradient >= 64 * depth_scale;
    let piecewise_offset = mean_span >= 5 * depth_scale;

    // The gate is deliberately conservative. A false negative merely keeps the
    // normal 2Nx2N path; a false positive triggers four complete mode searches.
    useful_dc_gain && (piecewise_offset || mixed_direction)
}

struct NxNChroma444<'a> {
    src_cb: &'a [u16],
    src_cr: &'a [u16],
    rec_cb: &'a mut [u16],
    rec_cr: &'a mut [u16],
    src_cw: usize,
    src_ch: usize,
    cw_stride: usize,
    coded_ch_h: usize,
    lu_row: usize,
    lu_col: usize,
    yw_stride: usize,
    coded_yh: usize,
    qp_slice: u8,
    bit_depth: crate::fmt::BitDepth,
    lambda: f32,
    luma_modes: [u8; 4],
    residual_ctx_after_luma: &'a ContextSet,
}

/// 4:4:4 keeps four chroma PUs in an 8×8 PART_NxN CU. Select and reconstruct
/// each 4×4 chroma mode in luma-PU coding order, then retain its winner-only
/// RDOQ coefficients for the inferred split transform tree.
fn encode_nxn_chroma_444<W: CabacWriter>(
    enc: &mut W,
    ictx: &mut IntraModeContexts,
    scratch: &mut CompressionContext,
    job: NxNChroma444<'_>,
) {
    const SIDE: usize = 4;
    const LEN: usize = SIDE * SIDE;

    let NxNChroma444 {
        src_cb,
        src_cr,
        rec_cb,
        rec_cr,
        src_cw,
        src_ch,
        cw_stride,
        coded_ch_h,
        lu_row,
        lu_col,
        yw_stride,
        coded_yh,
        qp_slice,
        bit_depth,
        lambda,
        luma_modes,
        residual_ctx_after_luma,
    } = job;
    let chroma = crate::fmt::ChromaFormat::Yuv444;
    let chroma_qp = chroma_qp_for(qp_slice, chroma) + bit_depth.qp_bd_offset();
    let chroma_lambda = lambda * chroma_lambda_scale(qp_slice, chroma);
    let max_val = bit_depth.max_val();
    let neutral = bit_depth.neutral();
    let mut residual_ctx = residual_ctx_after_luma.clone();

    for (child_index, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let row = lu_row + dy * SIDE;
        let col = lu_col + dx * SIDE;
        extract_block_dyn_into(
            src_cb,
            src_cw,
            src_ch,
            row,
            col,
            SIDE,
            &mut scratch.chroma_orig_cb[..LEN],
        );
        extract_block_dyn_into(
            src_cr,
            src_cw,
            src_ch,
            row,
            col,
            SIDE,
            &mut scratch.chroma_orig_cr[..LEN],
        );
        let ((bc0, ba, bl), (rc0, ra, rl)) = intra::get_reference_samples_chroma_pair(
            rec_cb,
            rec_cr,
            intra::ChromaRefGeometry {
                stride: cw_stride,
                block_row: row,
                block_col: col,
                chroma_h: coded_ch_h,
                n: SIDE,
                sub_w: 1,
                sub_h: 1,
                luma_w: yw_stride,
                luma_h: coded_yh,
                luma_ctus_x: yw_stride / 64,
                min_luma_pu: 4,
                cur_luma_row: row,
                cur_luma_col: col,
                neutral,
            },
        );

        let mut ranked = [ChromaModeCandidate {
            pred_mode: 0,
            syntax_idx: 0,
            cost: f32::MAX,
        }; 5];
        for mut candidate in chroma_mode_candidates(luma_modes[child_index], chroma) {
            let mut cost =
                chroma_lambda.sqrt() * estimated_chroma_mode_bins(candidate.syntax_idx) as f32;
            intra::predict_chroma_tb_into(
                candidate.pred_mode,
                bc0,
                &ba,
                &bl,
                SIDE,
                max_val as i32,
                &mut scratch.pred,
                &mut scratch.angular,
            );
            cost += scratch.satd(&scratch.chroma_orig_cb[..LEN], &scratch.pred[..LEN], SIDE) as f32;
            intra::predict_chroma_tb_into(
                candidate.pred_mode,
                rc0,
                &ra,
                &rl,
                SIDE,
                max_val as i32,
                &mut scratch.pred,
                &mut scratch.angular,
            );
            cost += scratch.satd(&scratch.chroma_orig_cr[..LEN], &scratch.pred[..LEN], SIDE) as f32;
            candidate.cost = cost;
            update_chroma_candidate(&mut ranked, candidate);
        }

        let exact_count = full_rdo_chroma_count(&ranked, chroma);
        let mut best = ranked[0];
        if exact_count > 1 {
            let mut best_cost = f32::MAX;
            for &candidate in &ranked[..exact_count] {
                let cost = evaluate_chroma_mode(
                    ChromaEvalOpts {
                        candidate,
                        estimate_rate: true,
                        winner_rdoq: false,
                        cost_limit: best_cost,
                    },
                    scratch,
                    ChromaPlanes {
                        src_cb,
                        src_cr,
                        rec_cb: &mut *rec_cb,
                        rec_cr: &mut *rec_cr,
                    },
                    ChromaSplitGeom {
                        src_cw,
                        src_ch,
                        cw_stride,
                        coded_ch_h,
                        lu_row: row,
                        lu_col: col,
                        parent_luma: SIDE,
                        yw_stride,
                        coded_yh,
                    },
                    ChromaCoding {
                        chroma,
                        chroma_qp,
                        bit_depth: bit_depth.bits(),
                        max_val,
                        lambda: chroma_lambda,
                    },
                    1,
                    &residual_ctx,
                    ictx,
                );
                if cost < best_cost {
                    best_cost = cost;
                    best = candidate;
                }
            }
        }
        let _ = evaluate_chroma_mode(
            ChromaEvalOpts {
                candidate: best,
                estimate_rate: false,
                winner_rdoq: true,
                cost_limit: f32::MAX,
            },
            scratch,
            ChromaPlanes {
                src_cb,
                src_cr,
                rec_cb: &mut *rec_cb,
                rec_cr: &mut *rec_cr,
            },
            ChromaSplitGeom {
                src_cw,
                src_ch,
                cw_stride,
                coded_ch_h,
                lu_row: row,
                lu_col: col,
                parent_luma: SIDE,
                yw_stride,
                coded_yh,
            },
            ChromaCoding {
                chroma,
                chroma_qp,
                bit_depth: bit_depth.bits(),
                max_val,
                lambda: chroma_lambda,
            },
            1,
            &residual_ctx,
            ictx,
        );

        let offset = child_index * LEN;
        let (cb_nz, cr_nz) = {
            let tb = &scratch.chroma_tbs[0];
            scratch.tu_tree.cb_zz[offset..offset + LEN].copy_from_slice(&tb.cb_zz[..LEN]);
            scratch.tu_tree.cr_zz[offset..offset + LEN].copy_from_slice(&tb.cr_zz[..LEN]);
            (tb.cb_nz, tb.cr_nz)
        };
        scratch.tu_tree.cb_nz[child_index] = cb_nz;
        scratch.tu_tree.cr_nz[child_index] = cr_nz;
        let scan_idx = dct::scan_idx_for(best.pred_mode, 2, false, true);
        scratch.tu_tree.chroma_scan_idx[child_index] = scan_idx;
        encode_chroma_mode(enc, ictx, best.syntax_idx);

        let _ = residual_ctx.cbf_chroma[1].estimate_and_update(cb_nz as u8);
        let _ = residual_ctx.cbf_chroma[1].estimate_and_update(cr_nz as u8);
        if cb_nz {
            advance_residual_contexts(
                &mut residual_ctx,
                &scratch.tu_tree.cb_zz[offset..offset + LEN],
                2,
                false,
                scan_idx,
                true,
            );
        }
        if cr_nz {
            advance_residual_contexts(
                &mut residual_ctx,
                &scratch.tu_tree.cr_zz[offset..offset + LEN],
                2,
                false,
                scan_idx,
                true,
            );
        }
    }

    // The real transform-tree syntax is emitted after all prediction modes.
}

#[allow(clippy::too_many_arguments)]
fn encode_cu_nxn<W: CabacWriter>(
    ent: Entropy<'_, W>,
    src: CuSrcPlanes<'_>,
    rec: CuRecPlanes<'_>,
    geom: NxnGeom,
    params: NxnParams,
    mode_map: &mut [u8],
    challenger_cost: f32,
    aq: &mut AqCtuState,
    scratch: &mut CompressionContext,
) -> bool {
    let Entropy { enc, ctx, ictx } = ent;
    let CuSrcPlanes {
        y: src_y,
        cb: src_cb,
        cr: src_cr,
        src_yw,
    } = src;
    let CuRecPlanes {
        y: rec_y,
        cb: rec_cb,
        cr: rec_cr,
    } = rec;
    let NxnGeom {
        lu_row,
        lu_col,
        yw_stride,
        src_yh,
        cw_stride,
        src_cw,
        src_ch,
        coded_yh,
        coded_ch_h,
        mode_stride,
    } = geom;
    let NxnParams {
        qp_slice,
        qp,
        chroma,
        bit_depth,
        lambda,
    } = params;
    const PU: usize = 4;
    const PU_LEN: usize = 16;
    let max_val = bit_depth.max_val();
    let neutral = bit_depth.neutral();
    let mut rdoq_ctx = ctx.clone();
    let mut luma_modes = [0u8; 4];
    // Each PU's MPM list, captured at decision time so the luma mode syntax can
    // be emitted in HEVC order (all four flags, then all four remainders) after
    // every PU has been reconstructed.
    let mut pu_mpm = [[0u8; 3]; 4];
    // Accumulated luma J = SSE + λ·rate across the four PUs; compared against the
    // 2Nx2N luma cost so PART_NxN is committed only when it is actually cheaper.
    // The per-PU loop below touches only rec_y / mode_map / cloned contexts (never
    // the real `enc`/`ctx`/`ictx`), so a losing NxN candidate leaves no bitstream
    // side effects and the 2Nx2N path overwrites its scratch reconstruction.
    let mut nxn_cost = 0.0f32;

    for (pu_index, (dy, dx)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let row = lu_row + dy * PU;
        let col = lu_col + dx * PU;
        extract_block_n_into::<PU>(src_y, src_yw, src_yh, row, col, &mut scratch.orig);

        let internal_left = dx != 0;
        let internal_above = dy != 0;
        let avail_left =
            col > 0 && (internal_left || is_block_decoded(row, col - 1, row, col, 64, yw_stride));
        let above_in_same_ctb = row > 0 && row > (row / 64) * 64;
        let avail_above = row > 0
            && above_in_same_ctb
            && (internal_above || is_block_decoded(row - 1, col, row, col, 64, yw_stride));
        let mode_at = |r: usize, c: usize| mode_map[(r / 4) * mode_stride + c / 4];
        let cand_a = if avail_left { mode_at(row, col - 1) } else { 1 };
        let cand_b = if avail_above {
            mode_at(row - 1, col)
        } else {
            1
        };
        let mpm = mpm_list(cand_a, cand_b);

        let (corner, above, left) = intra::get_reference_samples(
            rec_y,
            intra::LumaRefGeometry {
                stride: yw_stride,
                block_row: row,
                block_col: col,
                height: coded_yh,
                n: PU,
                ctu: 64,
                ctus_x: yw_stride / 64,
                min_pu: 4,
                neutral,
            },
        );
        let predict =
            |mode: u8, dst: &mut [u16; 1024], angular: &mut intra::AngularScratch| match mode {
                0 => intra::predict_planar_into(&above, &left, PU, dst),
                1 => intra::predict_dc_into(&above, &left, PU, true, dst),
                _ => intra::predict_angular_into(
                    corner,
                    &above,
                    &left,
                    PU,
                    mode,
                    true,
                    max_val as i32,
                    dst,
                    angular,
                ),
            };

        let mut rmd = [IntraModeCandidate {
            mode: 0,
            cost: f32::MAX,
        }; 8];
        let mut mode_costs = [f32::MAX; 35];
        let mut tested = [false; 35];
        let lambda_mode = lambda.sqrt();
        let mut test_mode = |mode: u8| -> Option<f32> {
            let index = mode as usize;
            if tested[index] {
                return None;
            }
            tested[index] = true;
            predict(mode, &mut scratch.pred, &mut scratch.angular);
            let cost = scratch.satd(&scratch.orig[..PU_LEN], &scratch.pred[..PU_LEN], PU) as f32
                + lambda_mode * estimated_luma_mode_bins(mode, &mpm) as f32;
            mode_costs[index] = cost;
            update_intra_candidate(&mut rmd, mode, cost);
            Some(cost)
        };

        // Four 4×4 PUs are reached only behind the conservative NxN gate, but a
        // full 35-direction RMD for each child would still dominate those CUs.
        // Sample the angular space at four-mode intervals, then refine ±1/±2
        // around the best coarse direction and always include every MPM.
        static COARSE_MODES: [u8; 11] = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34];
        let mut coarse_mode = 0u8;
        let mut coarse_cost = f32::MAX;
        for mode in COARSE_MODES {
            if let Some(cost) = test_mode(mode)
                && cost < coarse_cost
            {
                coarse_cost = cost;
                coarse_mode = mode;
            }
        }
        if coarse_mode >= 2 {
            for delta in [-2i16, -1, 1, 2] {
                let mode = coarse_mode as i16 + delta;
                if (2..=34).contains(&mode) {
                    let _ = test_mode(mode as u8);
                }
            }
        }
        for &mode in &mpm {
            let _ = test_mode(mode);
        }

        let mut candidates = [IntraModeCandidate {
            mode: 0,
            cost: f32::MAX,
        }; 11];
        let mut candidate_count = 0usize;
        for candidate in rmd {
            push_sorted_unique_candidate(&mut candidates, &mut candidate_count, candidate);
        }
        for mode in mpm {
            push_sorted_unique_candidate(
                &mut candidates,
                &mut candidate_count,
                IntraModeCandidate {
                    mode,
                    cost: mode_costs[mode as usize],
                },
            );
        }
        let full_count = full_rdo_candidate_count(&candidates[..candidate_count], PU);
        let mut best_mode = candidates[0].mode;
        let mut best_cost = f32::MAX;

        for candidate in &candidates[..full_count] {
            let mode = candidate.mode;
            let scan_idx = dct::scan_idx_for(mode, 2, true, false);
            let scan = dct::coeff_scan(2, scan_idx);
            let mut trial_ctx = rdoq_ctx.clone();
            let mut trial_ictx = ictx.clone();
            let mut rate = estimate_luma_mode_bits(&mut trial_ictx, mode, &mpm);
            predict(mode, &mut scratch.pred, &mut scratch.angular);
            intra::compute_residual_i32_into(
                &scratch.orig[..PU_LEN],
                &scratch.pred[..PU_LEN],
                PU,
                &mut scratch.residual,
            );
            crate::hevc_transform::run_fwd_transform(
                scratch.fwd_transform,
                &scratch.residual[..PU_LEN],
                PU,
                bit_depth.bits(),
                &mut scratch.coeff,
                &mut scratch.transform_tmp,
                true,
            );
            crate::hevc_transform::quantize_with_sign_hiding_into(
                &scratch.coeff,
                PU,
                qp,
                bit_depth.bits(),
                scan,
                &mut scratch.levels,
            );
            let mut nonzero = false;
            for (dst, &(scan_row, scan_col)) in scratch.scanned[..PU_LEN].iter_mut().zip(scan) {
                let level = scratch.levels[scan_row * PU + scan_col];
                *dst = level;
                nonzero |= level != 0;
            }
            rate += trial_ctx.cbf_luma[0].estimate_and_update(nonzero as u8);
            if nonzero {
                rate += estimate_residual_bits(
                    &mut trial_ctx,
                    &scratch.scanned[..PU_LEN],
                    2,
                    true,
                    scan_idx,
                    true,
                );
            }
            let rate_cost = lambda * rate;
            if rate_cost >= best_cost {
                continue;
            }
            crate::hevc_transform::run_dequantize(
                scratch.dequantize,
                &scratch.levels,
                PU,
                qp,
                bit_depth.bits(),
                &mut scratch.dequant,
            );
            crate::hevc_transform::run_inv_transform(
                scratch.inv_transform,
                &scratch.dequant,
                PU,
                bit_depth.bits(),
                &mut scratch.inverse,
                &mut scratch.transform_tmp,
                true,
            );
            intra::reconstruct_into(
                &scratch.pred[..PU_LEN],
                &scratch.inverse[..PU_LEN],
                PU,
                max_val,
                &mut scratch.reconstructed,
            );
            let cost = block_sse(
                &scratch.orig[..PU_LEN],
                &scratch.reconstructed[..PU_LEN],
                PU,
            ) + rate_cost;
            if cost < best_cost {
                best_cost = cost;
                best_mode = mode;
                core::mem::swap(&mut scratch.pred, &mut scratch.best_pred);
                core::mem::swap(&mut scratch.coeff, &mut scratch.best_coeff);
            }
        }

        // Syntax is emitted after the loop; here only record the decision so the
        // next PU's MPM derivation sees this PU's reconstructed mode.
        mode_map[(row / 4) * mode_stride + col / 4] = best_mode;
        luma_modes[pu_index] = best_mode;
        pu_mpm[pu_index] = mpm;
        nxn_cost += best_cost;

        let scan_idx = dct::scan_idx_for(best_mode, 2, true, false);
        let scan = dct::coeff_scan(2, scan_idx);
        let tb = crate::hevc_transform::RdoqTb {
            coeff: &scratch.best_coeff,
            n: PU,
            qp,
            bit_depth: bit_depth.bits(),
            scan,
            scan_idx,
            lambda,
        };
        crate::hevc_transform::rdoq_luma_at_depth_with_sign_hiding_into(
            &tb,
            1,
            &rdoq_ctx,
            &mut scratch.levels,
            &mut scratch.rdoq,
        );
        let offset = pu_index * PU_LEN;
        let packed = &mut scratch.tu_tree.y_zz[offset..offset + PU_LEN];
        let mut nonzero = false;
        for (dst, &(scan_row, scan_col)) in packed.iter_mut().zip(scan) {
            let level = scratch.levels[scan_row * PU + scan_col];
            *dst = level;
            nonzero |= level != 0;
        }
        scratch.tu_tree.y_nz[pu_index] = nonzero;
        scratch.tu_tree.y_scan_idx[pu_index] = scan_idx;
        let _ = rdoq_ctx.cbf_luma[0].estimate_and_update(nonzero as u8);
        if nonzero {
            advance_residual_contexts(&mut rdoq_ctx, packed, 2, true, scan_idx, true);
        }

        crate::hevc_transform::run_dequantize(
            scratch.dequantize,
            &scratch.levels,
            PU,
            qp,
            bit_depth.bits(),
            &mut scratch.dequant,
        );
        crate::hevc_transform::run_inv_transform(
            scratch.inv_transform,
            &scratch.dequant,
            PU,
            bit_depth.bits(),
            &mut scratch.inverse,
            &mut scratch.transform_tmp,
            true,
        );
        intra::reconstruct_into(
            &scratch.best_pred[..PU_LEN],
            &scratch.inverse[..PU_LEN],
            PU,
            max_val,
            &mut scratch.reconstructed,
        );
        let dst_start = row * yw_stride + col;
        for (src_row, dst_row) in scratch.reconstructed[..PU_LEN]
            .as_chunks::<PU>()
            .0
            .iter()
            .zip(rec_y[dst_start..].chunks_mut(yw_stride))
        {
            dst_row[..PU].copy_from_slice(src_row);
        }
    }

    // Real RD decision: keep PART_NxN only if its four-PU luma cost beats the
    // 2Nx2N winner. Nothing has been written to the bitstream yet, so bailing out
    // here simply hands control back to the 2Nx2N commit path.
    if nxn_cost >= challenger_cost {
        return false;
    }

    // min-CU intra part_mode: zero means PART_NxN. The root transform split is
    // then inferred by HEVC and is not separately CABAC-coded.
    enc.encode_bin(0, &mut ictx.part_mode);

    // HEVC §7.3.8.5: for PART_NxN all four prev_intra_luma_pred_flag bins are
    // coded first, then all four mpm_idx/rem_intra_luma_pred_mode bins.
    for pu in 0..4 {
        encode_luma_mode_flag(enc, ictx, luma_modes[pu], &pu_mpm[pu]);
    }
    for pu in 0..4 {
        encode_luma_mode_rem(enc, luma_modes[pu], &pu_mpm[pu]);
    }

    let shared_chroma = !chroma.is_monochrome() && split_chroma_is_shared(8, chroma);
    if matches!(chroma, crate::fmt::ChromaFormat::Yuv444) {
        encode_nxn_chroma_444(
            enc,
            ictx,
            scratch,
            NxNChroma444 {
                src_cb,
                src_cr,
                rec_cb,
                rec_cr,
                src_cw,
                src_ch,
                cw_stride,
                coded_ch_h,
                lu_row,
                lu_col,
                yw_stride,
                coded_yh,
                qp_slice,
                bit_depth,
                lambda,
                luma_modes,
                residual_ctx_after_luma: &rdoq_ctx,
            },
        );
    } else if !chroma.is_monochrome() {
        debug_assert!(shared_chroma);
        let chroma_qp = chroma_qp_for(qp_slice, chroma) + bit_depth.qp_bd_offset();
        let chroma_lambda = lambda * chroma_lambda_scale(qp_slice, chroma);
        let candidates = chroma_mode_candidates(luma_modes[0], chroma);
        let mut ranked = [ChromaModeCandidate {
            pred_mode: 0,
            syntax_idx: 0,
            cost: f32::MAX,
        }; 5];
        let side = 8 / chroma.sub_w();
        let stacked = chroma.chroma_tbs_per_cu();
        let block_len = side * side;
        let ch_row = lu_row / chroma.sub_h();
        let ch_col = lu_col / chroma.sub_w();

        for mut candidate in candidates {
            let mut cost =
                chroma_lambda.sqrt() * estimated_chroma_mode_bins(candidate.syntax_idx) as f32;
            for stack_index in 0..stacked {
                let sub_ch_row = ch_row + stack_index * side;
                let ((bc0, ba, bl), (rc0, ra, rl)) = intra::get_reference_samples_chroma_pair(
                    rec_cb,
                    rec_cr,
                    intra::ChromaRefGeometry {
                        stride: cw_stride,
                        block_row: sub_ch_row,
                        block_col: ch_col,
                        chroma_h: coded_ch_h,
                        n: side,
                        sub_w: chroma.sub_w(),
                        sub_h: chroma.sub_h(),
                        luma_w: yw_stride,
                        luma_h: coded_yh,
                        luma_ctus_x: yw_stride / 64,
                        min_luma_pu: 8,
                        // 4:2:2 stacks two square TBs; anchor each at its own luma row.
                        cur_luma_row: lu_row + stack_index * side * chroma.sub_h(),
                        cur_luma_col: lu_col,
                        neutral,
                    },
                );
                let satd = scratch.satd;
                for component in 0..2 {
                    let (source_plane, corner, above, left, orig) = if component == 0 {
                        (src_cb, bc0, &ba[..], &bl[..], &mut scratch.chroma_orig_cb)
                    } else {
                        (src_cr, rc0, &ra[..], &rl[..], &mut scratch.chroma_orig_cr)
                    };
                    extract_block_dyn_into(
                        source_plane,
                        src_cw,
                        src_ch,
                        sub_ch_row,
                        ch_col,
                        side,
                        &mut orig[..block_len],
                    );
                    intra::predict_chroma_tb_into(
                        candidate.pred_mode,
                        corner,
                        above,
                        left,
                        side,
                        max_val as i32,
                        &mut scratch.pred,
                        &mut scratch.angular,
                    );
                    // SAFETY: both buffers contain a complete supported square block.
                    cost += unsafe { satd(&orig[..block_len], &scratch.pred[..block_len], side) }
                        as f32;
                }
            }
            candidate.cost = cost;
            update_chroma_candidate(&mut ranked, candidate);
        }

        let exact_count = full_rdo_chroma_count(&ranked, chroma);
        let mut best = ranked[0];
        if exact_count > 1 {
            let mut best_cost = f32::MAX;
            for &candidate in &ranked[..exact_count] {
                let cost = evaluate_chroma_mode(
                    ChromaEvalOpts {
                        candidate,
                        estimate_rate: true,
                        winner_rdoq: false,
                        cost_limit: best_cost,
                    },
                    scratch,
                    ChromaPlanes {
                        src_cb,
                        src_cr,
                        rec_cb: &mut *rec_cb,
                        rec_cr: &mut *rec_cr,
                    },
                    ChromaSplitGeom {
                        src_cw,
                        src_ch,
                        cw_stride,
                        coded_ch_h,
                        lu_row,
                        lu_col,
                        parent_luma: 8,
                        yw_stride,
                        coded_yh,
                    },
                    ChromaCoding {
                        chroma,
                        chroma_qp,
                        bit_depth: bit_depth.bits(),
                        max_val,
                        lambda: chroma_lambda,
                    },
                    0,
                    &rdoq_ctx,
                    ictx,
                );
                if cost < best_cost {
                    best_cost = cost;
                    best = candidate;
                }
            }
        }

        let _ = evaluate_chroma_mode(
            ChromaEvalOpts {
                candidate: best,
                estimate_rate: false,
                winner_rdoq: true,
                cost_limit: f32::MAX,
            },
            scratch,
            ChromaPlanes {
                src_cb,
                src_cr,
                rec_cb: &mut *rec_cb,
                rec_cr: &mut *rec_cr,
            },
            ChromaSplitGeom {
                src_cw,
                src_ch,
                cw_stride,
                coded_ch_h,
                lu_row,
                lu_col,
                parent_luma: 8,
                yw_stride,
                coded_yh,
            },
            ChromaCoding {
                chroma,
                chroma_qp,
                bit_depth: bit_depth.bits(),
                max_val,
                lambda: chroma_lambda,
            },
            0,
            &rdoq_ctx,
            ictx,
        );
        let scan_idx = dct::scan_idx_for(best.pred_mode, side.trailing_zeros(), false, false);
        for index in 0..stacked {
            scratch.tu_tree.chroma_scan_idx[index] = scan_idx;
        }
        encode_chroma_mode(enc, ictx, best.syntax_idx);
    }

    encode_split_transform_tree(enc, ctx, scratch, 8, chroma, true, shared_chroma, true, aq);
    true
}

#[allow(clippy::too_many_arguments)]
fn encode_split_transform_tree<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    scratch: &CompressionContext,
    parent_luma: usize,
    chroma: crate::fmt::ChromaFormat,
    inferred_root_split: bool,
    shared_chroma: bool,
    sign_data_hiding: bool,
    aq: &mut AqCtuState,
) {
    if !inferred_root_split {
        let split_ctx = split_transform_context(parent_luma);
        enc.encode_bin(1, &mut ctx.split_transform_flag[split_ctx]);
    }

    let child = parent_luma / 2;
    let child_len = child * child;
    let child_log2 = child.trailing_zeros();
    let stacked = chroma.chroma_tbs_per_cu();

    // 4:2:2 represents each rectangular chroma TU as two vertically stacked
    // square TUs. Consequently it has two root CBFs, one for the upper pair of
    // luma children and one for the lower pair. Other chroma formats have one.
    let root_count = if matches!(chroma, crate::fmt::ChromaFormat::Yuv422) && parent_luma == 8 {
        2
    } else {
        1
    };
    let mut root_cb_nz = [false; 2];
    let mut root_cr_nz = [false; 2];
    if !chroma.is_monochrome() {
        if shared_chroma {
            for root_index in 0..root_count {
                root_cb_nz[root_index] = scratch.chroma_tbs[root_index].cb_nz;
                root_cr_nz[root_index] = scratch.chroma_tbs[root_index].cr_nz;
            }
        } else if root_count == 2 {
            for root_index in 0..2 {
                for child_index in root_index * 2..root_index * 2 + 2 {
                    for stack_index in 0..stacked {
                        let index = child_index * stacked + stack_index;
                        root_cb_nz[root_index] |= scratch.tu_tree.cb_nz[index];
                        root_cr_nz[root_index] |= scratch.tu_tree.cr_nz[index];
                    }
                }
            }
        } else {
            root_cb_nz[0] = scratch.tu_tree.cb_nz[..4 * stacked]
                .iter()
                .any(|&nonzero| nonzero);
            root_cr_nz[0] = scratch.tu_tree.cr_nz[..4 * stacked]
                .iter()
                .any(|&nonzero| nonzero);
        }

        for &nonzero in &root_cb_nz[..root_count] {
            encode_cbf_chroma(enc, ctx, nonzero, 0);
        }
        for &nonzero in &root_cr_nz[..root_count] {
            encode_cbf_chroma(enc, ctx, nonzero, 0);
        }
    }

    for child_index in 0..4 {
        // The SPS permits recursion to depth three. This bounded one-level tree
        // deliberately terminates each non-minimum child here.
        if child > 4 {
            let split_ctx = split_transform_context(child);
            enc.encode_bin(0, &mut ctx.split_transform_flag[split_ctx]);
        }

        if !chroma.is_monochrome() && !shared_chroma {
            let root_index = if root_count == 2 { child_index / 2 } else { 0 };
            if root_cb_nz[root_index] {
                for stack_index in 0..stacked {
                    let index = child_index * stacked + stack_index;
                    encode_cbf_chroma(enc, ctx, scratch.tu_tree.cb_nz[index], 1);
                }
            }
            if root_cr_nz[root_index] {
                for stack_index in 0..stacked {
                    let index = child_index * stacked + stack_index;
                    encode_cbf_chroma(enc, ctx, scratch.tu_tree.cr_nz[index], 1);
                }
            }
        }

        let y_nz = scratch.tu_tree.y_nz[child_index];
        encode_cbf_luma(enc, ctx, y_nz, 1);
        let mut need_qp = y_nz;
        if !chroma.is_monochrome() {
            if shared_chroma {
                if child_index == 3 {
                    need_qp = need_qp
                        || scratch.chroma_tbs[..stacked]
                            .iter()
                            .any(|tb| tb.cb_nz || tb.cr_nz);
                }
            } else {
                for stack_index in 0..stacked {
                    let index = child_index * stacked + stack_index;
                    need_qp =
                        need_qp || scratch.tu_tree.cb_nz[index] || scratch.tu_tree.cr_nz[index];
                }
            }
        }
        encode_cu_qp_delta_if_needed(enc, ctx, aq, need_qp);
        if y_nz {
            let offset = child_index * child_len;
            encode_residual(
                enc,
                ctx,
                &scratch.tu_tree.y_zz[offset..offset + child_len],
                child_log2,
                true,
                scratch.tu_tree.y_scan_idx[child_index],
                sign_data_hiding,
            );
        }

        if chroma.is_monochrome() {
            continue;
        }
        if shared_chroma {
            // When four 4x4 luma leaves share the parent chroma TU, HEVC emits
            // chroma once after the last luma leaf.  For 4:2:2 that one chroma
            // unit contains both stacked TBs, in component-major order.
            if child_index != 3 {
                continue;
            }
            let side = parent_luma / chroma.sub_w();
            let len = side * side;
            let log2_side = side.trailing_zeros();
            for stack_index in 0..stacked {
                let tb = &scratch.chroma_tbs[stack_index];
                if tb.cb_nz {
                    encode_residual(
                        enc,
                        ctx,
                        &tb.cb_zz[..len],
                        log2_side,
                        false,
                        scratch.tu_tree.chroma_scan_idx[stack_index],
                        sign_data_hiding,
                    );
                }
            }
            for stack_index in 0..stacked {
                let tb = &scratch.chroma_tbs[stack_index];
                if tb.cr_nz {
                    encode_residual(
                        enc,
                        ctx,
                        &tb.cr_zz[..len],
                        log2_side,
                        false,
                        scratch.tu_tree.chroma_scan_idx[stack_index],
                        sign_data_hiding,
                    );
                }
            }
            continue;
        }

        let side = child / chroma.sub_w();
        let len = side * side;
        let log2_side = side.trailing_zeros();
        for stack_index in 0..stacked {
            let index = child_index * stacked + stack_index;
            if scratch.tu_tree.cb_nz[index] {
                let offset = index * len;
                encode_residual(
                    enc,
                    ctx,
                    &scratch.tu_tree.cb_zz[offset..offset + len],
                    log2_side,
                    false,
                    scratch.tu_tree.chroma_scan_idx[index],
                    sign_data_hiding,
                );
            }
        }
        for stack_index in 0..stacked {
            let index = child_index * stacked + stack_index;
            if scratch.tu_tree.cr_nz[index] {
                let offset = index * len;
                encode_residual(
                    enc,
                    ctx,
                    &scratch.tu_tree.cr_zz[offset..offset + len],
                    log2_side,
                    false,
                    scratch.tu_tree.chroma_scan_idx[index],
                    sign_data_hiding,
                );
            }
        }
    }
}

fn encode_lossless_chroma_residuals<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    tree: &TuTreeScratch,
    node: usize,
    chroma: crate::fmt::ChromaFormat,
) {
    let side = tree.node_chroma_side[node] as usize;
    if side == 0 {
        return;
    }
    let len = side * side;
    let log2 = side.trailing_zeros();
    let stacked = chroma.chroma_tbs_per_cu();
    for t in 0..stacked {
        if tree.node_cb_nz[node][t] {
            let offset = tree.node_chroma_offset[node][t] as usize;
            encode_residual(
                enc,
                ctx,
                &tree.cb_zz[offset..offset + len],
                log2,
                false,
                tree.node_chroma_scan_idx[node],
                false,
            );
        }
    }
    for t in 0..stacked {
        if tree.node_cr_nz[node][t] {
            let offset = tree.node_chroma_offset[node][t] as usize;
            encode_residual(
                enc,
                ctx,
                &tree.cr_zz[offset..offset + len],
                log2,
                false,
                tree.node_chroma_scan_idx[node],
                false,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_lossless_transform_node<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    tree: &TuTreeScratch,
    node: usize,
    parent_node: usize,
    size: usize,
    depth: usize,
    quadrant: usize,
    coeff_offset: usize,
    chroma: crate::fmt::ChromaFormat,
    parent_cb: [bool; 2],
    parent_cr: [bool; 2],
) {
    let split = size > 4 && tree.split[node];
    if size > 4 {
        enc.encode_bin(
            split as u8,
            &mut ctx.split_transform_flag[split_transform_context(size)],
        );
    }

    let mut cb = parent_cb;
    let mut cr = parent_cr;
    if !chroma.is_monochrome() && (size > 4 || matches!(chroma, crate::fmt::ChromaFormat::Yuv444)) {
        cb = tree.node_cb_nz[node];
        cr = tree.node_cr_nz[node];
        let stacked = chroma.chroma_tbs_per_cu();
        let signal_second =
            matches!(chroma, crate::fmt::ChromaFormat::Yuv422) && (!split || size == 8);
        for t in 0..stacked {
            if t == 1 && !signal_second {
                cb[1] = cb[0];
                break;
            }
            if depth == 0 || parent_cb[t] {
                encode_cbf_chroma(enc, ctx, cb[t], depth);
            }
        }
        for t in 0..stacked {
            if t == 1 && !signal_second {
                cr[1] = cr[0];
                break;
            }
            if depth == 0 || parent_cr[t] {
                encode_cbf_chroma(enc, ctx, cr[t], depth);
            }
        }
    }

    if split {
        let child = size / 2;
        let child_len = child * child;
        for child_index in 0..4 {
            encode_lossless_transform_node(
                enc,
                ctx,
                tree,
                node * 4 + 1 + child_index,
                node,
                child,
                depth + 1,
                child_index,
                coeff_offset + child_index * child_len,
                chroma,
                cb,
                cr,
            );
        }
        return;
    }

    let y_nz = tree.node_y_nz[node];
    encode_cbf_luma(enc, ctx, y_nz, depth);
    if y_nz {
        encode_residual(
            enc,
            ctx,
            &tree.y_zz[coeff_offset..coeff_offset + size * size],
            size.trailing_zeros(),
            true,
            tree.node_y_scan_idx[node],
            false,
        );
    }
    if chroma.is_monochrome() {
        return;
    }
    if size > 4 || matches!(chroma, crate::fmt::ChromaFormat::Yuv444) {
        encode_lossless_chroma_residuals(enc, ctx, tree, node, chroma);
    } else if quadrant == 3 {
        encode_lossless_chroma_residuals(enc, ctx, tree, parent_node, chroma);
    }
}

fn encode_lossless_transform_tree<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    scratch: &CompressionContext,
    size: usize,
    chroma: crate::fmt::ChromaFormat,
) {
    encode_lossless_transform_node(
        enc,
        ctx,
        &scratch.tu_tree,
        0,
        0,
        size,
        0,
        0,
        0,
        chroma,
        [false; 2],
        [false; 2],
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_cu<W: CabacWriter>(
    ent: Entropy<'_, W>,
    src: &CuSrcPlanes<'_>,
    rec: &mut CuRecPlanes<'_>,
    geo: &CuGeometry,
    par: &CuParams,
    mode_map: &mut [u8],
    aq: &mut AqCtuState,
    scratch: &mut CompressionContext,
) {
    scratch.last_tu_layout = TuLayout::Unsplit;
    // destructure so the rest of the body is unchanged
    let Entropy { enc, ctx, ictx } = ent;
    let CuGeometry {
        lu_row,
        lu_col,
        ch_row,
        ch_col,
        yw_stride,
        src_yh,
        cw_stride,
        src_cw,
        src_ch,
        mode_stride,
    } = *geo;
    let CuSrcPlanes {
        y: src_y,
        cb: src_cb,
        cr: src_cr,
        src_yw,
    } = *src;
    let CuRecPlanes {
        y: rec_y,
        cb: rec_cb,
        cr: rec_cr,
    } = rec;
    let CuParams {
        qp,
        chroma,
        bit_depth,
        lu,
        lambda,
        lossless,
    } = *par;
    let neutral: u16 = bit_depth.neutral(); // 128 (8-bit) / 512 (10-bit)
    let max_val: u16 = bit_depth.max_val(); // 255 / 1023
    // HEVC §8.6.1: the decoder dequantizes at Qp' = Qp + QpBdOffset, where
    // QpBdOffset = 6*(bitDepth-8). The signaled slice/PPS QP stays the user QP; the
    // encoder must quantize AND dequantize at this same Qp' so its bitstream matches
    // the decoder's interpretation. Chroma derives its table mapping from the
    // un-offset luma QP, then adds the offset.
    let qp_bd_offset = bit_depth.qp_bd_offset();
    let qp_slice = qp; // user/slice QP (no bd offset)
    let qp = qp_slice + qp_bd_offset; // luma Qp' used for transform quant/dequant
    let n_chroma_tb = chroma.chroma_tbs_per_cu();
    let coded_yh = rec_y.len() / yw_stride;
    let coded_ch_h = if cw_stride > 0 {
        rec_cb.len() / cw_stride.max(1)
    } else {
        0
    };

    // ── Luma intra prediction + mode decision ────────────────────────────────
    const PLANAR: u8 = 0;
    const DC: u8 = 1;
    // MPM candidates from neighbor modes (HEVC §8.4.2): candA = left, candB =
    // above (DC if unavailable or in a different CTB row). Modes come from the
    // per-block mode map written by previously coded CUs.
    let ctb = 64usize;
    let avail_left =
        lu_col > 0 && is_block_decoded(lu_row, lu_col - 1, lu_row, lu_col, ctb, yw_stride);
    let above_in_same_ctb = lu_row > 0 && ((lu_row - 1) >= (lu_row / ctb) * ctb);
    let avail_above = lu_row > 0
        && above_in_same_ctb
        && is_block_decoded(lu_row - 1, lu_col, lu_row, lu_col, ctb, yw_stride);
    let mode_at = |r: usize, c: usize| mode_map[(r / 4) * mode_stride + c / 4];
    let cand_a = if avail_left {
        mode_at(lu_row, lu_col - 1)
    } else {
        DC
    };
    let cand_b = if avail_above {
        mode_at(lu_row - 1, lu_col)
    } else {
        DC
    };
    let mpm = mpm_list(cand_a, cand_b);

    let (yc0, ya, yl) = intra::get_reference_samples(
        rec_y,
        intra::LumaRefGeometry {
            stride: yw_stride,
            block_row: lu_row,
            block_col: lu_col,
            height: coded_yh,
            n: lu,
            ctu: 64,
            ctus_x: yw_stride / 64,
            min_pu: 8,
            neutral,
        },
    );

    // Two-stage intra mode decision:
    //   1. rank all 35 modes with SATD + sqrt(lambda) * estimated mode bins;
    //   2. reconstruction-RDO an adaptively bounded subset of the HM-style
    //      shortlist using fractional CABAC costs, not the real arithmetic coder.
    //
    // The selected winner alone enters RDOQ. Its prediction, residual, and forward
    // transform are cached here so committing the CU does not repeat that work.
    let num_luma = lu * lu;
    match lu {
        32 => extract_block_n_into::<32>(src_y, src_yw, src_yh, lu_row, lu_col, &mut scratch.orig),
        16 => extract_block_n_into::<16>(src_y, src_yw, src_yh, lu_row, lu_col, &mut scratch.orig),
        _ => extract_block_n_into::<8>(src_y, src_yw, src_yh, lu_row, lu_col, &mut scratch.orig),
    }
    let lambda_mode = lambda.sqrt();
    // The smoothed references depend only on the block, not the mode.
    let (fa, fl) = intra::filter_references(yc0, &ya, &yl, lu);
    let cf = ((ya[0] as i32 + 2 * yc0 as i32 + yl[0] as i32 + 2) >> 2) as u16;
    let predict_luma = |mode: u8, pred: &mut [u16; 1024], angular: &mut intra::AngularScratch| {
        let (corner, above, left) = if intra::should_filter_refs(mode, lu) {
            (cf, &fa[..], &fl[..])
        } else {
            (yc0, &ya[..], &yl[..])
        };
        match mode {
            PLANAR => intra::predict_planar_into(above, left, lu, pred),
            DC => intra::predict_dc_into(above, left, lu, true, pred),
            _ => intra::predict_angular_into(
                corner,
                above,
                left,
                lu,
                mode,
                true,
                max_val as i32,
                pred,
                angular,
            ),
        }
    };

    const MAX_RMD_MODES: usize = 8;
    // 8 RMD modes + up to 3 missing MPMs + the two implicit-RDPCM modes.
    const MAX_RD_MODES: usize = 13;
    let fast_mode_count = if lu == 8 { 8 } else { 3 };
    let mut rmd = [IntraModeCandidate {
        mode: PLANAR,
        cost: f32::MAX,
    }; MAX_RMD_MODES];
    let mut mode_costs = [f32::MAX; 35];

    // Every mode is visited exactly once, so avoid the tested-mode bitmap and
    // closure dispatch from the previous implementation.
    for mode in 0u8..35 {
        predict_luma(mode, &mut scratch.pred, &mut scratch.angular);
        let satd = scratch.satd(&scratch.orig[..num_luma], &scratch.pred[..num_luma], lu) as f32;
        let cost = satd + lambda_mode * estimated_luma_mode_bins(mode, &mpm) as f32;
        mode_costs[mode as usize] = cost;
        update_intra_candidate(&mut rmd[..fast_mode_count], mode, cost);
    }

    let mut rd_candidates = [IntraModeCandidate {
        mode: PLANAR,
        cost: f32::MAX,
    }; MAX_RD_MODES];
    let mut rd_mode_count = 0usize;
    for &candidate in &rmd[..fast_mode_count] {
        push_sorted_unique_candidate(&mut rd_candidates, &mut rd_mode_count, candidate);
    }
    for &mode in &mpm {
        push_sorted_unique_candidate(
            &mut rd_candidates,
            &mut rd_mode_count,
            IntraModeCandidate {
                mode,
                cost: mode_costs[mode as usize],
            },
        );
    }
    if lossless {
        // SATD does not model the second prediction stage introduced by implicit
        // RDPCM. Pure horizontal/vertical can therefore be poor SATD candidates
        // but excellent entropy candidates. Always retain both for exact rate RDO.
        for mode in [10u8, 26] {
            push_sorted_unique_candidate(
                &mut rd_candidates,
                &mut rd_mode_count,
                IntraModeCandidate {
                    mode,
                    cost: mode_costs[mode as usize],
                },
            );
        }
    }
    let mut full_rd_count = full_rdo_candidate_count(&rd_candidates[..rd_mode_count], lu);
    if lossless {
        // Move the two inferred-RDPCM modes into the evaluated prefix without
        // increasing the normal lossy candidate budget.
        for mode in [10u8, 26] {
            if rd_candidates[..full_rd_count]
                .iter()
                .any(|candidate| candidate.mode == mode)
            {
                continue;
            }
            if let Some(index) = rd_candidates[..rd_mode_count]
                .iter()
                .position(|candidate| candidate.mode == mode)
            {
                rd_candidates.swap(full_rd_count, index);
                full_rd_count += 1;
            }
        }
    }
    let luma_log2_ts = lu.trailing_zeros();
    let mut luma_mode = rd_candidates[0].mode;
    let mut best_rd_cost = f32::MAX;

    for candidate in &rd_candidates[..full_rd_count] {
        let mode = candidate.mode;
        let mut trial_ctx = ctx.clone();
        let mut trial_ictx = ictx.clone();
        let mut rate = 0.0f32;

        if lossless {
            rate += trial_ctx.cu_transquant_bypass_flag.estimate_and_update(1);
        }
        if lu == 8 {
            rate += trial_ictx.part_mode.estimate_and_update(1);
        }
        rate += estimate_luma_mode_bits(&mut trial_ictx, mode, &mpm);

        predict_luma(mode, &mut scratch.pred, &mut scratch.angular);
        intra::compute_residual_i32_into(
            &scratch.orig[..num_luma],
            &scratch.pred[..num_luma],
            lu,
            &mut scratch.residual,
        );
        let scan_idx = dct::scan_idx_for(mode, luma_log2_ts, true, false);
        let scan = dct::coeff_scan(luma_log2_ts, scan_idx);
        if lossless {
            forward_lossless_rdpcm_into(
                &scratch.residual[..num_luma],
                lu,
                implicit_rdpcm_mode(mode),
                &mut scratch.levels,
            );
        } else {
            crate::hevc_transform::run_fwd_transform(
                scratch.fwd_transform,
                &scratch.residual[..num_luma],
                lu,
                bit_depth.bits(),
                &mut scratch.coeff,
                &mut scratch.transform_tmp,
                false,
            );
            crate::hevc_transform::quantize_with_sign_hiding_into(
                &scratch.coeff,
                lu,
                qp,
                bit_depth.bits(),
                scan,
                &mut scratch.levels,
            );
        }
        let mut nonzero = false;
        for (dst, &(row, col)) in scratch.scanned[..num_luma].iter_mut().zip(scan) {
            let level = scratch.levels[row * lu + col];
            *dst = level;
            nonzero |= level != 0;
        }
        rate += trial_ctx.cbf_luma[1].estimate_and_update(nonzero as u8);
        if nonzero {
            rate += estimate_residual_bits(
                &mut trial_ctx,
                &scratch.scanned[..num_luma],
                luma_log2_ts,
                true,
                scan_idx,
                !lossless,
            );
        }

        let rate_cost = lambda * rate;
        // Distortion is non-negative. Once the complete syntax estimate alone
        // loses to the current winner, inverse transform and reconstruction cannot
        // recover the candidate. This is exact pruning, not a heuristic.
        if rate_cost >= best_rd_cost {
            continue;
        }
        let cost = if lossless {
            // Transquant bypass plus inverse RDPCM reconstructs the source exactly
            // for every prediction mode, so lossless mode RDO is a pure rate
            // decision. Avoid a redundant reconstruction and full-block SSE pass.
            rate_cost
        } else {
            crate::hevc_transform::run_dequantize(
                scratch.dequantize,
                &scratch.levels,
                lu,
                qp,
                bit_depth.bits(),
                &mut scratch.dequant,
            );
            crate::hevc_transform::run_inv_transform(
                scratch.inv_transform,
                &scratch.dequant,
                lu,
                bit_depth.bits(),
                &mut scratch.inverse,
                &mut scratch.transform_tmp,
                false,
            );
            intra::reconstruct_into(
                &scratch.pred[..num_luma],
                &scratch.inverse[..num_luma],
                lu,
                max_val,
                &mut scratch.reconstructed,
            );
            block_sse(
                &scratch.orig[..num_luma],
                &scratch.reconstructed[..num_luma],
                lu,
            ) + rate_cost
        };
        if cost < best_rd_cost {
            best_rd_cost = cost;
            luma_mode = mode;
            // Current and best buffers form a two-slot cache. Swapping ownership
            // is constant-time and avoids copying 2–6 KiB every time the winner
            // changes; the old winner is simply overwritten by the next trial.
            core::mem::swap(&mut scratch.pred, &mut scratch.best_pred);
            if lossless {
                core::mem::swap(&mut scratch.residual, &mut scratch.best_residual);
            } else {
                core::mem::swap(&mut scratch.coeff, &mut scratch.best_coeff);
            }
        }
    }

    // PART_NxN is considered only for the minimum 8×8 CU and only after the
    // regular 2Nx2N winner is known. `choose_nxn_proxy` is a cheap gate that keeps
    // the expensive four-PU search off smooth/ordinary blocks; encode_cu_nxn then
    // makes the real rate–distortion decision, committing PART_NxN only when its
    // four-PU luma J beats the 2Nx2N winner (`best_rd_cost`) and otherwise leaving
    // the bitstream untouched so the 2Nx2N path below runs.
    // PART_NxN's inferred transform split reaches four 4×4 chroma TBs. For 4:2:2
    // those TBs use the stacked square layout that the split path does not yet
    // model, so NxN is likewise restricted to the square 4:2:0/4:4:4 chroma
    // formats (and monochrome).
    if lu == 8
        && !lossless
        && !matches!(chroma, crate::fmt::ChromaFormat::Yuv422)
        && choose_nxn_proxy(
            scratch.satd,
            &scratch.orig[..num_luma],
            &scratch.best_pred[..num_luma],
            lambda,
            bit_depth.bits(),
        )
    {
        // The NxN search reuses scratch.orig / best_pred / best_coeff, so snapshot
        // the 2Nx2N winner state and restore it if NxN loses the RD comparison.
        let mut saved_orig = [0u16; 64];
        let mut saved_pred = [0u16; 64];
        let mut saved_coeff = [0i32; 64];
        saved_orig.copy_from_slice(&scratch.orig[..num_luma]);
        saved_pred.copy_from_slice(&scratch.best_pred[..num_luma]);
        saved_coeff.copy_from_slice(&scratch.best_coeff[..num_luma]);
        if encode_cu_nxn(
            Entropy {
                enc: &mut *enc,
                ctx: &mut *ctx,
                ictx: &mut *ictx,
            },
            CuSrcPlanes {
                y: src_y,
                cb: src_cb,
                cr: src_cr,
                src_yw,
            },
            CuRecPlanes {
                y: rec_y,
                cb: rec_cb,
                cr: rec_cr,
            },
            NxnGeom {
                lu_row,
                lu_col,
                yw_stride,
                src_yh,
                cw_stride,
                src_cw,
                src_ch,
                coded_yh,
                coded_ch_h,
                mode_stride,
            },
            NxnParams {
                qp_slice,
                qp,
                chroma,
                bit_depth,
                lambda,
            },
            mode_map,
            best_rd_cost,
            aq,
            scratch,
        ) {
            return;
        }
        scratch.orig[..num_luma].copy_from_slice(&saved_orig);
        scratch.best_pred[..num_luma].copy_from_slice(&saved_pred);
        scratch.best_coeff[..num_luma].copy_from_slice(&saved_coeff);
    }

    // Every supported chroma format uses the same luma quadtree. 4:2:2 maps each
    // rectangular chroma region to two stacked square TBs; transquant-bypass
    // children use direct residuals and inferred RDPCM.
    let split_allowed = lu > 4;

    // ── cu_transquant_bypass_flag ────────────────────────────────────────────
    // Per HEVC §7.3.8.5 this is the first element of coding_unit(), present only
    // when the PPS sets transquant_bypass_enabled_flag (i.e. lossless coding).
    if lossless {
        enc.encode_bin(1, &mut ctx.cu_transquant_bypass_flag);
    }

    // ── part_mode ──────────────────────────────────────────────────────────
    // The regular path is PART_2Nx2N. PART_NxN is handled by the dedicated 8×8
    // path below, where four independent 4×4 prediction modes are signalled.
    if lu == 8 {
        enc.encode_bin(1, &mut ictx.part_mode);
    }

    // ── Luma intra pred mode syntax ──────────────────────────────────────────
    encode_luma_mode(enc, ictx, luma_mode, &mpm);

    // Record this CU's luma mode on the minimum-PU 4×4 grid.
    for br in 0..(lu / 4) {
        let row = (lu_row / 4 + br) * mode_stride + lu_col / 4;
        mode_map[row..row + lu / 4].fill(luma_mode);
    }

    let luma_log2_ts = lu.trailing_zeros();
    let luma_scan_idx = dct::scan_idx_for(luma_mode, luma_log2_ts, true, false);
    let luma_scan = dct::coeff_scan(luma_log2_ts, luma_scan_idx);
    // ── Unsplit luma: reconstruct into scratch (rec_y written only if it wins) ──
    if lossless {
        forward_lossless_rdpcm_into(
            &scratch.best_residual[..num_luma],
            lu,
            implicit_rdpcm_mode(luma_mode),
            &mut scratch.levels,
        );
    } else {
        let tb = crate::hevc_transform::RdoqTb {
            coeff: &scratch.best_coeff,
            n: lu,
            qp,
            bit_depth: bit_depth.bits(),
            scan: luma_scan,
            scan_idx: luma_scan_idx,
            lambda,
        };
        crate::hevc_transform::rdoq_luma_with_sign_hiding_into(
            &tb,
            ctx,
            &mut scratch.levels,
            &mut scratch.rdoq,
        );
    }
    let mut y_nz_unsplit = false;
    for (dst, &(row, col)) in scratch.scanned[..num_luma].iter_mut().zip(luma_scan) {
        let level = scratch.levels[row * lu + col];
        *dst = level;
        y_nz_unsplit |= level != 0;
    }
    if lossless {
        for (dst, &residual) in scratch.inverse[..num_luma]
            .iter_mut()
            .zip(&scratch.best_residual[..num_luma])
        {
            *dst = residual;
        }
    } else {
        crate::hevc_transform::run_dequantize(
            scratch.dequantize,
            &scratch.levels,
            lu,
            qp,
            bit_depth.bits(),
            &mut scratch.dequant,
        );
        crate::hevc_transform::run_inv_transform(
            scratch.inv_transform,
            &scratch.dequant,
            lu,
            bit_depth.bits(),
            &mut scratch.inverse,
            &mut scratch.transform_tmp,
            false,
        );
    }
    intra::reconstruct_into(
        &scratch.best_pred[..num_luma],
        &scratch.inverse[..num_luma],
        lu,
        max_val,
        &mut scratch.reconstructed,
    );
    let d_unsplit = block_sse(
        &scratch.orig[..num_luma],
        &scratch.reconstructed[..num_luma],
        lu,
    );
    let r_unsplit = {
        let mut tctx = ctx.clone();
        let mut r = tctx.split_transform_flag[split_transform_context(lu)].estimate_and_update(0);
        r += tctx.cbf_luma[1].estimate_and_update(y_nz_unsplit as u8);
        if y_nz_unsplit {
            r += estimate_residual_bits(
                &mut tctx,
                &scratch.scanned[..num_luma],
                luma_log2_ts,
                true,
                luma_scan_idx,
                !lossless,
            );
        }
        r
    };

    // ── Split luma (if allowed): reconstruct into rec_y + tu_tree, cost it ──────
    // commit_split_luma leaves scratch.reconstructed / scratch.scanned untouched,
    // so the unsplit candidate above survives for a possible rollback.
    let (tu_layout, y_nz) = if lossless {
        scratch.tu_tree.split.fill(false);
        scratch.tu_tree.node_y_nz.fill(false);
        scratch.tu_tree.node_y_scan_idx.fill(0);
        let (_, _, nonzero) = choose_lossless_luma_tree(
            scratch, rec_y, yw_stride, coded_yh, lu_row, lu_col, lu, 0, 0, lu, luma_mode, neutral,
            max_val, 0, 0, 0, ctx,
        );
        (TuLayout::Recursive, nonzero)
    } else if split_allowed {
        let y_nz_split = commit_split_luma(
            scratch,
            rec_y,
            LumaSplitGeom {
                stride: yw_stride,
                coded_yh,
                block_row: lu_row,
                block_col: lu_col,
                parent: lu,
            },
            LumaTbCoding {
                mode: luma_mode,
                qp,
                bit_depth: bit_depth.bits(),
                max_val,
                neutral,
                lambda,
                lossless,
            },
            ctx,
        );
        let d_split = sse_plane(
            &scratch.orig[..num_luma],
            rec_y,
            lu_row,
            lu_col,
            yw_stride,
            lu,
        );
        let r_split = {
            let mut tctx = ctx.clone();
            let mut r =
                tctx.split_transform_flag[split_transform_context(lu)].estimate_and_update(1);
            let child = lu / 2;
            let child_len = child * child;
            let log2_child = child.trailing_zeros();
            for i in 0..4 {
                let nz = scratch.tu_tree.y_nz[i];
                r += tctx.cbf_luma[0].estimate_and_update(nz as u8);
                if nz {
                    let off = i * child_len;
                    r += estimate_residual_bits(
                        &mut tctx,
                        &scratch.tu_tree.y_zz[off..off + child_len],
                        log2_child,
                        true,
                        scratch.tu_tree.y_scan_idx[i],
                        !lossless,
                    );
                }
            }
            r
        };
        if d_split + lambda * r_split < d_unsplit + lambda * r_unsplit {
            (TuLayout::Split, y_nz_split)
        } else {
            // Unsplit wins: restore its reconstruction into the picture. Its
            // coefficients are still in scratch.scanned for the CABAC writer.
            for (src_row, dst_row) in scratch.reconstructed[..num_luma]
                .chunks_exact(lu)
                .zip(rec_y[lu_row * yw_stride + lu_col..].chunks_mut(yw_stride))
            {
                dst_row[..lu].copy_from_slice(src_row);
            }
            (TuLayout::Unsplit, y_nz_unsplit)
        }
    } else {
        for (src_row, dst_row) in scratch.reconstructed[..num_luma]
            .chunks_exact(lu)
            .zip(rec_y[lu_row * yw_stride + lu_col..].chunks_mut(yw_stride))
        {
            dst_row[..lu].copy_from_slice(src_row);
        }
        (TuLayout::Unsplit, y_nz_unsplit)
    };
    scratch.last_tu_layout = tu_layout;
    // ── Independent chroma intra-mode RDO ──────────────────────────────────
    // HEVC exposes four explicit chroma modes plus DM_CHROMA. As in HM, a
    // duplicate explicit mode is replaced by angular mode 34. All five modes are
    // ranked with prediction SATD and syntax rate. A clearly separated proxy
    // winner is committed directly; only ambiguous blocks run reconstruction +
    // fractional-CABAC RDO against one challenger. The chosen mode alone reaches
    // winner-only RDOQ and the real CABAC coder.
    let chroma_qp = chroma_qp_for(qp_slice, chroma) + qp_bd_offset;
    let chroma_lambda = lambda * chroma_lambda_scale(qp_slice, chroma);
    let sub_w = chroma.sub_w();
    let sub_h = chroma.sub_h();
    let luma_ctus_x = yw_stride / 64;
    let ctb = lu / sub_w; // chroma TB side: 4 through 32
    let log2_ctb = ctb.trailing_zeros();
    let is_444 = matches!(chroma, crate::fmt::ChromaFormat::Yuv444);
    let n_ch = ctb * ctb;

    let mut chroma_tb_scan_idx = 0u8;
    let shared_chroma = matches!(tu_layout, TuLayout::Split)
        && !chroma.is_monochrome()
        && split_chroma_is_shared(lu, chroma);
    let split_chroma_tree =
        matches!(tu_layout, TuLayout::Split) && !chroma.is_monochrome() && !shared_chroma;

    if !chroma.is_monochrome() {
        // Seed the not-yet-coded chroma region with source samples only for the
        // first-pass 4:2:2 proxy. The lower stacked TB then sees a realistic
        // above row without any transform work. Exact RDO immediately overwrites
        // the region with each candidate's reconstructed upper TB.
        if n_chroma_tb > 1 {
            let seed_upper_tb = |src_plane: &[u16], rec_plane: &mut [u16]| {
                for r in 0..ctb {
                    let sy = (ch_row + r).min(src_ch - 1);
                    let src_start = sy * src_cw + ch_col.min(src_cw - 1);
                    let available = src_cw.saturating_sub(ch_col).min(ctb);
                    let dst_start = (ch_row + r) * cw_stride + ch_col;
                    let dst = &mut rec_plane[dst_start..dst_start + ctb];
                    if available != 0 {
                        dst[..available]
                            .copy_from_slice(&src_plane[src_start..src_start + available]);
                        let last = dst[available - 1];
                        dst[available..].fill(last);
                    } else {
                        dst.fill(src_plane[sy * src_cw + src_cw - 1]);
                    }
                }
            };
            seed_upper_tb(src_cb, rec_cb);
            seed_upper_tb(src_cr, rec_cr);
        }

        debug_assert!(n_chroma_tb * n_ch <= 1024);
        // Extract each source chroma TB once. Proxy ranking, ambiguous-mode
        // trials and the committed winner all reuse these blocks instead of
        // repeatedly copying/clamping the same source rows.
        for t in 0..n_chroma_tb {
            let sub_ch_row = ch_row + t * ctb;
            let offset = t * n_ch;
            extract_block_dyn_into(
                src_cb,
                src_cw,
                src_ch,
                sub_ch_row,
                ch_col,
                ctb,
                &mut scratch.chroma_orig_cb[offset..offset + n_ch],
            );
            extract_block_dyn_into(
                src_cr,
                src_cw,
                src_ch,
                sub_ch_row,
                ch_col,
                ctb,
                &mut scratch.chroma_orig_cr[offset..offset + n_ch],
            );
        }

        let all_candidates = chroma_mode_candidates(luma_mode, chroma);
        let chroma_satd_lambda = chroma_lambda.sqrt();

        // Proxy references are candidate-independent. Gather availability and
        // substitute missing samples once per chroma TB instead of repeating the
        // Morton/decode-order walk for all five modes. For 4:2:2 the source-seeded
        // upper TB gives the lower proxy its candidate-independent top boundary.
        let first_proxy_refs = intra::get_reference_samples_chroma_pair(
            rec_cb,
            rec_cr,
            intra::ChromaRefGeometry {
                stride: cw_stride,
                block_row: ch_row,
                block_col: ch_col,
                chroma_h: coded_ch_h,
                n: ctb,
                sub_w,
                sub_h,
                luma_w: yw_stride,
                luma_h: coded_yh,
                luma_ctus_x,
                min_luma_pu: 8,
                cur_luma_row: lu_row,
                cur_luma_col: lu_col,
                neutral,
            },
        );
        let mut proxy_refs = [first_proxy_refs; 2];
        if n_chroma_tb > 1 {
            proxy_refs[1] = intra::get_reference_samples_chroma_pair(
                rec_cb,
                rec_cr,
                intra::ChromaRefGeometry {
                    stride: cw_stride,
                    block_row: ch_row + ctb,
                    block_col: ch_col,
                    chroma_h: coded_ch_h,
                    n: ctb,
                    sub_w,
                    sub_h,
                    luma_w: yw_stride,
                    luma_h: coded_yh,
                    luma_ctus_x,
                    min_luma_pu: 8,
                    // 4:2:2 lower TB: anchor decode-order at its own luma row so the
                    // reconstructed upper TB counts as an available above-neighbour.
                    cur_luma_row: lu_row + ctb * sub_h,
                    cur_luma_col: lu_col,
                    neutral,
                },
            );
        }

        let mut ranked = [ChromaModeCandidate {
            pred_mode: 0,
            syntax_idx: 0,
            cost: f32::MAX,
        }; 5];
        if lossless {
            // Every mode reconstructs exactly, so SATD is not an RD proxy at all.
            // Evaluate the five legal chroma modes directly by entropy rate; DM is
            // first because its one-bin mode syntax is the most likely early winner.
            for (dst, candidate_index) in ranked.iter_mut().zip([4usize, 0, 3, 1, 2]) {
                *dst = all_candidates[candidate_index];
                dst.cost = 0.0;
            }
        } else {
            // Test DM first so its cheap one-bin syntax establishes a useful top-two
            // cutoff early. A candidate whose partial SATD already exceeds the second
            // best complete proxy can be abandoned safely: all remaining terms are
            // non-negative and only the top two can enter reconstruction RDO.
            for candidate_index in [4usize, 0, 3, 1, 2] {
                let mut candidate = all_candidates[candidate_index];
                let mode = candidate.pred_mode;
                let mut proxy_cost =
                    chroma_satd_lambda * estimated_chroma_mode_bins(candidate.syntax_idx) as f32;
                #[allow(clippy::needless_range_loop)]
                'proxy_tbs: for t in 0..n_chroma_tb {
                    let filt = ctb > 4 && intra::should_filter_refs(mode, ctb);
                    let ((bc0, ba, bl), (rc0, ra, rl)) = &proxy_refs[t];
                    let cb_filtered = if filt {
                        Some(intra::filter_references(*bc0, ba, bl, ctb))
                    } else {
                        None
                    };
                    let cr_filtered = if filt {
                        Some(intra::filter_references(*rc0, ra, rl, ctb))
                    } else {
                        None
                    };
                    let bcf = if filt {
                        ((ba[0] as i32 + 2 * (*bc0 as i32) + bl[0] as i32 + 2) >> 2) as u16
                    } else {
                        *bc0
                    };
                    let rcf = if filt {
                        ((ra[0] as i32 + 2 * (*rc0 as i32) + rl[0] as i32 + 2) >> 2) as u16
                    } else {
                        *rc0
                    };
                    let (baf, blf) = match &cb_filtered {
                        Some((above, left)) => (&above[..], &left[..]),
                        None => (&ba[..], &bl[..]),
                    };
                    let (raf, rlf) = match &cr_filtered {
                        Some((above, left)) => (&above[..], &left[..]),
                        None => (&ra[..], &rl[..]),
                    };

                    let source_offset = t * n_ch;
                    for component in 0..2 {
                        let (orig, corner, above, left) = if component == 0 {
                            (
                                &scratch.chroma_orig_cb[source_offset..source_offset + n_ch],
                                bcf,
                                baf,
                                blf,
                            )
                        } else {
                            (
                                &scratch.chroma_orig_cr[source_offset..source_offset + n_ch],
                                rcf,
                                raf,
                                rlf,
                            )
                        };
                        intra::predict_chroma_tb_into(
                            mode,
                            corner,
                            above,
                            left,
                            ctb,
                            max_val as i32,
                            &mut scratch.pred,
                            &mut scratch.angular,
                        );
                        proxy_cost += scratch.satd(orig, &scratch.pred[..n_ch], ctb) as f32;
                        if proxy_cost >= ranked[1].cost {
                            break 'proxy_tbs;
                        }
                    }
                }
                candidate.cost = proxy_cost;
                update_chroma_candidate(&mut ranked, candidate);
            }
        }

        let full_rd_count = if lossless {
            ranked.len()
        } else {
            full_rdo_chroma_count(&ranked, chroma)
        };
        let mut residual_ctx_after_luma = ctx.clone();
        match tu_layout {
            TuLayout::Unsplit => {
                let _ = residual_ctx_after_luma.cbf_luma[1].estimate_and_update(y_nz as u8);
                if y_nz {
                    advance_residual_contexts(
                        &mut residual_ctx_after_luma,
                        &scratch.scanned[..num_luma],
                        luma_log2_ts,
                        true,
                        luma_scan_idx,
                        !lossless,
                    );
                }
            }
            TuLayout::Split => {
                let child = lu / 2;
                let child_len = child * child;
                let child_log2 = child.trailing_zeros();
                for index in 0..4 {
                    let nonzero = scratch.tu_tree.y_nz[index];
                    let _ = residual_ctx_after_luma.cbf_luma[0].estimate_and_update(nonzero as u8);
                    if !nonzero {
                        continue;
                    }
                    let offset = index * child_len;
                    advance_residual_contexts(
                        &mut residual_ctx_after_luma,
                        &scratch.tu_tree.y_zz[offset..offset + child_len],
                        child_log2,
                        true,
                        scratch.tu_tree.y_scan_idx[index],
                        !lossless,
                    );
                }
            }
            TuLayout::Recursive => {
                advance_lossless_luma_tree_context(
                    &scratch.tu_tree,
                    0,
                    lu,
                    0,
                    0,
                    &mut residual_ctx_after_luma,
                );
            }
        }

        let mut evaluate_chroma = |candidate: ChromaModeCandidate,
                                   estimate_rate: bool,
                                   winner_rdoq: bool,
                                   cost_limit: f32|
         -> f32 {
            let mode = candidate.pred_mode;
            let scan_idx = dct::scan_idx_for(mode, log2_ctb, false, is_444);
            let scan = dct::coeff_scan(log2_ctb, scan_idx);
            let mut distortion = 0.0f32;

            for t in 0..n_chroma_tb {
                let sub_ch_row = ch_row + t * ctb;
                let filt = ctb > 4 && intra::should_filter_refs(mode, ctb);
                let ((bc0, ba, bl), (rc0, ra, rl)) = intra::get_reference_samples_chroma_pair(
                    rec_cb,
                    rec_cr,
                    intra::ChromaRefGeometry {
                        stride: cw_stride,
                        block_row: sub_ch_row,
                        block_col: ch_col,
                        chroma_h: coded_ch_h,
                        n: ctb,
                        sub_w,
                        sub_h,
                        luma_w: yw_stride,
                        luma_h: coded_yh,
                        luma_ctus_x,
                        min_luma_pu: 8,
                        // 4:2:2 lower TB (t=1) anchors decode-order at its own luma
                        // row so the reconstructed upper TB is an available neighbour.
                        cur_luma_row: lu_row + t * ctb * sub_h,
                        cur_luma_col: lu_col,
                        neutral,
                    },
                );
                let (baf, blf, bcf) = if filt {
                    let (above, left) = intra::filter_references(bc0, &ba, &bl, ctb);
                    let corner = ((ba[0] as i32 + 2 * bc0 as i32 + bl[0] as i32 + 2) >> 2) as u16;
                    (above, left, corner)
                } else {
                    (ba, bl, bc0)
                };
                let (raf, rlf, rcf) = if filt {
                    let (above, left) = intra::filter_references(rc0, &ra, &rl, ctb);
                    let corner = ((ra[0] as i32 + 2 * rc0 as i32 + rl[0] as i32 + 2) >> 2) as u16;
                    (above, left, corner)
                } else {
                    (ra, rl, rc0)
                };

                let source_offset = t * n_ch;
                for component in 0..2 {
                    let (orig, rec_plane, corner, above, left) = if component == 0 {
                        (
                            &scratch.chroma_orig_cb[source_offset..source_offset + n_ch],
                            &mut rec_cb[..],
                            bcf,
                            &baf[..],
                            &blf[..],
                        )
                    } else {
                        (
                            &scratch.chroma_orig_cr[source_offset..source_offset + n_ch],
                            &mut rec_cr[..],
                            rcf,
                            &raf[..],
                            &rlf[..],
                        )
                    };

                    intra::predict_chroma_tb_into(
                        mode,
                        corner,
                        above,
                        left,
                        ctb,
                        max_val as i32,
                        &mut scratch.pred,
                        &mut scratch.angular,
                    );
                    intra::compute_residual_i32_into(
                        orig,
                        &scratch.pred[..n_ch],
                        ctb,
                        &mut scratch.residual,
                    );

                    if lossless {
                        forward_lossless_rdpcm_into(
                            &scratch.residual[..n_ch],
                            ctb,
                            implicit_rdpcm_mode(mode),
                            &mut scratch.levels,
                        );
                    } else {
                        crate::hevc_transform::run_fwd_transform(
                            scratch.fwd_transform,
                            &scratch.residual[..n_ch],
                            ctb,
                            bit_depth.bits(),
                            &mut scratch.coeff,
                            &mut scratch.transform_tmp,
                            false,
                        );
                        if winner_rdoq {
                            let tb = crate::hevc_transform::RdoqTb {
                                coeff: &scratch.coeff,
                                n: ctb,
                                qp: chroma_qp,
                                bit_depth: bit_depth.bits(),
                                scan,
                                scan_idx,
                                lambda: chroma_lambda,
                            };
                            crate::hevc_transform::rdoq_chroma_with_sign_hiding_into(
                                &tb,
                                &residual_ctx_after_luma,
                                &mut scratch.levels,
                                &mut scratch.rdoq,
                            );
                        } else {
                            crate::hevc_transform::quantize_with_sign_hiding_into(
                                &scratch.coeff,
                                ctb,
                                chroma_qp,
                                bit_depth.bits(),
                                scan,
                                &mut scratch.levels,
                            );
                        }
                    }

                    let tb = &mut scratch.chroma_tbs[t];
                    let (zigzag, nonzero) = if component == 0 {
                        (&mut tb.cb_zz, &mut tb.cb_nz)
                    } else {
                        (&mut tb.cr_zz, &mut tb.cr_nz)
                    };
                    *nonzero = false;
                    for (dst, &(scan_row, scan_col)) in zigzag[..n_ch].iter_mut().zip(scan) {
                        let level = scratch.levels[scan_row * ctb + scan_col];
                        *dst = level;
                        *nonzero |= level != 0;
                    }

                    if lossless {
                        // Transquant bypass plus inverse RDPCM reproduces `orig`
                        // exactly. Copy the source block directly into the reference
                        // picture (needed by the lower 4:2:2 TB) and skip inverse
                        // processing, reconstruction and an always-zero SSE pass.
                        for (src_row, dst_row) in orig
                            .chunks_exact(ctb)
                            .zip(rec_plane[sub_ch_row * cw_stride + ch_col..].chunks_mut(cw_stride))
                        {
                            dst_row[..ctb].copy_from_slice(src_row);
                        }
                    } else {
                        crate::hevc_transform::run_dequantize(
                            scratch.dequantize,
                            &scratch.levels,
                            ctb,
                            chroma_qp,
                            bit_depth.bits(),
                            &mut scratch.dequant,
                        );
                        crate::hevc_transform::run_inv_transform(
                            scratch.inv_transform,
                            &scratch.dequant,
                            ctb,
                            bit_depth.bits(),
                            &mut scratch.inverse,
                            &mut scratch.transform_tmp,
                            false,
                        );
                        intra::reconstruct_into(
                            &scratch.pred[..n_ch],
                            &scratch.inverse[..n_ch],
                            ctb,
                            max_val,
                            &mut scratch.reconstructed,
                        );
                        distortion += block_sse(orig, &scratch.reconstructed[..n_ch], ctb);
                        if estimate_rate && distortion >= cost_limit {
                            return distortion;
                        }
                        for (src_row, dst_row) in scratch.reconstructed[..n_ch]
                            .chunks_exact(ctb)
                            .zip(rec_plane[sub_ch_row * cw_stride + ch_col..].chunks_mut(cw_stride))
                        {
                            dst_row[..ctb].copy_from_slice(src_row);
                        }
                    }
                }
            }

            if !estimate_rate || distortion >= cost_limit {
                return distortion;
            }

            // Follow the real syntax order so chroma residual estimates see
            // the CABAC states produced by luma residual coding. Only the
            // fractional sink is used; the arithmetic coder is untouched.
            let mut trial_ctx = residual_ctx_after_luma.clone();
            let mut trial_ictx = ictx.clone();
            let mut rate = estimate_chroma_mode_bits(&mut trial_ictx, candidate.syntax_idx);
            for tb in &scratch.chroma_tbs[..n_chroma_tb] {
                rate += trial_ctx.cbf_chroma[0].estimate_and_update(tb.cb_nz as u8);
            }
            for tb in &scratch.chroma_tbs[..n_chroma_tb] {
                rate += trial_ctx.cbf_chroma[0].estimate_and_update(tb.cr_nz as u8);
            }
            // cbf_luma is candidate-independent and uses a separate context,
            // so omitting it preserves ordering and avoids a constant cost.
            for tb in &scratch.chroma_tbs[..n_chroma_tb] {
                if tb.cb_nz {
                    rate += estimate_residual_bits(
                        &mut trial_ctx,
                        &tb.cb_zz[..n_ch],
                        log2_ctb,
                        false,
                        scan_idx,
                        !lossless,
                    );
                }
            }
            for tb in &scratch.chroma_tbs[..n_chroma_tb] {
                if tb.cr_nz {
                    rate += estimate_residual_bits(
                        &mut trial_ctx,
                        &tb.cr_zz[..n_ch],
                        log2_ctb,
                        false,
                        scan_idx,
                        !lossless,
                    );
                }
            }
            distortion + chroma_lambda * rate
        };

        let best_chroma = if full_rd_count == 1 {
            let winner = ranked[0];
            if !split_chroma_tree {
                let _ = evaluate_chroma(winner, false, true, f32::MAX);
            }
            winner
        } else {
            let mut winner = ranked[0];
            let mut best_cost = f32::MAX;
            for &candidate in &ranked[..full_rd_count] {
                let cost = evaluate_chroma(candidate, true, false, best_cost);
                if cost < best_cost {
                    best_cost = cost;
                    winner = candidate;
                }
            }
            if !split_chroma_tree {
                let _ = evaluate_chroma(winner, false, true, f32::MAX);
            }
            winner
        };

        chroma_tb_scan_idx = dct::scan_idx_for(best_chroma.pred_mode, log2_ctb, false, is_444);
        if matches!(tu_layout, TuLayout::Recursive) {
            scratch.tu_tree.node_cb_nz.fill([false; 2]);
            scratch.tu_tree.node_cr_nz.fill([false; 2]);
            scratch.tu_tree.node_chroma_side.fill(0);
            let mut cursor = 0;
            commit_lossless_chroma_tree(
                scratch,
                src_cb,
                src_cr,
                rec_cb,
                rec_cr,
                src_cw,
                src_ch,
                cw_stride,
                coded_ch_h,
                yw_stride,
                coded_yh,
                lu_row,
                lu_col,
                lu,
                chroma,
                best_chroma.pred_mode,
                bit_depth.bits(),
                max_val,
                0,
                &mut cursor,
            );
            debug_assert!(cursor <= 1024);
            let _ = aggregate_lossless_chroma(&mut scratch.tu_tree, 0, lu, chroma);
        } else if shared_chroma {
            for index in 0..n_chroma_tb {
                scratch.tu_tree.chroma_scan_idx[index] = chroma_tb_scan_idx;
            }
        } else if split_chroma_tree {
            commit_split_chroma(
                scratch,
                ChromaPlanes {
                    src_cb,
                    src_cr,
                    rec_cb,
                    rec_cr,
                },
                ChromaSplitGeom {
                    src_cw,
                    src_ch,
                    cw_stride,
                    coded_ch_h,
                    lu_row,
                    lu_col,
                    parent_luma: lu,
                    yw_stride,
                    coded_yh,
                },
                ChromaCoding {
                    chroma,
                    chroma_qp,
                    bit_depth: bit_depth.bits(),
                    max_val,
                    lambda: chroma_lambda,
                },
                best_chroma.pred_mode,
                &residual_ctx_after_luma,
                lossless,
            );
        }
        encode_chroma_mode(enc, ictx, best_chroma.syntax_idx);
    }

    // ── CABAC: transform_tree() + transform_unit() ────────────────────────
    match tu_layout {
        TuLayout::Unsplit => {
            // The root split flag is present for every 8/16/32 PART_2Nx2N CU.
            // The unsplit path codes zero and retains the component-major root-TU syntax.
            let split_ctx = split_transform_context(lu);
            enc.encode_bin(0, &mut ctx.split_transform_flag[split_ctx]);
            for t in &scratch.chroma_tbs[..n_chroma_tb] {
                encode_cbf_chroma(enc, ctx, t.cb_nz, 0);
            }
            for t in &scratch.chroma_tbs[..n_chroma_tb] {
                encode_cbf_chroma(enc, ctx, t.cr_nz, 0);
            }
            encode_cbf_luma(enc, ctx, y_nz, 0);

            let need_qp = y_nz
                || scratch.chroma_tbs[..n_chroma_tb]
                    .iter()
                    .any(|tb| tb.cb_nz || tb.cr_nz);
            encode_cu_qp_delta_if_needed(enc, ctx, aq, need_qp);

            if y_nz {
                encode_residual(
                    enc,
                    ctx,
                    &scratch.scanned[..num_luma],
                    luma_log2_ts,
                    true,
                    luma_scan_idx,
                    !lossless,
                );
            }
            for t in &scratch.chroma_tbs[..n_chroma_tb] {
                if t.cb_nz {
                    encode_residual(
                        enc,
                        ctx,
                        &t.cb_zz[..ctb * ctb],
                        log2_ctb,
                        false,
                        chroma_tb_scan_idx,
                        !lossless,
                    );
                }
            }
            for t in &scratch.chroma_tbs[..n_chroma_tb] {
                if t.cr_nz {
                    encode_residual(
                        enc,
                        ctx,
                        &t.cr_zz[..ctb * ctb],
                        log2_ctb,
                        false,
                        chroma_tb_scan_idx,
                        !lossless,
                    );
                }
            }
        }
        TuLayout::Split => encode_split_transform_tree(
            enc,
            ctx,
            scratch,
            lu,
            chroma,
            false,
            shared_chroma,
            !lossless,
            aq,
        ),
        TuLayout::Recursive => encode_lossless_transform_tree(enc, ctx, scratch, lu, chroma),
    }
}

/// Encode one intra CU (luma side `lu` = 8, 16, or 32) at (lu_row,lu_col) into the
/// bitstream and reconstruction planes; chroma coords derive via subsampling.
#[allow(clippy::too_many_arguments)]
fn code_one_cu<W: CabacWriter>(
    ent: Entropy<'_, W>,
    yuv: &Yuv,
    rec: CuRecPlanes<'_>,
    pos: CuPos,
    strides: PlaneStrides,
    coding: SliceCoding,
    mode_map: &mut [u8],
    mode_stride: usize,
    aq: &mut AqCtuState,
    scratch: &mut CompressionContext,
) {
    let CuRecPlanes {
        y: rec_y,
        cb: rec_cb,
        cr: rec_cr,
    } = rec;
    let CuPos { lu_row, lu_col, lu } = pos;
    let SliceCoding {
        qp,
        lambda,
        lossless,
    } = coding;
    let PlaneStrides {
        w,
        src_yw,
        src_yh,
        cw,
        src_cw,
        src_ch,
        sub_w,
        sub_h,
    } = strides;
    let ch_row = lu_row / sub_h;
    let ch_col = lu_col / sub_w;
    let geo = CuGeometry {
        lu_row,
        lu_col,
        ch_row,
        ch_col,
        yw_stride: w,
        src_yh,
        cw_stride: cw,
        src_cw,
        src_ch,
        mode_stride,
    };
    let src = CuSrcPlanes {
        y: &yuv.y,
        cb: &yuv.cb,
        cr: &yuv.cr,
        src_yw,
    };
    let mut rec = CuRecPlanes {
        y: rec_y,
        cb: rec_cb,
        cr: rec_cr,
    };
    let par = CuParams {
        qp,
        chroma: yuv.chroma,
        bit_depth: yuv.bit_depth,
        lu,
        lambda,
        lossless,
    };
    encode_cu(ent, &src, &mut rec, &geo, &par, mode_map, aq, scratch);
}

/// Extract an N×N block into reusable storage. Rows that lie fully inside the
/// source are copied as slices; only the right/bottom edge needs scalar clamping.
#[inline]
fn extract_block_n_into<const N: usize>(
    plane: &[u16],
    src_w: usize,
    src_h: usize,
    row: usize,
    col: usize,
    out: &mut [u16],
) {
    debug_assert!(out.len() >= N * N);
    for (r, dst) in out[..N * N].as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let src_row = (row + r).min(src_h - 1);
        let available = src_w.saturating_sub(col).min(N);
        if available != 0 {
            let start = src_row * src_w + col.min(src_w - 1);
            dst[..available].copy_from_slice(&plane[start..start + available]);
            let last = dst[available - 1];
            dst[available..].fill(last);
        } else {
            dst.fill(plane[src_row * src_w + src_w - 1]);
        }
    }
}

/// Runtime-sized variant used by chroma TBs.
#[inline]
fn extract_block_dyn_into(
    plane: &[u16],
    src_w: usize,
    src_h: usize,
    row: usize,
    col: usize,
    n: usize,
    out: &mut [u16],
) {
    debug_assert!(out.len() >= n * n);
    for (r, dst) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let src_row = (row + r).min(src_h - 1);
        let available = src_w.saturating_sub(col).min(n);
        if available != 0 {
            let start = src_row * src_w + col.min(src_w - 1);
            dst[..available].copy_from_slice(&plane[start..start + available]);
            let last = dst[available - 1];
            dst[available..].fill(last);
        } else {
            dst.fill(plane[src_row * src_w + src_w - 1]);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_aq_moves_bits_from_flat_to_textured_ctus() {
        let (w, h) = (128usize, 64usize);
        let mut y = vec![128u16; w * h];
        for row in 0..h {
            for col in 64..w {
                y[row * w + col] = if (row + col) & 1 == 0 { 16 } else { 240 };
            }
        }
        let yuv = Yuv {
            y,
            cb: Vec::new(),
            cr: Vec::new(),
            width: w as u32,
            height: h as u32,
            display_w: w as u32,
            display_h: h as u32,
            chroma: crate::fmt::ChromaFormat::Monochrome,
            bit_depth: crate::fmt::BitDepth::Eight,
        };
        let offsets = activity_qp_offsets(&yuv, 2, 1, 38, false, crate::VarianceBoost::default());
        assert!(
            offsets[0] < 0,
            "flat CTU should spend more bits: {offsets:?}"
        );
        assert!(
            offsets[1] > 0,
            "textured CTU should spend fewer bits: {offsets:?}"
        );
        assert!(offsets.iter().all(|&offset| (-3..=3).contains(&offset)));
    }

    #[test]
    fn variance_boost_is_bounded_and_only_targets_coarse_low_contrast_blocks() {
        let config = crate::VarianceBoost::default();
        assert_eq!(variance_boost_qp(0.0, 32, config), 0.0);
        assert_eq!(variance_boost_qp(6.0, 39, config), 0.0);
        let boost = variance_boost_qp(2.0, 39, config);
        assert!(boost > 0.5 && boost <= 1.15, "unexpected boost {boost}");
    }

    #[test]
    fn cu_qp_delta_uses_truncated_unary_and_sign_bypass() {
        #[derive(Default)]
        struct Trace {
            regular: Vec<u8>,
            bypass: Vec<u8>,
        }
        impl CabacWriter for Trace {
            fn encode_bin(&mut self, bin_val: u8, ctx: &mut crate::cabac::engine::CtxModel) {
                self.regular.push(bin_val);
                ctx.update(bin_val);
            }
            fn encode_bypass(&mut self, bin_val: u8) {
                self.bypass.push(bin_val);
            }
        }

        let mut trace = Trace::default();
        let mut ctx = ContextSet::init_islice(38);
        let mut aq = AqCtuState {
            enabled: true,
            predictor: 38,
            target: 35,
            coded: false,
        };
        encode_cu_qp_delta(&mut trace, &mut ctx, &mut aq);
        assert_eq!(trace.regular, [1, 1, 1, 0]);
        assert_eq!(trace.bypass, [1]);
        assert!(aq.coded);
    }

    #[test]
    fn intra_candidate_list_keeps_best_costs_sorted() {
        let mut candidates = [IntraModeCandidate {
            mode: u8::MAX,
            cost: f32::MAX,
        }; 3];
        update_intra_candidate(&mut candidates, 10, 10.0);
        update_intra_candidate(&mut candidates, 20, 4.0);
        update_intra_candidate(&mut candidates, 30, 7.0);
        update_intra_candidate(&mut candidates, 40, 12.0);
        update_intra_candidate(&mut candidates, 50, 5.0);

        assert_eq!(candidates.map(|candidate| candidate.mode), [20, 50, 30]);
        assert!(
            candidates
                .array_windows::<2>()
                .all(|pair| pair[0].cost <= pair[1].cost)
        );
    }

    #[test]
    fn full_rdo_candidate_budget_is_bounded() {
        let close = [
            IntraModeCandidate {
                mode: 0,
                cost: 100.0,
            },
            IntraModeCandidate {
                mode: 1,
                cost: 105.0,
            },
            IntraModeCandidate {
                mode: 2,
                cost: 115.0,
            },
            IntraModeCandidate {
                mode: 3,
                cost: 150.0,
            },
        ];
        assert_eq!(full_rdo_candidate_count(&close, 8), 3);
        assert_eq!(full_rdo_candidate_count(&close, 16), 2);

        let separated = [
            IntraModeCandidate {
                mode: 0,
                cost: 100.0,
            },
            IntraModeCandidate {
                mode: 1,
                cost: 130.0,
            },
            IntraModeCandidate {
                mode: 2,
                cost: 140.0,
            },
        ];
        assert_eq!(full_rdo_candidate_count(&separated, 8), 2);
    }

    #[test]
    fn cu_depth_map_drives_split_context() {
        let stride = 8;
        let mut depths = [0u8; 64];
        fill_cu_depth(&mut depths, 0, 0, 32, 1, stride);
        assert!(depths[..4].iter().all(|&depth| depth == 1));

        fill_cu_depth(&mut depths, 0, 0, 16, 2, stride);
        assert_eq!(split_cu_context(&depths, 0, 16, 1, stride), 1);
        assert_eq!(split_cu_context(&depths, 16, 0, 1, stride), 1);
    }

    #[test]
    fn deblock_map_contains_cu_and_selected_tu_edges() {
        let stride = 16;
        let mut v = vec![false; stride * stride];
        let mut h = v.clone();
        mark_deblock_edges(
            &mut v,
            &mut h,
            stride,
            8,
            8,
            16,
            TuLayout::Split,
            &[false; 85],
        );
        assert!(v[(8 / 4) * stride + 8 / 4]);
        assert!(h[(8 / 4) * stride + 8 / 4]);
        assert!(v[(8 / 4) * stride + 16 / 4]);
        assert!(h[(16 / 4) * stride + 8 / 4]);
        assert!(!v[(8 / 4) * stride + 12 / 4]);
    }

    #[test]
    fn deblocking_reduces_actual_reconstruction_sse() {
        let (w, h) = (64usize, 64usize);
        let mut y = vec![0u16; w * h];
        for row in 0..h {
            for col in 0..w {
                // Smooth content with a weak diagonal component makes quantized
                // transform-boundary discontinuities measurable without giving
                // the filter an artificial step edge to erase.
                y[row * w + col] = (48 + (3 * col + 2 * row) / 2).min(255) as u16;
            }
        }
        let yuv = Yuv {
            y: y.clone(),
            cb: vec![128; (w / 2) * (h / 2)],
            cr: vec![128; (w / 2) * (h / 2)],
            width: w as u32,
            height: h as u32,
            display_w: w as u32,
            display_h: h as u32,
            chroma: crate::fmt::ChromaFormat::Yuv420,
            bit_depth: crate::fmt::BitDepth::Eight,
        };
        let pool = crate::pool::ThreadPool::new(0);
        let output = encode_region_substreams(
            &yuv,
            w as u32,
            h as u32,
            38,
            false,
            false,
            1,
            true,
            &pool,
            true,
            crate::VarianceBoost::default(),
        );
        let sse = |rec: &[u16]| -> u64 {
            y.iter()
                .zip(rec)
                .map(|(&a, &b)| {
                    let d = i64::from(a) - i64::from(b);
                    (d * d) as u64
                })
                .sum()
        };
        let before = sse(&output.unfiltered_y);
        let after = sse(&output.y);
        eprintln!("deblock luma SSE: {before} -> {after}");
        assert!(
            after < before,
            "deblock SSE did not improve: {before} -> {after}"
        );
    }

    #[test]
    fn split_444_chroma_at_picture_bottom_does_not_overrun() {
        let (w, h) = (64usize, 64usize);
        let mut seed = 0x9e37_79b9u32;
        let mut plane = || {
            (0..w * h)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u16
                })
                .collect::<Vec<_>>()
        };
        let yuv = Yuv {
            y: plane(),
            cb: plane(),
            cr: plane(),
            width: w as u32,
            height: h as u32,
            display_w: w as u32,
            display_h: h as u32,
            chroma: crate::fmt::ChromaFormat::Yuv444,
            bit_depth: crate::fmt::BitDepth::Eight,
        };
        let pool = crate::pool::ThreadPool::new(0);
        let output = encode_region_substreams(
            &yuv,
            w as u32,
            h as u32,
            30,
            false,
            false,
            1,
            true,
            &pool,
            true,
            crate::VarianceBoost::default(),
        );
        assert_eq!(output.cb.len(), w * h);
        assert_eq!(output.cr.len(), w * h);
    }

    #[test]
    fn luma_mode_bin_estimate_matches_mpm_binarization() {
        let mpm = [0, 1, 26];
        assert_eq!(estimated_luma_mode_bins(0, &mpm), 2);
        assert_eq!(estimated_luma_mode_bins(1, &mpm), 3);
        assert_eq!(estimated_luma_mode_bins(26, &mpm), 3);
        assert_eq!(estimated_luma_mode_bins(17, &mpm), 6);
    }

    #[test]
    fn chroma_candidates_replace_the_dm_duplicate() {
        let modes = chroma_mode_candidates(26, crate::fmt::ChromaFormat::Yuv444);
        assert_eq!(
            modes.map(|candidate| candidate.pred_mode),
            [0, 34, 10, 1, 26]
        );
        assert_eq!(modes.map(|candidate| candidate.syntax_idx), [0, 1, 2, 3, 4]);

        let mapped = chroma_mode_candidates(26, crate::fmt::ChromaFormat::Yuv422);
        assert_eq!(mapped[1].pred_mode, MODE_422_MAP[34]);
        assert_eq!(mapped[4].pred_mode, MODE_422_MAP[26]);
        for i in 0..mapped.len() {
            for j in i + 1..mapped.len() {
                assert_ne!(mapped[i].pred_mode, mapped[j].pred_mode);
            }
        }
    }

    #[test]
    fn chroma_mode_bin_estimate_matches_binarization() {
        assert_eq!(estimated_chroma_mode_bins(CHROMA_DM_SYNTAX_IDX), 1);
        for syntax_idx in 0..CHROMA_DM_SYNTAX_IDX {
            assert_eq!(estimated_chroma_mode_bins(syntax_idx), 3);
        }
    }

    #[test]
    fn chroma_mode_binarization_uses_dm_flag_then_two_bypass_bits() {
        #[derive(Default)]
        struct Recorder {
            regular: Vec<u8>,
            bypass: Vec<u8>,
        }
        impl CabacWriter for Recorder {
            fn encode_bin(&mut self, bin_val: u8, ctx: &mut crate::cabac::engine::CtxModel) {
                self.regular.push(bin_val);
                let _ = ctx.estimate_and_update(bin_val);
            }

            fn encode_bypass(&mut self, bin_val: u8) {
                self.bypass.push(bin_val);
            }
        }

        let mut ictx = IntraModeContexts::init_islice(26);
        let mut dm = Recorder::default();
        encode_chroma_mode(&mut dm, &mut ictx, CHROMA_DM_SYNTAX_IDX);
        assert_eq!(dm.regular, [0]);
        assert!(dm.bypass.is_empty());

        let mut ictx = IntraModeContexts::init_islice(26);
        let mut explicit = Recorder::default();
        encode_chroma_mode(&mut explicit, &mut ictx, 2);
        assert_eq!(explicit.regular, [1]);
        assert_eq!(explicit.bypass, [1, 0]);
    }

    #[test]
    fn chroma_full_rdo_is_only_expanded_for_close_proxy_modes() {
        let make = |first: f32, second: f32| {
            let mut candidates = [ChromaModeCandidate {
                pred_mode: 0,
                syntax_idx: 0,
                cost: f32::MAX,
            }; 5];
            candidates[0].cost = first;
            candidates[1].cost = second;
            candidates
        };

        assert_eq!(
            full_rdo_chroma_count(&make(100.0, 102.0), crate::fmt::ChromaFormat::Yuv420),
            2
        );
        assert_eq!(
            full_rdo_chroma_count(&make(100.0, 104.0), crate::fmt::ChromaFormat::Yuv420),
            1
        );
        assert_eq!(
            full_rdo_chroma_count(&make(100.0, 107.0), crate::fmt::ChromaFormat::Yuv444),
            2
        );
        assert_eq!(
            full_rdo_chroma_count(&make(100.0, 109.0), crate::fmt::ChromaFormat::Yuv444),
            1
        );
    }

    #[test]
    fn chroma_lambda_tracks_the_420_qp_mapping() {
        assert_eq!(
            chroma_lambda_scale(29, crate::fmt::ChromaFormat::Yuv420),
            1.0
        );
        assert_eq!(
            chroma_lambda_scale(43, crate::fmt::ChromaFormat::Yuv420),
            0.25
        );
        assert_eq!(
            chroma_lambda_scale(51, crate::fmt::ChromaFormat::Yuv420),
            0.25
        );
        assert_eq!(
            chroma_lambda_scale(43, crate::fmt::ChromaFormat::Yuv444),
            1.0
        );
    }

    #[test]
    fn lossless_profile_enables_rext() {
        let mut lossy = BitWriter::new();
        write_profile_tier_level(
            &mut lossy,
            93,
            crate::fmt::ChromaFormat::Yuv420,
            crate::fmt::BitDepth::Eight,
            false,
        );
        let lossy = lossy.finish();
        assert_eq!(lossy[0] & 0x1f, 3);

        let mut lossless = BitWriter::new();
        write_profile_tier_level(
            &mut lossless,
            93,
            crate::fmt::ChromaFormat::Yuv420,
            crate::fmt::BitDepth::Eight,
            true,
        );
        let lossless = lossless.finish();
        assert_eq!(lossless[0] & 0x1f, 4);
        // General RExt tools force the 4:4:4 constraint profile in HM: do not
        // claim max-4:2:2, max-4:2:0 or monochrome merely because the source is
        // 4:2:0. The intra constraint remains set.
        assert_eq!(lossless[5] & 0x01, 0); // max_422chroma_constraint_flag
        assert_eq!(lossless[6] & 0xc0, 0); // max_420 + max_monochrome
        assert_ne!(lossless[6] & 0x20, 0); // intra_constraint_flag
    }

    #[test]
    fn lossless_range_extension_sets_only_implicit_rdpcm() {
        let mut enabled = BitWriter::new();
        write_sps_range_extension(&mut enabled, true);
        assert_eq!(enabled.finish().as_slice(), &[0x20, 0x00]);

        let mut disabled = BitWriter::new();
        write_sps_range_extension(&mut disabled, false);
        assert_eq!(disabled.finish().as_slice(), &[0x00, 0x00]);
    }

    #[test]
    fn implicit_rdpcm_mode_is_inferred_from_final_intra_mode() {
        assert_eq!(implicit_rdpcm_mode(10), ImplicitRdpcm::Horizontal);
        assert_eq!(implicit_rdpcm_mode(26), ImplicitRdpcm::Vertical);
        assert_eq!(implicit_rdpcm_mode(0), ImplicitRdpcm::Off);
        assert_eq!(implicit_rdpcm_mode(34), ImplicitRdpcm::Off);
    }

    #[test]
    fn implicit_rdpcm_roundtrips_all_supported_tb_sizes() {
        for n in [4usize, 8, 16, 32] {
            let mut residual = [0i32; 1024];
            for (index, sample) in residual[..n * n].iter_mut().enumerate() {
                let row = index / n;
                let col = index % n;
                *sample = (((row * 977 + col * 613 + row * col * 29) % 8191) as i32) - 4095;
            }

            for mode in [
                ImplicitRdpcm::Off,
                ImplicitRdpcm::Horizontal,
                ImplicitRdpcm::Vertical,
            ] {
                let mut levels = [0i16; 1024];
                let mut decoded = [0i32; 1024];
                forward_lossless_rdpcm_into(&residual, n, mode, &mut levels);
                inverse_lossless_rdpcm_into(&levels, n, mode, &mut decoded);
                assert_eq!(
                    &decoded[..n * n],
                    &residual[..n * n],
                    "roundtrip failed for {n}x{n} {mode:?}"
                );
                assert!(
                    levels[..n * n]
                        .iter()
                        .all(|&level| (-8190..=8190).contains(&(level as i32)))
                );
            }
        }
    }

    #[test]
    fn implicit_rdpcm_differences_the_expected_axis() {
        let residual = [
            1, 3, 6, 10, //
            2, 5, 9, 14, //
            4, 8, 13, 19, //
            7, 12, 18, 25,
        ];
        let mut levels = [0i16; 1024];

        forward_lossless_rdpcm_into(&residual, 4, ImplicitRdpcm::Horizontal, &mut levels);
        assert_eq!(
            &levels[..16],
            &[1, 2, 3, 4, 2, 3, 4, 5, 4, 4, 5, 6, 7, 5, 6, 7]
        );

        forward_lossless_rdpcm_into(&residual, 4, ImplicitRdpcm::Vertical, &mut levels);
        assert_eq!(
            &levels[..16],
            &[1, 3, 6, 10, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]
        );
    }

    #[test]
    fn split_transform_context_matches_hevc_size_context() {
        assert_eq!(split_transform_context(32), 0);
        assert_eq!(split_transform_context(16), 1);
        assert_eq!(split_transform_context(8), 2);
    }

    #[test]
    fn split_chroma_sharing_stops_below_minimum_tb() {
        use crate::fmt::ChromaFormat::{Yuv420, Yuv422, Yuv444};

        assert!(split_chroma_is_shared(8, Yuv420));
        assert!(split_chroma_is_shared(8, Yuv422));
        assert!(!split_chroma_is_shared(8, Yuv444));
        assert!(!split_chroma_is_shared(16, Yuv420));
    }

    #[test]
    fn nxn_proxy_rejects_flat_and_accepts_piecewise_residual() {
        let flat = [96u16; 64];
        let satd = crate::cost::resolve_satd();
        assert!(!choose_nxn_proxy(satd, &flat, &flat, 4.0, 8));

        let mut piecewise = [0u16; 64];
        for row in 0..8 {
            for col in 0..8 {
                piecewise[row * 8 + col] = match (row >= 4, col >= 4) {
                    (false, false) => 48,
                    (false, true) => 112,
                    (true, false) => 176,
                    (true, true) => 240,
                };
            }
        }
        let parent = [128u16; 64];
        assert!(choose_nxn_proxy(satd, &piecewise, &parent, 4.0, 8));
    }

    #[test]
    fn bit_writer_basic() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b10110, 5);
        bw.rbsp_trailing_bits();
        assert_eq!(bw.finish()[0], 0b1011_0100);
    }

    #[test]
    fn ue_coding() {
        let mut bw = BitWriter::new();
        bw.write_ue(0); // = 1 → single '1' bit
        bw.rbsp_trailing_bits();
        assert_eq!(bw.finish()[0] >> 7, 1);
    }

    #[test]
    fn vps_starts_with_nalu_header() {
        let vps = build_vps(
            256,
            256,
            crate::fmt::ChromaFormat::Yuv420,
            crate::fmt::BitDepth::Eight,
            false,
        );
        assert_eq!(vps.data[0], 0x40, "VPS first byte should be 0x40");
    }

    #[test]
    fn sps_conformance_window() {
        let sps = build_sps(
            64,
            48,
            crate::fmt::ChromaFormat::Yuv420,
            crate::fmt::BitDepth::Eight,
            false,
            Some(&crate::color::Cicp::srgb()),
        );
        assert!(sps.data.len() > 10);
    }

    #[test]
    fn pps_builds_cleanly() {
        let pps = build_pps(30, false, false);
        assert_eq!(pps.data[0], 0x44, "PPS first byte should be 0x44");
    }
}
