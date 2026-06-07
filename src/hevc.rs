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
        CabacEncoder, ContextSet, IntraModeContexts, encode_cbf_chroma, encode_cbf_luma,
        encode_residual,
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

fn write_profile_tier_level(
    bw: &mut BitWriter,
    level_idc: u8,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
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
    let is_rext = !is_420 || bits > 10;

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
        bw.write_bit(bits <= 12); // max_12bit_constraint_flag
        bw.write_bit(bits <= 10); // max_10bit_constraint_flag
        bw.write_bit(bits <= 8); // max_8bit_constraint_flag
        bw.write_bit(!is_444 || is_mono); // max_422chroma_constraint_flag
        bw.write_bit(is_420 || is_mono); // max_420chroma_constraint_flag
        bw.write_bit(is_mono); // max_monochrome_constraint_flag
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

    write_profile_tier_level(&mut bw, level, chroma, bit_depth);

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

// ─── SPS ─────────────────────────────────────────────────────────────────────

pub(crate) fn build_sps(
    width: u32,
    height: u32,
    chroma: crate::fmt::ChromaFormat,
    bit_depth: crate::fmt::BitDepth,
) -> Nalu {
    let mut bw = BitWriter::new();
    nalu_header(&mut bw, 33);

    bw.write_bits(0, 4); // sps_video_parameter_set_id = 0
    bw.write_bits(0, 3); // sps_max_sub_layers_minus1 = 0
    bw.write_bit(true); // sps_temporal_id_nesting_flag

    let sps_level = level_idc_for((width + 63) & !63, (height + 63) & !63);
    write_profile_tier_level(&mut bw, sps_level, chroma, bit_depth);

    bw.write_ue(0); // sps_seq_parameter_set_id = 0

    bw.write_ue(chroma.idc()); // chroma_format_idc (1=4:2:0, 2=4:2:2, 3=4:4:4)
    // separate_colour_plane_flag is present only when chroma_format_idc == 3. We use
    // packed 4:4:4 (the three components share one coding tree), so the flag is 0.
    if chroma.idc() == 3 {
        bw.write_bit(false); // separate_colour_plane_flag = 0
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
    // max_transform_hierarchy_depth_intra = 0 (matches x265). With depth 0, an 8×8
    // intra CU's transform tree is a single 8×8 TU and split_transform_flag is
    // inferred (not coded).
    bw.write_ue(0);
    // max_transform_hierarchy_depth_inter = 0 (matches x265).
    bw.write_ue(0);

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
    bw.write_bit(true); // strong_intra_smoothing_enabled_flag = 1 (matches x265;
    // only affects 32×32 intra which we don't use, so output
    // is unchanged)

    // VUI parameters: colour info so decoders display correctly
    bw.write_bit(true); // vui_parameters_present_flag
    write_vui(&mut bw);

    // sps_extension: RExt profiles require sps_range_extension to be present
    // even when all flags within it are 0 (x265 always writes it for profile_idc=4).
    // Apple's decoder rejects 12-bit streams whose SPS lacks the range extension.
    let need_range_ext = bit_depth.bits() > 8;
    bw.write_bit(need_range_ext); // sps_extension_present_flag
    if need_range_ext {
        bw.write_bit(true); // sps_range_extension_flag = 1
        bw.write_bit(false); // sps_multilayer_extension_flag = 0
        bw.write_bit(false); // sps_3d_extension_flag = 0
        bw.write_bit(false); // sps_scc_extension_flag = 0
        bw.write_bits(0, 4); // sps_extension_4bits = 0
        // sps_range_extension() — all flags 0 (no RExt features used for intra-only)
        bw.write_bit(false); // transform_skip_rotation_enabled_flag
        bw.write_bit(false); // transform_skip_context_enabled_flag
        bw.write_bit(false); // implicit_rdpcm_enabled_flag
        bw.write_bit(false); // explicit_rdpcm_enabled_flag
        bw.write_bit(false); // extended_precision_processing_flag
        bw.write_bit(false); // intra_smoothing_disabled_flag
        bw.write_bit(false); // high_precision_offsets_enabled_flag
        bw.write_bit(false); // persistent_rice_adaptation_enabled_flag
        bw.write_bit(false); // cabac_bypass_alignment_enabled_flag
    }

    bw.rbsp_trailing_bits();
    Nalu {
        _nal_type: 33,
        data: bw.finish(),
    }
}

/// Write minimal VUI (Annex E §E.2.1) with BT.601 colour info.
fn write_vui(bw: &mut BitWriter) {
    bw.write_bit(false); // aspect_ratio_info_present_flag
    bw.write_bit(false); // overscan_info_present_flag

    // video_signal_type_present_flag = true
    bw.write_bit(true);
    bw.write_bits(5, 3); // video_format = 5 (unspecified)
    bw.write_bit(true); // video_full_range_flag = 1 (full range 0-255)
    bw.write_bit(true); // colour_description_present_flag
    bw.write_bits(1, 8); // colour_primaries         = 1 (BT.709) — matches libheif
    bw.write_bits(13, 8); // transfer_characteristics = 13 (sRGB / IEC 61966-2-1)
    bw.write_bits(1, 8); // matrix_coefficients      = 1 (BT.709) — matches colr

    bw.write_bit(false); // chroma_loc_info_present_flag
    bw.write_bit(false); // neutral_chroma_indication_flag
    bw.write_bit(false); // field_seq_flag
    bw.write_bit(false); // frame_field_info_present_flag
    bw.write_bit(false); // default_display_window_flag
    bw.write_bit(false); // vui_timing_info_present_flag
    bw.write_bit(false); // bitstream_restriction_flag
}

pub(crate) fn build_pps(qp: u8) -> Nalu {
    let mut bw = BitWriter::new();
    nalu_header(&mut bw, 34);

    bw.write_ue(0); // pps_pic_parameter_set_id = 0
    bw.write_ue(0); // pps_seq_parameter_set_id = 0
    bw.write_bit(false); // dependent_slice_segments_enabled_flag
    bw.write_bit(false); // output_flag_present_flag
    bw.write_bits(0, 3); // num_extra_slice_header_bits
    bw.write_bit(false); // sign_data_hiding_enabled_flag
    bw.write_bit(false); // cabac_init_present_flag
    bw.write_ue(0); // num_ref_idx_l0_default_active_minus1
    bw.write_ue(0); // num_ref_idx_l1_default_active_minus1
    bw.write_se(qp as i32 - 26); // init_qp_minus26: carry the full slice QP here
    bw.write_bit(false); // constrained_intra_pred_flag
    bw.write_bit(false); // transform_skip_enabled_flag

    // cu_qp_delta_enabled_flag = false  (fixed QP throughout)
    bw.write_bit(false);
    // No diff_cu_qp_delta_depth since cu_qp_delta_enabled_flag = false

    // pps_cb_qp_offset and pps_cr_qp_offset: ALWAYS present (HEVC spec §7.3.2.3)
    bw.write_se(0); // pps_cb_qp_offset = 0
    bw.write_se(0); // pps_cr_qp_offset = 0

    bw.write_bit(false); // pps_slice_chroma_qp_offsets_present_flag
    bw.write_bit(false); // weighted_pred_flag
    bw.write_bit(false); // weighted_bipred_flag
    bw.write_bit(false); // transquant_bypass_enabled_flag
    bw.write_bit(false); // tiles_enabled_flag
    bw.write_bit(false); // entropy_coding_sync_enabled_flag
    // No tile fields (tiles_enabled=0).
    // seq_loop_filter_across_slices_enabled_flag: ALWAYS present per HEVC spec and
    // ffmpeg decode_pps() unconditionally reads it after tiles/ecs flags.
    bw.write_bit(true); // pps_loop_filter_across_slices_enabled_flag = 1 (matches
    // x265). With a single slice it has no visible effect, but
    // x265 sets it and we keep the PPS identical.

    // Deblocking filter ENABLED with default beta/tc offsets (0). The encoder
    // applies the same in-loop deblocking to its reconstruction, so the output
    // matches conformant decoders (libde265/ffmpeg) and block-edge artifacts are
    // smoothed. We still emit the control-present block so the offsets are
    // explicit rather than relying on defaults.
    bw.write_bit(false); // deblocking_filter_control_present_flag (use defaults: enabled, offsets 0)
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

// ─── IDR slice ───────────────────────────────────────────────────────────────

/// Encode a still image as a single HEVC IDR picture.
pub(crate) fn encode_intra(
    yuv: &Yuv,
    width: u32,
    height: u32,
    quality: u8,
) -> Result<NaluStream, EncodeError> {
    let vps = build_vps(width, height, yuv.chroma, yuv.bit_depth);
    let sps = build_sps(width, height, yuv.chroma, yuv.bit_depth);
    let qp_val: u8 = ((100 - quality.clamp(1, 100) as u32) * 41 / 99 + 10).min(51) as u8;
    let pps = build_pps(qp_val);
    let (idr, _ry, _rcb, _rcr) = build_idr_slice(yuv, width, height, quality)?;
    Ok(NaluStream {
        nalus: vec![vps, sps, pps, idr],
    })
}

#[allow(clippy::type_complexity)]
fn build_idr_slice(
    yuv: &Yuv,
    width: u32,
    height: u32,
    quality: u8,
) -> Result<(Nalu, Vec<u16>, Vec<u16>, Vec<u16>), EncodeError> {
    // Map quality (1-100) to HEVC QP (0-51): quality=100→QP~10, quality=1→QP=51
    let qp_val: u8 = ((100 - quality.clamp(1, 100) as u32) * 41 / 99 + 10).min(51) as u8;
    let _ = quality; // used above

    // Coded dimensions: multiples of CTB size (64 luma). Chroma planes subsample
    // by sub_w horizontally and sub_h vertically (4:2:0 → /2,/2; 4:2:2 → /2,/1).
    let sub_w = yuv.chroma.sub_w();
    let sub_h = yuv.chroma.sub_h();
    // Per 16×16 node, choose between one 16×16 CU and four 8×8 CUs by RD cost.
    // Offered for 4:2:0 only (the validated chroma TB → 8×8 path).
    let rd16 = yuv.chroma == crate::fmt::ChromaFormat::Yuv420;
    let w = ((width + 63) & !63) as usize;
    let h = ((height + 63) & !63) as usize;
    let cw = w / sub_w;
    let ch = h / sub_h;
    let src_yw = yuv.width as usize;
    let src_yh = yuv.height as usize;
    let src_cw = (yuv.width as usize).div_ceil(sub_w);
    let src_ch = (yuv.height as usize).div_ceil(sub_h);

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
    hdr.write_bit(true); // slice_sao_luma_flag   = 1
    if !yuv.chroma.is_monochrome() {
        hdr.write_bit(true); // slice_sao_chroma_flag = 1
    }
    // QP is carried fully in the PPS init_qp_minus26, so slice_qp_delta = 0.
    hdr.write_se(0); // slice_qp_delta
    // slice_loop_filter_across_slices_enabled_flag — REQUIRED here by HEVC §7.3.6.1:
    // it is present whenever pps_loop_filter_across_slices_enabled_flag (set in our
    // PPS) is 1 and (slice_sao_luma_flag || slice_sao_chroma_flag || deblocking not
    // disabled) — and slice_sao_luma_flag is always 1 here, so it is always present.
    // Omitting it leaves the slice header one bit short: a strict decoder consumes
    // the byte_alignment() '1' bit as this flag, then reads the following padding
    // '0' as alignment_bit_equal_to_one and rejects the slice (the
    // "alignment_bit_equal_to_one=0 / undecodable NALU 20" seen in recent ffmpeg).
    // Value 1 matches x265 and is a no-op for a single-slice picture.
    hdr.write_bit(true); // slice_loop_filter_across_slices_enabled_flag = 1
    hdr.rbsp_trailing_bits();
    let header_bytes = hdr.finish();

    // HEVC slice_segment_data(): for each CTU row-major:
    //   coding_tree_unit()  → luma 8×8 CU + chroma 4×4 CU×2
    //   end_of_slice_segment_flag (terminate = 0 or 1)
    let qp: u8 = qp_val;
    let mut cab = CabacEncoder::new();
    let mut ctx = ContextSet::init_islice(qp);
    let mut ictx = IntraModeContexts::init_islice(qp);
    // HM-style intra Lagrange multiplier for J = SSE + λ·R (R in bits).
    let lambda = 0.57_f64 * 2f64.powf((qp as f64 - 12.0) / 3.0);
    // Per-8×8-block quadtree depth (2 = covered by a 16×16 CU, 3 = an 8×8 CU);
    // drives the 16-level split_cu_flag context (depends on neighbour depths).
    let mut cu_depth = vec![0u8; (w / 8) * (h / 8)];
    // Per-8×8-block luma intra mode (for neighbour MPM derivation).
    let blk_stride = w / 8;
    let mut mode_map = vec![0u8; (w / 8) * (h / 8)];

    // Padded reconstruction buffers (prediction uses coded dimensions). Monochrome
    // has no chroma planes.
    let mut rec_y = pad_plane(&yuv.y, src_yw, src_yh, w, h);
    let (mut rec_cb, mut rec_cr) = if yuv.chroma.is_monochrome() {
        (Vec::new(), Vec::new())
    } else {
        (
            pad_plane(&yuv.cb, src_cw, src_ch, cw, ch),
            pad_plane(&yuv.cr, src_cw, src_ch, cw, ch),
        )
    };

    // CTB grid: full 64×64 CTBs over the 64-multiple coded picture (no partial
    // boundary CTBs — the SPS declares the 64-multiple size and the conformance
    // window crops the padding). Each CTB: 64→32→16→8 quadtree, 64 leaf 8×8 CUs.
    let ctb_size_y = 64usize;
    let ctb_size_c = 32usize;
    let cu_size_y = 8usize;
    let cu_size_c = 4usize;
    let ctus_x = w / ctb_size_y;
    let ctus_y = h / ctb_size_y;
    let total_ctus = ctus_x * ctus_y;
    let mut ctu_idx = 0usize;

    for ctu_row in 0..ctus_y {
        for ctu_col in 0..ctus_x {
            let lu_row0 = ctu_row * ctb_size_y;
            let lu_col0 = ctu_col * ctb_size_y;
            let ch_row0 = ctu_row * ctb_size_c;
            let ch_col0 = ctu_col * ctb_size_c;

            // ── sao() — SPS enables SAO and the slice SAO flags are set, so each
            // CTB must carry SAO syntax. We signal SAO OFF everywhere:
            //   sao_merge_left_flag = 0 (when left CTB available)
            //   sao_merge_up_flag   = 0 (when up CTB available, not merged left)
            //   then sao_type_idx_luma = 0 and sao_type_idx_chroma = 0 (both OFF).
            // sao_type_idx is coded as: first bin context-coded, and value 0 means
            // "not applied" (a single 0 bin). With type_idx=0 no further SAO syntax
            // follows, so reconstruction is identical to SAO-disabled.
            if ctu_col > 0 {
                cab.encode_bin(0, &mut ctx.sao_merge_flag); // sao_merge_left_flag = 0
            }
            if ctu_row > 0 {
                cab.encode_bin(0, &mut ctx.sao_merge_flag); // sao_merge_up_flag = 0
            }
            // sao_type_idx_luma = 0 (OFF): single context-coded 0 bin.
            cab.encode_bin(0, &mut ctx.sao_type_idx);
            // sao_type_idx_chroma — only present when ChromaArrayType != 0.
            if !yuv.chroma.is_monochrome() {
                cab.encode_bin(0, &mut ctx.sao_type_idx); // sao_type_idx_chroma = 0 (OFF)
            }

            let cl0 = if ctu_col > 0 { 1 } else { 0 };
            let ca0 = if ctu_row > 0 { 1 } else { 0 };
            cab.encode_bin(1, &mut ctx.split_cu_flag[cl0 + ca0]);

            for (q1y, q1x) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                let l1_lu_r = lu_row0 + q1y * 32;
                let l1_lu_c = lu_col0 + q1x * 32;
                let l1_ch_r = ch_row0 + q1y * 16;
                let l1_ch_c = ch_col0 + q1x * 16;

                let cl1 = if q1x > 0 || ctu_col > 0 { 1 } else { 0 };
                let ca1 = if q1y > 0 || ctu_row > 0 { 1 } else { 0 };
                cab.encode_bin(1, &mut ctx.split_cu_flag[cl1 + ca1]);

                for (q2y, q2x) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                    let l2_lu_r = l1_lu_r + q2y * 16;
                    let l2_lu_c = l1_lu_c + q2x * 16;
                    let l2_ch_r = l1_ch_r + q2y * 8;
                    let l2_ch_c = l1_ch_c + q2x * 8;

                    let cl2 = if q2x > 0 || q1x > 0 || ctu_col > 0 {
                        1
                    } else {
                        0
                    };
                    let ca2 = if q2y > 0 || q1y > 0 || ctu_row > 0 {
                        1
                    } else {
                        0
                    };
                    let _ = (cl2, ca2, l2_ch_r, l2_ch_c, cu_size_c); // superseded by RD + depth ctx
                    // split_cu_flag ctxInc = (left CU deeper than this depth-2 node)
                    // + (above CU deeper), read from the per-block depth map.
                    let bx = w / cu_size_y;
                    let cond_l = l2_lu_c > 0
                        && cu_depth[(l2_lu_r / cu_size_y) * bx + (l2_lu_c - 1) / cu_size_y] > 2;
                    let cond_a = l2_lu_r > 0
                        && cu_depth[((l2_lu_r - 1) / cu_size_y) * bx + l2_lu_c / cu_size_y] > 2;
                    let split_ctx = cond_l as usize + cond_a as usize;

                    let strides8 = PlaneStrides {
                        w,
                        src_yw,
                        src_yh,
                        cw,
                        src_cw,
                        src_ch,
                        sub_w,
                        sub_h,
                    };
                    let l2_ch_rr = l2_lu_r / sub_h;
                    let l2_ch_cc = l2_lu_c / sub_w;

                    // Choose 16×16 vs four 8×8 by RD cost J = SSE + λ·bits. Both
                    // trials run in place on the real coder using cheap snapshots
                    // (scalar state + output length) rather than cloning the whole
                    // CABAC output buffer. Trial A's appended output bytes are saved
                    // so the region can be restored if the 16×16 trial wins (trial B
                    // rolls back to the pre-trial state and overwrites the region).
                    let mut chose_16 = false;
                    if rd16 {
                        // RD trials run in place on the real coder. The CABAC
                        // output buffer only grows by appending, so each trial is
                        // bounded by a cheap snapshot (scalar state + buffer length)
                        // and rolled back by truncating — no output-buffer clone.
                        // The context sets are small fixed stack arrays, so cloning
                        // them per trial costs no heap traffic.
                        let base_bits = cab.flushed_bits();
                        let base_snap = cab.snapshot();
                        let base_ctx = ctx.clone();
                        let base_ictx = ictx.clone();

                        // Trial A — single 16×16 CU (encoded in place).
                        cab.encode_bin(0, &mut ctx.split_cu_flag[split_ctx]);
                        code_one_cu(
                            Entropy {
                                enc: &mut cab,
                                ctx: &mut ctx,
                                ictx: &mut ictx,
                            },
                            yuv,
                            &mut rec_y,
                            &mut rec_cb,
                            &mut rec_cr,
                            l2_lu_r,
                            l2_lu_c,
                            16,
                            strides8,
                            qp,
                            &mut mode_map,
                            blk_stride,
                        );
                        let d_a = region_sse(
                            yuv, &rec_y, &rec_cb, &rec_cr, l2_lu_r, l2_lu_c, l2_ch_rr, l2_ch_cc,
                            strides8,
                        );
                        let bits_a = cab.flushed_bits().saturating_sub(base_bits) as f64;

                        // Snapshot trial A's coder + context so the winner can be
                        // restored after trial B overwrites them. Trial B is encoded
                        // by rolling the coder back to `base_snap` and re-running in
                        // place, which truncates away the output bytes trial A
                        // appended. Those bytes cannot be recovered by a later
                        // truncate (trial B may be shorter), so save trial A's
                        // output tail (one CU's worth) to splice back if A wins.
                        let a_snap = cab.snapshot();
                        let a_ctx = ctx.clone();
                        let a_ictx = ictx.clone();
                        let a_tail: Vec<u8> = cab.output[base_snap.output_len()..].to_vec();

                        // Snapshot trial A's reconstruction + mode_map for the region
                        // (16×16 luma, 8×8 chroma, 2×2 mode-map blocks).
                        let mut sa_y = [0u16; 256];
                        let mut sa_cb = [0u16; 64];
                        let mut sa_cr = [0u16; 64];
                        let mut sa_mode = [0u8; 4];
                        for r in 0..16 {
                            let o = (l2_lu_r + r) * w + l2_lu_c;
                            sa_y[r * 16..r * 16 + 16].copy_from_slice(&rec_y[o..o + 16]);
                        }
                        for r in 0..8 {
                            let o = (l2_ch_rr + r) * cw + l2_ch_cc;
                            sa_cb[r * 8..r * 8 + 8].copy_from_slice(&rec_cb[o..o + 8]);
                            sa_cr[r * 8..r * 8 + 8].copy_from_slice(&rec_cr[o..o + 8]);
                        }
                        for br in 0..2 {
                            for bc in 0..2 {
                                sa_mode[br * 2 + bc] = mode_map
                                    [((l2_lu_r / 8) + br) * blk_stride + (l2_lu_c / 8) + bc];
                            }
                        }

                        // Trial B — four 8×8 CUs. Roll the coder + contexts back to
                        // the pre-trial state and re-encode in place.
                        cab.restore(&base_snap);
                        ctx = base_ctx;
                        ictx = base_ictx;
                        cab.encode_bin(1, &mut ctx.split_cu_flag[split_ctx]);
                        for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                            code_one_cu(
                                Entropy {
                                    enc: &mut cab,
                                    ctx: &mut ctx,
                                    ictx: &mut ictx,
                                },
                                yuv,
                                &mut rec_y,
                                &mut rec_cb,
                                &mut rec_cr,
                                l2_lu_r + dy * cu_size_y,
                                l2_lu_c + dx * cu_size_y,
                                8,
                                strides8,
                                qp,
                                &mut mode_map,
                                blk_stride,
                            );
                        }
                        let d_b = region_sse(
                            yuv, &rec_y, &rec_cb, &rec_cr, l2_lu_r, l2_lu_c, l2_ch_rr, l2_ch_cc,
                            strides8,
                        );
                        let bits_b = cab.flushed_bits().saturating_sub(base_bits) as f64;

                        chose_16 = d_a + lambda * bits_a <= d_b + lambda * bits_b;
                        if chose_16 {
                            // Restore trial A's reconstruction + mode_map (trial B
                            // overwrote the region) and adopt trial A's coder state.
                            for r in 0..16 {
                                let o = (l2_lu_r + r) * w + l2_lu_c;
                                rec_y[o..o + 16].copy_from_slice(&sa_y[r * 16..r * 16 + 16]);
                            }
                            for r in 0..8 {
                                let o = (l2_ch_rr + r) * cw + l2_ch_cc;
                                rec_cb[o..o + 8].copy_from_slice(&sa_cb[r * 8..r * 8 + 8]);
                                rec_cr[o..o + 8].copy_from_slice(&sa_cr[r * 8..r * 8 + 8]);
                            }
                            for br in 0..2 {
                                for bc in 0..2 {
                                    mode_map
                                        [((l2_lu_r / 8) + br) * blk_stride + (l2_lu_c / 8) + bc] =
                                        sa_mode[br * 2 + bc];
                                    cu_depth[((l2_lu_r / cu_size_y) + br) * bx
                                        + (l2_lu_c / cu_size_y)
                                        + bc] = 2;
                                }
                            }
                            cab.reinstate_tail(&base_snap, &a_tail);
                            cab.restore(&a_snap);
                            ctx = a_ctx;
                            ictx = a_ictx;
                        } else {
                            // Trial B's reconstruction + mode_map + coder already in place.
                            for br in 0..2 {
                                for bc in 0..2 {
                                    cu_depth[((l2_lu_r / cu_size_y) + br) * bx
                                        + (l2_lu_c / cu_size_y)
                                        + bc] = 3;
                                }
                            }
                        }
                    } else {
                        // 4:2:2 / 4:4:4: always four 8×8 CUs (no 16×16 trial).
                        cab.encode_bin(1, &mut ctx.split_cu_flag[split_ctx]);
                        for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                            code_one_cu(
                                Entropy {
                                    enc: &mut cab,
                                    ctx: &mut ctx,
                                    ictx: &mut ictx,
                                },
                                yuv,
                                &mut rec_y,
                                &mut rec_cb,
                                &mut rec_cr,
                                l2_lu_r + dy * cu_size_y,
                                l2_lu_c + dx * cu_size_y,
                                8,
                                strides8,
                                qp,
                                &mut mode_map,
                                blk_stride,
                            );
                        }
                        for br in 0..2 {
                            for bc in 0..2 {
                                cu_depth[((l2_lu_r / cu_size_y) + br) * bx
                                    + (l2_lu_c / cu_size_y)
                                    + bc] = 3;
                            }
                        }
                    }
                    let _ = chose_16;
                }
            }

            let is_last_ctu = ctu_idx == total_ctus - 1;
            cab.encode_terminate(if is_last_ctu { 1 } else { 0 });
            ctu_idx += 1;
        }
    }

    let cabac_bytes = cab.finish();
    let mut nalu_data = header_bytes;
    nalu_data.extend_from_slice(&cabac_bytes);

    // In-loop deblocking filter (matches the decoder's post-decode filtering, so
    // the returned reconstruction equals a conformant decoder's output and block
    // edges are smoothed).
    // In-loop deblocking. Monochrome filters luma only.
    if yuv.chroma.is_monochrome() {
        crate::deblock::deblock_luma_only(&mut rec_y, w, h, qp_val, yuv.bit_depth);
    } else {
        crate::deblock::deblock(
            &mut rec_y,
            w,
            h,
            &mut rec_cb,
            &mut rec_cr,
            cw,
            ch,
            qp_val,
            yuv.bit_depth,
        );
    }

    Ok((
        Nalu {
            _nal_type: 20,
            data: nalu_data,
        },
        rec_y,
        rec_cb,
        rec_cr,
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

/// Encode one 8×8 luma CU + paired 4×4 chroma TUs (Cb+Cr).
///
/// HEVC intra CU syntax per §7.3.8.5/8.6/8.11:
///   [luma intra mode] [chroma intra mode] [cbf_cb] [cbf_cr] [cbf_luma]
///   [luma residual?] [Cb residual?] [Cr residual?]
#[allow(clippy::too_many_arguments)]
/// Build the 3-entry MPM candidate list from left (A) and above (B) modes,
/// per HEVC §8.4.2 (fillIntraPredModeCandidates).
/// Sum of absolute 4×4 Hadamard-transformed differences over an N×N block —
/// the standard fast distortion proxy for intra mode decision (correlates with
/// post-transform coded cost far better than raw SAD).
fn satd_block(orig: &[u16], pred: &[u16], n: usize) -> u32 {
    let mut total = 0u32;
    let mut d = [0i32; 16];
    for by in (0..n).step_by(4) {
        for bx in (0..n).step_by(4) {
            for r in 0..4 {
                let row = (by + r) * n + bx;
                let o = &orig[row..row + 4];
                let p = &pred[row..row + 4];
                let dr = &mut d[r * 4..r * 4 + 4];
                for c in 0..4 {
                    dr[c] = o[c] as i32 - p[c] as i32;
                }
            }
            // rows
            for r in 0..4 {
                let o = r * 4;
                let a0 = d[o] + d[o + 2];
                let a1 = d[o + 1] + d[o + 3];
                let a2 = d[o] - d[o + 2];
                let a3 = d[o + 1] - d[o + 3];
                d[o] = a0 + a1;
                d[o + 1] = a0 - a1;
                d[o + 2] = a2 + a3;
                d[o + 3] = a2 - a3;
            }
            // cols
            for c in 0..4 {
                let a0 = d[c] + d[8 + c];
                let a1 = d[4 + c] + d[12 + c];
                let a2 = d[c] - d[8 + c];
                let a3 = d[4 + c] - d[12 + c];
                d[c] = a0 + a1;
                d[4 + c] = a0 - a1;
                d[8 + c] = a2 + a3;
                d[12 + c] = a2 - a3;
            }
            let mut s = 0u32;
            for &v in d.iter() {
                s += v.unsigned_abs();
            }
            total += s.div_ceil(2);
        }
    }
    total
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

/// Decode-order availability for the block containing neighbour pixel (nr,nc),
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
    /// Stride (in 8×8 blocks) of the per-CU luma `mode_map` used for MPM.
    blk_stride: usize,
}

/// The three entropy-coding state objects, threaded together so the per-CU
/// coding functions take one argument instead of three. Holds mutable borrows
/// so callers keep ownership (the RD trials clone the underlying objects and
/// build a fresh bundle per trial).
struct Entropy<'a> {
    enc: &'a mut CabacEncoder,
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
    /// Luma CU/TU size: 8 (the min CB) or 16. Drives part_mode presence, the
    /// luma scan/transform size, and the chroma TB size.
    lu: usize,
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

fn encode_cu(
    ent: Entropy<'_>,
    src: &CuSrcPlanes<'_>,
    rec: &mut CuRecPlanes<'_>,
    geo: &CuGeometry,
    par: &CuParams,
    mode_map: &mut [u8],
) {
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
        blk_stride,
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
    // MPM candidates from neighbour modes (HEVC §8.4.2): candA = left, candB =
    // above (DC if unavailable or in a different CTB row). Modes come from the
    // per-block mode map written by previously coded CUs.
    let ctb = 64usize;
    let avail_left =
        lu_col > 0 && is_block_decoded(lu_row, lu_col - 1, lu_row, lu_col, ctb, yw_stride);
    let above_in_same_ctb = lu_row > 0 && ((lu_row - 1) >= (lu_row / ctb) * ctb);
    let avail_above = lu_row > 0
        && above_in_same_ctb
        && is_block_decoded(lu_row - 1, lu_col, lu_row, lu_col, ctb, yw_stride);
    let mode_at = |r: usize, c: usize| mode_map[(r / 8) * blk_stride + c / 8];
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
            neutral,
        },
    );

    // Rough mode decision: rank all 35 modes by SATD(orig − pred) + λ_m·mode_bits
    // and keep the cheapest. λ_m ≈ √λ matches the SATD (≈√SSE) domain.
    let y_orig_rmd = extract_block_dyn(src_y, src_yw, src_yh, lu_row, lu_col, lu);
    let lambda = 0.57_f64 * 2f64.powf((qp_slice as f64 - 12.0) / 3.0);
    let lambda_mode = lambda.sqrt();
    let mut best_mode = PLANAR;
    let mut best_cost = f64::INFINITY;
    // The smoothed references depend only on the block, not the mode, so compute
    // them once and reuse across all filtering modes instead of recomputing
    // the filter for each of the ~28 modes that smooth.
    let (fa, fl) = intra::filter_references(yc0, &ya, &yl, lu);
    let cf = ((ya[0] as i32 + 2 * yc0 as i32 + yl[0] as i32 + 2) >> 2) as u16;
    for mode in 0u8..35 {
        let pred = if intra::should_filter_refs(mode, lu) {
            match mode {
                PLANAR => intra::predict_planar(&fa, &fl, lu),
                DC => intra::predict_dc(&fa, &fl, lu, true),
                _ => intra::predict_angular(cf, &fa, &fl, lu, mode, true, max_val as i32),
            }
        } else {
            match mode {
                PLANAR => intra::predict_planar(&ya, &yl, lu),
                DC => intra::predict_dc(&ya, &yl, lu, true),
                _ => intra::predict_angular(yc0, &ya, &yl, lu, mode, true, max_val as i32),
            }
        };
        let satd = satd_block(&y_orig_rmd, &pred, lu) as f64;
        let mode_bits = if let Some(i) = mpm.iter().position(|&m| m == mode) {
            (1 + i + 1) as f64 // prev_flag=1 + mpm_idx unary
        } else {
            6.0 // prev_flag=0 + 5-bit rem
        };
        let cost = satd + lambda_mode * mode_bits;
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    let luma_mode = best_mode;
    let y_pred = if intra::should_filter_refs(luma_mode, lu) {
        match luma_mode {
            PLANAR => intra::predict_planar(&fa, &fl, lu),
            DC => intra::predict_dc(&fa, &fl, lu, true),
            _ => intra::predict_angular(cf, &fa, &fl, lu, luma_mode, true, max_val as i32),
        }
    } else {
        match luma_mode {
            PLANAR => intra::predict_planar(&ya, &yl, lu),
            DC => intra::predict_dc(&ya, &yl, lu, true),
            _ => intra::predict_angular(yc0, &ya, &yl, lu, luma_mode, true, max_val as i32),
        }
    };

    // ── part_mode ──────────────────────────────────────────────────────────
    // Present only when the CU equals the SPS minimum luma CB (8×8); we always
    // use PART_2Nx2N → single context bin = 1 when present.
    if lu == 8 {
        enc.encode_bin(1, &mut ictx.part_mode);
    }
    let _ = &ictx.part_mode;

    // ── Luma intra pred mode syntax ──────────────────────────────────────────
    if let Some(idx) = mpm.iter().position(|&m| m == luma_mode) {
        enc.encode_bin(1, &mut ictx.prev_intra_luma_pred_flag);
        // mpm_idx TR(cMax=2) bypass: 0→"0", 1→"10", 2→"11".
        match idx {
            0 => {
                enc.encode_bypass(0);
            }
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
        // rem_intra_luma_pred_mode (5-bit FL). The decoder reconstructs the mode
        // by adding 1 for each sorted MPM <= the running value, so the inverse is
        // rem = luma_mode − (number of MPM candidates strictly less than it).
        let mut rem = luma_mode as i32;
        for &m in mpm.iter() {
            if (m as i32) < luma_mode as i32 {
                rem -= 1;
            }
        }
        for i in (0..5).rev() {
            enc.encode_bypass(((rem >> i) & 1) as u8);
        }
    }

    // Record this CU's luma mode for neighbours' MPM derivation.
    for br in 0..(lu / 8) {
        for bc in 0..(lu / 8) {
            mode_map[((lu_row / 8) + br) * blk_stride + (lu_col / 8) + bc] = luma_mode;
        }
    }

    // Chroma mode = luma mode (DM_CHROMA), with the 4:2:2 remap (HEVC Table 8-3).
    let chroma_mode = if chroma.sub_w() == 2 && chroma.sub_h() == 1 {
        MODE_422_MAP[luma_mode as usize]
    } else {
        luma_mode
    };

    // ── Chroma intra pred mode (DM_CHROMA → single '0' bin) ──────────────
    if !chroma.is_monochrome() {
        enc.encode_bin(0, &mut ictx.intra_chroma_pred_mode);
    }

    // ── Chroma prediction + transform + quantise ─────────────────────────────
    // Chroma TB size depends on format: 4:2:0/4:2:2 use 4×4 TBs; 4:4:4 uses 8×8.
    // 4:2:2 stacks two 4×4 TBs vertically. HEVC predicts each chroma TB separately
    // (§8.4.4.2.1); a lower stacked TB uses the just-reconstructed upper TB as its
    // above reference. We predict → transform → reconstruct each TB in order.
    let chroma_qp = chroma_qp_for(qp_slice, chroma) + qp_bd_offset;
    let sub_w = chroma.sub_w();
    let sub_h = chroma.sub_h();
    let luma_ctus_x = yw_stride / 64;
    // Chroma TB size derives from the luma CU size and subsampling: LU/sub_w.
    // For LU=8 this reproduces chroma_tb_size() (4:2:0/4:2:2→4, 4:4:4→8); for
    // LU=16 in 4:2:0 it gives an 8×8 chroma TB.
    let ctb = lu / sub_w; // chroma TB side (4, or 8 for 16×16-luma 4:2:0)
    let log2_ctb = ctb.trailing_zeros(); // 2 or 3
    // Mode-dependent scan for chroma TBs: 4×4 chroma uses vertical/horizontal for
    // chroma modes 6..=14 / 22..=30 (else diagonal); 8×8 chroma (4:4:4) is always
    // diagonal. Must match the scan_idx passed to encode_residual for chroma.
    let is_444 = matches!(chroma, crate::fmt::ChromaFormat::Yuv444);
    let chroma_tb_scan_idx = dct::scan_idx_for(chroma_mode, log2_ctb, false, is_444);
    let chroma_scan: &[(usize, usize)] = dct::coeff_scan(log2_ctb, chroma_tb_scan_idx);

    #[derive(Clone, Copy)]
    struct ChromaTb {
        cb_zz: [i16; 64],
        cb_nz: bool,
        cr_zz: [i16; 64],
        cr_nz: bool,
    }

    // n_chroma_tb is 1 (4:2:0 / 4:4:4) or 2 (4:2:2) — keep the TBs on the stack.
    let mut tbs = [ChromaTb {
        cb_zz: [0i16; 64],
        cb_nz: false,
        cr_zz: [0i16; 64],
        cr_nz: false,
    }; 2];
    for (t, tbs) in tbs[..n_chroma_tb].iter_mut().enumerate() {
        let sub_ch_row = ch_row + t * ctb;
        // Predict this chroma TB (DM_CHROMA → chroma_mode). 4:4:4 (8×8) smooths
        // references like luma when the mode calls for it; 4:2:0/4:2:2 (4×4) do not.
        let filt = ctb > 4 && intra::should_filter_refs(chroma_mode, ctb);
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
                cur_luma_row: lu_row,
                cur_luma_col: lu_col,
                neutral,
            },
        );
        // When chroma references are smoothed (4:4:4, 8×8), libde265 filters the
        // corner too (pF[0] = (above[0]+2·corner+left[0]+2)>>2), so pass the
        // filtered corner — matching the luma path.
        let (baf, blf) = if filt {
            intra::filter_references(bc0, &ba, &bl, ctb)
        } else {
            (ba, bl)
        };
        let bcf = if filt {
            ((ba[0] as i32 + 2 * bc0 as i32 + bl[0] as i32 + 2) >> 2) as u16
        } else {
            bc0
        };
        let cb_pred = intra::predict_chroma_tb(chroma_mode, bcf, &baf, &blf, ctb, max_val as i32);
        let (raf, rlf) = if filt {
            intra::filter_references(rc0, &ra, &rl, ctb)
        } else {
            (ra, rl)
        };
        let rcf = if filt {
            ((ra[0] as i32 + 2 * rc0 as i32 + rl[0] as i32 + 2) >> 2) as u16
        } else {
            rc0
        };
        let cr_pred = intra::predict_chroma_tb(chroma_mode, rcf, &raf, &rlf, ctb, max_val as i32);
        let b_orig = extract_block_dyn(src_cb, src_cw, src_ch, sub_ch_row, ch_col, ctb);
        let r_orig = extract_block_dyn(src_cr, src_cw, src_ch, sub_ch_row, ch_col, ctb);
        let n_ch = ctb * ctb;
        let mut b_res = [0i32; 64];
        let mut r_res = [0i32; 64];
        for (d, (&o, &p)) in b_res[..n_ch]
            .iter_mut()
            .zip(b_orig[..n_ch].iter().zip(&cb_pred[..n_ch]))
        {
            *d = o as i32 - p as i32;
        }
        for (d, (&o, &p)) in r_res[..n_ch]
            .iter_mut()
            .zip(r_orig[..n_ch].iter().zip(&cr_pred[..n_ch]))
        {
            *d = o as i32 - p as i32;
        }
        let cb_level = crate::hevc_transform::quantize(
            &crate::hevc_transform::fwd_transform(&b_res, ctb, bit_depth.bits()),
            ctb,
            chroma_qp,
            bit_depth.bits(),
        );
        let cr_level = crate::hevc_transform::quantize(
            &crate::hevc_transform::fwd_transform(&r_res, ctb, bit_depth.bits()),
            ctb,
            chroma_qp,
            bit_depth.bits(),
        );
        let mut cb_zz = [0i16; 64];
        let mut cr_zz = [0i16; 64];
        for (&(r, c), dst) in chroma_scan.iter().zip(cb_zz.iter_mut()) {
            *dst = cb_level[r * ctb + c];
        }
        for (&(r, c), dst) in chroma_scan.iter().zip(cr_zz.iter_mut()) {
            *dst = cr_level[r * ctb + c];
        }
        let cb_nz = cb_zz.iter().any(|&x| x != 0);
        let cr_nz = cr_zz.iter().any(|&x| x != 0);

        // Reconstruct this TB so the next stacked TB (4:2:2) sees it as a reference.
        let b_dq = crate::hevc_transform::dequantize(&cb_level, ctb, chroma_qp, bit_depth.bits());
        let b_inv = crate::hevc_transform::inv_transform(&b_dq, ctb, bit_depth.bits());
        let b_rec = intra::reconstruct(&cb_pred, &b_inv, ctb, max_val);
        let r_dq = crate::hevc_transform::dequantize(&cr_level, ctb, chroma_qp, bit_depth.bits());
        let r_inv = crate::hevc_transform::inv_transform(&r_dq, ctb, bit_depth.bits());
        let r_rec = intra::reconstruct(&cr_pred, &r_inv, ctb, max_val);
        for r in 0..ctb {
            for c in 0..ctb {
                let (row, col) = (sub_ch_row + r, ch_col + c);
                if row < coded_ch_h && col < cw_stride {
                    rec_cb[row * cw_stride + col] = b_rec[r * ctb + c];
                    rec_cr[row * cw_stride + col] = r_rec[r * ctb + c];
                }
            }
        }

        *tbs = ChromaTb {
            cb_zz,
            cb_nz,
            cr_zz,
            cr_nz,
        };
    }

    // ── HEVC integer transform + quantize: luma LU×LU ─────────────────────
    // Const-generic extractor so the copy is specialized for each size.
    let y_orig = match lu {
        16 => extract_block_n::<16>(src_y, src_yw, src_yh, lu_row, lu_col),
        _ => extract_block_n::<8>(src_y, src_yw, src_yh, lu_row, lu_col),
    };
    let y_res = intra::compute_residual_i32(&y_orig, &y_pred, lu);
    let y_tcoeff = crate::hevc_transform::fwd_transform(&y_res[..lu * lu], lu, bit_depth.bits());
    let y_level = crate::hevc_transform::quantize(&y_tcoeff, lu, qp, bit_depth.bits()); // row-major levels
    // Reorder row-major levels into the mode-dependent scan for residual_coding.
    // 8×8 luma uses a vertical/horizontal scan for modes 6..=14 / 22..=30 (else
    // diagonal); 16×16 is always diagonal (HEVC §6.5).
    let luma_log2_ts: u32 = if lu == 16 { 4 } else { 3 };
    let luma_scan_idx = dct::scan_idx_for(luma_mode, luma_log2_ts, true, false);
    let luma_scan = dct::coeff_scan(luma_log2_ts, luma_scan_idx);
    let y_zigzag: Vec<i16> = luma_scan
        .iter()
        .map(|&(r, c)| y_level[r * lu + c])
        .collect();
    let y_nz = y_zigzag.iter().any(|&x| x != 0);

    // ── CABAC: transform_tree() syntax ─────────────────────────────────────
    // split_transform_flag is inferred 0 (max_transform_hierarchy_depth_intra = 0),
    // so it is not coded — matching x265's parsing.
    //
    // cbf order (HEVC §7.3.8.8): for ChromaArrayType==2 (4:2:2) the two stacked
    // chroma TBs each have their own cbf, signalled cb[0],cb[1] then cr[0],cr[1],
    // before cbf_luma. For 4:2:0 there is one of each.
    for t in &tbs[..n_chroma_tb] {
        encode_cbf_chroma(enc, ctx, t.cb_nz, 0);
    }
    for t in &tbs[..n_chroma_tb] {
        encode_cbf_chroma(enc, ctx, t.cr_nz, 0);
    }
    encode_cbf_luma(enc, ctx, y_nz, 0);

    // ── CABAC: residuals (HEVC §7.3.8.11) ─────────────────────────────────
    // Order: luma, then all Cb chroma TBs, then all Cr chroma TBs (component-major).
    // Chroma TB size is log2_ctb (2 for 4:2:0/4:2:2, 3 for 4:4:4).
    if y_nz {
        encode_residual(enc, ctx, &y_zigzag, luma_log2_ts, true, luma_scan_idx);
    }
    let chroma_scan_idx = chroma_tb_scan_idx;
    for t in &tbs[..n_chroma_tb] {
        if t.cb_nz {
            encode_residual(enc, ctx, &t.cb_zz, log2_ctb, false, chroma_scan_idx);
        }
    }
    for t in &tbs[..n_chroma_tb] {
        if t.cr_nz {
            encode_residual(enc, ctx, &t.cr_zz, log2_ctb, false, chroma_scan_idx);
        }
    }

    // ── Reconstruct luma (integer dequant + inverse transform) ─────────────
    let y_dq = crate::hevc_transform::dequantize(&y_level, lu, qp, bit_depth.bits());
    let y_res_rec = crate::hevc_transform::inv_transform(&y_dq, lu, bit_depth.bits());
    let y_rec = intra::reconstruct(&y_pred, &y_res_rec, lu, max_val);
    for r in 0..lu {
        for c in 0..lu {
            let (row, col) = (lu_row + r, lu_col + c);
            if row < coded_yh && col < yw_stride {
                rec_y[row * yw_stride + col] = y_rec[r * lu + c];
            }
        }
    }
    // Chroma was already reconstructed into rec_cb/rec_cr inside the per-sub-TB loop
    // above (so each stacked sub-TB could serve as the next one's intra reference).
}

/// Encode one intra CU (luma side `lu` = 8 or 16) at (lu_row,lu_col) into the
/// bitstream and reconstruction planes; chroma coords derive via subsampling.
/// Shared by the RD trial and commit paths.
#[allow(clippy::too_many_arguments)]
fn code_one_cu(
    ent: Entropy<'_>,
    yuv: &Yuv,
    rec_y: &mut [u16],
    rec_cb: &mut [u16],
    rec_cr: &mut [u16],
    lu_row: usize,
    lu_col: usize,
    lu: usize,
    strides: PlaneStrides,
    qp: u8,
    mode_map: &mut [u8],
    blk_stride: usize,
) {
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
        blk_stride,
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
    };
    encode_cu(ent, &src, &mut rec, &geo, &par, mode_map);
}

/// RD distortion: SSE (luma 16×16 + chroma 8×8 Cb/Cr) between source and
/// reconstruction over one 16×16-luma CU region, with the encoder's edge
/// clamping for padded regions. Differences are integer-valued samples, so the
/// sum of squares is computed exactly in `i64` (returned as `f64` for the
/// caller's Lagrangian, which mixes it with λ·bits).
#[allow(clippy::too_many_arguments)]
fn region_sse(
    yuv: &Yuv,
    rec_y: &[u16],
    rec_cb: &[u16],
    rec_cr: &[u16],
    lu_row: usize,
    lu_col: usize,
    ch_row: usize,
    ch_col: usize,
    strides: PlaneStrides,
) -> f64 {
    let PlaneStrides {
        w,
        src_yw,
        src_yh,
        cw,
        src_cw,
        src_ch,
        ..
    } = strides;
    let mut sse: i64 = 0;
    for r in 0..16 {
        let sy = (lu_row + r).min(src_yh - 1);
        for c in 0..16 {
            let sx = (lu_col + c).min(src_yw - 1);
            let s = yuv.y[sy * src_yw + sx] as i64;
            let d = rec_y[(lu_row + r) * w + (lu_col + c)] as i64;
            let e = s - d;
            sse += e * e;
        }
    }
    for r in 0..8 {
        let sy = (ch_row + r).min(src_ch - 1);
        for c in 0..8 {
            let sx = (ch_col + c).min(src_cw - 1);
            let sb = yuv.cb[sy * src_cw + sx] as i64;
            let db = rec_cb[(ch_row + r) * cw + (ch_col + c)] as i64;
            let eb = sb - db;
            sse += eb * eb;
            let sr = yuv.cr[sy * src_cw + sx] as i64;
            let dr = rec_cr[(ch_row + r) * cw + (ch_col + c)] as i64;
            let er = sr - dr;
            sse += er * er;
        }
    }
    sse as f64
}

/// Extract an N×N block from a plane (compile-time N, so the copy is specialized
/// per size). Returns a fixed 256-entry buffer (N up to 16); first N×N written.
fn extract_block_n<const N: usize>(
    plane: &[u16],
    src_w: usize,
    src_h: usize,
    row: usize,
    col: usize,
) -> [u16; 256] {
    let mut out = [128u16; 256];
    for r in 0..N {
        for c in 0..N {
            out[r * N + c] = plane[(row + r).min(src_h - 1) * src_w + (col + c).min(src_w - 1)];
        }
    }
    out
}

/// Extract an n×n block from a plane (runtime size). Used for chroma where the TB
/// side is 4 (4:2:0/4:2:2) or 8 (4:4:4).
fn extract_block_dyn(
    plane: &[u16],
    src_w: usize,
    src_h: usize,
    row: usize,
    col: usize,
    n: usize,
) -> [u16; 256] {
    let mut out = [128u16; 256];
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = plane[(row + r).min(src_h - 1) * src_w + (col + c).min(src_w - 1)];
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        );
        assert!(sps.data.len() > 10);
    }

    #[test]
    fn pps_builds_cleanly() {
        let pps = build_pps(30);
        assert_eq!(pps.data[0], 0x44, "PPS first byte should be 0x44");
    }
}
