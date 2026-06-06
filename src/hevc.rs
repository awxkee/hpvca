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

//! HEVC bitstream encoder — parameter sets, slice header, CABAC slice data.
//!
//! Produces a conformant HEVC still-picture bitstream:
//!   - VPS  NAL type 32  (fixed, one-layer, one-temporal-layer)
//!   - SPS  NAL type 33  (Main profile, level 3.1, 8-bit 4:2:0, conformance window)
//!   - PPS  NAL type 34  (minimal, cu_qp_delta disabled)
//!   - IDR  NAL type 19  (intra-only, CABAC, DC/Planar per CU, reconstruct loop)

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
    #[allow(unused)]
    pub(crate) nal_type: u8,
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

// ─── Bit writer ───────────────────────────────────────────────────────────────

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

// ─── NALU header ─────────────────────────────────────────────────────────────

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
    const TABLE: &[(u64, u8)] = &[
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
    // Profile/Tier/Level, matched to x265's RExt output. The RExt constraint flags
    // must be consistent with the actual chroma format: a 4:2:2 stream must NOT set
    // max_420chroma_constraint_flag.
    bw.write_bits(0, 2); // general_profile_space = 0
    bw.write_bit(false); // general_tier_flag = 0 (Main tier)
    bw.write_bits(4, 5); // general_profile_idc = 4 (Format Range Extensions)

    bw.write_bits(0x0800_0000, 32); // compatibility: bit 4 (RExt)

    bw.write_bit(true); // general_progressive_source_flag   = 1
    bw.write_bit(false); // general_interlaced_source_flag    = 0
    bw.write_bit(false); // general_non_packed_constraint_flag= 0
    bw.write_bit(true); // general_frame_only_constraint_flag= 1
    // RExt bit-depth constraint flags. Each `max_Nbit` flag asserts the stream uses
    // at most N bits, so it is set when bit_depth <= N (HEVC Annex A).
    let bits = bit_depth.bits();
    bw.write_bit(bits <= 12); // max_12bit_constraint_flag
    bw.write_bit(bits <= 10); // max_10bit_constraint_flag
    bw.write_bit(bits <= 8); // max_8bit_constraint_flag
    // For monochrome, the stream satisfies all the chroma constraints (max_422,
    // max_420 are true) and additionally max_monochrome = 1.
    let is_444 = matches!(chroma, crate::fmt::ChromaFormat::Yuv444);
    let is_420 = matches!(chroma, crate::fmt::ChromaFormat::Yuv420);
    let is_mono = matches!(chroma, crate::fmt::ChromaFormat::Monochrome);
    bw.write_bit(!is_444 || is_mono); // max_422chroma_constraint_flag
    bw.write_bit(is_420 || is_mono); // max_420chroma_constraint_flag
    bw.write_bit(is_mono); // max_monochrome_constraint_flag
    bw.write_bit(true); // intra_constraint_flag        = 1
    bw.write_bit(false); // one_picture_only_constraint_flag = 0
    bw.write_bit(true); // lower_bit_rate_constraint_flag   = 1
    bw.write_bits(0, 32); // reserved
    bw.write_bits(0, 3); // reserved (35 reserved bits total after 13 constraint bits)

    bw.write_bits(level_idc as u32, 8);
}

// ─── VPS ─────────────────────────────────────────────────────────────────────

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
    bw.write_ue(2); // vps_max_dec_pic_buffering_minus1[0] = 2 → DPB 3 (matches SPS)
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
        nal_type: 32,
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
    bw.write_ue(2); // sps_max_dec_pic_buffering_minus1[0] = 2 → DPB 3 (matches x265;
    // hardware decoders allocate the DPB from this and may reject a
    // value smaller than they expect for the pipeline).
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

    bw.write_bit(false); // sps_extension_present_flag

    bw.rbsp_trailing_bits();
    Nalu {
        nal_type: 33,
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
    // libheif uses full range. Our YUV conversion
    // produces studio-swing Y [16-235], but VideoToolbox
    // on macOS ignores limited-range signals and clips to
    // black. Signalling full range matches libheif and
    // makes the image display correctly on Apple devices.
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

// ─── PPS ─────────────────────────────────────────────────────────────────────

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
        nal_type: 34,
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

// /// Encode and also return the encoder's internal reconstruction (coded dimensions).
// /// Intended for validation: the reconstruction is exactly what a matching decoder
// /// produces, so comparing it to the source measures encode quality without any
// /// external decoder.
// pub(crate) fn encode_intra_with_recon(
//     yuv: &Yuv,
//     width: u32,
//     height: u32,
//     quality: u8,
// ) -> Result<(NaluStream, Vec<u16>, Vec<u16>, Vec<u16>), EncodeError> {
//     let vps = build_vps(width, height, yuv.chroma, yuv.bit_depth);
//     let sps = build_sps(width, height, yuv.chroma, yuv.bit_depth);
//     let qp_val: u8 = ((100 - quality.clamp(1, 100) as u32) * 41 / 99 + 10).min(51) as u8;
//     let pps = build_pps(qp_val);
//     let (idr, ry, rcb, rcr) = build_idr_slice(yuv, width, height, quality)?;
//     Ok((
//         NaluStream {
//             nalus: vec![vps, sps, pps, idr],
//         },
//         ry,
//         rcb,
//         rcr,
//     ))
// }

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
    hdr.rbsp_trailing_bits();
    let header_bytes = hdr.finish();

    // ── CABAC slice data ─────────────────────────────────────────────────────
    // HEVC slice_segment_data(): for each CTU row-major:
    //   coding_tree_unit()  → luma 8×8 CU + chroma 4×4 CU×2
    //   end_of_slice_segment_flag (terminate = 0 or 1)
    let qp: u8 = qp_val;
    let mut cab = CabacEncoder::new();
    let mut ctx = ContextSet::init_islice(qp);
    let mut ictx = IntraModeContexts::init_islice(qp);

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
                    cab.encode_bin(1, &mut ctx.split_cu_flag[cl2 + ca2]);

                    for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                        let lu_row = l2_lu_r + dy * cu_size_y;
                        let lu_col = l2_lu_c + dx * cu_size_y;
                        // Chroma coordinates derive from luma via the subsampling
                        // factors: column always /sub_w (=2); row /sub_h (4:2:0 → /2,
                        // 4:2:2 → /1, i.e. full-height chroma).
                        let ch_row = lu_row / sub_h;
                        let ch_col = lu_col / sub_w;
                        let _ = (l2_ch_r, l2_ch_c, cu_size_c); // superseded by derivation

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
                        };

                        let src = CuSrcPlanes {
                            y: &yuv.y,
                            cb: &yuv.cb,
                            cr: &yuv.cr,
                            src_yw,
                        };

                        let mut rec = CuRecPlanes {
                            y: &mut rec_y,
                            cb: &mut rec_cb,
                            cr: &mut rec_cr,
                        };

                        let par = CuParams {
                            qp,
                            chroma: yuv.chroma,
                            bit_depth: yuv.bit_depth,
                        };

                        encode_cu(&mut cab, &mut ctx, &mut ictx, &src, &mut rec, &geo, &par);
                    }
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
            nal_type: 20,
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
        for c in 0..dst_w {
            let sc = c.min(src_w - 1);
            out[r * dst_w + c] = src[sr * src_w + sc];
        }
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
}

fn encode_cu(
    enc: &mut CabacEncoder,
    ctx: &mut ContextSet,
    ictx: &mut IntraModeContexts,
    src: &CuSrcPlanes<'_>,
    rec: &mut CuRecPlanes<'_>,
    geo: &CuGeometry,
    par: &CuParams,
) {
    // destructure so the rest of the body is unchanged
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
    } = *par;
    const LU: usize = 8; // luma block size
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

    // ── Luma intra prediction ───────────────────────────────────────────────
    // Always use PLANAR. With PLANAR everywhere, every block's neighbours are
    // PLANAR (or unavailable→treated as DC), so the spec MPM derivation always
    // yields candidate list [PLANAR, DC, VERTICAL] with PLANAR at index 0. That
    // makes mpm_idx = 0 correct for every block, independent of position — no
    // neighbour-mode tracking needed and no risk of an MPM-index/scan mismatch.
    let (yc0, ya, yl) = intra::get_reference_samples(
        rec_y,
        yw_stride,
        lu_row,
        lu_col,
        coded_yh,
        LU,
        64,
        yw_stride / 64,
        neutral,
    );
    // Luma 8×8 PLANAR uses the smoothed ([1 2 1]/4) reference (HEVC §8.4.4.2.3).
    let (yaf, ylf) = intra::filter_references(yc0, &ya, &yl, LU);
    let y_pred = intra::predict_planar(&yaf, &ylf, LU);

    // ── part_mode ──────────────────────────────────────────────────────────
    // Our 8×8 CU equals the SPS minimum luma CB size (log2_min=3), so the spec
    // requires part_mode here. We always use PART_2Nx2N → single context bin = 1.
    enc.encode_bin(1, &mut ictx.part_mode);
    let _ = &ictx.part_mode;

    // ── Luma intra pred mode syntax (prev_intra_luma_pred_flag + mpm_idx) ──
    // Every block uses PLANAR (mode 0). The MPM candidate list depends on the
    // neighbour-derived candidates A (left) and B (above) per HEVC §8.4.2, so
    // PLANAR is not always at mpm_idx 0 — we must locate it in the real list.
    //
    // candA: DC if left neighbour unavailable, else its mode (PLANAR here).
    // candB: DC if above unavailable OR above lies in a different CTB row, else
    //        its mode (PLANAR here).
    let ctb = 64usize;
    let avail_left =
        lu_col > 0 && is_block_decoded(lu_row, lu_col - 1, lu_row, lu_col, ctb, yw_stride);
    let above_in_same_ctb = lu_row > 0 && ((lu_row - 1) >= (lu_row / ctb) * ctb);
    let avail_above = lu_row > 0
        && above_in_same_ctb
        && is_block_decoded(lu_row - 1, lu_col, lu_row, lu_col, ctb, yw_stride);
    // All decoded neighbours are PLANAR (0); unavailable/cross-CTB → DC (1).
    const PLANAR: u8 = 0;
    const DC: u8 = 1;
    let cand_a = if avail_left { PLANAR } else { DC };
    let cand_b = if avail_above { PLANAR } else { DC };
    let mpm = mpm_list(cand_a, cand_b);
    let planar_idx = mpm.iter().position(|&m| m == PLANAR);

    if let Some(idx) = planar_idx {
        // PLANAR is in the MPM list: prev_flag=1, then mpm_idx (truncated unary, cMax 2).
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
        // PLANAR not an MPM (cannot happen here since one cand is always DC and
        // the list always contains PLANAR), fall back to rem_intra coding.
        enc.encode_bin(0, &mut ictx.prev_intra_luma_pred_flag);
        // rem_intra_luma_pred_mode: 5-bit FL of PLANAR after removing MPMs.
        let mut sorted = mpm;
        sorted.sort_unstable();
        let mut rem = PLANAR as i32;
        for &m in sorted.iter() {
            if (m as i32) <= rem {
                rem += 1;
            }
        }
        for i in (0..5).rev() {
            enc.encode_bypass(((rem >> i) & 1) as u8);
        }
    }

    // ── Chroma intra pred mode (DM_CHROMA → single '0' bin) ──────────────
    // Present only when ChromaArrayType != 0 (HEVC §7.3.8.5). DM_CHROMA means chroma
    // uses the luma mode = PLANAR, so we predict chroma with PLANAR.
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
    let ctb = chroma.chroma_tb_size(); // 4 or 8
    let log2_ctb = ctb.trailing_zeros(); // 2 or 3
    // Diagonal scan for this chroma TB size (4×4 or 8×8).
    let chroma_scan: &[(usize, usize)] = if ctb == 4 {
        &dct::DIAG_SCAN_4X4
    } else {
        &dct::ZIGZAG
    };

    struct ChromaTb {
        cb_zz: Vec<i16>,
        cb_nz: bool,
        cr_zz: Vec<i16>,
        cr_nz: bool,
    }

    let mut tbs: Vec<ChromaTb> = Vec::with_capacity(n_chroma_tb);
    for t in 0..n_chroma_tb {
        let sub_ch_row = ch_row + t * ctb;
        // Predict this chroma TB (PLANAR, DM_CHROMA). Availability follows the luma
        // decode order; a lower stacked TB sees the reconstructed upper TB.
        // For 4:4:4 (ChromaArrayType==3) the 8×8 chroma reference is smoothed with the
        // [1 2 1]/4 filter exactly like luma (HEVC §8.4.4.2.3); 4×4 chroma is not.
        let filt = ctb > 4; // true only for 4:4:4 8×8 chroma
        let (bc0, ba, bl) = intra::get_reference_samples_chroma(
            rec_cb,
            cw_stride,
            sub_ch_row,
            ch_col,
            coded_ch_h,
            ctb,
            sub_w,
            sub_h,
            yw_stride,
            coded_yh,
            luma_ctus_x,
            lu_row,
            lu_col,
            neutral,
        );
        let (baf, blf) = if filt {
            intra::filter_references(bc0, &ba, &bl, ctb)
        } else {
            (ba, bl)
        };
        let cb_pred = intra::predict_planar(&baf, &blf, ctb);
        let (rc0, ra, rl) = intra::get_reference_samples_chroma(
            rec_cr,
            cw_stride,
            sub_ch_row,
            ch_col,
            coded_ch_h,
            ctb,
            sub_w,
            sub_h,
            yw_stride,
            coded_yh,
            luma_ctus_x,
            lu_row,
            lu_col,
            neutral,
        );
        let (raf, rlf) = if filt {
            intra::filter_references(rc0, &ra, &rl, ctb)
        } else {
            (ra, rl)
        };
        let cr_pred = intra::predict_planar(&raf, &rlf, ctb);

        let b_orig = extract_block_dyn(src_cb, src_cw, src_ch, sub_ch_row, ch_col, ctb);
        let r_orig = extract_block_dyn(src_cr, src_cw, src_ch, sub_ch_row, ch_col, ctb);
        let b_res: Vec<i32> = b_orig
            .iter()
            .zip(&cb_pred)
            .map(|(&o, &p)| o as i32 - p as i32)
            .collect();
        let r_res: Vec<i32> = r_orig
            .iter()
            .zip(&cr_pred)
            .map(|(&o, &p)| o as i32 - p as i32)
            .collect();
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
        let cb_zz: Vec<i16> = chroma_scan
            .iter()
            .map(|&(r, c)| cb_level[r * ctb + c])
            .collect();
        let cr_zz: Vec<i16> = chroma_scan
            .iter()
            .map(|&(r, c)| cr_level[r * ctb + c])
            .collect();
        let cb_nz = cb_zz.iter().any(|&x| x != 0);
        let cr_nz = cr_zz.iter().any(|&x| x != 0);

        // Reconstruct this TB so the next stacked TB (4:2:2) sees it as a reference.
        let b_dq = crate::hevc_transform::dequantize(&cb_level, ctb, chroma_qp, bit_depth.bits());
        let b_rec_f: Vec<f32> = crate::hevc_transform::inv_transform(&b_dq, ctb, bit_depth.bits())
            .iter()
            .map(|&v| v as f32)
            .collect();
        let b_rec = intra::reconstruct(&cb_pred, &b_rec_f, ctb, max_val);
        let r_dq = crate::hevc_transform::dequantize(&cr_level, ctb, chroma_qp, bit_depth.bits());
        let r_rec_f: Vec<f32> = crate::hevc_transform::inv_transform(&r_dq, ctb, bit_depth.bits())
            .iter()
            .map(|&v| v as f32)
            .collect();
        let r_rec = intra::reconstruct(&cr_pred, &r_rec_f, ctb, max_val);
        for r in 0..ctb {
            for c in 0..ctb {
                let (row, col) = (sub_ch_row + r, ch_col + c);
                if row < coded_ch_h && col < cw_stride {
                    rec_cb[row * cw_stride + col] = b_rec[r * ctb + c];
                    rec_cr[row * cw_stride + col] = r_rec[r * ctb + c];
                }
            }
        }

        tbs.push(ChromaTb {
            cb_zz,
            cb_nz,
            cr_zz,
            cr_nz,
        });
    }

    // ── HEVC integer transform + quantize: luma 8×8 ───────────────────────
    let y_orig = extract_block_n::<LU>(src_y, src_yw, src_yh, lu_row, lu_col);
    let y_res = intra::compute_residual(&y_orig, &y_pred, LU);
    let y_res_i: Vec<i32> = y_res.iter().map(|&v| v as i32).collect();
    let y_tcoeff = crate::hevc_transform::fwd_transform(&y_res_i, LU, bit_depth.bits());
    let y_level = crate::hevc_transform::quantize(&y_tcoeff, LU, qp, bit_depth.bits()); // row-major levels
    // Reorder row-major levels into HEVC diagonal scan order for residual_coding.
    let y_zigzag: Vec<i16> = dct::ZIGZAG
        .iter()
        .map(|&(r, c)| y_level[r * LU + c])
        .collect();
    let y_nz = y_zigzag.iter().any(|&x| x != 0);

    // ── CABAC: transform_tree() syntax ─────────────────────────────────────
    // split_transform_flag is inferred 0 (max_transform_hierarchy_depth_intra = 0),
    // so it is not coded — matching x265's parsing.
    //
    // cbf order (HEVC §7.3.8.8): for ChromaArrayType==2 (4:2:2) the two stacked
    // chroma TBs each have their own cbf, signalled cb[0],cb[1] then cr[0],cr[1],
    // before cbf_luma. For 4:2:0 there is one of each.
    for t in &tbs {
        encode_cbf_chroma(enc, ctx, t.cb_nz, 0);
    }
    for t in &tbs {
        encode_cbf_chroma(enc, ctx, t.cr_nz, 0);
    }
    // cbf_luma at trafoDepth=0 (intra, single TU → always coded)
    encode_cbf_luma(enc, ctx, y_nz, 0);

    // ── CABAC: residuals (HEVC §7.3.8.11) ─────────────────────────────────
    // Order: luma, then all Cb chroma TBs, then all Cr chroma TBs (component-major).
    // Chroma TB size is log2_ctb (2 for 4:2:0/4:2:2, 3 for 4:4:4).
    if y_nz {
        encode_residual(enc, ctx, &y_zigzag, 3, true);
    }
    for t in &tbs {
        if t.cb_nz {
            encode_residual(enc, ctx, &t.cb_zz, log2_ctb, false);
        }
    }
    for t in &tbs {
        if t.cr_nz {
            encode_residual(enc, ctx, &t.cr_zz, log2_ctb, false);
        }
    }

    // ── Reconstruct luma (integer dequant + inverse transform) ─────────────
    let y_dq = crate::hevc_transform::dequantize(&y_level, LU, qp, bit_depth.bits());
    let y_res_rec = crate::hevc_transform::inv_transform(&y_dq, LU, bit_depth.bits());
    let y_res_rec_f: Vec<f32> = y_res_rec.iter().map(|&v| v as f32).collect();
    let y_rec = intra::reconstruct(&y_pred, &y_res_rec_f, LU, max_val);
    for r in 0..LU {
        for c in 0..LU {
            let (row, col) = (lu_row + r, lu_col + c);
            if row < coded_yh && col < yw_stride {
                rec_y[row * yw_stride + col] = y_rec[r * LU + c];
            }
        }
    }
    // Chroma was already reconstructed into rec_cb/rec_cr inside the per-sub-TB loop
    // above (so each stacked sub-TB could serve as the next one's intra reference).
}

/// Extract an N×N block from a plane (compile-time N via const generic).
fn extract_block_n<const N: usize>(
    plane: &[u16],
    src_w: usize,
    src_h: usize,
    row: usize,
    col: usize,
) -> Vec<u16> {
    let mut out = vec![128u16; N * N];
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
) -> Vec<u16> {
    let mut out = vec![128u16; n * n];
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
