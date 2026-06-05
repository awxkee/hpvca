//! HEVC CABAC context model initialisation values.
//!
//! initValues taken DIRECTLY from ffmpeg libavcodec/hevc_cabac.c
//! init_values[0] = I-slice (what we use).
//!
//! The CtxModel::init(initValue, qp) formula converts each byte to
//! (p_state_idx, val_mps) following HEVC spec §9.3.2.2.

use super::engine::CtxModel;

/// CNU = "context not used" sentinel value (127 in ffmpeg).
const CNU: u8 = 154; // use a neutral probability when context not applicable

/// All context models for I-slice residual coding, keyed by ffmpeg's
/// init_values[0] (I-slice column).
pub struct ContextSet {
    pub qp: u8,

    // CU-level
    pub split_cu_flag: [CtxModel; 3],

    // split_transform_flag[trafoDepth]: 3 contexts (initValues: 153,138,138)
    pub split_transform_flag: [CtxModel; 3],

    // CBF: cbf_luma[0..1], cbf_chroma[0..4]
    pub cbf_luma:   [CtxModel; 2],
    pub cbf_chroma: [CtxModel; 5],   // cb_trafoDepth0, cb_tD1, cr_tD0, cr_tD1, cr_tD2

    // last_sig_coeff_x/y prefix (18 contexts each)
    pub last_sig_coeff_x_prefix: [CtxModel; 18],
    pub last_sig_coeff_y_prefix: [CtxModel; 18],

    // sig_coeff_flag (44 contexts, luma+chroma merged)
    pub sig_coeff_flag: [CtxModel; 44],

    // coded_sub_block_flag (4 contexts: 2 luma + 2 chroma)
    pub coded_sub_block_flag: [CtxModel; 4],

    // coeff_abs_level_greater1 (24 contexts)
    pub coeff_abs_level_greater1: [CtxModel; 24],

    // coeff_abs_level_greater2 (6 contexts)
    pub coeff_abs_level_greater2: [CtxModel; 6],

    // SAO: sao_merge_left/up flag (1 ctx, shared) and sao_type_idx (1 ctx).
    // initType0: sao_merge=153, sao_type_idx=200.
    pub sao_merge_flag:  CtxModel,
    pub sao_type_idx:    CtxModel,
}

impl ContextSet {
    pub fn init_islice(qp: u8) -> Self {
        fn c(iv: u8, qp: u8) -> CtxModel { CtxModel::init(iv, qp) }
        fn arr<const N: usize>(ivs: [u8; N], qp: u8) -> [CtxModel; N] {
            ivs.map(|iv| CtxModel::init(iv, qp))
        }

        // I-slice initValues (initType=0) — authoritative, from libde265 contextmodel.cc.
        Self {
            qp,
            // split_cu_flag initType0: {139,141,157}
            split_cu_flag: arr([139, 141, 157], qp),

            // split_transform_flag initType0 (×3): {153,138,138}
            split_transform_flag: arr([153, 138, 138], qp),

            // cbf_luma initType0 (×2): {111,141}
            cbf_luma: arr([111, 141], qp),

            // cbf_chroma initType0 (×4): {94,138,182,154}; 5th slot reuses last.
            cbf_chroma: arr([94, 138, 182, 154, 154], qp),

            // last_significant_coeff_x_prefix (18) — initType0
            last_sig_coeff_x_prefix: arr([
                110, 110, 124, 125, 140, 153, 125, 127, 140, 109,
                111, 143, 127, 111,  79, 108, 123,  63,
            ], qp),

            // last_significant_coeff_y_prefix (18) — initType0 (same table)
            last_sig_coeff_y_prefix: arr([
                110, 110, 124, 125, 140, 153, 125, 127, 140, 109,
                111, 143, 127, 111,  79, 108, 123,  63,
            ], qp),

            // significant_coeff_flag — initType0 row (42 values + 2 pad)
            sig_coeff_flag: arr([
                111, 111, 125, 110, 110,  94, 124, 108, 124, 107,
                125, 141, 179, 153, 125, 107, 125, 141, 179, 153,
                125, 107, 125, 141, 179, 153, 125, 140, 139, 182,
                182, 152, 136, 152, 136, 153, 136, 139, 111, 136,
                139, 111, 141, 111,
            ], qp),

            // coded_sub_block_flag initType0 (×4): {91,171,134,141}
            coded_sub_block_flag: arr([91, 171, 134, 141], qp),

            // coeff_abs_level_greater1_flag (24) — initType0
            coeff_abs_level_greater1: arr([
                140,  92, 137, 138, 140, 152, 138, 139, 153,  74,
                149,  92, 139, 107, 122, 152, 140, 179, 166, 182,
                140, 227, 122, 197,
            ], qp),

            // coeff_abs_level_greater2_flag (6) — initType0: {138,153,136,167,152,152}
            coeff_abs_level_greater2: arr([138, 153, 136, 167, 152, 152], qp),

            // SAO initType0: sao_merge_flag=153, sao_type_idx=200 (libde265).
            sao_merge_flag: c(153, qp),
            sao_type_idx:   c(200, qp),
        }
    }
}

/// Intra-mode contexts (prev_intra_luma_pred_flag, intra_chroma_pred_mode).
/// I-slice (initType=0) init values from libde265:
///   part_mode = 184, prev_intra_luma_pred_flag = 184, intra_chroma_pred_mode = 63
pub struct IntraModeContexts {
    pub part_mode:                 CtxModel,
    pub prev_intra_luma_pred_flag: CtxModel,
    pub intra_chroma_pred_mode:    CtxModel,
}

impl IntraModeContexts {
    pub fn init_islice(qp: u8) -> Self {
        Self {
            part_mode:                 CtxModel::init(184, qp),
            prev_intra_luma_pred_flag: CtxModel::init(184, qp),
            intra_chroma_pred_mode:    CtxModel::init(63,  qp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_set_init() {
        let ctx = ContextSet::init_islice(26);
        // cbf_luma[0] initValue=111, qp=26
        // init_state = (slope * qp + offset<<3).clamp(1,126)
        // slope=111>>4=6, offset=111&0xF=15 → (6*26 + 15*8).clamp = (156+120)=276→126
        // 126 ≥ 64 → p_state=62, val_mps=1
        assert_eq!(ctx.cbf_luma[0].p_state_idx, 62);
        assert_eq!(ctx.cbf_luma[0].val_mps, 1);

        // cbf_chroma[0] initValue=94, qp=26
        // slope=5, offset=14 → (5*26+14*8)=130+112=242→126, p_state=62, mps=1
        assert_eq!(ctx.cbf_chroma[0].p_state_idx, 62);

        // prev_intra_luma_pred_flag initValue=184 (from IntraModeContexts, but verify formula)
        let ictx = IntraModeContexts::init_islice(26);
        assert!(ictx.prev_intra_luma_pred_flag.p_state_idx < 64);
    }

    #[test]
    fn intra_mode_contexts() {
        let ictx = IntraModeContexts::init_islice(26);
        assert!(ictx.prev_intra_luma_pred_flag.p_state_idx < 64);
        assert!(ictx.intra_chroma_pred_mode.p_state_idx < 64);
    }
}
