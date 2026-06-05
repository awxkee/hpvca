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
use super::{contexts::ContextSet, engine::CabacEncoder};

/// Encode cbf_luma (trafo_depth 0 or 1).
/// ffmpeg: GET_CABAC(elem_offset[CBF_LUMA] + !trafo_depth)
/// cbf_luma[0] = trafo_depth != 0, cbf_luma[1] = trafo_depth == 0 (i.e. root TU).
/// We always call at trafo_depth=0 (root), so use cbf_luma[1].
pub(crate) fn encode_cbf_luma(
    enc: &mut CabacEncoder,
    ctx: &mut ContextSet,
    flag: bool,
    trafo_depth: usize,
) {
    let idx = if trafo_depth == 0 { 1 } else { 0 };
    enc.encode_bin(flag as u8, &mut ctx.cbf_luma[idx]);
}

/// Encode cbf_cb or cbf_cr at given trafo_depth (0..4).
/// ffmpeg: GET_CABAC(elem_offset[CBF_CB_CR] + trafo_depth)
pub(crate) fn encode_cbf_chroma(
    enc: &mut CabacEncoder,
    ctx: &mut ContextSet,
    flag: bool,
    trafo_depth: usize,
) {
    let idx = trafo_depth.min(4);
    enc.encode_bin(flag as u8, &mut ctx.cbf_chroma[idx]);
}

/// Encode a coefficient block using HEVC CABAC residual_coding() syntax.
///
/// `coeffs`  — coefficients in HEVC sub-block-major diagonal scan order
///             (64 for 8×8, 16 for 4×4); `coeffs[sb*16 + k]` is scan position
///             `k` within sub-block `sb`.
/// `log2_ts` — log2 of the TU size (3 = 8×8, 2 = 4×4)
/// `is_luma` — selects luma vs chroma contexts
pub(crate) fn encode_residual(
    enc: &mut CabacEncoder,
    ctx: &mut ContextSet,
    coeffs: &[i16],
    log2_ts: u32,
    is_luma: bool,
) {
    let n_coeffs = (1usize << log2_ts) * (1usize << log2_ts);
    debug_assert!(coeffs.len() >= n_coeffs);

    let last_scan_pos = match coeffs[..n_coeffs].iter().rposition(|&c| c != 0) {
        Some(p) => p,
        None => return, // CBF=0 handled by caller
    };

    // (row, col) of the last significant coefficient in the TU.
    let (last_row, last_col) = ZIGZAG_LOOKUP(last_scan_pos, log2_ts);
    encode_last_sig(enc, ctx, last_col as u32, last_row as u32, log2_ts, is_luma);

    let tu_side = 1usize << log2_ts; // 8 or 4
    let sb_side = (tu_side / 4).max(1); // sub-blocks per side (2 or 1)
    let num_sb = sb_side * sb_side; // 4 or 1
    let last_sb = last_scan_pos / 16;

    // Sub-block diagonal scan over the sb_side×sb_side grid (matches coeff layout).
    let sb_scan: &[(usize, usize)] = if sb_side == 2 {
        &SB_DIAG_2X2
    } else {
        &SB_DIAG_1X1
    };

    // coded_sub_block_neighbors[idx] holds the 2-bit prev_csbf code for each
    // sub-block: bit0 (=1) means the sub-block to the RIGHT is coded, bit1 (=2)
    // means the sub-block BELOW is coded. Filled in as we walk (last → DC), exactly
    // as libde265/zune do.
    let mut csbf_neighbors = vec![0u8; num_sb];

    // Helper: TU-position (col,row) for an absolute scan position.
    let pos_of = |abs_pos: usize| -> (usize, usize) {
        let (r, c) = ZIGZAG_LOOKUP(abs_pos, log2_ts);
        (c, r) // (xc, yc)
    };

    // Level-coding carry state across sub-blocks (libde265 c1 / first_subblock).
    let mut level_state = LevelState {
        c1: 1,
        first_subblock: true,
    };

    // Walk sub-blocks from the last-significant sub-block down to DC (matches
    // HEVC §7.3.8.11: for(i = lastSubBlock; i >= 0; i--)). Sub-blocks above the
    // last one are not present in the bitstream at all.
    for sb in (0..=last_sb).rev() {
        let sb_start = sb * 16;
        let (sbx, sby) = sb_scan[sb];
        let sb_grid = sbx + sby * sb_side;
        let has_nonzero = coeffs[sb_start..(sb_start + 16).min(n_coeffs)]
            .iter()
            .any(|&c| c != 0);

        let sub_block_coded;
        let mut infer_dc = false;
        if sb != 0 && sb != last_sb {
            // coded_sub_block_flag is explicitly coded; ctx from neighbor code.
            let nb = csbf_neighbors[sb_grid];
            let ctx_inc = (nb > 0) as usize;
            let cg_ctx = if is_luma { ctx_inc } else { 2 + ctx_inc };
            enc.encode_bin(has_nonzero as u8, &mut ctx.coded_sub_block_flag[cg_ctx]);
            sub_block_coded = has_nonzero;
            infer_dc = true;
        } else {
            // first (DC) and last sub-blocks are always coded
            sub_block_coded = sb == 0 || sb == last_sb;
        }

        if sub_block_coded {
            // Mark neighbors: left learns its right is coded; top learns its bottom is.
            if sbx > 0 {
                csbf_neighbors[(sbx - 1) + sby * sb_side] |= 1;
            }
            if sby > 0 {
                csbf_neighbors[sbx + (sby - 1) * sb_side] |= 2;
            }
        }
        if !sub_block_coded {
            continue;
        }

        let prev_csbf = csbf_neighbors[sb_grid];
        let scan_top = if sb == last_sb {
            last_scan_pos % 16
        } else {
            15
        };

        let mut sig_positions: Vec<usize> = Vec::new();
        let mut any_sig_in_sb = false;

        // For the last sub-block, the top position is the known last coeff.
        // We process from scan_top down. For positions n=scan_top..1 we code/derive
        // sig flags; position 0 (sub-block DC) may be inferred.
        let start = scan_top;
        for k in (0..=start).rev() {
            let abs_pos = sb_start + k;
            if abs_pos >= n_coeffs {
                continue;
            }

            // Last coefficient of the whole TU: known significant, not coded here.
            if sb == last_sb && k == scan_top {
                sig_positions.push(abs_pos);
                any_sig_in_sb = true;
                continue;
            }

            // Sub-block DC inference: when this sub-block's csbf was explicitly
            // coded (infer_dc) and nothing else in it was significant, k==0 is
            // inferred significant and its flag is NOT coded.
            if k == 0 && infer_dc && !any_sig_in_sb {
                sig_positions.push(abs_pos);
                any_sig_in_sb = true;
                continue;
            }

            let (xc, yc) = pos_of(abs_pos);
            let ci = sig_coeff_ctx(xc, yc, prev_csbf, log2_ts, 0 /*diag*/, is_luma)
                .min(ctx.sig_coeff_flag.len() - 1);
            let is_sig = (coeffs[abs_pos] != 0) as u8;
            enc.encode_bin(is_sig, &mut ctx.sig_coeff_flag[ci]);
            if is_sig != 0 {
                sig_positions.push(abs_pos);
                any_sig_in_sb = true;
            }
        }

        if sig_positions.is_empty() {
            continue;
        }
        // sig_positions are in high→low scan order already.
        encode_coeff_levels(
            enc,
            ctx,
            coeffs,
            &sig_positions,
            sb,
            is_luma,
            &mut level_state,
        );
    }
}

/// Sub-block diagonal scan for a 2×2 grid of 4×4 sub-blocks (an 8×8 TU).
/// Index = sub-block number used in coeffs[sb*16+..]; value = (sbx, sby).
static SB_DIAG_2X2: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
/// Trivial 1×1 grid (a 4×4 TU).
const SB_DIAG_1X1: [(usize, usize); 1] = [(0, 0)];

/// Look up (row, col) for a scan position, given the TU size.
#[allow(non_snake_case)]
fn ZIGZAG_LOOKUP(scan_pos: usize, log2_ts: u32) -> (usize, usize) {
    if log2_ts == 2 {
        crate::dct::DIAG_SCAN_4X4[scan_pos]
    } else {
        crate::dct::ZIGZAG[scan_pos]
    }
}

/// Encode last_significant_coeff_x/y prefix using ffmpeg's exact formula.
///
/// For luma log2_size=3:  ctx_offset=3, ctx_shift=1
/// For chroma log2_size=2: ctx_offset=15, ctx_shift=0
/// For chroma log2_size=3: ctx_offset=15, ctx_shift=1
fn encode_last_sig(
    enc: &mut CabacEncoder,
    ctx: &mut ContextSet,
    last_x: u32,
    last_y: u32,
    log2_size: u32,
    is_luma: bool,
) {
    let (ctx_offset, ctx_shift) = if is_luma {
        let off = 3 * (log2_size - 2) + ((log2_size - 1) >> 2);
        let shift = (log2_size + 1) >> 2;
        (off as usize, shift as usize)
    } else {
        (15usize, (log2_size - 2) as usize)
    };

    // HEVC last-significant-coeff binarization tables (group index per coordinate,
    // and the minimum coordinate in each group). The prefix is a truncated-unary
    // code of the group index; the suffix is the offset within the group.
    static G_GROUP_IDX: [u32; 32] = [
        0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
        9, 9,
    ];
    const G_MIN_IN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

    let size = 1u32 << log2_size;
    let max_group = G_GROUP_IDX[(size - 1) as usize];

    let encode_prefix = |enc: &mut CabacEncoder,
                         ctx_arr: &mut [super::engine::CtxModel],
                         v: u32,
                         ctx_offset: usize,
                         ctx_shift: usize| {
        let group = G_GROUP_IDX[v as usize];
        let n = ctx_arr.len();
        for i in 0..group {
            let ci = (ctx_offset + (i as usize >> ctx_shift)).min(n - 1);
            enc.encode_bin(1, &mut ctx_arr[ci]);
        }
        if group < max_group {
            let ci = (ctx_offset + (group as usize >> ctx_shift)).min(n - 1);
            enc.encode_bin(0, &mut ctx_arr[ci]);
        }
    };
    let encode_suffix = |enc: &mut CabacEncoder, v: u32| {
        let group = G_GROUP_IDX[v as usize];
        if group > 3 {
            let suffix = v - G_MIN_IN_GROUP[group as usize];
            let nbits = (group - 2) / 2;
            for i in (0..nbits).rev() {
                enc.encode_bypass(((suffix >> i) & 1) as u8);
            }
        }
    };

    // HEVC §7.3.8.11 order: x-prefix, y-prefix, then x-suffix, y-suffix.
    encode_prefix(
        enc,
        &mut ctx.last_sig_coeff_x_prefix,
        last_x,
        ctx_offset,
        ctx_shift,
    );
    encode_prefix(
        enc,
        &mut ctx.last_sig_coeff_y_prefix,
        last_y,
        ctx_offset,
        ctx_shift,
    );
    encode_suffix(enc, last_x);
    encode_suffix(enc, last_y);
}

/// sig_coeff_flag context increment — faithful port of libde265/zune logic.
///
/// `xc,yc` are the coefficient's position within the whole TU. `prev_csbf` is the
/// 2-bit neighbour code for this coefficient's sub-block (bit0 = right sub-block
/// coded, bit1 = bottom sub-block coded). `scan_idx` is 0=diag, 1=horiz, 2=vert.
fn sig_coeff_ctx(
    xc: usize,
    yc: usize,
    prev_csbf: u8,
    log2_ts: u32,
    scan_idx: u8,
    is_luma: bool,
) -> usize {
    static CTX_IDX_MAP_4X4: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 99];
    let sb_width = 1usize << (log2_ts - 2); // sub-blocks per side (1 for 4×4, 2 for 8×8)

    let mut sig_ctx: i32;
    if sb_width == 1 {
        // 4×4 block: indexed by raster position
        sig_ctx = CTX_IDX_MAP_4X4[(yc << 2) + xc] as i32;
    } else if xc + yc == 0 {
        // DC of larger block
        sig_ctx = 0;
    } else {
        let xp = xc & 3;
        let yp = yc & 3;
        let xs = xc >> 2;
        let ys = yc >> 2;
        sig_ctx = match prev_csbf {
            0 => {
                if xp + yp >= 3 {
                    0
                } else if xp + yp > 0 {
                    1
                } else {
                    2
                }
            }
            1 => {
                if yp == 0 {
                    2
                } else if yp == 1 {
                    1
                } else {
                    0
                }
            }
            2 => {
                if xp == 0 {
                    2
                } else if xp == 1 {
                    1
                } else {
                    0
                }
            }
            _ => 2,
        };
        if is_luma {
            if xs + ys > 0 {
                sig_ctx += 3;
            }
            if sb_width == 2 {
                // 8×8 luma: diag vs horiz/vert
                sig_ctx += if scan_idx == 0 { 9 } else { 15 };
            } else {
                sig_ctx += 21;
            }
        } else {
            if sb_width == 2 {
                sig_ctx += 9;
            } else {
                sig_ctx += 12;
            }
        }
    }

    if is_luma {
        sig_ctx as usize
    } else {
        27 + sig_ctx as usize
    }
}

/// Per-TU state carried across sub-blocks for level coding.
struct LevelState {
    c1: i32, // greater1 carry (the libde265 `c1`)
    first_subblock: bool,
}

/// Encode coeff_abs_level_greater1, greater2, sign flags, and remaining levels
/// for one sub-block. `sig_pos` is in high→low scan order. Mirrors libde265/zune
/// (and HM) exactly, including the cross-sub-block context carry.
fn encode_coeff_levels(
    enc: &mut CabacEncoder,
    ctx: &mut ContextSet,
    coeffs: &[i16],
    sig_pos: &[usize],
    sb: usize,
    is_luma: bool,
    st: &mut LevelState,
) {
    let n = sig_pos.len();
    if n == 0 {
        return;
    }

    // ctx_set: 0 for the DC sub-block or any chroma block, else 2; +1 if the
    // previous sub-block ended with c1 == 0.
    let mut ctx_set: i32 = if sb == 0 || !is_luma { 0 } else { 2 };
    if st.c1 == 0 {
        ctx_set += 1;
    }
    st.c1 = 1; // reset for this sub-block

    let chroma_off = if is_luma { 0 } else { 16 };

    // greater1 flags for up to the first 8 coefficients.
    let last_g1 = n.min(8);
    let mut greater1_flags = vec![false; n];
    let mut first_g1_one: Option<usize> = None;
    let mut has_max_base = vec![true; n]; // coeff_has_max_base_level

    for c in 0..last_g1 {
        let abs_val = coeffs[sig_pos[c]].unsigned_abs() as i32;
        let gr1 = abs_val > 1;
        let g1ctx = if st.c1 >= 3 { 3 } else { st.c1 };
        let ci = (ctx_set * 4 + g1ctx + chroma_off) as usize;
        let ci = ci.min(ctx.coeff_abs_level_greater1.len() - 1);
        enc.encode_bin(gr1 as u8, &mut ctx.coeff_abs_level_greater1[ci]);
        greater1_flags[c] = gr1;
        if gr1 {
            st.c1 = 0;
            if first_g1_one.is_none() {
                first_g1_one = Some(c);
            }
        } else {
            has_max_base[c] = false; // level is exactly 1
            if st.c1 < 3 && st.c1 > 0 {
                st.c1 += 1;
            }
        }
    }
    st.first_subblock = false;

    // greater2: only for the first coeff whose greater1 == 1.
    if let Some(c) = first_g1_one {
        let abs_val = coeffs[sig_pos[c]].unsigned_abs() as i32;
        let gr2 = abs_val > 2;
        let ci2 = (ctx_set + if is_luma { 0 } else { 4 }) as usize;
        let ci2 = ci2.min(ctx.coeff_abs_level_greater2.len() - 1);
        enc.encode_bin(gr2 as u8, &mut ctx.coeff_abs_level_greater2[ci2]);
        // If greater2 == 0, the level is fully determined (=2): no remaining.
        has_max_base[c] = gr2;
    }

    // Signs (bypass), high→low scan order. (sign_data_hiding disabled in our PPS.)
    for &pos in sig_pos.iter() {
        enc.encode_bypass((coeffs[pos] < 0) as u8);
    }

    // coeff_abs_level_remaining for every coeff whose level isn't fully determined.
    let mut rice: u32 = 0;
    for c in 0..n {
        if !has_max_base[c] {
            continue;
        }
        // base_level = 1 + greater1 + greater2 (the flags actually coded).
        let g1 = if c < last_g1 {
            greater1_flags[c] as i32
        } else {
            0
        };
        let g2 = if Some(c) == first_g1_one {
            (coeffs[sig_pos[c]].unsigned_abs() as i32 > 2) as i32
        } else {
            0
        };
        let base_level = 1 + g1 + g2;
        let abs_val = coeffs[sig_pos[c]].unsigned_abs() as i32;
        let remaining = (abs_val - base_level).max(0) as u32;
        encode_coeff_remaining(enc, remaining, rice);
        // Rice update: if (base_level + remaining) > 3<<rice, rice = min(rice+1,4).
        let total = base_level + remaining as i32;
        if total > (3 << rice) {
            rice = (rice + 1).min(4);
        }
    }
}

/// Bypass-coded coeff_abs_level_remaining using truncated Rice/Exp-Golomb.
fn encode_coeff_remaining(enc: &mut CabacEncoder, value: u32, rice_k: u32) {
    // Inverse of HEVC coeff_abs_level_remaining (§9.3.3.x): a unary prefix of
    // `prefix` ones terminated by a 0, then a fixed-length suffix.
    //   prefix <= 3:  value = (prefix << k) + suffix(k bits)
    //   prefix  > 3:  value = (((1 << (prefix-3)) + 2) << k) + suffix(prefix-3+k bits)
    if value < (4u32 << rice_k) {
        let prefix = value >> rice_k;
        for _ in 0..prefix {
            enc.encode_bypass(1);
        }
        enc.encode_bypass(0);
        for i in (0..rice_k).rev() {
            enc.encode_bypass(((value >> i) & 1) as u8);
        }
    } else {
        // Find prefix >= 4 such that base(prefix) <= value < base(prefix+1).
        let mut p = 4u32;
        loop {
            let base_next = ((1u32 << (p + 1 - 3)) + 2) << rice_k;
            if value < base_next {
                break;
            }
            p += 1;
        }
        for _ in 0..p {
            enc.encode_bypass(1);
        }
        enc.encode_bypass(0);
        let suffix_bits = p - 3 + rice_k;
        let codeword = value - (((1u32 << (p - 3)) + 2) << rice_k);
        for i in (0..suffix_bits).rev() {
            enc.encode_bypass(((codeword >> i) & 1) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::contexts::ContextSet;

    fn make_coeffs(vals: &[(usize, i16)], len: usize) -> Vec<i16> {
        let mut c = vec![0i16; len];
        for &(i, v) in vals {
            c[i] = v;
        }
        c
    }

    #[test]
    fn encode_single_dc_8x8() {
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextSet::init_islice(26);
        let coeffs = make_coeffs(&[(0, 8)], 64);
        encode_residual(&mut enc, &mut ctx, &coeffs, 3, true);
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty());
    }

    #[test]
    fn encode_single_dc_4x4() {
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextSet::init_islice(26);
        let coeffs = make_coeffs(&[(0, 5)], 16);
        encode_residual(&mut enc, &mut ctx, &coeffs, 2, false);
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty());
    }

    #[test]
    fn encode_multiple_coeffs() {
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextSet::init_islice(26);
        let coeffs = make_coeffs(&[(0, 12), (1, -3), (2, 1), (8, 5)], 64);
        encode_residual(&mut enc, &mut ctx, &coeffs, 3, true);
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty());
    }

    #[test]
    fn encode_all_zero_does_nothing() {
        let mut enc = CabacEncoder::new();
        let mut ctx = ContextSet::init_islice(26);
        let coeffs = vec![0i16; 64];
        encode_residual(&mut enc, &mut ctx, &coeffs, 3, true);
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(out.len() < 4); // only the terminate flush
    }
}
