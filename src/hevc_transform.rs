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

//! Spec-faithful HEVC integer transform, quantization, and dequantization.

use crate::cabac::ContextSet;
use crate::cabac::residual::sig_coeff_ctx;

/// 4×4 HEVC transform matrix.
static T4: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// 8×8 HEVC transform matrix.
static T8: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

/// 16×16 HEVC transform matrix (spec Table 8-6).
#[rustfmt::skip]
static T16: [[i32; 16]; 16] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9],
];

static QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

/// Largest supported transform: 16×16 → 256 coefficients per fixed buffer.
const MAX_TB: usize = 256;
static DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Forward integer transform of an N×N residual block (N = 4 or 8).
/// Returns a fixed 64-entry buffer; only the first `n*n` entries are written.
pub(crate) fn fwd_transform(res: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    match n {
        4 => fwd_transform_n::<4>(res, &T4, bit_depth, &mut out),
        8 => fwd_transform_n::<8>(res, &T8, bit_depth, &mut out),
        16 => fwd_transform_n::<16>(res, &T16, bit_depth, &mut out),
        _ => panic!("unsupported transform size {n}"),
    }
    out
}

#[inline]
fn fwd_transform_n<const N: usize>(
    res: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    let log2n = N.trailing_zeros() as i32;
    let bd = bit_depth as i32;
    let shift1 = log2n + bd - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    // i32 throughout: products (|coeff|≤90 · |residual|≤4095) and the ≤N-term
    // sums stay well inside i32 for every supported bit depth.
    let mut tmp = [0i32; MAX_TB]; // N*N <= 256
    // pass 1 (rows of res): tmp[j*N+i] = (Σ_k T[i][k]·res[j*N+k]) >> shift1
    for (j, res_row) in res.as_chunks::<N>().0.iter().enumerate().take(N) {
        for (i, trow) in t.iter().enumerate() {
            let mut s = 0i32;
            for k in 0..N {
                s += trow[k] * res_row[k];
            }
            tmp[j * N + i] = if shift1 > 0 { (s + add1) >> shift1 } else { s };
        }
    }
    // pass 2 (columns): coeff[i*N+j] = (Σ_k T[i][k]·tmp[k*N+j]) >> shift2
    let shift2 = log2n + 6;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; N];
    for j in 0..N {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = tmp[k * N + j];
        }
        for (i, trow) in t.iter().enumerate() {
            let mut s = 0i32;
            for k in 0..N {
                s += trow[k] * colv[k];
            }
            out[i * N + j] = (s + add2) >> shift2;
        }
    }
}

#[inline]
fn coeff_remaining_bits(value: u32, rice: u32) -> f32 {
    if value < (4u32 << rice) {
        (value >> rice) as f32 + 1.0 + rice as f32
    } else {
        let mut prefix = 4u32;
        while value >= (((1u32 << (prefix + 1 - 3)) + 2) << rice) {
            prefix += 1;
        }
        // Unary prefix + separator, followed by the extended suffix.
        (prefix + 1 + prefix - 3 + rice) as f32
    }
}

#[inline]
fn rdoq_level_bits(
    abs_level: u32,
    ctx_set: usize,
    c1: i32,
    c1_idx: u32,
    c2_idx: u32,
    rice: u32,
    ctx: &ContextSet,
) -> f32 {
    debug_assert!(abs_level > 0);
    const C1_FLAGS: u32 = 8;
    const C2_FLAGS: u32 = 1;

    let mut bits = 1.0; // sign bypass bin
    let one_ctx =
        (ctx_set * 4 + c1.clamp(0, 3) as usize).min(ctx.coeff_abs_level_greater1.len() - 1);
    let abs_ctx = ctx_set.min(ctx.coeff_abs_level_greater2.len() - 1);
    let base_level = if c1_idx < C1_FLAGS {
        2 + (c2_idx < C2_FLAGS) as u32
    } else {
        1
    };

    if abs_level >= base_level {
        bits += coeff_remaining_bits(abs_level - base_level, rice);
        if c1_idx < C1_FLAGS {
            bits += ctx.coeff_abs_level_greater1[one_ctx].estimated_bits(1);
            if c2_idx < C2_FLAGS {
                bits += ctx.coeff_abs_level_greater2[abs_ctx].estimated_bits(1);
            }
        }
    } else if abs_level == 1 {
        if c1_idx < C1_FLAGS {
            bits += ctx.coeff_abs_level_greater1[one_ctx].estimated_bits(0);
        }
    } else {
        debug_assert_eq!(abs_level, 2);
        if c1_idx < C1_FLAGS {
            bits += ctx.coeff_abs_level_greater1[one_ctx].estimated_bits(1);
            if c2_idx < C2_FLAGS {
                bits += ctx.coeff_abs_level_greater2[abs_ctx].estimated_bits(0);
            }
        }
    }
    bits
}

#[inline]
fn last_sig_bits(ctx: &ContextSet, x: usize, y: usize, log2_size: u32, scan_idx: u8) -> f32 {
    static GROUP_IDX: [usize; 32] = [
        0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
        9, 9,
    ];
    static MIN_IN_GROUP: [usize; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

    let (x, y) = if scan_idx == 2 { (y, x) } else { (x, y) };
    let ctx_offset = (3 * (log2_size - 2) + ((log2_size - 1) >> 2)) as usize;
    let ctx_shift = ((log2_size + 1) >> 2) as usize;
    let max_group = GROUP_IDX[(1usize << log2_size) - 1];

    let prefix_bits = |value: usize, models: &[crate::cabac::engine::CtxModel]| {
        let group = GROUP_IDX[value];
        let mut bits = 0.0;
        for i in 0..group {
            let ci = (ctx_offset + (i >> ctx_shift)).min(models.len() - 1);
            bits += models[ci].estimated_bits(1);
        }
        if group < max_group {
            let ci = (ctx_offset + (group >> ctx_shift)).min(models.len() - 1);
            bits += models[ci].estimated_bits(0);
        }
        if group > 3 {
            bits += ((group - 2) / 2) as f32;
            debug_assert!(value >= MIN_IN_GROUP[group]);
        }
        bits
    };

    prefix_bits(x, &ctx.last_sig_coeff_x_prefix) + prefix_bits(y, &ctx.last_sig_coeff_y_prefix)
}

#[inline]
fn rdoq_distortion_scale(n: usize, bit_depth: u8) -> f32 {
    // The forward transform represents an orthonormal coefficient multiplied by
    // 2^(15-bitDepth-log2(N)); invert that scale to express coefficient error in
    // the same spatial-SSE domain as the encoder's lambda.
    let exponent = 2 * (bit_depth as i32 + n.trailing_zeros() as i32 - 15);
    2f32.powi(exponent)
}

#[inline]
fn dequant_abs_level(level: u32, n: usize, qp: u8, bit_depth: u8) -> i64 {
    let log2n = n.trailing_zeros() as i64;
    let bd_shift = bit_depth as i64 + log2n - 5;
    let add = 1i64 << (bd_shift - 1);
    let factor = DEQUANT_SCALE[(qp % 6) as usize] * (1i64 << (qp as i64 / 6)) * 16;
    ((level as i64 * factor + add) >> bd_shift).clamp(0, 32767)
}

#[inline]
fn coefficient_distortion(
    coeff_abs: i64,
    level: u32,
    n: usize,
    qp: u8,
    bit_depth: u8,
    scale: f32,
) -> f32 {
    let error = coeff_abs - dequant_abs_level(level, n, qp, bit_depth);
    (error * error) as f32 * scale
}

/// HM-style rate-distortion optimized quantization for the committed luma mode.
///
/// The fast intra shortlist uses [`quantize_with_sign_hiding`]. Only its winner
/// enters this path. RDOQ considers the rounded level and one lower level per coefficient,
/// can zero weak coefficients and whole 4×4 coefficient groups, and optimizes
/// the last-significant position. CABAC costs use frozen probability states from
/// the current TU entrance, while c1/c2 and Rice adaptation follow the candidate
/// levels in reverse scan order, matching HM's normal RDOQ structure.
pub(crate) fn rdoq_luma_with_sign_hiding(
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    scan: &[(usize, usize)],
    scan_idx: u8,
    lambda: f32,
    ctx: &ContextSet,
) -> [i16; MAX_TB] {
    const GROUP_SIZE: usize = 16;
    const C1_FLAGS: u32 = 8;

    let num_coeffs = n * n;
    debug_assert_eq!(scan.len(), num_coeffs);
    debug_assert!(matches!(n, 8 | 16));
    let log2_size = n.trailing_zeros();
    let num_groups = num_coeffs / GROUP_SIZE;
    let sb_side = n / 4;
    let sb_scan = crate::dct::sb_scan_for(log2_size, scan_idx);
    let q_bits = 14 + qp as i64 / 6 + (15 - bit_depth as i64 - log2_size as i64);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let round = 1i64 << (q_bits - 1);
    let distortion_scale = rdoq_distortion_scale(n, bit_depth);

    let mut levels = [0i16; MAX_TB]; // absolute levels until final sign restore
    let mut cost_coeff = [0.0f32; MAX_TB];
    let mut cost_coeff0 = [0.0f32; MAX_TB];
    let mut cost_sig = [0.0f32; MAX_TB];
    let mut cost_group_sig = [0.0f32; 16];
    let mut group_flags = [0u8; 16]; // raster coefficient-group grid
    let mut block_uncoded_cost = 0.0;
    let mut base_cost = 0.0;
    let mut last_scan_pos = None;
    let mut last_group = 0usize;
    let mut carry_c1 = 1i32;

    for group in (0..num_groups).rev() {
        let group_start = group * GROUP_SIZE;
        let (sbx, sby) = sb_scan[group];
        let group_grid = sbx + sby * sb_side;
        let right = sbx + 1 < sb_side && group_flags[group_grid + 1] != 0;
        let below = sby + 1 < sb_side && group_flags[group_grid + sb_side] != 0;
        let prev_csbf = right as u8 | ((below as u8) << 1);

        let mut ctx_set = if group == 0 { 0usize } else { 2usize };
        if carry_c1 == 0 {
            ctx_set += 1;
        }
        let mut c1 = 1i32;
        let mut c1_idx = 0u32;
        let mut c2_idx = 0u32;
        let mut rice = 0u32;

        let mut group_sig_cost = 0.0;
        let mut group_coded_level_and_dist = 0.0;
        let mut group_uncoded_dist = 0.0;
        let mut nnz_before_pos0 = 0usize;
        let mut group_has_nonzero = false;

        for k in (0..GROUP_SIZE).rev() {
            let scan_pos = group_start + k;
            let (row, col) = scan[scan_pos];
            let pos = row * n + col;
            let coeff_abs = (coeff[pos] as i64).abs();
            let scaled = coeff_abs * q_scale;
            let max_abs = ((scaled + round) >> q_bits).clamp(0, i16::MAX as i64) as u32;
            let dist0 = coefficient_distortion(coeff_abs, 0, n, qp, bit_depth, distortion_scale);
            cost_coeff0[scan_pos] = dist0;
            block_uncoded_cost += dist0;

            if last_scan_pos.is_none() && max_abs == 0 {
                cost_coeff[scan_pos] = dist0;
                base_cost += dist0;
                continue;
            }
            if last_scan_pos.is_none() {
                last_scan_pos = Some(scan_pos);
                last_group = group;
            }
            let is_last = last_scan_pos == Some(scan_pos);
            let sig_ctx = sig_coeff_ctx(col, row, prev_csbf, log2_size, scan_idx, true)
                .min(ctx.sig_coeff_flag.len() - 1);
            let sig0 = if is_last {
                0.0
            } else {
                lambda * ctx.sig_coeff_flag[sig_ctx].estimated_bits(0)
            };
            let sig1 = if is_last {
                0.0
            } else {
                lambda * ctx.sig_coeff_flag[sig_ctx].estimated_bits(1)
            };

            let mut best_level = 0u32;
            let mut best_cost = dist0 + sig0;
            if max_abs > 0 {
                let min_abs = if max_abs > 1 { max_abs - 1 } else { 1 };
                for abs_level in [max_abs, min_abs] {
                    let dist = coefficient_distortion(
                        coeff_abs,
                        abs_level,
                        n,
                        qp,
                        bit_depth,
                        distortion_scale,
                    );
                    let rate = rdoq_level_bits(abs_level, ctx_set, c1, c1_idx, c2_idx, rice, ctx);
                    let coded_cost = dist + sig1 + lambda * rate;
                    if coded_cost < best_cost || is_last && best_level == 0 {
                        best_cost = coded_cost;
                        best_level = abs_level;
                    }
                    if min_abs == max_abs {
                        break;
                    }
                }
                // HM directly tests zero for small levels. Larger all-zero
                // choices are handled more efficiently by CG and last cleanup.
                if !is_last && max_abs >= 3 {
                    best_cost = f32::MAX;
                    for abs_level in [max_abs, min_abs] {
                        let dist = coefficient_distortion(
                            coeff_abs,
                            abs_level,
                            n,
                            qp,
                            bit_depth,
                            distortion_scale,
                        );
                        let rate =
                            rdoq_level_bits(abs_level, ctx_set, c1, c1_idx, c2_idx, rice, ctx);
                        let coded_cost = dist + sig1 + lambda * rate;
                        if coded_cost < best_cost {
                            best_cost = coded_cost;
                            best_level = abs_level;
                        }
                        if min_abs == max_abs {
                            break;
                        }
                    }
                }
            }

            levels[pos] = best_level as i16;
            cost_coeff[scan_pos] = best_cost;
            cost_sig[scan_pos] = if best_level == 0 { sig0 } else { sig1 };
            base_cost += best_cost;
            group_sig_cost += cost_sig[scan_pos];

            if best_level > 0 {
                group_has_nonzero = true;
                group_coded_level_and_dist += best_cost - cost_sig[scan_pos];
                group_uncoded_dist += dist0;
                if k != 0 {
                    nnz_before_pos0 += 1;
                }

                let base_level = if c1_idx < C1_FLAGS {
                    2 + (c2_idx == 0) as u32
                } else {
                    1
                };
                if best_level >= base_level && best_level > (3 << rice) {
                    rice = (rice + 1).min(4);
                }
                c1_idx += 1;
                if best_level > 1 {
                    c1 = 0;
                    c2_idx += 1;
                } else if (1..3).contains(&c1) {
                    c1 += 1;
                }
            }
        }
        carry_c1 = c1;

        let Some(_) = last_scan_pos else {
            continue;
        };
        if group == last_group {
            group_flags[group_grid] = 1;
            continue;
        }
        if group == 0 {
            group_flags[group_grid] = group_has_nonzero as u8;
            continue;
        }

        let cg_ctx = (prev_csbf != 0) as usize;
        let csbf0 = lambda * ctx.coded_sub_block_flag[cg_ctx].estimated_bits(0);
        let csbf1 = lambda * ctx.coded_sub_block_flag[cg_ctx].estimated_bits(1);
        if !group_has_nonzero {
            base_cost += csbf0 - group_sig_cost;
            cost_group_sig[group] = csbf0;
            group_flags[group_grid] = 0;
            continue;
        }

        // With an explicitly coded non-zero CG, coefficient zero is inferred
        // significant when no higher position in the group is significant.
        if nnz_before_pos0 == 0 {
            base_cost -= cost_sig[group_start];
            group_sig_cost -= cost_sig[group_start];
            cost_sig[group_start] = 0.0;
        }
        let coded_group_cost = base_cost + csbf1;
        let zero_group_cost =
            base_cost + csbf0 + group_uncoded_dist - group_coded_level_and_dist - group_sig_cost;
        if zero_group_cost < coded_group_cost {
            base_cost = zero_group_cost;
            cost_group_sig[group] = csbf0;
            group_flags[group_grid] = 0;
            for scan_pos in group_start..group_start + GROUP_SIZE {
                let (row, col) = scan[scan_pos];
                levels[row * n + col] = 0;
                cost_coeff[scan_pos] = cost_coeff0[scan_pos];
                cost_sig[scan_pos] = 0.0;
            }
        } else {
            base_cost = coded_group_cost;
            cost_group_sig[group] = csbf1;
            group_flags[group_grid] = 1;
        }
    }

    let Some(initial_last) = last_scan_pos else {
        return levels;
    };

    // Compare the non-zero TU against CBF=0, then move the last-significant
    // position down through trailing unit levels. HM stops once it reaches a
    // level greater than one because discarding it is rarely profitable.
    let cbf = ctx.cbf_luma[1];
    let mut best_cost = block_uncoded_cost + lambda * cbf.estimated_bits(0);
    base_cost += lambda * cbf.estimated_bits(1);
    let mut best_last_p1 = 0usize;
    let mut stop = false;
    for group in (0..=last_group).rev() {
        base_cost -= cost_group_sig[group];
        let (sbx, sby) = sb_scan[group];
        let group_grid = sbx + sby * sb_side;
        if group_flags[group_grid] == 0 {
            continue;
        }
        for k in (0..GROUP_SIZE).rev() {
            let scan_pos = group * GROUP_SIZE + k;
            if scan_pos > initial_last {
                continue;
            }
            let (row, col) = scan[scan_pos];
            let pos = row * n + col;
            let level = levels[pos].unsigned_abs() as u32;
            if level != 0 {
                let total = base_cost + lambda * last_sig_bits(ctx, col, row, log2_size, scan_idx)
                    - cost_sig[scan_pos];
                if total < best_cost {
                    best_cost = total;
                    best_last_p1 = scan_pos + 1;
                }
                if level > 1 {
                    stop = true;
                    break;
                }
                base_cost += cost_coeff0[scan_pos] - cost_coeff[scan_pos];
            } else {
                base_cost -= cost_sig[scan_pos];
            }
        }
        if stop {
            break;
        }
    }

    for (scan_pos, &(row, col)) in scan.iter().enumerate() {
        let pos = row * n + col;
        if scan_pos >= best_last_p1 {
            levels[pos] = 0;
        } else if coeff[pos] < 0 {
            levels[pos] = -levels[pos];
        }
    }
    apply_sign_hiding_to_levels(&mut levels, coeff, n, qp, bit_depth, scan);
    levels
}

/// Forward quantization followed by HEVC sign-data hiding.
///
/// `scan` is the TU's coefficient scan in sub-block-major order. The level
/// adjustment is the distortion-only `signBitHidingHDQ` method used by HM: for
/// each eligible 4×4 coefficient group, change the cheapest magnitude by one so
/// the parity of the absolute-level sum carries the first significant sign.
pub(crate) fn quantize_with_sign_hiding(
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    scan: &[(usize, usize)],
) -> [i16; MAX_TB] {
    debug_assert_eq!(scan.len(), n * n);
    quantize_impl(coeff, n, qp, bit_depth, Some(scan))
}

/// Apply HEVC sign-data hiding to an already selected set of coefficient levels.
///
/// Winner-only RDOQ changes some of the plain quantizer's levels, so it cannot
/// reuse the levels returned by [`quantize_with_sign_hiding`]. Recompute the
/// quantization-error deltas for the chosen levels and run the same HM-style
/// parity adjustment once, after level optimization has finished.
fn apply_sign_hiding_to_levels(
    levels: &mut [i16; MAX_TB],
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    scan: &[(usize, usize)],
) {
    let log2n = n.trailing_zeros() as i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bit_depth as i64 - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let q_bits_8 = q_bits - 8;
    let mut delta_u = [0i32; MAX_TB];
    let mut abs_sum = 0u32;

    for (i, (&c, &level)) in coeff[..n * n].iter().zip(&levels[..n * n]).enumerate() {
        let scaled = (c as i64).abs() * q_scale;
        let magnitude = level.unsigned_abs() as i64;
        delta_u[i] = ((scaled - (magnitude << q_bits)) >> q_bits_8) as i32;
        abs_sum = abs_sum.saturating_add(magnitude as u32);
    }

    if abs_sum >= 2 {
        sign_bit_hiding_hdq(levels, coeff, &delta_u, n, scan);
    }
}

fn quantize_impl(
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    sign_hiding_scan: Option<&[(usize, usize)]>,
) -> [i16; MAX_TB] {
    let log2n = n.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bd - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra
    let q_bits_8 = q_bits - 8;
    let mut out = [0i16; MAX_TB];
    let mut delta_u = [0i32; MAX_TB];
    let mut abs_sum = 0u32;

    for (i, (o, &c)) in out[..n * n].iter_mut().zip(coeff).enumerate() {
        // i64 product: |coeff| can reach ~2^16 and q_scale ~2^15, so the product
        // overflows i32; keep this one multiply in i64.
        let c = c as i64;
        let scaled = c.abs() * q_scale;
        let magnitude = (scaled + offset) >> q_bits;
        delta_u[i] = ((scaled - (magnitude << q_bits)) >> q_bits_8) as i32;
        let level = if c < 0 { -magnitude } else { magnitude };
        *o = level.clamp(-32768, 32767) as i16;
        abs_sum = abs_sum.saturating_add((*o).unsigned_abs() as u32);
    }

    // HM avoids testing the degenerate one-coefficient, unit-level TU.
    if abs_sum >= 2 {
        if let Some(scan) = sign_hiding_scan {
            sign_bit_hiding_hdq(&mut out, coeff, &delta_u, n, scan);
        }
    }
    out
}

/// HEVC/HM signBitHidingHDQ, distortion-only variant (no rate term).
///
/// One sign may be hidden per 4×4 coefficient group when the distance between
/// its first and last significant scan positions is at least four. If the
/// current parity does not encode the first sign, choose the ±1 level change
/// with the smallest quantization-error increase while preserving a decodable
/// first-significant coefficient.
fn sign_bit_hiding_hdq(
    levels: &mut [i16; MAX_TB],
    coeff: &[i32],
    delta_u: &[i32; MAX_TB],
    n: usize,
    scan: &[(usize, usize)],
) {
    const GROUP_SIZE: usize = 16;
    const SBH_THRESHOLD: usize = 4;

    let num_coeffs = n * n;
    debug_assert_eq!(num_coeffs % GROUP_SIZE, 0);
    debug_assert!(coeff.len() >= num_coeffs);
    debug_assert_eq!(scan.len(), num_coeffs);

    let mut found_last_group = false;
    for subset in (0..num_coeffs / GROUP_SIZE).rev() {
        let sub_pos = subset * GROUP_SIZE;
        let group = &scan[sub_pos..sub_pos + GROUP_SIZE];
        let row_major = |scan_pos: usize| {
            let (row, col) = group[scan_pos];
            row * n + col
        };

        let Some(first_nz) = (0..GROUP_SIZE).find(|&i| levels[row_major(i)] != 0) else {
            continue;
        };
        let last_nz = (0..GROUP_SIZE)
            .rev()
            .find(|&i| levels[row_major(i)] != 0)
            .expect("non-empty coefficient group has a last coefficient");

        let is_last_group = !found_last_group;
        found_last_group = true;

        if last_nz - first_nz < SBH_THRESHOLD {
            continue;
        }

        // Signed and absolute sums have identical parity in two's-complement
        // arithmetic, matching HM's signed pQCoef accumulation.
        let sum: i32 = (first_nz..=last_nz)
            .map(|i| levels[row_major(i)] as i32)
            .sum();
        let first_pos = row_major(first_nz);
        let sign_bit = (levels[first_pos] < 0) as i32;
        if sign_bit == (sum & 1) {
            continue;
        }

        let search_top = if is_last_group {
            last_nz
        } else {
            GROUP_SIZE - 1
        };
        let mut best_cost = i64::MAX;
        let mut best_pos = None;
        let mut best_change = 0i32;

        for i in (0..=search_top).rev() {
            let pos = row_major(i);
            let level = levels[pos];
            let delta = delta_u[pos] as i64;

            let candidate = if level != 0 {
                if delta > 0 {
                    Some((-delta, 1))
                } else if i == first_nz && level.unsigned_abs() == 1 {
                    // Removing the coefficient whose sign is being hidden would
                    // change which sign the decoder infers.
                    None
                } else {
                    Some((delta, -1))
                }
            } else if i < first_nz {
                // Creating a new first significant coefficient is legal only if
                // its natural sign equals the sign currently being hidden.
                let this_sign = (coeff[pos] < 0) as i32;
                if this_sign == sign_bit {
                    Some((-delta, 1))
                } else {
                    None
                }
            } else {
                Some((-delta, 1))
            };

            if let Some((cost, change)) = candidate {
                if cost < best_cost {
                    best_cost = cost;
                    best_pos = Some(pos);
                    best_change = change;
                }
            }
        }

        let pos = best_pos.expect("eligible sign-hiding group has a valid adjustment");
        if levels[pos] == i16::MAX || levels[pos] == i16::MIN {
            best_change = -1;
        }
        let adjusted = if coeff[pos] >= 0 {
            levels[pos] as i32 + best_change
        } else {
            levels[pos] as i32 - best_change
        };
        levels[pos] = adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Dequantisation: level → transform coefficient (spec 8.6.3). Fixed 64-entry buffer.
pub(crate) fn dequantize(level: &[i16], n: usize, qp: u8, bit_depth: u8) -> [i32; MAX_TB] {
    let log2n = n.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let bd_shift = bd + log2n - 5;
    let add = 1i64 << (bd_shift - 1);
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let factor = scale * per * 16;
    let mut out = [0i32; MAX_TB];
    for (o, &l) in out[..n * n].iter_mut().zip(level) {
        *o = ((l as i64 * factor + add) >> bd_shift).clamp(-32768, 32767) as i32;
    }
    out
}

/// Inverse integer transform (spec 8.6.4.2). Returns residual in a fixed buffer.
pub(crate) fn inv_transform(coeff: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    match n {
        4 => inv_transform_n::<4>(coeff, &T4, bit_depth, &mut out),
        8 => inv_transform_n::<8>(coeff, &T8, bit_depth, &mut out),
        16 => inv_transform_n::<16>(coeff, &T16, bit_depth, &mut out),
        _ => panic!("unsupported transform size {n}"),
    }
    out
}

#[inline]
fn inv_transform_n<const N: usize>(
    coeff: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    let bd = bit_depth as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bd;
    let add2 = 1i32 << (shift2 - 1);

    // i32 accumulation is exact (dequant clamps to ±32768); basis rows are read
    // contiguously and zero coefficients are skipped — residual blocks are
    // mostly zero, so the skip removes the bulk of the work.
    let mut tmp = [0i32; MAX_TB];
    let mut acc = [0i32; N];

    // Stage 1 (columns): tmp[m*N+c] = clip( Σ_k T[k][m]·coeff[k*N+c] ) >> 7
    for c in 0..N {
        acc[..N].fill(0);
        for k in 0..N {
            let ck = coeff[k * N + c];
            if ck == 0 {
                continue;
            }
            let trow = &t[k];
            for m in 0..N {
                acc[m] += trow[m] * ck;
            }
        }
        for m in 0..N {
            tmp[m * N + c] = ((acc[m] + add1) >> shift1).clamp(-32768, 32767);
        }
    }

    // Stage 2 (rows): out[r*N+m] = ( Σ_k T[k][m]·tmp[r*N+k] ) >> (20-bd)
    for r in 0..N {
        acc[..N].fill(0);
        let rowv = &tmp[r * N..r * N + N];
        for k in 0..N {
            let rk = rowv[k];
            if rk == 0 {
                continue;
            }
            let trow = &t[k];
            for m in 0..N {
                acc[m] += trow[m] * rk;
            }
        }
        for m in 0..N {
            out[r * N + m] = (acc[m] + add2) >> shift2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HEVC basis rows are *approximately* orthonormal: they're integer
    /// approximations, so norms cluster tightly around the ideal N·64² and
    /// off-diagonal correlations are small (tens–low hundreds), not zero. A
    /// structurally wrong row (e.g. permuted magnitudes) produces an
    /// off-diagonal dot orders of magnitude larger, which this catches.
    fn check_orthogonal<const N: usize>(t: &[[i32; N]; N]) {
        let ideal = (N as i64) * 64 * 64;
        for (i, row) in t.iter().enumerate() {
            let ni: i64 = row.iter().map(|&v| (v as i64) * (v as i64)).sum();
            assert!(
                (ni - ideal).abs() <= ideal / 500,
                "row {i} norm {ni} too far from ideal {ideal}"
            );
            for j in (i + 1)..N {
                let dot: i64 = (0..N).map(|k| (t[i][k] as i64) * (t[j][k] as i64)).sum();
                assert!(
                    dot.abs() < ideal / 50,
                    "rows {i},{j} insufficiently orthogonal: dot={dot} (limit {})",
                    ideal / 50
                );
            }
        }
    }

    #[test]
    fn t16_is_orthogonal() {
        check_orthogonal(&T16);
        check_orthogonal(&T8);
        check_orthogonal(&T4);
    }

    /// A flat (constant) residual must transform to a DC-only coefficient block:
    /// every AC coefficient is exactly zero. Strong structural check of the
    /// matrix + row/column pass layout for each N.
    fn flat_is_dc_only(n: usize) {
        let c = 100i32;
        let res = vec![c; n * n];
        let coeff = fwd_transform(&res, n, 8);
        assert_ne!(coeff[0], 0, "N={n}: DC should be non-zero");
        for (i, &v) in coeff[..n * n].iter().enumerate().skip(1) {
            assert_eq!(v, 0, "N={n}: AC coeff {i} should be zero, got {v}");
        }
    }

    #[test]
    fn flat_residual_dc_only() {
        flat_is_dc_only(4);
        flat_is_dc_only(8);
        flat_is_dc_only(16);
    }

    /// Forward quantization: coeff → level. Returns a fixed 256-entry buffer.
    fn quantize(coeff: &[i32], n: usize, qp: u8, bit_depth: u8) -> [i16; MAX_TB] {
        quantize_impl(coeff, n, qp, bit_depth, None)
    }

    /// Full pipeline residual → fwd → quant → dequant → inv reconstructs the
    /// residual within the quantization error. At low QP the error is small;
    /// a gross T16/scan/buffer bug would blow this far past tolerance.
    fn roundtrip_bounded(n: usize, qp: u8, max_mean_abs: f32) {
        // Smooth ramp + a mid-frequency component — exercises many coefficients.
        let mut res = vec![0i32; n * n];
        for r in 0..n {
            for col in 0..n {
                let v = 8 * (r as i32 + col as i32) + 20 * (((r + col) % 4) as i32) - 60;
                res[r * n + col] = v;
            }
        }
        let coeff = fwd_transform(&res, n, 8);
        let level = quantize(&coeff[..n * n], n, qp, 8);
        let dq = dequantize(&level[..n * n], n, qp, 8);
        let rec = inv_transform(&dq[..n * n], n, 8);
        let mean_abs: f32 = (0..n * n)
            .map(|i| (res[i] - rec[i]).unsigned_abs() as f32)
            .sum::<f32>()
            / (n * n) as f32;
        assert!(
            mean_abs <= max_mean_abs,
            "N={n} qp={qp}: mean|err|={mean_abs:.3} exceeds {max_mean_abs}"
        );
    }

    #[test]
    fn pipeline_roundtrip_low_qp() {
        // Low QP → tight reconstruction for every size, including the new 16×16.
        for &n in &[4usize, 8, 16] {
            roundtrip_bounded(n, 4, 3.0);
        }
    }

    #[test]
    fn pipeline_roundtrip_scales_with_qp() {
        // Error grows with QP but stays bounded; 16×16 must behave like 4/8.
        for &n in &[4usize, 8, 16] {
            roundtrip_bounded(n, 22, 12.0);
        }
    }

    /// The 16×16 sub-block-major scan must be a permutation of all 256 positions.
    #[test]
    fn zigzag16_is_permutation() {
        let mut seen = [false; 256];
        for &(r, c) in crate::dct::ZIGZAG_16X16.iter() {
            assert!(r < 16 && c < 16);
            let idx = r * 16 + c;
            assert!(!seen[idx], "duplicate scan position ({r},{c})");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b), "scan does not cover all positions");
    }

    #[test]
    fn post_rdoq_sign_hiding_matches_direct_quantizer() {
        let scan = crate::dct::coeff_scan(3, 0);
        let mut coeff = [0i32; MAX_TB];
        for (i, &(row, col)) in scan.iter().enumerate().take(32) {
            let magnitude = 80 + (i as i32 * 37) % 900;
            coeff[row * 8 + col] = if i % 3 == 0 { -magnitude } else { magnitude };
        }

        let direct = quantize_with_sign_hiding(&coeff, 8, 22, 8, scan);
        let mut split = quantize_impl(&coeff, 8, 22, 8, None);
        apply_sign_hiding_to_levels(&mut split, &coeff, 8, 22, 8, scan);
        assert_eq!(split, direct);
    }

    #[test]
    fn rdoq_zero_block_stays_zero() {
        let coeff = [0i32; MAX_TB];
        let scan = crate::dct::coeff_scan(3, 0);
        let ctx = ContextSet::init_islice(26);
        let levels = rdoq_luma_with_sign_hiding(
            &coeff,
            8,
            26,
            8,
            scan,
            0,
            0.57 * 2f32.powf((26.0 - 12.0) / 3.0),
            &ctx,
        );
        assert!(levels[..64].iter().all(|&level| level == 0));
    }

    #[test]
    fn rdoq_output_obeys_sign_hiding_parity() {
        let scan = crate::dct::coeff_scan(3, 0);
        let mut coeff = [0i32; MAX_TB];
        for (i, &(row, col)) in scan.iter().enumerate().take(24) {
            let magnitude = 600 + i as i32 * 53;
            coeff[row * 8 + col] = if i % 5 == 0 { -magnitude } else { magnitude };
        }
        let ctx = ContextSet::init_islice(4);
        let levels = rdoq_luma_with_sign_hiding(&coeff, 8, 4, 8, scan, 0, 0.001, &ctx);

        for group in scan.chunks_exact(16) {
            let first = group
                .iter()
                .position(|&(row, col)| levels[row * 8 + col] != 0);
            let last = group
                .iter()
                .rposition(|&(row, col)| levels[row * 8 + col] != 0);
            let (Some(first), Some(last)) = (first, last) else {
                continue;
            };
            if last - first < 4 {
                continue;
            }
            let sum: u32 = group[first..=last]
                .iter()
                .map(|&(row, col)| levels[row * 8 + col].unsigned_abs() as u32)
                .sum();
            let (row, col) = group[first];
            assert_eq!((sum & 1) as i32, (levels[row * 8 + col] < 0) as i32);
        }
    }

    #[test]
    fn sign_hiding_fixes_group_parity() {
        let scan = crate::dct::coeff_scan(2, 0);
        let mut levels = [0i16; MAX_TB];
        let mut coeff = [0i32; MAX_TB];
        let delta_u = [0i32; MAX_TB];

        let (r0, c0) = scan[0];
        let (r4, c4) = scan[4];
        let p0 = r0 * 4 + c0;
        let p4 = r4 * 4 + c4;
        levels[p0] = -1;
        levels[p4] = 3;
        coeff[p0] = -100;
        coeff[p4] = 300;

        sign_bit_hiding_hdq(&mut levels, &coeff, &delta_u, 4, scan);

        assert_eq!(levels[p0], -1);
        assert_eq!(levels[p4], 2);
        let parity = levels[p0].unsigned_abs() as u32 + levels[p4].unsigned_abs() as u32;
        assert_eq!((parity & 1) as i32, (levels[p0] < 0) as i32);
    }

    #[test]
    fn sign_hiding_uses_each_coefficient_group_independently() {
        let scan = crate::dct::coeff_scan(3, 0);
        let mut levels = [0i16; MAX_TB];
        let mut coeff = [0i32; MAX_TB];
        let delta_u = [0i32; MAX_TB];

        for group in 0..4 {
            let first = group * 16;
            let last = first + 4;
            let (r0, c0) = scan[first];
            let (r4, c4) = scan[last];
            let p0 = r0 * 8 + c0;
            let p4 = r4 * 8 + c4;
            levels[p0] = -1;
            levels[p4] = 3;
            coeff[p0] = -100;
            coeff[p4] = 300;
        }

        sign_bit_hiding_hdq(&mut levels, &coeff, &delta_u, 8, scan);

        for group in 0..4 {
            let start = group * 16;
            let group_scan = &scan[start..start + 16];
            let first = (0..16)
                .find(|&i| {
                    let (r, c) = group_scan[i];
                    levels[r * 8 + c] != 0
                })
                .unwrap();
            let last = (0..16)
                .rev()
                .find(|&i| {
                    let (r, c) = group_scan[i];
                    levels[r * 8 + c] != 0
                })
                .unwrap();
            assert!(last - first >= 4, "group {group}");
            let sum: u32 = (first..=last)
                .map(|i| {
                    let (r, c) = group_scan[i];
                    levels[r * 8 + c].unsigned_abs() as u32
                })
                .sum();
            let (r, c) = group_scan[first];
            assert_eq!(
                (sum & 1) as i32,
                (levels[r * 8 + c] < 0) as i32,
                "group {group}"
            );
        }
    }
}
