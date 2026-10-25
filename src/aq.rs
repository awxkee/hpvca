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
    Yuv,
    math::{FastRound, fmla},
};
use std::sync::OnceLock;

type BlockLogVariancesFn = unsafe fn(&Yuv, &mut [f32]);
type QgLogVariancesFn = unsafe fn(&[f32], usize, usize, usize, usize) -> [f32; QGS_PER_CTU];

struct AqFunctions {
    blocks: BlockLogVariancesFn,
    groups: QgLogVariancesFn,
}

fn resolve_aq() -> &'static AqFunctions {
    static AQ: OnceLock<AqFunctions> = OnceLock::new();
    AQ.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            AqFunctions {
                blocks: crate::neon::block_log_variances_neon,
                groups: crate::neon::ctu_qg_log_variances_neon,
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return AqFunctions {
                    blocks: crate::avx::block_log_variances_avx2_fma,
                    groups: crate::avx::ctu_qg_log_variances_avx2_fma,
                };
            }
            AqFunctions {
                blocks: block_log_variances_scalar,
                groups: ctu_qg_log_variances_scalar,
            }
        }
    })
}

const ACTIVITY_QP_FLOOR: u8 = 8;

/// QP offset per nat of 8×8 log-variance above/below the picture mean.
const ACTIVITY_STRENGTH: f32 = 1.625;
pub(crate) const MAX_AQ_OFFSET: i8 = 6;

pub(crate) const QG_LOG2: u32 = 5;
pub(crate) const QG_SIZE: usize = 1 << QG_LOG2;
const QGS_PER_SIDE: usize = 64 / QG_SIZE;
/// Quantization groups per 64×64 CTU; the AQ map holds this many offsets per
/// CTU in Z-scan order.
pub(crate) const QGS_PER_CTU: usize = QGS_PER_SIDE * QGS_PER_SIDE;

pub(crate) const LOG1P_COEFFICIENTS: [f32; 8] = [
    -0.100_935_325_026_512_15,
    0.164_151_370_525_360_1,
    -0.173_346_474_766_731_26,
    0.198_739_007_115_364_07,
    -0.249_593_034_386_634_83,
    0.333_361_357_450_485_23,
    -0.500_006_675_720_214_8,
    0.999_999_821_186_065_7,
];

/// Z-scan index of the quantization group at (`row`, `col`) — in QG units
/// within its CTU.
pub(crate) const fn qg_z(row: usize, col: usize) -> usize {
    let mut z = 0;
    let mut bit = 0;
    while (1 << bit) < QGS_PER_SIDE {
        z |= ((col >> bit) & 1) << (2 * bit);
        z |= ((row >> bit) & 1) << (2 * bit + 1);
        bit += 1;
    }
    z
}

/// Natural `log(1+x)` for the non-negative
///
/// `1+x` is reduced to `m * 2^e`, with
/// `m ∈ [sqrt(0.5), sqrt(2)]`, then `log(m)` is evaluated by an
/// eighth-degree float polynomial. The coefficients and error bound were
/// generated with this Sollya script (Sollya 8.0):
///
/// ```text
/// display = decimal;
/// I = [-0.2928932188134524; 0.4142135623730951];
/// P = fpminimax(log(1+x), [|1,2,3,4,5,6,7,8|],
///               [|SG...|], I, absolute);
/// P;
/// dirtyinfnorm(P-log(1+x), I);
/// // 3.3789931067747551148516505608241513991692761013372e-8
/// ```
#[inline]
pub(crate) fn log1p(x: f32) -> f32 {
    debug_assert!(x >= 0.0 && x.is_finite());
    let y = 1.0 + x;
    let bits = y.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if mantissa > std::f32::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let t = mantissa - 1.0;
    let mut polynomial = LOG1P_COEFFICIENTS[0];
    for &coefficient in &LOG1P_COEFFICIENTS[1..] {
        polynomial = fmla(polynomial, t, coefficient);
    }
    fmla(t, polynomial, exponent as f32 * std::f32::consts::LN_2)
}

#[inline]
pub(crate) fn activity_aq_enabled(qp: u8, lossless: bool) -> bool {
    !lossless && qp >= ACTIVITY_QP_FLOOR
}

/// Mean over CTUs of each CTU's mean `ln(1+variance)` of its 8×8 luma blocks
/// (blocks clipped to the picture). A coarse texture statistic for picking the
/// picture's chroma QP offset.
pub(crate) fn picture_mean_ctu_log_variance(yuv: &Yuv) -> f32 {
    let width = yuv.width as usize;
    let height = yuv.height as usize;
    let (blocks, bw, bh) = block_log_variances(yuv);
    let (ctus_x, ctus_y) = (width.div_ceil(64), height.div_ceil(64));
    let mut picture_sum = 0.0f32;
    for ctu_row in 0..ctus_y {
        for ctu_col in 0..ctus_x {
            let (row0, col0) = (ctu_row * 8, ctu_col * 8);
            let (row_end, col_end) = ((row0 + 8).min(bh), (col0 + 8).min(bw));
            let mut log_sum = 0.0f32;
            for r in row0..row_end {
                for &value in &blocks[r * bw + col0..r * bw + col_end] {
                    log_sum += value;
                }
            }
            picture_sum += log_sum / ((row_end - row0) * (col_end - col0)) as f32;
        }
    }
    picture_sum / (ctus_x * ctus_y).max(1) as f32
}

/// Temperature (nats) of the soft maximum that pools a quantization group's
/// 8×8 log-variances. A group takes most of its masking from its busiest
/// blocks; the mean or a soft minimum (sparing smooth parts of mixed groups)
/// both measured worse on ssimulacra2 and ColorVideoVDP.
pub(crate) const QG_SOFTMAX_TEMPERATURE: f32 = 2.0;

/// `ln(1+variance)` of every 8×8 luma block of the picture (blocks clipped to
/// the picture), row-major over a `div_ceil(8)` grid.
fn block_log_variances(yuv: &Yuv) -> (Vec<f32>, usize, usize) {
    block_log_variances_with(yuv, resolve_aq().blocks)
}

fn block_log_variances_with(yuv: &Yuv, blocks: BlockLogVariancesFn) -> (Vec<f32>, usize, usize) {
    let (bw, bh) = (
        (yuv.width as usize).div_ceil(8),
        (yuv.height as usize).div_ceil(8),
    );
    let mut out = vec![0.0; bw * bh];
    // AArch64 has baseline NEON; x86 requires both AVX2 and FMA. Resolve once
    // per picture, outside the block and quantization-group loops.
    unsafe { blocks(yuv, &mut out) };
    (out, bw, bh)
}

/// Integer moments are exact after normalization to eight bits: at most
/// 64 * 255², which also converts to f32 without rounding.
#[inline]
pub(crate) fn variance_from_moments(sum: u32, sum_sq: u32, count: usize) -> f32 {
    let mean = sum as f32 / count as f32;
    (sum_sq as f32 / count as f32 - mean * mean).max(0.0)
}

#[inline]
pub(crate) fn block_variance_scalar(yuv: &Yuv, row: usize, col: usize) -> f32 {
    let width = yuv.width as usize;
    let height = yuv.height as usize;
    let shift = yuv.bit_depth.bits().saturating_sub(8);
    let (rows, cols) = ((height - row).min(8), (width - col).min(8));
    let (mut sum, mut sum_sq) = (0u32, 0u32);
    for r in row..row + rows {
        for &sample in &yuv.y[r * width + col..r * width + col + cols] {
            let sample = u32::from(sample >> shift);
            sum += sample;
            sum_sq += sample * sample;
        }
    }
    variance_from_moments(sum, sum_sq, rows * cols)
}

#[cfg_attr(all(target_arch = "aarch64", feature = "neon"), allow(dead_code))]
fn block_log_variances_scalar(yuv: &Yuv, out: &mut [f32]) {
    let width = yuv.width as usize;
    let height = yuv.height as usize;
    let bw = width.div_ceil(8);
    assert_eq!(out.len(), bw * height.div_ceil(8));
    for row in (0..height).step_by(8) {
        for col in (0..width).step_by(8) {
            out[row / 8 * bw + col / 8] = log1p(block_variance_scalar(yuv, row, col));
        }
    }
}

/// Soft maximum of the 8×8 log-variances inside each quantization group of
/// one CTU (Z order). Groups fully outside the picture report `NaN` and are
/// excluded from picture-level normalization.
#[cfg_attr(all(target_arch = "aarch64", feature = "neon"), allow(dead_code))]
pub(crate) fn ctu_qg_log_variances_scalar(
    blocks: &[f32],
    bw: usize,
    bh: usize,
    ctu_row: usize,
    ctu_col: usize,
) -> [f32; QGS_PER_CTU] {
    const BLOCKS_PER_QG: usize = QG_SIZE / 8;
    let mut out = [f32::NAN; QGS_PER_CTU];
    for qr in 0..QGS_PER_SIDE {
        for qc in 0..QGS_PER_SIDE {
            let row0 = (ctu_row * 64 + qr * QG_SIZE) / 8;
            let col0 = (ctu_col * 64 + qc * QG_SIZE) / 8;
            if row0 >= bh || col0 >= bw {
                continue;
            }
            let (row_end, col_end) = (
                (row0 + BLOCKS_PER_QG).min(bh),
                (col0 + BLOCKS_PER_QG).min(bw),
            );
            out[qg_z(qr, qc)] =
                qg_log_variance_scalar(blocks, bw, row0, col0, row_end - row0, col_end - col0);
        }
    }
    out
}

#[inline]
pub(crate) fn qg_log_variance_scalar(
    blocks: &[f32],
    bw: usize,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> f32 {
    let values =
        (row..row + rows).flat_map(|r| blocks[r * bw + col..r * bw + col + cols].iter().copied());
    // Log-sum-exp anchored at the maximum, so every term is ≤ 1.
    let max = values.clone().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = values
        .map(|v| ((v - max) / QG_SOFTMAX_TEMPERATURE).exp())
        .sum();
    max + QG_SOFTMAX_TEMPERATURE * (sum / (rows * cols) as f32).ln()
}

/// Per-quantization-group QP offsets, indexed `ctu_index * QGS_PER_CTU +
/// qg_z(..)`: activity masking against the picture mean of the
/// per-group soft-maximum 8×8 log-variance.
/// SIMD approximates exponentials and reassociates softmax sums, so values at
/// rounding boundaries can produce adjacent offsets on different backends.
pub(crate) fn activity_qp_offsets(
    yuv: &Yuv,
    ctus_x: usize,
    ctus_y: usize,
    qp: u8,
    lossless: bool,
) -> Vec<i8> {
    activity_qp_offsets_clamped(yuv, ctus_x, ctus_y, qp, lossless, MAX_AQ_OFFSET)
}

/// [`activity_qp_offsets`] with a caller-chosen clamp. Direct coding requires
/// ±[`MAX_AQ_OFFSET`] (the lambda-scale table), but a gridded encode computes the picture map
/// with a wide clamp so each cell can lift its mean into its slice QP and
/// re-center the residuals — otherwise cell-level dynamic range is lost to
/// the clamp before the cells ever see the map.
pub(crate) fn activity_qp_offsets_clamped(
    yuv: &Yuv,
    ctus_x: usize,
    ctus_y: usize,
    qp: u8,
    lossless: bool,
    max_offset: i8,
) -> Vec<i8> {
    if !activity_aq_enabled(qp, lossless) {
        return Vec::new();
    }
    activity_qp_offsets_impl(yuv, ctus_x, ctus_y, max_offset, resolve_aq())
}

fn activity_qp_offsets_impl(
    yuv: &Yuv,
    ctus_x: usize,
    ctus_y: usize,
    max_offset: i8,
    functions: &AqFunctions,
) -> Vec<i8> {
    let clamp_hi = f32::from(max_offset);
    let (blocks, bw, bh) = block_log_variances_with(yuv, functions.blocks);
    let mut qg_log_variance = Vec::with_capacity(ctus_x * ctus_y * QGS_PER_CTU);
    let groups = functions.groups;
    for row in 0..ctus_y {
        for col in 0..ctus_x {
            qg_log_variance.extend_from_slice(&unsafe { groups(&blocks, bw, bh, row, col) });
        }
    }
    let valid = qg_log_variance.iter().filter(|v| !v.is_nan());
    let valid_count = valid.clone().count().max(1) as f32;
    let mean = valid.sum::<f32>() / valid_count;
    let mut offsets: Vec<i8> = qg_log_variance
        .iter()
        .map(|&log_variance| {
            let masking = if log_variance.is_nan() {
                0.0
            } else {
                (log_variance - mean) * ACTIVITY_STRENGTH
            };
            // Coding requires ±MAX_AQ_OFFSET (`code_one_ctu`'s λ-scale table);
            // gridded analysis widens this and re-clamps after cell rebasing.
            masking.fast_round().clamp(-clamp_hi, clamp_hi) as i8
        })
        .collect();
    let rounded_mean = (offsets.iter().map(|&v| i32::from(v)).sum::<i32>() as f32
        / offsets.len().max(1) as f32)
        .fast_round() as i8;
    for offset in &mut offsets {
        *offset = (*offset - rounded_mean).clamp(-max_offset, max_offset);
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitDepth, ChromaFormat};

    fn test_yuv(w: usize, h: usize, bit_depth: BitDepth, pattern: usize) -> Yuv {
        let shift = bit_depth.bits() - 8;
        let mask = (1u32 << bit_depth.bits()) - 1;
        let mut state = 0x1234_5678u32;
        let y = (0..w * h)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                match pattern {
                    0 => mask as u16,
                    1 => {
                        if (i / w + i % w).is_multiple_of(2) {
                            mask as u16
                        } else {
                            0
                        }
                    }
                    2 => {
                        if i == w * h / 2 {
                            mask as u16
                        } else {
                            0
                        }
                    }
                    3 => (254 + (i % 2) as u16) << shift,
                    4 => ((state & mask) >> ((i / w / 8 + i % w / 8) % 8)) as u16,
                    _ => (state & mask) as u16,
                }
            })
            .collect();
        Yuv {
            y,
            cb: Vec::new(),
            cr: Vec::new(),
            width: w as u32,
            height: h as u32,
            display_w: w as u32,
            display_h: h as u32,
            chroma: ChromaFormat::Monochrome,
            bit_depth,
        }
    }

    // Original floating-point pixel traversal, with libm for the logarithm.
    fn reference_block_logs(yuv: &Yuv) -> Vec<f32> {
        let (w, h) = (yuv.width as usize, yuv.height as usize);
        let shift = yuv.bit_depth.bits().saturating_sub(8);
        let mut out = Vec::new();
        for row in (0..h).step_by(8) {
            for col in (0..w).step_by(8) {
                let (mut sum, mut squares, mut count) = (0.0f32, 0.0f32, 0.0f32);
                for r in row..(row + 8).min(h) {
                    for c in col..(col + 8).min(w) {
                        let x = f32::from(yuv.y[r * w + c] >> shift);
                        sum += x;
                        squares = fmla(x, x, squares);
                        count += 1.0;
                    }
                }
                let mean = sum / count;
                out.push((squares / count - mean * mean).max(0.0).ln_1p());
            }
        }
        out
    }

    #[test]
    fn block_statistics_match_reference_at_all_edges_and_bit_depths() {
        for depth in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for w in [
                0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 71, 72, 73, 127, 129,
            ] {
                for h in [0, 1, 7, 8, 9, 31, 32, 33, 64, 65] {
                    for pattern in 0..6 {
                        let yuv = test_yuv(w, h, depth, pattern);
                        let reference = reference_block_logs(&yuv);
                        let (actual, bw, bh) = block_log_variances(&yuv);
                        let mut scalar = vec![0.0; bw * bh];
                        block_log_variances_scalar(&yuv, &mut scalar);
                        assert_eq!(actual.len(), reference.len());
                        for ((&actual, &scalar), &reference) in
                            actual.iter().zip(&scalar).zip(&reference)
                        {
                            assert!(
                                (actual - reference).abs() <= 2e-6,
                                "{w}x{h}, {depth:?}, pattern={pattern}: {actual} != {reference}"
                            );
                            assert!((scalar - reference).abs() <= 2e-6);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn softmax_matches_scalar_for_full_clipped_and_missing_groups() {
        for bw in 1usize..=17 {
            for bh in 1usize..=17 {
                for pattern in 0..4 {
                    let blocks: Vec<f32> = (0..bw * bh)
                        .map(|i| match pattern {
                            0 => 9.69,
                            1 => {
                                if i % 7 == 0 {
                                    9.69
                                } else {
                                    0.0
                                }
                            }
                            2 => (i % 97) as f32 * 0.1,
                            _ => ((i * 331 + i / bw * 733) % 970) as f32 * 0.01,
                        })
                        .collect();
                    for row in 0..=bh.div_ceil(8) {
                        for col in 0..=bw.div_ceil(8) {
                            let expected = ctu_qg_log_variances_scalar(&blocks, bw, bh, row, col);
                            let actual =
                                unsafe { (resolve_aq().groups)(&blocks, bw, bh, row, col) };
                            for (actual, expected) in actual.into_iter().zip(expected) {
                                assert!(
                                    if expected.is_nan() {
                                        actual.is_nan()
                                    } else {
                                        (actual - expected).abs() <= 3e-6
                                    },
                                    "grid={bw}x{bh}, CTU=({row},{col}): {actual} != {expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dispatched_aq_offsets_match_scalar() {
        let scalar = AqFunctions {
            blocks: block_log_variances_scalar,
            groups: ctu_qg_log_variances_scalar,
        };
        for depth in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for (w, h) in [
                (1, 1),
                (31, 65),
                (65, 31),
                (63, 63),
                (64, 64),
                (65, 65),
                (127, 129),
                (193, 137),
            ] {
                for pattern in 0..6 {
                    let yuv = test_yuv(w, h, depth, pattern);
                    for clamp in [MAX_AQ_OFFSET, 24] {
                        let expected = activity_qp_offsets_impl(
                            &yuv,
                            w.div_ceil(64),
                            h.div_ceil(64),
                            clamp,
                            &scalar,
                        );
                        assert_eq!(
                            activity_qp_offsets_clamped(
                                &yuv,
                                w.div_ceil(64),
                                h.div_ceil(64),
                                38,
                                false,
                                clamp
                            ),
                            expected,
                            "{w}x{h}, {depth:?}, pattern={pattern}, clamp={clamp}"
                        );
                    }
                    assert!(
                        activity_qp_offsets(&yuv, w.div_ceil(64), h.div_ceil(64), 7, false)
                            .is_empty()
                    );
                    assert!(
                        activity_qp_offsets(&yuv, w.div_ceil(64), h.div_ceil(64), 38, true)
                            .is_empty()
                    );
                }
            }
        }
    }

    #[test]
    fn picture_statistic_weights_ctus_equally_at_picture_edges() {
        for (w, h) in [
            (0, 0),
            (1, 1),
            (7, 9),
            (31, 65),
            (65, 31),
            (65, 65),
            (193, 137),
        ] {
            let yuv = test_yuv(w, h, BitDepth::Twelve, 4);
            let blocks = reference_block_logs(&yuv);
            let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
            let mut means = Vec::new();
            for row in (0..bh).step_by(8) {
                for col in (0..bw).step_by(8) {
                    let values: Vec<f32> = (row..(row + 8).min(bh))
                        .flat_map(|r| {
                            blocks[r * bw + col..r * bw + (col + 8).min(bw)]
                                .iter()
                                .copied()
                        })
                        .collect();
                    means.push(values.iter().sum::<f32>() / values.len() as f32);
                }
            }
            let expected = means.iter().sum::<f32>() / means.len().max(1) as f32;
            assert!((picture_mean_ctu_log_variance(&yuv) - expected).abs() <= 3e-6);
        }
    }

    #[test]
    fn local_log1p_tracks_libm_over_aq_range() {
        for bits in 0..=16_384u32 {
            let x = bits as f32;
            let error = (log1p(x) - x.ln_1p()).abs();
            assert!(error <= 2.0e-6, "x={x}, error={error}");
        }
    }

    #[test]
    fn activity_aq_moves_bits_from_flat_to_textured_ctus() {
        let (w, h) = (128usize, 64usize);
        let mut y = vec![128u16; w * h];
        for row in y.chunks_exact_mut(w) {
            for (index, sample) in row[64..].iter_mut().enumerate() {
                *sample = if index & 1 == 0 { 16 } else { 240 };
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
            chroma: ChromaFormat::Monochrome,
            bit_depth: BitDepth::Eight,
        };
        let offsets = activity_qp_offsets(&yuv, 2, 1, 38, false);
        // Offsets are per quantization group, QGS_PER_CTU per CTU in Z order.
        assert_eq!(offsets.len(), 2 * QGS_PER_CTU);
        assert!(
            offsets[..QGS_PER_CTU].iter().all(|&offset| offset < 0),
            "flat CTU groups should spend more bits: {offsets:?}"
        );
        assert!(
            offsets[QGS_PER_CTU..].iter().all(|&offset| offset > 0),
            "textured CTU groups should spend fewer bits: {offsets:?}"
        );
        assert!(
            offsets
                .iter()
                .all(|&offset| (-MAX_AQ_OFFSET..=MAX_AQ_OFFSET).contains(&offset))
        );
    }
}
