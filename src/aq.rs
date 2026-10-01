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
fn log1p(x: f32) -> f32 {
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
    let mut polynomial = -0.100_935_325_026_512_15_f32;
    polynomial = fmla(polynomial, t, 0.164_151_370_525_360_1);
    polynomial = fmla(polynomial, t, -0.173_346_474_766_731_26);
    polynomial = fmla(polynomial, t, 0.198_739_007_115_364_07);
    polynomial = fmla(polynomial, t, -0.249_593_034_386_634_83);
    polynomial = fmla(polynomial, t, 0.333_361_357_450_485_23);
    polynomial = fmla(polynomial, t, -0.500_006_675_720_214_8);
    polynomial = fmla(polynomial, t, 0.999_999_821_186_065_7);
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
    let shift = yuv.bit_depth.bits().saturating_sub(8);
    let (ctus_x, ctus_y) = (width.div_ceil(64), height.div_ceil(64));
    let mut picture_sum = 0.0f32;
    for ctu_row in 0..ctus_y {
        for ctu_col in 0..ctus_x {
            let (row0, col0) = (ctu_row * 64, ctu_col * 64);
            let (row_end, col_end) = ((row0 + 64).min(height), (col0 + 64).min(width));
            let mut log_sum = 0.0f32;
            let mut blocks = 0.0f32;
            for block_row in (row0..row_end).step_by(8) {
                let band_end = (block_row + 8).min(row_end);
                for block_col in (col0..col_end).step_by(8) {
                    let cols = (block_col + 8).min(col_end) - block_col;
                    let (mut sum, mut sum_sq, mut count) = (0.0f32, 0.0f32, 0.0f32);
                    for r in block_row..band_end {
                        for &sample in &yuv.y[r * width + block_col..r * width + block_col + cols] {
                            let sample = f32::from(sample >> shift);
                            sum += sample;
                            sum_sq = fmla(sample, sample, sum_sq);
                            count += 1.0;
                        }
                    }
                    let mean = sum / count;
                    log_sum += log1p((sum_sq / count - mean * mean).max(0.0));
                    blocks += 1.0;
                }
            }
            picture_sum += log_sum / blocks;
        }
    }
    picture_sum / (ctus_x * ctus_y).max(1) as f32
}

/// Mean `ln(1+variance)` of the 8×8 blocks inside each quantization group of
/// one CTU (Z order), from the unpadded source. Groups fully outside the
/// picture report `NaN` and are excluded from picture-level normalization.
fn ctu_qg_log_variances(yuv: &Yuv, ctu_row: usize, ctu_col: usize) -> [f32; QGS_PER_CTU] {
    let width = yuv.width as usize;
    let height = yuv.height as usize;
    let shift = yuv.bit_depth.bits().saturating_sub(8);
    let mut out = [f32::NAN; QGS_PER_CTU];
    for qr in 0..QGS_PER_SIDE {
        for qc in 0..QGS_PER_SIDE {
            let row0 = ctu_row * 64 + qr * QG_SIZE;
            let col0 = ctu_col * 64 + qc * QG_SIZE;
            if row0 >= height || col0 >= width {
                continue;
            }
            let row_end = (row0 + QG_SIZE).min(height);
            let col_end = (col0 + QG_SIZE).min(width);
            let mut log_sum = 0.0f32;
            let mut blocks = 0.0f32;
            for block_row in (row0..row_end).step_by(8) {
                let band_end = (block_row + 8).min(row_end);
                for block_col in (col0..col_end).step_by(8) {
                    let cols = (block_col + 8).min(col_end) - block_col;
                    let (mut sum, mut sum_sq, mut count) = (0.0f32, 0.0f32, 0.0f32);
                    for r in block_row..band_end {
                        for &sample in &yuv.y[r * width + block_col..r * width + block_col + cols] {
                            let sample = f32::from(sample >> shift);
                            sum += sample;
                            sum_sq = fmla(sample, sample, sum_sq);
                            count += 1.0;
                        }
                    }
                    let mean = sum / count;
                    log_sum += log1p((sum_sq / count - mean * mean).max(0.0));
                    blocks += 1.0;
                }
            }
            out[qg_z(qr, qc)] = log_sum / blocks.max(1.0);
        }
    }
    out
}

/// Per-quantization-group QP offsets, indexed `ctu_index * QGS_PER_CTU +
/// qg_z(..)`: activity masking against the picture mean of the
/// per-group mean 8×8 log-variance.
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
    let clamp_hi = f32::from(max_offset);
    let mut qg_log_variance = Vec::with_capacity(ctus_x * ctus_y * QGS_PER_CTU);
    for row in 0..ctus_y {
        for col in 0..ctus_x {
            qg_log_variance.extend_from_slice(&ctu_qg_log_variances(yuv, row, col));
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
