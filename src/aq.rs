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
    VarianceBoost, Yuv,
    math::{FastRound, fmla},
};
use std::sync::OnceLock;

pub(crate) type CtuActivityFn = unsafe fn(&Yuv, usize, usize, u8) -> CtuActivity;

static CTU_ACTIVITY: OnceLock<CtuActivityFn> = OnceLock::new();

#[derive(Clone, Copy)]
pub(crate) struct CtuActivity {
    pub(crate) mean_log_variance: f32,
    pub(crate) low_contrast_log_variance: f32,
    pub(crate) mean_luma: f32,
    pub(crate) mid_frequency_energy: f32,
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

pub(crate) fn log1p_slice_scalar(values: &mut [f32]) {
    for value in values {
        *value = log1p(*value);
    }
}

#[inline]
fn variance_boost_qp(low_contrast_log_variance: f32, qp: u8, config: VarianceBoost) -> f32 {
    const LOW_VARIANCE_LOG_LIMIT: f32 = 5.549_076; // ln(1 + 256)
    let low_contrast = ((LOW_VARIANCE_LOG_LIMIT - low_contrast_log_variance)
        / LOW_VARIANCE_LOG_LIMIT)
        .clamp(0.0, 1.0);
    let strength = ((f32::from(qp) - 32.0) / 7.0).clamp(0.0, 1.0);
    low_contrast * strength * config.strength * 0.575
}

#[inline]
fn dark_structure_boost_qp(
    mean_luma: f32,
    mid_frequency_energy: f32,
    qp: u8,
    config: VarianceBoost,
) -> f32 {
    const MEAN_FLOOR: f32 = 16.0;
    const DARK_REFERENCE: f32 = 56.0;
    const DARK_GAMMA: f32 = 1.2;
    const MAX_DARK_WEIGHT: f32 = 4.0;
    // ln(1 + 64), the normalization denominator for persistent structure.
    const LOG1P_ENERGY_REFERENCE: f32 = 4.174_387_5;

    let dark_weight = ((MEAN_FLOOR + DARK_REFERENCE) / (MEAN_FLOOR + mean_luma))
        .powf(DARK_GAMMA)
        .clamp(1.0, MAX_DARK_WEIGHT);
    let darkness = dark_weight - 1.0;
    let persistent_structure = log1p(mid_frequency_energy * darkness) / LOG1P_ENERGY_REFERENCE;
    let qp_strength = ((f32::from(qp) - 32.0) / 7.0).clamp(0.0, 1.0);
    persistent_structure.clamp(0.0, 1.0) * qp_strength * config.strength * 0.75
}

#[allow(dead_code)]
pub(crate) fn luma_laplacian_energy(
    rows: &[u16],
    stride: usize,
    col0: usize,
    col_end: usize,
    shift: u8,
) -> f32 {
    if rows.len() < stride * 3 || col_end - col0 < 3 {
        return 0.0;
    }
    let all_rows = rows.chunks_exact(stride);
    let above = all_rows.clone();
    let center = all_rows.clone().skip(1);
    let below = all_rows.skip(2);
    let mut energy = 0.0f32;
    let mut samples = 0usize;
    for ((above, center), below) in above.zip(center).zip(below) {
        let vertical = above[col0 + 1..col_end - 1]
            .iter()
            .zip(&below[col0 + 1..col_end - 1]);
        for (horizontal, (&top, &bottom)) in
            center[col0..col_end].array_windows::<3>().zip(vertical)
        {
            let [left, middle, right] = *horizontal;
            let center = i32::from(middle >> shift);
            let laplacian = 4 * center
                - i32::from(left >> shift)
                - i32::from(right >> shift)
                - i32::from(top >> shift)
                - i32::from(bottom >> shift);
            energy += laplacian.unsigned_abs() as f32;
            samples += 1;
        }
    }
    energy / samples.max(1) as f32
}

#[allow(dead_code)]
pub(crate) fn f32_laplacian_energy(samples: &[f32], stride: usize) -> f32 {
    if stride < 3 || samples.len() < stride * 3 {
        return 0.0;
    }
    let all_rows = samples.chunks_exact(stride);
    let above = all_rows.clone();
    let center = all_rows.clone().skip(1);
    let below = all_rows.skip(2);
    let mut energy = 0.0f32;
    let mut count = 0usize;
    for ((above, center), below) in above.zip(center).zip(below) {
        let vertical = above[1..stride - 1].iter().zip(&below[1..stride - 1]);
        for (horizontal, (&top, &bottom)) in center.array_windows::<3>().zip(vertical) {
            let [left, middle, right] = *horizontal;
            energy += (4.0 * middle - left - right - top - bottom).abs();
            count += 1;
        }
    }
    energy / count.max(1) as f32
}

pub(crate) fn finish_ctu_activity(
    log_variances: &mut [f32],
    luma_sum: f32,
    luma_samples: f32,
    full_energy: f32,
    coarse_energy: f32,
    octile: u8,
) -> CtuActivity {
    let blocks = log_variances.len();
    if blocks == 0 {
        return CtuActivity {
            mean_log_variance: 0.0,
            low_contrast_log_variance: 0.0,
            mean_luma: 0.0,
            mid_frequency_energy: 0.0,
        };
    }
    let log_sum = log_variances.iter().sum::<f32>();
    log_variances.sort_unstable_by(f32::total_cmp);
    let ranked = |n: usize| log_variances[(blocks * n).div_ceil(8).saturating_sub(1)];
    let center = usize::from(octile);
    let low_contrast = (ranked(center.saturating_sub(1).max(1))
        + 2.0 * ranked(center)
        + ranked((center + 1).min(8)))
        * 0.25;
    CtuActivity {
        mean_log_variance: log_sum / blocks as f32,
        low_contrast_log_variance: low_contrast,
        mean_luma: luma_sum / luma_samples,
        mid_frequency_energy: (full_energy * coarse_energy).sqrt(),
    }
}

#[allow(dead_code)]
pub(crate) fn ctu_activity_scalar(
    yuv: &Yuv,
    ctu_row: usize,
    ctu_col: usize,
    octile: u8,
) -> CtuActivity {
    let width = yuv.width as usize;
    let height = yuv.height as usize;
    let row0 = ctu_row * 64;
    let col0 = ctu_col * 64;
    if row0 >= height || col0 >= width {
        return CtuActivity {
            mean_log_variance: 0.0,
            low_contrast_log_variance: 0.0,
            mean_luma: 0.0,
            mid_frequency_energy: 0.0,
        };
    }
    let shift = yuv.bit_depth.bits().saturating_sub(8);
    let mut luma_sum = 0.0f32;
    let mut luma_samples = 0.0f32;
    let mut log_variances = [0.0f32; 64];
    let mut variance_slots = log_variances.iter_mut();
    let row_end = (row0 + 64).min(height);
    let col_end = (col0 + 64).min(width);
    let ctu_rows = &yuv.y[row0 * width..row_end * width];
    for row_band in ctu_rows.chunks(width * 8) {
        let mut sums = [0.0f32; 8];
        let mut sums_sq = [0.0f32; 8];
        let mut counts = [0.0f32; 8];
        for row in row_band.chunks_exact(width) {
            for (((block, sum), sum_sq), count) in row[col0..col_end]
                .chunks(8)
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
            luma_sum += sum;
            luma_samples += count;
            let mean = sum / count;
            let variance = (sum_sq / count - mean * mean).max(0.0);
            *variance_slots
                .next()
                .expect("one variance slot per 8x8 CTU block") = variance;
        }
    }
    let blocks = 64 - variance_slots.len();
    let values = &mut log_variances[..blocks];
    log1p_slice_scalar(values);

    let ctu_width = col_end - col0;
    let coarse_width = ctu_width.div_ceil(2);
    let mut coarse = [0.0f32; 32 * 32];
    let mut coarse_slots = coarse.iter_mut();
    for row_pair in ctu_rows.chunks(width * 2) {
        let mut rows = row_pair.chunks_exact(width);
        let top = rows.next().expect("2x band contains a row");
        let bottom = rows.next().unwrap_or(top);
        for (top, bottom) in top[col0..col_end]
            .chunks(2)
            .zip(bottom[col0..col_end].chunks(2))
        {
            let sum = top
                .iter()
                .chain(bottom)
                .map(|&sample| f32::from(sample >> shift))
                .sum::<f32>();
            *coarse_slots
                .next()
                .expect("one coarse slot per 2x2 CTU block") =
                sum / (top.len() + bottom.len()) as f32;
        }
    }
    let coarse_len = 32 * 32 - coarse_slots.len();
    drop(coarse_slots);
    let full_energy = luma_laplacian_energy(ctu_rows, width, col0, col_end, shift);
    let coarse_energy = f32_laplacian_energy(&coarse[..coarse_len], coarse_width);
    finish_ctu_activity(
        values,
        luma_sum,
        luma_samples,
        full_energy,
        coarse_energy,
        octile,
    )
}

#[inline]
pub(crate) fn resolve_ctu_activity() -> CtuActivityFn {
    *CTU_ACTIVITY.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::ctu_activity_neon as CtuActivityFn
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let mut f = ctu_activity_scalar as CtuActivityFn;
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                f = crate::avx::ctu_activity_avx2 as CtuActivityFn;
            }
            f
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            ctu_activity_scalar as CtuActivityFn
        }
    })
}

#[inline]
pub(crate) fn activity_aq_enabled(qp: u8, lossless: bool) -> bool {
    !lossless && qp >= 24
}

pub(crate) fn activity_qp_offsets(
    yuv: &Yuv,
    ctus_x: usize,
    ctus_y: usize,
    qp: u8,
    lossless: bool,
    variance_boost: VarianceBoost,
    ctu_activity: CtuActivityFn,
) -> Vec<i8> {
    if !activity_aq_enabled(qp, lossless) {
        return Vec::new();
    }
    let mut activity = Vec::with_capacity(ctus_x * ctus_y);
    for row in 0..ctus_y {
        for col in 0..ctus_x {
            // SAFETY: dispatch targets share the slice-based scalar contract and
            // handle partial CTUs at the coded-picture edges.
            activity.push(unsafe { ctu_activity(yuv, row, col, variance_boost.octile) });
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
            masking.fast_round().clamp(-3.0, 3.0) as i8
        })
        .collect();

    if !variance_boost.boost_only {
        let rounded_mean = (offsets.iter().map(|&v| i32::from(v)).sum::<i32>() as f32
            / offsets.len().max(1) as f32)
            .fast_round() as i8;
        for offset in &mut offsets {
            *offset = (*offset - rounded_mean).clamp(-3, 3);
        }
    }
    for (offset, value) in offsets.iter_mut().zip(&activity) {
        let flat_boost = variance_boost_qp(value.low_contrast_log_variance, qp, variance_boost);
        let dark_boost = dark_structure_boost_qp(
            value.mean_luma,
            value.mid_frequency_energy,
            qp,
            variance_boost,
        );
        let protection = flat_boost.max(dark_boost).fast_round() as i8;
        *offset = (*offset - protection).clamp(-3, 3);
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
        let offsets = activity_qp_offsets(
            &yuv,
            2,
            1,
            38,
            false,
            VarianceBoost::default(),
            ctu_activity_scalar,
        );
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
    fn variance_boost_is_bounded_and_targets_coarse_low_contrast_blocks() {
        let config = VarianceBoost {
            strength: 2.0,
            ..VarianceBoost::default()
        };
        assert_eq!(variance_boost_qp(0.0, 32, config), 0.0);
        assert_eq!(variance_boost_qp(6.0, 39, config), 0.0);
        let boost = variance_boost_qp(2.0, 39, config);
        assert!(boost > 0.5 && boost <= 1.15, "unexpected boost {boost}");
    }

    #[test]
    fn dark_protection_targets_dark_persistent_structure_only() {
        let config = VarianceBoost {
            strength: 2.0,
            ..VarianceBoost::default()
        };
        let dark_structure = dark_structure_boost_qp(32.0, 20.0, 39, config);
        assert!(dark_structure > 0.5, "dark structure was not protected");
        assert_eq!(dark_structure_boost_qp(128.0, 20.0, 39, config), 0.0);
        assert_eq!(dark_structure_boost_qp(32.0, 0.0, 39, config), 0.0);
        assert_eq!(dark_structure_boost_qp(32.0, 20.0, 32, config), 0.0);
    }
}
