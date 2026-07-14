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
    aq::{CtuActivity, finish_ctu_activity},
};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn log1p_avx2(x: __m256) -> __m256 {
    let y = _mm256_add_ps(x, _mm256_set1_ps(1.0));
    let bits = _mm256_castps_si256(y);
    let mut exponent = _mm256_sub_epi32(
        _mm256_and_si256(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(0xff)),
        _mm256_set1_epi32(127),
    );
    let mut mantissa = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f80_0000u32 as i32),
    ));
    let reduce = _mm256_cmp_ps::<_CMP_GT_OQ>(mantissa, _mm256_set1_ps(std::f32::consts::SQRT_2));
    mantissa = _mm256_blendv_ps(
        mantissa,
        _mm256_mul_ps(mantissa, _mm256_set1_ps(0.5)),
        reduce,
    );
    exponent = _mm256_add_epi32(
        exponent,
        _mm256_and_si256(_mm256_castps_si256(reduce), _mm256_set1_epi32(1)),
    );

    let t = _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0));
    let mut polynomial = _mm256_set1_ps(-0.100_935_325_026_512_15);
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(0.164_151_370_525_360_1));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(-0.173_346_474_766_731_26));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(0.198_739_007_115_364_07));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(-0.249_593_034_386_634_83));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(0.333_361_357_450_485_23));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(-0.500_006_675_720_214_8));
    polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(0.999_999_821_186_065_7));
    _mm256_fmadd_ps(
        t,
        polynomial,
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(exponent),
            _mm256_set1_ps(std::f32::consts::LN_2),
        ),
    )
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn log1p_slice_avx2(values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for values in chunks {
        let x = unsafe { _mm256_loadu_ps(values.as_ptr()) };
        let result = log1p_avx2(x);
        unsafe { _mm256_storeu_ps(values.as_mut_ptr(), result) };
    }
    crate::aq::log1p_slice_scalar(remainder);
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn shifted_u16x8(values: &[u16; 8], shift: u8) -> __m256i {
    let values = unsafe { _mm_loadu_si128(values.as_ptr().cast()) };
    _mm256_srlv_epi32(
        _mm256_cvtepu16_epi32(values),
        _mm256_set1_epi32(i32::from(shift)),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum_ps(v: __m128) -> f32 {
    let mut shuf = _mm_movehdup_ps(v);
    let mut sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn horizontal_sum(values: __m256) -> f32 {
    let halves = _mm_add_ps(
        _mm256_castps256_ps128(values),
        _mm256_extractf128_ps::<1>(values),
    );
    hsum_ps(halves)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn sum_and_square(values: __m256i) -> (f32, f32) {
    let values = _mm256_cvtepi32_ps(values);
    (
        horizontal_sum(values),
        horizontal_sum(_mm256_mul_ps(values, values)),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn laplacian8(
    top: &[u16; 8],
    left: &[u16; 8],
    middle: &[u16; 8],
    right: &[u16; 8],
    bottom: &[u16; 8],
    shift: u8,
) -> f32 {
    let top = shifted_u16x8(top, shift);
    let left = shifted_u16x8(left, shift);
    let middle = shifted_u16x8(middle, shift);
    let right = shifted_u16x8(right, shift);
    let bottom = shifted_u16x8(bottom, shift);
    let laplacian = _mm256_sub_epi32(
        _mm256_sub_epi32(_mm256_slli_epi32::<2>(middle), left),
        _mm256_add_epi32(right, _mm256_add_epi32(top, bottom)),
    );
    horizontal_sum(_mm256_cvtepi32_ps(_mm256_abs_epi32(laplacian)))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn laplacian_energy_u16(
    rows: &[u16],
    stride: usize,
    col0: usize,
    col_end: usize,
    shift: u8,
) -> f32 {
    if rows.len() < stride * 3 || col_end - col0 < 3 {
        return 0.0;
    }
    let image_rows = rows.chunks_exact(stride);
    let above_rows = image_rows.clone();
    let center_rows = image_rows.clone().skip(1);
    let below_rows = image_rows.skip(2);
    let mut energy = 0.0;
    let mut count = 0usize;
    for ((above, center), below) in above_rows.zip(center_rows).zip(below_rows) {
        let (top, top_remainder) = above[col0 + 1..col_end - 1].as_chunks::<8>();
        let left = center[col0..col_end - 2].as_chunks::<8>().0;
        let middle = center[col0 + 1..col_end - 1].as_chunks::<8>().0;
        let right = center[col0 + 2..col_end].as_chunks::<8>().0;
        let bottom = below[col0 + 1..col_end - 1].as_chunks::<8>().0;
        for ((((top, left), middle), right), bottom) in
            top.iter().zip(left).zip(middle).zip(right).zip(bottom)
        {
            energy += laplacian8(top, left, middle, right, bottom, shift);
        }
        let vector_len = top.len() * 8;
        for (offset, &top) in top_remainder.iter().enumerate() {
            let column = col0 + 1 + vector_len + offset;
            let middle = i32::from(center[column] >> shift);
            let laplacian = 4 * middle
                - i32::from(center[column - 1] >> shift)
                - i32::from(center[column + 1] >> shift)
                - i32::from(top >> shift)
                - i32::from(below[column] >> shift);
            energy += laplacian.unsigned_abs() as f32;
        }
        count += col_end - col0 - 2;
    }
    energy / count as f32
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn laplacian_energy_f32(samples: &[f32], stride: usize) -> f32 {
    if stride < 3 || samples.len() < stride * 3 {
        return 0.0;
    }
    let rows = samples.chunks_exact(stride);
    let above_rows = rows.clone();
    let center_rows = rows.clone().skip(1);
    let below_rows = rows.skip(2);
    let mut energy = 0.0;
    let mut count = 0usize;
    for ((above, center), below) in above_rows.zip(center_rows).zip(below_rows) {
        let (top, top_remainder) = above[1..stride - 1].as_chunks::<8>();
        let left = center[..stride - 2].as_chunks::<8>().0;
        let middle = center[1..stride - 1].as_chunks::<8>().0;
        let right = center[2..stride].as_chunks::<8>().0;
        let bottom = below[1..stride - 1].as_chunks::<8>().0;
        for ((((top, left), middle), right), bottom) in
            top.iter().zip(left).zip(middle).zip(right).zip(bottom)
        {
            let top = unsafe { _mm256_loadu_ps(top.as_ptr()) };
            let left = unsafe { _mm256_loadu_ps(left.as_ptr()) };
            let middle = unsafe { _mm256_loadu_ps(middle.as_ptr()) };
            let right = unsafe { _mm256_loadu_ps(right.as_ptr()) };
            let bottom = unsafe { _mm256_loadu_ps(bottom.as_ptr()) };
            let laplacian = _mm256_sub_ps(
                _mm256_sub_ps(_mm256_mul_ps(middle, _mm256_set1_ps(4.0)), left),
                _mm256_add_ps(right, _mm256_add_ps(top, bottom)),
            );
            energy += horizontal_sum(_mm256_andnot_ps(_mm256_set1_ps(-0.0), laplacian));
        }
        let vector_len = top.len() * 8;
        for offset in 0..top_remainder.len() {
            let column = 1 + vector_len + offset;
            energy += (4.0 * center[column]
                - center[column - 1]
                - center[column + 1]
                - above[column]
                - below[column])
                .abs();
        }
        count += stride - 2;
    }
    energy / count as f32
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn ctu_activity_avx2(
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
    let row_end = (row0 + 64).min(height);
    let col_end = (col0 + 64).min(width);
    let ctu_rows = &yuv.y[row0 * width..row_end * width];
    let blocks_wide = (col_end - col0).div_ceil(8);
    let mut variances = [0.0f32; 64];
    let mut blocks = 0usize;
    let mut luma_sum = 0.0;
    let mut luma_samples = 0.0;

    for row_band in ctu_rows.chunks(width * 8) {
        let mut sums = [0.0f32; 8];
        let mut sums_sq = [0.0f32; 8];
        let mut counts = [0.0f32; 8];
        for row in row_band.chunks_exact(width) {
            let region = &row[col0..col_end];
            let (vectors, remainder) = region.as_chunks::<8>();
            for (index, values) in vectors.iter().enumerate() {
                let (sum, sum_sq) = sum_and_square(shifted_u16x8(values, shift));
                sums[index] += sum;
                sums_sq[index] += sum_sq;
                counts[index] += 8.0;
            }
            if !remainder.is_empty() {
                let index = vectors.len();
                for &sample in remainder {
                    let sample = f32::from(sample >> shift);
                    sums[index] += sample;
                    sums_sq[index] += sample * sample;
                    counts[index] += 1.0;
                }
            }
        }
        for index in 0..blocks_wide {
            let mean = sums[index] / counts[index];
            variances[blocks] = (sums_sq[index] / counts[index] - mean * mean).max(0.0);
            luma_sum += sums[index];
            luma_samples += counts[index];
            blocks += 1;
        }
    }
    log1p_slice_avx2(&mut variances[..blocks]);

    let coarse_width = (col_end - col0).div_ceil(2);
    let mut coarse = [0.0f32; 32 * 32];
    let mut coarse_len = 0usize;
    for row_pair in ctu_rows.chunks(width * 2) {
        let mut rows = row_pair.chunks_exact(width);
        let top = rows.next().unwrap();
        let bottom = rows.next().unwrap_or(top);
        let top = &top[col0..col_end];
        let bottom = &bottom[col0..col_end];
        let (top_vectors, top_remainder) = top.as_chunks::<8>();
        let (bottom_vectors, bottom_remainder) = bottom.as_chunks::<8>();
        for (top, bottom) in top_vectors.iter().zip(bottom_vectors) {
            let rows = _mm256_add_epi32(shifted_u16x8(top, shift), shifted_u16x8(bottom, shift));
            let paired = _mm256_hadd_epi32(rows, rows);
            let ordered =
                _mm256_permutevar8x32_epi32(paired, _mm256_setr_epi32(0, 1, 4, 5, 0, 1, 4, 5));
            let averaged = _mm_mul_ps(
                _mm256_castps256_ps128(_mm256_cvtepi32_ps(ordered)),
                _mm_set1_ps(0.25),
            );
            unsafe { _mm_storeu_ps(coarse[coarse_len..].as_mut_ptr(), averaged) };
            coarse_len += 4;
        }
        for (top, bottom) in top_remainder.chunks(2).zip(bottom_remainder.chunks(2)) {
            let sum = top
                .iter()
                .chain(bottom)
                .map(|&sample| f32::from(sample >> shift))
                .sum::<f32>();
            coarse[coarse_len] = sum / (top.len() + bottom.len()) as f32;
            coarse_len += 1;
        }
    }

    let full_energy = laplacian_energy_u16(ctu_rows, width, col0, col_end, shift);
    let coarse_energy = laplacian_energy_f32(&coarse[..coarse_len], coarse_width);
    finish_ctu_activity(
        &mut variances[..blocks],
        luma_sum,
        luma_samples,
        full_energy,
        coarse_energy,
        octile,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitDepth, ChromaFormat};

    #[test]
    fn ctu_activity_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for bit_depth in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            let shift = bit_depth.bits() - 8;
            for (width, height) in [
                (1usize, 1usize),
                (2, 7),
                (7, 2),
                (9, 11),
                (64, 64),
                (79, 67),
                (129, 65),
            ] {
                let y = (0..width * height)
                    .map(|i| (((i * 73 + i / width * 29 + 17) & 255) as u16) << shift)
                    .collect();
                let yuv = Yuv {
                    y,
                    cb: Vec::new(),
                    cr: Vec::new(),
                    width: width as u32,
                    height: height as u32,
                    display_w: width as u32,
                    display_h: height as u32,
                    chroma: ChromaFormat::Monochrome,
                    bit_depth,
                };
                for row in 0..height.div_ceil(64) {
                    for col in 0..width.div_ceil(64) {
                        let actual = unsafe { ctu_activity_avx2(&yuv, row, col, 6) };
                        let expected = crate::aq::ctu_activity_scalar(&yuv, row, col, 6);
                        assert!(
                            (actual.mean_log_variance - expected.mean_log_variance).abs() < 2.0e-5
                        );
                        assert!(
                            (actual.low_contrast_log_variance - expected.low_contrast_log_variance)
                                .abs()
                                < 2.0e-5
                        );
                        assert!((actual.mean_luma - expected.mean_luma).abs() < 2.0e-5);
                        assert!(
                            (actual.mid_frequency_energy - expected.mid_frequency_energy).abs()
                                < 2.0e-5
                        );
                    }
                }
            }
        }
    }
}
