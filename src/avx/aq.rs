/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

use super::math::exp_f32x8;
use crate::Yuv;
use crate::aq::{
    LOG1P_COEFFICIENTS, QG_SIZE, QG_SOFTMAX_TEMPERATURE, QGS_PER_CTU, block_variance_scalar, log1p,
    qg_log_variance_scalar, qg_z, variance_from_moments,
};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn log1p_f32x8(x: __m256) -> __m256 {
    let bits = _mm256_castps_si256(_mm256_add_ps(x, _mm256_set1_ps(1.0)));
    let exponent = _mm256_sub_epi32(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(127));
    let mantissa = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f80_0000),
    ));
    let high = _mm256_cmp_ps::<_CMP_GT_OQ>(mantissa, _mm256_set1_ps(std::f32::consts::SQRT_2));
    let mantissa = _mm256_blendv_ps(mantissa, _mm256_mul_ps(mantissa, _mm256_set1_ps(0.5)), high);
    let exponent = _mm256_sub_epi32(exponent, _mm256_castps_si256(high));
    let t = _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0));
    let mut p = _mm256_set1_ps(LOG1P_COEFFICIENTS[0]);
    for &c in &LOG1P_COEFFICIENTS[1..] {
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(c));
    }
    _mm256_fmadd_ps(
        t,
        p,
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(exponent),
            _mm256_set1_ps(std::f32::consts::LN_2),
        ),
    )
}

/// Each 128-bit half accumulates an independent block's exact integer moments.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn block_pair_variances_avx2(yuv: &Yuv, row: usize, col: usize) -> [f32; 2] {
    let width = yuv.width as usize;
    let rows = (yuv.height as usize - row).min(8);
    let shift = _mm_cvtsi32_si128(i32::from(yuv.bit_depth.bits().saturating_sub(8)));
    let mut sum = _mm256_setzero_si256();
    let mut squares = _mm256_setzero_si256();
    for r in row..row + rows {
        // The caller ensures col + 16 <= width and validates the whole plane.
        let samples = _mm256_srl_epi16(
            unsafe { _mm256_loadu_si256(yuv.y.as_ptr().add(r * width + col).cast()) },
            shift,
        );
        sum = _mm256_add_epi16(sum, samples);
        squares = _mm256_add_epi32(squares, _mm256_madd_epi16(samples, samples));
    }
    let sum = _mm256_madd_epi16(sum, _mm256_set1_epi16(1));
    let moments = _mm256_hadd_epi32(sum, squares);
    let moments = _mm256_hadd_epi32(moments, moments);
    let lo = _mm256_castsi256_si128(moments);
    let hi = _mm256_extracti128_si256::<1>(moments);
    [
        variance_from_moments(
            _mm_cvtsi128_si32(lo) as u32,
            _mm_extract_epi32::<1>(lo) as u32,
            rows * 8,
        ),
        variance_from_moments(
            _mm_cvtsi128_si32(hi) as u32,
            _mm_extract_epi32::<1>(hi) as u32,
            rows * 8,
        ),
    ]
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn block_log_variances_avx2_fma(yuv: &Yuv, out: &mut [f32]) {
    let (width, height) = (yuv.width as usize, yuv.height as usize);
    let bw = width.div_ceil(8);
    assert!(yuv.y.len() >= width * height);
    assert_eq!(out.len(), bw * height.div_ceil(8));
    for row in (0..height).step_by(8) {
        let dst = &mut out[row / 8 * bw..(row / 8 + 1) * bw];
        let mut block = 0;
        while block + 8 <= width / 8 {
            let mut variances = [0.0; 8];
            for (i, pair) in variances.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                *pair = block_pair_variances_avx2(yuv, row, (block + i * 2) * 8);
            }
            unsafe {
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(block),
                    log1p_f32x8(_mm256_loadu_ps(variances.as_ptr())),
                );
            }
            block += 8;
        }
        while block + 2 <= width / 8 {
            let pair = block_pair_variances_avx2(yuv, row, block * 8);
            dst[block] = log1p(pair[0]);
            dst[block + 1] = log1p(pair[1]);
            block += 2;
        }
        while block < bw {
            dst[block] = log1p(block_variance_scalar(yuv, row, block * 8));
            block += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn ctu_qg_log_variances_avx2_fma(
    blocks: &[f32],
    bw: usize,
    bh: usize,
    ctu_row: usize,
    ctu_col: usize,
) -> [f32; QGS_PER_CTU] {
    assert!(blocks.len() >= bw * bh);
    let mut out = [f32::NAN; QGS_PER_CTU];
    for qr in 0..64 / QG_SIZE {
        for qc in 0..64 / QG_SIZE {
            let (row, col) = (
                ctu_row * 8 + qr * (QG_SIZE / 8),
                ctu_col * 8 + qc * (QG_SIZE / 8),
            );
            if row >= bh || col >= bw {
                continue;
            }
            let (rows, cols) = ((bh - row).min(QG_SIZE / 8), (bw - col).min(QG_SIZE / 8));
            if rows != 4 || cols != 4 {
                out[qg_z(qr, qc)] = qg_log_variance_scalar(blocks, bw, row, col, rows, cols);
                continue;
            }
            // Four values from each row, without reading into the next row.
            let values: [__m256; 2] = std::array::from_fn(|i| unsafe {
                let lo = _mm_loadu_ps(blocks.as_ptr().add((row + i * 2) * bw + col));
                let hi = _mm_loadu_ps(blocks.as_ptr().add((row + i * 2 + 1) * bw + col));
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi)
            });
            let max = _mm256_max_ps(values[0], values[1]);
            let max = _mm_max_ps(_mm256_castps256_ps128(max), _mm256_extractf128_ps::<1>(max));
            let max = _mm_max_ps(max, _mm_movehl_ps(max, max));
            let max = _mm_cvtss_f32(_mm_max_ss(max, _mm_shuffle_ps::<0x55>(max, max)));
            let mut sum = _mm256_setzero_ps();
            for v in values {
                sum = _mm256_add_ps(
                    sum,
                    exp_f32x8(_mm256_mul_ps(
                        _mm256_sub_ps(v, _mm256_set1_ps(max)),
                        _mm256_set1_ps(1.0 / QG_SOFTMAX_TEMPERATURE),
                    )),
                );
            }
            let sum = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
            out[qg_z(qr, qc)] = max + QG_SOFTMAX_TEMPERATURE * (sum * (1.0 / 16.0)).ln();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_log1p_matches_libm() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for base in 0..=2048 {
            let input = std::array::from_fn::<_, 8, _>(|i| (base * 8 + i) as f32);
            let mut output = [0.0; 8];
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr(),
                    log1p_f32x8(_mm256_loadu_ps(input.as_ptr())),
                )
            };
            for (x, actual) in input.into_iter().zip(output) {
                assert!(
                    (actual - x.ln_1p()).abs() <= 2.0e-6,
                    "x={x}, actual={actual}"
                );
            }
        }
    }
}
