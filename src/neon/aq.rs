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

use super::math::exp_f32x4;
use crate::Yuv;
use crate::aq::{
    LOG1P_COEFFICIENTS, QG_SIZE, QG_SOFTMAX_TEMPERATURE, QGS_PER_CTU, block_variance_scalar, log1p,
    qg_log_variance_scalar, qg_z, variance_from_moments,
};
use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn log1p_f32x4(x: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_u32_f32(vaddq_f32(x, vdupq_n_f32(1.0)));
    let exponent = vsubq_s32(
        vreinterpretq_s32_u32(vshrq_n_u32::<23>(bits)),
        vdupq_n_s32(127),
    );
    let mantissa = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    ));
    let high = vcgtq_f32(mantissa, vdupq_n_f32(std::f32::consts::SQRT_2));
    let mantissa = vbslq_f32(high, vmulq_n_f32(mantissa, 0.5), mantissa);
    let exponent = vsubq_s32(exponent, vreinterpretq_s32_u32(high));
    let t = vsubq_f32(mantissa, vdupq_n_f32(1.0));
    let mut p = vdupq_n_f32(LOG1P_COEFFICIENTS[0]);
    for &c in &LOG1P_COEFFICIENTS[1..] {
        p = vfmaq_f32(vdupq_n_f32(c), p, t);
    }
    vfmaq_f32(
        vmulq_n_f32(vcvtq_f32_s32(exponent), std::f32::consts::LN_2),
        t,
        p,
    )
}

/// Eight columns, including a clipped final band of rows. Normalized samples
/// are at most 255, so both the u16 column sums and u32 squares are exact.
#[inline]
#[target_feature(enable = "neon")]
fn block_variance_neon(yuv: &Yuv, row: usize, col: usize) -> f32 {
    let width = yuv.width as usize;
    let rows = (yuv.height as usize - row).min(8);
    let shift = vdupq_n_s16(-i16::from(yuv.bit_depth.bits().saturating_sub(8)));
    let mut sum = vdupq_n_u16(0);
    let mut squares = vdupq_n_u32(0);
    for r in row..row + rows {
        // The caller ensures col + 8 <= width and validates the whole plane.
        let samples = vshlq_u16(
            unsafe { vld1q_u16(yuv.y.as_ptr().add(r * width + col)) },
            shift,
        );
        sum = vaddq_u16(sum, samples);
        let low = vget_low_u16(samples);
        squares = vmlal_u16(squares, low, low);
        squares = vmlal_high_u16(squares, samples, samples);
    }
    variance_from_moments(vaddlvq_u16(sum), vaddvq_u32(squares), rows * 8)
}

#[target_feature(enable = "neon")]
pub(crate) fn block_log_variances_neon(yuv: &Yuv, out: &mut [f32]) {
    let (width, height) = (yuv.width as usize, yuv.height as usize);
    let bw = width.div_ceil(8);
    assert!(yuv.y.len() >= width * height);
    assert_eq!(out.len(), bw * height.div_ceil(8));
    for row in (0..height).step_by(8) {
        let dst = &mut out[row / 8 * bw..(row / 8 + 1) * bw];
        let mut block = 0;
        while block + 4 <= width / 8 {
            let variances: [f32; 4] =
                std::array::from_fn(|i| block_variance_neon(yuv, row, (block + i) * 8));
            unsafe {
                vst1q_f32(
                    dst.as_mut_ptr().add(block),
                    log1p_f32x4(vld1q_f32(variances.as_ptr())),
                );
            }
            block += 4;
        }
        while block < width / 8 {
            dst[block] = log1p(block_variance_neon(yuv, row, block * 8));
            block += 1;
        }
        if block < bw {
            dst[block] = log1p(block_variance_scalar(yuv, row, block * 8));
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn ctu_qg_log_variances_neon(
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
            let values: [float32x4_t; 4] = std::array::from_fn(|i| unsafe {
                vld1q_f32(blocks.as_ptr().add((row + i) * bw + col))
            });
            let max = vmaxvq_f32(vmaxq_f32(
                vmaxq_f32(values[0], values[1]),
                vmaxq_f32(values[2], values[3]),
            ));
            let mut sum = vdupq_n_f32(0.0);
            for v in values {
                sum = vaddq_f32(
                    sum,
                    exp_f32x4(vmulq_n_f32(
                        vsubq_f32(v, vdupq_n_f32(max)),
                        1.0 / QG_SOFTMAX_TEMPERATURE,
                    )),
                );
            }
            out[qg_z(qr, qc)] =
                max + QG_SOFTMAX_TEMPERATURE * (vaddvq_f32(sum) * (1.0 / 16.0)).ln();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_log1p_matches_scalar() {
        for base in 0..=4096 {
            let input = std::array::from_fn::<_, 4, _>(|i| (base * 4 + i) as f32);
            let mut output = [0.0; 4];
            unsafe { vst1q_f32(output.as_mut_ptr(), log1p_f32x4(vld1q_f32(input.as_ptr()))) };
            for (x, actual) in input.into_iter().zip(output) {
                assert!(
                    (actual - x.ln_1p()).abs() <= 2.0e-6,
                    "x={x}, actual={actual}"
                );
                assert_eq!(actual, log1p(x));
            }
        }
    }
}
