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
use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn log1p_neon(x: float32x4_t) -> float32x4_t {
    let y = vaddq_f32(x, vdupq_n_f32(1.0));
    let bits = vreinterpretq_u32_f32(y);
    let mut exponent = vsubq_s32(
        vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32::<23>(bits), vdupq_n_u32(0xff))),
        vdupq_n_s32(127),
    );
    let mut mantissa = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    ));
    let reduce = vcgtq_f32(mantissa, vdupq_n_f32(std::f32::consts::SQRT_2));
    mantissa = vbslq_f32(reduce, vmulq_n_f32(mantissa, 0.5), mantissa);
    exponent = vaddq_s32(
        exponent,
        vreinterpretq_s32_u32(vandq_u32(reduce, vdupq_n_u32(1))),
    );

    let t = vsubq_f32(mantissa, vdupq_n_f32(1.0));
    let mut polynomial = vdupq_n_f32(-0.100_935_325_026_512_15);
    polynomial = vfmaq_f32(vdupq_n_f32(0.164_151_370_525_360_1), polynomial, t);
    polynomial = vfmaq_f32(vdupq_n_f32(-0.173_346_474_766_731_26), polynomial, t);
    polynomial = vfmaq_f32(vdupq_n_f32(0.198_739_007_115_364_07), polynomial, t);
    polynomial = vfmaq_f32(vdupq_n_f32(-0.249_593_034_386_634_83), polynomial, t);
    polynomial = vfmaq_f32(vdupq_n_f32(0.333_361_357_450_485_23), polynomial, t);
    polynomial = vfmaq_f32(vdupq_n_f32(-0.500_006_675_720_214_8), polynomial, t);
    polynomial = vfmaq_f32(vdupq_n_f32(0.999_999_821_186_065_7), polynomial, t);
    let exponent = vmulq_n_f32(vcvtq_f32_s32(exponent), std::f32::consts::LN_2);
    vfmaq_f32(exponent, t, polynomial)
}

#[target_feature(enable = "neon")]
pub(crate) fn log1p_slice_neon(values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();
    for values in chunks {
        let x = unsafe { vld1q_f32(values.as_ptr()) };
        let result = log1p_neon(x);
        unsafe { vst1q_f32(values.as_mut_ptr(), result) };
    }
    crate::aq::log1p_slice_scalar(remainder);
}

#[inline]
#[target_feature(enable = "neon")]
fn shifted_u16x8(values: &[u16; 8], shift: u8) -> uint16x8_t {
    let values = unsafe { vld1q_u16(values.as_ptr()) };
    vshlq_u16(values, vdupq_n_s16(-i16::from(shift)))
}

#[inline]
#[target_feature(enable = "neon")]
fn sum_and_square(values: uint16x8_t) -> (f32, f32) {
    let low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(values)));
    let high = vcvtq_f32_u32(vmovl_high_u16(values));
    (
        vaddvq_f32(vaddq_f32(low, high)),
        vaddvq_f32(vaddq_f32(vmulq_f32(low, low), vmulq_f32(high, high))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
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
    let mut energy = vdupq_n_f32(0.);
    for (top, left, middle, right, bottom) in [
        (
            vmovl_u16(vget_low_u16(top)),
            vmovl_u16(vget_low_u16(left)),
            vmovl_u16(vget_low_u16(middle)),
            vmovl_u16(vget_low_u16(right)),
            vmovl_u16(vget_low_u16(bottom)),
        ),
        (
            vmovl_high_u16(top),
            vmovl_high_u16(left),
            vmovl_high_u16(middle),
            vmovl_high_u16(right),
            vmovl_high_u16(bottom),
        ),
    ] {
        let laplacian = vsubq_s32(
            vsubq_s32(
                vshlq_n_s32::<2>(vreinterpretq_s32_u32(middle)),
                vreinterpretq_s32_u32(left),
            ),
            vaddq_s32(
                vreinterpretq_s32_u32(right),
                vaddq_s32(vreinterpretq_s32_u32(top), vreinterpretq_s32_u32(bottom)),
            ),
        );
        energy = vaddq_f32(
            energy,
            vcvtq_f32_u32(vreinterpretq_u32_s32(vabsq_s32(laplacian))),
        );
    }
    vaddvq_f32(energy)
}

#[inline]
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
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
        let (top, top_remainder) = above[1..stride - 1].as_chunks::<4>();
        let left = center[..stride - 2].as_chunks::<4>().0;
        let middle = center[1..stride - 1].as_chunks::<4>().0;
        let right = center[2..stride].as_chunks::<4>().0;
        let bottom = below[1..stride - 1].as_chunks::<4>().0;
        for ((((top, left), middle), right), bottom) in
            top.iter().zip(left).zip(middle).zip(right).zip(bottom)
        {
            let top = unsafe { vld1q_f32(top.as_ptr()) };
            let left = unsafe { vld1q_f32(left.as_ptr()) };
            let middle = unsafe { vld1q_f32(middle.as_ptr()) };
            let right = unsafe { vld1q_f32(right.as_ptr()) };
            let bottom = unsafe { vld1q_f32(bottom.as_ptr()) };
            let laplacian = vsubq_f32(
                vsubq_f32(vmulq_n_f32(middle, 4.0), left),
                vaddq_f32(right, vaddq_f32(top, bottom)),
            );
            energy += vaddvq_f32(vabsq_f32(laplacian));
        }
        let vector_len = top.len() * 4;
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

#[target_feature(enable = "neon")]
pub(crate) fn ctu_activity_neon(
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
    log1p_slice_neon(&mut variances[..blocks]);

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
            let rows = vaddq_u16(shifted_u16x8(top, shift), shifted_u16x8(bottom, shift));
            let averaged = vmulq_n_f32(vcvtq_f32_u32(vpaddlq_u16(rows)), 0.25);
            unsafe { vst1q_f32(coarse[coarse_len..].as_mut_ptr(), averaged) };
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

    fn test_yuv(width: usize, height: usize, bit_depth: BitDepth) -> Yuv {
        let shift = bit_depth.bits() - 8;
        let y = (0..width * height)
            .map(|i| (((i * 73 + i / width * 29 + 17) & 255) as u16) << shift)
            .collect();
        Yuv {
            y,
            cb: Vec::new(),
            cr: Vec::new(),
            width: width as u32,
            height: height as u32,
            display_w: width as u32,
            display_h: height as u32,
            chroma: ChromaFormat::Monochrome,
            bit_depth,
        }
    }

    #[test]
    fn ctu_activity_neon_matches_scalar() {
        for bit_depth in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            for (width, height) in [
                (1usize, 1usize),
                (2, 7),
                (7, 2),
                (9, 11),
                (64, 64),
                (79, 67),
                (129, 65),
            ] {
                let yuv = test_yuv(width, height, bit_depth);
                for row in 0..height.div_ceil(64) {
                    for col in 0..width.div_ceil(64) {
                        let actual = unsafe { ctu_activity_neon(&yuv, row, col, 6) };
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
