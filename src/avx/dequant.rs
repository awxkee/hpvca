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

use crate::hevc_transform::{DEQUANT_SCALE, MAX_TB};
use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn dequantize_avx2(
    level: &[i16],
    n: usize,
    qp: u8,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    let len = n * n;
    let bd_shift = bit_depth as i32 + n.trailing_zeros() as i32 - 5;
    let net_shift = qp as i32 / 6 + 4 - bd_shift;
    debug_assert!(qp as i32 <= 51 + 6 * (bit_depth as i32 - 8));

    let scale = _mm256_set1_epi32(DEQUANT_SCALE[(qp % 6) as usize] as i32);
    let min = _mm256_set1_epi32(-32768);
    let max = _mm256_set1_epi32(32767);
    if net_shift < 0 {
        dequantize_right(level, out, len, scale, -net_shift, min, max);
    } else {
        dequantize_left(level, out, len, scale, net_shift, min, max);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequantize_left(
    level: &[i16],
    out: &mut [i32; MAX_TB],
    len: usize,
    scale: __m256i,
    shift: i32,
    min: __m256i,
    max: __m256i,
) {
    let shift = _mm256_set1_epi32(shift);
    let (level, level_remainder) = level[..len].as_chunks::<16>();
    let (out, out_remainder) = out[..len].as_chunks_mut::<16>();
    debug_assert!(level_remainder.is_empty() && out_remainder.is_empty());
    for (level, out) in level.iter().zip(out) {
        let packed = unsafe { _mm256_loadu_si256(level.as_ptr().cast()) };
        let low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(packed));
        let high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(packed));
        let low = clamp(
            _mm256_sllv_epi32(_mm256_mullo_epi32(low, scale), shift),
            min,
            max,
        );
        let high = clamp(
            _mm256_sllv_epi32(_mm256_mullo_epi32(high, scale), shift),
            min,
            max,
        );
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().cast(), low);
            _mm256_storeu_si256(out[8..].as_mut_ptr().cast(), high);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequantize_right(
    level: &[i16],
    out: &mut [i32; MAX_TB],
    len: usize,
    scale: __m256i,
    shift: i32,
    min: __m256i,
    max: __m256i,
) {
    let add = _mm256_set1_epi32(1 << (shift - 1));
    let shift = _mm256_set1_epi32(shift);
    let (level, level_remainder) = level[..len].as_chunks::<16>();
    let (out, out_remainder) = out[..len].as_chunks_mut::<16>();
    debug_assert!(level_remainder.is_empty() && out_remainder.is_empty());
    for (level, out) in level.iter().zip(out) {
        let packed = unsafe { _mm256_loadu_si256(level.as_ptr().cast()) };
        let low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(packed));
        let high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(packed));
        let low = clamp(
            _mm256_srav_epi32(_mm256_add_epi32(_mm256_mullo_epi32(low, scale), add), shift),
            min,
            max,
        );
        let high = clamp(
            _mm256_srav_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(high, scale), add),
                shift,
            ),
            min,
            max,
        );
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().cast(), low);
            _mm256_storeu_si256(out[8..].as_mut_ptr().cast(), high);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn clamp(value: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(value, min), max)
}

/// [`dequantize_avx2`] with a scaling list: the per-coefficient factor `m`
/// (1..=255) replaces the flat 16. The product `level · scale · m` stays below
/// 2³¹; before a left shift it is clamped to ±2²⁰ so the shift cannot wrap,
/// which leaves the final ±32767 clamp unchanged.
#[target_feature(enable = "avx2")]
pub(crate) fn dequantize_scaled_avx2(
    level: &[i16],
    n: usize,
    qp: u8,
    bit_depth: u8,
    m: &[i32],
    out: &mut [i32; MAX_TB],
) {
    let len = n * n;
    let bd_shift = bit_depth as i32 + n.trailing_zeros() as i32 - 5;
    let net_shift = qp as i32 / 6 - bd_shift;
    let scale = _mm256_set1_epi32(DEQUANT_SCALE[(qp % 6) as usize] as i32);
    let min = _mm256_set1_epi32(-32768);
    let max = _mm256_set1_epi32(32767);
    let pre_min = _mm256_set1_epi32(-(1 << 20));
    let pre_max = _mm256_set1_epi32(1 << 20);
    let (level, lr) = level[..len].as_chunks::<16>();
    let (m, mr) = m[..len].as_chunks::<16>();
    let (out, or) = out[..len].as_chunks_mut::<16>();
    debug_assert!(lr.is_empty() && mr.is_empty() && or.is_empty());
    let left = net_shift >= 0;
    let shift = _mm256_set1_epi32(net_shift.abs());
    let add = _mm256_set1_epi32(if left { 0 } else { 1 << (net_shift.abs() - 1) });
    for ((level, m), out) in level.iter().zip(m).zip(out) {
        let packed = unsafe { _mm256_loadu_si256(level.as_ptr().cast()) };
        let (m_lo, m_hi) = unsafe {
            (
                _mm256_loadu_si256(m.as_ptr().cast()),
                _mm256_loadu_si256(m[8..].as_ptr().cast()),
            )
        };
        let low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(packed));
        let high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(packed));
        let low = _mm256_mullo_epi32(_mm256_mullo_epi32(low, scale), m_lo);
        let high = _mm256_mullo_epi32(_mm256_mullo_epi32(high, scale), m_hi);
        let (low, high) = if left {
            (
                _mm256_sllv_epi32(clamp(low, pre_min, pre_max), shift),
                _mm256_sllv_epi32(clamp(high, pre_min, pre_max), shift),
            )
        } else {
            (
                _mm256_srav_epi32(_mm256_add_epi32(low, add), shift),
                _mm256_srav_epi32(_mm256_add_epi32(high, add), shift),
            )
        };
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().cast(), clamp(low, min, max));
            _mm256_storeu_si256(out[8..].as_mut_ptr().cast(), clamp(high, min, max));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for n in [4, 8, 16, 32] {
            for bit_depth in [8, 10, 12] {
                let max_qp = 51 + 6 * (bit_depth - 8);
                for qp in [0, 1, 5, 6, 22, max_qp] {
                    let mut level = [0i16; MAX_TB];
                    for (i, value) in level[..n * n].iter_mut().enumerate() {
                        *value = match i % 11 {
                            0 => i16::MIN,
                            1 => i16::MAX,
                            _ => ((i as i32 * 7919 + 104_729) as i16).wrapping_sub(16384),
                        };
                    }
                    let mut expected = [0x1234_5678; MAX_TB];
                    let mut actual = expected;
                    crate::hevc_transform::dequantize_into(&level, n, qp, bit_depth, &mut expected);
                    unsafe { dequantize_avx2(&level, n, qp, bit_depth, &mut actual) };
                    assert_eq!(actual, expected, "n={n}, bit_depth={bit_depth}, qp={qp}");
                }
            }
        }
    }

    #[test]
    fn dequantize_scaled_avx2_matches_scalar() {
        for n in [8usize, 16, 32] {
            for bit_depth in [8u8, 10, 12] {
                let max_qp = 51 + 6 * (bit_depth - 8);
                for qp in (0..=max_qp).step_by(5) {
                    let mut level = [0i16; MAX_TB];
                    let mut m = vec![16i32; n * n];
                    for i in 0..n * n {
                        level[i] = ((i as i32 * 7919 + qp as i32 * 13) % 65536 - 32768) as i16;
                        m[i] = 1 + (i as i32 * 37 + qp as i32) % 255;
                    }
                    let mut expected = [0i32; MAX_TB];
                    let mut actual = [0i32; MAX_TB];
                    crate::hevc_transform::dequantize_scaled_into(
                        &level,
                        n,
                        qp,
                        bit_depth,
                        &m,
                        &mut expected,
                    );
                    unsafe { dequantize_scaled_avx2(&level, n, qp, bit_depth, &m, &mut actual) };
                    assert_eq!(
                        &expected[..n * n],
                        &actual[..n * n],
                        "n{n} bd{bit_depth} qp{qp}"
                    );
                }
            }
        }
    }
}
