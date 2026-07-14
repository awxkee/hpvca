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

use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn satd_avx2(orig: &[u16], pred: &[u16], n: usize) -> u32 {
    assert!(matches!(n, 4 | 8 | 16 | 32));
    assert!(orig.len() >= n * n && pred.len() >= n * n);

    if n == 4 {
        return unsafe { satd_4x4(orig.as_ptr(), pred.as_ptr(), 4) };
    }

    let mut total = 0u32;
    for by in (0..n).step_by(4) {
        for bx in (0..n).step_by(8) {
            let offset = by * n + bx;
            // SAFETY: the slice checks above and 4-pixel tiling guarantee that
            // every 128-bit row load is in bounds. The resolver checks AVX2.
            total += unsafe { satd_4x8(orig.as_ptr().add(offset), pred.as_ptr().add(offset), n) };
        }
    }
    total
}

#[inline]
#[target_feature(enable = "avx2")]
fn hadamard4x2(row: __m256i) -> __m256i {
    let opposite = _mm256_shuffle_epi32(row, 0b01_00_11_10);
    let pair_add = _mm256_add_epi32(row, opposite);
    let pair_sub = _mm256_sub_epi32(row, opposite);
    let butterfly = _mm256_unpacklo_epi64(pair_add, pair_sub);
    let adjacent = _mm256_shuffle_epi32(butterfly, 0b10_11_00_01);
    let sum = _mm256_add_epi32(butterfly, adjacent);
    let difference = _mm256_sub_epi32(butterfly, adjacent);
    let difference = _mm256_shuffle_epi32(difference, 0b10_10_00_00);
    _mm256_blend_epi32(sum, difference, 0b1010_1010)
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose4x2(rows: [__m256i; 4]) -> [__m256i; 4] {
    let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
    let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
    let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
    let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
    [
        _mm256_unpacklo_epi64(t0, t2),
        _mm256_unpackhi_epi64(t0, t2),
        _mm256_unpacklo_epi64(t1, t3),
        _mm256_unpackhi_epi64(t1, t3),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn hadamard4(row: __m128i) -> __m128i {
    // [x0+x2, x1+x3, x0-x2, x1-x3]
    let opposite = _mm_shuffle_epi32(row, 0b01_00_11_10);
    let pair_add = _mm_add_epi32(row, opposite);
    let pair_sub = _mm_sub_epi32(row, opposite);
    let butterfly = _mm_unpacklo_epi64(pair_add, pair_sub);

    // [a0+a1, a0-a1, a2+a3, a2-a3]
    let adjacent = _mm_shuffle_epi32(butterfly, 0b10_11_00_01);
    let sum = _mm_add_epi32(butterfly, adjacent);
    let difference = _mm_sub_epi32(butterfly, adjacent);
    let difference = _mm_shuffle_epi32(difference, 0b10_10_00_00);
    _mm_blend_epi32(sum, difference, 0b1010)
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose4(rows: [__m128i; 4]) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(rows[0], rows[1]);
    let t1 = _mm_unpackhi_epi32(rows[0], rows[1]);
    let t2 = _mm_unpacklo_epi32(rows[2], rows[3]);
    let t3 = _mm_unpackhi_epi32(rows[2], rows[3]);
    [
        _mm_unpacklo_epi64(t0, t2),
        _mm_unpackhi_epi64(t0, t2),
        _mm_unpacklo_epi64(t1, t3),
        _mm_unpackhi_epi64(t1, t3),
    ]
}

#[target_feature(enable = "avx2")]
unsafe fn satd_4x4(orig: *const u16, pred: *const u16, stride: usize) -> u32 {
    let mut rows = [_mm_setzero_si128(); 4];
    for (row, dst) in rows.iter_mut().enumerate() {
        let o16 = unsafe { _mm_loadl_epi64(orig.add(row * stride).cast()) };
        let p16 = unsafe { _mm_loadl_epi64(pred.add(row * stride).cast()) };
        let difference = _mm_sub_epi32(_mm_cvtepu16_epi32(o16), _mm_cvtepu16_epi32(p16));
        *dst = hadamard4(difference);
    }

    // The second separable transform operates on columns. Transposing turns
    // those columns into SIMD rows and avoids scalar lane extraction.
    let rows = transpose4(rows);
    let mut coefficients = _mm_setzero_si128();
    for row in rows {
        coefficients = _mm_add_epi32(coefficients, _mm_abs_epi32(hadamard4(row)));
    }
    let pair = _mm_hadd_epi32(coefficients, coefficients);
    let sum = _mm_cvtsi128_si32(_mm_hadd_epi32(pair, pair)) as u32;
    (sum + 1) >> 1
}

#[target_feature(enable = "avx2")]
unsafe fn satd_4x8(orig: *const u16, pred: *const u16, stride: usize) -> u32 {
    let mut rows = [_mm256_setzero_si256(); 4];
    for (row, dst) in rows.iter_mut().enumerate() {
        let o16 = unsafe { _mm_loadu_si128(orig.add(row * stride).cast()) };
        let p16 = unsafe { _mm_loadu_si128(pred.add(row * stride).cast()) };
        let difference = _mm256_sub_epi32(_mm256_cvtepu16_epi32(o16), _mm256_cvtepu16_epi32(p16));
        *dst = hadamard4x2(difference);
    }

    let mut coefficients = _mm256_setzero_si256();
    for row in transpose4x2(rows) {
        coefficients = _mm256_add_epi32(coefficients, _mm256_abs_epi32(hadamard4x2(row)));
    }
    let pairs = _mm256_hadd_epi32(coefficients, coefficients);
    let sums = _mm256_hadd_epi32(pairs, pairs);
    let first = _mm_cvtsi128_si32(_mm256_castsi256_si128(sums)) as u32;
    let second = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(sums)) as u32;
    ((first + 1) >> 1) + ((second + 1) >> 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satd_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut orig = [0u16; 1024];
        let mut pred = [0u16; 1024];
        for seed in 0..32u32 {
            let mut state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            for (index, (orig, pred)) in orig.iter_mut().zip(&mut pred).enumerate() {
                state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
                *orig = if seed == 0 {
                    4095
                } else {
                    ((state >> 16) & 4095) as u16
                };
                state ^= (index as u32).wrapping_mul(277_803_737);
                *pred = if seed == 0 {
                    0
                } else {
                    ((state >> 12) & 4095) as u16
                };
            }
            for n in [4, 8, 16, 32] {
                let scalar = crate::cost::satd_scalar(&orig[..n * n], &pred[..n * n], n);
                let simd = unsafe { satd_avx2(&orig[..n * n], &pred[..n * n], n) };
                assert_eq!(simd, scalar, "SATD mismatch for seed={seed}, {n}x{n}");
            }
        }
    }
}
