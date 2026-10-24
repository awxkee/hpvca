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

use crate::cost::{DCT4, DISTORTION_WEIGHTS};
use core::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
pub(crate) fn weighted_sse_avx2_fma(h: usize, w: usize, err: impl Fn(usize, usize) -> i32) -> f32 {
    debug_assert!(h.is_multiple_of(4) && w.is_multiple_of(4));
    let basis: [__m256; 4] = std::array::from_fn(|j| {
        let column = std::array::from_fn::<_, 8, _>(|k| DCT4[k % 4][j]);
        unsafe { _mm256_loadu_ps(column.as_ptr()) }
    });
    let weights: [__m256; 4] = std::array::from_fn(|k| {
        let row = unsafe { _mm_loadu_ps(DISTORTION_WEIGHTS[k].as_ptr()) };
        _mm256_broadcast_ps(&row)
    });

    let mut total = [_mm256_setzero_ps(); 4];
    for r0 in (0..h).step_by(4) {
        // Each 128-bit half processes an independent 4×4 tile. The
        // final tile of widths 4, 12, ... uses only the low half.
        for c0 in (0..w).step_by(8) {
            let columns = (w - c0).min(8);
            let rows: [__m256; 4] = std::array::from_fn(|i| {
                let row = std::array::from_fn::<_, 8, _>(|j| {
                    if j < 4 || columns == 8 {
                        err(r0 + i, c0 + j)
                    } else {
                        0
                    }
                });
                _mm256_cvtepi32_ps(unsafe { _mm256_loadu_si256(row.as_ptr().cast()) })
            });
            for k in 0..4 {
                let mut t = _mm256_mul_ps(rows[0], _mm256_set1_ps(DCT4[k][0]));
                for (i, row) in rows.iter().enumerate().skip(1) {
                    t = _mm256_fmadd_ps(*row, _mm256_set1_ps(DCT4[k][i]), t);
                }
                let mut c = _mm256_mul_ps(basis[0], _mm256_shuffle_ps::<0x00>(t, t));
                c = _mm256_fmadd_ps(basis[1], _mm256_shuffle_ps::<0x55>(t, t), c);
                c = _mm256_fmadd_ps(basis[2], _mm256_shuffle_ps::<0xaa>(t, t), c);
                c = _mm256_fmadd_ps(basis[3], _mm256_shuffle_ps::<0xff>(t, t), c);
                total[k] = _mm256_fmadd_ps(_mm256_mul_ps(weights[k], c), c, total[k]);
            }
        }
    }
    let sum = _mm256_add_ps(
        _mm256_add_ps(total[0], total[1]),
        _mm256_add_ps(total[2], total[3]),
    );
    let sum = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
    let shuf = _mm_movehdup_ps(sum);
    let pairs = _mm_add_ps(sum, shuf);
    let high = _mm_movehl_ps(shuf, pairs);
    _mm_cvtss_f32(_mm_add_ss(pairs, high))
}
