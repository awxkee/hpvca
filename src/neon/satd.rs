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

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn satd_neon(orig: &[u16], pred: &[u16], n: usize) -> u32 {
    assert!(matches!(n, 4 | 8 | 16 | 32));
    assert!(orig.len() >= n * n && pred.len() >= n * n);

    let mut total = 0u32;
    for by in (0..n).step_by(4) {
        for bx in (0..n).step_by(4) {
            let offset = by * n + bx;
            // SAFETY: the slice checks above and 4-pixel tiling guarantee that
            // every four-sample row load is in bounds. NEON is mandatory on AArch64.
            total += unsafe { satd_4x4(orig.as_ptr().add(offset), pred.as_ptr().add(offset), n) };
        }
    }
    total
}

#[inline]
#[target_feature(enable = "neon")]
fn hadamard4(row: int32x4_t) -> int32x4_t {
    let lo = vget_low_s32(row);
    let hi = vget_high_s32(row);
    let pair_add = vadd_s32(lo, hi);
    let pair_sub = vsub_s32(lo, hi);

    let add_sum = vpadd_s32(pair_add, pair_add);
    let add_difference = vsub_s32(pair_add, vrev64_s32(pair_add));
    let sub_sum = vpadd_s32(pair_sub, pair_sub);
    let sub_difference = vsub_s32(pair_sub, vrev64_s32(pair_sub));
    vcombine_s32(
        vzip1_s32(add_sum, add_difference),
        vzip1_s32(sub_sum, sub_difference),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose4(rows: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let t0 = vtrnq_s32(rows[0], rows[1]);
    let t1 = vtrnq_s32(rows[2], rows[3]);
    [
        vreinterpretq_s32_s64(vtrn1q_s64(
            vreinterpretq_s64_s32(t0.0),
            vreinterpretq_s64_s32(t1.0),
        )),
        vreinterpretq_s32_s64(vtrn1q_s64(
            vreinterpretq_s64_s32(t0.1),
            vreinterpretq_s64_s32(t1.1),
        )),
        vreinterpretq_s32_s64(vtrn2q_s64(
            vreinterpretq_s64_s32(t0.0),
            vreinterpretq_s64_s32(t1.0),
        )),
        vreinterpretq_s32_s64(vtrn2q_s64(
            vreinterpretq_s64_s32(t0.1),
            vreinterpretq_s64_s32(t1.1),
        )),
    ]
}

#[target_feature(enable = "neon")]
unsafe fn satd_4x4(orig: *const u16, pred: *const u16, stride: usize) -> u32 {
    let mut rows = [vdupq_n_s32(0); 4];
    for (row, dst) in rows.iter_mut().enumerate() {
        let o16 = unsafe { vld1_u16(orig.add(row * stride)) };
        let p16 = unsafe { vld1_u16(pred.add(row * stride)) };
        let difference = vreinterpretq_s32_u32(vsubl_u16(o16, p16));
        *dst = hadamard4(difference);
    }

    // The second separable transform operates on columns. Transposing turns
    // those columns into SIMD rows and avoids scalar lane extraction.
    let rows = transpose4(rows);
    let mut coefficients = vdupq_n_u32(0);
    for row in rows {
        coefficients = vaddq_u32(
            coefficients,
            vreinterpretq_u32_s32(vabsq_s32(hadamard4(row))),
        );
    }
    (vaddvq_u32(coefficients) + 1) >> 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satd_neon_matches_scalar() {
        let mut orig = [0u16; 1024];
        let mut pred = [0u16; 1024];
        for i in 0..orig.len() {
            orig[i] = ((i * 251 + i / 5 * 17 + 4095) & 4095) as u16;
            pred[i] = ((i * 43 + i / 11 * 197) & 4095) as u16;
        }
        for n in [4, 8, 16, 32] {
            let scalar = unsafe { crate::cost::satd_scalar(&orig[..n * n], &pred[..n * n], n) };
            let simd = unsafe { satd_neon(&orig[..n * n], &pred[..n * n], n) };
            assert_eq!(simd, scalar, "SATD mismatch for {n}x{n}");
        }
    }
}
