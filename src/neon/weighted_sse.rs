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
use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn weighted_sse_neon(h: usize, w: usize, err: impl Fn(usize, usize) -> i32) -> f32 {
    debug_assert!(h.is_multiple_of(4) && w.is_multiple_of(4));
    let basis: [float32x4_t; 4] = std::array::from_fn(|j| {
        let column = std::array::from_fn::<_, 4, _>(|k| DCT4[k][j]);
        unsafe { vld1q_f32(column.as_ptr()) }
    });
    let weights: [float32x4_t; 4] =
        std::array::from_fn(|k| unsafe { vld1q_f32(DISTORTION_WEIGHTS[k].as_ptr()) });

    let mut total = [vdupq_n_f32(0.0); 4];
    for r0 in (0..h).step_by(4) {
        for c0 in (0..w).step_by(4) {
            let rows: [float32x4_t; 4] = std::array::from_fn(|i| {
                let row = std::array::from_fn::<_, 4, _>(|j| err(r0 + i, c0 + j));
                vcvtq_f32_s32(unsafe { vld1q_s32(row.as_ptr()) })
            });
            for k in 0..4 {
                // Columns: the lanes hold four independent column transforms.
                let mut t = vmulq_n_f32(rows[0], DCT4[k][0]);
                for (i, row) in rows.iter().enumerate().skip(1) {
                    t = vfmaq_n_f32(t, *row, DCT4[k][i]);
                }
                // Rows: lane l now holds coefficient (k, l).
                let mut c = vmulq_laneq_f32::<0>(basis[0], t);
                c = vfmaq_laneq_f32::<1>(c, basis[1], t);
                c = vfmaq_laneq_f32::<2>(c, basis[2], t);
                c = vfmaq_laneq_f32::<3>(c, basis[3], t);
                // Independent frequency-row accumulators avoid a serial chain
                // of scalar additions through every coefficient and tile.
                total[k] = vfmaq_f32(total[k], vmulq_f32(weights[k], c), c);
            }
        }
    }
    vaddvq_f32(vaddq_f32(
        vaddq_f32(total[0], total[1]),
        vaddq_f32(total[2], total[3]),
    ))
}
