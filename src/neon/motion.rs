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
//! NEON kernels for inter prediction: fractional-sample interpolation and
//! the sum of absolute differences. They compute exactly what the scalar
//! code in [`crate::motion`] does; the tests there hold them to it.

use crate::motion::{MID_OFFSET, Rounding, Weight};
use core::arch::aarch64::*;

/// `Σ taps[i] · base[i · step ..][lane]` for eight lanes: one FIR output
/// group, horizontal with `step` 1 or vertical with `step` the row stride.
/// Samples are at most 12 bits and intermediates fit 16, so both are read as
/// `i16` and widened by the multiply.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot<const N: usize>(
    base: *const i16,
    step: usize,
    taps: &[i16; N],
) -> (int32x4_t, int32x4_t) {
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for (i, &tap) in taps.iter().enumerate() {
        // SAFETY: the caller guarantees eight readable samples at every tap.
        let v = unsafe { vld1q_s16(base.add(i * step)) };
        lo = vmlal_n_s16(lo, vget_low_s16(v), tap);
        hi = vmlal_high_n_s16(hi, v, tap);
    }
    (lo, hi)
}

/// The rounding that turns filter sums into samples (weighted) or into the
/// intermediates of bi-prediction.
struct Finish {
    intermediate: bool,
    /// Negated right shift bringing a filter sum to the 14-bit intermediate.
    pre: int32x4_t,
    weight: int32x4_t,
    round: int32x4_t,
    /// Negated `log2_denom + 14 − bit_depth`.
    post: int32x4_t,
    offset: int32x4_t,
    max: int32x4_t,
}

impl Finish {
    #[inline]
    #[target_feature(enable = "neon")]
    fn new(pre_shift: u32, bit_depth: u8, rounding: Rounding) -> Self {
        let (weight, intermediate) = match rounding {
            Rounding::Samples(weight) => (weight, false),
            Rounding::Intermediate => (Weight::UNIT, true),
        };
        let log2_wd = weight.log2_denom + 14 - u32::from(bit_depth);
        Finish {
            intermediate,
            pre: vdupq_n_s32(-(pre_shift as i32)),
            weight: vdupq_n_s32(weight.w),
            round: vdupq_n_s32(1 << (log2_wd - 1)),
            post: vdupq_n_s32(-(log2_wd as i32)),
            offset: vdupq_n_s32(weight.offset),
            max: vdupq_n_s32((1 << bit_depth) - 1),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn half(&self, sum: int32x4_t) -> uint16x4_t {
        let v = vshlq_s32(sum, self.pre);
        if self.intermediate {
            // An `i16` bit pattern per lane.
            let v = vsubq_s32(v, vdupq_n_s32(MID_OFFSET));
            return vreinterpret_u16_s16(vmovn_s32(v));
        }
        let v = vshlq_s32(vmlaq_s32(self.round, v, self.weight), self.post);
        let v = vminq_s32(
            vmaxq_s32(vaddq_s32(v, self.offset), vdupq_n_s32(0)),
            self.max,
        );
        vmovn_u32(vreinterpretq_u32_s32(v))
    }

    /// Finish eight sums and store the first `lanes` (8 or 4) at `dst`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(&self, sums: (int32x4_t, int32x4_t), dst: *mut u16, lanes: usize) {
        let lo = self.half(sums.0);
        // SAFETY: the caller guarantees `lanes` writable samples at `dst`.
        unsafe {
            if lanes >= 8 {
                vst1q_u16(dst, vcombine_u16(lo, self.half(sums.1)));
            } else {
                vst1_u16(dst, lo);
            }
        }
    }
}

/// Filter the `window` into `out` (stride `w`):
/// the SIMD form of `motion::filter_scalar`, for at least one fractional
/// direction.
///
/// # Safety
///
/// `w` is a multiple of 4, `out` holds `w · h` samples, and `window` holds
/// the block with its filter margins at row stride `stride`, plus four readable
/// samples after the block if `w` is not a multiple of 8.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn interpolate_neon<const N: usize>(
    window: &[u16],
    stride: usize,
    horizontal: Option<&[i32; N]>,
    vertical: Option<&[i32; N]>,
    w: usize,
    h: usize,
    bit_depth: u8,
    rounding: Rounding,
    out: &mut [u16],
    rows: &mut Vec<i16>,
) {
    debug_assert!(w.is_multiple_of(4) && out.len() >= w * h);
    let narrow = |f: &[i32; N]| -> [i16; N] { f.map(|tap| tap as i16) };
    let shift1 = u32::from(bit_depth) - 8;
    let win = window.as_ptr().cast::<i16>();
    let dst = out.as_mut_ptr();
    // SAFETY: every load stays within the window and its padding, every
    // store within `out` (partial at a row's 4-sample tail) or within `rows`
    // and its padding.
    unsafe {
        match (horizontal.map(narrow), vertical.map(narrow)) {
            (Some(fh), None) => {
                let finish = Finish::new(shift1, bit_depth, rounding);
                for r in 0..h {
                    for c in (0..w).step_by(8) {
                        let sums = dot(win.add(r * stride + c), 1, &fh);
                        finish.store(sums, dst.add(r * w + c), w - c);
                    }
                }
            }
            (None, Some(fv)) => {
                let finish = Finish::new(shift1, bit_depth, rounding);
                for r in 0..h {
                    for c in (0..w).step_by(8) {
                        let sums = dot(win.add(r * stride + c), stride, &fv);
                        finish.store(sums, dst.add(r * w + c), w - c);
                    }
                }
            }
            (Some(fh), Some(fv)) => {
                let wh = h + N - 1;
                // Horizontal pass into 16-bit rows. A row's 4-sample tail is
                // stored as eight; the spill is overwritten by the next row,
                // or lands in the padding.
                rows.resize(w * wh + 8, 0);
                let mid = rows.as_mut_ptr();
                let down = vdupq_n_s32(-(shift1 as i32));
                for r in 0..wh {
                    for c in (0..w).step_by(8) {
                        let (lo, hi) = dot(win.add(r * stride + c), 1, &fh);
                        let v = vcombine_s16(
                            vmovn_s32(vshlq_s32(lo, down)),
                            vmovn_s32(vshlq_s32(hi, down)),
                        );
                        vst1q_s16(mid.add(r * w + c), v);
                    }
                }
                let finish = Finish::new(6, bit_depth, rounding);
                for r in 0..h {
                    for c in (0..w).step_by(8) {
                        let sums = dot(mid.add(r * w + c).cast_const(), w, &fv);
                        finish.store(sums, dst.add(r * w + c), w - c);
                    }
                }
            }
            (None, None) => unreachable!("a whole-sample block is not filtered"),
        }
    }
}

/// The SIMD form of `motion::bi_average_scalar`.
///
/// # Safety
///
/// `a`, `b` and `out` have the same length.
#[target_feature(enable = "neon")]
pub(crate) fn bi_average_neon(a: &[u16], b: &[u16], bit_depth: u8, out: &mut [u16]) {
    debug_assert!(a.len() == out.len() && b.len() == out.len());
    let shift = 15 - i32::from(bit_depth);
    let round = vdupq_n_s32((1 << (shift - 1)) + 2 * MID_OFFSET);
    let down = vdupq_n_s32(-shift);
    let max = vdupq_n_u16((1 << bit_depth) - 1);
    let whole = out.len() & !7;
    for c in (0..whole).step_by(8) {
        // SAFETY: `c + 8` is within all three runs.
        unsafe {
            let p = vld1q_s16(a.as_ptr().add(c).cast());
            let q = vld1q_s16(b.as_ptr().add(c).cast());
            let lo = vaddq_s32(vaddl_s16(vget_low_s16(p), vget_low_s16(q)), round);
            let hi = vaddq_s32(vaddl_high_s16(p, q), round);
            let v = vcombine_u16(
                vqmovun_s32(vshlq_s32(lo, down)),
                vqmovun_s32(vshlq_s32(hi, down)),
            );
            vst1q_u16(out.as_mut_ptr().add(c), vminq_u16(v, max));
        }
    }
    crate::motion::bi_average_scalar(&a[whole..], &b[whole..], bit_depth, &mut out[whole..]);
}

/// Sum of absolute differences of two `w`×`h` blocks given as sample runs
/// with row strides `a_stride` and `b_stride`.
///
/// # Safety
///
/// `w` is a multiple of 8 and both runs hold `h` rows of `w` samples at
/// their stride.
#[target_feature(enable = "neon")]
pub(crate) fn sad_neon(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    debug_assert!(w.is_multiple_of(8) && h > 0);
    debug_assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w);
    let mut acc = [vdupq_n_u32(0); 2];
    for r in 0..h {
        // SAFETY: row `r` holds `w` samples in both runs (checked above), and
        // each load of 8 stays within them because 8 divides `w`.
        unsafe {
            let (pa, pb) = (a.as_ptr().add(r * a_stride), b.as_ptr().add(r * b_stride));
            let mut c = 0;
            while c + 16 <= w {
                let d0 = vabdq_u16(vld1q_u16(pa.add(c)), vld1q_u16(pb.add(c)));
                let d1 = vabdq_u16(vld1q_u16(pa.add(c + 8)), vld1q_u16(pb.add(c + 8)));
                acc[0] = vpadalq_u16(acc[0], d0);
                acc[1] = vpadalq_u16(acc[1], d1);
                c += 16;
            }
            if c < w {
                let d = vabdq_u16(vld1q_u16(pa.add(c)), vld1q_u16(pb.add(c)));
                acc[0] = vpadalq_u16(acc[0], d);
            }
        }
    }
    vaddvq_u32(vaddq_u32(acc[0], acc[1]))
}
