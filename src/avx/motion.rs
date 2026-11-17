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
//! SSE4.1 kernels for inter prediction: fractional-sample interpolation and
//! the sum of absolute differences. They compute exactly what the scalar
//! code in [`crate::motion`] does; the tests there hold them to it.

use crate::motion::{MID_OFFSET, Rounding, Weight};
use core::arch::x86_64::*;

/// `Σ taps[i] · base[i · step ..][lane]` for eight lanes: one FIR output
/// group, horizontal with `step` 1 or vertical with `step` the row stride.
/// Samples are at most 12 bits and intermediates fit 16, so both are read as
/// `i16`; taps are taken in pairs through `pmaddwd`. `N` is even.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn dot<const N: usize>(
    base: *const i16,
    step: usize,
    taps: &[i16; N],
) -> (__m128i, __m128i) {
    let mut lo = _mm_setzero_si128();
    let mut hi = _mm_setzero_si128();
    for i in (0..N).step_by(2) {
        // SAFETY: the caller guarantees eight readable samples at every tap.
        let (a, b) = unsafe {
            (
                _mm_loadu_si128(base.add(i * step).cast()),
                _mm_loadu_si128(base.add((i + 1) * step).cast()),
            )
        };
        let pair = _mm_set1_epi32((i32::from(taps[i + 1]) << 16) | i32::from(taps[i] as u16));
        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), pair));
        hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), pair));
    }
    (lo, hi)
}

/// The rounding that turns filter sums into samples (weighted) or into the
/// intermediates of bi-prediction.
struct Finish {
    intermediate: bool,
    /// Right shift bringing a filter sum to the 14-bit intermediate.
    pre: __m128i,
    weight: __m128i,
    round: __m128i,
    /// `log2_denom + 14 − bit_depth`.
    post: __m128i,
    offset: __m128i,
    max: __m128i,
}

impl Finish {
    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn new(pre_shift: u32, bit_depth: u8, rounding: Rounding) -> Self {
        let (weight, intermediate) = match rounding {
            Rounding::Samples(weight) => (weight, false),
            Rounding::Intermediate => (Weight::UNIT, true),
        };
        let log2_wd = weight.log2_denom + 14 - u32::from(bit_depth);
        Finish {
            intermediate,
            pre: _mm_cvtsi32_si128(pre_shift as i32),
            weight: _mm_set1_epi32(weight.w),
            round: _mm_set1_epi32(1 << (log2_wd - 1)),
            post: _mm_cvtsi32_si128(log2_wd as i32),
            offset: _mm_set1_epi32(weight.offset),
            max: _mm_set1_epi32((1 << bit_depth) - 1),
        }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn half(&self, sum: __m128i) -> __m128i {
        let v = _mm_sra_epi32(sum, self.pre);
        let v = _mm_add_epi32(_mm_mullo_epi32(v, self.weight), self.round);
        let v = _mm_add_epi32(_mm_sra_epi32(v, self.post), self.offset);
        _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), self.max)
    }

    /// Finish eight sums and store the first `lanes` (8 or 4) at `dst`.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn store(&self, sums: (__m128i, __m128i), dst: *mut u16, lanes: usize) {
        let packed = if self.intermediate {
            // `i16` bit patterns; the values fit, so the pack never saturates.
            let offset = _mm_set1_epi32(MID_OFFSET);
            _mm_packs_epi32(
                _mm_sub_epi32(_mm_sra_epi32(sums.0, self.pre), offset),
                _mm_sub_epi32(_mm_sra_epi32(sums.1, self.pre), offset),
            )
        } else {
            _mm_packus_epi32(self.half(sums.0), self.half(sums.1))
        };
        // SAFETY: the caller guarantees `lanes` writable samples at `dst`.
        unsafe {
            if lanes >= 8 {
                _mm_storeu_si128(dst.cast(), packed);
            } else {
                _mm_storel_epi64(dst.cast(), packed);
            }
        }
    }
}

/// Filter the fetched `window` (row stride `ww`) into `out` (stride `w`):
/// the SIMD form of `motion::filter_scalar`, for at least one fractional
/// direction.
///
/// # Safety
///
/// SSE4.1 is available, `w` is a multiple of 4, `out` holds `w · h` samples,
/// and `window` holds the block with its filter margins at row stride `stride`,
/// plus four readable samples after the block if `w` is not a multiple of 8.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn interpolate_sse41<const N: usize>(
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
    debug_assert!(w.is_multiple_of(4) && out.len() >= w * h && N.is_multiple_of(2));
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
                let down = _mm_cvtsi32_si128(shift1 as i32);
                for r in 0..wh {
                    for c in (0..w).step_by(8) {
                        let (lo, hi) = dot(win.add(r * stride + c), 1, &fh);
                        let v = _mm_packs_epi32(_mm_sra_epi32(lo, down), _mm_sra_epi32(hi, down));
                        _mm_storeu_si128(mid.add(r * w + c).cast(), v);
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
/// SSE4.1 is available, and `a`, `b` and `out` have the same length.
#[target_feature(enable = "sse4.1")]
pub(crate) fn bi_average_sse41(a: &[u16], b: &[u16], bit_depth: u8, out: &mut [u16]) {
    debug_assert!(a.len() == out.len() && b.len() == out.len());
    let shift = 15 - i32::from(bit_depth);
    let round = _mm_set1_epi32((1 << (shift - 1)) + 2 * MID_OFFSET);
    let down = _mm_cvtsi32_si128(shift);
    let max = _mm_set1_epi16(((1i32 << bit_depth) - 1) as i16);
    let whole = out.len() & !7;
    for c in (0..whole).step_by(8) {
        // SAFETY: `c + 8` is within all three runs.
        unsafe {
            let p = _mm_loadu_si128(a.as_ptr().add(c).cast());
            let q = _mm_loadu_si128(b.as_ptr().add(c).cast());
            let lo = _mm_add_epi32(_mm_cvtepi16_epi32(p), _mm_cvtepi16_epi32(q));
            let hi = _mm_add_epi32(
                _mm_cvtepi16_epi32(_mm_srli_si128(p, 8)),
                _mm_cvtepi16_epi32(_mm_srli_si128(q, 8)),
            );
            let v = _mm_packus_epi32(
                _mm_sra_epi32(_mm_add_epi32(lo, round), down),
                _mm_sra_epi32(_mm_add_epi32(hi, round), down),
            );
            _mm_storeu_si128(out.as_mut_ptr().add(c).cast(), _mm_min_epu16(v, max));
        }
    }
    crate::motion::bi_average_scalar(&a[whole..], &b[whole..], bit_depth, &mut out[whole..]);
}

/// Sum of absolute differences of two `w`×`h` blocks given as sample runs
/// with row strides `a_stride` and `b_stride`.
///
/// # Safety
///
/// SSE4.1 is available, `w` is a multiple of 8 and both runs hold `h` rows
/// of `w` samples at their stride.
#[target_feature(enable = "sse4.1")]
pub(crate) fn sad_sse41(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    debug_assert!(w.is_multiple_of(8) && h > 0);
    debug_assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w);
    let ones = _mm_set1_epi16(1);
    let mut acc = _mm_setzero_si128();
    for r in 0..h {
        // SAFETY: row `r` holds `w` samples in both runs (checked above), and
        // each load of 8 stays within them because 8 divides `w`.
        unsafe {
            let (pa, pb) = (a.as_ptr().add(r * a_stride), b.as_ptr().add(r * b_stride));
            for c in (0..w).step_by(8) {
                let va = _mm_loadu_si128(pa.add(c).cast());
                let vb = _mm_loadu_si128(pb.add(c).cast());
                // |a − b| of unsigned samples; at most 12 bits, so the
                // pairwise sums of `pmaddwd` cannot overflow.
                let d = _mm_or_si128(_mm_subs_epu16(va, vb), _mm_subs_epu16(vb, va));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(d, ones));
            }
        }
    }
    let acc = _mm_add_epi32(acc, _mm_shuffle_epi32::<0b01_00_11_10>(acc));
    let acc = _mm_add_epi32(acc, _mm_shuffle_epi32::<0b00_00_00_01>(acc));
    _mm_cvtsi128_si32(acc) as u32
}
