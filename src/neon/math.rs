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

use core::arch::aarch64::*;

/// exp(x) for finite x in [-80, 80], with a normal result. AQ uses [-5, 0].
/// Reduce by ln(2), evaluate the degree-seven Taylor polynomial on
/// [-ln(2)/2, ln(2)/2], then reconstruct the power of two from its exponent.
/// This internal kernel deliberately omits libm's NaN/overflow/subnormal cases.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn exp_f32x4(x: float32x4_t) -> float32x4_t {
    let k = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
    let r = vfmsq_n_f32(x, k, 0.693_145_75);
    let r = vfmsq_n_f32(r, k, 1.428_606_8e-6);
    let mut p = vdupq_n_f32(1.0 / 5040.0);
    for c in [
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = vfmaq_f32(vdupq_n_f32(c), p, r);
    }
    let exponent = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127)));
    vmulq_f32(p, vreinterpretq_f32_s32(exponent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_libm() {
        for base in 0..40_000 {
            let input = std::array::from_fn::<_, 4, _>(|i| -80.0 + (base * 4 + i) as f32 * 0.001);
            let mut output = [0.0; 4];
            unsafe { vst1q_f32(output.as_mut_ptr(), exp_f32x4(vld1q_f32(input.as_ptr()))) };
            for (x, actual) in input.into_iter().zip(output) {
                let expected = x.exp();
                let relative = (actual - expected).abs() / expected;
                assert!(
                    relative <= 2.5e-7,
                    "exp({x}) = {actual}, expected {expected}"
                );
            }
        }
    }
}
