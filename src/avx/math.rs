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

use core::arch::x86_64::*;

/// exp(x) for finite x in [-80, 80], with a normal result. AQ uses [-5, 0].
/// See the NEON kernel for the range reduction and polynomial.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) fn exp_f32x8(x: __m256) -> __m256 {
    let k = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    let r = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.693_145_75), x);
    let r = _mm256_fnmadd_ps(k, _mm256_set1_ps(1.428_606_8e-6), r);
    let mut p = _mm256_set1_ps(1.0 / 5040.0);
    for c in [
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
    }
    let exponent = _mm256_slli_epi32::<23>(_mm256_add_epi32(
        _mm256_cvttps_epi32(k),
        _mm256_set1_epi32(127),
    ));
    _mm256_mul_ps(p, _mm256_castsi256_ps(exponent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_libm() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for base in 0..20_000 {
            let input = std::array::from_fn::<_, 8, _>(|i| -80.0 + (base * 8 + i) as f32 * 0.001);
            let mut output = [0.0; 8];
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr(),
                    exp_f32x8(_mm256_loadu_ps(input.as_ptr())),
                )
            };
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
