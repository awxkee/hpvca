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

use std::sync::OnceLock;

pub(crate) type SatdFn = unsafe fn(&[u16], &[u16], usize) -> u32;

static SATD: OnceLock<SatdFn> = OnceLock::new();

#[inline]
pub(crate) fn resolve_satd() -> SatdFn {
    *SATD.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::satd_neon as SatdFn
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let mut f = satd_scalar as SatdFn;
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::satd_avx2 as SatdFn;
            }
            f
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            satd_scalar as SatdFn
        }
    })
}

#[inline]
#[cfg_attr(
    any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ),
    allow(dead_code)
)]
pub(crate) fn satd_scalar(orig: &[u16], pred: &[u16], n: usize) -> u32 {
    match n {
        4 => satd_scalar_n::<4>(orig, pred),
        8 => satd_scalar_n::<8>(orig, pred),
        16 => satd_scalar_n::<16>(orig, pred),
        32 => satd_scalar_n::<32>(orig, pred),
        _ => panic!("unsupported SATD block size {n}"),
    }
}

#[inline]
#[cfg_attr(
    any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ),
    allow(dead_code)
)]
fn satd_scalar_n<const N: usize>(orig: &[u16], pred: &[u16]) -> u32 {
    assert!(orig.len() >= N * N && pred.len() >= N * N);

    let mut total = 0u32;
    let mut diff = [0i32; 16];

    for (orig_band, pred_band) in orig[..N * N]
        .chunks_exact(N * 4)
        .zip(pred[..N * N].chunks_exact(N * 4))
    {
        for bx in (0..N).step_by(4) {
            for ((dst_row, orig_row), pred_row) in diff
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(orig_band.as_chunks::<N>().0.iter())
                .zip(pred_band.as_chunks::<N>().0.iter())
            {
                for ((dst, &orig), &pred) in dst_row
                    .iter_mut()
                    .zip(&orig_row[bx..bx + 4])
                    .zip(&pred_row[bx..bx + 4])
                {
                    *dst = orig as i32 - pred as i32;
                }
            }

            for row in diff.as_chunks_mut::<4>().0 {
                let a0 = row[0] + row[2];
                let a1 = row[1] + row[3];
                let a2 = row[0] - row[2];
                let a3 = row[1] - row[3];
                row[0] = a0 + a1;
                row[1] = a0 - a1;
                row[2] = a2 + a3;
                row[3] = a2 - a3;
            }
            for col in 0..4 {
                let a0 = diff[col] + diff[8 + col];
                let a1 = diff[4 + col] + diff[12 + col];
                let a2 = diff[col] - diff[8 + col];
                let a3 = diff[4 + col] - diff[12 + col];
                diff[col] = a0 + a1;
                diff[4 + col] = a0 - a1;
                diff[8 + col] = a2 + a3;
                diff[12 + col] = a2 - a3;
            }
            let sum: u32 = diff.iter().map(|value| value.unsigned_abs()).sum();
            total += (sum + 1) >> 1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satd_known_4x4_dc_difference() {
        let orig = [12u16; 16];
        let pred = [4u16; 16];
        // Only the DC Hadamard coefficient is nonzero: 16 * 8 / 2.
        assert_eq!(unsafe { satd_scalar(&orig, &pred, 4) }, 64);
    }

    #[test]
    fn resolved_satd_matches_scalar() {
        let mut orig = [0u16; 1024];
        let mut pred = [0u16; 1024];
        for i in 0..orig.len() {
            orig[i] = ((i * 67 + i / 7 * 131) & 4095) as u16;
            pred[i] = ((i * 29 + i / 3 * 47 + 11) & 4095) as u16;
        }
        let resolved = resolve_satd();
        for n in [4, 8, 16, 32] {
            assert_eq!(
                unsafe { resolved(&orig[..n * n], &pred[..n * n], n) },
                unsafe { satd_scalar(&orig[..n * n], &pred[..n * n], n) },
            );
        }
    }
}
