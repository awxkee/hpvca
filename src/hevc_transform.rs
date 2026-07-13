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

//! Spec-faithful HEVC integer transform, quantization, and dequantization.

use crate::cabac::ContextSet;
use crate::cabac::residual::sig_coeff_ctx;
use std::sync::OnceLock;

/// 4×4 HEVC transform matrix.
pub(crate) static T4: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// HEVC 4×4 intra-luma integer DST matrix (spec Table 8-7). The transform is
/// normative for 4×4 luma TUs in intra CUs; chroma and larger luma TUs keep DCT.
pub(crate) static DST4: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// 8×8 HEVC transform matrix.
pub(crate) static T8: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

/// 16×16 HEVC transform matrix (spec Table 8-6).
#[rustfmt::skip]
pub(crate) static T16: [[i32; 16]; 16] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9],
];

#[rustfmt::skip]
pub(crate) static T32: [[i32; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

static QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

/// Largest supported transform: 32×32 → 1024 coefficients per fixed buffer.
pub(crate) const MAX_TB: usize = 1024;
pub(crate) static DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Persistent work area for winner-only RDOQ. The HM-style pass jointly
/// selects coefficient levels, coefficient-group significance, CBF and the last
/// position; no second magnitude replay is needed.
pub(crate) struct RdoqScratch {
    cost_coeff: [f32; MAX_TB],
    cost_coeff0: [f32; MAX_TB],
    cost_sig: [f32; MAX_TB],
    cost_group_sig: [f32; 64],
    group_flags: [u8; 64],
}

impl RdoqScratch {
    pub(crate) fn new() -> Self {
        Self {
            cost_coeff: [0.0; MAX_TB],
            cost_coeff0: [0.0; MAX_TB],
            cost_sig: [0.0; MAX_TB],
            cost_group_sig: [0.0; 64],
            group_flags: [0; 64],
        }
    }
}

pub(crate) type FwdTransformFn =
    unsafe fn(&[i32], usize, u8, &mut [i32; MAX_TB], &mut [i32; MAX_TB], bool);

static FWD_TRANSFORM: OnceLock<FwdTransformFn> = OnceLock::new();

pub(crate) type InvTransformFn =
    unsafe fn(&[i32], usize, u8, &mut [i32; MAX_TB], &mut [i32; MAX_TB], bool);

static INV_TRANSFORM: OnceLock<InvTransformFn> = OnceLock::new();

pub(crate) type DequantizeFn = unsafe fn(&[i16], usize, u8, u8, &mut [i32; MAX_TB]);

static DEQUANTIZE: OnceLock<DequantizeFn> = OnceLock::new();

#[inline]
pub(crate) fn resolve_fwd_transform() -> FwdTransformFn {
    *FWD_TRANSFORM.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::fwd_transform_neon as FwdTransformFn
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let mut f = fwd_transform_scalar as FwdTransformFn;
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::fwd_transform_avx2 as FwdTransformFn;
            }
            f
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            fwd_transform_scalar as FwdTransformFn
        }
    })
}

#[inline]
pub(crate) fn resolve_inv_transform() -> InvTransformFn {
    *INV_TRANSFORM.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::inv_transform_neon as InvTransformFn
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let mut f = inv_transform_scalar as InvTransformFn;
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::inv_transform_avx2 as InvTransformFn;
            }
            f
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            inv_transform_scalar as InvTransformFn
        }
    })
}

#[inline]
pub(crate) fn resolve_dequantize() -> DequantizeFn {
    *DEQUANTIZE.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::dequantize_neon as DequantizeFn
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let mut f = dequantize_into as DequantizeFn;
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::dequantize_avx2 as DequantizeFn;
            }
            f
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            dequantize_into as DequantizeFn
        }
    })
}

#[inline]
pub(crate) fn run_fwd_transform(
    f: FwdTransformFn,
    res: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
    intra_luma: bool,
) {
    // SAFETY: resolvers only return implementations with this slice-based
    // contract; each implementation validates the transform size and buffers.
    unsafe { f(res, n, bit_depth, out, tmp, intra_luma) }
}

#[inline]
pub(crate) fn run_inv_transform(
    f: InvTransformFn,
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
    intra_luma: bool,
) {
    // SAFETY: resolvers only return implementations with this slice-based
    // contract; each implementation validates the transform size and buffers.
    unsafe { f(coeff, n, bit_depth, out, tmp, intra_luma) }
}

#[inline]
pub(crate) fn run_dequantize(
    f: DequantizeFn,
    level: &[i16],
    n: usize,
    qp: u8,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    // SAFETY: resolvers only return implementations with this slice-based
    // contract; callers provide one complete supported transform block.
    unsafe { f(level, n, qp, bit_depth, out) }
}

#[inline]
#[cfg_attr(
    any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ),
    allow(dead_code)
)]
pub(crate) fn fwd_transform_scalar(
    res: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
    intra_luma: bool,
) {
    if intra_luma {
        fwd_transform_intra_luma_into(res, n, bit_depth, out, tmp);
    } else {
        fwd_transform_into(res, n, bit_depth, out, tmp);
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
pub(crate) fn inv_transform_scalar(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
    intra_luma: bool,
) {
    if intra_luma {
        inv_transform_intra_luma_into(coeff, n, bit_depth, out, tmp);
    } else {
        inv_transform_into(coeff, n, bit_depth, out, tmp);
    }
}

/// Forward integer transform of an N×N residual block (N = 4, 8, 16, or 32)
/// into caller-owned storage. `tmp` is reusable transpose/intermediate scratch;
/// only the first `n*n` entries of either buffer are touched.
#[inline]
pub(crate) fn fwd_transform_into(
    res: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    match n {
        4 => fwd_transform_n::<4>(res, &T4, bit_depth, out, tmp),
        8 => fwd_transform_n::<8>(res, &T8, bit_depth, out, tmp),
        16 => fwd_transform_n::<16>(res, &T16, bit_depth, out, tmp),
        32 => fwd_transform_32(res, bit_depth, out, tmp),
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Forward transform for intra luma. HEVC substitutes the integer DST for the
/// 4×4 case and uses the regular integer DCT for every larger transform size.
#[inline]
pub(crate) fn fwd_transform_intra_luma_into(
    res: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    if n == 4 {
        fwd_transform_n::<4>(res, &DST4, bit_depth, out, tmp);
    } else {
        fwd_transform_into(res, n, bit_depth, out, tmp);
    }
}

#[inline]
fn fwd_transform_n<const N: usize>(
    res: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    let log2n = N.trailing_zeros() as i32;
    let bd = bit_depth as i32;
    let shift1 = log2n + bd - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    // i32 throughout: products (|coeff|≤90 · |residual|≤4095) and the ≤N-term
    // sums stay well inside i32 for every supported bit depth.
    // pass 1 (rows of res): tmp[j*N+i] = (Σ_k T[i][k]·res[j*N+k]) >> shift1
    for (j, res_row) in res.as_chunks::<N>().0.iter().enumerate().take(N) {
        for (i, trow) in t.iter().enumerate() {
            let mut s = 0i32;
            for k in 0..N {
                s += trow[k] * res_row[k];
            }
            tmp[j * N + i] = if shift1 > 0 { (s + add1) >> shift1 } else { s };
        }
    }
    // pass 2 (columns): coeff[i*N+j] = (Σ_k T[i][k]·tmp[k*N+j]) >> shift2
    let shift2 = log2n + 6;
    let add2 = 1i32 << (shift2 - 1);
    let mut colv = [0i32; N];
    for j in 0..N {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = tmp[k * N + j];
        }
        for (i, trow) in t.iter().enumerate() {
            let mut s = 0i32;
            for k in 0..N {
                s += trow[k] * colv[k];
            }
            out[i * N + j] = (s + add2) >> shift2;
        }
    }
}

#[inline(always)]
fn round_shift_i32(value: i32, shift: i32) -> i32 {
    if shift > 0 {
        (value + (1i32 << (shift - 1))) >> shift
    } else {
        value
    }
}

/// HM-style partial-butterfly 32-point forward transform. A dense matrix
/// multiply needs 1024 multiplies per vector; symmetry reduces this to 344.
#[inline]
fn fwd_transform_1d_32(src: &[i32], dst: &mut [i32], shift: i32) {
    debug_assert!(src.len() >= 32 && dst.len() >= 32);
    let mut e = [0i32; 16];
    let mut o = [0i32; 16];
    for k in 0..16 {
        e[k] = src[k] + src[31 - k];
        o[k] = src[k] - src[31 - k];
    }

    let mut ee = [0i32; 8];
    let mut eo = [0i32; 8];
    for k in 0..8 {
        ee[k] = e[k] + e[15 - k];
        eo[k] = e[k] - e[15 - k];
    }

    let mut eee = [0i32; 4];
    let mut eeo = [0i32; 4];
    for k in 0..4 {
        eee[k] = ee[k] + ee[7 - k];
        eeo[k] = ee[k] - ee[7 - k];
    }

    let mut eeee = [0i32; 2];
    let mut eeeo = [0i32; 2];
    for k in 0..2 {
        eeee[k] = eee[k] + eee[3 - k];
        eeeo[k] = eee[k] - eee[3 - k];
    }

    for k in [0usize, 16] {
        let sum = T32[k][0] * eeee[0] + T32[k][1] * eeee[1];
        dst[k] = round_shift_i32(sum, shift);
    }
    for k in [8usize, 24] {
        let sum = T32[k][0] * eeeo[0] + T32[k][1] * eeeo[1];
        dst[k] = round_shift_i32(sum, shift);
    }
    for k in (4..32).step_by(8) {
        let mut sum = 0i32;
        for j in 0..4 {
            sum += T32[k][j] * eeo[j];
        }
        dst[k] = round_shift_i32(sum, shift);
    }
    for k in (2..32).step_by(4) {
        let mut sum = 0i32;
        for j in 0..8 {
            sum += T32[k][j] * eo[j];
        }
        dst[k] = round_shift_i32(sum, shift);
    }
    for k in (1..32).step_by(2) {
        let mut sum = 0i32;
        for j in 0..16 {
            sum += T32[k][j] * o[j];
        }
        dst[k] = round_shift_i32(sum, shift);
    }
}

#[inline]
fn fwd_transform_32(res: &[i32], bit_depth: u8, out: &mut [i32; MAX_TB], tmp: &mut [i32; MAX_TB]) {
    debug_assert!(res.len() >= 32 * 32);
    let shift1 = bit_depth as i32 - 4;
    let shift2 = 11;

    for row in 0..32 {
        let src = &res[row * 32..row * 32 + 32];
        let dst = &mut tmp[row * 32..row * 32 + 32];
        fwd_transform_1d_32(src, dst, shift1);
    }

    let mut col = [0i32; 32];
    let mut transformed = [0i32; 32];
    for c in 0..32 {
        for r in 0..32 {
            col[r] = tmp[r * 32 + c];
        }
        fwd_transform_1d_32(&col, &mut transformed, shift2);
        for r in 0..32 {
            out[r * 32 + c] = transformed[r];
        }
    }
}

#[inline]
fn coeff_remaining_bits(value: u32, rice: u32) -> f32 {
    if value < (4u32 << rice) {
        (value >> rice) as f32 + 1.0 + rice as f32
    } else {
        let mut prefix = 4u32;
        while value >= (((1u32 << (prefix + 1 - 3)) + 2) << rice) {
            prefix += 1;
        }
        // Unary prefix + separator, followed by the extended suffix.
        (prefix + 1 + prefix - 3 + rice) as f32
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn rdoq_level_bits(
    abs_level: u32,
    ctx_set: usize,
    c1: i32,
    c1_idx: u32,
    c2_idx: u32,
    rice: u32,
    is_luma: bool,
    ctx: &ContextSet,
) -> f32 {
    debug_assert!(abs_level > 0);
    const C1_FLAGS: u32 = 8;
    const C2_FLAGS: u32 = 1;

    let mut bits = 1.0; // sign bypass bin
    let greater1_offset = if is_luma { 0 } else { 16 };
    let one_ctx = (greater1_offset + ctx_set * 4 + c1.clamp(0, 3) as usize)
        .min(ctx.coeff_abs_level_greater1.len() - 1);
    let greater2_offset = if is_luma { 0 } else { 4 };
    let abs_ctx = (greater2_offset + ctx_set).min(ctx.coeff_abs_level_greater2.len() - 1);
    let base_level = if c1_idx < C1_FLAGS {
        2 + (c2_idx < C2_FLAGS) as u32
    } else {
        1
    };

    if abs_level >= base_level {
        bits += coeff_remaining_bits(abs_level - base_level, rice);
        if c1_idx < C1_FLAGS {
            bits += ctx.coeff_abs_level_greater1[one_ctx].estimated_bits(1);
            if c2_idx < C2_FLAGS {
                bits += ctx.coeff_abs_level_greater2[abs_ctx].estimated_bits(1);
            }
        }
    } else if abs_level == 1 {
        if c1_idx < C1_FLAGS {
            bits += ctx.coeff_abs_level_greater1[one_ctx].estimated_bits(0);
        }
    } else {
        debug_assert_eq!(abs_level, 2);
        if c1_idx < C1_FLAGS {
            bits += ctx.coeff_abs_level_greater1[one_ctx].estimated_bits(1);
            if c2_idx < C2_FLAGS {
                bits += ctx.coeff_abs_level_greater2[abs_ctx].estimated_bits(0);
            }
        }
    }
    bits
}

#[inline]
fn last_sig_bits(
    ctx: &ContextSet,
    x: usize,
    y: usize,
    log2_size: u32,
    scan_idx: u8,
    is_luma: bool,
) -> f32 {
    static GROUP_IDX: [usize; 32] = [
        0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
        9, 9,
    ];
    static MIN_IN_GROUP: [usize; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

    let (x, y) = if scan_idx == 2 { (y, x) } else { (x, y) };
    let (ctx_offset, ctx_shift) = if is_luma {
        (
            (3 * (log2_size - 2) + ((log2_size - 1) >> 2)) as usize,
            ((log2_size + 1) >> 2) as usize,
        )
    } else {
        (15usize, (log2_size - 2) as usize)
    };
    let max_group = GROUP_IDX[(1usize << log2_size) - 1];

    let prefix_bits = |value: usize, models: &[crate::cabac::engine::CtxModel]| {
        let group = GROUP_IDX[value];
        let mut bits = 0.0;
        for i in 0..group {
            let ci = (ctx_offset + (i >> ctx_shift)).min(models.len() - 1);
            bits += models[ci].estimated_bits(1);
        }
        if group < max_group {
            let ci = (ctx_offset + (group >> ctx_shift)).min(models.len() - 1);
            bits += models[ci].estimated_bits(0);
        }
        if group > 3 {
            bits += ((group - 2) / 2) as f32;
            debug_assert!(value >= MIN_IN_GROUP[group]);
        }
        bits
    };

    prefix_bits(x, &ctx.last_sig_coeff_x_prefix) + prefix_bits(y, &ctx.last_sig_coeff_y_prefix)
}

#[derive(Clone, Copy)]
struct RdoqDistortion {
    factor: i64,
    add: i64,
    shift: u32,
    scale: f32,
}

impl RdoqDistortion {
    #[inline]
    fn new(n: usize, qp: u8, bit_depth: u8) -> Self {
        let log2n = n.trailing_zeros() as i32;
        let shift = (bit_depth as i32 + log2n - 5) as u32;
        let exponent = 2 * (bit_depth as i32 + log2n - 15);
        let scale = if exponent >= 0 {
            (1u32 << exponent as u32) as f32
        } else {
            1.0 / (1u32 << (-exponent) as u32) as f32
        };
        Self {
            factor: DEQUANT_SCALE[(qp % 6) as usize] * (1i64 << (qp / 6)) * 16,
            add: 1i64 << (shift - 1),
            shift,
            scale,
        }
    }

    #[inline]
    fn coefficient_cost(self, coeff_abs: i64, level: u32) -> f32 {
        let dequant = ((level as i64 * self.factor + self.add) >> self.shift).clamp(0, 32767);
        let error = coeff_abs - dequant;
        (error * error) as f32 * self.scale
    }
}

pub(crate) struct RdoqTb<'a> {
    pub coeff: &'a [i32],
    pub n: usize,
    pub qp: u8,
    pub bit_depth: u8,
    pub scan: &'a [(usize, usize)],
    pub scan_idx: u8,
    pub lambda: f32,
}

/// Rate-distortion optimized quantization for a committed luma or chroma mode.
fn rdoq_with_sign_hiding_into(
    tb: &RdoqTb<'_>,
    is_luma: bool,
    cbf_depth: usize,
    ctx: &ContextSet,
    levels: &mut [i16; MAX_TB],
    scratch: &mut RdoqScratch,
) {
    let RdoqTb {
        coeff,
        n,
        qp,
        bit_depth,
        scan,
        scan_idx,
        lambda,
    } = *tb;
    const GROUP_SIZE: usize = 16;
    const C1_FLAGS: u32 = 8;

    let num_coeffs = n * n;
    debug_assert_eq!(scan.len(), num_coeffs);
    debug_assert!(matches!(n, 4 | 8 | 16 | 32));
    let log2_size = n.trailing_zeros();
    let num_groups = num_coeffs / GROUP_SIZE;
    let sb_side = n / 4;
    let sb_scan = crate::dct::sb_scan_for(log2_size, scan_idx);
    let q_bits = 14 + qp as i64 / 6 + (15 - bit_depth as i64 - log2_size as i64);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let round = 1i64 << (q_bits - 1);
    let distortion = RdoqDistortion::new(n, qp, bit_depth);

    let RdoqScratch {
        cost_coeff,
        cost_coeff0,
        cost_sig,
        cost_group_sig,
        group_flags,
    } = scratch;
    // Every coefficient and group entry is assigned on this reverse scan before
    // it can be observed. This avoids clearing the level and group arrays for
    // every winner-only TU.
    let mut block_uncoded_cost = 0.0;
    let mut base_cost = 0.0;
    let mut last_scan_pos = None;
    let mut last_group = 0usize;
    let mut carry_c1 = 1i32;

    for group in (0..num_groups).rev() {
        // The last/non-coded group has no coded_sub_block_flag contribution.
        // Set this entry lazily rather than clearing the complete array.
        cost_group_sig[group] = 0.0;
        let group_start = group * GROUP_SIZE;
        let (sbx, sby) = sb_scan[group];
        let group_grid = sbx + sby * sb_side;
        let right = sbx + 1 < sb_side && group_flags[group_grid + 1] != 0;
        let below = sby + 1 < sb_side && group_flags[group_grid + sb_side] != 0;
        let prev_csbf = right as u8 | ((below as u8) << 1);
        group_flags[group_grid] = 0;

        let mut ctx_set = if group == 0 || !is_luma {
            0usize
        } else {
            2usize
        };
        if carry_c1 == 0 {
            ctx_set += 1;
        }
        let mut c1 = 1i32;
        let mut c1_idx = 0u32;
        let mut c2_idx = 0u32;
        let mut rice = 0u32;

        let mut group_sig_cost = 0.0;
        let mut group_coded_level_and_dist = 0.0;
        let mut group_uncoded_dist = 0.0;
        let mut nnz_before_pos0 = 0usize;
        let mut group_has_nonzero = false;

        for k in (0..GROUP_SIZE).rev() {
            let scan_pos = group_start + k;
            let (row, col) = scan[scan_pos];
            let pos = row * n + col;
            let coeff_abs = (coeff[pos] as i64).abs();
            let scaled = coeff_abs * q_scale;
            let max_abs = ((scaled + round) >> q_bits).clamp(0, i16::MAX as i64) as u32;
            let dist0 = distortion.coefficient_cost(coeff_abs, 0);
            cost_coeff0[scan_pos] = dist0;
            block_uncoded_cost += dist0;

            if last_scan_pos.is_none() && max_abs == 0 {
                levels[pos] = 0;
                cost_coeff[scan_pos] = dist0;
                base_cost += dist0;
                continue;
            }
            if last_scan_pos.is_none() {
                last_scan_pos = Some(scan_pos);
                last_group = group;
            }
            let is_last = last_scan_pos == Some(scan_pos);
            let sig_ctx = sig_coeff_ctx(col, row, prev_csbf, log2_size, scan_idx, is_luma)
                .min(ctx.sig_coeff_flag.len() - 1);
            let sig0 = if is_last {
                0.0
            } else {
                lambda * ctx.sig_coeff_flag[sig_ctx].estimated_bits(0)
            };
            let sig1 = if is_last {
                0.0
            } else {
                lambda * ctx.sig_coeff_flag[sig_ctx].estimated_bits(1)
            };

            let mut best_level = 0u32;
            let mut best_cost = dist0 + sig0;
            if max_abs > 0 {
                let min_abs = if max_abs > 1 { max_abs - 1 } else { 1 };
                for abs_level in [max_abs, min_abs] {
                    let dist = distortion.coefficient_cost(coeff_abs, abs_level);
                    let rate =
                        rdoq_level_bits(abs_level, ctx_set, c1, c1_idx, c2_idx, rice, is_luma, ctx);
                    let coded_cost = dist + sig1 + lambda * rate;
                    if coded_cost < best_cost || is_last && best_level == 0 {
                        best_cost = coded_cost;
                        best_level = abs_level;
                    }
                    if min_abs == max_abs {
                        break;
                    }
                }
                // HM directly tests zero for small levels. Larger all-zero
                // choices are handled more efficiently by CG and last cleanup.
                if !is_last && max_abs >= 3 {
                    best_cost = f32::MAX;
                    for abs_level in [max_abs, min_abs] {
                        let dist = distortion.coefficient_cost(coeff_abs, abs_level);
                        let rate = rdoq_level_bits(
                            abs_level, ctx_set, c1, c1_idx, c2_idx, rice, is_luma, ctx,
                        );
                        let coded_cost = dist + sig1 + lambda * rate;
                        if coded_cost < best_cost {
                            best_cost = coded_cost;
                            best_level = abs_level;
                        }
                        if min_abs == max_abs {
                            break;
                        }
                    }
                }
            }

            levels[pos] = best_level as i16;
            cost_coeff[scan_pos] = best_cost;
            cost_sig[scan_pos] = if best_level == 0 { sig0 } else { sig1 };
            base_cost += best_cost;
            group_sig_cost += cost_sig[scan_pos];

            if best_level > 0 {
                group_has_nonzero = true;
                group_coded_level_and_dist += best_cost - cost_sig[scan_pos];
                group_uncoded_dist += dist0;
                if k != 0 {
                    nnz_before_pos0 += 1;
                }

                let base_level = if c1_idx < C1_FLAGS {
                    2 + (c2_idx == 0) as u32
                } else {
                    1
                };
                if best_level >= base_level && best_level > (3 << rice) {
                    rice = (rice + 1).min(4);
                }
                c1_idx += 1;
                if best_level > 1 {
                    c1 = 0;
                    c2_idx += 1;
                } else if (1..3).contains(&c1) {
                    c1 += 1;
                }
            }
        }
        carry_c1 = c1;

        let Some(_) = last_scan_pos else {
            continue;
        };
        if group == last_group {
            group_flags[group_grid] = 1;
            continue;
        }
        if group == 0 {
            group_flags[group_grid] = group_has_nonzero as u8;
            continue;
        }

        let cg_ctx = if is_luma {
            (prev_csbf != 0) as usize
        } else {
            2 + (prev_csbf != 0) as usize
        };
        let csbf0 = lambda * ctx.coded_sub_block_flag[cg_ctx].estimated_bits(0);
        let csbf1 = lambda * ctx.coded_sub_block_flag[cg_ctx].estimated_bits(1);
        if !group_has_nonzero {
            base_cost += csbf0 - group_sig_cost;
            cost_group_sig[group] = csbf0;
            group_flags[group_grid] = 0;
            continue;
        }

        // With an explicitly coded non-zero CG, coefficient zero is inferred
        // significant when no higher position in the group is significant.
        if nnz_before_pos0 == 0 {
            base_cost -= cost_sig[group_start];
            group_sig_cost -= cost_sig[group_start];
            cost_sig[group_start] = 0.0;
        }
        let coded_group_cost = base_cost + csbf1;
        let zero_group_cost =
            base_cost + csbf0 + group_uncoded_dist - group_coded_level_and_dist - group_sig_cost;
        if zero_group_cost < coded_group_cost {
            base_cost = zero_group_cost;
            cost_group_sig[group] = csbf0;
            group_flags[group_grid] = 0;
            for scan_pos in group_start..group_start + GROUP_SIZE {
                let (row, col) = scan[scan_pos];
                levels[row * n + col] = 0;
                cost_coeff[scan_pos] = cost_coeff0[scan_pos];
                cost_sig[scan_pos] = 0.0;
            }
        } else {
            base_cost = coded_group_cost;
            cost_group_sig[group] = csbf1;
            group_flags[group_grid] = 1;
        }
    }

    let Some(initial_last) = last_scan_pos else {
        return;
    };

    // Compare the non-zero TU against CBF=0, then move the last-significant
    // position down through trailing unit levels. HM stops once it reaches a
    // level greater than one because discarding it is rarely profitable.
    let cbf = if is_luma {
        ctx.cbf_luma[if cbf_depth == 0 { 1 } else { 0 }]
    } else {
        ctx.cbf_chroma[cbf_depth.min(4)]
    };
    let mut best_cost = block_uncoded_cost + lambda * cbf.estimated_bits(0);
    base_cost += lambda * cbf.estimated_bits(1);
    let mut best_last_p1 = 0usize;
    let mut stop = false;
    for group in (0..=last_group).rev() {
        base_cost -= cost_group_sig[group];
        let (sbx, sby) = sb_scan[group];
        let group_grid = sbx + sby * sb_side;
        if group_flags[group_grid] == 0 {
            continue;
        }
        for k in (0..GROUP_SIZE).rev() {
            let scan_pos = group * GROUP_SIZE + k;
            if scan_pos > initial_last {
                continue;
            }
            let (row, col) = scan[scan_pos];
            let pos = row * n + col;
            let level = levels[pos].unsigned_abs() as u32;
            if level != 0 {
                let total = base_cost
                    + lambda * last_sig_bits(ctx, col, row, log2_size, scan_idx, is_luma)
                    - cost_sig[scan_pos];
                if total < best_cost {
                    best_cost = total;
                    best_last_p1 = scan_pos + 1;
                }
                if level > 1 {
                    stop = true;
                    break;
                }
                base_cost += cost_coeff0[scan_pos] - cost_coeff[scan_pos];
            } else {
                base_cost -= cost_sig[scan_pos];
            }
        }
        if stop {
            break;
        }
    }

    for (scan_pos, &(row, col)) in scan.iter().enumerate() {
        if scan_pos >= best_last_p1 {
            levels[row * n + col] = 0;
        }
    }

    for &(row, col) in scan.iter().take(best_last_p1) {
        let pos = row * n + col;
        if coeff[pos] < 0 {
            levels[pos] = -levels[pos];
        }
    }
    apply_sign_hiding_to_levels(levels, coeff, n, qp, bit_depth, scan);
}

/// Winner-only RDOQ for a committed luma mode.
pub(crate) fn rdoq_luma_with_sign_hiding_into(
    tb: &RdoqTb<'_>,
    ctx: &ContextSet,
    levels: &mut [i16; MAX_TB],
    scratch: &mut RdoqScratch,
) {
    rdoq_with_sign_hiding_into(tb, true, 0, ctx, levels, scratch);
}

/// Winner-only RDOQ for a committed chroma mode. Chroma uses its own
/// significant-coefficient, coefficient-group, greater1/greater2 and CBF
/// contexts while sharing the same HM-style significance, level, CG, CBF and
/// last-position optimization as luma.
pub(crate) fn rdoq_chroma_with_sign_hiding_into(
    tb: &RdoqTb<'_>,
    ctx: &ContextSet,
    levels: &mut [i16; MAX_TB],
    scratch: &mut RdoqScratch,
) {
    rdoq_with_sign_hiding_into(tb, false, 0, ctx, levels, scratch);
}

/// Depth-aware luma RDOQ used by child TUs in the transform quadtree.
pub(crate) fn rdoq_luma_at_depth_with_sign_hiding_into(
    tb: &RdoqTb<'_>,
    trafo_depth: usize,
    ctx: &ContextSet,
    levels: &mut [i16; MAX_TB],
    scratch: &mut RdoqScratch,
) {
    rdoq_with_sign_hiding_into(tb, true, trafo_depth, ctx, levels, scratch);
}

/// Depth-aware chroma RDOQ used by child TUs in the transform quadtree.
pub(crate) fn rdoq_chroma_at_depth_with_sign_hiding_into(
    tb: &RdoqTb<'_>,
    trafo_depth: usize,
    ctx: &ContextSet,
    levels: &mut [i16; MAX_TB],
    scratch: &mut RdoqScratch,
) {
    rdoq_with_sign_hiding_into(tb, false, trafo_depth, ctx, levels, scratch);
}

/// Forward quantization followed by HEVC sign-data hiding.
///
/// `scan` is the TU's coefficient scan in sub-block-major order. The level
/// adjustment is the distortion-only `signBitHidingHDQ` method used by HM: for
/// each eligible 4×4 coefficient group, change the cheapest magnitude by one so
/// the parity of the absolute-level sum carries the first significant sign.
#[inline]
pub(crate) fn quantize_with_sign_hiding_into(
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    scan: &[(usize, usize)],
    out: &mut [i16; MAX_TB],
) {
    debug_assert_eq!(scan.len(), n * n);
    quantize_impl_into(coeff, n, qp, bit_depth, Some(scan), out);
}

/// Apply HEVC sign-data hiding to an already selected set of coefficient levels.
/// Winner-only RDOQ changes some of the plain quantizer's levels, so parity is
/// repaired once after level optimization has finished. Error deltas are computed
/// lazily only for the eligible coefficient groups instead of written to a full
/// 1024-entry temporary for every TU.
fn apply_sign_hiding_to_levels(
    levels: &mut [i16; MAX_TB],
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    scan: &[(usize, usize)],
) {
    let abs_sum = levels[..n * n].iter().fold(0u32, |sum, level| {
        sum.saturating_add(level.unsigned_abs() as u32)
    });
    if abs_sum >= 2 {
        sign_bit_hiding_hdq(levels, coeff, n, scan, qp, bit_depth, false);
    }
}

fn quantize_impl_into(
    coeff: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    sign_hiding_scan: Option<&[(usize, usize)]>,
    out: &mut [i16; MAX_TB],
) {
    let log2n = n.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bd - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra
    let mut abs_sum = 0u32;

    for (level, &coefficient) in out[..n * n].iter_mut().zip(coeff) {
        // i64 product: |coeff| can reach ~2^16 and q_scale ~2^15, so the product
        // overflows i32; keep this one multiply in i64.
        let coefficient = coefficient as i64;
        let magnitude = (coefficient.abs() * q_scale + offset) >> q_bits;
        let signed = if coefficient < 0 {
            -magnitude
        } else {
            magnitude
        };
        *level = signed.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        abs_sum = abs_sum.saturating_add((*level).unsigned_abs() as u32);
    }

    // HM avoids testing the degenerate one-coefficient, unit-level TU.
    if abs_sum >= 2
        && let Some(scan) = sign_hiding_scan
    {
        sign_bit_hiding_hdq(out, coeff, n, scan, qp, bit_depth, true);
    }
}

fn sign_bit_hiding_hdq(
    levels: &mut [i16; MAX_TB],
    coeff: &[i32],
    n: usize,
    scan: &[(usize, usize)],
    qp: u8,
    bit_depth: u8,
    use_rounded_magnitude: bool,
) {
    const GROUP_SIZE: usize = 16;
    const SBH_THRESHOLD: usize = 4;

    let num_coeffs = n * n;
    debug_assert_eq!(num_coeffs % GROUP_SIZE, 0);
    debug_assert!(coeff.len() >= num_coeffs);
    debug_assert_eq!(scan.len(), num_coeffs);
    let log2n = n.trailing_zeros() as i64;
    let q_bits = 14 + qp as i64 / 6 + (15 - bit_depth as i64 - log2n);
    let q_bits_8 = q_bits - 8;
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let quant_offset = 171i64 << (q_bits - 9);

    let mut found_last_group = false;
    for subset in (0..num_coeffs / GROUP_SIZE).rev() {
        let sub_pos = subset * GROUP_SIZE;
        let group = &scan[sub_pos..sub_pos + GROUP_SIZE];
        let row_major = |scan_pos: usize| {
            let (row, col) = group[scan_pos];
            row * n + col
        };

        let Some(first_nz) = (0..GROUP_SIZE).find(|&i| levels[row_major(i)] != 0) else {
            continue;
        };
        let last_nz = (0..GROUP_SIZE)
            .rev()
            .find(|&i| levels[row_major(i)] != 0)
            .expect("non-empty coefficient group has a last coefficient");

        let is_last_group = !found_last_group;
        found_last_group = true;

        if last_nz - first_nz < SBH_THRESHOLD {
            continue;
        }

        // Signed and absolute sums have identical parity in two's-complement
        // arithmetic, matching HM's signed pQCoef accumulation.
        let sum: i32 = (first_nz..=last_nz)
            .map(|i| levels[row_major(i)] as i32)
            .sum();
        let first_pos = row_major(first_nz);
        let sign_bit = (levels[first_pos] < 0) as i32;
        if sign_bit == (sum & 1) {
            continue;
        }

        let search_top = if is_last_group {
            last_nz
        } else {
            GROUP_SIZE - 1
        };
        let mut best_cost = i64::MAX;
        let mut best_pos = None;
        let mut best_change = 0i32;

        for i in (0..=search_top).rev() {
            let pos = row_major(i);
            let level = levels[pos];
            let scaled = (coeff[pos] as i64).abs() * q_scale;
            let magnitude = if use_rounded_magnitude {
                (scaled + quant_offset) >> q_bits
            } else {
                level.unsigned_abs() as i64
            };
            let delta = (scaled - (magnitude << q_bits)) >> q_bits_8;

            let candidate = if level != 0 {
                if delta > 0 {
                    Some((-delta, 1))
                } else if i == first_nz && level.unsigned_abs() == 1 {
                    // Removing the coefficient whose sign is being hidden would
                    // change which sign the decoder infers.
                    None
                } else {
                    Some((delta, -1))
                }
            } else if i < first_nz {
                // Creating a new first significant coefficient is legal only if
                // its natural sign equals the sign currently being hidden.
                let this_sign = (coeff[pos] < 0) as i32;
                if this_sign == sign_bit {
                    Some((-delta, 1))
                } else {
                    None
                }
            } else {
                Some((-delta, 1))
            };

            if let Some((cost, change)) = candidate
                && cost < best_cost
            {
                best_cost = cost;
                best_pos = Some(pos);
                best_change = change;
            }
        }

        let pos = best_pos.expect("eligible sign-hiding group has a valid adjustment");
        if levels[pos] == i16::MAX || levels[pos] == i16::MIN {
            best_change = -1;
        }
        let adjusted = if coeff[pos] >= 0 {
            levels[pos] as i32 + best_change
        } else {
            levels[pos] as i32 - best_change
        };
        levels[pos] = adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Dequantisation: level → transform coefficient (spec 8.6.3).
#[inline]
#[cfg_attr(
    any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ),
    allow(dead_code)
)]
pub(crate) fn dequantize_into(
    level: &[i16],
    n: usize,
    qp: u8,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    let log2n = n.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let bd_shift = bd + log2n - 5;
    let add = 1i64 << (bd_shift - 1);
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let factor = scale * per * 16;
    for (dst, &level) in out[..n * n].iter_mut().zip(&level[..n * n]) {
        *dst = ((level as i64 * factor + add) >> bd_shift).clamp(-32768, 32767) as i32;
    }
}

/// Inverse integer transform (spec 8.6.4.2) into reusable output/intermediate
/// buffers. Only the first `n*n` entries are touched.
#[inline]
pub(crate) fn inv_transform_into(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    match n {
        4 => inv_transform_n::<4>(coeff, &T4, bit_depth, out, tmp),
        8 => inv_transform_n::<8>(coeff, &T8, bit_depth, out, tmp),
        16 => inv_transform_n::<16>(coeff, &T16, bit_depth, out, tmp),
        32 => inv_transform_32(coeff, bit_depth, out, tmp),
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Inverse transform paired with [`fwd_transform_intra_luma_into`].
#[inline]
pub(crate) fn inv_transform_intra_luma_into(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    if n == 4 {
        inv_transform_n::<4>(coeff, &DST4, bit_depth, out, tmp);
    } else {
        inv_transform_into(coeff, n, bit_depth, out, tmp);
    }
}

#[inline]
fn inv_transform_n<const N: usize>(
    coeff: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    let bd = bit_depth as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bd;
    let add2 = 1i32 << (shift2 - 1);

    // i32 accumulation is exact (dequant clamps to ±32768); basis rows are read
    // contiguously and zero coefficients are skipped — residual blocks are
    // mostly zero, so the skip removes the bulk of the work.
    let mut acc = [0i32; N];

    // Stage 1 (columns): tmp[m*N+c] = clip( Σ_k T[k][m]·coeff[k*N+c] ) >> 7
    for c in 0..N {
        acc[..N].fill(0);
        for k in 0..N {
            let ck = coeff[k * N + c];
            if ck == 0 {
                continue;
            }
            let trow = &t[k];
            for m in 0..N {
                acc[m] += trow[m] * ck;
            }
        }
        for m in 0..N {
            tmp[m * N + c] = ((acc[m] + add1) >> shift1).clamp(-32768, 32767);
        }
    }

    // Stage 2 (rows): out[r*N+m] = ( Σ_k T[k][m]·tmp[r*N+k] ) >> (20-bd)
    for r in 0..N {
        acc[..N].fill(0);
        let rowv = &tmp[r * N..r * N + N];
        for k in 0..N {
            let rk = rowv[k];
            if rk == 0 {
                continue;
            }
            let trow = &t[k];
            for m in 0..N {
                acc[m] += trow[m] * rk;
            }
        }
        for m in 0..N {
            out[r * N + m] = (acc[m] + add2) >> shift2;
        }
    }
}

/// HM-style partial-butterfly inverse. For very sparse vectors, the existing
/// zero-skipping matrix path is cheaper, so retain it below eleven non-zeros.
#[inline]
fn inv_transform_1d_32(src: &[i32], dst: &mut [i32], shift: i32, output_min: i32, output_max: i32) {
    debug_assert!(src.len() >= 32 && dst.len() >= 32);
    let nonzero = src[..32].iter().filter(|&&value| value != 0).count();
    if nonzero <= 10 {
        for m in 0..32 {
            let mut sum = 0i32;
            for k in 0..32 {
                let value = src[k];
                if value != 0 {
                    sum += T32[k][m] * value;
                }
            }
            dst[m] = round_shift_i32(sum, shift).clamp(output_min, output_max);
        }
        return;
    }

    let mut o = [0i32; 16];
    for k in 0..16 {
        let mut sum = 0i32;
        for j in (1..32).step_by(2) {
            sum += T32[j][k] * src[j];
        }
        o[k] = sum;
    }

    let mut eo = [0i32; 8];
    for k in 0..8 {
        let mut sum = 0i32;
        for j in (2..32).step_by(4) {
            sum += T32[j][k] * src[j];
        }
        eo[k] = sum;
    }

    let mut eeo = [0i32; 4];
    for k in 0..4 {
        let mut sum = 0i32;
        for j in (4..32).step_by(8) {
            sum += T32[j][k] * src[j];
        }
        eeo[k] = sum;
    }

    let mut eeeo = [0i32; 2];
    let mut eeee = [0i32; 2];
    for k in 0..2 {
        eeeo[k] = T32[8][k] * src[8] + T32[24][k] * src[24];
        eeee[k] = T32[0][k] * src[0] + T32[16][k] * src[16];
    }

    let mut eee = [0i32; 4];
    for k in 0..2 {
        eee[k] = eeee[k] + eeeo[k];
        eee[k + 2] = eeee[1 - k] - eeeo[1 - k];
    }

    let mut ee = [0i32; 8];
    for k in 0..4 {
        ee[k] = eee[k] + eeo[k];
        ee[k + 4] = eee[3 - k] - eeo[3 - k];
    }

    let mut e = [0i32; 16];
    for k in 0..8 {
        e[k] = ee[k] + eo[k];
        e[k + 8] = ee[7 - k] - eo[7 - k];
    }

    for k in 0..16 {
        dst[k] = round_shift_i32(e[k] + o[k], shift).clamp(output_min, output_max);
        dst[k + 16] = round_shift_i32(e[15 - k] - o[15 - k], shift).clamp(output_min, output_max);
    }
}

#[inline]
fn inv_transform_32(
    coeff: &[i32],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
) {
    debug_assert!(coeff.len() >= 32 * 32);
    let shift1 = 7;
    let shift2 = 20 - bit_depth as i32;
    let mut col = [0i32; 32];
    let mut transformed = [0i32; 32];

    for c in 0..32 {
        for r in 0..32 {
            col[r] = coeff[r * 32 + c];
        }
        inv_transform_1d_32(&col, &mut transformed, shift1, -32768, 32767);
        for r in 0..32 {
            tmp[r * 32 + c] = transformed[r];
        }
    }

    for r in 0..32 {
        let src = &tmp[r * 32..r * 32 + 32];
        let dst = &mut out[r * 32..r * 32 + 32];
        inv_transform_1d_32(src, dst, shift2, i32::MIN, i32::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HEVC basis rows are *approximately* orthonormal: they're integer
    /// approximations, so norms cluster tightly around the ideal N·64² and
    /// off-diagonal correlations are small (tens–low hundreds), not zero. A
    /// structurally wrong row (e.g. permuted magnitudes) produces an
    /// off-diagonal dot orders of magnitude larger, which this catches.
    fn check_orthogonal<const N: usize>(t: &[[i32; N]; N]) {
        let ideal = (N as i64) * 64 * 64;
        for (i, row) in t.iter().enumerate() {
            let ni: i64 = row.iter().map(|&v| (v as i64) * (v as i64)).sum();
            assert!(
                (ni - ideal).abs() <= ideal / 500,
                "row {i} norm {ni} too far from ideal {ideal}"
            );
            for j in (i + 1)..N {
                let dot: i64 = (0..N).map(|k| (t[i][k] as i64) * (t[j][k] as i64)).sum();
                assert!(
                    dot.abs() < ideal / 50,
                    "rows {i},{j} insufficiently orthogonal: dot={dot} (limit {})",
                    ideal / 50
                );
            }
        }
    }

    #[test]
    fn t16_is_orthogonal() {
        check_orthogonal(&T16);
        check_orthogonal(&T8);
        check_orthogonal(&T4);
    }

    /// A flat (constant) residual must transform to a DC-only coefficient block:
    /// every AC coefficient is exactly zero. Strong structural check of the
    /// matrix + row/column pass layout for each N.
    fn flat_is_dc_only(n: usize) {
        let c = 100i32;
        let res = vec![c; n * n];
        let coeff = fwd_transform(&res, n, 8);
        assert_ne!(coeff[0], 0, "N={n}: DC should be non-zero");
        for (i, &v) in coeff[..n * n].iter().enumerate().skip(1) {
            assert_eq!(v, 0, "N={n}: AC coeff {i} should be zero, got {v}");
        }
    }

    #[test]
    fn flat_residual_dc_only() {
        flat_is_dc_only(4);
        flat_is_dc_only(8);
        flat_is_dc_only(16);
        flat_is_dc_only(32);
    }

    /// Forward quantization: coeff → level. Returns a fixed 1024-entry buffer.
    fn quantize(coeff: &[i32], n: usize, qp: u8, bit_depth: u8) -> [i16; MAX_TB] {
        quantize_impl(coeff, n, qp, bit_depth, None)
    }

    /// Compatibility wrapper for tests and non-hot callers.
    pub(crate) fn fwd_transform(res: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
        let mut out = [0i32; MAX_TB];
        let mut tmp = [0i32; MAX_TB];
        fwd_transform_into(res, n, bit_depth, &mut out, &mut tmp);
        out
    }

    /// Full pipeline residual → fwd → quant → dequant → inv reconstructs the
    /// residual within the quantization error. At low QP the error is small;
    /// a gross T16/scan/buffer bug would blow this far past tolerance.
    fn roundtrip_bounded(n: usize, qp: u8, max_mean_abs: f32) {
        // Smooth ramp + a mid-frequency component — exercises many coefficients.
        let mut res = vec![0i32; n * n];
        for r in 0..n {
            for col in 0..n {
                let v = 8 * (r as i32 + col as i32) + 20 * (((r + col) % 4) as i32) - 60;
                res[r * n + col] = v;
            }
        }
        let coeff = fwd_transform(&res, n, 8);
        let level = quantize(&coeff[..n * n], n, qp, 8);
        let dq = dequantize(&level[..n * n], n, qp, 8);
        let rec = inv_transform(&dq[..n * n], n, 8);
        let mean_abs: f32 = (0..n * n)
            .map(|i| (res[i] - rec[i]).unsigned_abs() as f32)
            .sum::<f32>()
            / (n * n) as f32;
        assert!(
            mean_abs <= max_mean_abs,
            "N={n} qp={qp}: mean|err|={mean_abs:.3} exceeds {max_mean_abs}"
        );
    }

    pub(crate) fn dequantize(level: &[i16], n: usize, qp: u8, bit_depth: u8) -> [i32; MAX_TB] {
        let mut out = [0i32; MAX_TB];
        dequantize_into(level, n, qp, bit_depth, &mut out);
        out
    }

    pub(crate) fn inv_transform(coeff: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
        let mut out = [0i32; MAX_TB];
        let mut tmp = [0i32; MAX_TB];
        inv_transform_into(coeff, n, bit_depth, &mut out, &mut tmp);
        out
    }

    #[test]
    fn pipeline_roundtrip_low_qp() {
        // Low QP → tight reconstruction for every supported transform size.
        for &n in &[4usize, 8, 16, 32] {
            roundtrip_bounded(n, 4, 3.0);
        }
    }

    #[test]
    fn pipeline_roundtrip_scales_with_qp() {
        // Error grows with QP but stays bounded across all transform sizes.
        for &n in &[4usize, 8, 16, 32] {
            roundtrip_bounded(n, 22, 12.0);
        }
    }

    /// The 16×16 sub-block-major scan must be a permutation of all 256 positions.
    #[test]
    fn zigzag16_is_permutation() {
        let mut seen = [false; 256];
        for &(r, c) in crate::dct::ZIGZAG_16X16.iter() {
            assert!(r < 16 && c < 16);
            let idx = r * 16 + c;
            assert!(!seen[idx], "duplicate scan position ({r},{c})");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b), "scan does not cover all positions");
    }

    #[test]
    fn t32_partial_butterfly_matches_matrix_transform() {
        let mut residual = [0i32; MAX_TB];
        for (i, value) in residual.iter_mut().enumerate() {
            let mixed = (i as i32 * 73 + (i / 32) as i32 * 19) & 511;
            *value = mixed - 256;
        }

        let mut expected = [0i32; MAX_TB];
        let mut tmp = [0i32; MAX_TB];
        fwd_transform_n::<32>(&residual, &T32, 8, &mut expected, &mut tmp);
        let actual = fwd_transform(&residual, 32, 8);
        assert_eq!(actual, expected);
    }

    #[test]
    fn t32_partial_butterfly_matches_sparse_inverse() {
        let mut coeff = [0i32; MAX_TB];
        for i in 0..MAX_TB {
            if i % 7 != 0 {
                coeff[i] = ((i as i32 * 41 + 17) & 1023) - 512;
            }
        }

        let mut expected = [0i32; MAX_TB];
        let mut tmp = [0i32; MAX_TB];
        inv_transform_n::<32>(&coeff, &T32, 10, &mut expected, &mut tmp);
        let actual = inv_transform(&coeff, 32, 10);
        assert_eq!(actual, expected);

        // Exercise the sparse matrix fallback as well as the dense butterfly.
        coeff.fill(0);
        for (index, value) in [(0usize, 1200), (32, -900), (97, 700), (511, -300)] {
            coeff[index] = value;
        }
        inv_transform_n::<32>(&coeff, &T32, 8, &mut expected, &mut tmp);
        let actual = inv_transform(&coeff, 32, 8);
        assert_eq!(actual, expected);
    }

    fn quantize_impl(
        coeff: &[i32],
        n: usize,
        qp: u8,
        bit_depth: u8,
        sign_hiding_scan: Option<&[(usize, usize)]>,
    ) -> [i16; MAX_TB] {
        let mut out = [0i16; MAX_TB];
        quantize_impl_into(coeff, n, qp, bit_depth, sign_hiding_scan, &mut out);
        out
    }

    pub(crate) fn quantize_with_sign_hiding(
        coeff: &[i32],
        n: usize,
        qp: u8,
        bit_depth: u8,
        scan: &[(usize, usize)],
    ) -> [i16; MAX_TB] {
        let mut out = [0i16; MAX_TB];
        quantize_with_sign_hiding_into(coeff, n, qp, bit_depth, scan, &mut out);
        out
    }

    #[test]
    fn post_rdoq_sign_hiding_matches_direct_quantizer() {
        let scan = crate::dct::coeff_scan(3, 0);
        let mut coeff = [0i32; MAX_TB];
        for (i, &(row, col)) in scan.iter().enumerate().take(32) {
            let magnitude = 80 + (i as i32 * 37) % 900;
            coeff[row * 8 + col] = if i % 3 == 0 { -magnitude } else { magnitude };
        }

        let direct = quantize_with_sign_hiding(&coeff, 8, 22, 8, scan);
        let mut split = quantize_impl(&coeff, 8, 22, 8, None);
        apply_sign_hiding_to_levels(&mut split, &coeff, 8, 22, 8, scan);
        assert_eq!(split, direct);
    }

    #[test]
    fn rdoq_zero_block_stays_zero() {
        let coeff = [0i32; MAX_TB];
        let scan = crate::dct::coeff_scan(3, 0);
        let ctx = ContextSet::init_islice(26);
        let levels = rdoq_luma_with_sign_hiding(
            &coeff,
            8,
            26,
            8,
            scan,
            0,
            0.57 * 2f32.powf((26.0 - 12.0) / 3.0),
            &ctx,
        );
        assert!(levels[..64].iter().all(|&level| level == 0));
    }

    pub(crate) fn rdoq_luma_with_sign_hiding(
        coeff: &[i32],
        n: usize,
        qp: u8,
        bit_depth: u8,
        scan: &[(usize, usize)],
        scan_idx: u8,
        lambda: f32,
        ctx: &ContextSet,
    ) -> [i16; MAX_TB] {
        let mut levels = [0i16; MAX_TB];
        let mut scratch = RdoqScratch::new();
        let tb = RdoqTb {
            coeff,
            n,
            qp,
            bit_depth,
            scan,
            scan_idx,
            lambda,
        };
        rdoq_luma_with_sign_hiding_into(&tb, ctx, &mut levels, &mut scratch);
        levels
    }

    #[test]
    fn chroma_rdoq_supports_4x4_and_chroma_contexts() {
        let scan = crate::dct::coeff_scan(2, 0);
        let mut coeff = [0i32; MAX_TB];
        for (i, &(row, col)) in scan.iter().enumerate() {
            let magnitude = 240 + i as i32 * 41;
            coeff[row * 4 + col] = if i % 4 == 0 { -magnitude } else { magnitude };
        }
        let ctx = ContextSet::init_islice(22);
        let mut levels = [0i16; MAX_TB];
        let mut scratch = RdoqScratch::new();
        let tb = RdoqTb {
            coeff: &coeff,
            n: 4,
            qp: 22,
            bit_depth: 8,
            scan,
            scan_idx: 0,
            lambda: 0.57 * 2f32.powf((22.0 - 12.0) / 3.0),
        };
        rdoq_chroma_with_sign_hiding_into(&tb, &ctx, &mut levels, &mut scratch);
        assert!(levels[..16].iter().any(|&level| level != 0));
        for (&level, &source) in levels[..16].iter().zip(&coeff[..16]) {
            assert!(level == 0 || level.signum() as i32 == source.signum());
        }
    }

    #[test]
    fn rdoq_output_obeys_sign_hiding_parity() {
        let scan = crate::dct::coeff_scan(3, 0);
        let mut coeff = [0i32; MAX_TB];
        for (i, &(row, col)) in scan.iter().enumerate().take(24) {
            let magnitude = 600 + i as i32 * 53;
            coeff[row * 8 + col] = if i % 5 == 0 { -magnitude } else { magnitude };
        }
        let ctx = ContextSet::init_islice(4);
        let levels = rdoq_luma_with_sign_hiding(&coeff, 8, 4, 8, scan, 0, 0.001, &ctx);

        for group in scan.chunks_exact(16) {
            let first = group
                .iter()
                .position(|&(row, col)| levels[row * 8 + col] != 0);
            let last = group
                .iter()
                .rposition(|&(row, col)| levels[row * 8 + col] != 0);
            let (Some(first), Some(last)) = (first, last) else {
                continue;
            };
            if last - first < 4 {
                continue;
            }
            let sum: u32 = group[first..=last]
                .iter()
                .map(|&(row, col)| levels[row * 8 + col].unsigned_abs() as u32)
                .sum();
            let (row, col) = group[first];
            assert_eq!((sum & 1) as i32, (levels[row * 8 + col] < 0) as i32);
        }
    }

    #[test]
    fn sign_hiding_fixes_group_parity() {
        let scan = crate::dct::coeff_scan(2, 0);
        let mut levels = [0i16; MAX_TB];
        let mut coeff = [0i32; MAX_TB];

        let (r0, c0) = scan[0];
        let (r4, c4) = scan[4];
        let p0 = r0 * 4 + c0;
        let p4 = r4 * 4 + c4;
        levels[p0] = -1;
        levels[p4] = 3;
        coeff[p0] = -100;
        coeff[p4] = 300;

        sign_bit_hiding_hdq(&mut levels, &coeff, 4, scan, 22, 8, false);

        assert_ne!(levels[p0], 0);
        let parity: u32 = scan[..16]
            .iter()
            .map(|&(row, col)| levels[row * 4 + col].unsigned_abs() as u32)
            .sum();
        assert_eq!((parity & 1) as i32, (levels[p0] < 0) as i32);
    }

    #[test]
    fn sign_hiding_uses_each_coefficient_group_independently() {
        let scan = crate::dct::coeff_scan(3, 0);
        let mut levels = [0i16; MAX_TB];
        let mut coeff = [0i32; MAX_TB];

        for group in 0..4 {
            let first = group * 16;
            let last = first + 4;
            let (r0, c0) = scan[first];
            let (r4, c4) = scan[last];
            let p0 = r0 * 8 + c0;
            let p4 = r4 * 8 + c4;
            levels[p0] = -1;
            levels[p4] = 3;
            coeff[p0] = -100;
            coeff[p4] = 300;
        }

        sign_bit_hiding_hdq(&mut levels, &coeff, 8, scan, 22, 8, false);

        for group in 0..4 {
            let start = group * 16;
            let group_scan = &scan[start..start + 16];
            let first = (0..16)
                .find(|&i| {
                    let (r, c) = group_scan[i];
                    levels[r * 8 + c] != 0
                })
                .unwrap();
            let last = (0..16)
                .rev()
                .find(|&i| {
                    let (r, c) = group_scan[i];
                    levels[r * 8 + c] != 0
                })
                .unwrap();
            assert!(last - first >= 4, "group {group}");
            let sum: u32 = (first..=last)
                .map(|i| {
                    let (r, c) = group_scan[i];
                    levels[r * 8 + c].unsigned_abs() as u32
                })
                .sum();
            let (r, c) = group_scan[first];
            assert_eq!(
                (sum & 1) as i32,
                (levels[r * 8 + c] < 0) as i32,
                "group {group}"
            );
        }
    }

    #[test]
    fn intra_luma_4x4_uses_dst_not_dct() {
        let residual = [
            31, 12, -7, -19, 24, 8, -11, -27, 15, 2, -13, -22, 5, -3, -9, -16,
        ];
        let mut dct = [0i32; MAX_TB];
        let mut dst = [0i32; MAX_TB];
        let mut tmp = [0i32; MAX_TB];
        fwd_transform_into(&residual, 4, 8, &mut dct, &mut tmp);
        fwd_transform_intra_luma_into(&residual, 4, 8, &mut dst, &mut tmp);
        assert_ne!(&dct[..16], &dst[..16]);

        let mut reconstructed = [0i32; MAX_TB];
        inv_transform_intra_luma_into(&dst, 4, 8, &mut reconstructed, &mut tmp);
        assert!(
            reconstructed[..16]
                .iter()
                .zip(residual)
                .all(|(&a, b)| (a - b).abs() <= 1)
        );
    }
}
