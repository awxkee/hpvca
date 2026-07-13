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

use crate::hevc_transform::{DST4, MAX_TB, T4, T8, T16, T32};
use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn fwd_transform_neon(
    res: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
    intra_luma: bool,
) {
    assert!(matches!(n, 4 | 8 | 16 | 32));
    assert!(res.len() >= n * n);
    let matrix = transform_matrix(n, intra_luma);
    let shift1 = n.trailing_zeros() as i32 + bit_depth as i32 - 9;
    let shift2 = n.trailing_zeros() as i32 + 6;

    transpose(res, tmp, n);
    forward_pass(tmp, out, matrix, n, shift1, intra_luma);
    transpose(out, tmp, n);
    forward_pass(tmp, out, matrix, n, shift2, intra_luma);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inv_transform_neon(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
    tmp: &mut [i32; MAX_TB],
    intra_luma: bool,
) {
    assert!(matches!(n, 4 | 8 | 16 | 32));
    assert!(coeff.len() >= n * n);
    let matrix = transform_matrix(n, intra_luma);

    inverse_pass(coeff, tmp, matrix, n, 7, true, intra_luma);
    transpose(tmp, out, n);
    inverse_pass(
        out,
        tmp,
        matrix,
        n,
        20 - bit_depth as i32,
        false,
        intra_luma,
    );
    transpose(tmp, out, n);
}

#[inline]
fn transform_matrix(n: usize, intra_luma: bool) -> &'static [i32] {
    match (n, intra_luma) {
        (4, true) => DST4.as_flattened(),
        (4, false) => T4.as_flattened(),
        (8, _) => T8.as_flattened(),
        (16, _) => T16.as_flattened(),
        (32, _) => T32.as_flattened(),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn round_shift(value: int32x4_t, shift: i32) -> int32x4_t {
    if shift > 0 {
        vshlq_s32(
            vaddq_s32(value, vdupq_n_s32(1 << (shift - 1))),
            vdupq_n_s32(-shift),
        )
    } else {
        value
    }
}

#[target_feature(enable = "neon")]
fn forward_pass(
    src: &[i32; MAX_TB],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    n: usize,
    shift: i32,
    intra_luma: bool,
) {
    match n {
        4 => forward_pass4(src, dst, matrix, shift, intra_luma),
        8 => forward_butterfly::<8>(src, dst, matrix, shift),
        16 => forward_butterfly::<16>(src, dst, matrix, shift),
        32 => forward_butterfly::<32>(src, dst, matrix, shift),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
fn forward_pass4(
    src: &[i32; MAX_TB],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    intra_luma: bool,
) {
    let x = core::array::from_fn::<_, 4, _>(|k| unsafe { vld1q_s32(src.as_ptr().add(k * 4)) });
    let mut output = [vdupq_n_s32(0); 4];
    if intra_luma {
        let c0 = vaddq_s32(x[0], x[3]);
        let c1 = vaddq_s32(x[1], x[3]);
        let c2 = vsubq_s32(x[0], x[1]);
        let c3 = vmulq_n_s32(x[2], 74);
        output[0] = vaddq_s32(vaddq_s32(vmulq_n_s32(c0, 29), vmulq_n_s32(c1, 55)), c3);
        output[1] = vmulq_n_s32(vsubq_s32(vaddq_s32(x[0], x[1]), x[3]), 74);
        output[2] = vsubq_s32(vaddq_s32(vmulq_n_s32(c2, 29), vmulq_n_s32(c0, 55)), c3);
        output[3] = vaddq_s32(vsubq_s32(vmulq_n_s32(c2, 55), vmulq_n_s32(c1, 29)), c3);
    } else {
        let even = [vaddq_s32(x[0], x[3]), vaddq_s32(x[1], x[2])];
        let odd = [vsubq_s32(x[0], x[3]), vsubq_s32(x[1], x[2])];
        for k in [0usize, 2] {
            for j in 0..2 {
                output[k] = vmlaq_n_s32(output[k], even[j], matrix[k * 4 + j]);
            }
        }
        for k in [1usize, 3] {
            for j in 0..2 {
                output[k] = vmlaq_n_s32(output[k], odd[j], matrix[k * 4 + j]);
            }
        }
    }
    for (k, value) in output.into_iter().enumerate() {
        unsafe { vst1q_s32(dst.as_mut_ptr().add(k * 4), round_shift(value, shift)) };
    }
}

#[target_feature(enable = "neon")]
fn forward_butterfly<const N: usize>(
    src: &[i32; MAX_TB],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
) {
    let mut x = [vdupq_n_s32(0); 32];
    let mut e = [vdupq_n_s32(0); 16];
    let mut o = [vdupq_n_s32(0); 16];
    let mut ee = [vdupq_n_s32(0); 8];
    let mut eo = [vdupq_n_s32(0); 8];
    let mut eee = [vdupq_n_s32(0); 4];
    let mut eeo = [vdupq_n_s32(0); 4];
    let mut eeee = [vdupq_n_s32(0); 2];
    let mut eeeo = [vdupq_n_s32(0); 2];
    let mut output = [vdupq_n_s32(0); 32];

    for column in (0..N).step_by(4) {
        for (k, value) in x[..N].iter_mut().enumerate() {
            *value = unsafe { vld1q_s32(src.as_ptr().add(k * N + column)) };
        }
        for k in 0..N / 2 {
            e[k] = vaddq_s32(x[k], x[N - 1 - k]);
            o[k] = vsubq_s32(x[k], x[N - 1 - k]);
        }
        for k in 0..N / 4 {
            ee[k] = vaddq_s32(e[k], e[N / 2 - 1 - k]);
            eo[k] = vsubq_s32(e[k], e[N / 2 - 1 - k]);
        }
        if N >= 16 {
            for k in 0..N / 8 {
                eee[k] = vaddq_s32(ee[k], ee[N / 4 - 1 - k]);
                eeo[k] = vsubq_s32(ee[k], ee[N / 4 - 1 - k]);
            }
        }
        if N == 32 {
            for k in 0..2 {
                eeee[k] = vaddq_s32(eee[k], eee[3 - k]);
                eeeo[k] = vsubq_s32(eee[k], eee[3 - k]);
            }
        }
        output.fill(vdupq_n_s32(0));
        accumulate_frequencies::<N>(&mut output, &o, matrix, 1, 2, N / 2);
        accumulate_frequencies::<N>(&mut output, &eo, matrix, 2, 4, N / 4);
        match N {
            8 => accumulate_frequencies::<N>(&mut output, &ee, matrix, 0, 4, 2),
            16 => {
                accumulate_frequencies::<N>(&mut output, &eeo, matrix, 4, 8, 2);
                accumulate_frequencies::<N>(&mut output, &eee, matrix, 0, 8, 2);
            }
            32 => {
                accumulate_frequencies::<N>(&mut output, &eeo, matrix, 4, 8, 4);
                accumulate_frequencies::<N>(&mut output, &eeeo, matrix, 8, 16, 2);
                accumulate_frequencies::<N>(&mut output, &eeee, matrix, 0, 16, 2);
            }
            _ => unreachable!(),
        }
        for (k, value) in output[..N].iter().enumerate() {
            unsafe {
                vst1q_s32(
                    dst.as_mut_ptr().add(k * N + column),
                    round_shift(*value, shift),
                )
            };
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_frequencies<const N: usize>(
    output: &mut [int32x4_t; 32],
    values: &[int32x4_t],
    matrix: &[i32],
    first: usize,
    step: usize,
    terms: usize,
) {
    for k in (first..N).step_by(step) {
        for j in 0..terms {
            output[k] = vmlaq_n_s32(output[k], values[j], matrix[k * N + j]);
        }
    }
}

#[target_feature(enable = "neon")]
fn inverse_pass(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    n: usize,
    shift: i32,
    clip: bool,
    intra_luma: bool,
) {
    match n {
        4 => inverse_pass4(src, dst, matrix, shift, clip, intra_luma),
        8 => inverse_butterfly8(src, dst, matrix, shift, clip),
        16 => inverse_butterfly16(src, dst, matrix, shift, clip),
        32 => inverse_butterfly32(src, dst, matrix, shift, clip),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
fn inverse_pass4(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
    intra_luma: bool,
) {
    let x = core::array::from_fn::<_, 4, _>(|k| unsafe { vld1q_s32(src.as_ptr().add(k * 4)) });
    let output = if intra_luma {
        let c0 = vaddq_s32(x[0], x[2]);
        let c1 = vaddq_s32(x[2], x[3]);
        let c2 = vsubq_s32(x[0], x[3]);
        let c3 = vmulq_n_s32(x[1], 74);
        [
            vaddq_s32(vaddq_s32(vmulq_n_s32(c0, 29), vmulq_n_s32(c1, 55)), c3),
            vsubq_s32(vaddq_s32(vmulq_n_s32(c2, 55), c3), vmulq_n_s32(c1, 29)),
            vmulq_n_s32(vaddq_s32(vsubq_s32(x[0], x[2]), x[3]), 74),
            vsubq_s32(vaddq_s32(vmulq_n_s32(c0, 55), vmulq_n_s32(c2, 29)), c3),
        ]
    } else {
        let even: [int32x4_t; 2] = core::array::from_fn(|k| {
            vaddq_s32(
                vmulq_n_s32(x[0], matrix[k]),
                vmulq_n_s32(x[2], matrix[8 + k]),
            )
        });
        let odd: [int32x4_t; 2] = core::array::from_fn(|k| {
            vaddq_s32(
                vmulq_n_s32(x[1], matrix[4 + k]),
                vmulq_n_s32(x[3], matrix[12 + k]),
            )
        });
        [
            vaddq_s32(even[0], odd[0]),
            vaddq_s32(even[1], odd[1]),
            vsubq_s32(even[1], odd[1]),
            vsubq_s32(even[0], odd[0]),
        ]
    };
    for (k, value) in output.into_iter().enumerate() {
        let mut result = round_shift(value, shift);
        if clip {
            result = vminq_s32(vmaxq_s32(result, vdupq_n_s32(-32768)), vdupq_n_s32(32767));
        }
        unsafe { vst1q_s32(dst.as_mut_ptr().add(k * 4), result) };
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inverse_dot<const N: usize>(
    input: &[int32x4_t; N],
    matrix: &[i32],
    output: usize,
    first: usize,
    step: usize,
) -> int32x4_t {
    let mut sum = vdupq_n_s32(0);
    for frequency in (first..N).step_by(step) {
        sum = vmlaq_n_s32(sum, input[frequency], matrix[frequency * N + output]);
    }
    sum
}

#[inline]
#[target_feature(enable = "neon")]
fn store_inverse<const N: usize>(
    dst: &mut [i32; MAX_TB],
    column: usize,
    output: &[int32x4_t; N],
    shift: i32,
    clip: bool,
) {
    for (row, value) in output.iter().enumerate() {
        let mut result = round_shift(*value, shift);
        if clip {
            result = vminq_s32(vmaxq_s32(result, vdupq_n_s32(-32768)), vdupq_n_s32(32767));
        }
        unsafe { vst1q_s32(dst.as_mut_ptr().add(row * N + column), result) };
    }
}

#[target_feature(enable = "neon")]
fn inverse_butterfly8(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
) {
    for column in (0..8).step_by(4) {
        let input: [int32x4_t; 8] =
            core::array::from_fn(|k| unsafe { vld1q_s32(src.as_ptr().add(k * 8 + column)) });
        let odd: [int32x4_t; 4] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 1, 2));
        let eo: [int32x4_t; 2] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 2, 4));
        let ee: [int32x4_t; 2] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 0, 4));
        let even = [
            vaddq_s32(ee[0], eo[0]),
            vaddq_s32(ee[1], eo[1]),
            vsubq_s32(ee[1], eo[1]),
            vsubq_s32(ee[0], eo[0]),
        ];
        let output: [int32x4_t; 8] = core::array::from_fn(|k| {
            if k < 4 {
                vaddq_s32(even[k], odd[k])
            } else {
                vsubq_s32(even[7 - k], odd[7 - k])
            }
        });
        store_inverse(dst, column, &output, shift, clip);
    }
}

#[target_feature(enable = "neon")]
fn inverse_butterfly16(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
) {
    for column in (0..16).step_by(4) {
        let input: [int32x4_t; 16] =
            core::array::from_fn(|k| unsafe { vld1q_s32(src.as_ptr().add(k * 16 + column)) });
        let odd: [int32x4_t; 8] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 1, 2));
        let eo: [int32x4_t; 4] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 2, 4));
        let eeo: [int32x4_t; 2] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 4, 8));
        let eee: [int32x4_t; 2] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 0, 8));
        let ee = [
            vaddq_s32(eee[0], eeo[0]),
            vaddq_s32(eee[1], eeo[1]),
            vsubq_s32(eee[1], eeo[1]),
            vsubq_s32(eee[0], eeo[0]),
        ];
        let even: [int32x4_t; 8] = core::array::from_fn(|k| {
            if k < 4 {
                vaddq_s32(ee[k], eo[k])
            } else {
                vsubq_s32(ee[7 - k], eo[7 - k])
            }
        });
        let output: [int32x4_t; 16] = core::array::from_fn(|k| {
            if k < 8 {
                vaddq_s32(even[k], odd[k])
            } else {
                vsubq_s32(even[15 - k], odd[15 - k])
            }
        });
        store_inverse(dst, column, &output, shift, clip);
    }
}

#[target_feature(enable = "neon")]
fn inverse_butterfly32(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
) {
    for column in (0..32).step_by(4) {
        let input: [int32x4_t; 32] =
            core::array::from_fn(|k| unsafe { vld1q_s32(src.as_ptr().add(k * 32 + column)) });
        let odd: [int32x4_t; 16] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 1, 2));
        let eo: [int32x4_t; 8] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 2, 4));
        let eeo: [int32x4_t; 4] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 4, 8));
        let eeeo: [int32x4_t; 2] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 8, 16));
        let eeee: [int32x4_t; 2] = core::array::from_fn(|k| inverse_dot(&input, matrix, k, 0, 16));
        let eee = [
            vaddq_s32(eeee[0], eeeo[0]),
            vaddq_s32(eeee[1], eeeo[1]),
            vsubq_s32(eeee[1], eeeo[1]),
            vsubq_s32(eeee[0], eeeo[0]),
        ];
        let ee: [int32x4_t; 8] = core::array::from_fn(|k| {
            if k < 4 {
                vaddq_s32(eee[k], eeo[k])
            } else {
                vsubq_s32(eee[7 - k], eeo[7 - k])
            }
        });
        let even: [int32x4_t; 16] = core::array::from_fn(|k| {
            if k < 8 {
                vaddq_s32(ee[k], eo[k])
            } else {
                vsubq_s32(ee[15 - k], eo[15 - k])
            }
        });
        let output: [int32x4_t; 32] = core::array::from_fn(|k| {
            if k < 16 {
                vaddq_s32(even[k], odd[k])
            } else {
                vsubq_s32(even[31 - k], odd[31 - k])
            }
        });
        store_inverse(dst, column, &output, shift, clip);
    }
}

#[target_feature(enable = "neon")]
fn transpose(src: &[i32], dst: &mut [i32; MAX_TB], n: usize) {
    for row in (0..n).step_by(4) {
        for column in (0..n).step_by(4) {
            let r0 = unsafe { vld1q_s32(src.as_ptr().add(row * n + column)) };
            let r1 = unsafe { vld1q_s32(src.as_ptr().add((row + 1) * n + column)) };
            let r2 = unsafe { vld1q_s32(src.as_ptr().add((row + 2) * n + column)) };
            let r3 = unsafe { vld1q_s32(src.as_ptr().add((row + 3) * n + column)) };
            let t0 = vtrnq_s32(r0, r1);
            let t1 = vtrnq_s32(r2, r3);
            let columns = [
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
            ];
            for (offset, values) in columns.into_iter().enumerate() {
                unsafe { vst1q_s32(dst.as_mut_ptr().add((column + offset) * n + row), values) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_transform_neon_matches_scalar() {
        compare(false);
        compare(true);
    }

    #[test]
    fn inverse_transform_neon_matches_scalar() {
        compare_inverse(false);
        compare_inverse(true);
    }

    fn compare(intra_luma: bool) {
        let mut residual = [0i32; MAX_TB];
        for (i, value) in residual.iter_mut().enumerate() {
            *value = ((i * 127 + i / 3 * 29 + 17) as i32 & 8191) - 4095;
        }
        for bit_depth in [8, 10, 12] {
            for n in [4, 8, 16, 32] {
                let mut expected = [0i32; MAX_TB];
                let mut expected_tmp = [0i32; MAX_TB];
                let mut actual = [0i32; MAX_TB];
                let mut actual_tmp = [0i32; MAX_TB];
                unsafe {
                    crate::hevc_transform::fwd_transform_scalar(
                        &residual,
                        n,
                        bit_depth,
                        &mut expected,
                        &mut expected_tmp,
                        intra_luma,
                    );
                    fwd_transform_neon(
                        &residual,
                        n,
                        bit_depth,
                        &mut actual,
                        &mut actual_tmp,
                        intra_luma,
                    );
                }
                assert_eq!(
                    &actual[..n * n],
                    &expected[..n * n],
                    "n={n}, depth={bit_depth}"
                );
            }
        }
    }

    fn compare_inverse(intra_luma: bool) {
        let mut coefficient = [0i32; MAX_TB];
        for (i, value) in coefficient.iter_mut().enumerate() {
            *value = ((i * 509 + i / 7 * 131 + 29) as i32 & 65535) - 32768;
        }
        for bit_depth in [8, 10, 12] {
            for n in [4, 8, 16, 32] {
                let mut expected = [0i32; MAX_TB];
                let mut expected_tmp = [0i32; MAX_TB];
                let mut actual = [0i32; MAX_TB];
                let mut actual_tmp = [0i32; MAX_TB];
                unsafe {
                    crate::hevc_transform::inv_transform_scalar(
                        &coefficient,
                        n,
                        bit_depth,
                        &mut expected,
                        &mut expected_tmp,
                        intra_luma,
                    );
                    inv_transform_neon(
                        &coefficient,
                        n,
                        bit_depth,
                        &mut actual,
                        &mut actual_tmp,
                        intra_luma,
                    );
                }
                assert_eq!(
                    &actual[..n * n],
                    &expected[..n * n],
                    "n={n}, depth={bit_depth}"
                );
            }
        }
    }
}
