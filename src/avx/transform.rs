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
use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn fwd_transform_avx2(
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

#[target_feature(enable = "avx2")]
pub(crate) fn inv_transform_avx2(
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
#[target_feature(enable = "avx2")]
fn round_shift4(value: __m128i, shift: i32) -> __m128i {
    if shift > 0 {
        _mm_srav_epi32(
            _mm_add_epi32(value, _mm_set1_epi32(1 << (shift - 1))),
            _mm_set1_epi32(shift),
        )
    } else {
        value
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_shift8(value: __m256i, shift: i32) -> __m256i {
    if shift > 0 {
        _mm256_srav_epi32(
            _mm256_add_epi32(value, _mm256_set1_epi32(1 << (shift - 1))),
            _mm256_set1_epi32(shift),
        )
    } else {
        value
    }
}

#[target_feature(enable = "avx2")]
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
        8 => forward_butterfly8::<8>(src, dst, matrix, shift),
        16 => forward_butterfly8::<16>(src, dst, matrix, shift),
        32 => forward_butterfly8::<32>(src, dst, matrix, shift),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
fn forward_pass4(
    src: &[i32; MAX_TB],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    intra_luma: bool,
) {
    let x = core::array::from_fn::<_, 4, _>(|k| unsafe {
        _mm_loadu_si128(src.as_ptr().add(k * 4).cast())
    });
    let mut output = [_mm_setzero_si128(); 4];
    if intra_luma {
        let c0 = _mm_add_epi32(x[0], x[3]);
        let c1 = _mm_add_epi32(x[1], x[3]);
        let c2 = _mm_sub_epi32(x[0], x[1]);
        let c3 = _mm_mullo_epi32(x[2], _mm_set1_epi32(74));
        output[0] = _mm_add_epi32(
            _mm_add_epi32(
                _mm_mullo_epi32(c0, _mm_set1_epi32(29)),
                _mm_mullo_epi32(c1, _mm_set1_epi32(55)),
            ),
            c3,
        );
        output[1] = _mm_mullo_epi32(
            _mm_sub_epi32(_mm_add_epi32(x[0], x[1]), x[3]),
            _mm_set1_epi32(74),
        );
        output[2] = _mm_sub_epi32(
            _mm_add_epi32(
                _mm_mullo_epi32(c2, _mm_set1_epi32(29)),
                _mm_mullo_epi32(c0, _mm_set1_epi32(55)),
            ),
            c3,
        );
        output[3] = _mm_add_epi32(
            _mm_sub_epi32(
                _mm_mullo_epi32(c2, _mm_set1_epi32(55)),
                _mm_mullo_epi32(c1, _mm_set1_epi32(29)),
            ),
            c3,
        );
    } else {
        let even = [_mm_add_epi32(x[0], x[3]), _mm_add_epi32(x[1], x[2])];
        let odd = [_mm_sub_epi32(x[0], x[3]), _mm_sub_epi32(x[1], x[2])];
        for k in [0usize, 2] {
            for j in 0..2 {
                output[k] = _mm_add_epi32(
                    output[k],
                    _mm_mullo_epi32(even[j], _mm_set1_epi32(matrix[k * 4 + j])),
                );
            }
        }
        for k in [1usize, 3] {
            for j in 0..2 {
                output[k] = _mm_add_epi32(
                    output[k],
                    _mm_mullo_epi32(odd[j], _mm_set1_epi32(matrix[k * 4 + j])),
                );
            }
        }
    }
    for (k, value) in output.into_iter().enumerate() {
        unsafe {
            _mm_storeu_si128(
                dst.as_mut_ptr().add(k * 4).cast(),
                round_shift4(value, shift),
            )
        };
    }
}

#[target_feature(enable = "avx2")]
fn forward_butterfly8<const N: usize>(
    src: &[i32; MAX_TB],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
) {
    let mut x = [_mm256_setzero_si256(); 32];
    let mut e = [_mm256_setzero_si256(); 16];
    let mut o = [_mm256_setzero_si256(); 16];
    let mut ee = [_mm256_setzero_si256(); 8];
    let mut eo = [_mm256_setzero_si256(); 8];
    let mut eee = [_mm256_setzero_si256(); 4];
    let mut eeo = [_mm256_setzero_si256(); 4];
    let mut eeee = [_mm256_setzero_si256(); 2];
    let mut eeeo = [_mm256_setzero_si256(); 2];
    let mut output = [_mm256_setzero_si256(); 32];

    for column in (0..N).step_by(8) {
        for (k, value) in x[..N].iter_mut().enumerate() {
            *value = unsafe { _mm256_loadu_si256(src.as_ptr().add(k * N + column).cast()) };
        }
        for k in 0..N / 2 {
            e[k] = _mm256_add_epi32(x[k], x[N - 1 - k]);
            o[k] = _mm256_sub_epi32(x[k], x[N - 1 - k]);
        }
        for k in 0..N / 4 {
            ee[k] = _mm256_add_epi32(e[k], e[N / 2 - 1 - k]);
            eo[k] = _mm256_sub_epi32(e[k], e[N / 2 - 1 - k]);
        }
        if N >= 16 {
            for k in 0..N / 8 {
                eee[k] = _mm256_add_epi32(ee[k], ee[N / 4 - 1 - k]);
                eeo[k] = _mm256_sub_epi32(ee[k], ee[N / 4 - 1 - k]);
            }
        }
        if N == 32 {
            for k in 0..2 {
                eeee[k] = _mm256_add_epi32(eee[k], eee[3 - k]);
                eeeo[k] = _mm256_sub_epi32(eee[k], eee[3 - k]);
            }
        }
        output.fill(_mm256_setzero_si256());
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
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(k * N + column).cast(),
                    round_shift8(*value, shift),
                )
            };
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn accumulate_frequencies<const N: usize>(
    output: &mut [__m256i; 32],
    values: &[__m256i],
    matrix: &[i32],
    first: usize,
    step: usize,
    terms: usize,
) {
    for k in (first..N).step_by(step) {
        for j in 0..terms {
            output[k] = _mm256_add_epi32(
                output[k],
                _mm256_mullo_epi32(values[j], _mm256_set1_epi32(matrix[k * N + j])),
            );
        }
    }
}

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
fn inverse_pass4(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
    intra_luma: bool,
) {
    let x = core::array::from_fn::<_, 4, _>(|k| unsafe {
        _mm_loadu_si128(src.as_ptr().add(k * 4).cast())
    });
    let output = if intra_luma {
        let c0 = _mm_add_epi32(x[0], x[2]);
        let c1 = _mm_add_epi32(x[2], x[3]);
        let c2 = _mm_sub_epi32(x[0], x[3]);
        let c3 = _mm_mullo_epi32(x[1], _mm_set1_epi32(74));
        [
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(c0, _mm_set1_epi32(29)),
                    _mm_mullo_epi32(c1, _mm_set1_epi32(55)),
                ),
                c3,
            ),
            _mm_sub_epi32(
                _mm_add_epi32(_mm_mullo_epi32(c2, _mm_set1_epi32(55)), c3),
                _mm_mullo_epi32(c1, _mm_set1_epi32(29)),
            ),
            _mm_mullo_epi32(
                _mm_add_epi32(_mm_sub_epi32(x[0], x[2]), x[3]),
                _mm_set1_epi32(74),
            ),
            _mm_sub_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(c0, _mm_set1_epi32(55)),
                    _mm_mullo_epi32(c2, _mm_set1_epi32(29)),
                ),
                c3,
            ),
        ]
    } else {
        let even: [__m128i; 2] = core::array::from_fn(|k| {
            _mm_add_epi32(
                _mm_mullo_epi32(x[0], _mm_set1_epi32(matrix[k])),
                _mm_mullo_epi32(x[2], _mm_set1_epi32(matrix[8 + k])),
            )
        });
        let odd: [__m128i; 2] = core::array::from_fn(|k| {
            _mm_add_epi32(
                _mm_mullo_epi32(x[1], _mm_set1_epi32(matrix[4 + k])),
                _mm_mullo_epi32(x[3], _mm_set1_epi32(matrix[12 + k])),
            )
        });
        [
            _mm_add_epi32(even[0], odd[0]),
            _mm_add_epi32(even[1], odd[1]),
            _mm_sub_epi32(even[1], odd[1]),
            _mm_sub_epi32(even[0], odd[0]),
        ]
    };
    for (k, value) in output.into_iter().enumerate() {
        let mut result = round_shift4(value, shift);
        if clip {
            result = _mm_min_epi32(
                _mm_max_epi32(result, _mm_set1_epi32(-32768)),
                _mm_set1_epi32(32767),
            );
        }
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(k * 4).cast(), result) };
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn inverse_dot8<const N: usize>(
    input: &[__m256i; N],
    matrix: &[i32],
    output: usize,
    first: usize,
    step: usize,
) -> __m256i {
    let mut sum = _mm256_setzero_si256();
    for frequency in (first..N).step_by(step) {
        sum = _mm256_add_epi32(
            sum,
            _mm256_mullo_epi32(
                input[frequency],
                _mm256_set1_epi32(matrix[frequency * N + output]),
            ),
        );
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_inverse8<const N: usize>(
    dst: &mut [i32; MAX_TB],
    column: usize,
    output: &[__m256i; N],
    shift: i32,
    clip: bool,
) {
    for (row, value) in output.iter().enumerate() {
        let mut result = round_shift8(*value, shift);
        if clip {
            result = _mm256_min_epi32(
                _mm256_max_epi32(result, _mm256_set1_epi32(-32768)),
                _mm256_set1_epi32(32767),
            );
        }
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(row * N + column).cast(), result) };
    }
}

#[target_feature(enable = "avx2")]
fn inverse_butterfly8(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
) {
    for column in (0..8).step_by(8) {
        let input: [__m256i; 8] = core::array::from_fn(|k| unsafe {
            _mm256_loadu_si256(src.as_ptr().add(k * 8 + column).cast())
        });
        let odd: [__m256i; 4] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 1, 2));
        let eo: [__m256i; 2] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 2, 4));
        let ee: [__m256i; 2] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 0, 4));
        let even = [
            _mm256_add_epi32(ee[0], eo[0]),
            _mm256_add_epi32(ee[1], eo[1]),
            _mm256_sub_epi32(ee[1], eo[1]),
            _mm256_sub_epi32(ee[0], eo[0]),
        ];
        let output: [__m256i; 8] = core::array::from_fn(|k| {
            if k < 4 {
                _mm256_add_epi32(even[k], odd[k])
            } else {
                _mm256_sub_epi32(even[7 - k], odd[7 - k])
            }
        });
        store_inverse8(dst, column, &output, shift, clip);
    }
}

#[target_feature(enable = "avx2")]
fn inverse_butterfly16(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
) {
    for column in (0..16).step_by(8) {
        let input: [__m256i; 16] = core::array::from_fn(|k| unsafe {
            _mm256_loadu_si256(src.as_ptr().add(k * 16 + column).cast())
        });
        let odd: [__m256i; 8] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 1, 2));
        let eo: [__m256i; 4] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 2, 4));
        let eeo: [__m256i; 2] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 4, 8));
        let eee: [__m256i; 2] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 0, 8));
        let ee = [
            _mm256_add_epi32(eee[0], eeo[0]),
            _mm256_add_epi32(eee[1], eeo[1]),
            _mm256_sub_epi32(eee[1], eeo[1]),
            _mm256_sub_epi32(eee[0], eeo[0]),
        ];
        let even: [__m256i; 8] = core::array::from_fn(|k| {
            if k < 4 {
                _mm256_add_epi32(ee[k], eo[k])
            } else {
                _mm256_sub_epi32(ee[7 - k], eo[7 - k])
            }
        });
        let output: [__m256i; 16] = core::array::from_fn(|k| {
            if k < 8 {
                _mm256_add_epi32(even[k], odd[k])
            } else {
                _mm256_sub_epi32(even[15 - k], odd[15 - k])
            }
        });
        store_inverse8(dst, column, &output, shift, clip);
    }
}

#[target_feature(enable = "avx2")]
fn inverse_butterfly32(
    src: &[i32],
    dst: &mut [i32; MAX_TB],
    matrix: &[i32],
    shift: i32,
    clip: bool,
) {
    for column in (0..32).step_by(8) {
        let input: [__m256i; 32] = core::array::from_fn(|k| unsafe {
            _mm256_loadu_si256(src.as_ptr().add(k * 32 + column).cast())
        });
        let odd: [__m256i; 16] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 1, 2));
        let eo: [__m256i; 8] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 2, 4));
        let eeo: [__m256i; 4] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 4, 8));
        let eeeo: [__m256i; 2] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 8, 16));
        let eeee: [__m256i; 2] = core::array::from_fn(|k| inverse_dot8(&input, matrix, k, 0, 16));
        let eee = [
            _mm256_add_epi32(eeee[0], eeeo[0]),
            _mm256_add_epi32(eeee[1], eeeo[1]),
            _mm256_sub_epi32(eeee[1], eeeo[1]),
            _mm256_sub_epi32(eeee[0], eeeo[0]),
        ];
        let ee: [__m256i; 8] = core::array::from_fn(|k| {
            if k < 4 {
                _mm256_add_epi32(eee[k], eeo[k])
            } else {
                _mm256_sub_epi32(eee[7 - k], eeo[7 - k])
            }
        });
        let even: [__m256i; 16] = core::array::from_fn(|k| {
            if k < 8 {
                _mm256_add_epi32(ee[k], eo[k])
            } else {
                _mm256_sub_epi32(ee[15 - k], eo[15 - k])
            }
        });
        let output: [__m256i; 32] = core::array::from_fn(|k| {
            if k < 16 {
                _mm256_add_epi32(even[k], odd[k])
            } else {
                _mm256_sub_epi32(even[31 - k], odd[31 - k])
            }
        });
        store_inverse8(dst, column, &output, shift, clip);
    }
}

#[target_feature(enable = "avx2")]
fn transpose(src: &[i32], dst: &mut [i32; MAX_TB], n: usize) {
    for row in (0..n).step_by(4) {
        for column in (0..n).step_by(4) {
            let r0 = unsafe { _mm_loadu_si128(src.as_ptr().add(row * n + column).cast()) };
            let r1 = unsafe { _mm_loadu_si128(src.as_ptr().add((row + 1) * n + column).cast()) };
            let r2 = unsafe { _mm_loadu_si128(src.as_ptr().add((row + 2) * n + column).cast()) };
            let r3 = unsafe { _mm_loadu_si128(src.as_ptr().add((row + 3) * n + column).cast()) };
            let t0 = _mm_unpacklo_epi32(r0, r1);
            let t1 = _mm_unpackhi_epi32(r0, r1);
            let t2 = _mm_unpacklo_epi32(r2, r3);
            let t3 = _mm_unpackhi_epi32(r2, r3);
            let columns = [
                _mm_unpacklo_epi64(t0, t2),
                _mm_unpackhi_epi64(t0, t2),
                _mm_unpacklo_epi64(t1, t3),
                _mm_unpackhi_epi64(t1, t3),
            ];
            for (offset, values) in columns.into_iter().enumerate() {
                unsafe {
                    _mm_storeu_si128(
                        dst.as_mut_ptr().add((column + offset) * n + row).cast(),
                        values,
                    )
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_transform_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        compare(false);
        compare(true);
    }

    #[test]
    fn inverse_transform_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
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
                    fwd_transform_avx2(
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
                    inv_transform_avx2(
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
