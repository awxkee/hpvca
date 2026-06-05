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

/// 4×4 HEVC transform matrix.
static T4: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// 8×8 HEVC transform matrix.
static T8: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

static QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
static DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Forward integer transform of an N×N residual block (N = 4 or 8).
pub(crate) fn fwd_transform(res: &[i32], n: usize, bit_depth: u8) -> Vec<i64> {
    match n {
        4 => fwd_transform_n::<4>(res, &T4, bit_depth),
        8 => fwd_transform_n::<8>(res, &T8, bit_depth),
        _ => panic!("unsupported transform size {n}"),
    }
}

#[inline]
fn fwd_transform_n<const N: usize>(res: &[i32], t: &[[i32; N]; N], bit_depth: u8) -> Vec<i64> {
    let log2n = N.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let shift1 = log2n + bd - 9;
    let add1 = if shift1 > 0 { 1i64 << (shift1 - 1) } else { 0 };
    let mut tmp = [0i64; 64]; // N*N <= 64
    // pass 1 (rows of res): tmp[j*N+i] = (sum_k T[i][k]*res[j*N+k]) >> shift1
    for (j, res_row) in res.chunks_exact(N).enumerate().take(N) {
        for (i, trow) in t.iter().enumerate() {
            let s: i64 = trow
                .iter()
                .zip(res_row)
                .map(|(&a, &b)| a as i64 * b as i64)
                .sum();
            tmp[j * N + i] = (s + add1) >> shift1;
        }
    }
    // pass 2 (columns): coeff[i*N+j] = (sum_k T[i][k]*tmp[k*N+j]) >> shift2
    let shift2 = log2n + 6;
    let add2 = 1i64 << (shift2 - 1);
    let mut coeff = vec![0i64; N * N];
    let mut colv = [0i64; N];
    for j in 0..N {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = tmp[k * N + j];
        }
        for (i, trow) in t.iter().enumerate() {
            let s: i64 = trow.iter().zip(&colv).map(|(&a, &b)| a as i64 * b).sum();
            coeff[i * N + j] = (s + add2) >> shift2;
        }
    }
    coeff
}

/// Forward quantisation: coeff → level. Intra rounding offset.
pub(crate) fn quantize(coeff: &[i64], n: usize, qp: u8, bit_depth: u8) -> Vec<i16> {
    let log2n = if n == 8 { 3 } else { 2 } as i64;
    let bd = bit_depth as i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bd - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra
    coeff
        .iter()
        .map(|&c| {
            let level = (c.abs() * q_scale + offset) >> q_bits;
            let level = if c < 0 { -level } else { level };
            level.clamp(-32768, 32767) as i16
        })
        .collect()
}

/// Dequantisation: level → transform coefficient (spec 8.6.3).
pub(crate) fn dequantize(level: &[i16], n: usize, qp: u8, bit_depth: u8) -> Vec<i64> {
    let log2n = if n == 8 { 3 } else { 2 } as i64;
    let bd = bit_depth as i64;
    let bd_shift = bd + log2n - 5;
    let add = 1i64 << (bd_shift - 1);
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let factor = scale * per * 16;
    level
        .iter()
        .map(|&l| ((l as i64 * factor + add) >> bd_shift).clamp(-32768, 32767))
        .collect()
}

/// Inverse integer transform (spec 8.6.4.2). Returns residual.
pub(crate) fn inv_transform(coeff: &[i64], n: usize, bit_depth: u8) -> Vec<i32> {
    match n {
        4 => inv_transform_n::<4>(coeff, &T4, bit_depth),
        8 => inv_transform_n::<8>(coeff, &T8, bit_depth),
        _ => panic!("unsupported transform size {n}"),
    }
}

#[inline]
fn inv_transform_n<const N: usize>(coeff: &[i64], t: &[[i32; N]; N], bit_depth: u8) -> Vec<i32> {
    let bd = bit_depth as i64;
    // Stage 1 (columns): tmp[m*N+c] = clip(sum_k T[k][m]*coeff[k*N+c]) >> 7
    let shift1 = 7i64;
    let add1 = 1i64 << (shift1 - 1);
    let mut tmp = [0i64; 64];
    let mut colv = [0i64; N];
    for c in 0..N {
        for (k, cv) in colv.iter_mut().enumerate() {
            *cv = coeff[k * N + c];
        }
        for m in 0..N {
            // column m of T: T[k][m]
            let s: i64 = t
                .iter()
                .zip(&colv)
                .map(|(trow, &v)| trow[m] as i64 * v)
                .sum();
            tmp[m * N + c] = ((s + add1) >> shift1).clamp(-32768, 32767);
        }
    }
    // Stage 2 (rows): out[r*N+m] = (sum_k T[k][m]*tmp[r*N+k]) >> (20-bd)
    let shift2 = 20 - bd;
    let add2 = 1i64 << (shift2 - 1);
    let mut out = vec![0i32; N * N];
    for r in 0..N {
        let rowv = &tmp[r * N..r * N + N];
        for m in 0..N {
            let s: i64 = t
                .iter()
                .zip(rowv)
                .map(|(trow, &v)| trow[m] as i64 * v)
                .sum();
            out[r * N + m] = ((s + add2) >> shift2) as i32;
        }
    }
    out
}
