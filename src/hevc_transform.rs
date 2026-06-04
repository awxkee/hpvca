//! Spec-faithful HEVC integer transform, quantization, and dequantization.
//!
//! Replaces the previous float-DCT approach. The float orthonormal DCT produced
//! coefficients in a different magnitude domain than HEVC's integer transform,
//! so a conforming decoder (e.g. ffmpeg) reconstructed wrong values and, for
//! high-energy blocks, the quantized levels could be large enough to make
//! `coeff_abs_level_remaining` emit illegal-length codes (CABAC_MAX_BIN).
//!
//! Using the exact integer transform makes the encoded coefficients land in the
//! same domain the decoder inverts, so encoder reconstruction == decoder output.

/// 4×4 HEVC transform matrix.
const T4: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// 8×8 HEVC transform matrix.
const T8: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
const DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

#[inline]
fn row(matrix_row: &[i32], v: &[i64], n: usize) -> i64 {
    let mut s = 0i64;
    for i in 0..n {
        s += matrix_row[i] as i64 * v[i];
    }
    s
}

fn t_row(n: usize, i: usize, j: usize) -> i64 {
    (if n == 8 { T8[i][j] } else { T4[i][j] }) as i64
}

/// Forward integer transform of an N×N residual block (N = 4 or 8), 8-bit.
/// Returns transform coefficients (pre-quantisation).
pub fn fwd_transform(res: &[i32], n: usize) -> Vec<i64> {
    let log2n = if n == 8 { 3 } else { 2 };
    let bd = 8i64;
    // Stage 1 (rows): tmp = T * res  (apply transform along columns of res rows)
    // We compute coeff = T * res * T^T with two 1-D passes.
    let shift1 = log2n + bd - 9; // 8x8:2, 4x4:1
    let add1 = if shift1 > 0 { 1i64 << (shift1 - 1) } else { 0 };
    // pass 1: along rows (horizontal): tmp[i][j] = sum_k T[i][k]*res[k][j]? 
    // Standard HM: first transform the rows of the residual, i.e.
    //   tmp = res * T^T  then  coeff = T * tmp ... order doesn't matter for square sep.
    // We do: tmp[i][j] = (sum_k T[i][k] * res[j][k]) >> shift1   (rows of res)
    let mut tmp = vec![0i64; n * n];
    for j in 0..n {
        // res row j
        let mut rowv = [0i64; 8];
        for k in 0..n {
            rowv[k] = res[j * n + k] as i64;
        }
        for i in 0..n {
            let mut s = 0i64;
            for k in 0..n {
                s += t_row(n, i, k) * rowv[k];
            }
            tmp[j * n + i] = (s + add1) >> shift1;
        }
    }
    // pass 2: along columns: coeff[i][j] = (sum_k T[i][k]*tmp[k][j]) >> shift2
    let shift2 = log2n + 6; // 8x8:9, 4x4:8
    let add2 = 1i64 << (shift2 - 1);
    let mut coeff = vec![0i64; n * n];
    for j in 0..n {
        let mut colv = [0i64; 8];
        for k in 0..n {
            colv[k] = tmp[k * n + j];
        }
        for i in 0..n {
            let mut s = 0i64;
            for k in 0..n {
                s += t_row(n, i, k) * colv[k];
            }
            coeff[i * n + j] = (s + add2) >> shift2;
        }
    }
    coeff
}

/// Forward quantisation: coeff → level. Intra rounding offset.
pub fn quantize(coeff: &[i64], n: usize, qp: u8) -> Vec<i16> {
    let log2n = if n == 8 { 3 } else { 2 } as i64;
    let bd = 8i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bd - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra
    let mut out = vec![0i16; n * n];
    for idx in 0..n * n {
        let c = coeff[idx];
        let level = (c.abs() * q_scale + offset) >> q_bits;
        let level = if c < 0 { -level } else { level };
        out[idx] = level.clamp(-32768, 32767) as i16;
    }
    out
}

/// Dequantisation: level → transform coefficient (spec 8.6.3).
pub fn dequantize(level: &[i16], n: usize, qp: u8) -> Vec<i64> {
    let log2n = if n == 8 { 3 } else { 2 } as i64;
    let bd = 8i64;
    let bd_shift = bd + log2n - 5; // 8x8:6, 4x4:5
    let add = 1i64 << (bd_shift - 1);
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let mut out = vec![0i64; n * n];
    for idx in 0..n * n {
        let v = (level[idx] as i64 * scale * per * 16 + add) >> bd_shift;
        out[idx] = v.clamp(-32768, 32767);
    }
    out
}

/// Inverse integer transform (spec 8.6.4.2), 8-bit. Returns residual.
pub fn inv_transform(coeff: &[i64], n: usize) -> Vec<i32> {
    let bd = 8i64;
    // Stage 1 (columns): out[n] = sum_k T[k][n] * coeff[k]  (= T^T @ col)
    let shift1 = 7i64;
    let add1 = 1i64 << (shift1 - 1);
    let mut tmp = vec![0i64; n * n];
    for c in 0..n {
        let mut colv = [0i64; 8];
        for k in 0..n {
            colv[k] = coeff[k * n + c];
        }
        for m in 0..n {
            let mut s = 0i64;
            for k in 0..n {
                s += t_row(n, k, m) * colv[k];
            }
            tmp[m * n + c] = ((s + add1) >> shift1).clamp(-32768, 32767);
        }
    }
    // Stage 2 (rows): out[n] = sum_k T[k][n] * tmp_row[k]
    let shift2 = 20 - bd; // 12
    let add2 = 1i64 << (shift2 - 1);
    let mut out = vec![0i32; n * n];
    for r in 0..n {
        let mut rowv = [0i64; 8];
        for k in 0..n {
            rowv[k] = tmp[r * n + k];
        }
        for m in 0..n {
            let mut s = 0i64;
            for k in 0..n {
                s += t_row(n, k, m) * rowv[k];
            }
            out[r * n + m] = ((s + add2) >> shift2) as i32;
        }
    }
    out
}
