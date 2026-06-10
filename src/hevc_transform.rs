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

/// 16×16 HEVC transform matrix (spec Table 8-6).
#[rustfmt::skip]
static T16: [[i32; 16]; 16] = [
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

static QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

/// Largest supported transform: 16×16 → 256 coefficients per fixed buffer.
const MAX_TB: usize = 256;
static DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Forward integer transform of an N×N residual block (N = 4 or 8).
/// Returns a fixed 64-entry buffer; only the first `n*n` entries are written.
pub(crate) fn fwd_transform(res: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    match n {
        4 => fwd_transform_n::<4>(res, &T4, bit_depth, &mut out),
        8 => fwd_transform_n::<8>(res, &T8, bit_depth, &mut out),
        16 => fwd_transform_n::<16>(res, &T16, bit_depth, &mut out),
        _ => panic!("unsupported transform size {n}"),
    }
    out
}

#[inline]
fn fwd_transform_n<const N: usize>(
    res: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    let log2n = N.trailing_zeros() as i32;
    let bd = bit_depth as i32;
    let shift1 = log2n + bd - 9;
    let add1 = if shift1 > 0 { 1i32 << (shift1 - 1) } else { 0 };
    // i32 throughout: products (|coeff|≤90 · |residual|≤4095) and the ≤N-term
    // sums stay well inside i32 for every supported bit depth.
    let mut tmp = [0i32; MAX_TB]; // N*N <= 256
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

/// Forward quantization: coeff → level. Returns a fixed 64-entry buffer.
pub(crate) fn quantize(coeff: &[i32], n: usize, qp: u8, bit_depth: u8) -> [i16; MAX_TB] {
    let log2n = n.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let q_bits = 14 + (qp as i64) / 6 + (15 - bd - log2n);
    let q_scale = QUANT_SCALE[(qp % 6) as usize];
    let offset = 171i64 << (q_bits - 9); // intra
    let mut out = [0i16; MAX_TB];
    for (o, &c) in out[..n * n].iter_mut().zip(coeff) {
        // i64 product: |coeff| can reach ~2^16 and q_scale ~2^15, so the product
        // overflows i32; keep this one multiply in i64.
        let c = c as i64;
        let level = (c.abs() * q_scale + offset) >> q_bits;
        let level = if c < 0 { -level } else { level };
        *o = level.clamp(-32768, 32767) as i16;
    }
    out
}

/// Dequantisation: level → transform coefficient (spec 8.6.3). Fixed 64-entry buffer.
pub(crate) fn dequantize(level: &[i16], n: usize, qp: u8, bit_depth: u8) -> [i32; MAX_TB] {
    let log2n = n.trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let bd_shift = bd + log2n - 5;
    let add = 1i64 << (bd_shift - 1);
    let scale = DEQUANT_SCALE[(qp % 6) as usize];
    let per = 1i64 << ((qp as i64) / 6);
    let factor = scale * per * 16;
    let mut out = [0i32; MAX_TB];
    for (o, &l) in out[..n * n].iter_mut().zip(level) {
        *o = ((l as i64 * factor + add) >> bd_shift).clamp(-32768, 32767) as i32;
    }
    out
}

/// Inverse integer transform (spec 8.6.4.2). Returns residual in a fixed buffer.
pub(crate) fn inv_transform(coeff: &[i32], n: usize, bit_depth: u8) -> [i32; MAX_TB] {
    let mut out = [0i32; MAX_TB];
    match n {
        4 => inv_transform_n::<4>(coeff, &T4, bit_depth, &mut out),
        8 => inv_transform_n::<8>(coeff, &T8, bit_depth, &mut out),
        16 => inv_transform_n::<16>(coeff, &T16, bit_depth, &mut out),
        _ => panic!("unsupported transform size {n}"),
    }
    out
}

#[inline]
fn inv_transform_n<const N: usize>(
    coeff: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32; MAX_TB],
) {
    let bd = bit_depth as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bd;
    let add2 = 1i32 << (shift2 - 1);

    // i32 accumulation is exact (dequant clamps to ±32768); basis rows are read
    // contiguously and zero coefficients are skipped — residual blocks are
    // mostly zero, so the skip removes the bulk of the work.
    let mut tmp = [0i32; MAX_TB];
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
    }

    /// Full pipeline residual → fwd → quant → dequant → inv reconstructs the
    /// residual within the quantization error. At low QP the error is small;
    /// a gross T16/scan/buffer bug would blow this far past tolerance.
    fn roundtrip_bounded(n: usize, qp: u8, max_mean_abs: f64) {
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
        let mean_abs: f64 = (0..n * n)
            .map(|i| (res[i] - rec[i]).unsigned_abs() as f64)
            .sum::<f64>()
            / (n * n) as f64;
        assert!(
            mean_abs <= max_mean_abs,
            "N={n} qp={qp}: mean|err|={mean_abs:.3} exceeds {max_mean_abs}"
        );
    }

    #[test]
    fn pipeline_roundtrip_low_qp() {
        // Low QP → tight reconstruction for every size, including the new 16×16.
        for &n in &[4usize, 8, 16] {
            roundtrip_bounded(n, 4, 3.0);
        }
    }

    #[test]
    fn pipeline_roundtrip_scales_with_qp() {
        // Error grows with QP but stays bounded; 16×16 must behave like 4/8.
        for &n in &[4usize, 8, 16] {
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
}
