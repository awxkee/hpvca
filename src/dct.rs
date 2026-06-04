//! DCT-II (8×8) and quantization for HEVC intra coding.
//!
//! HEVC uses a scaled integer approximation of the DCT.
//! We implement the classic AAN fast 8-point DCT here (sufficient for
//! a still-image encoder) and apply HEVC-style quantization.

/// Luminance quantization matrix (JPEG-derived, quality-scaled).
/// HEVC allows custom matrices; we embed a sensible default.
pub const LUMA_QUANT_MATRIX: [[u16; 8]; 8] = [
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
];

/// Chrominance quantization matrix.
pub const CHROMA_QUANT_MATRIX: [[u16; 8]; 8] = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
];

/// Compute quality-scaled quantization step for each coefficient.
///
/// `quality` is 1–100 (JPEG-style): 100 = best, 1 = worst.
pub fn scaled_quant_matrix(base: &[[u16; 8]; 8], quality: u8) -> [[u16; 8]; 8] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [[0u16; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            let v = ((base[r][c] as u32 * scale + 50) / 100).clamp(1, 255);
            out[r][c] = v as u16;
        }
    }
    out
}

/// Apply floating-point 1-D 8-point DCT-II (unnormalized) to `data` in-place.
fn dct1d(data: &mut [f32; 8]) {
    // AAN algorithm (Arai, Agui, Nakajima)
    const A1: f32 = 0.707_106_78; // cos(π/4)
    const A2: f32 = 0.541_196_1;  // cos(3π/8) - cos(π/8) … (approx)
    const A3: f32 = 0.707_106_78;
    const A4: f32 = 1.306_562_96; // cos(π/8) + cos(3π/8)
    const A5: f32 = 0.382_683_43; // cos(3π/8)

    // Stage 1
    let t0 = data[0] + data[7];
    let t1 = data[1] + data[6];
    let t2 = data[2] + data[5];
    let t3 = data[3] + data[4];
    let t4 = data[3] - data[4];
    let t5 = data[2] - data[5];
    let t6 = data[1] - data[6];
    let t7 = data[0] - data[7];

    // Even part
    let t8 = t0 + t3;
    let t9 = t1 + t2;
    let t10 = t1 - t2;
    let t11 = t0 - t3;

    data[0] = t8 + t9;
    data[4] = t8 - t9;
    let tmp = (t10 + t11) * A1;
    data[2] = t11 + tmp;
    data[6] = t11 - tmp;

    // Odd part
    let t12 = t4 + t5;
    let t13 = t5 + t6;
    let t14 = t6 + t7;
    let t15 = (t12 - t14) * A5;
    let t16 = A2 * t12 + t15;
    let t17 = A4 * t14 + t15;
    let t18 = A3 * t13;

    data[5] = t7 + t18 - t17;
    data[3] = t7 - t18 + t16;
    data[1] = t7 + t18 + t17 - t16;  // simplified chain
    data[7] = t7 - t18 - t16 + t17;
    // (The exact AAN normalization factors are absorbed into the quant matrix)

    // Re-order odd outputs to natural DCT order via simpler formula:
    // We use a straightforward Loeffler-style approach instead.
    // Reset with direct formula for correctness:
    *data = dct1d_direct_f32(*data);
}

/// Orthonormal 8-point DCT-II.
///
/// Uses the orthonormal scaling (1/√N for k=0, √(2/N) otherwise) so that the
/// inverse IDCT in the reconstruct loop — which is orthonormal — is an exact
/// inverse. Without this scaling the forward transform is √8 too large per
/// dimension (8× in 2-D), and the orthonormal inverse over-reconstructs by 8×.
fn dct1d_direct_f32(x: [f32; 8]) -> [f32; 8] {
    use std::f32::consts::PI;
    const N: usize = 8;
    let mut out = [0f32; 8];
    for k in 0..N {
        let wk = if k == 0 {
            (1.0 / N as f32).sqrt()
        } else {
            (2.0 / N as f32).sqrt()
        };
        let mut sum = 0.0f32;
        for n in 0..N {
            sum += x[n] * ((PI * k as f32 * (2 * n + 1) as f32) / 16.0).cos();
        }
        out[k] = wk * sum;
    }
    out
}

/// 2-D 8×8 DCT of an 8×8 block of `f32`.
pub fn dct2d(block: &mut [[f32; 8]; 8]) {
    // Row transforms
    for row in block.iter_mut() {
        *row = dct1d_direct_f32(*row);
    }
    // Column transforms
    for col in 0..8 {
        let mut col_data = [0f32; 8];
        for row in 0..8 {
            col_data[row] = block[row][col];
        }
        col_data = dct1d_direct_f32(col_data);
        for row in 0..8 {
            block[row][col] = col_data[row];
        }
    }
}

/// Extract an 8×8 block from a planar buffer (clamping at edges).
pub fn extract_block(
    plane: &[u8],
    stride: usize,
    block_row: usize,
    block_col: usize,
    height: usize,
) -> [[f32; 8]; 8] {
    let mut out = [[0.0f32; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            let row = (block_row + r).min(height - 1);
            let col_idx = block_col + c;
            // col is already within the horizontal extraction range
            let sample = plane[row * stride + col_idx.min(stride - 1)];
            out[r][c] = sample as f32 - 128.0; // level shift
        }
    }
    out
}

/// Quantize a DCT block, returning i16 coefficients in raster order.
pub fn quantize(dct_block: &[[f32; 8]; 8], qmat: &[[u16; 8]; 8]) -> [[i16; 8]; 8] {
    let mut out = [[0i16; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            let q = qmat[r][c] as f32;
            out[r][c] = (dct_block[r][c] / q).round() as i16;
        }
    }
    out
}

/// Dequantize and reconstruct — used in the decoder path and for verification.
pub fn dequantize(coeffs: &[[i16; 8]; 8], qmat: &[[u16; 8]; 8]) -> [[f32; 8]; 8] {
    let mut out = [[0.0f32; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            out[r][c] = coeffs[r][c] as f32 * qmat[r][c] as f32;
        }
    }
    out
}

/// HEVC up-right diagonal scan for an 8×8 transform block, in **sub-block-major**
/// order: the four 4×4 sub-blocks are visited in up-right diagonal order, and
/// within each sub-block the 16 positions are visited in up-right diagonal order.
/// Entries are `(row, col)`. This is the order HEVC `residual_coding` expects, so
/// `coeffs[sb*16 + k]` indexes sub-block `sb`, scan position `k` within it.
#[rustfmt::skip]
pub const ZIGZAG: [(usize, usize); 64] = [
    (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),
    (1,2),(0,3),(3,1),(2,2),(1,3),(3,2),(2,3),(3,3),
    (4,0),(5,0),(4,1),(6,0),(5,1),(4,2),(7,0),(6,1),
    (5,2),(4,3),(7,1),(6,2),(5,3),(7,2),(6,3),(7,3),
    (0,4),(1,4),(0,5),(2,4),(1,5),(0,6),(3,4),(2,5),
    (1,6),(0,7),(3,5),(2,6),(1,7),(3,6),(2,7),(3,7),
    (4,4),(5,4),(4,5),(6,4),(5,5),(4,6),(7,4),(6,5),
    (5,6),(4,7),(7,5),(6,6),(5,7),(7,6),(6,7),(7,7),
];

/// HEVC up-right diagonal scan for a single 4×4 block, `(row, col)`.
#[rustfmt::skip]
pub const DIAG_SCAN_4X4: [(usize, usize); 16] = [
    (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),
    (1,2),(0,3),(3,1),(2,2),(1,3),(3,2),(2,3),(3,3),
];

/// Flatten 8×8 coefficient block into a 64-element scan-ordered vector.
pub fn zigzag_scan(block: &[[i16; 8]; 8]) -> Vec<i16> {
    ZIGZAG.iter().map(|&(r, c)| block[r][c]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_dc_only() {
        // Flat block (all 127) → DC coeff large, all AC ≈ 0
        let mut block = [[127.0f32; 8]; 8];
        dct2d(&mut block);
        let dc = block[0][0].abs();
        assert!(dc > 100.0, "DC should dominate for flat block, got {dc}");
        for r in 0..8 {
            for c in 0..8 {
                if r == 0 && c == 0 { continue; }
                assert!(block[r][c].abs() < 1.0, "AC[{r}][{c}] should be ~0, got {}", block[r][c]);
            }
        }
    }

    #[test]
    fn quantize_round_trip() {
        let qmat = scaled_quant_matrix(&LUMA_QUANT_MATRIX, 90);
        let block = [[10.0f32; 8]; 8];
        let coeffs = quantize(&block, &qmat);
        let recon = dequantize(&coeffs, &qmat);
        // Check DC neighbourhood is preserved
        assert!((recon[0][0] - block[0][0]).abs() < 5.0);
    }
}

// ─── HEVC QP-based quantization ──────────────────────────────────────────────

/// HEVC level scale table: levelScale[QP % 6].
/// From HEVC spec §8.6.3 Table 8-15.
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Compute the HEVC encoder quantisation step for a given QP and log2 TU size.
///
/// HEVC spec §8.6.3:
///   qp_param = levelScale[QP % 6] << (QP / 6)
///   bd_shift = BitDepth + Log2TrafoSize - 5    (8-bit luma 8×8: 8+3-5=6)
///   quantStep = (m * qp_param) / 2^(bd_shift+1)
///   coeffLevel = (|transformCoeff| + add) / quantStep  [encoder approx]
///
/// For compatibility with the decoder, which dequantises as:
///   coeff_dq = (coeffLevel * m * qp_param + add_dec) >> bd_shift
///
/// Our float DCT produces unnormalized coefficients whose DC value is
/// sum(residual) × √N × √N = N × mean_residual for an N×N block.
/// The HEVC integer DCT is defined to have the same DC magnitude, so the
/// effective quantisation step for our float DCT output is:
///
///   float_quant_step = (m * qp_param) >> (bd_shift + 1 + extra_shift)
///
/// We use extra_shift=0 for 8×8 (the float DCT DC already equals N×mean),
/// and adjust for 4×4 (halving N → halving DC, so quant step halves too).
/// Quantisation step for our orthonormal float DCT, matched to a real HEVC
/// integer decoder.
///
/// Our forward DCT is orthonormal (DC of an N×N block = N × mean_residual) and
/// the inverse IDCT is its orthonormal inverse (DC weight 1/N). A spec-compliant
/// decoder, however, uses the integer dequant (`level · 16 · qpScale[qp%6] <<
/// (qp/6)`, right-shifted) followed by the integer inverse transform (basis
/// scaled by 64, with stage shifts of 7 and 12). Tracing a DC level through that
/// full integer pipeline gives a per-level reconstruction gain of
/// `16 · qpp / 2^(log2N+10)`, where `qpp = levelScale[qp%6] << (qp/6)`.
///
/// For the levels we emit to be reconstructed correctly by such a decoder, the
/// coefficient-domain step must be `N · gain`. The factor of N cancels the
/// 2^log2N in the gain's denominator, leaving a step that is independent of
/// block size:
///
///   step = qpp / 64
///
/// `dequantize` multiplies the level back by this same step, so the encoder's
/// own reconstruction loop reproduces exactly what the decoder will compute —
/// keeping intra-prediction references in sync between encoder and decoder.
pub fn hevc_qp_quant_step(qp: u8, _log2_size: u32) -> f32 {
    let qp = qp as i64;
    let qp_param = LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
    let step = qp_param as f32 / 64.0;
    step.max(1.0)
}

/// Quantize a DCT block: coeffLevel = round(coeff / step).
pub fn hevc_quantize(dct_block: &[[f32; 8]; 8], qp: u8, log2_size: u32) -> [[i16; 8]; 8] {
    let step = hevc_qp_quant_step(qp, log2_size);
    let mut out = [[0i16; 8]; 8];
    let n = 1usize << log2_size;
    for r in 0..n {
        for c in 0..n {
            out[r][c] = (dct_block[r][c] / step).round() as i16;
        }
    }
    out
}

/// Dequantize: the exact symmetric inverse of `hevc_quantize`.
///
/// Returns float coefficients (coeffLevel × step) in the transform domain, ready
/// for the orthonormal inverse DCT. Using the same `step` as the forward path
/// guarantees the encoder's reconstruction matches what it quantized, with only
/// the unavoidable rounding loss.
pub fn hevc_dequantize(coeffs: &[[i16; 8]; 8], qp: u8, log2_size: u32) -> [[f32; 8]; 8] {
    let step = hevc_qp_quant_step(qp, log2_size);
    let mut out = [[0.0f32; 8]; 8];
    let n = 1usize << log2_size;
    for r in 0..n {
        for c in 0..n {
            out[r][c] = coeffs[r][c] as f32 * step;
        }
    }
    out
}

/// HEVC 8×8 integer inverse DCT basis: rows mat_dct[4*j][0..8] of the spec's
/// 32×32 transform matrix (§8.6.4.2).
const IDCT8: [[i32; 8]; 8] = [
    [64,  64,  64,  64,  64,  64,  64,  64],
    [89,  75,  50,  18, -18, -50, -75, -89],
    [83,  36, -36, -83, -83, -36,  36,  83],
    [75, -18, -89, -50,  50,  89,  18, -75],
    [64, -64, -64,  64,  64, -64, -64,  64],
    [50, -89,  18,  75, -75, -18,  89, -50],
    [36, -83,  83, -36, -36,  83, -83,  36],
    [18, -50,  75, -89,  89, -75,  50, -18],
];

fn clip3(lo: i32, hi: i32, v: i32) -> i32 { v.max(lo).min(hi) }

/// Spec-exact HEVC reconstruction for an 8×8 luma TU: integer dequantization
/// (§8.6.3) followed by the integer inverse transform (§8.6.4), bit-identical to
/// conformant decoders (libde265/ffmpeg). Returns the residual samples as i32.
///
/// `coeffs[r][c]` are the quantized transform-coefficient levels (row-major).
pub fn hevc_inverse_8x8(coeffs: &[[i16; 8]; 8], qp: u8) -> [[i32; 8]; 8] {
    const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
    let bit_depth = 8i32;
    let log2 = 3i32;
    // Dequant: bdShift = BitDepth + Log2(nT) - 5, then -4 (m_x_y folded to 1).
    let bd_shift_dq = bit_depth + log2 - 5 - 4; // = 2 for 8-bit 8×8
    let fact = LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
    let offset_dq = 1i32 << (bd_shift_dq - 1);

    let mut d = [[0i32; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            let lvl = coeffs[r][c] as i32;
            d[r][c] = clip3(-32768, 32767, (lvl * fact + offset_dq) >> bd_shift_dq);
        }
    }

    // Inverse transform: vertical pass (shift 7), then horizontal pass (shift bdShift2).
    let rnd1 = 1i32 << (7 - 1);
    let coeff_max = (1i32 << 15) - 1;
    let coeff_min = -(1i32 << 15);
    // Vertical pass: g[i][c] = sum_j IDCT8[j][i] * d[j][c]
    let mut g = [[0i32; 8]; 8];
    for c in 0..8 {
        for i in 0..8 {
            let mut sum = 0i32;
            for j in 0..8 {
                sum += IDCT8[j][i] * d[j][c];
            }
            g[i][c] = clip3(coeff_min, coeff_max, (sum + rnd1) >> 7);
        }
    }
    // Horizontal pass: out[y][i] = sum_j IDCT8[j][i] * g[y][j]
    let bd_shift2 = 20 - bit_depth; // = 12
    let rnd2 = 1i32 << (bd_shift2 - 1);
    let mut out = [[0i32; 8]; 8];
    for y in 0..8 {
        for i in 0..8 {
            let mut sum = 0i32;
            for j in 0..8 {
                sum += IDCT8[j][i] * g[y][j];
            }
            out[y][i] = (sum + rnd2) >> bd_shift2;
        }
    }
    out
}

#[cfg(test)]
mod hevc_quant_tests {
    use super::*;

    #[test]
    fn quant_dequant_roundtrip_dc() {
        // Flat block of 50 → residual=50-128=-78 after level shift (done in caller)
        // We test the quant/dequant pair for a known DC value
        let mut block = [[0.0f32; 8]; 8];
        block[0][0] = 100.0; // DC-only
        let q = hevc_quantize(&block, 26, 3);
        let dq = hevc_dequantize(&q, 26, 3);
        // Dequantized DC should be within one quant step of original
        let step = hevc_qp_quant_step(26, 3);
        assert!((dq[0][0] - block[0][0]).abs() <= step,
            "dq={:.1} orig={:.1} step={:.1}", dq[0][0], block[0][0], step);
    }

    #[test]
    fn quant_step_reasonable() {
        // Step is qp_param/64; for QP=26 this is 12.75. It must be positive and
        // increase with QP (coarser quantisation = lower quality).
        let step = hevc_qp_quant_step(26, 3);
        assert!(step > 1.0 && step < 100.0, "step={}", step);
        let step_hi = hevc_qp_quant_step(37, 3);
        assert!(step_hi > step, "Higher QP should give larger step");
        // Step is independent of block size (the orthonormal-DCT N factor cancels
        // the decoder's 1/N reconstruction gain).
        assert_eq!(
            hevc_qp_quant_step(26, 3),
            hevc_qp_quant_step(26, 2),
            "step must not depend on block size"
        );
    }
}
