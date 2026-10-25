/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without
 * // modification, are permitted provided that the following conditions are met:
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

//! HEVC scaling lists (frequency-dependent quantization matrices, §7.3.4,
//! §7.4.5, §8.6.4.2).
//!
//! A scaling list multiplies the dequantization step of every transform
//! coefficient by `m / 16`, so high-frequency coefficients can be quantized
//! more coarsely than the DC without touching the QP. hpvca signals one set
//! of lists per picture in the SPS, chosen from the picture QP alone (no
//! content analysis), so every HEIF grid cell derives the identical SPS.
//!
//! Shape: the 4×4 lists stay flat (the HEVC default, signaled by reference,
//! so 4×4 transform-skip blocks — to which the decoder applies the 4×4 list
//! — are unaffected). The 8×8 list is an HEVC default matrix (Table 7-6)
//! raised to a strength exponent: `m = 16 · (D / 16)^s`, `s = 0` flat,
//! `s = 1` the HEVC default. The 16×16 and 32×32 lists replicate the same
//! 8×8 matrix (the only shape the syntax can express) with the DC pinned at
//! 16, which maps the curve onto absolute spatial frequency consistently
//! across TB sizes. Luma and chroma get their own strengths.
//!
//! Shipped (2026-10-25, 8-image corpus, q40–92, 4:2:0, piecewise-linear BD
//! over ssimulacra2 / ColorVideoVDP): luma = the *inter* default matrix at
//! strength 1.0 for 8×8 (replicated to 16/32) and its even-position
//! subsample at strength 1.5 for 4×4, chroma flat, full strength for QP ≤ 34
//! (q ≥ 40) tapering to off at QP 40: Fast −2.19% / −1.79%, Slow −1.46% /
//! −1.15%, better at every quality point. Measured and rejected: the intra
//! default (diagonal-heavy) at any strength (best −0.69 / −0.44 at 0.75);
//! 8×8 strengths 0.75–1.5 of the inter shape (flat optimum, within noise of
//! 1.0); chroma lists at any strength and size, in 4:2:0 and 4:4:4 (alone
//! +0.5 / +0.6, nothing on top of luma); gating the 4×4 lists off for
//! transform-skip pictures (the screenshot prefers them shaped, −3.1% vs
//! −2.1%); two radial shapes and the inverse of the RDOQ decision weights
//! `diag(H)` (+0.22 / +3.05); and keeping the lists on past QP 40 (+0.6% at
//! QP 41, +1.3% at QP 45).

use std::sync::OnceLock;

use crate::hevc::BitWriter;
use crate::hevc_transform::QUANT_SCALE;

/// HEVC Table 7-6, default 8×8 intra scaling list, raster order (row-major).
/// Symmetric in (row, col), so the encoder's coefficient orientation and the
/// decoder's `ScalingFactor[x][y]` agree without a transpose.
#[rustfmt::skip]
static DEFAULT_INTRA_8X8: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24,
    16, 16, 16, 16, 17, 19, 22, 25,
    16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36,
    17, 17, 20, 24, 30, 35, 41, 47,
    18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88,
    24, 25, 29, 36, 47, 65, 88, 115,
];

/// Up-right diagonal scan of an N×N block (§6.5.3), `(x, y)` per position.
/// `scaling_list_data()` sends the 4×4 and 8×8 coefficients in this order.
const fn diag_scan<const N: usize, const NN: usize>() -> [(usize, usize); NN] {
    let mut out = [(0usize, 0usize); NN];
    let (mut i, mut x, mut y) = (0usize, 0usize, 0isize);
    while i < NN {
        while y >= 0 {
            if x < N && (y as usize) < N {
                out[i] = (x, y as usize);
                i += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x as isize;
        x = 0;
    }
    out
}

static DIAG_4X4: [(usize, usize); 16] = diag_scan::<4, 16>();
static DIAG_8X8: [(usize, usize); 64] = diag_scan::<8, 64>();

/// One transform size's `ScalingFactor` for one component class, plus the
/// matching forward-quantizer multipliers.
pub(crate) struct ScalingTable {
    /// `ScalingFactor` in raster order (`row * n + col`), 1..=255; 16 = flat.
    pub(crate) m: Box<[i32]>,
    /// `(QUANT_SCALE[qp % 6] << 4) / m[pos]` for each QP remainder: the
    /// forward quantizer's per-coefficient replacement for `QUANT_SCALE`.
    pub(crate) qc: [Box<[i32]>; 6],
}

impl ScalingTable {
    fn new(m: Vec<i32>) -> Self {
        let qc = std::array::from_fn(|rem| {
            m.iter()
                .map(|&m| ((QUANT_SCALE[rem] << 4) / i64::from(m)) as i32)
                .collect()
        });
        Self {
            m: m.into_boxed_slice(),
            qc,
        }
    }
}

/// The picture's scaling lists: the signalled 8×8 matrices and the derived
/// per-size encoder tables.
pub(crate) struct ScalingLists {
    /// Signalled 4×4 lists in raster order, `[luma, chroma]`; all-16 = flat,
    /// signalled by reference to the Table 7-5 default.
    list4: [[u8; 16]; 2],
    /// Signalled 8×8 lists in raster order, `[luma, chroma]`.
    list8: [[u8; 64]; 2],
    /// Encoder tables for n = 4, `None` when flat.
    tables4: [Option<ScalingTable>; 2],
    /// Encoder tables for n = 8, 16, 32 (index `log2 n − 3`), `[luma, chroma]`.
    tables: [[ScalingTable; 2]; 3],
}

/// Per-size strengths of one component: the 8×8 matrix (also replicated to
/// 16×16/32×32) and the 4×4 one.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Strength {
    pub(crate) s8: f32,
    pub(crate) s4: f32,
}

/// HEVC Table 7-6, default 8×8 inter scaling list (gentler ramp), raster.
#[rustfmt::skip]
const DEFAULT_INTER_8X8: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24,
    16, 16, 16, 17, 18, 20, 24, 25,
    16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33,
    17, 18, 20, 24, 25, 28, 33, 41,
    18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71,
    24, 25, 28, 33, 41, 54, 71, 91,
];

/// Which HEVC default matrix the shape comes from: the shipped inter list,
/// or the intra one for `HPVCA_SL_SHAPE=0` experiments.
fn base_8x8() -> &'static [u8; 64] {
    static INTRA: OnceLock<bool> = OnceLock::new();
    if *INTRA.get_or_init(|| std::env::var("HPVCA_SL_SHAPE").as_deref() == Ok("0")) {
        &DEFAULT_INTRA_8X8
    } else {
        &DEFAULT_INTER_8X8
    }
}

/// `m = 16 · (D / 16)^s`, clamped to the 1..=255 the syntax can carry.
fn shaped_8x8(strength: f32) -> [u8; 64] {
    let mut out = [16u8; 64];
    if strength == 0.0 {
        return out;
    }
    for (o, &d) in out.iter_mut().zip(base_8x8()) {
        let v = 16.0 * (f32::from(d) / 16.0).powf(strength);
        *o = v.round().clamp(1.0, 255.0) as u8;
    }
    out
}

/// The 4×4 list: the 8×8 shape sampled at even positions, since 4×4 position
/// u covers the same spatial frequency (u/8 cycles per sample) as 8×8
/// position 2u. The decoder applies this list to 4×4 transform-skip blocks
/// too; measured, that still helps the screenshot (−3.1% vs −2.1% with the
/// 4×4 lists flattened for transform-skip pictures), so it is not gated.
fn shaped_4x4(strength: f32) -> [u8; 16] {
    let mut out = [16u8; 16];
    if strength == 0.0 {
        return out;
    }
    let base = base_8x8();
    for (i, o) in out.iter_mut().enumerate() {
        let d = base[(i / 4) * 2 * 8 + (i % 4) * 2];
        let v = 16.0 * (f32::from(d) / 16.0).powf(strength);
        *o = v.round().clamp(1.0, 255.0) as u8;
    }
    out
}

/// `ScalingFactor` for an n×n TB from an 8×8 list: each list entry covers an
/// (n/8)×(n/8) square (§7.4.5 replication), the DC overridden by `dc`.
fn replicate(list8: &[u8; 64], n: usize, dc: u8) -> Vec<i32> {
    let k = n / 8;
    let mut m = vec![0i32; n * n];
    for row in 0..n {
        for col in 0..n {
            m[row * n + col] = i32::from(list8[(row / k) * 8 + col / k]);
        }
    }
    if n > 8 {
        m[0] = i32::from(dc);
    }
    m
}

/// DC of the 16×16 and 32×32 lists: pinned at 16 so the lists only reshape
/// the AC steps and the picture QP keeps its meaning.
const DC: u8 = 16;

impl ScalingLists {
    /// Lists for the given luma and chroma strengths.
    pub(crate) fn from_strength(luma: Strength, chroma: Strength) -> Self {
        let list8 = [shaped_8x8(luma.s8), shaped_8x8(chroma.s8)];
        let list4 = [shaped_4x4(luma.s4), shaped_4x4(chroma.s4)];
        let tables = std::array::from_fn(|size| {
            let n = 8usize << size;
            std::array::from_fn(|class| ScalingTable::new(replicate(&list8[class], n, DC)))
        });
        let tables4 = std::array::from_fn(|class| {
            (list4[class] != [16u8; 16])
                .then(|| ScalingTable::new(list4[class].iter().map(|&m| i32::from(m)).collect()))
        });
        Self {
            list4,
            list8,
            tables4,
            tables,
        }
    }

    /// The encoder table for an n×n TB of the given component class, or
    /// `None` when that size is flat.
    #[inline]
    pub(crate) fn table(&self, n: usize, is_luma: bool) -> Option<&ScalingTable> {
        debug_assert!(matches!(n, 4 | 8 | 16 | 32));
        let class = usize::from(!is_luma);
        if n < 8 {
            return self.tables4[class].as_ref();
        }
        Some(&self.tables[n.trailing_zeros() as usize - 3][class])
    }

    /// `ScalingFactor` for an n×n TB, or `None` when flat.
    #[inline]
    pub(crate) fn factor(&self, n: usize, is_luma: bool) -> Option<&[i32]> {
        self.table(n, is_luma).map(|t| &t.m[..])
    }

    /// `scaling_list_data()` (§7.3.4). Flat 4×4 lists reference the Table
    /// 7-5 defaults; otherwise the luma intra list of each size is sent
    /// explicitly and every other matrix of that size is predicted from an
    /// earlier one.
    pub(crate) fn write(&self, bw: &mut BitWriter) {
        let same_chroma = [
            self.list4[0] == self.list4[1],
            self.list8[0] == self.list8[1],
        ];
        for size_id in 0..4u32 {
            let step = if size_id == 3 { 3 } else { 1 };
            for matrix_id in (0..6u32).step_by(step) {
                if size_id == 0 && self.list4[(matrix_id % 3).min(1) as usize] == [16u8; 16] {
                    // Table 7-5 default 4×4 list: flat 16.
                    bw.write_bit(false); // scaling_list_pred_mode_flag
                    bw.write_ue(0); // scaling_list_pred_matrix_id_delta
                    continue;
                }
                let explicit =
                    matrix_id == 0 || (matrix_id == 1 && !same_chroma[usize::from(size_id > 0)]);
                if !explicit {
                    // refMatrixId = matrixId − delta · (sizeId == 3 ? 3 : 1).
                    // Cr copies Cb, every inter matrix copies its intra twin,
                    // and Cb copies Y when luma and chroma lists coincide.
                    let delta = match matrix_id {
                        1 | 2 => 1,
                        _ => 3 / step,
                    };
                    bw.write_bit(false); // scaling_list_pred_mode_flag
                    bw.write_ue(delta as u32); // scaling_list_pred_matrix_id_delta
                    continue;
                }
                bw.write_bit(true); // scaling_list_pred_mode_flag
                let (list, scan): (&[u8], &[(usize, usize)]) = if size_id == 0 {
                    (&self.list4[matrix_id as usize], &DIAG_4X4)
                } else {
                    (&self.list8[matrix_id as usize], &DIAG_8X8)
                };
                let side = if size_id == 0 { 4 } else { 8 };
                let mut next = 8i32;
                if size_id > 1 {
                    bw.write_se(i32::from(DC) - 8); // scaling_list_dc_coef_minus8
                    next = i32::from(DC);
                }
                for &(x, y) in scan {
                    let v = i32::from(list[y * side + x]);
                    // nextCoef = (nextCoef + delta + 256) % 256, delta in −128..=127.
                    let delta = ((v - next + 128).rem_euclid(256)) - 128;
                    bw.write_se(delta); // scaling_list_delta_coef
                    next = v;
                }
            }
        }
    }
}

/// Picture QP up to which the full luma strength applies (q ≥ 40); above it
/// the strength tapers linearly to 0 at [`STRENGTH_ZERO_QP`]. Measured at
/// q15/25/35 (QP 45/41/37): the lists cost +1.3% / +0.6% at QP 45 / 41 and
/// are neutral at QP 37, so they are off from QP 40.
const FULL_STRENGTH_QP: u8 = 34;
const STRENGTH_ZERO_QP: u8 = 40;
/// 8×8 strength 1.0 and 4×4 strength 1.5 (0.5 / 1.0 / 1.5 / 2.0 measured:
/// the 4×4 optimum is 1.5 on both tiers; 2.0 gives back ss2 at q85–92).
const LUMA_STRENGTH: Strength = Strength { s8: 1.0, s4: 1.5 };

/// `NAME=a[,b]` as two floats, the second defaulting to the first.
fn env_pair(name: &str) -> Option<[f32; 2]> {
    let v = std::env::var(name).ok()?;
    let mut it = v.split(',').map(|s| s.trim().parse::<f32>().ok());
    let a = it.next()??;
    let b = it.next().flatten().unwrap_or(a);
    Some([a, b])
}

/// Luma/chroma strengths for a picture QP, or `None` to leave scaling lists
/// off. `HPVCA_SL=sy[,sc]` (8×8) and `HPVCA_SL4=sy[,sc]` (4×4) override the
/// table for experiments.
fn strengths_for_qp(qp: u8) -> Option<[Strength; 2]> {
    static ENV: OnceLock<[Option<[f32; 2]>; 2]> = OnceLock::new();
    let [env8, env4] = ENV.get_or_init(|| [env_pair("HPVCA_SL"), env_pair("HPVCA_SL4")]);
    let taper = f32::from(STRENGTH_ZERO_QP.saturating_sub(qp.max(FULL_STRENGTH_QP)))
        / f32::from(STRENGTH_ZERO_QP - FULL_STRENGTH_QP);
    let s8 = env8.unwrap_or([LUMA_STRENGTH.s8 * taper, 0.0]);
    let s4 = env4.unwrap_or([LUMA_STRENGTH.s4 * taper, 0.0]);
    let out = [
        Strength {
            s8: s8[0],
            s4: s4[0],
        },
        Strength {
            s8: s8[1],
            s4: s4[1],
        },
    ];
    out.iter().any(|s| s.s8 > 0.0 || s.s4 > 0.0).then_some(out)
}

/// The picture's scaling lists for a picture QP (`None` = flat, flag off).
/// Interned per QP: the lists are a pure function of the QP, and
/// [`crate::hevc::PictureTools`] stays `Copy` by holding a `'static` ref.
pub(crate) fn lists_for_qp(qp: u8) -> Option<&'static ScalingLists> {
    static CACHE: [OnceLock<Option<ScalingLists>>; 64] = [const { OnceLock::new() }; 64];
    CACHE[usize::from(qp.min(63))]
        .get_or_init(|| strengths_for_qp(qp).map(|s| ScalingLists::from_strength(s[0], s[1])))
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diag_scan_is_the_spec_order() {
        assert_eq!(
            &DIAG_8X8[..6],
            &[(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
        );
        assert_eq!(DIAG_8X8[63], (7, 7));
        let mut seen = [false; 64];
        for &(x, y) in &DIAG_8X8 {
            assert!(!std::mem::replace(&mut seen[y * 8 + x], true));
        }
    }

    #[test]
    fn strength_one_is_the_hevc_default() {
        assert_eq!(&shaped_8x8(1.0), base_8x8());
        assert_eq!(shaped_8x8(0.0), [16u8; 64]);
    }

    #[test]
    fn replication_follows_the_spec() {
        let l = ScalingLists::from_strength(
            Strength { s8: 1.0, s4: 0.0 },
            Strength { s8: 1.0, s4: 0.0 },
        );
        let list8 = shaped_8x8(1.0);
        let m16 = l.factor(16, true).unwrap();
        assert_eq!(m16[0], 16);
        assert_eq!(m16[15 * 16 + 15], i32::from(list8[63]));
        assert_eq!(m16[2 * 16 + 3], i32::from(list8[8 + 1]));
        let m32 = l.factor(32, false).unwrap();
        assert_eq!(m32[31 * 32 + 31], i32::from(list8[63]));
        assert_eq!(m32[4 * 32 + 7], i32::from(list8[8 + 1]));
        assert!(l.factor(4, true).is_none());
        let t = l.table(8, true).unwrap();
        assert_eq!(t.qc[0][0], (QUANT_SCALE[0] as i32));
        assert_eq!(
            t.qc[4][63],
            ((QUANT_SCALE[4] << 4) / i64::from(list8[63])) as i32
        );
    }

    /// Decode `scaling_list_data()` back with a bit reader and compare.
    #[test]
    fn written_lists_round_trip() {
        struct R<'a> {
            b: &'a [u8],
            pos: usize,
        }
        impl R<'_> {
            fn bit(&mut self) -> u32 {
                let v = (self.b[self.pos / 8] >> (7 - self.pos % 8)) & 1;
                self.pos += 1;
                u32::from(v)
            }
            fn ue(&mut self) -> u32 {
                let mut z = 0;
                while self.bit() == 0 {
                    z += 1;
                }
                let mut v = 1;
                for _ in 0..z {
                    v = (v << 1) | self.bit();
                }
                v - 1
            }
            fn se(&mut self) -> i32 {
                let u = self.ue() as i32;
                if u % 2 == 1 { (u + 1) / 2 } else { -u / 2 }
            }
        }
        let st = |s8, s4| Strength { s8, s4 };
        for (ly, lc) in [
            (st(1.0, 0.0), st(1.0, 0.0)),
            (st(0.7, 1.0), st(0.4, 0.0)),
            (st(2.0, 0.5), st(0.0, 0.5)),
            (st(1.0, 0.0), st(0.0, 1.0)),
        ] {
            let lists = ScalingLists::from_strength(ly, lc);
            let mut bw = BitWriter::new();
            lists.write(&mut bw);
            let bytes = bw.finish();
            let mut r = R { b: &bytes, pos: 0 };
            // ScalingList[sizeId][matrixId] in scan order; dc per size>1.
            let mut sl = vec![vec![vec![0i32; 64]; 6]; 4];
            let mut dc = vec![vec![16i32; 6]; 4];
            for size_id in 0..4usize {
                let step = if size_id == 3 { 3 } else { 1 };
                let coef_num = 64.min(1 << (4 + (size_id << 1)));
                for matrix_id in (0..6).step_by(step) {
                    if r.bit() == 0 {
                        let delta = r.ue() as usize;
                        if delta == 0 {
                            assert_eq!(size_id, 0, "only 4×4 uses the default");
                            sl[size_id][matrix_id] = vec![16; 64];
                        } else {
                            let reference = matrix_id - delta * step;
                            sl[size_id][matrix_id] = sl[size_id][reference].clone();
                            dc[size_id][matrix_id] = dc[size_id][reference];
                        }
                        continue;
                    }
                    let mut next = 8;
                    if size_id > 1 {
                        next = r.se() + 8;
                        dc[size_id][matrix_id] = next;
                    }
                    for entry in sl[size_id][matrix_id].iter_mut().take(coef_num) {
                        next = (next + r.se() + 256) % 256;
                        *entry = next;
                    }
                }
            }
            assert_eq!(r.pos.div_ceil(8), bytes.len());
            // 4×4: flat decodes to the default, shaped to the sent list.
            for class in 0..2usize {
                let expect: Vec<i32> = match lists.factor(4, class == 0) {
                    Some(m) => m.to_vec(),
                    None => vec![16; 16],
                };
                for (i, &(x, y)) in DIAG_4X4.iter().enumerate() {
                    assert_eq!(expect[y * 4 + x], sl[0][class][i], "4×4 class{class} {i}");
                }
                assert_eq!(sl[0][class], sl[0][class + 3]);
            }
            for (class, n) in [(0usize, 8usize), (1, 8), (0, 32), (1, 16), (0, 16)] {
                let matrix_id = class; // intra Y / Cb
                let size_id = n.trailing_zeros() as usize - 2;
                let m = lists.factor(n, class == 0).unwrap();
                let k = n / 8;
                for (i, &(x, y)) in DIAG_8X8.iter().enumerate() {
                    for j in 0..k {
                        for l in 0..k {
                            let (row, col) = (y * k + j, x * k + l);
                            let expect = if n > 8 && row == 0 && col == 0 {
                                dc[size_id][matrix_id]
                            } else {
                                sl[size_id][matrix_id][i]
                            };
                            assert_eq!(m[row * n + col], expect, "n{n} class{class} ({row},{col})");
                        }
                    }
                }
                // The inter twin decodes to the same matrix.
                assert_eq!(sl[size_id][matrix_id], sl[size_id][matrix_id + 3]);
            }
        }
    }
}
