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
//! Inter prediction building blocks for video P and B pictures: the motion
//! field entry format, merge and AMVP candidates, fractional-sample
//! interpolation, bi-prediction and the motion search. Everything here is
//! specialised for one reference picture per list and 2N×2N prediction
//! units.

use crate::ibc::{BV_INTRA, FieldEntry};

/// A motion vector in quarter luma samples, (x, y).
pub(crate) type Mv = (i16, i16);

/// Largest vector component the encoder uses, in quarter samples. The field
/// entry has 15 bits per component.
pub(crate) const MV_LIMIT: i16 = 8188;

/// `MaxNumMergeCand` of a video P or B slice.
pub(crate) const MAX_MERGE_CAND: usize = 3;

/// Motion of a prediction unit: which reference picture lists predict it and
/// with what vectors. Each list holds one picture, so there is no reference
/// index. The vector of an unused list is (0, 0), which makes `==` the
/// standard's "identical motion" comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PuMotion {
    pub(crate) pred: [bool; 2],
    pub(crate) mv: [Mv; 2],
}

impl PuMotion {
    /// Predicted from list `list` alone.
    #[inline]
    pub(crate) fn uni(list: usize, mv: Mv) -> Self {
        let mut pu = PuMotion::default();
        pu.pred[list] = true;
        pu.mv[list] = mv;
        pu
    }

    /// Predicted from both lists.
    #[inline]
    pub(crate) fn bi(mv0: Mv, mv1: Mv) -> Self {
        PuMotion {
            pred: [true; 2],
            mv: [mv0, mv1],
        }
    }

    #[inline]
    pub(crate) fn is_bi(self) -> bool {
        self.pred[0] && self.pred[1]
    }
}

/// What the motion field records about an inter CU: besides its motion, the
/// two facts its neighbors need — whether it was skipped (`cu_skip_flag`
/// context) and whether its luma transform block has coefficients
/// (deblocking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Motion {
    pub(crate) pu: PuMotion,
    pub(crate) skip: bool,
    pub(crate) luma_cbf: bool,
}

impl Motion {
    /// The `bv_map` entry: 15 bits per vector component (list 0 x, y, then
    /// list 1 x, y), the two prediction flags, then the two CU flags. An
    /// inter block sets at least one prediction flag, so no entry equals
    /// [`BV_INTRA`].
    #[inline]
    pub(crate) fn pack(self) -> FieldEntry {
        let field = |v: i16| {
            debug_assert!(v.abs() <= MV_LIMIT);
            i64::from(v) & 0x7FFF
        };
        debug_assert!(self.pu.pred[0] || self.pu.pred[1]);
        (field(self.pu.mv[0].0) << 49)
            | (field(self.pu.mv[0].1) << 34)
            | (field(self.pu.mv[1].0) << 19)
            | (field(self.pu.mv[1].1) << 4)
            | (i64::from(self.pu.pred[1]) << 3)
            | (i64::from(self.pu.pred[0]) << 2)
            | (i64::from(self.skip) << 1)
            | i64::from(self.luma_cbf)
    }

    /// `None` for an intra block.
    #[inline]
    pub(crate) fn unpack(entry: FieldEntry) -> Option<Motion> {
        // Sign-extend the 15-bit field whose lowest bit is `shift`.
        let field = |shift: u32| ((entry << (49 - shift)) >> 49) as i16;
        (entry != BV_INTRA).then_some(Motion {
            pu: PuMotion {
                pred: [entry & 4 != 0, entry & 8 != 0],
                mv: [(field(49), field(34)), (field(19), field(4))],
            },
            skip: entry & 2 != 0,
            luma_cbf: entry & 1 != 0,
        })
    }
}

/// Deblocking boundary strength (§8.7.2.4) across an edge that is both a
/// transform and a prediction edge, given the two blocks' field entries: 2
/// with an intra block on either side; else 1 when either luma transform
/// block has coefficients, the blocks are predicted from different pictures
/// or a different number of them, or vectors into the same picture differ by
/// a whole sample or more; else 0. The two lists never hold the same
/// picture, so "same pictures" is "same prediction flags".
#[inline]
pub(crate) fn boundary_strength(current: FieldEntry, neighbor: FieldEntry) -> u8 {
    match (Motion::unpack(current), Motion::unpack(neighbor)) {
        (Some(a), Some(b)) => {
            let apart = |list: usize| {
                let (p, q) = (a.pu.mv[list], b.pu.mv[list]);
                (i32::from(p.0) - i32::from(q.0)).abs() >= 4
                    || (i32::from(p.1) - i32::from(q.1)).abs() >= 4
            };
            u8::from(a.luma_cbf || b.luma_cbf || a.pu.pred != b.pu.pred || apart(0) || apart(1))
        }
        _ => 2,
    }
}

/// Merge candidate list of a `size`×`size` PART_2Nx2N prediction unit at
/// (`x`, `y`) (§8.5.3.2.2–5), with no temporal candidate and
/// `Log2ParMrgLevel` 2: the spatial candidates A1, B1, B0, A0, B2 after
/// pruning; in a B slice the combined bi-predictive candidates; then zero
/// vectors (bi-predictive in a B slice).
///
/// `motion(x, y)` is the motion of the inter block covering that sample, or
/// `None` when it is intra, outside the picture or not yet decoded.
pub(crate) fn merge_candidates(
    x: usize,
    y: usize,
    size: usize,
    b_slice: bool,
    motion: &dyn Fn(i32, i32) -> Option<PuMotion>,
) -> [PuMotion; MAX_MERGE_CAND] {
    let (x, y, n) = (x as i32, y as i32, size as i32);
    let a1 = motion(x - 1, y + n - 1);
    let b1 = motion(x + n - 1, y - 1);
    let b0 = motion(x + n, y - 1);
    let a0 = motion(x - 1, y + n);
    let b2 = motion(x - 1, y - 1);
    // Each candidate is compared with specific earlier *neighbors* (whether
    // or not those survived their own pruning).
    let same = |p: Option<PuMotion>, q: Option<PuMotion>| p.is_some() && p == q;
    let spatial = [
        a1,
        b1.filter(|_| !same(a1, b1)),
        b0.filter(|_| !same(b1, b0)),
        a0.filter(|_| !same(a1, a0)),
    ];
    let four = spatial.iter().all(Option::is_some);
    let b2 = b2.filter(|_| !four && !same(a1, b2) && !same(b1, b2));
    let zero = if b_slice {
        PuMotion::bi((0, 0), (0, 0))
    } else {
        PuMotion::uni(0, (0, 0))
    };
    let mut list = [zero; MAX_MERGE_CAND];
    let mut count = 0;
    for pu in spatial.into_iter().chain([b2]).flatten() {
        if count < MAX_MERGE_CAND {
            list[count] = pu;
            count += 1;
        }
    }
    // Combined bi-predictive candidates (§8.5.3.2.4): list 0 motion of one
    // candidate with list 1 motion of another. The lists hold different
    // pictures, so such a pair is never the same prediction twice.
    if b_slice && count > 1 {
        const L0_CAND: [usize; 12] = [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3];
        const L1_CAND: [usize; 12] = [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2];
        let original = count;
        for comb in 0..original * (original - 1) {
            if count == MAX_MERGE_CAND {
                break;
            }
            let (from0, from1) = (list[L0_CAND[comb]], list[L1_CAND[comb]]);
            if from0.pred[0] && from1.pred[1] {
                list[count] = PuMotion::bi(from0.mv[0], from1.mv[1]);
                count += 1;
            }
        }
    }
    list
}

/// A neighbor's vector into another picture, rescaled by the ratio of the
/// picture distances `tb / td` (§8.5.3.2.7, equations 8-179 to 8-183).
fn scale_mv(mv: Mv, tb: i32, td: i32) -> Mv {
    let (tb, td) = (tb.clamp(-128, 127), td.clamp(-128, 127));
    let tx = (16384 + td.abs() / 2) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let scale = |v: i16| {
        let product = factor * i32::from(v);
        (product.signum() * ((product.abs() + 127) >> 8)).clamp(-32768, 32767) as i16
    };
    (scale(mv.0), scale(mv.1))
}

/// AMVP predictors (§8.5.3.2.6–7) for list `list` of a `size`×`size`
/// PART_2Nx2N prediction unit at (`x`, `y`), without a temporal candidate.
///
/// `motion(x, y)` is as for [`merge_candidates`]. `distance[X]` is the POC
/// of the current picture minus that of the picture in list X: a neighbor
/// that only has the other list's vector lends it rescaled by the ratio of
/// the distances (or as it is when both lists hold the same picture, which
/// is never the case in a B slice here).
pub(crate) fn amvp_predictors(
    x: usize,
    y: usize,
    size: usize,
    list: usize,
    distance: [i32; 2],
    motion: &dyn Fn(i32, i32) -> Option<PuMotion>,
) -> [Mv; 2] {
    let (x, y, n) = (x as i32, y as i32, size as i32);
    let other = 1 - list;
    let same_picture = distance[0] == distance[1];
    // A vector that needs no scaling: the target list's own, or the other
    // list's when that is the same picture.
    let direct = |pu: PuMotion| -> Option<Mv> {
        if pu.pred[list] {
            Some(pu.mv[list])
        } else {
            (pu.pred[other] && same_picture).then_some(pu.mv[other])
        }
    };
    // Any vector, scaled to the target picture's distance.
    let scaled = |pu: PuMotion| -> Option<Mv> {
        if pu.pred[list] {
            Some(pu.mv[list])
        } else if pu.pred[other] {
            Some(if same_picture {
                pu.mv[other]
            } else {
                scale_mv(pu.mv[other], distance[list], distance[other])
            })
        } else {
            None
        }
    };
    // A0, A1 to the left; B0, B1, B2 above.
    let left = [motion(x - 1, y + n), motion(x - 1, y + n - 1)];
    let above = [
        motion(x + n, y - 1),
        motion(x + n - 1, y - 1),
        motion(x - 1, y - 1),
    ];
    let first = |blocks: &[Option<PuMotion>], pick: &dyn Fn(PuMotion) -> Option<Mv>| {
        blocks.iter().flatten().find_map(|&pu| pick(pu))
    };
    let is_scaled = left.iter().any(Option::is_some);
    let mut a = first(&left, &direct).or_else(|| first(&left, &scaled));
    let mut b = first(&above, &direct);
    if !is_scaled {
        // Nothing to the left: the unscaled candidate from above stands in
        // for it, and the one from above is derived again with scaling.
        if b.is_some() {
            a = b;
        }
        b = first(&above, &scaled);
    }
    let mut preds = [(0i16, 0i16); 2];
    let mut count = 0;
    if let Some(mv) = a {
        preds[count] = mv;
        count += 1;
    }
    if let Some(mv) = b
        && Some(mv) != a
    {
        preds[count] = mv;
    }
    preds
}

/// Luma interpolation filters by quarter-sample phase (Table 8-11).
static LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Chroma interpolation filters by eighth-sample phase (Table 8-12).
static CHROMA_FILTER: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Explicit weighted prediction of one component (§8.5.3.3.4.3): the
/// prediction is `sample · w / 2^log2_denom + offset`, with `offset` in
/// sample units at the picture's bit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Weight {
    pub(crate) w: i32,
    pub(crate) log2_denom: u32,
    pub(crate) offset: i32,
}

impl Weight {
    /// Plain prediction: the default weighted prediction of §8.5.3.3.4.2.
    pub(crate) const UNIT: Weight = Weight {
        w: 1,
        log2_denom: 0,
        offset: 0,
    };
}

/// What the 14-bit intermediates of bi-prediction are stored minus, so that
/// they fit 16 bits (a two-dimensional filter overshoots 14 bits by more
/// than a factor of two).
pub(crate) const MID_OFFSET: i32 = 8192;

/// What an interpolation leaves in its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rounding {
    /// Finished samples of a uni-directional prediction with this weight.
    Samples(Weight),
    /// The 14-bit intermediate `predSamplesLX` of §8.5.3.3.3 minus
    /// [`MID_OFFSET`], each stored as the bit pattern of an `i16`: one input
    /// of [`bi_average`].
    Intermediate,
}

impl Rounding {
    /// Plain uni-directional prediction.
    pub(crate) const PLAIN: Rounding = Rounding::Samples(Weight::UNIT);
}

/// One plane of a reference picture.
#[derive(Clone, Copy)]
pub(crate) struct RefPlane<'a> {
    pub(crate) data: &'a [u16],
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Scratch for [`predict_luma`] / [`predict_chroma`].
#[derive(Default)]
pub(crate) struct InterpScratch {
    window: Vec<u16>,
    rows: Vec<i32>,
    /// The SIMD kernels' 16-bit intermediate rows.
    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ))]
    rows16: Vec<i16>,
}

/// The eight-lane filters may read four extra samples at a four-sample tail.
const SIMD_PAD: usize = 4;

/// Prediction of a `w`×`h` block from one picture, whose top-left reference
/// sample is (`x0`, `y0`), filtered with the given phase filters (`None` for
/// a whole-sample phase), into `out` (stride `w`). This is §8.5.3.3.3,
/// followed for [`Rounding::Samples`] by the uni-directional weighted
/// prediction of §8.5.3.3.4; reference samples outside the picture repeat
/// its border.
#[allow(clippy::too_many_arguments)]
fn interpolate<const N: usize>(
    plane: RefPlane<'_>,
    x0: isize,
    y0: isize,
    horizontal: Option<&[i32; N]>,
    vertical: Option<&[i32; N]>,
    w: usize,
    h: usize,
    bit_depth: u8,
    rounding: Rounding,
    out: &mut [u16],
    scratch: &mut InterpScratch,
) {
    // A filter needs N / 2 − 1 samples before the block and N / 2 after it,
    // in its own direction only.
    let (halo_x, halo_y) = (
        if horizontal.is_some() { N / 2 - 1 } else { 0 },
        if vertical.is_some() { N / 2 - 1 } else { 0 },
    );
    let (ww, wh) = (
        w + if horizontal.is_some() { N - 1 } else { 0 },
        h + if vertical.is_some() { N - 1 } else { 0 },
    );
    let (pw, ph) = (plane.width as isize, plane.height as isize);
    let (left, top) = (x0 - halo_x as isize, y0 - halo_y as isize);
    let inside = left >= 0 && top >= 0 && left + ww as isize <= pw && top + wh as isize <= ph;
    // Borrow an in-bounds block at the plane's stride. Only borders need a
    // scratch copy, and their horizontal extension is the same on every row.
    let (window, stride) = if inside {
        let start = top as usize * plane.width + left as usize;
        (&plane.data[start..], plane.width)
    } else {
        let window_len = ww * wh;
        scratch.window.resize(window_len + SIMD_PAD, 0);
        let start = left.clamp(0, pw) as usize;
        let end = (left + ww as isize).clamp(0, pw) as usize;
        let prefix = (-left).clamp(0, ww as isize) as usize;
        let suffix = prefix + end - start;
        for (r, row) in scratch.window[..window_len]
            .chunks_exact_mut(ww)
            .enumerate()
        {
            let sy = (top + r as isize).clamp(0, ph - 1) as usize;
            let line = &plane.data[sy * plane.width..(sy + 1) * plane.width];
            row[..prefix].fill(line[0]);
            row[prefix..suffix].copy_from_slice(&line[start..end]);
            row[suffix..].fill(line[plane.width - 1]);
        }
        (scratch.window.as_slice(), ww)
    };
    #[cfg(test)]
    let simd = !tests::FORCE_SCALAR.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let simd = true;
    if simd
        && (horizontal.is_some() || vertical.is_some())
        && w.is_multiple_of(4)
        && window.len() >= (wh - 1) * stride + ww + w % 8
    {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            // SAFETY: each SIMD load is within the window, `w` is a multiple
            // of 4 and `out` holds the block.
            unsafe {
                crate::neon::interpolate_neon(
                    window,
                    stride,
                    horizontal,
                    vertical,
                    w,
                    h,
                    bit_depth,
                    rounding,
                    &mut out[..w * h],
                    &mut scratch.rows16,
                );
            }
            return;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("sse4.1") {
            // SAFETY: as above, and SSE4.1 was just detected.
            unsafe {
                crate::avx::interpolate_sse41(
                    window,
                    stride,
                    horizontal,
                    vertical,
                    w,
                    h,
                    bit_depth,
                    rounding,
                    &mut out[..w * h],
                    &mut scratch.rows16,
                );
            }
            return;
        }
    }
    filter_scalar(
        window,
        stride,
        horizontal,
        vertical,
        w,
        h,
        bit_depth,
        rounding,
        out,
        &mut scratch.rows,
    );
}

/// Filter the `window` (the block with its filter margins) into `out`
/// (stride `w`). The SIMD kernels must match this implementation bit for bit.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(all(target_arch = "aarch64", feature = "neon"), allow(dead_code))]
fn filter_scalar<const N: usize>(
    window: &[u16],
    stride: usize,
    horizontal: Option<&[i32; N]>,
    vertical: Option<&[i32; N]>,
    w: usize,
    h: usize,
    bit_depth: u8,
    rounding: Rounding,
    out: &mut [u16],
    rows: &mut Vec<i32>,
) {
    let wh = h + if vertical.is_some() { N - 1 } else { 0 };
    // Intermediates are kept at 14 bits; `shift1` brings a first filter pass
    // there, and the final rounding returns to the sample range.
    let shift1 = u32::from(bit_depth) - 8;
    let shift = 14 - u32::from(bit_depth);
    let max = (1i32 << bit_depth) - 1;
    // The weighted rounding of a 14-bit intermediate; with the unit weight
    // this is exactly the default `(v + 2^(shift−1)) >> shift`.
    let (weight, intermediate) = match rounding {
        Rounding::Samples(weight) => (weight, false),
        Rounding::Intermediate => (Weight::UNIT, true),
    };
    let log2_wd = weight.log2_denom + shift;
    let round = 1 << (log2_wd - 1);
    let finish = |v: i32| {
        if intermediate {
            (v - MID_OFFSET) as i16 as u16
        } else {
            (((v * weight.w + round) >> log2_wd) + weight.offset).clamp(0, max) as u16
        }
    };
    // Each pass accumulates one tap at a time over a whole row, which is the
    // shape the compiler vectorizes.
    let mut acc = [0i32; MAX_BLOCK];
    let acc = &mut acc[..w];
    match (horizontal, vertical) {
        (None, None) if rounding == Rounding::PLAIN => {
            for (r, dst) in out[..w * h].chunks_exact_mut(w).enumerate() {
                dst.copy_from_slice(&window[r * stride..r * stride + w]);
            }
        }
        (None, None) => {
            for (r, dst) in out[..w * h].chunks_exact_mut(w).enumerate() {
                for (px, &v) in dst.iter_mut().zip(&window[r * stride..r * stride + w]) {
                    *px = finish(i32::from(v) << shift);
                }
            }
        }
        (Some(f), None) => {
            for (r, dst) in out[..w * h].chunks_exact_mut(w).enumerate() {
                fir(acc, f, |i| &window[r * stride + i..r * stride + i + w]);
                for (px, &v) in dst.iter_mut().zip(acc.iter()) {
                    *px = finish(v >> shift1);
                }
            }
        }
        (None, Some(f)) => {
            for (r, dst) in out[..w * h].chunks_exact_mut(w).enumerate() {
                fir(acc, f, |i| &window[(r + i) * stride..(r + i) * stride + w]);
                for (px, &v) in dst.iter_mut().zip(acc.iter()) {
                    *px = finish(v >> shift1);
                }
            }
        }
        (Some(fh), Some(fv)) => {
            rows.resize(w * wh, 0);
            for (r, dst) in rows.chunks_exact_mut(w).enumerate() {
                fir(acc, fh, |i| &window[r * stride + i..r * stride + i + w]);
                for (v, &a) in dst.iter_mut().zip(acc.iter()) {
                    *v = a >> shift1;
                }
            }
            let rows = &*rows;
            for (r, dst) in out[..w * h].chunks_exact_mut(w).enumerate() {
                acc.fill(0);
                for (i, &tap) in fv.iter().enumerate() {
                    if tap != 0 {
                        let line = &rows[(r + i) * w..(r + i + 1) * w];
                        for (a, &v) in acc.iter_mut().zip(line) {
                            *a += tap * v;
                        }
                    }
                }
                for (px, &v) in dst.iter_mut().zip(acc.iter()) {
                    *px = finish(v >> 6);
                }
            }
        }
    }
}

/// Widest block the interpolation handles (a CTU).
const MAX_BLOCK: usize = 64;

/// `acc[c] = Σ taps[i] · line(i)[c]`: one FIR output row, where `line(i)` is
/// the row of input samples tap `i` reads.
#[inline]
fn fir<'a, const N: usize>(acc: &mut [i32], taps: &[i32; N], line: impl Fn(usize) -> &'a [u16]) {
    acc.fill(0);
    for (i, &tap) in taps.iter().enumerate() {
        if tap != 0 {
            for (a, &v) in acc.iter_mut().zip(line(i)) {
                *a += tap * i32::from(v);
            }
        }
    }
}

/// Predict the `n`×`n` luma block at (`x`, `y`) displaced by `mv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_luma(
    plane: RefPlane<'_>,
    x: usize,
    y: usize,
    mv: Mv,
    n: usize,
    bit_depth: u8,
    rounding: Rounding,
    out: &mut [u16],
    scratch: &mut InterpScratch,
) {
    let (fx, fy) = ((mv.0 & 3) as usize, (mv.1 & 3) as usize);
    interpolate(
        plane,
        x as isize + isize::from(mv.0 >> 2),
        y as isize + isize::from(mv.1 >> 2),
        (fx != 0).then(|| &LUMA_FILTER[fx]),
        (fy != 0).then(|| &LUMA_FILTER[fy]),
        n,
        n,
        bit_depth,
        rounding,
        out,
        scratch,
    );
}

/// Predict the `w`×`h` chroma block at chroma position (`x`, `y`) for the
/// luma vector `mv`; `sub_w` / `sub_h` are the chroma subsampling factors.
/// The chroma vector is in eighths of a chroma sample (§8.5.3.2.10).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_chroma(
    plane: RefPlane<'_>,
    x: usize,
    y: usize,
    mv: Mv,
    w: usize,
    h: usize,
    sub_w: usize,
    sub_h: usize,
    bit_depth: u8,
    rounding: Rounding,
    out: &mut [u16],
    scratch: &mut InterpScratch,
) {
    let mvc_x = isize::from(mv.0) * 2 / sub_w as isize;
    let mvc_y = isize::from(mv.1) * 2 / sub_h as isize;
    let (fx, fy) = ((mvc_x & 7) as usize, (mvc_y & 7) as usize);
    interpolate(
        plane,
        x as isize + (mvc_x >> 3),
        y as isize + (mvc_y >> 3),
        (fx != 0).then(|| &CHROMA_FILTER[fx]),
        (fy != 0).then(|| &CHROMA_FILTER[fy]),
        w,
        h,
        bit_depth,
        rounding,
        out,
        scratch,
    );
}

/// Bi-directional prediction (§8.5.3.3.4.2): the rounded average of two
/// [`Rounding::Intermediate`] predictions, as samples.
pub(crate) fn bi_average(a: &[u16], b: &[u16], bit_depth: u8, out: &mut [u16]) {
    assert!(a.len() == out.len() && b.len() == out.len());
    #[cfg(test)]
    let simd = !tests::FORCE_SCALAR.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let simd = true;
    if simd {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            // SAFETY: the three runs have the same length (asserted).
            unsafe { crate::neon::bi_average_neon(a, b, bit_depth, out) };
            return;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("sse4.1") {
            // SAFETY: as above, and SSE4.1 was just detected.
            unsafe { crate::avx::bi_average_sse41(a, b, bit_depth, out) };
            return;
        }
    }
    #[allow(unreachable_code)]
    bi_average_scalar(a, b, bit_depth, out);
}

/// The reference implementation of [`bi_average`]; also the tail of the SIMD
/// kernels.
pub(crate) fn bi_average_scalar(a: &[u16], b: &[u16], bit_depth: u8, out: &mut [u16]) {
    let shift = 15 - u32::from(bit_depth);
    let round = (1 << (shift - 1)) + 2 * MID_OFFSET;
    let max = (1i32 << bit_depth) - 1;
    for ((px, &p), &q) in out.iter_mut().zip(a).zip(b) {
        let sum = i32::from(p as i16) + i32::from(q as i16);
        *px = ((sum + round) >> shift).clamp(0, max) as u16;
    }
}

/// Sum of absolute differences between two `w`×`h` blocks, given as sample
/// runs with row strides `a_stride` and `b_stride`.
#[inline]
fn sad(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    assert!(h > 0 && a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w);
    #[cfg(test)]
    let simd = !tests::FORCE_SCALAR.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let simd = true;
    if simd && w.is_multiple_of(8) {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            // SAFETY: 8 divides `w` and both runs hold the block (asserted).
            return unsafe { crate::neon::sad_neon(a, a_stride, b, b_stride, w, h) };
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("sse4.1") {
            // SAFETY: as above, and SSE4.1 was just detected.
            return unsafe { crate::avx::sad_sse41(a, a_stride, b, b_stride, w, h) };
        }
    }
    #[allow(unreachable_code)]
    sad_scalar(a, a_stride, b, b_stride, w, h)
}

/// The reference implementation of [`sad`].
#[inline]
fn sad_scalar(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    (0..h)
        .map(|r| {
            a[r * a_stride..r * a_stride + w]
                .iter()
                .zip(&b[r * b_stride..r * b_stride + w])
                .map(|(&p, &q)| u32::from(p.abs_diff(q)))
                .sum::<u32>()
        })
        .sum()
}

/// SAD between two `n`×`n` blocks of stride `n`.
pub(crate) fn block_sad(a: &[u16], b: &[u16], n: usize) -> u32 {
    sad(a, n, b, n, n, n)
}

/// SAD between the `n`×`n` block `src` and the block of `plane` whose top-left
/// sample is (`x`, `y`); samples outside the plane repeat its border.
fn sad_at(plane: RefPlane<'_>, src: &[u16], x: isize, y: isize, n: usize) -> u32 {
    let (pw, ph) = (plane.width as isize, plane.height as isize);
    if x >= 0 && y >= 0 && x + n as isize <= pw && y + n as isize <= ph {
        let start = y as usize * plane.width + x as usize;
        return sad(src, n, &plane.data[start..], plane.width, n, n);
    }
    let mut total = 0;
    for (r, line) in src.chunks_exact(n).enumerate() {
        let row = (y + r as isize).clamp(0, ph - 1) as usize * plane.width;
        for (c, &p) in line.iter().enumerate() {
            let q = plane.data[row + (x + c as isize).clamp(0, pw - 1) as usize];
            total += u32::from(p.abs_diff(q));
        }
    }
    total
}

/// Approximate bits of one `mvd_coding()` component.
#[inline]
fn mvd_component_bits(d: i32) -> u32 {
    match d.unsigned_abs() {
        0 => 1,
        a => 2 * (32 - a.leading_zeros()) + 1,
    }
}

/// Approximate bits to code `mv` against the cheaper of the two predictors.
#[inline]
pub(crate) fn mvd_bits(mv: Mv, mvp: &[Mv; 2]) -> u32 {
    best_mvd(mv, mvp).2
}

/// The predictor index and difference selected by the motion search's bit cost.
pub(crate) fn amvp_coding(mv: Mv, mvp: &[Mv; 2]) -> (u8, (i32, i32)) {
    let (idx, mvd, _) = best_mvd(mv, mvp);
    (idx, mvd)
}

#[inline]
fn best_mvd(mv: Mv, mvp: &[Mv; 2]) -> (u8, (i32, i32), u32) {
    let [a, b] = mvp.map(|p| {
        // H.265 derives the luma MV from MVP + MVD modulo 2^16.
        let diff = (
            i32::from(mv.0.wrapping_sub(p.0)),
            i32::from(mv.1.wrapping_sub(p.1)),
        );
        (
            diff,
            mvd_component_bits(diff.0) + mvd_component_bits(diff.1),
        )
    });
    if b.1 < a.1 {
        (1, b.0, b.1)
    } else {
        (0, a.0, a.1)
    }
}

/// Whole samples the search may move away from its best starting point.
const SEARCH_RANGE: i32 = 48;

/// Motion search for the `n`×`n` luma block `src` (stride `n`) at (`x`, `y`):
/// the best of the `starts` (quarter-sample vectors, rounded to whole
/// samples), refined by a shrinking diamond and then to half and quarter
/// sample precision. Candidates are ranked by SAD + `lambda`·(vector bits
/// against `mvp`); the best vector is returned with that cost. `pred` is
/// scratch for one block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    plane: RefPlane<'_>,
    src: &[u16],
    x: usize,
    y: usize,
    n: usize,
    bit_depth: u8,
    weight: Weight,
    starts: &[Mv],
    mvp: &[Mv; 2],
    lambda: f32,
    pred: &mut [u16],
    scratch: &mut InterpScratch,
) -> (Mv, f32) {
    let src = &src[..n * n];
    let cost = |mv: Mv, pred: &mut [u16], scratch: &mut InterpScratch| -> f32 {
        let distortion = if (mv.0 | mv.1) & 3 == 0 && weight == Weight::UNIT {
            // Whole-sample position: compare against the plane directly.
            let (rx, ry) = (
                x as isize + isize::from(mv.0 >> 2),
                y as isize + isize::from(mv.1 >> 2),
            );
            sad_at(plane, src, rx, ry, n)
        } else {
            predict_luma(
                plane,
                x,
                y,
                mv,
                n,
                bit_depth,
                Rounding::Samples(weight),
                pred,
                scratch,
            );
            sad(src, n, &pred[..n * n], n, n, n)
        };
        distortion as f32 + lambda * mvd_bits(mv, mvp) as f32
    };
    // Keep the block within a block's width of the picture and the vector
    // inside what the motion field can hold.
    let legal = |mv: (i32, i32)| -> bool {
        let (px, py) = (x as i32 + (mv.0 >> 2), y as i32 + (mv.1 >> 2));
        mv.0.abs() <= i32::from(MV_LIMIT)
            && mv.1.abs() <= i32::from(MV_LIMIT)
            && px >= -(n as i32)
            && py >= -(n as i32)
            && px <= plane.width as i32
            && py <= plane.height as i32
    };
    let whole = |mv: Mv| (i32::from(mv.0) & !3, i32::from(mv.1) & !3);
    let mut best = ((0i32, 0i32), cost((0, 0), pred, scratch));
    for (i, &start) in starts.iter().enumerate() {
        let mv = whole(start);
        if mv == (0, 0) || !legal(mv) || starts[..i].iter().any(|&p| whole(p) == mv) {
            continue;
        }
        let c = cost((mv.0 as i16, mv.1 as i16), pred, scratch);
        if c < best.1 {
            best = (mv, c);
        }
    }
    let origin = best.0;
    let in_range = |mv: (i32, i32)| {
        legal(mv)
            && (mv.0 - origin.0).abs() <= 4 * SEARCH_RANGE
            && (mv.1 - origin.1).abs() <= 4 * SEARCH_RANGE
    };
    // Whole samples: a diamond of shrinking radius, re-centred on every
    // improvement.
    let mut axis_costs = None;
    for (step, diagonals) in [(32, false), (16, false), (8, false), (4, true)] {
        for _ in 0..16 {
            let centre = best.0;
            let mut local_axis_costs = [f32::INFINITY; 4];
            for (i, (dx, dy)) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
            .into_iter()
            .enumerate()
            {
                if dx != 0 && dy != 0 && !diagonals {
                    continue;
                }
                let mv = (centre.0 + dx * step, centre.1 + dy * step);
                if !in_range(mv) {
                    continue;
                }
                let c = cost((mv.0 as i16, mv.1 as i16), pred, scratch);
                if i < 4 {
                    local_axis_costs[i] = c;
                }
                if c < best.1 {
                    best = (mv, c);
                }
            }
            if best.0 == centre {
                if step == 4 {
                    axis_costs = Some(local_axis_costs);
                }
                break;
            }
        }
    }
    // Sub-sample precision. Every candidate here costs an interpolation, so
    // the whole-sample costs around the best position first say where the
    // minimum lies: a parabola through three costs along each axis. Most
    // blocks sit on a whole-sample minimum and are done without interpolating
    // anything.
    let (centre, centre_cost) = best;
    let mut at = |dx: i32, dy: i32, pred: &mut [u16], scratch: &mut InterpScratch| -> f32 {
        let mv = (centre.0 + dx, centre.1 + dy);
        if !in_range(mv) {
            return f32::INFINITY;
        }
        let c = cost((mv.0 as i16, mv.1 as i16), pred, scratch);
        if c < best.1 {
            best = (mv, c);
        }
        c
    };
    // Offset of the parabola's minimum in quarter samples, at most three.
    let vertex = |before: f32, after: f32| -> i32 {
        let curvature = before - 2.0 * centre_cost + after;
        if !curvature.is_finite() || curvature <= 0.0 {
            return 0;
        }
        // Rounded away from zero from 0.3 on: a wasted interpolation is
        // cheaper than a missed quarter sample, which every later picture
        // would inherit.
        let quarters = 2.0 * (before - after) / curvature;
        ((quarters.abs() + 0.2).round().min(3.0) * quarters.signum()) as i32
    };
    let [right, left, down, up] = axis_costs.unwrap_or_else(|| {
        let left = at(-4, 0, pred, scratch);
        let right = at(4, 0, pred, scratch);
        let up = at(0, -4, pred, scratch);
        let down = at(0, 4, pred, scratch);
        [right, left, down, up]
    });
    let (qx, qy) = (vertex(left, right), vertex(up, down));
    if (qx, qy) != (0, 0) {
        if at(qx, qy, pred, scratch) < centre_cost {
            // The estimate holds: settle the last quarter sample around it.
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                at(qx + dx, qy + dy, pred, scratch);
            }
        } else if qx != 0 && qy != 0 {
            at(qx, 0, pred, scratch);
            at(0, qy, pred, scratch);
        }
    }
    ((best.0.0 as i16, best.0.1 as i16), best.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Makes [`interpolate`] take the scalar reference path.
        pub(super) static FORCE_SCALAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn noise_plane(width: usize, height: usize, bit_depth: u8, seed: u32) -> Vec<u16> {
        let mut state = seed;
        (0..width * height)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Mostly full-range noise, with runs of the extremes.
                match (state >> 8) & 15 {
                    0 => 0,
                    1 => (1u16 << bit_depth) - 1,
                    _ => ((state >> 16) as u16) & ((1 << bit_depth) - 1),
                }
            })
            .collect()
    }

    /// The SIMD kernels against the scalar reference: every phase, block
    /// shape, bit depth, with and without weights, inside the picture and
    /// across each border.
    #[test]
    fn simd_interpolation_matches_scalar() {
        let (pw, ph) = (80usize, 72usize);
        let weighted = |w: i32, offset: i32| {
            Rounding::Samples(Weight {
                w,
                log2_denom: 6,
                offset,
            })
        };
        let roundings = [
            Rounding::PLAIN,
            weighted(96, 5),
            weighted(37, -9),
            weighted(191, 0),
            Rounding::Intermediate,
        ];
        let mut scratch = InterpScratch::default();
        let mut checked = 0;
        for bit_depth in [8u8, 10, 12] {
            let data = noise_plane(pw, ph, bit_depth, u32::from(bit_depth));
            let plane = RefPlane {
                data: &data,
                width: pw,
                height: ph,
            };
            for rounding in roundings {
                let weight = match rounding {
                    Rounding::Samples(weight) => Rounding::Samples(Weight {
                        offset: weight.offset << (bit_depth - 8),
                        ..weight
                    }),
                    Rounding::Intermediate => rounding,
                };
                // (x, y) positions: interior and straddling every border.
                for (x, y) in [(24usize, 20usize), (0, 0), (70, 3), (5, 66), (76, 68)] {
                    for n in [8usize, 16, 32, 64] {
                        for fy in 0..4 {
                            for fx in 0..4 {
                                let mv = (fx - 8, fy + 4);
                                let mut got = vec![0u16; n * n];
                                let mut want = vec![0u16; n * n];
                                predict_luma(
                                    plane,
                                    x,
                                    y,
                                    mv,
                                    n,
                                    bit_depth,
                                    weight,
                                    &mut got,
                                    &mut scratch,
                                );
                                FORCE_SCALAR.with(|f| f.set(true));
                                predict_luma(
                                    plane,
                                    x,
                                    y,
                                    mv,
                                    n,
                                    bit_depth,
                                    weight,
                                    &mut want,
                                    &mut scratch,
                                );
                                FORCE_SCALAR.with(|f| f.set(false));
                                assert_eq!(
                                    got, want,
                                    "luma {n} at ({x},{y}) mv {mv:?} {bit_depth}-bit {weight:?}"
                                );
                                checked += 1;
                            }
                        }
                    }
                    // Chroma shapes: 4:2:0, 4:2:2 (tall) and 4:4:4.
                    for (w, h, sub_w, sub_h) in [
                        (4usize, 4usize, 2usize, 2usize),
                        (4, 8, 2, 1),
                        (8, 8, 2, 2),
                        (8, 16, 2, 1),
                        (16, 16, 2, 2),
                        (32, 32, 1, 1),
                    ] {
                        for fy in 0..8 {
                            for fx in 0..8 {
                                // Chroma phase is the vector's low bits scaled by the subsampling.
                                let mv =
                                    ((fx * sub_w as i16) / 2 - 16, (fy * sub_h as i16) / 2 + 8);
                                let mut got = vec![0u16; w * h];
                                let mut want = vec![0u16; w * h];
                                predict_chroma(
                                    plane,
                                    x / 2,
                                    y / 2,
                                    mv,
                                    w,
                                    h,
                                    sub_w,
                                    sub_h,
                                    bit_depth,
                                    weight,
                                    &mut got,
                                    &mut scratch,
                                );
                                FORCE_SCALAR.with(|f| f.set(true));
                                predict_chroma(
                                    plane,
                                    x / 2,
                                    y / 2,
                                    mv,
                                    w,
                                    h,
                                    sub_w,
                                    sub_h,
                                    bit_depth,
                                    weight,
                                    &mut want,
                                    &mut scratch,
                                );
                                FORCE_SCALAR.with(|f| f.set(false));
                                assert_eq!(
                                    got, want,
                                    "chroma {w}x{h} at ({x},{y}) mv {mv:?} {bit_depth}-bit {weight:?}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000);
    }

    #[test]
    fn simd_sad_matches_scalar() {
        let a = noise_plane(96, 80, 12, 1);
        let b = noise_plane(96, 80, 12, 2);
        for (w, h) in [
            (8usize, 8usize),
            (16, 16),
            (32, 32),
            (64, 64),
            (8, 1),
            (24, 5),
            (40, 3),
        ] {
            for (a_stride, b_stride) in [(w, w), (96, w), (w, 96), (96, 96)] {
                assert_eq!(
                    sad(&a, a_stride, &b, b_stride, w, h),
                    sad_scalar(&a, a_stride, &b, b_stride, w, h),
                    "{w}x{h} strides {a_stride}/{b_stride}"
                );
            }
        }
        // The largest possible sum: 64×64 samples 4095 apart.
        let (lo, hi) = (vec![0u16; 4096], vec![4095u16; 4096]);
        assert_eq!(sad(&lo, 64, &hi, 64, 64, 64), 4096 * 4095);
    }

    #[test]
    fn four_sample_filter_at_end_of_reference() {
        let data = noise_plane(8, 8, 10, 1);
        let mut padded = data.clone();
        padded.extend([1023; 4]);
        let predict = |data: &[u16]| {
            let mut out = [0; 16];
            // Last tap of the bottom-right output is the last plane sample.
            predict_chroma(
                RefPlane {
                    data,
                    width: 8,
                    height: 8,
                },
                2,
                2,
                (1, 1),
                4,
                4,
                2,
                2,
                10,
                Rounding::PLAIN,
                &mut out,
                &mut InterpScratch::default(),
            );
            out
        };
        assert_eq!(predict(&data), predict(&padded));
    }

    #[test]
    fn amvp_coding_minimizes_bits_and_wraps_the_difference() {
        let mvp = [(-2, -2), (-5, 0)];
        assert_eq!(amvp_coding((0, 0), &mvp), (1, (5, 0)));
        assert_eq!(mvd_bits((0, 0), &mvp), 8);

        let mv = (-8188, 8188);
        let mvp = [(i16::MAX, i16::MIN); 2];
        let (idx, diff) = amvp_coding(mv, &mvp);
        assert_eq!(diff, (24581, -24580));
        let p = mvp[usize::from(idx)];
        assert_eq!(
            (
                p.0.wrapping_add(diff.0 as i16),
                p.1.wrapping_add(diff.1 as i16)
            ),
            mv
        );
    }

    #[test]
    fn motion_entries_round_trip() {
        let vectors = [
            (0, 0),
            (1, -1),
            (-4, 4),
            (MV_LIMIT, -MV_LIMIT),
            (-MV_LIMIT, MV_LIMIT),
            (-3, 0),
        ];
        for mv in vectors {
            for other in vectors {
                for pu in [
                    PuMotion::uni(0, mv),
                    PuMotion::uni(1, mv),
                    PuMotion::bi(mv, other),
                ] {
                    for (skip, luma_cbf) in [(false, false), (true, false), (false, true)] {
                        let m = Motion { pu, skip, luma_cbf };
                        assert_ne!(m.pack(), BV_INTRA);
                        assert_eq!(Motion::unpack(m.pack()), Some(m));
                    }
                }
            }
        }
        assert_eq!(Motion::unpack(BV_INTRA), None);
    }

    #[test]
    fn boundary_strength_rules() {
        let entry = |pu, luma_cbf| {
            Motion {
                pu,
                skip: false,
                luma_cbf,
            }
            .pack()
        };
        let e = |mv, luma_cbf| entry(PuMotion::uni(0, mv), luma_cbf);
        assert_eq!(boundary_strength(e((0, 0), false), BV_INTRA), 2);
        assert_eq!(boundary_strength(e((0, 0), false), e((0, 0), false)), 0);
        assert_eq!(boundary_strength(e((0, 0), false), e((3, -3), false)), 0);
        assert_eq!(boundary_strength(e((0, 0), false), e((4, 0), false)), 1);
        assert_eq!(boundary_strength(e((0, 0), false), e((0, -4), false)), 1);
        assert_eq!(boundary_strength(e((8, 8), true), e((8, 8), false)), 1);
        // Another picture, or another number of pictures.
        let l1 = entry(PuMotion::uni(1, (0, 0)), false);
        let bi = |mv0, mv1| entry(PuMotion::bi(mv0, mv1), false);
        assert_eq!(boundary_strength(e((0, 0), false), l1), 1);
        assert_eq!(boundary_strength(e((0, 0), false), bi((0, 0), (0, 0))), 1);
        assert_eq!(boundary_strength(l1, bi((0, 0), (0, 0))), 1);
        // Two pictures each: either vector a whole sample apart.
        assert_eq!(
            boundary_strength(bi((1, 1), (-2, 0)), bi((2, 0), (1, 3))),
            0
        );
        assert_eq!(
            boundary_strength(bi((1, 1), (-2, 0)), bi((5, 1), (-2, 0))),
            1
        );
        assert_eq!(
            boundary_strength(bi((1, 1), (-2, 0)), bi((1, 1), (-2, 4))),
            1
        );
    }

    /// A motion field with the five merge / AMVP neighbors of the 8×8 block
    /// at (16, 16): A1 = (15, 23), B1 = (23, 15), B0 = (24, 15),
    /// A0 = (15, 24), B2 = (15, 15).
    fn field(
        a1: Option<PuMotion>,
        b1: Option<PuMotion>,
        b0: Option<PuMotion>,
        a0: Option<PuMotion>,
        b2: Option<PuMotion>,
    ) -> impl Fn(i32, i32) -> Option<PuMotion> {
        move |x: i32, y: i32| match (x, y) {
            (15, 23) => a1,
            (23, 15) => b1,
            (24, 15) => b0,
            (15, 24) => a0,
            (15, 15) => b2,
            _ => None,
        }
    }

    #[test]
    fn merge_list_prunes_and_pads() {
        let l0 = |mv| PuMotion::uni(0, mv);
        let m = |f: &dyn Fn(i32, i32) -> Option<PuMotion>| merge_candidates(16, 16, 8, false, f);
        // Nothing available: zero vectors.
        assert_eq!(m(&field(None, None, None, None, None)), [l0((0, 0)); 3]);
        // B1 equal to A1 is pruned; B0 is compared with B1, not A1.
        let a = Some(l0((4, 0)));
        assert_eq!(
            m(&field(a, a, a, None, None)),
            [l0((4, 0)), l0((0, 0)), l0((0, 0))]
        );
        assert_eq!(
            m(&field(a, Some(l0((8, 0))), a, None, None)),
            [l0((4, 0)), l0((8, 0)), l0((4, 0))]
        );
        // B2 only joins when fewer than four of the others did.
        assert_eq!(
            m(&field(None, None, None, None, Some(l0((2, 2))))),
            [l0((2, 2)), l0((0, 0)), l0((0, 0))]
        );
    }

    #[test]
    fn b_slice_merge_list_combines_and_pads_with_bi_zero() {
        let m = |f: &dyn Fn(i32, i32) -> Option<PuMotion>| merge_candidates(16, 16, 8, true, f);
        let zero = PuMotion::bi((0, 0), (0, 0));
        assert_eq!(m(&field(None, None, None, None, None)), [zero; 3]);
        // One candidate: nothing to combine.
        let p = PuMotion::uni(0, (4, 0));
        assert_eq!(m(&field(Some(p), None, None, None, None)), [p, zero, zero]);
        // List 0 of the first with list 1 of the second.
        let q = PuMotion::uni(1, (-8, 2));
        assert_eq!(
            m(&field(Some(p), Some(q), None, None, None)),
            [p, q, PuMotion::bi((4, 0), (-8, 2))]
        );
        // The other order only yields the second combination (combIdx 1).
        assert_eq!(
            m(&field(Some(q), Some(p), None, None, None)),
            [q, p, PuMotion::bi((4, 0), (-8, 2))]
        );
        // Two list 0 candidates cannot be combined.
        let r = PuMotion::uni(0, (0, 12));
        assert_eq!(m(&field(Some(p), Some(r), None, None, None)), [p, r, zero]);
        // Bi-predictive candidates combine across: L0 of one, L1 of the other.
        let (u, v) = (PuMotion::bi((1, 1), (2, 2)), PuMotion::bi((3, 3), (4, 4)));
        assert_eq!(
            m(&field(Some(u), Some(v), None, None, None)),
            [u, v, PuMotion::bi((1, 1), (4, 4))]
        );
        // A pruned duplicate does not count as a second candidate.
        assert_eq!(
            m(&field(Some(u), Some(u), None, None, None)),
            [u, zero, zero]
        );
    }

    #[test]
    fn amvp_takes_own_list_first_and_scales_the_other() {
        let amvp = |list, distance, f: &dyn Fn(i32, i32) -> Option<PuMotion>| {
            amvp_predictors(16, 16, 8, list, distance, f)
        };
        let l0 = |mv| Some(PuMotion::uni(0, mv));
        let l1 = |mv| Some(PuMotion::uni(1, mv));
        // One list (a P slice): left then above, duplicates dropped, and a
        // lone candidate from above listed once.
        let p = [1, 1];
        assert_eq!(
            amvp(0, p, &field(l0((4, 0)), l0((8, 0)), None, None, None)),
            [(4, 0), (8, 0)]
        );
        assert_eq!(
            amvp(0, p, &field(l0((4, 0)), l0((4, 0)), None, None, None)),
            [(4, 0), (0, 0)]
        );
        assert_eq!(
            amvp(0, p, &field(None, l0((8, 0)), None, None, None)),
            [(8, 0), (0, 0)]
        );
        // A0 comes before A1, B0 before B1 before B2.
        assert_eq!(
            amvp(
                0,
                p,
                &field(
                    l0((4, 0)),
                    l0((8, 0)),
                    l0((12, 0)),
                    l0((16, 0)),
                    l0((20, 0))
                )
            ),
            [(16, 0), (12, 0)]
        );
        // B slice, pictures one before and one after: the other list's
        // vector is mirrored.
        let b = [1, -1];
        assert_eq!(
            amvp(0, b, &field(l1((8, -4)), None, None, None, None)),
            [(-8, 4), (0, 0)]
        );
        assert_eq!(
            amvp(1, b, &field(l1((8, -4)), None, None, None, None)),
            [(8, -4), (0, 0)]
        );
        // The target list's own vector wins over an earlier neighbor that
        // only has the other list's.
        assert_eq!(
            amvp(0, b, &field(l0((4, 4)), None, None, l1((8, 8)), None)),
            [(4, 4), (0, 0)]
        );
        // With a neighbor on the left, vectors above are never scaled.
        assert_eq!(
            amvp(0, b, &field(l0((4, 4)), l1((8, 8)), None, None, None)),
            [(4, 4), (0, 0)]
        );
        // Without one, the unscaled vector from above stands in on the left
        // and the candidate from above is derived again with scaling.
        assert_eq!(
            amvp(0, b, &field(None, l0((8, 8)), l1((2, 2)), None, None)),
            [(8, 8), (-2, -2)]
        );
        // Unequal distances: 1 before and 3 after scales by −1/3 and −3.
        let uneven = [1, -3];
        assert_eq!(
            amvp(0, uneven, &field(l1((12, -9)), None, None, None, None)),
            [(-4, 3), (0, 0)]
        );
        assert_eq!(
            amvp(1, uneven, &field(l0((4, -3)), None, None, None, None)),
            [(-12, 9), (0, 0)]
        );
    }

    #[test]
    fn simd_bi_average_matches_scalar_and_the_definition() {
        for bit_depth in [8u8, 10, 12] {
            let max = (1u32 << bit_depth) - 1;
            // Whole-sample intermediates of two blocks average to the
            // rounded mean of the samples.
            let p = noise_plane(40, 3, bit_depth, 7);
            let q = noise_plane(40, 3, bit_depth, 8);
            let mid = |v: &[u16]| -> Vec<u16> {
                v.iter()
                    .map(|&s| ((i32::from(s) << (14 - bit_depth)) - MID_OFFSET) as i16 as u16)
                    .collect()
            };
            let mut out = vec![0u16; p.len()];
            bi_average(&mid(&p), &mid(&q), bit_depth, &mut out);
            for ((&o, &a), &b) in out.iter().zip(&p).zip(&q) {
                assert_eq!(u32::from(o), (u32::from(a) + u32::from(b) + 1) >> 1);
            }
            // Arbitrary intermediates, overshoots included, at every length.
            let mut state = u32::from(bit_depth);
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 12) % 50_000) as i32 - 25_000
            };
            for len in [1usize, 4, 7, 8, 16, 23, 64, 4096] {
                let a: Vec<u16> = (0..len).map(|_| next() as i16 as u16).collect();
                let b: Vec<u16> = (0..len).map(|_| next() as i16 as u16).collect();
                let mut got = vec![0u16; len];
                let mut want = vec![0u16; len];
                bi_average(&a, &b, bit_depth, &mut got);
                bi_average_scalar(&a, &b, bit_depth, &mut want);
                assert_eq!(got, want, "{bit_depth}-bit, {len} samples");
                assert!(got.iter().all(|&v| u32::from(v) <= max));
            }
        }
    }

    #[test]
    fn intermediate_of_a_fractional_block_rounds_to_its_samples() {
        // Averaging a prediction with itself must give the uni-directional
        // prediction, wherever that is not clipped: the intermediate carries
        // the same value at higher precision.
        for bit_depth in [8u8, 10, 12] {
            let data: Vec<u16> = noise_plane(48, 48, bit_depth, 3)
                .iter()
                .map(|&v| v / 2 + (1 << (bit_depth - 2)))
                .collect();
            let plane = RefPlane {
                data: &data,
                width: 48,
                height: 48,
            };
            let mut scratch = InterpScratch::default();
            for mv in [(0, 0), (1, 0), (0, 2), (3, 1), (2, 2)] {
                let mut uni = [0u16; 256];
                let mut mid = [0u16; 256];
                let mut bi = [0u16; 256];
                predict_luma(
                    plane,
                    16,
                    16,
                    mv,
                    16,
                    bit_depth,
                    Rounding::PLAIN,
                    &mut uni,
                    &mut scratch,
                );
                predict_luma(
                    plane,
                    16,
                    16,
                    mv,
                    16,
                    bit_depth,
                    Rounding::Intermediate,
                    &mut mid,
                    &mut scratch,
                );
                bi_average(&mid, &mid, bit_depth, &mut bi);
                assert_eq!(uni, bi, "{bit_depth}-bit mv {mv:?}");
            }
        }
    }

    #[test]
    fn whole_sample_prediction_copies_and_extends_the_border() {
        let (w, h) = (16usize, 12usize);
        let data: Vec<u16> = (0..w * h).map(|i| (i % 251) as u16).collect();
        let plane = RefPlane {
            data: &data,
            width: w,
            height: h,
        };
        let mut out = [0u16; 16];
        let mut scratch = InterpScratch::default();
        predict_luma(
            plane,
            4,
            4,
            (8, -4),
            4,
            8,
            Rounding::PLAIN,
            &mut out,
            &mut scratch,
        );
        for r in 0..4 {
            assert_eq!(
                out[r * 4..r * 4 + 4],
                data[(3 + r) * w + 6..(3 + r) * w + 10]
            );
        }
        // Far outside to the top-left: every sample is the corner.
        predict_luma(
            plane,
            0,
            0,
            (-400, -400),
            4,
            8,
            Rounding::PLAIN,
            &mut out,
            &mut scratch,
        );
        assert_eq!(out, [data[0]; 16]);
    }

    #[test]
    fn fractional_prediction_of_a_flat_block_is_flat() {
        // Every filter sums to 64, so a constant plane must stay constant at
        // every phase and bit depth.
        for (bit_depth, value) in [(8u8, 200u16), (10, 801), (12, 3001)] {
            let data = vec![value; 32 * 32];
            let plane = RefPlane {
                data: &data,
                width: 32,
                height: 32,
            };
            let mut scratch = InterpScratch::default();
            for fy in 0..4 {
                for fx in 0..4 {
                    let mut out = [0u16; 64];
                    predict_luma(
                        plane,
                        8,
                        8,
                        (fx, fy),
                        8,
                        bit_depth,
                        Rounding::PLAIN,
                        &mut out,
                        &mut scratch,
                    );
                    assert_eq!(out, [value; 64], "luma phase ({fx}, {fy})");
                    let mut out = [0u16; 16];
                    predict_chroma(
                        plane,
                        4,
                        4,
                        (fx, fy),
                        4,
                        4,
                        2,
                        2,
                        bit_depth,
                        Rounding::PLAIN,
                        &mut out,
                        &mut scratch,
                    );
                    assert_eq!(out, [value; 16], "chroma phase ({fx}, {fy})");
                }
            }
        }
    }

    /// Timing of the kernels, for comparing implementations:
    /// `cargo test --release --lib kernel_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_timing() {
        let (w, h) = (256usize, 256usize);
        let data: Vec<u16> = (0..w * h)
            .map(|i| ((i * 2_654_435_761) >> 7) as u16 & 255)
            .collect();
        let plane = RefPlane {
            data: &data,
            width: w,
            height: h,
        };
        let mut scratch = InterpScratch::default();
        let mut out = vec![0u16; 64 * 64];
        let mut sink = 0u64;
        for n in [8usize, 16, 32] {
            for (name, mv) in [
                ("whole", (4, 4)),
                ("horizontal", (5, 4)),
                ("vertical", (4, 6)),
                ("both", (5, 6)),
            ] {
                let reps = 2_000_000 / (n * n);
                let start = std::time::Instant::now();
                for i in 0..reps {
                    predict_luma(
                        plane,
                        64 + i % 32,
                        64,
                        mv,
                        n,
                        8,
                        Rounding::PLAIN,
                        &mut out,
                        &mut scratch,
                    );
                    sink += u64::from(out[i % (n * n)]);
                }
                let ns = start.elapsed().as_nanos() as f64 / reps as f64;
                eprintln!(
                    "luma {n:2}x{n:<2} {name:10} {ns:8.1} ns  ({:.2} ns/sample)",
                    ns / (n * n) as f64
                );
            }
            let reps = 4_000_000 / (n * n);
            let src: Vec<u16> = data[..n * n].to_vec();
            let start = std::time::Instant::now();
            for i in 0..reps {
                sink += u64::from(sad_at(plane, &src, (64 + i % 32) as isize, 64, n));
            }
            let ns = start.elapsed().as_nanos() as f64 / reps as f64;
            eprintln!(
                "sad  {n:2}x{n:<2}            {ns:8.1} ns  ({:.2} ns/sample)",
                ns / (n * n) as f64
            );
        }
        for n in [4usize, 8, 16] {
            let reps = 2_000_000 / (n * n);
            let start = std::time::Instant::now();
            for i in 0..reps {
                predict_chroma(
                    plane,
                    64 + i % 32,
                    64,
                    (5, 3),
                    n,
                    n,
                    2,
                    2,
                    8,
                    Rounding::PLAIN,
                    &mut out,
                    &mut scratch,
                );
                sink += u64::from(out[i % (n * n)]);
            }
            let ns = start.elapsed().as_nanos() as f64 / reps as f64;
            eprintln!(
                "chroma {n:2}x{n:<2} both     {ns:8.1} ns  ({:.2} ns/sample)",
                ns / (n * n) as f64
            );
        }
        assert!(sink > 0);
    }

    #[test]
    fn weighted_prediction_scales_and_offsets() {
        // A flat plane through weight 96/64 and an offset of 4 (8-bit units):
        // the same value at every phase, scaled with the bit depth.
        for (bit_depth, value) in [(8u8, 100i32), (10, 400), (12, 1600)] {
            let data = vec![value as u16; 32 * 32];
            let plane = RefPlane {
                data: &data,
                width: 32,
                height: 32,
            };
            let weight = Weight {
                w: 96,
                log2_denom: 6,
                offset: 4 << (bit_depth - 8),
            };
            let want = (value * 96 / 64 + (4 << (bit_depth - 8))) as u16;
            let mut scratch = InterpScratch::default();
            for (fx, fy) in [(0, 0), (2, 0), (0, 1), (3, 2)] {
                let mut out = [0u16; 64];
                predict_luma(
                    plane,
                    8,
                    8,
                    (fx, fy),
                    8,
                    bit_depth,
                    Rounding::Samples(weight),
                    &mut out,
                    &mut scratch,
                );
                assert_eq!(out, [want; 64], "{bit_depth}-bit phase ({fx}, {fy})");
            }
        }
        // The result is clipped to the sample range.
        let data = vec![250u16; 16 * 16];
        let plane = RefPlane {
            data: &data,
            width: 16,
            height: 16,
        };
        let weight = Weight {
            w: 96,
            log2_denom: 6,
            offset: 0,
        };
        let mut out = [0u16; 16];
        predict_luma(
            plane,
            4,
            4,
            (0, 0),
            4,
            8,
            Rounding::Samples(weight),
            &mut out,
            &mut InterpScratch::default(),
        );
        assert_eq!(out, [255; 16]);
    }

    #[test]
    fn search_finds_a_translation() {
        // A smooth texture shifted by (+5, -3) whole samples.
        let (w, h) = (96usize, 96usize);
        let tex = |x: i32, y: i32| -> u16 {
            let (x, y) = (x as f32, y as f32);
            (128.0
                + 60.0 * (0.21 * x).sin() * (0.17 * y).cos()
                + 40.0 * (0.09 * x + 0.13 * y).sin()) as u16
        };
        let reference: Vec<u16> = (0..h as i32)
            .flat_map(|y| (0..w as i32).map(move |x| tex(x, y)))
            .collect();
        let plane = RefPlane {
            data: &reference,
            width: w,
            height: h,
        };
        let (x, y, n) = (40usize, 40usize, 16usize);
        let src: Vec<u16> = (0..n as i32)
            .flat_map(|r| (0..n as i32).map(move |c| tex(x as i32 + c + 5, y as i32 + r - 3)))
            .collect();
        let mut pred = vec![0u16; n * n];
        let mut scratch = InterpScratch::default();
        let (mv, _) = search(
            plane,
            &src,
            x,
            y,
            n,
            8,
            Weight::UNIT,
            &[],
            &[(0, 0); 2],
            1.0,
            &mut pred,
            &mut scratch,
        );
        assert_eq!(mv, (20, -12));
        // The same with half a sample more to the right, and with a quarter
        // sample less upwards.
        for (shift, want) in [((5.5f32, -3.0f32), (22, -12)), ((5.0, -2.75), (20, -11))] {
            let src: Vec<u16> = (0..n as i32)
                .flat_map(|r| {
                    (0..n as i32).map(move |c| {
                        let (fx, fy) = (
                            (x as i32 + c) as f32 + shift.0,
                            (y as i32 + r) as f32 + shift.1,
                        );
                        (128.0
                            + 60.0 * (0.21 * fx).sin() * (0.17 * fy).cos()
                            + 40.0 * (0.09 * fx + 0.13 * fy).sin()) as u16
                    })
                })
                .collect();
            let (mv, _) = search(
                plane,
                &src,
                x,
                y,
                n,
                8,
                Weight::UNIT,
                &[],
                &[(0, 0); 2],
                1.0,
                &mut pred,
                &mut scratch,
            );
            assert_eq!(mv, want, "shift {shift:?}");
        }
    }
}
