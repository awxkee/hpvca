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
//! Lookahead analysis on half-resolution luma: how much of a picture the
//! previous one cannot predict, which is what decides scene cuts.

use crate::Yuv;
use crate::pool::ThreadPool;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Side of the analysis blocks, in half-resolution samples (16 luma samples).
const BLOCK: usize = 8;
/// Half-resolution samples the block search may move from its start.
const RANGE: i32 = 24;
/// Prediction error per sample (8-bit scale) that counts as invisible. It
/// keeps flat, faintly noisy blocks — where even a tiny error is large
/// *relative* to the block's own activity — from reading as changed.
const VISIBILITY: f32 = 2.0;

/// Half-resolution luma of one picture, scaled to 8 bits.
pub(crate) struct Lowres {
    width: usize,
    height: usize,
    data: Vec<u8>,
    moments: OnceLock<(f32, f32)>,
}

impl Lowres {
    /// 2×2 box downscale of the luma plane, extended to full analysis blocks.
    pub(crate) fn new(yuv: &Yuv) -> Self {
        let (w, h) = (yuv.width as usize, yuv.height as usize);
        let (visible_w, visible_h) = (w.div_ceil(2), h.div_ceil(2));
        let (width, height) = (
            visible_w.next_multiple_of(BLOCK),
            visible_h.next_multiple_of(BLOCK),
        );
        let shift = 2 + u32::from(yuv.bit_depth.bits()) - 8;
        let mut data = Vec::<u8>::with_capacity(width * height);
        for r in 0..visible_h {
            let top = &yuv.y[2 * r * w..][..w];
            let bottom = &yuv.y[(2 * r + 1).min(h - 1) * w..][..w];
            for c in 0..visible_w {
                let (c0, c1) = (2 * c, (2 * c + 1).min(w - 1));
                let sum = u32::from(top[c0])
                    + u32::from(top[c1])
                    + u32::from(bottom[c0])
                    + u32::from(bottom[c1]);
                data.push(((sum + (1 << (shift - 1))) >> shift).min(255) as u8);
            }
            let last = data[data.len() - 1];
            data.resize((r + 1) * width, last);
        }
        let last_row = (visible_h - 1) * width..visible_h * width;
        for _ in visible_h..height {
            data.extend_from_within(last_row.clone());
        }
        Lowres {
            width,
            height,
            data,
            moments: OnceLock::new(),
        }
    }

    /// Mean and standard deviation of the picture.
    fn moments(&self) -> (f32, f32) {
        *self.moments.get_or_init(|| {
            let (sum, squares) = self.data.iter().fold((0u64, 0u64), |(sum, squares), &v| {
                let v = u64::from(v);
                (sum + v, squares + v * v)
            });
            let n = self.data.len() as f64;
            let mean = sum as f64 / n;
            let var = (squares as f64 / n - mean * mean).max(0.0);
            (mean as f32, var.sqrt() as f32)
        })
    }

    /// The picture as weighted prediction would present it: every sample
    /// times `weight / 64`, plus `offset`.
    pub(crate) fn weighted(&self, weight: i32, offset: i32) -> Lowres {
        Lowres {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .map(|&v| (((i32::from(v) * weight + 32) >> 6) + offset).clamp(0, 255) as u8)
                .collect(),
            moments: OnceLock::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Self {
        Lowres {
            width,
            height,
            data: (0..height)
                .flat_map(|r| (0..width).map(move |c| (r, c)))
                .map(|(r, c)| f(c, r))
                .collect(),
            moments: OnceLock::new(),
        }
    }
}

/// Sum of absolute differences of two rows of a block.
#[inline(always)]
fn row_sad(a: &[u8], b: &[u8]) -> u32 {
    let (a, b): (&[u8; BLOCK], &[u8; BLOCK]) = (
        a[..BLOCK].try_into().expect("a block row"),
        b[..BLOCK].try_into().expect("a block row"),
    );
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; both rows hold 8 bytes.
    unsafe {
        use std::arch::aarch64::*;
        u32::from(vaddlv_u8(vabd_u8(vld1_u8(a.as_ptr()), vld1_u8(b.as_ptr()))))
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is part of the x86_64 baseline; both rows hold 8 bytes.
    unsafe {
        use std::arch::x86_64::*;
        let (va, vb) = (
            _mm_loadl_epi64(a.as_ptr().cast()),
            _mm_loadl_epi64(b.as_ptr().cast()),
        );
        _mm_cvtsi128_si32(_mm_sad_epu8(va, vb)) as u32
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        a.iter()
            .zip(b)
            .map(|(&p, &q)| u32::from(p.abs_diff(q)))
            .sum()
    }
}

/// SAD between the block of `cur` at (`x`, `y`) and the block of `reference`
/// displaced by (`dx`, `dy`); samples outside `reference` repeat its border.
fn block_sad(cur: &Lowres, reference: &[u8], x: usize, y: usize, dx: i32, dy: i32) -> u32 {
    let (w, h) = (cur.width as i32, cur.height as i32);
    let (rx, ry) = (x as i32 + dx, y as i32 + dy);
    let inside = rx >= 0 && ry >= 0 && rx + BLOCK as i32 <= w && ry + BLOCK as i32 <= h;
    let mut sad = 0u32;
    for r in 0..BLOCK {
        let a = &cur.data[(y + r) * cur.width + x..][..BLOCK];
        if inside {
            let b = &reference[(ry as usize + r) * cur.width + rx as usize..][..BLOCK];
            sad += row_sad(a, b);
        } else {
            let line = (ry + r as i32).clamp(0, h - 1) as usize * cur.width;
            for (c, &p) in a.iter().enumerate() {
                let q = reference[line + (rx + c as i32).clamp(0, w - 1) as usize];
                sad += u32::from(p.abs_diff(q));
            }
        }
    }
    sad
}

/// [`block_sad`] for a displacement in quarter samples, the reference
/// interpolated bilinearly. Half-resolution pictures are too coarse for a
/// whole-sample match to stand for real motion: a pan of an odd number of
/// luma samples lands between their samples.
fn block_sad_quarter(cur: &Lowres, reference: &[u8], x: usize, y: usize, qx: i32, qy: i32) -> u32 {
    let (fx, fy) = ((qx & 3) as u32, (qy & 3) as u32);
    if fx == 0 && fy == 0 {
        return block_sad(cur, reference, x, y, qx >> 2, qy >> 2);
    }
    let (w, h) = (cur.width as i32, cur.height as i32);
    let (rx, ry) = (x as i32 + (qx >> 2), y as i32 + (qy >> 2));
    // Every sample the block reads, one more column and row than it has,
    // lies inside the reference: no border to repeat.
    let reach = BLOCK as i32 + 1;
    if rx >= 0 && ry >= 0 && rx + reach <= w && ry + reach <= h {
        const SPAN: usize = BLOCK + 1;
        let (fx, fy) = (fx as u16, fy as u16);
        // Each reference row interpolated horizontally (16 times a sample
        // at most 255 still fits), then pairs of them vertically.
        let horizontal = |r: usize| -> [u16; BLOCK] {
            let row: &[u8; SPAN] = reference[(ry as usize + r) * cur.width + rx as usize..][..SPAN]
                .try_into()
                .expect("a block row and one sample");
            std::array::from_fn(|c| (4 - fx) * u16::from(row[c]) + fx * u16::from(row[c + 1]))
        };
        let mut sad = 0u32;
        let mut top = horizontal(0);
        for r in 0..BLOCK {
            let bottom = horizontal(r + 1);
            let line: &[u8; BLOCK] = cur.data[(y + r) * cur.width + x..][..BLOCK]
                .try_into()
                .expect("a block row");
            let mut row = 0u16;
            for c in 0..BLOCK {
                let predicted = ((4 - fy) * top[c] + fy * bottom[c] + 8) >> 4;
                row += u16::from(line[c]).abs_diff(predicted);
            }
            sad += u32::from(row);
            top = bottom;
        }
        return sad;
    }
    let at = |c: i32, r: i32| {
        u32::from(reference[r.clamp(0, h - 1) as usize * cur.width + c.clamp(0, w - 1) as usize])
    };
    let mut sad = 0u32;
    for r in 0..BLOCK as i32 {
        let line = &cur.data[(y + r as usize) * cur.width + x..][..BLOCK];
        for (c, &p) in line.iter().enumerate() {
            let (sx, sy) = (rx + c as i32, ry + r);
            let top = (4 - fx) * at(sx, sy) + fx * at(sx + 1, sy);
            let bottom = (4 - fx) * at(sx, sy + 1) + fx * at(sx + 1, sy + 1);
            let predicted = ((4 - fy) * top + fy * bottom + 8) >> 4;
            sad += u32::from(p).abs_diff(predicted);
        }
    }
    sad
}

/// Cost of predicting the block at (`x`, `y`) from its own picture: the best
/// of DC, horizontal and vertical prediction from the neighboring samples.
fn intra_cost(cur: &Lowres, x: usize, y: usize) -> u32 {
    let at = |c: usize, r: usize| cur.data[r * cur.width + c];
    let row = |r: usize| &cur.data[r * cur.width + x..][..BLOCK];
    let mut dc_sum = 0u32;
    let mut dc_count = 0u32;
    if y > 0 {
        dc_sum += row(y - 1).iter().map(|&v| u32::from(v)).sum::<u32>();
        dc_count += BLOCK as u32;
    }
    if x > 0 {
        dc_sum += (0..BLOCK).map(|r| u32::from(at(x - 1, y + r))).sum::<u32>();
        dc_count += BLOCK as u32;
    }
    // No neighbors at the picture's corner: mid-gray.
    let dc = (dc_sum + dc_count / 2).checked_div(dc_count).unwrap_or(128) as u8;
    let mut costs = [0u32, u32::MAX, u32::MAX];
    if y > 0 {
        costs[1] = 0;
    }
    if x > 0 {
        costs[2] = 0;
    }
    let flat = [dc; BLOCK];
    for r in 0..BLOCK {
        let line = row(y + r);
        costs[0] += row_sad(line, &flat);
        if y > 0 {
            costs[1] += row_sad(line, row(y - 1));
        }
        if x > 0 {
            costs[2] += row_sad(line, &[at(x - 1, y + r); BLOCK]);
        }
    }
    costs.into_iter().min().unwrap_or(0)
}

/// The translation, in half-resolution samples, that best maps `reference`
/// onto `cur` as a whole: an exhaustive search within [`RANGE`] on pictures
/// reduced a further four times. A pan moves every block alike, and on
/// repetitive texture a block search that starts from zero stops at the
/// first false match; started from here it does not.
fn global_motion(cur: &Lowres, reference: &[u8], pool: Option<&ThreadPool>) -> (i32, i32) {
    const REDUCE: usize = 4;
    let (w, h) = (cur.width / REDUCE, cur.height / REDUCE);
    if w < 8 || h < 8 {
        return (0, 0);
    }
    let reduce = |data: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                let sum: u32 = (0..REDUCE)
                    .flat_map(|dr| (0..REDUCE).map(move |dc| (dr, dc)))
                    .map(|(dr, dc)| {
                        u32::from(data[(r * REDUCE + dr) * cur.width + c * REDUCE + dc])
                    })
                    .sum();
                out.push((sum / (REDUCE * REDUCE) as u32) as u8);
            }
        }
        out
    };
    let (a, b) = (reduce(&cur.data), reduce(reference));
    let reach = RANGE / REDUCE as i32;
    // The best vector of one row of the search; rows are independent.
    let search_row = |dy: i32| -> ((i32, i32), f32) {
        let mut best = ((0, 0), f32::INFINITY);
        for dx in -reach..=reach {
            // Mean absolute difference over the overlap of the two pictures.
            let (x0, x1) = ((-dx).max(0) as usize, (w as i32 - dx.max(0)) as usize);
            let (y0, y1) = ((-dy).max(0) as usize, (h as i32 - dy.max(0)) as usize);
            if x1 <= x0 + w / 2 || y1 <= y0 + h / 2 {
                continue;
            }
            let mut sad = 0u32;
            for r in y0..y1 {
                let p = &a[r * w + x0..r * w + x1];
                let q0 = (r as i32 + dy) as usize * w + (x0 as i32 + dx) as usize;
                let q = &b[q0..q0 + (x1 - x0)];
                sad += p
                    .iter()
                    .zip(q)
                    .map(|(&u, &v)| u32::from(u.abs_diff(v)))
                    .sum::<u32>();
            }
            // Prefer the smaller vector among equals.
            let cost =
                sad as f32 / ((x1 - x0) * (y1 - y0)) as f32 + 1e-3 * (dx.abs() + dy.abs()) as f32;
            if cost < best.1 {
                best = ((dx, dy), cost);
            }
        }
        best
    };
    let mut rows = vec![((0, 0), f32::INFINITY); (2 * reach + 1) as usize];
    match pool {
        Some(pool) if pool.parallelism() > 1 => pool.scoped(|scope| {
            let search_row = &search_row;
            for (slot, dy) in rows.iter_mut().zip(-reach..=reach) {
                scope.spawn(move || *slot = search_row(dy));
            }
        }),
        _ => {
            for (slot, dy) in rows.iter_mut().zip(-reach..=reach) {
                *slot = search_row(dy);
            }
        }
    }
    // In search order, so that ties resolve as in a single scan.
    let mut best = ((0, 0), f32::INFINITY);
    for row in rows {
        if row.1 < best.1 {
            best = row;
        }
    }
    (best.0.0 * REDUCE as i32, best.0.1 * REDUCE as i32)
}

/// The share of `cur`, between 0 and 1, that `prev` cannot predict.
///
/// Each 16×16 luma block (8×8 here) contributes its motion-compensated
/// prediction error from `prev` relative to its intra prediction error plus a
/// visibility floor, capped at 1. Unlike a ratio of whole-picture sums, every
/// part of the picture then weighs the same however much texture it holds: a
/// change of a flat wall counts as much as a change of foliage, and a few
/// very busy blocks cannot outvote the rest.
///
/// Brightness changes count toward this score. The encoder estimates and
/// tests weighted prediction separately.
pub(crate) fn change_score(prev: &Lowres, cur: &Lowres) -> f32 {
    analyse(prev, cur, None).score
}

/// The weight (in 64ths, minus 64) and offset that give `prev` the mean and
/// spread of `cur`: what a fade or a flash between them amounts to. `None`
/// when the two already agree. The fit uses each picture's statistics alone,
/// never sample pairs, so motion between them cannot disturb it; whether it
/// actually helps prediction is for the caller to check.
pub(crate) fn estimate_weights(prev: &Lowres, cur: &Lowres) -> Option<(i8, i8)> {
    let ((mean_p, dev_p), (mean_c, dev_c)) = (prev.moments(), cur.moments());
    // A flat picture has no spread to match: shift its level only.
    let gain = if dev_p > 1.0 && dev_c > 1.0 {
        (dev_c / dev_p).clamp(0.25, 2.9)
    } else {
        1.0
    };
    let weight = (gain * 64.0).round();
    let offset = (mean_c - weight / 64.0 * mean_p)
        .round()
        .clamp(-127.0, 127.0);
    let delta = weight as i32 - 64;
    (delta.abs() >= 2 || offset.abs() >= 2.0).then_some((delta as i8, offset as i8))
}

/// What the lookahead learns about a picture from the one before it.
pub(crate) struct Analysis {
    /// See [`change_score`].
    pub(crate) score: f32,
    /// Each block's contribution to the score, scaled to 0..=255, in raster
    /// order: how badly the previous picture predicts that 16×16 luma block.
    pub(crate) blocks: Vec<u8>,
    /// Blocks per row.
    pub(crate) stride: usize,
    /// The costs and motion behind each block's score.
    pub(crate) info: Vec<BlockInfo>,
}

/// What the analysis measured for one 16×16 luma block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BlockInfo {
    /// Cost of predicting the block from its own picture, plus the
    /// visibility floor.
    pub(crate) intra: f32,
    /// Cost of predicting it from the previous picture, at most `intra`.
    pub(crate) inter: f32,
    /// Its motion from the previous picture, in quarter half-resolution
    /// samples (32 to a block).
    pub(crate) mv: (i16, i16),
}

/// Blocks per row of a picture's analysis.
pub(crate) fn block_stride(picture: &Lowres) -> usize {
    picture.width / BLOCK
}

/// [`BlockInfo`] of a picture with nothing to predict it from: intra costs
/// only.
pub(crate) fn intra_info(cur: &Lowres) -> Vec<BlockInfo> {
    let (bw, bh) = (cur.width / BLOCK, cur.height / BLOCK);
    let floor = VISIBILITY * (BLOCK * BLOCK) as f32;
    (0..bw * bh)
        .map(|b| {
            let intra = intra_cost(cur, (b % bw) * BLOCK, (b / bw) * BLOCK) as f32 + floor;
            BlockInfo {
                intra,
                inter: intra,
                mv: (0, 0),
            }
        })
        .collect()
}

/// How much each block of a picture is worth to the pictures after it, as
/// `log2(1 + inherited / own)`: x264's macroblock-tree propagation.
///
/// `later` holds the following pictures in display order, each predicted
/// from the one before it. Walking them backwards, a block passes on to the
/// area of its reference that its vector points at the share of its own
/// information — its intra cost plus whatever was passed on to it — that the
/// reference supplied, `1 − inter / intra`. What arrives at `current` is the
/// information later pictures draw from each of its blocks.
pub(crate) fn propagation(
    current: &[BlockInfo],
    later: &[&[BlockInfo]],
    stride: usize,
) -> Vec<f32> {
    let n = current.len();
    let rows = n.checked_div(stride).unwrap_or(0);
    let mut incoming = vec![0f32; n];
    let mut outgoing = vec![0f32; n];
    for frame in later.iter().rev() {
        outgoing.fill(0.0);
        if frame.len() == n {
            for (b, info) in frame.iter().enumerate() {
                let amount = (info.intra + incoming[b]) * (1.0 - info.inter / info.intra);
                if amount <= 0.0 {
                    continue;
                }
                // Where the block's prediction comes from, in blocks; it
                // straddles up to four, weighted by overlap.
                let scale = 1.0 / (4 * BLOCK) as f32;
                let fx = (b % stride) as f32 + f32::from(info.mv.0) * scale;
                let fy = (b / stride) as f32 + f32::from(info.mv.1) * scale;
                let (x0, y0) = (fx.floor(), fy.floor());
                let (wx, wy) = (fx - x0, fy - y0);
                for (dx, dy, weight) in [
                    (0, 0, (1.0 - wx) * (1.0 - wy)),
                    (1, 0, wx * (1.0 - wy)),
                    (0, 1, (1.0 - wx) * wy),
                    (1, 1, wx * wy),
                ] {
                    let (x, y) = (x0 as i32 + dx, y0 as i32 + dy);
                    if weight > 0.0
                        && x >= 0
                        && y >= 0
                        && (x as usize) < stride
                        && (y as usize) < rows
                    {
                        outgoing[y as usize * stride + x as usize] += amount * weight;
                    }
                }
            }
        }
        std::mem::swap(&mut incoming, &mut outgoing);
    }
    current
        .iter()
        .zip(&incoming)
        .map(|(block, &inherited)| ((block.intra + inherited) / block.intra).log2())
        .collect()
}

/// [`change_score`] with its per-block breakdown.
///
/// With a `pool` the block rows run as a wavefront on it: a block starts
/// from the vectors of its left and upper neighbors when its co-located
/// match is imperfect. The result does not depend on the pool.
pub(crate) fn analyse(prev: &Lowres, cur: &Lowres, pool: Option<&ThreadPool>) -> Analysis {
    let (bw, bh) = (cur.width / BLOCK, cur.height / BLOCK);
    let same_size = prev.width == cur.width && prev.height == cur.height;
    if bw == 0 || bh == 0 || !same_size {
        return Analysis {
            score: if same_size { 0.0 } else { 1.0 },
            blocks: Vec::new(),
            stride: 0,
            info: Vec::new(),
        };
    }
    if prev.data == cur.data {
        let mut info = intra_info(cur);
        for block in &mut info {
            block.inter = 0.0;
        }
        return Analysis {
            score: 0.0,
            blocks: vec![0; bw * bh],
            stride: bw,
            info,
        };
    }
    let reference: &[u8] = &prev.data;
    let global = global_motion(cur, reference, pool);
    let mut blocks = vec![0u8; bw * bh];
    let mut shares = vec![0f32; bw * bh];
    let mut info = vec![
        BlockInfo {
            intra: 0.0,
            inter: 0.0,
            mv: (0, 0),
        };
        bw * bh
    ];
    // Whole-sample vectors as rows publish them, and how many each row has.
    let vectors: Vec<AtomicU32> = (0..bw * bh).map(|_| AtomicU32::new(0)).collect();
    let progress: Vec<AtomicUsize> = (0..bh).map(|_| AtomicUsize::new(0)).collect();
    let wave = Wave {
        cur,
        reference,
        global,
        bw,
        vectors: &vectors,
        progress: &progress,
    };
    let rows = blocks
        .chunks_mut(bw)
        .zip(shares.chunks_mut(bw))
        .zip(info.chunks_mut(bw))
        .enumerate();
    match pool {
        // Queued top row first: a row only ever waits for one that started
        // before it.
        Some(pool) if pool.parallelism() > 1 => pool.scoped(|scope| {
            let wave = &wave;
            for (by, ((blocks, shares), info)) in rows {
                scope.spawn(move || wave.row(by, blocks, shares, info));
            }
        }),
        _ => {
            for (by, ((blocks, shares), info)) in rows {
                wave.row(by, blocks, shares, info);
            }
        }
    }
    // Summed in raster order whatever order the rows finished in.
    let total: f32 = shares.iter().sum();
    Analysis {
        score: total / (bw * bh) as f32,
        blocks,
        stride: bw,
        info,
    }
}

/// What the rows of one [`analyse`] share.
struct Wave<'a> {
    cur: &'a Lowres,
    reference: &'a [u8],
    global: (i32, i32),
    bw: usize,
    vectors: &'a [AtomicU32],
    progress: &'a [AtomicUsize],
}

/// Marks a row complete when it ends, also by a panic, so that the row below
/// never waits for it forever.
struct RowDone<'a>(&'a AtomicUsize, usize);

impl Drop for RowDone<'_> {
    fn drop(&mut self) {
        self.0.store(self.1, Ordering::Release);
    }
}

impl Wave<'_> {
    fn vector(&self, index: usize) -> (i32, i32) {
        let packed = self.vectors[index].load(Ordering::Relaxed);
        (
            i32::from((packed >> 16) as u16 as i16),
            i32::from(packed as u16 as i16),
        )
    }

    /// Search the blocks of row `by`, behind the row above it.
    fn row(&self, by: usize, blocks: &mut [u8], shares: &mut [f32], info: &mut [BlockInfo]) {
        let (cur, reference, bw) = (self.cur, self.reference, self.bw);
        let floor = VISIBILITY * (BLOCK * BLOCK) as f32;
        let _done = RowDone(&self.progress[by], bw);
        for bx in 0..bw {
            let (x, y) = (bx * BLOCK, by * BLOCK);
            let sad = |v: (i32, i32)| block_sad(cur, reference, x, y, v.0, v.1);
            // Start from the picture's and the neighbors' vectors, then walk
            // a shrinking diamond.
            let mut best = ((0, 0), sad((0, 0)));
            if best.1 != 0 {
                if by > 0 {
                    while self.progress[by - 1].load(Ordering::Acquire) <= bx {
                        std::thread::yield_now();
                    }
                }
                let neighbors = [
                    Some(self.global),
                    (bx > 0).then(|| self.vector(by * bw + bx - 1)),
                    (by > 0).then(|| self.vector((by - 1) * bw + bx)),
                ];
                for v in neighbors.into_iter().flatten() {
                    if v != (0, 0) && v != best.0 {
                        let s = sad(v);
                        if s < best.1 {
                            best = (v, s);
                        }
                    }
                    if best.1 == 0 {
                        break;
                    }
                }
                for step in [4, 2, 1] {
                    if best.1 == 0 {
                        break;
                    }
                    for _ in 0..8 {
                        let centre = best.0;
                        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                            let v = (centre.0 + dx * step, centre.1 + dy * step);
                            if v.0.abs() > RANGE || v.1.abs() > RANGE {
                                continue;
                            }
                            let s = sad(v);
                            if s < best.1 {
                                best = (v, s);
                            }
                        }
                        if best.0 == centre || best.1 == 0 {
                            break;
                        }
                    }
                }
            }
            let packed =
                (u32::from(best.0.0 as i16 as u16) << 16) | u32::from(best.0.1 as i16 as u16);
            self.vectors[by * bw + bx].store(packed, Ordering::Relaxed);
            self.progress[by].store(bx + 1, Ordering::Release);
            // Half, then quarter samples around the whole-sample match,
            // unless that match is already exact.
            let mut fine = ((best.0.0 * 4, best.0.1 * 4), best.1);
            for step in [2, 1] {
                if fine.1 == 0 {
                    break;
                }
                let centre = fine.0;
                for (dx, dy) in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    let q = (centre.0 + dx * step, centre.1 + dy * step);
                    let s = block_sad_quarter(cur, reference, x, y, q.0, q.1);
                    if s < fine.1 {
                        fine = (q, s);
                    }
                }
            }
            let best = fine;
            let intra = intra_cost(cur, x, y) as f32 + floor;
            let share = (best.1 as f32 / intra).min(1.0);
            blocks[bx] = (share * 255.0) as u8;
            shares[bx] = share;
            info[bx] = BlockInfo {
                intra,
                inter: (best.1 as f32).min(intra),
                mv: (best.0.0 as i16, best.0.1 as i16),
            };
        }
    }
}

/// How far a cut's score must rise above the score of the frame before it,
/// and stay above that of the frame after it. A hard cut is one unpredictable
/// frame after predictable ones; a dissolve, a fade or a burst of fast motion
/// is several in a row, and a keyframe in the middle of one buys nothing. The
/// second margin is the smaller one because a new shot may well open in
/// motion.
const RISE: f32 = 0.3;
const SETTLE: f32 = 0.1;

/// Score a frame must reach to be a scene cut, `gop_size` frames after the
/// last keyframe. As in x264, a cut is taken more readily the longer the GOP
/// has run: a keyframe is due at `max_keyint` anyway, while one right after
/// `min_keyint` had better be certain.
///
/// On real footage hard cuts score 0.90–0.98 and frames within a shot stay
/// below about 0.3.
pub(crate) fn cut_threshold(gop_size: u32, min_keyint: u32, max_keyint: u32) -> f32 {
    const EAGER: f32 = 0.60;
    const STRICT: f32 = 0.85;
    if max_keyint <= min_keyint || gop_size <= min_keyint {
        return STRICT;
    }
    let run = (gop_size - min_keyint) as f32 / (max_keyint - min_keyint) as f32;
    STRICT + (EAGER - STRICT) * run.min(1.0)
}

/// Change score below which two frames show the same scene. Frames of one
/// shot score under about 0.3 against each other on real footage, also a few
/// frames apart.
pub(crate) const SAME_SCENE: f32 = 0.35;

/// Whether a frame with change score `score` — its neighbors scoring `before`
/// and `after` against their own predecessors — is a hard cut.
pub(crate) fn is_cut(score: f32, before: f32, after: f32, threshold: f32) -> bool {
    score >= threshold && score - before >= RISE && score - after >= SETTLE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_blocks_are_analysed_with_extended_edges() {
        let lowres = |samples, w, h| {
            Lowres::new(
                &Yuv::from_planes(
                    samples,
                    Vec::new(),
                    Vec::new(),
                    w,
                    h,
                    crate::ChromaFormat::Monochrome,
                    crate::BitDepth::Eight,
                )
                .unwrap(),
            )
        };
        let tiny = lowres((1..=9).map(|v| v * 10).collect(), 3, 3);
        assert_eq!(&tiny.data[..BLOCK], &[30, 45, 45, 45, 45, 45, 45, 45]);
        for row in tiny.data[BLOCK..].chunks_exact(BLOCK) {
            assert_eq!(row, &[75, 90, 90, 90, 90, 90, 90, 90]);
        }
        let black = lowres(vec![0; 9], 3, 3);
        assert!(change_score(&black, &tiny) > 0.4);

        let black = lowres(vec![0; 17 * 16], 17, 16);
        let right_edge = lowres(
            (0..17 * 16)
                .map(|i| if i % 17 == 16 { 255 } else { 0 })
                .collect(),
            17,
            16,
        );
        let analysis = analyse(&black, &right_edge, None);
        assert_eq!(analysis.stride, 2);
        assert_eq!(analysis.blocks[0], 0);
        assert!(analysis.blocks[1] > 250);
    }

    /// Smooth pseudo-random field: random values on a lattice, interpolated.
    fn value_noise(seed: u32, x: f32, y: f32) -> f32 {
        let lattice = |ix: i32, iy: i32| -> f32 {
            let h = (ix as u32)
                .wrapping_mul(374_761_393)
                .wrapping_add((iy as u32).wrapping_mul(668_265_263))
                .wrapping_add(seed.wrapping_mul(2_246_822_519));
            let h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
            ((h ^ (h >> 16)) & 0xFFFF) as f32 / 65535.0 - 0.5
        };
        let (ix, iy) = (x.floor(), y.floor());
        let (tx, ty) = (x - ix, y - iy);
        let (ix, iy) = (ix as i32, iy as i32);
        let top = lattice(ix, iy) * (1.0 - tx) + lattice(ix + 1, iy) * tx;
        let bottom = lattice(ix, iy + 1) * (1.0 - tx) + lattice(ix + 1, iy + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// A deterministic textured scene, displaced by `shift` samples, scaled
    /// by `gain`, with per-frame noise when `noise` is nonzero. Different
    /// seeds are unrelated pictures.
    fn scene(seed: u32, shift: (f32, f32), gain: f32, noise: u32) -> Lowres {
        Lowres::from_fn(160, 96, |x, y| {
            let (fx, fy) = (x as f32 + shift.0 + 1000.0, y as f32 + shift.1 + 1000.0);
            let v = 128.0
                + 110.0 * value_noise(seed, fx / 14.0, fy / 14.0)
                + 50.0 * value_noise(seed + 77, fx / 5.0, fy / 5.0);
            let n = if noise == 0 {
                0.0
            } else {
                let h = (x as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((y as u32).wrapping_mul(40_503))
                    .wrapping_add(noise.wrapping_mul(97_561));
                ((h >> 13) % 7) as f32 - 3.0
            };
            (v * gain + n).round().clamp(0.0, 255.0) as u8
        })
    }

    /// The block SADs, whole and quarter sample, against a plain per-sample
    /// definition: the vector rows and the border-free interpolation must
    /// agree with it everywhere, borders included.
    #[test]
    fn block_sads_match_their_definition() {
        let (cur, reference) = (scene(3, (0.0, 0.0), 1.0, 7), scene(4, (1.5, 0.5), 0.9, 9));
        let (w, h) = (cur.width as i32, cur.height as i32);
        let at = |c: i32, r: i32| {
            u32::from(
                reference.data[r.clamp(0, h - 1) as usize * cur.width + c.clamp(0, w - 1) as usize],
            )
        };
        let plain = |x: usize, y: usize, qx: i32, qy: i32| -> u32 {
            let (fx, fy) = ((qx & 3) as u32, (qy & 3) as u32);
            let mut sad = 0;
            for r in 0..BLOCK as i32 {
                for c in 0..BLOCK as i32 {
                    let (sx, sy) = (x as i32 + (qx >> 2) + c, y as i32 + (qy >> 2) + r);
                    let top = (4 - fx) * at(sx, sy) + fx * at(sx + 1, sy);
                    let bottom = (4 - fx) * at(sx, sy + 1) + fx * at(sx + 1, sy + 1);
                    let predicted = ((4 - fy) * top + fy * bottom + 8) >> 4;
                    let p = u32::from(cur.data[(y + r as usize) * cur.width + x + c as usize]);
                    sad += p.abs_diff(predicted);
                }
            }
            sad
        };
        let (bw, bh) = (cur.width / BLOCK, cur.height / BLOCK);
        for (bx, by) in [
            (0, 0),
            (bw - 1, 0),
            (0, bh - 1),
            (bw - 1, bh - 1),
            (bw / 2, bh / 2),
        ] {
            let (x, y) = (bx * BLOCK, by * BLOCK);
            for qy in (-40..=40).step_by(3) {
                for qx in (-40..=40).step_by(5) {
                    assert_eq!(
                        block_sad_quarter(&cur, &reference.data, x, y, qx, qy),
                        plain(x, y, qx, qy),
                        "block ({bx}, {by}) vector ({qx}, {qy})"
                    );
                }
            }
        }
    }

    #[test]
    fn a_still_or_moving_shot_scores_low_and_a_cut_scores_high() {
        let a = scene(1, (0.0, 0.0), 1.0, 0);
        assert!(change_score(&a, &a) < 0.01);
        // Sensor-like noise, a pan (whole and fractional samples), a fade.
        let noisy = change_score(&scene(1, (0.0, 0.0), 1.0, 1), &scene(1, (0.0, 0.0), 1.0, 2));
        assert!(noisy < 0.3, "noise {noisy}");
        let pan = change_score(&a, &scene(1, (5.0, -3.0), 1.0, 0));
        assert!(pan < 0.2, "pan {pan}");
        let pan = change_score(&a, &scene(1, (2.5, 1.5), 1.0, 0));
        assert!(pan < 0.4, "fractional pan {pan}");
        // A fade is unpredictable frame after frame, which is exactly why it
        // is not a cut: no frame of it stands out from its neighbors.
        let fade: Vec<f32> = (0..4)
            .map(|n| {
                let gain = |n: i32| 1.0 - 0.15 * n as f32;
                change_score(
                    &scene(1, (0.0, 0.0), gain(n), 0),
                    &scene(1, (0.0, 0.0), gain(n + 1), 0),
                )
            })
            .collect();
        assert!(!is_cut(fade[1], fade[0], fade[2], 0.6), "fade {fade:?}");
        assert!(!is_cut(fade[2], fade[1], fade[3], 0.6), "fade {fade:?}");
        // Unrelated pictures.
        for seed in 2..6 {
            let cut = change_score(&a, &scene(seed, (0.0, 0.0), 1.0, 0));
            // Smooth random fields match each other in places; real footage
            // scores 0.90 and up at a cut.
            assert!(cut > 0.6, "cut to scene {seed}: {cut}");
        }
    }

    #[test]
    fn importance_propagates_along_motion() {
        let block = |inter: f32, mv: (i16, i16)| BlockInfo {
            intra: 1000.0,
            inter,
            mv,
        };
        // A 4×1 strip. Nothing later: no block is worth more than itself.
        let current = [block(1000.0, (0, 0)); 4];
        assert_eq!(propagation(&current, &[], 4), [0.0; 4]);
        // Three later pictures that copy the strip exactly: each block is
        // drawn on three times, log2(1 + 3) = 2.
        let still = [block(0.0, (0, 0)); 4];
        let later = [&still[..], &still[..], &still[..]];
        assert_eq!(propagation(&current, &later, 4), [2.0; 4]);
        // A later picture that the strip does not predict at all inherits
        // nothing, and passes nothing on from the pictures after it.
        let cut = [block(1000.0, (0, 0)); 4];
        assert_eq!(
            propagation(&current, &[&still[..], &cut[..], &still[..]], 4),
            [1.0; 4]
        );
        // Motion: every block of the next picture comes from one block to
        // its left (−32 quarter samples), so block 3 is never drawn on and
        // block 0 only by block 1.
        let pan = [block(0.0, (-32, 0)); 4];
        assert_eq!(propagation(&current, &[&pan[..]], 4), [1.0, 1.0, 1.0, 0.0]);
        // Half a block of motion splits the inheritance between two blocks.
        let half = [block(0.0, (-16, 0)); 4];
        let got = propagation(&current, &[&half[..]], 4);
        assert!(
            (got[0] - 1.5f32.log2().max(1.0)).abs() < 0.6 && got[3] < got[1],
            "{got:?}"
        );
    }

    #[test]
    fn cuts_are_isolated_spikes() {
        assert!(is_cut(0.95, 0.04, 0.05, 0.85));
        // A new shot that opens with motion still stands out.
        assert!(is_cut(0.93, 0.04, 0.53, 0.85));
        assert!(is_cut(1.00, 0.05, 0.82, 0.82));
        // Two unpredictable frames in a row: a transition, not a cut.
        assert!(!is_cut(0.94, 0.10, 0.94, 0.60));
        // A dissolve or a fast animation: high for several frames in a row.
        assert!(!is_cut(0.80, 0.70, 0.44, 0.60));
        assert!(!is_cut(0.92, 0.69, 0.91, 0.60));
        // Below the threshold for this point of the GOP.
        assert!(!is_cut(0.70, 0.02, 0.02, 0.85));
        assert!(is_cut(0.70, 0.02, 0.02, 0.60));
    }

    #[test]
    fn the_threshold_relaxes_along_the_gop() {
        assert_eq!(cut_threshold(10, 25, 250), 0.85);
        assert_eq!(cut_threshold(25, 25, 250), 0.85);
        assert!(cut_threshold(100, 25, 250) < 0.85);
        assert!((cut_threshold(250, 25, 250) - 0.60).abs() < 1e-6);
        assert!((cut_threshold(400, 25, 250) - 0.60).abs() < 1e-6);
        assert_eq!(cut_threshold(30, 30, 30), 0.85);
    }
}
