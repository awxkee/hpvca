/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

//! Intra block copy (HEVC Screen Content Coding) — encoder side.

use std::collections::HashMap;

/// `bv_map` entry for a block that is not an IntraBC copy (intra or palette).
pub(crate) const BV_INTRA: i32 = i32::MIN;

/// Pack an integer-sample block vector into a `bv_map` entry.
#[inline]
pub(crate) fn pack_bv(x: i16, y: i16) -> i32 {
    ((x as i32) << 16) | (y as u16 as i32)
}

/// Unpack a `bv_map` entry; `None` for a non-IntraBC block.
#[inline]
pub(crate) fn unpack_bv(packed: i32) -> Option<(i16, i16)> {
    if packed == BV_INTRA {
        return None;
    }
    Some(((packed >> 16) as i16, (packed & 0xFFFF) as u16 as i16))
}

/// Deblocking boundary strength across an edge between two 4×4 blocks
/// (§8.7.2.4). Every IntraBC CU this encoder emits is a residual-free
/// PART_2Nx2N copy from the one reference picture, so the only inputs left are
/// "is either side intra" and the block-vector difference.
#[inline]
pub(crate) fn boundary_strength(current: i32, neighbor: i32) -> u8 {
    match (unpack_bv(current), unpack_bv(neighbor)) {
        (Some(a), Some(b)) => {
            // The threshold is 4 quarter-pel units, i.e. one integer sample.
            u8::from((a.0 - b.0).abs() >= 1 || (a.1 - b.1).abs() >= 1)
        }
        // An intra (or palette) block on either side is boundary strength 2.
        _ => 2,
    }
}

/// Geometry a block vector is validated against. `size` is the PART_2Nx2N CU
/// side, so the prediction block and the partition span coincide.
#[derive(Clone, Copy)]
pub(crate) struct CuGeom {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) size: usize,
    /// Coded (CTB-aligned) picture dimensions.
    pub(crate) pic_w: usize,
    pub(crate) pic_h: usize,
    pub(crate) ctb_log2: u32,
}

/// The two source corners whose availability §6.4.1 must accept.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SourceArea {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) x1: usize,
    pub(crate) y1: usize,
}

/// Geometry-only block-vector constraints from §8.5.3.2.1: the source must lie
/// inside the picture, must not overlap the block being predicted, and must not
/// reach past the reconstructed CTB wavefront. Returns the corner pair the
/// caller still has to check for decode-order availability.
///
/// This encoder only emits block vectors whose chroma counterpart is also
/// integer (see [`BvSearch::parity`]), so the equation 8-104/8-105 interpolation
/// margins are always zero.
pub(crate) fn source_area(cu: &CuGeom, bvx: i32, bvy: i32) -> Option<SourceArea> {
    let size = cu.size as i32;
    let src_x = cu.x as i32 + bvx;
    let src_y = cu.y as i32 + bvy;
    let x1 = src_x + size - 1;
    let y1 = src_y + size - 1;
    if src_x < 0 || src_y < 0 || x1 >= cu.pic_w as i32 || y1 >= cu.pic_h as i32 {
        return None;
    }
    // Source and destination prediction blocks must not overlap: the source is
    // either entirely left of, or entirely above, the current block.
    if bvx + size > 0 && bvy + size > 0 {
        return None;
    }
    // Equation 8-106: an above-row reference may only reach as far right as the
    // reconstructed CTB wavefront allows.
    let ctb = 1i32 << cu.ctb_log2;
    let left = x1 / ctb - (cu.x as i32) / ctb;
    let right = (cu.y as i32) / ctb - y1 / ctb;
    if left > right {
        return None;
    }
    Some(SourceArea {
        x0: src_x as usize,
        y0: src_y as usize,
        x1: x1 as usize,
        y1: y1 as usize,
    })
}

/// AMVP predictors for a PART_2Nx2N IntraBC prediction unit (§8.5.3.2.6–8),
/// specialised for this encoder's stream shape: one reference picture (the
/// current one), so every available inter neighbour matches the target
/// reference, and `slice_temporal_mvp_enabled_flag` is 0 for an IDR, so there
/// is no temporal candidate.
///
/// `available(x, y)` is the §6.4.1 z-scan/decode-order test; `motion` reads the
/// per-4×4 block-vector map.
pub(crate) fn amvp_predictors(
    x: usize,
    y: usize,
    size: usize,
    available: &dyn Fn(usize, usize) -> bool,
    motion: &dyn Fn(usize, usize) -> Option<(i16, i16)>,
) -> [(i16, i16); 2] {
    let xi = x as i32;
    let yi = y as i32;
    let n = size as i32;
    let at = |px: i32, py: i32| -> Option<(i16, i16)> {
        if px < 0 || py < 0 {
            return None;
        }
        let (px, py) = (px as usize, py as usize);
        if !available(px, py) {
            return None;
        }
        motion(px, py)
    };

    // A0 = (x-1, y+h), A1 = (x-1, y+h-1); B0 = (x+w, y-1), B1 = (x+w-1, y-1),
    // B2 = (x-1, y-1).
    let a = at(xi - 1, yi + n).or_else(|| at(xi - 1, yi + n - 1));
    let b = at(xi + n, yi - 1)
        .or_else(|| at(xi + n - 1, yi - 1))
        .or_else(|| at(xi - 1, yi - 1));

    // isScaledFlagLX is set when either A block is available and non-intra —
    // which, with a single reference picture, is exactly `a.is_some()`. When it
    // is clear the B result stands in as the A candidate and B is re-derived by
    // the scaling pass; that pass walks the same positions and, because every
    // candidate already references the current picture (POC distance 0, so
    // §8.5.3.2.8 leaves the vector unscaled), reproduces the same value. The
    // duplicate is then dropped, leaving a single non-zero predictor.
    let (a, b) = if a.is_some() { (a, b) } else { (b, b) };

    let mut preds = [(0i16, 0i16); 2];
    let mut count = 0usize;
    if let Some(mv) = a {
        preds[count] = mv;
        count += 1;
    }
    if let Some(mv) = b
        && Some(mv) != a
        && count < 2
    {
        preds[count] = mv;
    }
    preds
}

/// The motion-vector difference the bitstream carries for a chosen predictor.
/// `combine_mvp_mvd` in the decoder computes `((mvp >> 2) + mvd) * 4`, so the
/// difference is in integer luma samples against the *rounded* predictor.
#[inline]
pub(crate) fn mvd_for(bv: (i16, i16), mvp: (i16, i16)) -> (i32, i32) {
    (i32::from(bv.0 - mvp.0), i32::from(bv.1 - mvp.1))
}

/// Supported IntraBC CU sides, smallest first.
pub(crate) const SIZES: [usize; 3] = [8, 16, 32];

#[inline]
fn size_index(size: usize) -> Option<usize> {
    SIZES.iter().position(|&s| s == size)
}

/// Content hash of every 8-aligned source block, per CU size.
pub(crate) struct HashTable {
    levels: [Level; SIZES.len()],
    buckets: [HashMap<u64, Vec<(u32, u32)>>; SIZES.len()],
}

/// One block size's hash grid, indexed in 8-pixel units.
#[derive(Default)]
struct Level {
    cols: usize,
    rows: usize,
    hash: Vec<u64>,
    /// A block whose samples are all equal matches everywhere, which would make
    /// the buckets enormous; palette mode already codes those for a handful of
    /// bits, so they are left out of the index.
    flat: Vec<bool>,
    first: Vec<u16>,
}

impl Level {
    #[inline]
    fn index(&self, bx: usize, by: usize) -> usize {
        by * self.cols + bx
    }
}

/// Longest candidate list a single hash bucket contributes to one CU. Repeated
/// content produces very long buckets (a flat background matches everywhere);
/// the most recent entries are the closest ones, and therefore the cheapest to
/// code, so the tail is not worth walking.
pub(crate) const MAX_CANDIDATES: usize = 12;

/// Mix four child hashes into their parent's.
#[inline]
fn combine(children: [u64; 4]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for child in children {
        hash ^= child;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl HashTable {
    pub(crate) fn build(y: &[u16], src_w: usize, src_h: usize) -> Self {
        let mut levels: [Level; SIZES.len()] = Default::default();
        for (level, &size) in SIZES.iter().enumerate() {
            let cols = if src_w >= size {
                (src_w - size) / 8 + 1
            } else {
                0
            };
            let rows = if src_h >= size {
                (src_h - size) / 8 + 1
            } else {
                0
            };
            let count = cols * rows;
            let mut hash = vec![0u64; count];
            let mut flat = vec![false; count];
            let mut first = vec![0u16; count];
            if level == 0 {
                for by in 0..rows {
                    for bx in 0..cols {
                        let (h, is_flat, head) = hash_8x8(y, src_w, bx * 8, by * 8);
                        let i = by * cols + bx;
                        hash[i] = h;
                        flat[i] = is_flat;
                        first[i] = head;
                    }
                }
            } else {
                // The four sub-blocks of a `size` block sit `size/2` pixels
                // apart, i.e. `1 << (level - 1)` positions in the 8-unit grid.
                let step = 1usize << (level - 1);
                let child = &levels[level - 1];
                for by in 0..rows {
                    for bx in 0..cols {
                        let quad = [
                            child.index(bx, by),
                            child.index(bx + step, by),
                            child.index(bx, by + step),
                            child.index(bx + step, by + step),
                        ];
                        let i = by * cols + bx;
                        hash[i] = combine(quad.map(|q| child.hash[q]));
                        first[i] = child.first[quad[0]];
                        flat[i] = quad.iter().all(|&q| child.flat[q])
                            && quad.iter().all(|&q| child.first[q] == first[i]);
                    }
                }
            }
            levels[level] = Level {
                cols,
                rows,
                hash,
                flat,
                first,
            };
        }

        let mut buckets: [HashMap<u64, Vec<(u32, u32)>>; SIZES.len()] = Default::default();
        for (level, table) in levels.iter().enumerate() {
            for by in 0..table.rows {
                for bx in 0..table.cols {
                    let i = table.index(bx, by);
                    if table.flat[i] {
                        continue;
                    }
                    buckets[level]
                        .entry(table.hash[i])
                        .or_default()
                        .push((bx as u32 * 8, by as u32 * 8));
                }
            }
        }
        Self { levels, buckets }
    }

    /// Hash of the source block at `(x, y)`, which the build already computed.
    /// `None` when the block runs past the source or is flat.
    pub(crate) fn hash_at(&self, size: usize, x: usize, y: usize) -> Option<u64> {
        let level = size_index(size)?;
        let table = &self.levels[level];
        let (bx, by) = (x / 8, y / 8);
        if !x.is_multiple_of(8) || !y.is_multiple_of(8) || bx >= table.cols || by >= table.rows {
            return None;
        }
        let i = table.index(bx, by);
        if table.flat[i] {
            return None;
        }
        Some(table.hash[i])
    }

    /// Source positions whose block hashes equal `hash`.
    pub(crate) fn matches(&self, size: usize, hash: u64) -> &[(u32, u32)] {
        let Some(index) = size_index(size) else {
            return &[];
        };
        self.buckets[index]
            .get(&hash)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// FNV-1a over one 8×8 source block, with its flatness and first sample.
fn hash_8x8(y: &[u16], stride: usize, x: usize, y0: usize) -> (u64, bool, u16) {
    let first = y[y0 * stride + x];
    let mut flat = true;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for row in 0..8 {
        let base = (y0 + row) * stride + x;
        for &sample in &y[base..base + 8] {
            flat &= sample == first;
            hash ^= u64::from(sample);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    (hash, flat, first)
}

/// Chroma-position parity a block vector must satisfy so that its chroma
/// counterpart stays on integer samples. A fractional chroma vector would put
/// the decoder into the chroma interpolation path (and pull the §8.5.3.2.1
/// margins into the validity test); restricting the search costs almost nothing
/// on screen content, where matches are aligned anyway.
#[inline]
pub(crate) fn parity_ok(bvx: i32, bvy: i32, sub_w: usize, sub_h: usize) -> bool {
    (sub_w == 1 || bvx % 2 == 0) && (sub_h == 1 || bvy % 2 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(x: usize, y: usize, size: usize) -> CuGeom {
        CuGeom {
            x,
            y,
            size,
            pic_w: 256,
            pic_h: 256,
            ctb_log2: 6,
        }
    }

    #[test]
    fn block_vectors_must_not_overlap_the_current_block() {
        let cu = geom(64, 64, 16);
        // Entirely to the left.
        assert!(source_area(&cu, -16, 0).is_some());
        // One sample of overlap on the left.
        assert!(source_area(&cu, -15, 0).is_none());
        // Entirely above.
        assert!(source_area(&cu, 0, -16).is_some());
        assert!(source_area(&cu, 0, -15).is_none());
        // Diagonally up-left is fine as soon as either condition holds.
        assert!(source_area(&cu, -16, -16).is_some());
    }

    #[test]
    fn block_vectors_stay_inside_the_picture() {
        let cu = geom(0, 64, 16);
        assert!(source_area(&cu, -1, 0).is_none(), "source starts at x=-1");
        let cu = geom(64, 0, 16);
        assert!(source_area(&cu, 0, -1).is_none(), "source starts at y=-1");
    }

    #[test]
    fn the_ctb_wavefront_bounds_how_far_right_an_above_reference_reaches() {
        // CU at CTB (1,1). A reference one CTB row up may reach one CTB to the
        // right (equation 8-106), but not two.
        let cu = geom(64, 64, 16);
        assert!(source_area(&cu, 64, -64).is_some(), "one CTB right, one up");
        assert!(
            source_area(&cu, 128, -64).is_none(),
            "two CTBs right of the current CTB is past the wavefront"
        );
    }

    #[test]
    fn amvp_uses_the_left_group_first_and_drops_the_duplicate() {
        let always = |_: usize, _: usize| true;
        let motion = |x: usize, y: usize| -> Option<(i16, i16)> {
            if x < 64 {
                Some((-8, 0))
            } else if y < 64 {
                Some((0, -8))
            } else {
                None
            }
        };
        let preds = amvp_predictors(64, 64, 16, &always, &motion);
        assert_eq!(preds[0], (-8, 0));
        assert_eq!(preds[1], (0, -8));

        // With no left neighbour at all, the B candidate stands in as A and its
        // re-derivation is a duplicate, so the second predictor stays zero.
        let no_left = |x: usize, _y: usize| x >= 64;
        let above_only = |_: usize, _: usize| Some((0i16, -8i16));
        let preds = amvp_predictors(64, 64, 16, &no_left, &above_only);
        assert_eq!(preds[0], (0, -8));
        assert_eq!(preds[1], (0, 0));
    }

    #[test]
    fn intra_neighbors_give_boundary_strength_two() {
        assert_eq!(boundary_strength(BV_INTRA, pack_bv(-8, 0)), 2);
        assert_eq!(boundary_strength(pack_bv(-8, 0), BV_INTRA), 2);
        assert_eq!(boundary_strength(pack_bv(-8, 0), pack_bv(-8, 0)), 0);
        assert_eq!(boundary_strength(pack_bv(-8, 0), pack_bv(-9, 0)), 1);
        assert_eq!(boundary_strength(pack_bv(-8, 0), pack_bv(-8, -1)), 1);
    }

    #[test]
    fn packing_round_trips_negative_vectors() {
        for bv in [(-1i16, -1i16), (0, 0), (-4096, 4095), (1234, -5678)] {
            assert_eq!(unpack_bv(pack_bv(bv.0, bv.1)), Some(bv));
        }
        assert_eq!(unpack_bv(BV_INTRA), None);
    }

    #[test]
    fn the_hash_table_finds_a_repeated_block_and_skips_flat_ones() {
        let (w, h) = (64usize, 16usize);
        let mut y = vec![0u16; w * h];
        // A distinctive 8×8 pattern at (0,0), repeated at (32,0).
        for row in 0..8 {
            for col in 0..8 {
                let v = ((row * 8 + col) * 3) as u16;
                y[row * w + col] = v;
                y[row * w + 32 + col] = v;
            }
        }
        let table = HashTable::build(&y, w, h);
        let hash = table.hash_at(8, 32, 0).expect("pattern is not flat");
        let matches = table.matches(8, hash);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(32, 0)));
        // The all-zero block at (8,0) is flat and must not be indexed.
        assert!(table.hash_at(8, 8, 0).is_none());
        // Equal content must hash equal at every level, including the ones
        // built by combining sub-block hashes.
        assert_eq!(table.hash_at(8, 0, 0), table.hash_at(8, 32, 0));
    }

    #[test]
    fn chroma_parity_only_constrains_subsampled_axes() {
        assert!(parity_ok(-3, -3, 1, 1));
        assert!(!parity_ok(-3, -4, 2, 2));
        assert!(parity_ok(-4, -4, 2, 2));
        // 4:2:2 subsamples horizontally only.
        assert!(parity_ok(-4, -3, 2, 1));
        assert!(!parity_ok(-3, -4, 2, 1));
    }
}
