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

//! HEVC intra prediction for 8×8 luma and 4×4/8×8 chroma blocks.
//!
//! HEVC spec §8.4.4. The encoder uses PLANAR mode everywhere (mode 0), which keeps
//! the most-probable-mode derivation trivial and gives good results for both flat
//! and gradient regions in still images.

/// Predict an N×N block using Planar mode (HEVC §8.4.4.2.4).
///
/// `above[0..n]` = row above (above[0] is sample at x=0), `above[n]` = top-right.
/// `left[0..n]`  = column left, `left[n]` = bottom-left.
#[inline]
pub(crate) fn predict_planar(above: &[u16], left: &[u16], n: usize) -> Vec<u16> {
    let mut pred = vec![0u16; n * n];
    let top_right = above[n] as i32;
    let bottom_left = left[n] as i32;
    let log2 = n.trailing_zeros();

    for row in 0..n {
        for col in 0..n {
            let h = (n - 1 - col) as i32 * left[row] as i32 + (col + 1) as i32 * top_right;
            let v = (n - 1 - row) as i32 * above[col] as i32 + (row + 1) as i32 * bottom_left;
            pred[row * n + col] = ((h + v + n as i32) >> (log2 + 1)) as u16;
        }
    }
    pred
}

///
/// Inputs use the layout: `corner` = above-left sample, `above[0..=n]` =
/// above samples then top-right (length n+1), `left[0..=n]` = left samples
/// then bottom-left (length n+1). Returns filtered (above', left') in the same
/// layout that `predict_planar` consumes (length n+1 each). Endpoints that have
/// no outer neighbour (top-right, bottom-left) are filtered using the sample
/// beyond them, which for our restricted reference we approximate by clamping.
pub(crate) fn filter_references(
    corner: u16,
    above: &[u16],
    left: &[u16],
    n: usize,
) -> (Vec<u16>, Vec<u16>) {
    // `above` and `left` have length 2n+1 (indices 0..2n). The HEVC [1 2 1]/4
    // filter (§8.4.4.2.3) is applied to every interior reference sample; only the
    // two extreme endpoints (corner and the farthest above-right / below-left,
    // index 2n) are left unfiltered. predict_planar reads indices 0..=n, so we
    // need filtered values at 0..=n, which requires unfiltered neighbours up to
    // index n+1 — present because we gathered 2n samples.
    let mut fa = above.to_vec();
    let mut fl = left.to_vec();
    let ext = above.len() - 1; // = 2n
    // above[0] uses the corner as its previous neighbour.
    if ext >= 1 {
        fa[0] = ((corner as i32 + 2 * above[0] as i32 + above[1] as i32 + 2) >> 2) as u16;
    }
    for x in 1..ext {
        fa[x] = ((above[x - 1] as i32 + 2 * above[x] as i32 + above[x + 1] as i32 + 2) >> 2) as u16;
    }
    // index ext (=2n) left unfiltered (extreme endpoint).
    if ext >= 1 {
        fl[0] = ((corner as i32 + 2 * left[0] as i32 + left[1] as i32 + 2) >> 2) as u16;
    }
    for y in 1..ext {
        fl[y] = ((left[y - 1] as i32 + 2 * left[y] as i32 + left[y + 1] as i32 + 2) >> 2) as u16;
    }
    let _ = n;
    (fa, fl)
}

/// Decode-order ("z-scan") index of the 8×8 (luma) / 4×4 (chroma) block that
/// contains pixel (r, c). CTUs are coded in raster order; within a CTU the four
/// sub-blocks follow Z-scan [(0,0),(0,1),(1,0),(1,1)]. `blk` is the sub-block
/// size (8 for luma, 4 for chroma); `ctu` is the CTU size in that plane
/// (16 luma, 8 chroma); `ctus_x` is the number of CTUs per row.
fn decode_order(r: usize, c: usize, blk: usize, ctu: usize, ctus_x: usize) -> i64 {
    let ctu_r = r / ctu;
    let ctu_c = c / ctu;
    let ctu_idx = ctu_r * ctus_x + ctu_c;
    // Sub-block grid position within the CTU (in units of `blk`).
    let sub_r = (r % ctu) / blk;
    let sub_c = (c % ctu) / blk;
    // Hierarchical Z-scan (Morton order): interleave the bits of sub_r and sub_c.
    // For a 2×2 grid (16-CTU / 8-blk) this reduces to sub_r*2+sub_c.
    // For a 4×4 grid (32-CTU / 8-blk) it produces the correct nested Z-order:
    //   the CTB splits into four quadrants (Z-scan), each into four CUs (Z-scan).
    let grid = ctu / blk; // sub-blocks per side (2 or 4)
    let mut z: u64 = 0;
    let mut bit = 0;
    let mut sr = sub_r as u64;
    let mut sc = sub_c as u64;
    let mut g = grid;
    while g > 1 {
        z |= (sc & 1) << (2 * bit);
        z |= (sr & 1) << (2 * bit + 1);
        sr >>= 1;
        sc >>= 1;
        bit += 1;
        g >>= 1;
    }
    let cells = (grid * grid) as i64;
    (ctu_idx as i64) * cells + z as i64
}

/// Returns true if neighbour pixel (nr,nc) was already reconstructed when coding
/// the block whose top-left is (block_row, block_col).
#[allow(clippy::too_many_arguments)]
fn is_available(
    nr: i64,
    nc: i64,
    block_row: usize,
    block_col: usize,
    blk: usize,
    ctu: usize,
    ctus_x: usize,
    width: usize,
    height: usize,
) -> bool {
    if nr < 0 || nc < 0 || nr as usize >= height || nc as usize >= width {
        return false;
    }
    let cur = decode_order(block_row, block_col, blk, ctu, ctus_x);
    let nb = decode_order(nr as usize, nc as usize, blk, ctu, ctus_x);
    nb < cur
}

/// Extract reference samples for an N×N block at (block_row, block_col),
/// returning (corner, above[n+1], left[n+1]) following the HEVC reference-sample
/// substitution process (§8.4.4.2.2): unavailable samples are propagated from the
/// nearest available sample, scanning from bottom-left up the left column, through
/// the corner, then left-to-right along the top. If no sample is available, all
/// are set to 128 (1 << (bitDepth-1)).
///
/// `ctu` is the CTU size in this plane and `ctus_x` the CTUs per row; together
/// with the block size `n` these drive the decode-order availability test so the
/// encoder never references a neighbour the decoder has not yet reconstructed.
/// Public wrapper exposing luma decode order for chroma availability mapping.
pub(crate) fn luma_decode_order(r: usize, c: usize, ctus_x: usize) -> i64 {
    decode_order(r, c, 8, 64, ctus_x)
}

/// Reference samples for a chroma N×N block, with availability computed in LUMA
/// space. Chroma TBs are decoded in lockstep with their co-located luma CU, so a
/// chroma neighbour is available iff the luma block covering the corresponding luma
/// position was decoded before the current luma CU. This is correct for both 4:2:0
/// (sub_h = 2) and 4:2:2 (sub_h = 1), where the chroma CTB is non-square and a plain
/// chroma Morton scan would be wrong.
///
/// `sub_w`/`sub_h` are the subsampling factors; `luma_w`/`luma_h` the luma picture
/// dimensions; `luma_ctus_x` the luma CTUs per row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_reference_samples_chroma(
    plane: &[u16],
    stride: usize,
    block_row: usize,
    block_col: usize,
    chroma_h: usize,
    n: usize,
    sub_w: usize,
    sub_h: usize,
    luma_w: usize,
    luma_h: usize,
    luma_ctus_x: usize,
    cur_luma_row: usize,
    cur_luma_col: usize,
    neutral: u16,
) -> (u16, Vec<u16>, Vec<u16>) {
    let width = stride;
    let ext = 2 * n;
    const MAXE: usize = 17;
    let mut above = vec![0u16; ext + 1];
    let mut left = vec![0u16; ext + 1];
    let mut avail_above = [false; MAXE];
    let mut avail_left = [false; MAXE];
    let mut corner = 0u16;
    let mut avail_corner = false;

    let cur_luma = luma_decode_order(cur_luma_row, cur_luma_col, luma_ctus_x);
    let avail = |nr: i64, nc: i64, block_row: usize| -> bool {
        if nr < 0 || nc < 0 || nr as usize >= chroma_h || nc as usize >= width {
            return false;
        }
        let lr = (nr as usize) * sub_h;
        let lc = (nc as usize) * sub_w;
        if lr >= luma_h || lc >= luma_w {
            return false;
        }
        let nb_luma = luma_decode_order(lr, lc, luma_ctus_x);
        if nb_luma < cur_luma {
            return true;
        }
        nb_luma == cur_luma && (nr as usize) < block_row
    };

    {
        let nr = block_row as i64 - 1;
        let nc = block_col as i64 - 1;
        if avail(nr, nc, block_row) {
            corner = plane[(nr as usize) * stride + nc as usize];
            avail_corner = true;
        }
    }
    {
        let nr = block_row as i64 - 1;
        for i in 0..=ext {
            let nc = block_col as i64 + i as i64;
            if avail(nr, nc, block_row) {
                above[i] = plane[(nr as usize) * stride + nc as usize];
                avail_above[i] = true;
            }
        }
    }
    {
        let nc = block_col as i64 - 1;
        for i in 0..=ext {
            let nr = block_row as i64 + i as i64;
            if avail(nr, nc, block_row) {
                left[i] = plane[(nr as usize) * stride + nc as usize];
                avail_left[i] = true;
            }
        }
    }

    let any = avail_corner || avail_above.iter().any(|&a| a) || avail_left.iter().any(|&a| a);
    if !any {
        return (neutral, vec![neutral; ext + 1], vec![neutral; ext + 1]);
    }

    // Same substitution scan as the luma path.
    let total = (ext + 1) + 1 + (ext + 1);
    const MAXT: usize = 35;
    let mut vals = [0u16; MAXT];
    let mut av = [false; MAXT];
    for i in 0..=ext {
        vals[i] = left[ext - i];
        av[i] = avail_left[ext - i];
    }
    vals[ext + 1] = corner;
    av[ext + 1] = avail_corner;
    for i in 0..=ext {
        vals[(ext + 2) + i] = above[i];
        av[(ext + 2) + i] = avail_above[i];
    }
    let first = av[..total].iter().position(|&a| a).unwrap();
    let firstval = vals[first];
    for k in 0..first {
        vals[k] = firstval;
        av[k] = true;
    }
    for k in 1..total {
        if !av[k] {
            vals[k] = vals[k - 1];
            av[k] = true;
        }
    }
    for i in 0..=ext {
        left[ext - i] = vals[i];
    }
    corner = vals[ext + 1];
    for i in 0..=ext {
        above[i] = vals[(ext + 2) + i];
    }

    (corner, above, left)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_reference_samples(
    plane: &[u16],
    stride: usize,
    block_row: usize,
    block_col: usize,
    height: usize,
    n: usize,
    ctu: usize,
    ctus_x: usize,
    neutral: u16,
) -> (u16, Vec<u16>, Vec<u16>) {
    let width = stride;
    let ext = 2 * n; // gather 2N samples per side so the filter can process index N.
    // ext <= 16 (n <= 8); scratch availability flags live on the stack.
    const MAXE: usize = 17; // ext+1 <= 17
    let mut above = vec![0u16; ext + 1]; // above[0..2n] (returned)
    let mut left = vec![0u16; ext + 1]; // left[0..2n]  (returned)
    let mut avail_above = [false; MAXE];
    let mut avail_left = [false; MAXE];
    let mut corner = 0u16;
    let mut avail_corner = false;

    // Corner (above-left)
    {
        let nr = block_row as i64 - 1;
        let nc = block_col as i64 - 1;
        if is_available(nr, nc, block_row, block_col, n, ctu, ctus_x, width, height) {
            corner = plane[(nr as usize) * stride + nc as usize];
            avail_corner = true;
        }
    }
    // Above row: above[i] at (block_row-1, block_col+i), i=0..=2n
    {
        let nr = block_row as i64 - 1;
        for i in 0..=ext {
            let nc = block_col as i64 + i as i64;
            if is_available(nr, nc, block_row, block_col, n, ctu, ctus_x, width, height) {
                above[i] = plane[(nr as usize) * stride + nc as usize];
                avail_above[i] = true;
            }
        }
    }
    // Left col: left[i] at (block_row+i, block_col-1), i=0..=2n
    {
        let nc = block_col as i64 - 1;
        for i in 0..=ext {
            let nr = block_row as i64 + i as i64;
            if is_available(nr, nc, block_row, block_col, n, ctu, ctus_x, width, height) {
                left[i] = plane[(nr as usize) * stride + nc as usize];
                avail_left[i] = true;
            }
        }
    }

    let any = avail_corner || avail_above.iter().any(|&a| a) || avail_left.iter().any(|&a| a);
    if !any {
        return (neutral, vec![neutral; ext + 1], vec![neutral; ext + 1]);
    }

    // Ordered scan: bottom-most left (left[2n]) up to left[0], corner, above[0..2n].
    // total = 2*(ext+1)+1 <= 35; keep the scratch on the stack.
    let total = (ext + 1) + 1 + (ext + 1);
    const MAXT: usize = 35;
    let mut vals = [0u16; MAXT];
    let mut av = [false; MAXT];
    for i in 0..=ext {
        vals[i] = left[ext - i];
        av[i] = avail_left[ext - i];
    }
    vals[ext + 1] = corner;
    av[ext + 1] = avail_corner;
    for i in 0..=ext {
        vals[(ext + 2) + i] = above[i];
        av[(ext + 2) + i] = avail_above[i];
    }
    let first = av[..total].iter().position(|&a| a).unwrap();
    let firstval = vals[first];
    for k in 0..first {
        vals[k] = firstval;
        av[k] = true;
    }
    for k in 1..total {
        if !av[k] {
            vals[k] = vals[k - 1];
            av[k] = true;
        }
    }
    for i in 0..=ext {
        left[ext - i] = vals[i];
    }
    corner = vals[ext + 1];
    for i in 0..=ext {
        above[i] = vals[(ext + 2) + i];
    }

    (corner, above, left)
}

/// Subtract prediction from original to obtain the residual block.
///
/// Returns `orig[i] - pred[i]` as f32 (level-shifted by -128 already handled
/// at caller level, so we work in pixel domain here).
pub(crate) fn compute_residual(orig: &[u16], pred: &[u16], n: usize) -> Vec<f32> {
    debug_assert_eq!(orig.len(), n * n);
    debug_assert_eq!(pred.len(), n * n);
    orig.iter()
        .zip(pred.iter())
        .map(|(&o, &p)| o as f32 - p as f32)
        .collect()
}

/// Reconstruct pixels: clamp(pred[i] + residual[i]) to [0, max_val] → u16.
/// `max_val` is (1<<bit_depth)-1 (255 for 8-bit, 1023 for 10-bit).
pub(crate) fn reconstruct(pred: &[u16], residual: &[f32], n: usize, max_val: u16) -> Vec<u16> {
    let _ = n;
    pred.iter()
        .zip(residual.iter())
        .map(|(&p, &r)| (p as f32 + r).round().clamp(0.0, max_val as f32) as u16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_corners() {
        let mut above = vec![0u16; 9];
        let mut left = vec![0u16; 9];
        above[8] = 255; // top-right
        left[8] = 255; // bottom-left
        let pred = predict_planar(&above, &left, 8);
        // Bottom-left pixel should be close to bottom-left sample
        // Top-right pixel should be close to top-right sample
        assert!(pred[7] > 100); // top-right area of block
        assert!(pred[8 * 7] > 100); // bottom-left area
    }

    #[test]
    fn residual_zero_when_perfect() {
        let pixels = vec![128u16; 64];
        let pred = vec![128u16; 64];
        let res = compute_residual(&pixels, &pred, 8);
        assert!(res.iter().all(|&r| r == 0.0));
    }
}
