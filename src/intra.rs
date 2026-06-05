//! HEVC intra prediction for 8×8 luma and 4×4 chroma blocks.
//!
//! HEVC spec §8.4.4. We implement:
//!   - DC mode   (mode 1): mean of above + left reference samples
//!   - Planar mode (mode 0): bilinear blend from above-right + below-left
//!
//! For a still-image encoder with no inter prediction, DC mode gives good
//! results for uniform areas and Planar handles gradients well.
//! We select between the two based on variance of the reference samples.

/// Intra prediction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntraMode {
    /// Mode 0: bilinear ramp (Planar)
    Planar = 0,
    /// Mode 1: mean of boundary samples (DC)
    Dc = 1,
}

/// Predict an N×N block using DC mode.
///
/// Reference samples:
///   `above[0..n]` = row immediately above (left-to-right)
///   `left[0..n]`  = column immediately to the left (top-to-bottom)
///
/// Returns an N×N prediction block as a flat Vec (row-major).
pub fn predict_dc(above: &[u8], left: &[u8], n: usize) -> Vec<u8> {
    let sum: u32 =
        above.iter().map(|&x| x as u32).sum::<u32>() + left.iter().map(|&x| x as u32).sum::<u32>();
    let dc = ((sum + n as u32) / (2 * n as u32)) as u8; // rounded mean
    vec![dc; n * n]
}

/// Predict an N×N block using Planar mode (HEVC §8.4.4.2.4).
///
/// `above[0..n]` = row above (above[0] is sample at x=0), `above[n]` = top-right.
/// `left[0..n]`  = column left, `left[n]` = bottom-left.
pub fn predict_planar(above: &[u8], left: &[u8], n: usize) -> Vec<u8> {
    let mut pred = vec![0u8; n * n];
    let top_right = above[n] as i32;
    let bottom_left = left[n] as i32;
    let log2 = n.trailing_zeros();

    for row in 0..n {
        for col in 0..n {
            let h = (n - 1 - col) as i32 * left[row] as i32 + (col + 1) as i32 * top_right;
            let v = (n - 1 - row) as i32 * above[col] as i32 + (row + 1) as i32 * bottom_left;
            pred[row * n + col] = ((h + v + n as i32) >> (log2 + 1)) as u8;
        }
    }
    pred
}

/// Rectangular PLANAR prediction for a `w`×`h` block (HEVC §8.4.4.2.4 generalised
/// to non-square blocks, as used for 4:2:2 chroma where the block is 4 wide × 8
/// tall). `above` has length ≥ w+1 (above[0..w] plus top-right at above[w]); `left`
/// has length ≥ h+1 (left[0..h] plus bottom-left at left[h]). Returns row-major w*h.
pub fn predict_planar_rect(above: &[u8], left: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut pred = vec![0u8; w * h];
    let top_right = above[w] as i32;
    let bottom_left = left[h] as i32;
    // HEVC planar: predSamples[x][y] =
    //   ( (W-1-x)*p[-1][y] + (x+1)*p[W][-1] + (H-1-y)*p[x][-1] + (y+1)*p[-1][H]
    //     + W (when... ) ) — the standard uses normalisation by (W*H) with shift.
    // For W,H powers of two we use: (h_term*H + v_term*W ...) Simpler: use the
    // separable form with rounding by (w*h):
    let wh = (w * h) as i32;
    let shift = (w * h).trailing_zeros(); // log2(w*h)
    for y in 0..h {
        for x in 0..w {
            let horiz = (w as i32 - 1 - x as i32) * left[y] as i32 + (x as i32 + 1) * top_right;
            let vert = (h as i32 - 1 - y as i32) * above[x] as i32 + (y as i32 + 1) * bottom_left;
            // predV scaled by W, predH scaled by H → combine and normalise by 2*W*H.
            let val = (horiz * h as i32 + vert * w as i32 + wh) >> (shift + 1);
            pred[y * w + x] = val as u8;
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
pub fn filter_references(corner: u8, above: &[u8], left: &[u8], n: usize) -> (Vec<u8>, Vec<u8>) {
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
        fa[0] = ((corner as i32 + 2 * above[0] as i32 + above[1] as i32 + 2) >> 2) as u8;
    }
    for x in 1..ext {
        fa[x] = ((above[x - 1] as i32 + 2 * above[x] as i32 + above[x + 1] as i32 + 2) >> 2) as u8;
    }
    // index ext (=2n) left unfiltered (extreme endpoint).
    if ext >= 1 {
        fl[0] = ((corner as i32 + 2 * left[0] as i32 + left[1] as i32 + 2) >> 2) as u8;
    }
    for y in 1..ext {
        fl[y] = ((left[y - 1] as i32 + 2 * left[y] as i32 + left[y + 1] as i32 + 2) >> 2) as u8;
    }
    let _ = n;
    (fa, fl)
}

/// Choose DC vs Planar based on variance of the reference samples.
///
/// High variance → Planar (samples vary a lot → gradient likely).
/// Low variance  → DC (uniform area).
pub fn choose_mode(above: &[u8], left: &[u8]) -> IntraMode {
    let all: Vec<u8> = above.iter().chain(left.iter()).copied().collect();
    let n = all.len();
    let mean = all.iter().map(|&x| x as u32).sum::<u32>() / n as u32;
    let var = all
        .iter()
        .map(|&x| {
            let d = x as i64 - mean as i64;
            (d * d) as u64
        })
        .sum::<u64>()
        / n as u64;

    if var > 64 * 64 {
        IntraMode::Planar
    } else {
        IntraMode::Dc
    }
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
pub fn luma_decode_order(r: usize, c: usize, ctus_x: usize) -> i64 {
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
pub fn get_reference_samples_chroma(
    plane: &[u8],
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
) -> (u8, Vec<u8>, Vec<u8>) {
    let width = stride;
    let ext = 2 * n;
    let mut above = vec![0u8; ext + 1];
    let mut left = vec![0u8; ext + 1];
    let mut avail_above = vec![false; ext + 1];
    let mut avail_left = vec![false; ext + 1];
    let mut corner = 0u8;
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
        return (128, vec![128; ext + 1], vec![128; ext + 1]);
    }

    // Same substitution scan as the luma path.
    let total = (ext + 1) + 1 + (ext + 1);
    let mut vals = vec![0u8; total];
    let mut av = vec![false; total];
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
    let first = av.iter().position(|&a| a).unwrap();
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

pub fn get_reference_samples(
    plane: &[u8],
    stride: usize,
    block_row: usize,
    block_col: usize,
    height: usize,
    n: usize,
    ctu: usize,
    ctus_x: usize,
) -> (u8, Vec<u8>, Vec<u8>) {
    let width = stride;
    let ext = 2 * n; // gather 2N samples per side so the filter can process index N.
    let mut above = vec![0u8; ext + 1]; // above[0..2n]
    let mut left = vec![0u8; ext + 1]; // left[0..2n]
    let mut avail_above = vec![false; ext + 1];
    let mut avail_left = vec![false; ext + 1];
    let mut corner = 0u8;
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
        return (128, vec![128; ext + 1], vec![128; ext + 1]);
    }

    // Ordered scan: bottom-most left (left[2n]) up to left[0], corner, above[0..2n].
    let total = (ext + 1) + 1 + (ext + 1);
    let mut vals = vec![0u8; total];
    let mut av = vec![false; total];
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
    let first = av.iter().position(|&a| a).unwrap();
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
pub fn compute_residual(orig: &[u8], pred: &[u8], n: usize) -> Vec<f32> {
    debug_assert_eq!(orig.len(), n * n);
    debug_assert_eq!(pred.len(), n * n);
    orig.iter()
        .zip(pred.iter())
        .map(|(&o, &p)| o as f32 - p as f32)
        .collect()
}

/// Reconstruct pixels: clamp(pred[i] + residual[i]) → u8.
pub fn reconstruct(pred: &[u8], residual: &[f32], n: usize) -> Vec<u8> {
    pred.iter()
        .zip(residual.iter())
        .map(|(&p, &r)| (p as f32 + r).round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_flat_block() {
        let above = vec![100u8; 9]; // n=8, +1 corner
        let left = vec![100u8; 9];
        let pred = predict_dc(&above[..8], &left[..8], 8);
        assert_eq!(pred, vec![100u8; 64]);
    }

    #[test]
    fn dc_mixed() {
        let above = vec![200u8; 8];
        let left = vec![100u8; 8];
        let pred = predict_dc(&above, &left, 8);
        // mean = (200*8 + 100*8) / 16 = 150
        assert_eq!(pred[0], 150);
    }

    #[test]
    fn planar_corners() {
        let mut above = vec![0u8; 9];
        let mut left = vec![0u8; 9];
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
        let pixels = vec![128u8; 64];
        let pred = vec![128u8; 64];
        let res = compute_residual(&pixels, &pred, 8);
        assert!(res.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn choose_mode_uniform() {
        let above = vec![128u8; 8];
        let left = vec![128u8; 8];
        assert_eq!(choose_mode(&above, &left), IntraMode::Dc);
    }

    #[test]
    fn choose_mode_gradient() {
        let above: Vec<u8> = (0..8u8).map(|i| i * 30).collect();
        let left: Vec<u8> = (0..8u8).map(|i| 255 - i * 30).collect();
        assert_eq!(choose_mode(&above, &left), IntraMode::Planar);
    }
}
