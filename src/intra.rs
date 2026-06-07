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
pub(crate) fn predict_planar(above: &[u16], left: &[u16], n: usize) -> [u16; 256] {
    let mut pred = [0u16; 256];
    let top_right = above[n] as i32;
    let bottom_left = left[n] as i32;
    let log2 = n.trailing_zeros();

    for (row, &left) in left[..n].iter().enumerate() {
        let lr = left as i32;
        let v_w = (n - 1 - row) as i32;
        let v_b = (row + 1) as i32 * bottom_left;
        let arow = &above[..n];
        let prow = &mut pred[row * n..row * n + n];
        for (col, (prow, &arow)) in prow[..n].iter_mut().zip(arow.iter()).enumerate() {
            let h = (n - 1 - col) as i32 * lr + (col + 1) as i32 * top_right;
            let v = v_w * arow as i32 + v_b;
            *prow = ((h + v + n as i32) >> (log2 + 1)) as u16;
        }
    }
    pred
}

/// Whether the [1 2 1] reference-smoothing filter applies for this mode/size
/// (HEVC §8.4.4.2.3, luma). DC and 4×4 never filter; otherwise it depends on the
/// mode's angular distance from pure horizontal (10) / vertical (26).
pub(crate) fn should_filter_refs(mode: u8, n: usize) -> bool {
    if mode == DC || n == 4 {
        return false;
    }
    // PLANAR (mode 0) is treated as distance 10 (= min(|0-26|,|0-10|)).
    let dist = if mode == PLANAR {
        10
    } else {
        (mode as i32 - 26).abs().min((mode as i32 - 10).abs())
    };
    let thresh = match n {
        8 => 7,
        16 => 1,
        _ => 0, // 32
    };
    dist > thresh
}

/// intraPredAngle for angular modes 2..=34 (HEVC Table 8-5), indexed by mode.
static INTRA_PRED_ANGLE: [i32; 35] = [
    0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13,
    -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];
/// invAngle for modes 11..=25 (HEVC Table 8-6), indexed by mode (0 elsewhere).
static INV_ANGLE: [i32; 35] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390,
    -482, -630, -910, -1638, -4096, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub(crate) const PLANAR: u8 = 0;
pub(crate) const DC: u8 = 1;

/// Predict an N×N block with DC mode (HEVC §8.4.4.2.5). `filter_boundary`
/// applies the luma edge smoothing of the top row / left column / corner
/// (skipped for chroma and for n == 32).
pub(crate) fn predict_dc(
    above: &[u16],
    left: &[u16],
    n: usize,
    filter_boundary: bool,
) -> [u16; 256] {
    let mut sum = 0i32;
    for (&above, &left) in above[..n].iter().zip(left[..n].iter()) {
        sum += above as i32 + left as i32;
    }
    let log2 = n.trailing_zeros();
    let dc = (sum + n as i32) >> (log2 + 1);
    let mut pred = [0u16; 256];
    for v in pred[..n * n].iter_mut() {
        *v = dc as u16;
    }
    if filter_boundary && n < 32 {
        // Corner, first row and first column get a 3:1 / 2:1:1 blend with the refs.
        pred[0] = ((left[0] as i32 + 2 * dc + above[0] as i32 + 2) >> 2) as u16;
        for (pred, &above) in pred[1..n].iter_mut().zip(above[1..n].iter()) {
            *pred = ((above as i32 + 3 * dc + 2) >> 2) as u16;
        }
        for (y, &left) in (1..n).zip(left[1..n].iter()) {
            pred[y * n] = ((left as i32 + 3 * dc + 2) >> 2) as u16;
        }
    }
    pred
}

/// Predict an N×N block with an angular mode (2..=34, HEVC §8.4.4.2.6).
/// `corner` = p[-1][-1]; `above`/`left` hold p[x][-1] / p[-1][y] for x,y = 0..2n-1.
/// `filter_boundary` applies the pure-vertical (26) / horizontal (10) luma edge
/// filter (skipped for chroma and n == 32).
pub(crate) fn predict_angular(
    corner: u16,
    above: &[u16],
    left: &[u16],
    n: usize,
    mode: u8,
    filter_boundary: bool,
    max_val: i32,
) -> [u16; 256] {
    let angle = INTRA_PRED_ANGLE[mode as usize];
    let mut pred = [0u16; 256];

    // Build the main reference array `ref[i]`, i in 0..=2n, indexed from a base
    // offset so negative projections (angle < 0) are representable.
    // For vertical modes (>=18) the main reference is the above row; for
    // horizontal modes (<18) it is the left column (block predicted transposed).
    let vertical = mode >= 18;
    let (main, side): (&[u16], &[u16]) = if vertical {
        (above, left)
    } else {
        (left, above)
    };

    // r[OFF + i] = ref[i]; OFF lets i go negative down to -n.
    const OFF: usize = 32;
    let mut r = [0i32; OFF + 64 + 1];
    r[OFF] = corner as i32;
    r[OFF + 1..=OFF + 2 * n]
        .iter_mut()
        .zip(main[..2 * n].iter())
        .for_each(|(dst, &src)| *dst = src as i32);
    if angle < 0 {
        let inv = INV_ANGLE[mode as usize];
        let last = (n as i32 * angle) >> 5; // most-negative index needed
        let mut x = -1;
        while x >= last {
            let idx = (x * inv + 128) >> 8; // project onto the side array
            // ref[x] = p[-1][-1 + idx]: corner if idx==0, else side[idx-1].
            r[(OFF as i32 + x) as usize] = if idx <= 0 {
                corner as i32
            } else {
                side[(idx - 1) as usize] as i32
            };
            x -= 1;
        }
    }

    // Primary axis = rows for vertical modes, columns for horizontal modes. For
    // horizontal modes HEVC swaps the roles of x and y: iIdx/iFact derive from
    // the primary index, the reference index uses the secondary index.
    for p_outer in 0..n {
        let pos = (p_outer as i32 + 1) * angle;
        let i_idx = pos >> 5;
        let i_fact = pos & 31;
        for p_inner in 0..n {
            let base = OFF as i32 + p_inner as i32 + i_idx + 1;
            let a = r[base as usize];
            let b = r[(base + 1) as usize];
            let val = (((32 - i_fact) * a + i_fact * b + 16) >> 5) as u16;
            if vertical {
                pred[p_outer * n + p_inner] = val; // [row y][col x]
            } else {
                pred[p_inner * n + p_outer] = val; // [row y][col x], transposed
            }
        }
    }

    if filter_boundary && n < 32 {
        let max = max_val;
        if mode == 26 {
            // pure vertical: first column blended with the left reference
            for y in 0..n {
                let v = above[0] as i32 + ((left[y] as i32 - corner as i32) >> 1);
                pred[y * n] = v.clamp(0, max) as u16;
            }
        } else if mode == 10 {
            // pure horizontal: first row blended with the above reference
            for (pred, &above) in pred[..n].iter_mut().zip(above[..n].iter()) {
                let v = left[0] as i32 + ((above as i32 - corner as i32) >> 1);
                *pred = v.clamp(0, max) as u16;
            }
        }
    }
    pred
}

/// Predict an N×N block for any luma mode (0=PLANAR, 1=DC, 2..=34 angular),
/// given raw (unfiltered) references. Applies mode-dependent reference smoothing
/// internally. `boundary_filter` enables the DC/H/V luma edge filters.
/// Predict a chroma TB with `mode`, from references already prepared by the
/// caller (4:4:4 pre-filters them; 4:2:0/4:2:2 do not). Chroma never applies the
/// DC/H/V boundary edge filter (cIdx > 0).
pub(crate) fn predict_chroma_tb(
    mode: u8,
    corner: u16,
    above: &[u16],
    left: &[u16],
    n: usize,
    max_val: i32,
) -> [u16; 256] {
    match mode {
        PLANAR => predict_planar(above, left, n),
        DC => predict_dc(above, left, n, false),
        _ => predict_angular(corner, above, left, n, mode, false, max_val),
    }
}

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
) -> ([u16; 33], [u16; 33]) {
    // `above` and `left` have length 2n+1 (indices 0..2n). The HEVC [1 2 1]/4
    // filter (§8.4.4.2.3) is applied to every interior reference sample; only the
    // two extreme endpoints (corner and the farthest above-right / below-left,
    // index 2n) are left unfiltered. predict_planar reads indices 0..=n, so we
    // need filtered values at 0..=n, which requires unfiltered neighbours up to
    // index n+1 — present because we gathered 2n samples.
    let mut fa = [0u16; 33];
    let mut fl = [0u16; 33];
    let ext = 2 * n; // = 2n; derived from n (not array len) so larger buffers are safe
    fa[..=ext].copy_from_slice(&above[..=ext]);
    fl[..=ext].copy_from_slice(&left[..=ext]);
    // above[0] uses the corner as its previous neighbour.
    if ext >= 1 {
        fa[0] = ((corner as i32 + 2 * above[0] as i32 + above[1] as i32 + 2) >> 2) as u16;
    }
    // Filter interior samples 1..=2n-2. The farthest real reference sample, index
    // 2n-1 (HEVC pF[nTbS*2-1][-1]), is left UNFILTERED — it is the only sample the
    // extreme angular modes 2 and 34 read, and filtering it desyncs the encoder's
    // reconstruction from a conformant decoder. (It keeps its raw value from the
    // copy above.)
    for x in 1..ext - 1 {
        fa[x] = ((above[x - 1] as i32 + 2 * above[x] as i32 + above[x + 1] as i32 + 2) >> 2) as u16;
    }
    if ext >= 1 {
        fl[0] = ((corner as i32 + 2 * left[0] as i32 + left[1] as i32 + 2) >> 2) as u16;
    }
    for y in 1..ext - 1 {
        fl[y] = ((left[y - 1] as i32 + 2 * left[y] as i32 + left[y + 1] as i32 + 2) >> 2) as u16;
    }
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

/// Shared reference-sample substitution (HEVC §8.4.4.2.1): fill unavailable
/// reference positions from their available neighbours along the scan order
/// (bottom-left → corner → top-right). `above`/`left` carry the raw samples;
/// the `avail_*` flags say which were populated.
#[allow(clippy::too_many_arguments)]
fn substitute_refs(
    mut corner: u16,
    mut above: [u16; 33],
    mut left: [u16; 33],
    avail_corner: bool,
    avail_above: &[bool; 33],
    avail_left: &[bool; 33],
    ext: usize,
    neutral: u16,
) -> (u16, [u16; 33], [u16; 33]) {
    let any = avail_corner
        || avail_above[..=ext].iter().any(|&a| a)
        || avail_left[..=ext].iter().any(|&a| a);
    if !any {
        return (neutral, [neutral; 33], [neutral; 33]);
    }
    let total = (ext + 1) + 1 + (ext + 1);
    const MAXT: usize = 67;
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

/// Geometry inputs for chroma reference-sample gathering. Bundles the block
/// position, plane extents, subsampling factors and the current luma-block
/// coordinates that drive the decode-order availability test, so the gather
/// function takes two plane slices + this struct instead of 14 loose scalars.
#[derive(Clone, Copy)]
pub(crate) struct ChromaRefGeometry {
    pub(crate) stride: usize,
    pub(crate) block_row: usize,
    pub(crate) block_col: usize,
    pub(crate) chroma_h: usize,
    pub(crate) n: usize,
    pub(crate) sub_w: usize,
    pub(crate) sub_h: usize,
    pub(crate) luma_w: usize,
    pub(crate) luma_h: usize,
    pub(crate) luma_ctus_x: usize,
    pub(crate) cur_luma_row: usize,
    pub(crate) cur_luma_col: usize,
    pub(crate) neutral: u16,
}

/// Gather chroma reference samples for **both** the Cb and Cr planes in one
/// pass. The availability of each reference position depends only on geometry
/// (the luma decode order), not on the plane data, so the per-sample Morton
/// `decode_order` work — the dominant cost — is done once and shared, instead of
/// being recomputed identically for Cb and then Cr.
#[allow(clippy::type_complexity)]
pub(crate) fn get_reference_samples_chroma_pair(
    plane_cb: &[u16],
    plane_cr: &[u16],
    geo: ChromaRefGeometry,
) -> ((u16, [u16; 33], [u16; 33]), (u16, [u16; 33], [u16; 33])) {
    let ChromaRefGeometry {
        stride,
        block_row,
        block_col,
        chroma_h,
        n,
        sub_w,
        sub_h,
        luma_w,
        luma_h,
        luma_ctus_x,
        cur_luma_row,
        cur_luma_col,
        neutral,
    } = geo;
    let width = stride;
    let ext = 2 * n;
    const MAXE: usize = 33;
    let mut cb_above = [0u16; MAXE];
    let mut cb_left = [0u16; MAXE];
    let mut cr_above = [0u16; MAXE];
    let mut cr_left = [0u16; MAXE];
    let mut avail_above = [false; MAXE];
    let mut avail_left = [false; MAXE];
    let mut cb_corner = 0u16;
    let mut cr_corner = 0u16;
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
            let idx = (nr as usize) * stride + nc as usize;
            cb_corner = plane_cb[idx];
            cr_corner = plane_cr[idx];
            avail_corner = true;
        }
    }
    {
        let nr = block_row as i64 - 1;
        for i in 0..=ext {
            let nc = block_col as i64 + i as i64;
            if avail(nr, nc, block_row) {
                let idx = (nr as usize) * stride + nc as usize;
                cb_above[i] = plane_cb[idx];
                cr_above[i] = plane_cr[idx];
                avail_above[i] = true;
            }
        }
    }
    {
        let nc = block_col as i64 - 1;
        for i in 0..=ext {
            let nr = block_row as i64 + i as i64;
            if avail(nr, nc, block_row) {
                let idx = (nr as usize) * stride + nc as usize;
                cb_left[i] = plane_cb[idx];
                cr_left[i] = plane_cr[idx];
                avail_left[i] = true;
            }
        }
    }

    let cb = substitute_refs(
        cb_corner,
        cb_above,
        cb_left,
        avail_corner,
        &avail_above,
        &avail_left,
        ext,
        neutral,
    );
    let cr = substitute_refs(
        cr_corner,
        cr_above,
        cr_left,
        avail_corner,
        &avail_above,
        &avail_left,
        ext,
        neutral,
    );
    (cb, cr)
}

/// Geometry inputs for luma reference-sample gathering: block position, plane
/// extents, block/CTU sizes, and the unavailable-sample fill value. Lets the
/// gather function take one plane slice + this struct instead of 9 scalars.
#[derive(Clone, Copy)]
pub(crate) struct LumaRefGeometry {
    pub(crate) stride: usize,
    pub(crate) block_row: usize,
    pub(crate) block_col: usize,
    pub(crate) height: usize,
    pub(crate) n: usize,
    pub(crate) ctu: usize,
    pub(crate) ctus_x: usize,
    pub(crate) neutral: u16,
}

pub(crate) fn get_reference_samples(
    plane: &[u16],
    geo: LumaRefGeometry,
) -> (u16, [u16; 33], [u16; 33]) {
    let LumaRefGeometry {
        stride,
        block_row,
        block_col,
        height,
        n,
        ctu,
        ctus_x,
        neutral,
    } = geo;
    let width = stride;
    let ext = 2 * n; // gather 2N samples per side so the filter can process index N.
    // ext <= 16 (n <= 8); reference rows live entirely on the stack.
    const MAXE: usize = 33; // ext+1 <= 33 (n<=16)
    let mut above = [0u16; MAXE]; // above[0..2n] (returned)
    let mut left = [0u16; MAXE]; // left[0..2n]  (returned)
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
        return (neutral, [neutral; 33], [neutral; 33]);
    }

    // Ordered scan: bottom-most left (left[2n]) up to left[0], corner, above[0..2n].
    // total = 2*(ext+1)+1 <= 35; keep the scratch on the stack.
    let total = (ext + 1) + 1 + (ext + 1);
    const MAXT: usize = 67;
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

/// Integer residual `orig[i] - pred[i]` as `i32`, written directly into a stack
/// buffer. The HEVC forward transform is integer-domain, so the previous
/// f32 residual + per-block `Vec<i32>` round-trip was pure overhead — this
/// avoids both the float conversion and the heap allocation.
pub(crate) fn compute_residual_i32(orig: &[u16], pred: &[u16], n: usize) -> [i32; 256] {
    let mut res = [0i32; 256];
    for (r, (&o, &p)) in res[..n * n].iter_mut().zip(orig.iter().zip(pred.iter())) {
        *r = o as i32 - p as i32;
    }
    res
}

/// Reconstruct pixels: clamp(pred[i] + residual[i]) to [0, max_val] → u16.
/// `max_val` is (1<<bit_depth)-1 (255 for 8-bit, 1023 for 10-bit). The residual
/// from the inverse transform is integer-valued, so this is exact integer math
/// (the previous `(p as f32 + r).round()` was a no-op round on integers).
pub(crate) fn reconstruct(pred: &[u16], residual: &[i32], n: usize, max_val: u16) -> [u16; 256] {
    let mut out = [0u16; 256];
    let mx = max_val as i32;
    for (o, (&p, &r)) in out[..n * n]
        .iter_mut()
        .zip(pred.iter().zip(residual.iter()))
    {
        *o = (p as i32 + r).clamp(0, mx) as u16;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[test]
    fn angular_mode18_negative_diagonal() {
        // Mode 18 (angle −32, vertical): predSamples[x][y] = ref[x−y], where the
        // negative half of ref is the left column projected via invAngle.
        let mut above = [0u16; 33];
        let mut left = [0u16; 33];
        for i in 0..8 {
            above[i] = (10 * (i + 1)) as u16; // 10,20,30,40,...
            left[i] = (50 + 10 * i) as u16; // 50,60,70,80,...
        }
        let corner = 5;
        let p = predict_angular(corner, &above, &left, 4, 18, false, 255);
        let expect = [
            5, 10, 20, 30, // y=0: ref[x]   = corner, above[0..2]
            50, 5, 10, 20, // y=1: ref[x-1] = left[0], corner, above[0..1]
            60, 50, 5, 10, // y=2
            70, 60, 50, 5, // y=3
        ];
        assert_eq!(&p[..16], &expect, "mode-18 negative diagonal");
    }

    #[test]
    fn dc_of_flat_is_flat() {
        let above = [100u16; 33];
        let left = [100u16; 33];
        let p = predict_dc(&above, &left, 8, false);
        assert!(p[..64].iter().all(|&v| v == 100));
    }

    #[test]
    fn dc_value_is_average() {
        let mut above = [0u16; 33];
        let mut left = [0u16; 33];
        for i in 0..8 {
            above[i] = 80;
            left[i] = 120;
        }
        // DC = (8*80 + 8*120 + 8) >> 4 = (640+960+8)>>4 = 1608>>4 = 100
        let p = predict_dc(&above, &left, 8, false);
        assert_eq!(p[63], 100); // interior (unfiltered) sample
    }

    #[test]
    fn vertical_mode_copies_above_row() {
        // Mode 26 (pure vertical), no boundary filter → every row equals the above row.
        let mut above = [0u16; 33];
        let left = [50u16; 33];
        for i in 0..8 {
            above[i] = (10 * i) as u16;
        }
        let p = predict_angular(50, &above, &left, 8, 26, false, 255);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(p[r * 8 + c], above[c], "row {r} col {c}");
            }
        }
    }

    #[test]
    fn horizontal_mode_copies_left_col() {
        // Mode 10 (pure horizontal), no boundary filter → every col equals left col.
        let above = [50u16; 33];
        let mut left = [0u16; 33];
        for i in 0..8 {
            left[i] = (10 * i) as u16;
        }
        let p = predict_angular(50, &above, &left, 8, 10, false, 255);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(p[r * 8 + c], left[r], "row {r} col {c}");
            }
        }
    }

    #[test]
    fn angular_45_diagonal_shifts() {
        // Mode 34 has angle +32 (exact 45°): predSamples[y][x] = ref[x+y+2] = above[x+y+1].
        let mut above = [0u16; 33];
        let left = [0u16; 33];
        for i in 0..16 {
            above[i] = (i + 1) as u16;
        }
        let p = predict_angular(0, &above, &left, 8, 34, false, 255);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(p[y * 8 + x], above[x + y + 1], "y{y} x{x}");
            }
        }
    }

    #[test]
    fn ref_filter_decision() {
        assert!(!should_filter_refs(DC, 8)); // DC never
        assert!(!should_filter_refs(0, 4)); // 4x4 never
        assert!(should_filter_refs(PLANAR, 8)); // planar at 8 filters
        assert!(!should_filter_refs(26, 8)); // pure vertical never (dist 0)
        assert!(!should_filter_refs(10, 16)); // pure horizontal never
        assert!(should_filter_refs(2, 8)); // far diagonal at 8 filters
        assert!(should_filter_refs(18, 16)); // 45° diagonal at 16 filters (dist 8 > 1)
    }

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
        let res = compute_residual_i32(&pixels, &pred, 8);
        assert!(res.iter().all(|&r| r == 0));
    }
}
