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

//! HEVC in-loop deblocking filter (spec §8.7.2), specialized for this encoder's
//! intra-only, fixed-structure streams.

static BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

static TC_TABLE: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

#[inline]
fn clip_sample(v: i32, max_val: i32) -> u16 {
    v.max(0).min(max_val) as u16
}

/// Chroma QP mapping (HEVC Table 8-22) for 4:2:0.
fn chroma_qp(qpi: i32, chroma: crate::fmt::ChromaFormat) -> i32 {
    if !matches!(chroma, crate::fmt::ChromaFormat::Yuv420) {
        return qpi.min(51);
    }
    static TAB: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        TAB[(qpi - 30) as usize]
    }
}

/// Apply the full deblocking filter to a reconstructed picture.
///
/// `y` is the luma plane (`w`×`h`), `cb`/`cr` the chroma planes
/// (`w/2`×`h/2`, 4:2:0). `qp` is the (constant) luma QP. All dimensions are the
/// coded dimensions (multiples of the CTU size).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn deblock(
    y: &mut [u16],
    w: usize,
    h: usize,
    cb: &mut [u16],
    cr: &mut [u16],
    cw: usize,
    ch: usize,
    qp: u8,
    bit_depth: crate::fmt::BitDepth,
    chroma: crate::fmt::ChromaFormat,
    edge_v: &[bool],
    edge_h: &[bool],
) {
    deblock_impl(
        y, w, h, cb, cr, cw, ch, qp, bit_depth, chroma, edge_v, edge_h, None, 0,
    );
}

/// Deblock using the reconstructed per-4x4 luma QP map produced by adaptive
/// quantization. HEVC derives beta/tc from the average QP on the two sides of
/// each edge, so a single slice QP is insufficient when CU QP deltas are used.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_with_qp_map(
    y: &mut [u16],
    w: usize,
    h: usize,
    cb: &mut [u16],
    cr: &mut [u16],
    cw: usize,
    ch: usize,
    qp: u8,
    bit_depth: crate::fmt::BitDepth,
    chroma: crate::fmt::ChromaFormat,
    edge_v: &[bool],
    edge_h: &[bool],
    qp_map: &[u8],
    qp_stride: usize,
) {
    deblock_impl(
        y,
        w,
        h,
        cb,
        cr,
        cw,
        ch,
        qp,
        bit_depth,
        chroma,
        edge_v,
        edge_h,
        Some(qp_map),
        qp_stride,
    );
}

#[allow(clippy::too_many_arguments)]
fn deblock_impl(
    y: &mut [u16],
    w: usize,
    h: usize,
    cb: &mut [u16],
    cr: &mut [u16],
    cw: usize,
    ch: usize,
    qp: u8,
    bit_depth: crate::fmt::BitDepth,
    chroma: crate::fmt::ChromaFormat,
    edge_v: &[bool],
    edge_h: &[bool],
    qp_map: Option<&[u8]>,
    qp_stride: usize,
) {
    let edge_stride = w / 4;
    for &vertical in &[true, false] {
        let edges = if vertical { edge_v } else { edge_h };
        deblock_luma(
            y,
            w,
            h,
            qp,
            vertical,
            bit_depth.bits(),
            edges,
            edge_stride,
            qp_map,
            qp_stride,
        );
        if !chroma.is_monochrome() {
            deblock_chroma(
                cb,
                cw,
                ch,
                qp,
                vertical,
                bit_depth.bits(),
                chroma,
                edges,
                edge_stride,
                qp_map,
                qp_stride,
            );
            deblock_chroma(
                cr,
                cw,
                ch,
                qp,
                vertical,
                bit_depth.bits(),
                chroma,
                edges,
                edge_stride,
                qp_map,
                qp_stride,
            );
        }
    }
}

#[inline]
fn edge_qp(
    fallback: u8,
    qp_map: Option<&[u8]>,
    stride: usize,
    luma_x: usize,
    luma_y: usize,
    vertical: bool,
) -> i32 {
    let Some(map) = qp_map else {
        return i32::from(fallback);
    };
    let qx = luma_x / 4;
    let qy = luma_y / 4;
    let (px, py) = if vertical { (qx - 1, qy) } else { (qx, qy - 1) };
    (i32::from(map[py * stride + px]) + i32::from(map[qy * stride + qx]) + 1) >> 1
}

/// Luma-only deblocking, for monochrome (4:0:0) pictures that have no chroma.
/// Luma deblocking for all edges of one orientation on the 8-sample grid.
#[allow(clippy::too_many_arguments)]
fn deblock_luma(
    y: &mut [u16],
    w: usize,
    h: usize,
    qp: u8,
    vertical: bool,
    bit_depth: u8,
    edges: &[bool],
    edge_stride: usize,
    qp_map: Option<&[u8]>,
    qp_stride: usize,
) {
    let bdshift = (bit_depth - 8) as i32;
    let max_val = (1i32 << bit_depth) - 1;

    // Edges lie on the 8-sample grid; the boundary is at coordinate `e` for
    // e = 8, 16, 24, ... (never the picture border at 0).
    if vertical {
        // vertical edges at column x = 8,16,... ; filter 4-row segments down each.
        let mut x = 8;
        while x < w {
            let mut yk = 0;
            while yk + 4 <= h {
                if edges[(yk / 4) * edge_stride + x / 4] {
                    let qp_l = edge_qp(qp, qp_map, qp_stride, x, yk, true);
                    let beta = BETA_TABLE[clip3(0, 51, qp_l) as usize] << bdshift;
                    let tc = TC_TABLE[clip3(0, 53, qp_l + 2) as usize] << bdshift;
                    filter_luma_segment(y, w, x, yk, true, beta, tc, max_val);
                }
                yk += 4;
            }
            x += 8;
        }
    } else {
        let mut yy = 8;
        while yy < h {
            let mut xk = 0;
            while xk + 4 <= w {
                if edges[(yy / 4) * edge_stride + xk / 4] {
                    let qp_l = edge_qp(qp, qp_map, qp_stride, xk, yy, false);
                    let beta = BETA_TABLE[clip3(0, 51, qp_l) as usize] << bdshift;
                    let tc = TC_TABLE[clip3(0, 53, qp_l + 2) as usize] << bdshift;
                    filter_luma_segment(y, w, xk, yy, false, beta, tc, max_val);
                }
                xk += 4;
            }
            yy += 8;
        }
    }
}

/// Filter one 4-sample luma edge segment. For a vertical edge, (ex,ey) is the
/// top sample of the boundary column (q side at column ex, p side at ex-1), and
/// the segment spans rows ey..ey+4. For a horizontal edge it is transposed.
#[allow(clippy::too_many_arguments)]
fn filter_luma_segment(
    y: &mut [u16],
    stride: usize,
    ex: usize,
    ey: usize,
    vertical: bool,
    beta: i32,
    tc: i32,
    max_val: i32,
) {
    let at = |yref: &[u16], r: usize, c: usize| yref[r * stride + c] as i32;
    let mut p = [[0i32; 4]; 4];
    let mut q = [[0i32; 4]; 4];
    for k in 0..4 {
        for i in 0..4 {
            if vertical {
                let r = ey + k;
                q[k][i] = at(y, r, ex + i);
                p[k][i] = at(y, r, ex - 1 - i);
            } else {
                let c = ex + k;
                q[k][i] = at(y, ey + i, c);
                p[k][i] = at(y, ey.wrapping_sub(i + 1), c);
            }
        }
    }

    // Decision (8.7.2.4.3)
    let dp0 = (p[0][2] - 2 * p[0][1] + p[0][0]).abs();
    let dp3 = (p[3][2] - 2 * p[3][1] + p[3][0]).abs();
    let dq0 = (q[0][2] - 2 * q[0][1] + q[0][0]).abs();
    let dq3 = (q[3][2] - 2 * q[3][1] + q[3][0]).abs();
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    let d = dpq0 + dpq3;

    if d >= beta {
        return;
    }

    let d_sam0 = 2 * dpq0 < (beta >> 2)
        && (p[0][3] - p[0][0]).abs() + (q[0][0] - q[0][3]).abs() < (beta >> 3)
        && (p[0][0] - q[0][0]).abs() < ((5 * tc + 1) >> 1);
    let d_sam3 = 2 * dpq3 < (beta >> 2)
        && (p[3][3] - p[3][0]).abs() + (q[3][0] - q[3][3]).abs() < (beta >> 3)
        && (p[3][0] - q[3][0]).abs() < ((5 * tc + 1) >> 1);

    let de = if d_sam0 && d_sam3 { 2 } else { 1 };
    let dep = if dp < ((beta + (beta >> 1)) >> 3) {
        1
    } else {
        0
    };
    let deq = if dq < ((beta + (beta >> 1)) >> 3) {
        1
    } else {
        0
    };

    // Apply (8.7.2.4.4 / kernel). filterP=filterQ=true (no PCM/transquant-bypass).
    let set = |y: &mut [u16], r: usize, c: usize, v: u16| {
        y[r * stride + c] = v;
    };
    for k in 0..4 {
        let (p0, p1, p2, p3) = (p[k][0], p[k][1], p[k][2], p[k][3]);
        let (q0, q1, q2, q3) = (q[k][0], q[k][1], q[k][2], q[k][3]);
        if de == 2 {
            let pn0 = clip3(
                p0 - 2 * tc,
                p0 + 2 * tc,
                (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3,
            );
            let pn1 = clip3(p1 - 2 * tc, p1 + 2 * tc, (p2 + p1 + p0 + q0 + 2) >> 2);
            let pn2 = clip3(
                p2 - 2 * tc,
                p2 + 2 * tc,
                (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3,
            );
            let qn0 = clip3(
                q0 - 2 * tc,
                q0 + 2 * tc,
                (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3,
            );
            let qn1 = clip3(q1 - 2 * tc, q1 + 2 * tc, (p0 + q0 + q1 + q2 + 2) >> 2);
            let qn2 = clip3(
                q2 - 2 * tc,
                q2 + 2 * tc,
                (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3,
            );
            let pn = [pn0, pn1, pn2];
            let qn = [qn0, qn1, qn2];
            for i in 0..3 {
                if vertical {
                    set(y, ey + k, ex - 1 - i, clip_sample(pn[i], max_val));
                    set(y, ey + k, ex + i, clip_sample(qn[i], max_val));
                } else {
                    set(y, ey - 1 - i, ex + k, clip_sample(pn[i], max_val));
                    set(y, ey + i, ex + k, clip_sample(qn[i], max_val));
                }
            }
        } else {
            let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta.abs() < tc * 10 {
                let delta = clip3(-tc, tc, delta);
                if vertical {
                    set(y, ey + k, ex - 1, clip_sample(p0 + delta, max_val));
                    set(y, ey + k, ex, clip_sample(q0 - delta, max_val));
                } else {
                    set(y, ey - 1, ex + k, clip_sample(p0 + delta, max_val));
                    set(y, ey, ex + k, clip_sample(q0 - delta, max_val));
                }
                if dep == 1 {
                    let dpv = clip3(
                        -(tc >> 1),
                        tc >> 1,
                        (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
                    );
                    if vertical {
                        set(y, ey + k, ex - 2, clip_sample(p1 + dpv, max_val));
                    } else {
                        set(y, ey - 2, ex + k, clip_sample(p1 + dpv, max_val));
                    }
                }
                if deq == 1 {
                    let dqv = clip3(
                        -(tc >> 1),
                        tc >> 1,
                        (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
                    );
                    if vertical {
                        set(y, ey + k, ex + 1, clip_sample(q1 + dqv, max_val));
                    } else {
                        set(y, ey + 1, ex + k, clip_sample(q1 + dqv, max_val));
                    }
                }
            }
        }
    }
}

/// Chroma deblocking. Chroma edges sit on the 8-chroma-sample grid (= 16 luma),
/// i.e. every 8th chroma sample, processed in 4-sample segments. bS = 2 always.
#[allow(clippy::too_many_arguments)]
fn deblock_chroma(
    c: &mut [u16],
    cw: usize,
    chh: usize,
    qp: u8,
    vertical: bool,
    bit_depth: u8,
    chroma: crate::fmt::ChromaFormat,
    edges: &[bool],
    edge_stride: usize,
    qp_map: Option<&[u8]>,
    qp_stride: usize,
) {
    let bdshift = (bit_depth - 8) as i32;
    let max_val = (1i32 << bit_depth) - 1;

    if vertical {
        // chroma vertical edges at chroma column x = 8,16,... (every 16 luma)
        let mut x = 8;
        while x < cw {
            let mut yk = 0;
            while yk + 4 <= chh {
                let lx = x * chroma.sub_w();
                let ly = (yk + 1) * chroma.sub_h();
                if edges[(ly / 4) * edge_stride + lx / 4] {
                    let qpi = edge_qp(qp, qp_map, qp_stride, lx, ly, true);
                    let qp_c = chroma_qp(qpi, chroma);
                    let tc = TC_TABLE[clip3(0, 53, qp_c + 2) as usize] << bdshift;
                    filter_chroma_segment(c, cw, x, yk, true, tc, max_val);
                }
                yk += 4;
            }
            x += 8;
        }
    } else {
        let mut yy = 8;
        while yy < chh {
            let mut xk = 0;
            while xk + 4 <= cw {
                let lx = (xk + 1) * chroma.sub_w();
                let ly = yy * chroma.sub_h();
                if edges[(ly / 4) * edge_stride + lx / 4] {
                    let qpi = edge_qp(qp, qp_map, qp_stride, lx, ly, false);
                    let qp_c = chroma_qp(qpi, chroma);
                    let tc = TC_TABLE[clip3(0, 53, qp_c + 2) as usize] << bdshift;
                    filter_chroma_segment(c, cw, xk, yy, false, tc, max_val);
                }
                xk += 4;
            }
            yy += 8;
        }
    }
}

fn filter_chroma_segment(
    c: &mut [u16],
    stride: usize,
    ex: usize,
    ey: usize,
    vertical: bool,
    tc: i32,
    max_val: i32,
) {
    let at = |cref: &[u16], r: usize, col: usize| cref[r * stride + col] as i32;
    for k in 0..4 {
        let (p0, p1, q0, q1);
        if vertical {
            let r = ey + k;
            q0 = at(c, r, ex);
            q1 = at(c, r, ex + 1);
            p0 = at(c, r, ex - 1);
            p1 = at(c, r, ex - 2);
        } else {
            let col = ex + k;
            q0 = at(c, ey, col);
            q1 = at(c, ey + 1, col);
            p0 = at(c, ey - 1, col);
            p1 = at(c, ey - 2, col);
        }
        let delta = clip3(-tc, tc, (((q0 - p0) * 4) + p1 - q1 + 4) >> 3);
        if vertical {
            c[(ey + k) * stride + (ex - 1)] = clip_sample(p0 + delta, max_val);
            c[(ey + k) * stride + ex] = clip_sample(q0 - delta, max_val);
        } else {
            c[(ey - 1) * stride + (ex + k)] = clip_sample(p0 + delta, max_val);
            c[ey * stride + (ex + k)] = clip_sample(q0 - delta, max_val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_filters_only_marked_segments() {
        let w = 16;
        let h = 8;
        let mut y = vec![0u16; w * h];
        for row in y.chunks_exact_mut(w) {
            row[..8].fill(96);
            row[8..].fill(104);
        }
        let before = y.clone();
        let mut edge_v = vec![false; (w / 4) * (h / 4)];
        let edge_h = edge_v.clone();
        edge_v[2] = true; // x=8, rows 0..4 only
        deblock(
            &mut y,
            w,
            h,
            &mut [],
            &mut [],
            0,
            0,
            40,
            crate::fmt::BitDepth::Eight,
            crate::fmt::ChromaFormat::Monochrome,
            &edge_v,
            &edge_h,
        );
        assert_ne!(&y[..4 * w], &before[..4 * w]);
        assert_eq!(&y[4 * w..], &before[4 * w..]);
    }

    #[test]
    fn chroma_qp_mapping_matches_format_rules() {
        assert_eq!(chroma_qp(35, crate::fmt::ChromaFormat::Yuv420), 33);
        assert_eq!(chroma_qp(35, crate::fmt::ChromaFormat::Yuv422), 35);
        assert_eq!(chroma_qp(35, crate::fmt::ChromaFormat::Yuv444), 35);
    }
}
