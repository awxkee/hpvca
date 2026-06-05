//! HEVC in-loop deblocking filter (spec §8.7.2), specialized for this encoder's
//! intra-only, fixed-structure streams.
//!
//! Because every coding unit is intra and every 8×8 luma block is both a CU and a
//! TU, every edge that lies on the 8×8 luma sample grid is a prediction/transform
//! boundary with boundary strength bS = 2. Chroma (4:2:0) is filtered on the
//! corresponding 16-luma-sample grid, also at bS = 2.
//!
//! The filter is applied to the reconstruction in place: ALL vertical edges across
//! the whole picture first, then ALL horizontal edges (reading the already
//! vertically-filtered samples), exactly as a conformant decoder does. With the
//! deblocking filter enabled in the PPS (default offsets), the encoder's
//! reconstruction therefore matches the decoder's output bit-for-bit.

const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

const TC_TABLE: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.max(0).min(255) as u8
}

/// Chroma QP mapping (HEVC Table 8-22) for 4:2:0.
fn table8_22(qpi: i32) -> i32 {
    const TAB: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
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
pub fn deblock(
    y: &mut [u8],
    w: usize,
    h: usize,
    cb: &mut [u8],
    cr: &mut [u8],
    cw: usize,
    ch: usize,
    qp: u8,
) {
    // Vertical edges first, then horizontal — each pass over the whole picture.
    for &vertical in &[true, false] {
        deblock_luma(y, w, h, qp, vertical);
        deblock_chroma(cb, cw, ch, qp, vertical);
        deblock_chroma(cr, cw, ch, qp, vertical);
    }
}

/// Luma deblocking for all edges of one orientation on the 8-sample grid.
fn deblock_luma(y: &mut [u8], w: usize, h: usize, qp: u8, vertical: bool) {
    let q_qp = qp as i32;
    // qP_L = (QP_Q + QP_P + 1) >> 1; both sides have the same QP here.
    let qp_l = q_qp;
    let beta = BETA_TABLE[clip3(0, 51, qp_l) as usize]; // bitDepth 8 → no shift
    // bS = 2 → Q_tc = qp_l + 2*(2-1) = qp_l + 2
    let tc = TC_TABLE[clip3(0, 53, qp_l + 2) as usize];

    // Edges lie on the 8-sample grid; the boundary is at coordinate `e` for
    // e = 8, 16, 24, ... (never the picture border at 0).
    if vertical {
        // vertical edges at column x = 8,16,... ; filter 4-row segments down each.
        let mut x = 8;
        while x < w {
            let mut yk = 0;
            while yk + 4 <= h {
                filter_luma_segment(y, w, x, yk, true, beta, tc);
                yk += 4;
            }
            x += 8;
        }
    } else {
        let mut yy = 8;
        while yy < h {
            let mut xk = 0;
            while xk + 4 <= w {
                filter_luma_segment(y, w, xk, yy, false, beta, tc);
                xk += 4;
            }
            yy += 8;
        }
    }
}

/// Filter one 4-sample luma edge segment. For a vertical edge, (ex,ey) is the
/// top sample of the boundary column (q side at column ex, p side at ex-1), and
/// the segment spans rows ey..ey+4. For a horizontal edge it is transposed.
fn filter_luma_segment(
    y: &mut [u8],
    stride: usize,
    ex: usize,
    ey: usize,
    vertical: bool,
    beta: i32,
    tc: i32,
) {
    // Gather p[k][i], q[k][i] for k=0..3 (lines), i=0..3 (distance from edge).
    let at = |yref: &[u8], r: usize, c: usize| yref[r * stride + c] as i32;
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
    let set = |y: &mut [u8], r: usize, c: usize, v: u8| {
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
                    set(y, ey + k, ex - 1 - i, clip_u8(pn[i]));
                    set(y, ey + k, ex + i, clip_u8(qn[i]));
                } else {
                    set(y, ey - 1 - i, ex + k, clip_u8(pn[i]));
                    set(y, ey + i, ex + k, clip_u8(qn[i]));
                }
            }
        } else {
            let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta.abs() < tc * 10 {
                let delta = clip3(-tc, tc, delta);
                if vertical {
                    set(y, ey + k, ex - 1, clip_u8(p0 + delta));
                    set(y, ey + k, ex, clip_u8(q0 - delta));
                } else {
                    set(y, ey - 1, ex + k, clip_u8(p0 + delta));
                    set(y, ey, ex + k, clip_u8(q0 - delta));
                }
                if dep == 1 {
                    let dpv = clip3(
                        -(tc >> 1),
                        tc >> 1,
                        (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
                    );
                    if vertical {
                        set(y, ey + k, ex - 2, clip_u8(p1 + dpv));
                    } else {
                        set(y, ey - 2, ex + k, clip_u8(p1 + dpv));
                    }
                }
                if deq == 1 {
                    let dqv = clip3(
                        -(tc >> 1),
                        tc >> 1,
                        (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
                    );
                    if vertical {
                        set(y, ey + k, ex + 1, clip_u8(q1 + dqv));
                    } else {
                        set(y, ey + 1, ex + k, clip_u8(q1 + dqv));
                    }
                }
            }
        }
    }
}

/// Chroma deblocking. Chroma edges sit on the 8-chroma-sample grid (= 16 luma),
/// i.e. every 8th chroma sample, processed in 4-sample segments. bS = 2 always.
fn deblock_chroma(c: &mut [u8], cw: usize, chh: usize, qp: u8, vertical: bool) {
    // qP_i = ((QP_Q + QP_P + 1)>>1) + cQpPicOffset(0); both QP equal → qp.
    let qpi = qp as i32;
    let qp_c = table8_22(qpi);
    // Q = QP_C + 2*(bS-1) = QP_C + 2 ; tc' from table; tc = tc' (bitDepth 8).
    let tc = TC_TABLE[clip3(0, 53, qp_c + 2) as usize];
    if tc == 0 { /* still proceed; delta clamps to 0 */ }

    if vertical {
        // chroma vertical edges at chroma column x = 8,16,... (every 16 luma)
        let mut x = 8;
        while x < cw {
            let mut yk = 0;
            while yk + 4 <= chh {
                filter_chroma_segment(c, cw, x, yk, true, tc);
                yk += 4;
            }
            x += 8;
        }
    } else {
        let mut yy = 8;
        while yy < chh {
            let mut xk = 0;
            while xk + 4 <= cw {
                filter_chroma_segment(c, cw, xk, yy, false, tc);
                xk += 4;
            }
            yy += 8;
        }
    }
}

fn filter_chroma_segment(
    c: &mut [u8],
    stride: usize,
    ex: usize,
    ey: usize,
    vertical: bool,
    tc: i32,
) {
    let at = |cref: &[u8], r: usize, col: usize| cref[r * stride + col] as i32;
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
            c[(ey + k) * stride + (ex - 1)] = clip_u8(p0 + delta);
            c[(ey + k) * stride + ex] = clip_u8(q0 - delta);
        } else {
            c[(ey - 1) * stride + (ex + k)] = clip_u8(p0 + delta);
            c[ey * stride + (ex + k)] = clip_u8(q0 - delta);
        }
    }
}
