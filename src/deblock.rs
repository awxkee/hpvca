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

/// Chroma QP mapping including the PPS Cb/Cr offset that §8.7.2.5.5 adds to
/// the edge-average luma QP: HEVC Table 8-22 for 4:2:0, `Min(qPi, 51)` for
/// 4:2:2 and 4:4:4.
fn chroma_qp(qpi: i32, chroma: crate::fmt::ChromaFormat, cqo: i8) -> i32 {
    let qpi = clip3(0, 57, qpi + i32::from(cqo));
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
    cqo: i8,
    edge_v: &[u8],
    edge_h: &[u8],
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
        cqo,
        edge_v,
        edge_h,
        None,
        0,
        kernels(),
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
    cqo: i8,
    edge_v: &[u8],
    edge_h: &[u8],
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
        cqo,
        edge_v,
        edge_h,
        Some(qp_map),
        qp_stride,
        kernels(),
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
    cqo: i8,
    edge_v: &[u8],
    edge_h: &[u8],
    qp_map: Option<&[u8]>,
    qp_stride: usize,
    k: &Kernels,
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
            k,
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
                cqo,
                edges,
                edge_stride,
                qp_map,
                qp_stride,
                k,
            );
            deblock_chroma(
                cr,
                cw,
                ch,
                qp,
                vertical,
                bit_depth.bits(),
                chroma,
                cqo,
                edges,
                edge_stride,
                qp_map,
                qp_stride,
                k,
            );
        }
    }
}

/// `pps_tc_offset_div2` written by every PPS. Intra-only pictures never feed
/// deblocked samples back into prediction, so the filter is a pure post-filter
/// here; a slightly weaker tC clip preserves texture and measured BD-rate at identical bytes.
/// −1 and −3 are both worse; the β offset gains nothing and stays 0.
pub(crate) const TC_OFFSET_DIV2: i32 = -2;

/// §8.7.2.5.3: the tC table is indexed at
/// Q = QpL + 2·(bS − 1) + 2·slice_tc_offset_div2, so a boundary-strength-1
/// edge (an IntraBC copy against another copy, with no coded residual on
/// either side) filters with a strictly smaller clipping bound than an intra
/// edge.
#[inline]
fn tc_offset(bs: u8) -> i32 {
    2 * (i32::from(bs) - 1) + 2 * TC_OFFSET_DIV2
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

/// Per-4-line luma deblock decision (§8.7.2.5.3), computed once from lines 0 and
/// 3 and applied uniformly to all 4 lines. `Weak` carries the dEp/dEq gates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LumaDecision {
    Skip,
    Strong,
    Weak { do_p1: bool, do_q1: bool },
}

/// Compute the segment decision from lines 0 and 3. `g(line, tap)` reads a sample
/// (tap: -1=p0,-2=p1,-3=p2,-4=p3, 0=q0,1=q1,2=q2,3=q3).
#[inline]
fn luma_decision(beta: i32, tc: i32, g: impl Fn(usize, i32) -> i32) -> LumaDecision {
    let dp0 = (g(0, -3) - 2 * g(0, -2) + g(0, -1)).abs();
    let dp3 = (g(3, -3) - 2 * g(3, -2) + g(3, -1)).abs();
    let dq0 = (g(0, 2) - 2 * g(0, 1) + g(0, 0)).abs();
    let dq3 = (g(3, 2) - 2 * g(3, 1) + g(3, 0)).abs();
    if dp0 + dp3 + dq0 + dq3 >= beta {
        return LumaDecision::Skip;
    }
    let dsam = |line: usize, dpq: i32| -> bool {
        2 * dpq < (beta >> 2)
            && (g(line, -4) - g(line, -1)).abs() + (g(line, 0) - g(line, 3)).abs() < (beta >> 3)
            && (g(line, -1) - g(line, 0)).abs() < (5 * tc + 1) >> 1
    };
    if dsam(0, dp0 + dq0) && dsam(3, dp3 + dq3) {
        LumaDecision::Strong
    } else {
        let side_thr = (beta + (beta >> 1)) >> 3;
        LumaDecision::Weak {
            do_p1: dp0 + dp3 < side_thr,
            do_q1: dq0 + dq3 < side_thr,
        }
    }
}

type LumaPlaneFn = fn(&mut [u16], usize, usize, usize, usize, LumaDecision, i32, i32);
type LumaPairFn =
    fn(&mut [u16], usize, usize, usize, usize, LumaDecision, i32, LumaDecision, i32, i32);
type ChromaPlaneFn = fn(&mut [u16], usize, usize, usize, usize, i32, i32);
type ChromaPairFn = fn(&mut [u16], usize, usize, usize, usize, i32, i32, i32);

/// Filter kernels. Decisions are made scalar; the chosen filter is applied by
/// these (SIMD where available). Pair kernels filter two adjacent 4-sample
/// segments of the same edge at once.
#[derive(Clone, Copy)]
struct Kernels {
    luma_v: LumaPlaneFn,
    luma_h: LumaPlaneFn,
    luma_v_pair: Option<LumaPairFn>,
    luma_h_pair: Option<LumaPairFn>,
    chroma_v: ChromaPlaneFn,
    chroma_h: ChromaPlaneFn,
    chroma_v_pair: Option<ChromaPairFn>,
    chroma_h_pair: Option<ChromaPairFn>,
}

const SCALAR_KERNELS: Kernels = Kernels {
    luma_v: luma_vertical_plane_scalar,
    luma_h: luma_horizontal_plane_scalar,
    luma_v_pair: None,
    luma_h_pair: None,
    chroma_v: chroma_vertical_plane_scalar,
    chroma_h: chroma_horizontal_plane_scalar,
    chroma_v_pair: None,
    chroma_h_pair: None,
};

fn kernels() -> &'static Kernels {
    static K: std::sync::OnceLock<Kernels> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        #[allow(unused_mut)]
        let mut k = SCALAR_KERNELS;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            k.luma_v = crate::neon::luma_vertical_plane_neon;
            k.luma_h = crate::neon::luma_horizontal_plane_neon;
            k.chroma_v = crate::neon::chroma_vertical_plane_neon;
            k.chroma_h = crate::neon::chroma_horizontal_plane_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                k.luma_v = crate::avx::luma_vertical_plane_avx2;
                k.luma_h = crate::avx::luma_horizontal_plane_avx2;
                k.luma_v_pair = Some(crate::avx::luma_vertical_plane_pair_avx2);
                k.luma_h_pair = Some(crate::avx::luma_horizontal_plane_pair_avx2);
                k.chroma_v = crate::avx::chroma_vertical_plane_avx2;
                k.chroma_h = crate::avx::chroma_horizontal_plane_avx2;
                k.chroma_v_pair = Some(crate::avx::chroma_vertical_plane_pair_avx2);
                k.chroma_h_pair = Some(crate::avx::chroma_horizontal_plane_pair_avx2);
            }
        }
        k
    })
}

/// Run `single` over 4-sample segments `pos, pos+4, ...` while `pos + 4 <= end`,
/// pairing two adjacent filtered segments into one `pair` call when available.
/// `params(plane, pos)` returns `None` for segments that are not filtered.
/// Adjacent segments of one edge touch disjoint samples, so evaluating both
/// decisions before filtering is equivalent to the sequential order.
#[inline]
fn walk_segments<T: Copy>(
    plane: &mut [u16],
    end: usize,
    has_pair: bool,
    params: impl Fn(&[u16], usize) -> Option<T>,
    single: impl Fn(&mut [u16], usize, T),
    pair: impl Fn(&mut [u16], usize, T, T),
) {
    let mut pos = 0;
    while pos + 4 <= end {
        let Some(a) = params(plane, pos) else {
            pos += 4;
            continue;
        };
        if has_pair && pos + 8 <= end {
            match params(plane, pos + 4) {
                Some(b) => pair(plane, pos, a, b),
                // `pos + 4` was evaluated and is unfiltered.
                None => single(plane, pos, a),
            }
            pos += 8;
            continue;
        }
        single(plane, pos, a);
        pos += 4;
    }
}

/// Luma deblocking for all edges of one orientation on the 8-sample grid.
#[allow(clippy::too_many_arguments)]
fn deblock_luma(
    y: &mut [u16],
    w: usize,
    h: usize,
    qp: u8,
    vertical: bool,
    bit_depth: u8,
    edges: &[u8],
    edge_stride: usize,
    qp_map: Option<&[u8]>,
    qp_stride: usize,
    k: &Kernels,
) {
    let bdshift = (bit_depth - 8) as i32;
    let max_val = (1i32 << bit_depth) - 1;

    // Edges lie on the 8-sample grid; the boundary is at coordinate `e` for
    // e = 8, 16, 24, ... (never the picture border at 0). Segments run along
    // the edge in steps of 4 samples.
    let (edge_end, seg_end) = if vertical { (w, h) } else { (h, w) };
    let mut e = 8;
    while e < edge_end {
        let params = |y: &[u16], pos: usize| -> Option<(LumaDecision, i32)> {
            let (lx, ly) = if vertical { (e, pos) } else { (pos, e) };
            let bs = edges[(ly / 4) * edge_stride + lx / 4];
            if bs == 0 {
                return None;
            }
            let qp_l = edge_qp(qp, qp_map, qp_stride, lx, ly, vertical);
            let beta = BETA_TABLE[clip3(0, 51, qp_l) as usize] << bdshift;
            let tc = TC_TABLE[clip3(0, 53, qp_l + tc_offset(bs)) as usize] << bdshift;
            if tc == 0 {
                // Both the strong (±2tc) and weak (|Δ| < 10tc) filters are
                // identities at tc = 0.
                return None;
            }
            let dec = if vertical {
                luma_decision(beta, tc, |line, tap| {
                    y[(pos + line) * w + (e as isize + tap as isize) as usize] as i32
                })
            } else {
                luma_decision(beta, tc, |line, tap| {
                    y[(e as isize + tap as isize) as usize * w + pos + line] as i32
                })
            };
            (dec != LumaDecision::Skip).then_some((dec, tc))
        };
        let (single, pair) = if vertical {
            (k.luma_v, k.luma_v_pair)
        } else {
            (k.luma_h, k.luma_h_pair)
        };
        walk_segments(
            y,
            seg_end,
            pair.is_some(),
            params,
            |y, pos, (d, tc)| single(y, w, e, pos, 0, d, tc, max_val),
            |y, pos, (d0, tc0), (d1, tc1)| {
                pair.unwrap()(y, w, e, pos, 0, d0, tc0, d1, tc1, max_val)
            },
        );
        e += 8;
    }
}

/// Apply a precomputed per-segment `decision` uniformly to the 4 lines of a luma
/// edge segment. `at(line, tap)` gives the flat index (see `luma_decision`).
#[inline]
fn apply_luma_decision(
    plane: &mut [u16],
    decision: LumaDecision,
    tc: i32,
    max_val: i32,
    at: impl Fn(usize, i32) -> usize,
) {
    let (strong, do_p1, do_q1) = match decision {
        LumaDecision::Skip => return,
        LumaDecision::Strong => (true, false, false),
        LumaDecision::Weak { do_p1, do_q1 } => (false, do_p1, do_q1),
    };
    let tc2 = tc >> 1;
    let g = |plane: &[u16], line: usize, tap: i32| plane[at(line, tap)] as i32;
    for line in 0..4 {
        let p0 = g(plane, line, -1);
        let p1 = g(plane, line, -2);
        let p2 = g(plane, line, -3);
        let p3 = g(plane, line, -4);
        let q0 = g(plane, line, 0);
        let q1 = g(plane, line, 1);
        let q2 = g(plane, line, 2);
        let q3 = g(plane, line, 3);
        if strong {
            // §8.7.2.5.7: each strongly-filtered sample is clipped to ±2·tc
            // around its original value.
            let c = |orig: i32, v: i32| -> u16 {
                clip_sample(clip3(orig - 2 * tc, orig + 2 * tc, v), max_val)
            };
            plane[at(line, -1)] = c(p0, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            plane[at(line, -2)] = c(p1, (p2 + p1 + p0 + q0 + 2) >> 2);
            plane[at(line, -3)] = c(p2, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
            plane[at(line, 0)] = c(q0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            plane[at(line, 1)] = c(q1, (p0 + q0 + q1 + q2 + 2) >> 2);
            plane[at(line, 2)] = c(q2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
        } else {
            let delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta0.abs() < tc * 10 {
                let delta = clip3(-tc, tc, delta0);
                plane[at(line, -1)] = clip_sample(p0 + delta, max_val);
                plane[at(line, 0)] = clip_sample(q0 - delta, max_val);
                if do_p1 {
                    let dp1 = clip3(-tc2, tc2, (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1);
                    plane[at(line, -2)] = clip_sample(p1 + dp1, max_val);
                }
                if do_q1 {
                    let dq1 = clip3(-tc2, tc2, (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1);
                    plane[at(line, 1)] = clip_sample(q1 + dq1, max_val);
                }
            }
        }
    }
}

/// Scalar reference: apply `decision` to a vertical-edge segment (4 rows
/// starting at global row `s`; `pix` local row 0 is global row `row0`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_scalar(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    apply_luma_decision(pix, decision, tc, maxv, |line, tap| {
        let col = (edge as i32 + tap) as usize;
        (s + line - row0) * w + col
    });
}

/// Scalar reference: apply `decision` to a horizontal-edge segment (4 cols).
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_scalar(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    apply_luma_decision(pix, decision, tc, maxv, |line, tap| {
        let row = (edge as i32 + tap) as usize - row0;
        row * w + (scan + line)
    });
}

/// Chroma deblocking. Chroma edges sit on the 8-chroma-sample grid (16 luma
/// samples along a subsampled axis, 8 otherwise), processed in 4-sample
/// segments. Only bS = 2 edges are filtered.
#[allow(clippy::too_many_arguments)]
fn deblock_chroma(
    c: &mut [u16],
    cw: usize,
    chh: usize,
    qp: u8,
    vertical: bool,
    bit_depth: u8,
    chroma: crate::fmt::ChromaFormat,
    cqo: i8,
    edges: &[u8],
    edge_stride: usize,
    qp_map: Option<&[u8]>,
    qp_stride: usize,
    k: &Kernels,
) {
    let bdshift = (bit_depth - 8) as i32;
    let max_val = (1i32 << bit_depth) - 1;

    let (edge_end, seg_end) = if vertical { (cw, chh) } else { (chh, cw) };
    let (single, pair) = if vertical {
        (k.chroma_v, k.chroma_v_pair)
    } else {
        (k.chroma_h, k.chroma_h_pair)
    };
    let mut e = 8;
    while e < edge_end {
        let params = |pos: usize| -> Option<i32> {
            let (lx, ly) = if vertical {
                (e * chroma.sub_w(), (pos + 1) * chroma.sub_h())
            } else {
                ((pos + 1) * chroma.sub_w(), e * chroma.sub_h())
            };
            if edges[(ly / 4) * edge_stride + lx / 4] != 2 {
                return None;
            }
            let qpi = edge_qp(qp, qp_map, qp_stride, lx, ly, vertical);
            let qp_c = chroma_qp(qpi, chroma, cqo);
            let tc = TC_TABLE[clip3(0, 53, qp_c + tc_offset(2)) as usize] << bdshift;
            // Δ is clipped to ±tc, so tc = 0 is an identity.
            (tc != 0).then_some(tc)
        };
        walk_segments(
            c,
            seg_end,
            pair.is_some(),
            |_, pos| params(pos),
            |c, pos, tc| single(c, cw, e, pos, 0, tc, max_val),
            |c, pos, tc0, tc1| pair.unwrap()(c, cw, e, pos, 0, tc0, tc1, max_val),
        );
        e += 8;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_vertical_plane_scalar(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    s: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    for r in s..s + 4 {
        let lr = r - crow0;
        let p0 = pix[lr * cw + edge - 1] as i32;
        let p1 = pix[lr * cw + edge - 2] as i32;
        let q0 = pix[lr * cw + edge] as i32;
        let q1 = pix[lr * cw + edge + 1] as i32;
        let delta = clip3(-tc_c, tc_c, ((q0 - p0) * 4 + p1 - q1 + 4) >> 3);
        pix[lr * cw + edge - 1] = clip_sample(p0 + delta, maxv_c);
        pix[lr * cw + edge] = clip_sample(q0 - delta, maxv_c);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_horizontal_plane_scalar(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    scan: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    for c in scan..scan + 4 {
        let base = |gy: usize| (gy - crow0) * cw + c;
        let p0 = pix[base(edge - 1)] as i32;
        let p1 = pix[base(edge - 2)] as i32;
        let q0 = pix[base(edge)] as i32;
        let q1 = pix[base(edge + 1)] as i32;
        let delta = clip3(-tc_c, tc_c, ((q0 - p0) * 4 + p1 - q1 + 4) >> 3);
        pix[base(edge - 1)] = clip_sample(p0 + delta, maxv_c);
        pix[base(edge)] = clip_sample(q0 - delta, maxv_c);
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
        let mut edge_v = vec![0u8; (w / 4) * (h / 4)];
        let edge_h = edge_v.clone();
        edge_v[2] = 2; // x=8, rows 0..4 only
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
            0,
            &edge_v,
            &edge_h,
        );
        assert_ne!(&y[..4 * w], &before[..4 * w]);
        assert_eq!(&y[4 * w..], &before[4 * w..]);
    }

    /// SIMD filter kernels (NEON / AVX2 incl. pair kernels) must be
    /// bit-exact with the scalar reference across bit depths, chroma formats,
    /// boundary strengths and per-QG QPs. On builds without SIMD the dispatched
    /// kernels are scalar and this is trivially true.
    #[test]
    fn simd_kernels_match_scalar() {
        use crate::fmt::{BitDepth, ChromaFormat};
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (w, h) = (64usize, 48usize);
        let mut changed = 0usize;
        for bits in [8u8, 10, 12] {
            let max = (1u32 << bits) - 1;
            for chroma in [
                ChromaFormat::Monochrome,
                ChromaFormat::Yuv420,
                ChromaFormat::Yuv422,
                ChromaFormat::Yuv444,
            ] {
                for trial in 0..24 {
                    // Blocky content: per-8x8 DC plus small noise, so strong,
                    // weak and skip decisions all occur.
                    let mut plane = |pw: usize, ph: usize| -> Vec<u16> {
                        let bw = pw.div_ceil(8);
                        let dcs: Vec<u32> = (0..bw * ph.div_ceil(8))
                            .map(|_| rnd() as u32 % (max + 1))
                            .collect();
                        let noise = 1 + (trial as u32 % 4) * (max >> 6);
                        (0..pw * ph)
                            .map(|i| {
                                let (x, y) = (i % pw, i / pw);
                                let v = dcs[(y / 8) * bw + x / 8] as i64
                                    + (rnd() as u32 % (2 * noise + 1)) as i64
                                    - noise as i64;
                                v.clamp(0, max as i64) as u16
                            })
                            .collect()
                    };
                    let (cw, ch) = if chroma.is_monochrome() {
                        (0, 0)
                    } else {
                        (w / chroma.sub_w(), h / chroma.sub_h())
                    };
                    let y0 = plane(w, h);
                    let cb0 = plane(cw, ch);
                    let cr0 = plane(cw, ch);
                    let n4 = (w / 4) * (h / 4);
                    let edge_v: Vec<u8> = (0..n4).map(|_| (rnd() % 3) as u8).collect();
                    let edge_h: Vec<u8> = (0..n4).map(|_| (rnd() % 3) as u8).collect();
                    let qp_map: Vec<u8> = (0..n4).map(|_| 18 + (rnd() % 34) as u8).collect();
                    let use_map = trial % 2 == 1;
                    let qp = 22 + (trial as u8 % 6) * 4;
                    let run = |k: &Kernels| {
                        let (mut y, mut cb, mut cr) = (y0.clone(), cb0.clone(), cr0.clone());
                        deblock_impl(
                            &mut y,
                            w,
                            h,
                            &mut cb,
                            &mut cr,
                            cw,
                            ch,
                            qp,
                            BitDepth::from_bits(bits),
                            chroma,
                            -2,
                            &edge_v,
                            &edge_h,
                            use_map.then_some(&qp_map[..]),
                            w / 4,
                            k,
                        );
                        (y, cb, cr)
                    };
                    let reference = run(&SCALAR_KERNELS);
                    let simd = run(kernels());
                    changed += usize::from(reference.0 != y0)
                        + usize::from(reference.1 != cb0 || reference.2 != cr0);
                    assert_eq!(reference, simd, "bits={bits} {chroma:?} trial={trial}");
                }
            }
        }
        assert!(changed > 100, "test content barely filtered: {changed}");
    }

    #[test]
    fn chroma_qp_mapping_matches_format_rules() {
        // A −3 Cb/Cr offset shifts qPi before the mapping:
        // 35 − 3 = 32 → Table 8-22 gives 31 for 4:2:0, identity elsewhere.
        assert_eq!(chroma_qp(35, crate::fmt::ChromaFormat::Yuv420, -3), 31);
        assert_eq!(chroma_qp(35, crate::fmt::ChromaFormat::Yuv422, -3), 32);
        assert_eq!(chroma_qp(35, crate::fmt::ChromaFormat::Yuv444, -3), 32);
    }
}
