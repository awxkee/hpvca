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

use crate::cabac::{CabacWriter, ContextSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SaoParam {
    pub type_idx: u8, // 0=off, 1=BO, 2=EO
    pub offsets: [i8; 4],
    pub band_pos: u8,
    pub eo_class: u8,
}

/// One CTU's SAO parameters for all three components. `cb` and `cr` share
/// `type_idx` and, for edge offset, `eo_class` (§7.3.8.3: Cr inherits both
/// from Cb); their offsets and band positions are independent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SaoCtu {
    pub y: SaoParam,
    pub cb: SaoParam,
    pub cr: SaoParam,
}

/// One component plane as the SAO search sees it: statistics come from the
/// visible `width × height` samples, while the coded picture (`stride ×
/// coded_height`) bounds edge-offset neighbours exactly as the decoder
/// classifies them. `ctb_w × ctb_h` is the CTB size in this plane's samples.
#[derive(Clone, Copy)]
pub(crate) struct SaoPlane<'a> {
    pub original: &'a [u16],
    pub rec: &'a [u16],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    pub coded_height: usize,
    pub ctb_w: usize,
    pub ctb_h: usize,
    pub bit_depth: u8,
}

#[inline]
fn rounded_offset(sum: i64, count: u32, lo: i32, hi: i32) -> i32 {
    if count == 0 {
        return 0;
    }
    let c = i64::from(count);
    let value = if sum >= 0 {
        (sum + c / 2) / c
    } else {
        -((-sum + c / 2) / c)
    };
    (value as i32).clamp(lo, hi)
}

#[inline]
fn distortion_delta(count: u32, sum: i64, offset: i32) -> i64 {
    i64::from(count) * i64::from(offset * offset) - 2 * sum * i64::from(offset)
}

#[inline]
fn unary_bits(magnitude: i32, max: i32) -> u32 {
    magnitude as u32 + u32::from(magnitude < max)
}

fn select_offset(
    sum: i64,
    count: u32,
    lo: i32,
    hi: i32,
    max: i32,
    lambda: f32,
    explicit_sign: bool,
) -> (i32, i64, u32) {
    let initial = rounded_offset(sum, count, lo, hi);
    let mut best = (0, 0, unary_bits(0, max));
    let mut best_cost = f64::from(lambda) * f64::from(best.2);
    for magnitude in 1..=initial.abs() {
        let offset = magnitude * initial.signum();
        let bits = unary_bits(magnitude, max) + u32::from(explicit_sign);
        let delta = distortion_delta(count, sum, offset);
        let cost = delta as f64 + f64::from(lambda) * f64::from(bits);
        if cost < best_cost {
            best = (offset, delta, bits);
            best_cost = cost;
        }
    }
    best
}

#[inline]
fn eo_direction(eo_class: u8) -> (i32, i32) {
    match eo_class {
        0 => (1, 0),
        1 => (0, 1),
        2 => (1, 1),
        _ => (1, -1),
    }
}

/// One sample of `source` (a coded picture `stride` wide, `height` tall)
/// after SAO with `param`, exactly as [`apply_plane`] filters it.
#[allow(clippy::too_many_arguments)]
#[inline]
fn sao_sample(
    param: SaoParam,
    source: &[u16],
    stride: usize,
    height: usize,
    x: usize,
    y: usize,
    bit_depth: u8,
    max_value: i32,
) -> u16 {
    let width = stride;
    let s = source[y * stride + x];
    match param.type_idx {
        1 => {
            let rel = ((s >> (bit_depth - 5)) as u8).wrapping_sub(param.band_pos);
            if rel < 4 {
                (i32::from(s) + i32::from(param.offsets[rel as usize])).clamp(0, max_value) as u16
            } else {
                s
            }
        }
        2 => {
            let (dx, dy) = eo_direction(param.eo_class);
            let (xa, ya, xb, yb) = (x as i32 + dx, y as i32 + dy, x as i32 - dx, y as i32 - dy);
            if xa < 0
                || ya < 0
                || xb < 0
                || yb < 0
                || xa as usize >= width
                || xb as usize >= width
                || ya as usize >= height
                || yb as usize >= height
            {
                return s;
            }
            let sign = |p: u16, q: u16| (p > q) as i32 - (p < q) as i32;
            let edge = sign(s, source[ya as usize * stride + xa as usize])
                + sign(s, source[yb as usize * stride + xb as usize])
                + 2;
            let class = match edge {
                0 => 0,
                1 => 1,
                3 => 2,
                4 => 3,
                _ => return s,
            };
            (i32::from(s) + i32::from(param.offsets[class])).clamp(0, max_value) as u16
        }
        _ => s,
    }
}

/// A candidate parameter set for one CTB of one component with its
/// distortion change (relative to SAO off) and syntax bits. `bits` counts
/// the component's own offset syntax plus, for the component that carries
/// them, the type bypass bin and the band position / edge class.
#[derive(Clone, Copy)]
struct Candidate {
    param: SaoParam,
    delta: i64,
    bits: u32,
}

impl Candidate {
    fn cost(&self, lambda: f32) -> f64 {
        self.delta as f64 + f64::from(lambda) * f64::from(self.bits)
    }
}

/// The best band-offset parameters of one CTB (`x0`, `y0` in plane samples).
fn band_candidate(plane: SaoPlane<'_>, x0: usize, y0: usize, lambda: f32) -> Candidate {
    let x1 = (x0 + plane.ctb_w).min(plane.width);
    let y1 = (y0 + plane.ctb_h).min(plane.height);
    let max_offset = if plane.bit_depth <= 8 { 7 } else { 31 };
    let shift = plane.bit_depth - 5;
    let mut bo_count = [0u32; 32];
    let mut bo_sum = [0i64; 32];
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = y * plane.stride + x;
            let band = (plane.rec[idx] >> shift) as usize;
            bo_count[band] += 1;
            bo_sum[band] += i64::from(plane.original[idx]) - i64::from(plane.rec[idx]);
        }
    }
    let mut best = Candidate {
        param: SaoParam::default(),
        delta: 0,
        bits: 0,
    };
    let mut best_cost = 0.0f64;
    for band in 0..=28usize {
        let mut offsets = [0i8; 4];
        let mut delta = 0i64;
        let mut bits = 1 + 5; // type bypass + sao_band_position
        for k in 0..4 {
            let (o, class_delta, class_bits) = select_offset(
                bo_sum[band + k],
                bo_count[band + k],
                -max_offset,
                max_offset,
                max_offset,
                lambda,
                true,
            );
            offsets[k] = o as i8;
            delta += class_delta;
            bits += class_bits;
        }
        let candidate = Candidate {
            param: SaoParam {
                type_idx: 1,
                offsets,
                band_pos: band as u8,
                eo_class: 0,
            },
            delta,
            bits,
        };
        let cost = candidate.cost(lambda);
        if cost < best_cost {
            best_cost = cost;
            best = candidate;
        }
    }
    best
}

/// The best edge-offset parameters of one CTB for the given class.
fn edge_candidate(
    plane: SaoPlane<'_>,
    x0: usize,
    y0: usize,
    eo_class: u8,
    lambda: f32,
) -> Candidate {
    let x1 = (x0 + plane.ctb_w).min(plane.width);
    let y1 = (y0 + plane.ctb_h).min(plane.height);
    let max_offset = if plane.bit_depth <= 8 { 7 } else { 31 };
    let (dx, dy) = eo_direction(eo_class);
    let mut count = [0u32; 4];
    let mut sum = [0i64; 4];
    for y in y0..y1 {
        for x in x0..x1 {
            let xa = x as i32 + dx;
            let ya = y as i32 + dy;
            let xb = x as i32 - dx;
            let yb = y as i32 - dy;
            // Neighbor availability is the decoder's: the coded picture
            // (stride × coded_height), not the visible crop.
            if xa < 0
                || ya < 0
                || xb < 0
                || yb < 0
                || xa as usize >= plane.stride
                || xb as usize >= plane.stride
                || ya as usize >= plane.coded_height
                || yb as usize >= plane.coded_height
            {
                continue;
            }
            let idx = y * plane.stride + x;
            let s = plane.rec[idx];
            let a = plane.rec[ya as usize * plane.stride + xa as usize];
            let b = plane.rec[yb as usize * plane.stride + xb as usize];
            let sign = |p: u16, q: u16| (p > q) as i32 - (p < q) as i32;
            let edge = sign(s, a) + sign(s, b) + 2;
            let class = match edge {
                0 => Some(0),
                1 => Some(1),
                3 => Some(2),
                4 => Some(3),
                _ => None,
            };
            if let Some(class) = class {
                count[class] += 1;
                sum[class] += i64::from(plane.original[idx]) - i64::from(s);
            }
        }
    }
    let mut offsets = [0i8; 4];
    let mut delta = 0i64;
    let mut bits = 1 + 2; // type bypass + sao_eo_class
    for k in 0..4 {
        let (lo, hi) = if k < 2 {
            (0, max_offset)
        } else {
            (-max_offset, 0)
        };
        let (o, class_delta, class_bits) =
            select_offset(sum[k], count[k], lo, hi, max_offset, lambda, false);
        offsets[k] = o as i8;
        delta += class_delta;
        bits += class_bits;
    }
    Candidate {
        param: SaoParam {
            type_idx: 2,
            offsets,
            band_pos: 0,
            eo_class,
        },
        delta,
        bits,
    }
}

/// Luma SAO parameters of one CTU.
fn analyze_luma_ctu(plane: SaoPlane<'_>, x0: usize, y0: usize, lambda: f32) -> SaoParam {
    // OFF codes one context bin. Candidate costs are relative to OFF, so the
    // type prefix contributes roughly one extra bit for enabled SAO.
    let mut best = band_candidate(plane, x0, y0, lambda);
    let mut best_cost = best.cost(lambda).min(0.0);
    for eo_class in 0..4u8 {
        let candidate = edge_candidate(plane, x0, y0, eo_class, lambda);
        let cost = candidate.cost(lambda);
        if cost < best_cost {
            best_cost = cost;
            best = candidate;
        }
    }
    if best_cost >= 0.0 || best.param.type_idx == 0 {
        return SaoParam::default();
    }
    // The offsets above are chosen in plain SSE, but every other decision of
    // this encoder measures the perceptually weighted error, which discounts
    // the high-frequency corrections SAO mostly makes. Keep the parameters
    // only if they also lower the weighted J of the CTU (its coded area, as
    // the decoder filters it): −0.22% ssimulacra2 / −0.13% CVVDP BD-rate,
    // every image better, no measurable time. Re-choosing among the per-type
    // winners by weighted J measured the same.
    let (stride, coded_height, bit_depth) = (plane.stride, plane.coded_height, plane.bit_depth);
    let (x1, y1) = (
        (x0 + plane.ctb_w).min(stride),
        (y0 + plane.ctb_h).min(coded_height),
    );
    let (h, w) = (y1 - y0, x1 - x0);
    let max_value = ((1u32 << bit_depth) - 1) as i32;
    let (original, rec) = (plane.original, plane.rec);
    let off = crate::cost::weighted_sse(h, w, |r, c| {
        let idx = (y0 + r) * stride + x0 + c;
        i32::from(original[idx]) - i32::from(rec[idx])
    });
    let on = crate::cost::weighted_sse(h, w, |r, c| {
        let (y, x) = (y0 + r, x0 + c);
        let filtered = sao_sample(
            best.param,
            rec,
            stride,
            coded_height,
            x,
            y,
            bit_depth,
            max_value,
        );
        i32::from(original[y * stride + x]) - i32::from(filtered)
    });
    if f64::from(on) + f64::from(lambda) * f64::from(best.bits) < f64::from(off) {
        best.param
    } else {
        SaoParam::default()
    }
}

/// Joint Cb/Cr SAO parameters of one CTU: the type and the edge class are
/// shared, so each is chosen on the summed cost of both components, with
/// their offsets and band positions optimised independently inside it. Cr
/// carries neither the type bypass bin nor the band/class syntax, so its
/// candidate bits are reduced accordingly.
fn analyze_chroma_ctu(
    cb: SaoPlane<'_>,
    cr: SaoPlane<'_>,
    x0: usize,
    y0: usize,
    lambda: f32,
) -> (SaoParam, SaoParam) {
    let off = (SaoParam::default(), SaoParam::default());
    let mut best = off;
    let mut best_cost = 0.0f64;
    let (b_cb, b_cr) = (
        band_candidate(cb, x0, y0, lambda),
        band_candidate(cr, x0, y0, lambda),
    );
    if b_cb.param.type_idx == 1 || b_cr.param.type_idx == 1 {
        let (mut p_cb, mut p_cr) = (b_cb.param, b_cr.param);
        p_cb.type_idx = 1;
        p_cr.type_idx = 1;
        let bits = b_cb.bits.max(1 + 5) + b_cr.bits.max(1 + 5) - 1;
        let cost = (b_cb.delta + b_cr.delta) as f64 + f64::from(lambda) * f64::from(bits);
        if cost < best_cost {
            best_cost = cost;
            best = (p_cb, p_cr);
        }
    }
    for eo_class in 0..4u8 {
        let e_cb = edge_candidate(cb, x0, y0, eo_class, lambda);
        let e_cr = edge_candidate(cr, x0, y0, eo_class, lambda);
        let bits = e_cb.bits + e_cr.bits - 3;
        let cost = (e_cb.delta + e_cr.delta) as f64 + f64::from(lambda) * f64::from(bits);
        if cost < best_cost {
            best_cost = cost;
            best = (e_cb.param, e_cr.param);
        }
    }
    best
}

/// Per-CTU luma SAO parameters of a picture.
pub(crate) fn analyze_luma(plane: SaoPlane<'_>, lambda: f32) -> Vec<SaoParam> {
    let cols = plane.width.div_ceil(plane.ctb_w);
    let rows = plane.height.div_ceil(plane.ctb_h);
    let mut out = Vec::with_capacity(cols * rows);
    for ry in 0..rows {
        for rx in 0..cols {
            out.push(analyze_luma_ctu(
                plane,
                rx * plane.ctb_w,
                ry * plane.ctb_h,
                lambda,
            ));
        }
    }
    out
}

/// Per-CTU chroma SAO parameters of a picture, written into `out[i].cb` /
/// `out[i].cr` (one entry per CTU in raster order).
pub(crate) fn analyze_chroma(cb: SaoPlane<'_>, cr: SaoPlane<'_>, lambda: f32, out: &mut [SaoCtu]) {
    let cols = cb.width.div_ceil(cb.ctb_w);
    let rows = cb.height.div_ceil(cb.ctb_h);
    debug_assert_eq!(out.len(), cols * rows);
    for ry in 0..rows {
        for rx in 0..cols {
            let (p_cb, p_cr) = analyze_chroma_ctu(cb, cr, rx * cb.ctb_w, ry * cb.ctb_h, lambda);
            let ctu = &mut out[ry * cols + rx];
            ctu.cb = p_cb;
            ctu.cr = p_cr;
        }
    }
}

fn encode_offsets<W: CabacWriter>(enc: &mut W, param: SaoParam, bit_depth: u8) {
    let max = if bit_depth <= 8 { 7 } else { 31 };
    for &offset in &param.offsets {
        let magnitude = i32::from(offset).abs();
        for _ in 0..magnitude {
            enc.encode_bypass(1);
        }
        if magnitude < max {
            enc.encode_bypass(0);
        }
    }
    if param.type_idx == 1 {
        for &offset in &param.offsets {
            if offset != 0 {
                enc.encode_bypass(u8::from(offset < 0));
            }
        }
        for bit in (0..5).rev() {
            enc.encode_bypass((param.band_pos >> bit) & 1);
        }
    }
}

/// The `sao()` syntax of one CTU (§7.3.8.3). Merge flags are always coded
/// as 0. `chroma` is `slice_sao_chroma_flag`: Cb carries the shared type and
/// edge class, Cr only its offsets and band position.
pub(crate) fn encode_ctu<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    param: &SaoCtu,
    left_available: bool,
    up_available: bool,
    bit_depth: u8,
    chroma: bool,
) {
    if left_available {
        enc.encode_bin(0, &mut ctx.sao_merge_flag);
    }
    if up_available {
        enc.encode_bin(0, &mut ctx.sao_merge_flag);
    }
    enc.encode_bin(u8::from(param.y.type_idx != 0), &mut ctx.sao_type_idx);
    if param.y.type_idx != 0 {
        enc.encode_bypass(u8::from(param.y.type_idx == 2));
        encode_offsets(enc, param.y, bit_depth);
        if param.y.type_idx == 2 {
            enc.encode_bypass((param.y.eo_class >> 1) & 1);
            enc.encode_bypass(param.y.eo_class & 1);
        }
    }
    if !chroma {
        return;
    }
    debug_assert_eq!(param.cb.type_idx, param.cr.type_idx);
    debug_assert!(param.cb.type_idx != 2 || param.cb.eo_class == param.cr.eo_class);
    enc.encode_bin(u8::from(param.cb.type_idx != 0), &mut ctx.sao_type_idx);
    if param.cb.type_idx == 0 {
        return;
    }
    enc.encode_bypass(u8::from(param.cb.type_idx == 2));
    encode_offsets(enc, param.cb, bit_depth);
    if param.cb.type_idx == 2 {
        enc.encode_bypass((param.cb.eo_class >> 1) & 1);
        enc.encode_bypass(param.cb.eo_class & 1);
    }
    encode_offsets(enc, param.cr, bit_depth);
}

/// Apply SAO to one coded plane (`stride × height`, CTBs of `ctb_w × ctb_h`
/// samples); `param_at(i)` gives the parameters of CTB `i` in raster order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_plane(
    plane: &mut [u16],
    stride: usize,
    height: usize,
    ctb_w: usize,
    ctb_h: usize,
    bit_depth: u8,
    param_at: impl Fn(usize) -> SaoParam,
    source_scratch: &mut Vec<u16>,
) {
    source_scratch.clear();
    source_scratch.extend_from_slice(&plane[..stride * height]);
    let source: &[u16] = source_scratch;
    let width = stride;
    let cols = width.div_ceil(ctb_w);
    let rows = height.div_ceil(ctb_h);
    let max_value = ((1u32 << bit_depth) - 1) as i32;
    for index in 0..cols * rows {
        let param = param_at(index);
        if param.type_idx == 0 {
            continue;
        }
        let rx = index % cols;
        let ry = index / cols;
        let x0 = rx * ctb_w;
        let y0 = ry * ctb_h;
        let x1 = (x0 + ctb_w).min(width);
        let y1 = (y0 + ctb_h).min(height);
        if param.type_idx == 1 {
            let shift = bit_depth - 5;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * stride + x;
                    let band = (source[idx] >> shift) as u8;
                    let rel = band.wrapping_sub(param.band_pos);
                    if rel < 4 {
                        plane[idx] = (i32::from(source[idx])
                            + i32::from(param.offsets[rel as usize]))
                        .clamp(0, max_value) as u16;
                    }
                }
            }
        } else {
            let (dx, dy) = eo_direction(param.eo_class);
            for y in y0..y1 {
                for x in x0..x1 {
                    let xa = x as i32 + dx;
                    let ya = y as i32 + dy;
                    let xb = x as i32 - dx;
                    let yb = y as i32 - dy;
                    if xa < 0
                        || ya < 0
                        || xb < 0
                        || yb < 0
                        || xa as usize >= width
                        || xb as usize >= width
                        || ya as usize >= height
                        || yb as usize >= height
                    {
                        continue;
                    }
                    let idx = y * stride + x;
                    let s = source[idx];
                    let sign = |p: u16, q: u16| (p > q) as i32 - (p < q) as i32;
                    let edge = sign(s, source[ya as usize * stride + xa as usize])
                        + sign(s, source[yb as usize * stride + xb as usize])
                        + 2;
                    let class = match edge {
                        0 => Some(0),
                        1 => Some(1),
                        3 => Some(2),
                        4 => Some(3),
                        _ => None,
                    };
                    if let Some(class) = class {
                        plane[idx] = (i32::from(s) + i32::from(param.offsets[class]))
                            .clamp(0, max_value) as u16;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_offsets_obey_eo_sign_constraints() {
        assert_eq!(rounded_offset(17, 4, 0, 7), 4);
        assert_eq!(rounded_offset(-17, 4, 0, 7), 0);
        assert_eq!(rounded_offset(-17, 4, -7, 0), -4);
    }

    #[test]
    fn band_search_finds_quantization_bias() {
        let original = vec![100u16; 64 * 64];
        let rec = vec![96u16; 64 * 64];
        let plane = SaoPlane {
            original: &original,
            rec: &rec,
            stride: 64,
            width: 64,
            height: 64,
            coded_height: 64,
            ctb_w: 64,
            ctb_h: 64,
            bit_depth: 8,
        };
        let p = analyze_luma(plane, 1.0);
        assert_eq!(p[0].type_idx, 1);
        assert!(p[0].offsets.contains(&4));
        let mut filtered = rec;
        apply_plane(&mut filtered, 64, 64, 64, 64, 8, |i| p[i], &mut Vec::new());
        assert_eq!(filtered, original);
    }

    #[test]
    fn chroma_search_shares_the_type_and_corrects_both_components() {
        // 4:2:0 chroma CTBs are 32×32: Cb is biased by −3, Cr by +2.
        let orig_cb = vec![100u16; 32 * 32];
        let rec_cb = vec![97u16; 32 * 32];
        let orig_cr = vec![120u16; 32 * 32];
        let rec_cr = vec![122u16; 32 * 32];
        fn plane<'a>(original: &'a [u16], rec: &'a [u16]) -> SaoPlane<'a> {
            SaoPlane {
                original,
                rec,
                stride: 32,
                width: 32,
                height: 32,
                coded_height: 32,
                ctb_w: 32,
                ctb_h: 32,
                bit_depth: 8,
            }
        }
        let mut params = vec![SaoCtu::default(); 1];
        analyze_chroma(
            plane(&orig_cb, &rec_cb),
            plane(&orig_cr, &rec_cr),
            1.0,
            &mut params,
        );
        assert_eq!(params[0].cb.type_idx, 1);
        assert_eq!(params[0].cr.type_idx, 1);
        let mut cb = rec_cb;
        let mut cr = rec_cr;
        apply_plane(
            &mut cb,
            32,
            32,
            32,
            32,
            8,
            |i| params[i].cb,
            &mut Vec::new(),
        );
        apply_plane(
            &mut cr,
            32,
            32,
            32,
            32,
            8,
            |i| params[i].cr,
            &mut Vec::new(),
        );
        assert_eq!(cb, orig_cb);
        assert_eq!(cr, orig_cr);
    }

    #[test]
    fn chroma_syntax_codes_cr_offsets_without_type_or_class() {
        use crate::cabac::CabacEstimator;
        let mut ctx = ContextSet::init_islice(30);
        let luma_only = SaoCtu::default();
        let mut est = CabacEstimator::default();
        encode_ctu(&mut est, &mut ctx, &luma_only, false, false, 8, false);
        let luma_bits = est.bits();
        // Chroma off adds exactly the one context-coded sao_type_idx_chroma bin.
        let mut ctx2 = ContextSet::init_islice(30);
        let mut est2 = CabacEstimator::default();
        encode_ctu(&mut est2, &mut ctx2, &luma_only, false, false, 8, true);
        assert!(est2.bits() > luma_bits && est2.bits() < luma_bits + 2.0);
        // Edge offset: Cb carries the class, Cr only four offsets (the
        // zero offsets cost one bypass bin each).
        let eo = SaoParam {
            type_idx: 2,
            offsets: [1, 0, 0, -1],
            band_pos: 0,
            eo_class: 3,
        };
        let both = SaoCtu {
            y: SaoParam::default(),
            cb: eo,
            cr: SaoParam {
                offsets: [0, 0, 0, 0],
                ..eo
            },
        };
        let mut ctx3 = ContextSet::init_islice(30);
        let mut est3 = CabacEstimator::default();
        encode_ctu(&mut est3, &mut ctx3, &both, false, false, 8, true);
        // luma off + chroma type (1 ctx bin coded as 1, 1 bypass) + Cb offsets
        // (2+1+1+2 = 6 bypass) + class (2 bypass) + Cr offsets (4 bypass).
        let type_bits = {
            let mut ctx = ContextSet::init_islice(30);
            ctx.sao_type_idx.estimate_and_update(0);
            ctx.sao_type_idx.estimated_bits(1)
        };
        let bypass = 1.0 + 6.0 + 2.0 + 4.0;
        assert!((est3.bits() - luma_bits - type_bits - bypass).abs() < 0.01);
    }
}
