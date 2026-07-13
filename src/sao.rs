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

#[allow(clippy::too_many_arguments)]
fn analyze_ctu(
    original: &[u16],
    rec: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bit_depth: u8,
    lambda: f32,
) -> SaoParam {
    let x1 = (x0 + 64).min(width);
    let y1 = (y0 + 64).min(height);
    let max_offset = if bit_depth <= 8 { 7 } else { 31 };
    let shift = bit_depth - 5;

    let mut bo_count = [0u32; 32];
    let mut bo_sum = [0i64; 32];
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = y * stride + x;
            let band = (rec[idx] >> shift) as usize;
            bo_count[band] += 1;
            bo_sum[band] += i64::from(original[idx]) - i64::from(rec[idx]);
        }
    }

    // OFF codes one context bin. Candidate costs are relative to OFF, so the
    // type prefix contributes roughly one extra bit for enabled SAO.
    let mut best_cost = 0.0f64;
    let mut best = SaoParam::default();
    for band in 0..=28usize {
        let mut offsets = [0i8; 4];
        let mut delta = 0i64;
        let mut bits = 1 + 5; // enabled/type bypass + sao_band_position
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
        let cost = delta as f64 + f64::from(lambda) * f64::from(bits);
        if cost < best_cost {
            best_cost = cost;
            best = SaoParam {
                type_idx: 1,
                offsets,
                band_pos: band as u8,
                eo_class: 0,
            };
        }
    }

    for eo_class in 0..4u8 {
        let (dx, dy): (i32, i32) = match eo_class {
            0 => (1, 0),
            1 => (0, 1),
            2 => (1, 1),
            _ => (1, -1),
        };
        let mut count = [0u32; 4];
        let mut sum = [0i64; 4];
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
                let s = rec[idx];
                let a = rec[ya as usize * stride + xa as usize];
                let b = rec[yb as usize * stride + xb as usize];
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
                    sum[class] += i64::from(original[idx]) - i64::from(s);
                }
            }
        }
        let mut offsets = [0i8; 4];
        let mut delta = 0i64;
        let mut bits = 1 + 2; // enabled/type bypass + sao_eo_class
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
        let cost = delta as f64 + f64::from(lambda) * f64::from(bits);
        if cost < best_cost {
            best_cost = cost;
            best = SaoParam {
                type_idx: 2,
                offsets,
                band_pos: 0,
                eo_class,
            };
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_luma(
    original: &[u16],
    rec: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    lambda: f32,
) -> Vec<SaoParam> {
    let cols = width.div_ceil(64);
    let rows = height.div_ceil(64);
    let mut out = Vec::with_capacity(cols * rows);
    for ry in 0..rows {
        for rx in 0..cols {
            out.push(analyze_ctu(
                original,
                rec,
                stride,
                width,
                height,
                rx * 64,
                ry * 64,
                bit_depth,
                lambda,
            ));
        }
    }
    out
}

pub(crate) fn encode_luma<W: CabacWriter>(
    enc: &mut W,
    ctx: &mut ContextSet,
    param: SaoParam,
    left_available: bool,
    up_available: bool,
    bit_depth: u8,
) {
    if left_available {
        enc.encode_bin(0, &mut ctx.sao_merge_flag);
    }
    if up_available {
        enc.encode_bin(0, &mut ctx.sao_merge_flag);
    }
    enc.encode_bin(u8::from(param.type_idx != 0), &mut ctx.sao_type_idx);
    if param.type_idx == 0 {
        return;
    }
    enc.encode_bypass(u8::from(param.type_idx == 2));
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
    } else {
        enc.encode_bypass((param.eo_class >> 1) & 1);
        enc.encode_bypass(param.eo_class & 1);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_luma(
    plane: &mut [u16],
    stride: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    params: &[SaoParam],
) {
    let source = plane.to_vec();
    let cols = width.div_ceil(64);
    let max_value = ((1u32 << bit_depth) - 1) as i32;
    for (index, &param) in params.iter().enumerate() {
        if param.type_idx == 0 {
            continue;
        }
        let rx = index % cols;
        let ry = index / cols;
        let x0 = rx * 64;
        let y0 = ry * 64;
        let x1 = (x0 + 64).min(width);
        let y1 = (y0 + 64).min(height);
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
            let (dx, dy): (i32, i32) = match param.eo_class {
                0 => (1, 0),
                1 => (0, 1),
                2 => (1, 1),
                _ => (1, -1),
            };
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
        let p = analyze_luma(&original, &rec, 64, 64, 64, 8, 1.0);
        assert_eq!(p[0].type_idx, 1);
        assert!(p[0].offsets.contains(&4));
        let mut filtered = rec;
        apply_luma(&mut filtered, 64, 64, 64, 8, &p);
        assert_eq!(filtered, original);
    }
}
