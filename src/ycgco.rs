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

use crate::{BitDepth, ChromaFormat, Yuv};

/// YCgCo-R luma. `Y = t + (Cg >> 1)` where `t = B + ((R-B) >> 1)`. Range [0, 2ᴺ-1].
#[inline]
pub(crate) fn rgb_to_ycgco_y(r: i32, g: i32, b: i32) -> i32 {
    let hg0 = g >> 1;
    hg0 + ((r + b) >> 2)
}

/// YCgCo-R chrominance-green, offset by `neutral`. Un-offset range [-(2ᴺ-1), 2ᴺ-1].
#[inline]
pub(crate) fn rgb_to_ycgco_cg(r: i32, g: i32, b: i32, neutral: i32) -> i32 {
    ((g >> 1) - ((r + b) >> 2)) + neutral
}

/// YCgCo-R chrominance-orange, offset by `neutral`. Un-offset range [-(2ᴺ-1), 2ᴺ-1].
#[inline]
pub(crate) fn rgb_to_ycgco_co(r: i32, _g: i32, b: i32, neutral: i32) -> i32 {
    ((r - b) >> 1) + neutral
}

/// Convert planar RGB samples to planar YCgCo-R in the requested chroma format.
pub(crate) fn rgb_to_ycgco(
    rgb: &[u16],
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Yuv {
    let w = width as usize;
    let h = height as usize;
    let maxv = bit_depth.max_val() as i32;
    let neutral = bit_depth.neutral() as i32;

    if chroma.is_monochrome() {
        let channels = rgb.len() / (w * h);
        let y_plane: Vec<u16> = if channels == 1 {
            rgb.to_vec()
        } else if channels == 4 {
            rgb.as_chunks::<4>()
                .0
                .iter()
                .map(|px| {
                    let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                    rgb_to_ycgco_y(r, g, b).clamp(0, maxv) as u16
                })
                .collect()
        } else if channels == 3 {
            rgb.as_chunks::<3>()
                .0
                .iter()
                .map(|px| {
                    let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                    rgb_to_ycgco_y(r, g, b).clamp(0, maxv) as u16
                })
                .collect()
        } else {
            unimplemented!(
                "Amount of channels {} in 'rgb_to_ycgco' is not supported",
                channels
            )
        };
        return Yuv {
            y: y_plane,
            cb: Vec::new(),
            cr: Vec::new(),
            width,
            height,
            display_w: width,
            display_h: height,
            chroma,
            bit_depth,
        };
    }

    let sw = chroma.sub_w();
    let sh = chroma.sub_h();
    let cw = w.div_ceil(sw);
    let ch = h.div_ceil(sh);

    let mut y_plane = vec![0u16; w * h];
    let mut cg_plane = vec![0u16; cw * ch]; // Cg
    let mut co_plane = vec![0u16; cw * ch]; // Co

    let process_row =
        |src: &[u16], y_dst: &mut [u16], cg_dst: Option<&mut [u16]>, co_dst: Option<&mut [u16]>| {
            // Luma — every pixel.
            for (y_out, px) in y_dst.iter_mut().zip(src.as_chunks::<3>().0.iter()) {
                let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                *y_out = rgb_to_ycgco_y(r, g, b).clamp(0, maxv) as u16;
            }

            // Chroma — only when this row contributes a chroma row.
            if let (Some(cg_out), Some(co_out)) = (cg_dst, co_dst) {
                let pairs = src.as_chunks::<6>();
                let remainder = pairs.1; // 0 or 3 samples (odd width)

                for ((cg_out, co_out), pair) in
                    cg_out.iter_mut().zip(co_out.iter_mut()).zip(pairs.0.iter())
                {
                    let (r0, g0, b0) = (pair[0] as i32, pair[1] as i32, pair[2] as i32);
                    let (r1, g1, b1) = (pair[3] as i32, pair[4] as i32, pair[5] as i32);
                    // Horizontal average of two adjacent pixels (Q0).
                    *cg_out = ((rgb_to_ycgco_cg(r0, g0, b0, neutral)
                        + rgb_to_ycgco_cg(r1, g1, b1, neutral)
                        + 1)
                        >> 1)
                        .clamp(0, maxv) as u16;
                    *co_out = ((rgb_to_ycgco_co(r0, g0, b0, neutral)
                        + rgb_to_ycgco_co(r1, g1, b1, neutral)
                        + 1)
                        >> 1)
                        .clamp(0, maxv) as u16;
                }

                if !remainder.is_empty() {
                    let (r, g, b) = (
                        remainder[0] as i32,
                        remainder[1] as i32,
                        remainder[2] as i32,
                    );
                    if let Some(cg) = cg_out.last_mut() {
                        *cg = rgb_to_ycgco_cg(r, g, b, neutral).clamp(0, maxv) as u16;
                    }
                    if let Some(co) = co_out.last_mut() {
                        *co = rgb_to_ycgco_co(r, g, b, neutral).clamp(0, maxv) as u16;
                    }
                }
            }
        };

    let blend_chroma_row = |src: &[u16], cg_row: &mut [u16], co_row: &mut [u16]| {
        let pairs = src.as_chunks::<6>();
        let remainder = pairs.1;

        for ((cg_out, co_out), pair) in cg_row.iter_mut().zip(co_row.iter_mut()).zip(pairs.0.iter())
        {
            let (r0, g0, b0) = (pair[0] as i32, pair[1] as i32, pair[2] as i32);
            let (r1, g1, b1) = (pair[3] as i32, pair[4] as i32, pair[5] as i32);
            let cg1 =
                ((rgb_to_ycgco_cg(r0, g0, b0, neutral) + rgb_to_ycgco_cg(r1, g1, b1, neutral) + 1)
                    >> 1)
                    .clamp(0, maxv);
            let co1 =
                ((rgb_to_ycgco_co(r0, g0, b0, neutral) + rgb_to_ycgco_co(r1, g1, b1, neutral) + 1)
                    >> 1)
                    .clamp(0, maxv);
            // Vertical average with row0 value already stored.
            *cg_out = ((*cg_out as i32 + cg1 + 1) >> 1) as u16;
            *co_out = ((*co_out as i32 + co1 + 1) >> 1) as u16;
        }

        // Odd-width remainder.
        if !remainder.is_empty() {
            let (r, g, b) = (
                remainder[0] as i32,
                remainder[1] as i32,
                remainder[2] as i32,
            );
            if let Some(cg_out) = cg_row.last_mut() {
                let cg1 = rgb_to_ycgco_cg(r, g, b, neutral).clamp(0, maxv);
                *cg_out = ((*cg_out as i32 + cg1 + 1) >> 1) as u16;
            }
            if let Some(co_out) = co_row.last_mut() {
                let co1 = rgb_to_ycgco_co(r, g, b, neutral).clamp(0, maxv);
                *co_out = ((*co_out as i32 + co1 + 1) >> 1) as u16;
            }
        }
    };

    match chroma {
        ChromaFormat::Yuv444 => {
            for (row, ((y_row, cg_row), co_row)) in y_plane
                .chunks_exact_mut(w)
                .zip(cg_plane.chunks_exact_mut(cw))
                .zip(co_plane.chunks_exact_mut(cw))
                .enumerate()
            {
                let src = &rgb[row * w * 3..(row + 1) * w * 3];
                for (((y_out, cg_out), co_out), px) in y_row
                    .iter_mut()
                    .zip(cg_row.iter_mut())
                    .zip(co_row.iter_mut())
                    .zip(src.as_chunks::<3>().0.iter())
                {
                    let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                    *y_out = rgb_to_ycgco_y(r, g, b).clamp(0, maxv) as u16;
                    *cg_out = rgb_to_ycgco_cg(r, g, b, neutral).clamp(0, maxv) as u16;
                    *co_out = rgb_to_ycgco_co(r, g, b, neutral).clamp(0, maxv) as u16;
                }
            }
        }
        ChromaFormat::Yuv422 => {
            for (row, ((y_row, cg_row), co_row)) in y_plane
                .chunks_exact_mut(w)
                .zip(cg_plane.chunks_exact_mut(cw))
                .zip(co_plane.chunks_exact_mut(cw))
                .enumerate()
            {
                let src = &rgb[row * w * 3..(row + 1) * w * 3];
                process_row(src, y_row, Some(cg_row), Some(co_row));
            }
        }

        ChromaFormat::Yuv420 => {
            // Full pairs of luma rows.
            let full_pairs = h / 2;

            for chroma_row in 0..full_pairs {
                let luma_row0 = chroma_row * 2;
                let luma_row1 = luma_row0 + 1;

                let src0 = &rgb[luma_row0 * w * 3..luma_row1 * w * 3];
                let src1 = &rgb[luma_row1 * w * 3..(luma_row1 + 1) * w * 3];

                let y_dst = &mut y_plane[luma_row0 * w..(luma_row1 + 1) * w];
                let (y_row0, y_row1) = y_dst.split_at_mut(w);
                let cg_row = &mut cg_plane[chroma_row * cw..(chroma_row + 1) * cw];
                let co_row = &mut co_plane[chroma_row * cw..(chroma_row + 1) * cw];

                // Row 0: luma + first chroma estimate (horizontal pair average).
                process_row(src0, y_row0, Some(cg_row), Some(co_row));

                // Row 1: luma only.
                process_row(src1, y_row1, None, None);

                // Vertically blend row1's chroma into the row0 estimate.
                blend_chroma_row(src1, cg_row, co_row);
            }

            // Odd height: single trailing luma row with no vertical neighbor.
            // Treat as 4:2:2 — horizontal pair average only.
            if h & 1 != 0 {
                let last_row = h - 1;
                let last_chroma = ch - 1;
                let src = &rgb[last_row * w * 3..(last_row + 1) * w * 3];
                let y_row = &mut y_plane[last_row * w..last_row * w + w];
                let cg_row = &mut cg_plane[last_chroma * cw..last_chroma * cw + cw];
                let co_row = &mut co_plane[last_chroma * cw..last_chroma * cw + cw];
                process_row(src, y_row, Some(cg_row), Some(co_row));
            }
        }

        ChromaFormat::Monochrome => unreachable!("handled above"),
    }

    Yuv {
        y: y_plane,
        cb: cg_plane, // Cg
        cr: co_plane, // Co
        width,
        height,
        display_w: width,
        display_h: height,
        chroma,
        bit_depth,
    }
}

pub(crate) fn rgb_to_gbr(
    rgb: &[u16],
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Yuv {
    if chroma.is_monochrome() {
        return rgb_to_ycgco(rgb, width, height, chroma, bit_depth);
    }
    let n = (width as usize) * (height as usize);
    let channels = rgb.len() / n;
    let mut g = vec![0u16; n];
    let mut b = vec![0u16; n];
    let mut r = vec![0u16; n];
    match channels {
        3 => {
            let (src, _remainder) = rgb.as_chunks::<3>();
            for (&[red, green, blue], (r, (g, b))) in src
                .iter()
                .zip(r.iter_mut().zip(g.iter_mut().zip(b.iter_mut())))
            {
                *r = red;
                *g = green;
                *b = blue;
            }
        }
        4 => {
            let (src, _remainder) = rgb.as_chunks::<4>();
            for (&[red, green, blue, _alpha], (r, (g, b))) in src
                .iter()
                .zip(r.iter_mut().zip(g.iter_mut().zip(b.iter_mut())))
            {
                *r = red;
                *g = green;
                *b = blue;
            }
        }
        _ => unreachable!("unsupported channel count: {channels}"),
    }
    Yuv {
        y: g,
        cb: b,
        cr: r,
        width,
        height,
        display_w: width,
        display_h: height,
        chroma: ChromaFormat::Yuv444,
        bit_depth,
    }
}
