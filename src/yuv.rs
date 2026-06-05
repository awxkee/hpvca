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
use crate::fmt::{BitDepth, ChromaFormat};
use crate::{EncodeError, checked_buffer_size, validate_dims};

/// Planar YCbCr image
pub struct Yuv {
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
    pub width: u32,
    pub height: u32,
    pub display_w: u32,
    pub display_h: u32,
    pub chroma: ChromaFormat,
    pub bit_depth: BitDepth,
}

impl Yuv {
    /// Build a `Yuv` from caller-supplied planar samples, for the YUV-direct encode
    /// path ([`crate::encode_yuv`]). Validates that the plane lengths match the
    /// dimensions and chroma format. For monochrome, `cb`/`cr` must be empty.
    ///
    /// Samples must be at `bit_depth`'s native range. `width`/`height` are the visible
    /// dimensions; the luma plane must be exactly `width*height` and each chroma plane
    /// `ceil(width/sub_w) * ceil(height/sub_h)`.
    pub fn from_planes(
        y: Vec<u16>,
        cb: Vec<u16>,
        cr: Vec<u16>,
        width: u32,
        height: u32,
        chroma: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Result<Self, EncodeError> {
        let w = width as usize;
        let h = height as usize;
        if y.len() != w * h {
            return Err(EncodeError::InvalidInput);
        }
        if chroma.is_monochrome() {
            if !cb.is_empty() || !cr.is_empty() {
                return Err(EncodeError::InvalidInput);
            }
        } else {
            let cw = w.div_ceil(chroma.sub_w());
            let ch = h.div_ceil(chroma.sub_h());
            if cb.len() != cw * ch || cr.len() != cw * ch {
                return Err(EncodeError::InvalidInput);
            }
        }
        Ok(Yuv {
            y,
            cb,
            cr,
            width,
            height,
            display_w: width,
            display_h: height,
            chroma,
            bit_depth,
        })
    }

    /// Override the visible/display dimensions (reported via the HEIF `ispe` box),
    /// for sources whose true size is smaller than the coded planes — e.g. an odd
    /// width/height under a subsampled chroma format. Must be ≤ the coded
    /// `width`/`height`; otherwise the call is a no-op-safe clamp.
    pub fn with_display(mut self, display_w: u32, display_h: u32) -> Self {
        self.display_w = display_w.min(self.width);
        self.display_h = display_h.min(self.height);
        self
    }

    pub fn luma_stride(&self) -> usize {
        self.width as usize
    }
    pub fn chroma_stride(&self) -> usize {
        (self.width as usize).div_ceil(self.chroma.sub_w())
    }
    pub fn chroma_height(&self) -> usize {
        (self.height as usize).div_ceil(self.chroma.sub_h())
    }
}

impl Yuv {
    pub fn validate(&self) -> Result<(), EncodeError> {
        let w = self.width as usize;
        let h = self.height as usize;

        validate_dims(self.width, self.height)?;

        // Luma plane must hold exactly w × h samples.
        let expected_luma = checked_buffer_size::<u16>(w, h, 1)?;
        if self.y.len() < expected_luma {
            return Err(EncodeError::InvalidInput);
        }

        // Chroma planes: size depends on subsampling.
        if self.chroma.is_monochrome() {
            // 4:0:0: both chroma planes must be empty.
            if !self.cb.is_empty() || !self.cr.is_empty() {
                return Err(EncodeError::InvalidInput);
            }
        } else {
            let cw = w.div_ceil(self.chroma.sub_w());
            let ch = h.div_ceil(self.chroma.sub_h());
            let expected_chroma = checked_buffer_size::<u16>(cw, ch, 1)?;
            if self.cb.len() < expected_chroma {
                return Err(EncodeError::InvalidInput);
            }
            if self.cr.len() < expected_chroma {
                return Err(EncodeError::InvalidInput);
            }
        }

        // Display size must not exceed coded size.
        if self.display_w > self.width || self.display_h > self.height {
            return Err(EncodeError::InvalidInput);
        }

        // Coded dimensions must be on the chroma subsampling grid.
        let sw = self.chroma.sub_w() as u32;
        let sh = self.chroma.sub_h() as u32;
        if !self.width.is_multiple_of(sw) || !self.height.is_multiple_of(sh) {
            return Err(EncodeError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }

        Ok(())
    }
}

/// Q0.13 fixed-point scale.
const Q13: i32 = 1 << 13;
const Q13_HALF: i32 = 1 << 12;

/// BT.709 luma coefficients in Q0.13.
const KR: i32 = (0.2126_f64 * Q13 as f64) as i32; // 1742
const KG: i32 = (0.7152_f64 * Q13 as f64) as i32; // 5865
const KB: i32 = (0.0722_f64 * Q13 as f64) as i32; // 592

/// Chroma reciprocal scales in Q0.13.
/// diff is Q0 (±maxv), so diff * REC_*_Q13 fits in i32 at 12-bit:
/// 4095 * 5202 = 21_302_490 < 2^25. Safe.
const REC_CB_Q13: i32 = (Q13 as f64 / 1.8556_f64) as i32; // 4416
const REC_CR_Q13: i32 = (Q13 as f64 / 1.5748_f64) as i32; // 5202

/// Luma dot product in Q13. Call q13_round() to get a pixel value.
#[inline(always)]
fn rgb_to_y_q13(r: i32, g: i32, b: i32) -> i32 {
    KR * r + KG * g + KB * b
}

/// Luma pixel value (Q0, rounded).
#[inline(always)]
fn rgb_to_y(r: i32, g: i32, b: i32) -> i32 {
    (rgb_to_y_q13(r, g, b) + Q13_HALF) >> 13
}

/// Cb pixel value (Q0). neutral is the bit-depth midpoint (128 / 512 / 2048).
#[inline(always)]
fn rgb_to_cb(r: i32, g: i32, b: i32, neutral: i32) -> i32 {
    let y = rgb_to_y(r, g, b);
    neutral + (((b - y) * REC_CB_Q13 + Q13_HALF) >> 13)
}

/// Cr pixel value (Q0).
#[inline(always)]
fn rgb_to_cr(r: i32, g: i32, b: i32, neutral: i32) -> i32 {
    let y = rgb_to_y(r, g, b);
    neutral + (((r - y) * REC_CR_Q13 + Q13_HALF) >> 13)
}

/// Convert planar RGB samples to planar YCbCr in the requested chroma format.
///
/// For subsampled formats (4:2:0, 4:2:2) the dimensions do NOT need to be
/// pre-aligned — odd widths and heights are handled via the `chunks_exact`
/// remainder path exactly as the reference YUV library does.
pub(crate) fn rgb_to_yuv(
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

    // ── Monochrome ────────────────────────────────────────────────────────
    if chroma.is_monochrome() {
        let channels = rgb.len() / (w * h);
        let y_plane: Vec<u16> = if channels == 1 {
            rgb.to_vec()
        } else {
            rgb.chunks_exact(channels)
                .map(|px| {
                    let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                    rgb_to_y(r, g, b).clamp(0, maxv) as u16
                })
                .collect()
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

    // ── Plane allocation ──────────────────────────────────────────────────
    let sw = chroma.sub_w();
    let sh = chroma.sub_h();
    let cw = w.div_ceil(sw);
    let ch = h.div_ceil(sh);

    let mut y_plane = vec![0u16; w * h];
    let mut cb_plane = vec![0u16; cw * ch];
    let mut cr_plane = vec![0u16; cw * ch];

    // ── Inner helpers (capture maxv, neutral by value) ────────────────────

    // Write luma for every pixel in one source row.
    // Write chroma (horizontal average of pixel pairs) when cb/cr are Some.
    // Handles odd-width rows via chunks_exact remainder — no bounds issues.
    let process_row = |src: &[u16],
                       y_dst: &mut [u16],
                       cb_dst: Option<&mut [u16]>,
                       cr_dst: Option<&mut [u16]>| {
        // Luma — every pixel.
        for (y_out, px) in y_dst.iter_mut().zip(src.chunks_exact(3)) {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            *y_out = rgb_to_y(r, g, b).clamp(0, maxv) as u16;
        }

        // Chroma — only when this row contributes a chroma row.
        if let (Some(cb_out), Some(cr_out)) = (cb_dst, cr_dst) {
            let pairs = src.chunks_exact(6);
            let remainder = pairs.remainder(); // 0 or 3 samples (odd width)

            for ((cb_out, cr_out), pair) in cb_out.iter_mut().zip(cr_out.iter_mut()).zip(pairs) {
                let (r0, g0, b0) = (pair[0] as i32, pair[1] as i32, pair[2] as i32);
                let (r1, g1, b1) = (pair[3] as i32, pair[4] as i32, pair[5] as i32);
                // Horizontal average of two adjacent pixels (Q0).
                *cb_out = ((rgb_to_cb(r0, g0, b0, neutral) + rgb_to_cb(r1, g1, b1, neutral) + 1)
                    >> 1)
                    .clamp(0, maxv) as u16;
                *cr_out = ((rgb_to_cr(r0, g0, b0, neutral) + rgb_to_cr(r1, g1, b1, neutral) + 1)
                    >> 1)
                    .clamp(0, maxv) as u16;
            }

            // Odd-width trailing pixel: no horizontal neighbour, use as-is.
            if !remainder.is_empty() {
                let (r, g, b) = (
                    remainder[0] as i32,
                    remainder[1] as i32,
                    remainder[2] as i32,
                );
                if let Some(cb) = cb_out.last_mut() {
                    *cb = rgb_to_cb(r, g, b, neutral).clamp(0, maxv) as u16;
                }
                if let Some(cr) = cr_out.last_mut() {
                    *cr = rgb_to_cr(r, g, b, neutral).clamp(0, maxv) as u16;
                }
            }
        }
    };

    // Blend a second row's horizontal chroma into an already-written chroma
    // row (vertical average for 4:2:0). Values are Q0 throughout.
    let blend_chroma_row = |src: &[u16], cb_row: &mut [u16], cr_row: &mut [u16]| {
        let pairs = src.chunks_exact(6);
        let remainder = pairs.remainder();

        for ((cb_out, cr_out), pair) in cb_row.iter_mut().zip(cr_row.iter_mut()).zip(pairs) {
            let (r0, g0, b0) = (pair[0] as i32, pair[1] as i32, pair[2] as i32);
            let (r1, g1, b1) = (pair[3] as i32, pair[4] as i32, pair[5] as i32);
            let cb1 = ((rgb_to_cb(r0, g0, b0, neutral) + rgb_to_cb(r1, g1, b1, neutral) + 1) >> 1)
                .clamp(0, maxv);
            let cr1 = ((rgb_to_cr(r0, g0, b0, neutral) + rgb_to_cr(r1, g1, b1, neutral) + 1) >> 1)
                .clamp(0, maxv);
            // Vertical average with row0 value already stored.
            *cb_out = ((*cb_out as i32 + cb1 + 1) >> 1) as u16;
            *cr_out = ((*cr_out as i32 + cr1 + 1) >> 1) as u16;
        }

        // Odd-width remainder.
        if !remainder.is_empty() {
            let (r, g, b) = (
                remainder[0] as i32,
                remainder[1] as i32,
                remainder[2] as i32,
            );
            if let Some(cb_out) = cb_row.last_mut() {
                let cb1 = rgb_to_cb(r, g, b, neutral).clamp(0, maxv);
                *cb_out = ((*cb_out as i32 + cb1 + 1) >> 1) as u16;
            }
            if let Some(cr_out) = cr_row.last_mut() {
                let cr1 = rgb_to_cr(r, g, b, neutral).clamp(0, maxv);
                *cr_out = ((*cr_out as i32 + cr1 + 1) >> 1) as u16;
            }
        }
    };

    match chroma {
        // ── 4:4:4 ─────────────────────────────────────────────────────────
        // One chroma sample per luma pixel — no averaging at all.
        ChromaFormat::Yuv444 => {
            for (row, ((y_row, cb_row), cr_row)) in y_plane
                .chunks_exact_mut(w)
                .zip(cb_plane.chunks_exact_mut(cw))
                .zip(cr_plane.chunks_exact_mut(cw))
                .enumerate()
            {
                let src = &rgb[row * w * 3..(row + 1) * w * 3];
                for (((y_out, cb_out), cr_out), px) in y_row
                    .iter_mut()
                    .zip(cb_row.iter_mut())
                    .zip(cr_row.iter_mut())
                    .zip(src.chunks_exact(3))
                {
                    let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                    *y_out = rgb_to_y(r, g, b).clamp(0, maxv) as u16;
                    *cb_out = rgb_to_cb(r, g, b, neutral).clamp(0, maxv) as u16;
                    *cr_out = rgb_to_cr(r, g, b, neutral).clamp(0, maxv) as u16;
                }
            }
        }

        // ── 4:2:2 ─────────────────────────────────────────────────────────
        // One chroma sample per horizontal pair of luma pixels, full height.
        ChromaFormat::Yuv422 => {
            for (row, ((y_row, cb_row), cr_row)) in y_plane
                .chunks_exact_mut(w)
                .zip(cb_plane.chunks_exact_mut(cw))
                .zip(cr_plane.chunks_exact_mut(cw))
                .enumerate()
            {
                let src = &rgb[row * w * 3..(row + 1) * w * 3];
                process_row(src, y_row, Some(cb_row), Some(cr_row));
            }
        }

        // ── 4:2:0 ─────────────────────────────────────────────────────────
        // One chroma sample per 2×2 luma block.
        // Process two luma rows at a time → one chroma row.
        // Handles odd height: the last unpaired row is treated as 4:2:2.
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
                let cb_row = &mut cb_plane[chroma_row * cw..(chroma_row + 1) * cw];
                let cr_row = &mut cr_plane[chroma_row * cw..(chroma_row + 1) * cw];

                // Row 0: luma + first chroma estimate (horizontal pair average).
                process_row(src0, y_row0, Some(cb_row), Some(cr_row));

                // Row 1: luma only.
                process_row(src1, y_row1, None, None);

                // Vertically blend row1's chroma into the row0 estimate.
                blend_chroma_row(src1, cb_row, cr_row);
            }

            // Odd height: single trailing luma row with no vertical neighbour.
            // Treat as 4:2:2 — horizontal pair average only.
            if h & 1 != 0 {
                let last_row = h - 1;
                let last_chroma = ch - 1;
                let src = &rgb[last_row * w * 3..(last_row + 1) * w * 3];
                let y_row = &mut y_plane[last_row * w..last_row * w + w];
                let cb_row = &mut cb_plane[last_chroma * cw..last_chroma * cw + cw];
                let cr_row = &mut cr_plane[last_chroma * cw..last_chroma * cw + cw];
                process_row(src, y_row, Some(cb_row), Some(cr_row));
            }
        }

        ChromaFormat::Monochrome => unreachable!("handled above"),
    }

    Yuv {
        y: y_plane,
        cb: cb_plane,
        cr: cr_plane,
        width,
        height,
        display_w: width,
        display_h: height,
        chroma,
        bit_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_pixel_8bit() {
        let yuv = rgb_to_yuv(
            &[255u16, 255, 255],
            1,
            1,
            ChromaFormat::Yuv420,
            BitDepth::Eight,
        );
        assert!(yuv.y[0] > 250);
        assert!((yuv.cb[0] as i32 - 128).abs() < 5);
    }

    #[test]
    fn white_pixel_10bit() {
        // Native 10-bit white is 1023 per channel.
        let yuv = rgb_to_yuv(
            &[1023u16, 1023, 1023],
            1,
            1,
            ChromaFormat::Yuv420,
            BitDepth::Ten,
        );
        assert!(
            yuv.y[0] > 1000,
            "10-bit white Y should approach 1023, got {}",
            yuv.y[0]
        );
        assert!(
            (yuv.cb[0] as i32 - 512).abs() < 20,
            "10-bit neutral chroma ~512, got {}",
            yuv.cb[0]
        );
    }

    #[test]
    fn black_pixel() {
        let yuv = rgb_to_yuv(&[0u16, 0, 0], 1, 1, ChromaFormat::Yuv420, BitDepth::Eight);
        assert!(yuv.y[0] < 5);
    }

    #[test]
    fn dimensions_monochrome() {
        let yuv = rgb_to_yuv(
            &vec![128u16; 4 * 4 * 3],
            4,
            4,
            ChromaFormat::Monochrome,
            BitDepth::Eight,
        );
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.cb.len(), 0);
    }

    #[test]
    fn dimensions_444() {
        let yuv = rgb_to_yuv(
            &vec![128u16; 4 * 4 * 3],
            4,
            4,
            ChromaFormat::Yuv444,
            BitDepth::Eight,
        );
        assert_eq!(yuv.cb.len(), 16);
    }

    #[test]
    fn dimensions_422() {
        let yuv = rgb_to_yuv(
            &vec![128u16; 4 * 4 * 3],
            4,
            4,
            ChromaFormat::Yuv422,
            BitDepth::Eight,
        );
        assert_eq!(yuv.cb.len(), 8);
    }
}
