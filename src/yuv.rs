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

/// RGB→YCbCr coefficients in Q0.16 for the matrix and range the stream
/// signals, so decoders invert exactly the transform that was applied.
///
/// The luma weights sum exactly to the range scale (`g` absorbs the rounding)
/// and each chroma row sums exactly to 0, so greys stay neutral and white maps
/// to the top code value. Chroma is computed directly from RGB, never from the
/// rounded luma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct YcbcrMatrix {
    y: [i64; 3],
    cb: [i64; 3],
    cr: [i64; 3],
    /// Luma black level in code values (16·2^(bitDepth−8) for limited range).
    y_offset: i64,
}

impl YcbcrMatrix {
    /// Coefficients for `cicp` (BT.601 when absent, unspecified or not a
    /// Y'CbCr matrix this converter implements) at `bit_depth`.
    pub(crate) fn new(cicp: Option<crate::color::Cicp>, bit_depth: BitDepth) -> Self {
        use crate::color::MatrixCoefficients as M;
        let (kr, kb) = match cicp.map(|c| c.matrix) {
            Some(M::Bt709) => (0.2126, 0.0722),
            Some(M::Fcc) => (0.30, 0.11),
            Some(M::Smpte240m) => (0.212, 0.087),
            Some(M::Bt2020Ncl | M::Bt2020Cl) => (0.2627, 0.0593),
            _ => (0.299, 0.114),
        };
        let full_range = cicp.is_none_or(|c| c.full_range);
        let depth_scale = f64::from(1u32 << (bit_depth.bits() - 8));
        let max = f64::from(bit_depth.max_val());
        let (luma_scale, chroma_scale, y_offset) = if full_range {
            (1.0, 1.0, 0)
        } else {
            (
                219.0 * depth_scale / max,
                224.0 * depth_scale / max,
                16 << (bit_depth.bits() - 8),
            )
        };
        let q = |v: f64| (v * 65_536.0).round() as i64;
        let (yr, yb) = (q(kr * luma_scale), q(kb * luma_scale));
        let half = q(0.5 * chroma_scale);
        let cb_r = q(-0.5 * kr / (1.0 - kb) * chroma_scale);
        let cr_b = q(-0.5 * kb / (1.0 - kr) * chroma_scale);
        Self {
            y: [yr, q(luma_scale) - yr - yb, yb],
            cb: [cb_r, -half - cb_r, half],
            cr: [half, -half - cr_b, cr_b],
            y_offset,
        }
    }

    /// Luma code value (rounded, unclamped).
    #[inline(always)]
    fn luma(&self, r: i32, g: i32, b: i32) -> i32 {
        let [kr, kg, kb] = self.y;
        ((kr * r as i64 + kg * g as i64 + kb * b as i64 + (1 << 15)) >> 16) as i32
            + self.y_offset as i32
    }

    /// Unscaled Q16 chroma differences (Cb, Cr) of one pixel, before the offset.
    #[inline(always)]
    fn chroma_q16(&self, r: i32, g: i32, b: i32) -> (i64, i64) {
        let (r, g, b) = (r as i64, g as i64, b as i64);
        (
            self.cb[0] * r + self.cb[1] * g + self.cb[2] * b,
            self.cr[0] * r + self.cr[1] * g + self.cr[2] * b,
        )
    }
}

/// Final chroma sample from the Q16 sum over `1 << log2_count` pixels, rounded
/// once.
#[inline(always)]
fn chroma_from_sum(sum: i64, log2_count: u32, neutral: i32, maxv: i32) -> u16 {
    let shift = 16 + log2_count;
    let value = (sum + (1i64 << (shift - 1))) >> shift;
    (value as i32 + neutral).clamp(0, maxv) as u16
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
    matrix: &YcbcrMatrix,
) -> Yuv {
    rgb_to_yuv_into(
        rgb,
        width,
        height,
        chroma,
        bit_depth,
        matrix,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

/// [`rgb_to_yuv`] into donated plane buffers, reusing their retained capacity.
/// The buffers come back to the caller inside the returned [`Yuv`], so a
/// leased workspace can reclaim them after encoding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rgb_to_yuv_into(
    rgb: &[u16],
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    matrix: &YcbcrMatrix,
    mut y_plane: Vec<u16>,
    mut cb_plane: Vec<u16>,
    mut cr_plane: Vec<u16>,
) -> Yuv {
    let w = width as usize;
    let h = height as usize;
    let maxv = bit_depth.max_val() as i32;
    let neutral = bit_depth.neutral() as i32;

    if chroma.is_monochrome() {
        let channels = rgb.len() / (w * h);
        y_plane.clear();
        if channels == 1 {
            y_plane.extend_from_slice(rgb);
        } else if channels == 4 {
            y_plane.extend(rgb.as_chunks::<4>().0.iter().map(|px| {
                let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                matrix.luma(r, g, b).clamp(0, maxv) as u16
            }));
        } else if channels == 3 {
            y_plane.extend(rgb.as_chunks::<3>().0.iter().map(|px| {
                let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                matrix.luma(r, g, b).clamp(0, maxv) as u16
            }));
        } else {
            unimplemented!(
                "Amount of channels {} in 'rgb_to_yuv' is not supported",
                channels
            )
        }
        cb_plane.clear();
        cr_plane.clear();
        return Yuv {
            y: y_plane,
            cb: cb_plane,
            cr: cr_plane,
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

    y_plane.clear();
    y_plane.resize(w * h, 0u16);
    cb_plane.clear();
    cb_plane.resize(cw * ch, 0u16);
    cr_plane.clear();
    cr_plane.resize(cw * ch, 0u16);

    for (y_row, src) in y_plane.chunks_exact_mut(w).zip(rgb.chunks_exact(w * 3)) {
        for (y_out, px) in y_row.iter_mut().zip(src.as_chunks::<3>().0) {
            *y_out = matrix
                .luma(px[0] as i32, px[1] as i32, px[2] as i32)
                .clamp(0, maxv) as u16;
        }
    }

    // Each chroma sample is the box average of its sw×sh luma block (fewer at
    // odd right/bottom edges — always 1, 2 or 4 pixels), rounded once.
    let mut cb_acc = vec![0i64; cw];
    let mut cr_acc = vec![0i64; cw];
    for chroma_row in 0..ch {
        cb_acc.fill(0);
        cr_acc.fill(0);
        let row0 = chroma_row * sh;
        let rows = (h - row0).min(sh);
        for src in rgb[row0 * w * 3..(row0 + rows) * w * 3].chunks_exact(w * 3) {
            for (col, px) in src.as_chunks::<3>().0.iter().enumerate() {
                let (cb, cr) = matrix.chroma_q16(px[0] as i32, px[1] as i32, px[2] as i32);
                cb_acc[col / sw] += cb;
                cr_acc[col / sw] += cr;
            }
        }
        let cb_row = &mut cb_plane[chroma_row * cw..(chroma_row + 1) * cw];
        let cr_row = &mut cr_plane[chroma_row * cw..(chroma_row + 1) * cw];
        for (chroma_col, ((cb_out, cr_out), (&cb_sum, &cr_sum))) in cb_row
            .iter_mut()
            .zip(cr_row.iter_mut())
            .zip(cb_acc.iter().zip(&cr_acc))
            .enumerate()
        {
            let cols = (w - chroma_col * sw).min(sw);
            let log2_count = (cols * rows).trailing_zeros();
            *cb_out = chroma_from_sum(cb_sum, log2_count, neutral, maxv);
            *cr_out = chroma_from_sum(cr_sum, log2_count, neutral, maxv);
        }
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
            &YcbcrMatrix::new(None, BitDepth::Eight),
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
            &YcbcrMatrix::new(None, BitDepth::Ten),
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
        let yuv = rgb_to_yuv(
            &[0u16, 0, 0],
            1,
            1,
            ChromaFormat::Yuv420,
            BitDepth::Eight,
            &YcbcrMatrix::new(None, BitDepth::Eight),
        );
        assert!(yuv.y[0] < 5);
    }

    #[test]
    fn dimensions_monochrome() {
        let yuv = rgb_to_yuv(
            &[128u16; 4 * 4 * 3],
            4,
            4,
            ChromaFormat::Monochrome,
            BitDepth::Eight,
            &YcbcrMatrix::new(None, BitDepth::Eight),
        );
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.cb.len(), 0);
    }

    #[test]
    fn dimensions_444() {
        let yuv = rgb_to_yuv(
            &[128u16; 4 * 4 * 3],
            4,
            4,
            ChromaFormat::Yuv444,
            BitDepth::Eight,
            &YcbcrMatrix::new(None, BitDepth::Eight),
        );
        assert_eq!(yuv.cb.len(), 16);
    }

    #[test]
    fn dimensions_422() {
        let yuv = rgb_to_yuv(
            &[128u16; 4 * 4 * 3],
            4,
            4,
            ChromaFormat::Yuv422,
            BitDepth::Eight,
            &YcbcrMatrix::new(None, BitDepth::Eight),
        );
        assert_eq!(yuv.cb.len(), 8);
    }

    #[test]
    fn white_and_greys_are_exact() {
        for v in [0u16, 1, 77, 128, 200, 254, 255] {
            let yuv = rgb_to_yuv(
                &[v, v, v],
                1,
                1,
                ChromaFormat::Yuv444,
                BitDepth::Eight,
                &YcbcrMatrix::new(None, BitDepth::Eight),
            );
            assert_eq!((yuv.y[0], yuv.cb[0], yuv.cr[0]), (v, 128, 128), "grey {v}");
        }
        let yuv = rgb_to_yuv(
            &[1023, 1023, 1023],
            1,
            1,
            ChromaFormat::Yuv444,
            BitDepth::Ten,
            &YcbcrMatrix::new(None, BitDepth::Ten),
        );
        assert_eq!((yuv.y[0], yuv.cb[0], yuv.cr[0]), (1023, 512, 512));
    }

    #[test]
    fn subsampled_chroma_is_the_once_rounded_box_mean() {
        // A saturated 2×2 block: the four per-pixel chroma values straddle
        // rounding boundaries, so cascaded per-pixel/pairwise rounding drifts.
        let px: [[u16; 3]; 4] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let rgb: Vec<u16> = px.iter().flatten().copied().collect();
        let yuv = rgb_to_yuv(
            &rgb,
            2,
            2,
            ChromaFormat::Yuv420,
            BitDepth::Eight,
            &YcbcrMatrix::new(None, BitDepth::Eight),
        );
        let mean = |f: fn(f64, f64, f64) -> f64| {
            px.iter()
                .map(|p| f(p[0] as f64, p[1] as f64, p[2] as f64))
                .sum::<f64>()
                / 4.0
        };
        let cb = 128.0 + mean(|r, g, b| -0.168_736 * r - 0.331_264 * g + 0.5 * b);
        let cr = 128.0 + mean(|r, g, b| 0.5 * r - 0.418_688 * g - 0.081_312 * b);
        assert_eq!(yuv.cb[0], cb.round() as u16);
        assert_eq!(yuv.cr[0], cr.round() as u16);
    }
}
