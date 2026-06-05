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

/// Planar YCbCr image. Samples are u16 (valid range depends on bit depth).
///
/// `width`/`height` are the *coded/conformance* dimensions the planes are stored at
/// — rounded up to the chroma subsampling grid so chroma planes are well-defined.
/// `display_w`/`display_h` are the *true visible* dimensions (which may be odd); the
/// HEIF `ispe` box reports these so the decoder shows exactly the original size even
/// when the coded picture is one pixel larger. They equal `width`/`height` unless the
/// source had odd dimensions under a subsampled chroma format.
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
    ) -> Result<Self, crate::error::EncodeError> {
        let w = width as usize;
        let h = height as usize;
        if y.len() != w * h {
            return Err(crate::error::EncodeError::InvalidInput);
        }
        if chroma.is_monochrome() {
            if !cb.is_empty() || !cr.is_empty() {
                return Err(crate::error::EncodeError::InvalidInput);
            }
        } else {
            let cw = w.div_ceil(chroma.sub_w());
            let ch = h.div_ceil(chroma.sub_h());
            if cb.len() != cw * ch || cr.len() != cw * ch {
                return Err(crate::error::EncodeError::InvalidInput);
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

/// Convert planar RGB samples to planar YCbCr in the requested chroma format.
pub(crate) fn rgb_to_yuv(
    rgb: &[u16],
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Yuv {
    let w = width as usize;
    let h = height as usize;
    let maxv = bit_depth.max_val() as f32;
    let neutral = bit_depth.neutral() as f32;

    // Hoisted reciprocals — avoid repeated division in the hot loop.
    const REC_CB: f32 = 1.0 / 1.8556;
    const REC_CR: f32 = 1.0 / 1.5748;

    // ── Monochrome path ───────────────────────────────────────────────────
    if chroma.is_monochrome() {
        let channels = rgb.len() / (w * h);
        let y_plane: Vec<u16> = if channels == 1 {
            // 1-channel input: direct copy, no matrix.
            rgb.to_vec()
        } else {
            // Multi-channel input: derive luma via BT.709.
            rgb.chunks_exact(channels)
                .map(|px| {
                    let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
                    (0.2126 * r + 0.7152 * g + 0.0722 * b)
                        .round()
                        .clamp(0.0, maxv) as u16
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

    // Luma: one pass over all pixels, 3 samples at a time.
    let y_plane: Vec<u16> = rgb
        .chunks_exact(3)
        .map(|px| {
            let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
            (0.2126 * r + 0.7152 * g + 0.0722 * b)
                .round()
                .clamp(0.0, maxv) as u16
        })
        .collect();

    let sw = chroma.sub_w();
    let sh = chroma.sub_h();
    let cw = w.div_ceil(sw);
    let ch = h.div_ceil(sh);

    // Chroma: iterate over chroma block positions.
    // Each (crow, ccol) averages up to sw×sh luma-block pixels.
    let (cb_plane, cr_plane): (Vec<u16>, Vec<u16>) = (0..ch)
        .flat_map(|crow| (0..cw).map(move |ccol| (crow, ccol)))
        .map(|(crow, ccol)| {
            // Accumulate over the luma block covered by this chroma sample.
            let (sum_cb, sum_cr, count) = (0..sh)
                .flat_map(|dy| (0..sw).map(move |dx| (dy, dx)))
                .filter(|&(dy, dx)| {
                    let row = crow * sh + dy;
                    let col = ccol * sw + dx;
                    row < h && col < w
                })
                .fold((0.0f32, 0.0f32, 0u32), |(s_cb, s_cr, cnt), (dy, dx)| {
                    let row = crow * sh + dy;
                    let col = ccol * sw + dx;
                    let base = (row * w + col) * 3;
                    let (r, g, b) = (rgb[base] as f32, rgb[base + 1] as f32, rgb[base + 2] as f32);
                    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    (
                        s_cb + neutral + (b - y) * REC_CB,
                        s_cr + neutral + (r - y) * REC_CR,
                        cnt + 1,
                    )
                });

            if count > 0 {
                let rec_count = 1.0 / count as f32;
                let cb = (sum_cb * rec_count).round().clamp(0.0, maxv) as u16;
                let cr = (sum_cr * rec_count).round().clamp(0.0, maxv) as u16;
                (cb, cr)
            } else {
                (neutral as u16, neutral as u16)
            }
        })
        .unzip();

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
