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
#![deny(unreachable_pub)]
mod cabac;
mod color;
mod dct;
mod deblock;
mod error;
mod fmt;
mod hevc;
mod hevc_transform;
mod intra;
mod isobmff;
mod metadata;
mod yuv;

pub use color::{ColorEncoding, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
pub use error::EncodeError;
pub use fmt::{BitDepth, ChromaFormat};
pub use metadata::{ContentLightLevel, Metadata, Orientation};
pub use yuv::Yuv;

// ── constants ────────────────────────────────────────────────────────────────

/// Minimum accepted dimension (width or height) in pixels.
const MIN_DIM: u32 = 1;

/// Maximum accepted dimension. HEVC level 6.2 limits each axis to 16 384.
const MAX_DIM: u32 = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelLayout {
    /// Grayscale — 1 channel per pixel. Must be paired with
    /// [`ChromaFormat::Monochrome`] in the [`EncodeConfig`].
    Gray,
    /// Grayscale + alpha — 2 channels per pixel (Y, A). Must be paired with
    /// [`ChromaFormat::Monochrome`]. Alpha is ignored by [`encode`]; use
    /// [`encode_with_alpha`] to preserve it as a separate auxiliary image.
    GrayAlpha,
    /// Packed RGB — 3 channels per pixel (R, G, B).
    Rgb,
    /// Packed RGBA — 4 channels per pixel (R, G, B, A). Alpha is ignored by
    /// [`encode`]; use [`encode_with_alpha`] to preserve it.
    Rgba,
}

impl PixelLayout {
    pub fn channels(self) -> usize {
        match self {
            PixelLayout::Gray => 1,
            PixelLayout::GrayAlpha => 2,
            PixelLayout::Rgb => 3,
            PixelLayout::Rgba => 4,
        }
    }

    /// True when the layout has a dedicated alpha channel.
    pub fn has_alpha(self) -> bool {
        matches!(self, PixelLayout::GrayAlpha | PixelLayout::Rgba)
    }

    /// True when the layout is greyscale (no separate R/G/B channels).
    pub fn is_gray(self) -> bool {
        matches!(self, PixelLayout::Gray | PixelLayout::GrayAlpha)
    }
}

/// Encoder configuration. Build with [`EncodeConfig::new`] and the `with_*`
/// methods, then pass to [`encode`] / [`encode_with_alpha`] / [`encode_yuv`].
///
/// ```ignore
/// let cfg = EncodeConfig::new()
///     .with_quality(90)
///     .with_chroma(ChromaFormat::Yuv444)
///     .with_bit_depth(BitDepth::Ten)
///     .with_color(ColorMetadata::Cicp(ColorEncoding::bt2020_pq()));
/// let heic = encode(&rgb_u16, width, height, PixelLayout::Rgb, &cfg)?;
/// ```
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Visual quality, 1..=100 (higher = better quality, larger file).
    /// Maps to the HEVC QP via an internal table.
    pub quality: u8,
    /// Chroma subsampling format.
    pub chroma: ChromaFormat,
    /// Sample bit depth (8 / 10 / 12).
    pub bit_depth: BitDepth,
    /// Colour metadata written to the `colr` box and reflected in the VUI.
    /// Either enumerated CICP (`nclx`) or an embedded ICC profile (`prof`).
    pub color: ColorMetadata,
    /// Optional image metadata: orientation (`irot`/`imir`), HDR content
    /// light level (`clli`), and raw EXIF bytes. Empty by default.
    pub metadata: Metadata,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 90,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            color: ColorMetadata::default(), // sRGB ICC profile
            metadata: Metadata::default(),
        }
    }
}

impl EncodeConfig {
    /// A config with default settings (q = 90, 4:2:0, 8-bit, sRGB ICC).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the visual quality (1..=100).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Set the chroma subsampling format.
    pub fn with_chroma(mut self, chroma: ChromaFormat) -> Self {
        self.chroma = chroma;
        self
    }

    /// Set the sample bit depth.
    pub fn with_bit_depth(mut self, bit_depth: BitDepth) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set the colour metadata (CICP `nclx` or ICC `prof`).
    pub fn with_color(mut self, color: ColorMetadata) -> Self {
        self.color = color;
        self
    }

    /// Attach an ICC profile (shorthand for `with_color(ColorMetadata::Icc(..))`).
    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.color = ColorMetadata::Icc(icc);
        self
    }

    /// Use enumerated CICP signalling (shorthand for
    /// `with_color(ColorMetadata::Cicp(..))`).
    pub fn with_cicp(mut self, enc: ColorEncoding) -> Self {
        self.color = ColorMetadata::Cicp(enc);
        self
    }

    /// Set the full optional-metadata bundle (orientation, content light
    /// level, EXIF).
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the display orientation (`irot`/`imir`).
    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.metadata.orientation = o;
        self
    }

    /// Set the HDR content light level (`clli`).
    pub fn with_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.metadata.content_light_level = Some(cll);
        self
    }

    /// Attach raw EXIF/TIFF bytes (TIFF header onward, no `"Exif\0\0"` prefix).
    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.metadata.exif = Some(exif);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), EncodeError> {
        validate_quality(self.quality)?;
        Ok(())
    }
}

/// Encode a native-depth planar pixel buffer to HEIC using `cfg`.
///
/// `pixels` holds one `u16` per channel at `cfg.bit_depth`'s native range
/// (0..=255 / 0..=1023 / 0..=4095); the library does not rescale.
///
/// `layout` describes the channel order:
/// - [`PixelLayout::Rgb`]  — 3 channels per pixel (R, G, B).
/// - [`PixelLayout::Rgba`] — 4 channels per pixel; alpha is **discarded**.
///   Use [`encode_with_alpha`] to preserve the alpha plane.
/// - [`PixelLayout::Gray`] — 1 channel per pixel; `cfg.chroma` must be
///   [`ChromaFormat::Monochrome`].
///
/// The chroma format's subsampling grid may round the coded dimensions up by
/// one pixel; the conformance window in the bitstream crops back to the true
/// visible size so `ispe` reports `width × height` exactly.
pub fn encode(
    pixels: &[u16],
    width: u32,
    height: u32,
    layout: PixelLayout,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_pixel_buffer(pixels, width, height, layout)?;

    // Mono input requires a mono chroma config.
    if layout == PixelLayout::Gray && !cfg.chroma.is_monochrome() {
        return Err(EncodeError::InvalidInput);
    }
    if layout == PixelLayout::GrayAlpha && !cfg.chroma.is_monochrome() {
        return Err(EncodeError::InvalidInput);
    }

    // Strip alpha when the caller passes RGBA but only wants colour output.
    let rgb_owned: Vec<u16>;
    let rgb: &[u16] = match layout {
        PixelLayout::Rgb | PixelLayout::Gray => pixels,
        PixelLayout::Rgba => {
            rgb_owned = pixels
                .chunks_exact(4)
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            &rgb_owned
        }
        PixelLayout::GrayAlpha => {
            // Strip alpha; pass luma-only to rgb_to_yuv (Monochrome path).
            rgb_owned = pixels.chunks_exact(2).map(|px| px[0]).collect();
            &rgb_owned
        }
    };

    if layout.is_gray() && !cfg.chroma.is_monochrome() {
        return Err(EncodeError::InvalidInput);
    }

    let (enc_w, enc_h) = encoded_dims(width, height, cfg.chroma);
    let mut yuv = if enc_w != width || enc_h != height {
        let padded = pad_rgb_to_even(rgb, width, height, enc_w, enc_h);
        yuv::rgb_to_yuv(&padded, enc_w, enc_h, cfg.chroma, cfg.bit_depth)
    } else {
        yuv::rgb_to_yuv(rgb, width, height, cfg.chroma, cfg.bit_depth)
    };
    yuv = yuv.with_display(width, height);
    encode_yuv(&yuv, cfg)
}

/// Encode a native-depth RGBA pixel buffer to HEIC, preserving the alpha
/// channel as a separate monochrome auxiliary image per ISO/IEC 23008-12.
///
/// `layout` **must** be [`PixelLayout::Rgba`]; any other value is rejected
/// with [`EncodeError::InvalidInput`] because there is no alpha channel to
/// encode.
pub fn encode_with_alpha(
    pixels: &[u16],
    width: u32,
    height: u32,
    layout: PixelLayout,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_pixel_buffer(pixels, width, height, layout)?;

    if !layout.has_alpha() {
        return Err(EncodeError::InvalidInput);
    }

    let (enc_w, enc_h) = encoded_dims(width, height, cfg.chroma);
    let (w, h) = (width as usize, height as usize);
    let (nw, nh) = (enc_w as usize, enc_h as usize);
    let ch = layout.channels();

    // Alpha is always a monochrome plane — 1 sample per pixel.
    let mut alpha_plane = vec![0u16; nw * nh];

    let (_, color_stream) = match layout {
        PixelLayout::Rgba => {
            // Colour: 3-channel RGB buffer fed to rgb_to_yuv normally.
            let mut colour_buf = vec![0u16; nw * nh * 3];

            for (dst_row_idx, (col_row, alp_row)) in colour_buf
                .chunks_exact_mut(nw * 3)
                .zip(alpha_plane.chunks_exact_mut(nw))
                .enumerate()
            {
                let sr = dst_row_idx.min(h - 1);
                let src_row = &pixels[sr * w * ch..(sr * w + w) * ch];

                for (dst_col_idx, (col_px, alp_px)) in col_row
                    .chunks_exact_mut(3)
                    .zip(alp_row.iter_mut())
                    .enumerate()
                {
                    let sc = dst_col_idx.min(w - 1);
                    let src = &src_row[sc * 4..sc * 4 + 4];
                    col_px.copy_from_slice(&src[..3]);
                    *alp_px = src[3];
                }
            }

            let yuv = yuv::rgb_to_yuv(&colour_buf, enc_w, enc_h, cfg.chroma, cfg.bit_depth);
            let stream = hevc::encode_intra(&yuv, enc_w, enc_h, cfg.quality)?;
            (yuv, stream)
        }

        PixelLayout::GrayAlpha => {
            // Colour: 1-channel luma plane; build Yuv directly, no rgb_to_yuv needed.
            let mut luma = vec![0u16; nw * nh];

            for (dst_row_idx, (luma_row, alp_row)) in luma
                .chunks_exact_mut(nw)
                .zip(alpha_plane.chunks_exact_mut(nw))
                .enumerate()
            {
                let sr = dst_row_idx.min(h - 1);
                let src_row = &pixels[sr * w * 2..(sr * w + w) * 2];

                for (dst_col_idx, (luma_px, alp_px)) in
                    luma_row.iter_mut().zip(alp_row.iter_mut()).enumerate()
                {
                    let sc = dst_col_idx.min(w - 1);
                    *luma_px = src_row[sc * 2];
                    *alp_px = src_row[sc * 2 + 1];
                }
            }

            let yuv = Yuv {
                y: luma,
                cb: Vec::new(),
                cr: Vec::new(),
                width: enc_w,
                height: enc_h,
                display_w: width,
                display_h: height,
                chroma: ChromaFormat::Monochrome,
                bit_depth: cfg.bit_depth,
            };
            let stream = hevc::encode_intra(&yuv, enc_w, enc_h, cfg.quality)?;
            (yuv, stream)
        }

        _ => unreachable!(), // has_alpha() guard above
    };

    // Alpha is always monochrome — build its Yuv directly from the 1-channel plane.
    let alpha_yuv = Yuv {
        y: alpha_plane,
        cb: Vec::new(),
        cr: Vec::new(),
        width: enc_w,
        height: enc_h,
        display_w: width,
        display_h: height,
        chroma: ChromaFormat::Monochrome,
        bit_depth: cfg.bit_depth,
    };
    let alpha_stream = hevc::encode_intra(&alpha_yuv, enc_w, enc_h, cfg.quality)?;

    isobmff::wrap_hevc_image_with_alpha(
        &color_stream,
        &alpha_stream,
        width,
        height,
        cfg.bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode pre-converted planar YCbCr directly, skipping the RGB→YCbCr step.
///
/// The [`Yuv`] carries its own chroma format and bit depth; `cfg.chroma` and
/// `cfg.bit_depth` are **ignored** in favour of the planes' values (they must
/// be self-consistent). Useful when the caller already holds YCbCr data (e.g.
/// from a decoder or camera pipeline).
///
/// Plane dimensions must already satisfy the chroma subsampling grid; use
/// [`yuv::rgb_to_yuv`] or [`Yuv::from_planes`] to produce a conformant
/// [`Yuv`]. The visible `width`/`height` come from the [`Yuv`] itself.
pub fn encode_yuv(yuv: &Yuv, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    yuv.validate()?;
    cfg.validate()?;

    let enc_w = yuv.width;
    let enc_h = yuv.height;
    let nalu_stream = hevc::encode_intra(yuv, enc_w, enc_h, cfg.quality)?;

    // `ispe` carries the true visible size (may be odd); the SPS conformance
    // window crops the coded picture to the chroma-even plane size.
    isobmff::wrap_hevc_image(
        &nalu_stream,
        yuv.display_w,
        yuv.display_h,
        yuv.bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

// ── convenience wrappers ─────────────────────────────────────────────────────

/// Encode a packed 8-bit RGB image to HEIC (4:2:0, 8-bit, sRGB ICC).
///
/// `rgb` must hold exactly `width * height * 3` bytes in R, G, B order.
pub fn encode_heic(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, EncodeError> {
    encode_heic_fmt(rgb, width, height, quality, ChromaFormat::Yuv420)
}

/// Encode a packed 8-bit RGB image to HEIC with an explicit chroma format.
///
/// `rgb` must hold exactly `width * height * 3` bytes in R, G, B order.
pub fn encode_heic_fmt(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_quality(quality)?;
    validate_pixel_buffer_u8(rgb, width, height, PixelLayout::Rgb)?;

    let wide: Vec<u16> = rgb.iter().map(|&b| b as u16).collect();
    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(chroma)
        .with_bit_depth(BitDepth::Eight);
    encode(&wide, width, height, PixelLayout::Rgb, &cfg)
}

/// Encode a native-depth `u16` RGB image to HEIC with explicit chroma and
/// bit depth.
///
/// `rgb` must hold exactly `width * height * 3` samples in R, G, B order at
/// `bit_depth`'s native range.
pub fn encode_heic_fmt_bd(
    rgb: &[u16],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_quality(quality)?;
    validate_pixel_buffer(rgb, width, height, PixelLayout::Rgb)?;

    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(chroma)
        .with_bit_depth(bit_depth);
    encode(rgb, width, height, PixelLayout::Rgb, &cfg)
}

/// Encode a packed 8-bit grayscale + alpha image to HEIC with a separate
/// alpha auxiliary image. `ya` must hold exactly `width * height * 2` bytes
/// in Y, A order.
pub fn encode_heic_gray_alpha(
    ya: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_quality(quality)?;
    validate_pixel_buffer_u8(ya, width, height, PixelLayout::GrayAlpha)?;

    let wide: Vec<u16> = ya.iter().map(|&b| b as u16).collect();
    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(ChromaFormat::Monochrome);
    encode_with_alpha(&wide, width, height, PixelLayout::GrayAlpha, &cfg)
}

/// Encode a native-depth `u16` grayscale + alpha image to HEIC at an
/// explicit bit depth. `ya` must hold exactly `width * height * 2` samples
/// in Y, A order at `bit_depth`'s native range.
pub fn encode_heic_gray_alpha_bd(
    ya: &[u16],
    width: u32,
    height: u32,
    quality: u8,
    bit_depth: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_quality(quality)?;
    validate_pixel_buffer(ya, width, height, PixelLayout::GrayAlpha)?;

    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(ChromaFormat::Monochrome)
        .with_bit_depth(bit_depth);
    encode_with_alpha(ya, width, height, PixelLayout::GrayAlpha, &cfg)
}

/// Encode a packed 8-bit RGBA image to HEIC with a separate alpha auxiliary
/// image (4:2:0, 8-bit).
///
/// `rgba` must hold exactly `width * height * 4` bytes in R, G, B, A order.
pub fn encode_heic_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_quality(quality)?;
    validate_pixel_buffer_u8(rgba, width, height, PixelLayout::Rgba)?;

    let wide: Vec<u16> = rgba.iter().map(|&b| b as u16).collect();
    encode_heic_with_alpha_bd(&wide, width, height, quality, chroma, BitDepth::Eight)
}

/// Encode a native-depth `u16` RGBA image to HEIC with a separate alpha
/// auxiliary image at an explicit bit depth.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order
/// at `bit_depth`'s native range.
pub fn encode_heic_with_alpha_bd(
    rgba: &[u16],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    validate_quality(quality)?;
    validate_pixel_buffer(rgba, width, height, PixelLayout::Rgba)?;

    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(chroma)
        .with_bit_depth(bit_depth);
    encode_with_alpha(rgba, width, height, PixelLayout::Rgba, &cfg)
}

// ── internal helpers ─────────────────────────────────────────────────────────

/// Validate that `width` and `height` are within the accepted range.
pub(crate) fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM || width > MAX_DIM || height > MAX_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

/// Validate that `quality` is in 1..=100.
fn validate_quality(quality: u8) -> Result<(), EncodeError> {
    if quality == 0 || quality > 100 {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

/// Check that `buf` is large enough for `width × height` pixels at `layout`.
fn validate_pixel_buffer(
    buf: &[u16],
    width: u32,
    height: u32,
    layout: PixelLayout,
) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<u16>(width as usize, height as usize, layout.channels())?;
    if buf.len() != needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

/// Same check for packed `u8` buffers (8-bit convenience wrappers).
fn validate_pixel_buffer_u8(
    buf: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<u8>(width as usize, height as usize, layout.channels())?;
    if buf.len() < needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

/// Compute the required buffer length for `width × height × channels` samples
/// of type `T`, checking for overflow at every multiplication.
///
/// Also verifies that the total byte size fits in an `isize` (the Rust
/// allocation limit), catching pathological dimension combinations before any
/// allocation attempt.
pub(crate) fn checked_buffer_size<T>(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<usize, EncodeError> {
    let pixel_size = size_of::<T>();

    // Verify the byte total fits in isize (Rust allocation limit).
    let byte_total = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(channels))
        .and_then(|v| v.checked_mul(pixel_size))
        .and_then(|v| isize::try_from(v).ok())
        .ok_or(EncodeError::DimensionTooLarge { width, height })?;

    // Return the element count (not the byte count).
    width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(channels))
        .filter(|_| byte_total >= 0) // always true here, but keeps the type sound
        .ok_or(EncodeError::DimensionTooLarge { width, height })
}

/// Round visible dimensions up to the chroma subsampling grid.
fn encoded_dims(width: u32, height: u32, chroma: ChromaFormat) -> (u32, u32) {
    let sw = chroma.sub_w() as u32;
    let sh = chroma.sub_h() as u32;
    (width.div_ceil(sw) * sw, height.div_ceil(sh) * sh)
}

/// Replicate-pad planar RGB/mono from `(w, h)` to `(nw, nh)` by repeating
/// the last row / column. `nw >= w` and `nh >= h` must hold.
///
/// The output is a flat `Vec<u16>` with the same channel count as the input
/// (`input.len() / (w * h)` channels per pixel).
fn pad_rgb_to_even(rgb: &[u16], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u16> {
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    let channels = rgb.len() / (w * h); // 1, 3, or 4
    let row_stride = w * channels;
    let dst_row_stride = nw * channels;

    let mut out = vec![0u16; nw * nh * channels];

    for (dst_row_idx, dst_row) in out.chunks_exact_mut(dst_row_stride).enumerate() {
        let src_row_idx = dst_row_idx.min(h - 1);
        let src_row = &rgb[src_row_idx * row_stride..(src_row_idx + 1) * row_stride];

        let (dst_real, dst_pad) = dst_row.split_at_mut(w * channels);

        // Bulk copy the real columns.
        dst_real.copy_from_slice(src_row);

        // Replicate the last pixel into any padding columns.
        if !dst_pad.is_empty() {
            let last_px = &src_row[src_row.len() - channels..];
            for px in dst_pad.chunks_exact_mut(channels) {
                px.copy_from_slice(last_px);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_chains() {
        let cfg = EncodeConfig::new()
            .with_quality(80)
            .with_chroma(ChromaFormat::Yuv444)
            .with_bit_depth(BitDepth::Ten)
            .with_cicp(ColorEncoding::bt709());
        assert_eq!(cfg.quality, 80);
        assert_eq!(cfg.chroma, ChromaFormat::Yuv444);
        assert_eq!(cfg.bit_depth, BitDepth::Ten);
        assert!(matches!(cfg.color, ColorMetadata::Cicp(_)));
    }

    #[test]
    fn config_validate_rejects_quality_zero() {
        let cfg = EncodeConfig::new().with_quality(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_quality_over_100() {
        let cfg = EncodeConfig::new().with_quality(101);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn pixel_layout_channels() {
        assert_eq!(PixelLayout::Gray.channels(), 1);
        assert_eq!(PixelLayout::Rgb.channels(), 3);
        assert_eq!(PixelLayout::Rgba.channels(), 4);
    }

    #[test]
    fn validate_dims_rejects_zero_width() {
        assert!(validate_dims(0, 1).is_err());
    }

    #[test]
    fn validate_dims_rejects_zero_height() {
        assert!(validate_dims(1, 0).is_err());
    }

    #[test]
    fn validate_dims_rejects_oversized() {
        assert!(validate_dims(MAX_DIM + 1, 1).is_err());
        assert!(validate_dims(1, MAX_DIM + 1).is_err());
    }

    #[test]
    fn validate_dims_accepts_boundary() {
        assert!(validate_dims(MIN_DIM, MIN_DIM).is_ok());
        assert!(validate_dims(MAX_DIM, MAX_DIM).is_ok());
    }

    // ── validate_pixel_buffer ────────────────────────────────────────────

    #[test]
    fn pixel_buffer_rejects_short_rgb() {
        // 4×4 RGB needs 48 samples; 47 must be rejected.
        assert!(validate_pixel_buffer(&vec![0u16; 47], 4, 4, PixelLayout::Rgb).is_err());
    }

    #[test]
    fn pixel_buffer_accepts_exact_rgb() {
        assert!(validate_pixel_buffer(&vec![0u16; 48], 4, 4, PixelLayout::Rgb).is_ok());
    }

    #[test]
    fn pixel_buffer_rejects_short_rgba() {
        assert!(validate_pixel_buffer(&vec![0u16; 63], 4, 4, PixelLayout::Rgba).is_err());
    }

    #[test]
    fn from_planes_validates_sizes() {
        // 4×4, 4:2:0: luma 16, chroma 2×2 = 4 each.
        let ok = Yuv::from_planes(
            vec![0u16; 16],
            vec![0u16; 4],
            vec![0u16; 4],
            4,
            4,
            ChromaFormat::Yuv420,
            BitDepth::Eight,
        );
        assert!(ok.is_ok());

        // Wrong luma length.
        let bad = Yuv::from_planes(
            vec![0u16; 15],
            vec![0u16; 4],
            vec![0u16; 4],
            4,
            4,
            ChromaFormat::Yuv420,
            BitDepth::Eight,
        );
        assert!(bad.is_err());

        // Monochrome must have empty chroma.
        let mono_bad = Yuv::from_planes(
            vec![0u16; 16],
            vec![0u16; 4],
            vec![],
            4,
            4,
            ChromaFormat::Monochrome,
            BitDepth::Eight,
        );
        assert!(mono_bad.is_err());
    }

    // ── encode_yuv ────────────────────────────────────────────────────────

    #[test]
    fn encode_yuv_roundtrips() {
        let rgb = vec![128u16; 16 * 16 * 3];
        let yuv = yuv::rgb_to_yuv(&rgb, 16, 16, ChromaFormat::Yuv420, BitDepth::Eight);
        let cfg = EncodeConfig::new();
        let out = encode_yuv(&yuv, &cfg).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    // ── encode (RGB / RGBA / Mono) ────────────────────────────────────────

    #[test]
    fn encode_rgb_even_dims() {
        let rgb = vec![100u16; 16 * 16 * 3];
        let cfg = EncodeConfig::new();
        let out = encode(&rgb, 16, 16, PixelLayout::Rgb, &cfg).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_rgba_strips_alpha() {
        // RGBA with alpha = 0 should still produce a valid colour HEIC.
        let rgba = vec![100u16; 16 * 16 * 4];
        let cfg = EncodeConfig::new();
        let out = encode(&rgba, 16, 16, PixelLayout::Rgba, &cfg).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_mono_requires_monochrome_config() {
        let mono = vec![128u16; 8 * 8];

        let cfg_ok = EncodeConfig::new().with_chroma(ChromaFormat::Monochrome);
        let out = encode(&mono, 8, 8, PixelLayout::Gray, &cfg_ok).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_rejects_short_buffer() {
        let rgb = vec![0u16; 10]; // far too short for 16×16
        let cfg = EncodeConfig::new();
        assert!(encode(&rgb, 16, 16, PixelLayout::Rgb, &cfg).is_err());
    }

    // ── encode_with_alpha ─────────────────────────────────────────────────

    #[test]
    fn encode_with_alpha_rejects_rgb_layout() {
        let rgb = vec![128u16; 16 * 16 * 3];
        let cfg = EncodeConfig::new();
        assert!(encode_with_alpha(&rgb, 16, 16, PixelLayout::Rgb, &cfg).is_err());
    }

    #[test]
    fn encode_with_alpha_accepts_rgba() {
        let rgba = vec![200u16; 16 * 16 * 4];
        let cfg = EncodeConfig::new();
        let out = encode_with_alpha(&rgba, 16, 16, PixelLayout::Rgba, &cfg).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    // ── odd dimensions / ispe ─────────────────────────────────────────────

    #[test]
    fn odd_dimensions_reported_in_ispe() {
        // 281×181 under 4:2:0: planes round to 282×182, but ispe must report
        // the true odd size.
        let rgb = vec![100u16; 281 * 181 * 3];
        let cfg = EncodeConfig::new().with_chroma(ChromaFormat::Yuv420);
        let out = encode(&rgb, 281, 181, PixelLayout::Rgb, &cfg).unwrap();

        let ispe = out
            .windows(4)
            .position(|w| w == b"ispe")
            .expect("ispe present");
        // ispe payload: fullbox header (4) then width (4) height (4).
        let wpos = ispe + 4 + 4;
        let w = u32::from_be_bytes(out[wpos..wpos + 4].try_into().unwrap());
        let h = u32::from_be_bytes(out[wpos + 4..wpos + 8].try_into().unwrap());
        assert_eq!((w, h), (281, 181), "ispe must report the true odd size");
    }

    // ── checked_buffer_size ───────────────────────────────────────────────

    #[test]
    fn checked_buffer_size_normal() {
        assert_eq!(checked_buffer_size::<u16>(4, 4, 3).unwrap(), 48);
    }

    #[test]
    fn checked_buffer_size_overflow() {
        // usize::MAX * usize::MAX overflows.
        assert!(checked_buffer_size::<u16>(usize::MAX, usize::MAX, 3).is_err());
    }

    // ── pad_rgb_to_even ───────────────────────────────────────────────────

    #[test]
    fn pad_rgb_replicates_last_column() {
        // 1×1 → 2×1: the single pixel should be replicated to the right.
        let rgb = vec![10u16, 20, 30]; // one RGB pixel
        let out = pad_rgb_to_even(&rgb, 1, 1, 2, 1);
        assert_eq!(out, vec![10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn pad_rgb_replicates_last_row() {
        // 1×1 → 1×2: the single pixel should be replicated downward.
        let rgb = vec![10u16, 20, 30];
        let out = pad_rgb_to_even(&rgb, 1, 1, 1, 2);
        assert_eq!(out, vec![10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn pad_rgb_noop_when_already_even() {
        let rgb: Vec<u16> = (0..12).collect(); // 2×2 RGB
        let out = pad_rgb_to_even(&rgb, 2, 2, 2, 2);
        assert_eq!(out, rgb);
    }

    #[test]
    fn gray_alpha_layout_channels() {
        assert_eq!(PixelLayout::GrayAlpha.channels(), 2);
        assert!(PixelLayout::GrayAlpha.has_alpha());
        assert!(PixelLayout::GrayAlpha.is_gray());
        assert!(!PixelLayout::Rgb.is_gray());
        assert!(!PixelLayout::Rgba.is_gray());
    }

    #[test]
    fn encode_gray_alpha_rejects_non_mono_config() {
        let ya = vec![128u16; 8 * 8 * 2];
        let cfg = EncodeConfig::new().with_chroma(ChromaFormat::Yuv420);
        // encode (strip-alpha path) must reject non-Monochrome for gray layouts.
        assert!(encode(&ya, 8, 8, PixelLayout::GrayAlpha, &cfg).is_err());
    }

    #[test]
    fn encode_gray_alpha_strips_alpha() {
        let ya = vec![200u16; 8 * 8 * 2];
        let cfg = EncodeConfig::new().with_chroma(ChromaFormat::Monochrome);
        let out = encode(&ya, 8, 8, PixelLayout::GrayAlpha, &cfg).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_with_alpha_gray_alpha_roundtrip() {
        let ya = vec![180u16; 16 * 16 * 2];
        let cfg = EncodeConfig::new().with_chroma(ChromaFormat::Monochrome);
        let out = encode_with_alpha(&ya, 16, 16, PixelLayout::GrayAlpha, &cfg).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_with_alpha_rejects_mono_layout() {
        let mono = vec![128u16; 8 * 8];
        let cfg = EncodeConfig::new().with_chroma(ChromaFormat::Monochrome);
        assert!(encode_with_alpha(&mono, 8, 8, PixelLayout::Gray, &cfg).is_err());
    }
}
