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
use std::mem::size_of;
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

const MIN_DIM: u32 = 1;
const MAX_DIM: u32 = 16_384;

/// Images wider or taller than this are encoded as a HEIF grid of 512×512
/// tiles. The value matches Apple's tile size for compatibility.
const TILE_SIZE: u32 = 512;

// ── EncodeConfig ──────────────────────────────────────────────────────────────

/// Encoder configuration shared by all entry points.
///
/// Build with [`EncodeConfig::new`] and the `with_*` builder methods.
///
/// ```ignore
/// let cfg = EncodeConfig::new()
///     .with_quality(90)
///     .with_chroma(ChromaFormat::Yuv444)
///     .with_color(ColorMetadata::Cicp(ColorEncoding::bt2020_pq()));
/// ```
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Visual quality 1..=100 (higher = better, larger file). Maps to HEVC QP.
    pub quality: u8,
    /// Chroma subsampling format. Ignored by the `gray*` entry points, which
    /// always use [`ChromaFormat::Monochrome`].
    pub chroma: ChromaFormat,
    /// Color metadata written to the `colr` box / VUI.
    pub color: ColorMetadata,
    /// Optional image metadata (orientation, HDR light level, EXIF).
    pub metadata: Metadata,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 90,
            chroma: ChromaFormat::Yuv420,
            color: ColorMetadata::default(), // sRGB ICC profile
            metadata: Metadata::default(),
        }
    }
}

impl EncodeConfig {
    /// Default settings: q = 90, 4:2:0, sRGB ICC.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    pub fn with_chroma(mut self, chroma: ChromaFormat) -> Self {
        self.chroma = chroma;
        self
    }

    pub fn with_color(mut self, color: ColorMetadata) -> Self {
        self.color = color;
        self
    }

    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.color = ColorMetadata::Icc(icc);
        self
    }

    pub fn with_cicp(mut self, enc: ColorEncoding) -> Self {
        self.color = ColorMetadata::Cicp(enc);
        self
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.metadata.orientation = o;
        self
    }

    pub fn with_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.metadata.content_light_level = Some(cll);
        self
    }

    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.metadata.exif = Some(exif);
        self
    }

    fn validate(&self) -> Result<(), EncodeError> {
        validate_quality(self.quality)
    }
}

/// Encode a packed 8-bit RGB image to HEIC.
///
/// `rgb` must hold exactly `width * height * 3` bytes in R, G, B order.
pub fn encode_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgb, width, height, 3)?;
    let wide: Vec<u16> = rgb.iter().map(|&b| b as u16).collect();
    encode_rgb_wide(&wide, width, height, BitDepth::Eight, cfg)
}

/// Encode a packed 8-bit RGBA image to HEIC. Alpha is **discarded**.
/// Use [`encode_rgba_with_alpha`] to preserve it.
///
/// `rgba` must hold exactly `width * height * 4` bytes in R, G, B, A order.
pub fn encode_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgba, width, height, 4)?;
    let wide: Vec<u16> = rgba
        .chunks_exact(4)
        .flat_map(|px| [px[0] as u16, px[1] as u16, px[2] as u16])
        .collect();
    encode_rgb_wide(&wide, width, height, BitDepth::Eight, cfg)
}

/// Encode a packed 8-bit RGBA image to HEIC, writing alpha as a separate
/// monochrome auxiliary image per ISO/IEC 23008-12.
///
/// `rgba` must hold exactly `width * height * 4` bytes in R, G, B, A order.
///
/// # Large images
/// Images larger than 512×512 are not tiled when alpha is present. If you need
/// tiled output with alpha, pre-tile and call this function per tile, or use
/// [`encode_yuv`] on pre-converted YCbCr data.
pub fn encode_rgba_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgba, width, height, 4)?;
    let wide: Vec<u16> = rgba.iter().map(|&b| b as u16).collect();
    encode_rgba_with_alpha_wide(&wide, width, height, BitDepth::Eight, cfg)
}

/// Encode a 10-bit RGB image to HEIC.
///
/// `rgb` must hold exactly `width * height * 3` samples, each in `0..=1023`,
/// packed as `u16`.
pub fn encode_rgb10(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgb, width, height, 3)?;
    encode_rgb_wide(rgb, width, height, BitDepth::Ten, cfg)
}

/// Encode a 10-bit RGBA image to HEIC. Alpha is **discarded**.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=1023`, packed as `u16`.
pub fn encode_rgba10(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    let rgb: Vec<u16> = rgba
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    encode_rgb_wide(&rgb, width, height, BitDepth::Ten, cfg)
}

/// Encode a 10-bit RGBA image to HEIC with a separate alpha auxiliary image.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=1023`, packed as `u16`.
pub fn encode_rgba10_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    encode_rgba_with_alpha_wide(rgba, width, height, BitDepth::Ten, cfg)
}

/// Encode a 12-bit RGB image to HEIC.
///
/// `rgb` must hold exactly `width * height * 3` samples, each in `0..=4095`,
/// packed as `u16`.
pub fn encode_rgb12(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgb, width, height, 3)?;
    encode_rgb_wide(rgb, width, height, BitDepth::Twelve, cfg)
}

/// Encode a 12-bit RGBA image to HEIC. Alpha is **discarded**.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_rgba12(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    let rgb: Vec<u16> = rgba
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    encode_rgb_wide(&rgb, width, height, BitDepth::Twelve, cfg)
}

/// Encode a 12-bit RGBA image to HEIC with a separate alpha auxiliary image.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_rgba12_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    encode_rgba_with_alpha_wide(rgba, width, height, BitDepth::Twelve, cfg)
}

/// Encode a packed 8-bit greyscale image to HEIC (monochrome, no chroma).
///
/// `gray` must hold exactly `width * height` bytes.
pub fn encode_gray(
    gray: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(gray, width, height, 1)?;
    let wide: Vec<u16> = gray.iter().map(|&b| b as u16).collect();
    encode_gray_wide(&wide, width, height, BitDepth::Eight, cfg)
}

/// Encode a packed 8-bit greyscale + alpha image to HEIC. Alpha is **discarded**.
/// Use [`encode_gray_alpha_with_alpha`] to preserve it.
///
/// `ya` must hold exactly `width * height * 2` bytes in Y, A order.
pub fn encode_gray_alpha(
    ya: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(ya, width, height, 2)?;
    let wide: Vec<u16> = ya.chunks_exact(2).map(|px| px[0] as u16).collect();
    encode_gray_wide(&wide, width, height, BitDepth::Eight, cfg)
}

/// Encode a packed 8-bit greyscale + alpha image to HEIC with a separate
/// alpha auxiliary image per ISO/IEC 23008-12.
///
/// `ya` must hold exactly `width * height * 2` bytes in Y, A order.
pub fn encode_gray_alpha_with_alpha(
    ya: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(ya, width, height, 2)?;
    let wide: Vec<u16> = ya.iter().map(|&b| b as u16).collect();
    encode_gray_alpha_wide(&wide, width, height, BitDepth::Eight, cfg)
}

/// Encode a 10-bit greyscale image to HEIC.
///
/// `gray` must hold exactly `width * height` samples in `0..=1023`, packed as `u16`.
pub fn encode_gray10(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(gray, width, height, 1)?;
    encode_gray_wide(gray, width, height, BitDepth::Ten, cfg)
}

/// Encode a 10-bit greyscale + alpha image to HEIC. Alpha is **discarded**.
///
/// `ya` must hold exactly `width * height * 2` samples in Y, A order,
/// each in `0..=1023`, packed as `u16`.
pub fn encode_gray_alpha10(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(ya, width, height, 2)?;
    let luma: Vec<u16> = ya.chunks_exact(2).map(|px| px[0]).collect();
    encode_gray_wide(&luma, width, height, BitDepth::Ten, cfg)
}

/// Encode a 10-bit greyscale + alpha image to HEIC with a separate alpha
/// auxiliary image.
///
/// `ya` must hold exactly `width * height * 2` samples in Y, A order,
/// each in `0..=1023`, packed as `u16`.
pub fn encode_gray_alpha10_with_alpha(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(ya, width, height, 2)?;
    encode_gray_alpha_wide(ya, width, height, BitDepth::Ten, cfg)
}

/// Encode a 12-bit greyscale image to HEIC.
///
/// `gray` must hold exactly `width * height` samples in `0..=4095`, packed as `u16`.
pub fn encode_gray12(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(gray, width, height, 1)?;
    encode_gray_wide(gray, width, height, BitDepth::Twelve, cfg)
}

/// Encode a 12-bit greyscale + alpha image to HEIC. Alpha is **discarded**.
///
/// `ya` must hold exactly `width * height * 2` samples in Y, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_gray_alpha12(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(ya, width, height, 2)?;
    let luma: Vec<u16> = ya.chunks_exact(2).map(|px| px[0]).collect();
    encode_gray_wide(&luma, width, height, BitDepth::Twelve, cfg)
}

/// Encode a 12-bit greyscale + alpha image to HEIC with a separate alpha
/// auxiliary image.
///
/// `ya` must hold exactly `width * height * 2` samples in Y, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_gray_alpha12_with_alpha(
    ya: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(ya, width, height, 2)?;
    encode_gray_alpha_wide(ya, width, height, BitDepth::Twelve, cfg)
}

/// Encode pre-converted planar YCbCr directly, skipping the RGB→YCbCr step.
///
/// The [`Yuv`] carries its own chroma format and bit depth; `cfg.chroma` is
/// **ignored** in favour of what is stored in the planes. This entry point is
/// for callers that already hold YCbCr data (camera pipeline, decoder output,
/// etc.). The visible `width`/`height` are read from the [`Yuv`] itself.
///
/// Images wider or taller than 512 px are encoded as a HEIF grid of 512×512
/// tiles automatically.
///
/// Plane dimensions must satisfy the chroma subsampling grid. Use
/// [`Yuv::from_planes`] or [`yuv::rgb_to_yuv`] to produce a conformant
/// [`Yuv`]; those functions validate plane sizes on construction.
pub fn encode_yuv(yuv: &Yuv, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    cfg.validate()?;
    if needs_tiling(yuv.display_w, yuv.display_h) {
        return encode_yuv_tiled(yuv, cfg);
    }
    let nalu_stream = hevc::encode_intra(yuv, yuv.width, yuv.height, cfg.quality)?;
    isobmff::wrap_hevc_image(
        &nalu_stream,
        yuv.display_w,
        yuv.display_h,
        yuv.bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Returns true if the image is large enough to need grid tiling.
fn needs_tiling(width: u32, height: u32) -> bool {
    width > TILE_SIZE || height > TILE_SIZE
}

/// Core RGB path: dispatches to tiled grid for large images, single `hvc1`
/// item otherwise. `rgb` is always 3-channel u16 at this point.
fn encode_rgb_wide(
    rgb: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if needs_tiling(width, height) {
        return encode_rgb_tiled(rgb, width, height, bit_depth, cfg);
    }
    let (enc_w, enc_h) = encoded_dims(width, height, cfg.chroma);
    let mut yuv = if enc_w != width || enc_h != height {
        let padded = pad_buf(rgb, width, height, enc_w, enc_h, 3);
        yuv::rgb_to_yuv(&padded, enc_w, enc_h, cfg.chroma, bit_depth)
    } else {
        yuv::rgb_to_yuv(rgb, width, height, cfg.chroma, bit_depth)
    };
    yuv = yuv.with_display(width, height);
    encode_yuv_raw(&yuv, cfg)
}

/// Core RGBA-with-alpha path. Dispatches to a paired color+alpha grid for
/// images larger than [`TILE_SIZE`]; otherwise a single `hvc1`+auxl pair.
fn encode_rgba_with_alpha_wide(
    rgba: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if needs_tiling(width, height) {
        return encode_rgba_alpha_tiled(rgba, width, height, bit_depth, cfg);
    }
    let (enc_w, enc_h) = encoded_dims(width, height, cfg.chroma);
    let (w, h) = (width as usize, height as usize);
    let (nw, nh) = (enc_w as usize, enc_h as usize);

    let mut colour_buf = vec![0u16; nw * nh * 3];
    let mut alpha_plane = vec![0u16; nw * nh];

    for (dst_row_idx, (col_row, alp_row)) in colour_buf
        .chunks_exact_mut(nw * 3)
        .zip(alpha_plane.chunks_exact_mut(nw))
        .enumerate()
    {
        let sr = dst_row_idx.min(h - 1);
        let src_row = &rgba[sr * w * 4..(sr * w + w) * 4];
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

    let color_yuv = yuv::rgb_to_yuv(&colour_buf, enc_w, enc_h, cfg.chroma, bit_depth);
    let color_stream = hevc::encode_intra(&color_yuv, enc_w, enc_h, cfg.quality)?;

    let alpha_yuv = build_mono_yuv(alpha_plane, enc_w, enc_h, width, height, bit_depth);
    let alpha_stream = hevc::encode_intra(&alpha_yuv, enc_w, enc_h, cfg.quality)?;

    isobmff::wrap_hevc_image_with_alpha(
        &color_stream,
        &alpha_stream,
        width,
        height,
        bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Core greyscale path: dispatches to tiled grid for large images.
/// Always encodes as Monochrome regardless of `cfg.chroma`.
fn encode_gray_wide(
    gray: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if needs_tiling(width, height) {
        return encode_gray_tiled(gray, width, height, bit_depth, cfg);
    }
    // Greyscale is always Monochrome regardless of cfg.chroma.
    let (enc_w, enc_h) = encoded_dims(width, height, ChromaFormat::Monochrome);
    let luma = if enc_w != width || enc_h != height {
        pad_buf(gray, width, height, enc_w, enc_h, 1)
    } else {
        gray.to_vec()
    };
    let yuv = build_mono_yuv(luma, enc_w, enc_h, width, height, bit_depth);
    encode_yuv_raw(&yuv, cfg)
}

/// Core greyscale-with-alpha path. Dispatches to a paired luma+alpha grid
/// for images larger than [`TILE_SIZE`]; otherwise a single item+auxl pair.
fn encode_gray_alpha_wide(
    ya: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if needs_tiling(width, height) {
        return encode_gray_alpha_tiled(ya, width, height, bit_depth, cfg);
    }
    let (enc_w, enc_h) = encoded_dims(width, height, ChromaFormat::Monochrome);
    let (w, h) = (width as usize, height as usize);
    let (nw, nh) = (enc_w as usize, enc_h as usize);

    let mut luma_plane = vec![0u16; nw * nh];
    let mut alpha_plane = vec![0u16; nw * nh];

    for (dst_row_idx, (luma_row, alp_row)) in luma_plane
        .chunks_exact_mut(nw)
        .zip(alpha_plane.chunks_exact_mut(nw))
        .enumerate()
    {
        let sr = dst_row_idx.min(h - 1);
        let src_row = &ya[sr * w * 2..(sr * w + w) * 2];
        for (dst_col_idx, (luma_px, alp_px)) in
            luma_row.iter_mut().zip(alp_row.iter_mut()).enumerate()
        {
            let sc = dst_col_idx.min(w - 1);
            *luma_px = src_row[sc * 2];
            *alp_px = src_row[sc * 2 + 1];
        }
    }

    let color_yuv = build_mono_yuv(luma_plane, enc_w, enc_h, width, height, bit_depth);
    let color_stream = hevc::encode_intra(&color_yuv, enc_w, enc_h, cfg.quality)?;

    let alpha_yuv = build_mono_yuv(alpha_plane, enc_w, enc_h, width, height, bit_depth);
    let alpha_stream = hevc::encode_intra(&alpha_yuv, enc_w, enc_h, cfg.quality)?;

    isobmff::wrap_hevc_image_with_alpha(
        &color_stream,
        &alpha_stream,
        width,
        height,
        bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Wrap an already-built [`Yuv`] into a HEIC bitstream (no re-validation).
fn encode_yuv_raw(yuv: &Yuv, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    let nalu_stream = hevc::encode_intra(yuv, yuv.width, yuv.height, cfg.quality)?;
    isobmff::wrap_hevc_image(
        &nalu_stream,
        yuv.display_w,
        yuv.display_h,
        yuv.bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Construct a monochrome [`Yuv`] from a pre-built luma plane.
fn build_mono_yuv(
    y: Vec<u16>,
    coded_w: u32,
    coded_h: u32,
    display_w: u32,
    display_h: u32,
    bit_depth: BitDepth,
) -> Yuv {
    Yuv {
        y,
        cb: Vec::new(),
        cr: Vec::new(),
        width: coded_w,
        height: coded_h,
        display_w,
        display_h,
        chroma: ChromaFormat::Monochrome,
        bit_depth,
    }
}

// ── Tiling helpers ────────────────────────────────────────────────────────────

/// Encode a large RGB image as a HEIF grid of [`TILE_SIZE`]×[`TILE_SIZE`] tiles.
fn encode_rgb_tiled(
    rgb: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cols = width.div_ceil(TILE_SIZE);
    let rows = height.div_ceil(TILE_SIZE);
    // TILE_SIZE=512 is always chroma-even for sub_w/sub_h ∈ {1,2}, so
    // encoded_dims returns (512,512) for every subsampling format.
    let (enc_tw, enc_th) = encoded_dims(TILE_SIZE, TILE_SIZE, cfg.chroma);

    let mut tile_streams = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let tile = extract_rgb_tile(rgb, width, height, col, row, TILE_SIZE, 3);
            let yuv = yuv::rgb_to_yuv(&tile, enc_tw, enc_th, cfg.chroma, bit_depth);
            tile_streams.push(hevc::encode_intra(&yuv, enc_tw, enc_th, cfg.quality)?);
        }
    }
    isobmff::wrap_hevc_grid(
        &tile_streams,
        cols,
        rows,
        enc_tw,
        enc_th,
        width,
        height,
        bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a large greyscale image as a HEIF grid of monochrome tiles.
fn encode_gray_tiled(
    gray: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cols = width.div_ceil(TILE_SIZE);
    let rows = height.div_ceil(TILE_SIZE);

    let mut tile_streams = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let luma = extract_plane_tile(
                gray,
                width as usize,
                height as usize,
                (col * TILE_SIZE) as usize,
                (row * TILE_SIZE) as usize,
                TILE_SIZE as usize,
                TILE_SIZE as usize,
            );
            let yuv = build_mono_yuv(luma, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, bit_depth);
            tile_streams.push(hevc::encode_intra(&yuv, TILE_SIZE, TILE_SIZE, cfg.quality)?);
        }
    }
    isobmff::wrap_hevc_grid(
        &tile_streams,
        cols,
        rows,
        TILE_SIZE,
        TILE_SIZE,
        width,
        height,
        bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a large pre-converted [`Yuv`] image as a HEIF grid. Splits all
/// planes (Y, Cb, Cr) into tiles respecting the chroma subsampling ratio.
fn encode_yuv_tiled(yuv: &Yuv, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    let cols = yuv.display_w.div_ceil(TILE_SIZE);
    let rows = yuv.display_h.div_ceil(TILE_SIZE);
    let (enc_tw, enc_th) = encoded_dims(TILE_SIZE, TILE_SIZE, yuv.chroma);

    let sw = yuv.chroma.sub_w() as u32;
    let sh = yuv.chroma.sub_h() as u32;
    // Chroma tile dimensions after subsampling.
    let c_tw = (enc_tw / sw) as usize;
    let c_th = (enc_th / sh) as usize;
    // Full chroma plane dimensions (yuv.width is already chroma-even padded).
    let c_src_w = (yuv.width / sw) as usize;
    let c_src_h = (yuv.height / sh) as usize;

    let mut tile_streams = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let x0 = (col * TILE_SIZE) as usize;
            let y0 = (row * TILE_SIZE) as usize;

            let y_tile = extract_plane_tile(
                &yuv.y,
                yuv.width as usize,
                yuv.height as usize,
                x0,
                y0,
                enc_tw as usize,
                enc_th as usize,
            );
            let (cb_tile, cr_tile) = if yuv.chroma.is_monochrome() {
                (Vec::new(), Vec::new())
            } else {
                let cx0 = x0 / sw as usize;
                let cy0 = y0 / sh as usize;
                (
                    extract_plane_tile(&yuv.cb, c_src_w, c_src_h, cx0, cy0, c_tw, c_th),
                    extract_plane_tile(&yuv.cr, c_src_w, c_src_h, cx0, cy0, c_tw, c_th),
                )
            };

            let tile_yuv = Yuv {
                y: y_tile,
                cb: cb_tile,
                cr: cr_tile,
                width: enc_tw,
                height: enc_th,
                display_w: enc_tw,
                display_h: enc_th,
                chroma: yuv.chroma,
                bit_depth: yuv.bit_depth,
            };
            tile_streams.push(hevc::encode_intra(&tile_yuv, enc_tw, enc_th, cfg.quality)?);
        }
    }
    isobmff::wrap_hevc_grid(
        &tile_streams,
        cols,
        rows,
        enc_tw,
        enc_th,
        yuv.display_w,
        yuv.display_h,
        yuv.bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a large RGBA image as a paired HEIF grid: a color grid + an alpha
/// auxiliary grid, both with [`TILE_SIZE`]×[`TILE_SIZE`] tiles.
fn encode_rgba_alpha_tiled(
    rgba: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cols = width.div_ceil(TILE_SIZE);
    let rows = height.div_ceil(TILE_SIZE);
    let (enc_tw, enc_th) = encoded_dims(TILE_SIZE, TILE_SIZE, cfg.chroma);
    let ts2 = (TILE_SIZE * TILE_SIZE) as usize;

    let mut color_streams = Vec::with_capacity((cols * rows) as usize);
    let mut alpha_streams = Vec::with_capacity((cols * rows) as usize);

    for row in 0..rows {
        for col in 0..cols {
            // Extract a TILE_SIZE×TILE_SIZE RGBA tile (4 ch) with edge replication.
            let tile = extract_rgb_tile(rgba, width, height, col, row, TILE_SIZE, 4);

            // Deinterleave RGBA → color (3 ch) + alpha (1 ch).
            let mut color_buf = vec![0u16; ts2 * 3];
            let mut alpha_plane = vec![0u16; ts2];
            for ((px, colors), alpha) in tile
                .as_chunks::<4>()
                .0
                .iter()
                .zip(color_buf.as_chunks_mut::<3>().0.iter_mut())
                .zip(alpha_plane.iter_mut())
            {
                colors[0] = px[0];
                colors[1] = px[1];
                colors[2] = px[2];
                *alpha = px[3];
            }

            let color_yuv = yuv::rgb_to_yuv(&color_buf, enc_tw, enc_th, cfg.chroma, bit_depth);
            color_streams.push(hevc::encode_intra(&color_yuv, enc_tw, enc_th, cfg.quality)?);

            // Alpha is always monochrome; TILE_SIZE is already dimension-aligned.
            let alpha_yuv = build_mono_yuv(
                alpha_plane,
                TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
                bit_depth,
            );
            alpha_streams.push(hevc::encode_intra(
                &alpha_yuv,
                TILE_SIZE,
                TILE_SIZE,
                cfg.quality,
            )?);
        }
    }

    isobmff::wrap_hevc_grid_with_alpha(
        &color_streams,
        &alpha_streams,
        cols,
        rows,
        enc_tw,
        enc_th,
        width,
        height,
        bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Encode a large greyscale+alpha image as a paired HEIF grid: a luma grid +
/// an alpha auxiliary grid, both monochrome, with [`TILE_SIZE`]×[`TILE_SIZE`] tiles.
fn encode_gray_alpha_tiled(
    ya: &[u16],
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let cols = width.div_ceil(TILE_SIZE);
    let rows = height.div_ceil(TILE_SIZE);
    let ts2 = (TILE_SIZE * TILE_SIZE) as usize;

    let mut luma_streams = Vec::with_capacity((cols * rows) as usize);
    let mut alpha_streams = Vec::with_capacity((cols * rows) as usize);

    for row in 0..rows {
        for col in 0..cols {
            // Extract a TILE_SIZE×TILE_SIZE YA tile (2 ch) with edge replication.
            let tile = extract_rgb_tile(ya, width, height, col, row, TILE_SIZE, 2);

            // Deinterleave YA → luma (1 ch) + alpha (1 ch).
            let mut luma_plane = vec![0u16; ts2];
            let mut alpha_plane = vec![0u16; ts2];
            for ((px, luma), alpha) in tile
                .as_chunks::<2>()
                .0
                .iter()
                .zip(luma_plane.iter_mut())
                .zip(alpha_plane.iter_mut())
            {
                *luma = px[0];
                *alpha = px[1];
            }

            let luma_yuv = build_mono_yuv(
                luma_plane, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE, bit_depth,
            );
            luma_streams.push(hevc::encode_intra(
                &luma_yuv,
                TILE_SIZE,
                TILE_SIZE,
                cfg.quality,
            )?);

            let alpha_yuv = build_mono_yuv(
                alpha_plane,
                TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
                bit_depth,
            );
            alpha_streams.push(hevc::encode_intra(
                &alpha_yuv,
                TILE_SIZE,
                TILE_SIZE,
                cfg.quality,
            )?);
        }
    }

    isobmff::wrap_hevc_grid_with_alpha(
        &luma_streams,
        &alpha_streams,
        cols,
        rows,
        TILE_SIZE,
        TILE_SIZE,
        width,
        height,
        bit_depth,
        &cfg.color,
        &cfg.metadata,
    )
}

/// Extract a [`TILE_SIZE`]×[`TILE_SIZE`] interleaved-channel tile from a wide
/// pixel buffer, replication-padding at the right and bottom edges.
fn extract_rgb_tile(
    src: &[u16],
    src_w: u32,
    src_h: u32,
    col: u32,
    row: u32,
    tile_size: u32,
    channels: usize,
) -> Vec<u16> {
    let (sw, sh) = (src_w as usize, src_h as usize);
    let ts = tile_size as usize;
    let x0 = (col * tile_size) as usize;
    let y0 = (row * tile_size) as usize;
    let mut tile = vec![0u16; ts * ts * channels];
    for ty in 0..ts {
        let sy = (y0 + ty).min(sh - 1);
        let src_row = &src[sy * sw * channels..(sy * sw + sw) * channels];
        let dst_row = &mut tile[ty * ts * channels..(ty * ts + ts) * channels];
        for tx in 0..ts {
            let sx = (x0 + tx).min(sw - 1);
            dst_row[tx * channels..(tx + 1) * channels]
                .copy_from_slice(&src_row[sx * channels..(sx + 1) * channels]);
        }
    }
    tile
}

/// Extract a single-channel plane tile, replication-padding at edges.
fn extract_plane_tile(
    plane: &[u16],
    src_w: usize,
    src_h: usize,
    x0: usize,
    y0: usize,
    tile_w: usize,
    tile_h: usize,
) -> Vec<u16> {
    let mut tile = vec![0u16; tile_w * tile_h];
    for ty in 0..tile_h {
        let sy = (y0 + ty).min(src_h - 1);
        let src_row = &plane[sy * src_w..(sy + 1) * src_w];
        let dst_row = &mut tile[ty * tile_w..(ty + 1) * tile_w];
        for tx in 0..tile_w {
            dst_row[tx] = src_row[(x0 + tx).min(src_w - 1)];
        }
    }
    tile
}

// ── Validation helpers ────────────────────────────────────────────────────────

pub(crate) fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM || width > MAX_DIM || height > MAX_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

fn validate_quality(quality: u8) -> Result<(), EncodeError> {
    if quality == 0 || quality > 100 {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

fn validate_buf_u8(buf: &[u8], w: u32, h: u32, ch: usize) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<u8>(w as usize, h as usize, ch)?;
    if buf.len() != needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

fn validate_buf_u16(buf: &[u16], w: u32, h: u32, ch: usize) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<u16>(w as usize, h as usize, ch)?;
    if buf.len() != needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

pub(crate) fn checked_buffer_size<T>(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<usize, EncodeError> {
    let pixel_size = size_of::<T>();
    width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(channels))
        .and_then(|v| {
            v.checked_mul(pixel_size)
                .and_then(|b| isize::try_from(b).ok())?;
            Some(v)
        })
        .ok_or(EncodeError::InvalidInput)
}

fn encoded_dims(width: u32, height: u32, chroma: ChromaFormat) -> (u32, u32) {
    let sw = chroma.sub_w() as u32;
    let sh = chroma.sub_h() as u32;
    (width.div_ceil(sw) * sw, height.div_ceil(sh) * sh)
}

/// Replicate-pad a planar buffer from `(w, h)` to `(nw, nh)`.
/// `channels` is the number of interleaved u16 samples per pixel.
fn pad_buf(src: &[u16], w: u32, h: u32, nw: u32, nh: u32, channels: usize) -> Vec<u16> {
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    let src_stride = w * channels;
    let dst_stride = nw * channels;
    let mut out = vec![0u16; nw * nh * channels];

    for (dst_row_idx, dst_row) in out.chunks_exact_mut(dst_stride).enumerate() {
        let sr = dst_row_idx.min(h - 1);
        let src_row = &src[sr * src_stride..(sr + 1) * src_stride];
        let (real, pad) = dst_row.split_at_mut(src_stride);
        real.copy_from_slice(src_row);
        if !pad.is_empty() {
            let last_px = &src_row[src_row.len() - channels..];
            for px in pad.chunks_exact_mut(channels) {
                px.copy_from_slice(last_px);
            }
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EncodeConfig {
        EncodeConfig::new()
    }

    // ── validation ───────────────────────────────────────────────────────

    #[test]
    fn rejects_zero_dims() {
        assert!(validate_dims(0, 1).is_err());
        assert!(validate_dims(1, 0).is_err());
    }

    #[test]
    fn rejects_oversized_dims() {
        assert!(validate_dims(MAX_DIM + 1, 1).is_err());
        assert!(validate_dims(1, MAX_DIM + 1).is_err());
    }

    #[test]
    fn rejects_quality_bounds() {
        assert!(cfg().with_quality(0).validate().is_err());
        assert!(cfg().with_quality(101).validate().is_err());
        assert!(cfg().with_quality(1).validate().is_ok());
        assert!(cfg().with_quality(100).validate().is_ok());
    }

    #[test]
    fn rejects_wrong_buffer_size() {
        assert!(encode_rgb(&vec![0u8; 46], 4, 4, &cfg()).is_err());
        assert!(encode_rgb(&vec![0u8; 49], 4, 4, &cfg()).is_err());
        assert!(encode_rgb(&vec![0u8; 48], 4, 4, &cfg()).is_ok());
    }

    // ── 8-bit RGB / RGBA ─────────────────────────────────────────────────

    #[test]
    fn encode_rgb8_produces_heic() {
        let out = encode_rgb(&vec![100u8; 16 * 16 * 3], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_rgba8_strips_alpha() {
        let out = encode_rgba(&vec![100u8; 16 * 16 * 4], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_rgba8_with_alpha_produces_heic() {
        let out = encode_rgba_with_alpha(&vec![200u8; 16 * 16 * 4], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    // ── 10-bit RGB / RGBA ────────────────────────────────────────────────

    #[test]
    fn encode_rgb10_produces_heic() {
        let out = encode_rgb10(&vec![512u16; 16 * 16 * 3], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_rgba10_strips_alpha() {
        let out = encode_rgba10(&vec![512u16; 16 * 16 * 4], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_rgba10_with_alpha_produces_heic() {
        let out = encode_rgba10_with_alpha(&vec![512u16; 16 * 16 * 4], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_rgb12_produces_heic() {
        let out = encode_rgb12(&vec![2048u16; 16 * 16 * 3], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_rgba12_with_alpha_produces_heic() {
        let out = encode_rgba12_with_alpha(&vec![2048u16; 16 * 16 * 4], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_gray8_produces_heic() {
        let out = encode_gray(&vec![128u8; 16 * 16], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_gray_alpha8_strips_alpha() {
        let out = encode_gray_alpha(&vec![200u8; 16 * 16 * 2], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_gray_alpha8_with_alpha_produces_heic() {
        let out = encode_gray_alpha_with_alpha(&vec![180u8; 16 * 16 * 2], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_gray10_produces_heic() {
        let out = encode_gray10(&vec![512u16; 16 * 16], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_gray_alpha10_with_alpha_produces_heic() {
        let out =
            encode_gray_alpha10_with_alpha(&vec![512u16; 16 * 16 * 2], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_gray12_produces_heic() {
        let out = encode_gray12(&vec![2048u16; 16 * 16], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    #[test]
    fn encode_gray_alpha12_with_alpha_produces_heic() {
        let out =
            encode_gray_alpha12_with_alpha(&vec![2048u16; 16 * 16 * 2], 16, 16, &cfg()).unwrap();
        assert!(out.len() > 100);
    }

    // ── YUV direct API ───────────────────────────────────────────────────

    #[test]
    fn encode_yuv_roundtrips() {
        let rgb = vec![128u16; 16 * 16 * 3];
        let yuv = yuv::rgb_to_yuv(&rgb, 16, 16, ChromaFormat::Yuv420, BitDepth::Eight);
        let out = encode_yuv(&yuv, &cfg()).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn encode_yuv_rejects_invalid_planes() {
        let bad = Yuv::from_planes(
            vec![0u16; 15], // should be 16
            vec![0u16; 4],
            vec![0u16; 4],
            4,
            4,
            ChromaFormat::Yuv420,
            BitDepth::Eight,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn odd_dimensions_reported_in_ispe() {
        let rgb = vec![100u8; 281 * 181 * 3];
        let out = encode_rgb(&rgb, 281, 181, &cfg().with_chroma(ChromaFormat::Yuv420)).unwrap();

        let ispe = out.windows(4).position(|w| w == b"ispe").expect("ispe");
        let wpos = ispe + 4 + 4;
        let w = u32::from_be_bytes(out[wpos..wpos + 4].try_into().unwrap());
        let h = u32::from_be_bytes(out[wpos + 4..wpos + 8].try_into().unwrap());
        assert_eq!((w, h), (281, 181));
    }

    #[test]
    fn encode_1x1_rgb8() {
        assert!(encode_rgb(&[255, 0, 0], 1, 1, &cfg()).is_ok());
    }

    #[test]
    fn encode_1x1_gray8() {
        assert!(encode_gray(&[128], 1, 1, &cfg()).is_ok());
    }

    #[test]
    fn encode_1x1_rgba8_with_alpha() {
        assert!(encode_rgba_with_alpha(&[255, 0, 0, 255], 1, 1, &cfg()).is_ok());
    }

    #[test]
    fn tiled_rgb8_produces_grid_heic() {
        // 1024×768 triggers 2×2 grid tiling.
        let px: Vec<u8> = (0u32..1024 * 768 * 3).map(|i| (i % 256) as u8).collect();
        let out = encode_rgb(&px, 1024, 768, &cfg()).unwrap();
        assert!(out.len() > 1000);
        assert_eq!(&out[4..8], b"ftyp");
        // A grid HEIC has a 'grid' item type in iinf.
        assert!(out.windows(4).any(|w| w == b"grid"), "expected grid item");
    }

    #[test]
    fn tiled_rgb10_produces_grid_heic() {
        let px = vec![512u16; 1024 * 768 * 3];
        let out = encode_rgb10(&px, 1024, 768, &cfg()).unwrap();
        assert!(out.windows(4).any(|w| w == b"grid"));
    }

    #[test]
    fn tiled_gray8_produces_grid_heic() {
        let px: Vec<u8> = (0u32..1024 * 768).map(|i| (i % 256) as u8).collect();
        let out = encode_gray(&px, 1024, 768, &cfg()).unwrap();
        assert!(out.windows(4).any(|w| w == b"grid"));
    }

    #[test]
    fn tiled_yuv_produces_grid_heic() {
        let rgb = vec![200u16; 1024 * 768 * 3];
        let yuv = yuv::rgb_to_yuv(&rgb, 1024, 768, ChromaFormat::Yuv420, BitDepth::Eight);
        let out = encode_yuv(&yuv, &cfg()).unwrap();
        assert!(out.windows(4).any(|w| w == b"grid"));
    }

    #[test]
    fn tiled_grid_has_correct_ispe() {
        // ispe for the grid item should reflect the full image dimensions.
        let px: Vec<u8> = vec![128u8; 1024 * 768 * 3];
        let out = encode_rgb(&px, 1024, 768, &cfg()).unwrap();
        // Find the ispe that has 1024×768 (not the tile-size ispe 512×512).
        let mut found = false;
        let mut i = 0;
        while i + 16 <= out.len() {
            if &out[i..i + 4] == b"ispe" {
                let w = u32::from_be_bytes(out[i + 8..i + 12].try_into().unwrap());
                let h = u32::from_be_bytes(out[i + 12..i + 16].try_into().unwrap());
                if w == 1024 && h == 768 {
                    found = true;
                    break;
                }
            }
            i += 1;
        }
        assert!(found, "no ispe 1024×768 found in tiled output");
    }

    // ── tiled alpha ──────────────────────────────────────────────────────

    #[test]
    fn tiled_rgba8_with_alpha_produces_grid_heic() {
        let px: Vec<u8> = (0u32..1024 * 768 * 4).map(|i| (i % 256) as u8).collect();
        let out = encode_rgba_with_alpha(&px, 1024, 768, &cfg()).unwrap();
        assert!(out.windows(4).any(|w| w == b"grid"), "expected grid item");
        assert!(
            out.windows(4).any(|w| w == b"auxl"),
            "expected auxl reference"
        );
        assert!(
            out.windows(4).any(|w| w == b"auxC"),
            "expected auxC property"
        );
    }

    #[test]
    fn tiled_rgba10_with_alpha_produces_grid_heic() {
        let px = vec![512u16; 1024 * 768 * 4];
        let out = encode_rgba10_with_alpha(&px, 1024, 768, &cfg()).unwrap();
        assert!(out.windows(4).any(|w| w == b"grid"));
        assert!(out.windows(4).any(|w| w == b"auxl"));
    }

    #[test]
    fn tiled_gray_alpha8_with_alpha_produces_grid_heic() {
        let px: Vec<u8> = (0u32..1024 * 768 * 2).map(|i| (i % 256) as u8).collect();
        let out = encode_gray_alpha_with_alpha(&px, 1024, 768, &cfg()).unwrap();
        assert!(out.windows(4).any(|w| w == b"grid"));
        assert!(out.windows(4).any(|w| w == b"auxl"));
    }

    #[test]
    fn tiled_alpha_grid_has_correct_ispe() {
        let px: Vec<u8> = vec![200u8; 1024 * 768 * 4];
        let out = encode_rgba_with_alpha(&px, 1024, 768, &cfg()).unwrap();
        // Must contain an ispe 1024×768 for the color grid item.
        let mut found = false;
        let mut i = 0;
        while i + 16 <= out.len() {
            if &out[i..i + 4] == b"ispe" {
                let w = u32::from_be_bytes(out[i + 8..i + 12].try_into().unwrap());
                let h = u32::from_be_bytes(out[i + 12..i + 16].try_into().unwrap());
                if w == 1024 && h == 768 {
                    found = true;
                    break;
                }
            }
            i += 1;
        }
        assert!(found, "no ispe 1024×768 in tiled alpha output");
    }

    #[test]
    fn tiled_alpha_has_two_grid_items() {
        let px: Vec<u8> = vec![128u8; 1024 * 768 * 4];
        let out = encode_rgba_with_alpha(&px, 1024, 768, &cfg()).unwrap();
        // Two 'grid' entries in iinf: color grid + alpha grid.
        let count = out.windows(4).filter(|w| *w == b"grid").count();
        assert_eq!(
            count, 2,
            "expected 2 grid items (color + alpha), got {count}"
        );
    }
    // ── checked_buffer_size ──────────────────────────────────────────────

    #[test]
    fn buffer_size_correct() {
        assert_eq!(checked_buffer_size::<u16>(4, 4, 3).unwrap(), 48);
        assert_eq!(checked_buffer_size::<u8>(1, 1, 1).unwrap(), 1);
    }

    #[test]
    fn buffer_size_overflow() {
        assert!(checked_buffer_size::<u16>(usize::MAX, usize::MAX, 3).is_err());
    }

    #[test]
    fn pad_replicates_column() {
        let src = vec![10u16, 20, 30];
        assert_eq!(pad_buf(&src, 1, 1, 2, 1, 3), vec![10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn pad_replicates_row() {
        let src = vec![10u16, 20, 30];
        assert_eq!(pad_buf(&src, 1, 1, 1, 2, 3), vec![10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn pad_noop_when_aligned() {
        let src: Vec<u16> = (0..12).collect();
        assert_eq!(pad_buf(&src, 2, 2, 2, 2, 3), src);
    }
}
