mod cabac;
pub mod color;
pub mod dct;
pub mod deblock;
pub mod error;
pub mod fmt;
pub mod hevc;
pub mod hevc_transform;
mod icc_profile;
mod intra;
pub mod isobmff;
pub mod yuv;

pub use color::{ColorEncoding, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
pub use error::EncodeError;
pub use fmt::{BitDepth, ChromaFormat};
pub use yuv::Yuv;

/// Build identifier — bump when the encoder changes. Print this from your binary
/// (e.g. `eprintln!("hpvca {}", hpvca::BUILD_ID)`) to confirm you compiled the
/// latest source and aren't running a stale checkpoint.
pub const BUILD_ID: &str = "apple-compat-2026-06-04: 64x64 CTB, IDR_N_LP(20), \
slice-only mdat, iloc-before-iinf, array_completeness=0, BT.709 full-range, \
configurable-colr(nclx+ICC), even-dim rounding, level-scales-with-size, PTL-frame-only-constraints, 64-mult-full-CTB, DPB3, ICC-v2-colr-profile, SAO-enabled, x265-aligned-full, spec-EncodeFlush-termination, 4:0:0+4:2:0+4:2:2+4:4:4-chroma, alpha-aux-item, 8+10+12-bit, EncodeConfig-builder, YUV-direct-API";

/// Encoder configuration. Build with [`EncodeConfig::new`] and the `with_*` methods,
/// then pass to [`encode`] / [`encode_with_alpha`] / [`encode_yuv`].
///
/// ```ignore
/// let cfg = EncodeConfig::new()
///     .with_quality(90)
///     .with_chroma(ChromaFormat::Yuv444)
///     .with_bit_depth(BitDepth::Ten)
///     .with_color(ColorMetadata::Cicp(ColorEncoding::bt2020_pq()));
/// let heic = encode(&rgb_u16, w, h, &cfg)?;
/// ```
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Visual quality, 1..=100 (higher = better, larger). Maps to the HEVC QP.
    pub quality: u8,
    /// Chroma subsampling format.
    pub chroma: ChromaFormat,
    /// Sample bit depth (8/10/12).
    pub bit_depth: BitDepth,
    /// Colour metadata written to the `colr` box and reflected in the VUI. Either
    /// enumerated CICP (`nclx`) or an embedded ICC profile (`prof`).
    pub color: ColorMetadata,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 90,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            color: ColorMetadata::default(), // sRGB ICC profile
        }
    }
}

impl EncodeConfig {
    /// A config with default settings (q=90, 4:2:0, 8-bit, sRGB ICC).
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

    /// Use enumerated CICP signalling (shorthand for `with_color(ColorMetadata::Cicp(..))`).
    pub fn with_cicp(mut self, enc: ColorEncoding) -> Self {
        self.color = ColorMetadata::Cicp(enc);
        self
    }
}

/// Encode native-depth planar RGB (`u16` per channel) to HEIC using `cfg`.
///
/// `rgb` samples are at `cfg.bit_depth`'s native range (0..=255 / 0..=1023 / 0..=4095);
/// the library does not rescale. The chroma format's subsampling rounds the visible
/// dimensions; the conformance window crops the padding so 'ispe' matches.
pub fn encode(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (enc_w, enc_h) = encoded_dims(width, height, cfg.chroma);
    let yuv = if enc_w != width || enc_h != height {
        let padded = pad_rgb_to_even(rgb, width, height, enc_w, enc_h);
        yuv::rgb_to_yuv(&padded, enc_w, enc_h, cfg.chroma, cfg.bit_depth)
    } else {
        yuv::rgb_to_yuv(rgb, width, height, cfg.chroma, cfg.bit_depth)
    };
    encode_yuv(&yuv, cfg)
}

/// Encode pre-converted planar YCbCr directly, skipping RGB→YCbCr. The `Yuv` carries
/// its own chroma format and bit depth; `cfg.chroma`/`cfg.bit_depth` are ignored in
/// favour of the planes' own (they must be self-consistent). Useful when the caller
/// already has YCbCr (e.g. from a decoder or camera pipeline).
///
/// Plane dimensions must already be encode-ready (luma a multiple of the chroma
/// subsampling); use [`Yuv`]'s constructor via [`yuv::rgb_to_yuv`] or build planes to
/// the rounded size. The visible `width`/`height` come from the `Yuv`.
pub fn encode_yuv(yuv: &yuv::Yuv, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    let enc_w = yuv.width;
    let enc_h = yuv.height;
    let nalu_stream = hevc::encode_intra(yuv, enc_w, enc_h, cfg.quality)?;
    isobmff::wrap_hevc_image(&nalu_stream, enc_w, enc_h, yuv.bit_depth, &cfg.color)
}

/// Encode native-depth planar RGBA (`u16` per channel) to HEIC with an alpha channel,
/// using `cfg`. Alpha is encoded as a separate monochrome HEVC item per ISO/IEC
/// 23008-12 and linked as an auxiliary image.
pub fn encode_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let (enc_w, enc_h) = encoded_dims(width, height, cfg.chroma);
    let (w, h) = (width as usize, height as usize);
    let (nw, nh) = (enc_w as usize, enc_h as usize);
    let mut rgb = vec![0u16; nw * nh * 3];
    let mut alpha_rgb = vec![0u16; nw * nh * 3];
    for r in 0..nh {
        let sr = r.min(h - 1);
        for c in 0..nw {
            let sc = c.min(w - 1);
            let s = (sr * w + sc) * 4;
            let d = (r * nw + c) * 3;
            rgb[d..d + 3].copy_from_slice(&rgba[s..s + 3]);
            let a = rgba[s + 3];
            alpha_rgb[d] = a;
            alpha_rgb[d + 1] = a;
            alpha_rgb[d + 2] = a;
        }
    }
    let color_yuv = yuv::rgb_to_yuv(&rgb, enc_w, enc_h, cfg.chroma, cfg.bit_depth);
    let color_stream = hevc::encode_intra(&color_yuv, enc_w, enc_h, cfg.quality)?;
    let alpha_yuv = yuv::rgb_to_yuv(
        &alpha_rgb,
        enc_w,
        enc_h,
        ChromaFormat::Monochrome,
        cfg.bit_depth,
    );
    let alpha_stream = hevc::encode_intra(&alpha_yuv, enc_w, enc_h, cfg.quality)?;
    isobmff::wrap_hevc_image_with_alpha(
        &color_stream,
        &alpha_stream,
        enc_w,
        enc_h,
        cfg.bit_depth,
        &cfg.color,
    )
}

/// Round visible dimensions up to the chroma subsampling grid.
fn encoded_dims(width: u32, height: u32, chroma: ChromaFormat) -> (u32, u32) {
    let sw = chroma.sub_w() as u32;
    let sh = chroma.sub_h() as u32;
    ((width + sw - 1) / sw * sw, (height + sh - 1) / sh * sh)
}

// ── Convenience wrappers (8-bit packed u8 in / simple parameters) ────────────

/// Encode an RGBA image to HEIC with an alpha channel (8-bit, packed `u8`).
pub fn encode_heic_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
) -> Result<Vec<u8>, EncodeError> {
    let wide: Vec<u16> = rgba.iter().map(|&b| b as u16).collect();
    encode_heic_with_alpha_bd(&wide, width, height, quality, chroma, BitDepth::Eight)
}

/// Encode an RGBA image to HEIC with an alpha channel at an explicit bit depth.
/// `rgba` holds one `u16` per channel, already at `bit_depth`'s native range.
pub fn encode_heic_with_alpha_bd(
    rgba: &[u16],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(chroma)
        .with_bit_depth(bit_depth);
    encode_with_alpha(rgba, width, height, &cfg)
}

/// Encode an RGB image to HEIC bytes (4:2:0, 8-bit, packed `u8`).
pub fn encode_heic(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, EncodeError> {
    encode_heic_fmt(rgb, width, height, quality, ChromaFormat::Yuv420)
}

/// Encode an RGB image to HEIC bytes with an explicit chroma format (8-bit, `u8`).
pub fn encode_heic_fmt(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
) -> Result<Vec<u8>, EncodeError> {
    let wide: Vec<u16> = rgb.iter().map(|&b| b as u16).collect();
    encode_heic_fmt_bd(&wide, width, height, quality, chroma, BitDepth::Eight)
}

/// Encode native-depth `u16` RGB to HEIC with an explicit chroma format and bit depth.
pub fn encode_heic_fmt_bd(
    rgb: &[u16],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let cfg = EncodeConfig::new()
        .with_quality(quality)
        .with_chroma(chroma)
        .with_bit_depth(bit_depth);
    encode(rgb, width, height, &cfg)
}

/// Replicate-pad planar RGB from (w,h) to (nw,nh) by repeating the last row/column.
fn pad_rgb_to_even(rgb: &[u16], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u16> {
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    let mut out = vec![0u16; nw * nh * 3];
    for r in 0..nh {
        let sr = r.min(h - 1);
        for c in 0..nw {
            let sc = c.min(w - 1);
            let s = (sr * w + sc) * 3;
            let d = (r * nw + c) * 3;
            out[d..d + 3].copy_from_slice(&rgb[s..s + 3]);
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
    fn from_planes_validates_sizes() {
        // 4x4 4:2:0: luma 16, chroma 2x2=4 each.
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
        // Wrong luma length is rejected.
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

    #[test]
    fn encode_yuv_roundtrips() {
        let rgb = vec![128u16; 16 * 16 * 3];
        let yuv = yuv::rgb_to_yuv(&rgb, 16, 16, ChromaFormat::Yuv420, BitDepth::Eight);
        let cfg = EncodeConfig::new();
        let out = encode_yuv(&yuv, &cfg).unwrap();
        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
    }
}
