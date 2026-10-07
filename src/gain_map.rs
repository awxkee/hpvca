/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! HDR gain maps, packaged the way iPhone HEICs (iOS 18+) carry them.
//!
//! One monochrome HEVC image holds the gain samples and is referenced twice:
//!
//! ```text
//! primary ◄─auxl── gain map image (hidden hvc1, auxC urn:com:apple:photo:2020:aux:hdrgainmap)
//!                      ▲
//!                      └─cdsc── XMP (HDRGainMap:HDRGainMapVersion / HDRGainMapHeadroom)
//! tmap ──dimg──► [primary, gain map image]      ISO 21496-1 metadata, optional
//! grpl/altr { tmap, primary }                   SDR readers fall back to the primary
//! ```
//!
//! The samples use Apple's encoding: a Rec.709-OETF-coded linear gain `g`,
//! applied as `hdr = sdr * (1 + (headroom - 1) * g)` in linear light. That curve
//! is not exactly expressible in ISO 21496-1 (which is logarithmic), so the
//! `tmap` item carries a fitted approximation — the same compromise Apple's own
//! camera makes when it writes both. See [`GainMap::iso_metadata`] to override it.

use crate::color::Cicp;
use crate::hevc::NaluStream;
use crate::{
    BitDepth, ChromaFormat, ColorMetadata, EncodeConfig, EncodeError, MatrixCoefficients,
    Primaries, TransferFunction, Yuv, hevc,
};
use std::fmt;

/// `auxC` type of an Apple HDR gain map auxiliary image.
pub(crate) const APPLE_GAIN_MAP_URN: &[u8] = b"urn:com:apple:photo:2020:aux:hdrgainmap\0";

/// `HDRGainMapVersion` written to the XMP. 2.0 is the version that stores the
/// headroom in XMP rather than in the Apple MakerNote.
const APPLE_GAIN_MAP_VERSION: u32 = 0x2_0000;

/// Denominator of every ISO rational written for a fitted map (Apple uses 10⁶).
const ISO_DENOMINATOR: u32 = 1_000_000;

/// Sample storage of a gain map image (single channel).
#[derive(Clone)]
pub enum GainMapPixels {
    /// `width * height` 8-bit samples. What Apple writes.
    Gray8(Vec<u8>),
    /// `width * height` samples of `bit_depth` (10 or 12) bits each.
    Gray16 { data: Vec<u16>, bit_depth: BitDepth },
}

impl GainMapPixels {
    /// Bit depth of the samples.
    pub fn bit_depth(&self) -> BitDepth {
        match self {
            Self::Gray8(_) => BitDepth::Eight,
            Self::Gray16 { bit_depth, .. } => *bit_depth,
        }
    }

    /// Number of samples in the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Gray8(v) => v.len(),
            Self::Gray16 { data, .. } => data.len(),
        }
    }

    /// True when the buffer holds no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for GainMapPixels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::Gray8(_) => "Gray8",
            Self::Gray16 { .. } => "Gray16",
        };
        write!(
            f,
            "GainMapPixels::{kind} {{ samples: {}, bits: {} }}",
            self.len(),
            self.bit_depth().bits()
        )
    }
}

/// An HDR gain map to attach to an encode via [`EncodeConfig::with_gain_map`].
///
/// The primary image stays the SDR base rendition. The gain map describes, per
/// pixel, how much brighter the HDR rendition is; its dimensions are
/// independent of the primary image (Apple uses half the primary resolution).
///
/// ```no_run
/// use hpvca::{EncodeConfig, GainMap};
/// # let (rgb, w, h): (Vec<u8>, u32, u32) = (vec![], 0, 0);
/// # let ratios: Vec<f32> = vec![];
/// // `ratios[i]` = HDR / SDR linear luminance at gain-map pixel `i`.
/// let gain_map = GainMap::from_ratios(&ratios, w / 2, h / 2)?.with_quality(80);
/// let heic = hpvca::encode_rgb(&rgb, w, h, &EncodeConfig::new().with_gain_map(gain_map))?;
/// # Ok::<(), hpvca::EncodeError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GainMap {
    /// Gain map width in pixels.
    pub width: u32,
    /// Gain map height in pixels.
    pub height: u32,
    /// Apple-encoded gain samples (see the [module docs](self)).
    pub pixels: GainMapPixels,
    /// Linear HDR headroom: the peak HDR/SDR ratio a full-scale sample
    /// reaches. Must be finite and greater than 1.
    pub headroom: f32,
    /// Quality of the gain map image. `None` inherits the primary's quality.
    pub quality: Option<u8>,
    /// Encode the gain map image losslessly. Default false.
    pub lossless: bool,
    /// Also write an ISO 21496-1 `tmap` item. Default true.
    pub iso: bool,
    /// ISO metadata written into the `tmap` item. `None` fits it to Apple's
    /// curve for [`headroom`](Self::headroom); set it only if you know the
    /// samples follow a different curve under ISO readers.
    pub iso_metadata: Option<IsoGainMap>,
    /// Color of the alternate (HDR) rendition, written on the `tmap` item.
    /// `None` writes no `colr` there.
    pub alternate_color: Option<ColorMetadata>,
}

impl GainMap {
    fn new(pixels: GainMapPixels, width: u32, height: u32, headroom: f32) -> Self {
        Self {
            width,
            height,
            pixels,
            headroom,
            quality: None,
            lossless: false,
            iso: true,
            iso_metadata: None,
            alternate_color: None,
        }
    }

    /// 8-bit Apple-encoded gain map (`width * height` samples) for an HDR
    /// rendition with the given linear `headroom`.
    pub fn gray8(pixels: Vec<u8>, width: u32, height: u32, headroom: f32) -> Self {
        Self::new(GainMapPixels::Gray8(pixels), width, height, headroom)
    }

    /// 10- or 12-bit Apple-encoded gain map (`width * height` samples).
    pub fn gray16(
        pixels: Vec<u16>,
        bit_depth: BitDepth,
        width: u32,
        height: u32,
        headroom: f32,
    ) -> Self {
        Self::new(
            GainMapPixels::Gray16 {
                data: pixels,
                bit_depth,
            },
            width,
            height,
            headroom,
        )
    }

    /// Build an 8-bit gain map from linear HDR/SDR luminance ratios
    /// (`width * height` values). The headroom is the largest ratio; ratios
    /// below 1 are clamped, since the Apple curve cannot darken.
    pub fn from_ratios(ratios: &[f32], width: u32, height: u32) -> Result<Self, EncodeError> {
        let headroom = ratios
            .iter()
            .copied()
            .filter(|r| r.is_finite())
            .fold(1.0f32, f32::max);
        if headroom <= 1.0 {
            return Err(EncodeError::GainMap(
                "gain map ratios never exceed 1, there is no HDR to encode",
            ));
        }
        let pixels = ratios
            .iter()
            .map(|&r| {
                let r = if r.is_finite() { r } else { 1.0 };
                (apple_encode(r, headroom) * 255.0).round() as u8
            })
            .collect();
        Ok(Self::gray8(pixels, width, height, headroom))
    }

    /// Set the gain map image's own quality (1..=100).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Encode the gain map image losslessly.
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Write (default) or omit the ISO 21496-1 `tmap` item.
    pub fn with_iso(mut self, iso: bool) -> Self {
        self.iso = iso;
        self
    }

    /// Replace the fitted ISO metadata of the `tmap` item.
    pub fn with_iso_metadata(mut self, metadata: IsoGainMap) -> Self {
        self.iso_metadata = Some(metadata);
        self
    }

    /// Color of the alternate (HDR) rendition, written on the `tmap` item.
    pub fn with_alternate_color(mut self, color: ColorMetadata) -> Self {
        self.alternate_color = Some(color);
        self
    }

    /// ISO 21496-1 metadata the `tmap` item will carry: the override if set,
    /// otherwise the fit of Apple's curve for this headroom and bit depth.
    pub fn iso_metadata(&self) -> Result<IsoGainMap, EncodeError> {
        match self.iso_metadata {
            Some(m) => Ok(m),
            None => fit_iso_metadata(self.headroom, self.pixels.bit_depth()),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EncodeError> {
        crate::validate_dims(self.width, self.height)?;
        if crate::exceeds_single_picture(self.width, self.height) {
            return Err(EncodeError::GainMap(
                "gain map is too large for a single HEVC picture",
            ));
        }
        let expected = self.width as usize * self.height as usize;
        if self.pixels.len() != expected {
            return Err(EncodeError::InvalidInput);
        }
        if !self.headroom.is_finite() || self.headroom <= 1.0 {
            return Err(EncodeError::GainMap(
                "gain map headroom must be finite and greater than 1",
            ));
        }
        if let Some(q) = self.quality {
            crate::validate_quality(q)?;
        }
        if let GainMapPixels::Gray16 { data, bit_depth } = &self.pixels {
            let max = (1u16 << bit_depth.bits()) - 1;
            if data.iter().any(|&v| v > max) {
                return Err(EncodeError::InvalidInput);
            }
        }
        if let Some(m) = &self.iso_metadata {
            m.validate()?;
        }
        Ok(())
    }
}

/// Rec.709 OETF, Apple's gain sample encoding.
fn rec709_oetf(v: f64) -> f64 {
    if v < 0.018 {
        4.5 * v
    } else {
        1.099 * v.powf(0.45) - 0.099
    }
}

/// Inverse of [`rec709_oetf`].
fn rec709_eotf(v: f64) -> f64 {
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// Normalized Apple gain sample (`0..=1`) for an HDR/SDR `ratio`.
pub fn apple_encode(ratio: f32, headroom: f32) -> f32 {
    let g = ((ratio as f64 - 1.0) / (headroom as f64 - 1.0)).clamp(0.0, 1.0);
    rec709_oetf(g) as f32
}

/// HDR/SDR ratio Apple's readers apply for a normalized gain `sample`.
pub fn apple_decode(sample: f32, headroom: f32) -> f32 {
    let g = rec709_eotf((sample as f64).clamp(0.0, 1.0));
    (1.0 + (headroom as f64 - 1.0) * g) as f32
}

/// ISO 21496-1 (`use_base_color_space`, forward direction, single channel)
/// approximation of Apple's curve. `min` is pinned to 0 so samples without
/// gain stay exactly SDR under ISO readers too, `max` to `log2(headroom)`, and
/// the gamma minimizes the worst error in stops over every code value.
/// (Apple's own fit at headroom 6.74 is off by up to 0.093 stops; this by 0.074.)
fn fit_iso_metadata(headroom: f32, bit_depth: BitDepth) -> Result<IsoGainMap, EncodeError> {
    let max = (headroom as f64).log2();
    let levels = (1u32 << bit_depth.bits()) - 1;
    let samples: Vec<(f64, f64)> = (0..=levels)
        .map(|i| {
            let s = i as f64 / levels as f64;
            (s, (apple_decode(s as f32, headroom) as f64).log2())
        })
        .collect();
    let worst = |gamma: f64| -> f64 {
        samples
            .iter()
            .map(|&(s, y)| (max * s.powf(1.0 / gamma) - y).abs())
            .fold(0.0, f64::max)
    };
    // Golden-section search; the worst error is unimodal in gamma here.
    let (mut lo, mut hi) = (0.25f64, 4.0f64);
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let a = hi - phi * (hi - lo);
        let b = lo + phi * (hi - lo);
        if worst(a) < worst(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    let gamma = (lo + hi) / 2.0;
    let min = 0.0;

    let d = ISO_DENOMINATOR;
    let q = |v: f64| (v * d as f64).round();
    let (min_n, max_n, gamma_n) = (q(min), q(max), q(gamma));
    if !(i32::MIN as f64..=i32::MAX as f64).contains(&min_n)
        || !(1.0..=i32::MAX as f64).contains(&max_n)
        || !(1.0..=i32::MAX as f64).contains(&gamma_n)
    {
        return Err(EncodeError::GainMap(
            "gain map headroom cannot be represented as ISO metadata",
        ));
    }
    // Apple writes the same tiny offsets; they keep black finite in log space.
    let offset = 10;
    Ok(IsoGainMap {
        gain_map_min_n: [min_n as i32; 3],
        gain_map_min_d: [d; 3],
        gain_map_max_n: [max_n as i32; 3],
        gain_map_max_d: [d; 3],
        gain_map_gamma_n: [gamma_n as u32; 3],
        gain_map_gamma_d: [d; 3],
        base_offset_n: [offset; 3],
        base_offset_d: [d; 3],
        alternate_offset_n: [offset; 3],
        alternate_offset_d: [d; 3],
        base_hdr_headroom_n: 0,
        base_hdr_headroom_d: d,
        alternate_hdr_headroom_n: max_n as u32,
        alternate_hdr_headroom_d: d,
        backward_direction: false,
        use_base_color_space: true,
    })
}

/// Binary ISO 21496-1 gain map metadata, as carried by a HEIF `tmap` item.
///
/// Fields are rationals: `*_n` is the numerator, `*_d` the denominator.
/// Per-channel arrays hold one value per RGB channel; when all three are equal
/// the writer emits the compact single-channel form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsoGainMap {
    /// log2 of the minimum gain (per channel).
    pub gain_map_min_n: [i32; 3],
    pub gain_map_min_d: [u32; 3],
    /// log2 of the maximum gain (per channel).
    pub gain_map_max_n: [i32; 3],
    pub gain_map_max_d: [u32; 3],
    /// Gamma applied to the stored gain map samples (per channel).
    pub gain_map_gamma_n: [u32; 3],
    pub gain_map_gamma_d: [u32; 3],
    /// Offset added to the base image before applying the gain (per channel).
    pub base_offset_n: [i32; 3],
    pub base_offset_d: [u32; 3],
    /// Offset added to the alternate image before applying the gain (per channel).
    pub alternate_offset_n: [i32; 3],
    pub alternate_offset_d: [u32; 3],
    /// log2 of the base image HDR headroom.
    pub base_hdr_headroom_n: u32,
    pub base_hdr_headroom_d: u32,
    /// log2 of the alternate image HDR headroom.
    pub alternate_hdr_headroom_n: u32,
    pub alternate_hdr_headroom_d: u32,
    /// True when the gain map maps from the alternate image back to the base.
    pub backward_direction: bool,
    /// True when the gain map is applied in the base image color space.
    pub use_base_color_space: bool,
}

impl Default for IsoGainMap {
    /// Identity gain map: gain 1 everywhere, gamma 1, no offsets, no headroom.
    fn default() -> Self {
        Self {
            gain_map_min_n: [0; 3],
            gain_map_min_d: [1; 3],
            gain_map_max_n: [0; 3],
            gain_map_max_d: [1; 3],
            gain_map_gamma_n: [1; 3],
            gain_map_gamma_d: [1; 3],
            base_offset_n: [0; 3],
            base_offset_d: [1; 3],
            alternate_offset_n: [0; 3],
            alternate_offset_d: [1; 3],
            base_hdr_headroom_n: 0,
            base_hdr_headroom_d: 1,
            alternate_hdr_headroom_n: 0,
            alternate_hdr_headroom_d: 1,
            backward_direction: false,
            use_base_color_space: true,
        }
    }
}

const IS_MULTICHANNEL_MASK: u8 = 1 << 7;
const USE_BASE_COLORSPACE_MASK: u8 = 1 << 6;
const BACKWARD_DIRECTION_MASK: u8 = 1 << 2;
const USE_COMMON_DENOMINATOR_MASK: u8 = 1 << 3;

fn bad(msg: &'static str) -> EncodeError {
    EncodeError::GainMap(msg)
}

fn read_u32(arr: &[u8], pos: &mut usize) -> Result<u32, EncodeError> {
    let s = arr
        .get(*pos..*pos + 4)
        .ok_or_else(|| bad("gain map metadata truncated"))?;
    *pos += 4;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

impl IsoGainMap {
    /// True when at least one per-channel parameter differs between channels.
    pub fn is_multichannel(&self) -> bool {
        fn same<T: PartialEq>(n: &[T; 3], d: &[u32; 3]) -> bool {
            n[0] == n[1] && n[1] == n[2] && d[0] == d[1] && d[1] == d[2]
        }
        !(same(&self.gain_map_min_n, &self.gain_map_min_d)
            && same(&self.gain_map_max_n, &self.gain_map_max_d)
            && same(&self.gain_map_gamma_n, &self.gain_map_gamma_d)
            && same(&self.base_offset_n, &self.base_offset_d)
            && same(&self.alternate_offset_n, &self.alternate_offset_d))
    }

    fn validate(&self) -> Result<(), EncodeError> {
        if self.base_hdr_headroom_d == 0 || self.alternate_hdr_headroom_d == 0 {
            return Err(bad("zero denominator in HDR headroom"));
        }
        for c in 0..3 {
            if self.gain_map_min_d[c] == 0
                || self.gain_map_max_d[c] == 0
                || self.gain_map_gamma_d[c] == 0
                || self.base_offset_d[c] == 0
                || self.alternate_offset_d[c] == 0
            {
                return Err(bad("zero denominator in per-channel gain map parameter"));
            }
            if self.gain_map_gamma_n[c] == 0 {
                return Err(bad("gain map gamma must be non-zero"));
            }
        }
        Ok(())
    }

    /// Serialize to the ISO 21496-1 binary layout (big-endian):
    ///
    /// ```text
    /// u16 minimum_version = 0
    /// u16 writer_version  = 0
    /// u8  flags: bit7 multichannel, bit6 use_base_color_space,
    ///            bit2 backward_direction
    /// base_hdr_headroom, alternate_hdr_headroom        (u32 n, u32 d)
    /// per channel (1 or 3): min, max, gamma, base_offset, alternate_offset
    /// ```
    ///
    /// Denominators are always written explicitly: Apple ImageIO ignores a
    /// `tmap` whose metadata uses the common-denominator form (flag bit 3).
    pub fn to_metadata(&self) -> Result<Vec<u8>, EncodeError> {
        self.validate()?;
        let channels = if self.is_multichannel() { 3 } else { 1 };

        let mut flags = 0u8;
        if channels == 3 {
            flags |= IS_MULTICHANNEL_MASK;
        }
        if self.use_base_color_space {
            flags |= USE_BASE_COLORSPACE_MASK;
        }
        if self.backward_direction {
            flags |= BACKWARD_DIRECTION_MASK;
        }

        let mut out = Vec::with_capacity(5 + 16 + 40 * channels);
        out.extend_from_slice(&0u16.to_be_bytes()); // minimum_version
        out.extend_from_slice(&0u16.to_be_bytes()); // writer_version
        out.push(flags);
        let mut put = |v: u32| out.extend_from_slice(&v.to_be_bytes());
        put(self.base_hdr_headroom_n);
        put(self.base_hdr_headroom_d);
        put(self.alternate_hdr_headroom_n);
        put(self.alternate_hdr_headroom_d);
        for c in 0..channels {
            put(self.gain_map_min_n[c] as u32);
            put(self.gain_map_min_d[c]);
            put(self.gain_map_max_n[c] as u32);
            put(self.gain_map_max_d[c]);
            put(self.gain_map_gamma_n[c]);
            put(self.gain_map_gamma_d[c]);
            put(self.base_offset_n[c] as u32);
            put(self.base_offset_d[c]);
            put(self.alternate_offset_n[c] as u32);
            put(self.alternate_offset_d[c]);
        }
        Ok(out)
    }

    /// Parse the ISO 21496-1 binary layout. Single-channel metadata is
    /// expanded to three identical channels.
    pub fn from_metadata(data: &[u8]) -> Result<Self, EncodeError> {
        if data.len() < 5 {
            return Err(bad("gain map metadata too short"));
        }
        if u16::from_be_bytes([data[0], data[1]]) != 0 {
            return Err(bad("unsupported gain map metadata minimum version"));
        }
        let flags = data[4];
        let mut pos = 5;
        let channels = if flags & IS_MULTICHANNEL_MASK != 0 {
            3
        } else {
            1
        };
        let mut m = IsoGainMap {
            use_base_color_space: flags & USE_BASE_COLORSPACE_MASK != 0,
            backward_direction: flags & BACKWARD_DIRECTION_MASK != 0,
            ..IsoGainMap::default()
        };
        let common = if flags & USE_COMMON_DENOMINATOR_MASK != 0 {
            Some(read_u32(data, &mut pos)?)
        } else {
            None
        };
        // One rational: the numerator, then the denominator unless shared.
        let rational = |pos: &mut usize| -> Result<(u32, u32), EncodeError> {
            let n = read_u32(data, pos)?;
            let d = match common {
                Some(d) => d,
                None => read_u32(data, pos)?,
            };
            Ok((n, d))
        };
        (m.base_hdr_headroom_n, m.base_hdr_headroom_d) = rational(&mut pos)?;
        (m.alternate_hdr_headroom_n, m.alternate_hdr_headroom_d) = rational(&mut pos)?;
        for c in 0..channels {
            let (n, d) = rational(&mut pos)?;
            (m.gain_map_min_n[c], m.gain_map_min_d[c]) = (n as i32, d);
            let (n, d) = rational(&mut pos)?;
            (m.gain_map_max_n[c], m.gain_map_max_d[c]) = (n as i32, d);
            (m.gain_map_gamma_n[c], m.gain_map_gamma_d[c]) = rational(&mut pos)?;
            let (n, d) = rational(&mut pos)?;
            (m.base_offset_n[c], m.base_offset_d[c]) = (n as i32, d);
            let (n, d) = rational(&mut pos)?;
            (m.alternate_offset_n[c], m.alternate_offset_d[c]) = (n as i32, d);
        }
        for c in channels..3 {
            m.gain_map_min_n[c] = m.gain_map_min_n[0];
            m.gain_map_min_d[c] = m.gain_map_min_d[0];
            m.gain_map_max_n[c] = m.gain_map_max_n[0];
            m.gain_map_max_d[c] = m.gain_map_max_d[0];
            m.gain_map_gamma_n[c] = m.gain_map_gamma_n[0];
            m.gain_map_gamma_d[c] = m.gain_map_gamma_d[0];
            m.base_offset_n[c] = m.base_offset_n[0];
            m.base_offset_d[c] = m.base_offset_d[0];
            m.alternate_offset_n[c] = m.alternate_offset_n[0];
            m.alternate_offset_d[c] = m.alternate_offset_d[0];
        }
        m.validate()?;
        Ok(m)
    }
}

/// A gain map already coded, ready for the container writers.
pub(crate) struct EncodedGainMap {
    pub(crate) stream: NaluStream,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bit_depth: BitDepth,
    pub(crate) cicp: Cicp,
    /// Apple `HDRGainMap` XMP packet, stored as an `application/rdf+xml` item.
    pub(crate) xmp: Vec<u8>,
    /// `tmap` item payload (`version` byte + ISO metadata), if requested.
    pub(crate) tmap: Option<Vec<u8>>,
    pub(crate) alternate_color: Option<ColorMetadata>,
}

/// Gain samples are coded like Apple's: full range, nothing else signalled.
const GAIN_MAP_CICP: Cicp = Cicp {
    primaries: Primaries::Unspecified,
    transfer: TransferFunction::Unspecified,
    matrix: MatrixCoefficients::Unspecified,
    full_range: true,
};

fn apple_xmp(headroom: f32) -> Vec<u8> {
    format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="hpvca">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about=""
            xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">
         <HDRGainMap:HDRGainMapVersion>{APPLE_GAIN_MAP_VERSION}</HDRGainMap:HDRGainMapVersion>
         <HDRGainMap:HDRGainMapHeadroom>{headroom:.6}</HDRGainMap:HDRGainMapHeadroom>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>"#
    )
    .into_bytes()
}

/// Encode the gain map attached to `cfg` (if any). The gain map image is a
/// single monochrome picture coded with the primary's speed, SAO, threading and
/// Rice settings; it never uses the screen-content tools.
pub(crate) fn encode_gain_map(cfg: &EncodeConfig) -> Result<Option<EncodedGainMap>, EncodeError> {
    let Some(gm) = cfg.gain_map.as_ref() else {
        return Ok(None);
    };
    gm.validate()?;
    let bit_depth = gm.pixels.bit_depth();
    let y: Vec<u16> = match &gm.pixels {
        GainMapPixels::Gray8(p) => p.iter().map(|&v| v as u16).collect(),
        GainMapPixels::Gray16 { data, .. } => data.clone(),
    };
    let yuv = Yuv {
        y,
        cb: Vec::new(),
        cr: Vec::new(),
        width: gm.width,
        height: gm.height,
        display_w: gm.width,
        display_h: gm.height,
        chroma: ChromaFormat::Monochrome,
        bit_depth,
    };
    let stream = hevc::encode_intra_opts(
        &yuv,
        gm.width,
        gm.height,
        gm.quality.unwrap_or(cfg.quality),
        gm.lossless,
        Some(GAIN_MAP_CICP),
        cfg.parallelism.single_wpp(),
        false,
        crate::resolve_threads(cfg.threads),
        cfg.sao,
        cfg.speed,
        None,
        0,
        None,
        false,
        cfg.implicit_rdpcm,
        cfg.persistent_rice,
    )?;
    let tmap = if gm.iso {
        let mut payload = vec![0u8]; // ToneMapImage version
        payload.extend_from_slice(&gm.iso_metadata()?.to_metadata()?);
        Some(payload)
    } else {
        None
    };
    Ok(Some(EncodedGainMap {
        stream,
        width: gm.width,
        height: gm.height,
        bit_depth,
        cicp: GAIN_MAP_CICP,
        xmp: apple_xmp(gm.headroom),
        tmap,
        alternate_color: gm.alternate_color.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_curve_round_trips() {
        for h in [1.5f32, 4.0, 6.736285, 16.0] {
            for r in [1.0f32, 1.2, h * 0.5, h] {
                let r = r.max(1.0);
                let back = apple_decode(apple_encode(r, h), h);
                assert!((back - r).abs() < 1e-4 * h, "h={h} r={r} back={back}");
            }
        }
    }

    #[test]
    fn iso_fit_tracks_apple_curve() {
        // Apple's own fit for this headroom has a max error of ~0.093 stops.
        let h = 6.736285f32;
        let m = fit_iso_metadata(h, BitDepth::Eight).unwrap();
        let d = ISO_DENOMINATOR as f64;
        let min = m.gain_map_min_n[0] as f64 / d;
        let max = m.gain_map_max_n[0] as f64 / d;
        let gamma = m.gain_map_gamma_n[0] as f64 / d;
        assert!((max - (h as f64).log2()).abs() < 1e-6);
        assert_eq!(m.alternate_hdr_headroom_n as i32, m.gain_map_max_n[0]);
        let worst = (0..=255)
            .map(|i| {
                let s = i as f64 / 255.0;
                let t = s.powf(1.0 / gamma);
                let iso = min * (1.0 - t) + max * t;
                (iso - (apple_decode(s as f32, h) as f64).log2()).abs()
            })
            .fold(0.0, f64::max);
        assert_eq!(min, 0.0);
        assert!(worst < 0.075, "worst error {worst} stops");
    }

    #[test]
    fn iso_metadata_matches_apple_layout() {
        // Same layout as the tmap item of an iPhone HEIC (headroom 6.736285).
        let m = fit_iso_metadata(6.736285, BitDepth::Eight).unwrap();
        let bytes = m.to_metadata().unwrap();
        assert_eq!(&bytes[..5], &[0, 0, 0, 0, USE_BASE_COLORSPACE_MASK]);
        assert_eq!(bytes.len(), 5 + 16 + 40);
        assert_eq!(IsoGainMap::from_metadata(&bytes).unwrap(), m);
    }

    #[test]
    fn parses_iphone_tmap_metadata() {
        let apple: [u8; 61] = [
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x42, 0x40, 0x00,
            0x29, 0xfd, 0xd1, 0x00, 0x0f, 0x42, 0x40, 0xff, 0xff, 0xf3, 0x5d, 0x00, 0x0f, 0x42,
            0x40, 0x00, 0x29, 0xfd, 0xd1, 0x00, 0x0f, 0x42, 0x40, 0x00, 0x0d, 0x95, 0x19, 0x00,
            0x0f, 0x42, 0x40, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x0f, 0x42, 0x40, 0x00, 0x00, 0x00,
            0x0a, 0x00, 0x0f, 0x42, 0x40,
        ];
        let m = IsoGainMap::from_metadata(&apple).unwrap();
        assert_eq!(m.gain_map_min_n, [-3235; 3]);
        assert_eq!(m.gain_map_max_n, [2_751_953; 3]);
        assert_eq!(m.gain_map_gamma_n, [890_137; 3]);
        assert_eq!(m.alternate_hdr_headroom_n, 2_751_953);
        assert!(m.use_base_color_space && !m.backward_direction);
        assert_eq!(m.to_metadata().unwrap(), apple);
    }

    #[test]
    fn parses_common_denominator_form() {
        // libultrahdr-style compact blob: d = 4, one channel.
        let mut b = vec![0, 0, 0, 0, USE_COMMON_DENOMINATOR_MASK];
        for v in [4u32, 0, 8, (-2i32) as u32, 6, 4, 0, 0] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        let m = IsoGainMap::from_metadata(&b).unwrap();
        assert_eq!((m.gain_map_min_n, m.gain_map_min_d), ([-2; 3], [4; 3]));
        assert_eq!(
            (m.alternate_hdr_headroom_n, m.alternate_hdr_headroom_d),
            (8, 4)
        );
        assert!(!m.use_base_color_space);
        // Written back with explicit denominators.
        assert_eq!(
            IsoGainMap::from_metadata(&m.to_metadata().unwrap()).unwrap(),
            m
        );
    }

    #[test]
    fn from_ratios_derives_headroom() {
        let gm = GainMap::from_ratios(&[1.0, 2.0, 4.0, 0.5], 2, 2).unwrap();
        assert_eq!(gm.headroom, 4.0);
        let GainMapPixels::Gray8(p) = &gm.pixels else {
            unreachable!()
        };
        assert_eq!(p[0], 0);
        assert_eq!(p[2], 255);
        assert_eq!(p[3], 0);
        assert!(GainMap::from_ratios(&[1.0, 0.5], 2, 1).is_err());
    }

    #[test]
    fn validation_catches_bad_gain_maps() {
        assert!(GainMap::gray8(vec![0; 3], 2, 2, 2.0).validate().is_err());
        assert!(GainMap::gray8(vec![0; 4], 2, 2, 1.0).validate().is_err());
        assert!(GainMap::gray8(vec![0; 4], 2, 2, 2.0).validate().is_ok());
        assert!(
            GainMap::gray16(vec![1024; 4], BitDepth::Ten, 2, 2, 2.0)
                .validate()
                .is_err()
        );
        assert!(
            GainMap::gray8(vec![0; 4], 2, 2, 2.0)
                .with_quality(0)
                .validate()
                .is_err()
        );
    }
}
