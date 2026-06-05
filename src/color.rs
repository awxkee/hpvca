//! Colour signalling for the HEIF `colr` box and the HEVC VUI.
//!
//! Two forms are supported, mirroring the `colr` box's two `colour_type`s:
//!   - `nclx` — CICP enumerated coding-independent code points (primaries,
//!     transfer characteristics, matrix coefficients, full/limited range).
//!   - `prof` — an embedded ICC profile (raw bytes).
//!
//! The CICP values follow ISO/IEC 23091-2 (identical to the HEVC VUI enums), so a
//! single [`ColorEncoding`] also drives the VUI `colour_primaries`,
//! `transfer_characteristics`, `matrix_coeffs`, and `video_full_range_flag`.

/// CICP colour primaries (ISO/IEC 23091-2 Table 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Primaries {
    /// BT.709 / sRGB primaries (value 1).
    Bt709 = 1,
    /// Unspecified (value 2).
    Unspecified = 2,
    /// BT.2020 primaries (value 9).
    Bt2020 = 9,
    /// DCI-P3 / Display-P3 primaries (value 12, "P3-D65" via SMPTE RP 431-2 set 11→
    /// commonly signalled as 12 = SMPTE 432 Display-P3).
    DisplayP3 = 12,
}

/// CICP transfer characteristics (ISO/IEC 23091-2 Table 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferFunction {
    /// BT.709 transfer (value 1).
    Bt709 = 1,
    /// Unspecified (value 2).
    Unspecified = 2,
    /// sRGB / IEC 61966-2-1 transfer (value 13).
    Srgb = 13,
    /// SMPTE ST 2084 (PQ) — HDR10 (value 16).
    Pq = 16,
    /// ARIB STD-B67 (HLG) (value 18).
    Hlg = 18,
}

/// CICP matrix coefficients for RGB→YCbCr (ISO/IEC 23091-2 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixCoefficients {
    /// Identity — samples are GBR, no colour transform (value 0).
    Identity = 0,
    /// BT.709 (value 1).
    Bt709 = 1,
    /// Unspecified (value 2).
    Unspecified = 2,
    /// BT.2020 non-constant luminance (value 9).
    Bt2020Ncl = 9,
}

/// Whether sample values use the full code range (`true`, JPEG-style 0..=max) or the
/// studio/limited range (`false`, 16..235-style scaled to the bit depth).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullRange(pub bool);

/// CICP-style colour encoding: the primaries + transfer + matrix the image is
/// authored in, plus the sample range. Drives both the HEIF `colr` (nclx) box and
/// the HEVC VUI signalling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorEncoding {
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub matrix: MatrixCoefficients,
    pub full_range: bool,
}

impl ColorEncoding {
    /// sRGB: BT.709 primaries, sRGB transfer, BT.709 matrix, full range. This is the
    /// encoder's working colour space (full-range BT.709 RGB→YCbCr).
    pub const fn srgb() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }

    /// BT.709 video: BT.709 primaries/transfer/matrix, full range.
    pub const fn bt709() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }

    /// BT.2020 PQ (HDR10): BT.2020 primaries, PQ transfer, BT.2020 NCL matrix.
    pub const fn bt2020_pq() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Pq,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: true,
        }
    }

    /// The `nclx` payload for a HEIF `colr` box (without the box header):
    /// `colour_type` ('nclx') + the four CICP fields. The `full_range_flag` occupies
    /// the top bit of the final byte; the low 7 bits are reserved zero.
    pub fn nclx_payload(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(11);
        p.extend_from_slice(b"nclx");
        p.extend_from_slice(&(self.primaries as u16).to_be_bytes());
        p.extend_from_slice(&(self.transfer as u16).to_be_bytes());
        p.extend_from_slice(&(self.matrix as u16).to_be_bytes());
        p.push(if self.full_range { 0x80 } else { 0x00 });
        p
    }
}

impl Default for ColorEncoding {
    fn default() -> Self {
        ColorEncoding::srgb()
    }
}

/// How the output colour space is described in the file. Either an enumerated CICP
/// encoding (compact `nclx`) or an embedded ICC profile (`prof`).
///
/// Apple's ImageIO has historically rendered some `nclx`-only HEICs as black, so the
/// default ([`ColorMetadata::icc_srgb`]) embeds an sRGB ICC profile. Use
/// [`ColorMetadata::Cicp`] for compact CICP signalling (e.g. HDR) when the target
/// decoder honours it.
#[derive(Clone, Debug)]
pub enum ColorMetadata {
    /// Enumerated CICP code points → `colr` box of type `nclx`.
    Cicp(ColorEncoding),
    /// Embedded ICC profile bytes → `colr` box of type `prof`.
    Icc(Vec<u8>),
}

impl ColorMetadata {
    /// The default sRGB ICC profile (what libheif embeds), for broad compatibility.
    pub fn icc_srgb() -> Self {
        ColorMetadata::Icc(crate::icc_profile::SRGB_ICC.to_vec())
    }

    /// The colour-authoring encoding these metadata imply, used to drive the VUI.
    /// An ICC profile is treated as sRGB for VUI purposes (the working space).
    pub fn color_encoding(&self) -> ColorEncoding {
        match self {
            ColorMetadata::Cicp(c) => *c,
            ColorMetadata::Icc(_) => ColorEncoding::srgb(),
        }
    }
}

impl Default for ColorMetadata {
    fn default() -> Self {
        ColorMetadata::icc_srgb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nclx_payload_layout() {
        let p = ColorEncoding::bt709().nclx_payload();
        assert_eq!(&p[0..4], b"nclx");
        assert_eq!(u16::from_be_bytes([p[4], p[5]]), 1); // BT709 primaries
        assert_eq!(u16::from_be_bytes([p[6], p[7]]), 1); // BT709 transfer
        assert_eq!(u16::from_be_bytes([p[8], p[9]]), 1); // BT709 matrix
        assert_eq!(p[10] >> 7, 1); // full range
        assert_eq!(p.len(), 11);
    }

    #[test]
    fn srgb_default_is_icc() {
        match ColorMetadata::default() {
            ColorMetadata::Icc(bytes) => assert!(!bytes.is_empty()),
            _ => panic!("default should be ICC sRGB"),
        }
    }

    #[test]
    fn pq_encoding_values() {
        let e = ColorEncoding::bt2020_pq();
        assert_eq!(e.primaries as u8, 9);
        assert_eq!(e.transfer as u8, 16);
        assert_eq!(e.matrix as u8, 9);
    }
}
