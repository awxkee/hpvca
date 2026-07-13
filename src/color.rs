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

/// CICP color primaries (ISO/IEC 23091-2 Table 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Primaries {
    /// For future use by ITU-T | ISO/IEC
    Reserved,
    /// Rec. ITU-R BT.709-6<br />
    /// Rec. ITU-R BT.1361-0 conventional color gamut system and extended color gamut system (historical)<br />
    /// IEC 61966-2-1 sRGB or sYCC IEC 61966-2-4<br />
    /// Society of Motion Picture and Television Engineers (MPTE) RP 177 (1993) Annex B<br />
    Bt709 = 1,
    /// Unspecified<br />
    /// Image characteristics are unknown or are determined by the application.
    Unspecified = 2,
    /// Rec. ITU-R BT.470-6 System M (historical)<br />
    /// United States National Television System Committee 1953 Recommendation for transmission standards for color television<br />
    /// United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20)<br />
    Bt470M = 4,
    /// Rec. ITU-R BT.470-6 System B, G (historical) Rec. ITU-R BT.601-7 625<br />
    /// Rec. ITU-R BT.1358-0 625 (historical)<br />
    /// Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM<br />
    Bt470Bg = 5,
    /// Rec. ITU-R BT.601-7 525<br />
    /// Rec. ITU-R BT.1358-1 525 or 625 (historical) Rec. ITU-R BT.1700-0 NTSC<br />
    /// SMPTE 170M (2004)<br />
    /// (functionally the same as the value 7)<br />
    Bt601 = 6,
    /// SMPTE 240M (1999) (historical) (functionally the same as the value 6)<br />
    Smpte240 = 7,
    /// Generic film (color filters using Illuminant C)<br />
    GenericFilm = 8,
    /// Rec. ITU-R BT.2020-2<br />
    /// Rec. ITU-R BT.2100-0<br />
    Bt2020 = 9,
    /// SMPTE ST 428-1<br />
    /// (CIE 1931 XYZ as in ISO 11664-1)<br />
    Xyz = 10,
    /// SMPTE RP 431-2 (2011)<br />
    Smpte431 = 11,
    /// SMPTE EG 432-1 (2010)<br />
    Smpte432 = 12,
    /// EBU Tech. 3213-E (1975)<br />
    Ebu3213 = 22,
}

/// CICP transfer characteristics (ISO/IEC 23091-2 Table 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferFunction {
    /// For future use by ITU-T | ISO/IEC
    Reserved,
    /// Rec. ITU-R BT.709-6<br />
    /// Rec. ITU-R BT.1361-0 conventional color gamut system (historical)<br />
    /// (functionally the same as the values 6, 14 and 15)    <br />
    Bt709 = 1,
    /// Image characteristics are unknown or are determined by the application.<br />
    Unspecified = 2,
    /// Rec. ITU-R BT.470-6 System M (historical)<br />
    /// United States National Television System Committee 1953 Recommendation for transmission standards for color television<br />
    /// United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20)<br />
    /// Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM<br />
    Bt470M = 4,
    /// Rec. ITU-R BT.470-6 System B, G (historical)<br />
    Bt470Bg = 5,
    /// Rec. ITU-R BT.601-7 525 or 625<br />
    /// Rec. ITU-R BT.1358-1 525 or 625 (historical)<br />
    /// Rec. ITU-R BT.1700-0 NTSC SMPTE 170M (2004)<br />
    /// (functionally the same as the values 1, 14 and 15)<br />
    Bt601 = 6,
    /// SMPTE 240M (1999) (historical)<br />
    Smpte240 = 7,
    /// Linear transfer characteristics<br />
    Linear = 8,
    /// Logarithmic transfer characteristic (100:1 range)<br />
    Log100 = 9,
    /// Logarithmic transfer characteristic (100 * Sqrt( 10 ) : 1 range)<br />
    Log100sqrt10 = 10,
    /// IEC 61966-2-4<br />
    Iec61966 = 11,
    /// Rec. ITU-R BT.1361-0 extended color gamut system (historical)<br />
    Bt1361 = 12,
    /// IEC 61966-2-1 sRGB or sYCC<br />
    Srgb = 13,
    /// Rec. ITU-R BT.2020-2 (10-bit system)<br />
    /// (functionally the same as the values 1, 6 and 15)<br />
    Bt202010bit = 14,
    /// Rec. ITU-R BT.2020-2 (12-bit system)<br />
    /// (functionally the same as the values 1, 6 and 14)<br />
    Bt202012bit = 15,
    /// SMPTE ST 2084 for 10-, 12-, 14- and 16-bitsystems<br />
    /// Rec. ITU-R BT.2100-0 perceptual quantization (PQ) system<br />
    Smpte2084 = 16,
    /// SMPTE ST 428-1<br />
    Smpte428 = 17,
    /// ARIB STD-B67<br />
    /// Rec. ITU-R BT.2100-0 hybrid log- gamma (HLG) system<br />
    Hlg = 18,
}

/// CICP matrix coefficients for RGB→YCbCr (ISO/IEC 23091-2 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,                // RGB (Identity matrix)
    Bt709 = 1,                   // Rec. 709
    Unspecified = 2,             // Unspecified
    Reserved = 3,                // Reserved
    Fcc = 4,                     // FCC
    Bt470Bg = 5,                 // BT.470BG / BT.601-625
    Smpte170m = 6,               // SMPTE 170M / BT.601-525
    Smpte240m = 7,               // SMPTE 240M
    YCgCo = 8,                   // YCgCo
    Bt2020Ncl = 9,               // BT.2020 (non-constant luminance)
    Bt2020Cl = 10,               // BT.2020 (constant luminance)
    Smpte2085 = 11,              // SMPTE ST 2085
    ChromaticityDerivedNCL = 12, // Chromaticity-derived non-constant luminance
    ChromaticityDerivedCL = 13,  // Chromaticity-derived constant luminance
    ICtCp = 14,                  // ICtCp
    IPtC2 = 15,                  // Color representation developed in SMPTE as IPT-PQ-C2
    YCgCoRe = 16,                // YCgCo-Re (YCgCo-R type even),
    YCgCoRo = 17,                // YCgCo-Ro (YCgCo-R type odd),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub matrix: MatrixCoefficients,
    pub full_range: bool,
}

impl Cicp {
    /// sRGB: BT.709 primaries, sRGB transfer, BT.709 matrix, full range. This is the
    /// encoder's working color space (full-range BT.709 RGB→YCbCr).
    pub const fn srgb() -> Self {
        Cicp {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }

    /// BT.709 video: BT.709 primaries/transfer/matrix, full range.
    pub const fn bt709() -> Self {
        Cicp {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }

    /// BT.2020 PQ (HDR10): BT.2020 primaries, PQ transfer, BT.2020 NCL matrix.
    pub const fn bt2020_pq() -> Self {
        Cicp {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Smpte2084,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: true,
        }
    }

    pub const fn unspecified() -> Self {
        Cicp {
            primaries: Primaries::Unspecified,
            transfer: TransferFunction::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
            full_range: true,
        }
    }

    /// The `nclx` payload for a HEIF `colr` box (without the box header):
    /// `color_type` ('nclx') + the four CICP fields. The `full_range_flag` occupies
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

impl Default for Cicp {
    fn default() -> Self {
        Cicp::unspecified()
    }
}

/// How the output color space is described in the file.
#[derive(Clone, Debug)]
pub struct ColorMetadata {
    pub cicp: Option<Cicp>,
    pub icc: Option<Vec<u8>>,
}

impl ColorMetadata {
    /// CICP-only signaling (`nclx` box).
    pub fn cicp(enc: Cicp) -> Self {
        ColorMetadata {
            cicp: Some(enc),
            icc: None,
        }
    }

    /// ICC-only signaling (`prof` box).
    pub fn icc(profile: Vec<u8>) -> Self {
        ColorMetadata {
            cicp: None,
            icc: Some(profile),
        }
    }

    /// Both an enumerated CICP encoding and an embedded ICC profile. Emits an `nclx`
    /// `colr` box and a `prof` `colr` box.
    pub fn cicp_and_icc(enc: Cicp, profile: Vec<u8>) -> Self {
        ColorMetadata {
            cicp: Some(enc),
            icc: Some(profile),
        }
    }

    /// Set (or replace) the CICP encoding, leaving any ICC profile in place.
    pub fn with_cicp(mut self, enc: Cicp) -> Self {
        self.cicp = Some(enc);
        self
    }

    /// Set (or replace) the ICC profile, leaving any CICP encoding in place.
    pub fn with_icc(mut self, profile: Vec<u8>) -> Self {
        self.icc = Some(profile);
        self
    }

    /// The color-authoring encoding these metadata imply, used to drive the VUI.
    /// Falls back to sRGB when only an ICC profile is present (the working space).
    pub fn color_encoding(&self) -> Cicp {
        self.cicp.unwrap_or_else(Cicp::srgb)
    }
}

impl Default for ColorMetadata {
    fn default() -> Self {
        ColorMetadata::cicp(Cicp::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nclx_payload_layout() {
        let p = Cicp::bt709().nclx_payload();
        assert_eq!(&p[0..4], b"nclx");
        assert_eq!(u16::from_be_bytes([p[4], p[5]]), 1); // BT709 primaries
        assert_eq!(u16::from_be_bytes([p[6], p[7]]), 1); // BT709 transfer
        assert_eq!(u16::from_be_bytes([p[8], p[9]]), 1); // BT709 matrix
        assert_eq!(p[10] >> 7, 1); // full range
        assert_eq!(p.len(), 11);
    }

    #[test]
    fn pq_encoding_values() {
        let e = Cicp::bt2020_pq();
        assert_eq!(e.primaries as u8, 9);
        assert_eq!(e.transfer as u8, 16);
        assert_eq!(e.matrix as u8, 9);
    }
}
