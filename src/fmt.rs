//! Pixel-format description: chroma subsampling and bit depth.
//!
//! This module centralises the format parameters so the encoder can support
//! multiple chroma formats (4:0:0/4:2:0/4:2:2/4:4:4) and bit depths (8/10) without
//! scattering magic numbers through the codebase.

/// Sample bit depth. Owns the invariants that 8- and 10-bit code repeatedly needs,
/// so the magic `1<<(bd-1)` / `(1<<bd)-1` / `6*(bd-8)` arithmetic lives in one place
/// and a depth can never be silently mismatched with the samples it describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDepth {
    Eight,
    Ten,
}

impl BitDepth {
    /// Construct from a raw bit count (8 or 10). Panics on unsupported depths.
    pub fn from_bits(bits: u8) -> Self {
        match bits {
            8 => BitDepth::Eight,
            10 => BitDepth::Ten,
            other => panic!("unsupported bit depth: {other} (only 8 and 10 supported)"),
        }
    }

    /// Bit count (8 or 10).
    pub fn bits(self) -> u8 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
        }
    }

    /// `bit_depth - 8`, the value the SPS/hvcC store as `*_minus8`.
    pub fn minus8(self) -> u8 { self.bits() - 8 }

    /// Maximum representable sample value: `(1 << bits) - 1` (255 or 1023).
    pub fn max_val(self) -> u16 { (1u16 << self.bits()) - 1 }

    /// Neutral / midpoint sample: `1 << (bits - 1)` (128 or 512). Used as the
    /// unavailable-reference default in intra prediction.
    pub fn neutral(self) -> u16 { 1u16 << (self.bits() - 1) }

    /// QpBdOffset = `6 * (bit_depth - 8)` (0 or 12). The decoder dequantises at
    /// `SliceQp + QpBdOffset`, so the encoder must use the same effective QP.
    pub fn qp_bd_offset(self) -> u8 { 6 * self.minus8() }
}

/// Chroma subsampling mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 4:0:0 — monochrome (luma only, no chroma). ChromaArrayType = 0.
    Monochrome,
    /// 4:2:0 — chroma is half width, half height. ChromaArrayType = 1.
    Yuv420,
    /// 4:2:2 — chroma is half width, full height. ChromaArrayType = 2.
    Yuv422,
    /// 4:4:4 — chroma is full width, full height. ChromaArrayType = 3.
    Yuv444,
}

impl ChromaFormat {
    /// `chroma_format_idc` value written in the SPS (HEVC Table 6-1).
    pub fn idc(self) -> u32 {
        match self {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 3,
        }
    }

    /// True when there is no chroma (4:0:0).
    pub fn is_monochrome(self) -> bool {
        matches!(self, ChromaFormat::Monochrome)
    }

    /// Horizontal subsampling factor: luma_width / chroma_width. (1 for monochrome,
    /// which has no chroma; the value is unused but kept well-defined.)
    pub fn sub_w(self) -> usize {
        match self {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 | ChromaFormat::Monochrome => 1,
        }
    }

    /// Vertical subsampling factor: luma_height / chroma_height.
    pub fn sub_h(self) -> usize {
        match self {
            ChromaFormat::Yuv420 => 2,
            ChromaFormat::Yuv422 | ChromaFormat::Yuv444 | ChromaFormat::Monochrome => 1,
        }
    }

    /// Side length of each chroma transform block for an 8×8 luma CU.
    pub fn chroma_tb_size(self) -> usize {
        match self {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => 4,
            ChromaFormat::Yuv444 => 8,
            ChromaFormat::Monochrome => 0,
        }
    }

    /// Number of chroma transform blocks stacked vertically per 8×8 luma CU.
    pub fn chroma_tbs_per_cu(self) -> usize {
        match self {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 1,
        }
    }
}

/// Full pixel-format description: chroma subsampling and sample bit depth.
#[derive(Clone, Copy, Debug)]
pub struct PixelFormat {
    pub chroma: ChromaFormat,
    pub bit_depth: BitDepth,
}

impl PixelFormat {
    pub fn new(chroma: ChromaFormat, bit_depth: BitDepth) -> Self {
        PixelFormat { chroma, bit_depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitdepth_derived_quantities() {
        assert_eq!(BitDepth::Eight.bits(), 8);
        assert_eq!(BitDepth::Ten.bits(), 10);
        assert_eq!(BitDepth::Eight.minus8(), 0);
        assert_eq!(BitDepth::Ten.minus8(), 2);
        assert_eq!(BitDepth::Eight.max_val(), 255);
        assert_eq!(BitDepth::Ten.max_val(), 1023);
        assert_eq!(BitDepth::Eight.neutral(), 128);
        assert_eq!(BitDepth::Ten.neutral(), 512);
        assert_eq!(BitDepth::Eight.qp_bd_offset(), 0);
        assert_eq!(BitDepth::Ten.qp_bd_offset(), 12);
    }

    #[test]
    fn bitdepth_from_bits_roundtrip() {
        assert_eq!(BitDepth::from_bits(8), BitDepth::Eight);
        assert_eq!(BitDepth::from_bits(10), BitDepth::Ten);
    }

    #[test]
    #[should_panic]
    fn bitdepth_rejects_unsupported() {
        let _ = BitDepth::from_bits(12);
    }
}
