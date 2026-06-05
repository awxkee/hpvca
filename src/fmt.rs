//! Pixel-format description: chroma subsampling and bit depth.
//!
//! This module centralises the format parameters so the encoder can support
//! multiple chroma formats (4:2:0, 4:2:2) and — in a later step — higher bit
//! depths (10-bit) without scattering magic numbers through the codebase.

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

/// Full pixel-format description. Bit depth is fixed at 8 for now; the field is
/// present so the 10-bit extension only needs to flip this value and widen the
/// sample storage type (see the `u8`→`u16` seam comments in `yuv.rs` and the
/// transform/quantise paths).
#[derive(Clone, Copy, Debug)]
pub struct PixelFormat {
    pub chroma: ChromaFormat,
    pub bit_depth: u8,
}

impl PixelFormat {
    pub fn new(chroma: ChromaFormat) -> Self {
        PixelFormat { chroma, bit_depth: 8 }
    }
}
