//! YCbCr conversion from packed RGB, supporting 4:0:0/4:2:0/4:2:2/4:4:4 and 8/10-bit.
//!
//! Uses full-range BT.709 coefficients (consistent with the SPS VUI and the HEIF
//! colr ICC profile):
//!   Y  =  0.2126 R + 0.7152 G + 0.0722 B
//!   Cb = 128 + (B - Y) / 1.8556   (then scaled to the target bit depth)
//!   Cr = 128 + (R - Y) / 1.5748
//!
//! Samples are stored as u16 so 8-bit (0..255) and 10-bit (0..1023) share one path.
//! 8-bit RGB input is scaled to the target depth by bit-replication
//! (v10 = (v8<<2)|(v8>>6)), which maps 255→1023 exactly and is the standard upscale.

use crate::fmt::{BitDepth, ChromaFormat};

/// Planar YCbCr image. Samples are u16 (valid range depends on bit depth).
pub struct Yuv {
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
    pub width: u32,
    pub height: u32,
    pub chroma: ChromaFormat,
    pub bit_depth: BitDepth,
}

impl Yuv {
    pub fn luma_stride(&self) -> usize {
        self.width as usize
    }
    pub fn chroma_stride(&self) -> usize {
        (self.width as usize + self.chroma.sub_w() - 1) / self.chroma.sub_w()
    }
    pub fn chroma_height(&self) -> usize {
        (self.height as usize + self.chroma.sub_h() - 1) / self.chroma.sub_h()
    }
}

/// Convert planar RGB samples to planar YCbCr in the requested chroma format.
///
/// `rgb` holds one `u16` per channel (R,G,B interleaved), already at the target
/// `bit_depth`'s native range — i.e. 0..=255 for 8-bit, 0..=1023 for 10-bit. The
/// library does not guess or rescale the input range; a caller with an 8-bit source
/// that wants a 10-bit encode upscales the samples itself before calling.
///
/// Uses full-range BT.709: Y = 0.2126R + 0.7152G + 0.0722B, and chroma centred at
/// `neutral = 2^(bit_depth-1)` with the standard 1.8556 / 1.5748 divisors (which
/// operate on native-range sample differences, so they are bit-depth independent).
pub fn rgb_to_yuv(
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

    let mut y_plane = vec![0u16; w * h];
    for row in 0..h {
        for col in 0..w {
            let base = (row * w + col) * 3;
            let (r, g, b) = (rgb[base] as f32, rgb[base + 1] as f32, rgb[base + 2] as f32);
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            y_plane[row * w + col] = y.round().clamp(0.0, maxv) as u16;
        }
    }

    if chroma.is_monochrome() {
        return Yuv {
            y: y_plane,
            cb: Vec::new(),
            cr: Vec::new(),
            width,
            height,
            chroma,
            bit_depth,
        };
    }

    let sw = chroma.sub_w();
    let sh = chroma.sub_h();
    let cw = (w + sw - 1) / sw;
    let ch = (h + sh - 1) / sh;
    let mut cb_plane = vec![0u16; cw * ch];
    let mut cr_plane = vec![0u16; cw * ch];

    for crow in 0..ch {
        for ccol in 0..cw {
            let mut sum_cb = 0.0f32;
            let mut sum_cr = 0.0f32;
            let mut count = 0u32;
            for dy in 0..sh {
                for dx in 0..sw {
                    let row = crow * sh + dy;
                    let col = ccol * sw + dx;
                    if row < h && col < w {
                        let base = (row * w + col) * 3;
                        let (r, g, b) =
                            (rgb[base] as f32, rgb[base + 1] as f32, rgb[base + 2] as f32);
                        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        // Native-range chroma, centred at the bit-depth neutral.
                        sum_cb += neutral + (b - y) / 1.8556;
                        sum_cr += neutral + (r - y) / 1.5748;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                cb_plane[crow * cw + ccol] =
                    (sum_cb / count as f32).round().clamp(0.0, maxv) as u16;
                cr_plane[crow * cw + ccol] =
                    (sum_cr / count as f32).round().clamp(0.0, maxv) as u16;
            }
        }
    }

    Yuv {
        y: y_plane,
        cb: cb_plane,
        cr: cr_plane,
        width,
        height,
        chroma,
        bit_depth,
    }
}

/// Backwards-compatible 8-bit 4:2:0 helper (accepts packed 8-bit RGB).
pub fn rgb_to_yuv420(rgb: &[u8], width: u32, height: u32) -> Yuv {
    let wide: Vec<u16> = rgb.iter().map(|&b| b as u16).collect();
    rgb_to_yuv(&wide, width, height, ChromaFormat::Yuv420, BitDepth::Eight)
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
