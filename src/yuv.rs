//! YCbCr conversion from packed RGB, supporting 4:2:0 and 4:2:2 subsampling.
//!
//! Uses full-range BT.709 coefficients (consistent with the SPS VUI and the HEIF
//! colr ICC profile):
//!   Y  =  0.2126 R + 0.7152 G + 0.0722 B
//!   Cb = 128 + (B - Y) / 1.8556
//!   Cr = 128 + (R - Y) / 1.5748
//!
//! Sample storage is `u8` (8-bit). For the planned 10-bit extension this becomes
//! `u16`; the conversion math already produces wider intermediates, so the seam
//! is just the output type and the clamp range (255 -> 1023).

use crate::fmt::ChromaFormat;

/// Planar YCbCr image with explicit chroma plane dimensions.
pub struct Yuv {
    pub y: Vec<u8>,  // width * height
    pub cb: Vec<u8>, // chroma_width * chroma_height
    pub cr: Vec<u8>, // chroma_width * chroma_height
    pub width: u32,
    pub height: u32,
    pub chroma: ChromaFormat,
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

/// Convert packed 8-bit RGB to planar YCbCr in the requested chroma format.
/// For monochrome (4:0:0) only the luma plane is produced; cb/cr are empty.
pub fn rgb_to_yuv(rgb: &[u8], width: u32, height: u32, chroma: ChromaFormat) -> Yuv {
    let w = width as usize;
    let h = height as usize;

    let mut y_plane = vec![0u8; w * h];
    // Full-range BT.709 luma. (10-bit seam: widen clamp to 1023 and output u16.)
    for row in 0..h {
        for col in 0..w {
            let base = (row * w + col) * 3;
            let (r, g, b) = (rgb[base] as f32, rgb[base + 1] as f32, rgb[base + 2] as f32);
            let yv = (0.2126 * r + 0.7152 * g + 0.0722 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
            y_plane[row * w + col] = yv;
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
        };
    }

    let sw = chroma.sub_w();
    let sh = chroma.sub_h();
    let cw = (w + sw - 1) / sw;
    let ch = (h + sh - 1) / sh;
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    // Chroma: average over each sw x sh cell.
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
                        sum_cb += 128.0 + (b - y) / 1.8556;
                        sum_cr += 128.0 + (r - y) / 1.5748;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                cb_plane[crow * cw + ccol] =
                    (sum_cb / count as f32).round().clamp(0.0, 255.0) as u8;
                cr_plane[crow * cw + ccol] =
                    (sum_cr / count as f32).round().clamp(0.0, 255.0) as u8;
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
    }
}

/// Backwards-compatible 4:2:0 helper.
pub fn rgb_to_yuv420(rgb: &[u8], width: u32, height: u32) -> Yuv {
    rgb_to_yuv(rgb, width, height, ChromaFormat::Yuv420)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_pixel() {
        let rgb = vec![255u8, 255, 255];
        let yuv = rgb_to_yuv420(&rgb, 1, 1);
        assert!(yuv.y[0] > 250, "Y should be ~255 for white");
        assert!((yuv.cb[0] as i32 - 128).abs() < 5, "Cb~128 for white");
        assert!((yuv.cr[0] as i32 - 128).abs() < 5, "Cr~128 for white");
    }

    #[test]
    fn black_pixel() {
        let rgb = vec![0u8, 0, 0];
        let yuv = rgb_to_yuv420(&rgb, 1, 1);
        assert!(yuv.y[0] < 5, "Y should be ~0 for black (full range)");
    }

    #[test]
    fn dimensions_420() {
        let rgb = vec![128u8; 4 * 4 * 3];
        let yuv = rgb_to_yuv(&rgb, 4, 4, ChromaFormat::Yuv420);
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.cb.len(), 4);
        assert_eq!(yuv.cr.len(), 4);
    }

    #[test]
    fn dimensions_422() {
        let rgb = vec![128u8; 4 * 4 * 3];
        let yuv = rgb_to_yuv(&rgb, 4, 4, ChromaFormat::Yuv422);
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.cb.len(), 8);
        assert_eq!(yuv.cr.len(), 8);
    }

    #[test]
    fn dimensions_monochrome() {
        let rgb = vec![128u8; 4 * 4 * 3];
        let yuv = rgb_to_yuv(&rgb, 4, 4, ChromaFormat::Monochrome);
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.cb.len(), 0); // no chroma
        assert_eq!(yuv.cr.len(), 0);
    }

    #[test]
    fn dimensions_444() {
        let rgb = vec![128u8; 4 * 4 * 3];
        let yuv = rgb_to_yuv(&rgb, 4, 4, ChromaFormat::Yuv444);
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.cb.len(), 16); // full resolution
        assert_eq!(yuv.cr.len(), 16);
    }
}
