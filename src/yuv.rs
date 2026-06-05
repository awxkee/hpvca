//! YCbCr 4:2:0 conversion from packed RGB.
//!
//! Uses BT.601 coefficients (the standard for JPEG/still images).

/// Planar YCbCr 4:2:0 representation.
pub struct Yuv420 {
    pub y: Vec<u8>,  // width * height
    pub cb: Vec<u8>, // (width/2) * (height/2)
    pub cr: Vec<u8>, // (width/2) * (height/2)
    pub width: u32,
    pub height: u32,
}

impl Yuv420 {
    pub fn luma_stride(&self) -> usize {
        self.width as usize
    }
    pub fn chroma_stride(&self) -> usize {
        ((self.width as usize) + 1) / 2
    }
    pub fn chroma_height(&self) -> usize {
        ((self.height as usize) + 1) / 2
    }
}

/// Convert packed 8-bit RGB (R,G,B per pixel) to planar YCbCr 4:2:0.
///
/// BT.709 **full-range** coefficients — Y in [0, 255], matching matrix_coefficients=1
/// and full_range_flag=1 signalled in the SPS VUI and the HEIF colr box:
///   Y  =  0.2126 R + 0.7152 G + 0.0722 B
///   Cb = 128 + (B - Y) / 1.8556
///   Cr = 128 + (R - Y) / 1.5748
///
/// The colour pipeline (primaries, transfer, matrix) is fully consistent BT.709 so
/// Apple CoreGraphics/CGImage builds the correct colour space and renders properly.
pub fn rgb_to_yuv420(rgb: &[u8], width: u32, height: u32) -> Yuv420 {
    let w = width as usize;
    let h = height as usize;
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;

    let mut y_plane = vec![0u8; w * h];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    // Full-range BT.709 luma.
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

    for crow in 0..ch {
        for ccol in 0..cw {
            let mut sum_cb = 0.0f32;
            let mut sum_cr = 0.0f32;
            let mut count = 0u32;
            for dy in 0..2usize {
                for dx in 0..2usize {
                    let row = crow * 2 + dy;
                    let col = ccol * 2 + dx;
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

    Yuv420 {
        y: y_plane,
        cb: cb_plane,
        cr: cr_plane,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_pixel() {
        // Pure white → Y=255, Cb≈128, Cr≈128 (full range)
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
    fn dimensions_odd() {
        // 3x3 image — chroma should be 2x2
        let rgb = vec![128u8; 3 * 3 * 3];
        let yuv = rgb_to_yuv420(&rgb, 3, 3);
        assert_eq!(yuv.y.len(), 9);
        assert_eq!(yuv.cb.len(), 4);
        assert_eq!(yuv.cr.len(), 4);
    }
}
