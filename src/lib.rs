mod cabac;
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

pub use error::EncodeError;
pub use fmt::ChromaFormat;

/// Build identifier — bump when the encoder changes. Print this from your binary
/// (e.g. `eprintln!("hpvca {}", hpvca::BUILD_ID)`) to confirm you compiled the
/// latest source and aren't running a stale checkpoint.
pub const BUILD_ID: &str = "apple-compat-2026-06-04: 64x64 CTB, IDR_N_LP(20), \
slice-only mdat, iloc-before-iinf, array_completeness=0, BT.709 full-range, \
colr nclx present, even-dim rounding, level-scales-with-size, PTL-frame-only-constraints, 64-mult-full-CTB, DPB3, ICC-v2-colr-profile, SAO-enabled, x265-aligned-full, spec-EncodeFlush-termination, 4:0:0+4:2:0+4:2:2+4:4:4-chroma, alpha-aux-item";

/// Encode an RGBA image to HEIC with an alpha channel.
///
/// The color channels are encoded in the requested chroma format; the alpha channel
/// is encoded as a separate monochrome (4:0:0) HEVC image and linked as an auxiliary
/// item per ISO/IEC 23008-12. `rgba` is packed R,G,B,A (4 bytes per pixel).
pub fn encode_heic_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
) -> Result<Vec<u8>, EncodeError> {
    let sw = chroma.sub_w() as u32;
    let sh = chroma.sub_h() as u32;
    let enc_w = (width + sw - 1) / sw * sw;
    let enc_h = (height + sh - 1) / sh * sh;

    // Split RGBA into packed RGB and an alpha plane (replicate-padded to enc dims).
    let (w, h) = (width as usize, height as usize);
    let (nw, nh) = (enc_w as usize, enc_h as usize);
    let mut rgb = vec![0u8; nw * nh * 3];
    let mut alpha_rgb = vec![0u8; nw * nh * 3]; // alpha replicated into RGB for luma path
    for r in 0..nh {
        let sr = r.min(h - 1);
        for c in 0..nw {
            let sc = c.min(w - 1);
            let s = (sr * w + sc) * 4;
            let d = (r * nw + c) * 3;
            rgb[d..d + 3].copy_from_slice(&rgba[s..s + 3]);
            // Alpha → luma. We want the monochrome encoder's luma to equal the alpha
            // value. The luma transform is Y = 0.2126R+0.7152G+0.0722B; feeding equal
            // R=G=B=alpha yields Y≈alpha (full range), so replicate alpha into RGB.
            let a = rgba[s + 3];
            alpha_rgb[d] = a;
            alpha_rgb[d + 1] = a;
            alpha_rgb[d + 2] = a;
        }
    }

    // Color image in the requested chroma format.
    let color_yuv = yuv::rgb_to_yuv(&rgb, enc_w, enc_h, chroma);
    let color_stream = hevc::encode_intra(&color_yuv, enc_w, enc_h, quality)?;

    // Alpha image as monochrome.
    let alpha_yuv = yuv::rgb_to_yuv(&alpha_rgb, enc_w, enc_h, ChromaFormat::Monochrome);
    let alpha_stream = hevc::encode_intra(&alpha_yuv, enc_w, enc_h, quality)?;

    isobmff::wrap_hevc_image_with_alpha(&color_stream, &alpha_stream, enc_w, enc_h)
}

/// Encode an RGB image to HEIC bytes (4:2:0, 8-bit).
pub fn encode_heic(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, EncodeError> {
    encode_heic_fmt(rgb, width, height, quality, ChromaFormat::Yuv420)
}

/// Encode an RGB image to HEIC bytes with an explicit chroma format.
///
/// 4:2:0 and 4:2:2 are supported (8-bit). Both subsample chroma horizontally by
/// two, so the visible width is rounded up to even; 4:2:0 also subsamples
/// vertically, so its height is rounded to even as well. The conformance window
/// crops the padding and the HEIF 'ispe' box matches the SPS-cropped size.
pub fn encode_heic_fmt(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaFormat,
) -> Result<Vec<u8>, EncodeError> {
    // Round visible dimensions up so they are divisible by the chroma subsampling
    // factors (width by sub_w, height by sub_h). This keeps the SPS conformance
    // window, the chroma planes, and the 'ispe' box mutually consistent.
    let sw = chroma.sub_w() as u32;
    let sh = chroma.sub_h() as u32;
    let enc_w = (width + sw - 1) / sw * sw;
    let enc_h = (height + sh - 1) / sh * sh;

    // 1. RGB → YCbCr in the requested chroma format (replicate-pad if rounded).
    let yuv = if enc_w != width || enc_h != height {
        let padded = pad_rgb_to_even(rgb, width, height, enc_w, enc_h);
        yuv::rgb_to_yuv(&padded, enc_w, enc_h, chroma)
    } else {
        yuv::rgb_to_yuv(rgb, width, height, chroma)
    };

    // 2. HEVC-encode (intra-only, single IDR).
    let nalu_stream = hevc::encode_intra(&yuv, enc_w, enc_h, quality)?;

    // 3. Wrap in ISOBMF (HEIC brand). ispe matches the SPS-cropped size.
    let heic = isobmff::wrap_hevc_image(&nalu_stream, enc_w, enc_h)?;

    Ok(heic)
}

/// Replicate-pad packed RGB from (w,h) to (nw,nh) by repeating the last row/column.
fn pad_rgb_to_even(rgb: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    let mut out = vec![0u8; nw * nh * 3];
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
