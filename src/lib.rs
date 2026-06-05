mod cabac;
pub mod dct;
pub mod deblock;
pub mod error;
pub mod hevc;
pub mod hevc_transform;
mod icc_profile;
mod intra;
pub mod isobmff;
pub mod yuv;

pub use error::EncodeError;

/// Build identifier — bump when the encoder changes. Print this from your binary
/// (e.g. `eprintln!("hpvca {}", hpvca::BUILD_ID)`) to confirm you compiled the
/// latest source and aren't running a stale checkpoint.
pub const BUILD_ID: &str = "apple-compat-2026-06-04: 64x64 CTB, IDR_N_LP(20), \
slice-only mdat, iloc-before-iinf, array_completeness=0, BT.709 full-range, \
colr nclx present, even-dim rounding, level-scales-with-size, PTL-frame-only-constraints, 64-mult-full-CTB, DPB3, ICC-v2-colr-profile, SAO-enabled, x265-aligned-PTL-RExt-transform32-mvp-intrasmooth-loopfilter";

/// Encode an RGB image to HEIC bytes
pub fn encode_heic(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, EncodeError> {
    // 4:2:0 HEVC can only signal even visible dimensions (the conformance window
    // crops in chroma units of 2). Apple VideoToolbox requires the SPS-cropped
    // size and the HEIF 'ispe' box to match exactly. For odd requested sizes we
    // round up to even — the encoded image includes one extra real row/column
    // rather than producing an inconsistent SPS/ispe pair that Apple rejects.
    let enc_w = (width + 1) & !1;
    let enc_h = (height + 1) & !1;

    // 1. RGB → YCbCr 4:2:0. Pad to even dimensions if needed (replicate edge).
    let yuv = if enc_w != width || enc_h != height {
        let padded = pad_rgb_to_even(rgb, width, height, enc_w, enc_h);
        yuv::rgb_to_yuv420(&padded, enc_w, enc_h)
    } else {
        yuv::rgb_to_yuv420(rgb, width, height)
    };

    // 2. HEVC-encode each plane (intra-only, I-frame)
    let nalu_stream = hevc::encode_intra(&yuv, enc_w, enc_h, quality)?;

    // 3. Wrap in ISO Base Media File Format (HEIC brand). ispe matches the SPS.
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
