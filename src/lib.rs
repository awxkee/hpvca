pub mod yuv;
pub mod dct;
pub mod hevc;
pub mod isobmff;
pub mod error;
mod cabac;
mod intra;
mod hevc_transform;
mod deblock;

pub use error::EncodeError;

/// Encode an RGB image to HEIC bytes
pub fn encode_heic(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, EncodeError> {
    // 1. RGB → YCbCr 4:2:0
    let yuv = yuv::rgb_to_yuv420(rgb, width, height);

    // 2. HEVC-encode each plane (intra-only, I-frame)
    let nalu_stream = hevc::encode_intra(&yuv, width, height, quality)?;

    // 3. Wrap in ISO Base Media File Format (HEIC brand)
    let heic = isobmff::wrap_hevc_image(&nalu_stream, width, height)?;

    Ok(heic)
}
