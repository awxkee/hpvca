//! ISO Base Media File Format writer for HEIC still images.
//!
//! Box hierarchy:
//!   ftyp
//!   meta  (fullbox version=0)
//!     hdlr
//!     pitm
//!     iinf → infe
//!     iloc  (version=1, offset_size=4, length_size=4, base_offset_size=0)
//!     iprp → ipco → { hvcC, ispe }
//!            ipma
//!   mdat  ← offset stored in iloc is patched after mdat position is known

use crate::{error::EncodeError, hevc::NaluStream};
use std::io::Write;

fn w32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}
fn w16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn write_fullbox(buf: &mut Vec<u8>, cc: &[u8; 4], ver: u8, flags: u32) {
    w32(buf, 0); // size placeholder
    buf.extend_from_slice(cc);
    buf.push(ver);
    buf.push((flags >> 16) as u8);
    buf.push((flags >> 8) as u8);
    buf.push(flags as u8);
}

fn write_box(buf: &mut Vec<u8>, cc: &[u8; 4]) {
    w32(buf, 0);
    buf.extend_from_slice(cc);
}

fn patch(buf: &mut Vec<u8>, start: usize) {
    let size = (buf.len() - start) as u32;
    buf[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

pub fn wrap_hevc_image(
    stream: &NaluStream,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, EncodeError> {
    let hevc_sample = stream.to_length_prefixed_slices();
    let hvcC_data = build_hvcC(stream)?;

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ────────────────────────────────────────────────────────────────
    let s = f.len();
    write_box(&mut f, b"ftyp");
    f.extend_from_slice(b"heic"); // major brand
    w32(&mut f, 0); // minor version
    f.extend_from_slice(b"heic"); // HEIC still image
    f.extend_from_slice(b"mif1"); // HEIF base
    f.extend_from_slice(b"miaf"); // Multi-Image Application Format (required by Apple Preview)
    f.extend_from_slice(b"iso8"); // ISO base media
    patch(&mut f, s);

    // ── meta ─────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr
    {
        let s = f.len();
        write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0); // pre_defined
        f.extend_from_slice(b"pict"); // handler_type
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0); // reserved
        f.push(0); // name (empty)
        patch(&mut f, s);
    }

    // pitm
    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1); // item_ID = 1
        patch(&mut f, s);
    }

    // iloc — version=1 so construction_method field is present.
    // libheif places iloc BEFORE iinf; Apple parsers can be order-sensitive.
    // We write a placeholder for extent_offset and record its position.
    let iloc_offset_patch_pos;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 1, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, 1); // item_count = 1
        // item entry
        w16(&mut f, 1); // item_ID
        w16(&mut f, 0); // construction_method = 0 (file offset)
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1); // extent_count
        iloc_offset_patch_pos = f.len();
        w32(&mut f, 0); // extent_offset — patched later
        w32(&mut f, hevc_sample.len() as u32); // extent_length
        patch(&mut f, s);
    }

    // iinf
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, 1); // entry_count
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 1); // item_ID
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(b"hvc1"); // item_type
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iprp
    {
        let s = f.len();
        write_box(&mut f, b"iprp");

        // ipco
        {
            let si = f.len();
            write_box(&mut f, b"ipco");

            // prop 1: hvcC  (essential — decoder config)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&hvcC_data);
                patch(&mut f, sh);
            }
            // prop 2: ispe  (image spatial extents)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // prop 3: pixi  (pixel information — bits per channel)
            // libheif includes this; Apple Preview requires it.
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3); // num_channels = 3
                f.push(8); // bits_per_channel[0] = 8 (Y)
                f.push(8); // bits_per_channel[1] = 8 (Cb)
                f.push(8); // bits_per_channel[2] = 8 (Cr)
                patch(&mut f, sh);
            }
            // prop 4: colr (ICC profile). Apple's ImageIO renders nclx-tagged HEICs
            // as BLACK on current macOS even though it parses the nclx as sRGB; an
            // embedded ICC profile (exactly what libheif does) renders correctly.
            // The profile is sRGB IEC61966-2.1 (from littleCMS).
            {
                let sh = f.len();
                write_box(&mut f, b"colr");
                f.extend_from_slice(b"prof"); // colour_type = ICC profile
                f.extend_from_slice(&crate::icc_profile::SRGB_ICC); // the ICC profile bytes
                patch(&mut f, sh);
            }
            patch(&mut f, si);
        }

        // ipma — 4 properties (hvcC, ispe, pixi, colr)
        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 1); // entry_count
            w16(&mut f, 1); // item_ID
            f.push(4); // association_count = 4
            f.push(0x80 | 1); // essential=1, property_index=1 (hvcC)
            f.push(2); // essential=0, property_index=2 (ispe)
            f.push(3); // essential=0, property_index=3 (pixi)
            f.push(4); // essential=0, property_index=4 (colr)
            patch(&mut f, si);
        }

        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // ── mdat — now we know the exact offset ──────────────────────────────────
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let hevc_absolute_offset = f.len() as u32; // offset of first byte of payload
    f.extend_from_slice(&hevc_sample);
    patch(&mut f, mdat_start);

    // Patch iloc extent_offset with the real absolute file offset
    f[iloc_offset_patch_pos..iloc_offset_patch_pos + 4]
        .copy_from_slice(&hevc_absolute_offset.to_be_bytes());

    Ok(f)
}

fn build_hvcC(stream: &NaluStream) -> Result<Vec<u8>, EncodeError> {
    let mut vps: Vec<&[u8]> = Vec::new();
    let mut sps: Vec<&[u8]> = Vec::new();
    let mut pps: Vec<&[u8]> = Vec::new();
    for nalu in &stream.nalus {
        match (nalu.data[0] >> 1) & 0x3f {
            32 => vps.push(&nalu.data),
            33 => sps.push(&nalu.data),
            34 => pps.push(&nalu.data),
            _ => {}
        }
    }

    let mut r: Vec<u8> = Vec::new();
    r.push(1); // configurationVersion
    // profile_space=0, tier=0, profile_idc=3 (Main Still Picture)
    r.push(0b00_0_00011);
    // profile_compat_flags: flag[1]=Main, flag[2]=Main10, flag[3]=MainSP → 0x70000000
    r.extend_from_slice(&[0x70, 0x00, 0x00, 0x00]);
    // Mirror the level and constraint flags from the embedded SPS so the hvcC
    // summary matches the bitstream exactly (the SPS PTL is authoritative; the
    // level scales with picture size via hevc::level_idc_for).
    let (constraint6, level) = if let Some(s) = sps.first() {
        // SPS NALU: 2-byte NAL header, then vps_id/sublayers/tid byte, then PTL:
        // profile byte (1) + compat (4) + constraint (6) + level (1).
        // Constraint bytes begin at offset 2+1+1+4 = 8; level at offset 14.
        let mut c = [0u8; 6];
        if s.len() >= 15 {
            c.copy_from_slice(&s[8..14]);
            (c, s[14])
        } else {
            ([0xb0, 0, 0, 0, 0, 0], 93)
        }
    } else {
        ([0xb0, 0, 0, 0, 0, 0], 93)
    };
    r.extend_from_slice(&constraint6);
    r.push(level);
    r.push(0xF0);
    r.push(0x00); // min_spatial_segmentation
    r.push(0xFC); // parallelism type
    r.push(0xFD); // chroma_format_idc = 1 (4:2:0)
    r.push(0xF8); // bit_depth_luma_minus8 = 0
    r.push(0xF8); // bit_depth_chroma_minus8 = 0
    r.push(0x00);
    r.push(0x00); // avgFrameRate
    r.push(0b00_001_1_11); // cFR=0, numTL=1, tidNested=1, lengthSizeMinusOne=3 (4 bytes)

    let arrays: &[(u8, &Vec<&[u8]>)] = &[(32, &vps), (33, &sps), (34, &pps)];
    r.push(arrays.len() as u8);
    for &(nal_type, list) in arrays {
        // array_completeness=0 (high bit). libheif uses 0; Apple expects this.
        r.push(nal_type);
        r.push((list.len() >> 8) as u8);
        r.push(list.len() as u8);
        for &d in list {
            r.push((d.len() >> 8) as u8);
            r.push(d.len() as u8);
            r.extend_from_slice(d);
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::NaluStream;
    fn make_test_stream() -> NaluStream {
        use crate::hevc::{build_pps, build_sps, build_vps};
        NaluStream {
            nalus: vec![build_vps(16, 16), build_sps(16, 16), build_pps(30)],
        }
    }
    #[test]
    fn ftyp_brand() {
        let b = wrap_hevc_image(&make_test_stream(), 16, 16).unwrap();
        assert_eq!(&b[4..8], b"ftyp");
        assert_eq!(&b[8..12], b"heic");
    }
    #[test]
    fn box_order_ftyp_meta_mdat() {
        let b = wrap_hevc_image(&make_test_stream(), 16, 16).unwrap();
        let ftyp_size = u32::from_be_bytes(b[0..4].try_into().unwrap()) as usize;
        assert_eq!(
            &b[ftyp_size + 4..ftyp_size + 8],
            b"meta",
            "meta must follow ftyp"
        );
        let meta_size =
            u32::from_be_bytes(b[ftyp_size..ftyp_size + 4].try_into().unwrap()) as usize;
        assert_eq!(
            &b[ftyp_size + meta_size + 4..ftyp_size + meta_size + 8],
            b"mdat",
            "mdat must follow meta"
        );
    }
    #[test]
    fn iloc_offset_points_into_mdat() {
        let b = wrap_hevc_image(&make_test_stream(), 16, 16).unwrap();
        // Find mdat payload start
        let mut pos = 0usize;
        let mut mdat_payload_offset = 0u32;
        while pos + 8 <= b.len() {
            let sz = u32::from_be_bytes(b[pos..pos + 4].try_into().unwrap()) as usize;
            if &b[pos + 4..pos + 8] == b"mdat" {
                mdat_payload_offset = (pos + 8) as u32;
                break;
            }
            pos += sz;
        }
        assert!(mdat_payload_offset > 0, "mdat not found");
        // Find iloc box start (search for fourcc after size field)
        let iloc_box_pos = b.windows(4).position(|w| w == b"iloc").unwrap() - 4;
        // iloc layout:  box_hdr(8) + fullbox_ver_flags(4) = 12
        //               field_bytes(2) + item_count(2) = 4
        //               item: item_id(2) + constr_method(2) + data_ref(2) + ext_count(2) = 8
        //               extent_offset is here (4 bytes)
        let offset_pos = iloc_box_pos + 12 + 4 + 8;
        let extent_offset = u32::from_be_bytes(b[offset_pos..offset_pos + 4].try_into().unwrap());
        assert_eq!(
            extent_offset, mdat_payload_offset,
            "iloc extent_offset must point to start of mdat payload"
        );
    }
    #[test]
    fn ipco_has_pixi_and_colr() {
        let b = wrap_hevc_image(&make_test_stream(), 16, 16).unwrap();
        let s = b.as_slice();
        assert!(
            s.windows(4).any(|w| w == b"pixi"),
            "pixi box must be present"
        );
        // colr (nclx) is REQUIRED by Apple CoreGraphics to build a color space.
        assert!(
            s.windows(4).any(|w| w == b"colr"),
            "colr box must be present"
        );
        let colr_pos = s.windows(4).position(|w| w == b"colr").unwrap();
        assert_eq!(
            &s[colr_pos + 4..colr_pos + 8],
            b"prof",
            "colr must be ICC profile type"
        );
        // ipma should have exactly 4 associations (hvcC, ispe, pixi, colr)
        let ipma_pos = s.windows(4).position(|w| w == b"ipma").unwrap();
        let assoc_count = s[ipma_pos + 14];
        assert_eq!(assoc_count, 4, "ipma must associate 4 properties");
    }
}
