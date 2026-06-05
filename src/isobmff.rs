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

/// Write a `colr` box from the requested colour metadata: either an `nclx` payload
/// (enumerated CICP) or a `prof` payload (embedded ICC profile).
fn write_colr(f: &mut Vec<u8>, color: &crate::color::ColorMetadata) {
    use crate::color::ColorMetadata;
    let sh = f.len();
    write_box(f, b"colr");
    match color {
        ColorMetadata::Cicp(enc) => f.extend_from_slice(&enc.nclx_payload()),
        ColorMetadata::Icc(icc) => {
            f.extend_from_slice(b"prof");
            f.extend_from_slice(icc);
        }
    }
    patch(f, sh);
}

/// Wrap a color HEVC image plus a monochrome HEVC alpha image into a HEIC file.
///
/// The alpha channel is a standalone monochrome (4:0:0) HEVC image item, linked to
/// the master color item per ISO/IEC 23008-12:
///   - item 1 = color (primary), item 2 = alpha (auxiliary)
///   - `iref` box, reference type `auxl`: alpha → color
///   - the alpha item carries an `auxC` property with the alpha URN
///   - `ipma` associates {hvcC,ispe,pixi,colr} to color and {hvcC,ispe,pixi,auxC} to alpha
pub fn wrap_hevc_image_with_alpha(
    color: &NaluStream,
    alpha: &NaluStream,
    width: u32,
    height: u32,
    bit_depth: crate::fmt::BitDepth,
    color_meta: &crate::color::ColorMetadata,
) -> Result<Vec<u8>, EncodeError> {
    let color_sample = color.to_length_prefixed_slices();
    let alpha_sample = alpha.to_length_prefixed_slices();
    let color_hvcC = build_hvcC(color, bit_depth.bits())?;
    let alpha_hvcC = build_hvcC(alpha, bit_depth.bits())?;

    const ALPHA_URN: &[u8] = b"urn:mpeg:hevc:2015:auxid:1\0";

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ────────────────────────────────────────────────────────────────
    let s = f.len();
    write_box(&mut f, b"ftyp");
    f.extend_from_slice(b"heic");
    w32(&mut f, 0);
    f.extend_from_slice(b"heic");
    f.extend_from_slice(b"mif1");
    f.extend_from_slice(b"miaf");
    f.extend_from_slice(b"iso8");
    patch(&mut f, s);

    // ── meta ──────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr
    {
        let s = f.len();
        write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0);
        f.extend_from_slice(b"pict");
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0);
        f.push(0);
        patch(&mut f, s);
    }

    // pitm → primary (color) item is ID 1
    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }

    // iloc — two items. Offsets patched after mdat is laid out.
    let color_offset_patch_pos;
    let alpha_offset_patch_pos;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 1, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, 2); // item_count = 2
        // item 1: color
        w16(&mut f, 1);
        w16(&mut f, 0); // construction_method
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1); // extent_count
        color_offset_patch_pos = f.len();
        w32(&mut f, 0);
        w32(&mut f, color_sample.len() as u32);
        // item 2: alpha
        w16(&mut f, 2);
        w16(&mut f, 0);
        w16(&mut f, 0);
        w16(&mut f, 1);
        alpha_offset_patch_pos = f.len();
        w32(&mut f, 0);
        w32(&mut f, alpha_sample.len() as u32);
        patch(&mut f, s);
    }

    // iinf — two infe entries
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, 2); // entry_count
        for id in [1u16, 2u16] {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, id);
            w16(&mut f, 0);
            f.extend_from_slice(b"hvc1");
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref — alpha (2) is auxiliary-for color (1): 'auxl' from_id=2 → to_id=1
    {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        {
            let sr = f.len();
            write_box(&mut f, b"auxl");
            w16(&mut f, 2); // from_item_ID = alpha
            w16(&mut f, 1); // reference_count
            w16(&mut f, 1); // to_item_ID = color
            patch(&mut f, sr);
        }
        patch(&mut f, s);
    }

    // iprp
    {
        let s = f.len();
        write_box(&mut f, b"iprp");

        // ipco — property container. Indices (1-based):
        //   1 hvcC(color)  2 ispe  3 pixi(3ch)  4 colr
        //   5 hvcC(alpha)  6 pixi(1ch)  7 auxC
        {
            let si = f.len();
            write_box(&mut f, b"ipco");

            // 1: hvcC (color)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&color_hvcC);
                patch(&mut f, sh);
            }
            // 2: ispe (shared — same extents for color and alpha)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // 3: pixi (color, 3 channels)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3);
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 4: colr (color metadata — nclx CICP or ICC profile)
            write_colr(&mut f, color_meta);
            // 5: hvcC (alpha)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&alpha_hvcC);
                patch(&mut f, sh);
            }
            // 6: pixi (alpha, 1 channel)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(1);
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 7: auxC (alpha auxiliary type)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"auxC", 0, 0);
                f.extend_from_slice(ALPHA_URN); // aux_type (null-terminated URN)
                patch(&mut f, sh);
            }
            patch(&mut f, si);
        }

        // ipma — associations for both items
        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 2); // entry_count
            // color item 1: hvcC(1,essential), ispe(2), pixi(3), colr(4)
            w16(&mut f, 1);
            f.push(4);
            f.push(0x80 | 1);
            f.push(2);
            f.push(3);
            f.push(4);
            // alpha item 2: hvcC(5,essential), ispe(2), pixi(6), auxC(7)
            w16(&mut f, 2);
            f.push(4);
            f.push(0x80 | 5);
            f.push(2);
            f.push(6);
            f.push(7);
            patch(&mut f, si);
        }

        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // ── mdat — both samples concatenated; record each absolute offset ─────────
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let color_abs = f.len() as u32;
    f.extend_from_slice(&color_sample);
    let alpha_abs = f.len() as u32;
    f.extend_from_slice(&alpha_sample);
    patch(&mut f, mdat_start);

    f[color_offset_patch_pos..color_offset_patch_pos + 4].copy_from_slice(&color_abs.to_be_bytes());
    f[alpha_offset_patch_pos..alpha_offset_patch_pos + 4].copy_from_slice(&alpha_abs.to_be_bytes());

    Ok(f)
}

pub fn wrap_hevc_image(
    stream: &NaluStream,
    width: u32,
    height: u32,
    bit_depth: crate::fmt::BitDepth,
    color_meta: &crate::color::ColorMetadata,
) -> Result<Vec<u8>, EncodeError> {
    let hevc_sample = stream.to_length_prefixed_slices();
    let hvcC_data = build_hvcC(stream, bit_depth.bits())?;

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
                f.push(bit_depth.bits()); // bits_per_channel[0] (Y)
                f.push(bit_depth.bits()); // bits_per_channel[1] (Cb)
                f.push(bit_depth.bits()); // bits_per_channel[2] (Cr)
                patch(&mut f, sh);
            }
            // prop 4: colr — colour metadata. Either enumerated CICP ('nclx') or an
            // embedded ICC profile ('prof'). The default is an sRGB ICC profile,
            // because Apple's ImageIO has rendered some nclx-only HEICs as black even
            // when it parses the nclx as sRGB; an ICC profile (as libheif embeds)
            // renders correctly. CICP is available for compact/HDR signalling.
            write_colr(&mut f, color_meta);
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

/// Parse chroma_format_idc from an SPS NALU (returns None if it can't be parsed).
/// Layout after the 2-byte NAL header: sps_video_parameter_set_id(4) +
/// sps_max_sub_layers_minus1(3) + temporal_id_nesting(1) = 1 byte, then PTL
/// (profile byte + 4 compat + 6 constraint + 1 level = 12 bytes), then
/// sps_seq_parameter_set_id (ue) and chroma_format_idc (ue).
fn sps_chroma_format_idc(sps: Option<&[u8]>) -> Option<u8> {
    let s = sps?;
    // Bit position starts after NAL header (2) + 1 byte + PTL (12) = 15 bytes.
    let start_bit = 15 * 8;
    let mut pos = start_bit;
    let get_bit = |p: usize| -> u32 {
        if p / 8 >= s.len() {
            return 0;
        }
        ((s[p / 8] >> (7 - (p % 8))) & 1) as u32
    };
    let mut read_ue = |pos: &mut usize| -> u32 {
        let mut zeros = 0;
        while *pos < s.len() * 8 && get_bit(*pos) == 0 {
            zeros += 1;
            *pos += 1;
        }
        *pos += 1; // the terminating 1
        if zeros == 0 {
            return 0;
        }
        let mut val = 0u32;
        for _ in 0..zeros {
            val = (val << 1) | get_bit(*pos);
            *pos += 1;
        }
        (1 << zeros) - 1 + val
    };
    let _sps_id = read_ue(&mut pos);
    let cfi = read_ue(&mut pos);
    Some(cfi as u8)
}

fn build_hvcC(stream: &NaluStream, bit_depth: u8) -> Result<Vec<u8>, EncodeError> {
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

    // Mirror the profile byte, compatibility flags, constraint flags and level from
    // the embedded SPS PTL so the hvcC summary matches the bitstream exactly. The SPS
    // PTL is authoritative (RExt profile, with bit-depth/chroma constraint flags).
    // SPS NALU layout: 2-byte NAL header, vps_id/sublayers/tid byte, then PTL:
    //   profile byte (offset 3) + compat[4] (4..8) + constraint[6] (8..14) + level (14).
    let (profile_byte, compat, constraint6, level) = if let Some(s) = sps.first() {
        if s.len() >= 15 {
            let mut compat = [0u8; 4];
            compat.copy_from_slice(&s[4..8]);
            let mut c = [0u8; 6];
            c.copy_from_slice(&s[8..14]);
            (s[3], compat, c, s[14])
        } else {
            (0b00_0_00100, [0x00, 0, 0, 0], [0x90, 0, 0, 0, 0, 0], 93)
        }
    } else {
        (0b00_0_00100, [0x00, 0, 0, 0], [0x90, 0, 0, 0, 0, 0], 93)
    };
    r.push(profile_byte); // general_profile_space/tier/idc (RExt = 4)
    r.extend_from_slice(&compat); // general_profile_compatibility_flags
    r.extend_from_slice(&constraint6); // general_constraint_indicator_flags
    r.push(level);
    r.push(0xF0);
    r.push(0x00); // min_spatial_segmentation
    r.push(0xFC); // parallelism type
    // chroma_format_idc (low 2 bits; high 6 reserved = 1). Mirror the SPS value.
    let chroma_idc = sps_chroma_format_idc(sps.first().copied()).unwrap_or(1);
    r.push(0xFC | (chroma_idc & 0x3));
    r.push(0xF8 | (bit_depth - 8)); // bit_depth_luma_minus8
    r.push(0xF8 | (bit_depth - 8)); // bit_depth_chroma_minus8
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
            nalus: vec![
                build_vps(
                    16,
                    16,
                    crate::fmt::ChromaFormat::Yuv420,
                    crate::fmt::BitDepth::Eight,
                ),
                build_sps(
                    16,
                    16,
                    crate::fmt::ChromaFormat::Yuv420,
                    crate::fmt::BitDepth::Eight,
                ),
                build_pps(30),
            ],
        }
    }
    #[test]
    fn ftyp_brand() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            crate::fmt::BitDepth::Eight,
            &crate::color::ColorMetadata::default(),
        )
        .unwrap();
        assert_eq!(&b[4..8], b"ftyp");
        assert_eq!(&b[8..12], b"heic");
    }
    #[test]
    fn box_order_ftyp_meta_mdat() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            crate::fmt::BitDepth::Eight,
            &crate::color::ColorMetadata::default(),
        )
        .unwrap();
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
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            crate::fmt::BitDepth::Eight,
            &crate::color::ColorMetadata::default(),
        )
        .unwrap();
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
    fn alpha_container_structure() {
        let color = make_test_stream();
        let alpha = make_test_stream();
        let b = wrap_hevc_image_with_alpha(
            &color,
            &alpha,
            16,
            16,
            crate::fmt::BitDepth::Eight,
            &crate::color::ColorMetadata::default(),
        )
        .unwrap();
        let s = b.as_slice();
        // Two items, two hvcC, the auxl reference, the auxC property and the URN.
        assert!(s.windows(4).any(|w| w == b"iref"), "iref box required");
        assert!(
            s.windows(4).any(|w| w == b"auxl"),
            "auxl reference required"
        );
        assert!(s.windows(4).any(|w| w == b"auxC"), "auxC property required");
        assert!(
            s.windows(26).any(|w| w == b"urn:mpeg:hevc:2015:auxid:1"),
            "alpha URN required"
        );
        // ipma entry_count must be 2 (color + alpha).
        let ipma_pos = s.windows(4).position(|w| w == b"ipma").unwrap();
        let entry_count = u32::from_be_bytes(s[ipma_pos + 8..ipma_pos + 12].try_into().unwrap());
        assert_eq!(entry_count, 2, "ipma must have 2 entries (color + alpha)");
    }
}
