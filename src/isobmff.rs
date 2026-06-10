/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::{error::EncodeError, hevc::NaluStream};

/// Output-side image metadata common to every `wrap_hevc_*` entry point: the
/// sample bit depth plus the colour and image-metadata blocks written into the
/// `meta` box. Bundled so the wrappers take this instead of three loose params.
#[derive(Clone, Copy)]
pub(crate) struct ImageMeta<'a> {
    pub(crate) bit_depth: crate::fmt::BitDepth,
    pub(crate) color_meta: &'a crate::color::ColorMetadata,
    pub(crate) metadata: &'a crate::metadata::Metadata,
}

/// Geometry of a HEIF tile grid: the `cols`×`rows` tile layout, the common
/// per-tile coded size `(tile_w, tile_h)`, and the grid's visible size
/// `(full_w, full_h)` written to `ispe`.
#[derive(Clone, Copy)]
pub(crate) struct GridDims {
    pub(crate) cols: u32,
    pub(crate) rows: u32,
    pub(crate) tile_w: u32,
    pub(crate) tile_h: u32,
    pub(crate) full_w: u32,
    pub(crate) full_h: u32,
}

fn epb(nalu: &[u8]) -> Vec<u8> {
    if nalu.len() <= 2 {
        return nalu.to_vec();
    }
    let mut out = Vec::with_capacity(nalu.len() + 4);
    out.push(nalu[0]);
    out.push(nalu[1]); // NAL header excluded
    let mut zeros: u8 = 0;
    for &b in &nalu[2..] {
        if zeros >= 2 && b <= 3 {
            out.push(0x03);
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

fn w32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}
fn w16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn write_fullbox(buf: &mut Vec<u8>, cc: &[u8; 4], ver: u8, flags: u32) {
    w32(buf, 0);
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

fn patch(buf: &mut [u8], start: usize) {
    let size = (buf.len() - start) as u32;
    buf[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

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

/// Write ftyp. RExt profiles (4:2:2, 4:4:4, or 12-bit 4:2:0) use `heix`;
/// Main/Main10 (4:2:0 up to 10-bit) use `heic`.
fn write_ftyp(f: &mut Vec<u8>, chroma_idc: u8, bit_depth: u8) {
    let rext = chroma_idc > 1 || bit_depth > 10;
    let brand: &[u8; 4] = if rext { b"heix" } else { b"heic" };
    let s = f.len();
    write_box(f, b"ftyp");
    f.extend_from_slice(brand); // major brand
    w32(f, 0); // minor version
    f.extend_from_slice(brand); // compatible: same as major
    f.extend_from_slice(b"mif1");
    f.extend_from_slice(b"miaf");
    patch(f, s);
}

/// Write iloc version=0 with base_offset_size=4.
/// Apple strictly requires version=0; version=1 (which adds a construction_method
/// field) is for HEIF construction methods that simple still images never use.
/// Layout per item: item_id(2) + data_ref_index(2) + base_offset(4, = 0) +
///                  extent_count(2) + extent_offset(4, patched later) + extent_length(4).
/// Returns the byte positions of each extent_offset field to be patched later.
fn write_iloc_item(f: &mut Vec<u8>, item_id: u16, data_len: u32) -> usize {
    w16(f, item_id);
    w16(f, 0); // data_reference_index
    w32(f, 0); // base_offset = 0 (extent_offset will be absolute)
    w16(f, 1); // extent_count
    let patch_pos = f.len();
    w32(f, 0); // extent_offset — caller patches with absolute file position
    w32(f, data_len);
    patch_pos
}

pub(crate) fn wrap_hevc_image_with_alpha(
    color: &NaluStream,
    alpha: &NaluStream,
    width: u32,
    height: u32,
    img: ImageMeta<'_>,
) -> Result<Vec<u8>, EncodeError> {
    let ImageMeta {
        bit_depth,
        color_meta,
        metadata,
    } = img;
    let color_sample = color.to_length_prefixed_slices();
    let alpha_sample = alpha.to_length_prefixed_slices();
    let color_hvcc = build_hvcc(color, bit_depth.bits())?;
    let alpha_hvcc = build_hvcc(alpha, bit_depth.bits())?;
    let chroma_idc = sps_chroma_format_idc(color).unwrap_or(1);

    const ALPHA_URN: &[u8] = b"urn:mpeg:hevc:2015:auxid:1\0";

    let mut f: Vec<u8> = Vec::new();

    write_ftyp(&mut f, chroma_idc, bit_depth.bits());

    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

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

    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }

    let color_offset_patch_pos;
    let alpha_offset_patch_pos;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x40); // base_offset_size=4, index_size=0
        w16(&mut f, 2);
        color_offset_patch_pos = write_iloc_item(&mut f, 1, color_sample.len() as u32);
        alpha_offset_patch_pos = write_iloc_item(&mut f, 2, alpha_sample.len() as u32);
        patch(&mut f, s);
    }

    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, 2);
        for id in [1u16, 2u16] {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, id);
            w16(&mut f, 0);
            f.extend_from_slice(b"hvc1");
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        {
            let sr = f.len();
            write_box(&mut f, b"auxl");
            w16(&mut f, 2);
            w16(&mut f, 1);
            w16(&mut f, 1);
            patch(&mut f, sr);
        }
        patch(&mut f, s);
    }

    {
        let s = f.len();
        write_box(&mut f, b"iprp");

        // ipco indices (1-based):
        //   1 hvcC(color)  2 colr  3 ispe  4 pixi(3ch)
        //   5 hvcC(alpha)  6 pixi(1ch)  7 auxC   8+ optional
        let mut irot_idx = 0u8;
        let mut imir_idx = 0u8;
        let mut clli_idx = 0u8;
        {
            let si = f.len();
            write_box(&mut f, b"ipco");
            // 1: hvcC (color)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&color_hvcc);
                patch(&mut f, sh);
            }
            // 2: colr
            write_colr(&mut f, color_meta);
            // 3: ispe
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // 4: pixi (color, 3 ch)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3);
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 5: hvcC (alpha)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&alpha_hvcc);
                patch(&mut f, sh);
            }
            // 6: pixi (alpha, 1 ch)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(1);
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 7: auxC
            {
                let sh = f.len();
                write_fullbox(&mut f, b"auxC", 0, 0);
                f.extend_from_slice(ALPHA_URN);
                patch(&mut f, sh);
            }
            // 8+: optional
            let mut next_prop: u8 = 8;
            if metadata.orientation.irot_steps() != 0 {
                let sh = f.len();
                write_box(&mut f, b"irot");
                f.push(metadata.orientation.irot_steps() & 0x03);
                patch(&mut f, sh);
                irot_idx = next_prop;
                next_prop += 1;
            }
            if let Some(ax) = metadata.orientation.imir_axis() {
                let sh = f.len();
                write_box(&mut f, b"imir");
                f.push(if ax { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next_prop;
                next_prop += 1;
            }
            if let Some(cll) = metadata.content_light_level {
                let sh = f.len();
                write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next_prop;
                next_prop += 1;
            }
            let _ = next_prop;
            patch(&mut f, si);
        }

        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 2);
            // color: hvcC(1*) colr(2) ispe(3) pixi(4) + optionals
            let mut ca: Vec<u8> = vec![0x80 | 1, 2, 3, 4];
            if irot_idx != 0 {
                ca.push(0x80 | irot_idx);
            }
            if imir_idx != 0 {
                ca.push(0x80 | imir_idx);
            }
            if clli_idx != 0 {
                ca.push(clli_idx);
            }
            w16(&mut f, 1);
            f.push(ca.len() as u8);
            f.extend_from_slice(&ca);
            // alpha: hvcC(5*) ispe(3) pixi(6) auxC(7)
            w16(&mut f, 2);
            f.push(4);
            f.push(0x80 | 5);
            f.push(3);
            f.push(6);
            f.push(7);
            patch(&mut f, si);
        }

        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

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

pub(crate) fn wrap_hevc_image(
    stream: &NaluStream,
    width: u32,
    height: u32,
    img: ImageMeta<'_>,
) -> Result<Vec<u8>, EncodeError> {
    let ImageMeta {
        bit_depth,
        color_meta,
        metadata,
    } = img;
    let hevc_sample = stream.to_length_prefixed_slices();
    let hvcc_data = build_hvcc(stream, bit_depth.bits())?;
    let chroma_idc = sps_chroma_format_idc(stream).unwrap_or(1);

    let mut f: Vec<u8> = Vec::new();

    write_ftyp(&mut f, chroma_idc, bit_depth.bits());

    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

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

    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }

    let has_exif = metadata.exif.is_some();
    let exif_payload: Vec<u8> = metadata
        .exif
        .as_ref()
        .map(|e| {
            let mut p = Vec::with_capacity(e.len() + 4);
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(e);
            p
        })
        .unwrap_or_default();

    let iloc_offset_patch_pos;
    let mut iloc_exif_patch_pos = 0usize;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x40); // base_offset_size=4, index_size=0
        w16(&mut f, if has_exif { 2 } else { 1 });
        iloc_offset_patch_pos = write_iloc_item(&mut f, 1, hevc_sample.len() as u32);
        if has_exif {
            iloc_exif_patch_pos = write_iloc_item(&mut f, 2, exif_payload.len() as u32);
        }
        patch(&mut f, s);
    }

    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, if has_exif { 2 } else { 1 });
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 1);
            w16(&mut f, 0);
            f.extend_from_slice(b"hvc1");
            f.push(0);
            patch(&mut f, si);
        }
        if has_exif {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 2);
            w16(&mut f, 0);
            f.extend_from_slice(b"Exif");
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    if has_exif {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        {
            let si = f.len();
            write_box(&mut f, b"cdsc");
            w16(&mut f, 2);
            w16(&mut f, 1);
            w16(&mut f, 1);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    {
        let extra_props: (u8, u8, u8);
        let s = f.len();
        write_box(&mut f, b"iprp");

        {
            let si = f.len();
            write_box(&mut f, b"ipco");

            // ipco order matching libheif/ffmpeg: hvcC → colr → ispe → pixi → optionals
            // 1: hvcC (essential — decoder config record)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&hvcc_data);
                patch(&mut f, sh);
            }
            // 2: colr
            write_colr(&mut f, color_meta);
            // 3: ispe
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // 4: pixi
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3);
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }

            // 5+: optional
            let mut next_prop: u8 = 5;
            let mut irot_idx = 0u8;
            let mut imir_idx = 0u8;
            let mut clli_idx = 0u8;
            if metadata.orientation.irot_steps() != 0 {
                let sh = f.len();
                write_box(&mut f, b"irot");
                f.push(metadata.orientation.irot_steps() & 0x03);
                patch(&mut f, sh);
                irot_idx = next_prop;
                next_prop += 1;
            }
            if let Some(ax) = metadata.orientation.imir_axis() {
                let sh = f.len();
                write_box(&mut f, b"imir");
                f.push(if ax { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next_prop;
                next_prop += 1;
            }
            if let Some(cll) = metadata.content_light_level {
                let sh = f.len();
                write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next_prop;
                next_prop += 1;
            }
            let _ = next_prop;
            extra_props = (irot_idx, imir_idx, clli_idx);
            patch(&mut f, si);
        }

        {
            let (irot_idx, imir_idx, clli_idx) = extra_props;
            // ipma: hvcC(1*) colr(2) ispe(3) pixi(4) + optionals
            let mut assoc: Vec<u8> = vec![0x80 | 1, 2, 3, 4];
            if irot_idx != 0 {
                assoc.push(0x80 | irot_idx);
            }
            if imir_idx != 0 {
                assoc.push(0x80 | imir_idx);
            }
            if clli_idx != 0 {
                assoc.push(clli_idx);
            }
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 1);
            w16(&mut f, 1);
            f.push(assoc.len() as u8);
            f.extend_from_slice(&assoc);
            patch(&mut f, si);
        }

        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let hevc_abs = f.len() as u32;
    f.extend_from_slice(&hevc_sample);
    let exif_abs = f.len() as u32;
    if has_exif {
        f.extend_from_slice(&exif_payload);
    }
    patch(&mut f, mdat_start);

    f[iloc_offset_patch_pos..iloc_offset_patch_pos + 4].copy_from_slice(&hevc_abs.to_be_bytes());
    if has_exif {
        f[iloc_exif_patch_pos..iloc_exif_patch_pos + 4].copy_from_slice(&exif_abs.to_be_bytes());
    }

    Ok(f)
}

fn sps_chroma_format_idc(stream: &NaluStream) -> Option<u8> {
    let sps = stream
        .nalus
        .iter()
        .find(|n| (n.data[0] >> 1) & 0x3f == 33)
        .map(|n| n.data.as_slice())?;
    let start_bit = 15 * 8;
    let mut pos = start_bit;
    let get_bit = |p: usize| -> u32 {
        if p / 8 >= sps.len() {
            return 0;
        }
        ((sps[p / 8] >> (7 - (p % 8))) & 1) as u32
    };
    let read_ue = |pos: &mut usize| -> u32 {
        let mut zeros = 0;
        while *pos < sps.len() * 8 && get_bit(*pos) == 0 {
            zeros += 1;
            *pos += 1;
        }
        *pos += 1;
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
    Some(read_ue(&mut pos) as u8)
}

fn build_hvcc(stream: &NaluStream, bit_depth: u8) -> Result<Vec<u8>, EncodeError> {
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
    let (profile_byte, compat, constraint6, level) = if let Some(s) = sps.first() {
        if s.len() >= 15 {
            let mut cp = [0u8; 4];
            cp.copy_from_slice(&s[4..8]);
            let mut c = [0u8; 6];
            c.copy_from_slice(&s[8..14]);
            (s[3], cp, c, s[14])
        } else {
            (0b0000_0100, [0x00, 0, 0, 0], [0x90, 0, 0, 0, 0, 0], 93)
        }
    } else {
        (0b0000_0100, [0x00, 0, 0, 0], [0x90, 0, 0, 0, 0, 0], 93)
    };

    r.push(profile_byte);
    r.extend_from_slice(&compat);
    r.extend_from_slice(&constraint6);
    r.push(level);
    r.push(0xF0);
    r.push(0x00); // min_spatial_segmentation
    r.push(0xFC); // parallelism
    let chroma_idc = sps_chroma_format_idc(stream).unwrap_or(1);
    r.push(0xFC | (chroma_idc & 0x3));
    r.push(0xF8 | (bit_depth - 8)); // bit_depth_luma_minus8
    r.push(0xF8 | (bit_depth - 8)); // bit_depth_chroma_minus8
    r.push(0x00);
    r.push(0x00); // avgFrameRate
    r.push(0b0000_1111); // cFR=0, numTL=1, tidNested=1, lengthSizeM1=3

    let arrays: &[(u8, &Vec<&[u8]>)] = &[(32, &vps), (33, &sps), (34, &pps)];
    r.push(arrays.len() as u8);
    for &(nal_type, list) in arrays {
        r.push(nal_type); // array_completeness=0, matching libheif/Apple
        r.push((list.len() >> 8) as u8);
        r.push(list.len() as u8);
        for &d in list {
            // Parameter set NALUs stored in hvcC must include emulation prevention
            // bytes — Apple's decoder strips them during parameter set parsing.
            let ebsp = epb(d);
            r.push((ebsp.len() >> 8) as u8);
            r.push(ebsp.len() as u8);
            r.extend_from_slice(&ebsp);
        }
    }
    Ok(r)
}

/// Write a HEIC file whose primary item is an image grid assembled from
/// pre-encoded HEVC tile streams arranged in row-major order.
///
/// All tiles must be encoded at exactly `tile_w × tile_h` luma samples (edge
/// tiles are padded before encoding so every tile shares the same SPS/hvcC).
/// The grid's `ispe` carries the true visible dimensions `(full_w, full_h)`.
pub(crate) fn wrap_hevc_grid(
    tiles: &[NaluStream],
    dims: GridDims,
    img: ImageMeta<'_>,
) -> Result<Vec<u8>, EncodeError> {
    let GridDims {
        cols,
        rows,
        tile_w,
        tile_h,
        full_w,
        full_h,
    } = dims;
    let ImageMeta {
        bit_depth,
        color_meta,
        metadata,
    } = img;
    assert_eq!(
        tiles.len(),
        (cols * rows) as usize,
        "tile count must equal cols*rows"
    );

    // Build hvcC once from the first tile (all tiles share the same SPS/PPS).
    let hvcc_data = build_hvcc(&tiles[0], bit_depth.bits())?;
    let chroma_idc = sps_chroma_format_idc(&tiles[0]).unwrap_or(1);

    // Pre-encode all tile mdat samples.
    let tile_samples: Vec<Vec<u8>> = tiles
        .iter()
        .map(|t| t.to_length_prefixed_slices())
        .collect();

    // Grid item payload (HEIF §6.6.2.3.2):
    //   version(1) flags(1) rows_minus1(1) cols_minus1(1) output_width output_height
    //   flags bit0 = 1 → rows/cols are 16-bit (else 8-bit)
    //   flags bit1 = 1 → output dims are 32-bit (else 16-bit)
    //   ORDER IS: output_width FIRST, output_height SECOND
    let use_32bit = full_w > 65535 || full_h > 65535;
    let grid_payload: Vec<u8> = {
        let flags: u8 = if use_32bit { 2 } else { 0 }; // bit1 = 32-bit dims
        let mut g = vec![0u8, flags, (rows - 1) as u8, (cols - 1) as u8];
        if use_32bit {
            g.extend_from_slice(&full_w.to_be_bytes()); // width first
            g.extend_from_slice(&full_h.to_be_bytes()); // height second
        } else {
            g.extend_from_slice(&(full_w as u16).to_be_bytes()); // width first
            g.extend_from_slice(&(full_h as u16).to_be_bytes()); // height second
        }
        g
    };

    let n_tiles = tiles.len() as u16; // tile item IDs: 1..=n_tiles
    let grid_id = n_tiles + 1; // grid item ID
    let has_exif = metadata.exif.is_some();
    let exif_id = grid_id + 1;
    let exif_payload: Vec<u8> = metadata
        .exif
        .as_ref()
        .map(|e| {
            let mut p = Vec::with_capacity(e.len() + 4);
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(e);
            p
        })
        .unwrap_or_default();

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ─────────────────────────────────────────────────────────────────
    write_ftyp(&mut f, chroma_idc, bit_depth.bits());

    // ── meta ─────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

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

    // pitm: grid item is primary
    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, grid_id);
        patch(&mut f, s);
    }

    // iloc version=1 — required for mixed construction methods:
    //   tiles use construction_method=0 (mdat file offset)
    //   grid descriptor uses construction_method=1 (idat inline)
    // Apple's encoder always writes iloc v=1 with construction_method per item.
    let mut tile_patch_positions: Vec<usize> = Vec::with_capacity(n_tiles as usize);
    let mut exif_patch_pos = 0usize;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 1, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        let total_items = n_tiles + 1 + if has_exif { 1 } else { 0 };
        w16(&mut f, total_items);
        // Tile items: construction_method=0 (file offset into mdat)
        for i in 0..n_tiles {
            w16(&mut f, i + 1); // item_id
            w16(&mut f, 0); // construction_method=0 (file offset)
            w16(&mut f, 0); // data_reference_index
            w16(&mut f, 1); // extent_count
            let pp = f.len();
            w32(&mut f, 0); // extent_offset — patched later
            w32(&mut f, tile_samples[i as usize].len() as u32);
            tile_patch_positions.push(pp);
        }
        // Grid item: construction_method=1 (from idat, offset=0)
        w16(&mut f, grid_id);
        w16(&mut f, 1); // construction_method=1 (idat)
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1); // extent_count
        w32(&mut f, 0); // extent_offset = 0 (start of idat payload)
        w32(&mut f, grid_payload.len() as u32);
        if has_exif {
            w16(&mut f, exif_id);
            w16(&mut f, 0); // construction_method=0 (mdat)
            w16(&mut f, 0);
            w16(&mut f, 1);
            exif_patch_pos = f.len();
            w32(&mut f, 0);
            w32(&mut f, exif_payload.len() as u32);
        }
        patch(&mut f, s);
    }

    // iinf
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        let entry_count = n_tiles + 1 + if has_exif { 1 } else { 0 };
        w16(&mut f, entry_count);
        for i in 0..n_tiles {
            // Tile items referenced only via dimg MUST be hidden (HEIF spec §9.3.1.2)
            // Write infe fullbox header manually to ensure flags=1 (item_hidden_flag)
            let si = f.len();
            w32(&mut f, 0); // size (patched later)
            f.extend_from_slice(b"infe");
            f.push(2u8); // version=2
            f.push(0u8); // flags[2]=0
            f.push(0u8); // flags[1]=0
            f.push(1u8); // flags[0]=1 → item_hidden_flag SET
            w16(&mut f, i + 1); // item_ID
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(b"hvc1"); // item_type
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, grid_id);
            w16(&mut f, 0);
            f.extend_from_slice(b"grid");
            f.push(0);
            patch(&mut f, si);
        }
        if has_exif {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, exif_id);
            w16(&mut f, 0);
            f.extend_from_slice(b"Exif");
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref: dimg (grid→tiles) + cdsc (EXIF→grid)
    {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        {
            // dimg: grid references all tiles in row-major order
            let sr = f.len();
            write_box(&mut f, b"dimg");
            w16(&mut f, grid_id); // from_item_id
            w16(&mut f, n_tiles); // reference_count
            for i in 1..=n_tiles {
                w16(&mut f, i);
            }
            patch(&mut f, sr);
        }
        if has_exif {
            let sr = f.len();
            write_box(&mut f, b"cdsc");
            w16(&mut f, exif_id);
            w16(&mut f, 1);
            w16(&mut f, grid_id);
            patch(&mut f, sr);
        }
        patch(&mut f, s);
    }

    // iprp
    {
        let s = f.len();
        write_box(&mut f, b"iprp");

        // ipco property indices (1-based):
        //   1: hvcC (shared by all tiles)
        //   2: ispe (tile coded dimensions, shared by all tiles)
        //   3: ispe (full image dimensions, for grid item only)
        //   4: colr
        //   5: pixi
        //   6+: optional irot/imir/clli (for grid item)
        let mut irot_idx = 0u8;
        let mut imir_idx = 0u8;
        let mut clli_idx = 0u8;
        {
            let si = f.len();
            write_box(&mut f, b"ipco");
            // 1: hvcC
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&hvcc_data);
                patch(&mut f, sh);
            }
            // 2: ispe (tile coded dimensions)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, tile_w);
                w32(&mut f, tile_h);
                patch(&mut f, sh);
            }
            // 3: ispe (full image)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, full_w);
                w32(&mut f, full_h);
                patch(&mut f, sh);
            }
            // 4: colr
            write_colr(&mut f, color_meta);
            // 5: pixi
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3);
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 6+: orientation / HDR (applied to the grid item)
            let mut next: u8 = 6;
            if metadata.orientation.irot_steps() != 0 {
                let sh = f.len();
                write_box(&mut f, b"irot");
                f.push(metadata.orientation.irot_steps() & 3);
                patch(&mut f, sh);
                irot_idx = next;
                next += 1;
            }
            if let Some(ax) = metadata.orientation.imir_axis() {
                let sh = f.len();
                write_box(&mut f, b"imir");
                f.push(if ax { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next;
                next += 1;
            }
            if let Some(cll) = metadata.content_light_level {
                let sh = f.len();
                write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next;
                next += 1;
            }
            let _ = next;
            patch(&mut f, si);
        }

        // ipma
        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            let entry_count = n_tiles + 1; // tiles + grid (EXIF has no ipma entry)
            w32(&mut f, entry_count as u32);

            // Tile items: hvcC(1,essential) + ispe_tile(2,essential) + colr(4,essential)
            // Apple's encoder marks colr essential on each tile so VideoToolbox can
            // set up colour management per-tile during grid decoding.
            for i in 1..=n_tiles {
                w16(&mut f, i);
                f.push(3);
                f.push(0x80 | 1); // hvcC essential
                f.push(0x80 | 2); // ispe tile essential
                f.push(0x80 | 4); // colr essential
            }

            // Grid item: ispe_full(3) + colr(4,essential) + pixi(5) + optional transforms
            w16(&mut f, grid_id);
            let mut ga: Vec<u8> = vec![3, 0x80 | 4, 5]; // colr marked essential
            if irot_idx != 0 {
                ga.push(0x80 | irot_idx);
            }
            if imir_idx != 0 {
                ga.push(0x80 | imir_idx);
            }
            if clli_idx != 0 {
                ga.push(clli_idx);
            }
            f.push(ga.len() as u8);
            f.extend_from_slice(&ga);

            patch(&mut f, si);
        }

        patch(&mut f, s);
    }

    // ── idat — grid descriptor stored inline (construction_method=1) ──────────
    // idat is a child of meta (ISO 14496-12 §8.11.4). iloc item 5 uses
    // construction_method=1 with extent_offset=0 pointing here.
    {
        let s = f.len();
        write_box(&mut f, b"idat");
        f.extend_from_slice(&grid_payload);
        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // ── mdat — tile HEVC samples + optional EXIF ─────────────────────────────
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");

    let mut tile_abs: Vec<u32> = Vec::with_capacity(n_tiles as usize);
    for sample in &tile_samples {
        tile_abs.push(f.len() as u32);
        f.extend_from_slice(sample);
    }
    let exif_abs = f.len() as u32;
    if has_exif {
        f.extend_from_slice(&exif_payload);
    }
    patch(&mut f, mdat_start);

    // Patch tile iloc extent_offsets
    for (i, &abs) in tile_abs.iter().enumerate() {
        let pp = tile_patch_positions[i];
        f[pp..pp + 4].copy_from_slice(&abs.to_be_bytes());
    }
    if has_exif {
        f[exif_patch_pos..exif_patch_pos + 4].copy_from_slice(&exif_abs.to_be_bytes());
    }

    Ok(f)
}

/// Write a HEIC file with a tiled primary image **and** a tiled alpha auxiliary
/// image, both assembled as HEIF grids.
///
/// Item-ID layout:
/// - `1..=n_tiles`          : color tile items (hidden `hvc1`)
/// - `n_tiles+1`            : color grid item (primary)
/// - `n_tiles+2..=2*n_tiles+1` : alpha tile items (hidden `hvc1`)
/// - `2*n_tiles+2`          : alpha grid item (`auxl` → color grid)
///
/// Both grids share the same tile/image dimensions and use `construction_method=1`
/// (inline `idat`) for their grid descriptors; tile samples are in `mdat`.
pub(crate) fn wrap_hevc_grid_with_alpha(
    color_tiles: &[NaluStream],
    alpha_tiles: &[NaluStream],
    dims: GridDims,
    img: ImageMeta<'_>,
) -> Result<Vec<u8>, EncodeError> {
    let GridDims {
        cols,
        rows,
        tile_w,
        tile_h,
        full_w,
        full_h,
    } = dims;
    let ImageMeta {
        bit_depth,
        color_meta,
        metadata,
    } = img;
    assert_eq!(color_tiles.len(), (cols * rows) as usize);
    assert_eq!(alpha_tiles.len(), (cols * rows) as usize);

    const ALPHA_URN: &[u8] = b"urn:mpeg:hevc:2015:auxid:1\0";

    let color_hvcc = build_hvcc(&color_tiles[0], bit_depth.bits())?;
    let alpha_hvcc = build_hvcc(&alpha_tiles[0], bit_depth.bits())?;
    let chroma_idc = sps_chroma_format_idc(&color_tiles[0]).unwrap_or(1);

    let color_samples: Vec<Vec<u8>> = color_tiles
        .iter()
        .map(|t| t.to_length_prefixed_slices())
        .collect();
    let alpha_samples: Vec<Vec<u8>> = alpha_tiles
        .iter()
        .map(|t| t.to_length_prefixed_slices())
        .collect();

    // Build both grid descriptors (identical geometry, packed sequentially in idat).
    let use_32bit = full_w > 65535 || full_h > 65535;
    let grid_payload: Vec<u8> = {
        let flags: u8 = if use_32bit { 2 } else { 0 };
        let mut g = vec![0u8, flags, (rows - 1) as u8, (cols - 1) as u8];
        if use_32bit {
            g.extend_from_slice(&full_w.to_be_bytes());
            g.extend_from_slice(&full_h.to_be_bytes());
        } else {
            g.extend_from_slice(&(full_w as u16).to_be_bytes());
            g.extend_from_slice(&(full_h as u16).to_be_bytes());
        }
        g
    };
    // Alpha grid has identical geometry — reuse the same descriptor bytes.
    let alpha_grid_offset = grid_payload.len() as u32; // offset into idat

    let n = color_tiles.len() as u16; // tiles per image
    let color_grid_id = n + 1;
    let alpha_tile_base = n + 2; // first alpha tile item-id
    let alpha_grid_id = alpha_tile_base + n;

    let has_exif = metadata.exif.is_some();
    let exif_id = alpha_grid_id + 1;
    let exif_payload: Vec<u8> = metadata
        .exif
        .as_ref()
        .map(|e| {
            let mut p = Vec::with_capacity(e.len() + 4);
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(e);
            p
        })
        .unwrap_or_default();

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ─────────────────────────────────────────────────────────────────
    write_ftyp(&mut f, chroma_idc, bit_depth.bits());

    // ── meta ─────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

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

    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, color_grid_id);
        patch(&mut f, s);
    }

    // iloc version=1: mixed construction methods
    let mut color_tile_patches: Vec<usize> = Vec::with_capacity(n as usize);
    let mut alpha_tile_patches: Vec<usize> = Vec::with_capacity(n as usize);
    let mut exif_patch_pos = 0usize;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 1, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        let total = 2 * n + 2 + if has_exif { 1 } else { 0 };
        w16(&mut f, total);

        // Color tiles (cm=0, mdat)
        for i in 0..n {
            w16(&mut f, i + 1);
            w16(&mut f, 0);
            w16(&mut f, 0);
            w16(&mut f, 1);
            let pp = f.len();
            w32(&mut f, 0);
            w32(&mut f, color_samples[i as usize].len() as u32);
            color_tile_patches.push(pp);
        }
        // Color grid (cm=1, idat offset=0)
        w16(&mut f, color_grid_id);
        w16(&mut f, 1);
        w16(&mut f, 0);
        w16(&mut f, 1);
        w32(&mut f, 0);
        w32(&mut f, grid_payload.len() as u32);

        // Alpha tiles (cm=0, mdat)
        for i in 0..n {
            w16(&mut f, alpha_tile_base + i);
            w16(&mut f, 0);
            w16(&mut f, 0);
            w16(&mut f, 1);
            let pp = f.len();
            w32(&mut f, 0);
            w32(&mut f, alpha_samples[i as usize].len() as u32);
            alpha_tile_patches.push(pp);
        }
        // Alpha grid (cm=1, idat offset=after color descriptor)
        w16(&mut f, alpha_grid_id);
        w16(&mut f, 1);
        w16(&mut f, 0);
        w16(&mut f, 1);
        w32(&mut f, alpha_grid_offset);
        w32(&mut f, grid_payload.len() as u32);

        if has_exif {
            w16(&mut f, exif_id);
            w16(&mut f, 0);
            w16(&mut f, 0);
            w16(&mut f, 1);
            exif_patch_pos = f.len();
            w32(&mut f, 0);
            w32(&mut f, exif_payload.len() as u32);
        }
        patch(&mut f, s);
    }

    // iinf
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        let count = 2 * n + 2 + if has_exif { 1 } else { 0 };
        w16(&mut f, count);

        // Hidden color tiles
        for i in 0..n {
            let si = f.len();
            w32(&mut f, 0);
            f.extend_from_slice(b"infe");
            f.push(2);
            f.push(0);
            f.push(0);
            f.push(1); // version=2, flags=hidden
            w16(&mut f, i + 1);
            w16(&mut f, 0);
            f.extend_from_slice(b"hvc1");
            f.push(0);
            patch(&mut f, si);
        }
        // Color grid (visible, primary)
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, color_grid_id);
            w16(&mut f, 0);
            f.extend_from_slice(b"grid");
            f.push(0);
            patch(&mut f, si);
        }

        // Hidden alpha tiles
        for i in 0..n {
            let si = f.len();
            w32(&mut f, 0);
            f.extend_from_slice(b"infe");
            f.push(2);
            f.push(0);
            f.push(0);
            f.push(1);
            w16(&mut f, alpha_tile_base + i);
            w16(&mut f, 0);
            f.extend_from_slice(b"hvc1");
            f.push(0);
            patch(&mut f, si);
        }
        // Alpha grid (visible — alpha items are not hidden per HEIF spec §6.6.4.2)
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, alpha_grid_id);
            w16(&mut f, 0);
            f.extend_from_slice(b"grid");
            f.push(0);
            patch(&mut f, si);
        }

        if has_exif {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, exif_id);
            w16(&mut f, 0);
            f.extend_from_slice(b"Exif");
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref
    {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        // dimg: color grid → color tiles
        {
            let sr = f.len();
            write_box(&mut f, b"dimg");
            w16(&mut f, color_grid_id);
            w16(&mut f, n);
            for i in 1..=n {
                w16(&mut f, i);
            }
            patch(&mut f, sr);
        }
        // dimg: alpha grid → alpha tiles
        {
            let sr = f.len();
            write_box(&mut f, b"dimg");
            w16(&mut f, alpha_grid_id);
            w16(&mut f, n);
            for i in 0..n {
                w16(&mut f, alpha_tile_base + i);
            }
            patch(&mut f, sr);
        }
        // auxl: alpha grid is auxiliary for color grid
        {
            let sr = f.len();
            write_box(&mut f, b"auxl");
            w16(&mut f, alpha_grid_id);
            w16(&mut f, 1);
            w16(&mut f, color_grid_id);
            patch(&mut f, sr);
        }
        if has_exif {
            let sr = f.len();
            write_box(&mut f, b"cdsc");
            w16(&mut f, exif_id);
            w16(&mut f, 1);
            w16(&mut f, color_grid_id);
            patch(&mut f, sr);
        }
        patch(&mut f, s);
    }

    // iprp
    // ipco indices (1-based):
    //  1 hvcC(color)  2 ispe(tile)  3 ispe(full)  4 colr  5 pixi(color,3ch)
    //  6 hvcC(alpha)  7 pixi(alpha,1ch)  8 auxC  9+ irot/imir/clli
    let mut irot_idx = 0u8;
    let mut imir_idx = 0u8;
    let mut clli_idx = 0u8;
    {
        let s = f.len();
        write_box(&mut f, b"iprp");
        {
            let si = f.len();
            write_box(&mut f, b"ipco");
            // 1: hvcC (color)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&color_hvcc);
                patch(&mut f, sh);
            }
            // 2: ispe (tile)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, tile_w);
                w32(&mut f, tile_h);
                patch(&mut f, sh);
            }
            // 3: ispe (full)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, full_w);
                w32(&mut f, full_h);
                patch(&mut f, sh);
            }
            // 4: colr
            write_colr(&mut f, color_meta);
            // 5: pixi (color, 3ch)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3);
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 6: hvcC (alpha)
            {
                let sh = f.len();
                write_box(&mut f, b"hvcC");
                f.extend_from_slice(&alpha_hvcc);
                patch(&mut f, sh);
            }
            // 7: pixi (alpha, 1ch)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(1);
                f.push(bit_depth.bits());
                patch(&mut f, sh);
            }
            // 8: auxC
            {
                let sh = f.len();
                write_fullbox(&mut f, b"auxC", 0, 0);
                f.extend_from_slice(ALPHA_URN);
                patch(&mut f, sh);
            }
            // 9+: orientation / HDR (color grid only)
            let mut next: u8 = 9;
            if metadata.orientation.irot_steps() != 0 {
                let sh = f.len();
                write_box(&mut f, b"irot");
                f.push(metadata.orientation.irot_steps() & 3);
                patch(&mut f, sh);
                irot_idx = next;
                next += 1;
            }
            if let Some(ax) = metadata.orientation.imir_axis() {
                let sh = f.len();
                write_box(&mut f, b"imir");
                f.push(if ax { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next;
                next += 1;
            }
            if let Some(cll) = metadata.content_light_level {
                let sh = f.len();
                write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next;
                next += 1;
            }
            let _ = next;
            patch(&mut f, si);
        }
        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            let entry_count = 2 * n + 2; // color tiles + color grid + alpha tiles + alpha grid
            w32(&mut f, entry_count as u32);

            // Color tiles: hvcC(1*) ispe_tile(2*) colr(4*)
            for i in 1..=n {
                w16(&mut f, i);
                f.push(3);
                f.push(0x80 | 1);
                f.push(0x80 | 2);
                f.push(0x80 | 4);
            }
            // Color grid: ispe_full(3) colr(4*) pixi(5) + optionals
            w16(&mut f, color_grid_id);
            let mut ga: Vec<u8> = vec![3, 0x80 | 4, 5];
            if irot_idx != 0 {
                ga.push(0x80 | irot_idx);
            }
            if imir_idx != 0 {
                ga.push(0x80 | imir_idx);
            }
            if clli_idx != 0 {
                ga.push(clli_idx);
            }
            f.push(ga.len() as u8);
            f.extend_from_slice(&ga);

            // Alpha tiles: hvcC(6*) ispe_tile(2*)
            for i in 0..n {
                w16(&mut f, alpha_tile_base + i);
                f.push(2);
                f.push(0x80 | 6);
                f.push(0x80 | 2);
            }
            // Alpha grid: ispe_full(3) pixi(7) auxC(8)
            w16(&mut f, alpha_grid_id);
            f.push(3);
            f.push(3);
            f.push(7);
            f.push(8);

            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // idat — both grid descriptors inline (color at 0, alpha at alpha_grid_offset)
    {
        let s = f.len();
        write_box(&mut f, b"idat");
        f.extend_from_slice(&grid_payload); // color grid descriptor (offset=0)
        f.extend_from_slice(&grid_payload); // alpha grid descriptor (same geometry)
        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // mdat — all tile samples
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let mut color_abs: Vec<u32> = Vec::with_capacity(n as usize);
    for s in &color_samples {
        color_abs.push(f.len() as u32);
        f.extend_from_slice(s);
    }
    let mut alpha_abs: Vec<u32> = Vec::with_capacity(n as usize);
    for s in &alpha_samples {
        alpha_abs.push(f.len() as u32);
        f.extend_from_slice(s);
    }
    let exif_abs = f.len() as u32;
    if has_exif {
        f.extend_from_slice(&exif_payload);
    }
    patch(&mut f, mdat_start);

    for (i, &abs) in color_abs.iter().enumerate() {
        f[color_tile_patches[i]..color_tile_patches[i] + 4].copy_from_slice(&abs.to_be_bytes());
    }
    for (i, &abs) in alpha_abs.iter().enumerate() {
        f[alpha_tile_patches[i]..alpha_tile_patches[i] + 4].copy_from_slice(&abs.to_be_bytes());
    }
    if has_exif {
        f[exif_patch_pos..exif_patch_pos + 4].copy_from_slice(&exif_abs.to_be_bytes());
    }

    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::NaluStream;

    fn make_stream(chroma: crate::fmt::ChromaFormat, bd: crate::fmt::BitDepth) -> NaluStream {
        use crate::hevc::{build_pps, build_sps, build_vps};
        NaluStream {
            nalus: vec![
                build_vps(16, 16, chroma, bd),
                build_sps(16, 16, chroma, bd, crate::color::ColorEncoding::srgb()),
                build_pps(30, false),
            ],
        }
    }
    fn make_test_stream() -> NaluStream {
        make_stream(
            crate::fmt::ChromaFormat::Yuv420,
            crate::fmt::BitDepth::Eight,
        )
    }

    #[test]
    fn ftyp_heic_for_420() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        assert_eq!(&b[4..8], b"ftyp");
        assert_eq!(&b[8..12], b"heic", "4:2:0 8-bit must use heic brand");
    }

    #[test]
    fn ftyp_heix_for_422() {
        let b = wrap_hevc_image(
            &make_stream(crate::fmt::ChromaFormat::Yuv422, crate::fmt::BitDepth::Ten),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Ten,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        assert_eq!(&b[8..12], b"heix", "4:2:2 must use heix brand");
    }

    #[test]
    fn ftyp_heix_for_444() {
        let b = wrap_hevc_image(
            &make_stream(
                crate::fmt::ChromaFormat::Yuv444,
                crate::fmt::BitDepth::Eight,
            ),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        assert_eq!(&b[8..12], b"heix", "4:4:4 must use heix brand");
    }

    #[test]
    fn iloc_version_0() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        // windows().position() returns the start of the fourcc bytes directly.
        let iloc_fourcc = b.windows(4).position(|w| w == b"iloc").unwrap();
        let ver = b[iloc_fourcc + 4]; // version byte (first fullbox byte)
        let base_sz = (b[iloc_fourcc + 9] >> 4) & 0xF; // field2 high nibble = base_offset_size
        assert_eq!(ver, 0, "iloc must be version 0");
        assert_eq!(base_sz, 4, "base_offset_size must be 4");
    }

    #[test]
    fn box_order_ftyp_meta_mdat() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        let ftyp_size = u32::from_be_bytes(b[0..4].try_into().unwrap()) as usize;
        assert_eq!(&b[ftyp_size + 4..ftyp_size + 8], b"meta");
        let meta_size =
            u32::from_be_bytes(b[ftyp_size..ftyp_size + 4].try_into().unwrap()) as usize;
        assert_eq!(
            &b[ftyp_size + meta_size + 4..ftyp_size + meta_size + 8],
            b"mdat"
        );
    }

    #[test]
    fn iloc_offset_points_into_mdat() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        let mut pos = 0usize;
        let mut mdat_payload = 0u32;
        while pos + 8 <= b.len() {
            let sz = u32::from_be_bytes(b[pos..pos + 4].try_into().unwrap()) as usize;
            if &b[pos + 4..pos + 8] == b"mdat" {
                mdat_payload = (pos + 8) as u32;
                break;
            }
            pos += sz;
        }
        assert!(mdat_payload > 0, "mdat not found");
        // iloc v0+base4 layout:
        //   box(8) + fullbox(4) + fields(2) + item_count(2) = 16
        //   item: item_id(2) + data_ref(2) + base_offset(4) + extent_count(2) = 10
        //   extent_offset here (4 bytes)
        let iloc_pos = b.windows(4).position(|w| w == b"iloc").unwrap() - 4;
        let offset_pos = iloc_pos + 16 + 10;
        let extent_offset = u32::from_be_bytes(b[offset_pos..offset_pos + 4].try_into().unwrap());
        assert_eq!(extent_offset, mdat_payload);
    }

    #[test]
    fn ipco_order_hvcc_colr_ispe_pixi() {
        let b = wrap_hevc_image(
            &make_test_stream(),
            16,
            16,
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        let ipco_start = b.windows(4).position(|w| w == b"ipco").unwrap() + 4;
        let first_child = &b[ipco_start + 4..ipco_start + 8];
        assert_eq!(first_child, b"hvcC", "first ipco property must be hvcC");
        // second property should be colr
        let hvcc_sz =
            u32::from_be_bytes(b[ipco_start..ipco_start + 4].try_into().unwrap()) as usize;
        let second_child = &b[ipco_start + hvcc_sz + 4..ipco_start + hvcc_sz + 8];
        assert_eq!(second_child, b"colr", "second ipco property must be colr");
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
            ImageMeta {
                bit_depth: crate::fmt::BitDepth::Eight,
                color_meta: &crate::color::ColorMetadata::default(),
                metadata: &crate::metadata::Metadata::default(),
            },
        )
        .unwrap();
        let s = b.as_slice();
        assert!(s.windows(4).any(|w| w == b"iref"));
        assert!(s.windows(4).any(|w| w == b"auxl"));
        assert!(s.windows(4).any(|w| w == b"auxC"));
        assert!(s.windows(26).any(|w| w == b"urn:mpeg:hevc:2015:auxid:1"));
        let ipma_pos = s.windows(4).position(|w| w == b"ipma").unwrap();
        let entry_count = u32::from_be_bytes(s[ipma_pos + 8..ipma_pos + 12].try_into().unwrap());
        assert_eq!(entry_count, 2);
    }
}
