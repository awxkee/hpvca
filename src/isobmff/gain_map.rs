/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! Gain map items appended to any `wrap_hevc_*` layout. Each writer calls the
//! hooks below at the matching point of its own box sequence; the items take
//! the IDs after the writer's last one. Layout mirrors an iPhone HEIC:
//!
//! - gain map image: hidden `hvc1`, `auxC` Apple URN, `auxl` → primary,
//!   plus the primary's rotation/mirror (it is not cropped: its size is its own);
//! - XMP: hidden `mime` `application/rdf+xml`, `cdsc` → gain map image;
//! - `tmap` (optional): `dimg` → [primary, gain map image], ISO metadata as
//!   its payload, and an `altr` group preferring it over the primary.

use super::{ImageMeta, build_hvcc, hvcc_uses_rext, patch, w16, w32, write_box, write_fullbox};
use crate::error::EncodeError;
use crate::gain_map::{APPLE_GAIN_MAP_URN, EncodedGainMap};

/// `iloc` entry layout of the enclosing writer.
#[derive(Clone, Copy)]
pub(super) enum IlocLayout {
    /// Version 0, `base_offset_size = 4`.
    V0,
    /// Version 1, `base_offset_size = 0`, explicit construction method.
    V1,
}

const XMP_CONTENT_TYPE: &[u8] = b"application/rdf+xml\0";

pub(super) struct GainMapBoxes<'a> {
    gm: &'a EncodedGainMap,
    hvcc: Vec<u8>,
    sample: Vec<u8>,
    image_id: u16,
    xmp_id: u16,
    tmap_id: u16,
    /// `ipco` indices (1-based) of the gain map image's properties:
    /// hvcC, ispe, pixi, auxC, colr.
    image_props: [u8; 5],
    /// `ipco` indices of the tmap item's properties (ispe, pixi, colr…).
    tmap_props: Vec<u8>,
    /// `iloc` extent_offset fields of the image, XMP and tmap items.
    offset_fields: Vec<usize>,
}

impl<'a> GainMapBoxes<'a> {
    /// Items are numbered from `first_id`.
    pub(super) fn new(
        gm: Option<&'a EncodedGainMap>,
        first_id: u16,
    ) -> Result<Option<Self>, EncodeError> {
        let Some(gm) = gm else {
            return Ok(None);
        };
        let id = |k: u16| {
            first_id
                .checked_add(k)
                .ok_or_else(|| EncodeError::IsobmffError("too many HEIF items".into()))
        };
        Ok(Some(Self {
            gm,
            hvcc: build_hvcc(&gm.stream, gm.bit_depth.bits())?,
            sample: gm.stream.to_length_prefixed_slices(),
            image_id: id(0)?,
            xmp_id: id(1)?,
            tmap_id: if gm.tmap.is_some() { id(2)? } else { 0 },
            image_props: [0; 5],
            tmap_props: Vec::new(),
            offset_fields: Vec::new(),
        }))
    }

    pub(super) fn item_count(&self) -> u16 {
        2 + u16::from(self.gm.tmap.is_some())
    }

    pub(super) fn is_rext(&self) -> bool {
        hvcc_uses_rext(&self.hvcc)
    }

    pub(super) fn has_tmap(&self) -> bool {
        self.gm.tmap.is_some()
    }

    /// Items stored in `mdat`, in `iloc` order.
    fn blobs(&self) -> impl Iterator<Item = (u16, &[u8])> {
        [
            (self.image_id, self.sample.as_slice()),
            (self.xmp_id, self.gm.xmp.as_slice()),
        ]
        .into_iter()
        .chain(self.gm.tmap.as_deref().map(|t| (self.tmap_id, t)))
    }

    pub(super) fn write_iloc(&mut self, f: &mut Vec<u8>, layout: IlocLayout) {
        let mut fields = Vec::with_capacity(3);
        for (id, data) in self.blobs() {
            w16(f, id);
            match layout {
                IlocLayout::V0 => {
                    w16(f, 0); // data_reference_index
                    w32(f, 0); // base_offset
                }
                IlocLayout::V1 => {
                    w16(f, 0); // construction_method = 0 (file offset)
                    w16(f, 0); // data_reference_index
                }
            }
            w16(f, 1); // extent_count
            fields.push(f.len());
            w32(f, 0); // extent_offset, patched by `write_data`
            w32(f, data.len() as u32);
        }
        self.offset_fields = fields;
    }

    pub(super) fn write_iinf(&self, f: &mut Vec<u8>) {
        write_infe(f, self.image_id, b"hvc1", true, &[]);
        write_infe(f, self.xmp_id, b"mime", true, XMP_CONTENT_TYPE);
        if self.has_tmap() {
            write_infe(f, self.tmap_id, b"tmap", false, &[]);
        }
    }

    pub(super) fn write_iref(&self, f: &mut Vec<u8>, primary_id: u16) {
        write_ref(f, b"auxl", self.image_id, &[primary_id]);
        write_ref(f, b"cdsc", self.xmp_id, &[self.image_id]);
        if self.has_tmap() {
            write_ref(f, b"dimg", self.tmap_id, &[primary_id, self.image_id]);
        }
    }

    /// Append the properties to `ipco`; `next` is the next free 1-based index.
    /// `visible` is the primary's size before rotation (after any crop).
    pub(super) fn write_ipco(
        &mut self,
        f: &mut Vec<u8>,
        next: &mut u8,
        base: &ImageMeta<'_>,
        visible: (u32, u32),
    ) {
        let mut take = || {
            let i = *next;
            *next += 1;
            i
        };
        let gm = self.gm;
        let sh = f.len();
        write_box(f, b"hvcC");
        f.extend_from_slice(&self.hvcc);
        patch(f, sh);
        let hvcc = take();
        write_ispe(f, gm.width, gm.height);
        let ispe = take();
        write_pixi(f, &[gm.bit_depth.bits()]);
        let pixi = take();
        let sh = f.len();
        write_fullbox(f, b"auxC", 0, 0);
        f.extend_from_slice(APPLE_GAIN_MAP_URN);
        patch(f, sh);
        let auxc = take();
        super::write_colr_nclx(f, &gm.cicp);
        let colr = take();
        self.image_props = [hvcc, ispe, pixi, auxc, colr];

        if !self.has_tmap() {
            return;
        }
        // The tmap's inputs are the primary and gain map *after* their own
        // transforms, so its output size is the displayed (rotated) one.
        let (w, h) = if base.metadata.orientation.irot_steps() & 1 == 1 {
            (visible.1, visible.0)
        } else {
            visible
        };
        let mut props = Vec::with_capacity(4);
        write_ispe(f, w, h);
        props.push(take());
        let bits = base.bit_depth.bits().max(10);
        write_pixi(f, &[bits; 3]);
        props.push(take());
        if let Some(alt) = &gm.alternate_color {
            if alt.cicp.is_some() || alt.icc.is_some() {
                super::write_colr(f, alt);
                props.push(take());
            }
            if super::has_secondary_colr(alt) {
                super::write_secondary_colr(f, alt);
                props.push(take());
            }
        }
        self.tmap_props = props;
    }

    pub(super) fn ipma_count(&self) -> u32 {
        1 + u32::from(self.has_tmap())
    }

    /// `transforms` are the primary's rotation/mirror associations (essential
    /// bit set), applied to the gain map image too.
    pub(super) fn write_ipma(&self, f: &mut Vec<u8>, transforms: &[u8]) {
        let [hvcc, ispe, pixi, auxc, colr] = self.image_props;
        let mut assoc = vec![0x80 | hvcc, ispe, pixi, auxc, colr];
        assoc.extend_from_slice(transforms);
        w16(f, self.image_id);
        f.push(assoc.len() as u8);
        f.extend_from_slice(&assoc);
        if self.has_tmap() {
            w16(f, self.tmap_id);
            f.push(self.tmap_props.len() as u8);
            f.extend_from_slice(&self.tmap_props);
        }
    }

    /// `grpl` with an `altr` group: tmap first (preferred), primary as the
    /// fallback for readers without gain map support. Goes inside `meta`.
    pub(super) fn write_grpl(&self, f: &mut Vec<u8>, primary_id: u16) {
        if !self.has_tmap() {
            return;
        }
        let s = f.len();
        write_box(f, b"grpl");
        let sa = f.len();
        write_fullbox(f, b"altr", 0, 0);
        // group_id shares the item ID namespace; take the next free one.
        w32(f, self.tmap_id as u32 + 1);
        w32(f, 2);
        w32(f, self.tmap_id as u32);
        w32(f, primary_id as u32);
        patch(f, sa);
        patch(f, s);
    }

    /// Append the items' data to the open `mdat` and patch their `iloc` offsets.
    pub(super) fn write_data(&self, f: &mut Vec<u8>) {
        let fields = self.offset_fields.clone();
        for ((_, data), field) in self.blobs().zip(fields) {
            let abs = f.len() as u32;
            f.extend_from_slice(data);
            f[field..field + 4].copy_from_slice(&abs.to_be_bytes());
        }
    }

    /// Compatible brands the gain map adds to `ftyp`.
    pub(super) fn extra_brands(&self, primary_rext: bool) -> Vec<&'static [u8; 4]> {
        let mut brands = Vec::new();
        if self.is_rext() && !primary_rext {
            brands.push(b"heix");
        }
        if self.has_tmap() {
            brands.push(b"tmap");
        }
        brands
    }
}

fn write_infe(f: &mut Vec<u8>, id: u16, kind: &[u8; 4], hidden: bool, content_type: &[u8]) {
    let si = f.len();
    write_fullbox(f, b"infe", 2, u32::from(hidden));
    w16(f, id);
    w16(f, 0); // item_protection_index
    f.extend_from_slice(kind);
    f.push(0); // item_name
    f.extend_from_slice(content_type);
    patch(f, si);
}

fn write_ref(f: &mut Vec<u8>, kind: &[u8; 4], from: u16, to: &[u16]) {
    let sr = f.len();
    write_box(f, kind);
    w16(f, from);
    w16(f, to.len() as u16);
    for &id in to {
        w16(f, id);
    }
    patch(f, sr);
}

fn write_ispe(f: &mut Vec<u8>, w: u32, h: u32) {
    let sh = f.len();
    write_fullbox(f, b"ispe", 0, 0);
    w32(f, w);
    w32(f, h);
    patch(f, sh);
}

fn write_pixi(f: &mut Vec<u8>, bits: &[u8]) {
    let sh = f.len();
    write_fullbox(f, b"pixi", 0, 0);
    f.push(bits.len() as u8);
    f.extend_from_slice(bits);
    patch(f, sh);
}
