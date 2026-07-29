/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

//! Palette mode (HEVC Screen Content Coding, §7.3.8.13 / §8.4.4.2) — encoder side.
//!
pub(crate) const MAX_COMPONENTS: usize = 3;

/// `palette_max_size` signalled in the SPS.
pub(crate) const MAX_PALETTE_SIZE: usize = 64;
/// `delta_palette_max_predictor_size` signalled in the SPS.
pub(crate) const DELTA_MAX_PREDICTOR_SIZE: usize = 64;
/// PaletteMaxPredictorSize = palette_max_size + delta_palette_max_predictor_size.
pub(crate) const MAX_PREDICTOR_SIZE: usize = MAX_PALETTE_SIZE + DELTA_MAX_PREDICTOR_SIZE;

/// Largest palette CU side (MaxTbLog2SizeY = 5 in this encoder).
pub(crate) const MAX_CU: usize = 32;
const MAX_SAMPLES: usize = MAX_CU * MAX_CU;

/// Distinct exact colors tolerated before a block is declared photographic and
/// palette analysis bails out. Bailing early keeps the gate cheap on photos.
const MAX_DISTINCT: usize = 128;

/// Slots in the color lookup table. A power of two at least twice
/// [`MAX_DISTINCT`], so linear probing never passes a 50% load factor.
const COLOR_SLOTS: usize = 256;

/// Open-addressed "have I seen this color" table for the block histogram.
///
/// The histogram used to rescan every distinct color found so far for each
/// sample, which is O(n·distinct) — and it runs on *every* CU that survives the
/// luma pre-gate, whether or not palette mode ends up being evaluated. On a
/// photograph that single loop was most of the cost of the screen-content
/// search. Hashing makes it O(n) while preserving first-seen insertion order,
/// so the frequency sort downstream sees exactly the same input.
pub(crate) struct ColorIndex {
    keys: [u64; COLOR_SLOTS],
    entry: [u16; COLOR_SLOTS],
    /// Which call last wrote each slot; bumping `generation` retires the whole
    /// table without touching memory.
    stamp: [u32; COLOR_SLOTS],
    generation: u32,
}

impl ColorIndex {
    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            keys: [0; COLOR_SLOTS],
            entry: [0; COLOR_SLOTS],
            stamp: [0; COLOR_SLOTS],
            generation: 0,
        })
    }

    #[inline]
    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped: retire every slot explicitly, once every 4 billion CUs.
            self.stamp = [0; COLOR_SLOTS];
            self.generation = 1;
        }
    }

    /// Look `key` up, inserting `next` as its entry index when absent. Returns
    /// the stored index and whether the color had been seen already.
    #[inline]
    fn lookup_or_insert(&mut self, key: u64, next: usize) -> (usize, bool) {
        let mut slot =
            (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as usize & (COLOR_SLOTS - 1);
        loop {
            if self.stamp[slot] != self.generation {
                self.stamp[slot] = self.generation;
                self.keys[slot] = key;
                self.entry[slot] = next as u16;
                return (next, false);
            }
            if self.keys[slot] == key {
                return (self.entry[slot] as usize, true);
            }
            slot = (slot + 1) & (COLOR_SLOTS - 1);
        }
    }
}

/// Pack a color into the table key. Components are at most 16 bits.
#[inline]
fn color_key(sample: &[u16; MAX_COMPONENTS], num_comps: usize) -> u64 {
    let mut key = u64::from(sample[0]);
    if num_comps > 1 {
        key |= u64::from(sample[1]) << 16;
        key |= u64::from(sample[2]) << 32;
    }
    key
}

/// COPY_ABOVE_MODE (§7.4.9.13). `palette_run_type_flag` carries it directly, so
/// the writers pass `u8::from(copy_above)`; the constant names the value the
/// decoder's walk is replayed against in the tests.
#[cfg_attr(not(test), allow(dead_code))]
const COPY_ABOVE_MODE: u8 = 1;

/// Slice/tile/WPP-row persistent palette predictor (§9.3.2.3). Entries are
/// stored as component triples in predictor order.
#[derive(Clone, Default)]
pub(crate) struct PalettePredictor {
    entries: Vec<[u16; MAX_COMPONENTS]>,
    num_comps: usize,
}

impl PalettePredictor {
    /// Reset to the (empty) SPS/PPS initializer table. This encoder signals no
    /// palette predictor initializers, so the reset always empties the table.
    pub(crate) fn reset(&mut self, num_comps: usize) {
        self.entries.clear();
        self.num_comps = num_comps;
    }

    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub(crate) fn entry(&self, k: usize) -> [u16; MAX_COMPONENTS] {
        self.entries[k]
    }

    /// Palette predictor update (§8.4.4.2): the CU palette moves to the front,
    /// then the previous entries that were **not** reused, truncated to
    /// `max_pred`. Mirrors `hpvcd::palette::PalettePredictor::update`.
    pub(crate) fn update(
        &mut self,
        cu_palette: &[[u16; MAX_COMPONENTS]],
        reused: &[bool],
        max_pred: usize,
    ) {
        let mut next: Vec<[u16; MAX_COMPONENTS]> = Vec::with_capacity(max_pred);
        next.extend_from_slice(cu_palette);
        for (k, entry) in self.entries.iter().enumerate() {
            if next.len() >= max_pred {
                break;
            }
            if !reused.get(k).copied().unwrap_or(false) {
                next.push(*entry);
            }
        }
        next.truncate(max_pred);
        self.entries = next;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit sink
// ─────────────────────────────────────────────────────────────────────────────

/// The CABAC operations palette syntax needs, with the normative context
/// selection (§9.3.4.2.1) resolved by the caller's bridge.
pub(crate) trait PaletteBitWriter {
    /// `palette_run_type_flag` **and** `copy_above_indices_for_final_run_flag`;
    /// both use the same single context variable.
    fn run_type_flag(&mut self, bin: u8);
    /// `palette_transpose_flag`.
    fn transpose_flag(&mut self, bin: u8);
    /// `palette_run_prefix` bin `bin_idx` for a COPY_ABOVE run, or a non-first
    /// bin of a COPY_INDEX run.
    fn run_prefix_bin(&mut self, bin_idx: usize, copy_above: bool, bin: u8);
    /// First `palette_run_prefix` bin of a COPY_INDEX run (context depends on
    /// the *unadjusted* index symbol).
    fn run_prefix_index_bin(&mut self, palette_index: u32, bin: u8);
    /// One bypass bin.
    fn bypass(&mut self, bin: u8);
    /// `n` bypass bins carrying `value`, MSB first.
    fn bypass_bits(&mut self, value: u32, n: u32);
    /// `delta_qp()` / `chroma_qp_offset()` in the palette-escape position.
    fn escape_qp_syntax(&mut self);
}

/// SCC bypass Exp-Golomb order-`k`: a run of ones terminated by a zero, then
/// `prefix + k` refinement bits.
fn write_egk<B: PaletteBitWriter>(bits: &mut B, value: u32, k: u32) {
    let prefix = 31 - ((value >> k) + 1).leading_zeros();
    let base = ((1u32 << prefix) - 1) << k;
    for _ in 0..prefix {
        bits.bypass(1);
    }
    bits.bypass(0);
    bits.bypass_bits(value - base, prefix + k);
}

#[inline]
fn write_eg0<B: PaletteBitWriter>(bits: &mut B, value: u32) {
    write_egk(bits, value, 0);
}

/// Truncated binary over an alphabet of `alphabet` symbols.
fn write_truncated_binary<B: PaletteBitWriter>(bits: &mut B, value: u32, alphabet: u32) {
    if alphabet <= 1 {
        return;
    }
    let k = 31 - alphabet.leading_zeros();
    let u = (1u32 << (k + 1)) - alphabet;
    if value < u {
        bits.bypass_bits(value, k);
    } else {
        let coded = value + u;
        bits.bypass_bits(coded >> 1, k);
        bits.bypass((coded & 1) as u8);
    }
}

/// `num_palette_indices_minus1` (§9.3.3.14): truncated-Rice prefix capped at
/// four one-bins; only the all-ones prefix carries an EGk extension.
fn write_num_palette_indices<B: PaletteBitWriter>(
    bits: &mut B,
    count_minus1: u32,
    max_palette_index: u32,
) {
    let rice = 3 + ((max_palette_index + 1) >> 3);
    let cmax = 4u32 << rice;
    if count_minus1 < cmax {
        for _ in 0..(count_minus1 >> rice) {
            bits.bypass(1);
        }
        bits.bypass(0);
        bits.bypass_bits(count_minus1 & ((1u32 << rice) - 1), rice);
    } else {
        for _ in 0..4 {
            bits.bypass(1);
        }
        write_egk(bits, count_minus1 - cmax, rice + 1);
    }
}

/// `palette_run_prefix` + `palette_run_suffix` for a run of `run` *additional*
/// samples, bounded by `palette_max_run`.
fn write_run<B: PaletteBitWriter>(
    bits: &mut B,
    run: u32,
    palette_max_run: u32,
    copy_above: bool,
    raw_index: u32,
) {
    if palette_max_run == 0 {
        return;
    }
    let max_prefix = 32 - palette_max_run.leading_zeros();
    let prefix = if run < 2 {
        run
    } else {
        32 - run.leading_zeros()
    }
    .min(max_prefix);

    let emit = |bits: &mut B, idx: u32, bin: u8| {
        if !copy_above && idx == 0 {
            bits.run_prefix_index_bin(raw_index, bin);
        } else {
            bits.run_prefix_bin(idx as usize, copy_above, bin);
        }
    };
    for i in 0..prefix {
        emit(bits, i, 1);
    }
    if prefix < max_prefix {
        emit(bits, prefix, 0);
    }
    if prefix >= 2 {
        let base = 1u32 << (prefix - 1);
        if palette_max_run != base {
            let suffix_max = if (base << 1) > palette_max_run {
                palette_max_run - base
            } else {
                base - 1
            };
            write_truncated_binary(bits, run - base, suffix_max + 1);
        }
    }
}

/// `palette_predictor_run`-coded reuse flags (§7.3.8.13).
fn write_reuse_flags<B: PaletteBitWriter>(bits: &mut B, reused: &[bool], max_size: usize) {
    let pred_size = reused.len();
    let mut idx = 0usize;
    let mut num_reused = 0usize;
    for (position, _) in reused.iter().enumerate().filter(|&(_, &r)| r) {
        let gap = position - idx;
        write_eg0(bits, if gap == 0 { 0 } else { gap as u32 + 1 });
        idx = position + 1;
        num_reused += 1;
    }
    // The decoder's loop stops on its own once the predictor is exhausted or the
    // palette is full; only an early stop needs the terminating run of 1.
    if idx < pred_size && num_reused < max_size {
        write_eg0(bits, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan geometry
// ─────────────────────────────────────────────────────────────────────────────

/// Palette traverse scan position `i` → grid `(col, row)` for a `size × size`
/// block. Palette syntax always uses the fixed horizontal boustrophedon scan;
/// `palette_transpose_flag` acts at reconstruction, not on the scan.
#[inline]
pub(crate) fn scan_pos(i: usize, size: usize) -> (usize, usize) {
    let row = i / size;
    let x_in_row = i % size;
    let col = if row & 1 == 1 {
        size - 1 - x_in_row
    } else {
        x_in_row
    };
    (col, row)
}

#[inline]
pub(crate) fn scan_pos_inv(col: usize, row: usize, size: usize) -> usize {
    let x_in_row = if row & 1 == 1 { size - 1 - col } else { col };
    row * size + x_in_row
}

/// Actual index of the sample directly above scan position `scan`.
#[inline]
fn index_above(indices: &[u32], scan: usize, size: usize) -> u32 {
    let (col, row) = scan_pos(scan, size);
    if row == 0 {
        return 0;
    }
    indices[scan_pos_inv(col, row - 1, size)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Escape quantization
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn level_scale(rem: i32) -> i64 {
    const LS: [i64; 6] = [40, 45, 51, 57, 64, 72];
    LS[rem.rem_euclid(6) as usize]
}

/// Palette escape reconstruction (§8.4.4.2.2). `qp` is the component Qp′
/// (already including QpBdOffset). Mirrors `hpvcd`'s `dequant_escape`.
#[inline]
pub(crate) fn dequant_escape(level: i32, qp: i32, bit_depth: u8, tqb: bool) -> u16 {
    let max = (1i32 << bit_depth) - 1;
    if tqb {
        return level.clamp(0, max) as u16;
    }
    let qp = qp.max(0);
    let scaled = ((level as i64 * level_scale(qp % 6)) << (qp / 6)) + 32;
    (scaled >> 6).clamp(0, max as i64) as u16
}

/// Pick the escape level whose dequantized value is closest to `value`.
fn quantize_escape(value: u16, qp: i32, bit_depth: u8, tqb: bool) -> i32 {
    if tqb {
        return i32::from(value);
    }
    let qp = qp.max(0);
    let step = (level_scale(qp % 6) << (qp / 6)) as f64;
    let start = ((f64::from(value) * 64.0) / step).round() as i64;
    let mut best = 0i32;
    let mut best_err = i32::MAX;
    for candidate in (start - 1).max(0)..=(start + 1).max(0) {
        let level = candidate.min(i64::from(i32::MAX)) as i32;
        let rec = dequant_escape(level, qp, bit_depth, false);
        let err = (i32::from(rec) - i32::from(value)).abs();
        if err < best_err {
            best_err = err;
            best = level;
        }
    }
    best
}

/// One palette-coded CU: everything the syntax writer and the reconstruction
/// need. Boxed inside the per-worker scratch — the index/escape arrays are
/// sized for the largest palette CU (32×32).
#[derive(Clone)]
pub(crate) struct PaletteCu {
    pub(crate) size: usize,
    pub(crate) num_comps: usize,
    pub(crate) palette: [[u16; MAX_COMPONENTS]; MAX_PALETTE_SIZE],
    pub(crate) palette_size: usize,
    /// Per-predictor-entry reuse flags, `pred_size` long.
    pub(crate) reused: [bool; MAX_PREDICTOR_SIZE],
    pub(crate) pred_size: usize,
    pub(crate) num_signaled: usize,
    pub(crate) escape_present: bool,
    pub(crate) transpose: bool,
    /// Per-scan-position actual index; `palette_size` marks an escape.
    pub(crate) indices: [u32; MAX_SAMPLES],
    /// Per-scan-position escape levels (component-major triples).
    pub(crate) escapes: [[i32; MAX_COMPONENTS]; MAX_SAMPLES],
    /// Sum of squared error of the palette reconstruction against the source,
    /// measured over exactly the samples the decoder writes.
    pub(crate) sse: f32,
    /// How far a sample may sit from its palette entry before it escapes;
    /// carried between the palette build and the index assignment.
    tolerance: u32,
}

impl PaletteCu {
    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            size: 0,
            num_comps: 1,
            palette: [[0; MAX_COMPONENTS]; MAX_PALETTE_SIZE],
            palette_size: 0,
            reused: [false; MAX_PREDICTOR_SIZE],
            pred_size: 0,
            num_signaled: 0,
            escape_present: false,
            transpose: false,
            indices: [0; MAX_SAMPLES],
            escapes: [[0; MAX_COMPONENTS]; MAX_SAMPLES],
            sse: f32::MAX,
            tolerance: 0,
        })
    }

    #[inline]
    pub(crate) fn palette_entries(&self) -> &[[u16; MAX_COMPONENTS]] {
        &self.palette[..self.palette_size]
    }

    #[inline]
    pub(crate) fn reuse_flags(&self) -> &[bool] {
        &self.reused[..self.pred_size]
    }

    /// MaxPaletteIndex = CurrentPaletteSize − 1 + palette_escape_val_present_flag.
    #[inline]
    fn max_palette_index(&self) -> i64 {
        self.palette_size as i64 - 1 + i64::from(self.escape_present)
    }
}

/// Frame-constant palette parameters.
#[derive(Clone, Copy)]
pub(crate) struct PaletteConfig {
    pub(crate) num_comps: usize,
    pub(crate) chroma_idc: u8,
    pub(crate) sub_w: usize,
    pub(crate) sub_h: usize,
    pub(crate) bd_luma: u8,
    pub(crate) bd_chroma: u8,
    /// Component Qp′ values (including QpBdOffset) used for escape scaling.
    pub(crate) qp: [i32; MAX_COMPONENTS],
    pub(crate) lossless: bool,
}

impl PaletteConfig {
    #[inline]
    fn bit_depth(&self, comp: usize) -> u8 {
        if comp == 0 {
            self.bd_luma
        } else {
            self.bd_chroma
        }
    }

    /// Per-component subsampling as the decoder's reconstruction uses it.
    #[inline]
    fn sub(&self, comp: usize) -> (usize, usize) {
        if comp == 0 {
            (1, 1)
        } else {
            (self.sub_w, self.sub_h)
        }
    }
}

/// Read-only view of the source planes around one CU.
#[derive(Clone, Copy)]
pub(crate) struct SourceBlock<'a> {
    pub(crate) y: &'a [u16],
    pub(crate) cb: &'a [u16],
    pub(crate) cr: &'a [u16],
    pub(crate) yw: usize,
    pub(crate) yh: usize,
    pub(crate) cw: usize,
    pub(crate) chh: usize,
    /// Absolute luma position of the CU.
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) size: usize,
}

impl SourceBlock<'_> {
    #[inline]
    fn luma(&self, lx: usize, ly: usize) -> u16 {
        let x = (self.x0 + lx).min(self.yw - 1);
        let y = (self.y0 + ly).min(self.yh - 1);
        self.y[y * self.yw + x]
    }

    /// Chroma sample co-located with luma position `(lx, ly)`.
    #[inline]
    fn chroma(&self, comp: usize, lx: usize, ly: usize, sub_w: usize, sub_h: usize) -> u16 {
        if self.cw == 0 || self.chh == 0 {
            return 0;
        }
        let x = ((self.x0 + lx) / sub_w).min(self.cw - 1);
        let y = ((self.y0 + ly) / sub_h).min(self.chh - 1);
        let plane = if comp == 1 { self.cb } else { self.cr };
        plane[y * self.cw + x]
    }

    /// The color triple at index-grid cell `(row, col)`.
    #[inline]
    fn grid_sample(
        &self,
        row: usize,
        col: usize,
        transpose: bool,
        cfg: &PaletteConfig,
    ) -> [u16; MAX_COMPONENTS] {
        let (lx, ly) = if transpose { (row, col) } else { (col, row) };
        let mut out = [0u16; MAX_COMPONENTS];
        out[0] = self.luma(lx, ly);
        for (comp, slot) in out.iter_mut().enumerate().take(cfg.num_comps).skip(1) {
            *slot = self.chroma(comp, lx, ly, cfg.sub_w, cfg.sub_h);
        }
        out
    }
}

/// Reconstructed value of component `comp` at scan position `scan`.
#[inline]
fn reconstructed(cu: &PaletteCu, cfg: &PaletteConfig, scan: usize, comp: usize) -> u16 {
    let idx = cu.indices[scan];
    if idx == cu.palette_size as u32 {
        dequant_escape(
            cu.escapes[scan][comp],
            cfg.qp[comp],
            cfg.bit_depth(comp),
            cfg.lossless,
        )
    } else {
        cu.palette[idx as usize][comp]
    }
}

fn distinct_luma_exceeds(src: &SourceBlock<'_>, bd: u8, limit: u32) -> bool {
    let shift = u32::from(bd.saturating_sub(8));
    let mut seen = [0u64; 4];
    let mut count = 0u32;
    for ly in 0..src.size {
        for lx in 0..src.size {
            let v = usize::from(src.luma(lx, ly) >> shift).min(255);
            let (word, bit) = (v >> 6, 1u64 << (v & 63));
            if seen[word] & bit == 0 {
                seen[word] |= bit;
                count += 1;
                if count > limit {
                    return true;
                }
            }
        }
    }
    false
}

pub(crate) fn analyze_palette(
    src: &SourceBlock<'_>,
    predictor: &PalettePredictor,
    cfg: &PaletteConfig,
    seen: &mut ColorIndex,
    out: &mut PaletteCu,
) -> bool {
    let size = src.size;
    debug_assert!(size <= MAX_CU);
    // Palette mode only pays when colors actually repeat: it spends a full
    // sample value per new entry and buys back a short index run per sample. A
    // block whose colors are more than half distinct can never win that trade.
    // Luma alone already decides that for most photographic blocks.
    let repetition_limit = (size * size / 2) as u32;
    if distinct_luma_exceeds(src, cfg.bd_luma, repetition_limit.min(MAX_DISTINCT as u32)) {
        return false;
    }

    let mut colors = [[0u16; MAX_COMPONENTS]; MAX_DISTINCT];
    let mut counts = [0u32; MAX_DISTINCT];
    let mut distinct = 0usize;
    seen.clear();
    for row in 0..size {
        for col in 0..size {
            let sample = src.grid_sample(row, col, false, cfg);
            let (k, present) = seen.lookup_or_insert(color_key(&sample, cfg.num_comps), distinct);
            if present {
                counts[k] += 1;
            } else {
                if distinct == MAX_DISTINCT {
                    return false;
                }
                colors[distinct] = sample;
                counts[distinct] = 1;
                distinct += 1;
            }
        }
    }

    // Same rejection as the luma pre-gate above, now on the real color count.
    if distinct * 2 > size * size {
        return false;
    }

    let tol = color_tolerance(cfg);
    let mut order: [u8; MAX_DISTINCT] = std::array::from_fn(|i| i as u8);
    order[..distinct].sort_unstable_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));

    // Representative colors, most frequent first.
    let mut reps = [[0u16; MAX_COMPONENTS]; MAX_PALETTE_SIZE];
    let mut rep_count = 0usize;
    for &oi in &order[..distinct] {
        let color = colors[oi as usize];
        let merged = reps[..rep_count]
            .iter()
            .any(|rep| within(rep, &color, cfg.num_comps, tol));
        if merged {
            continue;
        }
        if rep_count == MAX_PALETTE_SIZE {
            break;
        }
        reps[rep_count] = color;
        rep_count += 1;
    }
    if rep_count == 0 {
        return false;
    }

    let pred_size = predictor.size().min(MAX_PREDICTOR_SIZE);
    out.pred_size = pred_size;
    out.reused[..pred_size].fill(false);
    let mut rep_taken = [false; MAX_PALETTE_SIZE];
    let mut palette_size = 0usize;
    for k in 0..pred_size {
        if palette_size == MAX_PALETTE_SIZE {
            break;
        }
        let entry = predictor.entry(k);
        let hit =
            (0..rep_count).find(|&r| !rep_taken[r] && within(&entry, &reps[r], cfg.num_comps, tol));
        if let Some(r) = hit {
            rep_taken[r] = true;
            out.reused[k] = true;
            out.palette[palette_size] = entry;
            palette_size += 1;
        }
    }
    let num_reused = palette_size;
    // Remaining representatives are signaled explicitly.
    for r in 0..rep_count {
        if rep_taken[r] || palette_size == MAX_PALETTE_SIZE {
            continue;
        }
        out.palette[palette_size] = reps[r];
        palette_size += 1;
    }
    out.palette_size = palette_size;
    out.num_signaled = palette_size - num_reused;
    out.size = size;
    out.num_comps = cfg.num_comps;
    out.tolerance = tol;
    palette_size != 0
}

/// Map every sample onto the CU palette (or an escape) for one setting of
/// `palette_transpose_flag`, and measure the resulting distortion. Returns
/// `false` when the block would be mostly escapes — an expensive way to spell
/// "intra".
pub(crate) fn assign_indices(
    src: &SourceBlock<'_>,
    cfg: &PaletteConfig,
    transpose: bool,
    out: &mut PaletteCu,
) -> bool {
    let size = out.size;
    let n = size * size;
    let palette_size = out.palette_size;
    let tol = out.tolerance;

    let mut escapes = 0usize;
    let escape_index = palette_size as u32;
    // The palette holds no duplicates, so a zero-distance entry is the unique
    // nearest one. That makes both shortcuts below — reusing the previous
    // sample's entry, and stopping the scan on an exact hit — free of any
    // effect on which index is picked.
    let mut previous = 0usize;
    for scan in 0..n {
        let (col, row) = scan_pos(scan, size);
        let sample = src.grid_sample(row, col, transpose, cfg);
        let mut best = 0usize;
        let mut best_err = u32::MAX;
        if max_abs(&out.palette[previous], &sample, cfg.num_comps) == 0 {
            best = previous;
            best_err = 0;
        } else {
            for (k, entry) in out.palette[..palette_size].iter().enumerate() {
                let err = max_abs(entry, &sample, cfg.num_comps);
                if err < best_err {
                    best_err = err;
                    best = k;
                    if err == 0 {
                        break;
                    }
                }
            }
        }
        previous = best;
        if best_err <= tol {
            out.indices[scan] = best as u32;
            out.escapes[scan] = [0; MAX_COMPONENTS];
        } else {
            out.indices[scan] = escape_index;
            escapes += 1;
            for (comp, &value) in sample[..cfg.num_comps].iter().enumerate() {
                out.escapes[scan][comp] =
                    quantize_escape(value, cfg.qp[comp], cfg.bit_depth(comp), cfg.lossless);
            }
        }
    }
    // A block that is mostly escapes is an expensive way to spell "intra".
    if escapes * 4 > n {
        return false;
    }
    out.escape_present = escapes != 0;
    out.transpose = transpose;

    // Unweighted, matching `cu_region_sse`: the palette J is compared directly
    // against the intra CU's J, so both must measure distortion the same way.
    let mut sse = 0.0f64;
    for comp in 0..cfg.num_comps {
        let (sw, sh) = cfg.sub(comp);
        for oy in 0..size / sh {
            for ox in 0..size / sw {
                let (lx, ly) = if transpose {
                    (oy * sh, ox * sw)
                } else {
                    (ox * sw, oy * sh)
                };
                let scan = scan_pos_inv(lx, ly, size);
                let rec = reconstructed(out, cfg, scan, comp);
                let orig = if comp == 0 {
                    src.luma(lx, ly)
                } else {
                    src.chroma(comp, lx, ly, cfg.sub_w, cfg.sub_h)
                };
                let d = f64::from(rec) - f64::from(orig);
                sse += d * d;
            }
        }
    }
    out.sse = sse as f32;
    true
}

#[inline]
fn within(a: &[u16; MAX_COMPONENTS], b: &[u16; MAX_COMPONENTS], comps: usize, tol: u32) -> bool {
    max_abs(a, b, comps) <= tol
}

#[inline]
fn max_abs(a: &[u16; MAX_COMPONENTS], b: &[u16; MAX_COMPONENTS], comps: usize) -> u32 {
    let mut worst = 0u32;
    for comp in 0..comps {
        let d = i32::from(a[comp]) - i32::from(b[comp]);
        worst = worst.max(d.unsigned_abs());
    }
    worst
}

/// How far two colors may differ and still share a palette entry: half the
/// luma reconstruction step, i.e. the error quantization would introduce anyway.
fn color_tolerance(cfg: &PaletteConfig) -> u32 {
    if cfg.lossless {
        return 0;
    }
    let qp = cfg.qp[0].max(0);
    let step = (level_scale(qp % 6) << (qp / 6)) as f64 / 64.0;
    ((step * 0.5).round() as i64).clamp(0, 16) as u32
}

#[derive(Clone, Copy)]
pub(crate) struct PaletteRun {
    copy_above: bool,
    /// Unadjusted `palette_index_idc` symbol (also the run-prefix context key).
    raw_index: u32,
    /// Number of *additional* samples after the first one.
    run: u32,
    scan: usize,
}

/// Segment the index map into COPY_INDEX / COPY_ABOVE runs, following exactly
/// the mode-availability rules the decoder's walk applies.
fn segment_runs(cu: &PaletteCu, runs: &mut Vec<PaletteRun>) {
    let size = cu.size;
    let n = size * size;
    let indices = &cu.indices[..n];
    runs.clear();

    let mut scan = 0usize;
    let mut prev_copy_above = false;
    while scan < n {
        let can_copy_above = scan >= size;
        let mut copy_above = false;
        let mut len = 1usize;

        // How far a COPY_INDEX run would reach.
        let value = indices[scan];
        let mut index_len = 1usize;
        while scan + index_len < n && indices[scan + index_len] == value {
            index_len += 1;
        }

        if can_copy_above && !prev_copy_above {
            let mut above_len = 0usize;
            while scan + above_len < n
                && indices[scan + above_len] == index_above(indices, scan + above_len, size)
            {
                above_len += 1;
            }
            if above_len > 0 && above_len >= index_len {
                copy_above = true;
                len = above_len;
            }
        }
        if !copy_above {
            len = index_len;
        }

        let raw_index = if copy_above {
            0
        } else if scan == 0 {
            value
        } else {
            let reference = if prev_copy_above {
                index_above(indices, scan, size)
            } else {
                indices[scan - 1]
            };
            debug_assert_ne!(value, reference, "run segmentation is not maximal");
            if value > reference { value - 1 } else { value }
        };

        runs.push(PaletteRun {
            copy_above,
            raw_index,
            run: (len - 1) as u32,
            scan,
        });
        prev_copy_above = copy_above;
        scan += len;
    }
}

/// Write `palette_coding()` (§7.3.8.13) for an analysed CU.
pub(crate) fn write_palette_cu<B: PaletteBitWriter>(
    bits: &mut B,
    cu: &PaletteCu,
    cfg: &PaletteConfig,
    x0: usize,
    y0: usize,
    runs: &mut Vec<PaletteRun>,
) {
    let size = cu.size;
    let n = size * size;

    // (1) predictor reuse runs
    write_reuse_flags(bits, cu.reuse_flags(), MAX_PALETTE_SIZE);

    // (2) num_signaled_palette_entries — present only when the palette is not
    // already full from reuse alone.
    let num_reused = cu.reuse_flags().iter().filter(|&&r| r).count();
    if num_reused < MAX_PALETTE_SIZE {
        write_eg0(bits, cu.num_signaled as u32);
    }

    // (3) new_palette_entries, component-major
    for comp in 0..cfg.num_comps {
        let nbits = u32::from(cfg.bit_depth(comp));
        for entry in &cu.palette[num_reused..cu.palette_size] {
            bits.bypass_bits(u32::from(entry[comp]), nbits);
        }
    }

    // (4) palette_escape_val_present_flag (inferred 1 for an empty palette)
    if cu.palette_size != 0 {
        bits.bypass(u8::from(cu.escape_present));
    }

    let max_palette_index = cu.max_palette_index();
    if max_palette_index > 0 {
        let mpi = max_palette_index as u32;
        segment_runs(cu, runs);

        // (5a) num_palette_indices_minus1 + palette_index_idc + final-run flag
        let index_runs = runs.iter().filter(|r| !r.copy_above).count();
        debug_assert!(index_runs >= 1);
        write_num_palette_indices(bits, index_runs as u32 - 1, mpi);
        let mut first = true;
        for run in runs.iter().filter(|r| !r.copy_above) {
            let alphabet = if first { mpi + 1 } else { mpi };
            write_truncated_binary(bits, run.raw_index, alphabet);
            first = false;
        }
        let final_copy_above = runs.last().map(|r| r.copy_above).unwrap_or(false);
        bits.run_type_flag(u8::from(final_copy_above));

        // (5b) transpose, then delta_qp / chroma_qp_offset, then the runs
        bits.transpose_flag(u8::from(cu.transpose));
        if cu.escape_present {
            bits.escape_qp_syntax();
        }

        let mut remaining_index_runs = index_runs;
        let mut prev_copy_above = false;
        for run in runs.iter() {
            let scan = run.scan;
            let can_copy_above = scan >= size;
            if can_copy_above && !prev_copy_above {
                if remaining_index_runs != 0 && scan + 1 < n {
                    bits.run_type_flag(u8::from(run.copy_above));
                } else {
                    debug_assert_eq!(
                        run.copy_above,
                        !(scan + 1 == n && remaining_index_runs != 0),
                        "inferred palette_run_type_flag disagrees with the chosen mode"
                    );
                }
            } else {
                debug_assert!(!run.copy_above);
            }
            if !run.copy_above {
                remaining_index_runs -= 1;
            }
            let last_run = remaining_index_runs == 0 && run.copy_above == final_copy_above;
            if !last_run {
                let reserved = remaining_index_runs + usize::from(final_copy_above);
                let max_run = (n - scan - 1 - reserved) as u32;
                write_run(bits, run.run, max_run, run.copy_above, run.raw_index);
            } else {
                debug_assert_eq!(scan + run.run as usize + 1, n);
            }
            prev_copy_above = run.copy_above;
        }
    } else if max_palette_index == 0 && cu.escape_present && cu.palette_size == 0 {
        // All-escape CU: the index map is inferred, but the QP syntax remains.
        bits.escape_qp_syntax();
    }

    // (6) palette_escape_val, component-major
    if cu.escape_present {
        let escape_index = cu.palette_size as u32;
        for comp in 0..cfg.num_comps {
            let nbits = u32::from(cfg.bit_depth(comp));
            for (scan, &idx) in cu.indices[..n].iter().enumerate() {
                if idx != escape_index {
                    continue;
                }
                if comp > 0 && !chroma_escape_present(cfg, cu.transpose, x0, y0, scan, size) {
                    continue;
                }
                let level = cu.escapes[scan][comp];
                if cfg.lossless {
                    bits.bypass_bits(level as u32, nbits);
                } else {
                    write_egk(bits, level as u32, 3);
                }
            }
        }
    }
}

/// §7.3.8.13 tests chroma-escape presence on the **absolute** luma coordinates
/// derived from the (untransposed) traverse scan.
#[inline]
fn chroma_escape_present(
    cfg: &PaletteConfig,
    transpose: bool,
    x0: usize,
    y0: usize,
    scan: usize,
    size: usize,
) -> bool {
    let (col, row) = scan_pos(scan, size);
    let xc = x0 + col;
    let yc = y0 + row;
    match cfg.chroma_idc {
        0 => false,
        1 => xc & 1 == 0 && yc & 1 == 0,
        2 if transpose => yc & 1 == 0,
        2 => xc & 1 == 0,
        3 => true,
        _ => false,
    }
}

/// Destination planes for palette reconstruction.
pub(crate) struct ReconTarget<'a> {
    pub(crate) y: &'a mut [u16],
    pub(crate) cb: &'a mut [u16],
    pub(crate) cr: &'a mut [u16],
    pub(crate) y_stride: usize,
    pub(crate) c_stride: usize,
    /// Coded (padded) plane heights.
    pub(crate) y_height: usize,
    pub(crate) c_height: usize,
}

/// Write the palette CU into the reconstruction planes exactly as §8.4.4.2.7
/// specifies, including the `palette_transpose_flag` remapping.
pub(crate) fn reconstruct(
    cu: &PaletteCu,
    cfg: &PaletteConfig,
    dst: &mut ReconTarget<'_>,
    x0: usize,
    y0: usize,
) {
    let size = cu.size;
    for comp in 0..cfg.num_comps {
        let (sw, sh) = cfg.sub(comp);
        let (stride, height) = if comp == 0 {
            (dst.y_stride, dst.y_height)
        } else {
            (dst.c_stride, dst.c_height)
        };
        if stride == 0 || height == 0 {
            continue;
        }
        let ox0 = x0 / sw;
        let oy0 = y0 / sh;
        for oy in 0..size / sh {
            let py = oy0 + oy;
            if py >= height {
                break;
            }
            for ox in 0..size / sw {
                let px = ox0 + ox;
                if px >= stride {
                    break;
                }
                let (lx, ly) = if cu.transpose {
                    (oy * sh, ox * sw)
                } else {
                    (ox * sw, oy * sh)
                };
                let scan = scan_pos_inv(lx, ly, size);
                let value = reconstructed(cu, cfg, scan, comp);
                let plane: &mut [u16] = match comp {
                    0 => dst.y,
                    1 => dst.cb,
                    _ => dst.cr,
                };
                plane[py * stride + px] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the bins a palette write produces, mirroring hpvcd's MockBits so
    /// the same buffer can be replayed through a hand-rolled reader.
    #[derive(Default)]
    struct Sink {
        bins: Vec<u8>,
        qp_syntax: usize,
    }

    impl PaletteBitWriter for Sink {
        fn run_type_flag(&mut self, bin: u8) {
            self.bins.push(bin);
        }
        fn transpose_flag(&mut self, bin: u8) {
            self.bins.push(bin);
        }
        fn run_prefix_bin(&mut self, _idx: usize, _copy_above: bool, bin: u8) {
            self.bins.push(bin);
        }
        fn run_prefix_index_bin(&mut self, _index: u32, bin: u8) {
            self.bins.push(bin);
        }
        fn bypass(&mut self, bin: u8) {
            self.bins.push(bin);
        }
        fn bypass_bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bins.push(((value >> i) & 1) as u8);
            }
        }
        fn escape_qp_syntax(&mut self) {
            self.qp_syntax += 1;
        }
    }

    struct Source {
        bins: Vec<u8>,
        pos: usize,
    }
    impl Source {
        fn bit(&mut self) -> u8 {
            let b = self.bins.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            b
        }
        fn bits(&mut self, n: u32) -> u32 {
            let mut v = 0;
            for _ in 0..n {
                v = (v << 1) | u32::from(self.bit());
            }
            v
        }
        fn eg0(&mut self) -> u32 {
            let mut prefix = 0;
            while self.bit() != 0 {
                prefix += 1;
            }
            let suffix = if prefix > 0 { self.bits(prefix) } else { 0 };
            ((1u32 << prefix) - 1) + suffix
        }
        fn tb(&mut self, alphabet: u32) -> u32 {
            if alphabet <= 1 {
                return 0;
            }
            let k = 31 - alphabet.leading_zeros();
            let u = (1u32 << (k + 1)) - alphabet;
            let prefix = self.bits(k);
            if prefix < u {
                prefix
            } else {
                (prefix << 1) + u32::from(self.bit()) - u
            }
        }
        fn num_indices(&mut self, max_palette_index: u32) -> u32 {
            let rice = 3 + ((max_palette_index + 1) >> 3);
            let mut prefix = 0;
            while prefix < 4 {
                if self.bit() == 0 {
                    return (prefix << rice) + self.bits(rice);
                }
                prefix += 1;
            }
            (4u32 << rice) + self.egk(rice + 1)
        }
        fn egk(&mut self, k: u32) -> u32 {
            let mut prefix = 0;
            while self.bit() != 0 {
                prefix += 1;
            }
            ((1u32 << prefix) - 1) * (1 << k) + self.bits(prefix + k)
        }
        fn run(&mut self, palette_max_run: u32) -> u32 {
            if palette_max_run == 0 {
                return 0;
            }
            let max_prefix = 32 - palette_max_run.leading_zeros();
            let mut prefix = 0;
            while prefix < max_prefix {
                if self.bit() == 0 {
                    break;
                }
                prefix += 1;
            }
            if prefix < 2 {
                return prefix;
            }
            let base = 1u32 << (prefix - 1);
            if palette_max_run == base {
                return base;
            }
            let suffix_max = if (base << 1) > palette_max_run {
                palette_max_run - base
            } else {
                base - 1
            };
            base + self.tb(suffix_max + 1)
        }
    }

    fn cfg444() -> PaletteConfig {
        PaletteConfig {
            num_comps: 3,
            chroma_idc: 3,
            sub_w: 1,
            sub_h: 1,
            bd_luma: 8,
            bd_chroma: 8,
            qp: [26, 26, 26],
            lossless: false,
        }
    }

    /// Replay a written index map through the decoder's walk (a direct port of
    /// hpvcd's `assign_index_runs`) and check it reproduces the input.
    fn replay_index_map(bins: Vec<u8>, size: usize, max_palette_index: u32) -> Vec<u32> {
        let n = size * size;
        let mut s = Source { bins, pos: 0 };
        let num_indices = (s.num_indices(max_palette_index) as usize + 1).min(n);
        let mut idc = Vec::with_capacity(num_indices);
        for i in 0..num_indices {
            let alphabet = if i == 0 {
                max_palette_index + 1
            } else {
                max_palette_index
            };
            idc.push(s.tb(alphabet));
        }
        let final_run_copy_above = s.bit() == COPY_ABOVE_MODE;
        let _transpose = s.bit();

        let mut indices = vec![0u32; n];
        let mut idc_pos = 0usize;
        let mut copy_index_runs = idc.len();
        let mut scan = 0usize;
        let mut prev_mode_copy_above = false;
        while scan < n {
            let can_copy_above = scan >= size;
            let copy_above = if can_copy_above && !prev_mode_copy_above {
                if copy_index_runs != 0 && scan + 1 < n {
                    s.bit() == COPY_ABOVE_MODE
                } else {
                    !(scan + 1 == n && copy_index_runs != 0)
                }
            } else {
                false
            };
            let (raw, value) = if copy_above {
                (0, 0)
            } else {
                let raw = idc.get(idc_pos).copied().unwrap_or(0);
                idc_pos += 1;
                copy_index_runs -= 1;
                let actual = if scan == 0 {
                    raw
                } else {
                    let reference = if prev_mode_copy_above {
                        index_above(&indices, scan, size)
                    } else {
                        indices[scan - 1]
                    };
                    if raw >= reference { raw + 1 } else { raw }
                };
                (raw, actual)
            };
            let _ = raw;
            let last_run = copy_index_runs == 0 && copy_above == final_run_copy_above;
            let run = if last_run {
                (n - scan - 1) as u32
            } else {
                let reserved = copy_index_runs + usize::from(final_run_copy_above);
                s.run((n - scan - 1 - reserved) as u32)
            };
            for _ in 0..=run {
                if scan >= n {
                    break;
                }
                indices[scan] = if copy_above {
                    index_above(&indices, scan, size)
                } else {
                    value
                };
                scan += 1;
            }
            prev_mode_copy_above = copy_above;
        }
        indices
    }

    fn write_index_map_only(cu: &PaletteCu) -> Vec<u8> {
        let mut sink = Sink::default();
        let mut runs = Vec::new();
        let mpi = cu.max_palette_index() as u32;
        segment_runs(cu, &mut runs);
        let index_runs = runs.iter().filter(|r| !r.copy_above).count();
        write_num_palette_indices(&mut sink, index_runs as u32 - 1, mpi);
        let mut first = true;
        for run in runs.iter().filter(|r| !r.copy_above) {
            write_truncated_binary(&mut sink, run.raw_index, if first { mpi + 1 } else { mpi });
            first = false;
        }
        let final_copy_above = runs.last().map(|r| r.copy_above).unwrap_or(false);
        sink.run_type_flag(u8::from(final_copy_above));
        sink.transpose_flag(u8::from(cu.transpose));

        let n = cu.size * cu.size;
        let mut remaining = index_runs;
        let mut prev_copy_above = false;
        for run in runs.iter() {
            let scan = run.scan;
            if scan >= cu.size && !prev_copy_above && remaining != 0 && scan + 1 < n {
                sink.run_type_flag(u8::from(run.copy_above));
            }
            if !run.copy_above {
                remaining -= 1;
            }
            let last_run = remaining == 0 && run.copy_above == final_copy_above;
            if !last_run {
                let reserved = remaining + usize::from(final_copy_above);
                write_run(
                    &mut sink,
                    run.run,
                    (n - scan - 1 - reserved) as u32,
                    run.copy_above,
                    run.raw_index,
                );
            }
            prev_copy_above = run.copy_above;
        }
        sink.bins
    }

    fn cu_with(size: usize, palette_size: usize, indices: &[u32]) -> Box<PaletteCu> {
        let mut cu = PaletteCu::new();
        cu.size = size;
        cu.num_comps = 3;
        cu.palette_size = palette_size;
        cu.escape_present = false;
        cu.indices[..indices.len()].copy_from_slice(indices);
        cu
    }

    #[test]
    fn exp_golomb_round_trips_through_the_decoder_binarization() {
        for k in 0..=3u32 {
            for value in 0..256u32 {
                let mut sink = Sink::default();
                write_egk(&mut sink, value, k);
                let mut source = Source {
                    bins: sink.bins,
                    pos: 0,
                };
                assert_eq!(source.egk(k), value, "k={k} value={value}");
            }
        }
    }

    #[test]
    fn truncated_binary_round_trips() {
        for alphabet in 2..40u32 {
            for value in 0..alphabet {
                let mut sink = Sink::default();
                write_truncated_binary(&mut sink, value, alphabet);
                let mut source = Source {
                    bins: sink.bins,
                    pos: 0,
                };
                assert_eq!(source.tb(alphabet), value, "alphabet={alphabet}");
            }
        }
    }

    #[test]
    fn num_palette_indices_round_trips_past_the_escape_threshold() {
        for max_palette_index in [1u32, 7, 8, 31] {
            for count in [0u32, 1, 5, 31, 32, 33, 200, 1023] {
                let mut sink = Sink::default();
                write_num_palette_indices(&mut sink, count, max_palette_index);
                let mut source = Source {
                    bins: sink.bins,
                    pos: 0,
                };
                assert_eq!(source.num_indices(max_palette_index), count);
            }
        }
    }

    #[test]
    fn run_length_round_trips_for_every_bound() {
        for palette_max_run in 0..40u32 {
            for run in 0..=palette_max_run {
                let mut sink = Sink::default();
                write_run(&mut sink, run, palette_max_run, false, 0);
                let mut source = Source {
                    bins: sink.bins,
                    pos: 0,
                };
                assert_eq!(source.run(palette_max_run), run, "max={palette_max_run}");
            }
        }
    }

    #[test]
    fn reuse_flags_round_trip() {
        let reused = [true, true, false, true, false, false];
        let mut sink = Sink::default();
        write_reuse_flags(&mut sink, &reused, MAX_PALETTE_SIZE);
        let mut source = Source {
            bins: sink.bins,
            pos: 0,
        };
        // Mirror of hpvcd::palette::decode_reuse_flags.
        let mut decoded = vec![false; reused.len()];
        let mut idx = 0usize;
        let mut num = 0usize;
        while idx < reused.len() && num < MAX_PALETTE_SIZE {
            let run = source.eg0();
            if run == 1 {
                break;
            }
            if run > 1 {
                idx += (run - 1) as usize;
            }
            if idx < reused.len() {
                decoded[idx] = true;
                num += 1;
                idx += 1;
            }
        }
        assert_eq!(decoded, reused.to_vec());
    }

    #[test]
    fn index_maps_round_trip_through_the_decoder_walk() {
        let cases: Vec<(usize, usize, Vec<u32>)> = vec![
            (2, 3, vec![0, 1, 2, 0]),
            (4, 4, vec![0, 1, 1, 2, 3, 3, 0, 0, 1, 2, 2, 2, 0, 0, 1, 3]),
            // A block whose second row copies the first exactly (COPY_ABOVE).
            (4, 2, vec![0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]),
            // Single run over the whole block.
            (4, 2, vec![1; 16]),
            // Alternating columns — worst case for run coding.
            (4, 2, vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]),
        ];
        for (size, palette_size, indices) in &cases {
            let cu = cu_with(*size, *palette_size, indices);
            let bins = write_index_map_only(&cu);
            let decoded = replay_index_map(bins, *size, cu.max_palette_index() as u32);
            assert_eq!(&decoded, indices, "size={size}");
        }
    }

    #[test]
    fn pseudo_random_index_maps_round_trip() {
        // Deterministic LCG so the case set is reproducible.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for size in [4usize, 8, 16] {
            for palette_size in [1usize, 2, 5, 9] {
                let n = size * size;
                let mut indices = vec![0u32; n];
                for slot in indices.iter_mut() {
                    *slot = (next() % palette_size as u64) as u32;
                }
                let cu = cu_with(size, palette_size, &indices);
                if cu.max_palette_index() <= 0 {
                    continue;
                }
                let bins = write_index_map_only(&cu);
                let decoded = replay_index_map(bins, size, cu.max_palette_index() as u32);
                assert_eq!(decoded, indices, "size={size} palette={palette_size}");
            }
        }
    }

    #[test]
    fn escape_quantization_is_the_inverse_of_the_normative_scaling() {
        for qp in [4i32, 16, 26, 37, 51] {
            let mut worst = 0i32;
            for value in 0..=255u16 {
                let level = quantize_escape(value, qp, 8, false);
                let rec = dequant_escape(level, qp, 8, false);
                worst = worst.max((i32::from(rec) - i32::from(value)).abs());
            }
            // Reconstruction error must stay inside one quantizer step.
            let step = ((level_scale(qp % 6) << (qp / 6)) as f64 / 64.0).ceil() as i32;
            assert!(worst <= step, "qp={qp} worst={worst} step={step}");
        }
    }

    #[test]
    fn predictor_update_moves_the_cu_palette_to_the_front() {
        let mut predictor = PalettePredictor::default();
        predictor.reset(3);
        predictor.update(&[[10, 11, 12], [20, 21, 22], [30, 31, 32]], &[], 64);
        predictor.update(&[[5, 5, 5], [6, 6, 6]], &[true, false, true], 64);
        // The CU palette leads; only the predictor entries that were not reused
        // survive behind it (entry 1 here).
        assert_eq!(predictor.size(), 3);
        assert_eq!(predictor.entry(0), [5, 5, 5]);
        assert_eq!(predictor.entry(1), [6, 6, 6]);
        assert_eq!(predictor.entry(2), [20, 21, 22]);
    }

    #[test]
    fn analysis_finds_an_exact_palette_for_a_two_color_block() {
        let size = 8;
        let mut y = vec![0u16; size * size];
        for (i, sample) in y.iter_mut().enumerate() {
            *sample = if (i / size) % 2 == 0 { 235 } else { 16 };
        }
        let cb = vec![128u16; size * size];
        let cr = vec![128u16; size * size];
        let src = SourceBlock {
            y: &y,
            cb: &cb,
            cr: &cr,
            yw: size,
            yh: size,
            cw: size,
            chh: size,
            x0: 0,
            y0: 0,
            size,
        };
        let cfg = cfg444();
        let predictor = PalettePredictor::default();
        let mut cu = PaletteCu::new();
        let mut seen = ColorIndex::new();
        assert!(analyze_palette(&src, &predictor, &cfg, &mut seen, &mut cu));
        assert!(assign_indices(&src, &cfg, false, &mut cu));
        assert_eq!(cu.palette_size, 2);
        assert!(!cu.escape_present);
        assert_eq!(cu.sse, 0.0);
    }

    #[test]
    fn analysis_rejects_photographic_blocks() {
        let size = 32;
        let mut y = vec![0u16; size * size];
        let mut state = 12345u32;
        for sample in y.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = ((state >> 16) & 0xFF) as u16;
        }
        let cb = vec![128u16; size * size];
        let cr = vec![128u16; size * size];
        let src = SourceBlock {
            y: &y,
            cb: &cb,
            cr: &cr,
            yw: size,
            yh: size,
            cw: size,
            chh: size,
            x0: 0,
            y0: 0,
            size,
        };
        let mut cfg = cfg444();
        cfg.qp = [4, 4, 4];
        let predictor = PalettePredictor::default();
        let mut cu = PaletteCu::new();
        let mut seen = ColorIndex::new();
        assert!(!analyze_palette(&src, &predictor, &cfg, &mut seen, &mut cu));
    }
}
