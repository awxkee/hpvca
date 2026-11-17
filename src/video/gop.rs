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
//! Closed-GOP planning and validation. Pure bookkeeping, no pixels: the
//! planner lays a GOP out in decode order and [`validate`] replays it through
//! the HEVC decoded picture buffer rules (§8.3.2 reference picture sets,
//! §C.5.2 output-order DPB) so an invalid structure is caught before a single
//! slice is coded.

// Whole-GOP planning and the temporal layer are exercised by the tests
// alone: the encoder plans one mini-GOP at a time and codes every picture in
// temporal layer 0.
#![allow(dead_code)]

use super::GopStructure;

/// Most pictures a reference picture set may hold (§7.4.8: at most
/// `sps_max_dec_pic_buffering_minus1`, itself at most 15).
const MAX_RPS_PICS: usize = 15;
/// Deepest B hierarchy the planner builds (mini-GOP of 32).
const MAX_B_DEPTH: u8 = 5;

/// `nal_unit_type` values the planner emits (Table 7-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum NalType {
    /// Trailing picture no later picture references.
    TrailN = 0,
    /// Trailing picture that may be referenced.
    TrailR = 1,
    /// IDR without leading pictures: closed GOPs never have any.
    IdrNLp = 20,
}

/// `slice_type` as coded in the slice header (Table 7-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SliceType {
    B = 0,
    P = 1,
    I = 2,
}

/// One picture of a GOP. A plan is a `Vec<FramePlan>` in decode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FramePlan {
    /// Picture order count: the output position inside the GOP, 0 for the IDR.
    pub(crate) poc: u32,
    pub(crate) nal_type: NalType,
    pub(crate) slice_type: SliceType,
    /// Depth in the B hierarchy; 0 for the IDR and P pictures. A picture only
    /// references pictures of an equal or lower value, so this is usable as
    /// `TemporalId`.
    pub(crate) temporal_id: u8,
    /// Slice QP offset relative to the keyframe QP.
    pub(crate) qp_offset: i8,
    /// Active RefPicList0 entries (POCs), nearest first.
    pub(crate) l0: Vec<u32>,
    /// Active RefPicList1 entries (POCs), nearest first.
    pub(crate) l1: Vec<u32>,
    /// The reference picture set, ascending POC: every earlier-decoded picture
    /// this one *or a later one* still references. A picture left out is
    /// dropped from the DPB for good (§8.3.2), so this is a superset of
    /// `l0 ∪ l1`.
    pub(crate) keep: Vec<u32>,
}

/// What a valid plan needs from the SPS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GopStats {
    /// `sps_max_dec_pic_buffering_minus1 + 1`.
    pub(crate) max_dec_pic_buffering: u8,
    /// `sps_max_num_reorder_pics`.
    pub(crate) max_num_reorder: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GopError {
    Empty,
    /// The first picture is not an IDR with POC 0, or a later one is an IDR.
    BadKeyframe {
        decode_idx: usize,
    },
    /// POCs are not a permutation of `0..len`.
    BadPoc {
        decode_idx: usize,
    },
    /// Reference lists do not fit the slice type, or are too long.
    BadRefLists {
        decode_idx: usize,
    },
    /// An active reference is missing from the picture's own RPS.
    RefNotInRps {
        decode_idx: usize,
        poc: u32,
    },
    /// The RPS names a picture that is not held for reference (not decoded
    /// yet, or left out of an earlier picture's RPS).
    MissingReference {
        decode_idx: usize,
        poc: u32,
    },
    /// A picture references one of a higher temporal layer.
    TemporalLayer {
        decode_idx: usize,
        poc: u32,
    },
    /// A `TRAIL_N` picture is referenced.
    NonReferenceUsed {
        decode_idx: usize,
        poc: u32,
    },
    /// The structure needs a larger DPB than the level allows.
    DpbOverflow {
        needed: usize,
        limit: u8,
    },
}

/// MaxDpbSize (§A.4.2) for a picture of `pic_samples` luma samples at
/// `level_idc`.
pub(crate) fn max_dpb_size(level_idc: u8, pic_samples: u64) -> u8 {
    let max_luma_ps: u64 = match level_idc {
        0..=30 => 36_864,
        31..=60 => 122_880,
        61..=63 => 245_760,
        64..=90 => 552_960,
        91..=93 => 983_040,
        94..=123 => 2_228_224,
        124..=156 => 8_912_896,
        _ => 35_651_584,
    };
    if pic_samples <= max_luma_ps >> 2 {
        16
    } else if pic_samples <= max_luma_ps >> 1 {
        12
    } else if pic_samples <= (3 * max_luma_ps) >> 2 {
        8
    } else {
        6
    }
}

/// Decides, frame by frame in display order, where GOPs start.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeyframePlacer {
    min_keyint: u32,
    max_keyint: u32,
    /// Frames since the last keyframe; `None` before the first frame.
    since_key: Option<u32>,
}

impl KeyframePlacer {
    pub(crate) fn new(min_keyint: u32, max_keyint: u32) -> Self {
        let max_keyint = max_keyint.max(1);
        KeyframePlacer {
            min_keyint: min_keyint.clamp(1, max_keyint),
            max_keyint,
            since_key: None,
        }
    }

    /// Frames coded since the last keyframe, itself not counted; 0 before
    /// the first frame.
    pub(crate) fn frames_since_key(&self) -> u32 {
        self.since_key.unwrap_or(0)
    }

    /// Whether the next frame starts a GOP. `scene_cut` is the detector's
    /// verdict and is ignored closer than `min_keyint` to the last keyframe;
    /// `force` (a caller request) is not.
    pub(crate) fn next(&mut self, scene_cut: bool, force: bool) -> bool {
        let key = match self.since_key {
            None => true,
            Some(n) => force || n + 1 >= self.max_keyint || (scene_cut && n + 1 >= self.min_keyint),
        };
        self.since_key = Some(if key {
            0
        } else {
            self.since_key.map_or(0, |n| n + 1)
        });
        key
    }
}

/// QP offset of P pictures against keyframes. Every later picture of the GOP
/// is predicted, directly or not, from the keyframe, so bits spent there go
/// furthest. Measured on real clips, BD-rate by VMAF: +1 / +2 / +3 gave −1.6
/// / −3.5 / −4.3% against none; later, with the lookahead lowering the QP of
/// what persists, +4 / +5 gave a further −2.0 / −2.5% against +3 (the gain
/// is on moving content; +5 costs 2% on a near-static clip).
const P_QP_OFFSET: i8 = 4;
/// QP offset of the first layer of B pictures against keyframes, two above
/// [`P_QP_OFFSET`]; each deeper layer adds one. On top of it B pictures take
/// the lookahead's QP deltas as P pictures do. Measured on ten clips at depth
/// 2 with those deltas, BD-rate by VMAF / PSNR against P pictures only: 5 / 6
/// / 7 gave −7.5 / −7.9 / −7.6% and −12.9 / −13.5 / −13.6%. Without the
/// deltas no value of this offset helped (3 to 6: within 1% of each other):
/// B pictures then came out 2 dB below the anchors beside them. No step
/// between layers cost 0.3% / 0.5%. In a GOP with B pictures a P offset of 3
/// instead of 4 gave −0.75 / −0.4%, with clips ±2.5% either way: not taken.
const B_QP_OFFSET_BASE: i8 = 6;

pub(crate) fn idr() -> FramePlan {
    FramePlan {
        poc: 0,
        nal_type: NalType::IdrNLp,
        slice_type: SliceType::I,
        temporal_id: 0,
        qp_offset: 0,
        l0: Vec::new(),
        l1: Vec::new(),
        keep: Vec::new(),
    }
}

fn p_frame(poc: u32, l0: Vec<u32>) -> FramePlan {
    FramePlan {
        poc,
        nal_type: NalType::TrailR,
        slice_type: SliceType::P,
        temporal_id: 0,
        qp_offset: P_QP_OFFSET,
        l0,
        l1: Vec::new(),
        keep: Vec::new(),
    }
}

/// The picture at `poc > 0` of a low-delay P GOP: it references the `refs`
/// pictures before it and keeps nothing else. A GOP of any length is these in
/// POC order after [`idr`], so a low-delay encoder needs no GOP length up
/// front.
pub(crate) fn low_delay_frame(poc: u32, refs: u8) -> FramePlan {
    let refs = u32::from(refs.max(1));
    let mut frame = p_frame(poc, (poc.saturating_sub(refs)..poc).rev().collect());
    frame.keep = (poc.saturating_sub(refs)..poc).collect();
    frame
}

/// The lowest `general_level_idc` (Table A.8) that fits a `width × height`
/// picture at `fps_num / fps_den` pictures per second in the Main tier, if
/// the picture and luma sample rate fit a defined level.
pub(crate) fn video_level_idc(width: u32, height: u32, fps_num: u32, fps_den: u32) -> Option<u8> {
    // (level_idc, MaxLumaPs, MaxLumaSr)
    static LEVELS: [(u8, u64, u64); 13] = [
        (30, 36_864, 552_960),
        (60, 122_880, 3_686_400),
        (63, 245_760, 7_372_800),
        (90, 552_960, 16_588_800),
        (93, 983_040, 33_177_600),
        (120, 2_228_224, 66_846_720),
        (123, 2_228_224, 133_693_440),
        (150, 8_912_896, 267_386_880),
        (153, 8_912_896, 534_773_760),
        (156, 8_912_896, 1_069_547_520),
        (180, 35_651_584, 1_069_547_520),
        (183, 35_651_584, 2_139_095_040),
        (186, 35_651_584, 4_278_190_080),
    ];
    let (w, h) = (u64::from(width), u64::from(height));
    let samples = w * h;
    let rate = u128::from(samples) * u128::from(fps_num);
    LEVELS
        .iter()
        .find(|&&(_, max_ps, max_sr)| {
            // §A.4.1: neither side may exceed sqrt(8 · MaxLumaPs).
            samples <= max_ps
                && rate <= u128::from(max_sr) * u128::from(fps_den)
                && w * w <= 8 * max_ps
                && h * h <= 8 * max_ps
        })
        .map(|&(idc, _, _)| idc)
}

/// Code the pictures strictly between the already-coded `lo` and `hi`:
/// the midpoint as a B picture predicted from both, then each half.
fn bisect(plan: &mut Vec<FramePlan>, lo: u32, hi: u32, temporal_id: u8) {
    if hi - lo < 2 {
        return;
    }
    let mid = lo + (hi - lo) / 2;
    let referenced = mid - lo > 1 || hi - mid > 1;
    plan.push(FramePlan {
        poc: mid,
        nal_type: if referenced {
            NalType::TrailR
        } else {
            NalType::TrailN
        },
        slice_type: SliceType::B,
        temporal_id,
        qp_offset: B_QP_OFFSET_BASE + temporal_id as i8 - 1,
        l0: vec![lo],
        l1: vec![hi],
        keep: Vec::new(),
    });
    bisect(plan, lo, mid, temporal_id + 1);
    bisect(plan, mid, hi, temporal_id + 1);
}

/// The pictures after an already-coded anchor (the IDR or a P picture) at
/// POC `anchor`, up to and including the next anchor `len` pictures later, in
/// decode order: that P picture, then the B pictures between the two. Every
/// GOP [`plan_gop`] lays out for [`GopStructure::HierarchicalB`] is its IDR
/// followed by such mini-GOPs, so an encoder can plan them one at a time
/// without knowing where the GOP ends: `len` is `1 << depth` except in front
/// of a keyframe or the end of the stream.
pub(crate) fn mini_gop(anchor: u32, len: u32) -> Vec<FramePlan> {
    debug_assert!(len >= 1);
    // Planned relative to the anchor, which stands at POC 0 like an IDR.
    let mut plan = vec![idr(), p_frame(len, vec![0])];
    bisect(&mut plan, 0, len, 1);
    assign_rps(&mut plan);
    plan.remove(0);
    for frame in &mut plan {
        frame.poc += anchor;
        for poc in frame
            .l0
            .iter_mut()
            .chain(&mut frame.l1)
            .chain(&mut frame.keep)
        {
            *poc += anchor;
        }
    }
    plan
}

/// The structure the encoder will actually use — `structure` with its B
/// hierarchy made shallower until the DPB can hold it — and what GOPs of any
/// length built from it need from the SPS.
pub(crate) fn steady_state(
    structure: GopStructure,
    max_dpb: u8,
) -> Result<(GopStructure, GopStats), GopError> {
    let mut structure = clamp_structure(structure, max_dpb);
    loop {
        let mini = match structure {
            GopStructure::LowDelayP { .. } => u32::from(max_dpb) + 1,
            GopStructure::HierarchicalB { depth } => 1 << depth,
        };
        // A full mini-GOP followed by every shorter one a GOP can end with.
        let mut worst: Result<GopStats, GopError> = Err(GopError::Empty);
        for tail in 1..=mini {
            match (
                validate(&build(1 + mini + tail, structure), max_dpb),
                &mut worst,
            ) {
                (Ok(stats), Ok(worst)) => {
                    worst.max_dec_pic_buffering =
                        worst.max_dec_pic_buffering.max(stats.max_dec_pic_buffering);
                    worst.max_num_reorder = worst.max_num_reorder.max(stats.max_num_reorder);
                }
                (Ok(stats), worst @ Err(GopError::Empty)) => *worst = Ok(stats),
                (Err(e), worst) => {
                    *worst = Err(e);
                    break;
                }
                (Ok(_), Err(_)) => break,
            }
        }
        match (worst, structure) {
            (Ok(stats), _) => return Ok((structure, stats)),
            (Err(GopError::DpbOverflow { .. }), GopStructure::HierarchicalB { depth })
                if depth > 0 =>
            {
                structure = GopStructure::HierarchicalB { depth: depth - 1 };
            }
            (Err(e), _) => return Err(e),
        }
    }
}

fn clamp_structure(structure: GopStructure, max_dpb: u8) -> GopStructure {
    match structure {
        GopStructure::LowDelayP { refs } => GopStructure::LowDelayP {
            refs: refs.clamp(1, max_dpb.saturating_sub(1).clamp(1, MAX_RPS_PICS as u8)),
        },
        GopStructure::HierarchicalB { depth } => GopStructure::HierarchicalB {
            depth: depth.min(MAX_B_DEPTH),
        },
    }
}

/// Fill every picture's `keep` from the active reference lists: a decoded
/// picture stays in the RPS up to the last picture (in decode order) that
/// references it. The last picture of a plan is never needed by a later GOP,
/// since GOPs are closed.
fn assign_rps(plan: &mut [FramePlan]) {
    let mut last_use = vec![0usize; plan.len()];
    for (i, f) in plan.iter().enumerate() {
        for &r in f.l0.iter().chain(&f.l1) {
            if let Some(slot) = last_use.get_mut(r as usize) {
                *slot = i;
            }
        }
    }
    let mut live: Vec<u32> = Vec::new();
    for (i, f) in plan.iter_mut().enumerate() {
        live.retain(|&p| last_use[p as usize] >= i);
        f.keep.clear();
        f.keep.extend_from_slice(&live);
        f.keep.sort_unstable();
        if last_use.get(f.poc as usize).is_some_and(|&u| u > i) {
            live.push(f.poc);
        }
    }
}

fn build(len: u32, structure: GopStructure) -> Vec<FramePlan> {
    let mut plan = Vec::with_capacity(len as usize);
    plan.push(idr());
    match structure {
        GopStructure::LowDelayP { refs } => {
            plan.extend((1..len).map(|poc| low_delay_frame(poc, refs)));
        }
        GopStructure::HierarchicalB { depth } => {
            let mini = 1u32 << depth.min(MAX_B_DEPTH);
            let mut anchor = 0;
            while anchor + 1 < len {
                let next = (anchor + mini).min(len - 1);
                plan.push(p_frame(next, vec![anchor]));
                bisect(&mut plan, anchor, next, 1);
                anchor = next;
            }
        }
    }
    assign_rps(&mut plan);
    plan
}

/// Plan one closed GOP of `len` frames whose DPB need stays within
/// `max_dpb` (see [`max_dpb_size`]). A structure too large for the DPB is
/// shrunk — fewer P references, shallower B hierarchy — rather than rejected.
pub(crate) fn plan_gop(
    len: u32,
    structure: GopStructure,
    max_dpb: u8,
) -> Result<(Vec<FramePlan>, GopStats), GopError> {
    if len == 0 {
        return Err(GopError::Empty);
    }
    let mut structure = clamp_structure(structure, max_dpb);
    loop {
        let plan = build(len, structure);
        match validate(&plan, max_dpb) {
            Ok(stats) => return Ok((plan, stats)),
            Err(error @ GopError::DpbOverflow { .. }) => match structure {
                GopStructure::HierarchicalB { depth } if depth > 0 => {
                    structure = GopStructure::HierarchicalB { depth: depth - 1 };
                }
                _ => return Err(error),
            },
            Err(e) => return Err(e),
        }
    }
}

/// Fenwick tree over POCs, for counting already-decoded pictures above a POC.
struct Counts(Vec<u32>);

impl Counts {
    fn add(&mut self, mut i: usize) {
        i += 1;
        while i < self.0.len() {
            self.0[i] += 1;
            i += i & i.wrapping_neg();
        }
    }
    /// Entries added at indices `0..=i`.
    fn prefix(&self, mut i: usize) -> u32 {
        i += 1;
        let mut sum = 0;
        while i > 0 {
            sum += self.0[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }
}

struct DpbPic {
    poc: u32,
    temporal_id: u8,
    nal_type: NalType,
    reference: bool,
    awaiting_output: bool,
}

/// Output ("bump") the lowest-POC pictures until at most `reorder` wait, and
/// free the ones no longer held for reference (§C.5.2.2 / §C.5.2.3).
fn bump(dpb: &mut Vec<DpbPic>, reorder: usize) {
    while dpb.iter().filter(|p| p.awaiting_output).count() > reorder {
        let Some(first) = dpb
            .iter_mut()
            .filter(|p| p.awaiting_output)
            .min_by_key(|p| p.poc)
        else {
            return;
        };
        first.awaiting_output = false;
        dpb.retain(|p| p.reference || p.awaiting_output);
    }
}

/// Check a decode-order plan of one closed GOP against the HEVC reference and
/// DPB rules, and derive the SPS values it needs. `max_dpb` is the level's
/// MaxDpbSize.
pub(crate) fn validate(plan: &[FramePlan], max_dpb: u8) -> Result<GopStats, GopError> {
    if plan.is_empty() {
        return Err(GopError::Empty);
    }
    let n = plan.len();

    // sps_max_num_reorder_pics: the most pictures that precede any picture in
    // decode order yet follow it in output order (§7.4.3.2.1).
    let mut seen = vec![false; n];
    let mut counts = Counts(vec![0; n + 1]);
    let mut reorder = 0usize;
    for (i, f) in plan.iter().enumerate() {
        let poc = f.poc as usize;
        if poc >= n || seen[poc] {
            return Err(GopError::BadPoc { decode_idx: i });
        }
        seen[poc] = true;
        reorder = reorder.max(i - counts.prefix(poc) as usize);
        counts.add(poc);
    }

    let mut dpb: Vec<DpbPic> = Vec::new();
    let mut needed = 1usize;
    for (i, f) in plan.iter().enumerate() {
        let is_idr = f.nal_type == NalType::IdrNLp;
        if is_idr != (i == 0) || (is_idr && f.poc != 0) {
            return Err(GopError::BadKeyframe { decode_idx: i });
        }
        let lists_ok = match f.slice_type {
            SliceType::I => f.l0.is_empty() && f.l1.is_empty(),
            SliceType::P => !f.l0.is_empty() && f.l1.is_empty(),
            SliceType::B => !f.l0.is_empty() && !f.l1.is_empty(),
        };
        if !lists_ok
            || (is_idr && (f.slice_type != SliceType::I || !f.keep.is_empty()))
            || f.l0.len() > MAX_RPS_PICS
            || f.l1.len() > MAX_RPS_PICS
            || f.keep.len() > MAX_RPS_PICS
        {
            return Err(GopError::BadRefLists { decode_idx: i });
        }
        for &poc in f.l0.iter().chain(&f.l1) {
            if !f.keep.contains(&poc) {
                return Err(GopError::RefNotInRps { decode_idx: i, poc });
            }
        }
        // Apply the RPS: everything it names must still be held for reference
        // (a picture an earlier RPS left out may linger awaiting output, but
        // can never become a reference again); everything else stops being one.
        for &poc in &f.keep {
            let Some(pic) = dpb.iter().find(|p| p.poc == poc && p.reference) else {
                return Err(GopError::MissingReference { decode_idx: i, poc });
            };
            if pic.temporal_id > f.temporal_id {
                return Err(GopError::TemporalLayer { decode_idx: i, poc });
            }
            if pic.nal_type == NalType::TrailN {
                return Err(GopError::NonReferenceUsed { decode_idx: i, poc });
            }
        }
        for pic in &mut dpb {
            pic.reference = f.keep.contains(&pic.poc);
        }
        dpb.retain(|p| p.reference || p.awaiting_output);
        bump(&mut dpb, reorder);
        // The current picture needs a free buffer on top of what is left.
        needed = needed.max(dpb.len() + 1);
        dpb.push(DpbPic {
            poc: f.poc,
            temporal_id: f.temporal_id,
            nal_type: f.nal_type,
            reference: true,
            awaiting_output: true,
        });
        bump(&mut dpb, reorder);
    }
    if needed > usize::from(max_dpb) {
        return Err(GopError::DpbOverflow {
            needed,
            limit: max_dpb,
        });
    }
    Ok(GopStats {
        max_dec_pic_buffering: needed as u8,
        max_num_reorder: reorder as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pocs(plan: &[FramePlan]) -> Vec<u32> {
        plan.iter().map(|f| f.poc).collect()
    }

    #[test]
    fn low_delay_p_is_valid_for_every_length() {
        for refs in [1u8, 2, 4] {
            for len in 1..=40 {
                let (plan, stats) = plan_gop(len, GopStructure::LowDelayP { refs }, 6).unwrap();
                assert_eq!(pocs(&plan), (0..len).collect::<Vec<_>>());
                assert_eq!(stats.max_num_reorder, 0);
                let expect = (u32::from(refs).min(len - 1) + 1) as u8;
                assert_eq!(stats.max_dec_pic_buffering, expect, "refs {refs} len {len}");
            }
        }
    }

    #[test]
    fn low_delay_p_refs_are_clamped_to_the_dpb() {
        let (plan, stats) = plan_gop(40, GopStructure::LowDelayP { refs: 16 }, 6).unwrap();
        assert_eq!(stats.max_dec_pic_buffering, 6);
        assert_eq!(plan[39].l0, vec![38, 37, 36, 35, 34]);
    }

    #[test]
    fn hierarchical_b_decode_order() {
        let (plan, stats) = plan_gop(9, GopStructure::HierarchicalB { depth: 3 }, 16).unwrap();
        assert_eq!(pocs(&plan), vec![0, 8, 4, 2, 1, 3, 6, 5, 7]);
        assert_eq!(stats.max_num_reorder, 3);
        // POC 1 is predicted from 0 and 2 while 4 and 8 wait for later pictures.
        assert_eq!(plan[4].keep, vec![0, 2, 4, 8]);
        assert_eq!(plan[4].nal_type, NalType::TrailN);
        assert_eq!(plan[2].nal_type, NalType::TrailR);
        assert_eq!(plan[4].temporal_id, 3);
    }

    #[test]
    fn hierarchical_b_is_valid_for_every_length() {
        for depth in 0..=MAX_B_DEPTH {
            for len in 1..=80 {
                let (plan, stats) =
                    plan_gop(len, GopStructure::HierarchicalB { depth }, 16).unwrap();
                assert_eq!(plan.len(), len as usize);
                assert!(stats.max_num_reorder <= depth, "depth {depth} len {len}");
                assert_eq!(validate(&plan, 16), Ok(stats));
            }
        }
    }

    #[test]
    fn mini_gops_make_up_the_planned_gop() {
        for depth in 0..=4u8 {
            let mini = 1u32 << depth;
            for len in 2..=40u32 {
                let (whole, _) = plan_gop(len, GopStructure::HierarchicalB { depth }, 16).unwrap();
                let mut pieces = vec![idr()];
                let mut anchor = 0;
                while anchor + 1 < len {
                    let step = mini.min(len - 1 - anchor);
                    pieces.extend(mini_gop(anchor, step));
                    anchor += step;
                }
                assert_eq!(pieces, whole, "depth {depth} len {len}");
            }
        }
    }

    #[test]
    fn steady_state_covers_every_gop_length() {
        for depth in 0..=4u8 {
            let structure = GopStructure::HierarchicalB { depth };
            let (used, stats) = steady_state(structure, 16).unwrap();
            assert_eq!(used, structure);
            for len in 1..=70 {
                let (_, needs) = plan_gop(len, structure, 16).unwrap();
                assert!(needs.max_dec_pic_buffering <= stats.max_dec_pic_buffering);
                assert!(needs.max_num_reorder <= stats.max_num_reorder);
            }
        }
        // A hierarchy the DPB cannot hold is made shallower.
        let (used, stats) = steady_state(GopStructure::HierarchicalB { depth: 5 }, 4).unwrap();
        let GopStructure::HierarchicalB { depth } = used else {
            panic!("still a B hierarchy");
        };
        assert!(depth < 5 && stats.max_dec_pic_buffering <= 4);
        let (used, stats) = steady_state(GopStructure::LowDelayP { refs: 1 }, 6).unwrap();
        assert_eq!(used, GopStructure::LowDelayP { refs: 1 });
        assert_eq!((stats.max_dec_pic_buffering, stats.max_num_reorder), (2, 0));

        let (used, _) = steady_state(GopStructure::HierarchicalB { depth: u8::MAX }, 16).unwrap();
        assert_eq!(used, GopStructure::HierarchicalB { depth: MAX_B_DEPTH });
    }

    #[test]
    fn hierarchical_b_depth_shrinks_to_fit_the_dpb() {
        let deep = plan_gop(65, GopStructure::HierarchicalB { depth: 5 }, 16)
            .unwrap()
            .1;
        assert!(deep.max_dec_pic_buffering > 4);
        let (plan, stats) = plan_gop(65, GopStructure::HierarchicalB { depth: 5 }, 4).unwrap();
        assert!(stats.max_dec_pic_buffering <= 4);
        assert!(stats.max_num_reorder < deep.max_num_reorder);
        assert_eq!(validate(&plan, 4), Ok(stats));
    }

    #[test]
    fn validator_rejects_broken_plans() {
        let good = plan_gop(9, GopStructure::HierarchicalB { depth: 3 }, 16)
            .unwrap()
            .0;

        // POC 4 dropped from the RPS at POC 1: POC 3 can no longer find it.
        let mut plan = good.clone();
        plan[4].keep.retain(|&p| p != 4);
        assert_eq!(
            validate(&plan, 16),
            Err(GopError::MissingReference {
                decode_idx: 5,
                poc: 4
            })
        );

        // Referencing a picture that is decoded later.
        let mut plan = good.clone();
        plan[2].l0 = vec![2];
        plan[2].keep.push(2);
        assert_eq!(
            validate(&plan, 16),
            Err(GopError::MissingReference {
                decode_idx: 2,
                poc: 2
            })
        );

        // Active reference missing from the picture's own RPS.
        let mut plan = good.clone();
        plan[2].keep.retain(|&p| p != 8);
        assert_eq!(
            validate(&plan, 16),
            Err(GopError::RefNotInRps {
                decode_idx: 2,
                poc: 8
            })
        );

        // A lower layer (POC 6) referencing a higher one (POC 3).
        let mut plan = good.clone();
        plan[6].l0 = vec![3];
        plan[6].keep = vec![3, 4, 8];
        assert_eq!(
            validate(&plan, 16),
            Err(GopError::TemporalLayer {
                decode_idx: 6,
                poc: 3
            })
        );

        // Referencing a TRAIL_N picture of the same layer (POC 3 from POC 1).
        let mut plan = good.clone();
        plan[5].l0 = vec![1];
        plan[5].keep = vec![1, 2, 4, 8];
        assert_eq!(
            validate(&plan, 16),
            Err(GopError::NonReferenceUsed {
                decode_idx: 5,
                poc: 1
            })
        );

        // Two IDRs, a duplicated POC, a DPB that is too small.
        let mut plan = good.clone();
        plan[1] = idr();
        assert_eq!(validate(&plan, 16), Err(GopError::BadPoc { decode_idx: 1 }));
        let mut plan = good.clone();
        plan[1].nal_type = NalType::IdrNLp;
        assert_eq!(
            validate(&plan, 16),
            Err(GopError::BadKeyframe { decode_idx: 1 })
        );
        assert!(matches!(
            validate(&good, 3),
            Err(GopError::DpbOverflow { limit: 3, .. })
        ));
    }

    #[test]
    fn levels_follow_size_and_frame_rate() {
        assert_eq!(video_level_idc(1920, 1080, 30, 1), Some(120)); // 4
        assert_eq!(video_level_idc(1920, 1080, 60, 1), Some(123)); // 4.1
        assert_eq!(video_level_idc(3840, 2160, 30, 1), Some(150)); // 5
        assert_eq!(video_level_idc(3840, 2160, 60, 1), Some(153)); // 5.1
        assert_eq!(video_level_idc(1280, 720, 30000, 1001), Some(93)); // 3.1
        assert_eq!(video_level_idc(352, 288, 30, 1), Some(60)); // 2
        // Small enough in area for level 1, but 1024 exceeds sqrt(8 · MaxLumaPs)
        // of levels 1 and 2.
        assert_eq!(video_level_idc(1024, 32, 1, 1), Some(63));
        assert_eq!(video_level_idc(8192, 8192, 1, 1), None);
        assert_eq!(video_level_idc(1920, 1080, 10000, 1), None);
    }

    #[test]
    fn level_dpb_sizes() {
        // 1080p at level 4 / 4.1 fills the level: 6 pictures.
        assert_eq!(max_dpb_size(120, 1920 * 1080), 6);
        assert_eq!(max_dpb_size(123, 1920 * 1080), 6);
        // 720p at level 4: under half of MaxLumaPs.
        assert_eq!(max_dpb_size(120, 1280 * 720), 12);
        // 1080p at level 5.1: under a quarter.
        assert_eq!(max_dpb_size(153, 1920 * 1080), 16);
        assert_eq!(max_dpb_size(150, 3840 * 2160), 6);
    }

    #[test]
    fn keyframe_placement() {
        let mut k = KeyframePlacer::new(3, 6);
        // First frame is always a keyframe; a cut one frame later is too close.
        assert!(k.next(false, false));
        assert!(!k.next(true, false));
        assert!(!k.next(false, false));
        // Third frame after the keyframe: a cut is honoured.
        assert!(k.next(true, false));
        // No cuts: forced at max_keyint.
        let run: Vec<bool> = (0..6).map(|_| k.next(false, false)).collect();
        assert_eq!(run, [false, false, false, false, false, true]);
        // A caller-forced keyframe ignores min_keyint.
        assert!(k.next(false, true));
        // max_keyint 1 is intra-only.
        let mut intra = KeyframePlacer::new(25, 1);
        assert!((0..4).all(|_| intra.next(false, false)));
    }
}
