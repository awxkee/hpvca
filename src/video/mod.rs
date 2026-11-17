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
//! Video (multi-picture) encoding. **Under construction.**
//!
//! [`VideoEncoder`] produces a valid HEVC elementary stream of closed GOPs
//! with motion-compensated P pictures and, with
//! [`GopStructure::HierarchicalB`], B pictures between them: each coding
//! unit is merged with a neighbor's motion (skipped or with a residual),
//! given searched quarter-sample vectors — into the picture before, the one
//! after, or both averaged — or intra coded, whichever costs less. A short
//! lookahead places keyframes at hard scene cuts, steers quantization toward
//! what later pictures keep, and finds the fades that weighted prediction
//! then compensates.
//!
//! Each reference picture list holds one picture and prediction units are
//! always the whole coding unit. Rate control is not implemented yet.

mod gop;
mod lookahead;
mod md5;

use crate::hevc::{
    InterPic, IntraHint, MotionHint, Nalu, PredWeights, QpHint, ReconFrame, RefPics, SeqInfo,
    Shortcuts, VideoStream,
};
use crate::{BitDepth, ChromaFormat, EncodeConfig, EncodeError, Yuv};
use std::collections::VecDeque;
use std::sync::Arc;

/// How scene changes move keyframes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SceneCut {
    /// Keyframes only every [`GopConfig::max_keyint`] frames or when forced.
    /// Also skip the lookahead's motion, weighting and persistence analysis.
    Off,
    /// Start a new GOP at hard cuts, subject to [`GopConfig::min_keyint`].
    ///
    /// A cut is a frame the previous one cannot predict, standing alone
    /// between frames that are predictable: fades, dissolves and bursts of
    /// fast motion are several such frames in a row and get no keyframe. With
    /// [`VideoConfig::lookahead`] frames to look at, a flash — the picture
    /// returning to what it was within that many frames — is not taken for a
    /// cut either.
    #[default]
    Adaptive,
}

/// Prediction structure between keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GopStructure {
    /// Keep up to `refs` preceding pictures in the decoded picture buffer.
    /// Each P picture predicts from the most recent one. Packets come in
    /// display order.
    LowDelayP { refs: u8 },
    /// Hierarchical B pictures between P anchors up to `1 << depth` frames
    /// apart: depth 1 is one B picture between two anchors, depth 2 three of
    /// them with the middle one predicting the other two. Anchors come
    /// closer in front of a keyframe and where the lookahead finds the far
    /// one too hard to predict. Packets come in decoding order, which
    /// differs from display order, and the encoder holds back up to
    /// `(1 << depth) - 1` more frames than the lookahead. The depth is at
    /// most 5, and is lowered automatically when the level's decoded picture
    /// buffer cannot hold the structure.
    HierarchicalB { depth: u8 },
}

impl Default for GopStructure {
    fn default() -> Self {
        GopStructure::LowDelayP { refs: 1 }
    }
}

/// Group-of-pictures layout. Every GOP is closed: it starts with an IDR
/// picture and nothing in it references a picture before that IDR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GopConfig {
    /// Largest distance between keyframes, in frames. `1` is intra-only.
    pub max_keyint: u32,
    /// Smallest distance between keyframes a scene change may produce. Keeps
    /// flashes and rapid cuts from closing a GOP right after it opened.
    /// Forced keyframes ignore it.
    pub min_keyint: u32,
    /// Scene-change handling.
    pub scene_cut: SceneCut,
    /// Prediction structure inside a GOP.
    pub structure: GopStructure,
}

impl Default for GopConfig {
    fn default() -> Self {
        GopConfig {
            max_keyint: 250,
            min_keyint: 25,
            scene_cut: SceneCut::default(),
            structure: GopStructure::default(),
        }
    }
}

/// Settings of a [`VideoEncoder`].
#[derive(Clone, Debug)]
pub struct VideoConfig {
    /// Per-picture settings shared with still images: quality, speed,
    /// threads, SAO, colour. The thread preference is capped at the available
    /// CPU parallelism. Lossless, screen content, gain maps, EXIF and
    /// orientation are not available for video; `chroma` is ignored because
    /// the frames carry their own format.
    pub base: EncodeConfig,
    /// Frames per second as a fraction, e.g. `(30000, 1001)`.
    pub frame_rate: (u32, u32),
    /// Keyframe placement and prediction structure.
    pub gop: GopConfig,
    /// Frames the encoder holds back to look ahead, which delays its output
    /// by as many and keeps a copy of each. Scene-cut detection uses them to
    /// tell a flash from a cut, and quantization favors what they show to
    /// persist; `0` still detects cuts. The default is 16: against it, 4 / 8 /
    /// 30 frames measured +6 / +1.6 / −0.5% bitrate at equal quality.
    pub lookahead: usize,
    /// Append a decoded picture hash SEI (MD5) to every picture, so a decoder
    /// can verify its output is bit-exact. Costs 49 bytes per picture.
    pub picture_hash: bool,
    /// How much of the mode decision to spend on P and B pictures.
    pub preset: VideoPreset,
}

/// How thoroughly P and B pictures are searched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VideoPreset {
    /// Every trial that has measured as worth its time.
    #[default]
    Quality,
    /// Leaves out trials that seldom win: about a quarter less encode time
    /// for about 2% more bitrate at the same quality.
    Fast,
}

impl Default for VideoConfig {
    fn default() -> Self {
        VideoConfig {
            base: EncodeConfig::default(),
            frame_rate: (30, 1),
            gop: GopConfig::default(),
            lookahead: 16,
            picture_hash: false,
            preset: VideoPreset::default(),
        }
    }
}

/// How a coded picture depends on others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameKind {
    /// IDR picture: decoding can start here. Its packet begins with the
    /// parameter sets.
    Key,
    /// Predicted from earlier pictures.
    P,
    /// Predicted from an earlier and a later picture.
    B,
}

/// One coded picture.
#[derive(Clone, Debug)]
pub struct Packet {
    /// The picture's NAL units in Annex B byte-stream format (start-code
    /// prefixed). Concatenating every packet gives a playable `.h265` file.
    pub data: Vec<u8>,
    /// Presentation timestamp, as passed to [`VideoEncoder::send_frame`].
    pub pts: i64,
    /// Decoding timestamp. Equal to `pts` while there is no reordering; with
    /// B pictures it runs behind by the reorder depth, starting below the
    /// first `pts`, so that it never exceeds a packet's `pts`.
    pub dts: i64,
    pub kind: FrameKind,
}

/// Change score ([`lookahead::change_score`]) below which SAO is not tried
/// for a P picture. Measured on four real clips with SAO otherwise always on:
/// 0.03 / 0.06 / 0.1 / 0.2 gave −1.9 / −6.6 / −5.7 / −5.8% BD-rate by VMAF.
/// One clip's frames score about 0.04 and gain 20% from losing SAO, which is
/// what separates 0.03 from 0.06; above the threshold the per-picture
/// decision in the encoder bounds the cost of being wrong.
const SAO_MIN_CHANGE: f32 = 0.06;

/// Weighted prediction is used when it lowers a frame's change score to
/// under this share of the plain score.
const WEIGHTED_GAIN: f32 = 0.9;

/// QP reduction per doubling of a block's worth to later pictures
/// ([`lookahead::propagation`]). Measured on six real clips with a 16-frame
/// lookahead, BD-rate by VMAF / PSNR against 1.5 on co-located blocks only:
/// 1.5 / 2.0 / 2.5 / 3.0 gave −0.4 / −2.7 / −3.6 / −3.8% and −0.3 / −1.9 /
/// −2.6 / −2.7%. (x264's macroblock tree uses 2.0.)
const PERSISTENCE_STRENGTH: f32 = 2.5;

/// With B pictures, the frames between two anchors (a mini-GOP) are cut short
/// when the lookahead's cost of predicting the would-be anchor straight from
/// the last one exceeds this multiple of what the frames since then cost,
/// each predicted from the one before it: a far anchor that is harder to
/// reach than walking there pays for its B pictures twice. Measured on ten
/// clips at depth 2, BD-rate by VMAF / PSNR against P pictures only: never
/// shortening −7.9 / −13.5%, 0.9 here −8.8 / −13.3%, 1.1 −8.7 / −13.9%.
/// (0.7 without [`FAR_ANCHOR_FLOOR`] took every B picture out of a pan and
/// of a noisy clip: −7.4 / −10.8%. The lookahead's costs are differences of
/// samples, noise included, which the encoder does not code; only a clear
/// excess means anything.)
const FAR_ANCHOR_CHAIN: f32 = 1.1;

/// ... and only when that cost is at least this share of the frame's intra
/// cost. Below it both plans are nearly free and their ratio is noise.
const FAR_ANCHOR_FLOOR: f32 = 0.1;

/// A mini-GOP is also cut short when the last anchor predicts the would-be
/// anchor hardly better than intra prediction (this share of its intra cost)
/// while the frame before it does at least twice as well: such an anchor
/// costs nearly a keyframe. Depth 2 / 3: a further −0.4 / −1.9% by VMAF,
/// nearly all of it on one clip of fast camera motion.
const FAR_ANCHOR_NEAR_INTRA: f32 = 0.75;

/// A frame waiting in the lookahead.
struct Pending {
    frame: Yuv,
    pts: i64,
    force_keyframe: bool,
    /// Analysis picture and its change score against the frame before it,
    /// without weighted prediction; `None` when scene-cut detection is off.
    /// Scene cuts are judged on these plain scores, a frame's and its
    /// neighbors' alike.
    analysis: Option<(Arc<lookahead::Lowres>, f32)>,
    /// The change score as the encoder will experience the frame: with
    /// weighted prediction where that was adopted.
    coded_score: f32,
    /// Weighted prediction shows this frame to be the same scene as the one
    /// before it, whatever its plain score says: a brightness change, not a
    /// cut.
    same_scene: bool,
    /// Per-block breakdown of that score ([`lookahead::Analysis`]): blocks
    /// and blocks per row.
    blocks: (Vec<u8>, usize),
    /// Costs and motion per block ([`lookahead::BlockInfo`]), and blocks per
    /// row; a frame with no predecessor has intra costs only.
    info: (Vec<lookahead::BlockInfo>, usize),
    /// Weighted prediction from the previous frame, when the analysis found
    /// that a brightness change separates the two.
    weights: Option<PredWeights>,
}

/// The stream format, fixed by the first frame.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Format {
    chroma: ChromaFormat,
    bit_depth: BitDepth,
}

/// HEVC video encoder. Feed frames in display order with
/// [`send_frame`](Self::send_frame) and drain coded pictures with
/// [`receive_packet`](Self::receive_packet).
///
/// See the [module documentation](self) for what the encoder does not do yet.
pub struct VideoEncoder {
    cfg: VideoConfig,
    width: u32,
    height: u32,
    seq: SeqInfo,
    /// Pictures retained in the decoded picture buffer in low-delay mode.
    refs: u8,
    /// Frames from one anchor (keyframe or P picture) to the next: 1 without
    /// B pictures.
    mini_len: usize,
    placer: gop::KeyframePlacer,
    pool: crate::pool::ThreadPool,
    threads: usize,
    stream: Option<VideoStream>,
    format: Option<Format>,
    /// Frames accepted but not yet coded, in display order.
    pending: VecDeque<Pending>,
    /// Analysis picture and change score of the last coded frame.
    last_coded: Option<(Arc<lookahead::Lowres>, f32)>,
    /// Frames past the last anchor whose place in the GOP is decided, in
    /// display order: coded once the next anchor is known.
    mini: Vec<Placed>,
    /// Frames left of a flash in progress, the one that returns to the
    /// scene included: none of them is a scene cut.
    flash_frames: usize,
    /// VPS, SPS and PPS in Annex B format.
    headers: Vec<u8>,
    /// POC of the next picture (in display order) within its GOP.
    poc: u32,
    /// Reconstructed reference pictures of the current GOP.
    dpb: Vec<DpbPic>,
    spare: Vec<ReconFrame>,
    packets: VecDeque<Packet>,
    /// Timestamps of the frames placed so far that no packet's `dts` has
    /// used yet, in display order.
    display_pts: VecDeque<i64>,
    /// Timestamp of the last accepted frame when packet order differs.
    last_pts: Option<i64>,
    /// Packets produced so far.
    coded: u64,
    /// `dts` of the first packet, and the timestamp step it was derived
    /// from.
    first_dts: Option<(i128, i128)>,
}

/// A frame with its position in the GOP decided.
struct Placed {
    frame: Pending,
    poc: u32,
    /// [`VideoEncoder::persistence_deltas`] and their blocks per row.
    persistence: Option<(Vec<i8>, usize)>,
    /// How the last anchor predicts this frame, when frames lie between the
    /// two: what it is coded with if it becomes the next anchor.
    from_anchor: Option<Prediction>,
}

/// A reference picture held for later pictures of the GOP.
struct DpbPic {
    poc: u32,
    recon: ReconFrame,
    /// Its analysis picture, when the lookahead makes them.
    lowres: Option<Arc<lookahead::Lowres>>,
}

/// What the lookahead finds out about predicting one picture from another.
struct Prediction {
    /// Change score without weighted prediction.
    plain_score: f32,
    /// Change score as the encoder will experience it: with weighted
    /// prediction where that is adopted.
    coded_score: f32,
    /// See [`Pending::same_scene`].
    same_scene: bool,
    blocks: Vec<u8>,
    info: Vec<lookahead::BlockInfo>,
    stride: usize,
    weights: Option<PredWeights>,
}

/// Analyse predicting `cur` from `prev`, with weighted prediction if a
/// brightness change separates the two.
fn predict_from(
    prev: &lookahead::Lowres,
    cur: &lookahead::Lowres,
    pool: &crate::pool::ThreadPool,
) -> Prediction {
    let analysis = lookahead::analyse(prev, cur, Some(pool));
    // A fade or a flash: if giving the reference this picture's brightness
    // makes it clearly more predictable, the encoder will predict that way,
    // and everything the lookahead says about the frame assumes it.
    if let Some((delta_weight, luma_offset)) = lookahead::estimate_weights(prev, cur) {
        let adjusted = prev.weighted(64 + i32::from(delta_weight), i32::from(luma_offset));
        let with_weights = lookahead::analyse(&adjusted, cur, Some(pool));
        if with_weights.score < WEIGHTED_GAIN * analysis.score {
            return Prediction {
                plain_score: analysis.score,
                coded_score: with_weights.score,
                // A cut into a low-contrast shot also "improves" under
                // weights (the reference is flattened towards the new
                // picture's mean) without becoming predictable. Only a
                // weighted score that shows the same scene may overrule the
                // plain one.
                same_scene: with_weights.score < lookahead::SAME_SCENE,
                blocks: with_weights.blocks,
                info: with_weights.info,
                stride: with_weights.stride,
                weights: Some(PredWeights {
                    delta_weight,
                    luma_offset,
                }),
            };
        }
    }
    Prediction {
        plain_score: analysis.score,
        coded_score: analysis.score,
        same_scene: false,
        blocks: analysis.blocks,
        info: analysis.info,
        stride: analysis.stride,
        weights: None,
    }
}

impl VideoEncoder {
    /// Create an encoder for `width`×`height` frames.
    pub fn new(width: u32, height: u32, cfg: VideoConfig) -> Result<Self, EncodeError> {
        crate::validate_dims(width, height)?;
        cfg.base.validate()?;
        if cfg.base.lossless {
            return Err(EncodeError::Unsupported("lossless video"));
        }
        if cfg.base.screen_content {
            return Err(EncodeError::Unsupported("screen content tools in video"));
        }
        let (fps_num, fps_den) = cfg.frame_rate;
        if fps_num == 0 || fps_den == 0 {
            return Err(EncodeError::InvalidInput);
        }
        let (coded_w, coded_h) = (
            crate::hevc::coded_dim(width),
            crate::hevc::coded_dim(height),
        );
        let level_idc = gop::video_level_idc(coded_w, coded_h, fps_num, fps_den).ok_or(
            EncodeError::Unsupported("coded dimensions or frame rate exceed HEVC level 6.2"),
        )?;
        let max_dpb = gop::max_dpb_size(level_idc, u64::from(coded_w) * u64::from(coded_h));
        // What the structure needs from the SPS in its steady state, and how
        // much of it the DPB allows.
        let intra_only = cfg.gop.max_keyint <= 1;
        let (structure, stats) = if intra_only {
            gop::plan_gop(1, cfg.gop.structure, max_dpb)
                .map(|(_, stats)| (GopStructure::LowDelayP { refs: 1 }, stats))
        } else {
            gop::steady_state(cfg.gop.structure, max_dpb)
        }
        .map_err(|_| EncodeError::InvalidInput)?;
        let (refs, mini_len) = match structure {
            GopStructure::LowDelayP { refs } => (
                refs.clamp(1, stats.max_dec_pic_buffering.saturating_sub(1).max(1)),
                1,
            ),
            GopStructure::HierarchicalB { depth } => (1, 1usize << depth),
        };
        let available = crate::resolve_threads(0);
        let threads = if cfg.base.threads == 0 {
            available
        } else {
            cfg.base.threads.min(available)
        };
        let pool = crate::pool::ThreadPool::new(threads - 1);
        let threads = pool.parallelism();
        Ok(VideoEncoder {
            width,
            height,
            seq: SeqInfo {
                video: true,
                level_idc: Some(level_idc),
                max_dec_pic_buffering: stats.max_dec_pic_buffering,
                max_num_reorder: stats.max_num_reorder,
                timing: Some((fps_den, fps_num)),
            },
            refs,
            mini_len,
            placer: gop::KeyframePlacer::new(cfg.gop.min_keyint, cfg.gop.max_keyint),
            pool,
            threads,
            stream: None,
            format: None,
            pending: VecDeque::new(),
            last_coded: None,
            mini: Vec::new(),
            flash_frames: 0,
            headers: Vec::new(),
            poc: 0,
            dpb: Vec::new(),
            spare: Vec::new(),
            packets: VecDeque::new(),
            display_pts: VecDeque::new(),
            last_pts: None,
            coded: 0,
            first_dts: None,
            cfg,
        })
    }

    /// Submit the next frame in display order. `force_keyframe` starts a new
    /// GOP at this frame regardless of [`GopConfig::min_keyint`].
    ///
    /// The frame must match the encoder's dimensions, and every frame must
    /// have the chroma format and bit depth of the first. With B pictures,
    /// timestamps must strictly increase and the first must leave at least
    /// the reorder depth in timestamp ticks above `i64::MIN` for initial
    /// decoding timestamps. Its packet becomes
    /// available once [`VideoConfig::lookahead`] further frames have been
    /// submitted — with B pictures, once the anchor after it has been coded,
    /// up to `(1 << depth) - 1` frames more — or after
    /// [`flush`](Self::flush).
    pub fn send_frame(
        &mut self,
        frame: &Yuv,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<(), EncodeError> {
        frame.validate()?;
        if frame.width != self.width || frame.height != self.height {
            return Err(EncodeError::InvalidDimensions {
                width: frame.width,
                height: frame.height,
            });
        }
        let reorder = i64::from(self.seq.max_num_reorder);
        if reorder > 0
            && self
                .last_pts
                .map_or(pts < i64::MIN + reorder, |last| pts <= last)
        {
            return Err(EncodeError::InvalidInput);
        }
        let format = Format {
            chroma: frame.chroma,
            bit_depth: frame.bit_depth,
        };
        if *self.format.get_or_insert(format) != format {
            return Err(EncodeError::InvalidInput);
        }
        if reorder > 0 {
            self.last_pts = Some(pts);
        }
        let mut blocks = (Vec::new(), 0);
        let mut info = (Vec::new(), 0);
        let mut weights = None;
        let (mut coded_score, mut same_scene) = (0.0, false);
        let analysis = (self.cfg.gop.scene_cut == SceneCut::Adaptive).then(|| {
            let lowres = lookahead::Lowres::new(frame);
            let previous = match self.pending.back() {
                Some(pending) => pending.analysis.as_ref(),
                None => self.last_coded.as_ref(),
            };
            let score = previous.map_or(0.0, |(prev, _)| {
                let prediction = predict_from(prev, &lowres, &self.pool);
                coded_score = prediction.coded_score;
                same_scene = prediction.same_scene;
                blocks = (prediction.blocks, prediction.stride);
                info = (prediction.info, prediction.stride);
                weights = prediction.weights;
                prediction.plain_score
            });
            if previous.is_none() {
                info = (
                    lookahead::intra_info(&lowres),
                    lookahead::block_stride(&lowres),
                );
            }
            (Arc::new(lowres), score)
        });
        self.pending.push_back(Pending {
            // The caller keeps its frame; the lookahead needs its own copy.
            frame: Yuv {
                y: frame.y.clone(),
                cb: frame.cb.clone(),
                cr: frame.cr.clone(),
                ..*frame
            },
            pts,
            force_keyframe,
            analysis,
            coded_score,
            same_scene,
            blocks,
            info,
            weights,
        });
        while self.pending.len() > self.cfg.lookahead {
            self.encode_next()?;
        }
        Ok(())
    }

    /// Whether `next`, the frame about to be coded, opens a new scene; `None`
    /// when there is nothing to compare it with.
    fn is_scene_cut(&mut self, next: &Pending) -> Option<bool> {
        let (_, score) = next.analysis.as_ref()?;
        let (before_cut, before) = self.last_coded.as_ref()?;
        if next.same_scene {
            return Some(false);
        }
        let gop = &self.cfg.gop;
        let threshold = lookahead::cut_threshold(
            self.placer.frames_since_key() + 1,
            gop.min_keyint,
            gop.max_keyint,
        );
        fn analysis(pending: &Pending) -> Option<&(Arc<lookahead::Lowres>, f32)> {
            pending.analysis.as_ref()
        }
        let after = self.pending.front().and_then(analysis).map_or(0.0, |a| a.1);
        if !lookahead::is_cut(*score, *before, after, threshold) {
            return Some(false);
        }
        // A flash: the picture is back to what it was before the "cut" while
        // still in sight. No keyframe, neither here nor in the `frames_away`
        // frames up to and including the one that returns (which looks like
        // a cut too): they are a detour from the scene this GOP already holds.
        let returns = self
            .pending
            .iter()
            .filter_map(analysis)
            .position(|(later, _)| {
                lookahead::change_score(before_cut, later) < lookahead::SAME_SCENE
            });
        if let Some(frames_away) = returns {
            self.flash_frames = frames_away + 1;
        }
        Some(returns.is_none())
    }

    /// QP deltas for `next`, the frame about to be coded, per 16×16 block,
    /// from how much the frames after it draw on each block
    /// ([`lookahead::propagation`]): what later pictures keep copying is
    /// worth coding well once, what changes again at once is not. Keyframes
    /// get it too, which is where most of the gain is.
    fn persistence_deltas(&self, next: &Pending) -> Option<(Vec<i8>, usize)> {
        let (current, stride) = (&next.info.0, next.info.1);
        if current.is_empty() {
            return None;
        }
        let later: Vec<&[lookahead::BlockInfo]> = self
            .pending
            .iter()
            .take((self.cfg.gop.max_keyint.max(1) - 1 - self.placer.frames_since_key()) as usize)
            .take_while(|frame| !frame.force_keyframe)
            .map(|frame| &frame.info.0[..])
            .take_while(|info| info.len() == current.len())
            .collect();
        if later.is_empty() {
            return None;
        }
        let deltas = lookahead::propagation(current, &later, stride)
            .iter()
            .map(|&worth| -(PERSISTENCE_STRENGTH * worth).round().min(12.0) as i8)
            .collect();
        Some((deltas, stride))
    }

    /// Decide where the oldest pending frame goes in its GOP, and code what
    /// that settles: a keyframe at once (after the frames waiting in front
    /// of it), any other frame once the anchor after it is known.
    fn encode_next(&mut self) -> Result<(), EncodeError> {
        let Some(next) = self.pending.pop_front() else {
            return Ok(());
        };
        // Nothing inside a flash is a scene cut, the frame that returns to
        // the scene included.
        let in_flash = self.flash_frames > 0;
        self.flash_frames = self.flash_frames.saturating_sub(1);
        let scene_cut = !in_flash && self.is_scene_cut(&next).unwrap_or(false);
        let key = self.placer.next(scene_cut, next.force_keyframe);
        let persistence = self.persistence_deltas(&next);
        self.last_coded.clone_from(&next.analysis);
        if self.seq.max_num_reorder > 0 {
            self.display_pts.push_back(next.pts);
        }
        if key {
            // The GOP before ends here: its last frames become a shorter
            // mini-GOP.
            self.code_mini_gop()?;
            self.poc = 0;
        }
        let mut placed = Placed {
            frame: next,
            poc: self.poc,
            persistence,
            from_anchor: None,
        };
        self.poc += 1;
        if key {
            return self.code_key(&placed);
        }
        placed.from_anchor = self.predict_from_anchor(&placed);
        if self.far_anchor_is_worse(&placed) {
            // The frame opens the next mini-GOP, right after its new anchor.
            self.code_mini_gop()?;
            placed.from_anchor = None;
        } else if let Some(prior) = self.mini.last_mut() {
            // Only the last waiting frame can become the next anchor.
            prior.from_anchor = None;
        }
        self.mini.push(placed);
        if self.mini.len() >= self.mini_len {
            self.code_mini_gop()?;
        }
        Ok(())
    }

    /// The lookahead's view of predicting `placed` from the last anchor,
    /// when frames wait between the two.
    fn predict_from_anchor(&self, placed: &Placed) -> Option<Prediction> {
        let anchor = self.mini.first()?.poc - 1;
        let past = self.dpb.iter().find(|pic| pic.poc == anchor)?;
        let (current, _) = placed.frame.analysis.as_ref()?;
        Some(predict_from(past.lowres.as_ref()?, current, &self.pool))
    }

    /// Whether making `placed` the next anchor, with the frames waiting in
    /// `mini` as B pictures in front of it, is a worse plan than ending the
    /// mini-GOP at the frame before it; see [`FAR_ANCHOR_CHAIN`] and
    /// [`FAR_ANCHOR_NEAR_INTRA`].
    fn far_anchor_is_worse(&self, placed: &Placed) -> bool {
        let Some(from_anchor) = &placed.from_anchor else {
            return false;
        };
        let cost = |info: &[lookahead::BlockInfo]| info.iter().map(|b| b.inter).sum::<f32>();
        let direct = cost(&from_anchor.info);
        let own = cost(&placed.frame.info.0);
        let chain = own + self.mini.iter().map(|f| cost(&f.frame.info.0)).sum::<f32>();
        let intra: f32 = placed.frame.info.0.iter().map(|b| b.intra).sum();
        (direct > FAR_ANCHOR_CHAIN * chain && direct > FAR_ANCHOR_FLOOR * intra)
            || (direct > FAR_ANCHOR_NEAR_INTRA * intra && 2.0 * own < direct)
    }

    /// Code the frames waiting past the last anchor: the last of them as
    /// the next anchor, the others as B pictures between the two.
    fn code_mini_gop(&mut self) -> Result<(), EncodeError> {
        let frames = std::mem::take(&mut self.mini);
        let Some(first) = frames.first() else {
            return Ok(());
        };
        let anchor = first.poc - 1;
        let plans = if self.mini_len > 1 {
            gop::mini_gop(anchor, frames.len() as u32)
        } else {
            vec![gop::low_delay_frame(first.poc, self.refs)]
        };
        let frame_for = |plan: &gop::FramePlan| &frames[(plan.poc - anchor - 1) as usize];
        let Some(stream) = &self.stream else {
            unreachable!("a GOP starts with a keyframe, which sets the stream up");
        };
        let pool = &self.pool;
        let shortcuts = match self.cfg.preset {
            VideoPreset::Quality => Shortcuts::NONE,
            VideoPreset::Fast => Shortcuts::FAST,
        };
        // Every picture the mini-GOP predicts from: the DPB as it stands and
        // the reference pictures coded along the way. What the reference
        // picture sets drop is only let go of at the end, so that pictures
        // which do not predict from one another can be coded at the same
        // time. Their packets still leave in decoding order, and each is
        // what coding them one by one gives.
        let mut held: Vec<DpbPic> = std::mem::take(&mut self.dpb);
        let initial = held.len();
        let mut coded: Vec<Option<(Vec<u8>, FrameKind)>> = plans.iter().map(|_| None).collect();
        let mut left: Vec<usize> = (0..plans.len()).collect();
        let mut failure = None;
        while !left.is_empty() && failure.is_none() {
            let have = |poc: u32| held.iter().find(|pic| pic.poc == poc);
            let references = |plan: &gop::FramePlan| {
                let l0 = have(plan.l0[0])?;
                match plan.l1.first() {
                    Some(&poc) => Some((l0, Some(have(poc)?))),
                    None => Some((l0, None)),
                }
            };
            let ready: Vec<usize> = left
                .iter()
                .copied()
                .filter(|&index| references(&plans[index]).is_some())
                .take(pool.parallelism())
                .collect();
            assert!(
                !ready.is_empty(),
                "the validated plan only references pictures in the DPB"
            );
            let mut results: Vec<_> = ready.iter().map(|_| None).collect();
            let spare = &mut self.spare;
            let jobs: Vec<_> = ready
                .iter()
                .zip(results.iter_mut())
                .map(|(&index, slot)| (index, slot, spare.pop().unwrap_or_default()))
                .collect();
            let encode = |index: usize, recon: ReconFrame| {
                let plan = &plans[index];
                let placed = frame_for(plan);
                let (l0, l1) = references(plan).expect("checked above");
                Self::encode_inter(stream, shortcuts, pool, placed, plan, l0, l1, recon)
            };
            if jobs.len() == 1 {
                for (index, slot, recon) in jobs {
                    *slot = Some(encode(index, recon));
                }
            } else {
                let encode = &encode;
                pool.scoped(|scope| {
                    for (index, slot, recon) in jobs {
                        scope.spawn(move || *slot = Some(encode(index, recon)));
                    }
                });
            }
            for (&index, result) in ready.iter().zip(results) {
                match result.expect("every job ran") {
                    Ok((mut data, recon, kind)) => {
                        let plan = &plans[index];
                        let pending = &frame_for(plan).frame;
                        if self.cfg.picture_hash {
                            write_annex_b(
                                &mut data,
                                &picture_hash_sei(&recon, pending.frame.bit_depth),
                            );
                        }
                        if plan.nal_type == gop::NalType::TrailN {
                            self.spare.push(recon);
                        } else {
                            held.push(DpbPic {
                                poc: plan.poc,
                                recon,
                                lowres: pending.analysis.as_ref().map(|(lowres, _)| lowres.clone()),
                            });
                        }
                        coded[index] = Some((data, kind));
                    }
                    Err(error) => failure = Some(error),
                }
            }
            left.retain(|index| coded[*index].is_none());
        }
        // The DPB as it was, and the reference pictures coded since.
        let mut fresh = held.split_off(initial);
        self.dpb = held;
        if let Some(error) = failure {
            return Err(error);
        }
        // In decoding order: each picture's reference picture set, whatever
        // it leaves out being gone for good, then its packet.
        for (plan, coded) in plans.iter().zip(coded) {
            let (data, kind) = coded.expect("every picture was coded");
            self.dpb.retain_mut(|pic| {
                let keep = plan.keep.contains(&pic.poc);
                if !keep {
                    self.spare.push(std::mem::take(&mut pic.recon));
                }
                keep
            });
            if let Some(at) = fresh.iter().position(|pic| pic.poc == plan.poc) {
                self.dpb.push(fresh.swap_remove(at));
            }
            self.queue_packet(data, frame_for(plan).frame.pts, kind);
        }
        Ok(())
    }

    /// Fix the stream's settings and parameter sets from its first frame.
    fn start_stream(&mut self, frame: &Yuv) {
        if self.stream.is_some() {
            return;
        }
        let base = &self.cfg.base;
        let stream = VideoStream::new(
            frame,
            base.quality,
            base.sao,
            base.speed,
            base.parallelism != crate::ParallelismStrategy::Single,
            self.threads,
            base.persistent_rice,
            self.seq,
            self.mini_len > 1,
        );
        for nalu in stream.parameter_sets(frame, base.color.cicp.as_ref()) {
            write_annex_b(&mut self.headers, &nalu);
        }
        self.stream = Some(stream);
    }

    /// Code `placed` as the IDR picture of a new GOP.
    fn code_key(&mut self, placed: &Placed) -> Result<(), EncodeError> {
        let frame = &placed.frame.frame;
        self.start_stream(frame);
        let Some(stream) = &self.stream else {
            unreachable!("the stream was set up above");
        };
        self.spare.extend(self.dpb.drain(..).map(|pic| pic.recon));
        let plan = gop::idr();
        let qp_hint = placed.persistence.as_ref().map(|(deltas, stride)| QpHint {
            deltas,
            stride: *stride,
        });
        let mut recon = self.spare.pop().unwrap_or_default();
        let mut data = self.headers.clone();
        let slice = stream.encode_picture(
            frame,
            None,
            &self.pool,
            &mut recon,
            plan.qp_offset,
            true,
            qp_hint,
        )?;
        write_annex_b(&mut data, &slice);
        if self.cfg.picture_hash {
            write_annex_b(&mut data, &picture_hash_sei(&recon, frame.bit_depth));
        }
        self.dpb.push(DpbPic {
            poc: plan.poc,
            recon,
            lowres: placed
                .frame
                .analysis
                .as_ref()
                .map(|(lowres, _)| lowres.clone()),
        });
        self.queue_packet(data, placed.frame.pts, FrameKind::Key);
        Ok(())
    }

    /// Code `placed` as the inter picture `plan` describes, predicted from
    /// `l0` and, for a B picture, `l1`. Returns its slice and what kind of
    /// picture it is; `recon` receives its reconstruction.
    #[allow(clippy::too_many_arguments)]
    fn encode_inter(
        stream: &VideoStream,
        shortcuts: Shortcuts,
        pool: &crate::pool::ThreadPool,
        placed: &Placed,
        plan: &gop::FramePlan,
        l0: &DpbPic,
        l1: Option<&DpbPic>,
        mut recon: ReconFrame,
    ) -> Result<(Vec<u8>, ReconFrame, FrameKind), EncodeError> {
        let frame = &placed.frame.frame;
        let before: Vec<(u32, bool)> = plan
            .keep
            .iter()
            .rev()
            .filter(|&&poc| poc < plan.poc)
            .map(|&poc| (poc, poc == l0.poc))
            .collect();
        let after: Vec<(u32, bool)> = plan
            .keep
            .iter()
            .filter(|&&poc| poc > plan.poc)
            .map(|&poc| (poc, l1.is_some_and(|pic| pic.poc == poc)))
            .collect();
        let distance = |pic: &DpbPic| i64::from(plan.poc) - i64::from(pic.poc);

        // What the lookahead knows about this picture is relative to the
        // frame before it in display order. That is the reference of a P
        // picture right after its anchor; any other picture is analysed
        // against what it is really predicted from, which for a P picture
        // was done when it was placed.
        let pending = &placed.frame;
        let current = pending.analysis.as_ref().map(|(lowres, _)| lowres);
        let adjacent = distance(l0) == 1;
        let (computed, past_analysis, future_analysis);
        let from_l0: Option<&Prediction> = match (&placed.from_anchor, current, &l0.lowres) {
            _ if adjacent => None,
            (Some(known), ..) if l1.is_none() => Some(known),
            (_, Some(current), Some(past)) if l1.is_none() => {
                computed = predict_from(past, current, pool);
                Some(&computed)
            }
            _ => None,
        };
        // Either picture may predict a block of a B picture.
        let both = match (current, &l0.lowres, l1.and_then(|pic| pic.lowres.as_ref())) {
            (Some(current), Some(past), Some(future)) => {
                // The adjacent past picture was already analysed when this
                // frame arrived. A weighted analysis only applies to P.
                let (blocks, stride, score, info) = if adjacent && pending.weights.is_none() {
                    (
                        &pending.blocks.0,
                        pending.blocks.1,
                        pending.coded_score,
                        &pending.info.0,
                    )
                } else {
                    past_analysis = lookahead::analyse(past, current, Some(pool));
                    (
                        &past_analysis.blocks,
                        past_analysis.stride,
                        past_analysis.score,
                        &past_analysis.info,
                    )
                };
                future_analysis = lookahead::analyse(future, current, Some(pool));
                let either: Vec<u8> = blocks
                    .iter()
                    .zip(&future_analysis.blocks)
                    .map(|(&p, &q)| p.min(q))
                    .collect();
                Some((
                    either,
                    stride,
                    score.min(future_analysis.score),
                    [info.as_slice(), future_analysis.info.as_slice()],
                ))
            }
            _ => None,
        };
        // Per block how badly the references predict it, its motion into
        // each list's picture, the picture's change score and its weights.
        let (blocks, stride, score, weights, info): (&[u8], _, _, _, [&[lookahead::BlockInfo]; 2]) =
            match (&both, from_l0) {
                (Some((either, stride, score, info)), _) => (either, *stride, *score, None, *info),
                (None, Some(p)) => (
                    &p.blocks,
                    p.stride,
                    p.coded_score,
                    p.weights,
                    [&p.info, &[]],
                ),
                (None, None) if adjacent && l1.is_none() => (
                    &pending.blocks.0,
                    pending.blocks.1,
                    pending.coded_score,
                    pending.weights,
                    [&pending.info.0, &[]],
                ),
                (None, None) => (&[], 0, 0.0, None, [&[], &[]]),
            };
        let intra_hint = (!blocks.is_empty()).then_some(IntraHint { blocks, stride });
        // The analysis measures motion in quarter samples at half
        // resolution, which is half samples of the picture.
        let vectors = info.map(|info| -> Vec<(i16, i16)> {
            info.iter()
                .map(|block| (block.mv.0.saturating_mul(2), block.mv.1.saturating_mul(2)))
                .collect()
        });
        let motion_hint = [0, 1].map(|list| {
            (!vectors[list].is_empty()).then_some(MotionHint {
                vectors: &vectors[list],
                stride,
            })
        });
        // A picture that is almost a copy of its reference gains nothing
        // from SAO, so it is not even analysed. Any other picture decides by
        // what SAO turns out to save and cost.
        let sao = current.is_none() || score >= SAO_MIN_CHANGE;
        // B pictures take the lookahead's QP deltas as P pictures do, though
        // little or nothing is predicted from them: next to anchors coded
        // that much finer they would otherwise stand out as the coarse
        // frames (−2.6% bitrate by VMAF, −1.3% by PSNR against none; 0.7 of
        // the deltas was no better than all of them).
        let qp_hint = placed.persistence.as_ref().map(|(deltas, stride)| QpHint {
            deltas,
            stride: *stride,
        });
        let pic = InterPic {
            nal_type: plan.nal_type as u8,
            poc: plan.poc,
            before: &before,
            after: &after,
            pics: RefPics {
                l0: &l0.recon,
                l1: l1.map(|pic| &pic.recon),
                distance: [
                    distance(l0) as i32,
                    l1.map_or(distance(l0), distance) as i32,
                ],
            },
            intra_hint,
            shortcuts,
            weights,
            motion_hint,
        };
        let slice = stream.encode_picture(
            frame,
            Some(&pic),
            pool,
            &mut recon,
            plan.qp_offset,
            sao,
            qp_hint,
        )?;
        let kind = if l1.is_some() {
            FrameKind::B
        } else {
            FrameKind::P
        };
        let mut data = Vec::new();
        write_annex_b(&mut data, &slice);
        Ok((data, recon, kind))
    }

    fn queue_packet(&mut self, data: Vec<u8>, pts: i64, kind: FrameKind) {
        // Decoding timestamps are the presentation timestamps delayed by
        // the reorder depth, extended below the first one.
        let reorder = u64::from(self.seq.max_num_reorder);
        let dts = if reorder == 0 {
            pts
        } else {
            let (first, step) = *self.first_dts.get_or_insert_with(|| {
                let pts = i128::from(pts);
                let reorder = i128::from(reorder);
                let step = self
                    .pending
                    .front()
                    .map_or(1, |next| i128::from(next.pts) - pts);
                let step = step.min((pts - i128::from(i64::MIN)) / reorder);
                (pts - reorder * step, step)
            });
            if self.coded < reorder {
                (first + i128::from(self.coded) * step) as i64
            } else {
                self.display_pts
                    .pop_front()
                    .expect("coded picture has a timestamp")
            }
        };
        self.coded += 1;
        debug_assert!(dts <= pts);
        self.packets.push_back(Packet {
            data,
            pts,
            dts,
            kind,
        });
    }

    /// The next coded picture, if one is ready.
    pub fn receive_packet(&mut self) -> Option<Packet> {
        self.packets.pop_front()
    }

    /// Signal the end of the stream: code every frame still held in the
    /// lookahead. Drain [`receive_packet`](Self::receive_packet) afterwards.
    pub fn flush(&mut self) -> Result<(), EncodeError> {
        while !self.pending.is_empty() {
            self.encode_next()?;
        }
        self.code_mini_gop()
    }

    /// VPS, SPS and PPS in Annex B format, as also written in front of every
    /// keyframe. Empty until the first frame has been coded, because the
    /// first frame fixes the stream settings.
    pub fn parameter_sets(&self) -> &[u8] {
        &self.headers
    }
}

/// Append a NAL unit in Annex B byte-stream format: a four-byte start code,
/// then the unit with emulation prevention bytes inserted (§7.4.2).
fn write_annex_b(out: &mut Vec<u8>, nalu: &Nalu) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    let mut zeros = 0;
    for &byte in &nalu.data {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
}

/// Decoded picture hash SEI (§D.2.19) carrying the MD5 of each plane of
/// `recon`, as a suffix SEI NAL unit.
fn picture_hash_sei(recon: &ReconFrame, bit_depth: BitDepth) -> Nalu {
    const SUFFIX_SEI: u8 = 40;
    const DECODED_PICTURE_HASH: u8 = 132;
    let planes: &[&[u16]] = if recon.cb.is_empty() {
        &[&recon.y]
    } else {
        &[&recon.y, &recon.cb, &recon.cr]
    };
    // nal_unit_header: type, nuh_layer_id 0, nuh_temporal_id_plus1 1.
    let mut data = vec![SUFFIX_SEI << 1, 1, DECODED_PICTURE_HASH];
    data.push(1 + 16 * planes.len() as u8); // payload_size
    data.push(0); // hash_type: MD5
    let mut bytes = [0; 4096];
    for plane in planes {
        // Samples are hashed as one byte each up to 8 bits, else two bytes
        // little-endian, row by row over the whole coded picture.
        let mut md5 = md5::Md5::new();
        if bit_depth.bits() <= 8 {
            for samples in plane.chunks(bytes.len()) {
                let dst = &mut bytes[..samples.len()];
                for (byte, &sample) in dst.iter_mut().zip(samples) {
                    *byte = sample as u8;
                }
                md5.update(dst);
            }
        } else {
            for samples in plane.chunks(bytes.len() / 2) {
                let dst = &mut bytes[..samples.len() * 2];
                for (pair, &sample) in dst.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
                    *pair = sample.to_le_bytes();
                }
                md5.update(dst);
            }
        }
        data.extend_from_slice(&md5.finish());
    }
    data.push(0x80); // rbsp_trailing_bits
    Nalu {
        _nal_type: SUFFIX_SEI,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 160;
    const H: usize = 128;

    fn hash(a: u32, b: u32, c: u32) -> u32 {
        let h = a
            .wrapping_mul(374_761_393)
            .wrapping_add(b.wrapping_mul(668_265_263))
            .wrapping_add(c.wrapping_mul(2_246_822_519));
        let h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        h ^ (h >> 16)
    }

    /// A monochrome frame of scene `scene`, with a little noise that differs
    /// per `frame`. The scenes differ in structure, not just in content, as
    /// real shots do: 1 is random tiles, 2 a smooth wave, 3 vertical stripes
    /// and anything else horizontal stripes.
    fn frame(scene: u32, frame: u32) -> Yuv {
        let y = (0..H)
            .flat_map(|r| (0..W).map(move |c| (r, c)))
            .map(|(r, c)| {
                let base = match scene {
                    1 => 40 + hash(scene, (c / 8) as u32, (r / 8) as u32) % 176,
                    2 => (128.0 + 90.0 * (0.05 * c as f32 + 0.03 * r as f32).sin()) as u32,
                    3 => 40 + hash(scene, (c / 6) as u32, 0) % 176,
                    _ => 40 + hash(scene, 0, (r / 6) as u32) % 176,
                };
                let noise = hash(frame + 1000 * scene, c as u32, r as u32) % 3;
                (base + noise) as u16
            })
            .collect();
        Yuv::from_planes(
            y,
            Vec::new(),
            Vec::new(),
            W as u32,
            H as u32,
            ChromaFormat::Monochrome,
            BitDepth::Eight,
        )
        .unwrap()
    }

    /// Encode one frame per entry of `scenes` and return where keyframes fell.
    fn keyframes(scenes: &[u32], lookahead: usize, gop: GopConfig) -> Vec<i64> {
        let cfg = VideoConfig {
            base: EncodeConfig::default().with_quality(30).with_threads(1),
            gop,
            lookahead,
            ..VideoConfig::default()
        };
        let mut enc = VideoEncoder::new(W as u32, H as u32, cfg).unwrap();
        let mut keys = Vec::new();
        let mut packets = 0;
        let mut drain = |enc: &mut VideoEncoder, sent: usize| {
            while let Some(packet) = enc.receive_packet() {
                assert_eq!(packet.pts, packets as i64, "packets come in display order");
                packets += 1;
                if packet.kind == FrameKind::Key {
                    keys.push(packet.pts);
                }
            }
            // The encoder holds back exactly `lookahead` frames.
            assert_eq!(packets, sent.saturating_sub(lookahead));
        };
        for (n, &scene) in scenes.iter().enumerate() {
            enc.send_frame(&frame(scene, n as u32), n as i64, false)
                .unwrap();
            drain(&mut enc, n + 1);
        }
        enc.flush().unwrap();
        drain(&mut enc, scenes.len() + lookahead);
        assert_eq!(packets, scenes.len());
        keys
    }

    fn gop(min_keyint: u32, max_keyint: u32) -> GopConfig {
        GopConfig {
            min_keyint,
            max_keyint,
            ..GopConfig::default()
        }
    }

    #[test]
    fn keyframes_follow_scene_cuts() {
        //            0  1  2  3  4  5  6  7  8  9 10 11
        let scenes = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3];
        for lookahead in [0, 1, 4] {
            assert_eq!(
                keyframes(&scenes, lookahead, gop(2, 250)),
                [0, 5, 9],
                "lookahead {lookahead}"
            );
        }
        // Detection off: only max_keyint.
        let off = GopConfig {
            scene_cut: SceneCut::Off,
            ..gop(2, 8)
        };
        assert_eq!(keyframes(&scenes, 4, off), [0, 8]);
        // A cut closer than min_keyint to the last keyframe is not taken.
        assert_eq!(keyframes(&scenes, 4, gop(6, 250)), [0, 9]);
    }

    #[test]
    fn a_flash_is_not_a_scene_cut() {
        //            0  1  2  3  4  5  6  7  8  9 10 11
        let scenes = [1, 1, 1, 1, 9, 9, 1, 1, 1, 2, 2, 2];
        // With frames to look ahead, neither the flash nor the return from
        // it starts a GOP — also when min_keyint would allow one — and the
        // real cut after it still does.
        assert_eq!(keyframes(&scenes, 4, gop(2, 250)), [0, 9]);
        assert_eq!(keyframes(&scenes, 4, gop(1, 250)), [0, 9]);
        assert_eq!(keyframes(&scenes, 2, gop(1, 250)), [0, 9]);
        // Without lookahead the flash cannot be told from a cut. The return
        // is then a cut as well, but falls inside min_keyint here.
        assert_eq!(keyframes(&scenes, 0, gop(3, 250)), [0, 4, 9]);
    }

    #[test]
    fn b_pictures_come_in_decoding_order() {
        //            0  1  2  3  4  5  6  7  8  9 10 11 12
        let scenes = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
        for depth in [1u8, 2, 3] {
            let cfg = VideoConfig {
                base: EncodeConfig::default().with_quality(30).with_threads(1),
                gop: GopConfig {
                    structure: GopStructure::HierarchicalB { depth },
                    ..gop(2, 250)
                },
                lookahead: 2,
                ..VideoConfig::default()
            };
            let mut enc = VideoEncoder::new(W as u32, H as u32, cfg).unwrap();
            let mut packets = Vec::new();
            for (n, &scene) in scenes.iter().enumerate() {
                // Timestamps 3 apart, starting at 100.
                let pts = 100 + 3 * n as i64;
                enc.send_frame(&frame(scene, n as u32), pts, false).unwrap();
                packets.extend(std::iter::from_fn(|| enc.receive_packet()));
                // Nothing is coded before its anchor has left the lookahead.
                assert!(packets.iter().all(|p| p.pts <= pts - 3 * 2));
            }
            enc.flush().unwrap();
            packets.extend(std::iter::from_fn(|| enc.receive_packet()));
            let frames: Vec<i64> = packets.iter().map(|p| (p.pts - 100) / 3).collect();
            let kinds: Vec<FrameKind> = packets.iter().map(|p| p.kind).collect();
            use FrameKind::{B, Key, P};
            let (order, kind): (&[i64], &[FrameKind]) = match depth {
                1 => (
                    &[0, 2, 1, 4, 3, 6, 5, 7, 9, 8, 11, 10, 12],
                    &[Key, P, B, P, B, P, B, Key, P, B, P, B, P],
                ),
                2 => (
                    &[0, 4, 2, 1, 3, 6, 5, 7, 11, 9, 8, 10, 12],
                    &[Key, P, B, B, B, P, B, Key, P, B, B, B, P],
                ),
                _ => (
                    &[0, 6, 3, 1, 2, 4, 5, 7, 12, 9, 8, 10, 11],
                    &[Key, P, B, B, B, B, B, Key, P, B, B, B, B],
                ),
            };
            assert_eq!(frames, order, "depth {depth}");
            assert_eq!(kinds, kind, "depth {depth}");
            // Decoding timestamps: the presentation timestamps, delayed by
            // the reorder depth and continued below the first.
            for pair in packets.windows(2) {
                assert!(pair[0].dts < pair[1].dts, "depth {depth}: dts increases");
            }
            assert!(packets.iter().all(|p| p.dts <= p.pts), "depth {depth}");
            assert!(packets[0].dts < 100 && (100 - packets[0].dts) % 3 == 0);
        }
    }

    #[test]
    fn b_packets_are_available_with_their_anchor() {
        let cfg = VideoConfig {
            base: EncodeConfig::default().with_threads(1),
            lookahead: 0,
            gop: GopConfig {
                scene_cut: SceneCut::Off,
                structure: GopStructure::HierarchicalB { depth: 2 },
                ..GopConfig::default()
            },
            ..VideoConfig::default()
        };
        let mut enc = VideoEncoder::new(W as u32, H as u32, cfg).unwrap();
        let picture = frame(1, 0);
        enc.send_frame(&picture, 0, false).unwrap();
        assert_eq!(enc.receive_packet().unwrap().kind, FrameKind::Key);
        for pts in 1..4 {
            enc.send_frame(&picture, pts, false).unwrap();
            assert!(enc.receive_packet().is_none());
        }
        enc.send_frame(&picture, 4, false).unwrap();
        let pts: Vec<_> =
            std::iter::from_fn(|| enc.receive_packet().map(|packet| packet.pts)).collect();
        assert_eq!(pts, [4, 2, 1, 3]);
    }

    #[test]
    fn reordered_timestamps_handle_i64_limits_and_reject_duplicates() {
        let cfg = VideoConfig {
            base: EncodeConfig::default().with_threads(1),
            lookahead: 1,
            gop: GopConfig {
                scene_cut: SceneCut::Off,
                structure: GopStructure::HierarchicalB { depth: 1 },
                ..GopConfig::default()
            },
            ..VideoConfig::default()
        };
        let mut enc = VideoEncoder::new(W as u32, H as u32, cfg).unwrap();
        let picture = frame(1, 0);
        let first = i64::MIN + 3;
        assert!(enc.send_frame(&picture, i64::MIN, false).is_err());
        enc.send_frame(&picture, first, false).unwrap();
        assert!(enc.send_frame(&picture, first, true).is_err());
        assert!(enc.send_frame(&picture, first - 1, false).is_err());
        enc.send_frame(&picture, i64::MAX - 1, false).unwrap();
        enc.send_frame(&picture, i64::MAX, false).unwrap();
        enc.flush().unwrap();
        let stamps: Vec<_> =
            std::iter::from_fn(|| enc.receive_packet().map(|p| (p.pts, p.dts))).collect();
        assert_eq!(
            stamps,
            [
                (first, i64::MIN),
                (i64::MAX, first),
                (i64::MAX - 1, i64::MAX - 1),
            ]
        );
    }

    #[test]
    fn rejects_what_it_cannot_encode() {
        let new = |cfg| VideoEncoder::new(W as u32, H as u32, cfg).err();
        let lossless = VideoConfig {
            base: EncodeConfig::default().with_lossless(true),
            ..VideoConfig::default()
        };
        assert!(matches!(new(lossless), Some(EncodeError::Unsupported(_))));
        // A frame of another size or format than the stream's.
        let mut enc = VideoEncoder::new(W as u32, H as u32, VideoConfig::default()).unwrap();
        enc.send_frame(&frame(1, 0), 0, false).unwrap();
        let small = Yuv::from_planes(
            vec![0; 64 * 64],
            Vec::new(),
            Vec::new(),
            64,
            64,
            ChromaFormat::Monochrome,
            BitDepth::Eight,
        )
        .unwrap();
        assert!(enc.send_frame(&small, 1, false).is_err());
        let mut ten_bit = frame(1, 1);
        ten_bit.bit_depth = BitDepth::Ten;
        assert!(enc.send_frame(&ten_bit, 1, false).is_err());
    }
}
