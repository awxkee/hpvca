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

//! HEVC CABAC arithmetic encoder.

/// LPS range table from HEVC spec Table 9-43.
#[rustfmt::skip]
pub(crate) static RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128,176,208,240],[128,167,197,227],[128,158,187,216],[123,150,178,205],
    [116,142,169,195],[111,135,160,185],[105,128,152,175],[100,122,144,166],
    [ 95,116,137,158],[ 90,110,130,150],[ 85,104,123,142],[ 81, 99,117,135],
    [ 77, 94,111,128],[ 73, 89,105,122],[ 69, 85,100,116],[ 66, 80, 95,110],
    [ 62, 76, 90,104],[ 59, 72, 86, 99],[ 56, 69, 81, 94],[ 53, 65, 77, 89],
    [ 51, 62, 73, 85],[ 48, 59, 69, 80],[ 46, 56, 66, 76],[ 43, 53, 63, 72],
    [ 41, 50, 59, 69],[ 39, 48, 56, 65],[ 37, 45, 54, 62],[ 35, 43, 51, 59],
    [ 33, 41, 48, 56],[ 32, 39, 46, 53],[ 30, 37, 43, 50],[ 29, 35, 41, 48],
    [ 27, 33, 39, 45],[ 26, 31, 37, 43],[ 24, 30, 35, 41],[ 23, 28, 33, 39],
    [ 22, 27, 32, 37],[ 21, 26, 30, 35],[ 20, 24, 29, 33],[ 19, 23, 27, 31],
    [ 18, 22, 26, 30],[ 17, 21, 25, 28],[ 16, 20, 23, 27],[ 15, 19, 22, 25],
    [ 14, 18, 21, 24],[ 14, 17, 20, 23],[ 13, 16, 19, 22],[ 12, 15, 18, 21],
    [ 12, 14, 17, 20],[ 11, 14, 16, 19],[ 11, 13, 15, 18],[ 10, 12, 15, 17],
    [ 10, 12, 14, 16],[  9, 11, 13, 15],[  9, 11, 12, 14],[  8, 10, 12, 14],
    [  8,  9, 11, 13],[  7,  9, 11, 12],[  7,  9, 10, 12],[  7,  8, 10, 11],
    [  6,  8,  9, 11],[  6,  7,  9, 10],[  6,  7,  8,  9],[  2,  2,  2,  2],
];

/// State transition: MPS path.
#[rustfmt::skip]
pub(crate) static TRANS_IDX_MPS: [u8; 64] = [
     1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63,
];

/// State transition: LPS path.
#[rustfmt::skip]
pub(crate) static TRANS_IDX_LPS: [u8; 64] = [
     0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9,11,11,12,
    13,13,15,15,16,16,18,18,19,19,21,21,22,22,23,24,
    24,25,26,26,27,27,28,29,29,30,30,30,31,32,32,33,
    33,33,34,34,35,35,35,36,36,36,37,37,37,38,38,63,
];

/// Number of renorm bits for a given LPS range value.
/// Index is lps>>3 (0..31). Values from x265/openHEVC renorm table.
/// A single CABAC context model: probability state index + MPS value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CtxModel {
    pub(crate) p_state_idx: u8,
    pub(crate) val_mps: u8,
}

// Fractional bit estimates for the 64 CABAC probability states. These use the
// average LPS probability represented by RANGE_TAB_LPS over the four range
// classes. HM uses the same kind of frozen-context entropy estimates for RDOQ:
// probability states are sampled at the TU entrance, while c1/c2/Rice syntax
// state still follows the candidate levels.
#[rustfmt::skip]
static EST_BITS_MPS: [f32; 64] = [
    0.962303, 0.907699, 0.859384, 0.804965, 0.749568, 0.701959, 0.655262, 0.614665,
    0.577596, 0.541455, 0.506198, 0.477201, 0.448776, 0.421908, 0.398025, 0.374826,
    0.351704, 0.332561, 0.313866, 0.295216, 0.280483, 0.263488, 0.250106, 0.235369,
    0.222108, 0.210290, 0.199249, 0.188472, 0.177775, 0.169542, 0.158983, 0.151692,
    0.142221, 0.135013, 0.127592, 0.120457, 0.115408, 0.109439, 0.103326, 0.097407,
    0.093239, 0.088284, 0.083223, 0.078302, 0.074190, 0.071407, 0.067314, 0.063232,
    0.060470, 0.057475, 0.054724, 0.051577, 0.049899, 0.045867, 0.044195, 0.041845,
    0.039124, 0.037063, 0.036173, 0.034351, 0.032179, 0.030361, 0.028708, 0.007823,
];
#[rustfmt::skip]
static EST_BITS_LPS: [f32; 64] = [
    1.038709, 1.098613, 1.155816, 1.225585, 1.303228, 1.376084, 1.453875, 1.527331,
    1.599_81, 1.676123, 1.756699, 1.828126, 1.903274, 1.979583, 2.052277, 2.127_83,
    2.208615, 2.280162, 2.354635, 2.434016, 2.500752, 2.582698, 2.651404, 2.731857,
    2.809_06, 2.882185, 2.954603, 3.029557, 3.108_62, 3.172997, 3.260_58, 3.324732,
    3.413088, 3.484573, 3.562481, 3.641983, 3.701272, 3.774937, 3.854851, 3.937_03,
    3.998051, 4.074383, 4.157046, 4.242541, 4.318337, 4.372123, 4.455251, 4.543465,
    4.606535, 4.678325, 4.747726, 4.831597, 4.878486, 4.998051, 5.050_78, 5.128427,
    5.224097, 5.301123, 5.335_76, 5.409426, 5.502581, 5.585541, 5.665_52, 7.530762,
];

impl CtxModel {
    /// Frozen-context fractional bit estimate used by winner-only RDOQ.
    #[inline]
    pub(crate) fn estimated_bits(self, bin: u8) -> f32 {
        let state = self.p_state_idx as usize;
        if (bin ^ self.val_mps) & 1 == 0 {
            EST_BITS_MPS[state]
        } else {
            EST_BITS_LPS[state]
        }
    }

    /// Apply the exact CABAC probability-state transition without calculating
    /// a fractional rate. This is used when a later RDO stage only needs the
    /// contexts produced by already-selected syntax.
    #[inline]
    pub(crate) fn update(&mut self, bin: u8) {
        let state = self.p_state_idx as usize;
        if (bin ^ self.val_mps) & 1 == 0 {
            self.p_state_idx = TRANS_IDX_MPS[state];
        } else {
            if self.p_state_idx == 0 {
                self.val_mps ^= 1;
            }
            self.p_state_idx = TRANS_IDX_LPS[state];
        }
    }

    /// Fractional CABAC bit cost plus the same probability-state transition as
    /// a real context-coded bin. Used by speculative encoder RDO so trials do
    /// not drive the arithmetic coder or touch its output buffer.
    #[inline]
    pub(crate) fn estimate_and_update(&mut self, bin: u8) -> f32 {
        let bits = self.estimated_bits(bin);
        self.update(bin);
        bits
    }

    /// Initialize from HEVC spec §9.3.2.2.
    pub(crate) fn init(init_value: u8, qp: u8) -> Self {
        // HEVC §9.3.2.2 context initialization.
        let slope_idx = (init_value >> 4) as i32;
        let offset_idx = (init_value & 0x0F) as i32;
        let m = slope_idx * 5 - 45;
        let n = (offset_idx << 3) - 16;
        let qpc = (qp as i32).clamp(0, 51);
        let pre = (((m * qpc) >> 4) + n).clamp(1, 126);
        if pre >= 64 {
            CtxModel {
                p_state_idx: (pre - 64) as u8,
                val_mps: 1,
            }
        } else {
            CtxModel {
                p_state_idx: (63 - pre) as u8,
                val_mps: 0,
            }
        }
    }

    /// Test-only constructor with an explicit probability state and MPS value.
    #[cfg(test)]
    pub(crate) fn fixed(p: u8, m: u8) -> Self {
        CtxModel {
            p_state_idx: p,
            val_mps: m,
        }
    }
}

/// Minimal CABAC writer interface shared by the real arithmetic coder and the
/// fractional-bit estimator used by speculative RD trials.
pub(crate) trait CabacWriter {
    fn encode_bin(&mut self, bin_val: u8, ctx: &mut CtxModel);
    fn encode_bypass(&mut self, bin_val: u8);
}

/// Fast fractional-bit CABAC sink. It updates context states exactly like the
/// arithmetic coder, but bypass bins simply add one bit and no output is
/// generated. This is the normal hot-path model for encoder mode RDO.
#[derive(Clone, Copy, Default)]
pub(crate) struct CabacEstimator {
    bits: f32,
}

impl CabacEstimator {
    #[inline]
    pub(crate) fn bits(&self) -> f32 {
        self.bits
    }
}

impl CabacWriter for CabacEstimator {
    #[inline]
    fn encode_bin(&mut self, bin_val: u8, ctx: &mut CtxModel) {
        self.bits += ctx.estimate_and_update(bin_val);
    }

    #[inline]
    fn encode_bypass(&mut self, _bin_val: u8) {
        self.bits += 1.0;
    }
}

/// Context-only CABAC sink. It follows all regular-bin state transitions but
/// deliberately ignores bypass bins and rate accumulation. This is much cheaper
/// than the fractional estimator when only the post-syntax context state is
/// required by a subsequent winner-only RDOQ pass.
#[derive(Clone, Copy, Default)]
pub(crate) struct CabacContextUpdater;

impl CabacWriter for CabacContextUpdater {
    #[inline]
    fn encode_bin(&mut self, bin_val: u8, ctx: &mut CtxModel) {
        ctx.update(bin_val);
    }

    #[inline]
    fn encode_bypass(&mut self, _bin_val: u8) {}
}

/// HEVC CABAC encoder.
///
/// Implements the standard arithmetic coder with a 9-bit `low` register and the
/// classic bit-FIFO renormalisation with outstanding-bit (carry) handling, as in
/// the H.264/HEVC reference software. This produces a bitstream exactly
/// compatible with the HEVC arithmetic decoder (verified by an independent
/// decoder over tens of thousands of random symbol sequences).
#[derive(Clone)]
pub(crate) struct CabacEncoder {
    low: u32,              // 10-bit working low register
    m_range: u32,          // current range [256, 510]
    bits_outstanding: u32, // count of pending carry-dependent bits
    first_bit: bool,       // suppress the very first put_bit (H.264/HEVC convention)
    bit_buffer: u8,        // partial output byte
    bit_count: u8,         // bits filled in bit_buffer (0..8)
    pub(crate) output: Vec<u8>,
}

impl CabacEncoder {
    pub(crate) fn new() -> Self {
        CabacEncoder {
            low: 0,
            m_range: 510,
            bits_outstanding: 0,
            first_bit: true,
            bit_buffer: 0,
            bit_count: 0,
            output: Vec::new(),
        }
    }

    #[inline]
    fn emit_bit(&mut self, b: u32) {
        self.bit_buffer = (self.bit_buffer << 1) | (b as u8 & 1);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.output.push(self.bit_buffer);
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
    }

    /// Output a resolved bit plus any outstanding (carry-deferred) opposite bits.
    #[inline]
    fn put_bit(&mut self, b: u32) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.emit_bit(b);
        }
        while self.bits_outstanding > 0 {
            self.emit_bit(1 - b);
            self.bits_outstanding -= 1;
        }
    }

    /// Renormalize after a context-coded bin.
    #[inline]
    fn renorm(&mut self) {
        while self.m_range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(1);
            } else {
                self.low -= 256;
                self.bits_outstanding += 1;
            }
            self.m_range <<= 1;
            self.low <<= 1;
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Context-adaptive binary encoding.
    #[inline]
    pub(crate) fn encode_bin(&mut self, bin_val: u8, ctx: &mut CtxModel) {
        let state = ctx.p_state_idx as usize;
        let lps = RANGE_TAB_LPS[state][(self.m_range >> 6) as usize & 3] as u32;
        self.m_range -= lps;

        if (bin_val ^ ctx.val_mps) & 1 == 0 {
            // MPS
            ctx.p_state_idx = TRANS_IDX_MPS[state];
        } else {
            // LPS
            self.low += self.m_range;
            self.m_range = lps;
            if ctx.p_state_idx == 0 {
                ctx.val_mps ^= 1;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[state];
        }
        self.renorm();
    }

    /// Equal-probability bypass encoding.
    #[inline]
    pub(crate) fn encode_bypass(&mut self, bin_val: u8) {
        self.low <<= 1;
        if bin_val != 0 {
            self.low += self.m_range;
        }
        if self.low >= 1024 {
            self.put_bit(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(0);
        } else {
            self.low -= 512;
            self.bits_outstanding += 1;
        }
    }

    /// Encode end_of_slice_segment_flag (terminate bin).
    pub(crate) fn encode_terminate(&mut self, flag: u8) {
        self.m_range -= 2;
        if flag != 0 {
            self.low += self.m_range;
            // Flush the arithmetic coder (encodeBinTrm + finish).
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// EncodeFlush per HEVC spec §9.3.4.3.5. Called when the terminate bin = 1.
    /// The spec procedure is precise about the trailing bits, and strict decoders
    /// (Apple VideoToolbox hardware) depend on this exact pattern:
    ///   ivlCurrRange = 2
    ///   RenormE()
    ///   PutBit( (ivlLow >> 9) & 1 )
    ///   WriteBits( ((ivlLow >> 7) & 3) | 1, 2 )
    fn flush(&mut self) {
        self.m_range = 2;
        self.renorm();
        self.put_bit((self.low >> 9) & 1);
        // Write the final 2 bits: bits [8:7] of low, with the low bit forced to 1
        // (the rbsp stop bit). Emit MSB first.
        let two = ((self.low >> 7) & 3) | 1;
        self.emit_bit((two >> 1) & 1);
        self.emit_bit(two & 1);
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        // Byte-align: pad the partial byte with zeros.
        if self.bit_count > 0 {
            self.bit_buffer <<= 8 - self.bit_count;
            self.output.push(self.bit_buffer);
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
        self.output
    }
}

impl CabacWriter for CabacEncoder {
    #[inline]
    fn encode_bin(&mut self, bin_val: u8, ctx: &mut CtxModel) {
        CabacEncoder::encode_bin(self, bin_val, ctx);
    }

    #[inline]
    fn encode_bypass(&mut self, bin_val: u8) {
        CabacEncoder::encode_bypass(self, bin_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminate_produces_output() {
        let mut enc = CabacEncoder::new();
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty(), "Termination should produce output");
    }

    #[test]
    fn ctx_model_init_valid() {
        let ctx = CtxModel::init(111, 26);
        assert!(ctx.p_state_idx < 64);
        assert!(ctx.val_mps <= 1);
    }

    #[test]
    fn bypass_zero() {
        let mut enc = CabacEncoder::new();
        for _ in 0..8 {
            enc.encode_bypass(0);
        }
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty());
    }

    #[test]
    fn mps_sequence_then_terminate() {
        let mut enc = CabacEncoder::new();
        let mut ctx = CtxModel::fixed(20, 0);
        for _ in 0..16 {
            enc.encode_bin(0, &mut ctx);
        }
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty());
    }

    #[test]
    fn fractional_estimator_matches_context_transitions() {
        let bins = [0u8, 0, 1, 0, 1, 1, 0, 0, 1];
        let mut real_ctx = CtxModel::fixed(20, 0);
        let mut estimated_ctx = real_ctx;
        let mut enc = CabacEncoder::new();
        let mut est = CabacEstimator::default();
        for bin in bins {
            enc.encode_bin(bin, &mut real_ctx);
            est.encode_bin(bin, &mut estimated_ctx);
        }
        assert_eq!(real_ctx.p_state_idx, estimated_ctx.p_state_idx);
        assert_eq!(real_ctx.val_mps, estimated_ctx.val_mps);
        assert!(est.bits().is_finite() && est.bits() > 0.0);
    }

    #[test]
    fn context_only_updater_matches_fractional_estimator_state() {
        let bins = [1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1];
        let mut estimated_ctx = CtxModel::fixed(13, 1);
        let mut updated_ctx = estimated_ctx;
        let mut est = CabacEstimator::default();
        let mut updater = CabacContextUpdater;
        for bin in bins {
            est.encode_bin(bin, &mut estimated_ctx);
            updater.encode_bin(bin, &mut updated_ctx);
        }
        assert_eq!(estimated_ctx.p_state_idx, updated_ctx.p_state_idx);
        assert_eq!(estimated_ctx.val_mps, updated_ctx.val_mps);
    }

    #[test]
    fn lps_produces_output() {
        // Encode one LPS bin and verify output is produced
        let mut enc = CabacEncoder::new();
        let mut ctx = CtxModel::fixed(0, 1); // state=0, mps=1 → LPS gives many renorm bits
        enc.encode_bin(0, &mut ctx); // LPS: bin=0 ≠ mps=1
        enc.encode_terminate(1);
        let out = enc.finish();
        assert!(!out.is_empty(), "LPS should produce output bytes");
    }
}
