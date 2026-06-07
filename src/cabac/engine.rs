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

impl CtxModel {
    /// Initialise from HEVC spec §9.3.2.2.
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

    /// Renormalise after a context-coded bin.
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

    /// Finish encoding and return the byte-aligned output buffer.
    /// Coded length in bits if the stream were terminated here (clones + flushes
    /// a copy). RDO differences two of these for a CU's true marginal coded cost.
    pub(crate) fn flushed_bits(&self) -> u64 {
        let mut probe = self.clone();
        probe.encode_terminate(1); // resolves `low` + outstanding into output/bit_buffer
        (probe.output.len() as u64) * 8 + probe.bit_count as u64
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
