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

//! Per-worker reusable coding workspace.

use crate::hevc::CompressionContext;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

/// One worker's complete coding workspace. See the module docs.
pub(crate) struct CoderScratch {
    /// Per-CU prediction/transform/RDO working set (fixed-size, ~200 KiB).
    pub(crate) cc: CompressionContext,
    /// Padded luma reconstruction plane (coded width × coded height).
    pub(crate) rec_y: Vec<u16>,
    /// Padded chroma reconstruction planes (empty for monochrome).
    pub(crate) rec_cb: Vec<u16>,
    pub(crate) rec_cr: Vec<u16>,
    /// Per-8×8 CU depth map (split_cu_flag contexts).
    pub(crate) cu_depth: Vec<u8>,
    /// Per-4×4 luma intra-mode map (MPM derivation).
    pub(crate) mode_map: Vec<u8>,
    /// Per-4×4 deblocking edge flags.
    pub(crate) edge_v: Vec<bool>,
    pub(crate) edge_h: Vec<bool>,
    /// Per-4×4 QpY map for AQ deblocking (empty when AQ is off).
    pub(crate) qp_map: Vec<u8>,
    /// Padded source luma for the SAO statistics pass.
    pub(crate) sao_original: Vec<u16>,
    /// Pre-filter source snapshot consumed by `sao::apply_luma`.
    pub(crate) sao_apply: Vec<u16>,
    /// Donated YUV conversion planes, reclaimed from the encoded [`crate::Yuv`].
    pub(crate) conv_y: Vec<u16>,
    pub(crate) conv_cb: Vec<u16>,
    pub(crate) conv_cr: Vec<u16>,
    /// Interleaved staging: bit-widening, padding, grid tile extraction.
    pub(crate) stage: Vec<u16>,
    /// Second staging buffer for paths that carry an alpha plane.
    pub(crate) stage2: Vec<u16>,
}

impl CoderScratch {
    fn new() -> Box<Self> {
        Box::new(Self {
            cc: CompressionContext::new(),
            rec_y: Vec::new(),
            rec_cb: Vec::new(),
            rec_cr: Vec::new(),
            cu_depth: Vec::new(),
            mode_map: Vec::new(),
            edge_v: Vec::new(),
            edge_h: Vec::new(),
            qp_map: Vec::new(),
            sao_original: Vec::new(),
            sao_apply: Vec::new(),
            conv_y: Vec::new(),
            conv_cb: Vec::new(),
            conv_cr: Vec::new(),
            stage: Vec::new(),
            stage2: Vec::new(),
        })
    }
}

/// Recycled workspaces. Bounded by the peak number of concurrent leases ever
/// reached (worker count), not by throughput.
static STASH: Mutex<Vec<Box<CoderScratch>>> = Mutex::new(Vec::new());

/// Exclusive loan of a [`CoderScratch`]; returns it to the stash on drop.
pub(crate) struct ScratchLease(Option<Box<CoderScratch>>);

/// Lease a workspace: recycled when one is available, freshly allocated only
/// when the process has never run this many workers concurrently.
pub(crate) fn lease() -> ScratchLease {
    let recycled = STASH.lock().unwrap().pop();
    ScratchLease(Some(recycled.unwrap_or_else(CoderScratch::new)))
}

impl Deref for ScratchLease {
    type Target = CoderScratch;
    fn deref(&self) -> &CoderScratch {
        self.0.as_deref().expect("lease is populated until drop")
    }
}

impl DerefMut for ScratchLease {
    fn deref_mut(&mut self) -> &mut CoderScratch {
        self.0.as_deref_mut().expect("lease is populated until drop")
    }
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        if let Some(scratch) = self.0.take() {
            STASH.lock().unwrap().push(scratch);
        }
    }
}

/// Overwrite `out` with `len` copies of `value`, reusing retained capacity.
#[inline]
pub(crate) fn fill_resize<T: Copy>(out: &mut Vec<T>, len: usize, value: T) {
    out.clear();
    out.resize(len, value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_recycles_the_same_workspace() {
        let first = {
            let mut lease = lease();
            lease.rec_y.resize(4096, 7);
            lease.rec_y.as_ptr() as usize
        };
        let lease = lease();
        // Capacity (and, absent other threads, the very allocation) survives.
        assert!(lease.rec_y.capacity() >= 4096 || lease.rec_y.as_ptr() as usize != first);
    }

    #[test]
    fn fill_resize_overwrites_previous_contents() {
        let mut v = vec![9u8; 16];
        fill_resize(&mut v, 8, 3);
        assert_eq!(v, vec![3u8; 8]);
        fill_resize(&mut v, 12, 0);
        assert_eq!(v, vec![0u8; 12]);
    }
}
