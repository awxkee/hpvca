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

use hpvca::{BitDepth, ChromaFormat, EncodeConfig, ParallelismStrategy, Speed};
use image::imageops::FilterType;
use std::fs;
use std::time::Instant;

fn main() {
    let img = image::open("./assets/spring_tree.png").unwrap().to_rgb8();
    let arr = img.to_vec(); //;.iter().map(|&x| x >> 6).collect::<Vec<_>>();

    let instant = Instant::now();
    let data = hpvca::encode_rgb(
        &arr,
        img.width(),
        img.height(),
        &EncodeConfig::default()
            .with_chroma(ChromaFormat::Yuv444)
            .with_parallelism(ParallelismStrategy::GridWpp)
            .with_sao(false)
            .with_lossless(false)
            .with_quality(90)
            .with_speed(Speed::Slow),
    )
    .unwrap();
    println!("Encoded time: {:?}", instant.elapsed());
    fs::write("results.heic", data).unwrap();
    println!("Hello, world!");
}
