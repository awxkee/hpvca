//! Multi-threaded encode coverage across picture sizes and parallelism
//! strategies. Written to be run under ThreadSanitizer:
//!
//! ```sh
//! RUSTFLAGS="-Zsanitizer=thread" cargo +nightly test --test thread_safety \
//!     --target aarch64-apple-darwin -Zbuild-std --release
//! ```
//!
//! The WPP encoder shares the reconstruction planes and per-4x4 maps across
//! worker threads through raw pointers (`SyncSlice`), relying on two
//! structural invariants: each CTU row is written by exactly one thread, and a
//! thread reads the rows above only after their progress atomics are
//! published. Grid cells and tiles add a second layer of concurrency on the
//! same pool. These tests exercise sizes that land on and off the CTU grid, on
//! and off the chroma grid, below and above the grid threshold, so a missing
//! synchronisation edge has somewhere to show up.

use hpvca::{BitDepth, ChromaFormat, EncodeConfig, ParallelismStrategy, Speed};

/// Sizes chosen to hit: sub-CTU, exact CTU multiples, one-past-CTU, odd
/// dimensions (chroma rounding), many-CTU-row wavefronts, and sizes above the
/// 512 px grid threshold.
const SIZES: &[(u32, u32)] = &[
    (8, 8),
    (17, 13),
    (64, 64),
    (65, 64),
    (64, 65),
    (127, 129),
    (256, 192),
    (320, 256),
    (513, 129),
    (640, 384),
    (700, 540),
];

const STRATEGIES: &[ParallelismStrategy] = &[
    ParallelismStrategy::Single,
    ParallelismStrategy::Wpp,
    ParallelismStrategy::TilesWpp,
    ParallelismStrategy::Grid,
    ParallelismStrategy::GridWpp,
];

/// Deterministic pseudo-random content with structure: gradients plus a
/// high-frequency component, so CU/TU splitting and the intra mode search
/// actually diverge across blocks rather than picking one flat answer.
fn make_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize) * 3];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for y in 0..h as usize {
        for x in 0..w as usize {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 56) as u8;
            let i = (y * w as usize + x) * 3;
            out[i] = ((x * 255 / w.max(1) as usize) as u8).wrapping_add(noise >> 3);
            out[i + 1] = ((y * 255 / h.max(1) as usize) as u8).wrapping_add(noise >> 4);
            out[i + 2] = noise;
        }
    }
    out
}

fn make_gray(w: u32, h: u32) -> Vec<u8> {
    make_rgb(w, h).chunks_exact(3).map(|p| p[0]).collect()
}

fn make_rgba(w: u32, h: u32) -> Vec<u8> {
    let rgb = make_rgb(w, h);
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        out.extend_from_slice(px);
        out.push((i % 251) as u8);
    }
    out
}

fn cfg(strategy: ParallelismStrategy, chroma: ChromaFormat, threads: usize) -> EncodeConfig {
    EncodeConfig::default()
        .with_quality(72)
        .with_chroma(chroma)
        .with_parallelism(strategy)
        .with_threads(threads)
}

#[test]
fn rgb_all_sizes_strategies_and_chroma_formats() {
    for &(w, h) in SIZES {
        let rgb = make_rgb(w, h);
        for &strategy in STRATEGIES {
            for &chroma in &[
                ChromaFormat::Yuv420,
                ChromaFormat::Yuv422,
                ChromaFormat::Yuv444,
            ] {
                for &threads in &[1usize, 4] {
                    let out = hpvca::encode_rgb(&rgb, w, h, &cfg(strategy, chroma, threads))
                        .unwrap_or_else(|e| panic!("{w}x{h} {strategy:?} {chroma:?} t{threads}: {e:?}"));
                    assert!(!out.is_empty(), "{w}x{h} {strategy:?} produced no output");
                }
            }
        }
    }
}

#[test]
fn wpp_output_is_thread_count_invariant() {
    // The wavefront must produce identical bitstreams regardless of how many
    // workers happen to claim rows — that is the whole soundness argument for
    // the shared-plane writes. `TilesWpp` is excluded on purpose: its tile grid
    // is chosen from the thread count (`tile_target = sqrt(threads)`), so its
    // bitstream is *meant* to vary; it is covered by the determinism test below.
    for &(w, h) in SIZES {
        let rgb = make_rgb(w, h);
        for &strategy in &[ParallelismStrategy::Wpp, ParallelismStrategy::GridWpp] {
            let one =
                hpvca::encode_rgb(&rgb, w, h, &cfg(strategy, ChromaFormat::Yuv420, 1)).unwrap();
            for &threads in &[2usize, 3, 8] {
                let many =
                    hpvca::encode_rgb(&rgb, w, h, &cfg(strategy, ChromaFormat::Yuv420, threads))
                        .unwrap();
                assert_eq!(
                    one, many,
                    "{w}x{h} {strategy:?}: output changed with {threads} threads"
                );
            }
        }
    }
}

#[test]
fn parallel_encodes_are_deterministic_at_a_fixed_thread_count() {
    // Every strategy must be reproducible: same input, same settings, same
    // bytes. A race in the shared planes or in tile/cell assembly would show up
    // here as an intermittent mismatch.
    for &(w, h) in SIZES {
        let rgb = make_rgb(w, h);
        for &strategy in STRATEGIES {
            let first =
                hpvca::encode_rgb(&rgb, w, h, &cfg(strategy, ChromaFormat::Yuv420, 4)).unwrap();
            for _ in 0..3 {
                let again =
                    hpvca::encode_rgb(&rgb, w, h, &cfg(strategy, ChromaFormat::Yuv420, 4)).unwrap();
                assert_eq!(first, again, "{w}x{h} {strategy:?}: non-deterministic output");
            }
        }
    }
}

#[test]
fn alpha_gray_and_high_bit_depth_paths() {
    for &(w, h) in SIZES {
        for &threads in &[1usize, 4] {
            let c = cfg(ParallelismStrategy::GridWpp, ChromaFormat::Yuv420, threads);
            let gray = make_gray(w, h);
            hpvca::encode_gray(&gray, w, h, &c).unwrap();

            let rgba = make_rgba(w, h);
            hpvca::encode_rgba_with_alpha(&rgba, w, h, &c).unwrap();

            let wide: Vec<u16> = make_rgb(w, h).iter().map(|&v| (v as u16) << 2).collect();
            hpvca::encode_rgb10(&wide, w, h, &c).unwrap();
        }
    }
    let _ = BitDepth::Ten;
}

#[test]
fn lossless_and_slow_effort_under_threads() {
    for &(w, h) in &[(64u32, 64u32), (320, 256), (640, 384)] {
        let rgb = make_rgb(w, h);
        for &threads in &[1usize, 4] {
            let lossless = EncodeConfig::default()
                .with_lossless(true)
                .with_parallelism(ParallelismStrategy::GridWpp)
                .with_threads(threads);
            hpvca::encode_rgb(&rgb, w, h, &lossless).unwrap();

            let slow = cfg(ParallelismStrategy::Wpp, ChromaFormat::Yuv420, threads)
                .with_speed(Speed::Slow);
            hpvca::encode_rgb(&rgb, w, h, &slow).unwrap();
        }
    }
}
