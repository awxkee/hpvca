# hpvca

A tiny HEVC encoder in Rust.

## Example

```rust
fn main() {
    let img = image::open("./assets/abstract_alpha.png")
        .unwrap()
        .to_rgba16();
    let arr = img.iter().map(|&x| x >> 6).collect::<Vec<_>>();

    let instant = Instant::now();
    let data = hpvca::encode_with_alpha(
        &arr,
        img.width(),
        img.height(),
        PixelLayout::Rgba,
        &EncodeConfig::default()
            .with_bit_depth(BitDepth::Ten)
            .with_chroma(ChromaFormat::Yuv420),
    )
        .unwrap();
    std::fs::write("output.heic", &data).expect("failed to write output");
}
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.