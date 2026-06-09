# hpvca

A tiny HEVC encoder in Rust.

## Example

```rust
fn main() {
    let img = image::open("./assets/abstract_alpha.png")
        .unwrap()
        .to_rgba8();
    let arr = img.to_vec();

    let data = hpvca::encode_rgba_with_alpha(
        &arr,
        img.width(),
        img.height(),
        &EncodeConfig::default().with_chroma(ChromaFormat::Yuv444),
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