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

## HDR gain maps

Attach a gain map to make the HEIC display as HDR on capable screens while
staying a plain SDR image everywhere else. It is written the way iPhone HEICs
carry it: an Apple `hdrgainmap` auxiliary image with its `HDRGainMap` XMP, plus
an ISO 21496-1 `tmap` item that references the same gain map image.

```rust
use hpvca::{EncodeConfig, GainMap};

// `ratios[i]`: HDR / SDR linear luminance at gain map pixel `i`
// (any resolution; Apple uses half the primary's).
let gain_map = GainMap::from_ratios(&ratios, width / 2, height / 2)?.with_quality(80);
let heic = hpvca::encode_rgb(&rgb, width, height, &EncodeConfig::new().with_gain_map(gain_map))?;
```

`GainMap::gray8(samples, w, h, headroom)` takes samples already in Apple's
encoding, e.g. a gain map decoded from an iPhone photo; see
[`app/src/bin/gainmap.rs`](app/src/bin/gainmap.rs). `with_iso(false)` writes
only the Apple form.

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.