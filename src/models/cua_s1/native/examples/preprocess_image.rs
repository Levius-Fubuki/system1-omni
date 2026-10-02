//! CPU-only RGB8 preprocessing; run from the repository root with:
//! cargo run -p omni-cua-s1-native --example preprocess_image -- 256 256 image.rgb

use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Write},
};

use anyhow::{Context, Result, ensure};
use omni_cua_s1_native::image_preprocess::preprocess_rgb8;

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3 || args.len() == 4,
        "usage: preprocess_image WIDTH HEIGHT RAW_RGB_PATH [OUTPUT_F32_PATH]"
    );
    let dimension = |index: usize| -> Result<usize> {
        let text = args[index]
            .to_str()
            .context("dimensions must be UTF-8 decimal integers")?;
        ensure!(
            !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()),
            "dimensions must be unsigned decimal integers"
        );
        text.parse().context("dimension is too large")
    };
    let width = dimension(0)?;
    let height = dimension(1)?;
    // Bound the file read before allocating. The library validates the complete
    // contract too; these checks keep malformed CLI inputs cheap to reject.
    ensure!(
        width > 0 && height > 0 && width <= 2048 && height <= 2048,
        "dimensions must be in 1..=2048"
    );
    let area = width.checked_mul(height).context("image area overflow")?;
    ensure!(
        area <= 1_048_576,
        "image area must not exceed 1048576 pixels"
    );
    ensure!(
        width.max(height) <= width.min(height) * 200,
        "image aspect ratio must not exceed 200"
    );
    let expected = area.checked_mul(3).context("RGB length overflow")?;
    let mut rgb = Vec::with_capacity(expected + 1);
    File::open(&args[2])
        .context("opening RGB input")?
        .take((expected + 1) as u64)
        .read_to_end(&mut rgb)
        .context("reading RGB input")?;
    ensure!(
        rgb.len() == expected,
        "RGB input must contain exactly {expected} bytes"
    );
    let image = preprocess_rgb8(width, height, &rgb)?;
    println!(
        "pixel_values shape: [{}, 1536]",
        image.pixel_values.len() / 1536
    );
    println!("image_grid_thw: {:?}", image.image_grid_thw);
    println!("resized: {}x{}", image.resized_width, image.resized_height);
    println!("image_tokens: {}", image.image_tokens());
    if let Some(path) = args.get(3) {
        let mut output = BufWriter::new(File::create(path).context("creating float32 output")?);
        for value in image.pixel_values {
            output
                .write_all(&value.to_le_bytes())
                .context("writing float32 output")?;
        }
        output.flush().context("flushing float32 output")?;
    }
    Ok(())
}
