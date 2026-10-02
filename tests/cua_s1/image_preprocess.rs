use omni_cua_s1_native::image_preprocess::preprocess_rgb8;

fn normalized(pixel: u8) -> f32 {
    (f32::from(pixel) - 127.5) / 127.5
}

#[test]
fn tiny_rgb_is_upscaled_and_temporally_repeated() {
    let image = preprocess_rgb8(1, 1, &[0, 127, 255]).unwrap();
    assert_eq!((image.resized_width, image.resized_height), (256, 256));
    assert_eq!(image.image_grid_thw, [1, 16, 16]);
    assert_eq!(image.image_tokens(), 64);
    assert_eq!(image.pixel_values.len(), 256 * 1536);
    for patch in image.pixel_values.as_chunks::<1536>().0 {
        for (channel, expected) in [0, 127, 255].into_iter().enumerate() {
            assert!(
                patch[channel * 512..(channel + 1) * 512]
                    .iter()
                    .all(|&value| value.to_bits() == normalized(expected).to_bits())
            );
        }
    }
}

#[test]
fn identity_resize_preserves_every_byte_and_patch_merge_order() {
    let width = 256;
    let height = 256;
    let rgb: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| [x as u8, y as u8, (x ^ y) as u8]))
        .collect();
    let image = preprocess_rgb8(width, height, &rgb).unwrap();
    for block_y in 0..8 {
        for block_x in 0..8 {
            for merge_y in 0..2 {
                for merge_x in 0..2 {
                    let patch = ((block_y * 8 + block_x) * 2 + merge_y) * 2 + merge_x;
                    for channel in 0..3 {
                        for temporal in 0..2 {
                            for py in 0..16 {
                                for px in 0..16 {
                                    let x = block_x * 32 + merge_x * 16 + px;
                                    let y = block_y * 32 + merge_y * 16 + py;
                                    let index = patch * 1536
                                        + channel * 512
                                        + temporal * 256
                                        + py * 16
                                        + px;
                                    assert_eq!(
                                        image.pixel_values[index].to_bits(),
                                        normalized(rgb[(y * width + x) * 3 + channel]).to_bits()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn smart_resize_uses_python_ties_even_rounding() {
    for (side, expected) in [(272, 256), (304, 320)] {
        let image = preprocess_rgb8(side, side, &vec![128; side * side * 3]).unwrap();
        assert_eq!(
            (image.resized_width, image.resized_height),
            (expected, expected)
        );
        assert!(
            image
                .pixel_values
                .iter()
                .all(|&value| value == normalized(128))
        );
    }
}

#[test]
fn rejects_invalid_geometry_before_buffer_length_validation() {
    for (width, height) in [(0, 1), (1, 0), (0, 0)] {
        assert_eq!(
            preprocess_rgb8(width, height, &[]).unwrap_err().to_string(),
            "image dimensions must be nonzero",
            "{width}x{height}"
        );
    }
    for (width, height) in [(usize::MAX, 1), (1, usize::MAX), (usize::MAX, usize::MAX)] {
        assert_eq!(
            preprocess_rgb8(width, height, &[]).unwrap_err().to_string(),
            "image sides must not exceed 2048",
            "{width}x{height}"
        );
    }
    for (width, height, expected_error) in [
        (2049, 32, "image sides must not exceed 2048"),
        (32, 2049, "image sides must not exceed 2048"),
        (1025, 1024, "image area must not exceed 1048576 pixels"),
        (201, 1, "image aspect ratio must not exceed 200"),
        (1, 201, "image aspect ratio must not exceed 200"),
    ] {
        // A valid byte length ensures the geometry check itself rejects this
        // image, rather than accidentally passing due to a truncated buffer.
        let rgb = vec![0; width * height * 3];
        assert_eq!(
            preprocess_rgb8(width, height, &rgb)
                .unwrap_err()
                .to_string(),
            expected_error,
            "{width}x{height}"
        );
    }
}

#[test]
fn rejects_incorrect_rgb_buffer_lengths() {
    for rgb in [&[][..], &[1, 2][..], &[1, 2, 3, 4][..]] {
        assert_eq!(
            preprocess_rgb8(1, 1, rgb).unwrap_err().to_string(),
            format!("RGB buffer length must be 3, got {}", rgb.len())
        );
    }
}

#[test]
fn accepts_input_limits_inclusively() {
    for (width, height) in [(200, 1), (1, 200), (2048, 512), (512, 2048)] {
        let image = preprocess_rgb8(width, height, &vec![255; width * height * 3]).unwrap();
        assert_eq!(image.resized_width % 32, 0);
        assert_eq!(image.resized_height % 32, 0);
        assert!(image.pixel_values.iter().all(|&value| value == 1.0));
    }
}

#[test]
fn matches_pinned_processor_full_output_hashes() {
    use sha2::{Digest, Sha256};
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/image_preprocess/manifest.json")).unwrap();
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "fixtures/image_preprocess/preprocessor_config.json"
            ))
        ),
        manifest["config_sha256"].as_str().unwrap(),
        "processor configuration must match the reference manifest"
    );
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let width = case["width"].as_u64().unwrap() as usize;
        let height = case["height"].as_u64().unwrap() as usize;
        let pattern = case["pattern"].as_str().unwrap();
        let mut state = case["seed"].as_u64().unwrap() as u32;
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    rgb.push(match pattern {
                        "noise" => {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            state as u8
                        }
                        "constant" => [0, 128, 255][channel],
                        "checker" => {
                            if (x + y + channel) % 2 == 0 {
                                0
                            } else {
                                255
                            }
                        }
                        "ramp" => ((x * 13 + y * 7 + channel * 83) % 256) as u8,
                        _ => panic!("unknown fixture pattern {pattern}"),
                    });
                }
            }
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(&rgb)),
            case["input_sha256"].as_str().unwrap(),
            "{name} input"
        );
        let image = preprocess_rgb8(width, height, &rgb).unwrap();
        assert_eq!(
            serde_json::json!(image.image_grid_thw),
            case["grid"],
            "{name} grid"
        );
        assert_eq!(
            serde_json::json!([image.pixel_values.len() / 1536, 1536]),
            case["shape"],
            "{name} shape"
        );
        assert_eq!(
            image.image_tokens(),
            case["image_tokens"].as_u64().unwrap() as usize,
            "{name} tokens"
        );
        assert_eq!(
            image.resized_width,
            case["resized_width"].as_u64().unwrap() as usize,
            "{name} width"
        );
        assert_eq!(
            image.resized_height,
            case["resized_height"].as_u64().unwrap() as usize,
            "{name} height"
        );
        let mut hash = Sha256::new();
        for value in image.pixel_values {
            hash.update(value.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            case["output_sha256"].as_str().unwrap(),
            "{name} output"
        );
    }
}
