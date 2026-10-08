use omni_qwen3_5_native::image_preprocess::{ImageLimits, preprocess_rgb8_with_limits};
use omni_qwen3_5_native::vision::{VisionConfig, VisionGeometry};
use serde_json::json;

fn config(h: usize, depth: usize, intermediate: usize, out: usize) -> serde_json::Value {
    json!({"depth":depth,"hidden_size":h,"intermediate_size":intermediate,"num_heads":16,
    "num_position_embeddings":2304,"out_hidden_size":out,"in_channels":3,"patch_size":16,
    "temporal_patch_size":2,"spatial_merge_size":2,"hidden_act":"gelu_pytorch_tanh",
    "deepstack_visual_indexes":[],"model_type":"qwen3_5"})
}
#[test]
fn accepts_only_supported_vision_layouts() {
    let small = VisionConfig::from_value(config(1024, 24, 4096, 2560)).unwrap();
    let large = VisionConfig::from_value(config(1152, 27, 4304, 5120)).unwrap();
    assert_eq!(small.head_dim(), 64);
    assert_eq!(large.head_dim(), 72);
    assert!(VisionConfig::from_value(config(1152, 24, 4304, 5120)).is_err());
    let mut changed = config(1152, 27, 4304, 5120);
    changed["deepstack_visual_indexes"] = json!([8]);
    assert!(VisionConfig::from_value(changed).is_err());
    assert_eq!(
        large.inventory()["model.visual.merger.linear_fc1.weight"],
        vec![4608, 4608]
    );
    assert_eq!(
        large.inventory()["model.visual.blocks.26.mlp.linear_fc1.weight"],
        vec![4304, 1152]
    );
}
#[test]
fn rotary_geometry_uses_36_frequencies_for_72_dimensions() {
    let config = VisionConfig::from_value(config(1152, 27, 4304, 5120)).unwrap();
    let g = VisionGeometry::new([1, 2, 4], &config).unwrap();
    assert_eq!(g.cos.len(), 8 * 36);
    assert_eq!(g.indices.len(), 8 * 4);
    // Processor block-major patches: (0,0), (0,1), (1,0), (1,1), ...
    assert_eq!(g.cos[36], 1.0);
    assert!((g.cos[36 + 18] - 1f32.cos()).abs() < 1e-6);
    assert!((g.cos[72] - 1f32.cos()).abs() < 1e-6);
    for weights in g.weights.as_chunks::<4>().0 {
        assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-6);
    }
    assert!(VisionGeometry::new([2, 2, 4], &config).is_err());
    assert!(VisionGeometry::new([1, 3, 4], &config).is_err());
}
#[test]
fn configurable_source_limits_allow_reference_jemm_screenshots() {
    let limits = ImageLimits {
        max_source_side: 16384,
        max_source_pixels: 3072 * 1024,
        min_pixels: 65536,
        max_pixels: 16777216,
        max_patches: 12288,
    };
    let image =
        preprocess_rgb8_with_limits(1920, 1080, &vec![127; 1920 * 1080 * 3], &limits).unwrap();
    assert_eq!(image.image_grid_thw, [1, 68, 120]);
    assert_eq!(image.image_tokens(), 2040);
    assert!(
        preprocess_rgb8_with_limits(1920, 1080, &vec![127; 1920 * 1080 * 3], &ImageLimits::cua())
            .is_err()
    );
    let narrow = preprocess_rgb8_with_limits(4800, 32, &vec![0; 4800 * 32 * 3], &limits).unwrap();
    assert_eq!(narrow.image_grid_thw, [1, 2, 300]);
    assert!(preprocess_rgb8_with_limits(1920, 1080, &[0; 3], &limits).is_err());
}
#[test]
fn patch_budget_is_checked_after_smart_resize() {
    let limits = ImageLimits {
        max_source_side: 4096,
        max_source_pixels: 3072 * 1024,
        min_pixels: 65536,
        max_pixels: 16777216,
        max_patches: 128,
    };
    assert!(preprocess_rgb8_with_limits(256, 256, &vec![0; 256 * 256 * 3], &limits).is_err());
}

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("omni-vision-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn write(&self, entries: std::collections::BTreeMap<String, Vec<usize>>, bad_dtype: bool) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in entries {
            let dtype = if bad_dtype && name == "model.visual.pos_embed.weight" {
                "F32"
            } else {
                "BF16"
            };
            let length = shape.iter().product::<usize>() * if dtype == "F32" { 4 } else { 2 };
            header.insert(
                name,
                json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+length]}),
            );
            offset += length;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ')
        }
        let mut file = std::fs::File::create(self.0.join("vision.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        // Sparse payload: structural loading must not allocate or read all checkpoint values.
        file.set_len((8 + header.len() + offset) as u64).unwrap();
        std::fs::write(
            self.0.join("config.json"),
            serde_json::to_vec(&json!({"model_type":"qwen3_5",
            "vision_config":config(1152,27,4304,5120),"text_config":{"hidden_size":5120}}))
            .unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn normalized_checkpoint_validates_inventory_dtypes_and_text_width_on_cpu() {
    use omni_qwen3_5_native::vision::VisionCheckpoint;
    let fixture = Fixture::new();
    let c = VisionConfig::from_value(config(1152, 27, 4304, 5120)).unwrap();
    fixture.write(c.inventory(), false);
    {
        let checkpoint = VisionCheckpoint::load(&fixture.0).unwrap();
        assert_eq!(checkpoint.base_names().count(), 333);
        assert_eq!(
            checkpoint
                .base_tensor("model.visual.pos_embed.weight")
                .unwrap()
                .shape(),
            &[2304, 1152]
        );
    }
    fixture.write(c.inventory(), true);
    assert!(
        VisionCheckpoint::load(&fixture.0)
            .err()
            .unwrap()
            .to_string()
            .contains("dtype mismatch")
    );
    let mut missing = c.inventory();
    missing.remove("model.visual.pos_embed.weight");
    fixture.write(missing, false);
    assert!(
        VisionCheckpoint::load(&fixture.0)
            .err()
            .unwrap()
            .to_string()
            .contains("missing visual tensor")
    );
    let mut extra = c.inventory();
    extra.insert("model.visual.unexpected.weight".into(), vec![1]);
    fixture.write(extra, false);
    assert!(
        VisionCheckpoint::load(&fixture.0)
            .err()
            .unwrap()
            .to_string()
            .contains("unexpected visual tensor")
    );
    fixture.write(c.inventory(), false);
    std::fs::write(
        fixture.0.join("config.json"),
        serde_json::to_vec(&json!({"model_type":"qwen3_5",
        "vision_config":config(1152,27,4304,5120),"text_config":{"hidden_size":2560}}))
        .unwrap(),
    )
    .unwrap();
    assert!(
        VisionCheckpoint::load(&fixture.0)
            .err()
            .unwrap()
            .to_string()
            .contains("vision/text config mismatch")
    );
}
#[test]
fn invalid_base_tensor_set_is_rejected_before_loading_cuda() {
    use omni_qwen3_5_native::vision::VisionModel;
    let c = VisionConfig::from_value(config(1152, 27, 4304, 5120)).unwrap();
    let error = VisionModel::from_tensors(
        c,
        Vec::new(),
        Vec::new(),
        std::path::Path::new("/missing-cuda-library"),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("incomplete base vision tensors"));
}
