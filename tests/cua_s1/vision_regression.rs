//! The same harness can run against original Cua and the extracted shared executor.
//! CUA_VISION_BASE, CUA_VISION_ADAPTER, CUA_S1_CUDA_LIB and CUA_VISION_DUMP_DIR are required.
use omni_cua_s1_native::{image_preprocess::preprocess_rgb8, vision::VisionModel};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
#[test]
#[ignore = "needs pinned Cua vision checkpoint, GPU, library and dump directory"]
fn dump_cua_vision_features_and_stage_hashes() {
    let required =
        |key| PathBuf::from(std::env::var_os(key).unwrap_or_else(|| panic!("{key} must be set")));
    let base = required("CUA_VISION_BASE");
    let adapter = required("CUA_VISION_ADAPTER");
    let library = required("CUA_S1_CUDA_LIB");
    let dir = required("CUA_VISION_DUMP_DIR");
    fs::create_dir_all(&dir).unwrap();
    let mut model = VisionModel::load(base, adapter, &library).unwrap();
    for (width, height) in [(256usize, 256usize), (512, 256), (512, 512)] {
        let rgb: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 37 + i / 7 * 13 + 17) % 256) as u8)
            .collect();
        let image = preprocess_rgb8(width, height, &rgb).unwrap();
        let case = format!("{width}x{height}");
        let mut stages = Vec::new();
        let output = model
            .forward_with_trace(&image, |name, values| {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                let hash = format!("{:x}", Sha256::digest(&bytes));
                stages
                    .push(serde_json::json!({"stage":name,"elements":values.len(),"sha256":hash}));
                Ok(())
            })
            .unwrap();
        assert_eq!(output.len(), image.image_tokens() * 2560);
        assert!(output.iter().all(|x| x.is_finite()));
        let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dir.join(format!("{case}.features.bf16")), &bytes).unwrap();
        let report = serde_json::json!({"case":case,"grid":image.image_grid_thw,"stages":stages,
            "feature_sha256":format!("{:x}",Sha256::digest(&bytes))});
        fs::write(
            dir.join(format!("{case}.json")),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    model.synchronize().unwrap();
}
