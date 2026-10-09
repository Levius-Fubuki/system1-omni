use omni_jemm_native::artifacts::verify_export;
use serde_json::json;
#[test]
fn export_hash_map_is_mandatory_and_rejects_traversal() {
    let dir = std::env::temp_dir();
    assert!(
        verify_export(&dir, &json!({}))
            .unwrap_err()
            .to_string()
            .contains("export_sha256")
    );
    assert!(
        verify_export(&dir, &json!({"export_sha256":{"../escape":"0".repeat(64)}}))
            .unwrap_err()
            .to_string()
            .contains("relative")
    );
}
#[test]
fn hashes_every_exported_file_and_requires_all_language_shards() {
    use sha2::{Digest, Sha256};
    let dir = std::env::temp_dir().join(format!("jemm-artifacts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let index = br#"{"weight_map":{"embed_tokens.weight":"model-1.safetensors"}}"#;
    let mut hashes = serde_json::Map::new();
    for name in [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "vision.safetensors",
        "jemm_lm_head.safetensors",
        "model-1.safetensors",
        "decision_config.json",
    ] {
        let bytes = if name == "model.safetensors.index.json" {
            index.as_slice()
        } else {
            b"original"
        };
        std::fs::write(dir.join(name), bytes).unwrap();
        hashes.insert(name.into(), json!(format!("{:x}", Sha256::digest(bytes))));
    }
    let manifest = json!({"export_sha256":hashes});
    verify_export(&dir, &manifest).unwrap();
    std::fs::write(dir.join("decision_config.json"), b"tampered").unwrap();
    assert!(
        verify_export(&dir, &manifest)
            .unwrap_err()
            .to_string()
            .contains("mismatch: decision_config.json")
    );
    let mut incomplete = manifest;
    incomplete["export_sha256"]
        .as_object_mut()
        .unwrap()
        .remove("model-1.safetensors");
    assert!(
        verify_export(&dir, &incomplete)
            .unwrap_err()
            .to_string()
            .contains("missing language shard")
    );
    std::fs::remove_dir_all(dir).unwrap();
}
