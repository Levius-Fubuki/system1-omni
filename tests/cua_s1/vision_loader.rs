use omni_cua_s1_native::vision::VisionCheckpoint;
use safetensors::Dtype;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const BASE: &str = "model.visual.";
const ADAPTER: &str = "base_model.model.model.visual.";
type Inventory = BTreeMap<String, Value>;
fn inventory(adapter: bool) -> Inventory {
    serde_json::from_str(if adapter {
        include_str!("fixtures/vision/adapter-tensors.json")
    } else {
        include_str!("fixtures/vision/base-tensors.json")
    })
    .unwrap()
}
fn sparse(path: &Path, tensors: &Inventory) {
    let mut offset = 0u64;
    let mut header = serde_json::Map::new();
    for (name, tensor) in tensors {
        let bytes = tensor["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .product::<u64>()
            * if tensor["dtype"] == "BF16" { 2 } else { 4 };
        header.insert(name.clone(), json!({"dtype":tensor["dtype"],"shape":tensor["shape"],"data_offsets":[offset, offset+bytes]}));
        offset += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + offset).unwrap();
    for tensor in serde_json::from_slice::<Value>(&header)
        .unwrap()
        .as_object()
        .unwrap()
        .values()
    {
        file.seek(SeekFrom::Start(
            8 + header.len() as u64 + tensor["data_offsets"][0].as_u64().unwrap(),
        ))
        .unwrap();
        file.write_all(if tensor["dtype"] == "BF16" {
            &[0x80, 0x3f]
        } else {
            &[0, 0, 0x80, 0x3f]
        })
        .unwrap();
    }
}
struct Fixture {
    root: PathBuf,
    base: PathBuf,
    adapter: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "cua-vision-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let base = root.join("base");
        let adapter = root.join("adapter");
        fs::create_dir_all(&base).unwrap();
        fs::create_dir_all(&adapter).unwrap();
        fs::write(
            base.join("config.json"),
            include_str!("fixtures/vision/config.json"),
        )
        .unwrap();
        fs::write(
            adapter.join("adapter_config.json"),
            include_str!("fixtures/vision/adapter_config.json"),
        )
        .unwrap();
        sparse(&base.join("model.safetensors"), &inventory(false));
        sparse(&adapter.join("adapter_model.safetensors"), &inventory(true));
        Self {
            root,
            base,
            adapter,
        }
    }
    fn load(&self) -> anyhow::Result<VisionCheckpoint> {
        VisionCheckpoint::load(&self.base, &self.adapter)
    }
    fn reject(&self, expected: &str) {
        let error = match self.load() {
            Ok(_) => panic!("accepted invalid checkpoint: {expected}"),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            error.contains(expected),
            "expected {expected:?}, got {error}"
        );
    }
    fn config(&self, adapter: bool, key: &str, value: Value) {
        let path = if adapter {
            self.adapter.join("adapter_config.json")
        } else {
            self.base.join("config.json")
        };
        let mut config: Value = serde_json::from_str(if adapter {
            include_str!("fixtures/vision/adapter_config.json")
        } else {
            include_str!("fixtures/vision/config.json")
        })
        .unwrap();
        if adapter {
            config[key] = value;
        } else if key == "text_hidden" {
            config["text_config"]["hidden_size"] = value;
        } else {
            config["vision_config"][key] = value;
        }
        fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }
    fn shards(&self) -> BTreeMap<String, String> {
        let (a, b): (Inventory, Inventory) = inventory(false)
            .into_iter()
            .partition(|(n, _)| n.contains("blocks."));
        sparse(&self.base.join("a.safetensors"), &a);
        sparse(&self.base.join("b.safetensors"), &b);
        let mut map: BTreeMap<String, String> = a
            .keys()
            .map(|n| (n.clone(), "a.safetensors".into()))
            .chain(b.keys().map(|n| (n.clone(), "b.safetensors".into())))
            .collect();
        map.insert(
            "model.language_model.weight".into(),
            "absent-language.safetensors".into(),
        );
        self.index(&map);
        fs::remove_file(self.base.join("model.safetensors")).unwrap();
        map
    }
    fn index(&self, map: &BTreeMap<String, String>) {
        fs::write(
            self.base.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map":map})).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn check(checkpoint: &VisionCheckpoint) {
    assert_eq!(checkpoint.config().depth, 24);
    assert_eq!(checkpoint.config().hidden_size, 1024);
    assert_eq!(checkpoint.adapter().rank, 16);
    assert_eq!(checkpoint.adapter().alpha, 32);
    assert_eq!(checkpoint.adapter().scale(), 2.0);
    for adapter in [false, true] {
        let expected = inventory(adapter);
        let names: Vec<_> = if adapter {
            checkpoint.adapter_names().collect()
        } else {
            checkpoint.base_names().collect()
        };
        assert_eq!(names.len(), if adapter { 100 } else { 297 });
        for (name, spec) in expected {
            let tensor = if adapter {
                checkpoint.adapter_tensor(&name)
            } else {
                checkpoint.base_tensor(&name)
            }
            .unwrap();
            assert_eq!(
                tensor.shape(),
                serde_json::from_value::<Vec<usize>>(spec["shape"].clone()).unwrap()
            );
            assert_eq!(
                tensor.dtype(),
                if adapter { Dtype::F32 } else { Dtype::BF16 }
            );
            assert_eq!(
                &tensor.data()[..if adapter { 4 } else { 2 }],
                if adapter {
                    &[0, 0, 0x80, 0x3f][..]
                } else {
                    &[0x80, 0x3f][..]
                }
            );
        }
    }
    assert!(checkpoint.base_tensor("missing").is_err());
}
#[test]
fn valid_single_matches_independent_inventory() {
    let f = Fixture::new();
    check(&f.load().unwrap());
}
#[test]
fn valid_shards_skip_language_only_files() {
    let f = Fixture::new();
    f.shards();
    check(&f.load().unwrap());
}
#[test]
fn ignores_language_tensors() {
    let f = Fixture::new();
    for adapter in [false, true] {
        let mut inv = inventory(adapter);
        inv.insert(
            "unrelated.language.weight".into(),
            json!({"shape":[1],"dtype":"F32"}),
        );
        sparse(
            &if adapter {
                f.adapter.join("adapter_model.safetensors")
            } else {
                f.base.join("model.safetensors")
            },
            &inv,
        );
    }
    check(&f.load().unwrap());
}
#[test]
fn rejects_missing_wrong_and_extra_tensors() {
    let f = Fixture::new();
    for adapter in [false, true] {
        let path = if adapter {
            f.adapter.join("adapter_model.safetensors")
        } else {
            f.base.join("model.safetensors")
        };
        let original = inventory(adapter);
        let name = original.keys().next().unwrap().clone();
        let mut inv = original.clone();
        inv.remove(&name);
        sparse(&path, &inv);
        f.reject("missing");
        let mut inv = original.clone();
        inv.get_mut(&name).unwrap()["shape"] = json!([1]);
        sparse(&path, &inv);
        f.reject("shape");
        let mut inv = original.clone();
        inv.get_mut(&name).unwrap()["dtype"] = json!(if adapter { "BF16" } else { "F32" });
        sparse(&path, &inv);
        f.reject("dtype");
        let mut inv = original.clone();
        inv.insert(
            format!("{}unexpected.weight", if adapter { ADAPTER } else { BASE }),
            json!({"shape":[1],"dtype":"F32"}),
        );
        sparse(&path, &inv);
        f.reject("unexpected");
        sparse(&path, &original);
    }
}
#[test]
fn rejects_truncated_and_malformed_safetensors() {
    let f = Fixture::new();
    for adapter in [false, true] {
        let path = if adapter {
            f.adapter.join("adapter_model.safetensors")
        } else {
            f.base.join("model.safetensors")
        };
        let len = fs::metadata(&path).unwrap().len();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 1)
            .unwrap();
        f.reject("safetensors");
        fs::write(&path, b"invalid").unwrap();
        f.reject("safetensors");
        sparse(&path, &inventory(adapter));
    }
}
#[test]
fn rejects_index_missing_extra_misrouting_and_duplicates() {
    let f = Fixture::new();
    let original = f.shards();
    let name = inventory(false).keys().next().unwrap().clone();
    let mut map = original.clone();
    map.remove(&name);
    f.index(&map);
    f.reject("missing");
    let mut map = original.clone();
    map.insert(format!("{BASE}unexpected"), "a.safetensors".into());
    f.index(&map);
    f.reject("unexpected");
    let mut map = original.clone();
    map.insert(name.clone(), "b.safetensors".into());
    f.index(&map);
    f.reject("index");
    f.index(&original);
    let mut b: Inventory = inventory(false)
        .into_iter()
        .filter(|(n, _)| !n.contains("blocks."))
        .collect();
    b.insert(name.clone(), inventory(false)[&name].clone());
    sparse(&f.base.join("b.safetensors"), &b);
    f.reject("index");
    fs::write(
        f.base.join("model.safetensors.index.json"),
        format!("{{\"weight_map\":{{\"{name}\":\"a.safetensors\",\"{name}\":\"b.safetensors\"}}}}"),
    )
    .unwrap();
    f.reject("duplicate");
}
#[test]
fn rejects_index_paths_outside_directory() {
    let f = Fixture::new();
    let original = f.shards();
    let name = inventory(false).keys().next().unwrap().clone();
    for path in ["../outside.safetensors", "/tmp/outside.safetensors"] {
        let mut map = original.clone();
        map.insert(name.clone(), path.into());
        f.index(&map);
        f.reject("checkpoint directory");
    }
}
#[cfg(unix)]
#[test]
fn rejects_symlink_escape() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    fs::rename(
        f.base.join("model.safetensors"),
        f.root.join("outside.safetensors"),
    )
    .unwrap();
    symlink(
        f.root.join("outside.safetensors"),
        f.base.join("model.safetensors"),
    )
    .unwrap();
    f.reject("checkpoint directory");
}
#[test]
fn validates_full_vision_configuration() {
    let f = Fixture::new();
    for (key, value) in [
        ("depth", json!(23)),
        ("hidden_size", json!(512)),
        ("intermediate_size", json!(2048)),
        ("num_heads", json!(8)),
        ("num_position_embeddings", json!(1024)),
        ("out_hidden_size", json!(2048)),
        ("in_channels", json!(1)),
        ("patch_size", json!(14)),
        ("temporal_patch_size", json!(1)),
        ("spatial_merge_size", json!(1)),
        ("hidden_act", json!("gelu")),
        ("deepstack_visual_indexes", json!([1])),
        ("text_hidden", json!(2048)),
    ] {
        f.config(false, key, value);
        f.reject("config");
    }
}
#[test]
fn rejects_adapter_math_and_target_changes() {
    let f = Fixture::new();
    for (key, value) in [
        ("r", json!(8)),
        ("lora_alpha", json!(16)),
        ("bias", json!("all")),
        ("lora_bias", json!(true)),
        ("use_dora", json!(true)),
        ("use_rslora", json!(true)),
        ("use_qalora", json!(true)),
        ("fan_in_fan_out", json!(true)),
        ("rank_pattern", json!({"linear_fc1":8})),
        ("alpha_pattern", json!({"linear_fc1":16})),
        ("exclude_modules", json!(["linear_fc1"])),
        ("target_modules", json!(["up_proj", "down_proj"])),
        ("modules_to_save", json!(["merger"])),
        ("layers_to_transform", json!([1])),
        ("target_parameters", json!(["weight"])),
        ("layer_replication", json!([[0, 1]])),
        ("alora_invocation_tokens", json!([1])),
        ("peft_type", json!("IA3")),
    ] {
        f.config(true, key, value);
        f.reject("adapter config");
    }
}
#[test]
fn requires_multimodal_adapter() {
    let f = Fixture::new();
    fs::remove_file(f.adapter.join("adapter_model.safetensors")).unwrap();
    f.reject("adapter_model.safetensors");
    sparse(
        &f.adapter.join("adapter_model.safetensors"),
        &BTreeMap::from([(
            "base_model.model.model.language_model.weight".into(),
            json!({"shape":[1],"dtype":"F32"}),
        )]),
    );
    f.reject("missing");
}

#[test]
fn rejects_duplicate_visual_keys_in_safetensors_header() {
    use std::io::Read;
    let f = Fixture::new();
    let path = f.base.join("model.safetensors");
    let mut file = File::open(&path).unwrap();
    let total = file.metadata().unwrap().len();
    let mut size = [0; 8];
    file.read_exact(&mut size).unwrap();
    let old_len = u64::from_le_bytes(size);
    let mut header = vec![0; old_len as usize];
    file.read_exact(&mut header).unwrap();
    drop(file);
    let entries: BTreeMap<String, Value> = serde_json::from_slice(&header).unwrap();
    let (name, info) = entries.first_key_value().unwrap();
    let mut duplicate = String::from_utf8(header).unwrap();
    duplicate.pop();
    duplicate.push_str(&format!(
        ",{}:{info}}}",
        serde_json::to_string(name).unwrap()
    ));
    let mut file = File::create(&path).unwrap();
    file.write_all(&(duplicate.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(duplicate.as_bytes()).unwrap();
    file.set_len(total - old_len + duplicate.len() as u64)
        .unwrap();
    drop(file);
    f.reject("duplicate");
}

#[test]
fn rejects_overflowing_payload_offsets_without_panicking() {
    let f = Fixture::new();
    let elements = usize::MAX / 8;
    let mut header = serde_json::Map::new();
    for i in 0..8 {
        header.insert(
            format!("unrelated.language.{i}.weight"),
            json!({
                "dtype": "U8", "shape": [elements],
                "data_offsets": [i * elements, (i + 1) * elements]
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = File::create(f.base.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    drop(file);
    f.reject("safetensors");
}
