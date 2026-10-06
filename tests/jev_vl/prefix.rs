//! R2d cache-structure tests: key stability, LRU/budget accounting, and the
//! manifest-faithful suffix-split equivalence (bit-exact vs full expansion on
//! the FROZEN R1 same-image prompts — the L1 hit path's correctness contract).

use std::path::PathBuf;
use std::sync::Arc;

use omni_jev_vl_native::caches::{CacheCfg, Caches, structure_key};
use omni_jev_vl_native::contract::{self, Part};
use omni_jev_vl_native::images::{ImageAsset, expand};

fn asset(grid: [i64; 3]) -> Arc<ImageAsset> {
    let [t, h, w] = grid;
    let n = (t * h * w / 4) as usize;
    Arc::new(ImageAsset {
        grid_thw: grid,
        embeddings: vec![half::bf16::ONE; n * 5120],
    })
}

#[test]
fn structure_key_is_sensitive_and_stable() {
    let parts_a = vec![Part::Text("state text".into()), Part::Image("u://a".into())];
    let parts_b = vec![Part::Text("state text".into()), Part::Image("u://b".into())];
    let parts_c = vec![Part::Image("u://a".into()), Part::Text("state text".into())];
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("score", &parts_a)
    );
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_b)
    );
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_c)
    );
    assert_eq!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_a)
    );
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_a[..1])
    );
}

fn hub(f: impl FnOnce(&mut CacheCfg)) -> Arc<Caches> {
    let mut cfg = CacheCfg {
        enabled: true,
        l1: true,
        l2: true,
        l3: true,
        l1_max: 64,
        l2_bytes: 1 << 20,
        l3_bytes: 1 << 20,
    };
    f(&mut cfg);
    Caches::new(cfg)
}

#[test]
fn l2_lru_hits_misses_and_evicition() {
    let c = hub(|cfg| cfg.l2_bytes = (2 * 5120 * 2 + 64) * 2 + 128);
    let (a, b, d) = (asset([1, 2, 4]), asset([1, 2, 4]), asset([1, 2, 4]));
    assert!(c.l2_get("k-a").is_none());
    c.l2_insert("k-a".into(), a.clone());
    assert!(c.l2_get("k-a").is_some());
    c.l2_insert("k-b".into(), b);
    c.l2_insert("k-d".into(), d); // a evicted: budget fits two
    assert!(c.l2_get("k-a").is_none());
    assert!(c.l2_get("k-b").is_some());
    let s = c.snapshot();
    assert_eq!(s.l2_hit, 2);
    assert_eq!(s.l2_miss, 2);
    assert_eq!(s.l2_records, 2);
    // Disabled: no reuse, all misses counted.
    let off = hub(|cfg| cfg.enabled = false);
    off.l2_insert("k-a".into(), asset([1, 2, 4]));
    assert!(off.l2_get("k-a").is_none());
    assert_eq!(off.snapshot().l2_miss, 1);
}

#[test]
fn records_are_lru_and_budgeted() {
    let c = hub(|cfg| cfg.l1_max = 2);
    let meta = |pads: usize| omni_jev_vl_native::caches::L1Meta {
        pads_start: pads,
        pads_end: pads + 96,
        p: 64,
        base_pad: pads as i64,
        advance: 12,
        asset_key: "k".into(),
        ids_prefix: vec![0u32; 64],
        positions_prefix: [vec![0i64; 64], vec![0i64; 64], vec![0i64; 64]],
    };
    let r1 = c.record_insert(1, meta(10));
    c.record_insert(2, meta(20));
    assert!(c.record_get(1).is_some());
    assert!(Arc::ptr_eq(&c.record_get(1).unwrap(), &r1));
    c.record_insert(3, meta(30)); // l1_max=2: one eviction of the oldest (2)
    assert!(c.record_get(2).is_none());
    assert!(c.record_get(1).is_some());
    let s = c.snapshot();
    assert!(s.l1_hit >= 3 && s.l1_miss >= 1 && s.l1_records <= 2);
}

/// The manifest-faithful equivalence: for every img entry of the FROZEN R1
/// manifest the suffix split (pads-tail + vision_end + fresh tail tokenization)
/// must equal `expand` on the whole prompt bit for bit — that's the L1 hit path.
#[test]
#[ignore = "needs JEV_VL_EXPORT (export dir with tokenizer + manifest)"]
fn manifest_suffix_split_matches_full_expand() {
    let dir = PathBuf::from(std::env::var_os("JEV_VL_EXPORT").expect("set JEV_VL_EXPORT"));
    let manifest_dir =
        PathBuf::from(std::env::var_os("JEV_VL_MANIFEST").expect("set JEV_VL_MANIFEST"));
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("jev_vl_export.json")).unwrap()).unwrap();
    let labels: Vec<String> = serde_json::from_value(manifest["labels"].clone()).unwrap();
    let image_pad = tokenizer.token_to_id("<|image_pad|>").unwrap();
    let vision_end = tokenizer.token_to_id("<|vision_end|>").unwrap();
    let mut checked = 0;
    for line in std::fs::read_to_string(manifest_dir).unwrap().lines() {
        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        if !entry["id"].as_str().unwrap_or_default().starts_with("img-") {
            continue;
        }
        let raw = serde_json::to_vec(&entry["request"]).unwrap();
        let compiled = contract::compile(&raw, &labels).unwrap();
        let tail = contract::tail_after_last_image(&compiled, &labels).unwrap();
        let grid = [1, 60, 60];
        let e = expand(
            &tokenizer
                .encode(compiled.prompt.as_str(), false)
                .unwrap()
                .get_ids()
                .to_vec(),
            image_pad,
            &[asset(grid)],
        )
        .unwrap();
        assert_eq!(e.blocks.len(), 1, "{}", entry["id"]);
        let p = e.blocks[0].end / 64 * 64;
        // Rebuild as the L1 hit path does.
        let b = &e.blocks[0];
        let suffix_pads = b.end - p;
        let tail_ids: Vec<u32> = tokenizer
            .encode(tail.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut ids: Vec<u32> = std::iter::repeat_n(image_pad, suffix_pads).collect();
        ids.push(vision_end);
        ids.extend_from_slice(&tail_ids);
        assert_eq!(&ids[..], &e.ids[p..], "suffix ids diverge: {}", entry["id"]);
        let pos =
            omni_jev_vl_native::images::meshgrid_positions(grid, b.base, p - b.start, suffix_pads);
        for a in 0..3 {
            assert_eq!(
                &pos[a][..],
                &e.positions[a][p..p + suffix_pads],
                "suffix meshgrid diverges: {}",
                entry["id"]
            );
        }
        assert_eq!(
            e.positions[0][p + suffix_pads],
            b.base + b.advance,
            "after-image base: {}",
            entry["id"]
        );
        // The structure key is shared across the 12 same-image entries per kind.
        checked += 1;
    }
    assert_eq!(checked, 12, "the frozen manifest carries 12 img entries");
}
