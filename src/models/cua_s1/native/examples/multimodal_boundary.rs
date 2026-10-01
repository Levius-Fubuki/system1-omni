//! Replay #53's exported language boundary through the native model.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_cua_s1_native::{inputs::MultimodalInput, model::Model};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};

fn integers(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<i64>> {
    let v = st.tensor(name)?;
    ensure!(
        v.dtype() == Dtype::I64 && v.shape() == shape,
        "{name}: expected I64 {shape:?}"
    );
    Ok(v.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect())
}

fn features(st: &SafeTensors<'_>, shape: &[usize]) -> Result<Vec<bf16>> {
    let v = st.tensor("image_features")?;
    ensure!(
        v.dtype() == Dtype::BF16 && v.shape() == shape,
        "image_features: expected BF16 {shape:?}"
    );
    Ok(v.data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect())
}

fn embedding_file(dir: &Path) -> Result<(PathBuf, String)> {
    let names = [
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ];
    if dir.join("model.safetensors.index.json").exists() {
        let index: Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        for name in names {
            if let Some(file) = index["weight_map"][name].as_str() {
                return Ok((dir.join(file), name.into()));
            }
        }
    } else {
        let path = dir.join("model.safetensors");
        let bytes = std::fs::read(&path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        for name in names {
            if st.tensor(name).is_ok() {
                return Ok((path, name.into()));
            }
        }
    }
    anyhow::bail!("missing embedding weight")
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: multimodal_boundary MODEL_DIR REFERENCE_BUNDLE OUTPUT_JSON"
    );
    let (dir, bundle, out) = (
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    );
    ensure!(!out.exists(), "output already exists");
    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").context("CUA_S1_CUDA_LIB")?);
    let manifest: Value = serde_json::from_slice(&std::fs::read(bundle.join("manifest.json"))?)?;
    ensure!(
        manifest["schema"] == "cua-s1-multimodal-reference-v1",
        "unsupported reference schema"
    );
    let mut model = Model::load(dir, &library)?;
    let (file, name) = embedding_file(dir)?;
    let file = std::fs::File::open(file)?;
    // SAFETY: the checkpoint is immutable while the example runs.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let weights = SafeTensors::deserialize(&map)?;
    let embed = weights.tensor(&name)?;
    ensure!(
        embed.dtype() == Dtype::BF16
            && embed.shape().len() == 2
            && embed.shape()[1] == model.cfg.hidden,
        "embedding shape/dtype"
    );
    let mut rows = Vec::new();
    for entry in manifest["questions"].as_array().context("questions")? {
        let relative = Path::new(entry["tensors_file"].as_str().context("tensors_file")?);
        ensure!(
            relative
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe tensor path"
        );
        let bytes = std::fs::read(bundle.join(relative))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let ids = st.tensor("input_ids")?;
        ensure!(
            ids.shape().len() == 2 && ids.shape()[0] == 1,
            "expected batch one"
        );
        let t = ids.shape()[1];
        let ids: Vec<u32> = integers(&st, "input_ids", &[1, t])?
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()?;
        let indices = st.tensor("image_token_indices")?;
        ensure!(
            indices.shape().len() == 1,
            "image indices must be one-dimensional"
        );
        let count = indices.shape()[0];
        let indices: Vec<usize> = integers(&st, "image_token_indices", &[count])?
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<_, _>>()?;
        let features = features(&st, &[count, model.cfg.hidden])?;
        let positions = integers(&st, "position_ids", &[3, 1, t])?;
        let input = MultimodalInput {
            token_ids: &ids,
            image_token_indices: &indices,
            image_embeddings: &features,
            position_ids: [&positions[..t], &positions[t..2 * t], &positions[2 * t..]],
        };
        let last = model.forward_multimodal(&input)?;
        ensure!(
            last.iter().all(|x| x.is_finite()),
            "non-finite hidden state"
        );
        // Repeating the same boundary in the same model checks buffer reuse.
        ensure!(
            last == model.forward_multimodal(&input)?,
            "repeat changed native hidden state"
        );
        let n = entry["option_keys"]
            .as_array()
            .context("option_keys")?
            .len();
        ensure!((1..=26).contains(&n), "candidate count");
        let candidates = integers(&st, "candidate_token_ids", &[n])?;
        let mut logits = Vec::new();
        for id in candidates {
            let id = usize::try_from(id)?;
            ensure!(id < embed.shape()[0], "candidate outside vocabulary");
            let row = &embed.data()[id * last.len() * 2..(id + 1) * last.len() * 2];
            let dot: f64 = row
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&last)
                .map(|(b, &h)| bf16::from_le_bytes(*b).to_f64() * h as f64)
                .sum();
            logits.push(dot as f32);
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probabilities: Vec<f32> = exps.iter().map(|e| (e / total) as f32).collect();
        ensure!(
            probabilities.iter().all(|x| x.is_finite()),
            "non-finite readout"
        );
        rows.push(json!({"case": entry["case"], "question": entry["question"], "sequence": t, "image_tokens": count, "last_hidden_state": last, "candidate_logits": logits, "probabilities": probabilities, "repeat_equal": true}));
    }
    std::fs::write(
        out,
        serde_json::to_vec_pretty(
            &json!({"schema": "cua-s1-native-language-boundary-v1", "questions": rows}),
        )?,
    )?;
    Ok(())
}
