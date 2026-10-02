//! Replay full screenshot requests and independently compare all CPU boundaries.
use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_cua_s1_native::{contract, image_request::parse_image_body, vision_engine::VisionEngine};
use safetensors::SafeTensors;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 5,
        "usage: native_vision BASE ADAPTER LANGUAGE CONTROL_DIR OUTPUT_JSON"
    );
    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").context("CUA_S1_CUDA_LIB")?);
    let bundle = Path::new(&args[3]);
    let out = Path::new(&args[4]);
    ensure!(!out.exists(), "output already exists");
    let manifest: Value = serde_json::from_slice(&std::fs::read(bundle.join("manifest.json"))?)?;
    ensure!(
        manifest["schema"] == "cua-s1-native-vision-controls-v1",
        "unsupported control schema"
    );
    let mut engine = VisionEngine::load(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        &library,
    )?;
    let mut records = Vec::new();
    for case in manifest["cases"].as_array().context("cases")? {
        let name = case["case"].as_str().context("case name")?;
        let read = |field: &str| -> Result<Vec<u8>> {
            let p = Path::new(case[field].as_str().context("file name")?);
            ensure!(
                p.components()
                    .all(|c| matches!(c, std::path::Component::Normal(_))),
                "invalid relative path"
            );
            Ok(std::fs::read(bundle.join(p))?)
        };
        let body =
            contract::parse_body(&read("request_file")?).map_err(|e| anyhow::anyhow!(e.message))?;
        let request = parse_image_body(&body)?;
        let reference_rgb = read("rgb_file")?;
        ensure!(
            request.rgb.len() == reference_rgb.len(),
            "decoded RGB shape mismatch"
        );
        let decode_max_abs = request
            .rgb
            .iter()
            .zip(&reference_rgb)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        let prepared = engine.prepare(
            request.width,
            request.height,
            &request.rgb,
            &request.questions,
        )?;
        let grid: Vec<usize> = serde_json::from_value(case["grid"].clone())?;
        ensure!(
            prepared.image.image_grid_thw.as_slice() == grid,
            "grid mismatch"
        );
        let start = std::time::Instant::now();
        let features = engine.vision.forward(&prepared.image)?;
        let vision_ms = start.elapsed().as_secs_f64() * 1000.;
        let reference = std::fs::read(bundle.join(format!("{name}.bfloat16.safetensors")))?;
        let st = SafeTensors::deserialize(&reference)?;
        let f = st.tensor("image_features")?;
        ensure!(
            f.dtype() == safetensors::Dtype::BF16 && f.data().len() == features.len() * 2,
            "feature shape/dtype mismatch"
        );
        let reference_features: Vec<f32> = f
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b).to_f32())
            .collect();
        let errors: Vec<f64> = features
            .iter()
            .zip(&reference_features)
            .map(|(a, &b)| a.to_f64() - b as f64)
            .collect();
        let feature_max_abs = errors.iter().copied().map(f64::abs).fold(0., f64::max);
        let feature_rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
        let mut questions = Vec::new();
        for ((q, prompt), control) in request
            .questions
            .iter()
            .zip(&prepared.prompts)
            .zip(case["questions"].as_array().context("questions")?)
        {
            let ids: Vec<u32> = serde_json::from_value(control["input_ids"].clone())?;
            let positions: [Vec<i64>; 3] = serde_json::from_value(control["position_ids"].clone())?;
            ensure!(
                prompt.token_ids == ids,
                "{name}: tokenization differs from reference"
            );
            ensure!(
                prompt.position_ids == positions,
                "{name}: positions differ from reference"
            );
            let score = engine.score(prompt, &features, q.keys.len())?;
            let repeated = engine.score(prompt, &features, q.keys.len())?;
            ensure!(
                score.hidden == repeated.hidden,
                "repeated native language execution changed"
            );
            questions.push(
                json!({"name":q.name, "sequence":ids.len(), "token_ids_equal":true,
                "position_ids_equal":true, "repeat_equal":true, "probabilities":score.probabilities,
                "logits":score.logits, "hidden":score.hidden}),
            );
        }
        // Replay the same image after LM calls to expose cross-model scratch corruption.
        ensure!(
            features == engine.vision.forward(&prepared.image)?,
            "vision replay changed features"
        );
        records.push(json!({"case":name,"decode_max_abs":decode_max_abs,"grid":grid,
            "image_tokens":features.len()/2560,"feature_max_abs":feature_max_abs,"feature_rmse":feature_rmse,
            "vision_ms_first_call":vision_ms,"questions":questions}));
        eprintln!("completed {name}: feature max {feature_max_abs}, RMS {feature_rmse}");
    }
    std::fs::write(
        out,
        serde_json::to_vec_pretty(
            &json!({"schema":"cua-s1-native-vision-results-v1","cases":records}),
        )?,
    )?;
    Ok(())
}
