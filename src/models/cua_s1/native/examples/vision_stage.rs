//! GPU parity stage runner. Raw RGB8 input; emits BF16 stage files and shapes JSON.
use anyhow::{Result, ensure};
use omni_cua_s1_native::{image_preprocess::preprocess_rgb8, vision::VisionModel};
use std::{env, fs, path::Path};
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    ensure!(
        args.len() == 8,
        "usage: vision_stage BASE ADAPTER LIB RGB WIDTH HEIGHT OUTDIR"
    );
    let image = preprocess_rgb8(args[5].parse()?, args[6].parse()?, &fs::read(&args[4])?)?;
    fs::create_dir_all(&args[7])?;
    let mut model = VisionModel::load(&args[1], &args[2], Path::new(&args[3]))?;
    let start = std::time::Instant::now();
    let output = model.forward_with_trace(&image, |name, values| {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        fs::write(Path::new(&args[7]).join(format!("{name}.bf16")), bytes)?;
        Ok(())
    })?;
    println!(
        "{}",
        serde_json::json!({"grid":image.image_grid_thw,"output_shape":[image.image_tokens(),2560],"values":output.len(),"elapsed_seconds":start.elapsed().as_secs_f64()})
    );
    Ok(())
}
