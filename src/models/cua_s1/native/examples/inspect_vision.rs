use anyhow::{Result, ensure};
use omni_cua_s1_native::vision::VisionCheckpoint;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: inspect_vision BASE_DIR MULTIMODAL_ADAPTER_DIR"
    );
    let checkpoint = VisionCheckpoint::load(&args[0], &args[1])?;
    println!("Vision: {:#?}", checkpoint.config());
    println!("Base: {} BF16 tensors", checkpoint.base_names().count());
    println!(
        "Adapter: {} FP32 tensors ({} pairs), rank {}, alpha {}, scale {}",
        checkpoint.adapter_names().count(),
        checkpoint.adapter_names().count() / 2,
        checkpoint.adapter().rank,
        checkpoint.adapter().alpha,
        checkpoint.adapter().scale()
    );
    Ok(())
}
