//! Cua-S1 4B 0.2 (`text` adapter) `/v1/systemone` worker on native CUDA kernels.
//!
//!     omni-cua-s1-native --model <merged text checkpoint> [--port 8000]
//!
//! See recipe/cua_s1/native.md for building the CUDA library and exporting the merged
//! checkpoint.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use clap::builder::RangedU64ValueParser;

use omni_cua_s1_native::contract;
use omni_cua_s1_native::cuda;
use omni_cua_s1_native::engine::Engine;
use omni_cua_s1_native::model::Options;
use omni_cua_s1_native::server::{self, App, Limits};

#[derive(Parser)]
#[command(about = "Cua-S1 4B 0.2 text worker on native CUDA kernels")]
struct Args {
    /// Merged text checkpoint: Qwen/Qwen3.5-4B with the `text` adapter merged, plus
    /// the cua_s1_export.json that recipe/cua_s1/export_text_merged.py writes.
    #[arg(long, env = "CUA_S1_MODEL")]
    model: PathBuf,
    /// libqwen3_5_cuda.so, built by src/backends/cuda/qwen3_5/build.sh [default: next
    /// to this executable].
    #[arg(long, env = "CUA_S1_CUDA_LIB")]
    cuda_lib: Option<PathBuf>,
    #[arg(long, env = "CUA_S1_HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(long, env = "CUA_S1_PORT", default_value_t = 8000)]
    port: u16,
    #[arg(long, env = "CUA_S1_MAX_BODY_BYTES", default_value_t = 4 << 20)]
    max_body_bytes: usize,
    #[arg(long, env = "CUA_S1_MAX_QUESTIONS", default_value_t = 64)]
    max_questions: usize,
    /// Per question; 0 disables the check.
    #[arg(long, env = "CUA_S1_MAX_PROMPT_TOKENS", default_value_t = 16384)]
    max_prompt_tokens: usize,
    /// Prompts up to this many tokens run as a CUDA graph captured for their length
    /// on first use; longer ones run eagerly. 0 runs everything eagerly, with
    /// cuBLASLt's first-choice GEMM algorithms and no tuning.
    #[arg(long, env = "CUA_S1_GRAPH_MAX_TOKENS", default_value_t = 2048)]
    graph_max_tokens: usize,
    /// How many prompt lengths keep their captured graph.
    #[arg(long, env = "CUA_S1_GRAPH_CACHE", default_value_t = 128,
          value_parser = RangedU64ValueParser::<usize>::new().range(1..))]
    graph_cache: usize,
    /// GEMM algorithm choices: read from this file if it exists, else tuned at
    /// startup and written to it, so later starts make the same choices. A file tuned
    /// on another GPU or cuBLASLt version, or for another --graph-max-tokens, is
    /// refused.
    #[arg(long, env = "CUA_S1_GEMM_PLANS")]
    gemm_plans: Option<PathBuf>,
    /// When tuning, time far more cuBLASLt configurations for prompts up to
    /// --graph-max-tokens (each algorithm with its tiles, stage counts, swizzles and
    /// several split-K factors) instead of the heuristic's shortlist. Takes about a
    /// minute; use it with --gemm-plans so that it runs once.
    #[arg(long, env = "CUA_S1_GEMM_SEARCH")]
    gemm_search: bool,
}

impl Args {
    fn options(&self) -> Result<Options> {
        ensure!(
            self.graph_max_tokens > 0 || (self.gemm_plans.is_none() && !self.gemm_search),
            "--gemm-plans and --gemm-search need --graph-max-tokens above 0"
        );
        Ok(Options {
            library: match &self.cuda_lib {
                Some(path) => path.clone(),
                None => cuda::default_library()?,
            },
            graph_max_tokens: self.graph_max_tokens,
            graph_cache: self.graph_cache,
            gemm_plans: self.gemm_plans.clone(),
            gemm_search: self.gemm_search,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let engine = Engine::load(&args.model, &args.options()?).await?;
    if engine.provenance.adapter_revision != contract::ADAPTER_REVISION {
        eprintln!(
            "warning: adapter revision {} is not the pinned {}",
            engine.provenance.adapter_revision,
            contract::ADAPTER_REVISION
        );
    }
    if engine.provenance.base_revision != contract::BASE_REVISION {
        eprintln!(
            "warning: base revision {} is not the pinned {}",
            engine.provenance.base_revision,
            contract::BASE_REVISION
        );
    }
    println!(
        "loaded in {:.1} s on {} (bfloat16, graphs up to {} tokens)",
        engine.load_seconds,
        engine.device,
        engine.graph_max_tokens()
    );
    let api_key = std::env::var_os("CUA_S1_API_KEY").map(|k| k.into_encoded_bytes());
    let limits = Limits {
        max_body_bytes: args.max_body_bytes,
        max_questions: args.max_questions,
        max_prompt_tokens: args.max_prompt_tokens,
    };
    let revision = engine.provenance.adapter_revision.clone();
    let app = Arc::new(App::new(engine, limits, api_key, &revision));
    let started = Instant::now();
    server::warmup(&app).await?;
    println!("warmed up in {:.1} s", started.elapsed().as_secs_f64());
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    println!("listening on {}:{}", args.host, args.port);
    axum::serve(listener, server::router(app)).await?;
    Ok(())
}
