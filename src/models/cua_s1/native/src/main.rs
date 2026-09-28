//! Cua-S1 4B 0.2 (`text` adapter) `/v1/systemone` worker on native CUDA kernels.
//!
//!     omni-cua-s1-native --model <merged text checkpoint> [--port 8000]
//!
//! See recipe/cua_s1/native.md for building the CUDA library, exporting the merged
//! checkpoint and checking the worker.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use clap::builder::RangedU64ValueParser;

use omni_cua_s1_native::contract::{self, map_request, parse_body};
use omni_cua_s1_native::cuda;
use omni_cua_s1_native::engine::{self, Engine, Mode, Options, Prompter};
use omni_cua_s1_native::server::{self, App, DecideError, Limits};

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
    /// Read request bodies on stdin (one JSON string per line), print the prompt
    /// token ids or the rejection for each, and exit. Loads only the tokenizer.
    #[arg(long)]
    encode_only: bool,
    /// Score every question of the request bodies in this JSON file (name -> body),
    /// eagerly and as served, print one JSON line per run, and exit.
    #[arg(long)]
    score_all: Option<PathBuf>,
    /// Time the forward pass over the token ids in each of these JSON files (lists
    /// of integers), without HTTP, and exit.
    #[arg(long, num_args = 1..)]
    bench: Vec<PathBuf>,
    #[arg(long, default_value_t = 50, value_parser = RangedU64ValueParser::<usize>::new().range(1..))]
    bench_repeat: usize,
    /// Idle time before each timed forward pass, as between separate requests.
    #[arg(long, default_value_t = 0)]
    bench_gap_ms: u64,
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

fn encode_only(args: &Args) -> Result<()> {
    let prompter = Prompter::load(&args.model)?;
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let body: String = serde_json::from_str(&line?)?;
        let result = parse_body(body.as_bytes())
            .and_then(|b| map_request(&b, args.max_questions))
            .map_err(DecideError::Request)
            .and_then(|r| {
                let ids = server::encode_all(&prompter, &r, args.max_prompt_tokens)?;
                Ok((r, ids))
            });
        let record = match result {
            Ok((request, ids)) => serde_json::json!({
                "status": 200,
                "questions": request.questions.iter().map(|q| &q.name).collect::<Vec<_>>(),
                "ids": ids,
            }),
            Err(DecideError::Request(e)) => {
                serde_json::json!({"status": e.status, "detail": e.message})
            }
            Err(DecideError::Internal(e)) => return Err(e),
        };
        writeln!(out, "{record}")?;
    }
    Ok(())
}

async fn score_all(args: &Args, inputs: &Path) -> Result<()> {
    let cases: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(inputs)?)?;
    let engine = Engine::load(&args.model, &args.options()?).await?;
    let graph_max = engine.graph_max_tokens();
    let mut out = std::io::stdout().lock();
    for (case, body) in &cases {
        let raw = serde_json::to_vec(body)?;
        let request = parse_body(&raw)
            .and_then(|b| map_request(&b, args.max_questions))
            .map_err(|e| anyhow::anyhow!("{case}: {}", e.message))?;
        for question in &request.questions {
            let ids = engine.prompter.encode(&request.state, question)?;
            for (mode, label) in [(Mode::Eager, "eager"), (Mode::Auto, "served")] {
                let probs = engine
                    .score_mode(ids.clone(), question.keys.len(), mode)
                    .await?;
                let probabilities: serde_json::Map<String, serde_json::Value> = question
                    .keys
                    .iter()
                    .zip(&probs)
                    .map(|(k, p)| (k.clone(), serde_json::json!(p)))
                    .collect();
                writeln!(
                    out,
                    "{}",
                    serde_json::json!({
                        "case": case,
                        "question": question.name,
                        "tokens": ids.len(),
                        "mode": label,
                        "graph": mode == Mode::Auto && ids.len() <= graph_max,
                        "probabilities": probabilities,
                    })
                )?;
            }
        }
    }
    Ok(())
}

async fn bench(args: &Args) -> Result<()> {
    let engine = Engine::load(&args.model, &args.options()?).await?;
    println!(
        "loaded in {:.1} s, graphs up to {} tokens",
        engine.load_seconds,
        engine.graph_max_tokens()
    );
    for file in &args.bench {
        let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(file)?)?;
        let mut times = Vec::with_capacity(args.bench_repeat);
        for i in 0..args.bench_repeat + 3 {
            std::thread::sleep(std::time::Duration::from_millis(args.bench_gap_ms));
            let started = Instant::now();
            engine.last_hidden_blocking(&ids)?;
            if i >= 3 {
                times.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(f64::total_cmp);
        let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        println!(
            "{}: {} tokens, forward p50 {:.2} ms p95 {:.2} ms min {:.2} ms",
            file.display(),
            ids.len(),
            at(0.5),
            at(0.95),
            times[0]
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.encode_only {
        return encode_only(&args);
    }
    if let Some(inputs) = &args.score_all {
        return score_all(&args, inputs).await;
    }
    if !args.bench.is_empty() {
        return bench(&args).await;
    }
    let provenance = engine::provenance(&args.model)?;
    if provenance.adapter_revision != contract::ADAPTER_REVISION {
        eprintln!(
            "warning: adapter revision {} is not the pinned {}",
            provenance.adapter_revision,
            contract::ADAPTER_REVISION
        );
    }
    if provenance.base_revision != contract::BASE_REVISION {
        eprintln!(
            "warning: base revision {} is not the pinned {}",
            provenance.base_revision,
            contract::BASE_REVISION
        );
    }
    let engine = Engine::load(&args.model, &args.options()?).await?;
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
    let app = Arc::new(App::new(
        engine,
        limits,
        api_key,
        &provenance.adapter_revision,
    ));
    let started = Instant::now();
    server::warmup(&app).await?;
    println!("warmed up in {:.1} s", started.elapsed().as_secs_f64());
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    println!("listening on {}:{}", args.host, args.port);
    axum::serve(listener, server::router(app)).await?;
    Ok(())
}
