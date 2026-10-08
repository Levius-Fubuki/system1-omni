//! Existing worker HTTP transport and diagnostic JSONL interface.
use crate::{executor::Executor, processing::Processor};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_qwen3_5_native::cuda;
use omni_runtime::SerialScheduler;
use serde_json::{Value, json};
use std::{
    io::{BufRead, Write},
    path::Path,
    sync::Arc,
};
const WARMUP:&[u8]=br#"{"state":"Update installed.","questions":{"q":{"instructions":"Choose the action.","criteria":{"close":"Close dialog","wait":"Wait"}}}}"#;
pub struct Engine {
    pub processor: Processor,
    pub executor: Executor,
    pub scheduler: SerialScheduler,
}
fn manifest(dir: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(
        &std::fs::read(dir.join("jemm_export.json"))
            .context("export JEMM first; see recipe/jemm/native.md")?,
    )?)
}
impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let export = manifest(dir)?;
        let processor = Processor::load(dir, &export)?;
        let executor = Executor::load(dir, library, &export).await?;
        let engine = Self {
            processor,
            executor,
            scheduler: SerialScheduler::default(),
        };
        let prepared = engine.processor.prepare(WARMUP)?;
        let rows = engine
            .executor
            .execute(&engine.scheduler, prepared.inputs)
            .await?;
        prepared.context.finish(rows)?;
        ensure!(engine.executor.available(), "warmup retired the model");
        Ok(engine)
    }
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error":message.to_string()}))).into_response()
}
pub async fn decide(engine: &Engine, raw: &[u8]) -> Response {
    if !engine.executor.available() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "model is unavailable");
    }
    let prepared = match engine.processor.prepare(raw) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    match engine
        .executor
        .execute(&engine.scheduler, prepared.inputs)
        .await
        .and_then(|rows| prepared.context.finish(rows))
    {
        Ok(body) => Json(body).into_response(),
        Err(e) => {
            eprintln!("inference failed: {e:#}");
            error(StatusCode::SERVICE_UNAVAILABLE, "model inference failed")
        }
    }
}
async fn systemone(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();
    if !content_type.eq_ignore_ascii_case("application/json") {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        );
    }
    match body {
        Ok(raw) => decide(&engine, &raw).await,
        Err(e) => error(e.status(), e.body_text()),
    }
}
pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|State(engine): State<Arc<Engine>>| async move {
                if engine.executor.available() {
                    Json(json!({"status":"ready","model":"JEMM"})).into_response()
                } else {
                    error(StatusCode::SERVICE_UNAVAILABLE, "model is unavailable")
                }
            }),
        )
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(crate::contract::MAX_BODY_BYTES))
        .with_state(engine)
}
pub async fn run() -> Result<()> {
    let dir = std::env::var_os("JEMM_MODEL").context("set JEMM_MODEL to the native export")?;
    let dir = Path::new(&dir);
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.is_empty() || args == ["--prepare-jsonl"] || args == ["--diagnostic-jsonl"],
        "usage: omni-jemm-native [--prepare-jsonl|--diagnostic-jsonl]"
    );
    if args == ["--prepare-jsonl"] {
        let processor = Processor::load(dir, &manifest(dir)?)?;
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let output = match processor.prepare_diagnostic(line.as_bytes()) {
                Ok(p) => p.diagnostics,
                Err(e) => json!({"error":e.to_string()}),
            };
            println!("{output}");
            std::io::stdout().flush()?;
        }
        return Ok(());
    }
    let library = std::env::var_os("JEMM_CUDA_LIB")
        .map(Into::into)
        .map_or_else(cuda::default_library, Ok)?;
    let engine = Arc::new(Engine::load(dir, &library).await?);
    if args == ["--diagnostic-jsonl"] {
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let output = async {
                let p = engine.processor.prepare_diagnostic(line.as_bytes())?;
                let rows = engine.executor.execute(&engine.scheduler, p.inputs).await?;
                let response = p.context.finish(rows.clone())?;
                Ok::<_, anyhow::Error>(
                    json!({"response":response,"logits":rows,"diagnostics":p.diagnostics}),
                )
            }
            .await
            .unwrap_or_else(|e| json!({"error":e.to_string()}));
            println!("{output}");
            std::io::stdout().flush()?;
        }
        return Ok(());
    }
    let host = std::env::var("JEMM_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("JEMM_PORT")
        .map_or(Ok(8000), |v| v.parse())
        .context("JEMM_PORT")?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    eprintln!("JEMM ready on {host}:{port}");
    axum::serve(listener, router(engine)).await?;
    Ok(())
}
