//! HTTP routes, matching the Python worker: `GET /health` and `POST /v1/systemone`,
//! one decision at a time, with the same status codes and response format.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use http_body_util::BodyExt;

use crate::contract::{self, Request, RequestError, detail_json, map_request, parse_body};
use crate::engine::Engine;
use crate::pyjson::{PyStr, repr_str, write_json_str};

pub struct Limits {
    pub max_body_bytes: usize,
    pub max_questions: usize,
    /// per question; 0 disables the check
    pub max_prompt_tokens: usize,
}

pub struct App {
    pub engine: Engine,
    pub limits: Limits,
    /// `Bearer <key>` as raw bytes, when a key is set
    expected_auth: Option<Vec<u8>>,
    pub identity: String,
    /// held for the whole decision, so forward passes run one at a time
    turn: tokio::sync::Mutex<()>,
}

impl App {
    /// `api_key` is the raw value of `CUA_S1_API_KEY`; empty means no key.
    pub fn new(engine: Engine, limits: Limits, api_key: Option<Vec<u8>>, revision: &str) -> Self {
        let expected_auth = api_key
            .filter(|k| !k.is_empty())
            .map(|k| [b"Bearer ".as_slice(), &k].concat());
        Self {
            engine,
            limits,
            expected_auth,
            identity: contract::model_identity(revision),
            turn: tokio::sync::Mutex::new(()),
        }
    }
}

fn json_response(status: StatusCode, body: String) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

fn error(status: u16, message: &str) -> Response {
    json_response(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        detail_json(message),
    )
}

enum DecideError {
    Request(RequestError),
    Internal(anyhow::Error),
}

/// Score each question and build the response body. Every prompt is tokenized and
/// checked against the prompt limit before any forward pass runs.
async fn decide(app: &App, request: &Request) -> Result<String, DecideError> {
    let limit = app.limits.max_prompt_tokens;
    let mut encoded = Vec::with_capacity(request.questions.len());
    for question in &request.questions {
        let ids = app
            .engine
            .prompter
            .encode(&request.state, question)
            .map_err(DecideError::Internal)?;
        if limit > 0 && ids.len() > limit {
            return Err(DecideError::Request(RequestError::new(
                413,
                format!(
                    "question {}: prompt is {} tokens, over the {limit}-token limit",
                    repr_str(&PyStr::new(&question.name)),
                    ids.len()
                ),
            )));
        }
        encoded.push(ids);
    }
    let mut answers = String::from("{");
    let mut prompt_tokens = 0;
    for (i, (question, ids)) in request.questions.iter().zip(encoded).enumerate() {
        prompt_tokens += ids.len();
        let probs = app
            .engine
            .score(ids, question.keys.len())
            .await
            .map_err(DecideError::Internal)?;
        let answer = contract::answer_json(question, &probs)
            .map_err(|e| DecideError::Internal(anyhow::anyhow!(e)))?;
        if i > 0 {
            answers.push(',');
        }
        write_json_str(&question.name, &mut answers);
        answers.push(':');
        answers.push_str(&answer);
    }
    answers.push('}');
    let mut out = String::from("{\"model\":");
    write_json_str(&app.identity, &mut out);
    out.push_str(",\"answers\":");
    out.push_str(&answers);
    out.push_str(&format!(
        ",\"usage\":{{\"input_tokens\":{prompt_tokens},\"output_tokens\":0}}}}"
    ));
    Ok(out)
}

/// The Python worker reads the header as Latin-1 text (Starlette) and encodes it back
/// as UTF-8 before `hmac.compare_digest`; the same bytes are compared here, so both
/// workers accept and reject the same headers.
fn authorized(app: &App, headers: &HeaderMap) -> bool {
    let Some(expected) = &app.expected_auth else {
        return true;
    };
    let raw = headers
        .get(header::AUTHORIZATION)
        .map(|v| v.as_bytes())
        .unwrap_or(b"");
    let mut supplied = Vec::with_capacity(raw.len());
    for &b in raw {
        if b < 0x80 {
            supplied.push(b);
        } else {
            supplied.extend_from_slice(&[0xc0 | (b >> 6), 0x80 | (b & 0x3f)]);
        }
    }
    supplied.len() == expected.len()
        && supplied
            .iter()
            .zip(expected)
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

async fn health(State(app): State<Arc<App>>) -> Response {
    let mut out = String::from("{\"status\":\"ready\",\"modality\":\"text\",\"model\":");
    write_json_str(&app.identity, &mut out);
    out.push_str(",\"device\":");
    write_json_str(&app.engine.device, &mut out);
    out.push_str(",\"dtype\":\"bfloat16\",\"mode\":\"native\"}");
    json_response(StatusCode::OK, out)
}

async fn systemone(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    if !authorized(&app, &headers) {
        return error(401, "invalid or missing bearer token");
    }
    let max = app.limits.max_body_bytes;
    if let Some(len) = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        && !len.is_empty()
        && len.bytes().all(|b| b.is_ascii_digit())
        && len.parse::<u128>().map_or(true, |n| n > max as u128)
    {
        return error(413, "request body too large");
    }
    let mut raw = Vec::new();
    let mut body = body;
    while let Some(frame) = body.frame().await {
        let Ok(frame) = frame else {
            return error(400, "request body could not be read");
        };
        if let Some(chunk) = frame.data_ref() {
            raw.extend_from_slice(chunk);
            if raw.len() > max {
                return error(413, "request body too large");
            }
        }
    }
    let request = match parse_body(&raw).and_then(|b| map_request(&b, app.limits.max_questions)) {
        Ok(r) => r,
        Err(e) => return error(e.status, &e.message),
    };
    let _turn = app.turn.lock().await;
    match decide(&app, &request).await {
        Ok(body) => json_response(StatusCode::OK, body),
        Err(DecideError::Request(e)) => error(e.status, &e.message),
        Err(DecideError::Internal(e)) => {
            eprintln!("inference failed: {e:#}");
            error(500, "inference failed")
        }
    }
}

async fn not_found() -> Response {
    error(404, "Not Found")
}

async fn method_not_allowed() -> Response {
    error(405, "Method Not Allowed")
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/systemone", post(systemone))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .with_state(app)
}

/// One decision through the whole request path, before the server listens.
pub async fn warmup(app: &App) -> anyhow::Result<()> {
    let request = map_request(
        &parse_body(contract::WARMUP_BODY.as_bytes()).map_err(|e| anyhow::anyhow!(e.message))?,
        64,
    )
    .map_err(|e| anyhow::anyhow!(e.message))?;
    let _turn = app.turn.lock().await;
    match decide(app, &request).await {
        Ok(_) => Ok(()),
        Err(DecideError::Request(e)) => anyhow::bail!(e.message),
        Err(DecideError::Internal(e)) => Err(e),
    }
}
