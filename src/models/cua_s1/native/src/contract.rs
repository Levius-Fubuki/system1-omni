//! Request mapping, prompt construction and answers for Cua-S1 4B 0.2, ported from
//! `src/models/cua_s1/text/contract.py`, so the two workers build the same prompts and
//! reject the same requests with the same status codes.

use std::fmt::Write as _;

use crate::pyjson::{self, PyStr, Value, dumps, float_repr, repr, repr_str, write_json_str};

pub const MODEL_NAME: &str = "cua-s1-4b-0.2";
pub const ADAPTER_REPO: &str = "cua-ai/cua-s1-4b-0.2";
pub const ADAPTER_REVISION: &str = "16818868b0cc7813808aae4e87b417657046ab79";
pub const BASE_REPO: &str = "Qwen/Qwen3.5-4B";
pub const BASE_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

pub const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const MAX_OPTIONS: usize = 26;

// The system message, the user message layout and the fixed values below are copied
// from trycua/cua at 0e75660ce4c2edda519e0c795fa3ad98abf4e76f:
// `libs/cua-s1/python/src/cua_s1/four_b.py` (SYSTEM_PROMPT, build_prompt,
// _describe_option) and `libs/cua-driver/examples/jev-use/python/decision_models.py`
// (S1DecisionModel.score).
//
// MIT License
//
// Copyright (c) 2025 Cua AI, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
pub const SYSTEM_PROMPT: &str = "You are a one-pass computer-use decision model. You are shown the \
current state of a screen and a fixed, closed list of candidate \
(element, action) options, each given a single letter. Choose exactly \
one option: the single best next action to take. Answer with ONLY that \
option's letter -- no words, no punctuation, no explanation.";
pub const APP: &str = "Cua Driver";
pub const TASK_FAMILY: &str = "closed-candidate decision";
pub const ROLE: &str = "Decision";
pub const ACTION: &str = "select";

/// A request the worker rejects, with the HTTP status to return.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestError {
    pub status: u16,
    pub message: String,
}

impl RequestError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn unprocessable(message: impl Into<String>) -> Self {
        Self::new(422, message)
    }
}

impl From<pyjson::JsonError> for RequestError {
    fn from(e: pyjson::JsonError) -> Self {
        RequestError::new(400, e.message())
    }
}

/// One `choice` question mapped onto the prompt fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub name: String,
    pub goal: String,
    pub keys: Vec<String>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub state: String,
    pub questions: Vec<Question>,
}

pub fn parse_body(raw: &[u8]) -> Result<Vec<(PyStr, Value)>, RequestError> {
    Ok(pyjson::parse(raw)?)
}

fn text(s: &PyStr) -> &str {
    s.as_str().expect("parse rejects lone surrogates")
}

/// `state` or `instructions` as prompt text: a string as is, anything else as
/// `json.dumps(value, ensure_ascii=False)`.
fn as_text(value: &Value) -> String {
    match value {
        Value::Str(s) => text(s).to_string(),
        other => dumps(other),
    }
}

/// An option label escaped the way upstream's chooser does:
/// `json.dumps(value, ensure_ascii=False)[1:-1]`.
fn escape_label(value: &str) -> String {
    let mut out = String::new();
    write_json_str(value, &mut out);
    out[1..out.len() - 1].to_string()
}

fn check_json_value(value: &Value, place: &str, allow_null: bool) -> Result<(), RequestError> {
    match value {
        Value::Null if !allow_null => Err(RequestError::unprocessable(format!(
            "{place} must not be null"
        ))),
        Value::Bool(_) | Value::Int(_) | Value::Float(_) => Err(RequestError::unprocessable(
            format!("{place} must be a string, an object or an array"),
        )),
        _ => Ok(()),
    }
}

fn get<'a>(pairs: &'a [(PyStr, Value)], key: &str) -> Option<&'a Value> {
    pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Validate a `/v1/systemone` body and map it onto prompt fields.
pub fn map_request(body: &[(PyStr, Value)], max_questions: usize) -> Result<Request, RequestError> {
    match get(body, "model") {
        Some(Value::Str(s)) if s == MODEL_NAME => {}
        _ => {
            return Err(RequestError::unprocessable(format!(
                "'model' must be {}",
                repr_str(&PyStr::new(MODEL_NAME))
            )));
        }
    }

    let state_value =
        get(body, "state").ok_or_else(|| RequestError::unprocessable("'state' is required"))?;
    check_json_value(state_value, "'state'", false)?;
    let empty = match state_value {
        Value::Str(s) => s == "",
        Value::Object(p) => p.is_empty(),
        Value::Array(a) => a.is_empty(),
        _ => false,
    };
    if empty {
        return Err(RequestError::unprocessable("'state' must not be empty"));
    }
    let state = as_text(state_value);

    let questions = match get(body, "questions") {
        Some(Value::Object(q)) if !q.is_empty() => q,
        _ => {
            return Err(RequestError::unprocessable(
                "'questions' must be a non-empty object",
            ));
        }
    };
    if questions.len() > max_questions {
        return Err(RequestError::new(
            413,
            format!("too many questions ({} > {max_questions})", questions.len()),
        ));
    }

    // Every question type is checked before the per-question checks, so a `score`
    // or `noul` question anywhere rejects the whole request with that reason.
    for (name, question) in questions {
        let Value::Object(q) = question else {
            return Err(RequestError::unprocessable(format!(
                "question {} must be an object",
                repr_str(name)
            )));
        };
        let missing = Value::Null;
        let kind = get(q, "type").unwrap_or(&missing);
        match kind {
            Value::Str(s) if s == "score" || s == "noul" => {
                return Err(RequestError::unprocessable(format!(
                    "question {}: type {} is not supported; Cua-S1 4B 0.2 answers 'choice' questions only",
                    repr_str(name),
                    repr(kind)
                )));
            }
            Value::Str(s) if s == "choice" => {}
            _ => {
                return Err(RequestError::unprocessable(format!(
                    "question {}: unknown type {}",
                    repr_str(name),
                    repr(kind)
                )));
            }
        }
    }

    let mut mapped = Vec::with_capacity(questions.len());
    for (name, question) in questions {
        let Value::Object(q) = question else {
            unreachable!("checked above")
        };
        let place = format!("question {}", repr_str(name));
        let instructions = get(q, "instructions").ok_or_else(|| {
            RequestError::unprocessable(format!("{place}: 'instructions' is required"))
        })?;
        check_json_value(instructions, &format!("{place}: 'instructions'"), true)?;
        let goal = match instructions {
            Value::Null => String::new(),
            other => as_text(other),
        };

        let criteria = match get(q, "criteria") {
            Some(Value::Object(c)) => c,
            _ => {
                return Err(RequestError::unprocessable(format!(
                    "{place}: 'criteria' must be an object"
                )));
            }
        };
        if criteria.is_empty() {
            return Err(RequestError::unprocessable(format!(
                "{place}: 'criteria' must have at least one option"
            )));
        }
        if criteria.len() > MAX_OPTIONS {
            return Err(RequestError::unprocessable(format!(
                "{place}: {} options; at most {MAX_OPTIONS} are supported",
                criteria.len()
            )));
        }
        let mut keys = Vec::with_capacity(criteria.len());
        let mut labels = Vec::with_capacity(criteria.len());
        for (key, value) in criteria {
            check_json_value(value, &format!("{place}: option {}", repr_str(key)), true)?;
            let label = match value {
                Value::Null => text(key).to_string(),
                other => as_text(other),
            };
            keys.push(text(key).to_string());
            labels.push(escape_label(&label));
        }
        mapped.push(Question {
            name: text(name).to_string(),
            goal,
            keys,
            labels,
        });
    }
    Ok(Request {
        state,
        questions: mapped,
    })
}

/// The user message for one question, matching upstream `build_prompt` (text).
pub fn user_message(state: &str, question: &Question) -> String {
    let mut user = String::new();
    if !question.goal.is_empty() {
        write!(user, "Goal: {}\n\n", question.goal).unwrap();
    }
    write!(user, "App: {APP}\nTask family: {TASK_FAMILY}\n\n").unwrap();
    write!(user, "Accessibility tree:\n{state}\n\n").unwrap();
    user.push_str("Options:\n");
    for (i, (letter, label)) in LETTERS.chars().zip(&question.labels).enumerate() {
        if i > 0 {
            user.push('\n');
        }
        write!(user, "{letter}. {ROLE} \"{label}\" -> {ACTION}").unwrap();
    }
    user.push_str("\n\nAnswer with a single letter.");
    user
}

/// The prompt text the Qwen3.5 chat template renders for the system and user messages
/// with `add_generation_prompt=True` (thinking left on). The template trims message
/// content, which changes nothing here: the system prompt is fixed, and the user
/// message starts with "Goal: " or "App: " and ends with "letter.".
pub fn chat_text(state: &str, question: &Question) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        user_message(state, question)
    )
}

/// CPython 3.12's `sum()` over floats (Neumaier compensated summation).
fn py_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut total, mut c) = (0.0f64, 0.0f64);
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    total
}

/// Normalized entropy, `1 - H(p) / ln(n)`, as the LAYA worker reports it.
pub fn confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n < 2 {
        return 1.0;
    }
    let entropy = -py_sum(probabilities.iter().map(|&p| p * p.clamp(1e-12, 1.0).ln()));
    (1.0 - entropy / (n as f64).ln()).clamp(0.0, 1.0)
}

/// One choice answer, already serialized the way the Python worker's response is
/// (`json.dumps(..., ensure_ascii=False, separators=(",", ":"))`). Ties go to the
/// earliest option.
pub fn answer_json(question: &Question, probabilities: &[f32]) -> Result<String, String> {
    let p: Vec<f64> = probabilities.iter().map(|&x| x as f64).collect();
    if p.len() != question.keys.len() || !p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x))
    {
        return Err(format!("model returned invalid probabilities: {p:?}"));
    }
    let total = py_sum(p.iter().copied());
    // math.isclose(total, 1.0, abs_tol=1e-5)
    if (total - 1.0).abs() > f64::max(1e-9 * total.abs().max(1.0), 1e-5) {
        return Err(format!("model probabilities do not sum to one: {p:?}"));
    }
    let mut best = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > p[best] {
            best = i;
        }
    }
    let mut out = String::from("{\"type\":\"choice\",\"choice\":");
    write_json_str(&question.keys[best], &mut out);
    out.push_str(",\"probabilities\":{");
    for (i, (key, x)) in question.keys.iter().zip(&p).enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_str(key, &mut out);
        out.push(':');
        out.push_str(&float_repr(*x));
    }
    out.push_str("},\"confidence\":");
    out.push_str(&float_repr(confidence(&p)));
    out.push('}');
    Ok(out)
}

pub fn model_identity(revision: &str) -> String {
    format!("{ADAPTER_REPO}@{revision}:text")
}

/// `{"detail": message}` as the Python worker serializes it.
pub fn detail_json(message: &str) -> String {
    let mut out = String::from("{\"detail\":");
    write_json_str(message, &mut out);
    out.push('}');
    out
}

pub const WARMUP_BODY: &str = r#"{"model": "cua-s1-4b-0.2", "state": "Dialog: 'Update installed.' Button: OK", "questions": {"warmup": {"type": "choice", "instructions": "Close the dialog.", "criteria": {"ok": "Click OK", "wait": "Wait"}}}}"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn map(body: &str) -> Result<Request, RequestError> {
        map_request(&parse_body(body.as_bytes())?, 64)
    }

    fn detail(body: &str) -> (u16, String) {
        let e = map(body).unwrap_err();
        (e.status, e.message)
    }

    const OK: &str = r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"q": {"type": "choice", "instructions": "go", "criteria": {"a": "A", "b": null}}}}"#;

    #[test]
    fn maps_a_request() {
        let r = map(OK).unwrap();
        assert_eq!(r.state, "S");
        assert_eq!(r.questions[0].keys, ["a", "b"]);
        assert_eq!(r.questions[0].labels, ["A", "b"]);
        let text = chat_text(&r.state, &r.questions[0]);
        assert!(text.ends_with(
            "Options:\nA. Decision \"A\" -> select\nB. Decision \"b\" -> select\n\nAnswer with a single letter.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        ));
        assert!(text.contains("<|im_start|>user\nGoal: go\n\nApp: Cua Driver\n"));
    }

    #[test]
    fn errors_match_python() {
        let cases: &[(&str, u16, &str)] = &[
            (r#"{"state": "S"}"#, 422, "'model' must be 'cua-s1-4b-0.2'"),
            (r#"{"model": "cua-s1-4b-0.2"}"#, 422, "'state' is required"),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": null}"#,
                422,
                "'state' must not be null",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": 1.5}"#,
                422,
                "'state' must be a string, an object or an array",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": {}}"#,
                422,
                "'state' must not be empty",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": []}"#,
                422,
                "'questions' must be a non-empty object",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"q": {"type": "choice"}, "r": {"type": "noul"}}}"#,
                422,
                "question 'r': type 'noul' is not supported; Cua-S1 4B 0.2 answers 'choice' questions only",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"it's": {"type": [1, {"a": null}]}}}"#,
                422,
                "question \"it's\": unknown type [1, {'a': None}]",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"q": {"type": "choice"}}}"#,
                422,
                "question 'q': 'instructions' is required",
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"q": {"type": "choice", "instructions": null, "criteria": {"a": true}}}}"#,
                422,
                "question 'q': option 'a' must be a string, an object or an array",
            ),
        ];
        for (body, status, message) in cases {
            assert_eq!(detail(body), (*status, message.to_string()), "{body}");
        }
    }

    #[test]
    fn confidence_matches_python() {
        // values from the Python worker
        let p = [0.00247262348420918f64, 0.9975274205207825];
        assert_eq!(float_repr(confidence(&p)), "0.9750249548256825");
    }
}
