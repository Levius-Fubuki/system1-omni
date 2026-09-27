"""Request mapping, prompt construction and answers for Cua-S1 4B 0.2.

This module follows the contract in `src/models/cua_s1/README.md`. It has no
torch or Transformers imports, so it can be tested without weights.
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass
from typing import Any

MODEL_NAME = "cua-s1-4b-0.2"
ADAPTER_REPO = "cua-ai/cua-s1-4b-0.2"
ADAPTER_REVISION = "16818868b0cc7813808aae4e87b417657046ab79"
BASE_REPO = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

LETTERS = string.ascii_uppercase
MAX_OPTIONS = len(LETTERS)

# The system message, the user message layout and the fixed values below are
# copied from trycua/cua at 0e75660ce4c2edda519e0c795fa3ad98abf4e76f:
# `libs/cua-s1/python/src/cua_s1/four_b.py` (SYSTEM_PROMPT, build_prompt,
# _describe_option) and `libs/cua-driver/examples/jev-use/python/
# decision_models.py` (S1DecisionModel.score). MIT License, Copyright (c) 2025
# Cua AI, Inc.; see THIRD_PARTY_NOTICES.md.
SYSTEM_PROMPT = (
    "You are a one-pass computer-use decision model. You are shown the "
    "current state of a screen and a fixed, closed list of candidate "
    "(element, action) options, each given a single letter. Choose exactly "
    "one option: the single best next action to take. Answer with ONLY that "
    "option's letter -- no words, no punctuation, no explanation."
)
APP = "Cua Driver"
TASK_FAMILY = "closed-candidate decision"
ROLE = "Decision"
ACTION = "select"


class RequestError(ValueError):
    """A request the worker rejects. `status` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise RequestError(f"duplicate key {key!r} in a JSON object", status=400)
        obj[key] = value
    return obj


def _reject_constant(name: str) -> Any:
    raise RequestError(f"{name} is not valid JSON", status=400)


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise RequestError(f"number {text} is out of range", status=400)
    return value


def _check_unicode(value: Any) -> None:
    """Reject lone surrogates (for example a `\\ud800` escape): they cannot be
    encoded as UTF-8, so they cannot be tokenized or echoed back."""
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, dict):
        for key, item in value.items():
            key.encode("utf-8")
            _check_unicode(item)
    elif isinstance(value, list):
        for item in value:
            _check_unicode(item)


def parse_body(raw: bytes) -> dict[str, Any]:
    """Decode a request body, keeping key order and rejecting duplicate keys."""
    try:
        text = raw.decode("utf-8")
        body = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        _check_unicode(body)
    except RequestError:
        raise
    except RecursionError as error:
        raise RequestError("request body is nested too deeply", status=400) from error
    except UnicodeError as error:
        raise RequestError(
            "request body must be valid UTF-8 text", status=400
        ) from error
    except ValueError as error:
        # json.JSONDecodeError, a UTF-8 byte order mark, or an integer too long
        # for Python to convert.
        raise RequestError("request body must be valid JSON", status=400) from error
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object", status=400)
    return body


def as_text(value: Any) -> str:
    """Render `state` or `instructions` as prompt text.

    A string is used as is; an object or array is serialized the way Python's
    `json.dumps(value, ensure_ascii=False)` does.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def escape_label(value: str) -> str:
    """Escape an option label the way upstream's chooser does."""
    return json.dumps(value, ensure_ascii=False)[1:-1]


@dataclass(frozen=True)
class Question:
    """One `choice` question mapped onto the prompt fields."""

    name: str
    goal: str
    keys: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Request:
    state: str
    questions: tuple[Question, ...]


def _check_json_value(value: Any, where: str, allow_null: bool) -> None:
    if value is None:
        if not allow_null:
            raise RequestError(f"{where} must not be null")
        return
    if isinstance(value, bool) or isinstance(value, (int, float)):
        raise RequestError(f"{where} must be a string, an object or an array")
    if not isinstance(value, (str, dict, list)):
        raise RequestError(f"{where} must be a string, an object or an array")


def map_request(body: dict[str, Any], *, max_questions: int = 64) -> Request:
    """Validate a `/v1/systemone` body and map it onto prompt fields."""
    model = body.get("model")
    if model != MODEL_NAME:
        raise RequestError(f"'model' must be {MODEL_NAME!r}")

    if "state" not in body:
        raise RequestError("'state' is required")
    state_value = body["state"]
    _check_json_value(state_value, "'state'", allow_null=False)
    if state_value in ("", {}, []):
        raise RequestError("'state' must not be empty")
    state = as_text(state_value)

    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("'questions' must be a non-empty object")
    if len(questions) > max_questions:
        raise RequestError(
            f"too many questions ({len(questions)} > {max_questions})", status=413
        )

    # Check every question type before the per-question checks, so a `score`
    # or `noul` question anywhere rejects the whole request with that reason.
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise RequestError(f"question {name!r} must be an object")
        kind = question.get("type")
        if kind in ("score", "noul"):
            raise RequestError(
                f"question {name!r}: type {kind!r} is not supported; "
                "Cua-S1 4B 0.2 answers 'choice' questions only"
            )
        if kind != "choice":
            raise RequestError(f"question {name!r}: unknown type {kind!r}")

    mapped = []
    for name, question in questions.items():
        where = f"question {name!r}"
        if "instructions" not in question:
            raise RequestError(f"{where}: 'instructions' is required")
        instructions = question["instructions"]
        _check_json_value(instructions, f"{where}: 'instructions'", allow_null=True)
        goal = "" if instructions is None else as_text(instructions)

        criteria = question.get("criteria")
        if not isinstance(criteria, dict):
            raise RequestError(f"{where}: 'criteria' must be an object")
        if not criteria:
            raise RequestError(f"{where}: 'criteria' must have at least one option")
        if len(criteria) > MAX_OPTIONS:
            raise RequestError(
                f"{where}: {len(criteria)} options; at most {MAX_OPTIONS} are supported"
            )
        keys, labels = [], []
        for key, value in criteria.items():
            _check_json_value(value, f"{where}: option {key!r}", allow_null=True)
            if value is None:
                text = key
            else:
                text = as_text(value)
            keys.append(key)
            labels.append(escape_label(text))
        mapped.append(
            Question(name=name, goal=goal, keys=tuple(keys), labels=tuple(labels))
        )
    return Request(state=state, questions=tuple(mapped))


def build_messages(state: str, question: Question) -> list[dict[str, str]]:
    """Chat messages for one question, matching upstream `build_prompt` (text)."""
    option_lines = "\n".join(
        f'{letter}. {ROLE} "{label}" -> {ACTION}'
        for letter, label in zip(LETTERS, question.labels, strict=False)
    )
    user = (
        (f"Goal: {question.goal}\n\n" if question.goal else "")
        + f"App: {APP}\nTask family: {TASK_FAMILY}\n\n"
        + f"Accessibility tree:\n{state}\n\n"
        + f"Options:\n{option_lines}\n\nAnswer with a single letter."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def confidence(probabilities: list[float]) -> float:
    """Normalized entropy, `1 - H(p) / ln(n)`, as the LAYA worker reports it."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(min(max(p, 1e-12), 1.0)) for p in probabilities)
    return min(max(1.0 - entropy / math.log(n), 0.0), 1.0)


def answer(question: Question, probabilities: list[float]) -> dict[str, Any]:
    """The Jev choice answer. Ties go to the earliest option."""
    if len(probabilities) != len(question.keys) or not all(
        math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities
    ):
        raise ValueError(f"model returned invalid probabilities: {probabilities}")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError(f"model probabilities do not sum to one: {probabilities}")
    best = 0
    for index, p in enumerate(probabilities):
        if p > probabilities[best]:
            best = index
    return {
        "type": "choice",
        "choice": question.keys[best],
        "probabilities": dict(zip(question.keys, probabilities, strict=True)),
        "confidence": confidence(probabilities),
    }


def model_identity(revision: str = ADAPTER_REVISION, modality: str = "text") -> str:
    return f"{ADAPTER_REPO}@{revision}:{modality}"
