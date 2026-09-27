"""Contract tests that need neither weights nor torch.

PYTHONPATH=src python -m pytest tests/cua_s1
"""

import json
import math
from pathlib import Path

import pytest

from models.cua_s1.text.contract import (
    RequestError,
    answer,
    build_messages,
    confidence,
    map_request,
    parse_body,
)

INPUTS = json.loads(
    (Path(__file__).parent / "data" / "text_inputs.json").read_text(encoding="utf-8")
)

# The user message upstream's chooser builds for
# libs/cua-driver/examples/jev-use/fixtures/jev-choice-request-v1.json at the
# pinned revision (FourBModel text modality).
FIXTURE_POSITIVE_USER = (
    "Goal: Submit the verified form.\n\n"
    "App: Cua Driver\nTask family: closed-candidate decision\n\n"
    "Accessibility tree:\n"
    'Visual-region-derived observation for capture "capture-fixture-1":\n'
    "\"submit-text\": text 'Submit' at (300,240,100,40) confidence=0.96 interactive=true\n\n"
    "Options:\n"
    'A. Decision "Submit using the unique validated visual region." -> select\n'
    'B. Decision "Discard this decision set and obtain a fresh observation." -> select\n'
    'C. Decision "Stop without acting if no supplied action is safe." -> select\n\n'
    "Answer with a single letter."
)


def mapped(name):
    return map_request(parse_body(json.dumps(INPUTS[name]).encode()))


def reject(body, status=422):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    with pytest.raises(RequestError) as info:
        map_request(parse_body(raw))
    assert info.value.status == status
    return str(info.value)


def base(**question):
    q = {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {"a": "A", "b": "B"},
    }
    q.update(question)
    return {"model": "cua-s1-4b-0.2", "state": "Screen", "questions": {"q": q}}


def test_fixture_prompt_matches_upstream():
    request = mapped("fixture_positive")
    messages = build_messages(request.state, request.questions[0])
    assert messages[1] == {"role": "user", "content": FIXTURE_POSITIVE_USER}
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(
        "You are a one-pass computer-use decision model."
    )


def test_every_input_maps():
    for name in INPUTS:
        request = mapped(name)
        assert request.questions
        for question in request.questions:
            assert 1 <= len(question.keys) <= 26


def test_goal_line_left_out_when_empty_or_null():
    request = mapped("no_goal")
    for question in request.questions:
        user = build_messages(request.state, question)[1]["content"]
        assert user.startswith("App: Cua Driver\n")


def test_structured_values_and_null_label():
    request = mapped("structured")
    state = INPUTS["structured"]["state"]
    assert request.state == json.dumps(state, ensure_ascii=False)
    question = request.questions[0]
    assert question.goal.startswith('{"question": "Which action turns off `target`?"')
    assert question.labels[0] == '{\\"action\\": \\"click\\", \\"element\\": \\"e1\\"}'
    assert question.labels[1] == '[\\"click\\", \\"e2\\"]'
    assert question.labels[3] == "abstain"


def test_label_escaping_matches_chooser():
    question = mapped("escaping").questions[0]
    assert question.labels[0] == 'Click \\"Submit\\"\\n(keeps the path)'
    assert question.labels[1] == "Click 'Clear'\\tthen retype C:\\\\Users"
    assert mapped("non_ascii").questions[0].labels[0] == "点击「保存」"


def test_score_or_noul_rejects_the_whole_request():
    body = base()
    body["questions"]["s"] = {
        "type": "score",
        "instructions": "Rate it.",
        "criteria": ["low", "high"],
    }
    assert "'score' is not supported" in reject(body)
    body = base()
    body["questions"]["n"] = {"type": "noul", "instructions": "Is it red?"}
    assert "'noul' is not supported" in reject(body)


def test_option_count_limits():
    assert "at least one option" in reject(base(criteria={}))
    many = {f"o{i}": f"Option {i}" for i in range(27)}
    assert "27 options" in reject(base(criteria=many))
    assert (
        len(
            map_request(base(criteria={f"o{i}": "x" for i in range(26)}))
            .questions[0]
            .keys
        )
        == 26
    )


def test_duplicate_keys_anywhere():
    raw = (
        b'{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"q": {"type": "choice",'
        b' "instructions": "I", "criteria": {"a": "A", "a": "B"}}}}'
    )
    assert "duplicate key 'a'" in reject(raw, status=400)
    raw = (
        b'{"model": "cua-s1-4b-0.2", "state": {"x": 1, "x": 2}, "questions": {"q": {"type":'
        b' "choice", "instructions": "I", "criteria": {"a": "A"}}}}'
    )
    assert "duplicate key 'x'" in reject(raw, status=400)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"model": "cua-s1-4b-0.2", "state": NaN}',
        b'{"model": "cua-s1-4b-0.2", "state": {"x": 1e400}}',
        b'{"model": "cua-s1-4b-0.2", "state": {"x": ' + b"9" * 5000 + b"}}",
        b'{"model": "cua-s1-4b-0.2", "state": "\\ud800"}',
        b"[" * 100000 + b"]" * 100000,
        b"\xff\xfe",
        '{"model": "cua-s1-4b-0.2", "state": "S"}'.encode("utf-16"),
        b"\xef\xbb\xbf" + b'{"model": "cua-s1-4b-0.2", "state": "S"}',
    ],
)
def test_malformed_bodies_are_400(raw):
    reject(raw, status=400)


def test_question_shape_errors():
    body = base()
    body["questions"]["q"] = "not an object"
    assert "must be an object" in reject(body)
    assert "'criteria' must be an object" in reject(base(criteria=["a", "b"]))
    body = base()
    del body["questions"]["q"]["instructions"]
    assert "'instructions' is required" in reject(body)
    assert "unknown type 'rank'" in reject(base(type="rank"))
    body = base()
    body["questions"] = {f"q{i}": body["questions"]["q"] for i in range(3)}
    with pytest.raises(RequestError) as info:
        map_request(body, max_questions=2)
    assert info.value.status == 413


@pytest.mark.parametrize("value", [1, 2.5, True, False])
def test_number_or_boolean_criteria_value(value):
    assert "must be a string, an object or an array" in reject(
        base(criteria={"a": value, "b": "B"})
    )


@pytest.mark.parametrize("state", ["", {}, [], None, 3, True])
def test_bad_state(state):
    body = base()
    body["state"] = state
    reject(body)


def test_model_name_and_body_shape():
    body = base()
    body["model"] = "english"
    assert "'model' must be" in reject(body)
    reject(b"not json", status=400)
    reject(b"[1, 2]", status=400)


def test_confidence_is_normalized_entropy():
    assert confidence([1.0]) == 1.0
    assert confidence([0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)
    p = [0.88, 0.12, 0.0]
    h = -(0.88 * math.log(0.88) + 0.12 * math.log(0.12))
    assert confidence(p) == pytest.approx(1 - h / math.log(3))


def test_answer_shape_and_ties():
    question = mapped("two_options").questions[0]
    result = answer(question, [0.5, 0.5])
    assert result["choice"] == "delete"
    assert result["type"] == "choice"
    assert list(result["probabilities"]) == ["delete", "cancel"]
    assert answer(question, [0.2, 0.8])["choice"] == "cancel"
