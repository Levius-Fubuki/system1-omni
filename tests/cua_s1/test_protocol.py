import base64
import io
import json

import pytest
from PIL import Image

from models.cua_s1.multimodal.protocol import (
    InvalidRequest,
    answer,
    build_messages,
    decode_request,
    parse_request,
)


def image_url(fmt="PNG", size=(32, 32)):
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, format=fmt)
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(out.getvalue()).decode()


def request():
    return {
        "model": "cua-s1-4b-0.2",
        "state": {"image": image_url()},
        "questions": {
            "next": {
                "type": "choice",
                "instructions": "Submit the form",
                "criteria": {"submit": "Submit", "cancel": "Cancel"},
            }
        },
    }


def test_image_and_order_are_preserved():
    r = parse_request(request())
    assert r.image.mode == "RGB" and r.image.size == (32, 32)
    assert r.questions[0].keys == ("submit", "cancel")
    assert r.questions[0].labels == ("Submit", "Cancel")


def test_prompt_keeps_image_block_and_upstream_text():
    q = parse_request(request()).questions[0]
    msg = build_messages(q)
    assert msg[1]["content"][0]["type"] == "image"
    assert msg[1]["content"][1]["text"] == (
        "Goal: Submit the form\n\nApp: Cua Driver\nTask family: closed-candidate decision\n\n"
        "The current screenshot is attached.\n\nOptions:\n"
        'A. Decision "Submit" -> select\nB. Decision "Cancel" -> select\n\n'
        "Answer with a single letter."
    )


def test_structured_values_and_escaping():
    r = request()
    r["questions"]["next"]["instructions"] = {"目标": "提交"}
    r["questions"]["next"]["criteria"] = {"fallback": None, "obj": {"x": '"\n'}}
    q = parse_request(r).questions[0]
    assert q.goal == '{"目标": "提交"}'
    assert q.labels == (
        "fallback",
        json.dumps(json.dumps({"x": '"\n'}, ensure_ascii=False), ensure_ascii=False)[
            1:-1
        ],
    )


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"image": "/etc/passwd"},
        {"image": "https://example.com/a.png"},
        {"image": "data:image/png;base64,!!"},
        {"image": image_url(), "text": "ignored"},
    ],
)
def test_invalid_images_and_unknown_state_fields(state):
    r = request()
    r["state"] = state
    with pytest.raises(InvalidRequest):
        parse_request(r)


def test_mime_mismatch_and_oversized_dimensions():
    for url in [
        image_url().replace("image/png", "image/jpeg"),
        image_url(size=(2049, 1)),
    ]:
        r = request()
        r["state"]["image"] = url
        with pytest.raises(InvalidRequest):
            parse_request(r)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("size", [(2048, 1), (1, 2048), (201, 1), (1, 201)])
def test_unsupported_image_aspect_ratio_is_rejected(fmt, size):
    r = request()
    r["state"]["image"] = image_url(fmt, size)
    with pytest.raises(InvalidRequest, match="aspect ratio"):
        parse_request(r)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("size", [(200, 1), (1, 200), (199, 1), (1, 199)])
def test_supported_image_aspect_ratio_boundary_is_preserved(fmt, size):
    r = request()
    r["state"]["image"] = image_url(fmt, size)
    assert parse_request(r).image.size == size


@pytest.mark.parametrize(
    "criteria", [{}, {str(i): "x" for i in range(27)}, {"a": 1}, {"a": True}]
)
def test_invalid_candidates(criteria):
    r = request()
    r["questions"]["next"]["criteria"] = criteria
    with pytest.raises(InvalidRequest):
        parse_request(r)


@pytest.mark.parametrize("kind", ["score", "noul"])
def test_unsupported_question_rejects_entire_request(kind):
    r = request()
    r["questions"]["bad"] = {"type": kind, "instructions": "x"}
    with pytest.raises(InvalidRequest):
        parse_request(r)


def test_duplicate_keys_and_nonfinite_json():
    for raw in [
        b'{"model":1,"model":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"[]",
        b"not json",
    ]:
        with pytest.raises(InvalidRequest):
            decode_request(raw)


def test_entropy_confidence_and_earliest_tie():
    q = parse_request(request()).questions[0]
    result = answer(q, [0.5, 0.5])
    assert result["choice"] == "submit"
    assert result["confidence"] == 0.0
    assert result["probabilities"] == {"submit": 0.5, "cancel": 0.5}
    r = request()
    r["questions"]["next"]["criteria"] = {"only": "Only"}
    assert answer(parse_request(r).questions[0], [1.0])["confidence"] == 1.0


def test_jpeg_supported():
    r = request()
    r["state"]["image"] = image_url("JPEG")
    assert parse_request(r).image.mode == "RGB"


@pytest.mark.parametrize(
    "token", ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
)
@pytest.mark.parametrize("field", ["instructions", "criteria"])
def test_media_control_tokens_are_rejected(token, field):
    value = request()
    value["questions"]["next"][field] = (
        token if field == "instructions" else {"a": {"text": token}}
    )
    with pytest.raises(InvalidRequest, match="control token"):
        parse_request(value)


@pytest.mark.parametrize("instructions", ["", None])
def test_explicit_empty_or_null_instructions_omits_goal(instructions):
    value = request()
    value["questions"]["next"]["instructions"] = instructions
    question = parse_request(value).questions[0]
    assert question.goal == ""
    assert "Goal:" not in build_messages(question)[1]["content"][1]["text"]
