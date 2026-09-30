"""HTTP tests for the worker with a fake model: no weights, no torch."""

import json
from dataclasses import dataclass

import pytest

# The worker's own requirements include fastapi and httpx; skip where only the
# contract tests' dependencies are installed.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from frontend.cua_s1_text import build_app  # noqa: E402


@dataclass
class _Ids:
    shape: tuple


class FakeModel:
    device = "cpu"
    dtype = "float32"

    def __init__(self, tokens=100, fail=False, nan=False):
        self.tokens = tokens
        self.fail = fail
        self.nan = nan
        self.forward_calls = 0

    def encode(self, state, question):
        return {"input_ids": _Ids(shape=(1, self.tokens))}

    def score_encoded(self, inputs, n_options):
        self.forward_calls += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")

        @dataclass
        class Scored:
            probabilities: list
            prompt_tokens: int

        probabilities = [0.1] * n_options
        probabilities[-1] = 1.0 - 0.1 * (n_options - 1)
        if self.nan:
            probabilities[0] = float("nan")
        return Scored(probabilities, inputs["input_ids"].shape[1])


def client(model=None, api_key=None, max_body_bytes=4 << 20, max_prompt_tokens=32768):
    app = build_app(
        model or FakeModel(),
        api_key=api_key,
        max_body_bytes=max_body_bytes,
        max_questions=64,
        max_prompt_tokens=max_prompt_tokens,
        revision="r",
    )
    return TestClient(app)


BODY = {
    "model": "cua-s1-4b-0.2",
    "state": "Screen",
    "questions": {
        "q": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"a": "A", "b": "B"},
        }
    },
}


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "cua-ai/cua-s1-4b-0.2@r:text"
    assert response.json()["status"] == "ready"
    assert response.json()["modality"] == "text"


def test_choice_answer():
    response = client().post("/v1/systemone", json=BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answers"]["q"]["type"] == "choice"
    assert body["answers"]["q"]["choice"] == "b"
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 0}


def test_keys_come_back_as_sent():
    body = json.loads(json.dumps(BODY))
    body["questions"] = {
        "_sample": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"_save": "Save", "b": "B"},
        }
    }
    response = client().post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    assert list(answers) == ["_sample"]
    assert list(answers["_sample"]["probabilities"]) == ["_save", "b"]


def test_chunked_upload():
    raw = json.dumps(BODY).encode()
    response = client().post(
        "/v1/systemone",
        content=iter([raw[:10], raw[10:]]),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200, response.text


def test_errors():
    c = client()
    bad = json.loads(json.dumps(BODY))
    bad["questions"]["q"]["type"] = "noul"
    response = c.post("/v1/systemone", json=bad)
    assert response.status_code == 422
    assert "'noul' is not supported" in response.json()["detail"]
    assert c.post("/v1/systemone", content=b"{").status_code == 400


def test_limits():
    assert client(max_body_bytes=50).post("/v1/systemone", json=BODY).status_code == 413
    raw = json.dumps(BODY).encode()
    streamed = client(max_body_bytes=50).post(
        "/v1/systemone",
        content=iter([raw[:40], raw[40:]]),
        headers={"content-type": "application/json"},
    )
    assert streamed.status_code == 413
    model = FakeModel(tokens=40000)
    body = json.loads(json.dumps(BODY))
    body["questions"]["r"] = body["questions"]["q"]
    response = client(model).post("/v1/systemone", json=body)
    assert response.status_code == 413
    assert "token limit" in response.json()["detail"]
    assert model.forward_calls == 0


@pytest.mark.parametrize("model", [FakeModel(fail=True), FakeModel(nan=True)])
def test_model_failure_is_json_500(model):
    response = client(model).post("/v1/systemone", json=BODY)
    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}


def test_warmup_runs_the_request_path():
    model = FakeModel()
    app = build_app(
        model,
        api_key=None,
        max_body_bytes=1 << 20,
        max_questions=64,
        max_prompt_tokens=32768,
        revision="r",
    )
    app.state.warmup()
    assert model.forward_calls == 1
    with pytest.raises(ValueError):
        build_app(
            FakeModel(nan=True),
            api_key=None,
            max_body_bytes=1 << 20,
            max_questions=64,
            max_prompt_tokens=32768,
            revision="r",
        ).state.warmup()


def test_bearer_token():
    c = client(api_key="secret")
    assert c.post("/v1/systemone", json=BODY).status_code == 401
    assert c.get("/health").status_code == 200
    ok = c.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
