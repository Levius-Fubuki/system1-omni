"""Unit tests for the Laya worker. A fake Router stands in for laya's; no model is loaded.

python -m pytest src/models/laya/tests
"""

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from frontend import laya_mps as worker
from models.laya import engine, optimize

ANSWER = {"type": "noul", "noul": 0.9, "confidence": 0.9}


def on_gpu(rows):
    """Stands in for input_ids on the GPU: the wrapper only reads its device type and number of rows."""
    return SimpleNamespace(device=SimpleNamespace(type="mps"), shape=(rows, 7))


class FakeAgent:
    def __init__(self, device="mps", dtype="torch.float16"):
        self.device = device
        self.dtype = dtype
        self.mps_amp_min_rows = 5


class FakeRouter:
    """The part of laya.router.Router the worker and laya.serve.create_app use."""

    def __init__(self, agent=None, fail_on_call=None, agents=None):
        self.agent = agent or FakeAgent()
        self.agents = agents if agents is not None else {"english": self.agent}
        self.calls = []
        self.loads = []
        self.fail_on_call = fail_on_call

    @property
    def loaded(self):
        return list(self.agents)

    def load(self, name):
        self.loads.append(name)
        return self.agents.setdefault(name, self.agent)

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions, model))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("MPS backend out of memory")
        return {
            "model": "laya-rl-agent",
            "answers": {qid: ANSWER for qid in questions},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "routing": {"model": model, "repo": "convaiinnovations/laya"},
        }


def test_warmup_covers_short_long_and_fp16_multi_question_shapes():
    router = FakeRouter()
    engine.warmup(router, "english")
    words = {len(state.split()) for state, _, _ in router.calls}
    rows = {len(questions) for _, questions, _ in router.calls}
    assert min(words) <= 20 and max(words) >= 400
    assert max(rows) >= router.agent.mps_amp_min_rows  # crosses laya's fp16 autocast threshold on MPS
    assert {q["type"] for _, questions, _ in router.calls for q in questions.values()} == {"choice", "score", "noul"}
    assert len(router.calls) == len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS
    assert all(model == "english" for _, _, model in router.calls)


def test_warmup_runs_before_the_app_exists():
    router = FakeRouter()
    worker.build_app(router, "english", "mps")
    assert len(router.calls) == len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS


def test_warmup_failure_raises_and_no_app_is_built():
    with pytest.raises(RuntimeError, match="out of memory"):
        worker.build_app(FakeRouter(fail_on_call=3), "english", "mps")


def test_health_reports_the_agent_device_not_the_requested_one():
    router = FakeRouter(FakeAgent(device="cpu", dtype="torch.float32"))
    health = TestClient(worker.build_app(router, "english", "mps")).get("/health").json()
    assert health["device"] == "cpu"
    assert health["requested_device"] == "mps"
    assert health["device_mismatch"] is True
    assert health["ready"] is True


def test_health_on_the_requested_device():
    health = TestClient(worker.build_app(FakeRouter(), "english", "mps")).get("/health").json()
    assert health["device"] == "mps"
    assert health["device_mismatch"] is False
    assert health["autocast_dtype"] == "torch.float16"
    assert health["checkpoint"] == "convaiinnovations/laya"
    assert health["warmup_ms"] >= 0


def test_device_index_is_not_a_mismatch():
    router = FakeRouter(FakeAgent(device="cuda:0"))
    assert TestClient(worker.build_app(router, "english", "cuda")).get("/health").json()["device_mismatch"] is False


def test_auto_device_is_never_a_mismatch():
    router = FakeRouter(FakeAgent(device="cpu"))
    health = TestClient(worker.build_app(router, "english", None)).get("/health").json()
    assert health["requested_device"] == "auto"
    assert health["device_mismatch"] is False


def test_require_device_refuses_to_serve_on_another_device():
    router = FakeRouter(FakeAgent(device="cpu"))
    with pytest.raises(RuntimeError, match="asked for mps, english is on cpu"):
        worker.build_app(router, "english", "mps", require_device=True)


def test_only_one_health_route_remains():
    app = worker.build_app(FakeRouter(), "english", "mps")
    assert [r.path for r in app.router.routes if getattr(r, "path", None) == "/health"] == ["/health"]


def test_decisions_still_go_through_laya_serve():
    router = FakeRouter()
    client = TestClient(worker.build_app(router, "english", "mps"))
    before = len(router.calls)
    response = client.post(
        "/v1/systemone",
        json={"model": "english", "state": "refund me", "questions": {"r": {"type": "noul", "instructions": "?"}}},
    )
    assert response.status_code == 200
    assert response.json()["answers"]["r"]["noul"] == 0.9
    assert len(router.calls) == before + 1


def test_main_exits_non_zero_when_warmup_fails(monkeypatch):
    monkeypatch.setattr(worker, "make_router", lambda device, model: FakeRouter(fail_on_call=1))
    monkeypatch.setattr(sys, "argv", ["laya_mps", "--device", "mps"])
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not bind"))
    with pytest.raises(SystemExit, match="not starting"):
        worker.main()


def test_compile_wraps_the_model_before_warmup(monkeypatch):
    router = FakeRouter()
    order = []
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: order.append((agent, len(router.calls))))
    worker.build_app(router, "english", "mps", compile=True, graph_counter=lambda: 3)
    assert order == [(router.agent, 0)]  # before the first warmup request


def test_health_reports_compile_off_by_default():
    health = TestClient(worker.build_app(FakeRouter(), "english", "mps")).get("/health").json()
    assert health["compile"] == {"enabled": False}


def test_health_flags_graphs_compiled_after_ready(monkeypatch):
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: None)
    graphs = iter([4, 4, 5])  # at readiness, first /health, second /health after a new shape compiled
    client = TestClient(
        worker.build_app(FakeRouter(), "english", "mps", compile=True, graph_counter=lambda: next(graphs))
    )
    first = client.get("/health").json()["compile"]
    assert first == {"enabled": True, "graphs_at_ready": 4, "graphs_now": 4, "recompiled_after_ready": False}
    assert client.get("/health").json()["compile"]["recompiled_after_ready"] is True


def test_compile_failure_means_no_app(monkeypatch):
    def broken(agent):
        raise RuntimeError("inductor: unsupported op on mps")

    monkeypatch.setattr(optimize, "compile_agent", broken)
    with pytest.raises(RuntimeError, match="unsupported op"):
        worker.build_app(FakeRouter(), "english", "mps", compile=True)


def test_compiled_paths_by_batch_rows(monkeypatch):
    import torch

    class Encoder(torch.nn.Module):
        def forward(self, input_ids):
            return "eager encoder"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.head = torch.nn.Linear(2, 2)

        def forward(self, input_ids):
            return ("head", self.encoder(input_ids))

    class Stub(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        def forward(self, *args):
            return self.kind

    def fake_compile(module, dynamic):
        assert dynamic is True
        return Stub("compiled encoder" if isinstance(module, Encoder) else "whole model compiled")

    monkeypatch.setattr(torch, "compile", fake_compile)
    agent = FakeAgent()
    model = Model()
    agent.model = model
    optimize.compile_agent(agent)
    assert agent.model(on_gpu(1)) == "whole model compiled"  # one question
    assert agent.model(on_gpu(3)) == ("head", "compiled encoder")  # several: eager head, compiled encoder
    assert agent.model(torch.zeros(1, 7)) == ("head", "eager encoder")  # on the CPU: laya's model as it is
    assert model.encoder(torch.zeros(1, 7)) == "eager encoder"  # the original model is left as it was
    assert len(list(agent.model.parameters())) == len(list(model.parameters()))  # one set of weights


def test_every_loaded_model_is_warmed_and_described():
    agents = {"english": FakeAgent(), "multilingual": FakeAgent(device="cpu", dtype="torch.float32")}
    router = FakeRouter(agents=agents)
    health = TestClient(worker.build_app(router, "english", "mps")).get("/health").json()
    per_model = len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS
    assert [m for _, _, m in router.calls].count("multilingual") == per_model
    assert [m for _, _, m in router.calls].count("english") == per_model
    assert set(health["models"]) == {"english", "multilingual"}
    assert health["device"] == "mps"  # the top level summarises --model
    assert health["models"]["multilingual"]["device"] == "cpu"
    assert health["device_mismatch"] is True  # one model off the requested device is enough


def test_a_model_that_is_not_preloaded_is_not_loaded_for_warmup():
    router = FakeRouter(agents={"multilingual": FakeAgent()})
    health = TestClient(worker.build_app(router, "english", "mps")).get("/health").json()
    assert "english" not in router.loads
    assert {m for _, _, m in router.calls} == {"multilingual"}
    assert set(health["models"]) == {"multilingual"}


def test_nothing_preloaded_warms_the_worker_model():
    router = FakeRouter(agents={})
    worker.build_app(router, "english", "mps")
    assert {m for _, _, m in router.calls} == {"english"}


def test_revision_comes_from_the_loaded_snapshot_not_a_guess():
    revisions = {"convaiinnovations/laya": "55cf4c4"}
    health = TestClient(worker.build_app(FakeRouter(), "english", "mps", revisions=revisions)).get("/health")
    assert health.json()["revision"] == "55cf4c4"
    unknown = TestClient(worker.build_app(FakeRouter(), "english", "mps")).get("/health").json()
    assert unknown["revision"] is None


def test_record_snapshot_revisions_reads_the_downloaded_path(monkeypatch):
    import huggingface_hub

    paths = {
        "convaiinnovations/laya": "/cache/models--convaiinnovations--laya/snapshots/55cf4c4abc/multilingual",
        "/local/checkpoint": "/local/checkpoint",
    }
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kwargs: paths[repo_id])
    revisions = engine.record_snapshot_revisions()
    assert (
        huggingface_hub.snapshot_download("convaiinnovations/laya", allow_patterns=["*"])
        == paths["convaiinnovations/laya"]
    )
    huggingface_hub.snapshot_download("/local/checkpoint")
    assert revisions == {"convaiinnovations/laya": "55cf4c4abc"}


def test_fp16_weights_keep_act_head_in_fp32():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.act_head = torch.nn.Linear(4, 2)

    agent = FakeAgent()
    agent.model = Model()
    optimize.use_fp16_weights(agent)
    assert agent.model.eager.encoder.weight.dtype == torch.float16
    assert agent.model.eager.act_head.weight.dtype == torch.float32


def test_fp16_weights_are_applied_to_every_loaded_model_before_warmup(monkeypatch):
    order = []
    agents = {"english": FakeAgent(), "multilingual": FakeAgent()}
    router = FakeRouter(agents=agents)
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: order.append((agent, len(router.calls))))
    worker.build_app(router, "english", "mps", fp16=True)
    assert order == [(agents["english"], 0), (agents["multilingual"], 0)]


def test_options_are_not_applied_to_a_model_on_the_cpu(monkeypatch, caplog):
    applied = []
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: applied.append("fp16"))
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: applied.append("compile"))
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(
            FakeRouter(FakeAgent(device="cpu")), "english", "cpu", fp16=True, compile=True, graph_counter=lambda: 0
        )
    assert applied == []
    assert "apply on the GPU only" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(FakeRouter(), "english", "mps", fp16=True, compile=True, graph_counter=lambda: 0)
    assert applied == ["fp16", "compile"]
    assert "GPU only" not in caplog.text


def test_after_a_fallback_to_cpu_the_model_runs_fp32_and_uncompiled(monkeypatch):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.act_head = torch.nn.Linear(4, 2)

        def forward(self, input_ids):
            return ("eager", self.encoder.weight.dtype)

    monkeypatch.setattr(torch, "compile", lambda module, dynamic: lambda *a, **k: "compiled")
    agent = FakeAgent()
    agent.model = Model()
    optimize.use_fp16_weights(agent)
    optimize.compile_agent(agent)
    assert agent.model(on_gpu(1)) == "compiled"
    assert next(agent.model.parameters()).dtype == torch.float16
    assert agent.model(torch.zeros(1, 7)) == ("eager", torch.float32)  # inputs on the CPU: laya fell back
    assert {p.dtype for p in agent.model.parameters()} == {torch.float32}
    assert agent.model(torch.zeros(3, 7)) == ("eager", torch.float32)


def test_health_follows_a_fallback_to_cpu_after_startup():
    agent = FakeAgent(device="mps")
    client = TestClient(worker.build_app(FakeRouter(agent), "english", "mps", require_device=True))
    assert client.get("/health").json()["device_mismatch"] is False
    agent.device = "cpu"  # what laya does when a request runs out of GPU memory
    agent.dtype = "torch.float32"
    health = client.get("/health").json()
    assert health["device"] == "cpu"
    assert health["device_mismatch"] is True
    assert health["models"]["english"]["device"] == "cpu"
    assert health["autocast_dtype"] == "torch.float32"
