"""Admission invariants independent of CUDA timing and model internals."""

import pytest

from models.cua_s1.multimodal import graph_runtime


def policy(**kwargs):
    from models.cua_s1.multimodal.graph_admission import AdmissionPolicy

    return AdmissionPolicy(graph_runtime.GraphConfig(**kwargs))


def test_duplicate_questions_are_one_request_observation():
    p = policy()
    p.begin_request()
    assert p.reason("a") == "warmup"
    assert p.reason("a") == "warmup"
    p.begin_request()
    assert p.reason("a") is None


def test_rotating_twelve_lengths_never_qualify_even_after_six_rounds():
    p = policy()
    for key in list(range(12)) * 6:
        p.begin_request()
        assert p.reason(key) == "warmup"
        assert p.reason(key) == "warmup"


def test_eviction_requires_cooldown_then_fresh_request_reuse():
    p = policy(cooldown_requests=2)
    p.begin_request()
    p.reason("a")
    p.begin_request()
    assert p.reason("a") is None
    p.evict("a")
    for _ in range(2):
        p.begin_request()
        assert p.reason("a") == "cooldown"
    p.begin_request()
    assert p.reason("a") == "warmup"
    p.begin_request()
    assert p.reason("a") is None


def test_attempt_count_budget_includes_failed_attempts_and_expires():
    p = policy(min_uses=1, max_captures=2, capture_window=3)
    p.begin_request()
    for key in ("a", "b"):
        assert p.reason(key) is None
        ticket = p.start_capture()
        p.finish_capture(ticket, 1)
    assert p.reason("c") == "capture_budget"
    for _ in range(2):
        p.begin_request()
        assert p.reason("c") == "capture_budget"
    p.begin_request()
    assert p.reason("c") is None


def test_time_budget_blocks_next_attempt_after_nonpreemptible_overshoot():
    p = policy(min_uses=1, capture_budget_ms=10)
    p.begin_request()
    ticket = p.start_capture()
    p.finish_capture(ticket, 11)
    assert p.reason("b") == "capture_budget"


def test_history_and_cooldown_metadata_are_bounded_and_resettable():
    p = policy()
    for key in range(200):
        p.begin_request()
        p.reason(key)
        p.evict(key)
    assert len(p.history) <= 128
    assert len(p.cooldowns) <= 128
    ticket = p.start_capture()
    p.finish_capture(ticket, 3000)
    p.reset()
    assert not p.history and not p.cooldowns and not p.attempts
    assert p.request_index == 0


@pytest.mark.parametrize(
    "field",
    [
        "admission_window",
        "cooldown_requests",
        "capture_window",
        "max_captures",
        "capture_budget_ms",
    ],
)
def test_new_limits_must_be_positive(field):
    with pytest.raises(ValueError, match="positive"):
        graph_runtime.GraphConfig(**{field: 0})


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_capture_time_budget_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        graph_runtime.GraphConfig(capture_budget_ms=value)


def runtime_fixture(monkeypatch, **config):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            no_grad=nullcontext,
            equal=lambda a, b: a == b,
            cuda=SimpleNamespace(
                synchronize=lambda *a: None, OutOfMemoryError=MemoryError
            ),
        ),
    )
    core = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa")
            )
        )
    )
    r = graph_runtime.GraphRuntime(
        SimpleNamespace(get_base_model=lambda: core),
        graph_runtime.GraphConfig(**config),
    )
    r._supported = lambda v: True

    def eager(v):
        r.stats["eager"] += 1
        return 1

    r._eager = eager
    r._run_segments = lambda v, entry: 1

    def values(length):
        tensor = SimpleNamespace(
            shape=(1, length, 2),
            stride=lambda: (length * 2, 2, 1),
            dtype="bf16",
            device="cuda:0",
        )
        return {"inputs_embeds": tensor, "position_ids": tensor, "attention_mask": tensor}

    return r, values


def test_runtime_capture_replay_and_standalone_forward(monkeypatch):
    r, values = runtime_fixture(monkeypatch)
    r.forward(values(3))
    assert r.stats["no_request"] == 1
    with r.request():
        r.forward(values(3))
        r.forward(values(3))
    assert r.stats["captures"] == 0
    with r.request():
        r.forward(values(3))
        r.forward(values(3))
    assert r.stats["captures"] == 1
    assert r.stats["replays"] == 1
    assert r.stats["capture_attempts"] == 1


def test_runtime_evicted_key_is_cooled_and_cache_hits_ignore_budget(monkeypatch):
    r, values = runtime_fixture(monkeypatch, min_uses=1, max_shapes=1, max_captures=2)
    with r.request():
        r.forward(values(1))
        r.forward(values(2))
        r.forward(values(1))
        r.forward(values(2))
        r.forward(values(3))
    assert r.stats["evictions"] == 1
    assert r.stats["cooldown"] == 1
    assert r.stats["replays"] == 1
    assert r.stats["capture_budget"] == 1


def test_runtime_failed_capture_spends_budget(monkeypatch):
    r, values = runtime_fixture(monkeypatch, min_uses=1, max_captures=1)

    def fail(v, entry):
        raise RuntimeError("recoverable capture failure")

    r._run_segments = fail
    with r.request():
        assert r.forward(values(1)) == 1
        assert r.forward(values(2)) == 1
    assert r.stats["capture_attempts"] == 1
    assert r.stats["capture_error"] == 1
    assert r.stats["capture_budget"] == 1
    assert len(r.admission.attempts) == 1


def test_request_error_restores_scope_and_invalidation_resets_policy(monkeypatch):
    r, values = runtime_fixture(monkeypatch)
    with pytest.raises(ValueError), r.request():
        r.forward(values(1))
        raise ValueError("request failed")
    assert not r._in_request
    with r.request():
        with pytest.raises(RuntimeError, match="nested"), r.request():
            pass
        with pytest.raises(RuntimeError, match="invalidate"):
            r.invalidate()
    r.invalidate()
    assert r.admission.request_index == 0
    assert not r.admission.history and not r.cache.entries


def test_engine_predict_opens_one_context_for_entire_request(monkeypatch):
    from models.cua_s1.multimodal.model import MultimodalEngine

    r, _ = runtime_fixture(monkeypatch)
    engine = object.__new__(MultimodalEngine)
    engine.graph_runtime = r

    def predict(request):
        assert r._in_request
        return "response"

    engine._predict = predict
    assert engine.predict(object()) == "response"
    assert r.stats["requests"] == 1
    assert not r._in_request
