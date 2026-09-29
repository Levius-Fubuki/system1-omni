"""Auto CLI, lifecycle, admission clocks and runtime wiring."""

from types import SimpleNamespace

import pytest

from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime


def test_auto_config_and_cli_are_opt_in():
    from models.cua_s1.multimodal.server import parse_args

    config = GraphConfig(mode="auto", bucket_width=128)
    assert config.mode == "auto"
    base = ["--base", "base", "--adapter", "adapter"]
    assert parse_args(base).graph_config is None
    assert parse_args(base + ["--graph"]).graph_config.mode == "exact"
    configured = parse_args(
        base + ["--graph-mode", "auto", "--graph-bucket-width", "128"]
    )
    assert configured.graph_config == config


def test_explicit_resource_injection_preserves_empty_shared_cache():
    from models.cua_s1.multimodal.graph_runtime import GraphCache

    cache = GraphCache(2, 100)
    runtime = GraphRuntime(object(), cache=cache)
    assert runtime.cache is cache


def test_auto_invalidation_clears_policy_and_pending_events():
    from models.cua_s1.multimodal.graph_auto import AutoGraphRuntime

    runtime = AutoGraphRuntime(object(), GraphConfig(mode="auto"))
    with runtime.request():
        runtime.selector.observe("key", "bucket", 256)
    runtime._pending.append(object())
    runtime.invalidate()
    assert not runtime.selector.history
    assert not runtime._pending
    assert not runtime.cache.entries
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"), runtime.request():
        pass


def test_auto_cannot_be_manually_switched():
    from models.cua_s1.multimodal.graph_auto import AutoGraphRuntime

    runtime = AutoGraphRuntime(object(), GraphConfig(mode="auto"))
    with pytest.raises(RuntimeError, match="automatic"):
        runtime.select_mode("exact")


def test_injected_key_namespace_and_close_release_every_owner():
    from models.cua_s1.multimodal.graph_shared import SharedGraphRuntime

    runtime = SharedGraphRuntime(object())
    closed = []
    for mode, child in runtime._runtimes.items():
        entry = SimpleNamespace(bytes=1, close=lambda m=mode: closed.append(m))
        child.cache.put("same", entry)
    assert len(runtime.cache) == 2
    runtime.close()
    assert set(closed) == {"exact", "rule-bucket"}


def fake_dispatch(monkeypatch):
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    from models.cua_s1.multimodal.graph_auto import AutoGraphRuntime

    runtime = AutoGraphRuntime(object())
    exact = runtime._runtimes["exact"]
    bucket = runtime._runtimes["rule-bucket"]
    exact._supported = lambda values: True
    exact._key = lambda values: values["key"]
    bucket._key = lambda values: "bucket"
    exact._eager = lambda values: "eager"
    events = []

    class Event:
        def __init__(self, **kwargs):
            events.append(self)

        def record(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 100

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "device", lambda value: nullcontext())
    values = {
        "key": "a",
        "inputs_embeds": SimpleNamespace(shape=(1, 256, 4), device="cuda"),
    }
    return runtime, exact, bucket, values, events


def test_auto_exact_dispatch_does_not_check_bucket_mask_values(monkeypatch):
    runtime, exact, bucket, values, _ = fake_dispatch(monkeypatch)

    def forbidden(values):
        raise AssertionError("bucket-only synchronization on exact path")

    bucket._dense = forbidden
    runtime.selector.choose = lambda *args, **kwargs: "exact"
    exact.forward = lambda values: "exact"
    with runtime.request():
        assert runtime.forward(values) == "exact"


def test_auto_decision_is_fixed_per_layout_within_request(monkeypatch):
    runtime, exact, _, values, _ = fake_dispatch(monkeypatch)
    choices = iter(["eager", "exact"])
    runtime.selector.choose = lambda *args, **kwargs: next(choices)
    exact.forward = lambda values: "exact"
    with runtime.request():
        assert runtime.forward(values) == runtime.forward(values) == "eager"
    with runtime.request():
        assert runtime.forward(values) == "exact"
    assert runtime.stats["requests"] == 2


def test_auto_sampling_and_request_decisions_are_bounded(monkeypatch):
    runtime, _, _, values, events = fake_dispatch(monkeypatch)
    runtime.selector.choose = lambda *args, **kwargs: "eager"
    with runtime.request():
        for i in range(300):
            runtime.forward(dict(values, key=str(i)))
    assert len(runtime._decisions) <= 128
    assert len(runtime.selector.history) <= 128
    assert len(runtime._pending) <= 16
    assert len(events) < 100
