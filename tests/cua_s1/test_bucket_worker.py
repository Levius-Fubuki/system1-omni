"""Worker mode selection and deterministic Graph lifecycle behavior."""

from types import SimpleNamespace

import pytest

from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "auto"},
        {"mode": "rule-bucket", "bucket_width": 0},
        {"mode": "rule-bucket", "bucket_width": 65},
        {"mode": "rule-bucket", "bucket_width": True},
        {"mode": "rule-bucket", "bucket_width": 4096},
        {"bucket_width": 128},
    ],
)
def test_graph_mode_configuration_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        GraphConfig(**kwargs)


def test_cli_default_and_legacy_and_explicit_modes():
    from models.cua_s1.multimodal.server import parse_args

    base = ["--base", "base", "--adapter", "adapter"]
    assert parse_args(base).graph_config is None
    assert parse_args(base + ["--graph"]).graph_config.mode == "exact"
    assert parse_args(base + ["--graph-mode", "exact"]).graph_config.mode == "exact"
    config = parse_args(
        base + ["--graph-mode", "rule-bucket", "--graph-bucket-width", "128"]
    ).graph_config
    assert config.mode == "rule-bucket" and config.bucket_width == 128


@pytest.mark.parametrize(
    "args",
    [
        ["--graph-bucket-width", "64"],
        ["--graph", "--graph-bucket-width", "128"],
        ["--graph-mode", "auto"],
        ["--graph-mode", "rule-bucket", "--graph-bucket-width", "65"],
    ],
)
def test_cli_invalid_modes_rejected_before_loading(args):
    from models.cua_s1.multimodal.server import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--base", "base", "--adapter", "adapter", *args])


def test_close_explicitly_retires_cache_and_rejects_new_requests():
    runtime = GraphRuntime(object())
    closed = []
    # Keep another reference alive: invalidation cannot rely on GC/destructors.
    entry = SimpleNamespace(bytes=1, close=lambda: closed.append(True))
    runtime.cache.put("key", entry)
    runtime.close()
    runtime.close()
    assert closed == [True]
    assert not runtime.cache.entries
    with pytest.raises(RuntimeError, match="closed"), runtime.request():
        pass


def test_close_during_request_does_not_destroy_live_entries():
    runtime = GraphRuntime(object())
    with runtime.request():
        with pytest.raises(RuntimeError, match="request"):
            runtime.close()
    with runtime.request():
        pass
    runtime.close()


def test_engine_close_is_idempotent_and_prevents_predict():
    from models.cua_s1.multimodal.model import MultimodalEngine

    engine = object.__new__(MultimodalEngine)
    closes = []
    engine.graph_runtime = SimpleNamespace(close=lambda: closes.append(True))
    engine.close()
    engine.close()
    assert closes == [True]
    with pytest.raises(RuntimeError, match="closed"):
        engine.predict(object())


def test_production_rule_packing_and_gate_match_contract():
    torch = pytest.importorskip("torch")
    from models.cua_s1.multimodal.graph_buckets import (
        RuleBucketRuntime,
        pack_rule_inputs,
    )

    values = {k: torch.ones(1, 63, 2, 4) for k in ("query", "key", "value")}
    values.update(g=torch.ones(1, 63, 2), beta=torch.ones(1, 63, 2))
    packed = pack_rule_inputs(values, 64)
    for name, tensor in packed.items():
        assert torch.equal(tensor[:, :63], values[name])
        assert torch.count_nonzero(tensor[:, 63:]) == 0
    r = RuleBucketRuntime(object())
    entry = SimpleNamespace(verified_lengths=set(), rejected_lengths=set())
    reference = torch.tensor([1.0, 2.0])
    r._eager = lambda v: reference
    v = {"inputs_embeds": torch.ones(1, 63, 4)}
    assert torch.equal(r._validate_replay(v, entry, reference), reference)
    v["inputs_embeds"] = torch.ones(1, 64, 4)
    assert torch.equal(r._validate_replay(v, entry, reference + 1), reference)
    assert entry.verified_lengths == {63} and entry.rejected_lengths == {64}


def test_rule_segment_rejects_layout_change_before_copy():
    torch = pytest.importorskip("torch")
    from models.cua_s1.multimodal.graph_buckets import RuleSegment
    from models.cua_s1.multimodal.graph_runtime import tensor_signature

    segment = object.__new__(RuleSegment)
    good = {k: torch.zeros(1, 64, 2, 4) for k in ("query", "key", "value")}
    segment.signatures = {k: tensor_signature(v) for k, v in good.items()}
    segment.static_extra = {"key": good["key"].clone(), "value": good["value"].clone()}
    bad = {**good, "value": torch.ones(1, 1, 2, 4)}
    with pytest.raises(ValueError, match="layout"):
        segment.replay_values(bad)
    assert torch.count_nonzero(segment.static_extra["value"]) == 0


def test_rule_packing_normalizes_strides_at_exact_bucket_boundary():
    torch = pytest.importorskip("torch")
    from models.cua_s1.multimodal.graph_buckets import pack_rule_inputs
    from models.cua_s1.multimodal.graph_runtime import tensor_signature

    signatures = []
    for length in [255, 256]:
        mixed = torch.arange(24 * length).reshape(1, 24, length).transpose(1, 2)
        values = {
            name: value.reshape(1, length, 2, 4)
            for name, value in zip(("query", "key", "value"), mixed.split(8, dim=-1))
        }
        packed = pack_rule_inputs(values, 64)
        for name, value in packed.items():
            assert torch.equal(value[:, :length], values[name])
            assert value.is_contiguous()
        signatures.append({k: tensor_signature(v) for k, v in packed.items()})
    assert signatures[0] == signatures[1]


def test_rejected_length_bypasses_replay_without_disabling_verified_length():
    torch = pytest.importorskip("torch")
    from models.cua_s1.multimodal.graph_buckets import RuleBucketRuntime

    runtime = RuleBucketRuntime(object())
    entry = SimpleNamespace(bytes=1, verified_lengths={63}, rejected_lengths=set())
    runtime.cache.put("bucket", entry)
    runtime._supported = lambda values: True
    runtime._key = lambda values: "bucket"
    reference = torch.tensor([1.0, 2.0])
    runtime._eager = lambda values: reference
    replayed = []

    def replay(values, cached):
        length = values["inputs_embeds"].shape[1]
        replayed.append(length)
        return reference if length == 63 else reference + 1

    runtime._run_segments = replay
    for length in [64, 64, 63]:
        values = {
            "inputs_embeds": torch.ones(1, length, 4),
            "attention_mask": torch.ones(1, length),
        }
        with runtime.request():
            assert torch.equal(runtime.forward(values), reference)
    assert replayed == [64, 63]
    assert entry.rejected_lengths == {64}
    assert entry.verified_lengths == {63}
    assert runtime.stats["length_rejections"] == 1
    assert runtime.stats["length_disabled"] == 1
    assert runtime.stats["replays"] == 2


def test_worker_close_waits_for_inflight_http_before_engine_close():
    import threading

    from test_protocol import request
    from test_server import call

    from models.cua_s1.multimodal.server import WorkerServer

    started, release, closing, closed = (threading.Event() for _ in range(4))

    class Engine:
        def predict(self, parsed):
            started.set()
            assert release.wait(5)
            assert not closed.is_set()
            return {"ok": True}

        def close(self):
            closed.set()

    server = WorkerServer(("127.0.0.1", 0), Engine())
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    responses = []
    client = threading.Thread(
        target=lambda: responses.append(
            call(f"http://127.0.0.1:{server.server_port}/v1/systemone", request())
        )
    )
    client.start()

    def stop():
        server.shutdown()
        closing.set()
        server.server_close()

    stopper = threading.Thread(target=stop)
    try:
        assert started.wait(5)
        stopper.start()
        assert closing.wait(5)
        assert not closed.wait(0.05)
        release.set()
        stopper.join(5)
        client.join(5)
        serving.join(5)
        assert closed.is_set() and not stopper.is_alive()
        assert responses == [(200, {"ok": True})]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        client.join(5)
        serving.join(5)
        if stopper.ident is not None:
            stopper.join(5)
