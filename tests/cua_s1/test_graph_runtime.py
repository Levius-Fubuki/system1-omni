"""Policy tests for the bounded CUDA Graph runtime (no GPU required)."""

from types import SimpleNamespace

import pytest

from models.cua_s1.multimodal.graph_runtime import (
    GraphCache,
    GraphConfig,
    GraphRuntime,
    tensor_signature,
)


@pytest.mark.parametrize(
    "setting",
    [
        {"max_shapes": 0},
        {"max_bytes": 0},
        {"min_uses": 0},
        {"max_tokens": 0},
    ],
)
def test_graph_config_rejects_nonpositive_limits(setting):
    with pytest.raises(ValueError, match="positive"):
        GraphConfig(**setting)


def test_tensor_signature_includes_layout_dtype_and_device():
    value = SimpleNamespace(
        shape=(1, 128, 2560),
        stride=lambda: (327680, 2560, 1),
        dtype="bfloat16",
        device="cuda:0",
    )
    assert tensor_signature(value) == (
        (1, 128, 2560),
        (327680, 2560, 1),
        "bfloat16",
        "cuda:0",
    )


def test_cache_evicts_least_recently_used_shape_atomically():
    cache = GraphCache(max_shapes=2, max_bytes=70)
    first = SimpleNamespace(bytes=20)
    second = SimpleNamespace(bytes=25)
    third = SimpleNamespace(bytes=30)
    assert cache.put("a", first) == []
    assert cache.put("b", second) == []
    assert cache.get("a") is first
    assert cache.put("c", third) == [second]
    assert cache.get("b") is None
    assert cache.get("a") is first
    assert cache.get("c") is third
    assert cache.bytes == 50


def test_cache_rejects_one_shape_exceeding_budget():
    cache = GraphCache(max_shapes=2, max_bytes=40)
    assert cache.put("huge", SimpleNamespace(bytes=41)) is None
    assert len(cache) == 0


def test_graph_runtime_falls_back_if_model_is_training():
    runtime = GraphRuntime(SimpleNamespace(training=True))
    values = {
        "inputs_embeds": SimpleNamespace(
            device=SimpleNamespace(type="cuda"), ndim=3, shape=(1, 4, 2560)
        ),
        "position_ids": SimpleNamespace(shape=(3, 1, 4)),
    }
    assert runtime._supported(values) is False
