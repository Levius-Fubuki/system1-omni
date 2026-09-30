"""Shared policy accounting and retirement without requiring a GPU."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recipe/cua_s1"))


def group(**limits):
    from graph_policy import SharedGraphRuntime

    from models.cua_s1.multimodal.graph_runtime import GraphConfig

    return SharedGraphRuntime(object(), GraphConfig(**limits))


def entry(size):
    value = SimpleNamespace(bytes=size, key=None, closed=False)

    def close():
        value.closed = True

    value.close = close
    return value


def test_cross_mode_eviction_closes_live_entry_and_namespaces_keys():
    runtime = group(max_shapes=2, max_bytes=70)
    exact, bucket = runtime._runtimes.values()
    a, b, c = entry(20), entry(25), entry(30)
    assert exact.cache.put("same", a) == []
    assert bucket.cache.put("same", b) == []
    assert exact.cache.get("same") is a
    assert bucket.cache.put("new", c) == [b]
    assert b.closed and not a.closed
    assert bucket.cache.get("same") is None
    assert runtime.cache.bytes == 50
    assert len(runtime.cache) == 2


def test_admission_and_capture_budget_are_shared_but_keys_are_not():
    runtime = group(min_uses=1, max_captures=1, capture_window=2)
    exact, bucket = runtime._runtimes.values()
    with runtime.request():
        assert exact.admission.reason("same") is None
        ticket = exact.admission.start_capture()
        exact.admission.finish_capture(ticket, 5)
        assert bucket.admission.reason("same") == "capture_budget"
        # Two mode calls in one prediction must not advance the request clock.
        assert runtime.admission.request_index == 1
    with runtime.request():
        assert bucket.admission.reason("same") == "capture_budget"
    with runtime.request():
        assert bucket.admission.reason("same") is None
    assert runtime.stats["requests"] == 3


def test_mode_frequency_does_not_satisfy_other_mode_admission():
    runtime = group(min_uses=2)
    exact, bucket = runtime._runtimes.values()
    with runtime.request():
        assert exact.admission.reason("key") == "warmup"
    with runtime.request():
        assert bucket.admission.reason("key") == "warmup"
        assert exact.admission.reason("key") is None


def test_cross_mode_time_budget_is_global():
    runtime = group(min_uses=1, capture_budget_ms=10)
    exact, bucket = runtime._runtimes.values()
    with runtime.request():
        ticket = exact.admission.start_capture()
        exact.admission.finish_capture(ticket, 11)
        assert bucket.admission.reason("x") == "capture_budget"


def test_invalidate_and_close_retire_both_modes_with_live_references():
    runtime = group()
    exact, bucket = runtime._runtimes.values()
    a, b = entry(20), entry(30)
    exact.cache.put("a", a)
    bucket.cache.put("b", b)
    exact.disabled["bad"] = None
    runtime.invalidate()
    assert a.closed and b.closed
    assert not exact.disabled
    assert runtime.cache.bytes == 0
    assert len(runtime.cache) == 0
    assert runtime.admission.request_index == 0
    c = entry(10)
    bucket.cache.put("c", c)
    runtime.close()
    runtime.close()
    assert c.closed and exact._closed and bucket._closed
    with pytest.raises(RuntimeError, match="closed"), runtime.request():
        pass


def test_oversize_entry_does_not_evict_residents():
    runtime = group(max_bytes=40)
    exact, bucket = runtime._runtimes.values()
    a, b = entry(20), entry(41)
    exact.cache.put("a", a)
    assert bucket.cache.put("b", b) is None
    assert not a.closed and exact.cache.get("a") is a
    assert runtime.cache.bytes == 20


def test_close_or_invalidate_during_request_is_rejected():
    runtime = group()
    with runtime.request():
        with pytest.raises(RuntimeError, match="request"):
            runtime.close()
        with pytest.raises(RuntimeError, match="request"):
            runtime.invalidate()
        with pytest.raises(RuntimeError, match="nested"), runtime.request():
            pass
    runtime.close()


def test_different_models_never_share_cache_or_budget():
    first, second = group(min_uses=1), group(min_uses=1)
    first._runtimes["exact"].cache.put("a", entry(20))
    with first.request():
        first.admission.start_capture()
    assert len(second.cache) == 0
    assert second.admission.request_index == 0
    assert len(second.admission.attempts) == 0
    first.close()
    with second.request():
        assert second._runtimes["exact"].admission.reason("a") is None


def test_bucket_width_is_not_applied_to_exact_runtime():
    runtime = group(mode="rule-bucket", bucket_width=128)
    assert runtime._runtimes["exact"].config.bucket_width == 64
    assert runtime._runtimes["rule-bucket"].width == 128


def test_eviction_cooldown_addresses_the_evicted_mode():
    runtime = group(min_uses=1, max_shapes=1)
    exact, bucket = runtime._runtimes.values()
    exact.cache.put("same", entry(20))
    with runtime.request():
        retired = bucket.cache.put("same", entry(30))
        for value in retired:
            bucket.admission.evict(value.key)
        assert exact.admission.reason("same") == "cooldown"
        assert bucket.admission.reason("same") is None
