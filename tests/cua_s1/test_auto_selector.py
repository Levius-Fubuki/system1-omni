"""Online selection uses bounded history, costs and resident resources only."""

from types import SimpleNamespace

from models.cua_s1.multimodal.graph_runtime import GraphConfig


def selector(**kwargs):
    from models.cua_s1.multimodal.graph_selector import AutoSelector

    return AutoSelector(GraphConfig(**kwargs))


def observe(policy, index, key, bucket="bucket", length=256, copies=2):
    policy.begin_request(index)
    for _ in range(copies):
        policy.observe(key, bucket, length)


def test_duplicates_within_one_request_do_not_admit_cold_layout():
    policy = selector()
    observe(policy, 1, "a", copies=8)
    assert policy.choose("a", "bucket", 256) == "eager"


def test_fast_recurring_layout_selects_exact_after_cost_check():
    policy = selector()
    observe(policy, 1, "a")
    observe(policy, 5, "a")
    assert policy.choose("a", "bucket", 256) == "exact"
    policy.record("bucket", "exact", capture_ms=100000)
    assert policy.choose("a", "bucket", 256) != "exact"
    policy.record("bucket", "rule-bucket", capture_ms=100000)
    assert policy.choose("a", "bucket", 256) == "eager"


def test_distributed_recurrence_selects_bucket():
    policy = selector()
    for i in range(1, 13):
        observe(policy, i, f"key{i}", length=230 + i)
        assert policy.choose(f"key{i}", "bucket", 230 + i) == "eager"
    observe(policy, 13, "key1", length=231)
    assert policy.choose("key1", "bucket", 231) == "rule-bucket"


def test_one_off_layout_does_not_pay_a_resident_bucket_length_gate():
    policy = selector()
    observe(policy, 1, "new", length=250)
    entry = SimpleNamespace(verified_lengths={240}, rejected_lengths=set())
    assert policy.choose("new", "bucket", 250, bucket_entry=entry) == "eager"
    observe(policy, 2, "new", length=250)
    assert policy.choose("new", "bucket", 250, bucket_entry=entry) in {
        "exact",
        "rule-bucket",
    }


def test_resident_exact_is_used_without_recapturing():
    policy = selector()
    observe(policy, 1, "a")
    assert policy.choose("a", "bucket", 256, exact_entry=object()) == "exact"


def test_rejected_bucket_length_is_never_selected():
    policy = selector()
    observe(policy, 1, "a")
    observe(policy, 13, "a")
    rejected = SimpleNamespace(verified_lengths=set(), rejected_lengths={256})
    assert policy.choose("a", "bucket", 256, bucket_entry=rejected) == "eager"


def test_cache_pressure_avoids_exact_capture_and_uses_bucket():
    policy = selector()
    for i in range(1, 11):
        key = str((i - 1) % 5)
        observe(policy, i, key, length=240 + (i - 1) % 5)
    assert policy.choose("4", "bucket", 244) == "rule-bucket"


def test_predicted_bytes_prevent_exact_capture_eviction():
    policy = selector(max_bytes=100)
    observe(policy, 1, "a")
    observe(policy, 2, "a")
    policy.record("bucket", "exact", owned_bytes=60)
    assert policy.choose("a", "bucket", 256, cache_bytes=50) == "rule-bucket"


def test_recent_bucket_selection_cools_down_exact_promotion():
    policy = selector()
    observe(policy, 1, "a")
    policy.selected("a", "rule-bucket")
    observe(policy, 2, "a")
    policy.record("bucket", "exact", capture_ms=1, latency_ms=1)
    policy.record("bucket", "rule-bucket", latency_ms=90)
    bucket = SimpleNamespace(verified_lengths={256}, rejected_lengths=set())
    assert policy.choose("a", "bucket", 256, bucket_entry=bucket) == "rule-bucket"


def test_history_costs_and_invalidation_are_bounded():
    policy = selector()
    for i in range(1, 301):
        observe(policy, i, str(i), bucket=str(i))
        policy.record(str(i), "eager", latency_ms=100)
    assert len(policy.history) <= 128
    assert len(policy.costs) <= 128
    assert "1" not in policy.history
    policy.reset()
    assert not policy.history and not policy.costs


def test_cost_record_ignores_nonfinite_or_negative_samples():
    policy = selector()
    policy.record("b", "eager", latency_ms=100)
    for value in [float("nan"), float("inf"), -1, 0]:
        policy.record("b", "eager", latency_ms=value)
    assert policy.costs["b"]["eager"] == 100


def test_resident_bucket_must_pay_new_length_gate_and_have_positive_saving():
    policy = selector()
    observe(policy, 1, "a", length=250, copies=1)
    observe(policy, 32, "a", length=250, copies=1)
    entry = SimpleNamespace(verified_lengths={240}, rejected_lengths=set())
    policy.record("bucket", "eager", latency_ms=100)
    for replay in (120, 99):
        policy.costs["bucket"]["rule-bucket"] = replay
        assert policy.choose("a", "bucket", 250, bucket_entry=entry) == "eager"


def test_slow_verified_bucket_is_not_replayed_for_cold_layout():
    policy = selector()
    observe(policy, 1, "a")
    policy.record("bucket", "eager", latency_ms=100)
    policy.record("bucket", "rule-bucket", latency_ms=120)
    entry = SimpleNamespace(verified_lengths={256}, rejected_lengths=set())
    assert policy.choose("a", "bucket", 256, bucket_entry=entry) == "eager"
