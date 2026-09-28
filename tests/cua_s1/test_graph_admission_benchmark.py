"""CPU checks for the admission experiment and independent report verifier."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recipe/cua_s1"))


def benchmark():
    import benchmark_graph_admission

    return benchmark_graph_admission


def test_exact_schedules():
    s = benchmark().schedules()
    assert s["hot_four"] == [1, 2, 4, 8] * 12
    assert s["churn_twelve"] == list(range(1, 13)) * 6
    assert s["hot_cold"] == [v for i in range(8) for v in [1, 2, 4, 8, 13 + i]]
    assert s["shifting_hot"] == [
        v
        for group in [range(1, 5), range(5, 9), range(9, 13), range(1, 5)]
        for v in list(group) * 8
    ]


def test_metrics_p95_and_sustained():
    events = [
        {"eager_ms": 10, "graph_ms": v, "graph_stats_delta": {"captures": int(i == 1)}}
        for i, v in enumerate([5, 30, 1, 1, 20, 1])
    ]
    m = benchmark().summarize_events(events)
    assert m["graph_p95_ms"] == 30
    assert m["first_cumulative_crossing_request"] == 1
    assert m["post_capture_sustained_break_even_request"] == 6


def test_legacy_dataclass_import_and_request_adapter(tmp_path):
    path = tmp_path / "legacy.py"
    path.write_text(
        "from dataclasses import dataclass\n@dataclass\nclass GraphConfig:\n    min_uses: int = 2\nclass GraphRuntime:\n    def __init__(self, model, config): self.config = config\n"
    )
    module, digest = benchmark().load_legacy(path)
    assert len(digest) == 64
    runtime = benchmark().legacy_runtime(module, object())
    with runtime.request():
        assert runtime.config.min_uses == 2


def test_response_equality_rejects_unchecked_field():
    a = {"model": "x", "usage": {}, "answers": {}, "extra": 1}
    b = dict(a, extra=2)
    with pytest.raises(ValueError, match="response"):
        benchmark().check_responses(a, b, lambda a, b: 0)


def test_order_reverses_each_request_and_run():
    b = benchmark()
    assert b.variant_order(0, 0) == ["eager", "legacy", "admission"]
    assert (
        b.variant_order(0, 1)
        == b.variant_order(1, 0)
        == ["admission", "legacy", "eager"]
    )


def test_verifier_rejects_dirty_source_and_nan():
    import verify_graph_admission as verify

    with pytest.raises(ValueError, match="clean"):
        verify.verify_report({"status": "complete", "repository": {"dirty": True}})
    with pytest.raises(ValueError, match="finite"):
        verify.metrics([10], [float("nan")], [0])
    metrics = verify.metrics([10] * 6, [5, 30, 1, 1, 20, 1], [0, 1, 0, 0, 0, 0])
    assert metrics["post_capture_sustained_break_even_request"] == 6
    assert metrics["graph_p95_ms"] == 30


def valid_report():
    import hashlib
    import json

    import verify_graph_admission as verifier

    schedule = benchmark().schedules()["hot_four"]
    events = []
    for i, count in enumerate(schedule):
        variants = {}
        for variant in ["eager", "legacy", "admission"]:
            variants[variant] = {
                "response": {"model": "fixture", "usage": {}, "answers": {}},
                "latency_ms": 10.0,
                "cache_shapes": 0,
                "cache_bytes": 0,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "stats_delta": {} if variant == "eager" else {"captures": 0},
                "max_probability_difference": 0,
            }
        events.append(
            {
                "index": i + 1,
                "goal_repetitions": count,
                "tokens_per_question": count + 100,
                "order": benchmark().variant_order(0, i),
                "variants": variants,
            }
        )
    stats = {"legacy": {"captures": 0}, "admission": {"captures": 0}}
    summary = {v: verifier.metrics([10.0] * 48, [10.0] * 48, [0] * 48) for v in stats}
    return {
        "schema_version": 1,
        "status": "complete",
        "repository": {"dirty": False, "revision": "a" * 40},
        "legacy": {"sha256": "b" * 64},
        "environment": {"python": "test"},
        "fixture": {},
        "fixture_sha256": hashlib.sha256(
            json.dumps({}, sort_keys=True).encode()
        ).hexdigest(),
        "config": {
            "cases": ["hot_four"],
            "runs": 1,
            "legacy": {
                "max_shapes": 8,
                "max_bytes": 1024**3,
                "min_uses": 2,
                "max_tokens": 2048,
            },
            "admission": {
                "max_shapes": 8,
                "max_bytes": 1024**3,
                "min_uses": 2,
                "max_tokens": 2048,
                "admission_window": 8,
                "cooldown_requests": 32,
                "capture_window": 32,
                "max_captures": 4,
                "capture_budget_ms": 2000,
            },
        },
        "token_counts": {str(v): v + 100 for v in set(schedule)},
        "workloads": {
            "hot_four": [
                {
                    "run": 1,
                    "status": "complete",
                    "schedule": schedule,
                    "events": events,
                    "stats_initial": stats,
                    "stats_final": stats,
                    "summary": summary,
                }
            ]
        },
    }


def test_verifier_accepts_valid_report_and_rejects_corruption():
    import copy

    import verify_graph_admission as verifier

    report = valid_report()
    assert verifier.verify_report(report)
    for mutate in [
        lambda r: r["config"]["admission"].update(min_uses=99),
        lambda r: r["workloads"]["hot_four"][0]["events"][0]["variants"]["admission"][
            "response"
        ].update(extra=1),
        lambda r: r["workloads"]["hot_four"][0]["events"][0]["variants"]["admission"][
            "stats_delta"
        ].update(captures=1),
        lambda r: r["workloads"]["hot_four"][0]["summary"]["admission"].update(
            graph_p95_ms=11
        ),
        lambda r: r["workloads"]["hot_four"][0]["schedule"].append(1),
        lambda r: r.update(fixture_sha256="x" * 64),
    ]:
        corrupted = copy.deepcopy(report)
        mutate(corrupted)
        with pytest.raises(ValueError):
            verifier.verify_report(corrupted)
