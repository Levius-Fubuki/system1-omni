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
                "stats_delta": {}
                if variant == "eager"
                else dict(
                    policy_event()["variants"][variant]["stats_delta"], captures=0
                ),
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
    stats = {
        v: dict.fromkeys(events[0]["variants"][v]["stats_delta"], 0)
        for v in ["legacy", "admission"]
    }
    final = {
        v: {
            k: sum(e["variants"][v]["stats_delta"][k] for e in events) for k in stats[v]
        }
        for v in stats
    }
    summary = {v: verifier.metrics([10.0] * 48, [10.0] * 48, [0] * 48) for v in stats}
    return {
        "schema_version": 1,
        "status": "complete",
        "repository": {"dirty": False, "revision": "a" * 40},
        "legacy": {
            "sha256": "b" * 64,
            "revision": "c" * 40,
            "source_bytes_verified_against_git": True,
            "current_graph_runtime_sha256": "d" * 64,
            "runtime_segment_override": "current exclusive-stream _GraphSegment; historical admission policy only",
        },
        "comparison_kind": "legacy-policy-with-stream-fix",
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
                    "stats_final": final,
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
        lambda r: r["workloads"]["hot_four"][0]["events"][0]["variants"]["legacy"][
            "stats_delta"
        ].update(rejected=1),
        lambda r: r["workloads"]["hot_four"][0]["events"][0]["variants"][
            "eager"
        ].update(allocated_bytes=1),
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


def policy_event(**changes):
    stats = dict.fromkeys(
        [
            "numerical_mismatch",
            "capture_error",
            "capture_oom",
            "rejected",
            "unsupported",
            "no_request",
            "capture_attempts",
            "captures",
            "replays",
            "capture_attempt_ms",
        ],
        0,
    )
    stats["requests"] = 1
    stats.update(changes)
    return {
        "variants": {
            "admission": {"stats_delta": stats},
            "legacy": {
                "stats_delta": {
                    k: 0
                    for k in [
                        "numerical_mismatch",
                        "capture_error",
                        "capture_oom",
                        "rejected",
                        "unsupported",
                    ]
                }
            },
        }
    }


@pytest.mark.parametrize(
    "events,case,message",
    [
        (
            [policy_event(capture_attempts=1, captures=1, capture_attempt_ms=1)] * 5,
            "hot_four",
            "count budget",
        ),
        (
            [
                policy_event(capture_attempts=1, captures=1, capture_attempt_ms=2000),
                policy_event(capture_attempts=1, captures=1),
            ],
            "hot_four",
            "time budget",
        ),
        ([policy_event(rejected=1)], "hot_four", "rejected"),
        ([policy_event(requests=2)], "hot_four", "request counter"),
        ([policy_event(capture_attempts=2, captures=2)], "hot_four", "one capture"),
        (
            [policy_event(capture_attempts=1, captures=0)],
            "hot_four",
            "successful captures",
        ),
        ([policy_event(replays=1)], "churn_twelve", "churn"),
    ],
)
def test_policy_verifier_rejects_recorded_violations(events, case, message):
    import verify_graph_admission as verifier

    with pytest.raises(ValueError, match=message):
        verifier.verify_policy(events, case, valid_report()["config"]["admission"])


def test_policy_window_expires_at_request_distance_32():
    import verify_graph_admission as verifier

    events = [policy_event(capture_attempts=1, captures=1, capture_attempt_ms=2001)]
    events += [policy_event() for _ in range(31)]
    events += [policy_event(capture_attempts=1, captures=1, capture_attempt_ms=1)]
    verifier.verify_policy(events, "hot_four", valid_report()["config"]["admission"])


def test_legacy_source_provenance_checks_git_bytes(tmp_path):
    import subprocess

    revision = subprocess.check_output(
        ["git", "rev-parse", "6e0a432"], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", revision + ":src/models/cua_s1/multimodal/graph_runtime.py"],
        cwd=ROOT,
    )
    path = tmp_path / "legacy.py"
    path.write_bytes(source)
    benchmark().verify_legacy_source(path, revision, ROOT)
    path.write_bytes(source + b"\n# changed\n")
    with pytest.raises(ValueError, match="bytes"):
        benchmark().verify_legacy_source(path, revision, ROOT)
    with pytest.raises(ValueError, match="40"):
        benchmark().verify_legacy_source(path, "6e0a432", ROOT)


def test_legacy_segment_override_and_diagnostic():
    from types import SimpleNamespace

    original, current = object(), object()
    module = SimpleNamespace(_GraphSegment=original)
    marker = benchmark().configure_legacy_segment(module, current)
    assert module._GraphSegment is current
    assert (
        marker
        == "current exclusive-stream _GraphSegment; historical admission policy only"
    )
    module = SimpleNamespace(_GraphSegment=original)
    marker = benchmark().configure_legacy_segment(module, current, unpatched=True)
    assert module._GraphSegment is original
    assert marker == "none; unpatched historical runtime diagnostic"


def test_verifier_requires_explicit_segment_provenance():
    import verify_graph_admission as verifier

    report = valid_report()
    report["legacy"]["runtime_segment_override"] = "unknown"
    with pytest.raises(ValueError, match="segment override"):
        verifier.verify_report(report)
