"""Recompute automatic-worker timing, shared budgets and parity evidence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from verify_graph_buckets import verify


def verify_auto(report):
    configs = report["config"]["variant_graph"]
    base = configs["exact"]
    assert configs["auto"] == dict(base, mode="auto")
    assert configs["bucket"] == dict(base, mode="rule-bucket")
    assert configs["exact_tuned"] == dict(base, admission_window=32)
    for records in report["workloads"].values():
        for record in records:
            events = record["events"]
            for stats in record["stats_final"].values():
                assert stats["requests"] == len(events)
                for key in (
                    "rejected",
                    "length_rejections",
                    "numerical_mismatch",
                    "capture_error",
                    "capture_oom",
                    "unsupported",
                    "no_request",
                ):
                    assert stats.get(key, 0) == 0
            for i, event in enumerate(events):
                auto = event["variants"]["auto"]
                stats = auto["stats_delta"]
                assert stats["requests"] == 1
                assert sum(
                    stats["selected_" + mode]
                    for mode in ("eager", "exact", "rule_bucket")
                ) == len(auto["response"]["answers"])
                assert len(auto["resident_modes"]) == auto["cache_shapes"]
                assert set(auto["resident_modes"]) <= {"exact", "rule-bucket"}
                for key in (
                    "rejected",
                    "length_rejections",
                    "capture_oom",
                    "capture_error",
                    "numerical_mismatch",
                ):
                    assert stats[key] == 0
                recent = events[max(0, i - base["capture_window"] + 1) : i]
                prior = [e["variants"]["auto"]["stats_delta"] for e in recent]
                attempts = stats["capture_attempts"]
                assert (
                    sum(s["capture_attempts"] for s in prior) + attempts
                    <= base["max_captures"]
                )
                # Synchronous captures may individually overshoot. With the
                # identical-question schedule at most one attempt occurs/request.
                assert attempts <= 1
                if attempts:
                    assert (
                        sum(s["capture_attempt_ms"] for s in prior)
                        < base["capture_budget_ms"]
                    )


def verify_parity(path):
    report = json.loads(path.read_text())
    assert report["status"] == "complete", report.get("error")
    assert report["repository"]["dirty"] is False
    root = Path(__file__).resolve().parents[2]
    for name, expected in report["source_sha256"].items():
        content = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "show",
                report["repository"]["revision"] + ":" + name,
            ]
        )
        assert hashlib.sha256(content).hexdigest() == expected
    parity(report)


def parity(report):
    assert report["status"] == "complete", report.get("error")
    assert report["explicit_close"]
    assert len(report["cases"]) == 5
    for case in report["cases"]:
        assert case["stats_delta"]["captures"] > 0
        assert case["stats_delta"]["replays"] > 0
    for key in (
        "rejected",
        "length_rejections",
        "numerical_mismatch",
        "capture_error",
        "capture_oom",
    ):
        assert report["stats"][key] == 0
    responses = [e for c in report["cases"] for e in c["events"]] + report[
        "churn_events"
    ]
    assert len(responses) == 61
    for event in responses:
        assert event["expected"] == event["actual"]
        assert event["stats_delta"]["requests"] == 1
    assert len(report["logit_checks"]) == 266
    assert all(c["equal"] and c["max_abs"] == 0 for c in report["logit_checks"])
    assert len(report["boundaries"]) == 24
    churn = report["churn_stats"]
    assert churn["selected_rule_bucket"] > 0
    # No exact routes occurred in this diagnostic, so actual captures/replays
    # (not just selected routes) establish that bucket execution was exercised.
    assert churn["selected_exact"] == 0
    assert churn["captures"] > 0 and churn["replays"] > 0
    assert sum(e["stats_delta"]["captures"] for e in report["boundaries"]) > 0
    assert sum(e["stats_delta"]["replays"] for e in report["boundaries"]) > 0
    print("verified 266 complete-logit checks, 61 responses and explicit retirement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--parity", type=Path)
    args = parser.parse_args()
    verify(args.report, require_replay=True)
    verify_auto(json.loads(args.report.read_text()))
    print("verified automatic routing counts, shared cache and capture budgets")
    if args.parity:
        verify_parity(args.parity)
