"""Independently audit the three mixed-length CUDA Graph reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SCHEDULES = {
    "hot_four": [1, 2, 4, 8] * 6,
    "churn_twelve": list(range(1, 13)) * 3,
}
REPORTS = {"default.json": 2, "min4.json": 4, "min8.json": 8}


def check_report(report, min_uses):
    assert report["status"] == "complete"
    assert report["repository"]["revision"]
    assert report["repository"]["dirty"] is False
    assert report["config"]["runs"] == 2
    assert report["config"]["graph"] == {
        "max_shapes": 8,
        "max_bytes": 1024**3,
        "min_uses": min_uses,
        "max_tokens": 2048,
    }
    tokens = {int(k): v for k, v in report["token_counts"].items()}
    assert sorted(tokens) == list(range(1, 13))
    assert len(set(tokens.values())) == 12
    assert max(tokens.values()) <= 2048
    assert set(report["workloads"]) == set(SCHEDULES)
    count = 0
    for name, schedule in SCHEDULES.items():
        runs = report["workloads"][name]
        assert len(runs) == 2
        for run_number, run in enumerate(runs, 1):
            assert run["status"] == "complete"
            assert run["run"] == run_number
            assert run["schedule"] == schedule
            events = run["events"]
            assert len(events) == len(schedule)
            eager_total = graph_total = 0.0
            first_break_even = None
            deltas = {field: 0 for field in run["stats_final"]}
            for index, (event, repetition) in enumerate(zip(events, schedule), 1):
                assert event["index"] == index
                assert event["goal_repetitions"] == repetition
                assert event["tokens_per_question"] == tokens[repetition]
                assert event["order"] == (
                    ["eager", "graph"]
                    if (run_number - 1 + index - 1) % 2 == 0
                    else ["graph", "eager"]
                )
                assert event["max_probability_difference"] == 0
                assert event["eager_ms"] > 0 and event["graph_ms"] > 0
                assert 0 <= event["cache_shapes"] <= 8
                assert 0 <= event["cache_bytes"] <= 1024**3
                assert 0 < event["allocated_bytes"] <= event["peak_allocated_bytes"]
                assert 0 < event["reserved_bytes"] <= event["peak_reserved_bytes"]
                assert event["graph_stats_delta"].keys() == deltas.keys()
                for field, value in event["graph_stats_delta"].items():
                    assert value >= 0
                    deltas[field] += value
                eager_total += event["eager_ms"]
                graph_total += event["graph_ms"]
                if first_break_even is None and graph_total <= eager_total:
                    first_break_even = index
                count += 1
            assert all(
                math.isclose(deltas[field], value, abs_tol=1e-5)
                for field, value in run["stats_final"].items()
            )
            assert run["stats_final"]["numerical_mismatch"] == 0
            assert run["stats_final"]["capture_error"] == 0
            assert run["stats_final"]["capture_oom"] == 0
            assert run["stats_final"]["rejected"] == 0
            summary = run["summary"]
            assert math.isclose(summary["eager_total_ms"], eager_total)
            assert math.isclose(summary["graph_total_ms"], graph_total)
            assert math.isclose(
                summary["total_reduction_percent"],
                100 * (1 - graph_total / eager_total),
            )
            assert math.isclose(
                summary["eager_p50_ms"],
                statistics.median(x["eager_ms"] for x in events),
            )
            assert math.isclose(
                summary["graph_p50_ms"],
                statistics.median(x["graph_ms"] for x in events),
            )
            assert summary["first_break_even_request"] == first_break_even
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    reports = {}
    total = 0
    for filename, min_uses in REPORTS.items():
        report = json.loads((args.directory / filename).read_text())
        total += check_report(report, min_uses)
        reports[filename] = report
    assert len({r["fixture_sha256"] for r in reports.values()}) == 1
    assert (
        len({json.dumps(r["token_counts"], sort_keys=True) for r in reports.values()})
        == 1
    )
    print(f"verified {len(REPORTS)} configurations and {total} paired requests")


if __name__ == "__main__":
    main()
