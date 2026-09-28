"""Independently recompute the segmented Graph report's timing and gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CASES = {
    "320x240-short-q2",
    "320x240-short-q8",
    "640x480-short-q8",
    "640x480-long-q8",
    "640x480-distinct-q8",
}


def verify(report, cache_report):
    assert report["status"] == "complete"
    assert report["repository"]["revision"]
    assert report["repository"]["dirty"] is False
    assert {case["id"] for case in report["cases"]} == CASES
    assert report["config"]["runs"] == 2
    assert report["config"]["iterations"] == 20
    total = 0
    for case in report["cases"]:
        assert case["status"] == "complete"
        for field in (
            "original_max_probability_difference",
            "changed_image_max_probability_difference",
            "changed_text_max_probability_difference",
        ):
            assert case[field] == 0, (case["id"], field)
        assert case["stats_final"]["captures"] >= 1
        assert case["stats_final"]["numerical_mismatch"] == 0
        assert case["stats_final"]["capture_error"] == 0
        assert case["cache_bytes_final"] <= 1024 * 1024 * 1024
        assert len(case["runs"]) == 2
        for run in case["runs"]:
            for name in ("eager", "graph"):
                item = run[name]
                samples = item["samples_ms"]
                assert len(samples) == 20
                assert all(value > 0 for value in samples)
                assert item["p50_ms"] == statistics.median(samples)
                ordered = sorted(samples)
                assert item["p95_ms"] == ordered[19]
                total += len(samples)

    for name in ("shape_eviction", "memory_fallback"):
        assert cache_report[name]["exact_response_parity"] is True
    assert cache_report["shape_eviction"]["stats"]["evictions"] >= 1
    assert cache_report["shape_eviction"]["cache_shapes"] == 1
    assert cache_report["memory_fallback"]["stats"]["memory_budget"] >= 1
    assert cache_report["memory_fallback"]["cache_shapes"] == 0
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--cache-report", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    cache_report = json.loads(args.cache_report.read_text())
    total = verify(report, cache_report)
    print(f"verified {len(report['cases'])} cases and {total} timed predictions")


if __name__ == "__main__":
    main()
