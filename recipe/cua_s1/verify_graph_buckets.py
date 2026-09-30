"""Independently verify raw timing, response and provenance evidence (stdlib)."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


def verify(path, require_replay=False):
    report = json.loads(path.read_text())
    assert report["status"] == "complete", report.get("error")
    assert report["repository"]["dirty"] is False
    root = Path(__file__).resolve().parents[2]
    revision = report["repository"]["revision"]
    for name, expected in report["source_sha256"].items():
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", revision + ":" + name]
        )
        assert hashlib.sha256(source).hexdigest() == expected
    variants_expected = set(
        report["config"].get("variants", ["eager", "exact", "bucket"])
    )
    count = 0
    for name, records in report["workloads"].items():
        assert len(records) == report["config"]["runs"]
        for record in records:
            assert record["status"] == "complete"
            events = record["events"]
            count += len(events) * len(variants_expected)
            for index, event in enumerate(events):
                assert event["index"] == index
                variants = event["variants"]
                assert set(variants) == variants_expected
                assert all(
                    v["response"] == variants["eager"]["response"]
                    for v in variants.values()
                )
                for v in variants.values():
                    assert math.isfinite(v["latency_ms"]) and v["latency_ms"] > 0
                    assert (
                        0
                        <= v["cache_shapes"]
                        <= report["config"]["graph"]["max_shapes"]
                    )
                    assert (
                        0 <= v["cache_bytes"] <= report["config"]["graph"]["max_bytes"]
                    )
            eager_total = sum(e["variants"]["eager"]["latency_ms"] for e in events)
            for variant in variants_expected:
                samples = [e["variants"][variant]["latency_ms"] for e in events]
                summary = record["summary"][variant]
                assert summary["requests"] == len(samples)
                metrics = {
                    "total_ms": sum(samples),
                    "p50_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
                    "total_reduction_vs_eager_pct": 100
                    * (1 - sum(samples) / eager_total),
                }
                for key, value in metrics.items():
                    assert math.isclose(
                        summary[key], value, rel_tol=1e-12, abs_tol=1e-9
                    )
                if variant != "eager":
                    stats = record["stats_final"][variant]
                    for key, total in stats.items():
                        assert math.isclose(
                            total,
                            sum(
                                e["variants"][variant]["stats_delta"][key]
                                for e in events
                            ),
                            rel_tol=1e-12,
                            abs_tol=1e-9,
                        )
            if require_replay:
                stats = record["stats_final"]["bucket"]
                assert stats["captures"] > 0 and stats["replays"] > 0
                assert stats["rejected"] == stats["length_rejections"] == 0
            print(name, record["run"], "verified", len(events), "request groups")
    print(
        "verified",
        count,
        "timed predictions; source hashes, responses, metrics and cache bounds",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--require-replay", action="store_true")
    args = p.parse_args()
    verify(args.report, args.require_replay)
