"""Independent stdlib validation of the controlled and shared-budget experiments."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def provenance(report, root):
    assert report["status"] == "complete", report.get("error")
    assert report["repository"]["dirty"] is False
    revision = report["repository"]["revision"]
    assert len(revision) == 40
    for name, expected in report["source_sha256"].items():
        data = subprocess.check_output(
            ["git", "-C", str(root), "show", revision + ":" + name]
        )
        assert hashlib.sha256(data).hexdigest() == expected, name


def clean_stats(stats):
    for key in [
        "capture_oom",
        "capture_error",
        "numerical_mismatch",
        "length_rejections",
        "unsupported",
        "no_request",
    ]:
        assert stats.get(key, 0) == 0, (key, stats.get(key))


def baseline(report):
    configs = report["config"]["variant_graph"]
    exact, tuned = dict(configs["exact"]), dict(configs["exact_tuned"])
    assert exact.pop("admission_window") == 8
    assert tuned.pop("admission_window") == 32
    assert exact == tuned, "tuned comparator changed more than admission window"
    bucket = dict(configs["bucket"])
    assert bucket.pop("mode") == "rule-bucket"
    reference = dict(configs["exact"])
    reference.pop("mode")
    assert bucket == reference
    assert report["config"]["runs"] == 2
    schedules = {
        "hot_four": [1, 2, 4, 8] * 12,
        "churn_twelve": list(range(1, 13)) * 6,
        "hot_cold": [v for i in range(8) for v in [1, 2, 4, 8, 13 + i]],
        "shifting_hot": [
            v
            for g in [range(1, 5), range(5, 9), range(9, 13), range(1, 5)]
            for v in list(g) * 8
        ],
    }
    assert report["config"]["schedules"] == schedules
    assert set(report["workloads"]) == set(schedules)
    count = 0
    for name, records in report["workloads"].items():
        assert len(records) == 2
        for record in records:
            assert [e["goal_repetitions"] for e in record["events"]] == schedules[name]
            for mode, stats in record["stats_final"].items():
                clean_stats(stats)
                assert stats["rejected"] == 0
                assert stats["requests"] == len(schedules[name])
                # Check count and time limits at each attempt. These workloads
                # have two identical questions, so at most one new shape/request.
                attempts = []
                config = configs[mode]
                for index, event in enumerate(record["events"]):
                    delta = event["variants"][mode]["stats_delta"]
                    assert delta["requests"] == 1
                    n = delta["capture_attempts"]
                    assert n in (0, 1)
                    assert delta["captures"] == n
                    attempts = [
                        (i, t)
                        for i, t in attempts
                        if index - i < config["capture_window"]
                    ]
                    if n:
                        assert len(attempts) < config["max_captures"]
                        assert sum(t for _, t in attempts) < config["capture_budget_ms"]
                        attempts.append((index, delta["capture_attempt_ms"]))
            count += len(record["events"]) * 4
    assert count == 2304
    return count


def cleanup(record):
    assert record["live_entry_references_retired"] is True
    assert record["after"]["cache_shapes"] == record["after"]["cache_bytes"] == 0


def shared(report):
    records = report["boundary_checks"]
    assert len(records) == 24
    assert [(e["mode"], e["tokens"]) for e in records] == [
        (mode, n)
        for n in [255, 256, 257, 319, 320, 321, 321, 320, 319, 257, 256, 255]
        for mode in ("exact", "rule-bucket")
    ]
    assert report["boundary_stats"]["captures"] == 9
    assert report["boundary_stats"]["replays"] == 15
    cleanup(report["boundary_cleanup"])
    groups = [(report["boundary_config"], records, report["boundary_stats"])]
    for record in report["policies"].values():
        groups.append((record["config"], record["events"], record["stats"]))
        cleanup(record["cleanup"])
    total = 0
    for config, events, stats in groups:
        clean_stats(stats)
        assert stats["requests"] == len(events)
        for key, value in stats.items():
            assert math.isclose(
                value,
                sum(e["stats_delta"][key] for e in events),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        for event in events:
            assert event["equal"] and event["max_abs"] == 0
            assert 0 <= event["cache_shapes"] <= config["max_shapes"]
            assert 0 <= event["cache_bytes"] <= config["max_bytes"]
            assert sum(e["bytes"] for e in event["entries"]) == event["cache_bytes"]
            assert len(event["entries"]) == event["cache_shapes"]
        total += len(events)
    for name in ("cross_mode_eviction", "resident_byte_limit"):
        record = report["policies"][name]
        assert record["stats"]["evictions"] == 1
        assert record["stats"]["cooldown"] == 1
        assert sum(e["retired_with_live_references"] for e in record["events"]) == 1
        assert record["events"][-1]["stats_delta"]["replays"] == 1
    for name in ("capture_count_limit", "capture_time_limit"):
        record = report["policies"][name]
        assert record["stats"]["captures"] == 1
        assert record["stats"]["capture_budget"] == 1
        assert record["events"][1]["mode"] == "rule-bucket"
        assert record["events"][1]["stats_delta"]["captures"] == 0
    assert report["policies"]["oversize_candidates"]["stats"]["memory_budget"] == 2
    assert report["policies"]["oversize_candidates"]["stats"]["captures"] == 0
    assert len(report["responses"]) == 16
    for event in report["responses"]:
        assert event["actual"] == event["expected"]
        assert event["stats_delta"]["requests"] == 1
        clean_stats(event["stats_delta"])
    assert len(report["response_logit_checks"]) == 80
    for event in report["response_logit_checks"]:
        assert event["equal"] and event["max_abs"] == 0
    clean_stats(report["response_stats"])
    assert report["response_stats"]["rejected"] == 0
    cleanup(report["response_cleanup"])
    return total + len(report["response_logit_checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--kind", choices=("baseline", "shared"), required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    provenance(report, Path(__file__).resolve().parents[2])
    if args.kind == "baseline":
        from verify_graph_buckets import verify

        verify(args.report, require_replay=True)
        print("controlled policy comparison verified:", baseline(report), "predictions")
    else:
        print(
            "shared resource comparison verified:",
            shared(report),
            "complete-logit comparisons",
        )


if __name__ == "__main__":
    main()
