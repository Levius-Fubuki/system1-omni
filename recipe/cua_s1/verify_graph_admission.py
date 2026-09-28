"""Independently recompute admission benchmark evidence, using only the stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(eager, graph, captures):
    require(
        all(math.isfinite(v) and v > 0 for v in eager + graph),
        "timings must be finite and positive",
    )
    balances = [sum(eager[:i]) - sum(graph[:i]) for i in range(1, len(eager) + 1)]
    first_capture = next((i for i, v in enumerate(captures) if v), None)
    crossing = next((i + 1 for i, v in enumerate(balances) if v >= 0), None)
    sustained = next(
        (
            i + 1
            for i in range(len(balances))
            if first_capture is not None
            and i >= first_capture
            and min(balances[i:]) >= 0
        ),
        None,
    )
    result = {
        "eager_total_ms": sum(eager),
        "graph_total_ms": sum(graph),
        "total_reduction_percent": 100 * (1 - sum(graph) / sum(eager)),
        "first_cumulative_crossing_request": crossing,
        "post_capture_sustained_break_even_request": sustained,
    }
    for name, values in [("eager", eager), ("graph", graph)]:
        result[name + "_p50_ms"] = statistics.median(values)
        result[name + "_p95_ms"] = sorted(values)[math.ceil(len(values) * 0.95) - 1]
    return result


def verify_report(report):
    require(report["status"] == "complete", "report incomplete")
    source = report["repository"]
    require(
        source.get("dirty") is False and len(source.get("revision", "")) == 40,
        "clean source SHA required",
    )
    require(report["schema_version"] == 1, "unknown schema")
    require(len(report["legacy"]["sha256"]) == 64, "legacy hash required")
    require(report["environment"], "environment missing")
    fixture = report["fixture"]
    require(
        hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
        == report["fixture_sha256"],
        "fixture hash mismatch",
    )
    expected = {
        "hot_four": [1, 2, 4, 8] * 12,
        "churn_twelve": list(range(1, 13)) * 6,
        "hot_cold": [v for i in range(8) for v in [1, 2, 4, 8, 13 + i]],
        "shifting_hot": list(range(1, 5)) * 8
        + list(range(5, 9)) * 8
        + list(range(9, 13)) * 8
        + list(range(1, 5)) * 8,
    }
    config = report["config"]
    legacy_config = {
        "max_shapes": 8,
        "max_bytes": 1024**3,
        "min_uses": 2,
        "max_tokens": 2048,
    }
    admission_config = dict(
        legacy_config,
        admission_window=8,
        cooldown_requests=32,
        capture_window=32,
        max_captures=4,
        capture_budget_ms=2000,
    )
    require(config["legacy"] == legacy_config, "legacy configuration mismatch")
    require(config["admission"] == admission_config, "admission configuration mismatch")
    require(
        set(report["workloads"]) == set(config["cases"]) and bool(config["cases"]),
        "selected workloads mismatch",
    )
    require(config["runs"] > 0, "runs must be positive")
    tokens = report["token_counts"]
    require(
        len(set(tokens.values())) == len(tokens)
        and all(isinstance(v, int) and 0 < v < 2048 for v in tokens.values()),
        "invalid token counts",
    )
    for name, runs in report["workloads"].items():
        require(
            name in expected and len(runs) == config["runs"], "invalid workload runs"
        )
        for run_index, record in enumerate(runs):
            require(
                record["run"] == run_index + 1 and record["status"] == "complete",
                "invalid run",
            )
            require(
                record["schedule"] == expected[name]
                and len(record["events"]) == len(expected[name]),
                "schedule mismatch",
            )
            for i, event in enumerate(record["events"]):
                count = expected[name][i]
                require(
                    event["index"] == i + 1
                    and event["goal_repetitions"] == count
                    and event["tokens_per_question"] == tokens[str(count)],
                    "event identity mismatch",
                )
                order = ["eager", "legacy", "admission"]
                require(
                    event["order"] == (order[::-1] if (i + run_index) % 2 else order),
                    "execution order mismatch",
                )
                variants = event["variants"]
                require(set(variants) == set(order), "variant missing")
                for variant, data in variants.items():
                    require(
                        data["response"] == variants["eager"]["response"],
                        "full response mismatch",
                    )
                    require(
                        math.isfinite(data["latency_ms"]) and data["latency_ms"] > 0,
                        "timings must be finite and positive",
                    )
                    for key in [
                        "cache_shapes",
                        "cache_bytes",
                        "allocated_bytes",
                        "reserved_bytes",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                    ]:
                        require(
                            isinstance(data[key], int) and data[key] >= 0,
                            "invalid memory counter",
                        )
                    require(
                        data["cache_shapes"]
                        <= (config[variant]["max_shapes"] if variant != "eager" else 0),
                        "cache shape limit exceeded",
                    )
                    require(
                        data["cache_bytes"]
                        <= (config[variant]["max_bytes"] if variant != "eager" else 0),
                        "cache byte limit exceeded",
                    )
                    if variant != "eager":
                        require(
                            data["max_probability_difference"] == 0,
                            "probabilities changed",
                        )
                        for k, v in data["stats_delta"].items():
                            require(
                                math.isfinite(v) and v >= 0, "invalid stats delta " + k
                            )
            for variant in ["legacy", "admission"]:
                initial = record["stats_initial"][variant]
                final = record["stats_final"][variant]
                require(
                    set(initial) == set(final)
                    and all(v == 0 for v in initial.values()),
                    "run must start with cold counters",
                )
                for event in record["events"]:
                    require(
                        set(event["variants"][variant]["stats_delta"]) == set(initial),
                        "stats keys mismatch",
                    )
                for key in initial:
                    total = initial[key] + sum(
                        e["variants"][variant]["stats_delta"][key]
                        for e in record["events"]
                    )
                    require(
                        math.isfinite(final[key])
                        and math.isclose(total, final[key], rel_tol=1e-9, abs_tol=1e-6),
                        "stats total mismatch " + key,
                    )
                calculated = metrics(
                    [e["variants"]["eager"]["latency_ms"] for e in record["events"]],
                    [e["variants"][variant]["latency_ms"] for e in record["events"]],
                    [
                        e["variants"][variant]["stats_delta"]["captures"]
                        for e in record["events"]
                    ],
                )
                saved = record["summary"][variant]
                require(set(saved) == set(calculated), "summary keys mismatch")
                for key, value in calculated.items():
                    require(
                        saved[key] == value
                        if value is None or key.endswith("_request")
                        else math.isclose(
                            saved[key], value, rel_tol=1e-9, abs_tol=1e-6
                        ),
                        "summary mismatch " + key,
                    )
                print(
                    f"{name} run {run_index + 1} {variant}: {calculated['graph_total_ms']:.2f} ms vs eager {calculated['eager_total_ms']:.2f} ms; reduction {calculated['total_reduction_percent']:.2f}%; p95 {calculated['graph_p95_ms']:.2f} ms; sustained {calculated['post_capture_sustained_break_even_request']}"
                )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    verify_report(json.loads(args.report.read_text()))


if __name__ == "__main__":
    main()
