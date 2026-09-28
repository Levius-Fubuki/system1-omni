"""Measure cold and warm segmented Graph behavior on changing prompt lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from itertools import chain
from pathlib import Path


def schedules():
    """Exercise a fitting hot set and a working set larger than the cache."""
    return {
        "hot_four": [1, 2, 4, 8] * 6,
        "churn_twelve": list(range(1, 13)) * 3,
    }


def summarize_events(events):
    eager_total = sum(item["eager_ms"] for item in events)
    graph_total = sum(item["graph_ms"] for item in events)
    eager_so_far = graph_so_far = 0.0
    break_even = None
    for index, item in enumerate(events, 1):
        eager_so_far += item["eager_ms"]
        graph_so_far += item["graph_ms"]
        if break_even is None and graph_so_far <= eager_so_far:
            break_even = index
    return {
        "eager_total_ms": eager_total,
        "graph_total_ms": graph_total,
        "total_reduction_percent": 100 * (1 - graph_total / eager_total),
        "eager_p50_ms": statistics.median(x["eager_ms"] for x in events),
        "graph_p50_ms": statistics.median(x["graph_ms"] for x in events),
        "first_break_even_request": break_even,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--graph-min-uses", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.runs, args.graph_min_uses) <= 0:
        parser.error("--runs and --graph-min-uses must be positive")
    if (args.output / "report.json").exists():
        parser.error("report already exists; choose a fresh output directory")
    return args


def main(argv=None):
    args = parse_args(argv)

    import torch
    from benchmark_multimodal_graph import measure, response_difference
    from evaluate_multimodal import environment
    from profile_multimodal import (
        GOAL,
        case_matrix,
        fixture,
        repository_state,
        write_json,
    )

    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "environment": environment(),
        "config": {
            "runs": args.runs,
            "graph": {
                "max_shapes": 8,
                "max_bytes": 1024**3,
                "min_uses": args.graph_min_uses,
                "max_tokens": 2048,
            },
            "timing": "synchronized engine.predict, including first use and capture",
            "scope": "two questions per request, one 320x240 synthetic image, serial, no HTTP",
        },
        "workloads": {},
    }
    write_json(report_path, report)
    try:
        engine = MultimodalEngine(
            str(args.weights / "Qwen3.5-4B"),
            str(args.weights / "cua-s1-4b-0.2/multimodal"),
        )
        case = next(x for x in case_matrix() if x["id"] == "320x240-short-q2")
        raw = fixture(case, args.output / "fixtures")
        base = parse_request(raw)
        repetitions = sorted(set(chain.from_iterable(schedules().values())))
        requests = {
            count: replace(
                base,
                questions=tuple(
                    replace(question, goal=" ".join([GOAL] * count))
                    for question in base.questions
                ),
            )
            for count in repetitions
        }
        token_counts = {
            count: int(
                engine.prepare_reused(request.image, request.questions)[0][
                    "input_ids"
                ].shape[-1]
            )
            for count, request in requests.items()
        }
        if (
            len(set(token_counts.values())) != len(token_counts)
            or max(token_counts.values()) > 2048
        ):
            raise ValueError("workload lengths must be distinct and Graph-supported")
        report["fixture_sha256"] = hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest()
        report["token_counts"] = token_counts

        # Initialize kernels and the vision encoder without creating Graph entries.
        engine.graph_runtime = None
        for _ in range(3):
            engine.predict(requests[1])
        torch.cuda.synchronize()
        for workload_name, schedule in schedules().items():
            report["workloads"][workload_name] = []
            for run_index in range(args.runs):
                runtime = GraphRuntime(
                    engine.model, GraphConfig(min_uses=args.graph_min_uses)
                )
                record = {
                    "run": run_index + 1,
                    "schedule": schedule,
                    "events": [],
                    "status": "running",
                }
                report["workloads"][workload_name].append(record)
                write_json(report_path, report)
                for index, count in enumerate(schedule):
                    request = requests[count]
                    before = dict(runtime.stats)
                    samples = {}
                    results = {}
                    order = ["eager", "graph"]
                    if (run_index + index) % 2:
                        order.reverse()
                    for variant in order:
                        engine.graph_runtime = None if variant == "eager" else runtime
                        if variant == "graph":
                            torch.cuda.reset_peak_memory_stats()
                        results[variant], samples[variant] = measure(
                            lambda request=request: engine.predict(request)
                        )
                        if variant == "graph":
                            peak_allocated = torch.cuda.max_memory_allocated()
                            peak_reserved = torch.cuda.max_memory_reserved()
                            allocated = torch.cuda.memory_allocated()
                            reserved = torch.cuda.memory_reserved()
                    difference = response_difference(results["eager"], results["graph"])
                    if difference != 0:
                        raise ValueError(
                            f"{workload_name} run {run_index + 1} request {index + 1}: response changed"
                        )
                    delta = {
                        name: runtime.stats[name] - value
                        for name, value in before.items()
                    }
                    event = {
                        "index": index + 1,
                        "goal_repetitions": count,
                        "tokens_per_question": token_counts[count],
                        "order": order,
                        "eager_ms": samples["eager"],
                        "graph_ms": samples["graph"],
                        "max_probability_difference": difference,
                        "graph_stats_delta": delta,
                        "cache_shapes": len(runtime.cache),
                        "cache_bytes": runtime.cache.bytes,
                        "allocated_bytes": allocated,
                        "reserved_bytes": reserved,
                        "peak_allocated_bytes": peak_allocated,
                        "peak_reserved_bytes": peak_reserved,
                    }
                    record["events"].append(event)
                    write_json(report_path, report)
                record["summary"] = summarize_events(record["events"])
                record["stats_final"] = dict(runtime.stats)
                record["status"] = "complete"
                engine.graph_runtime = None
                runtime.invalidate()
                del runtime
                torch.cuda.empty_cache()
                write_json(report_path, report)
                print(workload_name, run_index + 1, record["summary"], flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(report_path, report)


if __name__ == "__main__":
    main()
