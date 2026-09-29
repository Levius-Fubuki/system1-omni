"""Capture-inclusive eager / exact / bucket experiment on the pinned model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path


def schedules():
    return {
        "probe": list(range(1, 13)) * 2,
        "hot_four": [1, 2, 4, 8] * 12,
        "churn_twelve": list(range(1, 13)) * 6,
        "hot_cold": [v for i in range(8) for v in [1, 2, 4, 8, 13 + i]],
        "shifting_hot": [
            v
            for group in [range(1, 5), range(5, 9), range(9, 13), range(1, 5)]
            for v in list(group) * 8
        ],
    }


def variant_configs(kind, width, tuned_exact_window=None):
    from models.cua_s1.multimodal.graph_runtime import GraphConfig

    config = GraphConfig()
    configs = {
        "exact": config,
        "bucket": replace(config, mode="rule-bucket", bucket_width=width)
        if kind == "worker"
        else config,
    }
    if tuned_exact_window is not None:
        configs["exact_tuned"] = replace(config, admission_window=tuned_exact_window)
    return configs


def summary(events, variant):
    samples = [event["variants"][variant]["latency_ms"] for event in events]
    ordered = sorted(samples)
    eager_total = sum(e["variants"]["eager"]["latency_ms"] for e in events)
    return {
        "requests": len(samples),
        "total_ms": sum(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[math.ceil(len(samples) * 0.95) - 1],
        "total_reduction_vs_eager_pct": 100 * (1 - sum(samples) / eager_total),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--tuned-exact-window", type=int)
    p.add_argument("--kind", choices=("segment", "rule", "worker"), default="segment")
    p.add_argument(
        "--include-recipe",
        action="store_true",
        help="also time the prior recipe algorithm with current shared runtime",
    )
    p.add_argument("--case", choices=list(schedules()), action="append")
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output directory")
    if args.include_recipe and args.kind != "worker":
        p.error("--include-recipe requires --kind worker")
    if args.runs < 1:
        p.error("runs must be positive")
    if args.tuned_exact_window is not None and args.tuned_exact_window < 1:
        p.error("tuned exact window must be positive")

    import torch
    from benchmark_multimodal_graph import measure
    from evaluate_multimodal import environment
    from graph_buckets import BucketRuntime, RuleBucketRuntime, bucket_length
    from profile_multimodal import (
        GOAL,
        case_matrix,
        fixture,
        repository_state,
        write_json,
    )

    from models.cua_s1.multimodal.graph_runtime import GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    bucket_length(1, args.width)
    root = Path(__file__).resolve().parents[2]
    source = repository_state(root)
    if source["dirty"] or not source["revision"]:
        raise ValueError("clean committed source required")
    args.output.mkdir(parents=True)
    report_path = args.output / "report.json"
    selected = {k: v for k, v in schedules().items() if k in (args.case or ["probe"])}
    configs = variant_configs(args.kind, args.width, args.tuned_exact_window)
    config = configs["exact"]
    bucket_class = RuleBucketRuntime if args.kind == "rule" else BucketRuntime
    if args.kind == "worker":
        from models.cua_s1.multimodal.graph_buckets import (
            RuleBucketRuntime as WorkerRuntime,
        )

        def bucket_class(model, config, width):
            return WorkerRuntime(
                model, replace(config, mode="rule-bucket", bucket_width=width)
            )

    variants = (
        ("eager", "exact", "bucket", "recipe")
        if args.include_recipe
        else ("eager", "exact", "bucket")
    )
    if args.tuned_exact_window is not None:
        variants += ("exact_tuned",)
    if args.include_recipe:
        configs["recipe"] = config
    report = {
        "status": "running",
        "repository": source,
        "environment": environment(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).with_name("graph_buckets.py"),
                Path(__file__).with_name("benchmark_multimodal_graph.py"),
                Path(__file__).with_name("profile_multimodal.py"),
                Path(__file__).with_name("evaluate_multimodal.py"),
                *(
                    root / "src/models/cua_s1/multimodal" / name
                    for name in (
                        "graph_runtime.py",
                        "graph_admission.py",
                        "graph_buckets.py",
                        "rule_prefill.py",
                        "model.py",
                    )
                ),
            ]
        },
        "config": {
            "graph": asdict(config),
            "variant_graph": {k: asdict(v) for k, v in configs.items()},
            "schedules": selected,
            "variants": variants,
            "bucket_graph": asdict(
                replace(config, mode="rule-bucket", bucket_width=args.width)
            )
            if args.kind == "worker"
            else asdict(config),
            "width": args.width,
            "kind": args.kind,
            "runs": args.runs,
            "timing": "synchronized predict including cold capture and first-length checks",
            "memory": "paired caches coexist; allocator totals are combined, cache bytes are per variant",
            "scope": "serial, two identical questions, synthetic 320x240 image, no HTTP",
            "order": "all variant permutations cycle across requests, offset by half a cycle across runs",
        },
        "workloads": {},
    }
    write_json(report_path, report)
    try:
        engine = MultimodalEngine(
            str(args.weights / "Qwen3.5-4B"),
            str(args.weights / "cua-s1-4b-0.2/multimodal"),
        )
        case = next(c for c in case_matrix() if c["id"] == "320x240-short-q2")
        raw = fixture(case, args.output / "fixtures")
        report["fixture"] = raw
        base = parse_request(raw)
        counts = sorted({n for s in selected.values() for n in s})
        requests = {
            n: replace(
                base,
                questions=tuple(
                    replace(q, goal=" ".join([GOAL] * n)) for q in base.questions
                ),
            )
            for n in counts
        }
        tokens = {
            n: int(
                engine.prepare_reused(r.image, r.questions)[0]["input_ids"].shape[-1]
            )
            for n, r in requests.items()
        }
        report["tokens"] = tokens
        report["buckets"] = {n: bucket_length(v, args.width) for n, v in tokens.items()}
        print("lengths", tokens, "buckets", report["buckets"], flush=True)
        for _ in range(3):
            engine.predict(requests[counts[0]])
        orders = list(itertools.permutations(variants))
        for name, schedule in selected.items():
            report["workloads"][name] = []
            for run in range(args.runs):
                runtimes = {
                    "exact": GraphRuntime(engine.model, config),
                    "bucket": bucket_class(engine.model, config, width=args.width),
                }
                if args.tuned_exact_window is not None:
                    runtimes["exact_tuned"] = GraphRuntime(
                        engine.model, configs["exact_tuned"]
                    )
                if args.include_recipe:
                    runtimes["recipe"] = RuleBucketRuntime(
                        engine.model, config, width=args.width
                    )
                record = {"run": run + 1, "status": "running", "events": []}
                report["workloads"][name].append(record)
                for index, n in enumerate(schedule):
                    order = orders[(index + run * (len(orders) // 2)) % len(orders)]
                    event = {
                        "index": index,
                        "goal_repetitions": n,
                        "tokens": tokens[n],
                        "bucket": bucket_length(tokens[n], args.width),
                        "order": order,
                        "variants": {},
                    }
                    record["events"].append(event)
                    for variant in order:
                        runtime = runtimes.get(variant)
                        engine.graph_runtime = runtime
                        before = dict(runtime.stats) if runtime else {}
                        torch.cuda.reset_peak_memory_stats()
                        response, elapsed = measure(
                            lambda request=requests[n]: engine.predict(request)
                        )
                        event["variants"][variant] = {
                            "response": response,
                            "latency_ms": elapsed,
                            "stats_delta": {
                                k: runtime.stats[k] - v for k, v in before.items()
                            }
                            if runtime
                            else {},
                            "cache_shapes": len(runtime.cache) if runtime else 0,
                            "cache_bytes": runtime.cache.bytes if runtime else 0,
                            "reserved_bytes": torch.cuda.memory_reserved(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                    assert all(
                        v["response"] == event["variants"]["eager"]["response"]
                        for v in event["variants"].values()
                    ), (name, run, index)
                    write_json(report_path, report)
                    if index % 12 == 11:
                        print(
                            name,
                            run + 1,
                            index + 1,
                            {v: dict(r.stats) for v, r in runtimes.items()},
                            flush=True,
                        )
                record["summary"] = {v: summary(record["events"], v) for v in variants}
                record["stats_final"] = {v: dict(r.stats) for v, r in runtimes.items()}
                record["status"] = "complete"
                engine.graph_runtime = None
                for runtime in runtimes.values():
                    runtime.close()
                del runtime, runtimes
                gc.collect()
                torch.cuda.empty_cache()
                write_json(report_path, report)
                print(name, run + 1, record["summary"], flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(report_path, report)


if __name__ == "__main__":
    main()
