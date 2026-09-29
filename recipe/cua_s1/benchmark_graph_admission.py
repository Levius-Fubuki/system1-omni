"""Paired, capture-inclusive eager / historical / admission Graph experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

from benchmark_graph_mixed_shapes import summarize_events as mixed_summary


def schedules():
    return {
        "hot_four": [1, 2, 4, 8] * 12,
        "churn_twelve": list(range(1, 13)) * 6,
        "hot_cold": [v for i in range(8) for v in [1, 2, 4, 8, 13 + i]],
        "shifting_hot": [
            v
            for group in [range(1, 5), range(5, 9), range(9, 13), range(1, 5)]
            for v in list(group) * 8
        ],
    }


def summarize_events(events):
    result = mixed_summary(events)
    for variant in ("eager", "graph"):
        samples = sorted(e[f"{variant}_ms"] for e in events)
        result[f"{variant}_p95_ms"] = samples[math.ceil(0.95 * len(samples)) - 1]
    return result


def verify_legacy_source(path, revision, root):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "legacy revision must be a full 40-character lowercase Git SHA"
        )
    expected = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            revision + ":src/models/cua_s1/multimodal/graph_runtime.py",
        ]
    )
    if path.read_bytes() != expected:
        raise ValueError("legacy runtime bytes do not match the declared Git revision")


def configure_legacy_execution(module, current, *, unpatched=False):
    if unpatched:
        return "none; unpatched historical runtime diagnostic"
    module._GraphSegment = current._GraphSegment
    module._ShapeEntry = current._ShapeEntry
    module.GraphRuntime._run_segments = current.GraphRuntime._run_segments
    return "current _GraphSegment, _ShapeEntry, and GraphRuntime._run_segments; shared per-shape pool, owned stream, and reserved-memory accounting; historical admission/cache policy only"


def load_legacy(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "_cua_legacy_graph_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations through sys.modules.
    spec.loader.exec_module(module)
    return module, digest


def legacy_runtime(module, model):
    runtime = module.GraphRuntime(model, module.GraphConfig())
    runtime.request = nullcontext
    return runtime


def check_responses(expected, actual, difference):
    delta = difference(expected, actual)
    if delta != 0 or expected != actual:
        raise ValueError("full response changed")
    return delta


def variant_order(run_index, request_index):
    order = ["eager", "legacy", "admission"]
    return order[::-1] if (run_index + request_index) % 2 else order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-runtime", type=Path, required=True)
    parser.add_argument("--legacy-revision", required=True)
    parser.add_argument(
        "--unpatched-legacy",
        action="store_true",
        help="diagnostic only: use historical unsafe capture streams",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--case", choices=list(schedules()), action="append")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.legacy_revision):
        parser.error("--legacy-revision must be a full 40-character lowercase Git SHA")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if (args.output / "report.json").exists():
        parser.error("report exists; choose a fresh directory")
    root = Path(__file__).resolve().parents[2]
    if root in args.legacy_runtime.resolve().parents:
        parser.error("--legacy-runtime must be outside the measured repository")
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

    from models.cua_s1.multimodal import graph_runtime as current_runtime
    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    source = repository_state(Path(__file__).resolve().parents[2])
    if source["dirty"] is not False or not source["revision"]:
        raise ValueError("benchmark requires clean committed source")
    root = Path(__file__).resolve().parents[2]
    verify_legacy_source(args.legacy_runtime, args.legacy_revision, root)
    legacy, digest = load_legacy(args.legacy_runtime)
    override = configure_legacy_execution(
        legacy, current_runtime, unpatched=args.unpatched_legacy
    )
    current_runtime_sha256 = hashlib.sha256(
        (root / "src/models/cua_s1/multimodal/graph_runtime.py").read_bytes()
    ).hexdigest()
    selected = {k: v for k, v in schedules().items() if not args.case or k in args.case}
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "repository": source,
        "environment": environment(),
        "legacy": {
            "sha256": digest,
            "revision": args.legacy_revision,
            "source_bytes_verified_against_git": True,
            "runtime_segment_override": override,
            "current_graph_runtime_sha256": current_runtime_sha256,
        },
        "comparison_kind": "unpatched-legacy-diagnostic"
        if args.unpatched_legacy
        else "legacy-policy-with-execution-fix",
        "config": {
            "runs": args.runs,
            "cases": list(selected),
            "legacy": asdict(legacy.GraphConfig()),
            "admission": asdict(GraphConfig()),
            "timing": "synchronized engine.predict; includes cold admission and capture",
            "scope": "two identical questions; one synthetic 320x240 image; serial; no HTTP",
            "memory_limitation": "legacy and admission graph pools coexist; allocator totals include both pools and are not isolated variant memory measurements",
            "order": "reverse eager/legacy/admission on alternating requests and runs",
            "p95": "nearest rank",
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
        counts = sorted({v for schedule in selected.values() for v in schedule})
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
        if len(set(tokens.values())) != len(tokens) or max(tokens.values()) >= 2048:
            raise ValueError("token lengths must be distinct and below 2048")
        report["fixture_sha256"] = hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest()
        report["fixture"] = raw
        report["token_counts"] = tokens
        engine.graph_runtime = None
        for _ in range(3):
            engine.predict(requests[counts[0]])
        torch.cuda.synchronize()
        for name, schedule in selected.items():
            report["workloads"][name] = []
            for run in range(args.runs):
                runtimes = {
                    "legacy": legacy_runtime(legacy, engine.model),
                    "admission": GraphRuntime(engine.model, GraphConfig()),
                }
                record = {
                    "run": run + 1,
                    "schedule": schedule,
                    "status": "running",
                    "events": [],
                    "stats_initial": {v: dict(r.stats) for v, r in runtimes.items()},
                }
                report["workloads"][name].append(record)
                write_json(report_path, report)
                for index, count in enumerate(schedule):
                    event = {
                        "index": index + 1,
                        "goal_repetitions": count,
                        "tokens_per_question": tokens[count],
                        "order": variant_order(run, index),
                        "variants": {},
                    }
                    for variant in event["order"]:
                        runtime = runtimes.get(variant)
                        engine.graph_runtime = runtime
                        before = dict(runtime.stats) if runtime else {}
                        torch.cuda.reset_peak_memory_stats()
                        response, elapsed = measure(
                            lambda request=requests[count]: engine.predict(request)
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
                            "allocated_bytes": torch.cuda.memory_allocated(),
                            "reserved_bytes": torch.cuda.memory_reserved(),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                    for variant in runtimes:
                        event["variants"][variant]["max_probability_difference"] = (
                            check_responses(
                                event["variants"]["eager"]["response"],
                                event["variants"][variant]["response"],
                                response_difference,
                            )
                        )
                    record["events"].append(event)
                    write_json(report_path, report)
                record["summary"] = {
                    v: summarize_events(
                        [
                            {
                                "eager_ms": e["variants"]["eager"]["latency_ms"],
                                "graph_ms": e["variants"][v]["latency_ms"],
                                "graph_stats_delta": e["variants"][v]["stats_delta"],
                            }
                            for e in record["events"]
                        ]
                    )
                    for v in runtimes
                }
                record["stats_final"] = {v: dict(r.stats) for v, r in runtimes.items()}
                record["status"] = "complete"
                engine.graph_runtime = None
                for runtime in runtimes.values():
                    runtime.invalidate()
                del runtime, runtimes
                gc.collect()
                torch.cuda.empty_cache()  # only between cold runs, never paired variants
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
