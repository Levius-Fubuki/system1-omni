"""Compare full and final-token output projection on the reused Cua-S1 path."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from benchmark_image_reuse import distinct_fixture, paired_benchmark
from profile_multimodal import case_matrix, fixture, repository_state, write_json


@contextmanager
def projection_variants(engine):
    """Execute the same reuse path with only the output-head keyword changed."""
    original = engine.model.forward

    def full_forward(*args, **kwargs):
        kwargs.pop("logits_to_keep", None)
        return original(*args, **kwargs)

    def last_forward(*args, **kwargs):
        return original(*args, **kwargs)

    def execute(forward, request):
        engine.model.forward = forward
        return engine.predict(request)

    try:
        yield SimpleNamespace(
            predict_reference=lambda request: execute(full_forward, request),
            predict=lambda request: execute(last_forward, request),
        )
    finally:
        engine.model.forward = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args(argv)
    known = {case["id"] for case in case_matrix()} | {"640x480-distinct-q8"}
    if len(args.case) != len(set(args.case)) or set(args.case) - known:
        parser.error("--case contains duplicate or unknown case ids")
    if min(args.warmup, args.runs, args.iterations) <= 0:
        parser.error("warmup, runs, and iterations must be positive")
    report_path = args.output / "report.json"
    if report_path.exists():
        raise ValueError("output report already exists; use a fresh directory")

    from evaluate_multimodal import environment, measure

    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    report = {
        "schema_version": 1,
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "environment": environment(),
        "config": {
            "warmup": args.warmup,
            "runs": args.runs,
            "iterations": args.iterations,
            "case": args.case,
            "concurrency": 1,
            "timed_scope": "synchronized engine.predict with only logits_to_keep changed; excludes parsing, fixture creation, validation, and hooks",
        },
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    try:
        engine, report["load_ms"] = measure(
            lambda: MultimodalEngine(
                str(args.weights / "Qwen3.5-4B"),
                str(args.weights / "cua-s1-4b-0.2/multimodal"),
            )
        )
        report["adapter_modules"] = engine.adapter_modules
        selected = case_matrix() + [{"id": "640x480-distinct-q8"}]
        selected = [
            case for case in selected if not args.case or case["id"] in args.case
        ]
        for case in selected:
            value = (
                distinct_fixture(args.output / "fixtures")
                if case["id"] == "640x480-distinct-q8"
                else fixture(case, args.output / "fixtures")
            )
            request = parse_request(value)
            item = {
                "id": case["id"],
                "questions": len(request.questions),
                "fixture_sha256": hashlib.sha256(
                    json.dumps(value, sort_keys=True).encode()
                ).hexdigest(),
                "status": "running",
            }
            report["cases"].append(item)
            write_json(report_path, report)
            with projection_variants(engine) as variants:
                shapes = []
                head = engine.model.get_base_model().lm_head
                handle = head.register_forward_pre_hook(
                    lambda module, inputs, shapes=shapes: shapes.append(
                        list(inputs[0].shape)
                    )
                )
                try:
                    full = variants.predict_reference(request)
                    item["full_head_shapes"] = shapes[:]
                    shapes.clear()
                    last = variants.predict(request)
                    item["last_head_shapes"] = shapes[:]
                finally:
                    handle.remove()
                if full != last:
                    raise ValueError(f"{case['id']}: full response differs")
                if len(item["full_head_shapes"]) != len(request.questions) or len(
                    item["last_head_shapes"]
                ) != len(request.questions):
                    raise ValueError(f"{case['id']}: unexpected output-head calls")
                if len(request.questions) > 1 and any(
                    shape[1] != 1 for shape in item["last_head_shapes"]
                ):
                    raise ValueError(f"{case['id']}: output head was not limited")
                item["exact_response_parity"] = True
                runs = paired_benchmark(
                    variants,
                    request,
                    warmup=args.warmup,
                    runs=args.runs,
                    iterations=args.iterations,
                )
            for run in runs:
                run["full_projection"] = run.pop("baseline")
                run["last_token_projection"] = run.pop("reuse")
                run["orders"] = [
                    [
                        "full_projection"
                        if v == "baseline"
                        else "last_token_projection"
                        for v in order
                    ]
                    for order in run["orders"]
                ]
            item.update(status="complete", runs=runs)
            write_json(report_path, report)
            print(
                f"{case['id']}: "
                + json.dumps(
                    [
                        [
                            round(run[v]["p50_ms"], 2)
                            for v in ("full_projection", "last_token_projection")
                        ]
                        for run in runs
                    ]
                ),
                flush=True,
            )
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
