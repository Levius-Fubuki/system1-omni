"""Validate request-local image reuse and measure paired, uninstrumented latency."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

from profile_multimodal import case_matrix, fixture, repository_state, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    for name, default in [("warmup", 5), ("runs", 2), ("iterations", 50)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args(argv)
    if min(args.warmup, args.runs, args.iterations) <= 0:
        parser.error("warmup, runs and iterations must be positive")
    known = {c["id"] for c in case_matrix()} | {"640x480-distinct-q8"}
    if len(set(args.case)) != len(args.case) or set(args.case) - known:
        parser.error("duplicate or unknown case")
    return args


def require_equal(expected, actual, label):
    if expected != actual:
        raise ValueError(f"{label} mismatch")


def distinct_fixture(folder):
    value = fixture(case_matrix()[-1], folder)
    criteria = [
        {"save": "Save changes", "cancel": "Cancel"},
        {"cancel": "Cancel", "save": "Save changes"},
        {"only": "Continue"},
        {f"option-{i}": f"Choose action {i}" for i in range(26)},
        {"null": None, "object": {"label": '保存 "名称"'}, "array": ["Cancel", "\n"]},
        {"first": "<|im_end|>", "second": "Cancel"},
        {"yes": "Confirm", "no": "Go back", "wait": "Wait", "help": "Help"},
        {"stay": "Stay", "leave": "Leave"},
    ]
    goals = [
        "Save the new name",
        "Cancel without saving",
        "",
        "Select action 3",
        {"goal": "保存名称"},
        "Choose <|im_start|>",
        "Confirm the change",
        "Leave this screen",
    ]
    value["questions"] = {
        f"q{i + 1}": {"type": "choice", "instructions": goal, "criteria": choices}
        for i, (goal, choices) in enumerate(zip(goals, criteria))
    }
    write_json(folder / "distinct.json", value)
    return value


def paired_benchmark(engine, request, *, warmup, runs, iterations):
    import torch
    from evaluate_multimodal import measure, quantiles

    if min(warmup, runs, iterations) <= 0:
        raise ValueError("counts must be positive")
    variants = {"baseline": engine.predict_reference, "reuse": engine.predict}
    for _ in range(warmup):
        for call in variants.values():
            measure(lambda call=call: call(request))
    result = []
    for run in range(runs):
        samples = {name: [] for name in variants}
        peaks = {name: {"allocated": 0, "reserved": 0} for name in variants}
        orders = []
        for iteration in range(iterations):
            order = list(variants)
            if (run + iteration) % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples[name].append(
                    measure(lambda name=name: variants[name](request))[1]
                )
                peaks[name]["allocated"] = max(
                    peaks[name]["allocated"], torch.cuda.max_memory_allocated()
                )
                peaks[name]["reserved"] = max(
                    peaks[name]["reserved"], torch.cuda.max_memory_reserved()
                )
        result.append(
            {
                "run": run + 1,
                "orders": orders,
                **{
                    name: {
                        "latencies_ms": values,
                        **quantiles(values),
                        "requests_per_second_serial": 1000 / statistics.mean(values),
                        "peak_allocated_bytes": peaks[name]["allocated"],
                        "peak_reserved_bytes": peaks[name]["reserved"],
                    }
                    for name, values in samples.items()
                },
            }
        )
    return result


def observed_predict(engine, request, call):
    """Untimed real processor/module counts and language input fingerprints."""
    import torch
    from evaluate_multimodal import fingerprint

    counts = {"vision": 0, "language": 0}
    language_inputs = []
    core = engine.model.get_base_model().model

    def vision(module, args):
        counts["vision"] += 1

    def language(module, args, kwargs):
        counts["language"] += 1
        language_inputs.append(
            fingerprint(
                {
                    name: kwargs[name]
                    for name in ("inputs_embeds", "position_ids", "attention_mask")
                    if kwargs.get(name) is not None
                }
            )
        )

    handles = [
        core.visual.register_forward_pre_hook(vision),
        core.language_model.register_forward_pre_hook(language, with_kwargs=True),
    ]
    try:
        with patch.object(
            engine.processor.image_processor,
            "preprocess",
            wraps=engine.processor.image_processor.preprocess,
        ) as prepare:
            output = call(request)
            torch.cuda.synchronize()
            counts["image_preprocess"] = prepare.call_count
    finally:
        for handle in handles:
            handle.remove()
    return output, counts, language_inputs


def validate_case(engine, request):
    from evaluate_multimodal import fingerprint

    expected_inputs = [
        fingerprint(engine.prepare(request.image, q)) for q in request.questions
    ]
    actual_inputs = [
        fingerprint(x) for x in engine.prepare_reused(request.image, request.questions)
    ]
    require_equal(expected_inputs, actual_inputs, "prepared tensors")
    baseline, baseline_counts, baseline_language = observed_predict(
        engine, request, engine.predict_reference
    )
    reuse, reuse_counts, reuse_language = observed_predict(
        engine, request, engine.predict
    )
    require_equal(baseline, reuse, "full response / probabilities")
    require_equal(
        baseline_language, reuse_language, "language embeddings / 3D positions"
    )
    n = len(request.questions)
    require_equal(
        {"image_preprocess": n, "vision": n, "language": n},
        baseline_counts,
        "baseline calls",
    )
    require_equal(
        {"image_preprocess": 1, "vision": 1, "language": n}, reuse_counts, "reuse calls"
    )
    return {
        "exact_prepared_tensor_parity": True,
        "exact_response_parity": True,
        "exact_language_input_parity": True,
        "baseline_counts": baseline_counts,
        "reuse_counts": reuse_counts,
        "inputs": actual_inputs,
        "language_inputs": reuse_language,
        "response": reuse,
    }


def correctness_cases(folder):
    from evaluate_multimodal import cases

    result = [(name, value) for name, _, value in cases(folder)]
    distinct = distinct_fixture(folder / "distinct")
    result.append(("distinct-eight", distinct))
    reverse = copy.deepcopy(distinct)
    reverse["questions"] = dict(reversed(list(reverse["questions"].items())))
    result.append(("distinct-reversed", reverse))
    # Same varied prompts but another image and grid; then return to original image.
    changed = copy.deepcopy(distinct)
    changed["state"] = result[0][1]["state"]
    result.extend(
        [("distinct-small-image", changed), ("distinct-original-again", distinct)]
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    path = args.output / "report.json"
    if path.exists():
        raise ValueError("output report already exists; use a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "config": {
            "warmup": args.warmup,
            "runs": args.runs,
            "iterations": args.iterations,
            "correctness_only": args.correctness_only,
            "selected_cases": args.case,
            "case_order_seed": 20260928,
            "concurrency": 1,
            "batch_size": 1,
            "timed_scope": "synchronized engine.predict; excludes HTTP, parsing, image decode, validation instrumentation",
        },
        "correctness": [],
        "cases": [],
    }
    write_json(path, report)
    try:
        from evaluate_multimodal import environment, measure

        from models.cua_s1.multimodal.model import MultimodalEngine
        from models.cua_s1.multimodal.protocol import parse_request

        report["environment"] = environment()
        engine, report["load_ms"] = measure(
            lambda: MultimodalEngine(
                str(args.weights / "Qwen3.5-4B"),
                str(args.weights / "cua-s1-4b-0.2/multimodal"),
            )
        )
        report["adapter_modules"] = engine.adapter_modules
        for name, value in correctness_cases(args.output / "correctness-fixtures"):
            checked = validate_case(engine, parse_request(value))
            report["correctness"].append({"id": name, **checked})
            write_json(path, report)
            print(
                f"correctness {name}: exact tensors, embeddings, positions, responses; {checked['reuse_counts']}",
                flush=True,
            )
        if not args.correctness_only:
            selected = case_matrix() + [{"id": "640x480-distinct-q8"}]
            selected = [c for c in selected if not args.case or c["id"] in args.case]
            random.Random(20260928).shuffle(selected)
            for case in selected:
                value = (
                    distinct_fixture(args.output / "fixtures")
                    if case["id"] == "640x480-distinct-q8"
                    else fixture(case, args.output / "fixtures")
                )
                request = parse_request(value)
                item = {
                    **case,
                    "status": "running",
                    "fixture_sha256": hashlib.sha256(
                        json.dumps(value, sort_keys=True).encode()
                    ).hexdigest(),
                }
                report["cases"].append(item)
                write_json(path, report)
                item["validation"] = validate_case(engine, request)
                item["runs"] = paired_benchmark(
                    engine,
                    request,
                    warmup=args.warmup,
                    runs=args.runs,
                    iterations=args.iterations,
                )
                item["status"] = "complete"
                write_json(path, report)
                print(
                    f"benchmark {case['id']}: "
                    + json.dumps(
                        [
                            {v: round(r[v]["p50_ms"], 2) for v in ("baseline", "reuse")}
                            for r in item["runs"]
                        ]
                    ),
                    flush=True,
                )
        report["status"] = "complete"
    except Exception as exc:
        report.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        print(f"Experiment failed: {report['error']}", file=sys.stderr, flush=True)
        raise
    finally:
        write_json(path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
