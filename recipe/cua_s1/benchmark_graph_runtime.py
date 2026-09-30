"""Validate and time the serving worker's segmented CUDA Graph runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from benchmark_image_reuse import distinct_fixture
from benchmark_multimodal_graph import (
    changed_text_request,
    measure,
    response_difference,
    summarize,
)
from profile_multimodal import case_matrix, fixture, repository_state, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", action="append", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--graph-max-tokens", type=int, default=2048)
    args = p.parse_args(argv)
    if min(args.warmup, args.runs, args.iterations, args.graph_max_tokens) <= 0:
        p.error("run limits must be positive")
    known = {case["id"] for case in case_matrix()} | {"640x480-distinct-q8"}
    if len(args.case) != len(set(args.case)) or set(args.case) - known:
        p.error("duplicate or unknown case")
    if (args.output / "report.json").exists():
        p.error("report already exists; select a fresh directory")

    import torch
    from evaluate_multimodal import environment

    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "environment": environment(),
        "config": vars(args)
        | {"weights": str(args.weights), "output": str(args.output)},
        "cases": [],
    }
    write_json(report_path, report)
    try:
        engine = MultimodalEngine(
            str(args.weights / "Qwen3.5-4B"),
            str(args.weights / "cua-s1-4b-0.2/multimodal"),
        )
        selected = case_matrix() + [{"id": "640x480-distinct-q8"}]
        for case in selected:
            if case["id"] not in args.case:
                continue
            raw = (
                distinct_fixture(args.output / "fixtures")
                if case["id"] == "640x480-distinct-q8"
                else fixture(case, args.output / "fixtures")
            )
            request = parse_request(raw)
            if len(request.questions) == 1:
                raise ValueError(f"{case['id']}: single-question path is outside scope")
            item = {
                "id": case["id"],
                "status": "running",
                "fixture_sha256": hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode()
                ).hexdigest(),
                "runs": [],
            }
            report["cases"].append(item)
            write_json(report_path, report)
            engine.graph_runtime = None
            eager_response = engine.predict(request)
            runtime = GraphRuntime(
                engine.model,
                GraphConfig(max_tokens=args.graph_max_tokens),
            )
            engine.graph_runtime = runtime
            graph_response = engine.predict(request)
            # A second call crosses the default two-use capture threshold.
            graph_response = engine.predict(request)
            item["original_max_probability_difference"] = response_difference(
                eager_response, graph_response
            )
            if item["original_max_probability_difference"] != 0:
                raise ValueError(f"{case['id']}: Graph changed original response")
            from PIL import Image

            changed = replace(
                request, image=Image.new("RGB", request.image.size, "black")
            )
            engine.graph_runtime = None
            changed_eager = engine.predict(changed)
            engine.graph_runtime = runtime
            changed_graph = engine.predict(changed)
            item["changed_image_max_probability_difference"] = response_difference(
                changed_eager, changed_graph
            )
            if item["changed_image_max_probability_difference"] != 0:
                raise ValueError(f"{case['id']}: Graph changed image response")
            if case["id"] == "640x480-distinct-q8":
                # This fixture intentionally has unrelated prompts, so change
                # only its first question while keeping every tensor layout.
                first = request.questions[0]
                changed_text = replace(
                    request,
                    questions=(
                        replace(first, goal=first.goal.replace("Save", "Open")),
                        *request.questions[1:],
                    ),
                )
                old = engine.prepare_reused(request.image, request.questions)
                new = engine.prepare_reused(request.image, changed_text.questions)
                if not all(
                    before["input_ids"].shape == after["input_ids"].shape
                    for before, after in zip(old, new)
                ) or torch.equal(old[0]["input_ids"], new[0]["input_ids"]):
                    raise ValueError("distinct changed text did not retain layouts")
                verb = "Open"
            else:
                changed_text, verb = changed_text_request(engine, request)
            engine.graph_runtime = None
            text_eager = engine.predict(changed_text)
            engine.graph_runtime = runtime
            text_graph = engine.predict(changed_text)
            item["changed_text_verb"] = verb
            item["changed_text_max_probability_difference"] = response_difference(
                text_eager, text_graph
            )
            if item["changed_text_max_probability_difference"] != 0:
                raise ValueError(f"{case['id']}: Graph changed text response")
            item["stats_after_validation"] = dict(runtime.stats)
            item["cache_bytes_after_validation"] = runtime.cache.bytes
            item["cache_shapes_after_validation"] = len(runtime.cache)
            for _ in range(args.warmup):
                engine.graph_runtime = None
                engine.predict(request)
                engine.graph_runtime = runtime
                engine.predict(request)
            for run in range(args.runs):
                samples = {"eager": [], "graph": []}
                for iteration in range(args.iterations):
                    order = ["eager", "graph"]
                    if (run + iteration) % 2:
                        order.reverse()
                    for name in order:
                        engine.graph_runtime = None if name == "eager" else runtime
                        result, elapsed = measure(lambda: engine.predict(request))
                        if response_difference(eager_response, result) != 0:
                            raise ValueError(
                                f"{case['id']}: timed {name} response changed"
                            )
                        samples[name].append(elapsed)
                item["runs"].append(
                    {name: summarize(values) for name, values in samples.items()}
                )
                write_json(report_path, report)
            item["stats_final"] = dict(runtime.stats)
            item["cache_bytes_final"] = runtime.cache.bytes
            item["cache_shapes_final"] = len(runtime.cache)
            item["status"] = "complete"
            engine.graph_runtime = None
            runtime.invalidate()
            del runtime
            torch.cuda.empty_cache()
            write_json(report_path, report)
            print(
                case["id"],
                [
                    (run["eager"]["p50_ms"], run["graph"]["p50_ms"])
                    for run in item["runs"]
                ],
                flush=True,
            )
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
