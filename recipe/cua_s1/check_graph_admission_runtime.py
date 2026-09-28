"""GPU regression checks for input changes, admission and eager fallback parity."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output path")

    import torch
    from benchmark_image_reuse import distinct_fixture
    from benchmark_multimodal_graph import changed_text_request
    from evaluate_multimodal import environment
    from PIL import Image
    from profile_multimodal import case_matrix, fixture, repository_state, write_json

    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    source = repository_state(Path(__file__).resolve().parents[2])
    if source["dirty"]:
        raise ValueError("GPU checks require clean committed source")
    report = {
        "status": "running",
        "repository": source,
        "environment": environment(),
        "cases": [],
        "policies": {},
    }
    fixtures = args.output.parent / "parity-fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    engine = MultimodalEngine(
        str(args.weights / "Qwen3.5-4B"), str(args.weights / "cua-s1-4b-0.2/multimodal")
    )
    selected = {
        "320x240-short-q2",
        "320x240-short-q8",
        "640x480-short-q8",
        "640x480-long-q8",
    }
    cases = [c for c in case_matrix() if c["id"] in selected] + [
        {"id": "640x480-distinct-q8"}
    ]
    for case in cases:
        raw = (
            distinct_fixture(fixtures)
            if case["id"] == "640x480-distinct-q8"
            else fixture(case, fixtures)
        )
        request = parse_request(raw)
        # Expanded validation limits ensure all distinct layouts and the long
        # prompt actually replay; performance measurements use worker defaults.
        config = GraphConfig(max_tokens=4096, max_captures=16, capture_budget_ms=10000)
        runtime = GraphRuntime(engine.model, config)
        changed_image = replace(
            request, image=Image.new("RGB", request.image.size, "black")
        )
        if case["id"] == "640x480-distinct-q8":
            changed_text = replace(
                request,
                questions=(
                    replace(
                        request.questions[0],
                        goal=request.questions[0].goal.replace("Save", "Open"),
                    ),
                    *request.questions[1:],
                ),
            )
            old = engine.prepare_reused(request.image, request.questions)
            new = engine.prepare_reused(request.image, changed_text.questions)
            assert all(
                a["input_ids"].shape == b["input_ids"].shape for a, b in zip(old, new)
            )
            assert not torch.equal(old[0]["input_ids"], new[0]["input_ids"])
        else:
            changed_text, _ = changed_text_request(engine, request)
        records = []
        for name, value in [
            ("cold", request),
            ("capture", request),
            ("image", changed_image),
            ("text", changed_text),
        ]:
            engine.graph_runtime = None
            expected = engine.predict(value)
            engine.graph_runtime = runtime
            before = dict(runtime.stats)
            actual = engine.predict(value)
            assert actual == expected, (case["id"], name)
            records.append(
                {
                    "variant": name,
                    "eager_response": expected,
                    "graph_response": actual,
                    "stats_delta": {k: runtime.stats[k] - v for k, v in before.items()},
                }
            )
        assert records[0]["stats_delta"]["captures"] == 0
        assert records[1]["stats_delta"]["captures"] > 0
        assert records[2]["stats_delta"]["replays"] == len(request.questions)
        assert records[3]["stats_delta"]["replays"] == len(request.questions)
        assert runtime.stats["rejected"] == 0
        report["cases"].append(
            {
                "id": case["id"],
                "config": asdict(config),
                "records": records,
                "stats": dict(runtime.stats),
            }
        )
        engine.graph_runtime = None
        runtime.invalidate()
        del runtime
        torch.cuda.empty_cache()
        write_json(args.output, report)

    request = parse_request(distinct_fixture(fixtures))
    request = replace(request, questions=request.questions[:2])
    engine.graph_runtime = None
    expected = engine.predict(request)
    for name, config, counter in [
        ("eviction_cooldown", GraphConfig(min_uses=1, max_shapes=1), "cooldown"),
        ("memory", GraphConfig(min_uses=1, max_bytes=1 << 20), "memory_budget"),
        ("attempt_budget", GraphConfig(min_uses=1, max_captures=1), "capture_budget"),
        (
            "time_budget",
            GraphConfig(min_uses=1, capture_budget_ms=0.001),
            "capture_budget",
        ),
    ]:
        runtime = GraphRuntime(engine.model, config)
        engine.graph_runtime = runtime
        responses = [engine.predict(request), engine.predict(request)]
        assert all(r == expected for r in responses)
        assert runtime.stats[counter] > 0
        assert (
            len(runtime.cache) <= config.max_shapes
            and runtime.cache.bytes <= config.max_bytes
        )
        if name in ("attempt_budget", "time_budget"):
            assert runtime.stats["capture_attempts"] == 1
        if name == "eviction_cooldown":
            assert runtime.stats["evictions"] == 1
        report["policies"][name] = {
            "config": asdict(config),
            "stats": dict(runtime.stats),
            "eager_response": expected,
            "graph_responses": responses,
        }
        engine.graph_runtime = None
        runtime.invalidate()
        assert (
            not runtime.admission.history
            and not runtime.admission.attempts
            and not runtime.cache.entries
        )
        del runtime
        torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(args.output, report)
    print("verified changed inputs on five cases and four admission/fallback policies")


if __name__ == "__main__":
    main()
