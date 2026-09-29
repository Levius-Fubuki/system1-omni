"""Strict full-logit/response checks and isolated reservation for rule buckets."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output")

    import torch
    from benchmark_image_reuse import distinct_fixture
    from benchmark_multimodal_graph import changed_text_request
    from evaluate_multimodal import environment
    from graph_buckets import RuleBucketRuntime
    from PIL import Image
    from profile_multimodal import case_matrix, fixture, repository_state, write_json

    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    source = repository_state(Path(__file__).resolve().parents[2])
    if source["dirty"]:
        raise ValueError("clean source required")
    report = {
        "status": "running",
        "repository": source,
        "environment": environment(),
        "cases": [],
        "memory": {},
    }
    engine = MultimodalEngine(
        str(args.weights / "Qwen3.5-4B"), str(args.weights / "cua-s1-4b-0.2/multimodal")
    )
    fixtures = args.output.parent / "check-fixtures"
    selected = {
        "320x240-short-q2",
        "320x240-short-q8",
        "640x480-short-q8",
        "640x480-long-q8",
    }
    cases = [c for c in case_matrix() if c["id"] in selected] + [
        {"id": "640x480-distinct-q8"}
    ]
    config = GraphConfig(
        max_tokens=4096, max_captures=16, capture_budget_ms=10000, max_bytes=4 * 1024**3
    )
    try:
        for case in cases:
            raw = (
                distinct_fixture(fixtures)
                if "distinct" in case["id"]
                else fixture(case, fixtures)
            )
            request = parse_request(raw)
            runtime = RuleBucketRuntime(engine.model, config)
            original_forward = runtime.forward
            checked = []

            def checked_forward(values):
                expected = engine.model(
                    **values, logits_to_keep=1, use_cache=False
                ).logits[0, -1, :]
                actual = original_forward(values)
                equal = torch.equal(expected, actual)
                checked.append(
                    {
                        "tokens": values["inputs_embeds"].shape[1],
                        "equal": equal,
                        "max_abs": (expected.float() - actual.float())
                        .abs()
                        .max()
                        .item(),
                    }
                )
                assert equal, "full logits differ on changed input"
                return actual

            runtime.forward = checked_forward
            changed_image = replace(
                request, image=Image.new("RGB", request.image.size, "black")
            )
            if "distinct" in case["id"]:
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
            else:
                changed_text, _ = changed_text_request(engine, request)
            record = {
                "id": case["id"],
                "config": asdict(config),
                "records": [],
                "logit_checks": checked,
            }
            report["cases"].append(record)
            with torch.no_grad():
                for name, item in [
                    ("cold", request),
                    ("capture", request),
                    ("image", changed_image),
                    ("text", changed_text),
                ]:
                    engine.graph_runtime = None
                    expected = engine.predict(item)
                    engine.graph_runtime = runtime
                    before = dict(runtime.stats)
                    actual = engine.predict(item)
                    delta = {k: runtime.stats[k] - v for k, v in before.items()}
                    record["records"].append(
                        {
                            "variant": name,
                            "expected": expected,
                            "actual": actual,
                            "stats_delta": delta,
                        }
                    )
                    assert actual == expected
                    assert (
                        runtime.stats["rejected"]
                        == runtime.stats["length_rejections"]
                        == 0
                    )
                    if name in ("image", "text"):
                        assert delta["replays"] == len(item.questions), (
                            case["id"],
                            name,
                            delta,
                        )
            engine.graph_runtime = None
            runtime.invalidate()
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
            write_json(args.output, report)
            print(
                case["id"],
                "verified",
                len(checked),
                "full-logit comparisons",
                flush=True,
            )

        # Each mode owns the only live Graph pool; allocator is emptied between
        # variants. This is a separate diagnostic, not paired performance data.
        case = next(c for c in case_matrix() if c["id"] == "320x240-short-q2")
        request = parse_request(fixture(case, fixtures))
        for name, runtime_class in [
            ("eager", None),
            ("exact", GraphRuntime),
            ("rule", RuleBucketRuntime),
        ]:
            runtime = (
                runtime_class(engine.model, GraphConfig()) if runtime_class else None
            )
            engine.graph_runtime = runtime
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                engine.predict(request)
            torch.cuda.synchronize()
            report["memory"][name] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "cache_bytes": runtime.cache.bytes if runtime else 0,
                "stats": dict(runtime.stats) if runtime else {},
            }
            engine.graph_runtime = None
            if runtime:
                runtime.invalidate()
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
