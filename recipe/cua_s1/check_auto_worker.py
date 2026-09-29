"""Independent full-logit checks of the actual automatic worker runtime."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path")

    import torch
    from benchmark_image_reuse import distinct_fixture
    from benchmark_multimodal_graph import changed_text_request
    from diagnose_graph_buckets import prepare_values
    from evaluate_multimodal import environment
    from PIL import Image
    from profile_multimodal import (
        GOAL,
        case_matrix,
        fixture,
        repository_state,
        write_json,
    )

    from models.cua_s1.multimodal.graph_runtime import GraphConfig
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    root = Path(__file__).resolve().parents[2]
    source = repository_state(root)
    if source["dirty"] or not source["revision"]:
        raise ValueError("clean committed source required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = GraphConfig(
        mode="auto",
        max_tokens=4096,
        max_bytes=4 << 30,
        max_captures=16,
        capture_budget_ms=10000,
    )
    files = [
        Path(__file__),
        *sorted((root / "src/models/cua_s1/multimodal").glob("*.py")),
    ]
    report = {
        "status": "running",
        "repository": source,
        "environment": environment(),
        "config": asdict(config),
        "cases": [],
        "logit_checks": [],
        "boundaries": [],
        "churn_events": [],
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    write_json(args.output, report)
    engine = MultimodalEngine(
        str(args.weights / "Qwen3.5-4B"),
        str(args.weights / "cua-s1-4b-0.2/multimodal"),
        graph_config=config,
    )
    runtime = engine.graph_runtime
    assert runtime.__class__.__module__ == "models.cua_s1.multimodal.graph_auto"
    original = runtime.forward

    def checked(values):
        expected = engine.model(**values, logits_to_keep=1, use_cache=False).logits[
            0, -1, :
        ]
        actual = original(values)
        equal = torch.equal(expected, actual)
        report["logit_checks"].append(
            {
                "tokens": values["inputs_embeds"].shape[1],
                "equal": equal,
                "max_abs": (actual.float() - expected.float()).abs().max().item(),
            }
        )
        assert equal
        return actual

    runtime.forward = checked
    fixtures = args.output.parent / "auto-fixtures"
    try:
        with torch.no_grad():
            cases = [
                c
                for c in case_matrix()
                if c["id"]
                in {
                    "320x240-short-q2",
                    "320x240-short-q8",
                    "640x480-short-q8",
                    "640x480-long-q8",
                }
            ]
            cases.append({"id": "640x480-distinct-q8"})
            for case in cases:
                raw = (
                    distinct_fixture(fixtures)
                    if "distinct" in case["id"]
                    else fixture(case, fixtures)
                )
                request = parse_request(raw)
                if "distinct" in case["id"]:
                    changed = replace(
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
                    changed, _ = changed_text_request(engine, request)
                black = replace(
                    request, image=Image.new("RGB", request.image.size, "black")
                )
                record = {"id": case["id"], "events": []}
                report["cases"].append(record)
                before_case = runtime.stats
                for name, item in [
                    ("cold", request),
                    ("admit", request),
                    ("warm", request),
                    ("image", black),
                    ("text", changed),
                ]:
                    engine.graph_runtime = None
                    expected = engine.predict(item)
                    engine.graph_runtime = runtime
                    before = runtime.stats
                    actual = engine.predict(item)
                    delta = {k: v - before[k] for k, v in runtime.stats.items()}
                    assert delta["requests"] == 1
                    assert expected == actual
                    record["events"].append(
                        {
                            "name": name,
                            "expected": expected,
                            "actual": actual,
                            "stats_delta": delta,
                        }
                    )
                record["stats_delta"] = {
                    k: v - before_case[k] for k, v in runtime.stats.items()
                }
                assert record["stats_delta"]["captures"] > 0
                entries = list(runtime.cache.entries.values())
                runtime.invalidate()
                assert all(not entry.blocks and entry.pool is None for entry in entries)
                assert not runtime.selector.history and not runtime._pending
                write_json(args.output, report)

            base = parse_request(
                fixture(
                    next(c for c in cases if c["id"] == "320x240-short-q2"), fixtures
                )
            )
            before_churn = runtime.stats
            for repetitions in list(range(1, 13)) * 3:
                item = replace(
                    base,
                    questions=tuple(
                        replace(q, goal=" ".join([GOAL] * repetitions))
                        for q in base.questions
                    ),
                )
                engine.graph_runtime = None
                expected = engine.predict(item)
                engine.graph_runtime = runtime
                before = runtime.stats
                actual = engine.predict(item)
                assert actual == expected
                report["churn_events"].append(
                    {
                        "repetitions": repetitions,
                        "expected": expected,
                        "actual": actual,
                        "stats_delta": {
                            k: v - before[k] for k, v in runtime.stats.items()
                        },
                    }
                )
            report["churn_stats"] = {
                k: v - before_churn[k] for k, v in runtime.stats.items()
            }
            assert report["churn_stats"]["selected_rule_bucket"] > 0
            runtime.invalidate()

            long_request = replace(
                base,
                questions=tuple(
                    replace(q, goal=" ".join([GOAL] * 30)) for q in base.questions
                ),
            )
            prepared = prepare_values(engine, long_request)
            for length in [255, 256, 257, 319, 320, 321] * 3 + [
                321,
                320,
                319,
                257,
                256,
                255,
            ]:
                values = {
                    "inputs_embeds": prepared["inputs_embeds"][:, :length].clone(),
                    "position_ids": prepared["position_ids"][:, :, :length].clone(),
                    "attention_mask": prepared["attention_mask"][:, :length].clone(),
                }
                before = runtime.stats
                with runtime.request():
                    runtime.forward(values)
                report["boundaries"].append(
                    {
                        "length": length,
                        "stats_delta": {
                            k: v - before[k] for k, v in runtime.stats.items()
                        },
                    }
                )
            assert runtime.stats["rejected"] == runtime.stats["length_rejections"] == 0
            report["stats"] = runtime.stats
            report["selector_reasons"] = dict(runtime.decisions)
            entries = list(runtime.cache.entries.values())
            engine.close()
            assert not runtime.cache.entries and runtime.cache.bytes == 0
            assert all(not entry.blocks and entry.pool is None for entry in entries)
            report["explicit_close"] = True
        report["status"] = "complete"
        print(
            "auto parity complete:",
            len(report["logit_checks"]),
            "complete-logit comparisons;",
            sum(len(c["events"]) for c in report["cases"]),
            "case responses; 36 churn responses;",
            len(report["boundaries"]),
            "boundaries",
            flush=True,
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        engine.graph_runtime = runtime
        engine.close()
        write_json(args.output, report)


if __name__ == "__main__":
    main()
