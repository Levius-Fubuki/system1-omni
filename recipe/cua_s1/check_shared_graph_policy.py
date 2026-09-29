"""GPU proof of shared Graph budgets, cross-mode eviction and explicit close."""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import asdict, replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path")

    import torch
    from benchmark_multimodal_graph import changed_text_request
    from diagnose_graph_buckets import prepare_values
    from evaluate_multimodal import environment
    from graph_policy import SharedGraphRuntime
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
    files = [
        Path(__file__),
        Path(__file__).with_name("graph_policy.py"),
        Path(__file__).with_name("diagnose_graph_buckets.py"),
        Path(__file__).with_name("benchmark_multimodal_graph.py"),
        Path(__file__).with_name("profile_multimodal.py"),
        *sorted((root / "src/models/cua_s1/multimodal").glob("*.py")),
    ]
    report = {
        "status": "running",
        "repository": source,
        "environment": environment(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "boundary_checks": [],
        "policies": {},
        "responses": [],
        "response_logit_checks": [],
    }
    write_json(args.output, report)
    engine = MultimodalEngine(
        str(args.weights / "Qwen3.5-4B"),
        str(args.weights / "cua-s1-4b-0.2/multimodal"),
    )
    runtime = None

    def cache_snapshot(group):
        torch.cuda.synchronize()
        return {
            "cache_shapes": len(group.cache),
            "cache_bytes": group.cache.bytes,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "entries": [
                {"mode": key[0], "bytes": entry.bytes}
                for key, entry in group.cache.entries.items()
            ],
        }

    def close_and_check(group):
        entries = list(group.cache.entries.values())
        before = cache_snapshot(group)
        group.close()
        assert all(not entry.blocks and entry.pool is None for entry in entries)
        assert not group.cache.entries and group.cache.bytes == 0
        assert not group.admission.attempts and not group.admission.history
        gc.collect()
        torch.cuda.empty_cache()
        after = cache_snapshot(group)
        return {"before": before, "after": after, "live_entry_references_retired": True}

    def compare(group, mode, values, records):
        group.select_mode(mode)
        expected = engine.model(**values, logits_to_keep=1, use_cache=False).logits[
            0, -1, :
        ]
        before = group.stats
        previous = list(group.cache.entries.values())
        with group.request():
            actual = group.forward(values)
        equal = torch.equal(expected, actual)
        current_ids = {id(entry) for entry in group.cache.entries.values()}
        retired = [entry for entry in previous if id(entry) not in current_ids]
        assert all(not entry.blocks and entry.pool is None for entry in retired)
        delta = {k: value - before[k] for k, value in group.stats.items()}
        records.append(
            {
                "mode": mode,
                "tokens": values["inputs_embeds"].shape[1],
                "equal": equal,
                "max_abs": (expected.float() - actual.float()).abs().max().item(),
                "stats_delta": delta,
                "retired_with_live_references": len(retired),
                **cache_snapshot(group),
            }
        )
        assert equal
        assert len(group.cache) <= group.config.max_shapes
        assert group.cache.bytes <= group.config.max_bytes
        assert (
            group.stats["numerical_mismatch"] == group.stats["length_rejections"] == 0
        )

    try:
        with torch.no_grad():
            case = next(c for c in case_matrix() if c["id"] == "320x240-short-q2")
            request = parse_request(fixture(case, args.output.parent / "fixtures"))
            long_request = replace(
                request,
                questions=tuple(
                    replace(q, goal=" ".join([GOAL] * 30)) for q in request.questions
                ),
            )
            prepared = prepare_values(engine, long_request)
            lengths = [255, 256, 257, 319, 320, 321]
            values = {
                n: {
                    "inputs_embeds": prepared["inputs_embeds"][:, :n].clone(),
                    "position_ids": prepared["position_ids"][:, :, :n].clone(),
                    "attention_mask": prepared["attention_mask"][:, :n].clone(),
                }
                for n in lengths
            }
            config = GraphConfig(
                min_uses=1,
                max_shapes=12,
                max_bytes=4 << 30,
                max_captures=32,
                capture_budget_ms=20000,
            )
            report["boundary_config"] = asdict(config)
            runtime = SharedGraphRuntime(engine.model, config)
            torch.cuda.reset_peak_memory_stats()
            for n in lengths + lengths[::-1]:
                for mode in ("exact", "rule-bucket"):
                    compare(runtime, mode, values[n], report["boundary_checks"])
            assert runtime.stats["captures"] == 9
            assert runtime.stats["replays"] == 15
            assert runtime.stats["requests"] == 24
            report["boundary_stats"] = runtime.stats
            report["boundary_cleanup"] = close_and_check(runtime)
            for name, policy, sequence, counter in [
                (
                    "cross_mode_eviction",
                    replace(config, max_shapes=1),
                    [
                        ("exact", 255),
                        ("rule-bucket", 255),
                        ("exact", 255),
                        ("rule-bucket", 256),
                    ],
                    "cooldown",
                ),
                (
                    "resident_byte_limit",
                    replace(config, max_bytes=300 << 20),
                    [
                        ("exact", 255),
                        ("rule-bucket", 255),
                        ("exact", 255),
                        ("rule-bucket", 256),
                    ],
                    "evictions",
                ),
                (
                    "capture_count_limit",
                    replace(config, max_captures=1),
                    [("exact", 255), ("rule-bucket", 255), ("exact", 255)],
                    "capture_budget",
                ),
                (
                    "capture_time_limit",
                    replace(config, capture_budget_ms=0.001),
                    [("exact", 255), ("rule-bucket", 255)],
                    "capture_budget",
                ),
                (
                    "oversize_candidates",
                    replace(config, max_bytes=1 << 20),
                    [("exact", 255), ("rule-bucket", 255)],
                    "memory_budget",
                ),
            ]:
                runtime = SharedGraphRuntime(engine.model, policy)
                record = {"config": asdict(policy), "events": []}
                report["policies"][name] = record
                torch.cuda.reset_peak_memory_stats()
                for mode, n in sequence:
                    compare(runtime, mode, values[n], record["events"])
                assert runtime.stats[counter] > 0, (name, runtime.stats)
                record["stats"] = runtime.stats
                record["cleanup"] = close_and_check(runtime)
                write_json(args.output, report)
            del values, prepared

            config = replace(config, min_uses=2)
            report["response_config"] = asdict(config)
            runtime = SharedGraphRuntime(engine.model, config)
            original_forward = runtime.forward

            def checked_forward(v):
                expected = engine.model(**v, logits_to_keep=1, use_cache=False).logits[
                    0, -1, :
                ]
                actual = original_forward(v)
                equal = torch.equal(expected, actual)
                report["response_logit_checks"].append(
                    {
                        "mode": runtime._mode,
                        "tokens": v["inputs_embeds"].shape[1],
                        "equal": equal,
                        "max_abs": (expected.float() - actual.float())
                        .abs()
                        .max()
                        .item(),
                    }
                )
                assert equal
                return actual

            runtime.forward = checked_forward
            for case in [
                c
                for c in case_matrix()
                if c["id"] in {"320x240-short-q2", "640x480-short-q8"}
            ]:
                request = parse_request(fixture(case, args.output.parent / "fixtures"))
                changed_text, _ = changed_text_request(engine, request)
                changed_image = replace(
                    request, image=Image.new("RGB", request.image.size, "black")
                )
                for name, item in [
                    ("cold", request),
                    ("capture", request),
                    ("image", changed_image),
                    ("text", changed_text),
                ]:
                    engine.graph_runtime = None
                    expected = engine.predict(item)
                    for mode in ("exact", "rule-bucket"):
                        runtime.select_mode(mode)
                        engine.graph_runtime = runtime
                        before = runtime.stats
                        actual = engine.predict(item)
                        delta = {k: v - before[k] for k, v in runtime.stats.items()}
                        assert delta["requests"] == 1
                        assert actual == expected
                        if name in {"image", "text"}:
                            assert delta["replays"] == len(item.questions)
                        report["responses"].append(
                            {
                                "case": case["id"],
                                "variant": name,
                                "mode": mode,
                                "expected": expected,
                                "actual": actual,
                                "stats_delta": delta,
                            }
                        )
            report["response_stats"] = runtime.stats
            assert runtime.stats["rejected"] == runtime.stats["length_rejections"] == 0
            report["response_cleanup"] = close_and_check(runtime)
            engine.graph_runtime = None
        report["status"] = "complete"
        print(
            "shared resource checks complete",
            len(report["boundary_checks"]),
            "boundary checks;",
            len(report["responses"]),
            "responses;",
            len(report["response_logit_checks"]),
            "response logit checks",
            flush=True,
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if runtime is not None:
            runtime.close()
        engine.graph_runtime = None
        engine.close()
        write_json(args.output, report)


if __name__ == "__main__":
    main()
