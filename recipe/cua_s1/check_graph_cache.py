"""GPU check of atomic shape eviction and memory-budget eager fallback."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from benchmark_image_reuse import distinct_fixture
from benchmark_multimodal_graph import response_difference
from profile_multimodal import write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("output already exists; select a fresh path")

    import torch

    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    engine = MultimodalEngine(
        str(args.weights / "Qwen3.5-4B"),
        str(args.weights / "cua-s1-4b-0.2/multimodal"),
    )
    request = parse_request(distinct_fixture(args.output.parent / "cache-fixtures"))
    request = replace(request, questions=request.questions[:2])
    baseline = engine.predict(request)
    report = {}
    for name, config in (
        ("shape_eviction", GraphConfig(max_shapes=1, max_bytes=512 << 20, min_uses=1)),
        ("memory_fallback", GraphConfig(max_shapes=1, max_bytes=1 << 20, min_uses=1)),
    ):
        runtime = GraphRuntime(engine.model, config)
        engine.graph_runtime = runtime
        first = engine.predict(request)
        second = engine.predict(request)
        assert response_difference(baseline, first) == 0
        assert response_difference(baseline, second) == 0
        if name == "shape_eviction":
            assert runtime.stats["evictions"] >= 1
            assert len(runtime.cache) == 1
        else:
            assert runtime.stats["memory_budget"] >= 1
            assert runtime.stats["disabled"] >= 1
            assert len(runtime.cache) == 0
        assert runtime.cache.bytes <= config.max_bytes
        report[name] = {
            "stats": dict(runtime.stats),
            "cache_shapes": len(runtime.cache),
            "cache_bytes": runtime.cache.bytes,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "exact_response_parity": True,
        }
        engine.graph_runtime = None
        runtime.invalidate()
        del runtime
        torch.cuda.empty_cache()
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    raise SystemExit(main())
