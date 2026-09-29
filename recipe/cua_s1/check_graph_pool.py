"""GPU regression for sequential segments sharing one bounded shape pool."""

import argparse
import json
from pathlib import Path


def main():
    import torch

    from models.cua_s1.multimodal.graph_runtime import (
        GraphCache,
        _GraphPool,
        _GraphSegment,
        _ShapeEntry,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output")
    torch.manual_seed(42)

    class Layer:
        def __init__(self):
            self.weight = (
                torch.randn((640, 640), device="cuda", dtype=torch.bfloat16) / 25
            )

        def __call__(self, hidden, **kwargs):
            return torch.relu(hidden @ self.weight)

    layers = [Layer() for _ in range(3)]
    hidden = torch.randn((1, 512, 640), device="cuda", dtype=torch.bfloat16)

    def capture():
        entry = _ShapeEntry(pool=_GraphPool(hidden.device))
        value = hidden
        for index in range(len(layers)):
            block = _GraphSegment(
                layers, index, index + 1, value, None, pool=entry.pool
            )
            entry.blocks[index] = block
            value = block.replay(value, None)
        entry.update_bytes()
        return entry

    def replay(entry, value):
        for block in entry.blocks.values():
            value = block.replay(value, None)
        return value

    entries = [capture() for _ in range(40)]
    handles = [entry.pool.capture_stream.cuda_stream for entry in entries]
    assert len(set(handles)) == 40
    reserved = [entry.pool.reserved_bytes() for entry in entries]
    assert all(value > 0 for value in reserved)
    for entry, size in zip(entries, reserved):
        assert entry.bytes == size + sum(
            block.external_bytes for block in entry.blocks.values()
        )
    changed = hidden + 0.5
    expected = changed
    for layer in layers:
        expected = layer(expected)
    for entry in entries:
        assert torch.equal(replay(entry, changed), expected)
    retired_ids = [entry.pool.pool_id for entry in entries[::2]]
    for entry in entries[::2]:
        entry.close()
    torch.cuda.empty_cache()
    assert all(
        not torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
        for pool in retired_ids
    )
    for entry in entries[1::2]:
        assert torch.equal(replay(entry, changed), expected)
    # Reserved footprint, not live tensor deltas, drives cache eviction.
    cache = GraphCache(max_shapes=8, max_bytes=entries[1].bytes * 2)
    evictions = 0
    for index, entry in enumerate(entries[1::2]):
        retired = cache.put(index, entry)
        assert retired is not None
        evictions += len(retired)
        for item in retired:
            item.close()
        assert cache.bytes <= cache.max_bytes
        assert len(cache) <= 2
    for entry in entries:
        entry.close()
    torch.cuda.empty_cache()
    result = {
        "status": "complete",
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "shape_count": 40,
        "segments_per_shape": 3,
        "unique_shape_streams": len(set(handles)),
        "exact_changed_input_shape_replays": 60,
        "pool_reserved_bytes": reserved,
        "cache_budget_bytes": cache.max_bytes,
        "budget_driven_evictions": evictions,
        "retired_pools_released": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
