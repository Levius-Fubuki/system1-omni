"""GPU regression: >32 owned capture streams, eviction, surviving replay."""

import argparse
import json
from pathlib import Path


def main():
    import torch

    from models.cua_s1.multimodal.graph_runtime import _GraphSegment

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output")
    torch.manual_seed(42)

    class Layer:
        def __init__(self):
            self.weight = torch.randn((640, 640), device="cuda", dtype=torch.bfloat16)

        def __call__(self, hidden, **kwargs):
            return hidden @ self.weight

    layers = [Layer()]
    hidden = torch.randn((1, 512, 640), device="cuda", dtype=torch.bfloat16)
    expected = layers[0](hidden)
    segments = [_GraphSegment(layers, 0, 1, hidden, None) for _ in range(40)]
    handles = [segment.capture_stream.cuda_stream for segment in segments]
    assert len(set(handles)) == 40
    for segment in segments:
        assert torch.equal(segment.replay(hidden, None), expected)
    # Release alternating graphs and their pools, then exercise survivors with
    # changed inputs. A pooled 32-stream workaround cannot satisfy uniqueness.
    for segment in segments[::2]:
        segment.close()
        segment.close()
    torch.cuda.empty_cache()
    changed = hidden + 1
    expected_changed = layers[0](changed)
    for _ in range(3):
        for segment in segments[1::2]:
            assert torch.equal(segment.replay(changed, None), expected_changed)
    for segment in segments:
        segment.close()
    torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "torch": torch.__version__,
                "torch_git": torch.version.git_version,
                "live_unique_streams": len(set(handles)),
                "initial_exact_replays": 40,
                "survivor_changed_input_exact_replays": 60,
                "idempotent_cleanup": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("40 unique streams; 100 exact replays across graph retirement")


if __name__ == "__main__":
    main()
