"""Real-model worker HTTP parity, instance isolation and explicit shutdown."""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import threading
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output")
    from evaluate_multimodal import environment
    from PIL import Image
    from profile_multimodal import case_matrix, fixture, repository_state, write_json
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request
    from models.cua_s1.multimodal.server import WorkerServer, parse_args

    source = repository_state(Path(__file__).resolve().parents[2])
    if source["dirty"]:
        raise ValueError("clean committed source required")
    config = parse_args(
        [
            "--base",
            str(args.weights / "Qwen3.5-4B"),
            "--adapter",
            str(args.weights / "cua-s1-4b-0.2/multimodal"),
            "--graph-mode",
            "rule-bucket",
        ]
    )
    engine = MultimodalEngine(
        config.base, config.adapter, graph_config=config.graph_config
    )
    runtime = engine.graph_runtime
    assert runtime.__class__.__module__ == "models.cua_s1.multimodal.graph_buckets"
    original_rule = upstream.torch_chunk_gated_delta_rule
    modules = [
        layer.linear_attn
        for layer in engine.model.get_base_model().model.language_model.layers
        if layer.block_type == "linear_attention"
    ]
    forwards = [getattr(m.forward, "__func__", m.forward) for m in modules]
    case = next(c for c in case_matrix() if c["id"] == "320x240-short-q2")
    raw = fixture(case, args.output.parent / "http-fixtures")
    black = copy.deepcopy(raw)
    image = io.BytesIO()
    Image.new("RGB", (320, 240), "black").save(image, format="PNG")
    black["state"]["image"] = (
        "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    )
    changed = copy.deepcopy(raw)
    for q in changed["questions"].values():
        q["instructions"] = q["instructions"].replace("Save", "Open")
    assert changed != raw
    requests = [("cold", raw), ("capture", raw), ("image", black), ("text", changed)]
    engine.graph_runtime = None
    expected = [engine.predict(parse_request(body)) for _, body in requests]
    engine.graph_runtime = runtime
    server = WorkerServer(("127.0.0.1", 0), engine)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    report = {
        "status": "running",
        "repository": source,
        "environment": environment(),
        "records": [],
    }
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(url + "/health", timeout=10) as response:
            report["health"] = {"status": response.status, "body": json.load(response)}
        for (name, body), reference in zip(requests, expected):
            before = dict(runtime.stats)
            request = urllib.request.Request(
                url + "/v1/systemone",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                actual = json.load(response)
                assert response.status == 200
            delta = {k: runtime.stats[k] - v for k, v in before.items()}
            report["records"].append(
                {
                    "name": name,
                    "expected": reference,
                    "actual": actual,
                    "stats_delta": delta,
                }
            )
            assert actual == reference
            assert runtime.stats["rejected"] == runtime.stats["length_rejections"] == 0
            if name in {"image", "text"}:
                assert delta["replays"] == 2
            assert upstream.torch_chunk_gated_delta_rule is original_rule
            assert all(
                getattr(m.forward, "__func__", m.forward) is f
                for m, f in zip(modules, forwards)
            )
        assert report["records"][0]["stats_delta"]["captures"] == 0
        assert report["records"][1]["stats_delta"]["captures"] == 1
        report["global_and_instance_methods_unchanged"] = True
        entries = list(runtime.cache.entries.values())
        server.shutdown()
        server.server_close()
        thread.join(10)
        assert not thread.is_alive() and runtime._closed and engine._closed
        assert entries and all(
            not entry.blocks and entry.pool is None for entry in entries
        )
        assert not runtime.cache.entries
        report["explicit_close_with_live_entry_references"] = True
        try:
            engine.predict(parse_request(raw))
        except RuntimeError as exc:
            assert "closed" in str(exc)
        else:
            raise AssertionError("closed engine accepted prediction")
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
        write_json(args.output, report)
    print(
        "verified HTTP cold/capture/changed-image/changed-text parity, isolation and shutdown"
    )


if __name__ == "__main__":
    main()
