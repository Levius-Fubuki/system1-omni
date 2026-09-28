"""Capture concrete tensor layout and verify the actual HTTP worker on CUDA."""

import json
import sys
import threading
import urllib.request
from pathlib import Path

from benchmark_image_reuse import distinct_fixture, require_equal
from evaluate_multimodal import environment
from profile_multimodal import repository_state, write_json

from models.cua_s1.multimodal.model import MultimodalEngine
from models.cua_s1.multimodal.protocol import parse_request
from models.cua_s1.multimodal.server import WorkerServer

root, weights, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
report = {"repository": repository_state(root), "environment": environment()}
engine = MultimodalEngine(
    str(weights / "Qwen3.5-4B"), str(weights / "cua-s1-4b-0.2/multimodal")
)
value = distinct_fixture(output / "fixture")
request = parse_request(value)
prepared = engine.prepare_reused(request.image, request.questions)
features = engine.encode_image(prepared[0])


def metadata(t):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "stride": list(t.stride()),
        "contiguous": t.is_contiguous(),
        "requires_grad": t.requires_grad,
        "bytes": t.numel() * t.element_size(),
    }


report["cpu_inputs"] = {k: metadata(t) for k, t in prepared[0].items()}
report["shared_features"] = metadata(features)
core = engine.model.get_base_model().model
positions = []


def capture(module, args, kwargs):
    positions.append(
        {
            k: metadata(kwargs[k])
            for k in ("inputs_embeds", "position_ids", "attention_mask")
        }
    )


handle = core.language_model.register_forward_pre_hook(capture, with_kwargs=True)
try:
    expected = engine.predict(request)
finally:
    handle.remove()
report["per_question_language_layouts"] = positions
del prepared, features
server = WorkerServer(("127.0.0.1", 0), engine)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    address = f"http://127.0.0.1:{server.server_port}"
    with urllib.request.urlopen(address + "/health", timeout=30) as response:
        report["health"] = {"status": response.status, "body": json.load(response)}
    req = urllib.request.Request(
        address + "/v1/systemone",
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        actual = json.load(response)
        require_equal(expected, actual, "HTTP response")
        report["http"] = {
            "status": response.status,
            "exact_engine_parity": True,
            "question_count": len(actual["answers"]),
            "response": actual,
        }
finally:
    server.shutdown()
    server.server_close()
    thread.join()
report["status"] = "complete"
write_json(output / "postflight.json", report)
print("CUDA tensor layout and HTTP q8 exact parity complete", flush=True)
