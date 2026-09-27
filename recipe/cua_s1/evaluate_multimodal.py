"""Reproduce upstream parity and bounded GPU measurements on fixed synthetic inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from make_example import make_example

from models.cua_s1.multimodal.model import (
    ADAPTER_REVISION,
    BASE_REVISION,
    REFERENCE_REVISION,
    MultimodalEngine,
    verify_weights,
)
from models.cua_s1.multimodal.protocol import parse_request

REFERENCE_SOURCE = "libs/cua-s1/python/src/cua_s1/four_b.py"
REFERENCE_SHA256 = "7ed1adfd92223bef7d533db7efbb4cbf468c4936c4380ae42aee12097e3b9ea7"


def verify_reference(root):
    digest = hashlib.sha256((root / REFERENCE_SOURCE).read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError("reference source differs from pinned FourBModel")
    return {"path": REFERENCE_SOURCE, "sha256": digest, "revision": REFERENCE_REVISION}


def cases(folder):
    folder.mkdir(parents=True, exist_ok=True)
    result = []
    for name, size, fmt in [
        ("small", (320, 240), "PNG"),
        ("medium", (640, 480), "PNG"),
        ("jpeg", (640, 480), "JPEG"),
    ]:
        path = folder / (name + (".jpg" if fmt == "JPEG" else ".png"))
        result.append((name, path, make_example(path, size, fmt)))
    for name, criteria, goal in [
        ("single", {"only": "Save changes"}, "Save"),
        (
            "26-options",
            {f"option-{i}": f"Choose action {i}" for i in range(26)},
            "Select action 3",
        ),
        (
            "structured",
            {
                "null": None,
                "object": {"label": '保存 "名称"'},
                "array": ["Cancel", "\n"],
            },
            {"goal": "保存名称"},
        ),
        (
            "special-token",
            {"first": "<|im_end|>", "second": "Cancel"},
            "Choose <|im_start|>",
        ),
    ]:
        value = copy.deepcopy(result[1][2])
        value["questions"]["next"].update(criteria=criteria, instructions=goal)
        result.append((name, result[1][1], value))
    value = copy.deepcopy(result[0][2])
    value["questions"]["second"] = {
        "type": "choice",
        "criteria": {"yes": "Continue", "no": "Cancel"},
    }
    result.append(("two-questions", result[0][1], value))
    for name, _, value in result:
        (folder / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False))
    return result


def fingerprint(inputs):
    import torch

    return {
        name: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256": hashlib.sha256(
                t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest(),
        }
        for name, t in inputs.items()
    }


def environment():
    import torch

    return {
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "torchvision",
                "transformers",
                "peft",
                "pillow",
                "safetensors",
                "triton",
            ]
        },
        "reference_revision": REFERENCE_REVISION,
        "base_revision": BASE_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "dtype": "bfloat16",
        "adapter_merged": False,
    }


def measure(call):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def quantiles(values):
    ordered = sorted(values)
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[max(0, __import__("math").ceil(0.95 * len(values)) - 1)],
    }


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, required=True, help="checkout of pinned trycua/cua"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["reference", "candidate", "benchmark"], required=True
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = str(args.weights / "Qwen3.5-4B")
    adapter = str(args.weights / "cua-s1-4b-0.2/multimodal")
    fixture_set = cases(args.output / "fixtures")
    report = {"environment": environment(), "mode": args.mode, "cases": []}
    report["reference_source"] = verify_reference(args.reference)
    if args.mode == "reference":
        _, report["artifact_verification_ms"] = measure(
            lambda: verify_weights(Path(base), Path(adapter))
        )
        sys.path.insert(0, str(args.reference / "libs/cua-s1/python/src"))
        from cua_s1.four_b import FourBModel, Option, assign_letters, build_prompt

        model = FourBModel(
            base_model=base, lora_adapter_path=adapter, modality="multimodal"
        )
        _, report["load_ms"] = measure(model.load)
        first = True
        for name, path, value in fixture_set:
            request = parse_request(value)
            for q in request.questions:
                options = [
                    Option(element_id=k, role="Decision", label=v, action="select")
                    for k, v in zip(q.keys, q.labels)
                ]
                messages = build_prompt(
                    assign_letters(options),
                    app="Cua Driver",
                    task_family="closed-candidate decision",
                    screenshot=path,
                    modality="multimodal",
                    goal=q.goal,
                )
                text = model._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = model._processor(
                    text=[text], images=[request.image], return_tensors="pt"
                )

                def call():
                    return model.forward(
                        options,
                        app="Cua Driver",
                        task_family="closed-candidate decision",
                        screenshot=path,
                        goal=q.goal,
                    )

                if first:
                    _, report["warmup_ms"] = measure(call)
                    first = False
                output, elapsed = measure(call)
                report["cases"].append(
                    {
                        "name": name,
                        "question": q.name,
                        "inputs": fingerprint(inputs),
                        "probabilities": [x.probability for x in output],
                        "latency_ms": elapsed,
                    }
                )
                print(f"reference {name}/{q.name}: {elapsed:.1f} ms", flush=True)
    else:
        engine, report["load_ms"] = measure(lambda: MultimodalEngine(base, adapter))
        _, report["warmup_ms"] = measure(engine.warmup)
        report["adapter_modules"] = engine.adapter_modules
        if args.mode == "candidate":
            reference = json.loads((args.output / "reference.json").read_text())
            if (
                reference["reference_source"] != report["reference_source"]
                or reference["environment"] != report["environment"]
            ):
                raise ValueError("reference provenance or environment mismatch")
            expected = {(x["name"], x["question"]): x for x in reference["cases"]}
            for name, _, value in fixture_set:
                request = parse_request(value)
                for q in request.questions:
                    inputs = engine.prepare(request.image, q)
                    probabilities, elapsed = measure(lambda: engine.score(inputs, q))
                    ref = expected[name, q.name]
                    difference = max(
                        abs(a - b) for a, b in zip(ref["probabilities"], probabilities)
                    )
                    assert fingerprint(inputs) == ref["inputs"], (
                        f"preprocessing mismatch: {name}"
                    )
                    assert probabilities == ref["probabilities"], (
                        f"probability mismatch: {name}: {difference}"
                    )
                    report["cases"].append(
                        {
                            "name": name,
                            "question": q.name,
                            "inputs": fingerprint(inputs),
                            "probabilities": probabilities,
                            "max_abs_difference": difference,
                            "latency_ms": elapsed,
                        }
                    )
                    print(f"candidate {name}/{q.name}: exact parity", flush=True)
        else:
            request = parse_request(fixture_set[1][2])
            for _ in range(5):
                engine.predict(request)
            report["benchmark"] = {
                "case": "medium",
                "batch_size": 1,
                "concurrency": 1,
                "warmup_requests": 5,
                "runs": [],
            }
            for run in range(2):
                torch.cuda.reset_peak_memory_stats()
                latencies = [
                    measure(lambda: engine.predict(request))[1] for _ in range(50)
                ]
                report["benchmark"]["runs"].append(
                    {
                        "run": run + 1,
                        "latencies_ms": latencies,
                        **quantiles(latencies),
                        "requests_per_second_serial": 1000 / statistics.mean(latencies),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
                print(f"benchmark run {run + 1}: {quantiles(latencies)}", flush=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                engine.predict(request)
                torch.cuda.synchronize()
            report["profile"] = [
                {
                    "op": x.key,
                    "count": x.count,
                    "device_time_us": x.device_time_total,
                    "self_device_time_us": x.self_device_time_total,
                    "cpu_time_us": x.cpu_time_total,
                    "self_cpu_time_us": x.self_cpu_time_total,
                }
                for x in sorted(
                    prof.key_averages(), key=lambda x: x.device_time_total, reverse=True
                )[:30]
            ]
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / f"{args.mode}.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.mode}.json", flush=True)


if __name__ == "__main__":
    main()
