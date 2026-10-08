"""Load the unmodified pinned JEMM reference and serve its official HTTP handler.

Use a local clone/download of ypcypc/JEMM at the pinned revision. This wrapper
only validates provenance, chooses the public model name JEMM, captures optional
preprocessing fixtures, and records the classes/functions actually loaded.
It never merges the adapter or replaces an upstream kernel.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from protocol import FROZEN_CORPUS_SHA256, load_corpus

INVENTORY_PATH = Path(__file__).with_name("pinned_inventory.json")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_source(source):
    expected = json.loads(INVENTORY_PATH.read_text())["upstream_source_sha256"]
    for filename, checksum in expected.items():
        path = source / filename
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"pinned upstream source mismatch: {filename}")
    return expected


def validate_checkpoints(base, adapter):
    """Reference inputs must be the same byte-verified pinned artifacts as export."""
    inventory = json.loads(INVENTORY_PATH.read_text())
    verified = {}
    for kind, directory in (("base", base), ("adapter", adapter)):
        checksums = {}
        for filename, expected in inventory[kind]["files"].items():
            path = directory / filename
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"pinned checkpoint mismatch: {kind}/{filename}")
            checksums[filename] = expected
        verified[kind] = checksums
    return verified


def function_record(function):
    function = getattr(function, "__func__", function)
    result = {"module": getattr(function, "__module__", None), "qualname": getattr(function, "__qualname__", str(type(function)))}
    try:
        filename = inspect.getsourcefile(function)
        if filename and Path(filename).is_file():
            result.update(file=filename, file_sha256=digest(filename))
    except (TypeError, OSError):
        pass
    return result


def observed_peft_state(model):
    """Inspect loaded tensors and adapter merge state; do not infer them from CLI intent."""
    dtype_inventory, lora_inventory, merged = {}, {}, {}
    parameter_count = 0
    for name, parameter in model.named_parameters():
        count, dtype = parameter.numel(), str(parameter.dtype)
        parameter_count += count
        summary = dtype_inventory.setdefault(dtype, {"tensors": 0, "elements": 0})
        summary["tensors"] += 1
        summary["elements"] += count
        if "lora_" in name:
            lora_inventory[name] = {"shape": list(parameter.shape), "dtype": dtype, "elements": count, "device": str(parameter.device)}
    for name, module in model.named_modules():
        state = getattr(module, "merged_adapters", None)
        if isinstance(state, (list, tuple)):
            merged[name] = list(state)
    peft_configs = getattr(model, "peft_config", {})
    return {"peft_class": type(model).__module__ + "." + type(model).__qualname__, "parameter_count": parameter_count,
            "parameter_dtype_inventory": dtype_inventory, "lora_parameter_inventory": lora_inventory,
            "merged_adapters": merged, "reference_adapter_merged": any(merged.values()) if merged else None,
            "observed_peft_adapters": {name: {"class": type(config).__module__ + "." + type(config).__qualname__,
                                             "peft_type": str(getattr(config, "peft_type", None)), "r": getattr(config, "r", None),
                                             "lora_alpha": getattr(config, "lora_alpha", None)} for name, config in peft_configs.items()}}


def capture_position_ids(model, inputs):
    """Call the loaded model's unmodified RoPE index method on CPU tensors."""
    import torch
    method = next((getattr(module, "get_rope_index") for _, module in model.named_modules()
                   if callable(getattr(module, "get_rope_index", None))), None)
    if method is None:
        raise ValueError("loaded reference exposes no get_rope_index method")
    ids = inputs["input_ids"].cpu()
    available = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "mm_token_type_ids": inputs.get("mm_token_type_ids", torch.zeros_like(ids)).cpu(),
                 "image_grid_thw": inputs["image_grid_thw"].cpu() if "image_grid_thw" in inputs else None,
                 "video_grid_thw": None}
    signature = inspect.signature(method)
    kwargs = {key: value for key, value in available.items() if key in signature.parameters}
    missing = [key for key, spec in signature.parameters.items() if spec.default is inspect.Parameter.empty
               and spec.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and key not in kwargs]
    if missing:
        raise ValueError(f"unsupported observed get_rope_index signature {signature}: missing {missing}")
    with torch.inference_mode():
        output = method(**kwargs)
    positions = output[0] if isinstance(output, tuple) else output
    if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != (3, 1, ids.shape[1]):
        raise ValueError("reference RoPE positions must have shape [3,1,T]")
    return positions.cpu().tolist(), {**function_record(method), "signature": str(signature), "cpu_kwargs": list(kwargs)}


def loaded_environment(decision, source, load_seconds):
    modules = []
    files = {}
    for name, module in decision.model.named_modules():
        # Record actual attention and vision modules, including the first layer
        # of each attention type. Names/classes expose wrapped PEFT modules too.
        if not name or name.endswith(("linear_attn", "self_attn", "visual", "lm_head")):
            forward = function_record(module.forward)
            if forward.get("file"):
                files[forward["file"]] = forward["file_sha256"]
            globals_ = getattr(getattr(module.forward, "__func__", module.forward), "__globals__", {})
            functions = {key: function_record(value) for key, value in globals_.items()
                         if callable(value) and any(term in key.lower() for term in ("delta_rule", "attention_forward", "flash_attn"))}
            availability = {key: value for key, value in globals_.items() if isinstance(value, bool)
                            and any(term in key.lower() for term in ("available", "fla", "flash", "fast_path"))}
            modules.append({"name": name, "class": type(module).__module__ + "." + type(module).__qualname__,
                            "forward": forward, "kernel_functions": functions, "availability_flags": availability})
    dependencies = {}
    for name in ("torch", "transformers", "peft", "flash-linear-attention", "safetensors", "pillow", "numpy", "huggingface-hub"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    import torch
    runtime_config = decision.model.config
    attention_implementation = getattr(runtime_config, "_attn_implementation", None)
    text_config = getattr(runtime_config, "text_config", None)
    return {"python": platform.python_version(), "platform": platform.platform(), "dependencies": dependencies,
            "cuda": torch.version.cuda, "device": str(decision.device), "model": decision.name,
            "load_seconds": load_seconds, **observed_peft_state(decision.model), "upstream_source_sha256": source,
            "loaded_forward_modules": modules, "forward_file_sha256": files,
            "attention_implementation": attention_implementation,
            "text_attention_implementation": getattr(text_config, "_attn_implementation", None),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "torch_threads": torch.get_num_threads(), "calibration": {key: getattr(decision, key) for key in ("temperature", "mm_temperature", "threshold")}}


def capture_contract(decision, corpus, destination):
    """Capture the official input boundary, with exact tokens and image grids.

    Preserve the complete processor tensors as safetensors, alongside readable
    identities, prompt text, labels, token IDs, grid and modality token types.
    No model forward occurs here.
    """
    from jemm.contract import mm_messages, single_messages
    from jemm.systemone import decode_images, single_requests
    from safetensors.torch import save_file
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for case in corpus:
        body = case["request"]
        images = decode_images(body.get("images"))
        for index, (qid, kind, request) in enumerate(single_requests(body.get("state", ""), body["questions"])):
            inputs = decision._inputs(request, images)
            positions, position_method = capture_position_ids(decision.model, inputs)
            messages = mm_messages(request, len(images)) if images else single_messages(request)
            processor = decision.processor if images else decision.tokenizer
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False, preserve_thinking=False)
            filename = f"{case['id']}-{index:02d}.safetensors"
            save_file({key: value.cpu().contiguous() for key, value in inputs.items()}, str(destination / filename))
            record = {"id": case["id"], "question_id": qid, "type": kind, "candidate_ids": [c["id"] for c in request["candidates"]],
                      "prompt": prompt, "input_ids": inputs["input_ids"][0].tolist(), "label_token_ids": decision.labels,
                      "position_ids": positions, "position_method": position_method,
                      "tensor_file": filename, "tensor_file_sha256": digest(destination / filename),
                      "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in inputs.items()}}
            for key in ("image_grid_thw", "mm_token_type_ids"):
                if key in inputs:
                    record[key] = inputs[key].tolist()
            records.append(record)
    (destination / "contract.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--capture-corpus", type=Path)
    parser.add_argument("--expected-corpus-sha256", default=FROZEN_CORPUS_SHA256, help="only override for a separately predeclared frozen workload")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8790, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    capture_cases, capture_hash = load_corpus(args.capture_corpus, args.expected_corpus_sha256) if args.capture_corpus else (None, None)
    source = args.source.resolve()
    provenance = validate_source(source)
    verification_started = time.perf_counter()
    checkpoint_hashes = validate_checkpoints(args.base.resolve(), args.adapter.resolve())
    verification_seconds = time.perf_counter() - verification_started
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    from jemm.model import DecisionModel
    from jemm.serve import make_handler
    if Path(inspect.getfile(DecisionModel)).resolve() != source / "jemm" / "model.py":
        raise ValueError("imported JEMM is not the validated pinned upstream source")
    started = time.perf_counter()
    decision = DecisionModel.from_pretrained(str(args.adapter.resolve()), base=str(args.base.resolve()), device=args.device, name="JEMM")
    environment = loaded_environment(decision, provenance, time.perf_counter() - started)
    environment.update(checkpoint_sha256=checkpoint_hashes, checkpoint_verification_seconds=verification_seconds,
                       captured_corpus_sha256=capture_hash)
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "reference-environment.json").write_text(json.dumps(environment, indent=2, allow_nan=False) + "\n")
    if args.capture_corpus:
        capture_contract(decision, capture_cases, args.evidence / "contract")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(decision))
    server.daemon_threads = True
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "model": "JEMM", "load_seconds": environment["load_seconds"]}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
