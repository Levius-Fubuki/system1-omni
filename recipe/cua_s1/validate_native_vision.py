"""Export fresh, full-vision unmerged BF16/FP32 controls for native RGB replay.

Run with PYTHONPATH=src in the pinned reference environment. Output contains
RGB inputs, token/position oracles, vision traces and full-model probabilities.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--requests", required=True, help="directory of screenshot request JSON files"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from safetensors.torch import save_file

    from models.cua_s1.multimodal.model import (
        ADAPTER_REVISION,
        BASE_REVISION,
        MultimodalEngine,
        letter_ids,
    )
    from models.cua_s1.multimodal.protocol import decode_request, parse_request

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = []
    for path in sorted(Path(args.requests).glob("*.json")):
        raw = path.read_bytes()
        request = parse_request(decode_request(raw))
        rgb = request.image.tobytes()
        name = path.stem
        (out / f"{name}.rgb").write_bytes(rgb)
        (out / f"{name}.json").write_bytes(raw)
        cases.append(
            (
                name,
                request,
                {
                    "case": name,
                    "width": request.image.width,
                    "height": request.image.height,
                    "rgb_file": f"{name}.rgb",
                    "request_file": f"{name}.json",
                    "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                    "questions": [],
                },
            )
        )
    for dtype in ["bfloat16", "float32"]:
        engine = MultimodalEngine(args.base, args.adapter, dtype=dtype)
        core = engine.model.get_base_model().model
        for name, request, record in cases:
            prepared = engine.prepare_reused(request.image, request.questions)
            trace = {}
            handles = []

            def hook(key, storage=trace):
                def keep(module, ins, output):
                    if hasattr(output, "pooler_output"):
                        output = output.pooler_output
                    storage[key] = output.detach().cpu().contiguous().clone()

                return keep

            if name == "small" and dtype == "bfloat16":
                handles.append(
                    core.visual.patch_embed.register_forward_hook(hook("patch_embed"))
                )

                def keep_position(module, ins, kwargs, storage=trace):
                    x = ins[0] if ins else kwargs["hidden_states"]
                    storage["position"] = x.detach().cpu().contiguous().clone()

                handles.append(
                    core.visual.blocks[0].register_forward_pre_hook(
                        keep_position, with_kwargs=True
                    )
                )
                for i, block in enumerate(core.visual.blocks):
                    handles.append(block.register_forward_hook(hook(f"blocks.{i}")))
                handles.append(
                    core.visual.merger.norm.register_forward_hook(hook("merger.norm"))
                )
                handles.append(
                    core.visual.merger.linear_fc1.register_forward_hook(
                        hook("merger.fc1")
                    )
                )
            with torch.no_grad():
                features = engine.encode_image(prepared[0])
            for handle in handles:
                handle.remove()
            tensors = {"image_features": features.detach().cpu().contiguous()}
            if trace:
                trace["merger.output"] = tensors["image_features"].clone()
                save_file(trace, str(out / "vision-trace.safetensors"))
            for idx, (q, inputs) in enumerate(zip(request.questions, prepared)):
                with torch.no_grad():
                    device_inputs = {
                        k: v.cuda() for k, v in inputs.items() if k != "pixel_values"
                    }
                    ids = device_inputs["input_ids"]
                    pos, _ = core.get_rope_index(
                        input_ids=ids,
                        mm_token_type_ids=device_inputs["mm_token_type_ids"],
                        image_grid_thw=device_inputs["image_grid_thw"],
                        attention_mask=device_inputs["attention_mask"],
                    )
                    embeds = core.get_input_embeddings()(ids)
                    mask = ids == engine.model.config.image_token_id
                    embeds[mask] = features.to(embeds.dtype)
                    hidden = core.language_model(
                        inputs_embeds=embeds,
                        position_ids=pos,
                        attention_mask=device_inputs["attention_mask"],
                        use_cache=False,
                    ).last_hidden_state[0, -1]
                    candidate_ids = letter_ids(engine.tokenizer, len(q.keys))
                    rows = engine.model.get_output_embeddings().weight[candidate_ids]
                    logits = torch.nn.functional.linear(hidden, rows)
                    probs = logits.float().softmax(-1).tolist()
                if dtype == "bfloat16":
                    record["grid"] = inputs["image_grid_thw"][0].tolist()
                    record["questions"].append(
                        {
                            "name": q.name,
                            "input_ids": ids[0].tolist(),
                            "position_ids": pos[:, 0].tolist(),
                            "candidate_ids": candidate_ids,
                            "bf16_probabilities": probs,
                        }
                    )
                    tensors["pixel_values"] = inputs["pixel_values"].contiguous()
                else:
                    record["questions"][idx]["fp32_probabilities"] = probs
                tensors[f"hidden.{idx}"] = hidden.detach().cpu().contiguous()
                print(dtype, name, q.name, probs, flush=True)
            save_file(tensors, str(out / f"{name}.{dtype}.safetensors"))
            del (
                features,
                hidden,
                embeds,
                rows,
                tensors,
                logits,
                trace,
                inputs,
                device_inputs,
                ids,
                pos,
            )
        del engine, core
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {
        "schema": "cua-s1-native-vision-controls-v1",
        "base_revision": BASE_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "tf32": False,
        "control": "full-vision unmerged adapter, candidate projection at last position",
        "acceptance": "max native probability error <= 2 * max BF16 probability error + 0.01; same top choice for FP32 margin >= 0.05",
        "cases": [r for _, _, r in cases],
    }
    manifest["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file()
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
