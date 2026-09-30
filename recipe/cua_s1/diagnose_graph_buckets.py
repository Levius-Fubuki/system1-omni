"""Locate padding-induced rounding before changing bucket acceptance criteria."""

from __future__ import annotations

import argparse
from pathlib import Path


def prepare_values(engine, request):

    inputs = engine.prepare_reused(request.image, request.questions)[0]
    features = engine.encode_image(inputs)
    text_inputs = {
        k: v.to(engine.model.device) for k, v in inputs.items() if k != "pixel_values"
    }
    core = engine.model.get_base_model().model
    ids = text_inputs.pop("input_ids")
    embeds = core.get_input_embeddings()(ids)
    image_embeds = features.to(embeds.device, embeds.dtype)
    mask, _ = core.get_placeholder_mask(
        ids, inputs_embeds=embeds, image_features=image_embeds
    )
    embeds = embeds.masked_scatter(mask, image_embeds)
    positions, _ = core.get_rope_index(
        input_ids=ids,
        mm_token_type_ids=text_inputs.pop("mm_token_type_ids"),
        image_grid_thw=text_inputs.pop("image_grid_thw"),
        attention_mask=text_inputs["attention_mask"],
    )
    return {"inputs_embeds": embeds, "position_ids": positions, **text_inputs}


def difference(a, b):
    import torch

    delta = (a.float() - b.float()).abs()
    return {
        "equal": torch.equal(a, b),
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "unequal": int((a != b).sum().item()),
        "elements": a.numel(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output")
    from dataclasses import replace

    import torch
    from evaluate_multimodal import environment
    from graph_buckets import BucketRuntime, pack_hidden
    from profile_multimodal import (
        GOAL,
        case_matrix,
        fixture,
        repository_state,
        write_json,
    )

    from models.cua_s1.multimodal.graph_runtime import _GraphSegment, _ShapeEntry
    from models.cua_s1.multimodal.model import MultimodalEngine, letter_ids
    from models.cua_s1.multimodal.protocol import parse_request

    report = {
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "environment": environment(),
        "cases": [],
    }
    if report["repository"]["dirty"]:
        raise ValueError("clean source required")
    engine = MultimodalEngine(
        str(args.weights / "Qwen3.5-4B"), str(args.weights / "cua-s1-4b-0.2/multimodal")
    )
    case = next(c for c in case_matrix() if c["id"] == "320x240-short-q2")
    base = parse_request(fixture(case, args.output.parent / "diagnostic-fixtures"))
    layers = engine.model.get_base_model().model.language_model.layers

    def run(hidden):
        for layer in layers[:3]:
            hidden = layer(
                hidden,
                position_embeddings=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
            )
        return hidden

    try:
        with torch.no_grad():
            for count in (1, 2, 5, 6):
                request = replace(
                    base,
                    questions=tuple(
                        replace(q, goal=" ".join([GOAL] * count))
                        for q in base.questions
                    ),
                )
                values = prepare_values(engine, request)
                hidden = values["inputs_embeds"]
                length = hidden.shape[1]
                padded = pack_hidden(hidden, 64)
                record = {"length": length, "padded_length": padded.shape[1]}
                observed = {}
                handles = []
                target = {}

                def hook(name):
                    def save(module, inputs, output):
                        if isinstance(output, torch.Tensor):
                            target[name] = output.detach().clone()

                    return save

                for i in range(3):
                    for name, module in layers[i].named_modules():
                        if not list(module.children()):
                            handles.append(
                                module.register_forward_hook(hook(f"{i}.{name}"))
                            )
                target = observed
                exact = run(hidden)
                padded_observed = {}
                target = padded_observed
                padded_eager = run(padded)[:, :length]
                for handle in handles:
                    handle.remove()
                modules = []
                for name, a in observed.items():
                    b = padded_observed[name]
                    if a.shape != b.shape:
                        # Leaf modules here contain one token dimension, or a
                        # flattened (batch * tokens * heads) norm dimension.
                        slices = tuple(slice(0, size) for size in a.shape)
                        b = b[slices]
                    modules.append(
                        {"name": name, "shape": list(a.shape), **difference(a, b)}
                    )
                record["modules"] = modules
                record["segment_padded_eager"] = difference(exact, padded_eager)
                exact_graph = _GraphSegment(layers, 0, 3, hidden, None)
                record["segment_exact_graph"] = difference(
                    exact, exact_graph.replay(hidden, None)
                )
                exact_graph.close()
                padded_graph = _GraphSegment(layers, 0, 3, padded, None)
                record["segment_padded_graph_vs_padded_eager"] = difference(
                    padded_eager, padded_graph.replay(padded, None)[:, :length]
                )
                padded_graph.close()
                runtime = BucketRuntime(engine.model)
                reference = runtime._eager(values)
                entry = _ShapeEntry()
                unsafe = runtime._run_segments(values, entry)
                record["full_logits"] = difference(reference, unsafe)
                ids = letter_ids(engine.tokenizer, len(request.questions[0].keys))
                a = torch.softmax(reference[ids].float(), -1)
                b = torch.softmax(unsafe[ids].float(), -1)
                record["candidate_probabilities"] = {
                    "reference": a.tolist(),
                    "padded": b.tolist(),
                    **difference(a, b),
                }
                entry.close()
                report["cases"].append(record)
                write_json(args.output, report)
                print(
                    length,
                    "first difference",
                    next((m for m in modules if not m["equal"]), None),
                    "full logits",
                    record["full_logits"],
                    flush=True,
                )
        report["status"] = "complete"
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
