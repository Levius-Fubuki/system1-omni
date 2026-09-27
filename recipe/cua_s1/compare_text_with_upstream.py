"""Compare the Cua-S1 worker with upstream `FourBModel` on the fixed input set.

Needs a checkout of trycua/cua at the pinned commit (for `cua_s1.four_b` and
the jev-use chooser) and the pinned weights. The worker's model scores every
question first and is freed; then `FourBModel` is loaded with the same device
and dtype and scores the same questions. For every question it checks:

- prompt token ids: worker vs upstream `build_prompt` plus the chat template;
- probabilities: worker vs `FourBModel.forward`, exact fp32 equality;
- for the two upstream fixtures, also worker vs the chooser's own path
  (`S1DecisionModel.score`), matched by option key.

Upstream `build_prompt` is given the worker's mapped labels, state and goal,
so the id check covers the prompt layout, chat template and tokenizer. The
request mapping itself (escaping, structured values, `null` labels) is
covered by the unit tests and, independently, by the two fixtures.

Only one model is resident at a time. Both compute logits for every prompt
position, so the longest input (15,446 tokens) needs about 8 GB for logits in
bfloat16 and 15 GB in float32, on top of the weights.

Run from the repository root:

    python recipe/cua_s1/compare_text_with_upstream.py --upstream ../cua \\
        --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = {
    "fixture_positive": "jev-choice-request-v1.json",
    "fixture_negative": "jev-choice-negative-v1.json",
}


def free(device: str) -> None:
    import torch

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def peak_gib(device: str) -> float | None:
    import torch

    if not device.startswith("cuda"):
        return None
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return round(peak, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upstream", required=True, help="trycua/cua checkout at the pinned commit"
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--inputs", default=str(ROOT / "tests/cua_s1/data/text_inputs.json")
    )
    parser.add_argument("--out", help="write one JSON line per question here")
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="disable TF32 in cuBLAS and cuDNN (use for fp32 reference runs)",
    )
    args = parser.parse_args()

    upstream = Path(args.upstream)
    sys.path.insert(0, str(upstream / "libs/cua-s1/python/src"))
    sys.path.insert(0, str(upstream / "libs/cua-driver/examples/jev-use/python"))
    import torch

    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from cua_s1.four_b import FourBModel, Option, assign_letters, build_prompt
    from decision_models import DecisionRequest, S1DecisionModel

    from models.cua_s1.text.contract import (
        ACTION,
        APP,
        ROLE,
        TASK_FAMILY,
        map_request,
        parse_body,
    )
    from models.cua_s1.text.engine import TextEngine

    cases = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    questions = []
    for name, body in cases.items():
        request = map_request(parse_body(json.dumps(body).encode()))
        questions += [(name, request, question) for question in request.questions]

    # Pass 1: the worker.
    engine = TextEngine(args.base, args.adapter, args.device, args.dtype)
    print(f"worker loaded in {engine.load_seconds:.1f} s", flush=True)
    worker = {}
    for name, request, question in questions:
        worker[name, question.name] = (
            engine.prompt_ids(request.state, question),
            engine.score(request.state, question).probabilities,
        )
    worker_peak = peak_gib(args.device)
    del engine
    free(args.device)

    # Pass 2: upstream FourBModel, and the chooser for the two fixtures.
    started = time.perf_counter()
    reference = FourBModel(
        base_model=args.base,
        lora_adapter_path=args.adapter,
        device=args.device,
        dtype=args.dtype,
        modality="text",
    )
    reference.load()
    print(f"upstream loaded in {time.perf_counter() - started:.1f} s", flush=True)
    fixture_dir = upstream / "libs/cua-driver/examples/jev-use/fixtures"
    rows, failures = [], 0
    for name, request, question in questions:
        options = [
            Option(element_id=k, role=ROLE, label=label, action=ACTION)
            for k, label in zip(question.keys, question.labels, strict=True)
        ]
        kwargs = dict(
            app=APP,
            task_family=TASK_FAMILY,
            ax_tree=request.state,
            modality="text",
            goal=question.goal or None,
        )
        messages = build_prompt(assign_letters(options), **kwargs)
        chat = reference._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        upstream_ids = reference._tokenizer(chat)["input_ids"]
        upstream_p = [r.probability for r in reference.forward(options, **kwargs)]
        worker_ids, worker_p = worker[name, question.name]
        row = {
            "case": name,
            "question": question.name,
            "options": len(options),
            "prompt_tokens": len(worker_ids),
            "ids_equal": worker_ids == upstream_ids,
            "probs_equal": worker_p == upstream_p,
            "max_abs_diff": max(
                abs(a - b) for a, b in zip(worker_p, upstream_p, strict=True)
            ),
            "worker": dict(zip(question.keys, worker_p, strict=True)),
            "upstream": dict(zip(question.keys, upstream_p, strict=True)),
        }
        if name in FIXTURES:
            raw = json.loads((fixture_dir / FIXTURES[name]).read_text(encoding="utf-8"))
            chooser = (
                S1DecisionModel(reference, modality="text")
                .score(DecisionRequest.from_validated(raw))
                .probabilities
            )
            row["chooser_equal"] = all(
                chooser.get(k) == p for k, p in row["worker"].items()
            )
        ok = row["ids_equal"] and row["probs_equal"] and row.get("chooser_equal", True)
        failures += not ok
        rows.append(row)
        print(
            f"{'ok  ' if ok else 'FAIL'} {name}/{question.name}: {len(options)} options, "
            f"{len(worker_ids)} tokens, max |diff| {row['max_abs_diff']:.3g}",
            flush=True,
        )
    upstream_peak = peak_gib(args.device)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if worker_peak is not None:
        print(f"peak allocated: worker {worker_peak} GiB, upstream {upstream_peak} GiB")
    print(
        f"{len(rows) - failures}/{len(rows)} questions identical "
        f"(device {args.device}, dtype {args.dtype}, torch {torch.__version__}, "
        f"tf32 {'off' if args.no_tf32 else 'default'})"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
