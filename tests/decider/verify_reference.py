#!/usr/bin/env python3
"""Pinned full-checkpoint eager parity; no latency or task-quality claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

GATES = {"max_probability_drift": 0.02, "max_score_drift": 0.1, "discrete_margin": 0.05}
REFERENCE_REVISION = "50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f"
REFERENCE_FILES = {
    "model.py": "74e0ca564bb52f7c2bd74c23cb34b16e0004ceadf7a13816f223c68829b5e1c5",
    "prompt.py": "f21ae1016a11ca3500b5a8e2e8c3ef524434d196bbd4891199b96a33b3ad6314",
    "prompt_fast.py": "fbfcc3a381a89785174cb9d29083ed14d404c7273d123ba8fe2b69584649ef25",
    "systemone.py": "884e4b0bae36f5e09dbd689b4390e94320cbbd77f7b8d675e50490b1d7c61d34",
    "temperature.py": "b940c253e5626145e806195cf5b2bc0de61461de633c07b174a924d18ceeab88",
}


def verify_sources(directory, expected):
    verified = {}
    for name, checksum in expected.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f"reference checksum mismatch: {name}")
        verified[name] = actual
    return verified


def workloads():
    cases = []
    for n in (2, 3, 10, 11, 26, 255):
        cases.append((f"choice_{n}", {"state": "The customer asks for a refund for a damaged order.", "questions": {
            "category": {"type": "choice", "instructions": "Choose the matching category.", "criteria": {
                f"option_{i}": "Refund request" if i == n - 1 else f"Unrelated category {i}" for i in range(n)}}}}))
    for n in (2, 3, 10):
        cases.append((f"score_{n}", {"state": "The customer received a damaged item and requests help immediately.", "questions": {
            "urgency": {"type": "score", "instructions": "How urgent is the customer request?", "criteria": [f"{i}. Urgency level {i}" for i in range(n)]}}}))
    cases.extend([
        ("noul_true", {"state": "The customer requests a refund.", "questions": {"refund": {"type": "noul", "instructions": "The customer requests a refund."}}}),
        ("noul_false", {"state": "The customer asks when the package will arrive.", "questions": {"refund": {"type": "noul", "instructions": "The customer requests a refund."}}}),
        ("noul_descriptions", {"state": "The parcel is delivered.", "questions": {"delivered": {"type": "noul", "criteria": {"true": "The parcel is delivered.", "false": "The parcel is still in transit."}}}}),
        ("structured_unicode", {"state": {"客户": "订单损坏，请退款", "orders": [{"id": i, "status": "damaged" if i == 7 else "delivered"} for i in range(9)]}, "questions": {
            "z_last": {"type": "choice", "instructions": {"task": "Choose the status of orders[7]."}, "criteria": {"damaged": "Damaged", "delivered": "Delivered"}},
            "a_first": {"type": "score", "instructions": "Severity of the damage", "criteria": {"0": "Low", "1": "Medium", "2": "High"}},
            "m_middle": {"type": "noul", "instructions": "The customer requests a refund."}}}),
        ("mixed", {"state": "A customer requests a refund because a delivery was damaged.", "questions": {
            "route": {"type": "choice", "instructions": "Choose the team.", "criteria": {"billing": "Refund and billing", "delivery": "Shipping tracking", "sales": "New purchases"}},
            "refund": {"type": "noul", "instructions": "The customer requests a refund."},
            "severity": {"type": "score", "instructions": "Severity of the damage", "criteria": ["Minor", "Moderate", "Severe"]}}}),
        ("long_context", {"state": "The item is damaged. " * 700 + "The customer requests a refund.", "questions": {"route": {"type": "choice", "instructions": "Choose the team.", "criteria": {"billing": "Refund and billing", "shipping": "Delivery status"}}}}),
        ("empty", {"state": {}, "questions": {}}),
    ])
    # Repetition and reverse question order test request/state isolation.
    cases.append(("mixed_reordered", {"state": cases[13][1]["state"], "questions": dict(reversed(list(cases[13][1]["questions"].items())))}))
    cases.append(("choice_repeat", cases[0][1]))
    return cases


def compare(reference, native):
    assert native["model"] == "decider-2b-v11"
    assert native["usage"] == reference["usage"]
    assert list(native["answers"]) == list(reference["answers"])
    worst_p, worst_score = 0.0, 0.0
    mismatches, low_margin = [], []
    for key, ref in reference["answers"].items():
        got = native["answers"][key]
        assert set(got) == set(ref), (key, set(got), set(ref))
        assert got["type"] == ref["type"]
        if ref["type"] == "noul":
            drift = abs(got["noul"] - ref["noul"])
        else:
            assert list(got["probabilities"]) == list(ref["probabilities"])
            drift = max(abs(got["probabilities"][k] - v) for k, v in ref["probabilities"].items())
            if ref["type"] == "choice":
                probabilities = sorted(ref["probabilities"].values(), reverse=True)
                if ref["choice"] != got["choice"]:
                    (mismatches if probabilities[0] - probabilities[1] >= GATES["discrete_margin"] else low_margin).append(key)
            else:
                assert got["legend"] == ref["legend"]
                worst_score = max(worst_score, abs(got["score"] - ref["score"]))
                drift = max(drift, max(abs(got["level_fit"][k] - v) for k, v in ref["level_fit"].items()))
        worst_p = max(worst_p, drift)
    assert worst_p <= GATES["max_probability_drift"], ("probability", worst_p)
    assert worst_score <= GATES["max_score_drift"], ("score", worst_score)
    assert not mismatches, ("discrete", mismatches)
    return {"max_probability_drift": worst_p, "max_score_drift": worst_score, "low_margin_disagreements": low_margin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = verify_sources(args.reference / "decider", REFERENCE_FILES)
    cases = workloads()
    manifest = {"reference_revision": REFERENCE_REVISION, "reference_source_sha256": sources, "gates": GATES, "runs": 1, "cases": [{"name": name, "request": request} for name, request in cases]}
    # Freeze the manifest before executing either implementation.
    (args.output / "protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    inputs = "".join(json.dumps(request, ensure_ascii=False) + "\n" for _, request in cases)
    (args.output / "requests.jsonl").write_text(inputs)
    with (args.output / "native.stderr.log").open("w") as stderr:
        native = subprocess.run([str(args.binary), str(args.model), str(args.library)], input=inputs, capture_output=False,
                                stdout=subprocess.PIPE, stderr=stderr, text=True, check=True)
    (args.output / "native.jsonl").write_text(native.stdout)
    native = [json.loads(line) for line in native.stdout.splitlines()]
    assert len(native) == len(cases)

    # Import only the pinned reference package, with no remote-code execution.
    package = types.ModuleType("decider")
    package.__path__ = [str(args.reference / "decider")]
    sys.modules["decider"] = package
    import torch
    from transformers import AutoTokenizer
    from decider import systemone as s1, prompt_fast, temperature as tt, prompt
    from decider.model import DecisionModel

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    scalar, temperatures = tt.from_config(json.loads((args.model / "decider_config.json").read_text()))[0]
    del scalar
    labels = prompt.label_table(tokenizer)[1]
    model = DecisionModel(str(args.model), dtype=torch.bfloat16, grad_ckpt=False).to("cuda").eval()
    model.lm.model.config.use_cache = False
    results = []
    with (args.output / "reference.jsonl").open("w") as records, torch.inference_mode():
        for (name, request), got in zip(cases, native):
            rendered = {key: s1.render_question(value) for key, value in request["questions"].items()}
            plans, index = s1.plan_rows(rendered, True)
            items, _ = prompt_fast.build_rows(tokenizer, s1.render_state(request["state"]),
                                              [[(plan["question"], plan["options"])] for plan in plans], max_ctx_tokens=32768)
            kinds = s1.row_types(rendered, index)
            assert len(got["rows"]) == len(items), name
            logits, probabilities = [], []
            max_logit_drift = 0.0
            for i, (item, kind) in enumerate(zip(items, kinds)):
                count = item["nopts"][0]
                row = got["rows"][i]
                assert row["ids"] == item["ids"], (name, i, "tokens")
                assert row["candidate_ids"] == labels[:count], (name, i, "candidate IDs")
                assert row["readout_position"] == item["slots"][0] == len(item["ids"]) - 1
                ids = torch.tensor([item["ids"]], device="cuda", dtype=torch.long)
                lg = model.slot_logits(ids, torch.ones_like(ids), torch.tensor([item["slots"][0]], device="cuda"),
                                       torch.tensor([0], device="cuda"), torch.tensor([count], device="cuda"))[0, :count]
                logits.append(lg.cpu().tolist())
                probabilities.append(tt.scaled_softmax(lg[None], [temperatures[kind]])[0].cpu().tolist())
                max_logit_drift = max(max_logit_drift, max(abs(a - b) for a, b in zip(logits[-1], got["logits"][i])))
            response = {"model": "decider-2b-v11", "answers": s1.assemble(rendered, index, probabilities),
                        "usage": {"input_tokens": s1.unique_tokens(items), "output_tokens": 0}}
            records.write(json.dumps({"name": name, "logits": logits, "probabilities": probabilities, "response": response}, ensure_ascii=False) + "\n")
            records.flush()
            result = {"name": name, "rows": len(items), "processed_tokens": sum(len(item["ids"]) for item in items),
                      "max_logit_drift": max_logit_drift, **compare(response, got["response"])}
            results.append(result)
            print(json.dumps(result), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summary = {"passed": len(results), "cases": results, "gates": GATES,
               "torch": torch.__version__, "transformers": __import__("transformers").__version__,
               "gpu": torch.cuda.get_device_name(), "native_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
