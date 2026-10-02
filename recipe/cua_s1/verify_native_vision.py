"""Fail closed on missing coverage or violations of the repository parity rule."""

import argparse
import json
import math
from pathlib import Path


def verify(control, native):
    if (
        control["schema"] != "cua-s1-native-vision-controls-v1"
        or native["schema"] != "cua-s1-native-vision-results-v1"
    ):
        raise ValueError("unsupported schema")
    refs, results = control["cases"], native["cases"]
    if not refs or len(refs) != len(results):
        raise ValueError("case coverage mismatch")
    errors, bf_errors, choices = [], [], []
    for ref, result in zip(refs, results):
        if ref["case"] != result["case"] or ref["grid"] != result["grid"]:
            raise ValueError("case/grid mismatch")
        if len(ref["questions"]) != len(result["questions"]):
            raise ValueError("question coverage mismatch")
        for r, n in zip(ref["questions"], result["questions"]):
            if r["name"] != n["name"] or not all(
                n[k] is True
                for k in ["token_ids_equal", "position_ids_equal", "repeat_equal"]
            ):
                raise ValueError("CPU boundary or replay mismatch")
            fp, bf, actual = (
                r["fp32_probabilities"],
                r["bf16_probabilities"],
                n["probabilities"],
            )
            if not 1 <= len(fp) <= 26 or len(fp) != len(bf) or len(fp) != len(actual):
                raise ValueError("candidate count mismatch")
            for p in [fp, bf, actual]:
                if (
                    not all(math.isfinite(x) and 0 <= x <= 1 for x in p)
                    or abs(sum(p) - 1) > 1e-5
                ):
                    raise ValueError("invalid probability distribution")
            error = max(abs(x - y) for x, y in zip(fp, actual))
            errors.append(error)
            bf_errors.append(max(abs(x - y) for x, y in zip(fp, bf)))
            order = sorted(range(len(fp)), key=lambda i: -fp[i])
            margin = fp[order[0]] - fp[order[1]] if len(fp) > 1 else 1.0
            chosen = max(range(len(actual)), key=actual.__getitem__)
            if margin >= 0.05 and chosen != order[0]:
                raise ValueError(f"choice mismatch: {ref['case']}/{r['name']}")
            choices.append(
                {
                    "case": ref["case"],
                    "question": r["name"],
                    "fp32_margin": margin,
                    "same_choice": chosen == order[0],
                    "max_probability_error": error,
                }
            )
    allowance = 2 * max(bf_errors) + 0.01
    if max(errors) > allowance:
        raise ValueError(f"probability error {max(errors)} exceeds {allowance}")
    return {
        "passed": True,
        "questions": len(errors),
        "max_native_error": max(errors),
        "max_bf16_reference_error": max(bf_errors),
        "allowance": allowance,
        "choices": choices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(
        json.loads(args.control.read_text()), json.loads(args.native.read_text())
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
