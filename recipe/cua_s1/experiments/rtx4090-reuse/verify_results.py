"""Independently check saved samples and parity evidence, then print the table."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads((ROOT / "paired.json").read_text())
    reference = json.loads((ROOT.parent / "rtx4090-profile/reference.json").read_text())
    assert report["status"] == "complete"
    assert report["repository"]["dirty"] is False
    assert report["environment"] == reference["environment"]
    expected_cases = {
        f"{size}-{goal}-q{questions}"
        for size in ("320x240", "640x480")
        for goal in ("short", "long")
        for questions in (1, 2, 4, 8)
    } | {"640x480-distinct-q8"}
    assert {case["id"] for case in report["cases"]} == expected_cases
    assert len(report["cases"]) == 17
    assert len(report["correctness"]) == 13
    original = {case["id"]: case for case in report["correctness"]}
    for case in reference["cases"]:
        checked = original[case["name"]]
        answers = checked["response"]["answers"]
        assert (
            list(answers[case["question"]]["probabilities"].values())
            == case["probabilities"]
        )
        index = list(answers).index(case["question"])
        assert checked["inputs"][index] == case["inputs"]
    count = 0
    for checked in report["correctness"] + [c["validation"] for c in report["cases"]]:
        assert checked["exact_prepared_tensor_parity"]
        assert checked["exact_language_input_parity"]
        assert checked["exact_response_parity"]
        n = len(checked["response"]["answers"])
        assert checked["baseline_counts"] == {
            "image_preprocess": n,
            "vision": n,
            "language": n,
        }
        assert checked["reuse_counts"] == {
            "image_preprocess": 1,
            "vision": 1,
            "language": n,
        }
    print(
        "| Case | Baseline p50 ms | Reuse p50 ms | p50 reduction | Baseline p95 ms | Reuse p95 ms | Peak allocated GiB, baseline → reuse |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in sorted(report["cases"], key=lambda c: c["id"]):
        assert case["status"] == "complete"
        assert len(case["runs"]) == report["config"]["runs"] == 2
        for index, run in enumerate(case["runs"]):
            for iteration, order in enumerate(run["orders"]):
                expected = ["baseline", "reuse"]
                if (index + iteration) % 2:
                    expected.reverse()
                assert order == expected
            for name in ("baseline", "reuse"):
                values = run[name]["latencies_ms"]
                assert (
                    len(values)
                    == len(run["orders"])
                    == report["config"]["iterations"]
                    == 50
                )
                assert all(math.isfinite(v) and v > 0 for v in values)
                assert statistics.median(values) == run[name]["p50_ms"]
                assert (
                    sorted(values)[math.ceil(0.95 * len(values)) - 1]
                    == run[name]["p95_ms"]
                )
                assert (
                    run[name]["peak_reserved_bytes"]
                    >= run[name]["peak_allocated_bytes"]
                    > 0
                )
                count += len(values)

        def times(name, percentile):
            return " / ".join(f"{r[name][percentile]:.2f}" for r in case["runs"])

        reduction = " / ".join(
            f"{100 * (1 - r['reuse']['p50_ms'] / r['baseline']['p50_ms']):.1f}%"
            for r in case["runs"]
        )
        memory = " → ".join(
            f"{max(r[name]['peak_allocated_bytes'] for r in case['runs']) / 2**30:.3f}"
            for name in ("baseline", "reuse")
        )
        print(
            f"| {case['id']} | {times('baseline', 'p50_ms')} | {times('reuse', 'p50_ms')} | {reduction} | {times('baseline', 'p95_ms')} | {times('reuse', 'p95_ms')} | {memory} |"
        )
    assert count == 3400
    print(
        f"\nVerified {count} timed requests, 13 correctness fixtures, 17 paired cases, and all 9 saved upstream forwards."
    )


if __name__ == "__main__":
    main()
