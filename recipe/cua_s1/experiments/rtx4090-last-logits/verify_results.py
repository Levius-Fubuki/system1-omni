"""Independently audit every saved final-token projection sample."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ("full_projection", "last_token_projection")


def main():
    report = json.loads((ROOT / "paired.json").read_text())
    correctness = json.loads((ROOT / "correctness.json").read_text())
    postflight = json.loads((ROOT / "postflight.json").read_text())
    assert report["status"] == "complete"
    assert report["repository"]["dirty"] is False
    assert correctness["status"] == postflight["status"] == "complete"
    for checked in (correctness, postflight):
        assert checked["repository"]["revision"] == report["repository"]["revision"]
        assert checked["repository"]["dirty"] is False
        assert checked["environment"] == report["environment"]
    assert postflight["health"]["status"] == postflight["http"]["status"] == 200
    assert postflight["http"]["exact_engine_parity"]
    assert postflight["http"]["question_count"] == 8
    assert len(correctness["correctness"]) == 13
    assert (
        sum(len(case["response"]["answers"]) for case in correctness["correctness"])
        == 49
    )
    for case in correctness["correctness"]:
        assert case["exact_prepared_tensor_parity"]
        assert case["exact_language_input_parity"]
        assert case["exact_response_parity"]
        n = len(case["response"]["answers"])
        assert case["baseline_counts"] == {
            "image_preprocess": n,
            "vision": n,
            "language": n,
        }
        assert case["reuse_counts"] == {
            "image_preprocess": 1,
            "vision": 1,
            "language": n,
        }
    assert report["config"]["case"] == []
    assert (report["config"]["runs"], report["config"]["iterations"]) == (2, 30)
    expected = {
        f"{size}-{goal}-q{questions}"
        for size in ("320x240", "640x480")
        for goal in ("short", "long")
        for questions in (1, 2, 4, 8)
    } | {"640x480-distinct-q8"}
    cases = report["cases"]
    assert len(cases) == 17
    assert {case["id"] for case in cases} == expected
    total = 0
    print(
        "| Case | Full p50 ms | Last p50 ms | p50 reduction | Full p95 ms | Last p95 ms | Peak allocated GiB, full → last |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in sorted(cases, key=lambda value: value["id"]):
        assert case["status"] == "complete" and case["exact_response_parity"]
        n = case["questions"]
        assert len(case["full_head_shapes"]) == len(case["last_head_shapes"]) == n
        for full, last in zip(case["full_head_shapes"], case["last_head_shapes"]):
            assert len(full) == len(last) == 3
            assert full[0] == last[0] == 1 and full[2] == last[2] == 2560
            assert full[1] > 1
            assert last[1] == (full[1] if n == 1 else 1)
        assert len(case["runs"]) == 2
        for run_index, run in enumerate(case["runs"]):
            assert run["run"] == run_index + 1
            assert len(run["orders"]) == 30
            for iteration, order in enumerate(run["orders"]):
                selected = list(VARIANTS)
                if (run_index + iteration) % 2:
                    selected.reverse()
                assert order == selected
            for name in VARIANTS:
                values = run[name]["latencies_ms"]
                assert len(values) == 30
                assert all(math.isfinite(value) and value > 0 for value in values)
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
                total += len(values)

        def times(name, metric, case=case):
            return " / ".join(f"{run[name][metric]:.2f}" for run in case["runs"])

        reduction = " / ".join(
            f"{100 * (1 - run['last_token_projection']['p50_ms'] / run['full_projection']['p50_ms']):.1f}%"
            for run in case["runs"]
        )
        memory = " → ".join(
            f"{max(run[name]['peak_allocated_bytes'] for run in case['runs']) / 2**30:.3f}"
            for name in VARIANTS
        )
        print(
            f"| {case['id']} | {times('full_projection', 'p50_ms')} | "
            f"{times('last_token_projection', 'p50_ms')} | {reduction} | "
            f"{times('full_projection', 'p95_ms')} | "
            f"{times('last_token_projection', 'p95_ms')} | {memory} |"
        )
    assert total == 2040
    print(f"Verified {len(cases)} cases and {total} timed predictions.")


if __name__ == "__main__":
    main()
