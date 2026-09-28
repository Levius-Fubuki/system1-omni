"""Recompute the published CUDA Graph experiment from saved raw samples."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check():
    report = json.loads((ROOT / "paired.json").read_text())
    rejected = json.loads((ROOT / "distinct-rejected.json").read_text())
    long_rejected = json.loads((ROOT / "long-rejected.json").read_text())
    assert report["status"] == "complete"
    assert report["repository"]["dirty"] is False
    assert len(report["cases"]) == 3
    assert report["config"]["runs"] == 2
    assert report["config"]["iterations"] == 20
    count = 0
    for case in report["cases"]:
        assert case["status"] == "complete"
        assert case["exact_eager_response_parity"]
        assert case["changed_image_replay_checked"]
        assert case["changed_text_replay_checked"]
        assert case["graph_count"] == 1
        assert len(case["capture_ms"]) == 1
        assert len(case["capture_allocated_delta_bytes"]) == 1
        assert len(case["capture_reserved_delta_bytes"]) == 1
        assert case["max_graph_probability_difference"] <= 0.002
        assert case["changed_image_max_probability_difference"] <= 0.002
        assert case["changed_text_max_probability_difference"] <= 0.002
        assert len(case["runs"]) == 2
        for run_index, run in enumerate(case["runs"]):
            assert len(run["orders"]) == 20
            for sample_index, order in enumerate(run["orders"]):
                expected = ["eager", "graph"]
                if (run_index + sample_index) % 2:
                    expected.reverse()
                assert order == expected
            for variant in ("eager", "graph"):
                values = run[variant]["samples_ms"]
                assert len(values) == 20 and all(value > 0 for value in values)
                ordered = sorted(values)
                assert run[variant]["p50_ms"] == statistics.median(values)
                assert run[variant]["p95_ms"] == ordered[19]
                assert run[variant]["peak_allocated_bytes"] > 0
                count += len(values)

    assert rejected["status"] == "failed"
    assert rejected["repository"]["dirty"] is False
    assert len(rejected["cases"]) == 1
    failure = rejected["cases"][0]
    assert failure["id"] == "640x480-distinct-q8"
    assert failure["status"] == "rejected_graph"
    assert failure["max_graph_probability_difference"] > 0.002
    assert len(failure["parity_diagnostic"]["graph_shapes"]) > 1
    assert long_rejected["status"] == "failed"
    assert long_rejected["repository"]["dirty"] is False
    assert len(long_rejected["cases"]) == 1
    failure = long_rejected["cases"][0]
    assert failure["id"] == "640x480-long-q8"
    assert failure["status"] == "rejected_graph"
    assert failure["max_graph_probability_difference"] <= 0.002
    assert failure["changed_image_max_probability_difference"] <= 0.002
    assert failure["changed_text_max_probability_difference"] > 0.002
    print(f"Verified {count} paired samples across three cases and two rejected cases.")


if __name__ == "__main__":
    check()
