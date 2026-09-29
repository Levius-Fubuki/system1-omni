"""Require the complete publication matrix and verify its Git source hashes."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))


def main():
    from verify_graph_admission import verify_report

    report = json.loads((HERE / "report.json").read_text())
    verify_report(report)
    measured = "9f2c7ef1cb1778b573d0c8f844c2b5be0710cf1a"
    historical = "2336a085fc090c1c6ac3a297f91d826b66949f30"
    assert report["repository"]["revision"] == measured
    assert report["legacy"]["revision"] == historical
    assert report["comparison_kind"] == "legacy-policy-with-execution-fix"
    assert report["config"]["runs"] == 2
    expected = {"hot_four": 48, "churn_twelve": 72, "hot_cold": 40, "shifting_hot": 128}
    assert set(report["workloads"]) == set(expected)
    count = 0
    for name, length in expected.items():
        runs = report["workloads"][name]
        assert len(runs) == 2
        for run in runs:
            assert len(run["events"]) == length
            count += length
    assert count == 576
    for revision, key in [
        (historical, "sha256"),
        (measured, "current_graph_runtime_sha256"),
    ]:
        source = subprocess.check_output(
            [
                "git",
                "show",
                revision + ":src/models/cua_s1/multimodal/graph_runtime.py",
            ],
            cwd=ROOT,
        )
        assert hashlib.sha256(source).hexdigest() == report["legacy"][key]
    print(
        "Publication gate passed: four schedules x two runs; 576 triplets / 1728 predictions; Git hashes match"
    )


if __name__ == "__main__":
    main()
