"""Recompute review evidence invariants and check each measured runtime against its PR commit."""

import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path("/root/cua-review-20260930")
PY = "/root/autodl-tmp/system1-chain-validation-20260929/venv/bin/python"


def read(pr, path):
    value = json.loads((ROOT / "results" / f"pr{pr}" / path).read_text())
    assert value["status"] == "complete", (pr, path)
    if "repository" in value:
        assert value["repository"]["dirty"] is False
        revision = value["repository"]["revision"]
        repo = ROOT / f"pr{pr}"
        expected = manifest[str(pr)]["runtime_revision"]
        assert not subprocess.check_output(
            ["git", "diff", expected, revision, "--", "src"], cwd=repo
        )
        for name, digest in value.get("source_sha256", {}).items():
            content = subprocess.check_output(
                ["git", "show", revision + ":" + name], cwd=repo
            )
            assert hashlib.sha256(content).hexdigest() == digest
    return value


manifest = json.loads((ROOT / "source-manifest.json").read_text())
checks = json.loads((ROOT / "results/checks.json").read_text())
latest = {(c["pr"], c["name"]): c for c in checks}
assert len(latest) == 25, len(latest)
assert all(c["exit"] == 0 for c in latest.values())
summary = {
    "runtime_sources": manifest,
    "completed_checks": len(latest),
    "historical_failed_attempts": [c for c in checks if c["exit"]],
    "cpu": {},
    "survivors": {},
}
for n in manifest:
    summary["cpu"][n] = (
        (ROOT / "results" / f"pr{n}" / "cpu-tests.log").read_text().splitlines()[-1]
    )
for n in (22, 33, 37, 38):
    d = read(n, "shared-survivor.json" if n == 38 else "eviction.json")
    assert len(d["checks"]) == 7
    assert all(c["equal"] and c["max_abs"] == 0 for c in d["checks"])
    assert d["surviving_graph_replays_after_eviction"] == 3
    assert d["victim_released"] and d["referenced_entries_explicitly_retired"]
    assert all(p["accounted"] == p["reserved"] + p["external"] for p in d["pools"])
    assert any(p["reserved"] > p["allocated"] for p in d["pools"])
    summary["survivors"][str(n)] = {
        "comparisons": len(d["checks"]),
        "pools": d["pools"],
    }
d = read(22, "streams.json")
assert (
    d["live_unique_streams"] == 40 and d["survivor_changed_input_exact_replays"] == 60
)
d = read(22, "pools.json")
assert d["shape_count"] == 40 and d["segments_per_shape"] == 3
assert d["retired_pools_released"] and d["budget_driven_evictions"] > 0
d = read(17, "image-reuse/report.json")
assert d["correctness"]
assert all(
    c[k]
    for c in d["correctness"]
    for k in (
        "exact_prepared_tensor_parity",
        "exact_response_parity",
        "exact_language_input_parity",
    )
)
summary["image_reuse_cases"] = len(d["correctness"])
d = read(18, "last-logits/report.json")
assert len(d["cases"]) == 2
assert all(
    c["exact_response_parity"] and all(s[1] == 1 for s in c["last_head_shapes"])
    for c in d["cases"]
)
summary["projection_cases"] = len(d["cases"])
d = read(22, "graph-matrix/report.json")
assert len(d["cases"]) == 5
for c in d["cases"]:
    assert c["status"] == "complete"
    assert all(
        c[k] == 0
        for k in (
            "original_max_probability_difference",
            "changed_image_max_probability_difference",
            "changed_text_max_probability_difference",
        )
    )
    assert c["stats_final"]["captures"] > 0 and c["stats_final"]["replays"] > 0
summary["exact_graph_cases"] = len(d["cases"])
d = read(33, "admission-parity.json")
assert len(d["cases"]) == 5 and len(d["policies"]) == 4
assert all(
    r["eager_response"] == r["graph_response"] for c in d["cases"] for r in c["records"]
)
for n, path in [
    (37, "bucket-parity.json"),
    (37, "boundaries.json"),
    (37, "http.json"),
    (38, "auto-parity.json"),
    (38, "shared.json"),
    (38, "http.json"),
]:
    read(n, path)

commands = [
    (
        37,
        [
            "verify_graph_buckets.py",
            str(ROOT / "results/pr37/bucket-matrix/report.json"),
            "--require-replay",
        ],
    ),
    (
        38,
        [
            "verify_auto_worker.py",
            str(ROOT / "results/pr38/auto-matrix/report.json"),
            "--parity",
            str(ROOT / "results/pr38/auto-parity.json"),
        ],
    ),
    (
        38,
        [
            "verify_graph_policy.py",
            str(ROOT / "results/pr38/shared.json"),
            "--kind",
            "shared",
        ],
    ),
]
for n, command in commands:
    result = subprocess.run(
        [PY, "recipe/cua_s1/" + command[0], *command[1:]],
        cwd=ROOT / f"pr{n}",
        text=True,
        capture_output=True,
    )
    (ROOT / "results" / f"pr{n}" / (command[0] + ".log")).write_text(
        result.stdout + result.stderr
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
summary["performance"] = {}
for n, path in [(37, "bucket-matrix/report.json"), (38, "auto-matrix/report.json")]:
    d = read(n, path)
    summary["performance"][str(n)] = {}
    for workload, runs in d["workloads"].items():
        rows = []
        for run in runs:
            metrics = run["summary"]
            totals = {k: v["total_ms"] / 1000 for k, v in metrics.items()}
            row = {
                "run": run["run"],
                "total_seconds": totals,
                "p95_ms": {k: v["p95_ms"] for k, v in metrics.items()},
            }
            if n == 38:
                fixed = min(v for k, v in totals.items() if k != "auto")
                row["auto_extra_vs_best_fixed_pct"] = 100 * (totals["auto"] / fixed - 1)
                row["auto_reduction_vs_eager_pct"] = 100 * (
                    1 - totals["auto"] / totals["eager"]
                )
            rows.append(row)
        summary["performance"][str(n)][workload] = rows
(ROOT / "results/verified-summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(
    "All current-runtime provenance, parity, memory ownership and benchmark checks verified."
)
