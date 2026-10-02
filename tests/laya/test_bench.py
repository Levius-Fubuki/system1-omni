"""Checks on the benchmark inputs, the run header and the documented commands in recipe/laya. No model."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "recipe/laya/bench"
sys.path.insert(0, str(BENCH))

import env as bench_env  # noqa: E402


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_header_records_what_makes_two_runs_comparable(monkeypatch):
    record = bench_env.header("some/repo", extra=1)
    assert record["type"] == "env" and record["extra"] == 1 and record["checkpoint"] == "some/repo"
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert record["omni_sha"] == head and isinstance(record["omni_dirty"], bool)
    import laya  # noqa: F401
    from importlib.metadata import version

    assert (record["laya"], record["torch"], record["transformers"]) == tuple(
        version(package) for package in ("laya", "torch", "transformers")
    )
    assert record["argv"] == sys.argv and record["python"] == ".".join(map(str, sys.version_info[:3]))
    assert record["power"] and record["loadavg_1m"] >= 0 and record["chip"] and record["mem_gb"] > 0
    assert record["utc"].endswith("+00:00")


def test_the_fixed_inputs_span_question_types_lengths_and_option_counts():
    workloads = rows(BENCH / "workloads.jsonl")
    assert len({w["id"] for w in workloads}) == len(workloads)
    questions = [q for w in workloads for q in w["questions"].values()]
    assert {q["type"] for q in questions} == {"choice", "score", "noul"}
    assert {len(q["criteria"]) for q in questions if q["type"] == "choice"} >= {2, 5, 10}
    bench = [w for w in workloads if w["kind"] == "bench"]
    words = sorted(len(w["state"].split()) for w in bench)
    assert (
        words[0] < 20 and 100 < max(w for w in words if w < 200) and words[-1] > 300
    )  # short, medium, near the window
    assert {len(w["questions"]) for w in bench} >= {1, 3, 6}  # below and above laya's autocast threshold of 5 rows
    parity = [w for w in workloads if w["kind"] == "parity"]
    assert {q["type"] for w in parity for q in w["questions"].values()} == {"choice", "score", "noul"}


DOCUMENTS = ["recipe/laya/apple-silicon.md", "recipe/laya/bench/README.md", "src/models/laya/README.md"]
SCRIPTS = {
    "frontend.laya_mps": "src/frontend/laya_mps.py",
    **{f"recipe/laya/bench/{name}.py": f"recipe/laya/bench/{name}.py"
       for name in ("bench_http", "bench_inproc", "paired", "profile_mps", "report")},
}  # fmt: skip


def documented_commands():
    import re

    for document in DOCUMENTS:
        text = (REPO / document).read_text()
        for block in re.findall(r"```sh\n(.*?)```", text, flags=re.S):
            for command in block.replace("\\\n", " ").splitlines():
                for name, source in SCRIPTS.items():
                    if name in command:
                        yield document, command, name, source


def test_documented_commands_use_flags_and_files_that_exist():
    import re

    commands = list(documented_commands())
    assert len(commands) >= 15
    for document, command, name, source in commands:
        assert (REPO / source).exists(), f"{document}: {source}"
        options = set(re.findall(r'add_argument\(\s*"(--?[a-z][a-z-]*)"', (REPO / source).read_text()))
        ours = command.split("--spawn")[0] if name.startswith("recipe") else command.split(name, 1)[1]
        if name == "recipe/laya/bench/paired.py":
            ours = re.sub(r'"[^"]*"', "", ours)  # --a/--b carry flags of the worker, checked below
            for worker_flags in re.findall(r'--[ab] "([^"]*)"', command):
                assert set(re.findall(r"--[a-z-]+", worker_flags)) <= set(
                    re.findall(r'add_argument\(\s*"(--[a-z-]+)"', (REPO / SCRIPTS["frontend.laya_mps"]).read_text())
                ), f"{document}: {command}"
        used = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", ours))
        assert used <= options, f"{document}: {command}: unknown {sorted(used - options)}"
