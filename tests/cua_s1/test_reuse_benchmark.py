"""Contracts for paired measurements, fixture diversity, and saved failures."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "recipe/cua_s1"


@pytest.fixture
def reuse_benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))
    path = RECIPE / "benchmark_image_reuse.py"
    assert path.exists(), "reuse benchmark is not implemented"
    spec = importlib.util.spec_from_file_location("reuse_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distinct_fixture_changes_goals_and_candidate_layout(reuse_benchmark, tmp_path):
    from models.cua_s1.multimodal.protocol import parse_request

    value = reuse_benchmark.distinct_fixture(tmp_path)
    request = parse_request(value)
    assert len(request.questions) == 8
    assert len({q.goal for q in request.questions}) == 8
    assert len({len(q.keys) for q in request.questions}) >= 3
    assert any(len(q.keys) == 26 for q in request.questions)


def test_paired_measurements_alternate_and_keep_variant_peaks(
    reuse_benchmark, monkeypatch
):
    events = []
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        reset_peak_memory_stats=lambda: events.append("reset"),
        max_memory_allocated=lambda: 123,
        max_memory_reserved=lambda: 456,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    engine = SimpleNamespace(
        predict_reference=lambda request: events.append("baseline"),
        predict=lambda request: events.append("reuse"),
    )
    result = reuse_benchmark.paired_benchmark(
        engine, None, warmup=1, runs=2, iterations=2
    )
    calls = [event for event in events if event in {"baseline", "reuse"}]
    assert calls == [
        "baseline",
        "reuse",
        "baseline",
        "reuse",
        "reuse",
        "baseline",
        "reuse",
        "baseline",
        "baseline",
        "reuse",
    ]
    assert events.count("reset") == 8
    assert len(result) == 2
    for run in result:
        for variant in ("baseline", "reuse"):
            assert len(run[variant]["latencies_ms"]) == 2
            assert run[variant]["peak_allocated_bytes"] == 123
            assert run[variant]["peak_reserved_bytes"] == 456


@pytest.mark.parametrize("option", ["--warmup", "--runs", "--iterations"])
def test_nonpositive_benchmark_counts_rejected(reuse_benchmark, option):
    with pytest.raises(SystemExit):
        reuse_benchmark.parse_args(
            ["--weights", "/tmp/w", "--output", "/tmp/o", option, "0"]
        )


def test_input_mismatch_does_not_pass_validation(reuse_benchmark):
    with pytest.raises(ValueError, match="mismatch"):
        reuse_benchmark.require_equal({"x": 1}, {"x": 2}, "inputs")


def test_failure_report_is_preserved(reuse_benchmark, monkeypatch, tmp_path):
    import json

    import evaluate_multimodal

    from models.cua_s1.multimodal import model

    monkeypatch.setattr(evaluate_multimodal, "environment", lambda: {"gpu": "fake"})
    monkeypatch.setattr(evaluate_multimodal, "measure", lambda call: (call(), 0.0))
    monkeypatch.setattr(
        model, "MultimodalEngine", lambda *args: SimpleNamespace(adapter_modules=178)
    )
    monkeypatch.setattr(
        reuse_benchmark, "correctness_cases", lambda folder: [("bad", {})]
    )
    with pytest.raises(ValueError):
        reuse_benchmark.main(
            [
                "--weights",
                str(tmp_path),
                "--output",
                str(tmp_path),
                "--correctness-only",
            ]
        )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]["type"] == "InvalidRequest"
    with pytest.raises(ValueError, match="already exists"):
        reuse_benchmark.main(["--weights", str(tmp_path), "--output", str(tmp_path)])
