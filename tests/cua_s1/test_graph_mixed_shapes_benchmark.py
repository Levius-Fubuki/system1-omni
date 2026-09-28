"""Check the mixed-shape experiment's workload and amortized cost accounting."""

import importlib.util
from pathlib import Path

import pytest


def load_benchmark():
    path = (
        Path(__file__).resolve().parents[2]
        / "recipe/cua_s1/benchmark_graph_mixed_shapes.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_graph_mixed_shapes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workloads_have_hot_reuse_and_cache_churn():
    schedules = load_benchmark().schedules()
    assert schedules["hot_four"] == [1, 2, 4, 8] * 6
    assert schedules["churn_twelve"] == list(range(1, 13)) * 3
    assert len(set(schedules["hot_four"])) <= 8
    assert len(set(schedules["churn_twelve"])) > 8


def test_summary_includes_capture_cost_and_reports_first_break_even():
    summarize = load_benchmark().summarize_events
    events = [
        {"eager_ms": 10.0, "graph_ms": 30.0},
        {"eager_ms": 10.0, "graph_ms": 5.0},
        {"eager_ms": 10.0, "graph_ms": 5.0},
        {"eager_ms": 10.0, "graph_ms": 5.0},
        {"eager_ms": 10.0, "graph_ms": 5.0},
    ]
    result = summarize(events)
    assert result["eager_total_ms"] == 50.0
    assert result["graph_total_ms"] == 50.0
    assert result["first_break_even_request"] == 5
    assert summarize(events[:4])["first_break_even_request"] is None


def test_capture_threshold_must_be_positive(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        load_benchmark().parse_args(
            [
                "--weights",
                str(tmp_path / "weights"),
                "--output",
                str(tmp_path / "results"),
                "--graph-min-uses",
                "0",
            ]
        )
