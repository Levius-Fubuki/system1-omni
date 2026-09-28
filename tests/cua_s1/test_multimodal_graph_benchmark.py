"""Guard the graph experiment's response comparison against changed choices."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def graph_benchmark(monkeypatch):
    recipe = Path(__file__).resolve().parents[2] / "recipe/cua_s1"
    monkeypatch.syspath_prepend(str(recipe))
    spec = importlib.util.spec_from_file_location(
        "benchmark_multimodal_graph", recipe / "benchmark_multimodal_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response():
    return {
        "model": "pinned",
        "usage": {"input_tokens": 3, "output_tokens": 0},
        "answers": {
            "next": {
                "type": "choice",
                "choice": "save",
                "probabilities": {"save": 0.8, "cancel": 0.2},
            }
        },
    }


def test_graph_comparison_tracks_probability_difference(graph_benchmark):
    actual = deepcopy(response())
    actual["answers"]["next"]["probabilities"]["save"] = 0.799
    assert graph_benchmark.response_difference(response(), actual) == pytest.approx(
        0.001
    )


def test_graph_comparison_rejects_changed_choice(graph_benchmark):
    actual = deepcopy(response())
    actual["answers"]["next"]["choice"] = "cancel"
    with pytest.raises(ValueError, match="selected choice changed"):
        graph_benchmark.response_difference(response(), actual)


def test_graph_signature_includes_layout_and_dtype(graph_benchmark):
    value = SimpleNamespace(
        shape=(1, 4, 8),
        stride=lambda: (32, 8, 1),
        dtype="bfloat16",
        device="cuda:0",
    )
    assert graph_benchmark.tensor_signature(value) == (
        (1, 4, 8),
        (32, 8, 1),
        "bfloat16",
        "cuda:0",
    )
