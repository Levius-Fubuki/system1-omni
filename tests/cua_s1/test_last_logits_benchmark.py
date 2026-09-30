"""The paired experiment must compare only output projection scope."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_projection_variants_change_only_forward_keyword(monkeypatch):
    recipe = Path(__file__).resolve().parents[2] / "recipe/cua_s1"
    monkeypatch.syspath_prepend(str(recipe))
    spec = importlib.util.spec_from_file_location(
        "benchmark_last_logits", recipe / "benchmark_last_logits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        return kwargs

    model = SimpleNamespace(forward=forward)
    engine = SimpleNamespace(model=model)
    engine.predict = lambda request: engine.model.forward(
        inputs_embeds="same", logits_to_keep=1
    )
    with module.projection_variants(engine) as variants:
        assert variants.predict_reference(None) == {"inputs_embeds": "same"}
        assert variants.predict(None) == {
            "inputs_embeds": "same",
            "logits_to_keep": 1,
        }
    assert model.forward is forward
    assert calls == [
        {"inputs_embeds": "same"},
        {"inputs_embeds": "same", "logits_to_keep": 1},
    ]
