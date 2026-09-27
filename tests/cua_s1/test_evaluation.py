"""Provenance checks must fail before a changed oracle can certify parity."""

import importlib.util
from pathlib import Path

import pytest


def test_modified_reference_is_rejected(tmp_path, monkeypatch):
    recipe = Path(__file__).resolve().parents[2] / "recipe/cua_s1"
    monkeypatch.syspath_prepend(str(recipe))
    spec = importlib.util.spec_from_file_location(
        "evaluation", recipe / "evaluate_multimodal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / module.REFERENCE_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("modified oracle")
    with pytest.raises(ValueError, match="reference source differs"):
        module.verify_reference(tmp_path)


def test_environment_survives_json_roundtrip(monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    recipe = Path(__file__).resolve().parents[2] / "recipe/cua_s1"
    monkeypatch.syspath_prepend(str(recipe))
    spec = importlib.util.spec_from_file_location(
        "evaluation", recipe / "evaluate_multimodal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_name=lambda: "GPU", get_device_capability=lambda: (8, 9)
        ),
        version=SimpleNamespace(cuda="13.0"),
        get_num_threads=lambda: 16,
        get_num_interop_threads=lambda: 16,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "pinned")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "driver\n")
    environment = module.environment()
    assert json.loads(json.dumps(environment)) == environment
