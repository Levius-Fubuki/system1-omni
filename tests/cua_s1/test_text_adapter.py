"""Adapter directory checks: no weights, no torch."""

import json

import pytest

from models.cua_s1.text.adapter import downloaded_revision, text_adapter_dir

REV = "16818868b0cc7813808aae4e87b417657046ab79"


def write_adapter(path, targets):
    path.mkdir(parents=True)
    config = {"base_model_name_or_path": "Qwen/Qwen3.5-4B", "target_modules": targets}
    (path / "adapter_config.json").write_text(json.dumps(config))


def test_text_adapter_dir(tmp_path):
    write_adapter(tmp_path / "text", ["q_proj", "down_proj"])
    write_adapter(tmp_path / "multimodal", ["q_proj", "linear_fc1"])
    assert text_adapter_dir(tmp_path) == tmp_path / "text"
    assert text_adapter_dir(tmp_path / "text") == tmp_path / "text"
    with pytest.raises(RuntimeError, match="multimodal adapter"):
        text_adapter_dir(tmp_path / "multimodal")


def test_downloaded_revision_from_repository_root(tmp_path):
    write_adapter(tmp_path / "text", ["q_proj"])
    meta = (
        tmp_path / ".cache/huggingface/download/text/adapter_model.safetensors.metadata"
    )
    meta.parent.mkdir(parents=True)
    meta.write_text(f"{REV}\nabc\n1\n")
    assert downloaded_revision(tmp_path) == REV
    assert downloaded_revision(tmp_path / "text") == REV
    assert downloaded_revision(tmp_path / "missing") is None
