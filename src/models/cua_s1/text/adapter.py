"""Locate and check the local Cua-S1 `text` adapter. No torch imports."""

from __future__ import annotations

import json
import re
from pathlib import Path


def text_adapter_dir(adapter_root: str | Path) -> Path:
    """Return the `text` adapter directory under the adapter root.

    Accepts the repository root (`<root>/text`) or the `text/` directory
    itself, and refuses the `multimodal/` adapter: PEFT only warns about keys
    it cannot place, so loading the wrong adapter would otherwise go unnoticed.
    """
    root = Path(adapter_root)
    path = root / "text" if (root / "text" / "adapter_config.json").exists() else root
    config_file = path / "adapter_config.json"
    if not config_file.exists():
        raise RuntimeError(f"no adapter_config.json under {root}")
    config = json.loads(config_file.read_text())
    if config.get("base_model_name_or_path") != "Qwen/Qwen3.5-4B":
        raise RuntimeError(f"{config_file}: base model is not Qwen/Qwen3.5-4B")
    if {"linear_fc1", "linear_fc2"} & set(config.get("target_modules") or []):
        raise RuntimeError(
            f"{config_file}: this is the multimodal adapter, not the text adapter"
        )
    return path


def downloaded_revision(adapter_root: str | Path) -> str | None:
    """The commit that `hf download --local-dir` recorded for the text adapter, if any.

    `hf download` keeps its metadata under the repository root, so this also
    looks one level up when `adapter_root` is the `text/` directory itself.
    """
    root = Path(adapter_root)
    places = [(root, "text/"), (root, "")]
    if root.name == "text":
        places.insert(0, (root.parent, "text/"))
    for base, prefix in places:
        cache = base / ".cache" / "huggingface" / "download"
        try:
            meta = (cache / f"{prefix}adapter_model.safetensors.metadata").read_text()
            first = meta.splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if re.fullmatch(r"[0-9a-f]{40}", first):
            return first
    return None
