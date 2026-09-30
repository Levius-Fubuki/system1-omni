"""Load Qwen3.5-4B with the Cua-S1 `text` adapter and score one prompt.

The calls mirror upstream `cua_s1.four_b.FourBModel` (text modality): the same
model class, an unmerged PEFT adapter, the chat template with its default
generation prompt, full logits, and a fp32 softmax over the letter logits at
the last position. Keeping them the same is what makes the worker's
probabilities bitwise identical to the reference in the same environment.

Torch, Transformers and PEFT are imported only when a model is loaded, so the
adapter checks can run without them.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .contract import BASE_REPO, LETTERS, Question, build_messages


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
    if config.get("base_model_name_or_path") != BASE_REPO:
        raise RuntimeError(f"{config_file}: base model is not {BASE_REPO}")
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


@dataclass
class Scored:
    probabilities: list[float]
    prompt_tokens: int


class TextModel:
    def __init__(
        self, base_model: str, adapter_root: str, device: str, dtype: str
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.dtype = dtype
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=getattr(torch, dtype), device_map=device
        )
        model = PeftModel.from_pretrained(model, str(text_adapter_dir(adapter_root)))
        model.eval()
        self.model = model
        self.load_seconds = time.perf_counter() - started
        self.letter_ids = self._letter_ids()

    def _letter_ids(self) -> list[int]:
        ids = []
        for letter in LETTERS:
            tokens = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(tokens) != 1:
                raise RuntimeError(f"letter {letter!r} is not a single token: {tokens}")
            ids.append(tokens[0])
        return ids

    def encode(self, state: str, question: Question):
        """Tokenized prompt for one question, on CPU.

        The Qwen3.5 tokenizer adds no special tokens here (contract point 4);
        the chat template already contains them.
        """
        chat_text = self.tokenizer.apply_chat_template(
            build_messages(state, question), tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(chat_text, return_tensors="pt")

    def score_encoded(self, inputs, n_options: int) -> Scored:
        import torch

        with torch.no_grad():
            inputs = inputs.to(self.model.device)
            out = self.model(**inputs)
            final_logits = out.logits[0, -1, :]
            letter_ids = self.letter_ids[:n_options]
            option_logits = final_logits[
                torch.tensor(letter_ids, device=final_logits.device)
            ]
            probabilities = torch.softmax(option_logits.float(), dim=-1).tolist()
        return Scored(
            probabilities=probabilities, prompt_tokens=int(inputs["input_ids"].shape[1])
        )
