"""Load Qwen3.5-4B with the Cua-S1 `text` adapter and score one prompt.

The calls mirror upstream `cua_s1.four_b.FourBModel` (text modality): the same
model class, an unmerged PEFT adapter, the chat template with its default
generation prompt, full logits, and a fp32 softmax over the letter logits at
the last position. Keeping them the same is what makes the worker's
probabilities bitwise identical to the reference in the same environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import text_adapter_dir
from .contract import LETTERS, Question, build_messages


@dataclass
class Scored:
    probabilities: list[float]
    prompt_tokens: int


class TextEngine:
    def __init__(
        self, base_model: str, adapter_root: str, device: str, dtype: str
    ) -> None:
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

    def prompt_ids(self, state: str, question: Question) -> list[int]:
        return self.encode(state, question)["input_ids"][0].tolist()

    @torch.no_grad()
    def score_encoded(self, inputs, n_options: int) -> Scored:
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

    def score(self, state: str, question: Question) -> Scored:
        return self.score_encoded(self.encode(state, question), len(question.keys))
