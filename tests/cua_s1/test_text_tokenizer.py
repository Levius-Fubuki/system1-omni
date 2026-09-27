"""Tokenizer checks against the pinned base model (tokenizer files only, no weights).

Set CUA_S1_BASE to a local Qwen/Qwen3.5-4B directory to run them.
"""

import json
import os
from pathlib import Path

import pytest

from models.cua_s1.text.contract import LETTERS, build_messages, map_request, parse_body

BASE = os.environ.get("CUA_S1_BASE")
pytestmark = pytest.mark.skipif(not BASE, reason="set CUA_S1_BASE to run")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE)


def test_letter_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert ids == [[32 + i] for i in range(26)]


def test_fixture_prompt(tokenizer):
    inputs = json.loads(
        (Path(__file__).parent / "data" / "text_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    request = map_request(parse_body(json.dumps(inputs["fixture_positive"]).encode()))
    text = tokenizer.apply_chat_template(
        build_messages(request.state, request.questions[0]),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    ids = tokenizer(text)["input_ids"]
    assert len(ids) == 218
    assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]
