import pytest

from models.cua_s1.multimodal.model import letter_ids, validate_adapter_config


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(text) - 33]


def test_letter_readout_is_in_candidate_order():
    assert letter_ids(Tokenizer(), 3) == [32, 33, 34]


def test_multitoken_letters_are_rejected():
    class Bad:
        def encode(self, *args, **kwargs):
            return [1, 2]

    with pytest.raises(ValueError, match="single token"):
        letter_ids(Bad(), 2)


def test_text_adapter_is_rejected_before_model_load():
    with pytest.raises(ValueError, match="multimodal"):
        validate_adapter_config(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj"],
            }
        )


def test_multimodal_adapter_contract():
    validate_adapter_config(
        {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "linear_fc1",
                "linear_fc2",
            ],
        }
    )


def test_unlisted_model_files_cannot_override_verified_shards(tmp_path, monkeypatch):
    import hashlib
    import json

    from models.cua_s1.multimodal import model

    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_bytes(b"{}")
    manifest = tmp_path / "weights.lock.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "role": "base",
                        "files": {
                            "config.json": {
                                "size": 2,
                                "sha256": hashlib.sha256(b"{}").hexdigest(),
                            }
                        },
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        model,
        "WEIGHTS_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    model.verify_weights(base, adapter)
    (base / "model.safetensors").write_bytes(b"override")
    with pytest.raises(ValueError, match="unlisted"):
        model.verify_weights(base, adapter)


def test_all_question_lengths_are_checked_before_inference():
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import InvalidRequest, Question, Request

    engine = object.__new__(MultimodalEngine)
    first = Question("first", ("a",), ("A",), "")
    second = Question("second", ("b",), ("B",), "")
    forwarded = []

    def prepare(image, question):
        if question.name == "second":
            raise InvalidRequest("processed prompt exceeds 4096 tokens")
        return {}

    engine.prepare = prepare
    engine.score = lambda inputs, q: forwarded.append(q)
    with pytest.raises(InvalidRequest, match="4096"):
        engine.predict_reference(Request(None, (first, second)))
    assert forwarded == []


def test_weights_manifest_must_match_pinned_upstream_digest(tmp_path):
    from models.cua_s1.multimodal.model import verify_weights

    (tmp_path / "weights.lock.json").write_text('{"artifacts": []}')
    with pytest.raises(ValueError, match="manifest checksum"):
        verify_weights(tmp_path / "base", tmp_path / "adapter")
