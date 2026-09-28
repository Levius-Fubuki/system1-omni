"""Request ownership tests, plus tensor integration tests when torch is installed."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from models.cua_s1.multimodal.model import MAX_TOKENS, MultimodalEngine
from models.cua_s1.multimodal.protocol import InvalidRequest, Question, Request

QUESTIONS = (
    Question("first", ("yes", "no"), ("Yes", "No"), "Confirm"),
    Question("second", ("back", "next", "stop"), ("Back", "Next", "Stop"), "Go back"),
)


class ImageProcessor:
    merge_size = 2

    def __init__(self):
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append(images[0])
        return {"pixel_values": images[0], "image_grid_thw": (1, 4, 4)}


class Processor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.calls = []
        self.fail = False

    def apply_chat_template(self, messages, **kwargs):
        return repr(messages)

    def __call__(self, text, images, **kwargs):
        assert self.image_processor.merge_size == 2
        image = self.image_processor(images, return_tensors="pt")
        # Processor implementations may consume/mutate their returned mapping.
        pixels = image.pop("pixel_values")
        self.calls.append(text[0])
        if self.fail:
            raise ValueError("tokenizer failure")
        return {
            "pixel_values": pixels,
            "image_grid_thw": image["image_grid_thw"],
            "input_ids": SimpleNamespace(shape=(1, len(text[0]))),
            "text": text[0],
        }


def engine_with_processor():
    engine = object.__new__(MultimodalEngine)
    engine.processor = Processor()
    return engine


def test_preprocessing_reuses_only_image_and_keeps_question_prompts():
    engine = engine_with_processor()
    original = engine.processor.image_processor
    image = object()
    prepared = engine.prepare_reused(image, QUESTIONS)
    assert original.calls == [image]
    assert prepared[0]["pixel_values"] is prepared[1]["pixel_values"] is image
    assert prepared[0]["text"] != prepared[1]["text"]
    assert engine.processor.image_processor is original
    for question, inputs in zip(QUESTIONS, prepared):
        reference = engine.prepare(image, question)
        assert inputs["text"] == reference["text"]
        assert inputs["input_ids"].shape == reference["input_ids"].shape
        assert inputs["image_grid_thw"] == reference["image_grid_thw"]


def test_image_cache_does_not_survive_requests_or_failed_preparation():
    engine = engine_with_processor()
    images = [object(), object(), object()]
    engine.prepare_reused(images[0], QUESTIONS)
    engine.processor.fail = True
    with pytest.raises(ValueError, match="tokenizer failure"):
        engine.prepare_reused(images[1], QUESTIONS)
    engine.processor.fail = False
    prepared = engine.prepare_reused(images[2], QUESTIONS)
    assert engine.processor.image_processor.calls == images
    assert all(x["pixel_values"] is images[2] for x in prepared)


@pytest.mark.parametrize("failure_stage", ["encode", "second_question"])
def test_failed_inference_releases_request_state_before_different_image(failure_stage):
    engine = engine_with_processor()
    processor = engine.processor
    image_processor = processor.image_processor
    failed_image, next_image = object(), object()
    encoded_images, feature_refs, scored = [], [], []

    class Features:
        def __init__(self, image):
            self.image = image

    def encode(inputs):
        image = inputs["pixel_values"]
        encoded_images.append(image)
        features = Features(image)
        feature_refs.append(weakref.ref(features))
        if image is failed_image and failure_stage == "encode":
            raise RuntimeError("encoder failure")
        return features

    def score(inputs, question, features):
        image = inputs["pixel_values"]
        assert features.image is image
        assert features is feature_refs[-1]()
        scored.append((image, question.name))
        if image is failed_image and question is QUESTIONS[1]:
            raise RuntimeError("later question failure")
        return [1 / len(question.keys)] * len(question.keys)

    engine.encode_image = encode
    engine.score_reused = score
    original_engine_state = dict(vars(engine))
    message = (
        "encoder failure" if failure_stage == "encode" else "later question failure"
    )
    with pytest.raises(RuntimeError, match=message):
        engine.predict(Request(failed_image, QUESTIONS))
    gc.collect()
    assert all(ref() is None for ref in feature_refs)
    assert vars(engine) == original_engine_state
    assert engine.processor is processor
    assert processor.image_processor is image_processor
    assert image_processor.calls == [failed_image]
    assert scored == (
        [] if failure_stage == "encode" else [(failed_image, q.name) for q in QUESTIONS]
    )

    result = engine.predict(Request(next_image, QUESTIONS))
    gc.collect()
    assert encoded_images == [failed_image, next_image]
    assert image_processor.calls == [failed_image, next_image]
    assert scored[-2:] == [(next_image, q.name) for q in QUESTIONS]
    assert list(result["answers"]) == [q.name for q in QUESTIONS]
    assert len(feature_refs) == 2
    assert all(ref() is None for ref in feature_refs)
    assert vars(engine) == original_engine_state
    assert engine.processor is processor
    assert processor.image_processor is image_processor


def test_all_lengths_checked_before_vision_or_language_execution():
    engine = engine_with_processor()
    forwarded = []
    engine.encode_image = lambda inputs: forwarded.append("vision")
    engine.score_reused = lambda *args: forwarded.append("language")
    long = Question("long", ("a",), ("A",), "x" * MAX_TOKENS)
    with pytest.raises(InvalidRequest, match="4096"):
        engine.predict(Request(object(), (QUESTIONS[0], long)))
    assert forwarded == []


def test_multi_question_prediction_encodes_once_and_keeps_usage():
    engine = engine_with_processor()
    encoded, scored = [], []
    features = object()

    def encode(inputs):
        encoded.append(inputs)
        return features

    def score(inputs, question, shared):
        assert shared is features
        scored.append(question)
        return [1 / len(question.keys)] * len(question.keys)

    engine.encode_image = encode
    engine.score_reused = score
    request = Request(object(), QUESTIONS)
    result = engine.predict(request)
    assert len(encoded) == 1
    assert scored == list(QUESTIONS)
    assert list(result["answers"]) == [q.name for q in QUESTIONS]
    assert result["usage"]["input_tokens"] == sum(
        x["input_ids"].shape[-1]
        for x in engine.prepare_reused(request.image, QUESTIONS)
    )


def test_single_question_uses_reference_path():
    engine = object.__new__(MultimodalEngine)
    request = Request(object(), QUESTIONS[:1])
    calls = []
    engine.predict_reference = lambda value: calls.append(value) or {"reference": True}
    assert engine.predict(request) == {"reference": True}
    assert calls == [request]


def test_reused_tensor_path_preserves_scatter_positions_and_peft_forward():
    torch = pytest.importorskip("torch")
    engine = object.__new__(MultimodalEngine)
    calls = []
    features = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    positions = torch.tensor([[[0, 1, 1, 3]], [[0, 1, 1, 3]], [[0, 1, 2, 3]]])

    class Core:
        def get_image_features(self, pixels, grid, return_dict):
            assert not torch.is_grad_enabled()
            calls.append("vision-with-adapters")
            return SimpleNamespace(pooler_output=(features,))

        def get_input_embeddings(self):
            return lambda ids: ids.unsqueeze(-1).expand(-1, -1, 2).float()

        def get_placeholder_mask(self, ids, inputs_embeds, image_features):
            assert torch.equal(image_features, features)
            return (ids == 9).unsqueeze(-1).expand_as(inputs_embeds), None

        def get_rope_index(self, **kwargs):
            assert torch.equal(
                kwargs["mm_token_type_ids"], torch.tensor([[0, 1, 1, 0]])
            )
            assert torch.equal(kwargs["input_ids"], torch.tensor([[1, 9, 9, 2]]))
            calls.append("positions")
            return positions, torch.tensor([[0]])

    class Peft:
        device = "cpu"

        def get_base_model(self):
            return SimpleNamespace(model=Core())

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert "input_ids" not in kwargs and "pixel_values" not in kwargs
            assert "image_grid_thw" not in kwargs
            assert torch.equal(kwargs["position_ids"], positions)
            assert torch.equal(
                kwargs["inputs_embeds"],
                torch.tensor([[[1.0, 1.0], [10.0, 20.0], [30.0, 40.0], [2.0, 2.0]]]),
            )
            calls.append("peft-forward")
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 2.0]]]))

    engine.model = Peft()
    engine.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(text) - 65])
    inputs = {
        "input_ids": torch.tensor([[1, 9, 9, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "pixel_values": torch.ones(2, 4),
        "image_grid_thw": torch.tensor([[1, 2, 4]]),
    }
    original = {key: value.clone() for key, value in inputs.items()}
    shared = engine.encode_image(inputs)
    for question in QUESTIONS:
        result = engine.score_reused(inputs, question, shared)
        assert (
            result
            == torch.softmax(torch.arange(len(question.keys)).float(), dim=-1).tolist()
        )
    assert calls == [
        "vision-with-adapters",
        "positions",
        "peft-forward",
        "positions",
        "peft-forward",
    ]
    assert all(torch.equal(value, original[key]) for key, value in inputs.items())
