"""CPU checks for the offline reference export; tensor checks need optional Torch."""

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "export_multimodal_reference", ROOT / "recipe/cua_s1/export_multimodal_reference.py"
)


def recipe():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_cases_reproduce_images_and_requests(tmp_path):
    module = recipe()
    left = module.make_cases(tmp_path / "left")
    right = module.make_cases(tmp_path / "right")
    assert len(left) == len(right) == 7
    assert sum(len(case["request"]["questions"]) for case in left) == 8
    for a, b in zip(left, right):
        assert a["name"] == b["name"]
        assert a["request"] == b["request"]
        image = base64.b64decode(a["request"]["state"]["image"].split(",")[1])
        assert (tmp_path / "left" / a["image"]).read_bytes() == image
        assert (tmp_path / "right" / b["image"]).read_bytes() == image
    from models.cua_s1.multimodal.protocol import parse_request

    requests = [parse_request(case["request"]) for case in left]
    assert len({request.image.size for request in requests}) >= 4
    assert {len(q.keys) for request in requests for q in request.questions} >= {
        1,
        3,
        26,
    }


def test_existing_output_is_rejected_before_model_load(tmp_path):
    module = recipe()
    with pytest.raises(FileExistsError):
        module.export(tmp_path / "missing-weights", tmp_path)
    assert not list(tmp_path.iterdir())


def test_fingerprint_preserves_bfloat16_bytes():
    torch = pytest.importorskip("torch")
    module = recipe()
    tensor = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    info = module.tensor_info(tensor)
    assert info["shape"] == [1, 2]
    assert info["dtype"] == "bfloat16"
    assert info["sha256"] == module.sha256(tensor.view(torch.uint8).numpy().tobytes())
    assert module.tensor_info(tensor.t().contiguous().t()) == info


def toy_engine(torch, fail=False):
    class Visual(torch.nn.Module):
        def forward(self, pixels, **kwargs):
            return SimpleNamespace(pooler_output=pixels.to(torch.bfloat16))

    class Language(torch.nn.Module):
        def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
            if fail:
                raise RuntimeError("deliberate forward failure")
            return SimpleNamespace(last_hidden_state=inputs_embeds + 1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.visual = Visual()
            self.model.language_model = Language()
            self.model.rope_deltas = torch.tensor([[-1]])
            self.device = torch.device("cpu")
            self.config = SimpleNamespace(image_token_id=99)

        def get_base_model(self):
            return self

        def forward(self, input_ids, pixel_values, **kwargs):
            features = self.model.visual(pixel_values).pooler_output
            embeds = torch.zeros(1, input_ids.shape[1], 2, dtype=torch.bfloat16)
            embeds[0, input_ids[0] == 99] = features
            positions = torch.arange(input_ids.shape[1]).expand(3, 1, -1)
            result = self.model.language_model(
                inputs_embeds=embeds, position_ids=positions
            )
            logits = result.last_hidden_state[..., :1].expand(-1, -1, 100).clone()
            return SimpleNamespace(logits=logits)

    model = Model()
    tokenizer = SimpleNamespace(encode=lambda letter, **kwargs: [ord(letter) - 33])
    return SimpleNamespace(model=model, tokenizer=tokenizer)


def test_capture_observes_forward_and_removes_hooks():
    torch = pytest.importorskip("torch")
    module = recipe()
    from models.cua_s1.multimodal.protocol import Question

    engine = toy_engine(torch)
    inputs = {
        "input_ids": torch.tensor([[1, 99, 99, 2]]),
        "pixel_values": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
    }
    question = Question("test", ("a", "b"), ("First", "Second"), "Choose")
    tensors = module.capture(engine, inputs, question)
    assert torch.equal(tensors["inputs_embeds"][0, 1:3], tensors["image_features"])
    assert tensors["position_ids"].shape == (3, 1, 4)
    assert tensors["last_hidden_state"].shape == (1, 2)
    assert tensors["candidate_probabilities"].tolist() == [0.5, 0.5]
    assert all(t.device.type == "cpu" for t in tensors.values())
    assert not engine.model.model.visual._forward_hooks
    assert not engine.model.model.language_model._forward_pre_hooks
    assert not engine.model.model.language_model._forward_hooks


def test_forward_failure_also_removes_hooks():
    torch = pytest.importorskip("torch")
    module = recipe()
    from models.cua_s1.multimodal.protocol import Question

    engine = toy_engine(torch, fail=True)
    with pytest.raises(RuntimeError, match="deliberate forward failure"):
        module.capture(
            engine,
            {"input_ids": torch.tensor([[99]]), "pixel_values": torch.ones(1, 2)},
            Question("test", ("a",), ("First",), "Choose"),
        )
    assert not engine.model.model.visual._forward_hooks
    assert not engine.model.model.language_model._forward_pre_hooks
    assert not engine.model.model.language_model._forward_hooks


def test_bundle_roundtrip_and_independent_storage(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    module = recipe()
    original = torch.tensor([[1, 2]], dtype=torch.int64)
    entries = module.save_tensors(
        tmp_path / "q.safetensors", {"a": original, "b": original}
    )
    loaded = load_file(str(tmp_path / "q.safetensors"))
    assert torch.equal(loaded["a"], original)
    assert entries["a"] == module.tensor_info(loaded["a"])
    assert entries["b"] == entries["a"]
    json.dumps(entries, allow_nan=False)


def test_verifier_checks_files_and_rejects_corruption(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "recipe/cua_s1"))
    import verify_multimodal_reference as verifier

    module = recipe()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    tensor_path = bundle / "q.safetensors"
    metadata = module.save_tensors(tensor_path, {"example": torch.ones(2)})
    manifest = {
        "schema": module.SCHEMA,
        "files": {
            "q.safetensors": {
                "size": tensor_path.stat().st_size,
                "sha256": module.sha256(tensor_path.read_bytes()),
            }
        },
        "questions": [{"tensors_file": "q.safetensors", "tensors": metadata}],
    }
    module.write_json(bundle / "manifest.json", manifest)
    # File integrity is independently testable before semantic validation.
    verifier.check_files(bundle, manifest)
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="file checksum"):
        verifier.check_files(bundle, manifest)


def test_verifier_does_not_accept_missing_tensors(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "recipe/cua_s1"))
    import verify_multimodal_reference as verifier

    with pytest.raises(ValueError, match="missing tensors"):
        verifier.check_tensors({}, {}, {})


def test_readout_allows_cpu_roundoff_but_rejects_drift(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "recipe/cua_s1"))
    import verify_multimodal_reference as verifier

    module = recipe()
    tensors = {
        "input_ids": torch.tensor([[1, 99, 99, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.int64),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        "pixel_values": torch.ones(2, 3),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
        "image_features": torch.ones(2, 2, dtype=torch.bfloat16),
        "image_token_indices": torch.tensor([1, 2]),
        "inputs_embeds": torch.ones(1, 4, 2, dtype=torch.bfloat16),
        "position_ids": torch.arange(4).expand(3, 1, -1),
        "rope_deltas": torch.tensor([[0]]),
        "last_hidden_state": torch.ones(1, 2, dtype=torch.bfloat16),
        "candidate_token_ids": torch.tensor([32, 33]),
        "candidate_logits": torch.tensor([0, 1], dtype=torch.bfloat16),
        "candidate_probabilities": torch.softmax(torch.tensor([0.0, 1.0]), dim=-1),
    }
    tensors["candidate_probabilities"][0] += 3e-8
    configs = {
        "base": {
            "text_config": {"hidden_size": 2},
            "vision_config": {
                "spatial_merge_size": 1,
                "patch_size": 1,
                "temporal_patch_size": 1,
            },
            "image_token_id": 99,
        }
    }
    entry = {
        "option_keys": ["a", "b"],
        "tensors": {k: module.tensor_info(v) for k, v in tensors.items()},
    }
    verifier.check_tensors(tensors, entry, configs)
    tensors["candidate_probabilities"][0] += 1e-4
    entry["tensors"] = {k: module.tensor_info(v) for k, v in tensors.items()}
    with pytest.raises(ValueError, match="readout mismatch"):
        verifier.check_tensors(tensors, entry, configs)


@pytest.mark.parametrize("unsafe", [False, True])
def test_verifier_rejects_unlisted_files_and_unsafe_paths(
    tmp_path, monkeypatch, unsafe
):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "recipe/cua_s1"))
    import verify_multimodal_reference as verifier

    module = recipe()
    folder = tmp_path / "bundle"
    folder.mkdir()
    extra = (tmp_path if unsafe else folder) / "extra.json"
    extra.write_text("{}")
    manifest = {"files": {}}
    if unsafe:
        manifest["files"]["../extra.json"] = {"size": 2, "sha256": module.sha256(b"{}")}
    with pytest.raises(
        ValueError, match="unsafe file path" if unsafe else "file inventory"
    ):
        verifier.check_files(folder, manifest)
