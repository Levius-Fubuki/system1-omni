"""CPU export and fidelity gates; never download weights or require a GPU."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    path = ROOT / "recipe" / "jemm" / (name + ".py")
    assert path.exists(), f"missing recipe feature: {name}"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.path.remove(str(path.parent))
    return loaded


def test_safe_merge_matches_peft_bf16_rounding_and_preserves_input():
    export = module("export")
    base = torch.tensor([[0.1, -0.2, 3.0], [0.5, 0.25, -2.0]], dtype=torch.bfloat16)
    a = torch.tensor([[0.07, 0.3, -0.1]], dtype=torch.float32)
    b = torch.tensor([[0.2], [-0.9]], dtype=torch.float32)
    before = base.clone()
    actual = export.safe_merge(base, a, b, 2.0, chunk_rows=1)
    # PEFT computes the F32 adapter delta, then adds it to BF16 base with +=.
    expected = base.clone()
    expected += (b @ a) * 2.0
    assert torch.equal(actual, expected)
    assert torch.equal(base, before)
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("bad", ["shape", "nan", "overflow"])
def test_safe_merge_rejects_invalid_adapter(bad):
    export = module("export")
    base = torch.ones((2, 3), dtype=torch.bfloat16)
    a = torch.ones((1, 3))
    b = torch.ones((2, 1))
    if bad == "shape":
        b = torch.ones((3, 1))
    elif bad == "nan":
        a[0, 0] = float("nan")
    else:
        b.fill_(3e38)
    with pytest.raises(ValueError):
        export.safe_merge(base, a, b, 2.0)


def test_corpus_is_deterministic_and_covers_candidate_limits():
    corpus = module("corpus")
    first, second = corpus.build_cases(), corpus.build_cases()
    assert first == second
    assert len({c["id"] for c in first}) == len(first)
    sizes = set()
    kinds = set()
    for case in first:
        for question in case["request"]["questions"].values():
            kinds.add(question["type"])
            if question["type"] != "noul":
                sizes.add(len(question["criteria"]))
    assert {2, 3, 10, 26, 32} <= sizes
    assert kinds == {"choice", "noul", "score"}
    from PIL import Image
    import base64
    import io
    sizes = {Image.open(io.BytesIO(base64.b64decode(c["request"]["images"][0]))).size
             for c in first if c["request"].get("images")}
    assert sizes == {(256, 256), (512, 256)}


def response(probs, choice="a", tokens=7):
    return {"model": "JEMM", "answers": {"q": {"type": "choice", "choice": choice,
            "probabilities": probs, "confidence": max(probs.values())}},
            "usage": {"input_tokens": tokens, "output_tokens": 0, "latency_ms": 1.0}}


def test_fidelity_gate_retains_low_margin_cases_but_rejects_large_drift():
    compare = module("compare")
    assert compare.compare_response(response({"a": .51, "b": .49}),
                                    response({"a": .49, "b": .51}, "b"))["passed"]
    assert not compare.compare_response(response({"a": .8, "b": .2}),
                                        response({"a": .77, "b": .23}))["passed"]
    assert not compare.compare_response(response({"a": .51, "b": .49}),
                                        response({"a": .51, "b": .49}, tokens=8))["passed"]


def test_fidelity_gate_checks_candidate_identity_and_score():
    compare = module("compare")
    ref = response({"a": .8, "b": .2})
    assert not compare.compare_response(ref, response({"a": .8, "c": .2}))["passed"]
    for value in (float("nan"), 1.1):
        assert not compare.compare_response(ref, response({"a": value, "b": .2}))["passed"]
    ref["answers"]["q"].update(type="score", expected_value=1.5)
    got = json.loads(json.dumps(ref))
    got["answers"]["q"]["expected_value"] = 1.61
    assert not compare.compare_response(ref, got)["passed"]


def test_fidelity_gate_rejects_malformed_and_unnormalized_answers():
    compare = module("compare")
    ref = response({"a": .51, "b": .49})
    for got in ({"answers": None}, {"answers": []}, response({"a": .52, "b": .5}), response({"a": True, "b": False})):
        assert not compare.compare_response(ref, got)["passed"]


def test_exact_contract_gate_rejects_changed_grid_and_token_order():
    compare = module("compare")
    ref = [{"id": "image", "question_id": "q", "candidate_ids": ["one", "two"],
            "prompt": "prompt", "input_ids": [1, 2, 3], "image_grid_thw": [[1, 8, 16]], "mm_token_type_ids": [[0, 1, 1]]}]
    got = json.loads(json.dumps(ref))
    ref[0].update(type="choice", label_token_ids=list(range(32, 58)) + list(range(15, 21)), position_ids=[[[0, 1, 2]]] * 3)
    got = json.loads(json.dumps(ref))
    cases = [{"id": "image", "request": {"images": ["image"], "questions": {"q": {"type": "choice", "criteria": {"one": "", "two": ""}}}}}]
    assert compare.compare_contract(ref, got, cases=cases)["passed"]
    got[0]["image_grid_thw"] = [[1, 16, 8]]
    assert not compare.compare_contract(ref, got, cases=cases)["passed"]
    got = json.loads(json.dumps(ref))
    got[0]["input_ids"] = [1, 3, 2]
    assert not compare.compare_contract(ref, got, cases=cases)["passed"]


def frozen_cases():
    return [json.loads(line) for line in (ROOT / "tests/jemm/fixtures/corpus.jsonl").read_text().splitlines()]


def measured_records():
    import hashlib
    records = []
    for repetition in range(2):
        for index, case in enumerate(frozen_cases()):
            answers = {}
            for qid, q in case["request"]["questions"].items():
                kind = q["type"]
                ids = ["yes", "no"] if kind == "noul" else ([str(i) for i in range(len(q["criteria"]))] if kind == "score" else list(q["criteria"]))
                probs = {key: 1 / len(ids) for key in ids}
                answers[qid] = {"type": kind, "probabilities": probs, "confidence": max(probs.values())}
                if kind == "choice":
                    answers[qid]["choice"] = ids[0]
                elif kind == "score":
                    answers[qid]["expected_value"] = (len(ids) - 1) / 2
                else:
                    answers[qid]["noul"] = .5
            raw = json.dumps({**case["request"], "model": "JEMM"}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            records.append({"id": case["id"], "repetition": repetition, "index": index, "phase": "measured", "status": 200,
                            "modality": case["modality"], "elapsed_ms": 1.0, "request_sha256": hashlib.sha256(raw).hexdigest(),
                            "response": {"model": "JEMM", "answers": answers, "usage": {"input_tokens": 7, "output_tokens": 0, "latency_ms": 1.0}}})
    return records


@pytest.mark.parametrize("mutation", ["empty", "truncated", "duplicate", "missing_rep", "http_failure", "request_hash", "missing_question"])
def test_run_gate_rejects_incomplete_or_invalid_equal_runs(mutation):
    compare = module("compare")
    records = measured_records()
    if mutation == "empty":
        records = []
    elif mutation == "truncated":
        records.pop()
    elif mutation == "duplicate":
        records[1] = records[0]
    elif mutation == "missing_rep":
        records[0].pop("repetition")
    elif mutation == "http_failure":
        records[0]["status"] = 500
    elif mutation == "request_hash":
        records[0]["request_sha256"] = "0" * 64
    else:
        records[0]["response"]["answers"] = {}
    assert not compare.compare_runs(records, records, frozen_cases())["passed"]


def test_run_gate_accepts_complete_frozen_two_repetition_run():
    compare = module("compare")
    records = measured_records()
    assert len(records) == 26
    assert compare.compare_runs(records, records, frozen_cases())["passed"]


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": None, "output_tokens": 0, "latency_ms": 1},
    {"input_tokens": True, "output_tokens": 0, "latency_ms": 1}, {"input_tokens": 7., "output_tokens": 0, "latency_ms": 1},
    {"input_tokens": -7, "output_tokens": 0, "latency_ms": 1}, {"input_tokens": 7, "output_tokens": 0, "latency_ms": 0},
    {"input_tokens": 7, "output_tokens": 0, "latency_ms": float("nan")}, {"input_tokens": 7, "output_tokens": 0, "latency_ms": 1, "extra": 0}])
def test_response_gate_rejects_invalid_usage_even_when_equal(usage):
    compare = module("compare")
    ref = response({"a": .6, "b": .4})
    ref["usage"] = usage
    assert not compare.compare_response(ref, ref)["passed"]


def test_response_gate_does_not_compare_latency_values():
    compare = module("compare")
    ref, got = response({"a": .6, "b": .4}), response({"a": .6, "b": .4})
    got["usage"]["latency_ms"] = 9000
    assert compare.compare_response(ref, got)["passed"]


@pytest.mark.parametrize("field", ["choice", "confidence", "expected_value", "noul"])
def test_response_gate_rejects_equal_missing_required_answer_fields(field):
    compare = module("compare")
    ref = response({"a": .6, "b": .4})
    if field == "expected_value":
        ref["answers"]["q"].update(type="score", expected_value=1.0)
    elif field == "noul":
        ref["answers"]["q"].update(type="noul", noul=.6)
    ref["answers"]["q"].pop(field)
    assert not compare.compare_response(ref, ref)["passed"]


def test_contract_gate_rejects_empty_equal_records():
    compare = module("compare")
    assert not compare.compare_contract([], [])["passed"]


def test_contract_gate_rejects_missing_fields_and_asymmetric_modality():
    compare = module("compare")
    ref = [{"id": "image", "question_id": "q", "type": "choice", "candidate_ids": ["one", "two"],
            "prompt": "prompt", "input_ids": [1, 2, 3], "label_token_ids": list(range(32, 58)) + list(range(15, 21)),
            "image_grid_thw": [[1, 8, 16]], "mm_token_type_ids": [[0, 1, 1]], "position_ids": [[[0, 1, 2]]] * 3}]
    cases = [{"id": "image", "request": {"images": ["image"], "questions": {"q": {"type": "choice", "criteria": {"one": "", "two": ""}}}}}]
    assert compare.compare_contract(ref, ref, cases=cases)["passed"]
    for key in ("prompt", "input_ids", "image_grid_thw", "mm_token_type_ids"):
        got = json.loads(json.dumps(ref))
        got[0].pop(key)
        assert not compare.compare_contract(ref, got, cases=cases)["passed"]
        assert not compare.compare_contract(got, got, cases=cases)["passed"]


def test_contract_gate_requires_all_frozen_questions_and_symmetric_positions():
    compare = module("compare")
    protocol = module("protocol")
    records = []
    for control in protocol.expected_contract(frozen_cases()):
        record = {key: control[key] for key in ("id", "question_id", "type", "candidate_ids")}
        record.update(prompt="prompt", input_ids=[1, 2, 3], label_token_ids=protocol.LABEL_TOKEN_IDS,
                      position_ids=[[[0, 1, 2]]] * 3)
        if control["image_count"]:
            record.update(image_grid_thw=[[1, 8, 16]], mm_token_type_ids=[[0, 1, 1]])
        records.append(record)
    assert len(records) == 16
    assert compare.compare_contract(records, records)["passed"]
    missing_positions = [{key: value for key, value in record.items() if key != "position_ids"} for record in records]
    assert not compare.compare_contract(missing_positions, missing_positions)["passed"]
    assert not compare.compare_contract(records[:-1], records[:-1])["passed"]
    got = json.loads(json.dumps(records))
    got[0].pop("position_ids")
    assert not compare.compare_contract(records, got)["passed"]
    got = json.loads(json.dumps(records))
    got[0]["position_ids"] = []
    assert not compare.compare_contract(got, got)["passed"]


def test_frozen_corpus_hash_rejects_mutation_before_run(tmp_path):
    protocol = module("protocol")
    original = ROOT / "tests/jemm/fixtures/corpus.jsonl"
    assert len(protocol.load_corpus(original)[0]) == 13
    altered = tmp_path / "altered.jsonl"
    altered.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="corpus SHA256"):
        protocol.load_corpus(altered)


def test_reference_evidence_observes_lora_merge_and_parameter_dtypes():
    reference = module("reference_server")
    class AdapterLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.bfloat16))
            self.lora_A = torch.nn.Parameter(torch.ones((1, 3), dtype=torch.float32))
            self.merged_adapters = []
    layer = AdapterLayer()
    observed = reference.observed_peft_state(layer)
    assert observed["parameter_count"] == 9
    assert observed["parameter_dtype_inventory"]["torch.bfloat16"]["elements"] == 6
    assert observed["lora_parameter_inventory"]["lora_A"]["dtype"] == "torch.float32"
    assert observed["reference_adapter_merged"] is False
    layer.merged_adapters = ["default"]
    assert reference.observed_peft_state(layer)["reference_adapter_merged"] is True


def test_reference_rope_capture_uses_model_method_on_cpu_without_forward():
    reference = module("reference_server")
    class Backbone(torch.nn.Module):
        def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None, mm_token_type_ids=None):
            assert input_ids.device.type == "cpu"
            assert torch.equal(attention_mask, torch.ones_like(input_ids))
            return torch.arange(input_ids.shape[1]).expand(3, 1, -1), torch.zeros((1, 1), dtype=torch.long)
        def forward(self, *args, **kwargs):
            raise AssertionError("rope capture must not perform a forward")
    model = Backbone()
    positions, method = reference.capture_position_ids(model, {"input_ids": torch.tensor([[1, 2, 3]])})
    assert positions == [[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]]
    assert method["qualname"].endswith("get_rope_index")


def test_compare_cli_rejects_empty_and_truncated_equal_files(tmp_path):
    import subprocess
    for records in ([], measured_records()[:-1]):
        path = tmp_path / "run.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        out = tmp_path / "report.json"
        process = subprocess.run([sys.executable, str(ROOT / "recipe/jemm/compare.py"), "--reference", str(path), "--actual", str(path), "--out", str(out)], capture_output=True, text=True)
        assert process.returncode == 1
        assert json.loads(out.read_text())["passed"] is False


def test_benchmark_cli_rejects_changed_corpus_before_http(tmp_path):
    import subprocess
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes((ROOT / "tests/jemm/fixtures/corpus.jsonl").read_bytes() + b"\n")
    out = tmp_path / "output"
    process = subprocess.run([sys.executable, str(ROOT / "recipe/jemm/benchmark.py"), "--url", "http://127.0.0.1:1", "--corpus", str(corpus), "--out", str(out)], capture_output=True, text=True)
    assert process.returncode != 0
    assert "corpus SHA256 mismatch" in process.stderr
    assert not out.exists()


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [39 - "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345".index(text)]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["preserve_thinking"] is False
        return "system:" + messages[0]["content"] + "\nuser:" + messages[1]["content"] + "\nassistant:<think>\n\n</think>\n\n"


def toy_checkpoint(tmp_path):
    export = module("export")
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    tensors = {
        "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(6).reshape(2, 3).to(torch.bfloat16),
        "model.language_model.embed_tokens.weight": torch.zeros((40, 3), dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(3, dtype=torch.bfloat16),
        "lm_head.weight": torch.arange(120).reshape(40, 3).to(torch.bfloat16),
        "model.visual.patch_embed.proj.weight": torch.ones((2, 3), dtype=torch.bfloat16),
    }
    shard_map = {}
    for i, names in enumerate((list(tensors)[:2], list(tensors)[2:])):
        name = f"model-{i:05d}.safetensors"
        save_file({key: tensors[key] for key in names}, base / name)
        shard_map.update({key: name for key in names})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shard_map}))
    (base / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 3, "vocab_size": 40, "tie_word_embeddings": False}, "tie_word_embeddings": False, "vision_config": {"hidden_size": 2}}))
    (base / "tokenizer.json").write_text("{}")
    lora = {
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight": torch.tensor([[.1, .2, .3]]),
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight": torch.tensor([[.4], [.5]]),
    }
    save_file(lora, adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 2, "base_model_name_or_path": export.BASE_MODEL_ID}))
    (adapter / "decision_config.json").write_text(json.dumps(export.CALIBRATION))
    inventory = {
        "base": {"files": {p.name: export.sha256(p) for p in base.iterdir()}, "tensors": {k: {"shape": list(v.shape), "dtype": "BF16", "file": shard_map[k]} for k, v in tensors.items()}},
        "adapter": {"files": {p.name: export.sha256(p) for p in adapter.iterdir()}, "tensors": {k: {"shape": list(v.shape), "dtype": "F32", "file": "adapter_model.safetensors"} for k, v in lora.items()}},
    }
    return base, adapter, tensors, inventory


def test_export_shards_selects_untied_head_and_records_hashes(tmp_path):
    export = module("export")
    base, adapter, tensors, inventory = toy_checkpoint(tmp_path)
    out = tmp_path / "out"
    result = export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    assert result["format"] == "jemm-native/1"
    assert result["preserve_thinking"] is False
    assert result["untied_lm_head"] is True
    assert result["label_token_ids"] == list(range(39, 7, -1))
    assert torch.equal(load_file(out / "jemm_lm_head.safetensors")["weight"], tensors["lm_head.weight"][result["label_token_ids"]])
    weights = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert "lm_head.weight" not in weights
    assert "layers.0.mlp.up_proj.weight" in weights
    assert "model.visual.patch_embed.proj.weight" in load_file(out / "vision.safetensors")
    merged = load_file(out / weights["layers.0.mlp.up_proj.weight"])["layers.0.mlp.up_proj.weight"]
    expected = tensors["model.language_model.layers.0.mlp.up_proj.weight"].clone()
    expected += torch.tensor([[.4], [.5]]) @ torch.tensor([[.1, .2, .3]]) * 2
    assert torch.equal(merged, expected)
    for name, digest in result["export_sha256"].items():
        assert export.sha256(out / name) == digest
    assert export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory) == result


@pytest.mark.parametrize("corruption", ["bytes", "shape", "index", "adapter_inventory"])
def test_export_rejects_corruption_before_writing(tmp_path, corruption):
    export = module("export")
    base, adapter, _, inventory = toy_checkpoint(tmp_path)
    if corruption == "bytes":
        (base / "config.json").write_text("{}")
    elif corruption == "shape":
        inventory["base"]["tensors"]["lm_head.weight"]["shape"] = [39, 3]
    elif corruption == "index":
        path = base / "model.safetensors.index.json"
        index = json.loads(path.read_text())
        index["weight_map"]["lm_head.weight"] = "model-00000.safetensors"
        path.write_text(json.dumps(index))
        inventory["base"]["files"][path.name] = export.sha256(path)
    else:
        inventory["adapter"]["tensors"].pop(next(iter(inventory["adapter"]["tensors"])))
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    assert not out.exists()


def test_completed_export_refuses_tampered_output(tmp_path):
    export = module("export")
    base, adapter, _, inventory = toy_checkpoint(tmp_path)
    out = tmp_path / "out"
    export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    (out / "jemm_lm_head.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)


def test_partial_export_reuses_only_verified_completed_shards(tmp_path, monkeypatch):
    export = module("export")
    base, adapter, _, inventory = toy_checkpoint(tmp_path)
    out = tmp_path / "out"
    original = export.atomic_tensors

    def interrupted(path, tensors):
        if path.name == "model-00002.safetensors":
            raise OSError("simulated storage interruption")
        return original(path, tensors)

    monkeypatch.setattr(export, "atomic_tensors", interrupted)
    with pytest.raises(OSError, match="storage interruption"):
        export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    assert not (out / "jemm_export.json").exists()
    completed = out / "model-00001.safetensors"
    before = completed.stat().st_mtime_ns
    monkeypatch.setattr(export, "atomic_tensors", original)
    result = export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    assert completed.stat().st_mtime_ns == before
    assert result["format"] == "jemm-native/1"


@pytest.mark.parametrize("state", ["partial", "complete"])
@pytest.mark.parametrize("change", ["script", "runtime"])
def test_export_refuses_cached_shards_from_different_producer(tmp_path, monkeypatch, state, change):
    export = module("export")
    base, adapter, _, inventory = toy_checkpoint(tmp_path)
    out = tmp_path / "out"
    original_write = export.atomic_tensors
    if state == "partial":
        def interrupt(path, tensors):
            if path.name == "model-00002.safetensors":
                raise OSError("interrupt")
            original_write(path, tensors)
        monkeypatch.setattr(export, "atomic_tensors", interrupt)
        with pytest.raises(OSError, match="interrupt"):
            export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
        monkeypatch.setattr(export, "atomic_tensors", original_write)
    else:
        export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    if change == "script":
        original_hash = export.sha256
        producer_path = Path(export.__file__).resolve()
        monkeypatch.setattr(export, "sha256", lambda path: "0" * 64 if Path(path).resolve() == producer_path else original_hash(path))
    else:
        assert hasattr(export, "producer_runtime"), "missing exporter runtime provenance"
        old_runtime = export.producer_runtime()
        monkeypatch.setattr(export, "producer_runtime", lambda: {**old_runtime, "torch_version": "changed"})
    with pytest.raises(ValueError, match="different producer|different source"):
        export.export_checkpoint(base, adapter, out, export.PINS, tokenizer=TinyTokenizer(), inventory=inventory)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize("environment_proxy", [False, True])
def test_benchmark_preserves_post_redirect_and_ignores_environment_proxy(tmp_path, monkeypatch, environment_proxy):
    bench = module("benchmark")
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(("POST", self.path))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(302)
            self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(b'{"error":"redirect"}')
        def do_GET(self):
            seen.append(("GET", self.path))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response({"a": .6, "b": .4})).encode())
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # No bypass: the default urllib opener would route even loopback via this proxy.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1" if environment_proxy else "")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1" if environment_proxy else "")
    monkeypatch.setenv("no_proxy", "" if environment_proxy else "*")
    monkeypatch.setenv("NO_PROXY", "" if environment_proxy else "*")
    out = tmp_path / "redirect"
    try:
        with pytest.raises(RuntimeError, match="HTTP 302"):
            bench.run(f"http://127.0.0.1:{server.server_port}", [{"id": "one", "modality": "text", "request": {"state": "", "questions": {"q": {}}}}], out, repetitions=1)
    finally:
        server.shutdown()
        thread.join()
    assert seen == [("POST", "/v1/systemone")]
    record = json.loads((out / "feasibility.jsonl").read_text())
    assert record["status"] == 302
    assert record["response"] == {"error": "redirect"}
    assert not (out / "measured.jsonl").exists()


def test_reference_loader_validates_pinned_source_before_import(tmp_path):
    reference = module("reference_server")
    with pytest.raises(ValueError, match="pinned upstream"):
        reference.validate_source(tmp_path)


def test_reference_loader_refuses_missing_pinned_checkpoint(tmp_path):
    reference = module("reference_server")
    with pytest.raises(ValueError, match="pinned checkpoint"):
        reference.validate_checkpoints(tmp_path, tmp_path)


def test_text_fixture_generator_refuses_unpinned_source(tmp_path):
    fixtures = module("text_fixtures")
    with pytest.raises(ValueError, match="pinned upstream"):
        fixtures.generate(tmp_path, tmp_path, [])


def test_benchmark_writes_failure_response_and_stops(tmp_path):
    bench = module("benchmark")
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(422)
            self.end_headers()
            self.wfile.write(b'{"error":"invalid"}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    out = tmp_path / "evidence"
    try:
        with pytest.raises(RuntimeError, match="422"):
            bench.run(f"http://127.0.0.1:{server.server_port}", [{"id": "one", "modality": "text", "request": {"state": "", "questions": {"q": {}}}}], out, repetitions=2)
    finally:
        server.shutdown()
        thread.join()
    raw = [json.loads(line) for line in (out / "feasibility.jsonl").read_text().splitlines()]
    assert raw[0]["status"] == 422
    assert raw[0]["response"] == {"error": "invalid"}
    assert not (out / "measured.jsonl").exists()


@pytest.mark.parametrize("status", [200, 422])
def test_benchmark_latency_includes_response_parse_excludes_request_serialization(tmp_path, monkeypatch, status):
    bench = module("benchmark")
    import urllib.error
    import io

    clock = {"seconds": 0.0}
    real_loads, real_dumps = json.loads, json.dumps
    response_bytes = real_dumps(response({"a": .6, "b": .4})).encode()

    def serialize(value, *args, **kwargs):
        encoded = real_dumps(value, *args, **kwargs)
        if isinstance(value, dict) and "state" in value and "questions" in value:
            clock["seconds"] += .25
        return encoded

    def parse(value, *args, **kwargs):
        decoded = real_loads(value, *args, **kwargs)
        if value == response_bytes:
            clock["seconds"] += .125
        return decoded

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return response_bytes

    def open_response(request, timeout):
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "failed", {}, io.BytesIO(response_bytes))
        return Reply()

    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock["seconds"])
    monkeypatch.setattr(bench.json, "dumps", serialize)
    monkeypatch.setattr(bench.json, "loads", parse)
    from types import SimpleNamespace
    monkeypatch.setattr(bench, "direct_opener", lambda: SimpleNamespace(open=open_response))
    cases = [{"id": "one", "modality": "text", "request": {"state": "", "questions": {"q": {}}}}]
    out = tmp_path / "timing"
    if status == 200:
        summary = bench.run("http://localhost", cases, out, repetitions=1)
        assert summary["latency"]["text"]["mean_ms"] == pytest.approx(125)
    else:
        with pytest.raises(RuntimeError, match="422"):
            bench.run("http://localhost", cases, out, repetitions=1)
    record = real_loads((out / "feasibility.jsonl").read_text().splitlines()[0])
    assert record["elapsed_ms"] == pytest.approx(125)
    assert record["status"] == status
