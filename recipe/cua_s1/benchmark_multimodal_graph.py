"""Compare eager and CUDA Graph language forwards on the reused multimodal path."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from benchmark_image_reuse import distinct_fixture
from profile_multimodal import case_matrix, fixture, repository_state, write_json


def language_inputs(engine, inputs, features):
    """Prepare per-question tensors exactly as the production reuse path does."""
    core = engine.model.get_base_model().model
    text = {
        key: value.to(engine.model.device)
        for key, value in inputs.items()
        if key != "pixel_values"
    }
    ids = text.pop("input_ids")
    embeds = core.get_input_embeddings()(ids)
    image_embeds = features.to(embeds.device, embeds.dtype)
    mask, _ = core.get_placeholder_mask(
        ids, inputs_embeds=embeds, image_features=image_embeds
    )
    embeds = embeds.masked_scatter(mask, image_embeds)
    positions, _ = core.get_rope_index(
        input_ids=ids,
        mm_token_type_ids=text.pop("mm_token_type_ids"),
        image_grid_thw=text.pop("image_grid_thw"),
        attention_mask=text.get("attention_mask"),
    )
    return {
        "inputs_embeds": embeds,
        "position_ids": positions,
        "attention_mask": text["attention_mask"],
    }


class GraphForward:
    """One exact-shape graph; static buffers remain alive for every replay."""

    def __init__(self, model, example):
        import torch

        self.model = model
        self.buffers = {name: value.clone() for name, value in example.items()}
        self.signatures = {
            name: tensor_signature(value) for name, value in example.items()
        }
        self.capture_ms = 0.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.static_logits = self._forward()
        torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - start) * 1000
        self.capture_allocated_delta_bytes = (
            torch.cuda.memory_allocated() - allocated_before
        )
        self.capture_reserved_delta_bytes = (
            torch.cuda.memory_reserved() - reserved_before
        )

    def _forward(self):
        return self.model(**self.buffers, logits_to_keep=1, use_cache=False).logits[
            0, -1, :
        ]

    def forward(self, values):
        for name, value in values.items():
            if tensor_signature(value) != self.signatures[name]:
                raise ValueError(f"graph tensor layout changed for {name}")
            self.buffers[name].copy_(value)
        self.graph.replay()
        return self.static_logits


def tensor_signature(value):
    return (
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
    )


class Scorer:
    def __init__(self, engine, graph=False):
        self.engine = engine
        self.graph = graph
        self.runners = {}
        self.capture_ms = []

    def score(self, inputs, question, features):
        import torch

        from models.cua_s1.multimodal.model import letter_ids

        with torch.no_grad():
            values = language_inputs(self.engine, inputs, features)
            if self.graph:
                key = tuple(
                    (name, tensor_signature(value)) for name, value in values.items()
                )
                if key not in self.runners:
                    self.runners[key] = GraphForward(self.engine.model, values)
                    self.capture_ms.append(self.runners[key].capture_ms)
                logits = self.runners[key].forward(values)
            else:
                logits = self.engine.model(
                    **values, logits_to_keep=1, use_cache=False
                ).logits[0, -1, :]
            ids = letter_ids(self.engine.tokenizer, len(question.keys))
            return torch.softmax(
                logits[torch.tensor(ids, device=logits.device)].float(), dim=-1
            ).tolist()


def run_variant(engine, scorer, request):
    original = engine.score_reused
    engine.score_reused = scorer.score
    try:
        return engine.predict(request)
    finally:
        engine.score_reused = original


def measure(call):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def summarize(values):
    ordered = sorted(values)
    return {
        "samples_ms": values,
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def response_difference(expected, actual):
    if expected["model"] != actual["model"] or expected["usage"] != actual["usage"]:
        raise ValueError("model identity or usage changed")
    if expected["answers"].keys() != actual["answers"].keys():
        raise ValueError("answer keys changed")
    worst = 0.0
    for name, before in expected["answers"].items():
        after = actual["answers"][name]
        if before["type"] != after["type"] or before["choice"] != after["choice"]:
            raise ValueError(f"{name}: selected choice changed")
        if before["probabilities"].keys() != after["probabilities"].keys():
            raise ValueError(f"{name}: candidate keys changed")
        worst = max(
            worst,
            *(
                abs(value - after["probabilities"][key])
                for key, value in before["probabilities"].items()
            ),
        )
    return worst


def changed_text_request(engine, request):
    """Find different words that retain every prepared question's tensor shape."""
    import torch

    before = engine.prepare_reused(request.image, request.questions)
    for verb in ("Open", "Close", "View", "Read", "Edit", "Keep"):
        questions = tuple(
            replace(question, goal=question.goal.replace("Save", verb))
            for question in request.questions
        )
        if questions == request.questions:
            continue
        after = engine.prepare_reused(request.image, questions)
        if all(
            old["input_ids"].shape == new["input_ids"].shape
            and not torch.equal(old["input_ids"], new["input_ids"])
            for old, new in zip(before, after)
        ):
            return replace(request, questions=questions), verb
    raise ValueError("no same-shape changed-text fixture available")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    known = {case["id"] for case in case_matrix()} | {"640x480-distinct-q8"}
    if len(args.case) != len(set(args.case)) or set(args.case) - known:
        parser.error("duplicate or unknown case")
    if not args.case:
        parser.error("select at least one --case for this bounded experiment")
    if min(args.warmup, args.runs, args.iterations) <= 0:
        parser.error("warmup, runs and iterations must be positive")
    if (args.output / "report.json").exists():
        parser.error("report already exists; use a fresh output directory")

    import torch
    from evaluate_multimodal import environment

    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "environment": environment(),
        "config": {
            "warmup": args.warmup,
            "runs": args.runs,
            "iterations": args.iterations,
            "cases": args.case,
            "scope": "synchronized engine.predict; Graph captures model forward after vision, text embeddings and 3D positions; both variants use_cache=False",
        },
        "cases": [],
    }
    write_json(report_path, report)
    try:
        engine = MultimodalEngine(
            str(args.weights / "Qwen3.5-4B"),
            str(args.weights / "cua-s1-4b-0.2/multimodal"),
        )
        selected = case_matrix() + [{"id": "640x480-distinct-q8"}]
        selected = [case for case in selected if case["id"] in args.case]
        for case in selected:
            value = (
                distinct_fixture(args.output / "fixtures")
                if case["id"] == "640x480-distinct-q8"
                else fixture(case, args.output / "fixtures")
            )
            request = parse_request(value)
            item = {"id": case["id"], "status": "running", "runs": []}
            report["cases"].append(item)
            write_json(report_path, report)
            if len(request.questions) == 1:
                item["status"] = "skipped_single_question"
                write_json(report_path, report)
                continue
            eager, graph = Scorer(engine), Scorer(engine, graph=True)
            baseline = engine.predict(request)
            eager_response = run_variant(engine, eager, request)
            graph_response = run_variant(engine, graph, request)
            if baseline != eager_response:
                item["parity_diagnostic"] = {
                    "baseline": baseline,
                    "eager": eager_response,
                    "graph": graph_response,
                }
                write_json(report_path, report)
                raise ValueError(f"{case['id']}: eager response differs")
            item["max_graph_probability_difference"] = response_difference(
                baseline, graph_response
            )
            if item["max_graph_probability_difference"] > 0.002:
                item["parity_diagnostic"] = {
                    "baseline": baseline,
                    "graph": graph_response,
                    "graph_shapes": [list(key) for key in graph.runners],
                }
                item["status"] = "rejected_graph"
                write_json(report_path, report)
                raise ValueError(
                    f"{case['id']}: graph probability difference exceeds 0.002"
                )
            item["exact_eager_response_parity"] = True
            # Replay the same graph after a substantial image change, then return
            # to the original input. This detects stale captured input pointers.
            from PIL import Image

            changed = replace(
                request, image=Image.new("RGB", request.image.size, "black")
            )
            changed_eager = run_variant(engine, eager, changed)
            graphs_before = len(graph.runners)
            changed_graph = run_variant(engine, graph, changed)
            original_again = run_variant(engine, graph, request)
            if len(graph.runners) != graphs_before:
                raise ValueError(
                    f"{case['id']}: image change unexpectedly changed graph shapes"
                )
            item["changed_image_max_probability_difference"] = response_difference(
                changed_eager, changed_graph
            )
            if item["changed_image_max_probability_difference"] > 0.002:
                raise ValueError(f"{case['id']}: changed-image graph response differs")
            if response_difference(baseline, original_again) > 0.002:
                raise ValueError(f"{case['id']}: original input was not restored")
            if changed_eager == baseline:
                raise ValueError(
                    f"{case['id']}: changed image did not change eager response"
                )
            item["changed_image_replay_checked"] = True
            changed_text, verb = changed_text_request(engine, request)
            text_eager = run_variant(engine, eager, changed_text)
            graphs_before = len(graph.runners)
            text_graph = run_variant(engine, graph, changed_text)
            original_again = run_variant(engine, graph, request)
            if len(graph.runners) != graphs_before:
                raise ValueError(
                    f"{case['id']}: text change unexpectedly changed graph shapes"
                )
            item["changed_text_verb"] = verb
            item["changed_text_max_probability_difference"] = response_difference(
                text_eager, text_graph
            )
            if item["changed_text_max_probability_difference"] > 0.002:
                item["status"] = "rejected_graph"
                write_json(report_path, report)
                raise ValueError(f"{case['id']}: changed-text graph response differs")
            if response_difference(baseline, original_again) > 0.002:
                raise ValueError(
                    f"{case['id']}: original input was not restored after text change"
                )
            if text_eager == baseline:
                raise ValueError(
                    f"{case['id']}: changed text did not change eager response"
                )
            item["changed_text_replay_checked"] = True
            item["fixture_sha256"] = hashlib.sha256(
                json.dumps(value, sort_keys=True).encode()
            ).hexdigest()
            item["capture_ms"] = graph.capture_ms
            item["graph_count"] = len(graph.runners)
            item["capture_allocated_delta_bytes"] = [
                runner.capture_allocated_delta_bytes
                for runner in graph.runners.values()
            ]
            item["capture_reserved_delta_bytes"] = [
                runner.capture_reserved_delta_bytes for runner in graph.runners.values()
            ]
            for _ in range(args.warmup):
                run_variant(engine, eager, request)
                run_variant(engine, graph, request)
            for run in range(args.runs):
                samples = {"eager": [], "graph": []}
                orders = []
                peaks = {"eager": 0, "graph": 0}
                for iteration in range(args.iterations):
                    order = ["eager", "graph"]
                    if (run + iteration) % 2:
                        order.reverse()
                    orders.append(order)
                    for name in order:
                        scorer = eager if name == "eager" else graph
                        torch.cuda.reset_peak_memory_stats()
                        response, elapsed = measure(
                            lambda scorer=scorer, request=request: run_variant(
                                engine, scorer, request
                            )
                        )
                        if name == "eager" and response != baseline:
                            raise ValueError(
                                f"{case['id']}: timed eager response differs"
                            )
                        if (
                            name == "graph"
                            and response_difference(baseline, response) > 0.002
                        ):
                            raise ValueError(
                                f"{case['id']}: timed graph response differs"
                            )
                        samples[name].append(elapsed)
                        peaks[name] = max(
                            peaks[name], torch.cuda.max_memory_allocated()
                        )
                item["runs"].append(
                    {
                        "orders": orders,
                        **{
                            name: {
                                **summarize(values),
                                "peak_allocated_bytes": peaks[name],
                            }
                            for name, values in samples.items()
                        },
                    }
                )
                write_json(report_path, report)
            item["status"] = "complete"
            del graph
            torch.cuda.empty_cache()
            write_json(report_path, report)
            print(
                case["id"],
                [
                    (round(x["eager"]["p50_ms"], 2), round(x["graph"]["p50_ms"], 2))
                    for x in item["runs"]
                ],
                flush=True,
            )
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
