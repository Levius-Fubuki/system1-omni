"""Profile a deterministic Cua-S1 workload matrix without instrumenting latency samples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

GOAL = "Save the changed display name."


def case_matrix():
    return [
        {
            "id": f"{w}x{h}-{goal}-q{questions}",
            "size": [w, h],
            "goal": goal,
            "goal_repetitions": 1 if goal == "short" else 64,
            "questions": questions,
        }
        for w, h in [(320, 240), (640, 480)]
        for goal in ["short", "long"]
        for questions in [1, 2, 4, 8]
    ]


def fixture(case, folder):
    from make_example import make_example

    folder.mkdir(parents=True, exist_ok=True)
    value = make_example(folder / f"{case['id']}.png", tuple(case["size"]))
    question = value["questions"]["next"]
    question["instructions"] = " ".join([GOAL] * case["goal_repetitions"])
    value["questions"] = {
        f"q{i + 1}": copy.deepcopy(question) for i in range(case["questions"])
    }
    write_json(folder / f"{case['id']}.json", value)
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="trace selected --profile cases without baseline measurements",
    )
    parser.add_argument(
        "--case", action="append", default=[], help="case id; repeatable"
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="case id for one traced request; repeatable",
    )
    for name, default in [("warmup", 5), ("runs", 2), ("iterations", 50)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args(argv)
    known = {case["id"] for case in case_matrix()}
    for name in ("warmup", "runs", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("case", "profile"):
        selected = getattr(args, name)
        if len(selected) != len(set(selected)) or set(selected) - known:
            parser.error(f"--{name} contains duplicate or unknown case ids")
    if set(args.profile) - set(args.case or known):
        parser.error("--profile cases must also be selected by --case")
    if args.trace_only and (
        not args.profile or (args.case and set(args.case) != set(args.profile))
    ):
        parser.error(
            "--trace-only requires --profile; when supplied, --case must match --profile exactly"
        )
    if not args.list_cases and (args.weights is None or args.output is None):
        parser.error("--weights and --output are required unless --list-cases is used")
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def repository_state(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.PIPE
        )

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain=v1", "-z")
        digest = hashlib.sha256(git("diff", "HEAD", "--binary"))
        untracked = git("ls-files", "--others", "--exclude-standard", "-z")
        for name in sorted(filter(None, untracked.split(b"\0"))):
            path = root / name.decode()
            content = path.read_bytes()
            digest.update(name + b"\0" + hashlib.sha256(content).digest())
        return {
            "revision": revision,
            "dirty": bool(status),
            "diff_sha256": digest.hexdigest(),
            "untracked_files": [x.decode() for x in untracked.split(b"\0") if x],
        }
    except (subprocess.CalledProcessError, OSError) as exc:
        return {"revision": None, "dirty": None, "diff_sha256": None, "error": str(exc)}


def benchmark(engine, request, *, warmup, runs, iterations):
    import torch
    from evaluate_multimodal import measure, quantiles

    if min(warmup, runs, iterations) <= 0:
        raise ValueError("warmup, runs and iterations must be positive")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(warmup):
        engine.predict(request)
    torch.cuda.synchronize()
    warmup_total_ms = (time.perf_counter() - start) * 1000
    result = []
    for run in range(runs):
        torch.cuda.reset_peak_memory_stats()
        # Indexing drops the response immediately; measured calls have no hooks/ranges.
        latencies = [
            measure(lambda: engine.predict(request))[1] for _ in range(iterations)
        ]
        result.append(
            {
                "run": run + 1,
                "latencies_ms": latencies,
                **quantiles(latencies),
                "requests_per_second_serial": 1000 / statistics.mean(latencies),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    return {"warmup_total_ms": warmup_total_ms, "runs": result}


@contextmanager
def instrument(engine, record):
    """Scope wrappers/hooks around existing inference, including exception cleanup."""
    modules = list(engine.model.named_modules())
    discovered = {}
    for stage, leaf in [
        ("vision", "visual"),
        ("language", "language_model"),
        ("output_projection", "lm_head"),
    ]:
        matches = [
            (name, module) for name, module in modules if name.split(".")[-1] == leaf
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{stage}: expected one {leaf} module, found {[n for n, _ in matches]}"
            )
        discovered[stage] = matches[0][0]
    active, handles, readout = [], [], []
    missing = object()
    originals = {
        name: engine.__dict__.get(name, missing) for name in ("prepare", "score")
    }
    prepare, score = engine.prepare, engine.score

    def enter(name):
        span = record("cua." + name)
        span.__enter__()
        active.append(span)
        return span

    def leave(span):
        active.remove(span)
        span.__exit__(*sys.exc_info())

    def attach(module, stage, is_root=False):
        spans = []

        def before(module, inputs):
            spans.append(enter(stage))

        def after(module, inputs, output):
            if spans:
                leave(spans.pop())
            if is_root and output is not None:
                readout.append(enter("readout"))

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))

    class Transfer:
        def __init__(self, inputs):
            self.inputs = inputs

        def to(self, *args, **kwargs):
            with record("cua.transfer"):
                return self.inputs.to(*args, **kwargs)

    def traced_prepare(*args, **kwargs):
        with record("cua.prepare"):
            return prepare(*args, **kwargs)

    def traced_score(inputs, question):
        try:
            return score(Transfer(inputs), question)
        finally:
            while readout:
                leave(readout.pop())

    try:
        attach(engine.model, "forward", is_root=True)
        for stage, name in discovered.items():
            attach(dict(modules)[name], stage)
        engine.prepare, engine.score = traced_prepare, traced_score
        yield discovered
    finally:
        for handle in reversed(handles):
            handle.remove()
        while active:
            leave(active[-1])
        for name, original in originals.items():
            if original is missing:
                engine.__dict__.pop(name, None)
            else:
                setattr(engine, name, original)


def profile_request(engine, request, trace):
    import torch

    expected = engine.predict(request)
    torch.cuda.synchronize()
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof,
        instrument(engine, torch.profiler.record_function) as modules,
    ):
        actual = engine.predict(request)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    if actual != expected:
        raise ValueError("instrumentation parity failed (not an upstream parity check)")
    operators = [
        {
            "op": item.key,
            "device_type": str(item.device_type),
            "is_user_annotation": item.is_user_annotation,
            "input_shapes": item.input_shapes,
            "count": item.count,
            "cpu_time_us": item.cpu_time_total,
            "self_cpu_time_us": item.self_cpu_time_total,
            "cuda_time_us": item.device_time_total,
            "self_cuda_time_us": item.self_device_time_total,
        }
        for item in prof.key_averages(group_by_input_shape=True)
    ]
    stage_counts = {}
    for stage in (
        "prepare",
        "transfer",
        "forward",
        "vision",
        "language",
        "output_projection",
        "readout",
    ):
        count = sum(
            item["count"]
            for item in operators
            if item["op"] == "cua." + stage
            and item["device_type"] == "DeviceType.CPU"
            and item["is_user_annotation"]
        )
        if count != len(request.questions):
            raise ValueError(
                f"{stage}: observed {count} ranges; expected {len(request.questions)}"
            )
        stage_counts[stage] = count
    with trace.open("rb") as handle:
        trace_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "trace": trace.name,
        "trace_sha256": trace_sha256,
        "trace_bytes": trace.stat().st_size,
        "instrumentation_exact_parity": True,
        "modules": modules,
        "request_count": 1,
        "uninstrumented_reference_requests": 1,
        "forward_count": stage_counts["forward"],
        "stage_counts": stage_counts,
        "timing_note": "Count invocations from CPU user annotations only. CPU and CUDA rows with the same name are distinct views, not additional invocations; do not add their durations. Inclusive nested ranges overlap. CUDA fields are raw profiler aggregates and may include synthetic annotations, not disjoint kernel time or wall time. Readout starts after root forward; transfer covers inputs.to().",
        "operators": sorted(
            operators, key=lambda x: x["self_cuda_time_us"], reverse=True
        ),
    }


def input_metadata(engine, request):
    from evaluate_multimodal import fingerprint

    from models.cua_s1.multimodal.model import MAX_TOKENS

    result = []
    for question in request.questions:
        inputs = engine.prepare(request.image, question)
        tokens = int(inputs["input_ids"].shape[-1])
        if tokens > MAX_TOKENS:
            raise ValueError(
                f"{question.name}: processed prompt exceeds {MAX_TOKENS} tokens"
            )
        result.append(
            {
                "question": question.name,
                "input_tokens": tokens,
                "goal_tokens": len(
                    engine.tokenizer.encode(question.goal, add_special_tokens=False)
                ),
                "inputs": fingerprint(inputs),
            }
        )
    return result


def main(argv=None):
    args = parse_args(argv)
    selectors = args.profile if args.trace_only else args.case
    selected = [
        case for case in case_matrix() if not selectors or case["id"] in selectors
    ]
    if args.list_cases:
        print(json.dumps(selected, indent=2))
        return 0
    if (args.output / "report.json").exists():
        raise ValueError(
            "output already contains report.json; use a fresh output directory"
        )
    report = {
        "schema_version": 1,
        "mode": "trace-only" if args.trace_only else "benchmark",
        "status": "running",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "config": {
            "warmup": 0 if args.trace_only else args.warmup,
            "runs": 0 if args.trace_only else args.runs,
            "iterations": 0 if args.trace_only else args.iterations,
            "weights": str(args.weights.resolve()),
            "profile": args.profile,
            "batch_size": 1,
            "concurrency": 1,
            "timed_scope": None
            if args.trace_only
            else "engine.predict(request), synchronized; excludes parse_request and metadata preparation",
        },
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", report)
    current = None
    try:
        from evaluate_multimodal import environment, measure

        from models.cua_s1.multimodal.model import MultimodalEngine
        from models.cua_s1.multimodal.protocol import parse_request

        report["environment"] = environment()
        engine, report["load_ms"] = measure(
            lambda: MultimodalEngine(
                str(args.weights / "Qwen3.5-4B"),
                str(args.weights / "cua-s1-4b-0.2/multimodal"),
            )
        )
        report["adapter_modules"] = engine.adapter_modules
        for case in selected:
            current = {**case, "status": "running"}
            report["cases"].append(current)
            write_json(args.output / "report.json", report)
            value = fixture(case, args.output / "fixtures")
            request = parse_request(value)
            current["fixture_sha256"] = hashlib.sha256(
                json.dumps(value, sort_keys=True).encode()
            ).hexdigest()
            current["questions_metadata"] = input_metadata(engine, request)
            current["input_tokens"] = sum(
                q["input_tokens"] for q in current["questions_metadata"]
            )
            if not args.trace_only:
                current.update(
                    benchmark(
                        engine,
                        request,
                        warmup=args.warmup,
                        runs=args.runs,
                        iterations=args.iterations,
                    )
                )
            current["status"] = "prepared" if args.trace_only else "complete"
            write_json(args.output / f"{case['id']}.json", current)
            write_json(args.output / "report.json", report)
            print(
                f"{case['id']}: {current['status']} ({0 if args.trace_only else args.runs * args.iterations} latency samples)",
                flush=True,
            )
        # Every baseline finishes before profiler state can affect later measurements.
        for current in report["cases"]:
            if current["id"] not in args.profile:
                continue
            value = json.loads(
                (args.output / "fixtures" / f"{current['id']}.json").read_text()
            )
            request = parse_request(value)
            current["profile"] = profile_request(
                engine, request, args.output / f"{current['id']}.trace.json"
            )
            current["status"] = "complete"
            write_json(args.output / f"{current['id']}.json", current)
            write_json(args.output / "report.json", report)
            print(f"{current['id']}: trace complete", flush=True)
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - preserve partial results for any run failure.
        error = {"type": type(exc).__name__, "message": str(exc)}
        report.update(status="failed", error=error)
        if current is not None:
            current.update(status="failed", error=error)
            write_json(args.output / f"{current['id']}.json", current)
        print(f"Profiling failed: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        write_json(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
