"""CPU contracts for the profiling harness; no weights or torch installation required."""

import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "recipe/cua_s1"


@pytest.fixture
def profiling(monkeypatch):
    path = RECIPE / "profile_multimodal.py"
    assert path.exists(), "profiling harness is not implemented"
    monkeypatch.syspath_prepend(str(RECIPE))
    spec = importlib.util.spec_from_file_location("profiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_is_deterministic_and_valid(profiling, tmp_path):
    from models.cua_s1.multimodal.protocol import parse_request

    matrix = profiling.case_matrix()
    assert len(matrix) == len({c["id"] for c in matrix}) == 16
    assert matrix[-1]["id"] == "640x480-long-q8"
    for case in matrix:
        one = profiling.fixture(case, tmp_path / "one")
        two = profiling.fixture(case, tmp_path / "two")
        assert one == two
        request = parse_request(one)
        assert request.image.size == tuple(case["size"])
        assert len(request.questions) == case["questions"]
        assert len({q.goal for q in request.questions}) == 1
    assert len(request.questions[0].goal) < 16384


@pytest.mark.parametrize("option", ["--warmup", "--runs", "--iterations"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_counts_are_rejected(profiling, option, value):
    with pytest.raises(SystemExit):
        profiling.parse_args(["--list-cases", option, value])


@pytest.mark.parametrize(
    "selectors",
    [
        ["--case", "unknown"],
        ["--case", "320x240-short-q1", "--case", "320x240-short-q1"],
        ["--profile", "unknown"],
        ["--profile", "320x240-short-q1", "--profile", "320x240-short-q1"],
        ["--case", "320x240-short-q1", "--profile", "640x480-long-q8"],
    ],
)
def test_invalid_selection_rejected(profiling, selectors):
    with pytest.raises(SystemExit):
        profiling.parse_args(["--list-cases", *selectors])


def test_help_and_list_cases_do_not_import_torch(profiling):
    for flag in ["--help", "--list-cases"]:
        command = (
            "import runpy,sys;sys.modules['torch']=None;"
            f"sys.path.insert(0,{str(RECIPE)!r});"
            f"sys.argv=['profile_multimodal.py',{flag!r}];"
            f"runpy.run_path({str(RECIPE / 'profile_multimodal.py')!r},run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "320x240-short-q1" in result.stdout or flag == "--help"


def test_measurement_accounting_and_synchronization(profiling, monkeypatch):
    events = []
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        reset_peak_memory_stats=lambda: events.append("reset"),
        max_memory_allocated=lambda: 100,
        max_memory_reserved=lambda: 200,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    engine = SimpleNamespace(predict=lambda request: events.append("predict"))
    result = profiling.benchmark(engine, object(), warmup=2, runs=2, iterations=3)
    assert result["warmup_total_ms"] >= 0
    assert len(result["runs"]) == 2
    assert all(len(run["latencies_ms"]) == 3 for run in result["runs"])
    assert (
        events
        == ["sync", "predict", "predict", "sync", "reset"]
        + ["sync", "predict", "sync"] * 3
        + ["reset"]
        + ["sync", "predict", "sync"] * 3
    )
    assert all(run["peak_allocated_bytes"] == 100 for run in result["runs"])


class Module:
    def __init__(self, name, events, fail=False):
        self.name, self.events, self.fail = name, events, fail
        self.pre, self.post = [], []
        self.children = []

    def named_modules(self):
        return iter([(self.name, self), *self.children])

    def register_forward_pre_hook(self, hook):
        self.pre.append(hook)
        return SimpleNamespace(remove=lambda: self.pre.remove(hook))

    def register_forward_hook(self, hook, always_call=False):
        assert always_call
        self.post.append(hook)
        return SimpleNamespace(remove=lambda: self.post.remove(hook))

    def __call__(self):
        output = None
        try:
            for hook in self.pre:
                hook(self, ())
            for _, child in self.children:
                child()
            if self.fail:
                raise RuntimeError("forward failed")
            output = "output"
            return output
        finally:
            for hook in self.post:
                hook(self, (), output)


def fake_engine(events, fail=False):
    root = Module("", events)
    root.children = [
        (name, Module(name, events, fail=fail and name == "visual"))
        for name in ("visual", "language_model", "lm_head")
    ]

    class Engine:
        model = root

        def prepare(self, image, question):
            return SimpleNamespace(to=lambda device: events.append("transfer"))

        def score(self, inputs, question):
            inputs.to("cuda")
            self.model()
            events.append("readout")
            return [1.0]

        def predict(self, request):
            return self.score(self.prepare(None, None), None)

    return Engine()


@pytest.mark.parametrize("fail", [False, True])
def test_instrumentation_is_scoped_and_ranges_close(profiling, fail):
    events, active = [], []
    engine = fake_engine(events, fail)
    original = dict(engine.__dict__)

    @contextmanager
    def record(name):
        active.append(name)
        events.append("enter:" + name)
        try:
            yield
        finally:
            assert active.pop() == name
            events.append("exit:" + name)

    try:
        with profiling.instrument(engine, record) as discovered:
            assert set(discovered) >= {"vision", "language", "output_projection"}
            assert engine.predict(None) == [1.0]
    except RuntimeError:
        assert fail
    assert not active
    assert engine.__dict__ == original
    assert all(not m.pre and not m.post for _, m in engine.model.named_modules())
    assert "enter:cua.prepare" in events
    assert "enter:cua.transfer" in events
    if not fail:
        assert (
            events.index("enter:cua.readout")
            < events.index("readout")
            < events.index("exit:cua.readout")
        )
        events.clear()
        engine.predict(None)
        assert events == ["transfer", "readout"]


def test_missing_modules_fail_before_installing_hooks(profiling):
    engine = fake_engine([])
    engine.model.children.pop()
    with (
        pytest.raises(ValueError, match="output_projection"),
        profiling.instrument(engine, None),
    ):
        pass
    assert all(not m.pre and not m.post for _, m in engine.model.named_modules())


def test_provenance_hash_includes_untracked_sources(profiling, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    path = tmp_path / "new.py"
    path.write_text("one")
    first = profiling.repository_state(tmp_path)
    path.write_text("two")
    second = profiling.repository_state(tmp_path)
    assert first["dirty"] and second["dirty"]
    assert first["diff_sha256"] != second["diff_sha256"]
    assert first["revision"] == second["revision"]
    assert json.loads(json.dumps(second)) == second


@pytest.mark.parametrize("profile_fails", [False, True])
def test_all_baselines_precede_profiles_and_results_survive_failure(
    profiling, monkeypatch, tmp_path, profile_fails
):
    import evaluate_multimodal

    from models.cua_s1.multimodal import model

    events = []
    engine = SimpleNamespace(adapter_modules=178)
    monkeypatch.setattr(model, "MultimodalEngine", lambda *args: engine)
    monkeypatch.setattr(evaluate_multimodal, "environment", lambda: {"gpu": "fake"})
    monkeypatch.setattr(evaluate_multimodal, "measure", lambda call: (call(), 1.0))
    monkeypatch.setattr(
        profiling, "repository_state", lambda root: {"revision": "fake"}
    )
    monkeypatch.setattr(
        profiling, "input_metadata", lambda *args: [{"input_tokens": 42}]
    )

    def benchmark(*args, **kwargs):
        events.append("baseline")
        return {"warmup_total_ms": 1.0, "runs": [{"latencies_ms": [1.0]}]}

    def profile(*args):
        events.append("profile")
        if profile_fails:
            raise RuntimeError("profile failed")
        return {"instrumentation_exact_parity": True}

    monkeypatch.setattr(profiling, "benchmark", benchmark)
    monkeypatch.setattr(profiling, "profile_request", profile)
    first, second = "320x240-short-q1", "320x240-short-q2"
    code = profiling.main(
        [
            "--weights",
            str(tmp_path / "weights"),
            "--output",
            str(tmp_path),
            "--case",
            first,
            "--case",
            second,
            "--profile",
            first,
        ]
    )
    assert events == ["baseline", "baseline", "profile"]
    assert code == int(profile_fails)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == ("failed" if profile_fails else "complete")
    assert json.loads((tmp_path / f"{second}.json").read_text())["status"] == "complete"
    saved = json.loads((tmp_path / f"{first}.json").read_text())
    assert saved["runs"] == [{"latencies_ms": [1.0]}]
    if profile_fails:
        assert saved["error"]["message"] == "profile failed"


@pytest.mark.parametrize("missing_stage", [None, "vision"])
def test_trace_has_shapes_times_checksum_and_exact_parity(
    profiling, monkeypatch, tmp_path, missing_stage
):
    import hashlib

    events = []
    engine = fake_engine(events)

    @contextmanager
    def record(name):
        yield

    class Profiler:
        def __enter__(self):
            events.append("profiler")
            return self

        def __exit__(self, *args):
            pass

        def export_chrome_trace(self, path):
            Path(path).write_bytes(b"trace")

        def key_averages(self, group_by_input_shape):
            assert group_by_input_shape
            return [
                SimpleNamespace(
                    key=key,
                    input_shapes=[[1, 2]],
                    count=1,
                    cpu_time_total=8,
                    self_cpu_time_total=3,
                    device_time_total=6,
                    self_device_time_total=2,
                )
                for key in ["op"]
                + [
                    "cua." + name
                    for name in [
                        "prepare",
                        "transfer",
                        "forward",
                        "vision",
                        "language",
                        "output_projection",
                        "readout",
                    ]
                    if name != missing_stage
                ]
            ]

    def start(**kwargs):
        assert kwargs["record_shapes"] is True
        return Profiler()

    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
        profiler=SimpleNamespace(
            profile=start,
            record_function=record,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    if missing_stage:
        with pytest.raises(ValueError, match="vision.*0.*1"):
            profiling.profile_request(
                engine, SimpleNamespace(questions=[None]), tmp_path / "trace.json"
            )
        return
    result = profiling.profile_request(
        engine, SimpleNamespace(questions=[None]), tmp_path / "trace.json"
    )
    assert events[:3] == ["transfer", "readout", "profiler"]
    assert result["instrumentation_exact_parity"] is True
    assert result["forward_count"] == 1
    assert result["stage_counts"] == {
        name: 1
        for name in [
            "prepare",
            "transfer",
            "forward",
            "vision",
            "language",
            "output_projection",
            "readout",
        ]
    }
    assert result["trace_sha256"] == hashlib.sha256(b"trace").hexdigest()
    assert result["trace_bytes"] == 5
    assert result["operators"][0]["input_shapes"] == [[1, 2]]
    assert result["operators"][0]["self_cuda_time_us"] == 2
