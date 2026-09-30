"""Exercise real GraphRuntime._capture cleanup without a CUDA installation.

REVIEW_GRAPH_REF selects the source revision independently of the checkout.
Only GPU primitives, model execution, and owned GPU resources are faked.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPOSITORY = Path(os.environ.get("REVIEW_REPOSITORY", Path.cwd()))
SOURCE_PATH = "src/models/cua_s1/multimodal/graph_runtime.py"


@pytest.fixture
def runtime_module(monkeypatch):
    revision = os.environ.get("REVIEW_GRAPH_REF", "codex/review-gpu-pr38")
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{SOURCE_PATH}"],
        cwd=REPOSITORY,
        text=True,
    )
    package = ModuleType("review_candidate_cleanup")
    package.__path__ = []
    admission = ModuleType("review_candidate_cleanup.graph_admission")
    admission.AdmissionPolicy = object
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, admission.__name__, admission)
    name = "review_candidate_cleanup.graph_runtime"
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, None)
    )
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, f"{revision}:{SOURCE_PATH}", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize(
    ("failure", "stat"),
    [
        ("oom", "capture_oom"),
        ("runtime_error", "capture_error"),
        ("numerical_mismatch", "numerical_mismatch"),
        ("oversize", "memory_budget"),
    ],
)
def test_rejected_candidate_closes_despite_retained_reference(
    runtime_module, monkeypatch, failure, stat
):
    module = runtime_module
    events = []

    class OutOfMemoryError(RuntimeError):
        pass

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            OutOfMemoryError=OutOfMemoryError,
            synchronize=lambda device: events.append("synchronize"),
            empty_cache=lambda: events.append("empty_cache"),
        ),
        equal=lambda reference, output: failure != "numerical_mismatch",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Block:
        closed = False

        def close(self):
            self.closed = True
            events.append("block_close")

    class Pool:
        closed = False

        def close(self):
            self.closed = True
            events.append("pool_close")

    retained = []
    original_shape_entry = module._ShapeEntry

    def candidate_factory(**kwargs):
        block, pool = Block(), Pool()
        candidate = original_shape_entry(
            **kwargs, blocks={0: block}, pool=pool, bytes=2
        )
        retained.append((candidate, block, pool))
        return candidate

    monkeypatch.setattr(module, "_ShapeEntry", candidate_factory)
    runtime = object.__new__(module.GraphRuntime)
    runtime.stats = dict.fromkeys(
        [
            "capture_oom",
            "capture_error",
            "numerical_mismatch",
            "memory_budget",
            "rejected",
        ],
        0,
    )
    runtime.disabled = module.OrderedDict()
    runtime.cache = module.GraphCache(max_shapes=2, max_bytes=1)
    reference = object()
    runtime._eager = lambda values: reference

    def run_segments(values, candidate):
        if failure == "oom":
            raise OutOfMemoryError("injected healthy-context allocation failure")
        if failure == "runtime_error":
            raise RuntimeError("injected recoverable capture failure")
        return object()

    runtime._run_segments = run_segments
    values = {"inputs_embeds": SimpleNamespace(device="cuda:0")}
    key = ("candidate", failure)
    try:
        assert runtime._capture(values, key) is reference
        candidate, block, pool = retained[0]
        assert runtime.stats[stat] == 1
        assert runtime.stats["rejected"] == 1
        assert key in runtime.disabled
        assert len(runtime.cache) == 0
        assert not candidate.blocks, (
            "retained rejected candidate still owns graph blocks"
        )
        assert candidate.pool is None, "retained rejected candidate still owns its pool"
        assert block.closed and pool.closed
        assert events.index("block_close") < events.index("pool_close")
        if failure == "oom":
            assert events.index("pool_close") < events.index("empty_cache")
    finally:
        # Keep test teardown deterministic even against the unfixed revision.
        for candidate, _, _ in retained:
            candidate.close()
