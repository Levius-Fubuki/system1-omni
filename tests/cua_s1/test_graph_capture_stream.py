"""CUDA stream ownership contracts, exercised without a GPU."""

import ctypes
import sys
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from models.cua_s1.multimodal.graph_runtime import _GraphSegment


@pytest.fixture
def cuda(monkeypatch):
    state = SimpleNamespace(
        events=[], next_handle=100, captures=[], pools=[], fail=False
    )

    class Tensor:
        shape = (1, 2, 3)
        dtype = "bf16"
        device = "cuda:0"

        def stride(self):
            return (6, 3, 1)

        def clone(self):
            return Tensor()

        def untyped_storage(self):
            return SimpleNamespace(nbytes=lambda: 12)

    class Stream:
        def wait_stream(self, other):
            pass

    class Graph:
        def reset(self):
            state.events.append("reset")

    def create(address):
        state.next_handle += 1
        ctypes.c_void_p.from_address(address).value = state.next_handle
        state.events.append("create")
        return 0

    def destroy(handle):
        state.events.append(("destroy", handle))
        return 0

    def check_error(result):
        if result:
            raise RuntimeError("CUDA failure")

    @contextmanager
    def graph(g, stream=None, pool=None):
        state.captures.append(stream)
        state.pools.append(pool)
        if state.fail:
            raise ValueError("capture failed")
        yield

    state.torch = SimpleNamespace(
        no_grad=nullcontext,
        cuda=SimpleNamespace(
            memory_allocated=lambda device: 0,
            get_allocator_backend=lambda: "native",
            graph_pool_handle=lambda: (0, object()),
            memory_snapshot=lambda **kwargs: [],
            Stream=lambda **kwargs: Stream(),
            current_stream=lambda device: Stream(),
            stream=lambda stream: nullcontext(),
            device=lambda device: nullcontext(),
            ExternalStream=lambda handle, **kwargs: SimpleNamespace(cuda_stream=handle),
            cudart=lambda: SimpleNamespace(
                cudaStreamCreate=create, cudaStreamDestroy=destroy
            ),
            check_error=check_error,
            CUDAGraph=Graph,
            graph=graph,
            synchronize=lambda device: state.events.append("sync"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", state.torch)
    state.make = lambda **kwargs: _GraphSegment(
        [lambda *a, **kw: Tensor()], 0, 1, Tensor(), Tensor(), **kwargs
    )
    return state


def test_live_segments_own_unique_capture_streams_beyond_pool_size(cuda):
    segments = [cuda.make() for _ in range(40)]
    assert all(stream is not None for stream in cuda.captures)
    assert len({stream.cuda_stream for stream in cuda.captures}) == 40
    for segment in segments:
        segment.close()


def test_close_waits_for_replay_then_resets_releases_and_destroys(cuda):
    segment = cuda.make()
    tensors = [
        weakref.ref(getattr(segment, name))
        for name in ("static_hidden", "static_mask", "static_output")
    ]
    handle = cuda.captures[0].cuda_stream
    original_destroy = segment._torch.cuda.cudart().cudaStreamDestroy

    def destroy(value):
        assert all(ref() is None for ref in tensors)
        return original_destroy(value)

    cuda.torch.cuda.cudart = lambda: SimpleNamespace(cudaStreamDestroy=destroy)
    cuda.events.clear()
    segment.close()
    assert cuda.events == ["sync", "reset", ("destroy", handle)]
    segment.close()
    assert cuda.events == ["sync", "reset", ("destroy", handle)]


def test_capture_exception_resets_graph_before_destroying_stream(cuda):
    cuda.fail = True
    with pytest.raises(ValueError, match="capture failed"):
        cuda.make()
    assert "reset" in cuda.events
    assert cuda.events.index("sync") < cuda.events.index("reset")
    assert cuda.events[-1][0] == "destroy"


def test_cleanup_failure_preserves_original_capture_exception(cuda):
    cuda.fail = True
    cuda.torch.cuda.synchronize = lambda device: (_ for _ in ()).throw(
        RuntimeError("cleanup failed")
    )
    with pytest.raises(ValueError, match="capture failed"):
        cuda.make()


def test_destructor_tolerates_partial_initialization():
    segment = object.__new__(_GraphSegment)
    segment.__del__()


def test_shape_segments_share_pool_and_stream_but_shapes_are_isolated(cuda):
    from models.cua_s1.multimodal.graph_runtime import _GraphPool, _ShapeEntry

    first = _ShapeEntry(pool=_GraphPool("cuda:0"))
    second = _ShapeEntry(pool=_GraphPool("cuda:0"))
    first.blocks[0] = cuda.make(pool=first.pool)
    first.blocks[1] = cuda.make(pool=first.pool)
    second.blocks[0] = cuda.make(pool=second.pool)
    assert cuda.captures[0] is cuda.captures[1]
    assert cuda.captures[0] is not cuda.captures[2]
    assert cuda.pools[0] == cuda.pools[1] != cuda.pools[2]
    cuda.events.clear()
    first.close()
    assert cuda.events.count("reset") == 2
    assert (
        len([e for e in cuda.events if isinstance(e, tuple) and e[0] == "destroy"]) == 1
    )
    assert cuda.events[-1][0] == "destroy"
    assert second.blocks[0].graph is not None
    second.close()


def test_shape_counts_all_reserved_pool_bytes_and_external_inputs_once(cuda):
    from models.cua_s1.multimodal.graph_runtime import _GraphPool, _ShapeEntry

    entry = _ShapeEntry(pool=_GraphPool("cuda:0"))
    entry.blocks[0] = cuda.make(pool=entry.pool)
    entry.blocks[1] = cuda.make(pool=entry.pool)
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return [
            {"total_size": 1000, "allocated_size": 10},
            {"total_size": 2000, "allocated_size": 20},
        ]

    cuda.torch.cuda.memory_snapshot = snapshot
    entry.update_bytes()
    # Two independently cloned inputs and masks, each with 12 bytes of storage.
    assert entry.bytes == 3000 + 4 * 12
    assert calls == [{"mempool_id": entry.pool.pool_id, "include_traces": False}]
    entry.close()


def test_unsupported_allocator_fails_before_allocating_unaccounted_pool(cuda):
    from models.cua_s1.multimodal.graph_runtime import _GraphPool

    cuda.torch.cuda.get_allocator_backend = lambda: "cudaMallocAsync"
    with pytest.raises(RuntimeError, match="native CUDA allocator"):
        _GraphPool("cuda:0")
    assert not cuda.events
