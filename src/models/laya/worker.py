"""Laya worker: laya-serve with warmup before readiness and the actual device in /health.

laya-serve (laya 0.3.20) answers /health as soon as it binds, before any forward pass, and reports
LAYA_DEVICE as configured rather than where the model ended up. This worker reuses laya's app and
request handling unchanged and fixes both:

- It binds only after a warmup of every loaded model that covers short, long and multi-question
  requests (the last one crosses laya's fp16 autocast threshold on MPS), so a reachable worker is a
  warm one whichever model a request is routed to.
- /health reports, per loaded model and read on every call, the device, weight and autocast dtypes, the
  checkpoint and the revision its weights were downloaded from, and whether the device differs from the
  one requested. laya moves a model to the CPU on a GPU out-of-memory error and keeps serving.

Configuration is laya-serve's (LAYA_HOST, LAYA_PORT, LAYA_DEVICE, LAYA_MODELS, LAYA_API_KEY, ...) plus:

    LAYA_WORKER_MODEL          model summarised at the top of /health; english
                               also the one loaded when nothing is
                               preloaded (LAYA_PRELOAD=0)
    LAYA_REQUIRE_DEVICE        exit instead of serving on another     0
                               device than LAYA_DEVICE asked for
    LAYA_WORKER_COMPILE        off or on: torch.compile (dynamic=True)  off
                               before warmup; see compile_agent
    LAYA_WORKER_WEIGHTS        fp32 or fp16: keep the checkpoint's     fp32
                               fp16 weights instead of laya's fp32
                               upcast

Both options apply on the GPU only. If laya falls back to the CPU after a GPU out-of-memory error, the
worker runs laya's fp32 model uncompiled from then on, and /health shows the CPU.

The warmup also compiles the graphs, so the worker takes longer to become ready (35–39 s instead of
8–10 s on an M1 Pro). /health counts compiled graphs at readiness and now; `recompiled_after_ready` means
a request hit a shape class the warmup did not cover.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("laya-worker")

# (words of state, questions): each shape runs twice. Short, mid-length and near-window states, then
# 3 and 6 questions (6 is at or above laya's MPS fp16 autocast threshold of 5 rows).
_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges and refunds", "technical": "Software problems", "other": "Anything else"},
}
_SCORE = {"type": "score", "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]}
_NOUL = {"type": "noul", "instructions": "Does the customer ask for a refund?"}
WARMUP_SHAPES = [
    (10, {"q": _CHOICE}),
    (150, {"q": _CHOICE}),
    (400, {"q": _CHOICE}),
    (10, {"a": _CHOICE, "b": _SCORE, "c": _NOUL}),
    (10, {f"q{i}": q for i, q in enumerate([_CHOICE, _SCORE, _NOUL, _CHOICE, _SCORE, _NOUL])}),
]
WARMUP_REPEATS = 2


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def warmup(router: Any, model: str, shapes=WARMUP_SHAPES, repeats: int = WARMUP_REPEATS) -> dict[str, Any]:
    """Run every shape `repeats` times. Any failure propagates: a worker that cannot answer must not bind."""
    started = time.perf_counter()
    routing = None
    for words, questions in shapes:
        state = " ".join(["refund"] * words)
        for _ in range(repeats):
            result = router.predict(state, questions, model=model)
            routing = result.get("routing") or routing
    return {"warmup_ms": round((time.perf_counter() - started) * 1000, 1), "routing": routing}


def compiled_graphs() -> int:
    """Graphs torch.compile has produced in this process."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


COMPILE_MODES = {"": "off", "0": "off", "off": "off", "1": "on", "on": "on"}


WEIGHT_MODES = {"": "fp32", "fp32": "fp32", "fp16": "fp16"}


def _served(agent: Any) -> Any:
    """The module the worker puts in place of laya's model, created on first use.

    On the GPU it runs the compiled paths when there are any, otherwise laya's model. On the CPU it always
    runs laya's model in fp32: laya moves the model to the CPU when a request runs out of GPU memory, and
    there fp16 weights are slower (334 ms against 138 ms for a 68-token request) and the compiled graphs
    would first recompile (28 s measured). So after a fallback the worker behaves like plain laya.
    """
    import torch

    if getattr(agent.model, "laya_worker_wrapper", False):
        return agent.model
    eager = agent.model

    class Served(torch.nn.Module):
        laya_worker_wrapper = True

        def __init__(self):
            super().__init__()
            self.eager = eager
            self.fp16 = False
            self.paths = None  # (whole model compiled, encoder-only compiled); a tuple is not a submodule

        def forward(self, input_ids, *args, **kwargs):
            if input_ids.device.type == "cpu":
                if self.fp16:
                    self.eager.float()
                    self.fp16 = False
                return self.eager(input_ids, *args, **kwargs)
            if self.paths is None:
                return self.eager(input_ids, *args, **kwargs)
            whole, encoder_only = self.paths
            return (whole if input_ids.shape[0] == 1 else encoder_only)(input_ids, *args, **kwargs)

    agent.model = Served()
    return agent.model


def use_fp16_weights(agent: Any) -> None:
    """Keep the weights in fp16, the checkpoint's own precision, so the conversion is exact. laya 0.3.20
    upcasts them to fp32 on MPS and CPU. `act_head` stays fp32 because laya feeds it `.float()` features.
    For the GPU only: see _served for what happens on the CPU."""
    served = _served(agent)
    served.eager.half()
    act_head = getattr(served.eager, "act_head", None)
    if act_head is not None:
        act_head.float()
    served.fp16 = True


def compile_agent(agent: Any) -> None:
    """Compile the model for the batches where it pays off on MPS, sharing the same parameters.

    A batch of one row (one question) runs the whole model compiled. A batch of several rows runs only
    the encoder compiled and laya's decision head eagerly: the head is two nn.TransformerEncoderLayer
    with a key padding mask, which lose PyTorch's fused fast path when compiled and were about 50 ms
    slower on padded multi-row batches.
    """
    import copy

    import torch

    served = _served(agent)
    eager = served.eager
    whole = torch.compile(eager, dynamic=True)
    encoder_only = copy.copy(eager)  # same parameters and submodules ...
    encoder_only._modules = dict(eager._modules)  # ... except the encoder slot
    encoder_only._modules["encoder"] = torch.compile(eager.encoder, dynamic=True)
    served.paths = (whole, encoder_only)


def record_snapshot_revisions() -> dict[str, str]:
    """Record the commit each Hugging Face checkpoint is loaded from, keyed by repo id.

    laya calls huggingface_hub.snapshot_download while loading and keeps only the repo id; the
    returned path (.../snapshots/<commit>/...) is the only place the loaded revision appears. Call this
    before the router loads anything. A checkpoint loaded from a local path records nothing.
    """
    import huggingface_hub

    revisions: dict[str, str] = {}
    original = huggingface_hub.snapshot_download

    def recording(repo_id, *args, **kwargs):
        path = original(repo_id, *args, **kwargs)
        parts = Path(path).parts
        if "snapshots" in parts[:-1]:
            revisions[repo_id] = parts[parts.index("snapshots") + 1]
        return path

    huggingface_hub.snapshot_download = recording
    return revisions


def describe(
    agent: Any, requested: str | None, routing: dict[str, Any] | None, revisions: dict[str, str] | None = None
) -> dict[str, Any]:
    """What /health reports about one loaded agent."""
    device = str(getattr(agent, "device", "unknown"))
    model = getattr(agent, "model", None)
    try:
        weights = str(next(model.parameters()).dtype) if model is not None else None
    except (AttributeError, StopIteration, TypeError):
        weights = None
    repo = (routing or {}).get("repo")
    requested_type = requested.split(":")[0] if requested else None
    return {
        "device": device,
        "requested_device": requested or "auto",
        "device_mismatch": bool(requested_type) and device.split(":")[0] != requested_type,
        "weights_dtype": weights,
        "autocast_dtype": str(getattr(agent, "dtype", None)),
        "mps_amp_min_rows": getattr(agent, "mps_amp_min_rows", None),
        "checkpoint": repo,
        "revision": (revisions or {}).get(repo),
    }


def create_worker_app(
    router: Any,
    model: str,
    requested: str | None,
    require_device: bool = False,
    compile: str = "off",
    graph_counter=compiled_graphs,
    revisions: dict[str, str] | None = None,
    weights: str = "fp32",
):
    """Optionally compile, warm up every loaded model, then return laya's app with /health replaced.
    `model` is the one summarised at the top of /health, and the one loaded if nothing is preloaded.
    Raises if compiling or warmup fails."""
    from laya.serve import create_app

    if compile not in ("off", "on"):
        raise ValueError(f"compile must be off or on, not {compile!r}")
    if weights not in ("fp32", "fp16"):
        raise ValueError(f"weights must be fp32 or fp16, not {weights!r}")
    names = list(router.loaded) or [model]  # never load a model the worker was not asked to serve
    for name in names:
        agent = router.load(name)
        if str(getattr(agent, "device", "")).startswith("cpu"):
            if weights == "fp16" or compile != "off":
                log.warning(
                    "%s is on the CPU: LAYA_WORKER_WEIGHTS=fp16 and LAYA_WORKER_COMPILE=on apply on the GPU only", name
                )
            continue
        if weights == "fp16":
            use_fp16_weights(agent)
        if compile != "off":
            compile_agent(agent)
    warmed = {name: warmup(router, name) for name in names}
    agents = {name: router.load(name) for name in names}
    primary = model if model in agents else names[0]

    def current() -> dict[str, Any]:
        """Describe the agents as they are now: laya moves a model to the CPU when a request runs out of
        GPU memory, so the device at startup is not necessarily the device serving the next request."""
        models = {
            name: {
                **describe(agent, requested, warmed[name]["routing"], revisions),
                "warmup_ms": warmed[name]["warmup_ms"],
            }
            for name, agent in agents.items()
        }
        return {
            **models[primary],
            "device_mismatch": any(m["device_mismatch"] for m in models.values()),
            "warmup_ms": round(sum(m["warmup_ms"] for m in models.values()), 1),
            "models": models,
        }

    info = current()
    models = info["models"]
    graphs_at_ready = graph_counter() if compile != "off" else None
    if info["device_mismatch"]:
        wrong = ", ".join(f"{n} is on {m['device']}" for n, m in models.items() if m["device_mismatch"])
        message = f"asked for {info['requested_device']}, {wrong}"
        if require_device:
            raise RuntimeError(message)
        log.warning(message)

    app = create_app(router)
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        compiled = {"mode": compile}
        if compile != "off":
            now = graph_counter()
            compiled.update(
                graphs_at_ready=graphs_at_ready, graphs_now=now, recompiled_after_ready=now > graphs_at_ready
            )
        return {"status": "ok", "ready": True, "loaded": router.loaded, **current(), "compile": compiled}

    return app


def main() -> None:
    import uvicorn
    from laya.serve import _resolve_port, build_router

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    model = os.environ.get("LAYA_WORKER_MODEL", "english")
    requested = os.environ.get("LAYA_DEVICE") or None
    revisions = record_snapshot_revisions()
    try:
        app = create_worker_app(
            build_router(),
            model,
            requested,
            require_device=_env_bool("LAYA_REQUIRE_DEVICE"),
            compile=COMPILE_MODES.get(os.environ.get("LAYA_WORKER_COMPILE", "").strip().lower(), "invalid"),
            revisions=revisions,
            weights=WEIGHT_MODES.get(os.environ.get("LAYA_WORKER_WEIGHTS", "").strip().lower(), "invalid"),
        )
    except Exception as exc:  # noqa: BLE001 -- any failure before binding means not ready, ever
        sys.exit(f"laya-worker: not starting: {exc}")
    uvicorn.run(
        app,
        host=os.environ.get("LAYA_HOST", "127.0.0.1"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
