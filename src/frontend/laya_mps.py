"""HTTP worker for Laya on Apple Silicon (PyTorch MPS) and CPU: `GET /health` and `POST /v1/systemone`.

PYTHONPATH=src python -m frontend.laya_mps --device mps --model english [--compile] [--weights fp16]

It is laya-serve (`laya[serve]==0.3.20`) with its request handling unchanged and three changes:

- It binds only after every loaded model has run a warmup over short, long and multi-question requests,
  so a reachable worker is a warm one. laya-serve answers /health before any forward pass.
- /health describes the loaded models as they are now: device, weight and autocast dtypes, checkpoint
  and the revision the weights were loaded from, and `device_mismatch` when a model is not on the
  requested device. laya-serve reports the configured device, and laya moves a model to the CPU on a
  GPU out-of-memory error and keeps serving. `--require-device` exits at startup instead of serving
  from another device.
- `--compile` and `--weights fp16` make the GPU path faster (models/laya/optimize.py). Both apply on
  the GPU only; on the CPU, including after a fallback, the worker runs laya's fp32 model uncompiled.

laya-serve's own environment variables still apply, notably LAYA_API_KEY for bearer authentication.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from models.laya import engine, optimize

log = logging.getLogger("laya-worker")


def build_app(
    router: Any,
    model: str,
    requested: str | None,
    *,
    require_device: bool = False,
    compile: bool = False,
    fp16: bool = False,
    graph_counter=optimize.compiled_graphs,
    revisions: dict[str, str] | None = None,
):
    """Apply the options, warm up every loaded model, then return laya's app with /health replaced.
    `model` is the one summarised at the top of /health, and the one loaded if nothing is preloaded.
    Raises if an option or the warmup fails, so the caller never binds a worker that cannot answer."""
    from laya.serve import create_app

    names = list(router.loaded) or [model]  # never load a model the worker was not asked to serve
    if fp16 or compile:
        for name in names:
            if not optimize.apply(router.load(name), fp16=fp16, compile=compile):
                log.warning("%s is on the CPU: --compile and --weights fp16 apply on the GPU only", name)
    warmed = {name: engine.warmup(router, name) for name in names}
    agents = {name: router.load(name) for name in names}
    primary = model if model in agents else names[0]

    def current() -> dict[str, Any]:
        """The agents as they are now, not as they were at startup (see the module docstring)."""
        models = {
            name: {
                **engine.describe(agent, requested, warmed[name]["routing"], revisions),
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
    graphs_at_ready = graph_counter() if compile else None
    if info["device_mismatch"]:
        wrong = ", ".join(f"{n} is on {m['device']}" for n, m in info["models"].items() if m["device_mismatch"])
        message = f"asked for {info['requested_device']}, {wrong}"
        if require_device:
            raise RuntimeError(message)
        log.warning(message)

    app = create_app(router)
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        compiled: dict[str, Any] = {"enabled": compile}
        if compile:
            now = graph_counter()
            compiled.update(
                graphs_at_ready=graphs_at_ready, graphs_now=now, recompiled_after_ready=now > graphs_at_ready
            )
        return {"status": "ok", "ready": True, "loaded": router.loaded, **current(), "compile": compiled}

    return app


def make_router(device: str | None, model: str) -> Any:
    """laya's Router with one checkpoint preloaded."""
    from laya.router import Router

    router = Router(device=device)
    router.preload([model])
    return router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="torch device for laya: mps, cpu (default: laya's choice)")
    parser.add_argument("--model", default="english", help="laya checkpoint to serve: english, multilingual, ...")
    parser.add_argument("--compile", action="store_true", help="torch.compile the GPU path during warmup")
    parser.add_argument("--weights", default="fp32", choices=["fp32", "fp16"], help="weight precision on the GPU")
    parser.add_argument("--require-device", action="store_true", help="exit if a model is not on --device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    revisions = engine.record_snapshot_revisions()
    try:
        app = build_app(
            make_router(args.device, args.model),
            args.model,
            args.device,
            require_device=args.require_device,
            compile=args.compile,
            fp16=args.weights == "fp16",
            revisions=revisions,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure before binding means not ready, ever
        sys.exit(f"laya-worker: not starting: {exc}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
