"""HTTP worker for Cua-S1 4B 0.2 (`text` adapter) behind the Rust frontend.

Routes: `GET /health` and `POST /v1/systemone`. The model is loaded before the
server starts listening, and one forward pass runs at a time.

    PYTHONPATH=src python -m models.cua_s1.text.server --base <dir> --adapter <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# FastAPI reads the handler annotations at runtime, so `Request` must be a
# module-level name while `from __future__ import annotations` is in effect.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .contract import (
    ADAPTER_REVISION,
    MODEL_NAME,
    RequestError,
    answer,
    map_request,
    model_identity,
    parse_body,
)

WARMUP_REQUEST = {
    "model": MODEL_NAME,
    "state": "Dialog: 'Update installed.' Button: OK",
    "questions": {
        "warmup": {
            "type": "choice",
            "instructions": "Close the dialog.",
            "criteria": {"ok": "Click OK", "wait": "Wait"},
        }
    },
}


def build_app(
    engine: Any,
    *,
    api_key: str | None,
    max_body_bytes: int,
    max_questions: int,
    max_prompt_tokens: int,
    revision: str,
):
    app = FastAPI()
    pool = ThreadPoolExecutor(max_workers=1)
    identity = model_identity(revision)
    expected_auth = (
        f"Bearer {api_key}".encode("utf-8", "surrogateescape") if api_key else b""
    )

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"detail": message}, status_code=status)

    def authorized(request: Request) -> bool:
        if not api_key:
            return True
        supplied = request.headers.get("authorization", "").encode(
            "utf-8", "surrogateescape"
        )
        return hmac.compare_digest(supplied, expected_auth)

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "modality": "text",
            "model": identity,
            "device": engine.device,
            "dtype": engine.dtype,
        }

    def decide(mapped):
        # Tokenize every question first, so an over-long prompt is rejected
        # before any forward pass runs.
        encoded = []
        for question in mapped.questions:
            inputs = engine.encode(mapped.state, question)
            n = int(inputs["input_ids"].shape[1])
            if max_prompt_tokens and n > max_prompt_tokens:
                raise RequestError(
                    f"question {question.name!r}: prompt is {n} tokens, "
                    f"over the {max_prompt_tokens}-token limit",
                    status=413,
                )
            encoded.append((question, inputs))
        answers, prompt_tokens = {}, 0
        for question, inputs in encoded:
            scored = engine.score_encoded(inputs, len(question.keys))
            answers[question.name] = answer(question, scored.probabilities)
            prompt_tokens += scored.prompt_tokens
        return {
            "model": identity,
            "answers": answers,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if not authorized(request):
            return error(401, "invalid or missing bearer token")
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body_bytes:
            return error(413, "request body too large")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > max_body_bytes:
                return error(413, "request body too large")
        try:
            mapped = map_request(parse_body(bytes(raw)), max_questions=max_questions)
            loop = asyncio.get_running_loop()
            # Returned as a JSONResponse: FastAPI's default encoder would drop
            # every key that starts with "_sa", and question names and option
            # keys come from the request.
            return JSONResponse(await loop.run_in_executor(pool, decide, mapped))
        except RequestError as exc:
            return error(exc.status, str(exc))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return error(500, "inference failed")

    def warmup() -> None:
        """Run one decision on the worker thread through the full request path."""
        mapped = map_request(WARMUP_REQUEST)
        json.dumps(pool.submit(decide, mapped).result(), allow_nan=False)

    app.state.warmup = warmup
    return app


def main(argv: list[str] | None = None) -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default=env("CUA_S1_BASE"),
        help="local Qwen/Qwen3.5-4B directory (env CUA_S1_BASE)",
    )
    parser.add_argument(
        "--adapter",
        default=env("CUA_S1_ADAPTER"),
        help="local cua-ai/cua-s1-4b-0.2 directory (env CUA_S1_ADAPTER)",
    )
    parser.add_argument(
        "--adapter-revision",
        default=env("CUA_S1_ADAPTER_REVISION"),
        help="adapter revision to report when the download metadata is missing",
    )
    parser.add_argument("--device", default=env("CUA_S1_DEVICE", "cuda"))
    parser.add_argument(
        "--dtype",
        default=env("CUA_S1_DTYPE", "bfloat16"),
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--host", default=env("CUA_S1_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env("CUA_S1_PORT", "8000")))
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(env("CUA_S1_MAX_BODY_BYTES", str(4 << 20))),
    )
    parser.add_argument(
        "--max-questions", type=int, default=int(env("CUA_S1_MAX_QUESTIONS", "64"))
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=int(env("CUA_S1_MAX_PROMPT_TOKENS", "16384")),
        help="per question; 0 disables the check",
    )
    args = parser.parse_args(argv)
    if not args.base or not args.adapter:
        parser.error("--base and --adapter are required")

    import uvicorn

    from .adapter import downloaded_revision, text_adapter_dir
    from .engine import TextEngine

    # Fail before loading weights if this is not the text adapter.
    text_adapter_dir(args.adapter)
    detected = downloaded_revision(args.adapter)
    if detected and args.adapter_revision and detected != args.adapter_revision:
        parser.error(
            f"--adapter-revision {args.adapter_revision} does not match the "
            f"downloaded revision {detected}"
        )
    revision = detected or args.adapter_revision or ADAPTER_REVISION
    if revision != ADAPTER_REVISION:
        print(
            f"warning: adapter revision {revision} is not the pinned {ADAPTER_REVISION}",
            flush=True,
        )
    if not detected:
        print(
            "note: no download metadata under --adapter; the adapter revision is not verified",
            flush=True,
        )

    engine = TextEngine(args.base, args.adapter, args.device, args.dtype)
    print(
        f"loaded in {engine.load_seconds:.1f} s on {args.device} ({args.dtype})",
        flush=True,
    )
    app = build_app(
        engine,
        api_key=env("CUA_S1_API_KEY") or None,
        max_body_bytes=args.max_body_bytes,
        max_questions=args.max_questions,
        max_prompt_tokens=args.max_prompt_tokens,
        revision=revision,
    )
    # One decision before listening, so the first real request does not pay
    # for lazy weight loading or first-call kernel setup on the worker thread.
    started = time.perf_counter()
    app.state.warmup()
    print(f"warmed up in {time.perf_counter() - started:.1f} s", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
