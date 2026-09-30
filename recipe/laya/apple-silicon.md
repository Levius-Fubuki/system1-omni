# Laya on Apple Silicon

This recipe serves Laya on the GPU of an Apple Silicon Mac (PyTorch MPS) with the Laya worker
from [`src/models/laya/`](../../src/models/laya/), puts the Rust frontend in front of it and runs
the benchmark suite. The [Laya text worker](README.md) recipe covers the plain CPU setup.

Validated on an M1 Pro (16 GB, 16-core GPU), macOS 26.1, Python 3.12, `laya[serve]==0.3.20`,
torch 2.14.0 and the `english` checkpoint (`convaiinnovations/laya` at `55cf4c4`), and by another
contributor on an M5 (10-core GPU, 32 GB, macOS 26.5.2). Other M-series Macs have not been tested.

Run all commands from the repository root.

## Install

Use Python 3.12. If `python3.12` is not on your `PATH` and you have uv, replace the first command below
with `uv venv --python 3.12 --seed .venv` (`--seed` puts pip in the environment).

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'laya[serve]==0.3.20' pytest httpx2
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

The last command must print `True`. The standard macOS arm64 wheel of torch includes MPS. `pytest` and
`httpx2` are only for the tests: Starlette's `TestClient` needs `httpx2` (or, deprecated, `httpx`).

## Start the worker

```sh
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=mps LAYA_MODELS=english \
LAYA_REQUIRE_DEVICE=1 \
  .venv/bin/python src/models/laya/worker.py
```

First startup downloads the checkpoint (about 850 MB). The worker loads the model, runs a warmup
over short, long and multi-question requests, and only then listens on port 8000, so the first
request it accepts is already warm: on an M1 Pro the first request after ready took 70–81 ms, against
0.7–1.1 s from plain laya-serve. `LAYA_REQUIRE_DEVICE=1` makes it exit instead of silently serving on the
CPU when the model cannot be placed on MPS; without it the worker logs a warning and serves from the CPU.

Check what it is running on:

```sh
curl -s http://127.0.0.1:8000/health
```

`device` must be `mps` and `device_mismatch` `false`. The response also names the checkpoint and
revision, the weight dtype (`torch.float32`; Laya upcasts the fp16 checkpoint on MPS), the autocast
dtype Laya uses for requests with at least `mps_amp_min_rows` questions, and the warmup time. The device
and dtypes are read on every call: if a request runs out of GPU memory, Laya moves the model to the CPU
and keeps serving, and `/health` then shows `device: cpu` and `device_mismatch: true`.

### Faster: compile and fp16 weights

```sh
LAYA_WORKER_COMPILE=on LAYA_WORKER_WEIGHTS=fp16 LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=mps \
LAYA_MODELS=english LAYA_REQUIRE_DEVICE=1 \
  .venv/bin/python src/models/laya/worker.py
```

`LAYA_WORKER_COMPILE=on` compiles the model during warmup: one-question requests run the whole model
compiled, requests with several questions run the encoder compiled and Laya's decision head as it is.
`LAYA_WORKER_WEIGHTS=fp16` keeps the checkpoint's fp16 weights instead of Laya's fp32 upcast on MPS.

On the M1 Pro, with both workers running and every request sent to each back to back, the two options
together lowered warm p50 against the worker without them by 37–38% for a 68-token one-question
request (about 57 → 35 ms in those runs), 17–20% at 198–484 tokens, 14% for three questions and 18% for
six. Answers stayed within 0.0031 of the fp32 worker's. A worker running on its own uses about 3 GB
with the options instead of 4.2 GB (2.8 GB against 3.5 GB in those paired runs, where the two workers
shared the machine). The price is startup: the worker became ready after 35–39 s instead of 8–10 s, and
its first request after that took 62–78 ms.

On an M5 the same paired comparison gave median ratios of 0.51–0.53 for one-question requests at
47–68 tokens, 0.30–0.33 at 198–484 tokens, 0.37 for three questions and 0.60 for six, most of it from
the fp16 weights, which on that GPU speed up every input even without compile. There the worker was
ready after 19 s instead of 3 s; its first request took 21–36 ms in 21 of 23 fresh starts and 327 and
409 ms in the other two, not yet explained (132–143 ms from plain laya-serve).

`/health` reports under `compile` how many graphs existed when the worker became ready and how many
exist now; `recompiled_after_ready: true` means a request shape was not covered by the warmup.

## Start the frontend

In another terminal:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

## Send a request

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

The frontend forwards the worker's response unchanged; `compare_with_backend.py` from the
[Laya text worker](README.md#compare-responses) recipe checks that against this setup as well.

## Test

```sh
.venv/bin/python -m pytest src/models/laya/tests                    # unit tests, no model
LAYA_CONTRACT=1 .venv/bin/python -m pytest src/models/laya/tests    # plus contract tests against a CPU worker
```

## Benchmark

Stop the worker and frontend first; the benchmark starts its own. The scripts are listed in
[`bench/`](bench/README.md). A first pass that checks everything runs:

```sh
.venv/bin/python recipe/laya/bench/check_workloads.py
.venv/bin/python recipe/laya/bench/bench_inproc.py --device mps --config C2 --run feasibility
.venv/bin/python recipe/laya/bench/bench_http.py --config C3 --run feasibility --spawn .venv/bin/laya-serve
.venv/bin/python recipe/laya/bench/bench_http.py --config C4 --run feasibility \
  --url http://127.0.0.1:8080 --frontend target/release/omni-jev --spawn .venv/bin/laya-serve
.venv/bin/python recipe/laya/bench/report.py recipe/laya/bench/results/*_feasibility.jsonl
```

Runs labelled anything other than `feasibility` refuse to start on battery power or when the
1-minute load average is above 2, so close other heavy applications and plug the Mac in first.

## Troubleshooting

- `device_mismatch: true` at startup, or the worker exits with `asked for mps, english is on cpu`: MPS
  is not available to this Python. Check the `torch.backends.mps.is_available()` line above (an x86_64
  Python under Rosetta, for example, has no MPS).
- `device_mismatch: true` on a worker that started on MPS: Laya fell back to the CPU after a GPU
  out-of-memory error. Free memory and restart the worker.
- The worker process uses about 4 GB, or 3 GB with fp16 weights (Activity Monitor's Memory column,
  which counts MPS allocations). On a 16 GB Mac, close other large applications before benchmarking.
- `Address already in use`: another worker or frontend still holds port 8000 or 8080.
