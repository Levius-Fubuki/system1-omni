# Laya on Apple Silicon

This recipe serves Laya on the GPU of an Apple Silicon Mac (PyTorch MPS) with the Laya worker
from [`src/models/laya/`](../../src/models/laya/), puts the Rust frontend in front of it and runs
the benchmark suite. The [Laya text worker](README.md) recipe covers the plain CPU setup.

Validated on an M1 Pro (16 GB, 16-core GPU), macOS 26.1, Python 3.12, `laya[serve]==0.3.20`,
torch 2.14.0 and the `english` checkpoint (`convaiinnovations/laya` at `55cf4c4`), and by another
contributor on an M5 (10-core GPU, 32 GB, macOS 26.5.2). Other M-series Macs have not been tested.

Run all commands from the repository root.

## Install

Use Python 3.12. If `python3.12` is not on your `PATH`, install it with `brew install python@3.12`
or `uv python install 3.12`.

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
request it accepts is already warm: on an M1 Pro the first request after ready took 62–81 ms, against
0.7–1.1 s from plain laya-serve. On an M5, 21 of 23 fresh starts gave 21–36 ms and two gave 327 and
409 ms, not yet explained. `LAYA_REQUIRE_DEVICE=1` makes it exit instead of silently serving on the CPU
when the model cannot be placed on MPS.

Check what it is running on:

```sh
curl -s http://127.0.0.1:8000/health
```

`device` must be `mps` and `device_mismatch` `false`. The response also names the checkpoint and
revision, the weight dtype (`torch.float32`; Laya upcasts the fp16 checkpoint on MPS), the autocast
dtype Laya uses for requests with at least `mps_amp_min_rows` questions, and the warmup time.

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
six, and cut the worker's memory from 3.5 GB to 2.8 GB. Answers stayed within 0.0031 of the fp32
worker's. The price is startup: the worker becomes ready after 20–30 s instead of about 8 s.

On an M5 the same paired comparison gave median ratios of 0.51–0.53 for one-question requests at
47–68 tokens, 0.30–0.33 at 198–484 tokens, 0.37 for three questions and 0.60 for six, most of it from
the fp16 weights, which on that GPU speed up every input even without compile.

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
.venv/bin/python recipe/laya/bench/report.py recipe/laya/bench/results/*.jsonl
```

Runs labelled anything other than `feasibility` refuse to start on battery power or when the
1-minute load average is above 2, so close other heavy applications and plug the Mac in first.

## Troubleshooting

- `device_mismatch: true`, or the worker exits with `asked for mps, english is on cpu`: MPS is not
  available to this Python. Check the `torch.backends.mps.is_available()` line above; an x86_64
  Python running under Rosetta cannot use MPS.
- The worker process uses about 4 GB (Activity Monitor's Memory column, which counts MPS
  allocations). On a 16 GB Mac, close other large applications before benchmarking.
- `Address already in use`: another worker or frontend still holds port 8000 or 8080.
