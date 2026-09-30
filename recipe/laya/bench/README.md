# Laya benchmark scripts

Scripts behind the numbers in the [Apple Silicon recipe](../apple-silicon.md). Each run writes raw
JSONL to `results/`; `report.py` and `parity.py` build the tables from it.

| file | purpose |
| --- | --- |
| `workloads.src.py` → `workloads.jsonl` | fixed inputs: W1–W6 timed, P* parity only |
| `check_workloads.py` | token count of each input with Laya's tokenizer |
| `bench_inproc.py` | Laya in-process: load, warmup, first request, warm latency, memory |
| `bench_http.py` | a `/v1/systemone` worker, optionally behind the frontend: time to ready, first request, warm latency, throughput |
| `paired.py` | two worker configurations alive at once, each request sent to both back to back; median ratio with a bootstrap interval |
| `frontend_overhead.py` | frontend cost, each request sent directly and through the frontend back to back |
| `profile_mps.py` | where a request's time goes on MPS |
| `parity.py` | answers of every run against a reference run |
| `report.py` | tables from the JSONL |
| `env.py` | versions, checkpoint, hardware and load recorded with each run |

## Run

From the repository root, in the environment of the recipe:

```sh
python recipe/laya/bench/check_workloads.py
python recipe/laya/bench/bench_inproc.py --device cpu --config C1 --run m1
python recipe/laya/bench/bench_inproc.py --device mps --config C2 --run m1
python recipe/laya/bench/bench_http.py --config C3 --run m1 --spawn .venv/bin/laya-serve
python recipe/laya/bench/bench_http.py --config C4 --run m1 --url http://127.0.0.1:8080 \
  --frontend target/release/omni-jev --spawn .venv/bin/laya-serve
LAYA_WORKER_COMPILE=off python recipe/laya/bench/bench_http.py --config C3w --run m1 \
  --spawn .venv/bin/python src/models/laya/worker.py
LAYA_WORKER_COMPILE=single python recipe/laya/bench/bench_http.py --config C3s --run m1 \
  --spawn .venv/bin/python src/models/laya/worker.py
python recipe/laya/bench/report.py recipe/laya/bench/results/*_m[0-9].jsonl
python recipe/laya/bench/parity.py recipe/laya/bench/results/*_m[0-9].jsonl --ref C1
```

Repeat with `--run m2` for a second measured run. Runs refuse to start on battery power or above a
1-minute load average of `--max-load` (default 2) unless labelled `--run feasibility`. Memory is the
process's physical footprint, which on Apple Silicon includes MPS allocations.

Two worker configurations can also be compared request by request, which holds up under background
load better than separate runs:

```sh
python recipe/laya/bench/paired.py --run p1 --a "LAYA_WORKER_COMPILE=single" \
  --b "LAYA_WORKER_COMPILE=single LAYA_WORKER_WEIGHTS=fp16"
python recipe/laya/bench/paired.py --summarize recipe/laya/bench/results/paired_p1.jsonl
```

## Results

`results/` holds the reports from the measured runs on an M1 Pro: `measured-report.md`,
`measured-parity.md`, `frontend_overhead_m1.md` and `paired-fp16.md`. The raw JSONL is published as
release assets:

```sh
curl -LO https://github.com/cacheline999/system1-omni/releases/download/laya-mps-results-2026-09-28/laya-mps-results-2026-09-28.tar.gz
shasum -a 256 laya-mps-results-2026-09-28.tar.gz   # 611ed30707ac8c98875b5aa5382360b5a7d760da166d61c626eb07ebe1ee6404
tar xzf laya-mps-results-2026-09-28.tar.gz -C recipe/laya/bench/results
python recipe/laya/bench/report.py recipe/laya/bench/results/*_m[0-9].jsonl
```

The paired fp16 runs (`paired_e4a.jsonl`, `paired_e4b.jsonl`) are in
`laya-mps-paired-fp16-2026-09-30.tar.gz` on the same release (sha256 `cc0d6f5bda6e3e0ee1f40c6966f84429e902a2f585b8e1ee33658a9be139326e`); rebuild the summary with
`paired.py --summarize`.
