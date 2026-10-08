"""Sequential c1 replay through the same Rust frontend, preserving every response.

One full-corpus feasibility pass precedes two measured repetitions. Model
loading and readiness are recorded separately by the server. Run reference and
native deployments sequentially, with exclusive GPU ownership.
Client latency starts after request JSON serialization and ends after reading
and parsing the complete response JSON, for successful and failed HTTP status
codes alike. Evidence writes remain outside that interval.
The private HTTP opener disables redirects and environment proxies so each
record describes the exact POST endpoint supplied by the caller.
"""
import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from protocol import FROZEN_CORPUS_SHA256, load_corpus


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def direct_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def run(url, cases, out, repetitions=2, timeout=600, corpus_sha256=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    url = url.rstrip("/")
    endpoint = url if url.endswith("/v1/systemone") else url + "/v1/systemone"
    opener = direct_opener()
    controls = {"concurrency": 1, "measured_repetitions": repetitions, "feasibility_repetitions": 1,
                "model": "JEMM", "endpoint": endpoint, "timeout_seconds": timeout,
                "corpus_sha256": corpus_sha256, "case_order": [case["id"] for case in cases], "cases": cases,
                "follow_redirects": False, "environment_proxies": False}
    (out / "controls.json").write_text(json.dumps(controls, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if corpus_sha256:
        (out / "corpus.sha256").write_text(corpus_sha256 + "\n")
    samples = []
    for phase, reps in (("feasibility", 1), ("measured", repetitions)):
        for repetition in range(reps):
            for index, case in enumerate(cases):
                body = {**case["request"], "model": "JEMM"}
                raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                request = urllib.request.Request(endpoint, data=raw, headers={"Content-Type": "application/json"}, method="POST")
                started = time.perf_counter()
                try:
                    with opener.open(request, timeout=timeout) as response:
                        status, returned = response.status, response.read()
                except urllib.error.HTTPError as exc:
                    status, returned = exc.code, exc.read()
                except Exception as exc:
                    record = {"id": case["id"], "repetition": repetition, "index": index, "phase": phase,
                              "modality": case["modality"], "elapsed_ms": (time.perf_counter() - started) * 1000,
                              "status": None, "error": repr(exc), "request_sha256": hashlib.sha256(raw).hexdigest()}
                    append_record(out / (phase + ".jsonl"), record)
                    raise
                try:
                    parsed = json.loads(returned)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    parsed = None
                elapsed = (time.perf_counter() - started) * 1000
                record = {"id": case["id"], "repetition": repetition, "index": index, "phase": phase,
                          "modality": case["modality"], "elapsed_ms": elapsed, "status": status,
                          "response": parsed, "response_raw": returned.decode("utf-8", errors="replace"),
                          "request_sha256": hashlib.sha256(raw).hexdigest()}
                append_record(out / (phase + ".jsonl"), record)
                print(f"{phase} {repetition + 1}/{reps} {case['id']} status={status} latency_ms={elapsed:.3f}", flush=True)
                if status != 200 or not isinstance(parsed, dict) or "answers" not in parsed:
                    raise RuntimeError(f"case {case['id']} failed with HTTP {status}; raw response preserved")
                if phase == "measured":
                    samples.append(record)
    summary = {"concurrency": 1, "repetitions": repetitions, "case_order": [c["id"] for c in cases],
               "endpoint": endpoint, "feasibility_repetitions": 1, "latency": {}}
    for modality in ("text", "image"):
        values = [sample["elapsed_ms"] for sample in samples if sample["modality"] == modality]
        if values:
            summary["latency"][modality] = {"samples": len(values), "median_ms": statistics.median(values),
                                              "min_ms": min(values), "max_ms": max(values), "mean_ms": statistics.mean(values)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Rust frontend URL")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--expected-corpus-sha256", default=FROZEN_CORPUS_SHA256, help="only override for a separately predeclared frozen workload")
    args = parser.parse_args()
    cases, corpus_hash = load_corpus(args.corpus, args.expected_corpus_sha256)
    run(args.url, cases, args.out, timeout=args.timeout, corpus_sha256=corpus_hash)


if __name__ == "__main__":
    main()
