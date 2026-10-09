"""Frozen-plan HTTP matrix client; Python 3.11+, standard library only."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit, urlunsplit


def decode_json(data):
    def reject_constant(value):
        raise ValueError(f"non-JSON numeric constant: {value}")
    return json.loads(data, parse_constant=reject_constant)


def case_variants(case):
    return case["variants"] if "variants" in case else [case]


def validate_plan(plan):
    try:
        json.dumps(plan, allow_nan=False)
    except (ValueError, RecursionError) as exc:
        raise ValueError("plan must contain only finite JSON values within serialization limits") from exc
    for key in ("endpoint", "health_endpoint"):
        if key not in plan and key == "health_endpoint":
            continue
        parts = urlsplit(plan[key])
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise ValueError(f"{key} must be an HTTP(S) URL without credentials or fragment")
        _ = parts.port
    cases = plan["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    names = set()
    for case in cases:
        name = case["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("case names must be nonempty and unique")
        names.add(name)
        variants = case_variants(case)
        if "variants" in case and any(key in case for key in ("request", "expected_response", "expected_response_text")):
            raise ValueError("variants replaces the case's request and expectation")
        if not isinstance(variants, list) or not variants:
            raise ValueError("variants must be a nonempty list")
        variant_names = set()
        for variant in variants:
            variant_name = variant["name"]
            if not isinstance(variant_name, str) or not variant_name or variant_name in variant_names:
                raise ValueError("variant names must be nonempty and unique within a case")
            variant_names.add(variant_name)
            if not isinstance(variant["request"], dict):
                raise ValueError("request must be a JSON object")
            if ("expected_response" in variant) == ("expected_response_text" in variant):
                raise ValueError("each variant needs exactly one expected response")
            if "expected_response_text" in variant and not isinstance(variant["expected_response_text"], str):
                raise ValueError("expected_response_text must be a string")
            if "expected_response_text" in variant:
                try:
                    variant["expected_response_text"].encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError("expected_response_text must be valid UTF-8 text") from exc
    count = plan["requests_per_case"]
    concurrency = plan["concurrency"]
    if type(count) is not int or count < 1 or not isinstance(concurrency, list) or not concurrency:
        raise ValueError("requests_per_case and concurrency must be positive")
    if any(type(c) is not int or c < 1 or c > count for c in concurrency) or len(set(concurrency)) != len(concurrency):
        raise ValueError("concurrency values must be unique positive integers <= requests_per_case")
    if type(plan["repetitions"]) is not int or plan["repetitions"] != 2:
        raise ValueError("repetitions must be exactly 2")
    if type(plan["warmup_per_case"]) is not int or plan["warmup_per_case"] != 2:
        raise ValueError("warmup_per_case must be exactly 2")
    timeout = plan["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not isinstance(plan["metadata"], dict):
        raise ValueError("metadata must be a pinned JSON object")
    for name in (
        "gpu", "gpu_ids", "driver", "cuda", "precision", "model_revision",
        "runtime_revision", "cache_policy", "cuda_evidence", "reservation",
    ):
        value = plan["metadata"].get(name)
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"metadata requires {name}")


def exact_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_json(a, b) for a, b in zip(left, right))
    return left == right


def exchange(endpoint, timeout, case=None, request_headers=None):
    """One connection, no proxy, redirect, pooling or retry; retain raw bytes."""
    started = time.perf_counter()
    parts = urlsplit(endpoint)
    path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
    connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parts.hostname, parts.port, timeout=timeout)
    expired = threading.Event()
    timer = None
    payload = bytearray()
    status = None
    headers = {}
    error = None
    decisions = 0
    try:
        connection.connect()
        remaining = timeout - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("connection exceeded deadline")
        transport = connection.sock
        transport.settimeout(remaining)

        def expire():
            expired.set()
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        body = json.dumps(case["request"], allow_nan=False).encode() if case else None
        connection.request("POST" if case else "GET", path, body=body,
                           headers={**(request_headers or {}), **({"Content-Type": "application/json"} if case else {})})
        response = connection.getresponse()
        status = response.status
        headers = dict(response.getheaders())
        while chunk := response.read1(65536):
            payload.extend(chunk)
        if expired.is_set():
            raise TimeoutError("response exceeded deadline")
        if response.length not in (None, 0):
            raise http.client.IncompleteRead(bytes(payload), response.length)
        if status != 200:
            error = {"kind": "http", "message": f"HTTP {status}"}
        elif case:
            try:
                decoded = decode_json(bytes(payload))
            except (ValueError, UnicodeError, RecursionError):
                decoded = None
            if "expected_response_text" in case:
                matches = bytes(payload) == case["expected_response_text"].encode("utf-8")
            else:
                matches = exact_json(decoded, case["expected_response"])
                # Invalid JSON must fail even when the oracle is JSON null.
                if decoded is None:
                    try:
                        decode_json(bytes(payload))
                    except (ValueError, UnicodeError, RecursionError):
                        matches = False
            if not matches:
                error = {"kind": "correctness", "message": "response differs from frozen expectation"}
            elif isinstance(decoded, dict) and isinstance(decoded.get("answers"), (dict, list)):
                decisions = len(decoded["answers"])
    except RecursionError:
        error = {"kind": "correctness", "message": "response validation exceeds recursion limit"}
    except (OSError, http.client.HTTPException) as exc:
        kind = "timeout" if expired.is_set() or isinstance(exc, TimeoutError) else "transport"
        error = {"kind": kind, "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if timer:
            timer.cancel()
        connection.close()
    return {
        "endpoint": endpoint, "status": status, "headers": headers,
        "raw_response_base64": base64.b64encode(payload).decode("ascii"),
        "raw_response_text": bytes(payload).decode("utf-8", errors="replace"),
        "latency_seconds": time.perf_counter() - started,
        "successful_decisions": decisions, "error": error,
    }


def round_summary(records, elapsed, case, concurrency, repetition, planned):
    successful = [r for r in records if r["error"] is None]
    latencies = sorted(r["latency_seconds"] for r in successful)
    quantiles = {f"p{p}": latencies[math.ceil(p / 100 * len(latencies)) - 1] if latencies else None for p in (50, 95)}
    decisions = sum(r["successful_decisions"] for r in successful)
    return {
        "case": case, "concurrency": concurrency, "repetition": repetition,
        "planned_requests": planned, "attempted_requests": len(records),
        "elapsed_seconds": elapsed, "successful_requests": len(successful),
        "successful_decisions": decisions,
        "successful_requests_per_second": len(successful) / elapsed,
        "successful_decisions_per_second": decisions / elapsed,
        "successful_latency_seconds": quantiles,
        "failures": dict(Counter(r["error"]["kind"] for r in records if r["error"])),
        "failure_latency_seconds": [r["latency_seconds"] for r in records if r["error"]],
        "complete": len(records) == planned and len(successful) == planned,
    }


def run(plan_path, output):
    snapshot = plan_path.read_bytes()
    plan = decode_json(snapshot)
    validate_plan(plan)
    request_headers = {}
    if token := os.environ.get("OMNI_JEV_TEST_TOKEN"):
        if any(ord(c) < 32 or ord(c) > 126 for c in token):
            raise ValueError("invalid OMNI_JEV_TEST_TOKEN: expected printable ASCII")
        request_headers["Authorization"] = f"Bearer {token}"
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    (output / "plan.json").write_bytes(snapshot)
    save("config.json", {
        "plan_sha256": hashlib.sha256(snapshot).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan": plan, "timing": "client wall time including response validation",
        "retries": 0, "redirects": False, "environment_proxy": False,
    })
    parts = urlsplit(plan["endpoint"])
    health_endpoint = plan.get("health_endpoint", urlunsplit((parts.scheme, parts.netloc, "/health", "", "")))
    health = exchange(health_endpoint, plan["timeout_seconds"], request_headers=request_headers)
    save("health.json", health)
    all_records = []
    rounds = []
    stopped = threading.Event()
    if health["error"]:
        stopped.set()
    lock = threading.Lock()
    with (output / "responses.jsonl").open("w") as sink:
        def request(case, phase, index, concurrency=1, repetition=None, variant_index=None):
            variants = case_variants(case)
            if variant_index is None:
                variant_index = index % len(variants)
            variant = variants[variant_index]
            record = exchange(plan["endpoint"], plan["timeout_seconds"], variant, request_headers)
            record.update(case=case["name"], phase=phase, index=index,
                          concurrency=concurrency, repetition=repetition,
                          variant=variant["name"], variant_index=variant_index)
            with lock:
                if record["error"]:
                    stopped.set()
                all_records.append(record)
                sink.write(json.dumps(record, allow_nan=False) + "\n")
                sink.flush()
            return record

        for case in plan["cases"]:
            if stopped.is_set():
                break
            for variant_index in range(len(case_variants(case))):
                request(case, "readiness", 0, variant_index=variant_index)
                if stopped.is_set():
                    break
        for case in plan["cases"]:
            for variant_index in range(len(case_variants(case))):
                for index in range(plan["warmup_per_case"]):
                    if stopped.is_set():
                        break
                    request(case, "warmup", index, variant_index=variant_index)

        def wave(case, phase, concurrency, count, repetition=None):
            next_index = 0
            completed = []

            def worker():
                nonlocal next_index
                while True:
                    with lock:
                        if stopped.is_set() or next_index == count:
                            return
                        index = next_index
                        next_index += 1
                    record = request(case, phase, index, concurrency, repetition)
                    with lock:
                        completed.append(record)

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(worker) for _ in range(concurrency)]
                for future in futures:
                    future.result()
            return completed, time.perf_counter() - start

        for case in plan["cases"]:
            for concurrency in plan["concurrency"]:
                if stopped.is_set():
                    break
                wave(case, "feasibility", concurrency, min(concurrency, plan["requests_per_case"]))
                for repetition in range(1, plan["repetitions"] + 1):
                    if stopped.is_set():
                        break
                    records, elapsed = wave(case, "measured", concurrency, plan["requests_per_case"], repetition)
                    rounds.append(round_summary(records, elapsed, case["name"], concurrency, repetition, plan["requests_per_case"]))
    failures = Counter(r["error"]["kind"] for r in [health, *all_records] if r["error"])
    summary = {
        "complete": not bool(failures), "failures": dict(failures),
        "inference_requests": len(all_records), "health_requests": 1,
        "rounds": rounds,
        "excluded_phases": ["health", "readiness", "warmup", "feasibility"],
    }
    save("summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args.plan, args.output)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
