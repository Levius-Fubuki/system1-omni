"""Frontend overhead, paired: each request goes to the worker directly and through the frontend, one right
after the other, alternating which goes first. Pairing cancels background load that shifts separate
runs, so the per-request difference is the frontend's cost.

Start a worker and the frontend first (recipe/laya/apple-silicon.md), then:

    python benchmarks/laya/frontend_overhead.py --direct http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
"""

import argparse
import http.client
import json
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def connect(url):
    parts = urlsplit(url)
    return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=120)


def quantile(sorted_values, p):
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


def call(conn, body):
    started = time.perf_counter()
    conn.request("POST", "/v1/systemone", body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    response.read()
    if response.status != 200:
        raise SystemExit(f"status {response.status}")
    return (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--direct", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="english")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("--only", nargs="*", help="bench workload ids (default: all)")
    parser.add_argument("-n", type=int, default=120, help="pairs per workload")
    parser.add_argument("--discard", type=int, default=10)
    args = parser.parse_args()

    with open(args.workloads) as f:
        workloads = [json.loads(line) for line in f if line.strip()]
    bench = [w for w in workloads if w["kind"] == "bench" and (not args.only or w["id"] in args.only)]
    direct, frontend = connect(args.direct), connect(args.frontend)

    print("| workload | body bytes | direct p50 | frontend p50 | Δ p10 | Δ p50 | Δ p90 | Δ > 10 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for w in bench:
        body = json.dumps({"model": args.model, "state": w["state"], "questions": w["questions"]}).encode()
        for _ in range(args.discard):
            call(direct, body)
            call(frontend, body)
        d_ms, f_ms, delta = [], [], []
        for i in range(args.n):
            if i % 2 == 0:
                a, b = call(direct, body), call(frontend, body)
            else:
                b, a = call(frontend, body), call(direct, body)
            d_ms.append(a)
            f_ms.append(b)
            delta.append(b - a)
        delta.sort()
        print(
            f"| {w['id']} | {len(body)} | {statistics.median(d_ms):.1f} | {statistics.median(f_ms):.1f} "
            f"| {quantile(delta, 0.1):+.1f} | {statistics.median(delta):+.1f} | {quantile(delta, 0.9):+.1f} | {sum(x > 10 for x in delta)}/{args.n} |"
        )


if __name__ == "__main__":
    main()
