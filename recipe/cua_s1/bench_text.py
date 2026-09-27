"""Send the fixed input set to a running worker, directly and through the frontend.

For each case it checks that the frontend returns the same status, content
type and body bytes as the worker, then measures warm end-to-end latency on
both paths. Warmup requests are sent first and reported separately. Requests
are sequential (concurrency 1).

    python recipe/cua_s1/bench_text.py --direct http://127.0.0.1:8000 \
        --frontend http://127.0.0.1:8080 --warmup 3 --repeat 20 --out results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Ignore http_proxy and friends: the worker and the frontend are local.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, body: bytes, token: str | None) -> tuple[int, str, bytes, float]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url + "/v1/systemone", data=body, headers=headers)
    started = time.perf_counter()
    try:
        with OPENER.open(request, timeout=120) as response:
            data = response.read()
            status, ctype = response.status, response.headers.get("content-type", "")
    except urllib.error.HTTPError as error:
        data, status, ctype = (
            error.read(),
            error.code,
            error.headers.get("content-type", ""),
        )
    return status, ctype, data, (time.perf_counter() - started) * 1000


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--direct", required=True, help="worker base URL")
    parser.add_argument(
        "--frontend", help="frontend base URL; omit to measure the worker only"
    )
    parser.add_argument(
        "--inputs", default=str(ROOT / "tests/cua_s1/data/text_inputs.json")
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--token", help="bearer token, if the worker requires one")
    parser.add_argument("--out")
    args = parser.parse_args()

    cases = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    paths = {"direct": args.direct}
    if args.frontend:
        paths["frontend"] = args.frontend
    results, mismatches = {}, 0
    for name, body in cases.items():
        raw = json.dumps(body, ensure_ascii=False).encode()
        status, ctype, direct_body, _ = post(args.direct, raw, args.token)
        row = {"status": status, "content_type": ctype}
        if status == 200:
            reply = json.loads(direct_body)
            row["answers"], row["input_tokens"] = (
                reply["answers"],
                reply["usage"]["input_tokens"],
            )
        else:
            row["body"] = direct_body.decode("utf-8", "replace")[:500]
        if args.frontend:
            f_status, f_ctype, f_body, _ = post(args.frontend, raw, args.token)
            row["frontend_identical"] = (f_status, f_ctype, f_body) == (
                status,
                ctype,
                direct_body,
            )
            mismatches += not row["frontend_identical"]
        for label, url in paths.items():
            warm = [post(url, raw, args.token)[3] for _ in range(args.warmup)]
            times = [post(url, raw, args.token)[3] for _ in range(args.repeat)]
            row[label] = {
                "warmup_ms": [round(t, 2) for t in warm],
                "p50_ms": round(statistics.median(times), 2),
                "p95_ms": round(pct(times, 0.95), 2),
                "min_ms": round(min(times), 2),
                "raw_ms": [round(t, 2) for t in times],
            }
        results[name] = row
        line = f"{name}: status {status}, tokens {row.get('input_tokens')}"
        for label in paths:
            line += (
                f", {label} p50 {row[label]['p50_ms']} ms p95 {row[label]['p95_ms']} ms"
            )
        if args.frontend:
            line += f", identical {row['frontend_identical']}"
        print(line, flush=True)
    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=1) + "\n"
        )
    if args.frontend:
        print(
            f"{len(cases) - mismatches}/{len(cases)} cases byte-identical through the frontend"
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
