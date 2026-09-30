"""Compare HTTP response status, content type and bytes through Rust and directly."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


def exchange(base, route, body=None):
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, response.headers.get("Content-Type"), response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:8080")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = sorted(args.fixtures.glob("*.json"))
    if not fixtures:
        raise ValueError("at least one valid fixture is required")
    checks = [("health", "/health", None)]
    checks += [(p.stem, "/v1/systemone", p.read_bytes()) for p in fixtures]
    checks += [
        ("invalid", "/v1/systemone", b"{}"),
        ("duplicate", "/v1/systemone", b'{"model":1,"model":2}'),
    ]
    missing = json.loads(fixtures[0].read_bytes())
    next(iter(missing["questions"].values())).pop("instructions")
    checks.append(
        ("missing-instructions", "/v1/systemone", json.dumps(missing).encode())
    )
    report = []
    for name, route, body in checks:
        direct = exchange(args.worker, route, body)
        proxied = exchange(args.frontend, route, body)
        assert direct == proxied, f"frontend changed response: {name}"
        expected_status = {
            "invalid": 422,
            "duplicate": 400,
            "missing-instructions": 422,
        }.get(name, 200)
        assert direct[0] == expected_status, f"unexpected status: {name}: {direct[0]}"
        if expected_status >= 400:
            assert set(json.loads(direct[2])) == {"detail"}, (
                f"wrong error envelope: {name}"
            )
        report.append(
            {
                "name": name,
                "status": direct[0],
                "content_type": direct[1],
                "body_sha256": hashlib.sha256(direct[2]).hexdigest(),
                "identical": True,
            }
        )
        print(f"{name}: HTTP {direct[0]}, identical response", flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
