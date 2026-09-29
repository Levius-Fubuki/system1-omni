"""Send every corpus body to the Python and the native worker and compare the answers.

    python recipe/cua_s1/diff_workers.py corpus.jsonl <python port> <native port> out.jsonl

Errors must match exactly: status, content type and body bytes. For answers, the
model identity, usage, question and option order and the answer type must match;
probabilities are compared numerically, and the choice may differ only where the
Python worker's top-two margin is under 0.05.
"""

from __future__ import annotations

import base64
import http.client
import json
import sys
import time


def request(port, method, path, body=None, headers=None, chunked=False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    headers = dict(headers or {})
    if body is not None and not chunked:
        headers.setdefault("content-type", "application/json")
    started = time.perf_counter()
    if chunked:
        conn.putrequest(method, path)
        conn.putheader("transfer-encoding", "chunked")
        conn.putheader("content-type", "application/json")
        conn.endheaders()
        step = 1 << 16
        try:
            for i in range(0, len(body), step):
                part = body[i : i + step]
                conn.send(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            conn.send(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
    else:
        try:
            conn.request(method, path, body=body, headers=headers)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the server answered before reading the whole body
    try:
        r = conn.getresponse()
    except (ConnectionResetError, http.client.RemoteDisconnected) as e:
        conn.close()
        return -1, None, repr(e).encode(), (time.perf_counter() - started) * 1000
    data = r.read()
    ms = (time.perf_counter() - started) * 1000
    ctype = r.getheader("content-type")
    conn.close()
    return r.status, ctype, data, ms


def ordered(data: bytes):
    return json.loads(data, object_pairs_hook=lambda pairs: pairs)


def compare(name, py, rs):
    """Return (ok, note, max_prob_diff)."""
    ps, pc, pb, _ = py
    rs_, rc, rb, _ = rs
    if ps != 200 or rs_ != 200:
        same = (ps, pc, pb) == (rs_, rc, rb)
        return (
            same,
            "" if same else f"python {ps} {pb[:200]!r} | native {rs_} {rb[:200]!r}",
            0.0,
        )
    p, r = ordered(pb), ordered(rb)
    pd, rd = dict(p), dict(r)
    notes, worst = [], 0.0
    if [k for k, _ in p] != [k for k, _ in r]:
        notes.append("top-level keys differ")
    if pd["model"] != rd["model"] or pd["usage"] != rd["usage"]:
        notes.append(
            f"model/usage differ: {pd['model']} {pd['usage']} vs {rd['model']} {rd['usage']}"
        )
    pa, ra = pd["answers"], rd["answers"]
    if [k for k, _ in pa] != [k for k, _ in ra]:
        notes.append("question order differs")
    for (qn, pans), (_, rans) in zip(pa, ra):
        pans, rans = dict(pans), dict(rans)
        pp, rp = pans["probabilities"], rans["probabilities"]
        if [k for k, _ in pp] != [k for k, _ in rp] or pans["type"] != rans["type"]:
            notes.append(f"{qn}: option keys or type differ")
            continue
        pv, rv = [v for _, v in pp], [v for _, v in rp]
        worst = max(worst, max(abs(a - b) for a, b in zip(pv, rv)))
        top = sorted(pv, reverse=True)
        margin = top[0] - (top[1] if len(top) > 1 else 0.0)
        if pans["choice"] != rans["choice"] and margin >= 0.05:
            notes.append(
                f"{qn}: choice {pans['choice']!r} vs {rans['choice']!r} (margin {margin:.3f})"
            )
    return not notes, "; ".join(notes), worst


def median(values):
    return sorted(values)[len(values) // 2]


def main() -> None:
    corpus, pport, rport, out_path = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
    )
    cases = [json.loads(line) for line in open(corpus)]
    extra = []
    max_body = 4 << 20
    big = b'{"model": "cua-s1-4b-0.2", "state": "' + b"x" * (max_body + 1) + b'"}'
    extra.append(("http/too_large_content_length", "POST", "/v1/systemone", big, False))
    extra.append(("http/too_large_chunked", "POST", "/v1/systemone", big, True))
    fill = max_body - len(
        b'{"model": "cua-s1-4b-0.2", "state": "", "questions": {"q": {"type": "choice", "instructions": "go", "criteria": {"a": "A"}}}}'
    )
    exact = (
        b'{"model": "cua-s1-4b-0.2", "state": "'
        + b"ab " * (fill // 3)
        + b"a" * (fill % 3)
        + b'", "questions": {"q": {"type": "choice", "instructions": "go", "criteria": {"a": "A"}}}}'
    )
    assert len(exact) == max_body, len(exact)
    extra.append(("http/exactly_max_body", "POST", "/v1/systemone", exact, False))
    extra.append(
        (
            "http/chunked_small",
            "POST",
            "/v1/systemone",
            base64.b64decode(cases[0]["body"]),
            True,
        )
    )
    extra.append(("http/get_systemone", "GET", "/v1/systemone", None, False))
    extra.append(("http/post_health", "POST", "/health", b"{}", False))
    extra.append(("http/not_found", "GET", "/nope", None, False))

    results, failures, worst, times = [], [], 0.0, {"py": [], "rs": []}
    items = [
        (c["name"], "POST", "/v1/systemone", base64.b64decode(c["body"]), False)
        for c in cases
    ] + extra
    for i, (name, method, path, body, chunked) in enumerate(items):
        py = request(pport, method, path, body, chunked=chunked)
        rs = request(rport, method, path, body, chunked=chunked)
        ok, note, diff = compare(name, py, rs)
        worst = max(worst, diff)
        if py[0] == 200 and rs[0] == 200:
            times["py"].append(py[3])
            times["rs"].append(rs[3])
        results.append(
            {
                "name": name,
                "ok": ok,
                "note": note,
                "python_status": py[0],
                "native_status": rs[0],
                "max_prob_diff": diff,
                "python_ms": py[3],
                "native_ms": rs[3],
            }
        )
        if not ok:
            failures.append((name, note))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(items)} done, {len(failures)} failures", flush=True)

    hp = request(pport, "GET", "/health")
    hr = request(rport, "GET", "/health")
    hpj, hrj = json.loads(hp[2]), json.loads(hr[2])
    health_note = {
        k: (hpj.get(k), hrj.get(k)) for k in {**hpj, **hrj} if hpj.get(k) != hrj.get(k)
    }

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    statuses = {}
    for r in results:
        statuses[r["python_status"]] = statuses.get(r["python_status"], 0) + 1
    print(f"{len(results)} requests, python statuses {statuses}")
    print(f"mismatches: {len(failures)}")
    for name, note in failures[:40]:
        print(f"- {name}: {note}")
    print(f"largest probability difference on answered requests: {worst:.4f}")
    if times["py"]:
        py_ms, rs_ms = median(times["py"]), median(times["rs"])
        print(
            f"answered requests: python median {py_ms:.1f} ms, native median {rs_ms:.1f} ms"
        )
    print(f"/health differences (python, native): {health_note}")


if __name__ == "__main__":
    main()
