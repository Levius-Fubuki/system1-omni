import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from benchmarks import serving_matrix


EXPECTED = {"answers": {"a": {"choice": "yes"}, "b": {"noul": 0.75}}}


@contextmanager
def server(actions=None, health_status=200, barrier_indices=(), responses_by_tag=None):
    state = {"gets": 0, "posts": 0, "active": 0, "peak": 0, "redirect_hits": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(len(barrier_indices)) if barrier_indices else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["gets"] += 1
            if self.path == "/redirected":
                state["redirect_hits"] += 1
            self.send_response(health_status)
            self.end_headers()
            self.wfile.write(b'{"ready":true}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["posts"] += 1
                index = state["posts"]
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            status, payload, delay = (actions or {}).get(
                index, (200, json.dumps(EXPECTED).encode(), 0.015)
            )
            if responses_by_tag is not None:
                payload = json.dumps(responses_by_tag[body["tag"]]).encode()
            try:
                if index in barrier_indices:
                    barrier.wait(timeout=5)
                time.sleep(delay)
                self.send_response(status)
                if status == 302:
                    self.send_header("Location", "/redirected")
                self.end_headers()
                if isinstance(payload, list):
                    for chunk in payload:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        time.sleep(delay)
                else:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    state["active"] -= 1

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def plan(self, origin, **overrides):
        plan = {
            "endpoint": origin + "/inference",
            "cases": [{"name": "two-decisions", "request": {"model": "test"},
                       "expected_response": EXPECTED}],
            "concurrency": [1, 2], "requests_per_case": 4,
            "repetitions": 2, "warmup_per_case": 2,
            "timeout_seconds": 2, "metadata": {"revision": "pinned"},
        }
        plan.update(overrides)
        path = self.root / "input.json"
        path.write_text(json.dumps(plan, indent=2) + "\n")
        return path

    def execute(self, path):
        output = self.root / "results"
        code = serving_matrix.main([str(path), "--output", str(output)])
        self.assertTrue((output / "summary.json").is_file(), "runner must write a summary")
        summary = json.loads((output / "summary.json").read_text())
        records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
        return code, summary, records, output

    def test_closed_loop_counts_metrics_and_provenance(self):
        with server() as (origin, state):
            path = self.plan(origin)
            code, summary, records, output = self.execute(path)
        self.assertEqual(code, 0)
        self.assertEqual(len(records), 22)
        self.assertEqual(state["posts"], 22)
        self.assertEqual(state["peak"], 2)
        self.assertEqual(summary["failures"], {})
        self.assertEqual(len(summary["rounds"]), 4)
        for result in summary["rounds"]:
            self.assertEqual(result["successful_requests"], 4)
            self.assertEqual(result["successful_decisions"], 8)
            self.assertAlmostEqual(result["successful_requests_per_second"], 4 / result["elapsed_seconds"])
            self.assertAlmostEqual(result["successful_decisions_per_second"], 8 / result["elapsed_seconds"])
            latencies = sorted(r["latency_seconds"] for r in records if r["phase"] == "measured" and r["concurrency"] == result["concurrency"] and r["repetition"] == result["repetition"])
            self.assertEqual(result["successful_latency_seconds"]["p50"], latencies[1])
            self.assertEqual(result["successful_latency_seconds"]["p95"], latencies[3])
        self.assertEqual((output / "plan.json").read_bytes(), path.read_bytes())
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["plan_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(config["runner_sha256"], hashlib.sha256(Path(serving_matrix.__file__).read_bytes()).hexdigest())

    def test_correctness_failure_stops_dispatch_but_drains_in_flight(self):
        with server({6: (200, b'{"answers":{}}', 0.005), 7: (200, json.dumps(EXPECTED).encode(), 0.04)}, barrier_indices=(6, 7)) as (origin, state):
            code, summary, records, _ = self.execute(self.plan(origin, concurrency=[2]))
        self.assertEqual(code, 1)
        self.assertEqual(state["posts"], 7)
        measured = [r for r in records if r["phase"] == "measured"]
        self.assertEqual(len(measured), 2)
        self.assertEqual(summary["failures"], {"correctness": 1})
        result = summary["rounds"][0]
        self.assertEqual(result["attempted_requests"], 2)
        self.assertEqual(result["successful_requests"], 1)
        self.assertEqual(result["successful_decisions"], 2)
        self.assertAlmostEqual(result["successful_requests_per_second"], 1 / result["elapsed_seconds"])
        self.assertEqual(len(result["failure_latency_seconds"]), 1)

    def test_http_failure_preserves_body_and_nonzero_status(self):
        with server({5: (503, b"unavailable", 0.001)}) as (origin, _):
            code, summary, records, _ = self.execute(self.plan(origin, concurrency=[1]))
        self.assertEqual(code, 1)
        self.assertEqual(summary["failures"], {"http": 1})
        self.assertEqual(records[-1]["status"], 503)
        self.assertEqual(records[-1]["raw_response_text"], "unavailable")
        self.assertGreater(records[-1]["latency_seconds"], 0)
        result = summary["rounds"][0]
        self.assertEqual(result["successful_requests_per_second"], 0)
        self.assertEqual(result["successful_latency_seconds"], {"p50": None, "p95": None})

    def test_timeout_has_failure_latency_and_no_retry(self):
        with server({5: (200, b"late", 0.2)}) as (origin, state):
            code, summary, records, _ = self.execute(self.plan(origin, concurrency=[1], timeout_seconds=0.05))
        self.assertEqual(code, 1)
        self.assertEqual(state["posts"], 5)
        self.assertEqual(summary["failures"], {"timeout": 1})
        self.assertGreaterEqual(records[-1]["latency_seconds"], 0.04)

    def test_total_body_deadline_retains_partial_raw_bytes(self):
        with server({1: (200, [b'{"answers":', b"{}", b"}"], 0.03)}) as (origin, _):
            code, summary, records, _ = self.execute(self.plan(origin, timeout_seconds=0.05))
        self.assertEqual(code, 1)
        self.assertEqual(summary["failures"], {"timeout": 1})
        self.assertEqual(records[0]["raw_response_text"], '{"answers":')
        self.assertLess(records[0]["latency_seconds"], 0.15)

    def test_environment_proxy_is_ignored(self):
        proxies = {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""}
        with server() as (origin, _), patch.dict(os.environ, proxies):
            code, _, _, _ = self.execute(self.plan(origin, concurrency=[1]))
        self.assertEqual(code, 0)

    def test_does_not_follow_redirect(self):
        with server({1: (302, b"redirect", 0)}) as (origin, state):
            code, summary, records, output = self.execute(self.plan(origin))
        self.assertEqual(code, 1)
        self.assertEqual(state["redirect_hits"], 0)
        self.assertEqual(records[0]["status"], 302)
        self.assertTrue((output / "health.json").exists())
        self.assertEqual(summary["failures"], {"http": 1})

    def test_health_failure_stops_before_readiness(self):
        with server(health_status=503) as (origin, state):
            code, summary, records, output = self.execute(self.plan(origin))
        self.assertEqual(code, 1)
        self.assertEqual(state["posts"], 0)
        self.assertEqual(records, [])
        self.assertEqual(summary["rounds"], [])
        self.assertEqual(json.loads((output / "health.json").read_text())["status"], 503)
        self.assertEqual(summary["failures"], {"http": 1})

    def test_each_case_has_readiness_and_exact_text_is_supported(self):
        second = {"name": "text", "request": {}, "expected_response_text": "literal\n"}
        actions = {2: (200, b"literal\n", 0)}
        with server(actions) as (origin, _):
            path = self.plan(origin, cases=[{"name": "first", "request": {}, "expected_response": EXPECTED}, second])
            code, summary, records, _ = self.execute(path)
        self.assertEqual([r["case"] for r in records if r["phase"] == "readiness"], ["first", "text"])
        self.assertEqual(code, 1)  # Subsequent text warmup uses a JSON body and must fail.
        self.assertEqual(summary["failures"], {"correctness": 1})

    def test_raw_text_expectation_completes_and_preserves_base64(self):
        text = "literal\n"
        actions = {index: (200, text.encode(), 0) for index in range(1, 13)}
        with server(actions) as (origin, _):
            code, summary, records, _ = self.execute(self.plan(origin, concurrency=[1], cases=[{"name": "text", "request": {}, "expected_response_text": text}]))
        self.assertEqual(code, 0)
        self.assertEqual(summary["rounds"][0]["successful_decisions"], 0)
        self.assertEqual(records[0]["raw_response_base64"], "bGl0ZXJhbAo=")

    def test_json_comparison_rejects_bool_number_equivalence(self):
        with server({1: (200, b'{"answers":{"a":{"choice":"yes"},"b":{"noul":true}}}', 0)}) as (origin, _):
            expected = {"answers": {"a": {"choice": "yes"}, "b": {"noul": 1}}}
            code, summary, _, _ = self.execute(self.plan(origin, cases=[{"name": "typed", "request": {}, "expected_response": expected}]))
        self.assertEqual(code, 1)
        self.assertEqual(summary["failures"], {"correctness": 1})

    def test_rejects_invalid_budget_before_creating_output(self):
        path = self.plan("http://127.0.0.1:1", concurrency=[8])
        with self.assertRaises(ValueError):
            self.execute(path)
        self.assertFalse((self.root / "results").exists())
        path = self.plan("http://127.0.0.1:1", repetitions=3)
        with self.assertRaises(ValueError):
            self.execute(path)

    def test_nonfinite_plan_values_fail_before_http_or_output(self):
        with server() as (origin, state):
            for section in ("request", "metadata", "expected_response"):
                with self.subTest(section=section):
                    path = self.plan(origin)
                    plan = json.loads(path.read_text())
                    if section == "metadata":
                        plan["metadata"]["nested"] = {"overflow": [float("inf")]}
                    else:
                        plan["cases"][0][section]["overflow"] = [float("inf")]
                    path.write_text(json.dumps(plan).replace("Infinity", "1e309"))
                    output = self.root / f"results-{section}"
                    with self.assertRaisesRegex(ValueError, "finite JSON"):
                        serving_matrix.main([str(path), "--output", str(output)])
                    self.assertFalse(output.exists())
                    self.assertEqual(state["gets"], 0)
                    self.assertEqual(state["posts"], 0)

    def test_surrogate_text_expectation_fails_before_http_or_output(self):
        case = {"name": "text", "request": {}, "expected_response_text": "\ud800"}
        with server({1: (200, b"ok", 0)}) as (origin, state):
            path = self.plan(origin, cases=[case])
            output = self.root / "results"
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                serving_matrix.main([str(path), "--output", str(output)])
            self.assertFalse(output.exists())
            self.assertEqual(state["gets"], 0)
            self.assertEqual(state["posts"], 0)

    def test_deep_json_response_is_retained_as_correctness_failure(self):
        payload = b"[" * 20000 + b"0" + b"]" * 20000
        with server({1: (200, payload, 0)}) as (origin, state):
            try:
                code, summary, records, _ = self.execute(self.plan(origin))
            except RecursionError:
                self.fail("deep response must be retained as a correctness failure")
        self.assertEqual(code, 1)
        self.assertEqual(state["posts"], 1)
        self.assertEqual(summary["failures"], {"correctness": 1})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], 200)
        self.assertEqual(records[0]["raw_response_text"].encode(), payload)

    def test_deep_json_comparison_is_retained_as_correctness_failure(self):
        payload = b"[" * 500 + b"0" + b"]" * 500
        case = {"name": "deep", "request": {}, "expected_response": json.loads(payload)}
        with server({1: (200, payload, 0)}) as (origin, state):
            try:
                code, summary, records, _ = self.execute(self.plan(origin, cases=[case]))
            except RecursionError:
                self.fail("deep comparison must be retained as a correctness failure")
        self.assertEqual(code, 1)
        self.assertEqual(state["posts"], 1)
        self.assertEqual(summary["failures"], {"correctness": 1})
        self.assertEqual(records[0]["raw_response_text"].encode(), payload)

    def test_refuses_existing_output(self):
        path = self.plan("http://127.0.0.1:1")
        output = self.root / "results"
        output.mkdir()
        (output / "keep").write_text("original")
        with self.assertRaises(FileExistsError):
            self.execute(path)
        self.assertEqual((output / "keep").read_text(), "original")

    def test_mixed_variants_alternate_and_count_actual_decisions(self):
        responses = {
            "short": {"answers": {"a": {"choice": "yes"}}},
            "long": {"answers": {"a": {"choice": "yes"}, "b": {"noul": 0.25}, "c": {"score": 2}}},
        }
        case = {"name": "mixed", "variants": [
            {"name": name, "request": {"tag": name}, "expected_response": response}
            for name, response in responses.items()
        ]}
        with server(responses_by_tag=responses) as (origin, state):
            code, summary, records, _ = self.execute(self.plan(origin, cases=[case], concurrency=[2]))
        self.assertEqual(code, 0)
        self.assertEqual(state["posts"], 16)
        self.assertEqual([(r["variant"], r["variant_index"]) for r in records if r["phase"] == "readiness"], [("short", 0), ("long", 1)])
        self.assertEqual([(r["variant"], r["index"]) for r in records if r["phase"] == "warmup"], [("short", 0), ("short", 1), ("long", 0), ("long", 1)])
        for record in records:
            self.assertEqual(record["case"], "mixed")
            self.assertEqual(json.loads(record["raw_response_text"]), responses[record["variant"]])
            if record["phase"] in ("feasibility", "measured"):
                self.assertEqual(record["variant_index"], record["index"] % 2)
                self.assertEqual(record["variant"], ["short", "long"][record["index"] % 2])
        for result in summary["rounds"]:
            self.assertEqual(result["case"], "mixed")
            self.assertEqual(result["successful_requests"], 4)
            self.assertEqual(result["successful_decisions"], 8)
            self.assertAlmostEqual(result["successful_decisions_per_second"], 8 / result["elapsed_seconds"])


if __name__ == "__main__":
    unittest.main()
