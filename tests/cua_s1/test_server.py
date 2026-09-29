import http.client
import json
import threading
import urllib.error
import urllib.request

import pytest
from test_protocol import image_url, request

from models.cua_s1.multimodal.protocol import MAX_BODY
from models.cua_s1.multimodal.server import WorkerServer


class Engine:
    def predict(self, parsed):
        return {
            "model": "test:multimodal",
            "answers": {q.name: {"type": "choice"} for q in parsed.questions},
        }


@pytest.fixture
def worker():
    server = WorkerServer(("127.0.0.1", 0), Engine())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def call(url, body=None, **headers):
    data = (
        body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    )
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as response:
        return response.code, json.load(response)


def test_health_and_prediction(worker):
    _, url = worker
    assert call(url + "/health")[0] == 200
    status, body = call(url + "/v1/systemone", request())
    assert status == 200 and "next" in body["answers"]


def test_invalid_question_never_reaches_model(worker):
    _, url = worker
    r = request()
    r["questions"]["next"]["type"] = "noul"
    status, body = call(url + "/v1/systemone", r)
    assert status == 422 and set(body) == {"detail"}


@pytest.mark.parametrize("size", [(2048, 1), (1, 2048)])
def test_unsupported_image_aspect_ratio_returns_422_without_inference(worker, size):
    server, url = worker
    calls = []

    def unexpected(parsed):
        calls.append(parsed)
        raise AssertionError("invalid image reached inference")

    server.engine.predict = unexpected
    value = request()
    value["state"]["image"] = image_url(size=size)
    status, body = call(url + "/v1/systemone", value)
    assert status == 422 and set(body) == {"detail"}
    assert "aspect ratio" in body["detail"]
    assert not calls
    assert server.inference_lock.acquire(timeout=1)
    server.inference_lock.release()


def test_busy_worker_rejects_instead_of_queueing_gpu_work(worker):
    server, url = worker
    server.inference_lock.acquire()
    try:
        assert call(url + "/health")[0] == 200
        assert call(url + "/v1/systemone", request())[0] == 503
    finally:
        server.inference_lock.release()


def test_body_limit_checked_before_reading(worker):
    _, url = worker
    assert (
        call(url + "/v1/systemone", {}, **{"Content-Length": str(MAX_BODY + 1)})[0]
        == 413
    )


def test_model_failure_is_not_reported_as_success(worker):
    server, url = worker

    def fail(_):
        raise RuntimeError("private file or input must not leak")

    server.engine.predict = fail
    status, body = call(url + "/v1/systemone", request())
    assert status == 500 and body == {"detail": "inference failed"}
    assert server.inference_lock.acquire(timeout=1)
    server.inference_lock.release()


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b'{"x":1,"x":2}',
        b'{"x":{"y":1,"y":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e400}',
        b"[]",
        b"null",
        b'{"x":"\xff"}',
        b'{"x":"\\ud800"}',
        b'{"\\ud800":1}',
        b"[" * 2000 + b"]" * 2000,
        "{}".encode("utf-16"),
    ],
)
def test_malformed_json_returns_400_without_inference(worker, raw):
    server, url = worker

    def unexpected(_):
        raise AssertionError("malformed JSON reached inference")

    server.engine.predict = unexpected
    status, body = call(url + "/v1/systemone", raw)
    assert status == 400 and set(body) == {"detail"}
    assert server.inference_lock.acquire(timeout=1)
    server.inference_lock.release()


def test_missing_instructions_rejects_whole_request(worker):
    server, url = worker
    value = request()
    value["questions"]["second"] = {"type": "choice", "criteria": {"a": "A"}}

    def unexpected(_):
        raise AssertionError("invalid question reached inference")

    server.engine.predict = unexpected
    status, body = call(url + "/v1/systemone", value)
    assert status == 422 and set(body) == {"detail"}
    assert "instructions" in body["detail"]


@pytest.mark.parametrize(
    "route,body,headers,status",
    [
        ("/missing", None, {}, 404),
        ("/missing", {}, {}, 404),
        ("/v1/systemone", {}, {"Content-Type": "text/plain"}, 415),
        ("/v1/systemone", {}, {"Transfer-Encoding": "chunked"}, 411),
        ("/v1/systemone", {}, {"Content-Length": "invalid"}, 400),
        ("/v1/systemone", {}, {"Content-Length": str(MAX_BODY + 1)}, 413),
    ],
)
def test_transport_errors_use_detail(worker, route, body, headers, status):
    server, _ = worker
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.putrequest("GET" if body is None else "POST", route)
        request_headers = {"Content-Type": "application/json", **headers}
        if body is not None and "Transfer-Encoding" not in headers:
            request_headers.setdefault("Content-Length", str(len(json.dumps(body))))
        for name, value in request_headers.items():
            connection.putheader(name, value)
        # Every case rejects from headers. Sending a body (especially a final
        # chunk terminator) can race the server's early response and close.
        connection.endheaders()
        with connection.getresponse() as response:
            assert response.status == status
            assert set(json.load(response)) == {"detail"}
    finally:
        connection.close()
