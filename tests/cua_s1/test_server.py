import json
import threading
import urllib.error
import urllib.request

import pytest
from test_protocol import request

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
    data = None if body is None else json.dumps(body).encode()
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
    assert call(url + "/v1/systemone", r)[0] == 422


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
    assert status == 500 and "private" not in json.dumps(body)
    assert not server.inference_lock.locked()
