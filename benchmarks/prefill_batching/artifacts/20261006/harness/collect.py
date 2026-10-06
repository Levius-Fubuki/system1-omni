"""Frozen current-main versus packed-prefill HTTP comparison."""
import ctypes
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
PLAN = json.loads((RUN / "plan.json").read_text())
C = PLAN["controls"]
WORKLOADS = json.loads((RUN / "harness/requests.json").read_text())
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROCESSES, LOGS = [], []
STATUS = {"status": "running", "phases": []}
REFERENCE = {}
MAX_DRIFT = 0.0


def save(name, data):
    (RUN / "analysis" / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def phase(name, **extra):
    STATUS["phase"] = name
    STATUS["phases"].append({"name": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
    save("status.json", STATUS)
    print(name, json.dumps(extra), flush=True)


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(args, env, name):
    log = (RUN / "analysis" / (name + ".log")).open("w")
    LOGS.append(log)
    start = time.perf_counter()
    process = subprocess.Popen(args, env=env, stdout=log, stderr=log)
    PROCESSES.append(process)
    save("owned-processes.json", [{"pid": p.pid, "args": p.args} for p in PROCESSES])
    return process, start


def ready(process, start, url):
    while time.perf_counter() - start < 600:
        if process.poll() is not None:
            raise RuntimeError(f"worker {process.pid} exited with {process.returncode}")
        try:
            with OPENER.open(url + "/health", timeout=1) as response:
                body = json.load(response)
            assert body["status"] == "ready", body
            return {"seconds": time.perf_counter() - start, "body": body}
        except (OSError, urllib.error.URLError):
            time.sleep(.2)
    raise TimeoutError("readiness")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    assert process.poll() is not None


def compare(a, b, path="answers"):
    global MAX_DRIFT
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            compare(a[k], b[k], path + "." + k)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b), path
        delta = abs(a - b)
        MAX_DRIFT = max(MAX_DRIFT, delta)
        assert delta <= PLAN["acceptance"]["maximum_probability_or_score_drift"], (path, a, b, delta)
        if path.endswith(".noul"):
            assert (a >= .5) == (b >= .5), (path, a, b)
    else:
        assert a == b, (path, a, b)


def post(url, workload):
    request = urllib.request.Request(url + "/v1/systemone", data=json.dumps(workload["body"], ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with OPENER.open(request, timeout=180) as response:
        body = response.read().decode("utf-8")
    elapsed = time.perf_counter() - start
    raw = json.loads(body)
    assert raw["answers"].keys() == workload["body"]["questions"].keys(), workload["id"]
    assert raw["usage"]["output_tokens"] == 0
    return {"id": workload["id"], "latency_s": elapsed, "raw": raw}


def check(row, reference):
    compare(row["raw"]["answers"], reference["raw"]["answers"])
    assert row["raw"]["usage"] == reference["raw"]["usage"], row["id"]
    assert row["raw"]["model"] == reference["raw"]["model"]
    ma, mb = dict(row["raw"]["metadata"]), dict(reference["raw"]["metadata"])
    ma.pop("inference_seconds"); mb.pop("inference_seconds")
    assert ma == mb, (row["id"], ma, mb)


def stats(rows):
    values = sorted(r["latency_s"] * 1000 for r in rows)
    return {"requests": len(values), "mean_ms": statistics.mean(values), "p50_ms": statistics.median(values),
            "p95_ms": values[math.ceil(.95 * len(values)) - 1],
            "successful_decisions_per_second": sum(len(r["raw"]["answers"]) for r in rows) / sum(r["latency_s"] for r in rows)}


def interrupted(sig, frame):
    raise RuntimeError(f"interrupted by {sig}")


signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
try:
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert sorted(os.sched_getaffinity(0)) == C["cpus"]
    for path, expected in PLAN["hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    cuda = ctypes.CDLL("libcuda.so.1")
    assert cuda.cuInit(0) == 0
    count, device = ctypes.c_int(), ctypes.c_int()
    assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 1
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
    uuid = (ctypes.c_ubyte * 16)()
    assert cuda.cuDeviceGetUuid(ctypes.byref(uuid), device) == 0
    h = bytes(uuid).hex()
    actual = "GPU-" + "-".join([h[:8], h[8:12], h[12:16], h[16:20], h[20:]])
    assert actual == C["gpu_uuid"], actual
    reservation = next(r for r in json.loads(subprocess.check_output(["gpu", "status", "--json"], text=True)) if r["gpu_id"] == 2)
    assert reservation["status"] == "IN_USE" and reservation["user"] == "hsliu2", reservation
    save("reservation.json", reservation)
    save("gpu-probe.json", {"uuid": actual, "visibility": os.environ["CUDA_VISIBLE_DEVICES"], "cpus": sorted(os.sched_getaffinity(0))})
    summaries = {}
    for arm in ["baseline", "candidate"]:
        phase(arm + "-startup")
        wp, fp = port(), port()
        assert wp != fp
        worker_url, frontend_url = f"http://127.0.0.1:{wp}", f"http://127.0.0.1:{fp}"
        env = {**os.environ, "OPEN_JEV_MODEL": C["model"], "OPEN_JEV_HOST": "127.0.0.1", "OPEN_JEV_PORT": str(wp),
               "OPEN_JEV_CUDA_LIB": C["library"], "CUA_S1_GRAPH": "0"}
        worker, started = launch([C[arm + "_worker"]], env, arm + "-worker")
        readiness = {"worker": ready(worker, started, worker_url)}
        longest = max(WORKLOADS["single"], key=lambda w: len(json.dumps(w["body"])))
        save(arm + "-first-inference.json", post(worker_url, longest))
        frontend, started = launch([C["frontend"]], {**env, "OMNI_JEV_BIND": f"127.0.0.1:{fp}", "OMNI_JEV_BACKEND_URL": worker_url}, arm + "-frontend")
        readiness["frontend"] = ready(frontend, started, frontend_url)
        save(arm + "-readiness.json", readiness)
        summaries[arm] = {}
        for name, workloads in WORKLOADS.items():
            summaries[arm][name] = []
            for repetition in [0, 1, 2]:
                tag = "feasibility" if repetition == 0 else f"measured-{repetition}"
                phase(arm + "-" + name + "-" + tag)
                rows = []
                with (RUN / "analysis" / f"{arm}-{name}-{tag}.jsonl").open("w") as output:
                    for workload in workloads:
                        row = post(frontend_url, workload)
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); output.flush()
                        rows.append(row)
                        key = (name, row["id"])
                        if arm == "baseline" and repetition == 0:
                            REFERENCE[key] = row
                        else:
                            check(row, REFERENCE[key])
                measured = stats(rows)
                phase(arm + "-" + name + "-" + tag + "-done", **measured)
                if repetition:
                    summaries[arm][name].append(measured)
            save("summary-partial.json", {"results": summaries, "maximum_drift": MAX_DRIFT})
        stop(frontend); stop(worker)
    reductions = {}
    for name in WORKLOADS:
        a = summaries["baseline"][name]; b = summaries["candidate"][name]
        reductions[name] = {"mean_reduction_percent": 100 * (1 - statistics.mean(x["mean_ms"] for x in b) / statistics.mean(x["mean_ms"] for x in a)),
                            "p95_reduction_percent": 100 * (1 - statistics.mean(x["p95_ms"] for x in b) / statistics.mean(x["p95_ms"] for x in a))}
    gates = {"multi_question": all(reductions[n]["mean_reduction_percent"] >= 15 for n in ["questions-4", "questions-8"]),
             "single_mean": reductions["single"]["mean_reduction_percent"] >= -2,
             "single_p95": reductions["single"]["p95_reduction_percent"] >= -2,
             "fidelity": MAX_DRIFT <= .001}
    save("summary.json", {"status": "complete", "results": summaries, "reductions": reductions, "gates": gates,
                          "maximum_drift": MAX_DRIFT, "failures": 0, "decision_flips": 0})
    STATUS["status"] = "complete"
    phase("complete", reductions=reductions, gates=gates, maximum_drift=MAX_DRIFT)
except BaseException as e:
    STATUS.update({"status": "failed", "error": str(e), "maximum_drift": MAX_DRIFT})
    save("status.json", STATUS)
    traceback.print_exc()
    raise
finally:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for process in reversed(PROCESSES):
        stop(process)
    for log in LOGS:
        log.close()
    save("cleanup.json", {"all_task_processes_exited": all(p.poll() is not None for p in PROCESSES),
                          "processes": [{"pid": p.pid, "returncode": p.returncode} for p in PROCESSES]})
