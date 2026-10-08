# Frozen HTTP serving matrix

`serving_matrix.py` is a Python 3.11+ standard-library client for one already
ready server. It does not launch servers, reserve devices, or collect GPU data.
Freeze model, server, hardware, workload and cache provenance in `metadata` before
running; these fields are retained verbatim, not independently verified.

```sh
python3 benchmarks/serving_matrix.py /path/to/frozen-plan.json \
  --output /path/to/new-results-directory
```

The output directory must not exist. A plan has this shape (replace the example
request and exact response with the workload's reference):

```json
{
  "endpoint": "http://127.0.0.1:8000/v1/systemone",
  "health_endpoint": "http://127.0.0.1:8000/health",
  "cases": [
    {
      "name": "short",
      "request": {"model": "served-alias", "state": "frozen input", "questions": {}},
      "expected_response": {"answers": {}}
    }
  ],
  "concurrency": [1, 8, 16],
  "requests_per_case": 32,
  "repetitions": 2,
  "warmup_per_case": 2,
  "timeout_seconds": 120,
  "metadata": {"server_revision": "exact commit", "workload_revision": "exact revision"}
}
```

`health_endpoint` is optional and defaults to the endpoint origin's `/health`.
Case names and positive concurrency values must be unique. `requests_per_case`
must be at least the largest concurrency value; repetitions and warmup count
must each equal two. Requests are POSTed unchanged as JSON. There is no token
injection or additional model selection. The complete decoded plan, including
requests, expectations and metadata, must serialize as finite JSON. Nonfinite
values, including exponent overflow such as `1e309`, fail before creating output
or making HTTP requests.

Each case must specify exactly one of `expected_response` or
`expected_response_text`. JSON expectations compare decoded values recursively,
including scalar types, with no tolerance or ignored fields. Object key order
does not matter; array order does. Text expectations compare the response bytes
with the expectation's UTF-8 encoding, including whitespace. Invalid JSON fails
a JSON expectation. An HTTP status other than 200 always fails.

The client captures one global health snapshot and one first inference per case,
then two sequential warmup requests per case. A case may instead provide a
nonempty `variants` list, replacing its top-level request and expectation:

```json
{
  "name": "mixed-shapes",
  "variants": [
    {"name": "short", "request": {"state": "short input"}, "expected_response": {"answers": {}}},
    {"name": "long", "request": {"state": "longer input"}, "expected_response": {"answers": {}}}
  ]
}
```

Each variant needs its own unique name within the case, request and exact
expectation. Each variant receives an independent first inference and two
warmup requests. Feasibility and measured requests cycle through variants using
`index % len(variants)`, starting at variant zero in each wave/round. Summaries
aggregate the mixed case; decision counts use each actual successful response.
Every inference record includes `variant` and its zero-based `variant_index`;
an original single-request case uses its case name as its sole variant name.
Choose a measured request count divisible by the variant count for equal shares.

Readiness, warmup and feasibility phases are excluded from
performance metrics. Each case/concurrency pair receives one excluded feasibility
wave of `min(concurrency, requests_per_case)` requests, followed by two measured
rounds with `requests_per_case` requests each. Fixed worker threads send their
next request only after the prior one completes; at most `concurrency` client
requests are in flight. There is no arrival-rate simulation.

Any error stops new dispatch and subsequent phases. Requests already in flight
finish and are retained. A failed health snapshot is saved with the summary and
stops the matrix before any readiness inference is dispatched. No exchange is
retried, redirected, or routed through an environment proxy. Socket connection
operations use the declared timeout; once connected, the remaining deadline
also covers request transmission, headers and the complete response body.
Standard-library hostname resolution is not bounded by the socket timeout;
use a resolved IP endpoint when a strict DNS-independent deadline is required.

Outputs are:

- `plan.json`: exact input bytes.
- `config.json`: plan SHA256, runner SHA256, metadata and client settings.
- `health.json`: health status, raw bytes, latency and any failure.
- `responses.jsonl`: every inference exchange in completion order, with case,
  phase, zero-based request index, concurrency, measured repetition (1 or 2),
  status, headers, base64 raw bytes, readable UTF-8 text, latency and error.
- `summary.json`: global failure counts and a summary for every attempted
  measured round, including an incomplete round when execution stops.

Round wall time starts before worker creation and ends after all workers finish
and write their records. Successful requests/s and decisions/s divide successful
counts by this complete elapsed time, including failures and their drain time.
A decision is one member of the successful response's `answers` map or list;
non-JSON text responses have zero counted decisions. Client request latency
includes transport and response validation. Successful p50/p95 use nearest-rank
quantiles; failure latencies are listed separately and remain in raw records.
Round summaries include attempted and planned counts, successful counts, failure
counts and completion status. Excluded-phase failures appear in global counts.
Exit status is zero only for a complete error-free matrix; any exchange failure
returns one after writing the summary. Invalid plans and existing directories
fail before network requests or output overwrite.

CPU-only verification:

```sh
python3 -m unittest discover -s tests/benchmarks -p test_serving_matrix.py -v
```
