"""Check live native screenshot HTTP answers against native replay outputs."""

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28003")
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.controls / "manifest.json").read_text())
    native = json.loads(args.native.read_text())
    if not manifest["cases"] or len(manifest["cases"]) != len(native["cases"]):
        raise ValueError("case coverage mismatch")
    records = []

    def post(raw):
        request = urllib.request.Request(
            args.url + "/v1/systemone",
            data=raw,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    with urllib.request.urlopen(args.url + "/health") as response:
        health = json.load(response)
        assert (
            health["status"] == "ready"
            and health["model"]
            == "cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:multimodal"
        )
    for case, result in zip(manifest["cases"], native["cases"]):
        assert case["case"] == result["case"]
        assert len(case["questions"]) == len(result["questions"])
        assert [q["name"] for q in case["questions"]] == [
            q["name"] for q in result["questions"]
        ]
        raw = (args.controls / case["request_file"]).read_bytes()
        body = json.loads(raw)
        assert list(body["questions"]) == [q["name"] for q in case["questions"]]
        status, answer = post(raw)
        assert status == 200, answer
        assert answer["model"] == health["model"]
        assert answer["usage"]["input_tokens"] == sum(
            len(q["input_ids"]) for q in case["questions"]
        )
        assert answer["usage"]["output_tokens"] == 0
        assert list(answer["answers"]) == list(body["questions"])
        for q in result["questions"]:
            probs = list(answer["answers"][q["name"]]["probabilities"].values())
            keys = list(body["questions"][q["name"]]["criteria"])
            response = answer["answers"][q["name"]]
            assert list(response["probabilities"]) == keys
            assert response["type"] == "choice"
            assert (
                response["choice"]
                == keys[max(range(len(probs)), key=probs.__getitem__)]
            )
            confidence = (
                1.0
                if len(probs) == 1
                else max(
                    0.0,
                    1.0
                    + sum(p * math.log(p) for p in probs if p) / math.log(len(probs)),
                )
            )
            assert abs(response["confidence"] - confidence) < 1e-7
            assert len(probs) == len(q["probabilities"])
            assert max(abs(a - b) for a, b in zip(probs, q["probabilities"])) < 1e-7
        records.append(
            {
                "case": case["case"],
                "status": status,
                "usage": answer["usage"],
                "answers": answer["answers"],
            }
        )
    # Run input failures after successful requests to exercise the live loaded worker.
    valid = json.loads(
        (args.controls / manifest["cases"][0]["request_file"]).read_bytes()
    )
    bad = []
    bad.append((b'{"a":1,"a":2}', 400))
    bad.append((b'{"x":NaN}', 400))
    bad.append((b"[]", 400))
    bad.append((b"{}", 422))
    invalid = json.loads(json.dumps(valid))
    invalid["state"]["image"] = "https://example.invalid/a.png"
    bad.append((json.dumps(invalid).encode(), 422))
    invalid = json.loads(json.dumps(valid))
    next(iter(invalid["questions"].values()))["instructions"] = "<|image_pad|>"
    bad.append((json.dumps(invalid).encode(), 422))
    invalid = json.loads(json.dumps(valid))
    invalid["questions"] = {
        str(i): next(iter(valid["questions"].values())) for i in range(9)
    }
    bad.append((json.dumps(invalid).encode(), 422))
    invalid = json.loads(json.dumps(valid))
    invalid["extra"] = True
    bad.append((json.dumps(invalid).encode(), 422))
    bad.append((b" " * (8 * 1024 * 1024 + 1), 413))
    for raw, expected in bad:
        status, response = post(raw)
        assert status == expected and isinstance(response.get("detail"), str), (
            status,
            response,
        )
    args.out.write_text(
        json.dumps(
            {"passed": True, "requests": records, "invalid_inputs_checked": len(bad)},
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: {len(records)} screenshot requests and {len(bad)} input failures")


if __name__ == "__main__":
    main()
