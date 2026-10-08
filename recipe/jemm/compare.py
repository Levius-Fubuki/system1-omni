"""Strict frozen JEMM response parity gates; retain all cases including ties."""
import argparse
import json
import math
import hashlib
from pathlib import Path

from protocol import DEFAULT_CORPUS, FROZEN_CORPUS_SHA256, LABEL_TOKEN_IDS, MEASURED_REPETITIONS, candidate_ids, expected_contract, load_corpus, request_bytes

PROBABILITY_TOLERANCE = 0.02
SCORE_TOLERANCE = 0.1
WINNER_MARGIN = 0.05


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_usage(usage):
    if not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens", "latency_ms"}:
        return ["usage must have exactly input_tokens, output_tokens, latency_ms"]
    failures = []
    for key in ("input_tokens", "output_tokens"):
        if type(usage[key]) is not int or usage[key] < 0:
            failures.append(f"usage.{key} must be a nonnegative integer")
    if usage["output_tokens"] != 0:
        failures.append("usage.output_tokens must be zero")
    if not number(usage["latency_ms"]) or usage["latency_ms"] <= 0:
        failures.append("usage.latency_ms must be finite and positive")
    return failures


def compare_response(reference, actual):
    failures, questions = [], {}
    if not isinstance(reference, dict) or not isinstance(actual, dict):
        return {"passed": False, "failures": ["response is not an object"], "questions": {}}
    if reference.get("model") != "JEMM" or actual.get("model") != "JEMM":
        failures.append("model identity")
    ref_usage, got_usage = reference.get("usage"), actual.get("usage")
    failures.extend("reference " + value for value in validate_usage(ref_usage))
    failures.extend("actual " + value for value in validate_usage(got_usage))
    if not isinstance(ref_usage, dict) or not isinstance(got_usage, dict):
        failures.append("malformed usage")
        ref_usage, got_usage = ref_usage or {}, got_usage or {}
        ref_usage = ref_usage if isinstance(ref_usage, dict) else {}
        got_usage = got_usage if isinstance(got_usage, dict) else {}
    for key in ("input_tokens", "output_tokens"):
        if ref_usage.get(key) != got_usage.get(key):
            failures.append("usage." + key)
    ref_answers, got_answers = reference.get("answers", {}), actual.get("answers", {})
    if not isinstance(ref_answers, dict) or not isinstance(got_answers, dict):
        return {"passed": False, "failures": failures + ["answers is not an object"], "questions": {}}
    if list(ref_answers) != list(got_answers):
        failures.append("question identity/order")
    if not ref_answers or not got_answers:
        failures.append("empty answers")
    for qid, ref in ref_answers.items():
        got = got_answers.get(qid, {})
        problem = []
        if not isinstance(ref, dict) or not isinstance(got, dict):
            failures.append(qid + ": answer is not an object")
            continue
        specific = {"choice": "choice", "score": "expected_value", "noul": "noul"}
        for side, answer in (("reference", ref), ("actual", got)):
            kind = answer.get("type")
            if kind not in specific:
                problem.append(side + " unsupported answer type")
            required = {"type", "probabilities", "confidence"} | ({specific[kind]} if kind in specific else set())
            if not required <= set(answer):
                problem.append(side + " missing required answer fields")
        if ref.get("type") != got.get("type"):
            problem.append("question type")
        rp, gp = ref.get("probabilities", {}), got.get("probabilities", {})
        if not isinstance(rp, dict) or not isinstance(gp, dict) or not rp or not gp:
            failures.append(qid + ": malformed probabilities")
            continue
        if list(rp) != list(gp):
            problem.append("candidate identity/order")
        drift = 0.0
        for cid, value in rp.items():
            observed = gp.get(cid)
            if not number(value) or not 0 <= value <= 1:
                problem.append("invalid reference probability: " + cid)
                continue
            if not number(observed) or not 0 <= observed <= 1:
                problem.append("invalid probability: " + cid)
                continue
            drift = max(drift, abs(observed - value))
        if drift > PROBABILITY_TOLERANCE + 1e-12:
            problem.append("probability drift")
        if all(number(v) for v in gp.values()) and abs(sum(gp.values()) - 1.0) > 1e-6:
            problem.append("probabilities do not sum to one")
        if not all(number(v) for v in rp.values()):
            failures.extend(qid + ": " + value for value in problem)
            continue
        ordered = sorted(rp, key=rp.get, reverse=True)
        margin = rp[ordered[0]] - rp[ordered[1]] if len(ordered) > 1 else 0.0
        winner = max(gp, key=gp.get) if gp and all(number(v) for v in gp.values()) else None
        if margin >= WINNER_MARGIN and ordered and ordered[0] != winner:
            problem.append("winner disagreement")
        if got.get("type") == "choice" and got.get("choice") != winner:
            problem.append("choice does not match probabilities")
        for key, tolerance in (("expected_value", SCORE_TOLERANCE), ("noul", PROBABILITY_TOLERANCE), ("confidence", PROBABILITY_TOLERANCE)):
            if key in ref:
                value = got.get(key)
                if not number(value) or not number(ref[key]) or abs(ref[key] - value) > tolerance + 1e-12:
                    problem.append(key + " drift")
        questions[qid] = {"probability_max_abs": drift, "reference_margin": margin, "low_margin": margin < WINNER_MARGIN, "failures": problem}
        failures.extend(qid + ": " + value for value in problem)
    return {"passed": not failures, "failures": failures, "questions": questions}


def compare_contract(reference, actual, *, cases=None):
    failures = []
    if cases is None:
        cases, _ = load_corpus()
    expected = expected_contract(cases)
    if not expected or not isinstance(reference, list) or not isinstance(actual, list) or not reference or not actual:
        return {"passed": False, "failures": ["contract records must be nonempty complete arrays"]}
    if any(not isinstance(r, dict) for r in reference + actual):
        return {"passed": False, "failures": ["contract record is not an object"]}
    identities = lambda records: [(r.get("id"), r.get("question_id")) for r in records]
    for side, records in (("reference", reference), ("actual", actual)):
        if identities(records) != identities(expected):
            failures.append(side + " case/question completeness or order")
        for index, (record, control) in enumerate(zip(records, expected)):
            required = {"id", "question_id", "type", "candidate_ids", "prompt", "input_ids", "label_token_ids", "position_ids"}
            if not required <= set(record):
                failures.append(f"{side} record {index}: missing mandatory fields")
            if any(record.get(key) != control[key] for key in ("id", "question_id", "type", "candidate_ids")):
                failures.append(f"{side} record {index}: contract identity/type/candidates")
            if not isinstance(record.get("prompt"), str) or not record["prompt"]:
                failures.append(f"{side} record {index}: invalid prompt")
            tokens = record.get("input_ids")
            if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
                failures.append(f"{side} record {index}: invalid input_ids")
            if record.get("label_token_ids") != LABEL_TOKEN_IDS:
                failures.append(f"{side} record {index}: pinned label token IDs")
            if "position_ids" in record:
                positions = record["position_ids"]
                if not isinstance(positions, list) or len(positions) != 3 or any(not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], list) or not isinstance(tokens, list) or len(axis[0]) != len(tokens) or any(type(n) is not int or n < 0 for n in axis[0]) for axis in positions):
                    failures.append(f"{side} record {index}: position_ids must have shape [3,1,T]")
            image_fields = {"image_grid_thw", "mm_token_type_ids"}
            if control["image_count"]:
                if not image_fields <= set(record):
                    failures.append(f"{side} record {index}: missing image fields")
                else:
                    grids, types = record["image_grid_thw"], record["mm_token_type_ids"]
                    if not isinstance(grids, list) or len(grids) != control["image_count"] or any(not isinstance(g, list) or len(g) != 3 or any(type(n) is not int or n < 1 for n in g) for g in grids):
                        failures.append(f"{side} record {index}: invalid image grids")
                    if not isinstance(types, list) or len(types) != 1 or not isinstance(types[0], list) or not isinstance(tokens, list) or len(types[0]) != len(tokens) or any(type(n) is not int or n not in (0, 1) for n in types[0]):
                        failures.append(f"{side} record {index}: invalid modality token types")
            elif image_fields & set(record):
                failures.append(f"{side} record {index}: image fields on text question")
    for index, (ref, got) in enumerate(zip(reference, actual)):
        for key in ("candidate_ids", "type", "prompt", "input_ids", "label_token_ids", "image_grid_thw", "mm_token_type_ids", "position_ids"):
            if (key in ref) != (key in got) or (key in ref and ref[key] != got[key]):
                failures.append(f"{ref.get('id', index)}/{ref.get('question_id')}: {key}")
    return {"passed": not failures, "expected_records": len(expected), "failures": failures}


def compare_runs(reference, actual, cases, repetitions=MEASURED_REPETITIONS):
    failures, results = [], []
    expected = [(case["id"], repetition) for repetition in range(repetitions) for case in cases]
    if not expected:
        return {"passed": False, "failures": ["empty expected workload"], "results": []}
    for side, records in (("reference", reference), ("actual", actual)):
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            failures.append(side + " records must be an array of objects")
            continue
        if [(r.get("id"), r.get("repetition")) for r in records] != expected:
            failures.append(side + " incomplete, duplicate or out-of-order measured records")
        for row, record in enumerate(records):
            case = cases[row % len(cases)]
            if type(record.get("repetition")) is not int or record.get("repetition") != row // len(cases):
                failures.append(f"{side} row {row}: invalid repetition")
            if type(record.get("index")) is not int or record.get("index") != row % len(cases) or record.get("phase") != "measured" or record.get("modality") != case["modality"]:
                failures.append(f"{side} row {row}: invalid index/phase/modality")
            if type(record.get("status")) is not int or record.get("status") != 200:
                failures.append(f"{side} row {row}: HTTP status is not 200")
            if not number(record.get("elapsed_ms")) or record["elapsed_ms"] <= 0:
                failures.append(f"{side} row {row}: invalid client latency")
            expected_hash = hashlib.sha256(request_bytes(case)).hexdigest()
            if record.get("request_sha256") != expected_hash:
                failures.append(f"{side} row {row}: request SHA256 differs from frozen request")
            response = record.get("response")
            if not isinstance(response, dict) or response.get("model") != "JEMM" or not isinstance(response.get("answers"), dict):
                failures.append(f"{side} row {row}: malformed response/model")
                continue
            if list(response["answers"]) != list(case["request"]["questions"]):
                failures.append(f"{side} row {row}: missing or reordered questions")
            for qid, question in case["request"]["questions"].items():
                answer = response["answers"].get(qid)
                if not isinstance(answer, dict) or answer.get("type") != question.get("type", "choice") or not isinstance(answer.get("probabilities"), dict) or list(answer["probabilities"]) != candidate_ids(question):
                    failures.append(f"{side} row {row}/{qid}: question type or candidate identity/order")
    if isinstance(reference, list) and isinstance(actual, list):
        for r, a in zip(reference, actual):
            if not isinstance(r, dict) or not isinstance(a, dict):
                continue
            if r.get("request_sha256") != a.get("request_sha256"):
                failures.append("paired request hashes differ")
            result = compare_response(r.get("response"), a.get("response"))
            results.append({"id": r.get("id"), "repetition": r.get("repetition"), **result})
    return {"passed": not failures and len(results) == len(expected) and all(x["passed"] for x in results),
            "failures": failures, "expected_records": len(expected), "results": results,
            "probability_tolerance": PROBABILITY_TOLERANCE, "score_tolerance": SCORE_TOLERANCE, "winner_margin": WINNER_MARGIN}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", action="store_true", help="compare preprocessing contract JSON arrays exactly")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--expected-corpus-sha256", default=FROZEN_CORPUS_SHA256, help="only override for a separately predeclared frozen workload")
    args = parser.parse_args()
    cases, corpus_hash = load_corpus(args.corpus, args.expected_corpus_sha256)
    if args.contract:
        report = compare_contract(json.loads(args.reference.read_text()), json.loads(args.actual.read_text()), cases=cases)
        report["corpus_sha256"] = corpus_hash
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise SystemExit(0 if report["passed"] else 1)
    ref = [json.loads(line) for line in args.reference.read_text().splitlines() if line]
    got = [json.loads(line) for line in args.actual.read_text().splitlines() if line]
    report = compare_runs(ref, got, cases)
    report["corpus_sha256"] = corpus_hash
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
