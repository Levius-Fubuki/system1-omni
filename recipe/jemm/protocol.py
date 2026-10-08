"""Frozen JEMM workload identity shared by benchmark and evidence gates."""
import hashlib
import json
from pathlib import Path

FROZEN_CORPUS_SHA256 = "5e729330b148aacec058abda8f5e0d81f9b9064a2e2bb8401426fd97e9815c4c"
DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "tests/jemm/fixtures/corpus.jsonl"
MEASURED_REPETITIONS = 2
LABEL_TOKEN_IDS = list(range(32, 58)) + list(range(15, 21))


def candidate_ids(question):
    kind = question.get("type", "choice")
    if kind == "noul":
        return ["yes", "no"]
    if kind == "score" and isinstance(question.get("criteria"), list):
        return [str(i) for i in range(len(question["criteria"]))]
    if kind == "choice" and isinstance(question.get("criteria"), dict):
        return list(question["criteria"])
    raise ValueError("invalid frozen question criteria")


def load_corpus(path=DEFAULT_CORPUS, expected_sha256=FROZEN_CORPUS_SHA256):
    raw = Path(path).read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != expected_sha256:
        raise ValueError(f"corpus SHA256 mismatch: expected {expected_sha256}, observed {checksum}")
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not cases or any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"] for c in cases):
        raise ValueError("corpus needs nonempty case IDs")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate corpus case IDs")
    for case in cases:
        body = case.get("request")
        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise ValueError("corpus request needs state and nonempty questions")
        if "model" in body:
            raise ValueError("corpus model alias is supplied by the runner")
        if case.get("modality") != ("image" if body.get("images") else "text"):
            raise ValueError("corpus modality disagrees with images")
        for question in body["questions"].values():
            if not isinstance(question, dict) or not 2 <= len(candidate_ids(question)) <= 32:
                raise ValueError("corpus question needs 2..32 candidates")
    if expected_sha256 == FROZEN_CORPUS_SHA256 and len(cases) != 13:
        raise ValueError("pinned JEMM corpus must contain 13 cases")
    return cases, checksum


def request_bytes(case):
    return json.dumps({**case["request"], "model": "JEMM"}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def expected_contract(cases):
    return [{"id": case["id"], "question_id": qid, "type": question.get("type", "choice"),
             "candidate_ids": candidate_ids(question), "image_count": len(case["request"].get("images") or [])}
            for case in cases for qid, question in case["request"]["questions"].items()]
