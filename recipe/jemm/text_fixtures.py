"""Capture CPU text fixtures directly from pinned upstream JEMM and tokenizer."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def generate(source, base, corpus):
    inventory = json.loads(Path(__file__).with_name("pinned_inventory.json").read_text())
    source, base = Path(source).resolve(), Path(base).resolve()
    for name, expected in inventory["upstream_source_sha256"].items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned upstream source mismatch: {name}")
    token_files = {name: inventory["base"]["files"][name] for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")}
    for name, expected in token_files.items():
        path = base / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned tokenizer mismatch: {name}")
    sys.path.insert(0, str(source))
    from jemm.contract import encode_single, label_token_ids, single_messages
    from jemm.systemone import single_requests
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    cases = []
    for case in corpus:
        if case["request"].get("images"):
            continue
        body = case["request"]
        questions = []
        for qid, kind, request in single_requests(body["state"], body["questions"]):
            tokens = encode_single(tokenizer, request)
            prompt = tokenizer.apply_chat_template(single_messages(request), tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False, preserve_thinking=False)
            questions.append({"question_id": qid, "type": kind, "candidate_ids": [c["id"] for c in request["candidates"]],
                              "prompt": prompt, "input_ids": tokens, "input_tokens": len(tokens)})
        cases.append({"id": case["id"], "request": body, "questions": questions})
    return {"provenance": {**inventory["provenance"], "source_sha256": inventory["upstream_source_sha256"],
                           "tokenizer_sha256": token_files, "transformers_version": importlib.metadata.version("transformers"),
                           "enable_thinking": False, "preserve_thinking": False},
            "label_token_ids": label_token_ids(tokenizer), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line]
    result = generate(args.source, args.base, corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
