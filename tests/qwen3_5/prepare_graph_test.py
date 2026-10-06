#!/usr/bin/env python3
"""Prepare private-item graph tests without editing the production executor."""

import argparse
import hashlib
import json
from pathlib import Path
import re


WIRING = (
    b'\n#[cfg(test)]\n'
    b'#[path = "multimodal_graph_cases.rs"]\n'
    b'mod graph_tests;\n'
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    test_dir = Path(__file__).resolve().parent
    repo = test_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-source",
        type=Path,
        default=repo / "src/models/qwen3_5/native/src/model.rs",
        help="executor source to copy byte for byte",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=test_dir / "multimodal_graph_sources.json",
        help="output JSON recording source hashes and the byte-prefix check",
    )
    args = parser.parse_args()

    source = args.model_source.resolve()
    source_bytes = source.read_bytes()
    if re.search(rb"(?m)^\s*mod\s+graph_tests\s*[;{]", source_bytes):
        parser.error("executor still contains graph_tests; remove old inline tests first")
    generated = test_dir / "model_under_test.rs"
    generated.write_bytes(source_bytes + WIRING)
    generated_bytes = generated.read_bytes()
    prefix = generated_bytes[: len(source_bytes)]
    if prefix != source_bytes or generated_bytes[len(source_bytes) :] != WIRING:
        raise RuntimeError("generated executor must preserve the complete source prefix")
    if source.read_bytes() != source_bytes:
        raise RuntimeError("executor source changed during test preparation; retry")

    cases = test_dir / "multimodal_graph_cases.rs"
    harness = test_dir / "multimodal_graph.rs"
    evidence = {
        "core_source": str(source),
        "core_source_bytes": len(source_bytes),
        "core_source_sha256": sha256(source_bytes),
        "generated_model": str(generated),
        "generated_model_bytes": len(generated_bytes),
        "generated_model_sha256": sha256(generated_bytes),
        "generated_core_prefix_sha256": sha256(prefix),
        "generated_core_prefix_matches": True,
        "appended_test_wiring": WIRING.decode("utf-8"),
        "root_cases": str(cases),
        "root_cases_sha256": sha256(cases.read_bytes()),
        "root_harness": str(harness),
        "root_harness_sha256": sha256(harness.read_bytes()),
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
