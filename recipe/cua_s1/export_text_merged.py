"""Export Qwen3.5-4B with the Cua-S1 `text` adapter merged, for the native worker.

    PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_text_merged.py \
        --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
        --out weights/cua-s1-4b-0.2-text-merged

Run it in the environment of the text worker (recipe/cua_s1/text.md). It loads the
model as that worker does, merges the adapter into the bfloat16 weights with PEFT's
`merge_and_unload` on the given device, and writes the checkpoint and tokenizer
files to --out, plus `cua_s1_export.json` with the revisions it was made from (read
from the Hugging Face download metadata) and the SHA-256 of the tokenizer.json it
wrote. The native worker reads that record and checks the tokenizer against it:
Transformers writes the pre-tokenizer rule it actually uses into this file, which
is not the one in the base repository's tokenizer.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import peft
import torch
import transformers

from models.cua_s1.text.adapter import downloaded_revision
from models.cua_s1.text.contract import ADAPTER_REPO, BASE_REPO, LETTERS
from models.cua_s1.text.engine import TextEngine


def base_revision(base: Path) -> str | None:
    """The commit Hugging Face recorded when it downloaded config.json."""
    meta = base / ".cache/huggingface/download/config.json.metadata"
    return meta.read_text().splitlines()[0].strip() if meta.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.perf_counter()
    engine = TextEngine(str(args.base), str(args.adapter), args.device, "bfloat16")
    model = engine.model.merge_and_unload().eval()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    engine.tokenizer.save_pretrained(args.out)
    tokenizer = (args.out / "tokenizer.json").read_bytes()
    record = {
        "format": "cua-s1-text-merged/1",
        "base": {"repo": BASE_REPO, "revision": base_revision(args.base)},
        "adapter": {
            "repo": ADAPTER_REPO,
            "revision": downloaded_revision(args.adapter),
            "subfolder": "text",
        },
        "merge": {
            "method": "peft merge_and_unload",
            "device": args.device,
            "dtype": "bfloat16",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "letters": LETTERS,
        "tokenizer": {
            "file": "tokenizer.json",
            "sha256": hashlib.sha256(tokenizer).hexdigest(),
            "saved_by": f"transformers {transformers.__version__}",
        },
    }
    (args.out / "cua_s1_export.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"exported to {args.out} in {time.perf_counter() - started:.1f} s")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
