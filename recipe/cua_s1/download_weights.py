"""Download the upstream-pinned artifacts and verify their checksums."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from models.cua_s1.multimodal.model import verify_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    lock = (
        Path(__file__).resolve().parents[2]
        / "src/models/cua_s1/multimodal/weights.lock.json"
    )
    for artifact in json.loads(lock.read_text())["artifacts"]:
        snapshot_download(
            repo_id=artifact["repo_id"],
            revision=artifact["revision"],
            local_dir=args.dest / artifact["name"],
            allow_patterns=list(artifact["files"]),
            token=False,
        )
    verify_weights(args.dest / "Qwen3.5-4B", args.dest / "cua-s1-4b-0.2/multimodal")
    print("Pinned base and multimodal adapter checksums verified.")


if __name__ == "__main__":
    main()
