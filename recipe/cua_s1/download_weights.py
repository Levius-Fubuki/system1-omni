"""Download the upstream-pinned artifacts and verify their checksums."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

from models.cua_s1.multimodal.model import (
    REFERENCE_REVISION,
    parse_weights_manifest,
    verify_weights,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    url = (
        f"https://raw.githubusercontent.com/trycua/cua/{REFERENCE_REVISION}/"
        "libs/cua-s1/ci/weights.lock.json"
    )
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    manifest = parse_weights_manifest(raw)
    for artifact in manifest["artifacts"]:
        snapshot_download(
            repo_id=artifact["repo_id"],
            revision=artifact["revision"],
            local_dir=args.dest / artifact["name"],
            allow_patterns=list(artifact["files"]),
            token=False,
        )
    (args.dest / "weights.lock.json").write_bytes(raw)
    verify_weights(args.dest / "Qwen3.5-4B", args.dest / "cua-s1-4b-0.2/multimodal")
    print("Pinned base and multimodal adapter checksums verified.")


if __name__ == "__main__":
    main()
