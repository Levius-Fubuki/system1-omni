"""Regenerate CPU parity hashes using the actual pinned Hugging Face processor.

Run in a Python 3.12 environment with torch==2.14.0, torchvision==0.29.0,
transformers==5.17.0, numpy==2.5.3 and Pillow==11.3.0. No model weights or GPU.
"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

PACKAGES = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "transformers": "5.17.0",
    "numpy": "2.5.3",
    "Pillow": "11.3.0",
}
CASES = [
    ("single_pixel", 1, 1, "constant", 1),
    ("tiny_noise", 17, 19, "noise", 17),
    ("aligned_noise", 256, 256, "noise", 123),
    ("odd_noise", 319, 241, "noise", 42),
    ("tie_down", 272, 256, "noise", 999),
    ("tie_up", 304, 256, "noise", 101),
    ("both_down", 271, 271, "checker", 1),
    ("one_axis_down", 256, 257, "ramp", 1),
    ("wide", 640, 320, "ramp", 1),
    ("portrait", 319, 641, "noise", 23),
    ("aspect_limit", 200, 1, "noise", 47),
    ("tall_aspect_limit", 7, 1400, "checker", 1),
    ("input_pixel_limit", 2048, 512, "noise", 55),
    ("large_round_up", 1600, 600, "noise", 77),
]


def pixels(width, height, pattern, seed):
    """Only input generation is duplicated in Rust; outputs come from HF."""
    if pattern == "noise":
        out = bytearray(width * height * 3)
        state = seed
        for i in range(len(out)):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            out[i] = state & 255
        return bytes(out)
    return bytes(
        (
            [0, 128, 255][c]
            if pattern == "constant"
            else (255 if (x + y + c) % 2 else 0)
            if pattern == "checker"
            else (x * 13 + y * 7 + c * 83) % 256
        )
        for y in range(height)
        for x in range(width)
        for c in range(3)
    )


def evaluate(processor, name, width, height, pattern, seed):
    raw = pixels(width, height, pattern, seed)
    array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    result = processor(images=Image.fromarray(array), return_tensors="pt", device="cpu")
    tensor = result["pixel_values"].contiguous()
    grid = result["image_grid_thw"][0].tolist()
    assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
    return {
        "name": name,
        "width": width,
        "height": height,
        "pattern": pattern,
        "seed": seed,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "grid": grid,
        "shape": list(tensor.shape),
        "image_tokens": grid[1] * grid[2] // 4,
        "resized_width": grid[2] * 16,
        "resized_height": grid[1] * 16,
        "output_sha256": hashlib.sha256(
            tensor.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest(),
    }


def main():
    for package, version in PACKAGES.items():
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != version:
            raise RuntimeError(f"{package}: expected {version}, found {actual}")
    folder = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
    cases = [evaluate(processor, *case) for case in CASES]
    manifest = {
        "schema": "cua-s1-rgb-preprocess-v1",
        "packages": PACKAGES,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor_class": type(processor).__name__,
        "config_sha256": hashlib.sha256(
            (folder / "preprocessor_config.json").read_bytes()
        ).hexdigest(),
        "source": "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/preprocessor_config.json",
        "cases": cases,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(cases)} full-output reference hashes")


if __name__ == "__main__":
    main()
