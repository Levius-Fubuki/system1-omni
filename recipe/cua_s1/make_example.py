"""Create redistributable synthetic GUI fixtures and inline-image requests."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_example(path: Path, size=(640, 480), fmt="PNG") -> dict:
    image = Image.new("RGB", size, "#f4f6f8")
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle(
        (w // 10, h // 8, w * 9 // 10, h * 7 // 8),
        fill="white",
        outline="#8899aa",
        width=2,
    )
    draw.text(
        (w // 7, h // 5), "Account settings", fill="black", font_size=max(12, w // 25)
    )
    draw.text(
        (w // 7, h // 3),
        "Display name: Alice",
        fill="black",
        font_size=max(10, w // 32),
    )
    draw.rectangle((w // 7, h // 2, w * 4 // 7, h * 2 // 3), fill="#1460b4")
    draw.text(
        (w // 6, h * 13 // 24), "Save changes", fill="white", font_size=max(10, w // 32)
    )
    draw.text(
        (w * 5 // 8, h * 13 // 24), "Cancel", fill="black", font_size=max(10, w // 32)
    )
    image.save(path, format=fmt)
    mime = "jpeg" if fmt == "JPEG" else "png"
    return {
        "model": "cua-s1-4b-0.2",
        "state": {
            "image": f"data:image/{mime};base64,"
            + base64.b64encode(path.read_bytes()).decode()
        },
        "questions": {
            "next": {
                "type": "choice",
                "instructions": "Save the changed display name.",
                "criteria": {
                    "save": "Click Save changes",
                    "cancel": "Click Cancel",
                    "wait": "Wait",
                },
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = make_example(args.output.with_suffix(".png"))
    args.output.write_text(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
