"""Deterministic screenshot requests for native end-to-end validation.

Standard cases preserve the #53 reference suite; boundary cases add maximum
area, a 200:1 image, odd dimensions, and eight questions sharing one tiny image.
"""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def make_cases(folder):
    folder.mkdir(parents=True)
    cases = []
    for name, size, fmt in [
        ("small", (320, 240), "PNG"),
        ("wide", (640, 320), "PNG"),
        ("portrait", (320, 640), "PNG"),
        ("jpeg", (640, 480), "JPEG"),
        ("single-option", (320, 240), "PNG"),
        ("26-options", (256, 256), "PNG"),
        ("two-questions", (320, 240), "PNG"),
    ]:
        image = Image.new("RGB", size, "#f4f6f8")
        draw = ImageDraw.Draw(image)
        width, height = size
        draw.rectangle(
            (16, 16, width - 16, height - 16), fill="white", outline="#8899aa"
        )
        draw.text((24, 24), "Account settings", fill="black")
        draw.text((24, 48), "Display name: Alice", fill="black")
        draw.rectangle((24, height // 2, width // 2, height // 2 + 32), fill="#1460b4")
        draw.text((28, height // 2 + 8), "Save", fill="white")
        draw.text((width // 2 + 16, height // 2 + 8), "Cancel", fill="black")
        image_path = folder / (name + (".jpg" if fmt == "JPEG" else ".png"))
        image.save(image_path, format=fmt)
        criteria = {"save": "Click Save", "cancel": "Click Cancel", "wait": "Wait"}
        if name == "single-option":
            criteria = {"save": "Click Save"}
        elif name == "26-options":
            criteria = {f"option-{i}": f"Choose action {i}" for i in range(26)}
        questions = {
            "next": {
                "type": "choice",
                "instructions": "Save the changed display name.",
                "criteria": criteria,
            }
        }
        if name == "two-questions":
            questions["second"] = {
                "type": "choice",
                "instructions": {"goal": "保存名称"},
                "criteria": {"continue": {"label": "Save"}, "cancel": None},
            }
        mime = "jpeg" if fmt == "JPEG" else "png"
        request = {
            "model": "cua-s1-4b-0.2",
            "state": {
                "image": f"data:image/{mime};base64,"
                + base64.b64encode(image_path.read_bytes()).decode()
            },
            "questions": questions,
        }
        write_json(folder / f"{name}.json", request)
        cases.append({"name": name, "image": image_path.name, "request": request})
    return cases


def make_boundaries(folder):
    folder.mkdir(parents=True, exist_ok=False)
    for name, w, h, count in [
        ("maximum", 1024, 1024, 1),
        ("narrow", 200, 1, 1),
        ("noise", 383, 257, 1),
        ("tiny-eight", 1, 1, 8),
    ]:
        rgb = bytes((i * 73 + (i // 3) * 17) % 256 for i in range(w * h * 3))
        image = Image.frombytes("RGB", (w, h), rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        questions = {
            f"q{i}": {
                "type": "choice",
                "instructions": f"Choose a useful next action. Case {i}.",
                "criteria": {
                    "save": "Click Save",
                    "cancel": "Click Cancel",
                    "wait": "Wait",
                },
            }
            for i in range(count)
        }
        write_json(
            folder / (name + ".json"),
            {
                "model": "cua-s1-4b-0.2",
                "state": {
                    "image": "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode()
                },
                "questions": questions,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--boundary", action="store_true")
    args = parser.parse_args()
    (make_boundaries if args.boundary else make_cases)(args.output)
