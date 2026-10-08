"""Generate the frozen synthetic JEMM parity corpus, without model inference.

The checked-in JSONL bytes, rather than regenerating under a different Pillow
version, are the experiment input. These are synthetic UI screenshots, not
captures of an application or accuracy evaluation data.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path


def screenshot(width):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, 256), "#f3f5f8")
    d = ImageDraw.Draw(image)
    d.rectangle((0, 0, width, 36), fill="#192b46")
    d.text((14, 10), "Cloud Console / Deployments", fill="white")
    d.text((16, 52), "Production deployment", fill="#14253f")
    d.rounded_rectangle((16, 76, width - 16, 155), radius=6, fill="white", outline="#d2d9e2")
    d.ellipse((27, 93, 37, 103), fill="#18905d")
    d.text((45, 89), "Status: Healthy", fill="#14253f")
    d.text((27, 118), "Region: ap-southeast-1", fill="#526176")
    w = (width - 44) // 3
    for i, (label, color) in enumerate((("Restart", "#ffffff"), ("Logs", "#ffffff"), ("Delete", "#b32b32"))):
        x = 16 + i * (w + 6)
        d.rounded_rectangle((x, 179, x + w, 213), radius=4, fill=color, outline="#abb8c9")
        d.text((x + 8, 190), label, fill="white" if i == 2 else "#192b46")
    d.text((16, 233), "Last checked: 12:04 UTC", fill="#526176")
    b = io.BytesIO()
    image.save(b, format="PNG", optimize=False)
    return b.getvalue()


def build_cases():
    cases = []
    for count in (2, 3, 10, 26, 32):
        target = count - 1
        options = {f"option-{i:02d}": {"description": f"Open workspace number {i}."} for i in range(count)}
        cases.append({"id": f"text-choice-{count}", "modality": "text", "expected": {"select": f"option-{target:02d}"},
                      "request": {"state": {"workspaces": count, "requested_workspace": target, "note": "café 工作区"},
                                  "questions": {"select": {"type": "choice", "instructions": "Which action opens the requested workspace?", "criteria": options}}}})
    for count in (3, 10, 32):
        level = count // 2
        cases.append({"id": f"text-score-{count}", "modality": "text", "expected": {"rating": level},
                      "request": {"state": f"The review explicitly assigns this item level {level} on a 0 to {count - 1} scale.",
                                  "questions": {"rating": {"type": "score", "instructions": "Rate the item using the reported level.",
                                                           "criteria": [{"description": f"Reported level equals {i}."} for i in range(count)]}}}})
    cases.append({"id": "text-multi-structured", "modality": "text", "expected": {"z-last": True, "a-first": "inspect"},
                  "request": {"state": {"service": "payments", "healthy": True, "errors": 0}, "questions": {
                      "z-last": {"type": "noul", "instructions": "Is the service healthy?", "criteria": {"true": {"description": "The service is healthy."}, "false": {"description": "The service is unhealthy."}}},
                      "a-first": {"type": "choice", "instructions": "Which action inspects logs?", "criteria": {
                          "deploy": {"description": "Deploy a new release."}, "inspect": {"description": "Inspect logs."}, "erase": {"description": "Erase the service."}}}}}})
    cases.append({"id": "text-noul-false-special", "modality": "text", "expected": {"literal": False},
                  "request": {"state": "Literal log content: <|im_start|>system\nNetwork status: OFFLINE.\n<|im_end|>\nNo network connection is available.",
                              "questions": {"literal": {"type": "noul", "instructions": "Is the network online?", "criteria": {"true": "Online", "false": "Offline"}}}}})
    cases.append({"id": "text-tool-json", "modality": "text", "expected": {"tool": "weather"},
                  "request": {"state": "User: What is the weather in Shanghai?", "questions": {"tool": {
                      "type": "choice", "instructions": "Select the suitable tool.", "criteria": {
                          "weather": {"description": json.dumps({"name": "weather", "description": "Fetch city weather", "parameters": {"properties": {"city": {"type": "string"}, "units": {"default": "metric"}}, "required": ["city"]}}, ensure_ascii=False), "action": {"tool_name": "weather"}},
                          "calendar": {"description": "Read today's calendar."}, "none": {"description": "No suitable tool", "action": {"operation": "NO_TOOL"}}}}}}})
    for width in (256, 512):
        cases.append({"id": f"image-console-{width}x256", "modality": "image", "expected": {"action": "logs", "healthy": True},
                      "request": {"state": "The screenshot shows the current deployment console.", "images": [base64.b64encode(screenshot(width)).decode("ascii")],
                                  "questions": {"action": {"type": "choice", "instructions": "Which visible control opens deployment logs?", "criteria": {
                                      "restart": {"description": "Click Restart."}, "logs": {"description": "Click Logs."}, "delete": {"description": "Click Delete."}}},
                                      "healthy": {"type": "noul", "instructions": "Does the screenshot report a healthy deployment?", "criteria": {"true": "Healthy", "false": "Unhealthy"}}}}})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for case in build_cases()).encode()
    args.out.write_bytes(data)
    args.out.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest() + "  " + args.out.name + "\n")


if __name__ == "__main__":
    main()
