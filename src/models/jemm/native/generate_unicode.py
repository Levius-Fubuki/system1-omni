#!/usr/bin/env python3
"""Generate Python 3.12/Unicode 15.0 printability boundaries from pinned UCD.
Source https://www.unicode.org/Public/15.0.0/ucd/UnicodeData.txt
License: LICENSE.unicode. Run: python3 generate_unicode.py /path/UnicodeData.txt
"""
import hashlib
import pathlib
import struct
import sys
EXPECTED = "806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73"
raw = pathlib.Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != EXPECTED:
    raise SystemExit("expected pinned Unicode 15.0 UnicodeData.txt")
# Unlisted points are Cn. Python treats all C and Z categories as nonprintable,
# except U+0020 ASCII space; UnicodeData First/Last pairs expand ranges.
printable = bytearray(0x110000)
start = None
for line in raw.decode("utf-8").splitlines():
    code, name, category, *_ = line.split(";")
    point = int(code, 16)
    value = not category.startswith(("C", "Z")) or point == 0x20
    if name.endswith(", First>"):
        start = point
    elif name.endswith(", Last>"):
        if start is None:
            raise SystemExit("Last without First")
        printable[start:point + 1] = bytes([value]) * (point + 1 - start)
        start = None
    else:
        printable[point] = value
boundaries = []
previous = False
for point, value in enumerate(printable):
    if bool(value) != previous:
        boundaries.append(point)
        previous = bool(value)
if previous:
    boundaries.append(0x110000)
out = pathlib.Path(__file__).parent / "src/unicode15_printable.bin"
out.write_bytes(b"".join(struct.pack("<I", x) for x in boundaries))
print(f"{len(boundaries)} boundaries; sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
