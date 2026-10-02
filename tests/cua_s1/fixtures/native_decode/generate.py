"""Tiny lossless PNG oracles; expectations checked with pinned Pillow 11.3.0."""

import struct
import zlib
from pathlib import Path


def chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


for name, kind, pixels, transparent in [
    ("l16", 0, [1000], None),
    ("l16-trns", 0, [1000], struct.pack(">H", 1000)),
    ("la16", 4, [1000, 0], None),
    ("rgb16", 2, [1000, 255, 32768], None),
    ("rgb16-trns", 2, [1000, 255, 32768], struct.pack(">HHH", 1000, 255, 32768)),
    ("rgba16", 6, [1000, 255, 32768, 0], None),
]:
    data = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, kind, 0, 0, 0)
    )
    if transparent is not None:
        data += chunk(b"tRNS", transparent)
    data += chunk(
        b"IDAT", zlib.compress(b"\0" + struct.pack(">" + "H" * len(pixels), *pixels))
    ) + chunk(b"IEND", b"")
    Path(__file__).with_name(name + ".png").write_bytes(data)
