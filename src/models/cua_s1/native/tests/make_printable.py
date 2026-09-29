"""Write src/printable.rs: the code points >= 0x80 that Python's repr escapes.

    python3.12 src/models/cua_s1/native/tests/make_printable.py > src/models/cua_s1/native/src/printable.rs

Use the Python version the reference worker runs on; the table follows its Unicode
database.
"""

import sys
import unicodedata

ranges = []
for cp in range(0x80, sys.maxunicode + 1):
    if not chr(cp).isprintable():
        if ranges and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])
version = ".".join(map(str, sys.version_info[:3]))
print(
    f"//! Generated from Python {version} (Unicode {unicodedata.unidata_version}): code points >= 0x80 for which"
)
print(
    "//! `str.isprintable()` is false, as inclusive ranges. Python's `repr` escapes these."
)
print("//! Regenerate with tests/make_printable.py.")
print()
print("pub const NON_PRINTABLE: &[(u32, u32)] = &[")
for lo, hi in ranges:
    print(f"    ({lo:#x}, {hi:#x}),")
print("];")
