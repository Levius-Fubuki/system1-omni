"""Write float formatting cases for the pyjson tests: "<bits as hex> <repr(float)>" lines.

    python3 src/models/cua_s1/native/tests/make_float_vectors.py floats.txt
    CUA_S1_FLOAT_VECTORS=floats.txt cargo test -p omni-cua-s1-native float_repr -- --ignored

Special values, powers of two and ten, values next to the points where repr switches
between plain and exponent notation, random bit patterns, and random short decimals.
"""

import math
import random
import struct
import sys


def bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def main() -> None:
    rng = random.Random(20260928)
    values = [0.0, -0.0, 1.0, -1.0, 0.5, 0.1, 0.2, 0.3, 1 / 3, 2 / 3, math.pi, math.e,
              5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 2.0**53, 2.0**53 + 2]
    values += [2.0**e for e in range(-1074, 1024, 7)]
    values += [10.0**e for e in range(-323, 309)]
    for e in (-5, -4, 15, 16, 17):
        base = 10.0**e
        values += [math.nextafter(base, 0.0), base, math.nextafter(base, math.inf)]
    while len(values) < 30000:
        pick = rng.random()
        if pick < 0.5:
            x = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        elif pick < 0.8:
            x = float(f"{rng.randint(1, 10**rng.randint(1, 17))}e{rng.randint(-30, 30)}")
        else:
            x = rng.uniform(-1e6, 1e6)
        if math.isfinite(x):
            values.append(x)
    with open(sys.argv[1], "w") as f:
        for x in values:
            f.write(f"{bits(x)} {x!r}\n")


if __name__ == "__main__":
    main()
