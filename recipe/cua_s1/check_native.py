"""Check `omni-cua-s1-native --score-all` output against the float32 reference.

    python recipe/cua_s1/check_native.py scores.jsonl <parity dir> [scores from another start.jsonl]

The parity directory holds parity_float32.jsonl and parity_bfloat16.jsonl, written by
compare_text_with_upstream.py --dtype float32 / bfloat16 --out (see native.md).

The rule is the one src/models/cua_s1/README.md declares for a native engine: the
largest per-option difference from the float32 worker is at most 2 x (bfloat16
worker vs float32) + 0.01, and the top option matches float32 wherever float32's
top-two margin is at least 0.05. The served result (from a graph when the
prompt fits) must be bitwise identical to the eager one. With a second file (another
process start), both runs must be bitwise identical too.
"""
import json
import sys
from pathlib import Path


def load_parity(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    return {(r["case"], r["question"]): r["worker"] for r in rows}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]


runs = load(sys.argv[1])
parity = Path(sys.argv[2])
fp32, bf16 = load_parity(parity / "parity_float32.jsonl"), load_parity(parity / "parity_bfloat16.jsonl")
diff = lambda a, b: max(abs(a[k] - b[k]) for k in b)
allowance = 2 * max(diff(bf16[k], fp32[k]) for k in fp32) + 0.01
eager = {(r["case"], r["question"]): r for r in runs if r["mode"] == "eager"}
served = {(r["case"], r["question"]): r for r in runs if r["mode"] == "served"}
missing = sorted(set(fp32) - set(eager))
worst, worst_at, flips = 0.0, None, []
for key, r in eager.items():
    ref, got = fp32[key], r["probabilities"]
    d = diff(got, ref)
    if d > worst:
        worst, worst_at = d, f"{key[0]}/{key[1]}"
    top = sorted(ref.values(), reverse=True)
    margin = top[0] - (top[1] if len(top) > 1 else 0.0)
    if margin >= 0.05 and max(ref, key=ref.get) != max(got, key=got.get):
        flips.append(f"{key[0]}/{key[1]}")
graph_keys = [k for k, r in served.items() if r["graph"]]
mismatch = [f"{k[0]}/{k[1]}" for k in served if served[k]["probabilities"] != eager[k]["probabilities"]]
print(f"{len(eager)} questions; allowance {allowance:.4f}")
print(f"largest |eager - fp32| {worst:.4f} ({worst_at}); top-option changes: {flips or 'none'}; missing: {missing or 'none'}")
print(f"served from a graph: {len(graph_keys)}; served differs from eager: {mismatch or 'none'}")
ok = worst <= allowance and not flips and not missing and not mismatch
if len(sys.argv) > 3:
    other = load(sys.argv[3])
    same = len(other) == len(runs) and all(a == b for a, b in zip(runs, other))
    print(f"identical to the other start: {same}")
    ok = ok and same
print("PASS" if ok else "FAIL")
