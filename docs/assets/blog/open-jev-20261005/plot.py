"""Regenerate the historical Open-Jev figures; CPU only, no model loading."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "source-data.json").read_text())
GRAY, BLUE, TEAL, ORANGE = "#9CA3AF", "#2563EB", "#0F766E", "#D97706"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "svg.fonttype": "none", "svg.hashsalt": "open-jev-20261005",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#CBD5E1",
    "text.color": "#172033", "axes.labelcolor": "#334155",
    "xtick.color": "#475569", "ytick.color": "#172033",
})


def bars(ax, rows, title, limit, colors, digits=3):
    for i, (row, color) in enumerate(zip(rows, colors)):
        ax.barh(i, row["mean_ms"], height=0.52, color=color, zorder=2)
        ax.scatter(row["pass_mean_ms"], [i] * 2, s=23, color="#172033", zorder=3)
        ax.text(row["mean_ms"] + limit * 0.018, i,
                f'{row["mean_ms"]:.{digits}f}', va="center", fontsize=11)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel("Mean wall time (ms) — lower is better")
    ax.set_title(title, loc="left", pad=16, weight="bold")
    ax.grid(axis="x", color="#E2E8F0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.margins(y=0.3)


def save(fig, name, footer):
    fig.text(0.03, 0.035, footer, fontsize=9, color="#475569")
    svg = HERE / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(HERE / f"{name}.png", dpi=170, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(name)


fig, ax = plt.subplots(figsize=(11.6, 4.5))
fig.subplots_adjust(left=0.22, right=0.93, top=0.79, bottom=0.22)
bars(ax, DATA["backend"]["rows"], "Matched full-backend comparison", 415,
     [GRAY, BLUE, TEAL])
fig.suptitle("Open-Jev on H200: 362.21 → 48.50 ms versus raw HF",
             x=0.03, y=0.96, ha="left", fontsize=17, weight="bold")
save(fig, "backend-http",
     "2026-10-03 · BF16 · 74 single-candidate requests/pass · concurrency 1\n"
     "Bars: reported aggregate means. Black dots: two pass means. Historical summary; no confidence intervals.")

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
fig.subplots_adjust(left=0.13, right=0.96, top=0.75, bottom=0.25, wspace=0.62)
bars(axes[0], DATA["graph"]["mixed"], "74 mixed requests: cache capacity matters", 112,
     [GRAY, ORANGE, BLUE])
bars(axes[1], DATA["graph"]["short"], "Fixed 107-token request: warm replay", 23.5,
     [GRAY, ORANGE, BLUE])
fig.suptitle("CUDA Graph replay: warm gains depend on retaining the workload's lengths",
             x=0.03, y=0.97, ha="left", fontsize=16, weight="bold")
save(fig, "graph-cache",
     "2026-10-03 · H200 · BF16 · concurrency 1 · independent graph-cache experiment\n"
     "Mixed: 74 requests/pass; short: 32 requests/pass. Black dots: two pass means. Different x-axis limits; both start at zero.")

fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
fig.subplots_adjust(left=0.15, right=0.96, top=0.83, bottom=0.18, hspace=0.82, wspace=0.62)
for ax, shape, limit in zip(axes.flat, DATA["gdn_kernel"]["shapes"], [0.063, 0.30, 0.94]):
    bars(ax, shape["rows"], f'{shape["tokens"]:,} tokens · complete GDN call', limit,
         [GRAY, BLUE], digits=5)
bars(axes[1, 1], DATA["gdn_http"]["rows"], "74 requests · warm HTTP", 57,
     [GRAY, BLUE])
fig.suptitle("GDN preparation: ~13% longer-call gains, 0.42% HTTP gain",
             x=0.03, y=0.97, ha="left", fontsize=17, weight="bold")
save(fig, "gdn-kernel-http",
     "2026-10-03 · H200 · two separate A/B experiments · CUDA Graph disabled\n"
     "Kernel: seeded synthetic inputs, 100 calls/pass. HTTP: 74 real requests/pass, concurrency 1.\n"
     "Bars recomputed from raw records; black dots are two pass means. Each panel has its own zero-based scale.")
