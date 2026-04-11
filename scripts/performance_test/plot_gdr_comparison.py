#!/usr/bin/env python3
"""Generate GDR vs Baseline comparison chart from A/B test CSV results."""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
WARMUP_ITERS = 1  # skip first iteration (warmup)


def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def avg_stable(rows, field):
    """Average field over non-warmup iterations."""
    vals = [float(r[field]) for r in rows[WARMUP_ITERS:]]
    return np.mean(vals)


# Load data
baseline_small = load_csv("mooncake_baseline_small.csv")
baseline_medium = load_csv("mooncake_baseline_medium.csv")
gdr_small = load_csv("mooncake_gdr_small.csv")
gdr_medium = load_csv("mooncake_gdr_medium.csv")

# Compute averages (skip warmup iter 1)
data = {
    "Small\n(0.28 GB)": {
        "Baseline PUT": avg_stable(baseline_small, "put_gbit_per_sec"),
        "GDR PUT": avg_stable(gdr_small, "put_gbit_per_sec"),
        "Baseline GET": avg_stable(baseline_small, "get_gbit_per_sec"),
        "GDR GET": avg_stable(gdr_small, "get_gbit_per_sec"),
    },
    "Medium\n(7.5 GB)": {
        "Baseline PUT": avg_stable(baseline_medium, "put_gbit_per_sec"),
        "GDR PUT": avg_stable(gdr_medium, "put_gbit_per_sec"),
        "Baseline GET": avg_stable(baseline_medium, "get_gbit_per_sec"),
        "GDR GET": avg_stable(gdr_medium, "get_gbit_per_sec"),
    },
}

# Also compute per-iteration data for the detailed subplot
iter_labels_med = [f"Iter {i+1}" for i in range(len(baseline_medium))]

# ── Figure: 2 subplots ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]})
fig.suptitle("TransferQueue: GDR Staging Buffer vs Baseline (CPU memcpy)\n2-node RDMA cluster",
             fontsize=14, fontweight="bold", y=0.98)

# ── Left: Grouped bar chart (averaged, warmup excluded) ──
scenarios = list(data.keys())
x = np.arange(len(scenarios))
width = 0.18

colors = {"Baseline PUT": "#5B9BD5", "GDR PUT": "#2E75B6",
          "Baseline GET": "#ED7D31", "GDR GET": "#C55A11"}

for i, metric in enumerate(["Baseline PUT", "GDR PUT", "Baseline GET", "GDR GET"]):
    vals = [data[s][metric] for s in scenarios]
    bars = ax1.bar(x + (i - 1.5) * width, vals, width, label=metric, color=colors[metric],
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax1.set_ylabel("Throughput (Gb/s)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=10)
ax1.set_title("Average Throughput (iters 2-4)", fontsize=11)
ax1.legend(fontsize=8, ncol=2, loc="upper left")
ax1.set_ylim(0, max(max(data[s].values()) for s in scenarios) * 1.25)
ax1.grid(axis="y", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Right: Per-iteration line chart for Medium ──
iters = list(range(1, len(baseline_medium) + 1))

bl_put = [float(r["put_gbit_per_sec"]) for r in baseline_medium]
bl_get = [float(r["get_gbit_per_sec"]) for r in baseline_medium]
gdr_put = [float(r["put_gbit_per_sec"]) for r in gdr_medium]
gdr_get = [float(r["get_gbit_per_sec"]) for r in gdr_medium]

ax2.plot(iters, bl_put, "s--", color="#5B9BD5", label="Baseline PUT", markersize=7, linewidth=1.5)
ax2.plot(iters, gdr_put, "s-", color="#2E75B6", label="GDR PUT", markersize=7, linewidth=2)
ax2.plot(iters, bl_get, "o--", color="#ED7D31", label="Baseline GET", markersize=7, linewidth=1.5)
ax2.plot(iters, gdr_get, "o-", color="#C55A11", label="GDR GET", markersize=7, linewidth=2)

# Annotate key values
for it, val in zip(iters, gdr_put):
    ax2.annotate(f"{val:.1f}", (it, val), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=8, color="#2E75B6", fontweight="bold")
for it, val in zip(iters, gdr_get):
    ax2.annotate(f"{val:.1f}", (it, val), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=8, color="#C55A11", fontweight="bold")

ax2.axvspan(0.5, 1.5, alpha=0.08, color="gray")
ax2.text(1, 2, "warmup", ha="center", fontsize=8, color="gray", fontstyle="italic")

ax2.set_xlabel("Iteration", fontsize=11)
ax2.set_ylabel("Throughput (Gb/s)", fontsize=11)
ax2.set_title("Medium (7.5 GB) — Per-Iteration Detail", fontsize=11)
ax2.set_xticks(iters)
ax2.legend(fontsize=8, loc="center right")
ax2.set_ylim(0, max(max(gdr_put), max(gdr_get)) * 1.25)
ax2.grid(alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Speedup annotation box ──
# Compute speedups for medium (iters 2-4 avg)
bl_put_avg = np.mean(bl_put[1:])
gdr_put_avg = np.mean(gdr_put[1:])
bl_get_avg = np.mean(bl_get[1:])
gdr_get_avg = np.mean(gdr_get[1:])

speedup_text = (
    f"Medium (iters 2-4 avg):\n"
    f"  PUT: {bl_put_avg:.1f} -> {gdr_put_avg:.1f} Gb/s  ({gdr_put_avg/bl_put_avg:.1f}x)\n"
    f"  GET: {bl_get_avg:.1f} -> {gdr_get_avg:.1f} Gb/s  ({gdr_get_avg/bl_get_avg:.1f}x)"
)
fig.text(0.5, 0.01, speedup_text, ha="center", fontsize=10,
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#E2EFDA", edgecolor="#70AD47", alpha=0.9))

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
out_path = os.path.join(RESULTS_DIR, "gdr_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Chart saved to: {out_path}")

# Also print summary table
print("\n" + "=" * 70)
print("SUMMARY: GDR Staging Buffer vs Baseline (averages, iters 2-4)")
print("=" * 70)
for scenario in scenarios:
    d = data[scenario]
    put_speedup = d["GDR PUT"] / d["Baseline PUT"]
    get_speedup = d["GDR GET"] / d["Baseline GET"]
    print(f"\n{scenario.replace(chr(10), ' ')}:")
    print(f"  PUT: {d['Baseline PUT']:6.1f} -> {d['GDR PUT']:6.1f} Gb/s  ({put_speedup:.2f}x)")
    print(f"  GET: {d['Baseline GET']:6.1f} -> {d['GDR GET']:6.1f} Gb/s  ({get_speedup:.2f}x)")
print()
