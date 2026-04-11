#!/usr/bin/env python3
"""
Generate side-by-side GDR vs Baseline breakdown comparison chart.

Reads two CSV files produced by breakdown_test.py:
  1. Baseline run (--device gpu, use_gdr=false)  → GPU data with CPU memcpy path
  2. GDR run      (--device gpu, use_gdr=true)   → GPU data with GDR staging buffer

Usage:
  python plot_gdr_breakdown_comparison.py \
      --baseline breakdown_baseline_gpu.csv \
      --gdr      breakdown_gdr_gpu.csv \
      [--scales small medium] \
      [--output  gdr_breakdown_comparison.png]
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: _auto_num(v) for k, v in row.items()})
    return rows


def _auto_num(v: str):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def aggregate(rows: list[dict], scales: list[str]) -> dict:
    """Aggregate rows per scale, return mean timing dicts."""
    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        sc = r["scale"]
        if sc not in scales:
            continue
        agg[sc]["data_gb"].append(r["data_gb"])

        p_total    = r.get("put.kv_batch_put.total_s", 0) * 1e3
        p_rmeta    = r.get("put.kv_batch_put.retrieve_meta_s", 0) * 1e3
        p_kvgen    = r.get("put.storage_mgr.put.key_val_gen_s", 0) * 1e3
        p_classify = r.get("put.mooncake.put.classify_s", 0) * 1e3
        p_metap    = r.get("put.storage_mgr.put.meta_processing_s", 0) * 1e3
        p_notify   = r.get("put.storage_mgr.put.notify_s", 0) * 1e3
        p_scm      = r.get("put.client.put.set_custom_meta_s", 0) * 1e3

        # Detect GDR vs baseline PUT
        if r.get("put.mooncake.gdr_batch_put.count", 0) > 0:
            p_d2d  = r.get("put.mooncake.gdr_put_loop.d2d_total_s", 0) * 1e3
            p_rdma = r.get("put.mooncake.gdr_put_loop.rdma_total_s", 0) * 1e3
            p_loop = r.get("put.mooncake.gdr_put_loop.loop_total_s", 0) * 1e3
            p_pyoh = max(0, p_loop - p_d2d - p_rdma)
        else:
            p_d2d  = 0.0
            p_rdma = r.get("put.mooncake.put_loop.rdma_total_s", 0) * 1e3
            p_pyoh = r.get("put.mooncake.put_loop.py_overhead_s", 0) * 1e3

        p_other = max(0, p_total - p_rmeta - p_kvgen - p_classify - p_d2d - p_rdma - p_pyoh - p_metap - p_notify - p_scm)

        for k, v in [
            ("put.total", p_total), ("put.classify", p_classify),
            ("put.d2d", p_d2d), ("put.rdma", p_rdma), ("put.pyoh", p_pyoh),
            ("put.other", p_rmeta + p_kvgen + p_metap + p_notify + p_scm),
            ("put.gap", p_other),
        ]:
            agg[sc][k].append(v)

        g_total    = r.get("get.kv_batch_get.total_s", 0) * 1e3
        g_rmeta    = r.get("get.kv_batch_get.retrieve_meta_s", 0) * 1e3
        g_keygen   = r.get("get.storage_mgr.key_gen_s", 0) * 1e3
        g_classify = r.get("get.mooncake.get.classify_s", 0) * 1e3
        g_scatter  = r.get("get.mooncake.get.scatter_s", 0) * 1e3
        g_gather   = r.get("get.mooncake.get.gather_s", 0) * 1e3
        g_merge    = r.get("get.storage_mgr.merge_to_tensordict_s", 0) * 1e3

        if r.get("get.mooncake.gdr_batch_get.count", 0) > 0:
            g_d2d  = r.get("get.mooncake.gdr_get_loop.d2d_total_s", 0) * 1e3
            g_rdma = r.get("get.mooncake.gdr_get_loop.rdma_total_s", 0) * 1e3
            g_loop = r.get("get.mooncake.gdr_get_loop.loop_total_s", 0) * 1e3
            g_pyoh = max(0, g_loop - g_d2d - g_rdma)
        else:
            g_d2d  = 0.0
            g_rdma = r.get("get.mooncake.get_loop.rdma_total_s", 0) * 1e3
            g_pyoh = r.get("get.mooncake.get_loop.py_overhead_s", 0) * 1e3

        g_other = max(0, g_total - g_rmeta - g_keygen - g_classify - g_scatter - g_d2d - g_rdma - g_pyoh - g_gather - g_merge)

        for k, v in [
            ("get.total", g_total), ("get.classify", g_classify + g_scatter),
            ("get.d2d", g_d2d), ("get.rdma", g_rdma), ("get.pyoh", g_pyoh),
            ("get.merge", g_merge),
            ("get.other", g_rmeta + g_keygen + g_gather),
            ("get.gap", g_other),
        ]:
            agg[sc][k].append(v)

    # Mean per scale
    result = {}
    for sc in scales:
        if sc not in agg:
            continue
        result[sc] = {k: statistics.mean(v) for k, v in agg[sc].items()}
    return result


def draw_comparison(bl_agg: dict, gdr_agg: dict, scales: list[str], out_path: str):
    """Draw 4-panel breakdown: PUT baseline/GDR, GET baseline/GDR."""
    common_scales = [s for s in scales if s in bl_agg and s in gdr_agg]
    n = len(common_scales)
    if n == 0:
        print("No common scales found between baseline and GDR data.")
        return

    # Colors
    c_classify = "#E97451"  # coral  - classify (D2H for baseline)
    c_d2d      = "#17BECF"  # cyan   - D2D copy
    c_rdma     = "#70AD47"  # green  - RDMA
    c_pyoh     = "#D9534F"  # red    - Python overhead
    c_merge    = "#FFD966"  # yellow - merge
    c_other    = "#9DC3E6"  # light blue - ZMQ + key_gen + etc
    c_gap      = "#888888"  # gray

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def draw_panel(ax, mode_agg, scales_list, direction, title, rdma_color):
        """Draw a single stacked bar panel."""
        x = np.arange(len(scales_list))
        bw = 0.5
        bottom = np.zeros(len(scales_list))

        def _bar(vals, label, color, hatch=None):
            nonlocal bottom
            arr = np.array(vals)
            if arr.max() > 0.1:
                ax.bar(x, arr, bw, bottom=bottom, label=label, color=color,
                       zorder=3, edgecolor="white", linewidth=0.5, hatch=hatch)
            bottom += arr

        dir_prefix = direction  # "put" or "get"
        totals = np.array([mode_agg[s][f"{dir_prefix}.total"] for s in scales_list])
        gb_arr = np.array([mode_agg[s]["data_gb"] for s in scales_list])

        other_vals   = [mode_agg[s][f"{dir_prefix}.other"] for s in scales_list]
        classify_vals = [mode_agg[s][f"{dir_prefix}.classify"] for s in scales_list]
        d2d_vals     = [mode_agg[s][f"{dir_prefix}.d2d"] for s in scales_list]
        rdma_vals    = [mode_agg[s][f"{dir_prefix}.rdma"] for s in scales_list]
        pyoh_vals    = [mode_agg[s][f"{dir_prefix}.pyoh"] for s in scales_list]
        gap_vals     = [mode_agg[s][f"{dir_prefix}.gap"] for s in scales_list]

        _bar(other_vals,    "ZMQ + key_gen + meta",   c_other)

        # classify label depends on whether D2H is involved
        has_d2d = max(d2d_vals) > 0
        if has_d2d:
            _bar(classify_vals, "classify (no D2H)",   c_classify)
        else:
            _bar(classify_vals, "classify (incl. D2H)", c_classify)

        _bar(d2d_vals,      "D2D copy (GPU staging)",  c_d2d)
        _bar(rdma_vals,     "RDMA transfer",           rdma_color)
        _bar(pyoh_vals,     "Python loop overhead",    c_pyoh, hatch="//")

        if direction == "get":
            merge_vals = [mode_agg[s].get("get.merge", 0) for s in scales_list]
            _bar(merge_vals, "merge_to_tensordict",    c_merge)

        if max(gap_vals) > 0.5:
            _bar(gap_vals,  "remaining gap",           c_gap, hatch="xx")

        # Annotations
        for i in range(len(scales_list)):
            bw_val = gb_arr[i] * 8 / (totals[i] / 1e3 + 1e-9)
            ax.text(i, totals[i] + max(totals) * 0.02,
                    f"{totals[i]:.0f}ms\n{bw_val:.1f} Gb/s",
                    ha="center", fontsize=9, fontweight="bold")

            # Percentage labels for large segments
            cum = 0
            for vals, color in [
                (other_vals, None), (classify_vals, "white"),
                (d2d_vals, "black"), (rdma_vals, "white"), (pyoh_vals, "white"),
            ]:
                v = vals[i]
                pct = v / totals[i] * 100 if totals[i] else 0
                if pct > 8 and color is not None:
                    ax.text(i, cum + v / 2, f"{pct:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color=color, fontweight="bold")
                cum += v

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n{gb_arr[i]:.2f} GB" for i, s in enumerate(scales_list)])
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Top row: PUT comparison
    draw_panel(axes[0, 0], bl_agg,  common_scales, "put", "PUT — Baseline (GPU→CPU→RDMA)", c_rdma)
    draw_panel(axes[0, 1], gdr_agg, common_scales, "put", "PUT — GDR (D2D→GDR RDMA)", c_rdma)
    # Bottom row: GET comparison
    draw_panel(axes[1, 0], bl_agg,  common_scales, "get", "GET — Baseline (RDMA→CPU, needs H2D)", "#FF6B35")
    draw_panel(axes[1, 1], gdr_agg, common_scales, "get", "GET — GDR (GDR RDMA→D2D→GPU)", "#FF6B35")

    # Match y-axis limits across left-right pairs
    for row in range(2):
        ymax = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        axes[row, 0].set_ylim(0, ymax * 1.15)
        axes[row, 1].set_ylim(0, ymax * 1.15)

    # Speedup annotation
    speedup_lines = []
    for sc in common_scales:
        bl_put = bl_agg[sc]["put.total"]
        gdr_put = gdr_agg[sc]["put.total"]
        bl_get = bl_agg[sc]["get.total"]
        gdr_get = gdr_agg[sc]["get.total"]
        gb = bl_agg[sc]["data_gb"]
        speedup_lines.append(
            f"{sc} ({gb:.2f}GB):  "
            f"PUT {bl_put:.0f}→{gdr_put:.0f}ms ({bl_put/gdr_put:.1f}x)  "
            f"GET {bl_get:.0f}→{gdr_get:.0f}ms ({bl_get/gdr_get:.1f}x)"
        )

    fig.suptitle(
        "MooncakeStore Breakdown: Baseline (GPU data, CPU memcpy) vs GDR (staging buffer)\n"
        "2-node RDMA cluster, --device gpu",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.01, "\n".join(speedup_lines),
        ha="center", fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E2EFDA", edgecolor="#70AD47", alpha=0.9),
    )

    fig.tight_layout(rect=[0, 0.03 + 0.02 * len(speedup_lines), 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GDR vs Baseline breakdown comparison chart")
    parser.add_argument("--baseline", required=True, help="Baseline breakdown CSV (--device gpu, use_gdr=false)")
    parser.add_argument("--gdr",      required=True, help="GDR breakdown CSV (--device gpu, use_gdr=true)")
    parser.add_argument("--scales",   nargs="+", default=["small", "medium"],
                        help="Scales to compare (default: small medium)")
    parser.add_argument("--output",   default=None, help="Output PNG path")
    args = parser.parse_args()

    bl_rows  = load_csv(args.baseline)
    gdr_rows = load_csv(args.gdr)

    bl_agg  = aggregate(bl_rows,  args.scales)
    gdr_agg = aggregate(gdr_rows, args.scales)

    out = args.output or os.path.join(os.path.dirname(args.baseline), "gdr_breakdown_comparison.png")
    draw_comparison(bl_agg, gdr_agg, args.scales, out)

    # Print summary table
    print(f"\n{'='*80}")
    print("BREAKDOWN COMPARISON: Baseline (GPU data) vs GDR")
    print(f"{'='*80}")
    for sc in args.scales:
        if sc not in bl_agg or sc not in gdr_agg:
            continue
        bl, gdr = bl_agg[sc], gdr_agg[sc]
        gb = bl["data_gb"]
        print(f"\n{sc} ({gb:.2f} GB):")
        for direction, label in [("put", "PUT"), ("get", "GET")]:
            bl_t = bl[f"{direction}.total"]
            gdr_t = gdr[f"{direction}.total"]
            bl_bw = gb * 8 / (bl_t / 1e3 + 1e-9)
            gdr_bw = gb * 8 / (gdr_t / 1e3 + 1e-9)
            print(f"  {label}: {bl_t:.0f}ms ({bl_bw:.1f} Gb/s) → {gdr_t:.0f}ms ({gdr_bw:.1f} Gb/s)  [{bl_t/gdr_t:.1f}x]")
            print(f"    Baseline: classify(D2H)={bl[f'{direction}.classify']:.0f}ms  RDMA={bl[f'{direction}.rdma']:.0f}ms  pyoh={bl[f'{direction}.pyoh']:.0f}ms")
            print(f"    GDR:      classify={gdr[f'{direction}.classify']:.0f}ms  D2D={gdr[f'{direction}.d2d']:.0f}ms  RDMA={gdr[f'{direction}.rdma']:.0f}ms  pyoh={gdr[f'{direction}.pyoh']:.0f}ms")
    print()


if __name__ == "__main__":
    main()
