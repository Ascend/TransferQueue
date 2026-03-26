#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_dir = Path(__file__).resolve().parent / "results"
csv_files = list(results_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {results_dir}")

size_order = ["Small", "Medium", "Large"]

# Filename -> display name mapping for backends.
# All normalization lives here so the shell script keeps simple lowercase names.
BACKEND_DISPLAY_NAMES = {
    "simplestorage": "SimpleStorage",
    "yuanrong": "Yuanrong",
    "mooncakestore": "MooncakeStore",
    "ray_baseline": "Ray",
}


def format_size(size_gb: float) -> str:
    """Format a data size in GB to a human-readable string with appropriate unit."""
    if size_gb >= 1.0:
        return f"{size_gb:.2f} GB"
    size_mb = size_gb * 1024
    if size_mb >= 1.0:
        return f"{size_mb:.2f} MB"
    size_kb = size_mb * 1024
    return f"{size_kb:.2f} KB"


dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Parse size label and backend from filename: {backend}_{size_label}.csv
    # Size label is always the last _-separated segment (lowercase).
    # Backend is everything before the last underscore.
    # e.g. "simplestorage_small.csv" -> backend_key="simplestorage", size_label="Small"
    # e.g. "ray_baseline_small.csv"  -> backend_key="ray_baseline", size_label="Small"
    stem = csv_file.stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        print(f"Warning: skipping {csv_file.name}, unexpected filename format")
        continue
    raw_backend, raw_size = parts
    size_label = raw_size.capitalize()
    if size_label not in size_order:
        print(f"Warning: skipping {csv_file.name}, unrecognized size label '{raw_size}'")
        continue
    df["backend_parsed"] = BACKEND_DISPLAY_NAMES.get(raw_backend, raw_backend)
    df["size_label"] = size_label
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

existing_sizes = [s for s in size_order if s in df["size_label"].unique()]

# Build composite X-axis label: "SizeLabel\n<human-readable size>"
size_to_gb = df.groupby("size_label")["total_data_size_gb"].first().to_dict()


def make_xlabel(size_label: str) -> str:
    return f"{size_label}\n{format_size(size_to_gb.get(size_label, 0))}"


df["X_label"] = df["size_label"].apply(make_xlabel)

# Make X_label categorical with the correct ordering
df["X_label"] = pd.Categorical(
    df["X_label"],
    categories=[make_xlabel(s) for s in existing_sizes],
    ordered=True,
)

df["Bandwidth"] = df["total_gbit_per_sec"]
df["Scenario"] = df["backend_parsed"]

# Set backend display order
backend_order = ["Ray", "SimpleStorage", "Yuanrong", "MooncakeStore"]

df["Scenario"] = pd.Categorical(df["Scenario"], categories=backend_order, ordered=True)

# ========== Plotting ==========
sns.set_theme(style="white", palette="husl")

fig, ax = plt.subplots(figsize=(12, 7))

# Use the backend order to ensure consistent coloring
existing_backends = df["Scenario"].unique()
palette = sns.color_palette("Set2", n_colors=len(existing_backends))
barplot = sns.barplot(data=df, x="X_label", y="Bandwidth", hue="Scenario", ax=ax, alpha=0.8, palette=palette)

# Legend: match old style — at the top center, horizontal, with frame
handles, labels = ax.get_legend_handles_labels()
# Move legend above the plot
ax.get_legend().remove()
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.5, 1.0),
    loc="upper center",
    ncol=len(handles),
    title="",
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=13,
)

# Annotations on bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=0,
        )

# Axis formatting
ax.set_title("Performance Comparison (Total Throughput)", fontsize=16, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Bandwidth (Gbps)", fontsize=16)

# Adjust y range to leave room for annotations
y_max = df["Bandwidth"].max() * 1.15
ax.set_ylim(0, y_max)

ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=13)

# Unified x-label at the bottom
fig.text(0.5, 0.02, "Data Volume", ha="center", fontsize=20)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])  # room for legend + x-label
plt.savefig(results_dir / "performance_comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("Performance comparison plot generated and saved as 'performance_comparison.pdf'")
