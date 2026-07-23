#!/usr/bin/env python3
"""Plots the cube-domain thread-scaling curve before and after the
halo-exchange fix (PERFORMANCE.md), reading
benchmarks/thread_scaling_before_fix.csv (saved from commit 87480c4) and
benchmarks/thread_scaling.csv (current).

Usage:
    python3 benchmarks/plot_halo_fix_before_after.py

Requires matplotlib (not a crate dependency -- this is a one-off analysis
script, not part of the Rust build).
"""

import csv
import pathlib

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
BEFORE_CSV = HERE / "thread_scaling_before_fix.csv"
AFTER_CSV = HERE / "thread_scaling.csv"
OUT_PATH = HERE / "halo_fix_before_after.png"

# Same categorical palette as the other benchmarks/validation plots.
BEFORE_COLOR = "#eb6834"
AFTER_COLOR = "#2a78d6"
GRID_COLOR = "#d8d8d4"
TEXT_COLOR = "#26261f"


def load(path, shape):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["shape"] == shape:
                rows.append({"threads": int(row["threads"]), "steps_per_s": float(row["steps_per_s"])})
    rows.sort(key=lambda r: r["threads"])
    return rows


def main():
    before = load(BEFORE_CSV, "cube")
    after = load(AFTER_CSV, "cube")

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(
        [r["threads"] for r in before], [r["steps_per_s"] for r in before],
        marker="o", markersize=7, linewidth=2, color=BEFORE_COLOR,
        label="Before: crossbeam-channel clone per boundary/phase/step",
    )
    ax.plot(
        [r["threads"] for r in after], [r["steps_per_s"] for r in after],
        marker="s", markersize=7, linewidth=2, color=AFTER_COLOR,
        label="After: direct CrossSlabPtr reads, no clone/allocation",
    )

    ax.set_xlabel("rayon threads", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("steps/s", color=TEXT_COLOR, fontsize=11)
    ax.set_title(
        "256³ cube: halo-exchange fix turns a scaling regression into speedup",
        color=TEXT_COLOR, fontsize=11.5, pad=12,
    )
    ax.set_ylim(bottom=0)

    ax.grid(True, which="both", linewidth=0.6, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    legend = ax.legend(loc="lower center", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PATH}")

    before_1, before_12 = before[0]["steps_per_s"], before[-1]["steps_per_s"]
    after_1, after_12 = after[0]["steps_per_s"], after[-1]["steps_per_s"]
    print(f"before: {before_1:.2f} -> {before_12:.2f} steps/s ({before_12/before_1:.3f}x)")
    print(f"after:  {after_1:.2f} -> {after_12:.2f} steps/s ({after_12/after_1:.3f}x)")


if __name__ == "__main__":
    main()
