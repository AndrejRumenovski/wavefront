#!/usr/bin/env python3
"""Plots benchmarks/thread_scaling.csv (written by
benchmarks/thread_scaling.sh) as steps/s vs. thread count, for both domain
shapes.

Usage:
    python3 benchmarks/plot_thread_scaling.py

Requires matplotlib (not a crate dependency -- this is a one-off analysis
script, not part of the Rust build).
"""

import csv
import pathlib

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
CSV_PATH = HERE / "thread_scaling.csv"
OUT_PATH = HERE / "thread_scaling.png"

# Same categorical palette as validation/plot_convergence.py and
# validation/plot_pml_reflection.py -- slot 1 blue, slot 2 orange, assigned
# by fixed category order (cube first, tall second), not cycled.
CUBE_COLOR = "#2a78d6"
TALL_COLOR = "#eb6834"
GRID_COLOR = "#d8d8d4"
TEXT_COLOR = "#26261f"


def main():
    rows_by_shape = {"cube": [], "tall": []}
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows_by_shape[row["shape"]].append(
                {"threads": int(row["threads"]), "steps_per_s": float(row["steps_per_s"])}
            )
    for shape in rows_by_shape:
        rows_by_shape[shape].sort(key=lambda r: r["threads"])

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    cube = rows_by_shape["cube"]
    tall = rows_by_shape["tall"]
    ax.plot(
        [r["threads"] for r in cube], [r["steps_per_s"] for r in cube],
        marker="o", markersize=7, linewidth=2, color=CUBE_COLOR,
        label="256³ cube (large halo-exchange plane)",
    )
    ax.plot(
        [r["threads"] for r in tall], [r["steps_per_s"] for r in tall],
        marker="s", markersize=7, linewidth=2, color=TALL_COLOR,
        label="64×64×4096 (same voxels, 16× smaller plane)",
    )

    ax.set_xlabel("rayon threads", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("steps/s", color=TEXT_COLOR, fontsize=11)
    ax.set_title(
        "Thread scaling: halo-exchange plane size dominates, not core count",
        color=TEXT_COLOR, fontsize=12, pad=12,
    )
    ax.set_ylim(bottom=0)

    ax.grid(True, which="both", linewidth=0.6, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    legend = ax.legend(loc="lower left", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PATH}")
    for shape, rows in rows_by_shape.items():
        one = next(r["steps_per_s"] for r in rows if r["threads"] == 1)
        last = rows[-1]
        print(
            f"{shape}: {one:.2f} steps/s @ 1 thread -> {last['steps_per_s']:.2f} steps/s "
            f"@ {last['threads']} threads ({last['steps_per_s'] / one:.3f}x)"
        )


if __name__ == "__main__":
    main()
