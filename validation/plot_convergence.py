#!/usr/bin/env python3
"""Plots validation/convergence_data.csv (written by
`cargo run --release --example convergence_study`) as a log-log phase
velocity error vs. grid resolution figure.

Usage:
    python3 validation/plot_convergence.py

Requires matplotlib (not a crate dependency -- this is a one-off analysis
script, not part of the Rust build).
"""

import csv
import math
import pathlib

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
CSV_PATH = HERE / "convergence_data.csv"
OUT_PATH = HERE / "convergence.png"

# Categorical palette (validated for colorblind- and contrast-safety; see
# the dataviz skill's reference palette): slot 1 blue, slot 2 orange.
MEASURED_COLOR = "#2a78d6"
THEORY_COLOR = "#eb6834"
GRID_COLOR = "#d8d8d4"
TEXT_COLOR = "#26261f"
REFERENCE_COLOR = "#9a9a90"


def log_log_slope(xs, ys):
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    n = len(lx)
    mean_x = sum(lx) / n
    mean_y = sum(ly) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(lx, ly))
    var = sum((x - mean_x) ** 2 for x in lx)
    return cov / var


def main():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["dx_m"])

    dx = [r["dx_m"] for r in rows]
    ppw = [r["points_per_wavelength"] for r in rows]
    measured_err = [r["measured_relative_error"] for r in rows]
    theory_err = [r["theoretical_relative_error"] for r in rows]

    measured_slope = log_log_slope(dx, measured_err)
    theory_slope = log_log_slope(dx, theory_err)

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.loglog(
        dx, theory_err, marker="s", markersize=7, linewidth=2,
        color=THEORY_COLOR, label=f"Closed-form Yee dispersion relation (slope {theory_slope:.2f})",
    )
    ax.loglog(
        dx, measured_err, marker="o", markersize=7, linewidth=2,
        color=MEASURED_COLOR, label=f"Measured (this simulator, quadrature demodulation) (slope {measured_slope:.2f})",
    )

    # O(dx^2) reference line, anchored to the finest measured point.
    ref_x = [min(dx), max(dx)]
    anchor_x, anchor_y = dx[0], theory_err[0]
    ref_y = [anchor_y * (x / anchor_x) ** 2 for x in ref_x]
    ax.loglog(ref_x, ref_y, linestyle="--", linewidth=1.5, color=REFERENCE_COLOR, label="O(dx²) reference", zorder=1)

    ax.set_xlabel("Cell size dx (m)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Relative phase velocity error |v_p − c| / c", color=TEXT_COLOR, fontsize=11)
    ax.set_title(
        "Yee-scheme numerical dispersion: measured vs. closed-form prediction",
        color=TEXT_COLOR, fontsize=12, pad=12,
    )

    for ppw_val, dx_val, err_val in zip(ppw, dx, measured_err):
        ax.annotate(
            f"{int(ppw_val)} pts/λ",
            xy=(dx_val, err_val),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#52514e",
        )

    ax.grid(True, which="both", linewidth=0.6, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    legend = ax.legend(loc="upper left", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PATH}")
    print(f"measured convergence order: {measured_slope:.3f}")
    print(f"theoretical convergence order: {theory_slope:.3f}")


if __name__ == "__main__":
    main()
