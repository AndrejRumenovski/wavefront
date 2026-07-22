#!/usr/bin/env python3
"""Plots validation/pml_reflection_data.csv (written by
`cargo run --release --example pml_reflection_study`) as a log-linear
reflection coefficient vs. PML thickness figure.

Usage:
    python3 validation/plot_pml_reflection.py

Requires matplotlib (not a crate dependency -- this is a one-off analysis
script, not part of the Rust build).
"""

import csv
import pathlib

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
CSV_PATH = HERE / "pml_reflection_data.csv"
OUT_PATH = HERE / "pml_reflection.png"

# Same categorical palette as plot_convergence.py.
MEASURED_COLOR = "#2a78d6"
TARGET_COLOR = "#eb6834"
GRID_COLOR = "#d8d8d4"
TEXT_COLOR = "#26261f"

# PmlConfig::default().target_reflection -- kept in sync manually with
# src/layout.rs since this is a plotting script, not Rust.
TARGET_REFLECTION = 1.0e-6


def main():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "thickness_blocks": int(row["thickness_blocks"]),
                    "thickness_voxels": int(row["thickness_voxels"]),
                    "reflection_coeff": float(row["reflection_coeff"]),
                }
            )
    rows.sort(key=lambda r: r["thickness_voxels"])

    thickness = [r["thickness_voxels"] for r in rows]
    reflection = [r["reflection_coeff"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.semilogy(
        thickness, reflection, marker="o", markersize=7, linewidth=2,
        color=MEASURED_COLOR, label="Measured |R| (two-run subtraction + quadrature demodulation)",
    )
    ax.axhline(
        TARGET_REFLECTION, linestyle="--", linewidth=1.5, color=TARGET_COLOR,
        label=f"PmlConfig::default target R0 = {TARGET_REFLECTION:.0e}",
    )

    ax.set_xlabel("PML thickness (voxels)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Reflection coefficient |R|", color=TEXT_COLOR, fontsize=11)
    ax.set_title(
        "CPML absorbing boundary: measured reflection vs. layer thickness",
        color=TEXT_COLOR, fontsize=12, pad=12,
    )

    for x, y, r in zip(thickness, reflection, rows):
        ax.annotate(
            f"{r['thickness_blocks']} block{'s' if r['thickness_blocks'] != 1 else ''}",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#52514e",
        )

    ax.grid(True, which="both", linewidth=0.6, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    legend = ax.legend(loc="upper right", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PATH}")
    for r in rows:
        print(f"{r['thickness_voxels']} voxels: |R| = {r['reflection_coeff']:.3e}")


if __name__ == "__main__":
    main()
