#!/usr/bin/env python3
"""Render appendix figures for the Case-1 and Case-3 tangent-oracle tests.

The horizontal axis is the prescribed closure-tail discrepancy evaluated on
the linear-PROM reference trajectory.  For state-dependent closures, that is
not generally identical to the tail discrepancy seen along the new online
trajectory, so the distinction is retained in the manuscript text.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
DIAGNOSTIC_ROOT = PAPER / "tmp_case13_tangent_oracle_sensitivity"
FIGURE_ROOT = PAPER / "Figures" / "appendix"
MAX_REFERENCE_TAIL_PERCENT = 50.0

POINTS = (
    ("verification", r"$\mu^{(v)}$: $(4.875,0.0225)$"),
    ("offgrid1", r"$\mu^{(1)}$: $(4.560,0.0190)$"),
    ("offgrid2", r"$\mu^{(2)}$: $(5.190,0.0260)$"),
    ("extrapolation20pct", r"$\mu^{(3)}$: $(4.000,0.0330)$"),
)
POINT_COLORS = {
    "verification": "#1F77B4",
    "offgrid1": "#FF7F0E",
    "offgrid2": "#2CA02C",
    "extrapolation20pct": "#9467BD",
}

# Fixed ranges make the Case-1 and Case-3 figures directly comparable.
STATE_YLIM = (-0.15, 11.5)
PRIMARY_YLIM = (-0.25, 15.2)

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.85,
    }
)


def read_rows(case: str) -> dict[str, list[dict[str, float | str]]]:
    path = DIAGNOSTIC_ROOT / case / f"{case}_tangent_oracle_sensitivity_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                row[key] = value if key in {"case", "point"} else float(value)
            if float(row["reference_secondary_error_percent"]) <= MAX_REFERENCE_TAIL_PERCENT + 1.0e-9:
                grouped[str(row["point"])].append(row)
    for point_key, rows in grouped.items():
        rows.sort(key=lambda row: float(row["reference_secondary_error_percent"]))
        if not rows:
            raise ValueError(f"No retained rows for {case}/{point_key}")
    missing = [key for key, _ in POINTS if key not in grouped]
    if missing:
        raise ValueError(f"{case} diagnostic is missing points: {missing}")
    return grouped


def native_ann_row(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    ann_error = float(rows[0]["ann_secondary_error_percent"])
    return min(rows, key=lambda row: abs(float(row["requested_secondary_error_percent"]) - ann_error))


def plot_case(case: str) -> Path:
    grouped = read_rows(case)
    figure, axes = plt.subplots(len(POINTS), 2, figsize=(12.6, 12.8), sharex=True)
    columns = (
        ("state_error_percent_vs_hdm", r"state error against HDM (\%)"),
        ("primary_q_error_percent_vs_linear_prom", r"solved-primary error against linear PROM (\%)"),
    )
    for row_index, (point_key, point_label) in enumerate(POINTS):
        rows = grouped[point_key]
        x = np.asarray([float(row["reference_secondary_error_percent"]) for row in rows])
        native = native_ann_row(rows)
        for column_index, (metric, ylabel) in enumerate(columns):
            axis = axes[row_index, column_index]
            y = np.asarray([float(row[metric]) for row in rows])
            axis.plot(x, y, marker="o", markersize=3.8, color=POINT_COLORS[point_key])
            axis.scatter(
                [float(native["reference_secondary_error_percent"])],
                [float(native[metric])],
                marker="*",
                s=92,
                color=POINT_COLORS[point_key],
                edgecolor="#222222",
                linewidth=0.55,
                zorder=5,
            )
            axis.set_xlim(-1.0, MAX_REFERENCE_TAIL_PERCENT + 1.0)
            axis.grid(True, alpha=0.26)
            axis.text(
                0.98,
                0.93,
                point_label,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.6},
            )
            if column_index == 0:
                axis.set_ylabel(ylabel)
                axis.set_ylim(*STATE_YLIM)
            else:
                axis.set_ylim(*PRIMARY_YLIM)
            if row_index == 0:
                axis.set_title(ylabel)
            if row_index == len(POINTS) - 1:
                axis.set_xlabel(r"reference-path closure-tail error (\%)")

    line_handles = [
        Line2D([0], [0], color=POINT_COLORS[key], marker="o", markersize=4, label=label.split(":")[0])
        for key, label in POINTS
    ]
    line_handles.append(
        Line2D([0], [0], color="#333333", marker="*", linestyle="none", markersize=9, label="native ANN tail")
    )
    figure.legend(
        handles=line_handles,
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )
    display_case = "Case--1" if case == "case1" else "Case--3"
    figure.suptitle(
        rf"{display_case}: tangent-preserving oracle closure-tail diagnostic",
        y=1.055,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.972), h_pad=1.15, w_pad=1.4)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    output = FIGURE_ROOT / f"prom_{case}_tangent_oracle_sensitivity.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    outputs = [plot_case("case1"), plot_case("case3")]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
