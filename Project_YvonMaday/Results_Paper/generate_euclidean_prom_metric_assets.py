#!/usr/bin/env python3
"""Generate the Euclidean-versus-LSPG Case--2 PROM tail diagnostic figure.

Both datasets prescribe a relative error in the secondary coefficient block
and solve the same ten primary coordinates with the full residual.  The
coordinate bases and the learned tail-error directions are distinct, so the
figure compares controlled response curves rather than coefficient entries
across the two bases.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manuscript_plot_style import METHOD_COLORS


SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
LSPG_CSV = PAPER / "Prom_MasterANN_Diagnostic" / "case2_secondary_sensitivity_summary.csv"
EUCLIDEAN_CSV = (
    PAPER / "tmp_euclidean_case2_secondary_sensitivity" / "case2_secondary_sensitivity_summary.csv"
)
OUTPUT = PAPER / "Figures" / "appendix" / "euclidean_vs_lspg_case2_n10_tail_sensitivity.png"

POINTS = (
    ("verification", r"verification: $\mu=(4.875,0.0225)$"),
    ("offgrid1", r"off-grid: $\mu=(4.560,0.0190)$"),
    ("offgrid2", r"off-grid: $\mu=(5.190,0.0260)$"),
    ("extrapolation20pct", r"extrapolation: $\mu=(4.000,0.0330)$"),
)
LEVELS = (0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)

LSPG_COLOR = METHOD_COLORS["case1"]
EUCLIDEAN_COLOR = METHOD_COLORS["podnn"]

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
    }
)


def read_rows(path: Path) -> dict[str, list[dict[str, float | str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic summary: {path}")
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                row[key] = value if key == "point" else float(value)
            grouped[str(row["point"])].append(row)
    return grouped


def standard_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Return exactly the prescribed, shared levels, excluding ANN-level rows."""
    selected: list[dict[str, float | str]] = []
    for level in LEVELS:
        matches = [
            row
            for row in rows
            if abs(float(row["requested_secondary_error_percent"]) - level) < 1.0e-8
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one row at {level:g}% but found {len(matches)} in point {rows[0]['point']}."
            )
        selected.append(matches[0])
    return selected


def ann_row(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    target = float(rows[0]["ann_secondary_error_percent"])
    return min(rows, key=lambda row: abs(float(row["requested_secondary_error_percent"]) - target))


def metric_key(column: int) -> tuple[str, str]:
    if column == 0:
        return "state_error_percent_vs_hdm", r"state error against HDM (\%)"
    return "primary_q_error_percent_vs_linear_prom", r"primary-coordinate error (\%)"


def plot() -> Path:
    lspg = read_rows(LSPG_CSV)
    euclidean = read_rows(EUCLIDEAN_CSV)

    for key, _ in POINTS:
        if key not in lspg or key not in euclidean:
            raise ValueError(f"Missing point {key} in one of the diagnostics.")

    figure, axes = plt.subplots(len(POINTS), 2, figsize=(12.6, 12.8), sharex=True)
    all_values: list[list[float]] = [[], []]

    for row_index, (point_key, title) in enumerate(POINTS):
        datasets = (
            ("LSPG-sensitive POD", lspg[point_key], LSPG_COLOR, "o"),
            ("Euclidean POD", euclidean[point_key], EUCLIDEAN_COLOR, "s"),
        )
        for column in range(2):
            key, ylabel = metric_key(column)
            ax = axes[row_index, column]
            for label, rows, color, marker in datasets:
                curves = standard_rows(rows)
                x = np.asarray([float(row["actual_secondary_error_percent"]) for row in curves])
                y = np.asarray([float(row[key]) for row in curves])
                all_values[column].extend(y.tolist())
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=4.0,
                    lw=1.85,
                    alpha=0.96,
                    label=label if row_index == 0 and column == 0 else None,
                )

                actual = ann_row(rows)
                actual_x = float(actual["actual_secondary_error_percent"])
                actual_y = float(actual[key])
                ax.scatter(
                    actual_x,
                    actual_y,
                    marker="*",
                    s=80,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.5,
                    zorder=6,
                    label=(f"actual {label} tail" if row_index == 0 and column == 0 else None),
                )

            ax.set_xlim(-1.0, 51.0)
            ax.grid(True, alpha=0.26)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(
                    r"state error $\|u_{\mathrm{HDM}}-u\|_F/\|u_{\mathrm{HDM}}\|_F$"
                    if column == 0
                    else r"solved-primary error $\|q_{1:10}-q_{1:10}^{\mathrm{lin}}\|_F/\|q_{1:10}^{\mathrm{lin}}\|_F$"
                )
            if row_index == len(POINTS) - 1:
                ax.set_xlabel(r"imposed relative error in $q_{11:151}$ (\%)")
            ax.text(
                0.985,
                0.93,
                title,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
            )

    # Share a fixed, data-derived ordinate range down each column so the two
    # bases and all four parameters can be compared directly.
    for column, values in enumerate(all_values):
        upper = 1.08 * max(values)
        lower = -0.03 * upper
        for ax in axes[:, column]:
            ax.set_ylim(lower, upper)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    figure.suptitle(
        r"Case--2, $n=10$: Euclidean-POD and LSPG-sensitive-POD tail sensitivity",
        y=1.025,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.12, w_pad=1.35)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT


if __name__ == "__main__":
    print(plot())
