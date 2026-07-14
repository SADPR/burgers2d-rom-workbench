#!/usr/bin/env python3
"""Generate matched PROM--HPROM Case--2 diagnostic appendix figures.

The PROM and HPROM diagnostics each use their own full linear reference
trajectory.  This script compares their response to the same prescribed
secondary-coordinate error and their Case--2 primary-dimension sweeps without
mixing those references.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from generate_hprom_baseline_assets import CASE2_N_SWEEP, POINTS, case2_diagnostic_paths
from generate_prom_master_ann_diagnostic import (
    POINTS as PROM_POINTS,
    coeff_curves as prom_coeff_curves,
    load_q as load_prom_q,
    n_sweep_q_path as prom_n_sweep_q_path,
    q_path as prom_q_path,
)
from manuscript_plot_style import CASE2_SWEEP_COLORS, METHOD_COLORS


SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
PROM_DIAGNOSTIC = PAPER / "Prom_MasterANN_Diagnostic"
HPROM_DIAGNOSTIC = PAPER / "tmp_case2_hprom_diagnostics"
FIG_DIR = PAPER / "Figures" / "appendix"

PROM_COLOR = METHOD_COLORS["case2_n10"]
PROM_STYLE = "-"
HPROM_STYLE = "--"
MAX_TAIL_PERCENT = 50.0

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def read_csv_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                if key == "point":
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def point_title(point: object) -> str:
    return rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$"


def grouped_tail_rows(path: Path, primary_key: str) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        if float(row["actual_secondary_error_percent"]) <= MAX_TAIL_PERCENT + 1.0e-10:
            if primary_key not in row:
                raise KeyError(f"Missing {primary_key} in {path}")
            grouped[str(row["point"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["actual_secondary_error_percent"]))
    missing = [point.key for point in POINTS if point.key not in grouped]
    if missing:
        raise ValueError(f"Tail diagnostic missing points: {missing}")
    return grouped


def nearest_ann_row(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    ann_tail = float(rows[0]["ann_secondary_error_percent"])
    return min(rows, key=lambda row: abs(float(row["actual_secondary_error_percent"]) - ann_tail))


def plot_tail_sensitivity() -> Path:
    """Plot matched n=10 secondary-tail diagnostics up to 50 percent."""
    prom = grouped_tail_rows(
        PROM_DIAGNOSTIC / "case2_secondary_sensitivity_summary.csv",
        "primary_q_error_percent_vs_linear_prom",
    )
    hprom = grouped_tail_rows(
        HPROM_DIAGNOSTIC / "secondary_sensitivity_n10" / "case2_secondary_sensitivity_summary.csv",
        "primary_q_error_percent_vs_linear_hprom",
    )
    figure, axes = plt.subplots(len(POINTS), 2, figsize=(12.8, 13.0), sharex=True)
    for row_index, point in enumerate(POINTS):
        prom_rows = prom[point.key]
        hprom_rows = hprom[point.key]
        for column, (metric, ylabel) in enumerate((
            ("state", r"state error against HDM (\%)"),
            ("primary", r"primary-coordinate error (\%)"),
        )):
            ax = axes[row_index, column]
            if metric == "state":
                prom_key = "state_error_percent_vs_hdm"
                hprom_key = "state_error_percent_vs_hdm"
            else:
                prom_key = "primary_q_error_percent_vs_linear_prom"
                hprom_key = "primary_q_error_percent_vs_linear_hprom"
            for rows, key, style, marker, label in (
                (prom_rows, prom_key, PROM_STYLE, "o", "PROM"),
                (hprom_rows, hprom_key, HPROM_STYLE, "s", "HPROM"),
            ):
                tail = np.asarray([float(item["actual_secondary_error_percent"]) for item in rows])
                value = np.asarray([float(item[key]) for item in rows])
                ax.plot(
                    tail, value, color=PROM_COLOR, ls=style, marker=marker,
                    markersize=4.1, lw=1.8, alpha=0.96,
                    label=label if row_index == 0 and column == 0 else None,
                )
                ann_row = nearest_ann_row(rows)
                ann_x = float(ann_row["actual_secondary_error_percent"])
                ann_y = float(ann_row[key])
                ax.scatter(
                    ann_x, ann_y, marker="*" if label == "PROM" else "X", s=80,
                    color=PROM_COLOR, edgecolor="#222222", linewidth=0.55, zorder=5,
                    label=("actual PROM tail" if label == "PROM" else "actual HPROM tail")
                    if row_index == 0 and column == 0 else None,
                )
            ax.grid(True, alpha=0.26)
            ax.set_xlim(-1.0, MAX_TAIL_PERCENT + 1.0)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(
                    r"state error $\|u_{\mathrm{HDM}}-u\|_F/\|u_{\mathrm{HDM}}\|_F$"
                    if column == 0 else r"solved-primary error $\|q_{1:10}-q_{1:10}^{\rm lin}\|_F/\|q_{1:10}^{\rm lin}\|_F$"
                )
            if row_index == len(POINTS) - 1:
                ax.set_xlabel(r"imposed relative error in $q_{11:151}$ (\%)")
            ax.text(0.985, 0.94, point_title(point), transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.3,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.7})

    for ax in axes[:, 0]:
        ax.set_ylim(-0.5, 19.5)
    for ax in axes[:, 1]:
        ax.set_ylim(-1.0, 45.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    figure.suptitle(r"Case--2, $n=10$: matched PROM--HPROM secondary-tail sensitivity", y=1.025)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.15, w_pad=1.4)
    output = FIG_DIR / "prom_hprom_case2_n10_tail_sensitivity.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def hprom_coeff_curves(q: np.ndarray, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    difference = q - q_ref
    absolute = np.linalg.norm(difference, axis=1)
    relative = 100.0 * absolute / np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
    return absolute, relative


def log_limits(curves: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([curve[np.isfinite(curve) & (curve > 0.0)] for curve in curves])
    return (
        10.0 ** np.floor(np.log10(np.min(values))),
        10.0 ** np.ceil(np.log10(np.max(values))),
    )


def plot_n_sweep_coefficient_comparison() -> Path:
    """Compare PROM and HPROM coefficient errors for each common Case--2 n."""
    n_values = tuple(n for n in CASE2_N_SWEEP if n < 151)
    prom_points = {point.key: point for point in PROM_POINTS}
    x = np.arange(1, 152)
    figure, axes = plt.subplots(2, len(POINTS), figsize=(17.6, 8.2), sharex=True)
    absolute_curves: list[np.ndarray] = []
    relative_curves: list[np.ndarray] = []

    for column, point in enumerate(POINTS):
        prom_point = prom_points[point.key]
        prom_reference = load_prom_q(prom_q_path("linear", prom_point))
        _, hprom_reference_path = case2_diagnostic_paths(151, point)
        hprom_reference = np.asarray(np.load(hprom_reference_path), dtype=np.float64)
        ax_absolute, ax_relative = axes[0, column], axes[1, column]

        for n in n_values:
            prom_q = load_prom_q(prom_n_sweep_q_path(n, prom_point))
            _, hprom_q_path = case2_diagnostic_paths(n, point)
            hprom_q = np.asarray(np.load(hprom_q_path), dtype=np.float64)
            prom_absolute, prom_relative = prom_coeff_curves(prom_q, prom_reference)
            hprom_absolute, hprom_relative = hprom_coeff_curves(hprom_q, hprom_reference)
            color = CASE2_SWEEP_COLORS[n]
            label = r"$n=0$ direct" if n == 0 else rf"$n={n}$"
            ax_absolute.semilogy(
                x, np.maximum(prom_absolute, 1.0e-14), color=color, ls=PROM_STYLE,
                lw=2.15 if n == 0 else 1.5, alpha=0.96,
                label=label if column == 0 else None,
            )
            ax_absolute.semilogy(
                x, np.maximum(hprom_absolute, 1.0e-14), color=color, ls=HPROM_STYLE,
                lw=1.35, alpha=0.96,
            )
            ax_relative.semilogy(
                x, np.maximum(prom_relative, 1.0e-14), color=color, ls=PROM_STYLE,
                lw=2.15 if n == 0 else 1.5, alpha=0.96,
            )
            ax_relative.semilogy(
                x, np.maximum(hprom_relative, 1.0e-14), color=color, ls=HPROM_STYLE,
                lw=1.35, alpha=0.96,
            )
            absolute_curves.extend((prom_absolute, hprom_absolute))
            relative_curves.extend((prom_relative, hprom_relative))

        for axis in (ax_absolute, ax_relative):
            axis.axvline(10, color="0.25", ls="--", lw=0.95, alpha=0.68)
            axis.grid(True, which="both", alpha=0.24)
            axis.set_xlim(1, 151)
        ax_absolute.set_title(point_title(point))
        ax_relative.set_xlabel(r"coefficient index $i$")

    abs_limits = log_limits(absolute_curves)
    rel_limits = log_limits(relative_curves)
    for axis in axes[0]:
        axis.set_ylim(*abs_limits)
    for axis in axes[1]:
        axis.set_ylim(*rel_limits)
    axes[0, 0].set_ylabel(r"$\|q_i-q_i^{\mathrm{lin}}\|_2$")
    axes[1, 0].set_ylabel(r"relative coefficient error (\%)")

    n_handles, _ = axes[0, 0].get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="#222222", ls=PROM_STYLE, lw=1.8, label="PROM"),
        Line2D([0], [0], color="#222222", ls=HPROM_STYLE, lw=1.8, label="HPROM"),
    ]
    figure.legend(
        style_handles + n_handles,
        ["PROM", "HPROM"] + [handle.get_label() for handle in n_handles],
        loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.015),
    )
    figure.suptitle(
        r"Case--2 coefficient-error comparison: PROM (solid) and HPROM (dashed)",
        y=1.08,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), w_pad=1.2, h_pad=1.0)
    output = FIG_DIR / "prom_hprom_case2_n_sweep_coefficients_comparison.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "prom_hprom_case2_n_sweep_comparison.png").unlink(missing_ok=True)
    outputs = (plot_tail_sensitivity(), plot_n_sweep_coefficient_comparison())
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
