#!/usr/bin/env python3
"""PROM-first master-ANN diagnostic.

This diagnostic reads existing PROM and ANN outputs only.  It does not run any
online solver and does not modify manuscript.tex.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
RUNS = PAPER / "mlspg_prom_main" / "Runs"
SENS = PAPER / "tmp_case2_secondary_sensitivity"
OUT = PAPER / "Prom_MasterANN_Diagnostic"
FIG_DIR = OUT / "figures"
NTOT = 151

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    }
)


@dataclass(frozen=True)
class Point:
    key: str
    label: str
    mu1: float
    mu2: float


POINTS = (
    Point("verification", r"$\mu^{(v)}$", 4.875, 0.0225),
    Point("offgrid1", r"$\mu^{(1)}$", 4.560, 0.0190),
    Point("offgrid2", r"$\mu^{(2)}$", 5.190, 0.0260),
    Point("extrapolation20pct", r"$\mu^{(3)}$", 4.000, 0.0330),
)
POINT_COLORS = {
    "verification": "#4c78a8",
    "offgrid1": "#f58518",
    "offgrid2": "#54a24b",
    "extrapolation20pct": "#b279a2",
}
N_SWEEP = (0, 3, 5, 10, 20, 30, 50, 100, 151)
N_COEFF_SWEEP = tuple(n for n in N_SWEEP if n != NTOT)


def mu_tag(point: Point) -> str:
    return f"mu1_{point.mu1:.3f}_mu2_{point.mu2:.4f}"


def linear_dir(point: Point) -> Path:
    return RUNS / "Linear" / f"linear_prom_{mu_tag(point)}_ntot151"


def data_dir(point: Point) -> Path:
    return RUNS / "ROM" / "DataDriven_MasterANN" / f"rom_data_driven_{mu_tag(point)}_ntot151"


def case2_sweep_dir(n: int) -> Path:
    return RUNS / "PROM" / "Case2_MasterANN_NSweep" / f"np{n}"


def case2_sweep_stem(point: Point, n: int) -> str:
    return f"case2_prom_ann_master_qtot_{mu_tag(point)}_n{n}_ntot151"


def q_path(method: str, point: Point) -> Path:
    if method == "linear":
        return linear_dir(point) / "qN.npy"
    if method == "data_driven":
        return data_dir(point) / "qN.npy"
    raise KeyError(method)


def summary_path(method: str, point: Point) -> Path:
    if method == "linear":
        return linear_dir(point) / "summary.txt"
    if method == "data_driven":
        return data_dir(point) / "rom_data_driven_summary.txt"
    raise KeyError(method)


def read_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def load_q(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    q = np.load(path, allow_pickle=False)
    if q.shape != (NTOT, 501):
        raise ValueError(f"Unexpected qN shape {q.shape}: {path}")
    return np.asarray(q, dtype=np.float64)


def rel_frob(q: np.ndarray, q_ref: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref))


def coeff_curves(q: np.ndarray, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = q - q_ref
    abs_curve = np.linalg.norm(diff, axis=1)
    denom = np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
    rel_curve = 100.0 * abs_curve / denom
    return abs_curve, rel_curve


def n_sweep_summary_path(n: int, point: Point) -> Path:
    if n == 0:
        return summary_path("data_driven", point)
    if n == NTOT:
        return summary_path("linear", point)
    return case2_sweep_dir(n) / f"{case2_sweep_stem(point, n)}_summary.txt"


def n_sweep_q_path(n: int, point: Point) -> Path:
    if n == 0:
        return q_path("data_driven", point)
    if n == NTOT:
        return q_path("linear", point)
    return case2_sweep_dir(n) / f"{case2_sweep_stem(point, n)}_qN.npy"


def n_sweep_state_errors() -> list[dict[str, object]]:
    rows = []
    for n in N_SWEEP:
        for p in POINTS:
            kv = read_kv(n_sweep_summary_path(n, p))
            rows.append(
                {
                    "n": n,
                    "point": p.key,
                    "label": p.label,
                    "mu1": p.mu1,
                    "mu2": p.mu2,
                    "relative_error_percent": float(kv["relative_error_percent"]),
                }
            )
    return rows


def n_sweep_coefficient_errors() -> list[dict[str, object]]:
    rows = []
    for n in N_SWEEP:
        for p in POINTS:
            q_ref = load_q(q_path("linear", p))
            q = load_q(n_sweep_q_path(n, p))
            rows.append(
                {
                    "n": n,
                    "point": p.key,
                    "label": p.label,
                    "mu1": p.mu1,
                    "mu2": p.mu2,
                    "relative_q_error_percent": rel_frob(q, q_ref),
                }
            )
    return rows


def read_secondary_sensitivity_rows() -> list[dict[str, object]]:
    src = SENS / "case2_secondary_sensitivity_summary.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing secondary sensitivity CSV: {src}\n"
            "Run run_case2_secondary_sensitivity_tmp.py first."
        )
    point_order = {p.key: i for i, p in enumerate(POINTS)}
    rows: list[dict[str, object]] = []
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item: dict[str, object] = dict(r)
            for key in (
                "mu1",
                "mu2",
                "requested_secondary_error_percent",
                "actual_secondary_error_percent",
                "ann_secondary_error_percent",
                "state_error_percent_vs_hdm",
                "state_error_percent_vs_linear_prom",
                "primary_q_error_percent_vs_linear_prom",
                "total_q_error_percent_vs_linear_prom",
            ):
                item[key] = float(r[key])
            item["n_primary"] = int(r["n_primary"])
            item["n_tot"] = int(r["n_tot"])
            rows.append(item)
    rows.sort(key=lambda r: (point_order[str(r["point"])], float(r["actual_secondary_error_percent"])))
    return rows


def write_csvs(
    n_sweep_rows: list[dict[str, object]],
    n_sweep_coeff_rows: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    n_sweep_csv = OUT / "case2_n_sweep_state_errors.csv"
    n_sweep_coeff_csv = OUT / "case2_n_sweep_coeff_errors_vs_linear_prom.csv"
    sensitivity_csv = OUT / "case2_secondary_sensitivity_summary.csv"

    with n_sweep_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "point", "mu1", "mu2", "relative_error_percent"],
        )
        writer.writeheader()
        for r in n_sweep_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    with n_sweep_coeff_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "point", "mu1", "mu2", "relative_q_error_percent"],
        )
        writer.writeheader()
        for r in n_sweep_coeff_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    shutil.copyfile(SENS / "case2_secondary_sensitivity_summary.csv", sensitivity_csv)
    return n_sweep_csv, n_sweep_coeff_csv, sensitivity_csv


def plot_n_sweep_state_errors(rows: list[dict[str, object]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ns = np.asarray(N_SWEEP, dtype=float)
    all_vals = []
    for p in POINTS:
        vals = [float(r["relative_error_percent"]) for r in rows if r["point"] == p.key]
        if len(vals) != len(N_SWEEP):
            raise ValueError(f"Missing n-sweep values for {p.key}")
        all_vals.append(vals)
        ax.plot(
            ns,
            vals,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=POINT_COLORS[p.key],
            label=rf"{p.label}: $\mu=({p.mu1:.3f},{p.mu2:.4f})$",
        )

    mean_curve = np.mean(np.asarray(all_vals, dtype=float), axis=0)
    ax.plot(ns, mean_curve, marker="s", linewidth=2.7, markersize=5.5, color="black", label="Mean")
    ax.axvline(0, color="0.55", linewidth=1.0, linestyle=":")
    ax.axvline(NTOT, color="0.55", linewidth=1.0, linestyle=":")
    ax.set_xticks(list(N_SWEEP))
    ax.set_xlim(-4, NTOT + 4)
    ax.set_xlabel(r"solved PROM dimension $n$")
    ax.set_ylabel("state relative error against HDM (%)")
    ax.set_title("Case 2 master-ANN sweep: state error vs solved dimension")
    ax.grid(True, which="major", alpha=0.30)
    ax.legend(loc="upper right", fontsize=8.7, frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "prom_case2_n_sweep_state_errors.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_n_sweep_coeff_curves() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, NTOT + 1)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(N_COEFF_SWEEP)))
    fig, axes = plt.subplots(2, len(POINTS), figsize=(18.0, 8.2), sharex=True)

    for c, p in enumerate(POINTS):
        q_ref = load_q(q_path("linear", p))
        ax_abs, ax_rel = axes[0, c], axes[1, c]
        for color, n in zip(colors, N_COEFF_SWEEP):
            q = load_q(n_sweep_q_path(n, p))
            abs_curve, rel_curve = coeff_curves(q, q_ref)
            label = r"$n=0$ data-driven" if n == 0 else rf"$n={n}$"
            ax_abs.semilogy(
                x,
                abs_curve + 1.0e-14,
                color=color,
                linewidth=2.4 if n == 0 else 1.65,
                label=label if c == 0 else None,
            )
            ax_rel.semilogy(
                x,
                rel_curve / 100.0 + 1.0e-14,
                color=color,
                linewidth=2.4 if n == 0 else 1.65,
            )

        for ax in (ax_abs, ax_rel):
            ax.axvline(10, color="#333333", linewidth=1.0, linestyle="--", alpha=0.65)
            ax.grid(True, which="major", alpha=0.32)
            ax.set_xlim(1, NTOT)
        ax_abs.set_title(rf"{p.label}: $\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax_rel.set_xlabel(r"coefficient index $i$")

    axes[0, 0].set_ylabel(r"$\|q_i-q_i^{\mathrm{PROM}}\|_2$")
    axes[1, 0].set_ylabel(r"$\|q_i-q_i^{\mathrm{PROM}}\|_2/\|q_i^{\mathrm{PROM}}\|_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Case 2 master-ANN sweep: coefficient errors vs linear PROM reference", y=1.075)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.6, h_pad=1.0)
    out = FIG_DIR / "prom_case2_n_sweep_coeff_abs_rel_all_points.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_secondary_sensitivity(rows: list[dict[str, object]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)

    for p in POINTS:
        subset = [r for r in rows if r["point"] == p.key]
        x = np.asarray([float(r["actual_secondary_error_percent"]) for r in subset])
        uerr = np.asarray([float(r["state_error_percent_vs_hdm"]) for r in subset])
        qerr = np.asarray([float(r["primary_q_error_percent_vs_linear_prom"]) for r in subset])
        ann_err = float(subset[0]["ann_secondary_error_percent"])
        idx_ann = int(np.argmin(np.abs(x - ann_err)))
        label = rf"{p.label}: $\mu=({p.mu1:.3f},{p.mu2:.4f})$"

        for ax, y in ((axes[0], uerr), (axes[1], qerr)):
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.8, color=POINT_COLORS[p.key], label=label)
            ax.scatter([x[idx_ann]], [y[idx_ann]], marker="*", s=125, color=POINT_COLORS[p.key], edgecolor="black", linewidth=0.45, zorder=5)

    axes[0].set_ylabel("state relative error against HDM (%)")
    axes[1].set_ylabel("primary coefficient error vs linear PROM (%)")
    for ax in axes:
        ax.set_xlabel(r"imposed relative error in $q_{11:151}$ only (%)")
        ax.grid(True, which="major", alpha=0.30)
        ax.set_xlim(left=-0.4)
    axes[0].set_title(r"Effect on state error $\|u_{HDM}-u_{approx}\|/\|u_{HDM}\|$")
    axes[1].set_title(r"Effect on solved coordinates $q_1,\ldots,q_{10}$")
    axes[0].legend(loc="upper left", fontsize=8.0, frameon=True)
    fig.suptitle("Case 2 n=10: sensitivity to prescribed secondary-coordinate error", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "case2_secondary_sensitivity_state_and_primary_error.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def fmt(x: float) -> str:
    return f"{x:.3f}"


def latex_n_sweep_table(rows: list[dict[str, object]], value_key: str, value_label: str) -> str:
    lines = [
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        rf"$n$ & {POINTS[0].label} & {POINTS[1].label} & {POINTS[2].label} & {POINTS[3].label} & Mean \\",
        r"\midrule",
    ]
    for n in N_SWEEP:
        vals = []
        for p in POINTS:
            matches = [r for r in rows if r["n"] == n and r["point"] == p.key]
            if len(matches) != 1:
                raise ValueError(f"Expected one row for n={n}, point={p.key}; got {len(matches)}")
            vals.append(float(matches[0][value_key]))
        mean_val = float(np.mean(vals))
        lines.append(rf"{n} & {fmt(vals[0])}\% & {fmt(vals[1])}\% & {fmt(vals[2])}\% & {fmt(vals[3])}\% & {fmt(mean_val)}\% \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\captionof{{table}}{{{value_label}}}",
    ]
    return "\n".join(lines)


def write_tex(
    n_sweep_rows: list[dict[str, object]],
    n_sweep_coeff_rows: list[dict[str, object]],
    n_sweep_fig_path: Path,
    n_sweep_coeff_fig_path: Path,
    sensitivity_fig_path: Path,
) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rel_n_sweep_fig = n_sweep_fig_path.relative_to(OUT)
    rel_n_sweep_coeff_fig = n_sweep_coeff_fig_path.relative_to(OUT)
    rel_sens_fig = sensitivity_fig_path.relative_to(OUT)
    tex = OUT / "prom_master_ann_diagnostic.tex"
    content = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath,bm}
\usepackage{caption}

\title{PROM-first master-ANN diagnostic}
\date{}

\begin{document}
\maketitle

\section*{Setup}
This note analyzes the PROM-first version of the master ANN experiment.  The
linear PROM with $n_{\mathrm{tot}}=151$ is used as the coefficient reference.
The case $n=0$ denotes the purely data-driven prediction
$\mathbf q_{\mathrm{tot}}=\mathcal G(\bm\mu,t)$, while intermediate Case~2
runs solve the first $n$ PROM coordinates and inject the remaining
coordinates from the same master ANN:
\[
\widetilde{\mathbf u}
=
\mathbf u_{\mathrm{ref}}
+\mathbf V_{1:n}\mathbf q_{1:n}
+\mathbf V_{n+1:151}\mathcal G_{n+1:151}(\bm\mu,t).
\]
This file is separate from the manuscript and is meant only to inspect the
current PROM-level evidence.

\section*{Case 2 sweep in solved PROM dimension}
The following tables and figures show how the state error and the full
coefficient-trajectory error vary as the number of solved coordinates grows
from the data-driven limit ($n=0$) to the linear PROM reference ($n=151$).

\begin{center}
\small
__N_SWEEP_STATE_TABLE__
\end{center}

\begin{center}
\small
__N_SWEEP_COEFF_TABLE__
\end{center}

\begin{figure}[h!]
\centering
\includegraphics[width=0.88\textwidth]{__N_SWEEP_FIG_PATH__}
\caption{State relative error against HDM as a function of the number of PROM coordinates solved online.}
\end{figure}

\begin{figure}[h!]
\centering
\includegraphics[width=0.98\textwidth]{__N_SWEEP_COEFF_FIG_PATH__}
\caption{Per-coefficient absolute and relative errors for the same $n$-sweep.  The linear PROM case $n=151$ is not plotted because it is the zero-error coefficient reference.}
\end{figure}

\clearpage
\section*{Case 2 $n=10$: sensitivity to secondary-coordinate error}
This diagnostic isolates one mechanism in Case~2.  The secondary coordinates
$q_{11},\ldots,q_{151}$ are prescribed from the linear PROM and then perturbed
in the direction of the actual master-ANN error.  The horizontal axis is the
imposed relative error in those prescribed secondary coordinates.  The star on
each curve marks the actual ANN secondary error for that test point.
This is not the global validation error of the ANN.  The trained master ANN has
a full-trajectory validation error of about $4.39\%$ on
$\mathbf q_{\mathrm{tot}}$, whereas the horizontal-axis values here measure
only
\[
\frac{\|q_{11:151}^{ANN}-q_{11:151}^{PROM}\|_F}
{\|q_{11:151}^{PROM}\|_F}
\]
on the four test points.  Since the secondary block carries only about
$20\%$ of the full coefficient norm, this secondary-only relative error can be
much larger than the full $\mathbf q_{\mathrm{tot}}$ error.

\begin{figure}[h!]
\centering
\includegraphics[width=0.98\textwidth]{__SENSITIVITY_FIG_PATH__}
\caption{Effect of the prescribed error in $q_{11},\ldots,q_{151}$ on the state error and on the recovered primary coefficients for Case~2 with $n=10$.  The horizontal axis is a secondary-block error only, not the global ANN validation error.  At zero perturbation, Case~2 uses the exact linear-PROM secondary coordinates; therefore it should recover the linear PROM lower-bound behavior up to nonlinear-solver tolerance.}
\end{figure}

\section*{Reading of the diagnostic}
The $n$-sweep tests whether solving more PROM coordinates compensates for the
master-ANN coefficient error.  The perturbation test checks the same question
more directly for fixed $n=10$: if the prescribed secondary coefficients are
made exact, the recovered primary coordinates stay close to the linear PROM
trajectory; as the secondary-coordinate perturbation grows, both the state
error and the primary-coordinate error increase.

\end{document}
"""
    content = content.replace(
        "__N_SWEEP_STATE_TABLE__",
        latex_n_sweep_table(
            n_sweep_rows,
            "relative_error_percent",
            r"State relative error against HDM for the Case~2 $n$-sweep.",
        ),
    )
    content = content.replace(
        "__N_SWEEP_COEFF_TABLE__",
        latex_n_sweep_table(
            n_sweep_coeff_rows,
            "relative_q_error_percent",
            r"Full coefficient-trajectory error against the linear PROM reference for the same $n$-sweep.",
        ),
    )
    content = content.replace("__N_SWEEP_FIG_PATH__", rel_n_sweep_fig.as_posix())
    content = content.replace("__N_SWEEP_COEFF_FIG_PATH__", rel_n_sweep_coeff_fig.as_posix())
    content = content.replace("__SENSITIVITY_FIG_PATH__", rel_sens_fig.as_posix())
    tex.write_text(content)
    return tex


def main() -> None:
    n_sweep_rows = n_sweep_state_errors()
    n_sweep_coeff_rows = n_sweep_coefficient_errors()
    sensitivity_rows = read_secondary_sensitivity_rows()
    n_sweep_fig = plot_n_sweep_state_errors(n_sweep_rows)
    n_sweep_coeff_fig = plot_n_sweep_coeff_curves()
    sensitivity_fig = plot_secondary_sensitivity(sensitivity_rows)
    n_sweep_csv, n_sweep_coeff_csv, sensitivity_csv = write_csvs(n_sweep_rows, n_sweep_coeff_rows)
    tex = write_tex(
        n_sweep_rows,
        n_sweep_coeff_rows,
        n_sweep_fig,
        n_sweep_coeff_fig,
        sensitivity_fig,
    )
    print(f"[figure] {n_sweep_fig}")
    print(f"[figure] {n_sweep_coeff_fig}")
    print(f"[figure] {sensitivity_fig}")
    print(f"[csv] {n_sweep_csv}")
    print(f"[csv] {n_sweep_coeff_csv}")
    print(f"[csv] {sensitivity_csv}")
    print(f"[tex] {tex}")


if __name__ == "__main__":
    main()
