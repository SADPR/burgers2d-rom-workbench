#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot q_tot coefficient traces: linear PROM teacher vs data-driven master ANN."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_mu456_qtot_diag")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

MU_LABEL = "mu1_4.560_mu2_0.0190"
TRUE_QN = ROOT / "mlspg_prom_main" / "Runs" / "Linear" / f"linear_prom_{MU_LABEL}_ntot151" / "qN.npy"
PRED_QN = ROOT / "mlspg_prom_main" / "Runs" / "DataDriven_MasterANN" / f"rom_data_driven_{MU_LABEL}_ntot151" / "qN.npy"
PRED_T = ROOT / "mlspg_prom_main" / "Runs" / "DataDriven_MasterANN" / f"rom_data_driven_{MU_LABEL}_ntot151" / "t.npy"

OUT_DIR = ROOT / "Prom_MasterANN_Diagnostic" / "mu456_data_driven_qtot_coefficients"


def _load_q(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing qN file: {path}")
    q = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"Expected 2D qN array, got shape {q.shape} from {path}")
    # Standardize to (n_coeff, n_time).
    if q.shape[0] == 501 and q.shape[1] == 151:
        q = q.T
    if q.shape[0] != 151:
        raise ValueError(f"Expected 151 coefficients, got shape {q.shape} from {path}")
    return q


def _load_time(n_time: int) -> np.ndarray:
    if PRED_T.exists():
        t = np.asarray(np.load(PRED_T, allow_pickle=False), dtype=np.float64).reshape(-1)
        if t.size == n_time:
            return t
    return np.arange(n_time, dtype=np.float64)


def _write_errors_csv(path: Path, true_q: np.ndarray, pred_q: np.ndarray) -> list[dict[str, float]]:
    err = pred_q - true_q
    true_norm = np.linalg.norm(true_q, axis=1)
    err_norm = np.linalg.norm(err, axis=1)
    rel = np.full_like(err_norm, np.nan, dtype=np.float64)
    mask = true_norm > 1e-14
    rel[mask] = 100.0 * err_norm[mask] / true_norm[mask]

    total_true_energy = float(np.linalg.norm(true_q, ord="fro") ** 2)
    rows = []
    for j in range(true_q.shape[0]):
        rows.append(
            {
                "coefficient": j + 1,
                "true_l2_norm": float(true_norm[j]),
                "error_l2_norm": float(err_norm[j]),
                "relative_error_percent": float(rel[j]),
                "max_abs_error": float(np.max(np.abs(err[j]))),
                "rms_abs_error": float(np.sqrt(np.mean(err[j] ** 2))),
                "true_energy_share_percent": float(
                    100.0 * np.sum(true_q[j] ** 2) / total_true_energy
                    if total_true_energy > 0.0
                    else np.nan
                ),
            }
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot_overview(out_dir: Path, rows: list[dict[str, float]], global_rel: float) -> None:
    coeff = np.array([r["coefficient"] for r in rows], dtype=np.int64)
    rel = np.array([r["relative_error_percent"] for r in rows], dtype=np.float64)
    true_norm = np.array([r["true_l2_norm"] for r in rows], dtype=np.float64)
    err_norm = np.array([r["error_l2_norm"] for r in rows], dtype=np.float64)
    energy = np.array([r["true_energy_share_percent"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.8), sharex=True)
    axes[0].semilogy(coeff, rel + 1e-14, color="#1f77b4", linewidth=1.9)
    axes[0].set_ylabel("rel. error (%)")
    axes[0].set_title(
        rf"$\mu=(4.560,0.0190)$, data-driven ANN vs linear PROM: global rel. q error = {global_rel:.2f}%"
    )
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].semilogy(coeff, true_norm + 1e-14, color="black", linewidth=1.7, label=r"$\|q_j\|_2$")
    axes[1].semilogy(coeff, err_norm + 1e-14, color="#d62728", linewidth=1.5, label=r"$\|\Delta q_j\|_2$")
    axes[1].set_ylabel("time L2 norm")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, which="both", alpha=0.25)

    axes[2].semilogy(coeff, energy + 1e-16, color="#2ca02c", linewidth=1.7)
    axes[2].set_ylabel("energy share (%)")
    axes[2].set_xlabel("coefficient index")
    axes[2].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "mu456_qtot_ann_vs_linear_overview.png", dpi=220)
    plt.close(fig)


def _plot_pages(out_dir: Path, true_q: np.ndarray, pred_q: np.ndarray, t: np.ndarray, rows: list[dict[str, float]]) -> None:
    pdf_path = out_dir / "mu456_qtot_ann_vs_linear_coefficients_10_per_page.pdf"
    png_dir = out_dir / "pages_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for page, start in enumerate(range(0, true_q.shape[0], 10), start=1):
            stop = min(start + 10, true_q.shape[0])
            n = stop - start
            fig, axes = plt.subplots(n, 1, figsize=(8.6, 1.55 * n), sharex=True)
            if n == 1:
                axes = [axes]
            for ax, j in zip(axes, range(start, stop)):
                rel = rows[j]["relative_error_percent"]
                max_abs = rows[j]["max_abs_error"]
                energy = rows[j]["true_energy_share_percent"]
                ax.plot(t, true_q[j], color="black", linewidth=1.45, label="linear PROM")
                ax.plot(t, pred_q[j], color="#1f77b4", linewidth=1.15, linestyle="--", label="data-driven ANN")
                ax.set_ylabel(rf"$q_{{{j + 1}}}$", rotation=0, labelpad=18)
                ax.grid(True, alpha=0.22)
                ax.text(
                    0.985,
                    0.78,
                    f"rel={rel:.2f}% | max={max_abs:.2e} | E={energy:.2e}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.2,
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.75", "alpha": 0.85},
                )
                if j == start:
                    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
            axes[-1].set_xlabel("time")
            fig.suptitle(
                rf"$\mu=(4.560,0.0190)$: coefficient traces $q_{{{start + 1}}}$--$q_{{{stop}}}$",
                y=0.997,
                fontsize=12,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
            pdf.savefig(fig)
            fig.savefig(png_dir / f"mu456_qtot_coefficients_page_{page:02d}_q{start + 1:03d}_to_q{stop:03d}.png", dpi=190)
            plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    true_q = _load_q(TRUE_QN)
    pred_q = _load_q(PRED_QN)
    if true_q.shape != pred_q.shape:
        raise ValueError(f"Shape mismatch: true {true_q.shape}, pred {pred_q.shape}")

    t = _load_time(true_q.shape[1])
    global_rel = 100.0 * np.linalg.norm(pred_q - true_q, ord="fro") / np.linalg.norm(true_q, ord="fro")

    rows = _write_errors_csv(OUT_DIR / "mu456_qtot_ann_vs_linear_coeff_errors.csv", true_q, pred_q)
    _plot_overview(OUT_DIR, rows, global_rel)
    _plot_pages(OUT_DIR, true_q, pred_q, t, rows)

    summary_path = OUT_DIR / "summary.txt"
    worst = sorted(rows, key=lambda r: r["relative_error_percent"], reverse=True)[:10]
    with summary_path.open("w") as f:
        f.write(f"true_qN: {TRUE_QN}\n")
        f.write(f"pred_qN: {PRED_QN}\n")
        f.write(f"shape: {true_q.shape}\n")
        f.write(f"global_relative_q_error_percent: {global_rel:.12g}\n")
        f.write("worst_10_by_relative_error:\n")
        for r in worst:
            f.write(
                "  "
                f"q{int(r['coefficient']):03d}: "
                f"rel={r['relative_error_percent']:.6g}% "
                f"err_l2={r['error_l2_norm']:.6e} "
                f"true_l2={r['true_l2_norm']:.6e} "
                f"energy_share={r['true_energy_share_percent']:.6e}%\n"
            )

    print(f"[mu456-qtot-diagnostic] global_relative_q_error_percent = {global_rel:.4f}%")
    print(f"[mu456-qtot-diagnostic] output_dir = {OUT_DIR}")
    print(f"[mu456-qtot-diagnostic] pdf = {OUT_DIR / 'mu456_qtot_ann_vs_linear_coefficients_10_per_page.pdf'}")
    print(f"[mu456-qtot-diagnostic] csv = {OUT_DIR / 'mu456_qtot_ann_vs_linear_coeff_errors.csv'}")
    print(f"[mu456-qtot-diagnostic] overview = {OUT_DIR / 'mu456_qtot_ann_vs_linear_overview.png'}")


if __name__ == "__main__":
    main()
