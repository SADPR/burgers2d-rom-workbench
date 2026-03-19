#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 2: QUADRATIC MANIFOLD PROJECTION CHECK

Loads the quadratic manifold from stage1 and evaluates pure projection quality
against HDM snapshots at a target parameter.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
quadratic_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(quadratic_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "figure.figsize": (12, 8),
    })


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def build_Q_symmetric_matrix(q_mat):
    n, _ = q_mat.shape
    i_triu, j_triu = np.triu_indices(n)
    return q_mat[i_triu, :] * q_mat[j_triu, :]


def qm_reconstruct(V, H, u_ref, snaps):
    q_mat = V.T @ (snaps - u_ref[:, None])
    Q = build_Q_symmetric_matrix(q_mat)
    return u_ref[:, None] + V @ q_mat + H @ Q


def load_qm_files():
    V_path = os.path.join(quadratic_dir, "qm_V.npy")
    H_path = os.path.join(quadratic_dir, "qm_H.npy")
    uref_path = os.path.join(quadratic_dir, "qm_uref.npy")
    sigma_path = os.path.join(quadratic_dir, "qm_sigma.npy")
    metadata_path = os.path.join(quadratic_dir, "qm_metadata.npz")

    for path in (V_path, H_path, uref_path, sigma_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file '{path}'. Run stage1_quadratic_offline.py first."
            )

    V = np.load(V_path, allow_pickle=False)
    H = np.load(H_path, allow_pickle=False)
    u_ref = np.load(uref_path, allow_pickle=False).reshape(-1)
    sigma = np.load(sigma_path, allow_pickle=False).reshape(-1)

    metadata = {}
    if os.path.exists(metadata_path):
        data = np.load(metadata_path, allow_pickle=True)
        for key in data.files:
            val = data[key]
            metadata[key] = val.item() if np.asarray(val).shape == () else val
    return V, H, u_ref, sigma, metadata


def main(
    mu1=4.56,
    mu2=0.019,
    dt=DT,
    num_steps=NUM_STEPS,
):
    results_dir = os.path.join(parent_dir, "Results")
    snap_folder = os.path.join(results_dir, "param_snaps")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    V, H, u_ref, sigma, metadata = load_qm_files()
    n = V.shape[1]
    m = H.shape[1]

    w0 = np.asarray(W0, dtype=np.float64).copy()
    mu_rom = [mu1, mu2]

    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(
        mu_rom,
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder=snap_folder,
    )
    elapsed_hdm = time.time() - t0

    t0 = time.time()
    qm_snaps = qm_reconstruct(V, H, u_ref, hdm_snaps)
    elapsed_qm = time.time() - t0

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - qm_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2

    qm_path = os.path.join(
        quadratic_dir,
        f"qm_projection_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(qm_path, qm_snaps)

    inds_to_plot = range(0, num_steps + 1, 100)
    fig, ax1, ax2 = plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snaps,
        inds_to_plot,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        GRID_X,
        GRID_Y,
        qm_snaps,
        inds_to_plot,
        label=f"Quadratic Manifold (n={n})",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )
    fig.suptitle(rf"$\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$", y=0.98)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    plt.tight_layout()

    fig_path = os.path.join(
        quadratic_dir,
        f"qm_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(
        quadratic_dir,
        f"qm_projection_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("mu1", mu1),
                    ("mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "manifold",
                [
                    ("V_shape", V.shape),
                    ("H_shape", H.shape),
                    ("u_ref_shape", u_ref.shape),
                    ("sigma_shape", sigma.shape),
                    ("n", n),
                    ("m", m),
                    ("metadata_n_trad", metadata.get("n_trad")),
                    ("metadata_pod_tol", metadata.get("pod_tol")),
                    ("metadata_zeta_qua", metadata.get("zeta_qua")),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("hdm_load_or_solve_time", elapsed_hdm),
                    ("qm_projection_time", elapsed_qm),
                    ("avg_qm_projection_time_per_step", elapsed_qm / num_steps),
                ],
            ),
            (
                "error_metrics",
                [
                    ("relative_l2_error", rel_err_l2),
                    ("relative_error_percent", relative_error),
                ],
            ),
            (
                "outputs",
                [
                    ("qm_snapshots_npy", qm_path),
                    ("comparison_plot_png", fig_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )

    print(f"[STAGE2] Relative projection error: {relative_error:.3f}%")
    print(f"[STAGE2] Saved snapshots: {qm_path}")
    print(f"[STAGE2] Saved plot: {fig_path}")
    print(f"[STAGE2] Saved summary: {report_path}")

    return elapsed_qm, relative_error


if __name__ == "__main__":
    main()
