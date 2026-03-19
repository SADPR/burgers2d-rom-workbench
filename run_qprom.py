#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global quadratic-manifold PROM (QM-LSPG) for the 2D inviscid Burgers
problem and compare against HDM.
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.quadratic_manifold import inviscid_burgers_implicit2D_LSPG_qm
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


def load_qm_model(qm_dir):
    V_path = os.path.join(qm_dir, "qm_V.npy")
    H_path = os.path.join(qm_dir, "qm_H.npy")
    uref_path = os.path.join(qm_dir, "qm_uref.npy")
    sigma_path = os.path.join(qm_dir, "qm_sigma.npy")
    metadata_path = os.path.join(qm_dir, "qm_metadata.npz")

    for path in (V_path, H_path, uref_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing quadratic manifold file: {path}")

    V = np.load(V_path, allow_pickle=False)
    H = np.load(H_path, allow_pickle=False)
    u_ref = np.load(uref_path, allow_pickle=False).reshape(-1)
    sigma = np.load(sigma_path, allow_pickle=False) if os.path.exists(sigma_path) else None

    metadata = {}
    if os.path.exists(metadata_path):
        data = np.load(metadata_path, allow_pickle=True)
        for key in data.files:
            val = data[key]
            metadata[key] = val.item() if np.asarray(val).shape == () else val

    return V, H, u_ref, sigma, metadata, V_path, H_path, uref_path, sigma_path, metadata_path


def main(
    mu1=4.75,
    mu2=0.02,
    qm_dir="Quadratic",
    dt=DT,
    num_steps=NUM_STEPS,
    snap_folder=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    max_its_q0=20,
    tol_q0=1e-6,
):
    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()
    mu_rom = [mu1, mu2]

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1

    (
        V,
        H,
        u_ref,
        sigma,
        metadata,
        V_path,
        H_path,
        uref_path,
        sigma_path,
        metadata_path,
    ) = load_qm_model(qm_dir)

    N, n = V.shape
    m = H.shape[1]
    print(f"[QPROM] Loaded manifold from '{qm_dir}'")
    print(f"[QPROM] V shape={V.shape}, H shape={H.shape}, u_ref shape={u_ref.shape}")

    t0 = time.time()
    rom_snaps_qm, stats = inviscid_burgers_implicit2D_LSPG_qm(
        grid_x,
        grid_y,
        w0,
        dt,
        num_steps,
        mu_rom,
        V,
        H,
        u_ref=u_ref,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        max_its_q0=max_its_q0,
        tol_q0=tol_q0,
    )
    elapsed_qprom = time.time() - t0
    num_its, jac_time, res_time, ls_time = stats

    print(f"[QPROM] Elapsed QPROM time: {elapsed_qprom:.3e} seconds")
    print(f"[QPROM] Gauss-Newton iterations: {num_its}")
    print(
        "[QPROM] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )

    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(
        mu_rom,
        grid_x,
        grid_y,
        w0,
        dt,
        num_steps,
        snap_folder=snap_folder,
    )
    elapsed_hdm = time.time() - t0
    print(f"[QPROM] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"qprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps_qm)
    print(f"[QPROM] ROM snapshots saved to: {rom_path}")

    snaps_to_plot = range(0, num_steps + 1, 100)
    fig, ax1, ax2 = plot_snaps(
        grid_x,
        grid_y,
        hdm_snaps,
        snaps_to_plot,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        grid_x,
        grid_y,
        rom_snaps_qm,
        snaps_to_plot,
        label="QPROM",
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
        results_dir,
        f"qprom_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[QPROM] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps_qm) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[QPROM] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"qprom_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("qm_dir", qm_dir),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("max_its_q0", max_its_q0),
                    ("tol_q0", tol_q0),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "discretization",
                [
                    ("num_cells_x", num_cells_x),
                    ("num_cells_y", num_cells_y),
                    ("full_state_size", w0.size),
                ],
            ),
            (
                "manifold",
                [
                    ("V_shape", V.shape),
                    ("H_shape", H.shape),
                    ("u_ref_shape", u_ref.shape),
                    ("sigma_shape", None if sigma is None else sigma.shape),
                    ("n", n),
                    ("m", m),
                    ("u_ref_l2_norm", np.linalg.norm(u_ref)),
                    ("metadata_n_trad", metadata.get("n_trad")),
                    ("metadata_n_final", metadata.get("n_final")),
                    ("metadata_pod_tol", metadata.get("pod_tol")),
                    ("metadata_zeta_qua", metadata.get("zeta_qua")),
                ],
            ),
            (
                "qprom_timing",
                [
                    ("total_qprom_time_seconds", elapsed_qprom),
                    ("avg_qprom_time_per_step_seconds", elapsed_qprom / num_steps),
                    ("gn_iterations_total", num_its),
                    ("avg_gn_iterations_per_step", num_its / num_steps),
                    ("jacobian_time_seconds", jac_time),
                    ("residual_time_seconds", res_time),
                    ("linear_solve_time_seconds", ls_time),
                    ("hdm_load_or_solve_time_seconds", elapsed_hdm),
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
                    ("qprom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("summary_txt", report_path),
                    ("qm_V_npy", V_path),
                    ("qm_H_npy", H_path),
                    ("qm_uref_npy", uref_path),
                    ("qm_sigma_npy", sigma_path if os.path.exists(sigma_path) else None),
                    ("qm_metadata_npz", metadata_path if os.path.exists(metadata_path) else None),
                ],
            ),
        ],
    )
    print(f"[QPROM] Text summary saved to: {report_path}")

    return elapsed_qprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019)
