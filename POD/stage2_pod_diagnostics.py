#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 2: POD PROJECTION / RECONSTRUCTION DIAGNOSTICS

This stage evaluates reconstruction quality of the POD basis on a parameter
set (default: all training parameters). It does not run online PROM; it only
checks projection quality.

Outputs (inside POD):
  - stage2_pod_diagnostics.npz
  - stage2_pod_diagnostics.png
  - stage2_pod_diagnostics_summary.txt
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, compute_error, get_snapshot_params
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "lines.linewidth": 2.2,
            "axes.linewidth": 1.1,
            "grid.alpha": 0.35,
            "figure.figsize": (11, 7),
        }
    )


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


def _param_label(mu):
    return f"({mu[0]:.3f}, {mu[1]:.4f})"


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    sigma_file=os.path.join(script_dir, "sigma.npy"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    stage1_metadata_file=os.path.join(script_dir, "stage1_pod_metadata.npz"),
    diagnostics_npz=os.path.join(script_dir, "stage2_pod_diagnostics.npz"),
    diagnostics_plot=os.path.join(script_dir, "stage2_pod_diagnostics.png"),
    report_file=os.path.join(script_dir, "stage2_pod_diagnostics_summary.txt"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    diagnostic_params=None,
):
    for path in (basis_file, sigma_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing POD file: {path}. Run stage1 first.")

    os.makedirs(os.path.dirname(diagnostics_npz), exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    sigma = np.asarray(np.load(sigma_file, allow_pickle=False), dtype=np.float64)

    if os.path.exists(uref_file):
        u_ref = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(basis.shape[0], dtype=np.float64)

    if basis.ndim != 2:
        raise ValueError(f"Basis must be 2D, got shape {basis.shape}.")
    if u_ref.size != basis.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {basis.shape[0]}."
        )

    if diagnostic_params is None:
        diagnostic_params = get_snapshot_params()

    if len(diagnostic_params) == 0:
        raise RuntimeError("diagnostic_params is empty.")

    print("\n====================================================")
    print("      STAGE 2: POD PROJECTION DIAGNOSTICS")
    print("====================================================")
    print(f"[POD-STAGE2] basis shape: {basis.shape}")
    print(f"[POD-STAGE2] evaluating {len(diagnostic_params)} parameter points")

    w0 = np.asarray(W0, dtype=np.float64).copy()

    rel_l2_global = []
    rel_l2_mean_time = []
    rel_l2_max_time = []
    elapsed_load_list = []
    elapsed_project_list = []

    for mu in diagnostic_params:
        mu = [float(mu[0]), float(mu[1])]

        t0 = time.time()
        snaps = np.asarray(
            load_or_compute_snaps(mu, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        elapsed_load = time.time() - t0

        if snaps.shape[0] != basis.shape[0]:
            raise RuntimeError(
                f"Snapshot size mismatch for mu={mu}: {snaps.shape[0]} vs basis {basis.shape[0]}."
            )

        t0 = time.time()
        q = basis.T @ (snaps - u_ref[:, None])
        snaps_rec = u_ref[:, None] + basis @ q
        elapsed_proj = time.time() - t0

        hdm_norm = np.linalg.norm(snaps)
        if hdm_norm > 0.0:
            rel_global = np.linalg.norm(snaps - snaps_rec) / hdm_norm
        else:
            rel_global = np.nan

        rel_t, rel_mean = compute_error(snaps_rec, snaps)
        rel_max = float(np.max(rel_t)) if rel_t.size > 0 else np.nan

        rel_l2_global.append(float(rel_global))
        rel_l2_mean_time.append(float(rel_mean))
        rel_l2_max_time.append(float(rel_max))
        elapsed_load_list.append(float(elapsed_load))
        elapsed_project_list.append(float(elapsed_proj))

        print(
            f"[POD-STAGE2] mu={_param_label(mu)} | "
            f"global={100.0*rel_global:.4f}% | "
            f"mean_t={100.0*rel_mean:.4f}% | max_t={100.0*rel_max:.4f}%"
        )

    params_array = np.asarray(diagnostic_params, dtype=np.float64)
    rel_l2_global = np.asarray(rel_l2_global, dtype=np.float64)
    rel_l2_mean_time = np.asarray(rel_l2_mean_time, dtype=np.float64)
    rel_l2_max_time = np.asarray(rel_l2_max_time, dtype=np.float64)
    elapsed_load_list = np.asarray(elapsed_load_list, dtype=np.float64)
    elapsed_project_list = np.asarray(elapsed_project_list, dtype=np.float64)

    np.savez(
        diagnostics_npz,
        params=params_array,
        rel_l2_global=rel_l2_global,
        rel_l2_mean_time=rel_l2_mean_time,
        rel_l2_max_time=rel_l2_max_time,
        elapsed_load_seconds=elapsed_load_list,
        elapsed_projection_seconds=elapsed_project_list,
    )

    x = np.arange(1, params_array.shape[0] + 1)
    plt.figure()
    plt.plot(x, 100.0 * rel_l2_global, "o-", label=r"global $\|e\|_2 / \|u\|_2$")
    plt.plot(x, 100.0 * rel_l2_mean_time, "s-", label="mean in time")
    plt.plot(x, 100.0 * rel_l2_max_time, "^-", label="max in time")
    plt.xlabel("Parameter index")
    plt.ylabel(r"Relative error [\%]")
    plt.title("POD projection/reconstruction diagnostics")
    plt.grid(True)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(diagnostics_plot, dpi=300, bbox_inches="tight")
    plt.close()

    worst_idx = int(np.nanargmax(rel_l2_global))
    best_idx = int(np.nanargmin(rel_l2_global))

    stage1_meta = {}
    if os.path.exists(stage1_metadata_file):
        m = np.load(stage1_metadata_file, allow_pickle=True)
        for key in m.files:
            val = m[key]
            stage1_meta[key] = val.item() if np.asarray(val).shape == () else val

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage2_pod_diagnostics.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("sigma_file", sigma_file),
                    ("u_ref_file", uref_file if os.path.exists(uref_file) else None),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("num_diagnostic_parameters", int(params_array.shape[0])),
                ],
            ),
            (
                "pod",
                [
                    ("basis_shape", basis.shape),
                    ("sigma_shape", sigma.shape),
                    ("basis_size", basis.shape[1]),
                    ("stage1_n_keep", stage1_meta.get("n_keep")),
                    ("stage1_energy_captured", stage1_meta.get("energy_captured")),
                    ("stage1_energy_lost", stage1_meta.get("energy_lost")),
                ],
            ),
            (
                "diagnostics",
                [
                    ("global_rel_error_percent_mean", 100.0 * float(np.nanmean(rel_l2_global))),
                    ("global_rel_error_percent_max", 100.0 * float(np.nanmax(rel_l2_global))),
                    ("global_rel_error_percent_min", 100.0 * float(np.nanmin(rel_l2_global))),
                    ("mean_time_rel_error_percent_mean", 100.0 * float(np.nanmean(rel_l2_mean_time))),
                    ("max_time_rel_error_percent_mean", 100.0 * float(np.nanmean(rel_l2_max_time))),
                    ("worst_param", params_array[worst_idx].tolist()),
                    ("worst_param_global_rel_error_percent", 100.0 * float(rel_l2_global[worst_idx])),
                    ("best_param", params_array[best_idx].tolist()),
                    ("best_param_global_rel_error_percent", 100.0 * float(rel_l2_global[best_idx])),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("load_snapshots_total", float(np.sum(elapsed_load_list))),
                    ("projection_total", float(np.sum(elapsed_project_list))),
                    ("load_snapshots_mean_per_param", float(np.mean(elapsed_load_list))),
                    ("projection_mean_per_param", float(np.mean(elapsed_project_list))),
                    (
                        "total",
                        float(np.sum(elapsed_load_list) + np.sum(elapsed_project_list)),
                    ),
                ],
            ),
            (
                "outputs",
                [
                    ("diagnostics_npz", diagnostics_npz),
                    ("diagnostics_plot_png", diagnostics_plot),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )

    print(f"[POD-STAGE2] Diagnostics saved: {diagnostics_npz}")
    print(f"[POD-STAGE2] Plot saved: {diagnostics_plot}")
    print(f"[POD-STAGE2] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
