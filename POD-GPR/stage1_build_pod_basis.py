#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 1: BUILD POD BASIS FOR POD-GPR

This stage builds the POD basis and stores artifacts needed by later stages.

Outputs (inside POD-GPR):
  - basis.npy
  - sigma.npy
  - u_ref.npy
  - stage1_pod_singular_value_decay.png
  - stage1_pod_basis_summary.txt
"""

import os
import sys
import time
from datetime import datetime

import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, POD, plot_singular_value_decay, get_snapshot_params
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


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


def build_snapshot_matrix(mu_samples, grid_x, grid_y, w0, dt, num_steps, snap_folder):
    first = np.asarray(
        load_or_compute_snaps(
            mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder
        ),
        dtype=np.float64,
    )
    n_dofs, n_time = first.shape

    snaps = np.zeros((n_dofs, n_time * len(mu_samples)), dtype=np.float64)
    snaps[:, :n_time] = first

    col = n_time
    for mu in mu_samples[1:]:
        s_mu = np.asarray(
            load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        if s_mu.shape != (n_dofs, n_time):
            raise RuntimeError(
                "Snapshot shape mismatch while aggregating matrix: "
                f"expected {(n_dofs, n_time)}, got {s_mu.shape} for mu={mu}."
            )
        snaps[:, col:col + n_time] = s_mu
        col += n_time

    return snaps


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    sigma_file=os.path.join(script_dir, "sigma.npy"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    decay_plot_file=os.path.join(script_dir, "stage1_pod_singular_value_decay.png"),
    report_file=os.path.join(script_dir, "stage1_pod_basis_summary.txt"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    pod_method="svd",
    pod_tol=1e-6,
    num_modes=None,
    center=True,
):
    if pod_method not in ("svd", "rsvd"):
        raise ValueError("pod_method must be 'svd' or 'rsvd'.")

    if pod_method == "rsvd" and num_modes is None:
        raise ValueError(
            "For pod_method='rsvd', num_modes must be provided explicitly. "
            "Use pod_method='svd' for tolerance-based truncation."
        )

    os.makedirs(os.path.dirname(basis_file), exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    print("\n====================================================")
    print("         STAGE 1: POD BASIS FOR POD-GPR")
    print("====================================================")
    print(f"[STAGE1] pod_method={pod_method}, center={center}")
    if num_modes is None:
        print(f"[STAGE1] truncation by pod_tol={pod_tol:.1e}")
    else:
        print(f"[STAGE1] truncation by num_modes={num_modes}")

    mu_samples = get_snapshot_params()
    print(f"[STAGE1] Number of training parameters: {len(mu_samples)}")

    w0 = np.asarray(W0, dtype=np.float64).copy()

    t0 = time.time()
    snaps = build_snapshot_matrix(
        mu_samples,
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder,
    )
    elapsed_snapshots = time.time() - t0

    print(f"[STAGE1] Snapshot matrix shape: {snaps.shape}")

    t0 = time.time()
    if num_modes is None:
        basis, sigma, info, u_ref = POD(
            snaps,
            method=pod_method,
            energy_loss=pod_tol,
            return_truncation_info=True,
            center=center,
            return_reference=True,
        )
    else:
        basis, sigma, info, u_ref = POD(
            snaps,
            method=pod_method,
            num_modes=num_modes,
            return_truncation_info=True,
            center=center,
            return_reference=True,
        )
    elapsed_pod = time.time() - t0

    if u_ref is None:
        u_ref = np.zeros(snaps.shape[0], dtype=np.float64)

    np.save(basis_file, basis)
    np.save(sigma_file, sigma)
    np.save(uref_file, u_ref)

    plot_singular_value_decay(
        sigma,
        out_path=decay_plot_file,
        max_modes=min(1000, sigma.size),
        label="POD-GPR Stage1",
        title="POD residual energy decay",
        use_latex=True,
    )

    print(f"[STAGE1] Saved basis: {basis_file}")
    print(f"[STAGE1] Saved singular values: {sigma_file}")
    print(f"[STAGE1] Saved reference state: {uref_file}")
    print(f"[STAGE1] Saved decay plot: {decay_plot_file}")

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage1_build_pod_basis.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("pod_method", pod_method),
                    ("pod_tol", pod_tol if num_modes is None else None),
                    ("num_modes_requested", num_modes),
                    ("center", center),
                    ("num_training_parameters", len(mu_samples)),
                ],
            ),
            (
                "snapshot_matrix",
                [
                    ("shape", snaps.shape),
                    ("state_size", snaps.shape[0]),
                    ("num_snapshots", snaps.shape[1]),
                ],
            ),
            (
                "pod_truncation",
                [
                    ("n_keep", int(info["n_keep"])),
                    ("energy_captured", float(info["energy_captured"])),
                    ("energy_lost", float(info["energy_lost"])),
                    ("centered", bool(info["centered"])),
                    ("reference_source", info["reference_source"]),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_snapshot_matrix", elapsed_snapshots),
                    ("compute_pod", elapsed_pod),
                    ("total", elapsed_snapshots + elapsed_pod),
                ],
            ),
            (
                "outputs",
                [
                    ("basis_npy", basis_file),
                    ("sigma_npy", sigma_file),
                    ("u_ref_npy", uref_file),
                    ("singular_value_decay_png", decay_plot_file),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )
    print(f"[STAGE1] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
