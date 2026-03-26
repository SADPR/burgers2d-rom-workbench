#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1:
- build centered POD basis using shared burgers.core utilities
- save POD artifacts into PROJECT_ROOT/Project_YvonMaday
- run a quick POD reconstruction check on a selected test parameter
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import load_or_compute_snaps, POD, get_snapshot_params, plot_singular_value_decay
from burgers.config import (
    GRID_X,
    GRID_Y,
    W0,
    DT,
    NUM_STEPS,
    MU1_RANGE,
    MU2_RANGE,
    SAMPLES_PER_MU,
)
try:
    from project_layout import STAGE1_DIR, ensure_layout_dirs
except ModuleNotFoundError:
    from .project_layout import STAGE1_DIR, ensure_layout_dirs


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


def choose_snap_folder(project_root):
    candidates = [
        os.path.join(project_root, "Results", "param_snaps"),
        os.path.join(project_root, "param_snaps"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def build_snapshot_matrix(mu_samples, grid_x, grid_y, w0, dt, num_steps, snap_folder):
    first = np.asarray(
        load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )
    n_dofs, n_time = first.shape

    snaps = np.zeros((n_dofs, n_time * len(mu_samples)), dtype=np.float64)
    snaps[:, :n_time] = first

    col = n_time
    for idx, mu in enumerate(mu_samples[1:], start=2):
        print(f"[STAGE1] loading snapshots {idx}/{len(mu_samples)} for mu={mu}")
        s_mu = np.asarray(
            load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        if s_mu.shape != (n_dofs, n_time):
            raise RuntimeError(
                "Snapshot shape mismatch while building POD matrix: "
                f"expected {(n_dofs, n_time)}, got {s_mu.shape} for mu={mu}."
            )
        snaps[:, col:col + n_time] = s_mu
        col += n_time

    return snaps


def main(
    pod_dir=STAGE1_DIR,
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    num_modes=None,
    pod_tol=1e-6,
    pod_method="svd",
    center=True,
    random_state=0,
    rebuild_basis=True,
    test_mu=(4.56, 0.019),
):
    set_latex_plot_style()
    ensure_layout_dirs()

    if pod_method not in ("svd", "rsvd"):
        raise ValueError("pod_method must be 'svd' or 'rsvd'.")

    if num_modes is not None and pod_tol is not None:
        raise ValueError("Choose either num_modes or pod_tol, not both.")

    if pod_method == "rsvd" and num_modes is None:
        raise ValueError(
            "For pod_method='rsvd', num_modes must be provided explicitly. "
            "Use pod_method='svd' for tolerance-based truncation."
        )

    if snap_folder is None:
        snap_folder = choose_snap_folder(PROJECT_ROOT)

    os.makedirs(pod_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    basis_file = os.path.join(pod_dir, "basis.npy")
    sigma_file = os.path.join(pod_dir, "sigma.npy")
    uref_file = os.path.join(pod_dir, "u_ref.npy")
    decay_plot_file = os.path.join(pod_dir, "stage1_pod_singular_value_decay.png")
    metadata_file = os.path.join(pod_dir, "stage1_pod_metadata.npz")
    report_file = os.path.join(pod_dir, "stage1_pod_summary.txt")

    print("\n====================================================")
    print("      PROJECT_YVONMADAY - STAGE 1 POD BASIS")
    print("====================================================")
    print(f"[STAGE1] pod_dir={pod_dir}")
    print(f"[STAGE1] snap_folder={snap_folder}")
    print(f"[STAGE1] method={pod_method} | center={center} | dt={dt} | num_steps={num_steps}")
    if num_modes is None:
        print(f"[STAGE1] truncation by pod_tol={pod_tol:.1e}")
    else:
        print(f"[STAGE1] truncation by num_modes={num_modes}")

    w0 = np.asarray(W0, dtype=np.float64).copy()

    if rebuild_basis:
        mu_samples = get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        )
        if len(mu_samples) == 0:
            raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

        print(f"[STAGE1] Number of training parameters: {len(mu_samples)}")

        t0 = time.time()
        snaps = build_snapshot_matrix(
            mu_samples=mu_samples,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
        )
        elapsed_snapshots = time.time() - t0
        print(f"[STAGE1] Snapshot matrix shape: {snaps.shape}")

        t0 = time.time()
        if num_modes is None:
            basis, sigma, info, u_ref = POD(
                snaps,
                method=pod_method,
                energy_loss=pod_tol,
                random_state=random_state,
                return_truncation_info=True,
                center=center,
                return_reference=True,
            )
        else:
            basis, sigma, info, u_ref = POD(
                snaps,
                num_modes=num_modes,
                method=pod_method,
                random_state=random_state,
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
            label="POD Stage 1",
            title="POD residual energy decay",
            use_latex=True,
        )

        np.savez(
            metadata_file,
            n_keep=np.asarray(int(info["n_keep"]), dtype=np.int64),
            n_available=np.asarray(int(info["n_available"]), dtype=np.int64),
            energy_captured=np.asarray(float(info["energy_captured"]), dtype=np.float64),
            energy_lost=np.asarray(float(info["energy_lost"]), dtype=np.float64),
            centered=np.asarray(bool(info["centered"]), dtype=np.int64),
            reference_source=np.asarray(str(info["reference_source"])),
            pod_method=np.asarray(str(pod_method)),
            pod_tol=np.asarray(np.nan if pod_tol is None else float(pod_tol), dtype=np.float64),
            num_modes_requested=np.asarray(-1 if num_modes is None else int(num_modes), dtype=np.int64),
            num_training_parameters=np.asarray(len(mu_samples), dtype=np.int64),
            state_size=np.asarray(snaps.shape[0], dtype=np.int64),
            num_training_snapshots=np.asarray(snaps.shape[1], dtype=np.int64),
            dt=np.asarray(float(dt), dtype=np.float64),
            num_steps=np.asarray(int(num_steps), dtype=np.int64),
            snap_folder=np.asarray(str(snap_folder)),
        )

        print(f"[STAGE1] Saved basis: {basis_file}")
        print(f"[STAGE1] Saved sigma: {sigma_file}")
        print(f"[STAGE1] Saved u_ref: {uref_file}")
        print(f"[STAGE1] Saved decay plot: {decay_plot_file}")
        print(f"[STAGE1] Saved metadata: {metadata_file}")
    else:
        elapsed_snapshots = None
        elapsed_pod = None
        mu_samples = get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        )

        if not (os.path.exists(basis_file) and os.path.exists(sigma_file) and os.path.exists(uref_file)):
            raise FileNotFoundError(
                "Cannot load POD artifacts because one or more files are missing: "
                f"{basis_file}, {sigma_file}, {uref_file}"
            )

        basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
        sigma = np.asarray(np.load(sigma_file, allow_pickle=False), dtype=np.float64)
        u_ref = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
        info = {
            "n_keep": int(basis.shape[1]),
            "n_available": int(sigma.size),
            "energy_captured": np.nan,
            "energy_lost": np.nan,
            "centered": bool(center),
            "reference_source": "loaded_file",
        }
        print("[STAGE1] Loaded existing POD artifacts from Project_YvonMaday.")

    # POD reconstruction check on one parameter
    mu_test = [float(test_mu[0]), float(test_mu[1])]
    print(f"[STAGE1] POD reconstruction check for mu={mu_test}")

    hdm_snap = np.asarray(
        load_or_compute_snaps(mu_test, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )

    hdm_centered = hdm_snap - u_ref[:, None]
    q = basis.T @ hdm_centered
    pod_rec = u_ref[:, None] + basis @ q

    rel_err = float(np.linalg.norm(hdm_snap - pod_rec) / np.linalg.norm(hdm_snap))
    print(f"[STAGE1] POD reconstruction relative error: {rel_err:.6e}")

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "Project_YvonMaday/stage1_pod.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("pod_dir", pod_dir),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("pod_method", pod_method),
                    ("pod_tol", pod_tol if num_modes is None else None),
                    ("num_modes_requested", num_modes),
                    ("center", center),
                    ("random_state", random_state),
                    ("rebuild_basis", rebuild_basis),
                    ("num_training_parameters", len(mu_samples)),
                    ("test_mu", mu_test),
                ],
            ),
            (
                "pod",
                [
                    ("basis_shape", basis.shape),
                    ("sigma_size", sigma.size),
                    ("n_keep", int(info["n_keep"])),
                    ("n_available", int(info["n_available"])),
                    ("energy_captured", info["energy_captured"]),
                    ("energy_lost", info["energy_lost"]),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                ],
            ),
            (
                "diagnostics",
                [
                    ("pod_reconstruction_relative_error", rel_err),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_snapshot_matrix", elapsed_snapshots),
                    ("compute_pod", elapsed_pod),
                    ("total_stage1_work", None if elapsed_snapshots is None else elapsed_snapshots + elapsed_pod),
                ],
            ),
            (
                "outputs",
                [
                    ("basis_npy", basis_file),
                    ("sigma_npy", sigma_file),
                    ("u_ref_npy", uref_file),
                    ("singular_value_decay_png", decay_plot_file),
                    ("metadata_npz", metadata_file),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )

    print(f"[STAGE1] Summary written to: {report_file}")
    print("[STAGE1] Done.")


if __name__ == "__main__":
    main()
