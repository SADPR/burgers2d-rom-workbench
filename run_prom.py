#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global affine PROM (LSPG) for the 2D inviscid Burgers problem
using a precomputed POD basis from `POD/`.

Offline POD construction is now separated into:
  - POD/stage1_build_pod_basis.py
  - POD/stage2_pod_diagnostics.py
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG
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


def _load_pod_data(pod_dir, legacy_pod_dir, state_size):
    basis_path = os.path.join(pod_dir, "basis.npy")
    sigma_path = os.path.join(pod_dir, "sigma.npy")
    u_ref_path = os.path.join(pod_dir, "u_ref.npy")
    metadata_path = os.path.join(pod_dir, "stage1_pod_metadata.npz")

    if not (os.path.exists(basis_path) and os.path.exists(sigma_path)):
        legacy_basis = os.path.join(legacy_pod_dir, "basis.npy")
        legacy_sigma = os.path.join(legacy_pod_dir, "sigma.npy")
        legacy_u_ref = os.path.join(legacy_pod_dir, "u_ref.npy")
        legacy_meta = os.path.join(legacy_pod_dir, "stage1_pod_metadata.npz")

        if os.path.exists(legacy_basis) and os.path.exists(legacy_sigma):
            print(
                f"[PROM] Warning: POD files not found in '{pod_dir}'. "
                f"Using legacy location '{legacy_pod_dir}'."
            )
            basis_path = legacy_basis
            sigma_path = legacy_sigma
            u_ref_path = legacy_u_ref
            metadata_path = legacy_meta
        else:
            raise FileNotFoundError(
                "POD basis files not found. Run POD/stage1_build_pod_basis.py first. "
                f"Checked: '{basis_path}' and legacy '{legacy_basis}'."
            )

    basis_full = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    sigma = np.asarray(np.load(sigma_path, allow_pickle=False), dtype=np.float64)

    if basis_full.ndim != 2:
        raise ValueError(f"Loaded basis has shape {basis_full.shape}, expected 2D array.")
    if basis_full.shape[0] != state_size:
        raise ValueError(
            f"Basis/state mismatch: basis has {basis_full.shape[0]} rows, expected {state_size}."
        )

    if os.path.exists(u_ref_path):
        u_ref = np.asarray(np.load(u_ref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if u_ref.size != state_size:
            raise ValueError(
                f"Loaded u_ref has size {u_ref.size}, expected {state_size}."
            )
        centered_basis = not np.allclose(u_ref, 0.0)
        ref_source = "loaded_file"
    else:
        u_ref = np.zeros(state_size, dtype=np.float64)
        centered_basis = False
        ref_source = "none"

    metadata = {}
    if os.path.exists(metadata_path):
        m = np.load(metadata_path, allow_pickle=True)
        for key in m.files:
            val = m[key]
            metadata[key] = val.item() if np.asarray(val).shape == () else val

    return {
        "basis_full": basis_full,
        "sigma": sigma,
        "u_ref": u_ref,
        "centered_basis": centered_basis,
        "reference_source": ref_source,
        "basis_path": basis_path,
        "sigma_path": sigma_path,
        "u_ref_path": u_ref_path,
        "metadata_path": metadata_path if os.path.exists(metadata_path) else None,
        "metadata": metadata,
    }


def main(
    mu1=4.56,
    mu2=0.019,
    pod_dir="POD",
    num_modes=None,
    dt=DT,
    num_steps=NUM_STEPS,
    snap_folder=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
):
    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")

    legacy_pod_dir = os.path.join(results_dir, "POD")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()
    mu_rom = [float(mu1), float(mu2)]

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1

    pod_data = _load_pod_data(pod_dir, legacy_pod_dir, w0.size)
    basis_full = pod_data["basis_full"]
    sigma = pod_data["sigma"]
    u_ref = pod_data["u_ref"]

    n_available = basis_full.shape[1]
    if num_modes is None:
        n_keep = n_available
    else:
        n_keep = int(num_modes)
        if n_keep < 1 or n_keep > n_available:
            raise ValueError(f"Requested num_modes={n_keep}, available modes={n_available}.")

    basis_trunc = basis_full[:, :n_keep]

    energy_captured = None
    energy_lost = None
    if sigma.size > 0 and n_keep <= sigma.size:
        sigma_sq = sigma**2
        total_energy = float(np.sum(sigma_sq))
        if total_energy > 0.0:
            energy_captured = float(np.sum(sigma_sq[:n_keep]) / total_energy)
            energy_lost = 1.0 - energy_captured

    print(f"[PROM] Loaded POD basis from: {pod_data['basis_path']}")
    print(f"[PROM] Loaded singular values from: {pod_data['sigma_path']}")
    print(f"[PROM] Basis size used: {n_keep}")
    print(
        f"[PROM] Centered basis: {pod_data['centered_basis']} "
        f"(reference: {pod_data['reference_source']})"
    )
    if energy_captured is not None:
        print(f"[PROM] Estimated captured energy at {n_keep} modes: {energy_captured:.8f}")
        print(f"[PROM] Estimated discarded energy at {n_keep} modes: {energy_lost:.8e}")

    t0 = time.time()
    rom_snaps, rom_times = inviscid_burgers_implicit2D_LSPG(
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu=mu_rom,
        basis=basis_trunc,
        u_ref=u_ref,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
    )
    elapsed_rom = time.time() - t0

    num_its, jac_time, res_time, ls_time = rom_times
    print(f"[PROM] Elapsed PROM time: {elapsed_rom:.3e} seconds")
    print(f"[PROM] Gauss-Newton iterations: {num_its}")
    print(
        "[PROM] Timing breakdown (s): "
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
    print(f"[PROM] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"prom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[PROM] ROM snapshots saved to: {rom_path}")

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
        rom_snaps,
        snaps_to_plot,
        label="PROM",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )

    fig.suptitle(rf"$\mu_1 = {mu_rom[0]:.2f}, \mu_2 = {mu_rom[1]:.3f}$", y=0.98)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    plt.tight_layout()

    fig_path = os.path.join(
        results_dir,
        f"prom_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PROM] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[PROM] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"prom_summary_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("mu1", mu_rom[0]),
                    ("mu2", mu_rom[1]),
                ],
            ),
            (
                "configuration",
                [
                    ("pod_dir_requested", pod_dir),
                    ("pod_basis_file", pod_data["basis_path"]),
                    ("pod_sigma_file", pod_data["sigma_path"]),
                    ("pod_u_ref_file", pod_data["u_ref_path"] if os.path.exists(pod_data["u_ref_path"]) else None),
                    ("pod_stage1_metadata", pod_data["metadata_path"]),
                    ("num_modes_requested", num_modes),
                    ("num_modes_used", n_keep),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("snap_folder", snap_folder),
                    ("max_its", max_its),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
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
                "pod",
                [
                    ("basis_size", n_keep),
                    ("n_available_modes", n_available),
                    ("energy_captured", energy_captured),
                    ("energy_lost", energy_lost),
                    ("centered_basis_used", pod_data["centered_basis"]),
                    ("reference_source", pod_data["reference_source"]),
                    ("u_ref_l2_norm", np.linalg.norm(u_ref)),
                    ("stage1_num_training_parameters", pod_data["metadata"].get("num_training_parameters")),
                    ("stage1_num_training_snapshots", pod_data["metadata"].get("num_training_snapshots")),
                ],
            ),
            (
                "prom_timing",
                [
                    ("total_prom_time_seconds", elapsed_rom),
                    ("avg_prom_time_per_step_seconds", elapsed_rom / num_steps),
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
                    ("rom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[PROM] Text summary saved to: {report_path}")

    return elapsed_rom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019)
