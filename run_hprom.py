#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global affine HPROM (ECSW-LSPG) for the 2D inviscid Burgers problem
using the modern `burgers/` modules and save outputs consistently with
run_prom.py.
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import (
    make_2D_grid,
    plot_snaps,
    load_or_compute_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
)
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
from burgers.linear_manifold import compute_ECSW_training_matrix_2D
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import (
    RandomizedSingularValueDecomposition,
)


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


def main(
    mu1=4.56,
    mu2=0.019,
    compute_ecsw=True,
    num_modes=None,
    pod_dir="POD",
    snap_sample_factor=10,
    snap_time_offset=3,
    mu_samples=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Parameters
    ----------
    mu1, mu2 : float
        Out-of-sample test parameter for online HPROM.
    compute_ecsw : bool
        If True, build ECSW weights from training snapshots.
        If False, load previously saved weights.
    num_modes : int or None
        Number of POD modes to keep from the saved basis.
        If None, use all available modes.
    pod_dir : str
        Preferred POD artifact directory (default: "POD").
        Legacy fallback "Results/POD" is used if needed.
    snap_sample_factor : int
        Temporal subsampling stride for ECSW training snapshots.
    snap_time_offset : int
        Time offset between "current" and "previous" snapshots in ECSW
        training blocks.
    mu_samples : sequence of [mu1, mu2] or None
        Parameter points used to build ECSW weights.
    """

    if mu_samples is None:
        mu_samples = [[4.25, 0.0225]]
    mu_samples = [list(mu) for mu in mu_samples]

    if snap_sample_factor < 1:
        raise ValueError("snap_sample_factor must be >= 1.")
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")

    # ------------------------------------------------------------------
    # Folders
    # ------------------------------------------------------------------
    results_dir = "Results"
    pod_dir_requested = pod_dir
    snap_folder = os.path.join(results_dir, "param_snaps")
    legacy_pod_dir = os.path.join(results_dir, "POD")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------
    set_latex_plot_style()

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------
    dt = 0.05
    num_steps = 500
    num_cells_x = 100
    num_cells_y = 100
    xl, xu, yl, yu = 0.0, 100.0, 0.0, 100.0

    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)

    u0 = np.ones((num_cells_y, num_cells_x), dtype=np.float64)
    v0 = np.ones((num_cells_y, num_cells_x), dtype=np.float64)
    w0 = np.concatenate((u0.ravel(), v0.ravel()))

    mu_rom = [mu1, mu2]

    # ------------------------------------------------------------------
    # Load POD basis / sigma / reference
    # ------------------------------------------------------------------
    basis_path = os.path.join(pod_dir, "basis.npy")
    sigma_path = os.path.join(pod_dir, "sigma.npy")
    u_ref_path = os.path.join(pod_dir, "u_ref.npy")
    weights_path = os.path.join(pod_dir, "ecsw_weights_lspg.npy")

    if not (os.path.exists(basis_path) and os.path.exists(sigma_path)):
        legacy_basis_path = os.path.join(legacy_pod_dir, "basis.npy")
        legacy_sigma_path = os.path.join(legacy_pod_dir, "sigma.npy")
        legacy_u_ref_path = os.path.join(legacy_pod_dir, "u_ref.npy")
        legacy_weights_path = os.path.join(legacy_pod_dir, "ecsw_weights_lspg.npy")
        if os.path.exists(legacy_basis_path) and os.path.exists(legacy_sigma_path):
            print(
                f"Warning: POD files not found in '{pod_dir}'. "
                f"Using legacy location '{legacy_pod_dir}'."
            )
            pod_dir = legacy_pod_dir
            basis_path = legacy_basis_path
            sigma_path = legacy_sigma_path
            u_ref_path = legacy_u_ref_path
            weights_path = legacy_weights_path
        else:
            raise FileNotFoundError(
                "POD basis files not found. Run POD/stage1_build_pod_basis.py first. "
                f"Checked '{basis_path}' and legacy '{legacy_basis_path}'."
            )

    os.makedirs(pod_dir, exist_ok=True)

    basis_full = np.load(basis_path, allow_pickle=False)
    sigma = np.load(sigma_path, allow_pickle=False)
    if basis_full.ndim != 2:
        raise ValueError(f"Loaded basis has shape {basis_full.shape}, expected 2D array.")

    n_available = basis_full.shape[1]
    if num_modes is None:
        n_keep = n_available
    else:
        n_keep = int(num_modes)
        if n_keep < 1 or n_keep > n_available:
            raise ValueError(f"Requested num_modes={n_keep}, available modes={n_available}.")
    basis_trunc = basis_full[:, :n_keep]

    if os.path.exists(u_ref_path):
        u_ref = np.load(u_ref_path, allow_pickle=False).reshape(-1)
        if u_ref.size != basis_full.shape[0]:
            raise ValueError(
                f"Loaded u_ref has size {u_ref.size}, expected {basis_full.shape[0]}."
            )
        centered_basis = not np.allclose(u_ref, 0.0)
        ref_source = "loaded_file"
    else:
        u_ref = np.zeros(basis_full.shape[0], dtype=np.float64)
        centered_basis = False
        ref_source = "none"
        print(f"Warning: {u_ref_path} not found. Using zero affine reference.")

    energy_captured = None
    energy_lost = None
    if sigma.size > 0 and n_keep <= sigma.size:
        sigma_sq = sigma**2
        total_energy = float(np.sum(sigma_sq))
        if total_energy > 0.0:
            energy_captured = float(np.sum(sigma_sq[:n_keep]) / total_energy)
            energy_lost = 1.0 - energy_captured

    print(f"Loaded POD basis from {basis_path}")
    print(f"Loaded singular values from {sigma_path}")
    print(f"Using basis size: {n_keep}")
    print(f"Reduced linear solver: {linear_solver}")
    if str(linear_solver).strip().lower() == "normal_eq":
        print(f"normal_eq_reg: {float(normal_eq_reg):.3e}")
    print(f"Centered basis: {centered_basis} (reference: {ref_source})")
    if energy_captured is not None:
        print(f"Estimated captured energy at {n_keep} modes: {energy_captured:.8f}")
        print(f"Estimated discarded energy at {n_keep} modes: {energy_lost:.8e}")

    # ------------------------------------------------------------------
    # ECSW weights: build or load
    # ------------------------------------------------------------------
    C_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None
    n_ecsw_elements = None

    if compute_ecsw:
        Clist = []
        t0 = time.time()

        for mu_train in mu_samples:
            mu_snaps = load_or_compute_snaps(
                mu_train,
                grid_x,
                grid_y,
                w0,
                dt,
                num_steps,
                snap_folder=snap_folder,
            )

            start_col = snap_time_offset
            stop_col = num_steps
            snaps_now = mu_snaps[:, start_col:stop_col:snap_sample_factor]
            snaps_prev = mu_snaps[:, 0:stop_col - snap_time_offset:snap_sample_factor]

            if snaps_now.shape[1] != snaps_prev.shape[1]:
                raise RuntimeError(
                    "ECSW snapshot alignment failed: "
                    f"snaps_now has {snaps_now.shape[1]} columns, "
                    f"snaps_prev has {snaps_prev.shape[1]} columns."
                )
            if snaps_now.shape[1] == 0:
                raise RuntimeError(
                    "ECSW training produced zero columns. "
                    "Adjust snap_time_offset or snap_sample_factor."
                )

            print(f"Generating ECSW training block for mu={mu_train}")
            Ci = compute_ECSW_training_matrix_2D(
                snaps_now,
                snaps_prev,
                basis_trunc,
                inviscid_burgers_res2D,
                inviscid_burgers_exact_jac2D,
                grid_x,
                grid_y,
                dt,
                mu_train,
            )
            Clist.append(Ci)

        C = np.vstack(Clist)
        C_shape = C.shape
        print(f"Stacked ECSW training matrix C shape: {C_shape}")

        C_ecm = np.ascontiguousarray(C, dtype=np.float64)
        b = np.ascontiguousarray(C_ecm.sum(axis=1), dtype=np.float64)

        # Build reduced basis for ECM from C^T
        rsvd = RandomizedSingularValueDecomposition()
        u, _, _, _ = rsvd.Calculate(C_ecm.T, 1e-8)

        selector = EmpiricalCubatureMethod()
        selector.SetUp(
            u,
            InitialCandidatesSet=None,
            constrain_sum_of_weights=True,
            constrain_conditions=False,
        )
        selector.Run()

        num_cells = (grid_x.size - 1) * (grid_y.size - 1)
        weights = np.zeros(num_cells, dtype=np.float64)
        weights[selector.z] = selector.w

        elapsed_ecsw = time.time() - t0
        denom = np.linalg.norm(b)
        if denom > 0.0:
            ecsw_residual = float(np.linalg.norm(C_ecm @ weights - b) / denom)
        else:
            ecsw_residual = np.nan

        np.save(weights_path, weights)
        print(f"ECSW weights saved to: {weights_path}")
        print(f"ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "hprom_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("HPROM Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"ECSW weights file not found: {weights_path}. "
                "Run with compute_ecsw=True first."
            )
        weights = np.load(weights_path, allow_pickle=False)
        print(f"Loaded ECSW weights from: {weights_path}")

    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)
    if weights.size != expected_num_cells:
        raise ValueError(
            f"ECSW weights size mismatch: got {weights.size}, "
            f"expected {expected_num_cells}."
        )

    n_ecsw_elements = int(np.sum(weights > 0.0))
    print(f"N_e (nonzero ECSW weights): {n_ecsw_elements}")

    # ------------------------------------------------------------------
    # Run HPROM
    # ------------------------------------------------------------------
    t0 = time.time()
    rom_red, hprom_stats = inviscid_burgers_implicit2D_LSPG_ecsw(
        grid_x=grid_x,
        grid_y=grid_y,
        weights=weights,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu=mu_rom,
        basis=basis_trunc,
        u_ref=u_ref,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )
    elapsed_hprom = time.time() - t0
    num_its, jac_time, res_time, ls_time = hprom_stats
    print(f"Elapsed HPROM time: {elapsed_hprom:.3e} seconds")
    print(f"HPROM Gauss-Newton iterations: {num_its}")
    print(f"HPROM timing breakdown (s): jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}")

    rom_snaps = u_ref[:, None] + basis_trunc @ rom_red

    # ------------------------------------------------------------------
    # Load HDM for comparison
    # ------------------------------------------------------------------
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
    print(f"Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    # ------------------------------------------------------------------
    # Save HPROM snapshots
    # ------------------------------------------------------------------
    rom_path = os.path.join(
        results_dir,
        f"hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"HPROM snapshots saved to: {rom_path}")

    # ------------------------------------------------------------------
    # Plot HDM vs HPROM
    # ------------------------------------------------------------------
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
        label="HPROM",
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
        f"hprom_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"HPROM comparison plot saved to: {fig_path}")

    # ------------------------------------------------------------------
    # Relative error
    # ------------------------------------------------------------------
    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"Relative error: {relative_error:.2f}%")

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(
        results_dir,
        f"hprom_summary_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.txt",
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
                    ("pod_dir_requested", pod_dir_requested),
                    ("pod_dir_used", pod_dir),
                    ("compute_ecsw", compute_ecsw),
                    ("num_modes_requested", num_modes),
                    ("snap_sample_factor", snap_sample_factor),
                    ("snap_time_offset", snap_time_offset),
                    ("mu_samples", mu_samples),
                    ("linear_solver", linear_solver),
                    (
                        "normal_eq_reg",
                        normal_eq_reg if str(linear_solver).strip().lower() == "normal_eq" else None,
                    ),
                ],
            ),
            (
                "discretization",
                [
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("num_cells_x", num_cells_x),
                    ("num_cells_y", num_cells_y),
                    ("full_state_size", w0.size),
                ],
            ),
            (
                "pod_basis",
                [
                    ("basis_size", n_keep),
                    ("n_available_modes", n_available),
                    ("energy_captured", energy_captured),
                    ("energy_lost", energy_lost),
                    ("centered_basis_used", centered_basis),
                    ("reference_source", ref_source),
                    ("u_ref_l2_norm", np.linalg.norm(u_ref)),
                ],
            ),
            (
                "ecsw",
                [
                    ("num_nonzero_weights", n_ecsw_elements),
                    ("weights_sum", float(np.sum(weights))),
                    ("ecsw_time_seconds", elapsed_ecsw),
                    ("ecsw_residual", ecsw_residual),
                    ("training_matrix_shape", C_shape),
                ],
            ),
            (
                "hprom_timing",
                [
                    ("total_hprom_time_seconds", elapsed_hprom),
                    ("avg_hprom_time_per_step_seconds", elapsed_hprom / num_steps),
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
                    ("hprom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("basis_npy", basis_path),
                    ("sigma_npy", sigma_path),
                    ("u_ref_npy", u_ref_path),
                    ("ecsw_weights_npy", weights_path),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                ],
            ),
        ],
    )
    print(f"HPROM text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)
