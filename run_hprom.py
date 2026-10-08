#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global affine HPROM (ECSW-LSPG) for the 2D inviscid Burgers problem
using the modern `burgers/` modules and save outputs consistently with
run_prom.py.
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import (
    plot_snaps,
    load_or_compute_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
    get_snapshot_params,
)
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
from burgers.linear_manifold import compute_ECSW_training_matrix_2D
from burgers.ecsw_utils import (
    build_ecsw_snapshot_plan,
    save_ecsw_sampling_3d_plot,
    greedy_nnls_ecsw_weights,
    greedy_nnls_bm_ecsw_weights,
    method_output_path,
    selection_output_tag,
    normalize_selection_method,
)
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import (
    RandomizedSingularValueDecomposition,
)
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


def main(
    mu1=4.56,
    mu2=0.019,
    compute_ecsw=True,
    num_modes=None,
    pod_dir="POD",
    results_dir="Results",
    snap_time_offset=3,
    mu_samples=None,
    ecsw_snapshot_percent=2.0,
    ecsw_random_seed=42,
    selection_method="ecm",
    max_cells=None,
    ecsw_tol_squared=1e-10,
    ecm_svd_tolerance=1e-8,
    ecm_svd_seed=None,
    linear_solver="normal_eq",
    normal_eq_reg=1e-12,
    bm_max_its=20,
    ecsw_snapshot_count=None,
    bm_jacobian="exact",
    bm_min_delta=1e-2,
    bm_candidate_score="signed_sum",
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
    results_dir : str
        Output directory for plots, summaries, and ROM snapshots.
    snap_time_offset : int
        Time offset between "current" and "previous" snapshots in ECSW
        training blocks.
    mu_samples : sequence of [mu1, mu2] or None
        Parameter points used to build ECSW weights.
    ecsw_snapshot_count : int or None
        Exact total training-pair count. Overrides percentage sampling when set.
    """

    selection_method = normalize_selection_method(selection_method)
    if bm_candidate_score not in ("signed_sum", "positive_norm"):
        raise ValueError("bm_candidate_score must be signed_sum or positive_norm.")
    if selection_method != "bm_ecsw" and bm_candidate_score != "signed_sum":
        raise ValueError("bm_candidate_score applies only to BM-ECSW selection.")
    if bm_jacobian not in ("exact", "gauss_newton"):
        raise ValueError("bm_jacobian must be exact or gauss_newton.")
    if not np.isfinite(bm_min_delta) or not 0 <= bm_min_delta < 1:
        raise ValueError("bm_min_delta must be in [0, 1); zero disables plateau stopping.")
    if int(bm_max_its) < 1:
        raise ValueError("bm_max_its must be positive.")
    if max_cells is not None and int(max_cells) < 1:
        raise ValueError("max_cells must be positive.")
    if selection_method == "ecm" and max_cells is not None:
        raise ValueError("max_cells is a safety cap for ECSW or BM-ECSW only.")
    if not np.isfinite(ecm_svd_tolerance) or not 0 < ecm_svd_tolerance < 1:
        raise ValueError("ecm_svd_tolerance must be in (0, 1).")

    if mu_samples is None:
        mu_samples = get_snapshot_params()
    mu_samples = [list(mu) for mu in mu_samples]

    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")
    ecsw_snapshot_percent = float(ecsw_snapshot_percent)
    if not np.isfinite(ecsw_snapshot_percent) or ecsw_snapshot_percent <= 0.0:
        raise ValueError("ecsw_snapshot_percent must be a finite value > 0.")
    if ecsw_snapshot_count is not None:
        if int(ecsw_snapshot_count) != ecsw_snapshot_count or ecsw_snapshot_count < 1:
            raise ValueError("ecsw_snapshot_count must be a positive integer.")
        ecsw_snapshot_count = int(ecsw_snapshot_count)
    ecsw_snapshot_mode = "global_param_time_stratified"
    ecsw_total_snapshots = ecsw_snapshot_count
    ecsw_total_snapshots_percent = (
        ecsw_snapshot_percent if ecsw_snapshot_count is None else None
    )
    ecsw_ensure_mu_coverage = True

    # ------------------------------------------------------------------
    # Folders
    # ------------------------------------------------------------------
    results_dir = str(results_dir)
    if not results_dir.strip():
        raise ValueError("results_dir must be a non-empty path.")
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
    # Problem setup from config
    # ------------------------------------------------------------------
    dt = DT
    num_steps = NUM_STEPS
    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()
    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1

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

    output_tag = selection_output_tag(
        selection_method, max_cells, ecsw_tol_squared,
        ecm_svd_tolerance, ecm_svd_seed, bm_candidate_score,
    )
    weights_path = method_output_path(
        weights_path, selection_method, max_cells, ecsw_tol_squared,
        ecm_svd_tolerance, ecm_svd_seed, bm_candidate_score,
    )
    plan_tag = ""
    if snap_time_offset != 3:
        plan_tag += f"_offset{snap_time_offset}"
    if ecsw_snapshot_count is not None:
        plan_tag += f"_snap{ecsw_snapshot_count}pairs"
    elif ecsw_snapshot_percent != 2.0:
        plan_tag += f"_snap{ecsw_snapshot_percent:g}pct"
    if plan_tag:
        output_tag += plan_tag
        stem, extension = os.path.splitext(weights_path)
        weights_path = f"{stem}{plan_tag}{extension}"
    run_tag = (
        f"{output_tag}_iter{bm_max_its}"
        if selection_method == "bm_ecsw"
        else output_tag
    )
    if selection_method == "bm_ecsw" and bm_jacobian == "gauss_newton":
        run_tag += "_gauss_newton"
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
    online_solver = linear_solver
    if selection_method == "bm_ecsw":
        online_solver = (
            "Newton for mode-weighted stationarity" if bm_jacobian == "exact"
            else "Gauss-Newton-type BM Jacobian (no residual Hessians)"
        )
    print(f"Reduced linear solver: {online_solver}")
    if selection_method != "bm_ecsw" and str(linear_solver).strip().lower() == "normal_eq":
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
    ecsw_residual_squared = None
    ecm_svd_rank = None
    ecm_svd_error = None
    ecm_internal_residual = None
    reduced_mesh_plot_path = None
    ecsw_sampling_3d_plot_path = None
    n_ecsw_elements = None
    ecsw_plan = None

    if compute_ecsw:
        Clist = []
        t0 = time.time()
        ecsw_plan = build_ecsw_snapshot_plan(
            num_steps=num_steps,
            snap_time_offset=snap_time_offset,
            num_mu=len(mu_samples),
            mode=ecsw_snapshot_mode,
            total_snapshots=ecsw_total_snapshots,
            total_snapshots_percent=ecsw_total_snapshots_percent,
            random_seed=ecsw_random_seed,
            ensure_mu_coverage=ecsw_ensure_mu_coverage,
            mu_points=mu_samples,
        )
        print(
            "[ECSW] Snapshot selection mode="
            f"{ecsw_plan['mode']}, selected {ecsw_plan['num_selected_total']} / "
            f"{ecsw_plan['num_candidates_total']} candidate pairs."
        )
        print(f"[ECSW] Selected snapshots per mu: {ecsw_plan['num_selected_per_mu']}")
        sampling_plot_name = (
            "hprom_ecsw_sampling_3d.png"
            if selection_method == "ecm" and not output_tag
            else f"hprom{output_tag}_sampling_3d.png"
        )
        ecsw_sampling_3d_plot_path = os.path.join(results_dir, sampling_plot_name)
        save_ecsw_sampling_3d_plot(
            mu_points=mu_samples,
            dt=dt,
            ecsw_plan=ecsw_plan,
            out_path=ecsw_sampling_3d_plot_path,
            title="HPROM ECSW Snapshot Selection in $(\\mu_1,\\mu_2,t)$",
        )
        print(f"[ECSW] 3D sampling plot saved to: {ecsw_sampling_3d_plot_path}")

        for imu, mu_train in enumerate(mu_samples):
            mu_snaps = load_or_compute_snaps(
                mu_train,
                grid_x,
                grid_y,
                w0,
                dt,
                num_steps,
                snap_folder=snap_folder,
            )

            now_cols = np.asarray(ecsw_plan["selected_now_cols_by_mu"][imu], dtype=int)
            prev_cols = now_cols - snap_time_offset
            snaps_now = mu_snaps[:, now_cols]
            snaps_prev = mu_snaps[:, prev_cols]

            if snaps_now.shape[1] == 0:
                continue

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
                u_ref=u_ref,
            )
            Clist.append(Ci)

        if not Clist:
            raise RuntimeError(
                "ECSW training produced zero columns for all mu samples. "
                "Increase ecsw_snapshot_percent or adjust snap_time_offset."
            )

        C = np.vstack(Clist)
        C_shape = C.shape
        print(f"Stacked ECSW training matrix C shape: {C_shape}")

        C_fit = np.ascontiguousarray(C, dtype=np.float64)
        b = np.ascontiguousarray(C_fit.sum(axis=1), dtype=np.float64)
        num_cells = C_fit.shape[1]

        if selection_method == "ecm":
            rsvd = RandomizedSingularValueDecomposition(RANDOM_SEED=ecm_svd_seed)
            u, _, _, ecm_svd_error = rsvd.Calculate(
                C_fit.T, ecm_svd_tolerance
            )
            ecm_svd_rank = u.shape[1]
            print(
                f"[ECM] SVD tolerance={ecm_svd_tolerance:.3e}, "
                f"retained rank={ecm_svd_rank}", flush=True,
            )
            selector = EmpiricalCubatureMethod()
            selector.SetUp(
                u,
                InitialCandidatesSet=None,
                constrain_sum_of_weights=True,
                constrain_conditions=False,
            )
            selector.Run()
            ecm_internal_residual = float(selector.nerrorACTUAL)
            weights = np.zeros(num_cells, dtype=np.float64)
            weights[selector.z] = selector.w
        elif selection_method == "ecsw":
            weights, ecsw_error_squared = greedy_nnls_ecsw_weights(
                C_fit,
                tolerance_squared=ecsw_tol_squared,
                max_cells=max_cells,
            )
            if ecsw_error_squared > ecsw_tol_squared:
                print(
                    "[ECSW] Training stopped before reaching tolerance: "
                    f"{ecsw_error_squared:.3e} > {ecsw_tol_squared:.3e}."
                )
        else:
            weights, ecsw_error_squared = greedy_nnls_bm_ecsw_weights(
                C_fit, n_keep,
                tolerance_squared=ecsw_tol_squared,
                max_cells=max_cells,
                candidate_score=bm_candidate_score,
            )
            if ecsw_error_squared > ecsw_tol_squared:
                print(
                    "[BM-ECSW] Training stopped before reaching tolerance: "
                    f"{ecsw_error_squared:.3e} > {ecsw_tol_squared:.3e}."
                )

        elapsed_ecsw = time.time() - t0
        denom = np.linalg.norm(b)
        if denom > 0.0:
            if selection_method == "bm_ecsw":
                support = np.flatnonzero(np.any(weights > 0, axis=1))
                fitted = np.zeros_like(b)
                for mode in range(n_keep):
                    fitted[mode::n_keep] = (
                        C_fit[mode::n_keep, support] @ weights[support, mode]
                    )
            else:
                fitted = C_fit @ weights
            ecsw_residual = float(np.linalg.norm(fitted - b) / denom)
            ecsw_residual_squared = ecsw_residual**2
        else:
            ecsw_residual = np.nan
            ecsw_residual_squared = np.nan

        np.save(weights_path, weights)
        print(f"ECSW weights saved to: {weights_path}")
        print(f"ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, f"hprom{output_tag}_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        mesh_mask = (
            np.any(weights > 0, axis=1)
            if selection_method == "bm_ecsw" else weights > 0
        )
        plt.spy(mesh_mask.reshape((num_cells_y, num_cells_x)))
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
    expected_shape = (
        (expected_num_cells, n_keep)
        if selection_method == "bm_ecsw" else (expected_num_cells,)
    )
    if weights.shape != expected_shape:
        raise ValueError(
            f"Cell weights shape mismatch: got {weights.shape}, "
            f"expected {expected_shape}."
        )

    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Cell weights must be finite and nonnegative.")
    n_ecsw_elements = int(np.sum(
        np.any(weights > 0.0, axis=1)
        if selection_method == "bm_ecsw" else weights > 0.0
    ))
    n_mode_weights = int(np.count_nonzero(weights)) if selection_method == "bm_ecsw" else None
    print(f"N_e (sampled cells): {n_ecsw_elements}")

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
        max_its=bm_max_its if selection_method == "bm_ecsw" else 20,
        bm_jacobian=bm_jacobian,
        min_delta=bm_min_delta if selection_method == "bm_ecsw" else 1e-2,
    )
    elapsed_hprom = time.time() - t0
    num_its, jac_time, res_time, ls_time = hprom_stats[:4]
    nonconverged_steps = hprom_stats[4] if selection_method == "bm_ecsw" else None
    bm_median_final_ratio = hprom_stats[5] if selection_method == "bm_ecsw" else None
    bm_max_final_ratio = hprom_stats[6] if selection_method == "bm_ecsw" else None
    bm_stop_reasons = hprom_stats[7] if selection_method == "bm_ecsw" else None
    print(f"Elapsed HPROM time: {elapsed_hprom:.3e} seconds")
    iteration_name = "Gauss-Newton"
    if selection_method == "bm_ecsw":
        iteration_name = "Newton" if bm_jacobian == "exact" else "Gauss-Newton-type"
    print(f"HPROM {iteration_name} iterations: {num_its}")
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
        f"hprom{run_tag}_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy",
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
        f"hprom{run_tag}_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.png",
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
        f"hprom{run_tag}_summary_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.txt",
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
                    ("results_dir", results_dir),
                    ("pod_dir_requested", pod_dir_requested),
                    ("pod_dir_used", pod_dir),
                    ("compute_ecsw", compute_ecsw),
                    ("selection_method", selection_method),
                    ("max_cells", max_cells),
                    ("ecsw_tol_squared", ecsw_tol_squared),
                    ("ecm_svd_tolerance", ecm_svd_tolerance if selection_method == "ecm" else None),
                    ("ecm_svd_seed", ecm_svd_seed if selection_method == "ecm" else None),
                    ("num_modes_requested", num_modes),
                    ("snap_time_offset", snap_time_offset),
                    ("ecsw_sampling_policy", ecsw_snapshot_mode),
                    ("ecsw_snapshot_percent", ecsw_total_snapshots_percent),
                    ("ecsw_snapshot_count", ecsw_total_snapshots),
                    ("ecsw_random_seed", ecsw_random_seed),
                    ("mu_samples", mu_samples),
                    ("linear_solver", online_solver),
                    ("bm_max_its", bm_max_its if selection_method == "bm_ecsw" else None),
                    ("bm_jacobian", bm_jacobian if selection_method == "bm_ecsw" else None),
                    ("bm_candidate_score", bm_candidate_score if selection_method == "bm_ecsw" else None),
                    ("bm_plateau_tolerance", bm_min_delta if selection_method == "bm_ecsw" else None),
                    (
                        "normal_eq_reg",
                        normal_eq_reg if selection_method != "bm_ecsw" and str(linear_solver).strip().lower() == "normal_eq" else None,
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
                    ("num_nonzero_mode_weights", n_mode_weights),
                    ("weights_sum", float(np.sum(weights))),
                    ("ecsw_time_seconds", elapsed_ecsw),
                    ("ecsw_residual", ecsw_residual),
                    ("ecsw_residual_squared", ecsw_residual_squared),
                    ("ecm_svd_rank", ecm_svd_rank),
                    ("ecm_svd_error", ecm_svd_error),
                    ("ecm_internal_residual", ecm_internal_residual),
                    ("ecsw_tolerance_met", ecsw_residual_squared <= ecsw_tol_squared if selection_method in ("ecsw", "bm_ecsw") and ecsw_residual_squared is not None else None),
                    ("training_matrix_shape", C_shape),
                    (
                        "snapshot_candidates_total",
                        ecsw_plan["num_candidates_total"] if ecsw_plan is not None else None,
                    ),
                    (
                        "snapshot_selected_total",
                        ecsw_plan["num_selected_total"] if ecsw_plan is not None else None,
                    ),
                    (
                        "snapshot_selected_per_mu",
                        ecsw_plan["num_selected_per_mu"] if ecsw_plan is not None else None,
                    ),
                ],
            ),
            (
                "hprom_timing",
                [
                    ("total_hprom_time_seconds", elapsed_hprom),
                    ("avg_hprom_time_per_step_seconds", elapsed_hprom / num_steps),
                    ("gn_iterations_total", num_its if selection_method != "bm_ecsw" else None),
                    ("bm_newton_iterations_total", num_its if selection_method == "bm_ecsw" and bm_jacobian == "exact" else None),
                    ("bm_gauss_newton_iterations_total", num_its if selection_method == "bm_ecsw" and bm_jacobian == "gauss_newton" else None),
                    ("bm_nonlinear_iterations_total", num_its if selection_method == "bm_ecsw" else None),
                    ("bm_projected_tolerance_not_met_steps", nonconverged_steps),
                    ("bm_median_final_projected_ratio", bm_median_final_ratio),
                    ("bm_max_final_projected_ratio", bm_max_final_ratio),
                    ("bm_stop_reasons", bm_stop_reasons),
                    ("bm_plateau_steps", bm_stop_reasons.get("plateau", 0) if bm_stop_reasons is not None else None),
                    ("bm_unsuccessful_steps", sum(count for reason, count in bm_stop_reasons.items() if reason not in ("projected_tolerance", "plateau")) if bm_stop_reasons is not None else None),
                    ("avg_nonlinear_iterations_per_step", num_its / num_steps),
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
                    ("ecsw_sampling_3d_png", ecsw_sampling_3d_plot_path),
                ],
            ),
        ],
    )
    print(f"HPROM text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run global linear HPROM (ECSW-LSPG) for a single parameter."
    )
    parser.add_argument("--mu1", type=float, default=4.56, help="First parameter value.")
    parser.add_argument("--mu2", type=float, default=0.019, help="Second parameter value.")
    parser.add_argument(
        "--num-modes",
        type=int,
        default=None,
        help="Number of POD modes to use (default: all available modes).",
    )
    parser.add_argument(
        "--pod-dir",
        type=str,
        default="POD",
        help="Directory containing POD artifacts (basis/sigma/u_ref/weights).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="Results",
        help="Output directory for snapshots, plots, and summaries.",
    )
    parser.add_argument(
        "--snap-time-offset",
        type=int,
        default=3,
        help="ECSW snapshot time offset.",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--ecsw-snapshot-percent",
        type=float,
        default=2.0,
        help="Percent of candidate ECSW snapshot pairs to select.",
    )
    sampling_group.add_argument(
        "--ecsw-snapshot-count",
        type=int,
        default=None,
        help="Exact total training-pair count, instead of a percentage.",
    )
    parser.add_argument(
        "--ecsw-random-seed",
        type=int,
        default=42,
        help="Random seed for ECSW snapshot selection.",
    )
    parser.add_argument(
        "--selection-method", choices=("ecm", "ecsw", "bm_ecsw"), default="ecm",
        help="Cell selector: ECM, classical ECSW, or mode-weighted BM-ECSW.",
    )
    parser.add_argument(
        "--bm-candidate-score", choices=("signed_sum", "positive_norm"),
        default="signed_sum",
        help="BM cell selection: signed correlation sum or norm of positive modal correlations.",
    )
    parser.add_argument(
        "--max-cells", type=int, default=None,
        help="Optional safety cap on selected cells; ECSW stops by tolerance.",
    )
    parser.add_argument(
        "--ecsw-tol-squared", type=float, default=1e-10,
        help="Stop ECSW or BM-ECSW at this global squared relative training error.",
    )
    parser.add_argument(
        "--ecm-svd-tolerance", type=float, default=1e-8,
        help="Relative SVD truncation tolerance used before ECM selection.",
    )
    parser.add_argument(
        "--ecm-svd-seed", type=int, default=None,
        help="Optional random seed for the ECM SVD.",
    )
    parser.add_argument(
        "--linear-solver",
        choices=("lstsq", "normal_eq"),
        default="lstsq",
        help="Reduced Gauss-Newton linear solver.",
    )
    parser.add_argument(
        "--normal-eq-reg",
        type=float,
        default=1e-12,
        help="Regularization used when linear_solver=normal_eq.",
    )
    parser.add_argument(
        "--bm-max-its",
        type=int,
        default=20,
        help="Maximum nonlinear iterations per online BM-ECSW time step.",
    )

    parser.add_argument(
        "--bm-jacobian", choices=("exact", "gauss_newton"), default="exact",
        help="BM derivative: exact Newton or omit residual Hessians.",
    )

    parser.add_argument(
        "--bm-min-delta", type=float, default=1e-2,
        help="BM projected-norm plateau tolerance; zero disables this stopping test.",
    )

    ecsw_group = parser.add_mutually_exclusive_group()
    ecsw_group.add_argument(
        "--compute-ecsw",
        dest="compute_ecsw",
        action="store_true",
        help="Build ECSW weights from training snapshots.",
    )
    ecsw_group.add_argument(
        "--no-compute-ecsw",
        dest="compute_ecsw",
        action="store_false",
        help="Reuse previously saved ECSW weights.",
    )
    parser.set_defaults(compute_ecsw=True)

    args = parser.parse_args()

    main(
        mu1=args.mu1,
        mu2=args.mu2,
        compute_ecsw=args.compute_ecsw,
        num_modes=args.num_modes,
        pod_dir=args.pod_dir,
        results_dir=args.results_dir,
        snap_time_offset=args.snap_time_offset,
        ecsw_snapshot_percent=args.ecsw_snapshot_percent,
        ecsw_random_seed=args.ecsw_random_seed,
        selection_method=args.selection_method,
        max_cells=args.max_cells,
        ecsw_tol_squared=args.ecsw_tol_squared,
        ecm_svd_tolerance=args.ecm_svd_tolerance,
        ecm_svd_seed=args.ecm_svd_seed,
        linear_solver=args.linear_solver,
        normal_eq_reg=args.normal_eq_reg,
        bm_max_its=args.bm_max_its,
        bm_jacobian=args.bm_jacobian,
        bm_min_delta=args.bm_min_delta,
        bm_candidate_score=args.bm_candidate_score,
        ecsw_snapshot_count=args.ecsw_snapshot_count,
    )
