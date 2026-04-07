#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global quadratic-manifold HPROM (QM-ECSW-LSPG) for the 2D inviscid Burgers
problem and compare against HDM.
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import (
    load_or_compute_snaps,
    plot_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
    get_snapshot_params,
)
from burgers.quadratic_manifold import (
    compute_ECSW_training_matrix_2D_qm,
    inviscid_burgers_implicit2D_LSPG_qm_ecsw,
)
from burgers.quadratic_manifold_utils import u_qm
from burgers.ecsw_utils import build_ecsw_snapshot_plan
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
    mu1=4.56,
    mu2=0.019,
    qm_dir="Quadratic",
    compute_ecsw=False,
    compute_ecm=None,
    weights_file=None,
    dt=DT,
    num_steps=NUM_STEPS,
    snap_time_offset=3,
    mu_samples=None,
    ecsw_snapshot_percent=2.0,
    ecsw_random_seed=42,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    max_its_q0=20,
    tol_q0=1e-6,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    if compute_ecm is not None:
        compute_ecsw = bool(compute_ecm)

    if mu_samples is None:
        mu_samples = get_snapshot_params()
    mu_samples = [list(mu) for mu in mu_samples]

    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")
    ecsw_snapshot_percent = float(ecsw_snapshot_percent)
    if not np.isfinite(ecsw_snapshot_percent) or ecsw_snapshot_percent <= 0.0:
        raise ValueError("ecsw_snapshot_percent must be a finite value > 0.")
    ecsw_snapshot_mode = "global_param_time_stratified"
    ecsw_total_snapshots = None
    ecsw_total_snapshots_percent = ecsw_snapshot_percent
    ecsw_ensure_mu_coverage = True

    results_dir = "Results"
    snap_folder = os.path.join(results_dir, "param_snaps")
    if weights_file is None:
        weights_file = os.path.join(results_dir, "hqprom_ecsw_weights.npy")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()
    mu_rom = [mu1, mu2]

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1
    num_cells = num_cells_x * num_cells_y

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

    n = V.shape[1]
    m = H.shape[1]
    print(f"[HQPROM] Loaded manifold from '{qm_dir}'")
    print(f"[HQPROM] V shape={V.shape}, H shape={H.shape}, u_ref shape={u_ref.shape}")
    print(f"[HQPROM] Reduced linear solver: {linear_solver}")
    if str(linear_solver).strip().lower() == "normal_eq":
        print(f"[HQPROM] normal_eq_reg: {float(normal_eq_reg):.3e}")

    # ------------------------------------------------------------------
    # ECSW weights: build or load
    # ------------------------------------------------------------------
    C_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None
    ecsw_plan = None

    if compute_ecsw:
        print("[HQPROM] Computing ECSW weights...")
        t0 = time.time()
        Clist = []
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
            "[HQPROM] ECSW snapshot selection mode="
            f"{ecsw_plan['mode']}, selected {ecsw_plan['num_selected_total']} / "
            f"{ecsw_plan['num_candidates_total']} candidate pairs."
        )
        print(f"[HQPROM] Selected snapshots per mu: {ecsw_plan['num_selected_per_mu']}")

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

            print(f"[HQPROM] Building ECSW training block for mu={mu_train}")
            Ci = compute_ECSW_training_matrix_2D_qm(
                snaps_now,
                snaps_prev,
                V,
                H,
                u_ref,
                inviscid_burgers_res2D,
                inviscid_burgers_exact_jac2D,
                grid_x,
                grid_y,
                dt,
                mu_train,
            )
            Clist.append(Ci)

        if not Clist:
            raise RuntimeError(
                "ECSW training produced zero columns for all mu samples. "
                "Increase ecsw_snapshot_percent or adjust snap_time_offset."
            )

        C = np.vstack(Clist)
        C_shape = C.shape
        print(f"[HQPROM] Stacked ECSW training matrix C shape: {C_shape}")

        C_ecm = np.ascontiguousarray(C, dtype=np.float64)
        b = np.ascontiguousarray(C_ecm.sum(axis=1), dtype=np.float64)

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

        weights = np.zeros(num_cells, dtype=np.float64)
        weights[selector.z] = selector.w

        elapsed_ecsw = time.time() - t0
        denom = np.linalg.norm(b)
        if denom > 0.0:
            ecsw_residual = float(np.linalg.norm(C_ecm @ weights - b) / denom)
        else:
            ecsw_residual = np.nan

        np.save(weights_file, weights)
        print(f"[HQPROM] ECSW weights saved to: {weights_file}")
        print(f"[HQPROM] ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"[HQPROM] ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "hqprom_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("HQPROM Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[HQPROM] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"ECSW weights file not found: {weights_file}. "
                "Run with compute_ecsw=True first."
            )
        weights = np.load(weights_file, allow_pickle=False)
        print(f"[HQPROM] Loaded ECSW weights from: {weights_file}")

    if weights.size != num_cells:
        raise ValueError(
            f"ECSW weights size mismatch: got {weights.size}, expected {num_cells}."
        )

    n_ecsw_elements = int(np.count_nonzero(weights))
    print(f"[HQPROM] Nonzero ECSW elements: {n_ecsw_elements} / {num_cells}")

    # ------------------------------------------------------------------
    # HQPROM solve
    # ------------------------------------------------------------------
    t0 = time.time()
    red_coords, stats = inviscid_burgers_implicit2D_LSPG_qm_ecsw(
        grid_x,
        grid_y,
        weights,
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
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )
    elapsed_hqprom = time.time() - t0

    num_its, jac_time, res_time, ls_time = stats
    print(f"[HQPROM] Total solve time: {elapsed_hqprom:.3e} seconds")
    print(f"[HQPROM] Gauss-Newton iterations: {num_its}")
    print(
        "[HQPROM] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )

    # Reconstruct full snapshots from reduced coordinates
    rom_snaps_qm = np.zeros((w0.size, num_steps + 1), dtype=np.float64)
    for k in range(num_steps + 1):
        rom_snaps_qm[:, k] = u_qm(red_coords[:, k], V, H, u_ref)

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
    print(f"[HQPROM] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    # ------------------------------------------------------------------
    # Save snapshots and plot
    # ------------------------------------------------------------------
    rom_path = os.path.join(
        results_dir,
        f"hqprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps_qm)
    print(f"[HQPROM] ROM snapshots saved to: {rom_path}")

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
        label="HQPROM",
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
        f"hqprom_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[HQPROM] Comparison plot saved to: {fig_path}")

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------
    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps_qm) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[HQPROM] Relative error: {relative_error:.2f}%")

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(
        results_dir,
        f"hqprom_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("compute_ecsw", compute_ecsw),
                    ("qm_dir", qm_dir),
                    ("weights_file", weights_file),
                    ("snap_time_offset", snap_time_offset),
                    ("ecsw_sampling_policy", ecsw_snapshot_mode),
                    ("ecsw_snapshot_percent", ecsw_snapshot_percent),
                    ("ecsw_random_seed", ecsw_random_seed),
                    ("mu_samples", mu_samples),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("max_its_q0", max_its_q0),
                    ("tol_q0", tol_q0),
                    ("linear_solver", linear_solver),
                    (
                        "normal_eq_reg",
                        normal_eq_reg if str(linear_solver).strip().lower() == "normal_eq" else None,
                    ),
                    ("snap_folder", snap_folder),
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
                "ecsw",
                [
                    ("num_nonzero_weights", n_ecsw_elements),
                    ("weights_sum", float(np.sum(weights))),
                    ("ecsw_time_seconds", elapsed_ecsw),
                    ("ecsw_residual", ecsw_residual),
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
                "hqprom_timing",
                [
                    ("total_hqprom_time_seconds", elapsed_hqprom),
                    ("avg_hqprom_time_per_step_seconds", elapsed_hqprom / num_steps),
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
                    ("hqprom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("summary_txt", report_path),
                    ("ecsw_weights_npy", weights_file),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                    ("qm_V_npy", V_path),
                    ("qm_H_npy", H_path),
                    ("qm_uref_npy", uref_path),
                    ("qm_sigma_npy", sigma_path if os.path.exists(sigma_path) else None),
                    ("qm_metadata_npz", metadata_path if os.path.exists(metadata_path) else None),
                ],
            ),
        ],
    )
    print(f"[HQPROM] Text summary saved to: {report_path}")

    return elapsed_hqprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)
