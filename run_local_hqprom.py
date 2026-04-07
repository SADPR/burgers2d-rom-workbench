#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run local quadratic-manifold HPROM (piecewise quadratic + ECSW) and compare
against HDM.
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
    compute_ECSW_training_matrix_2D_qm_local,
    inviscid_burgers_implicit2D_LSPG_local_qm_ecsw,
)
from burgers.ecsw_utils import (
    build_ecsw_snapshot_plan,
    select_local_cluster_count_snapshot_cols,
    save_ecsw_sampling_3d_plot,
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


def load_local_qm_model(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Local quadratic model not found: {filename}")

    data = np.load(filename, allow_pickle=True)

    S_shape = tuple(data["S_shape"])
    u0_list = [np.asarray(u0, dtype=np.float64) for u0 in data["u0_list"]]
    uc_list = [np.asarray(uc, dtype=np.float64) for uc in data["uc_list"]]
    V_list = [np.asarray(V, dtype=np.float64) for V in data["V_list"]]
    H_list = [np.asarray(H, dtype=np.float64) for H in data["H_list"]]
    cluster_indices = [np.asarray(idx, dtype=int) for idx in data["cluster_indices"]]

    d_const = np.asarray(data["d_const"], dtype=np.float64)

    g_raw = np.asarray(data["g_list"], dtype=object)
    K = g_raw.shape[0]
    g_list = [[None for _ in range(K)] for _ in range(K)]
    for k in range(K):
        for l in range(K):
            g_list[k][l] = np.asarray(g_raw[k, l], dtype=np.float64).reshape(-1)

    m_list = None
    if "m_list" in data.files:
        m_raw = np.asarray(data["m_list"], dtype=object)
        m_list = [[None for _ in range(K)] for _ in range(K)]
        for k in range(K):
            for l in range(K):
                m_list[k][l] = np.asarray(m_raw[k, l], dtype=np.float64).reshape(-1)

    n_list = (
        np.asarray(data["n_list"], dtype=int).tolist()
        if "n_list" in data.files
        else [int(V.shape[1]) for V in V_list]
    )
    n_trad_list = (
        np.asarray(data["n_trad_list"], dtype=int).tolist()
        if "n_trad_list" in data.files
        else None
    )

    return (
        S_shape,
        u0_list,
        uc_list,
        V_list,
        H_list,
        cluster_indices,
        d_const,
        g_list,
        m_list,
        n_list,
        n_trad_list,
    )


def main(
    mu1=4.56,
    mu2=0.019,
    local_model_file=os.path.join("LocalQuadratic", "local_qm_data.npz"),
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
    init_cluster=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
    selector_mode="quadratic",
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
    ecsw_snapshot_mode = "local_cluster_param_time_stratified"
    ecsw_total_snapshots = None
    ecsw_total_snapshots_percent = ecsw_snapshot_percent
    ecsw_ensure_mu_coverage = True
    ecsw_cluster_min_per_cluster = 1

    results_dir = "Results"
    snap_folder = os.path.join(results_dir, "param_snaps")
    if weights_file is None:
        weights_file = os.path.join(results_dir, "local_hqprom_ecsw_weights.npy")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()
    mu_rom = [mu1, mu2]
    selector_mode = str(selector_mode).strip().lower()
    if selector_mode not in ("linear", "quadratic"):
        raise ValueError("selector_mode must be one of: 'linear', 'quadratic'.")

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1
    num_cells = num_cells_x * num_cells_y

    print("\n====================================================")
    print("      LOCAL HQPROM (Piecewise Quadratic + ECSW)")
    print("====================================================")
    print(f"[LOCAL-HQPROM] mu1={mu1:.3f}, mu2={mu2:.4f}")
    print(f"[LOCAL-HQPROM] dt={dt}, num_steps={num_steps}")
    print(f"[LOCAL-HQPROM] grid={num_cells_x}x{num_cells_y}")

    (
        S_shape,
        u0_list,
        uc_list,
        V_list,
        H_list,
        cluster_indices,
        d_const,
        g_list,
        m_list,
        n_list,
        n_trad_list,
    ) = load_local_qm_model(local_model_file)

    K = len(V_list)
    mode_counts = [int(V.shape[1]) for V in V_list]
    print(f"[LOCAL-HQPROM] Loaded local model: {local_model_file}")
    print(f"[LOCAL-HQPROM] Number of clusters: {K}")
    print(
        "[LOCAL-HQPROM] Retained modes per cluster: "
        f"min={np.min(mode_counts)}, max={np.max(mode_counts)}, avg={np.mean(mode_counts):.2f}"
    )
    print(f"[LOCAL-HQPROM] Reduced linear solver: {linear_solver}")
    if str(linear_solver).strip().lower() == "normal_eq":
        print(f"[LOCAL-HQPROM] normal_eq_reg: {float(normal_eq_reg):.3e}")
    print(f"[LOCAL-HQPROM] Cluster selector mode: {selector_mode}")

    # ------------------------------------------------------------------
    # ECSW weights: build or load
    # ------------------------------------------------------------------
    C_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None
    ecsw_sampling_3d_plot_path = None
    ecsw_plan = None

    if compute_ecsw:
        print("[LOCAL-HQPROM] Computing ECSW weights...")
        t0 = time.time()
        Clist = []
        candidate_now_cols = np.arange(snap_time_offset, num_steps, dtype=int)
        n_candidates_per_mu = int(candidate_now_cols.size)
        if n_candidates_per_mu == 0:
            raise ValueError(
                "No valid ECSW snapshot pairs with current (num_steps, snap_time_offset)."
            )
        ecsw_plan = {
            "mode": ecsw_snapshot_mode,
            "candidate_now_cols": candidate_now_cols,
            "selected_now_cols_by_mu": [],
            "num_candidates_per_mu": n_candidates_per_mu,
            "num_candidates_total": int(n_candidates_per_mu * len(mu_samples)),
            "num_selected_per_mu": [],
            "num_selected_total": 0,
            "cluster_candidates_per_mu": [],
            "cluster_selected_per_mu": [],
            "cluster_min_per_cluster": int(ecsw_cluster_min_per_cluster),
            "cluster_reference": "uc_list",
        }
        mu_plan = build_ecsw_snapshot_plan(
            num_steps=num_steps,
            snap_time_offset=snap_time_offset,
            num_mu=len(mu_samples),
            mode="global_param_time_stratified",
            total_snapshots=ecsw_total_snapshots,
            total_snapshots_percent=ecsw_total_snapshots_percent,
            random_seed=ecsw_random_seed,
            ensure_mu_coverage=ecsw_ensure_mu_coverage,
            mu_points=mu_samples,
        )
        ecsw_plan["mu_target_selected_per_mu"] = [
            int(v) for v in mu_plan["num_selected_per_mu"]
        ]
        print(
            "[LOCAL-HQPROM] ECSW snapshot selection mode="
            "local_cluster_param_time_stratified "
            f"(target per mu: {ecsw_plan['mu_target_selected_per_mu']})."
        )

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

            target_count = int(ecsw_plan["mu_target_selected_per_mu"][imu])
            sel = select_local_cluster_count_snapshot_cols(
                mu_snaps=mu_snaps,
                candidate_now_cols=ecsw_plan["candidate_now_cols"],
                cluster_centers=uc_list,
                target_count=target_count,
                random_seed=int(ecsw_random_seed) + 10007 * int(imu),
                min_per_cluster=ecsw_cluster_min_per_cluster,
            )
            now_cols = np.asarray(sel["selected_now_cols"], dtype=int)
            cand_counts = np.asarray(sel["candidate_counts_by_cluster"], dtype=int)
            sel_counts = np.asarray(sel["selected_counts_by_cluster"], dtype=int)

            ecsw_plan["selected_now_cols_by_mu"].append(now_cols)
            ecsw_plan["num_selected_per_mu"].append(int(now_cols.size))
            ecsw_plan["cluster_candidates_per_mu"].append(cand_counts.tolist())
            ecsw_plan["cluster_selected_per_mu"].append(sel_counts.tolist())
            print(
                f"[LOCAL-HQPROM] mu={mu_train}: selected {int(now_cols.size)} "
                f"snapshot pairs (cluster counts: {sel_counts.tolist()})."
            )
            prev_cols = now_cols - snap_time_offset
            snaps_now = mu_snaps[:, now_cols]
            snaps_prev = mu_snaps[:, prev_cols]

            if snaps_now.shape[1] == 0:
                continue

            print(f"[LOCAL-HQPROM] Building ECSW training block for mu={mu_train}")
            Ci = compute_ECSW_training_matrix_2D_qm_local(
                snaps_now,
                snaps_prev,
                u0_list,
                uc_list,
                V_list,
                H_list,
                d_const,
                g_list,
                inviscid_burgers_res2D,
                inviscid_burgers_exact_jac2D,
                grid_x,
                grid_y,
                dt,
                mu_train,
                selector_mode=selector_mode,
                m_list=m_list,
            )
            Clist.append(Ci)

        ecsw_plan["num_selected_total"] = int(np.sum(ecsw_plan["num_selected_per_mu"]))
        print(
            "[LOCAL-HQPROM] ECSW snapshot selection mode="
            f"{ecsw_plan['mode']}, selected {ecsw_plan['num_selected_total']} / "
            f"{ecsw_plan['num_candidates_total']} candidate pairs."
        )
        print(
            f"[LOCAL-HQPROM] Selected snapshots per mu: {ecsw_plan['num_selected_per_mu']}"
        )
        ecsw_sampling_3d_plot_path = os.path.join(
            results_dir, "local_hqprom_ecsw_sampling_3d.png"
        )
        save_ecsw_sampling_3d_plot(
            mu_points=mu_samples,
            dt=dt,
            ecsw_plan=ecsw_plan,
            out_path=ecsw_sampling_3d_plot_path,
            title="Local HQPROM ECSW Snapshot Selection in $(\\mu_1,\\mu_2,t)$",
        )
        print(f"[LOCAL-HQPROM] 3D sampling plot saved to: {ecsw_sampling_3d_plot_path}")

        if not Clist:
            raise RuntimeError(
                "ECSW training produced zero columns for all mu samples. "
                "Increase snapshot selection percentage/count or adjust snap_time_offset."
            )

        C = np.vstack(Clist)
        C_shape = C.shape
        print(f"[LOCAL-HQPROM] Stacked ECSW training matrix C shape: {C_shape}")

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
        print(f"[LOCAL-HQPROM] ECSW weights saved to: {weights_file}")
        print(f"[LOCAL-HQPROM] ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"[LOCAL-HQPROM] ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "local_hqprom_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("Local HQPROM Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[LOCAL-HQPROM] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"ECSW weights file not found: {weights_file}. "
                "Run with compute_ecsw=True first."
            )
        weights = np.load(weights_file, allow_pickle=False)
        print(f"[LOCAL-HQPROM] Loaded ECSW weights from: {weights_file}")

    if weights.size != num_cells:
        raise ValueError(
            f"ECSW weights size mismatch: got {weights.size}, expected {num_cells}."
        )

    n_ecsw_elements = int(np.count_nonzero(weights))
    print(f"[LOCAL-HQPROM] Nonzero ECSW elements: {n_ecsw_elements} / {num_cells}")

    # ------------------------------------------------------------------
    # Local HQPROM solve
    # ------------------------------------------------------------------
    t0 = time.time()
    rom_snaps, stats = inviscid_burgers_implicit2D_LSPG_local_qm_ecsw(
        grid_x,
        grid_y,
        weights,
        w0,
        dt,
        num_steps,
        mu_rom,
        u0_list,
        V_list,
        H_list,
        uc_list,
        d_const,
        g_list,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        init_cluster=init_cluster,
        max_its=max_its,
        max_its_q0=max_its_q0,
        tol_q0=tol_q0,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
        selector_mode=selector_mode,
        m_list=m_list,
    )
    elapsed_hqprom = time.time() - t0

    num_its = int(stats["num_its"])
    jac_time = float(stats["jac_time"])
    res_time = float(stats["res_time"])
    ls_time = float(stats["ls_time"])
    cluster_history = np.asarray(stats["cluster_history"], dtype=int)
    cluster_counts = np.bincount(cluster_history, minlength=K)
    num_switches = int(np.sum(cluster_history[1:] != cluster_history[:-1]))
    red_coords = np.asarray(stats.get("red_coords"), dtype=np.float64)

    print(f"[LOCAL-HQPROM] Total solve time: {elapsed_hqprom:.3e} seconds")
    print(f"[LOCAL-HQPROM] Gauss-Newton iterations: {num_its}")
    print(
        "[LOCAL-HQPROM] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )
    print(f"[LOCAL-HQPROM] Cluster switches: {num_switches}")

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
    print(f"[LOCAL-HQPROM] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    # ------------------------------------------------------------------
    # Save HQPROM snapshots
    # ------------------------------------------------------------------
    rom_path = os.path.join(
        results_dir,
        f"local_hqprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[LOCAL-HQPROM] HQPROM snapshots saved to: {rom_path}")

    # ------------------------------------------------------------------
    # Plot HDM vs Local HQPROM
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
        label="Local HQPROM",
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
        f"local_hqprom_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[LOCAL-HQPROM] Comparison plot saved to: {fig_path}")

    # ------------------------------------------------------------------
    # Relative error
    # ------------------------------------------------------------------
    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[LOCAL-HQPROM] Relative error: {relative_error:.2f}%")

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(
        results_dir,
        f"local_hqprom_summary_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.txt",
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
                    ("compute_ecsw", compute_ecsw),
                    ("local_model_file", local_model_file),
                    ("weights_file", weights_file),
                    ("snap_time_offset", snap_time_offset),
                    ("ecsw_sampling_policy", ecsw_snapshot_mode),
                    ("ecsw_snapshot_percent", ecsw_snapshot_percent),
                    ("ecsw_random_seed", ecsw_random_seed),
                    ("ecsw_cluster_min_per_cluster", ecsw_cluster_min_per_cluster),
                    ("mu_samples", mu_samples),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("max_its_q0", max_its_q0),
                    ("tol_q0", tol_q0),
                    ("init_cluster", init_cluster),
                    ("selector_mode", selector_mode),
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
                "local_model",
                [
                    ("offline_snapshot_matrix_shape", S_shape),
                    ("num_clusters", K),
                    ("cluster_sizes_after_overlap", [int(idx.size) for idx in cluster_indices]),
                    ("n_qm_per_cluster", n_list),
                    ("retained_modes_per_cluster", mode_counts),
                    ("n_trad_per_cluster", n_trad_list),
                    ("d_const_shape", d_const.shape),
                    ("m_list_available_in_npz", m_list is not None),
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
                    (
                        "snapshot_cluster_candidates_per_mu",
                        ecsw_plan.get("cluster_candidates_per_mu") if ecsw_plan is not None else None,
                    ),
                    (
                        "snapshot_cluster_selected_per_mu",
                        ecsw_plan.get("cluster_selected_per_mu") if ecsw_plan is not None else None,
                    ),
                ],
            ),
            (
                "local_hqprom_timing",
                [
                    ("total_local_hqprom_time_seconds", elapsed_hqprom),
                    ("avg_local_hqprom_time_per_step_seconds", elapsed_hqprom / num_steps),
                    ("gn_iterations_total", num_its),
                    ("avg_gn_iterations_per_step", num_its / num_steps),
                    ("jacobian_time_seconds", jac_time),
                    ("residual_time_seconds", res_time),
                    ("linear_solve_time_seconds", ls_time),
                    ("hdm_load_or_solve_time_seconds", elapsed_hdm),
                ],
            ),
            (
                "cluster_metrics",
                [
                    ("initial_cluster", int(cluster_history[0])),
                    ("final_cluster", int(cluster_history[-1])),
                    ("cluster_switches", num_switches),
                    ("cluster_visit_counts", cluster_counts.tolist()),
                ],
            ),
            (
                "reduced_coordinates",
                [
                    ("red_coords_shape", red_coords.shape),
                    ("max_active_n_qm", int(red_coords.shape[0])),
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
                    ("local_hqprom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("ecsw_weights_npy", weights_file),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                    ("ecsw_sampling_3d_png", ecsw_sampling_3d_plot_path),
                    ("local_qm_model_npz", local_model_file),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[LOCAL-HQPROM] Text summary saved to: {report_path}")

    return elapsed_hqprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)
