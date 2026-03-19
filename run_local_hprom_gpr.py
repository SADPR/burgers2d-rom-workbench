#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run local POD-GPR HPROM (piecewise manifold + ECSW) and compare against HDM.
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
)
from burgers.pod_gpr_manifold import (
    compute_ECSW_training_matrix_2D_gpr_local,
    inviscid_burgers_implicit2D_LSPG_local_pod_gpr_ecsw,
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


def _resolved_cluster_jac_mode(jacobian_mode, model):
    mode = str(jacobian_mode).strip().lower()
    has_gpr = bool(model.get("has_gpr", False))
    analytic_ok = bool(model.get("analytic_jacobian_compatible", False))

    if not has_gpr:
        return "linear"

    if mode == "auto":
        return "analytic" if analytic_ok else "forward_fd"
    return mode


def load_local_pod_gpr_model(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Local POD-GPR model file not found: {filename}")

    data = np.load(filename, allow_pickle=True)

    u0_list = [np.asarray(u0, dtype=np.float64) for u0 in data["u0_list"]]
    V_list = [np.asarray(V, dtype=np.float64) for V in data["V_list"]]
    cluster_indices = [np.asarray(idx, dtype=int) for idx in data["cluster_indices"]]
    n_primary = int(data["n_primary"])

    models = [m for m in data["models"]]

    d_const = np.asarray(data["d_const"], dtype=np.float64)
    g_raw = np.asarray(data["g_list"], dtype=object)

    K = int(g_raw.shape[0])
    g_list = [[None for _ in range(K)] for _ in range(K)]
    for k in range(K):
        for l in range(K):
            g_list[k][l] = np.asarray(g_raw[k, l], dtype=np.float64).reshape(-1)

    return u0_list, V_list, cluster_indices, n_primary, models, d_const, g_list


def main(
    mu1=4.56,
    mu2=0.019,
    local_model_file=os.path.join("LocalPOD-GPR", "local_pod_gpr_all_offline.npz"),
    compute_ecsw=False,
    compute_ecm=None,
    weights_file=None,
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    snap_sample_factor=10,
    snap_time_offset=3,
    mu_samples=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-10,
    init_cluster=None,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
    verbose=True,
):
    if compute_ecm is not None:
        compute_ecsw = bool(compute_ecm)

    if mu_samples is None:
        mu_samples = [[4.25, 0.0225]]
    mu_samples = [list(mu) for mu in mu_samples]

    if snap_sample_factor < 1:
        raise ValueError("snap_sample_factor must be >= 1.")
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")

    jacobian_mode = str(jacobian_mode).strip().lower()
    if jacobian_mode not in ("auto", "analytic", "forward_fd", "central_fd"):
        raise ValueError(
            "jacobian_mode must be one of: 'auto', 'analytic', 'forward_fd', 'central_fd'."
        )

    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")

    if weights_file is None:
        weights_file = os.path.join(results_dir, "local_hprom_gpr_ecsw_weights.npy")
    legacy_weights_files = [
        os.path.join("LocalPOD-GPR", "ecsw_weights_local_pod_gpr.npy"),
        os.path.join("LocalPOD-GPR", "ecm_weights_local_pod_gpr.npy"),
    ]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1).copy()
    mu_rom = [float(mu1), float(mu2)]

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1
    num_cells = num_cells_x * num_cells_y

    print("\n====================================================")
    print("        LOCAL HPROM-GPR (Piecewise + ECSW)")
    print("====================================================")
    print(f"[LOCAL-HPROM-GPR] mu1={mu1:.3f}, mu2={mu2:.4f}")
    print(f"[LOCAL-HPROM-GPR] dt={dt}, num_steps={num_steps}")
    print(f"[LOCAL-HPROM-GPR] grid={num_cells_x}x{num_cells_y}")

    (
        u0_list,
        V_list,
        cluster_indices,
        n_primary,
        models,
        d_const,
        g_list,
    ) = load_local_pod_gpr_model(local_model_file)

    K = len(V_list)
    mode_counts = [int(V.shape[1]) for V in V_list]
    has_gpr_flags = [bool(m.get("has_gpr", False)) for m in models]
    analytic_ok_flags = [bool(m.get("analytic_jacobian_compatible", False)) for m in models]
    resolved_modes = [_resolved_cluster_jac_mode(jacobian_mode, m) for m in models]

    if jacobian_mode == "analytic":
        incompatible = [
            k
            for k, (has_gpr_k, analytic_ok_k) in enumerate(zip(has_gpr_flags, analytic_ok_flags))
            if has_gpr_k and not analytic_ok_k
        ]
        if incompatible:
            raise ValueError(
                "jacobian_mode='analytic' is not compatible with all local GPR models. "
                f"Incompatible clusters: {incompatible}. "
                "Use jacobian_mode='auto', 'forward_fd', or 'central_fd'."
            )

    print(f"[LOCAL-HPROM-GPR] Loaded local model: {local_model_file}")
    print(f"[LOCAL-HPROM-GPR] Number of clusters: {K}")
    print(
        "[LOCAL-HPROM-GPR] Retained modes per cluster: "
        f"min={np.min(mode_counts)}, max={np.max(mode_counts)}, avg={np.mean(mode_counts):.2f}"
    )
    print(
        "[LOCAL-HPROM-GPR] Clusters with active GPR closure: "
        f"{sum(has_gpr_flags)}/{K}"
    )
    print(f"[LOCAL-HPROM-GPR] Jacobian mode: {jacobian_mode}")

    C_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None
    weights_source = None

    if compute_ecsw:
        print("[LOCAL-HPROM-GPR] Computing ECSW weights...")
        t0 = time.time()
        Clist = []

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

            stop_col = num_steps
            snaps_now = mu_snaps[:, snap_time_offset:stop_col:snap_sample_factor]
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

            print(f"[LOCAL-HPROM-GPR] Building ECSW training block for mu={mu_train}")
            Ci = compute_ECSW_training_matrix_2D_gpr_local(
                snaps_now,
                snaps_prev,
                u0_list,
                V_list,
                models,
                n_primary,
                d_const,
                g_list,
                inviscid_burgers_res2D,
                inviscid_burgers_exact_jac2D,
                grid_x,
                grid_y,
                dt,
                mu_train,
                use_custom_predict=use_custom_predict,
                jacobian_mode=jacobian_mode,
                fd_eps=fd_eps,
                max_gn_its=max_its_ic,
                tol_rel=min_delta,
                init_cluster=init_cluster,
            )
            Clist.append(Ci)

        C = np.vstack(Clist)
        C_shape = C.shape
        print(f"[LOCAL-HPROM-GPR] Stacked ECSW training matrix C shape: {C_shape}")

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
        weights_source = "computed"

        print(f"[LOCAL-HPROM-GPR] ECSW weights saved to: {weights_file}")
        print(f"[LOCAL-HPROM-GPR] ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"[LOCAL-HPROM-GPR] ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "local_hprom_gpr_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("Local HPROM-GPR Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[LOCAL-HPROM-GPR] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if os.path.exists(weights_file):
            weights = np.asarray(np.load(weights_file, allow_pickle=False), dtype=np.float64)
            weights_source = "loaded"
            print(f"[LOCAL-HPROM-GPR] Loaded ECSW weights from: {weights_file}")
        else:
            weights = None
            for legacy in legacy_weights_files:
                if os.path.exists(legacy):
                    weights = np.asarray(np.load(legacy, allow_pickle=False), dtype=np.float64)
                    weights_source = f"loaded_legacy:{legacy}"
                    print(f"[LOCAL-HPROM-GPR] Loaded legacy ECSW weights from: {legacy}")
                    break
            if weights is None:
                raise FileNotFoundError(
                    f"ECSW weights file not found: {weights_file}. "
                    "Run with compute_ecsw=True first."
                )

    if weights.size != num_cells:
        raise ValueError(
            f"ECSW weights size mismatch: got {weights.size}, expected {num_cells}."
        )

    n_ecsw_elements = int(np.sum(weights > 0.0))
    print(f"[LOCAL-HPROM-GPR] N_e (nonzero ECSW weights): {n_ecsw_elements}")

    t0 = time.time()
    rom_snaps, stats = inviscid_burgers_implicit2D_LSPG_local_pod_gpr_ecsw(
        grid_x,
        grid_y,
        weights,
        w0,
        dt,
        num_steps,
        mu_rom,
        u0_list,
        V_list,
        models,
        n_primary,
        d_const,
        g_list,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        verbose=verbose,
        init_cluster=init_cluster,
        use_custom_predict=use_custom_predict,
        jacobian_mode=jacobian_mode,
        fd_eps=fd_eps,
        max_its=max_its,
        max_its_ic=max_its_ic,
        tol_ic=tol_ic,
    )
    elapsed_hprom = time.time() - t0

    num_its = int(stats["num_its"])
    jac_time = float(stats["jac_time"])
    res_time = float(stats["res_time"])
    ls_time = float(stats["ls_time"])
    cluster_history = np.asarray(stats.get("cluster_history", []), dtype=int)
    red_coords = np.asarray(stats.get("red_coords"), dtype=np.float64)

    if cluster_history.size > 0:
        cluster_counts = np.bincount(cluster_history, minlength=K)
        num_switches = int(np.sum(cluster_history[1:] != cluster_history[:-1]))
        initial_cluster = int(cluster_history[0])
        final_cluster = int(cluster_history[-1])
    else:
        cluster_counts = np.zeros(K, dtype=int)
        num_switches = 0
        initial_cluster = -1
        final_cluster = -1

    print(f"[LOCAL-HPROM-GPR] Elapsed HPROM time: {elapsed_hprom:.3e} seconds")
    print(f"[LOCAL-HPROM-GPR] Gauss-Newton iterations: {num_its}")
    print(
        "[LOCAL-HPROM-GPR] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )
    print(f"[LOCAL-HPROM-GPR] Cluster switches: {num_switches}")

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
    print(f"[LOCAL-HPROM-GPR] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"local_hprom_gpr_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[LOCAL-HPROM-GPR] HPROM snapshots saved to: {rom_path}")

    cluster_hist_path = os.path.join(
        results_dir,
        f"local_hprom_gpr_cluster_history_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(cluster_hist_path, cluster_history)

    red_coords_path = os.path.join(
        results_dir,
        f"local_hprom_gpr_red_coords_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(red_coords_path, red_coords)

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
        label="Local HPROM-GPR",
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
        f"local_hprom_gpr_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[LOCAL-HPROM-GPR] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[LOCAL-HPROM-GPR] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"local_hprom_gpr_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("local_model_file", local_model_file),
                    ("compute_ecsw", bool(compute_ecsw)),
                    ("weights_file", weights_file),
                    ("weights_source", weights_source),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("snap_sample_factor", snap_sample_factor),
                    ("snap_time_offset", snap_time_offset),
                    ("mu_samples", mu_samples),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("max_its_ic", max_its_ic),
                    ("tol_ic", tol_ic),
                    ("init_cluster", init_cluster),
                    ("use_custom_predict", bool(use_custom_predict)),
                    ("jacobian_mode", jacobian_mode),
                    ("fd_eps", fd_eps),
                    ("verbose", bool(verbose)),
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
                "local_gpr_model",
                [
                    ("num_clusters", K),
                    ("cluster_sizes_after_overlap", [int(idx.size) for idx in cluster_indices]),
                    ("retained_modes_per_cluster", mode_counts),
                    ("n_primary", n_primary),
                    ("clusters_with_gpr", int(sum(has_gpr_flags))),
                    ("clusters_analytic_compatible", int(sum(analytic_ok_flags))),
                    ("cluster_jacobian_modes", resolved_modes),
                    ("d_const_shape", d_const.shape),
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
                    ("avg_hprom_time_per_step_seconds", elapsed_hprom / max(1, num_steps)),
                    ("gn_iterations_total", num_its),
                    ("avg_gn_iterations_per_step", num_its / max(1, num_steps)),
                    ("jacobian_time_seconds", jac_time),
                    ("residual_time_seconds", res_time),
                    ("linear_solve_time_seconds", ls_time),
                    ("hdm_load_or_solve_time_seconds", elapsed_hdm),
                ],
            ),
            (
                "cluster_metrics",
                [
                    ("initial_cluster", initial_cluster),
                    ("final_cluster", final_cluster),
                    ("cluster_switches", num_switches),
                    ("cluster_visit_counts", cluster_counts.tolist()),
                    ("cluster_history_length", int(cluster_history.size)),
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
                    ("cluster_history_npy", cluster_hist_path),
                    ("red_coords_npy", red_coords_path),
                    ("comparison_plot_png", fig_path),
                    ("ecsw_weights_npy", weights_file),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[LOCAL-HPROM-GPR] Text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)

