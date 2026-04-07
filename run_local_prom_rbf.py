#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run local POD-RBF PROM (piecewise manifold) using LocalPOD-RBF offline data
and compare against HDM.
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.pod_rbf_manifold import inviscid_burgers_implicit2D_LSPG_local_pod_rbf
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


def load_local_pod_rbf_model(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Local POD-RBF model file not found: {filename}")

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
    local_model_file=os.path.join("LocalPOD-RBF", "local_pod_rbf_all_offline.npz"),
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-10,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
    init_cluster=None,
    verbose=True,
    selector_mode="nonlinear",
):
    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1).copy()
    mu_rom = [float(mu1), float(mu2)]
    selector_mode = str(selector_mode).strip().lower()
    if selector_mode not in ("linear", "nonlinear"):
        raise ValueError("selector_mode must be one of: 'linear', 'nonlinear'.")

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1

    print("\n====================================================")
    print("           LOCAL POD-RBF PROM")
    print("====================================================")
    print(f"[LOCAL-PROM-RBF] mu1={mu1:.3f}, mu2={mu2:.4f}")
    print(f"[LOCAL-PROM-RBF] dt={dt}, num_steps={num_steps}")
    print(f"[LOCAL-PROM-RBF] grid={num_cells_x}x{num_cells_y}")

    (
        u0_list,
        V_list,
        cluster_indices,
        n_primary,
        models,
        d_const,
        g_list,
    ) = load_local_pod_rbf_model(local_model_file)

    K = len(V_list)
    mode_counts = [int(V.shape[1]) for V in V_list]
    has_rbf_flags = [bool(m.get("has_rbf", False)) for m in models]

    print(f"[LOCAL-PROM-RBF] Loaded local model: {local_model_file}")
    print(f"[LOCAL-PROM-RBF] Number of clusters: {K}")
    print(
        "[LOCAL-PROM-RBF] Retained modes per cluster: "
        f"min={np.min(mode_counts)}, max={np.max(mode_counts)}, avg={np.mean(mode_counts):.2f}"
    )
    print(
        "[LOCAL-PROM-RBF] Clusters with active RBF closure: "
        f"{sum(has_rbf_flags)}/{K}"
    )
    print(f"[LOCAL-PROM-RBF] Cluster selector mode: {selector_mode}")
    print(f"[LOCAL-PROM-RBF] Reduced linear solver: {linear_solver}")
    if str(linear_solver).strip().lower() == "normal_eq":
        print(f"[LOCAL-PROM-RBF] normal_eq_reg: {float(normal_eq_reg):.3e}")

    t0 = time.time()
    rom_snaps, stats = inviscid_burgers_implicit2D_LSPG_local_pod_rbf(
        grid_x,
        grid_y,
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
        max_its=max_its,
        max_its_ic=max_its_ic,
        tol_ic=tol_ic,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
        selector_mode=selector_mode,
    )
    elapsed_rom = time.time() - t0

    num_its = int(stats["num_its"])
    jac_time = float(stats["jac_time"])
    res_time = float(stats["res_time"])
    ls_time = float(stats["ls_time"])

    cluster_history = np.asarray(stats.get("cluster_history", []), dtype=int)
    red_coords = stats.get("red_coords", None)

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

    print(f"[LOCAL-PROM-RBF] Total solve time: {elapsed_rom:.3e} seconds")
    print(f"[LOCAL-PROM-RBF] Gauss-Newton iterations: {num_its}")
    print(
        "[LOCAL-PROM-RBF] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )
    print(f"[LOCAL-PROM-RBF] Cluster switches: {num_switches}")

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
    print(f"[LOCAL-PROM-RBF] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"local_prom_rbf_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[LOCAL-PROM-RBF] ROM snapshots saved to: {rom_path}")

    cluster_hist_path = os.path.join(
        results_dir,
        f"local_prom_rbf_cluster_history_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(cluster_hist_path, cluster_history)

    red_coords_path = None
    if red_coords is not None:
        red_coords_path = os.path.join(
            results_dir,
            f"local_prom_rbf_red_coords_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
        )
        np.save(red_coords_path, np.asarray(red_coords, dtype=np.float64))

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
        label="Local PROM-RBF",
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
        f"local_prom_rbf_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[LOCAL-PROM-RBF] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[LOCAL-PROM-RBF] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"local_prom_rbf_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("max_its_ic", max_its_ic),
                    ("tol_ic", tol_ic),
                    ("linear_solver", linear_solver),
                    (
                        "normal_eq_reg",
                        normal_eq_reg if str(linear_solver).strip().lower() == "normal_eq" else None,
                    ),
                    ("init_cluster", init_cluster),
                    ("selector_mode", selector_mode),
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
                "local_rbf_model",
                [
                    ("num_clusters", K),
                    ("cluster_sizes_after_overlap", [int(idx.size) for idx in cluster_indices]),
                    ("retained_modes_per_cluster", mode_counts),
                    ("n_primary", n_primary),
                    ("clusters_with_rbf", int(sum(has_rbf_flags))),
                    ("d_const_shape", d_const.shape),
                ],
            ),
            (
                "local_prom_rbf_timing",
                [
                    ("total_local_prom_rbf_time_seconds", elapsed_rom),
                    (
                        "avg_local_prom_rbf_time_per_step_seconds",
                        elapsed_rom / max(1, num_steps),
                    ),
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
                    ("local_prom_rbf_snapshots_npy", rom_path),
                    ("cluster_history_npy", cluster_hist_path),
                    ("red_coords_npy", red_coords_path),
                    ("comparison_plot_png", fig_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[LOCAL-PROM-RBF] Text summary saved to: {report_path}")

    return elapsed_rom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019)
