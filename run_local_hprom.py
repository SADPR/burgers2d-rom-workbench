#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run local affine HPROM (piecewise-linear manifold + ECSW) and compare against HDM.
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
from burgers.linear_manifold import (
    compute_ECSW_training_matrix_2D_local,
    inviscid_burgers_implicit2D_LSPG_local_ecsw,
)
from burgers.cluster_utils import select_cluster_reduced
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


def load_local_pod_model(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Local POD model not found: {filename}")

    data = np.load(filename, allow_pickle=True)

    S_shape = tuple(data["S_shape"])
    u0_list = [np.asarray(u0, dtype=np.float64) for u0 in data["u0_list"]]
    uc_list = [np.asarray(uc, dtype=np.float64) for uc in data["uc_list"]]
    V_list = [np.asarray(V, dtype=np.float64) for V in data["V_list"]]
    cluster_indices = [np.asarray(idx, dtype=int) for idx in data["cluster_indices"]]

    d_const = np.asarray(data["d_const"], dtype=np.float64)
    g_raw = np.asarray(data["g_list"], dtype=object)
    K = g_raw.shape[0]
    g_list = [[None for _ in range(K)] for _ in range(K)]
    for k in range(K):
        for l in range(K):
            g_list[k][l] = np.asarray(g_raw[k, l], dtype=np.float64).reshape(-1)

    return S_shape, u0_list, uc_list, V_list, cluster_indices, d_const, g_list


def main(
    mu1=4.56,
    mu2=0.019,
    local_model_file=os.path.join("LocalPOD", "local_pod_data.npz"),
    compute_ecsw=False,
    compute_ecm=None,
    weights_file=None,
    dt=DT,
    num_steps=NUM_STEPS,
    snap_sample_factor=10,
    snap_time_offset=3,
    mu_samples=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
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

    results_dir = "Results"
    snap_folder = os.path.join(results_dir, "param_snaps")
    if weights_file is None:
        weights_file = os.path.join(results_dir, "local_hprom_ecsw_weights.npy")

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

    print("\n====================================================")
    print("         LOCAL HPROM (Affine Piecewise ECSW-LSPG)")
    print("====================================================")
    print(f"[LOCAL-HPROM] mu1={mu1:.3f}, mu2={mu2:.4f}")
    print(f"[LOCAL-HPROM] dt={dt}, num_steps={num_steps}")
    print(f"[LOCAL-HPROM] grid={num_cells_x}x{num_cells_y}")

    (
        S_shape,
        u0_list,
        uc_list,
        V_list,
        cluster_indices,
        d_const,
        g_list,
    ) = load_local_pod_model(local_model_file)

    K = len(V_list)
    mode_counts = [int(V.shape[1]) for V in V_list]
    print(f"[LOCAL-HPROM] Loaded local model: {local_model_file}")
    print(f"[LOCAL-HPROM] Number of clusters: {K}")
    print(
        "[LOCAL-HPROM] Retained modes per cluster: "
        f"min={np.min(mode_counts)}, max={np.max(mode_counts)}, avg={np.mean(mode_counts):.2f}"
    )

    # ------------------------------------------------------------------
    # ECSW weights: build or load
    # ------------------------------------------------------------------
    C_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None

    if compute_ecsw:
        print("[LOCAL-HPROM] Computing ECSW weights...")
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

            print(f"[LOCAL-HPROM] Building ECSW training block for mu={mu_train}")
            Ci = compute_ECSW_training_matrix_2D_local(
                snaps_now,
                snaps_prev,
                u0_list,
                V_list,
                d_const,
                g_list,
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
        print(f"[LOCAL-HPROM] Stacked ECSW training matrix C shape: {C_shape}")

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
        print(f"[LOCAL-HPROM] ECSW weights saved to: {weights_file}")
        print(f"[LOCAL-HPROM] ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"[LOCAL-HPROM] ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "local_hprom_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("Local HPROM Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[LOCAL-HPROM] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"ECSW weights file not found: {weights_file}. "
                "Run with compute_ecsw=True first."
            )
        weights = np.load(weights_file, allow_pickle=False)
        print(f"[LOCAL-HPROM] Loaded ECSW weights from: {weights_file}")

    if weights.size != num_cells:
        raise ValueError(
            f"ECSW weights size mismatch: got {weights.size}, expected {num_cells}."
        )

    n_ecsw_elements = int(np.count_nonzero(weights))
    print(f"[LOCAL-HPROM] Nonzero ECSW elements: {n_ecsw_elements} / {num_cells}")

    # ------------------------------------------------------------------
    # Local HPROM solve
    # ------------------------------------------------------------------
    t0 = time.time()
    rom_snaps, stats = inviscid_burgers_implicit2D_LSPG_local_ecsw(
        grid_x,
        grid_y,
        weights,
        w0,
        dt,
        num_steps,
        mu_rom,
        u0_list,
        V_list,
        uc_list,
        cluster_select_fun=select_cluster_reduced,
        d_const=d_const,
        g_list=g_list,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        max_its=max_its,
    )
    elapsed_hprom = time.time() - t0

    num_its = int(stats["num_its"])
    jac_time = float(stats["jac_time"])
    res_time = float(stats["res_time"])
    ls_time = float(stats["ls_time"])
    cluster_history = np.asarray(stats["cluster_history"], dtype=int)
    cluster_counts = np.bincount(cluster_history, minlength=K)
    num_switches = int(np.sum(cluster_history[1:] != cluster_history[:-1]))

    print(f"[LOCAL-HPROM] Total solve time: {elapsed_hprom:.3e} seconds")
    print(f"[LOCAL-HPROM] Gauss-Newton iterations: {num_its}")
    print(
        "[LOCAL-HPROM] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )
    print(f"[LOCAL-HPROM] Cluster switches: {num_switches}")

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
    print(f"[LOCAL-HPROM] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    # ------------------------------------------------------------------
    # Save HPROM snapshots
    # ------------------------------------------------------------------
    rom_path = os.path.join(
        results_dir,
        f"local_hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[LOCAL-HPROM] HPROM snapshots saved to: {rom_path}")

    # ------------------------------------------------------------------
    # Plot HDM vs Local HPROM
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
        label="Local HPROM",
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
        f"local_hprom_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[LOCAL-HPROM] Comparison plot saved to: {fig_path}")

    # ------------------------------------------------------------------
    # Relative error
    # ------------------------------------------------------------------
    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[LOCAL-HPROM] Relative error: {relative_error:.2f}%")

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(
        results_dir,
        f"local_hprom_summary_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.txt",
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
                    ("cluster_selector", "select_cluster_reduced"),
                    ("snap_sample_factor", snap_sample_factor),
                    ("snap_time_offset", snap_time_offset),
                    ("mu_samples", mu_samples),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
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
                    ("retained_modes_per_cluster", mode_counts),
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
                "local_hprom_timing",
                [
                    ("total_local_hprom_time_seconds", elapsed_hprom),
                    ("avg_local_hprom_time_per_step_seconds", elapsed_hprom / num_steps),
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
                "error_metrics",
                [
                    ("relative_l2_error", rel_err_l2),
                    ("relative_error_percent", relative_error),
                ],
            ),
            (
                "outputs",
                [
                    ("local_hprom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("ecsw_weights_npy", weights_file),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[LOCAL-HPROM] Text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=False)
