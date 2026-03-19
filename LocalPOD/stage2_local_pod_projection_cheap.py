#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage2_local_pod_projection_cheap.py

Load the offline local POD model (local_pod_data.npz) and test:

  • load HDM trajectory for a given (mu1, mu2)
  • at each time step:
        - use Grimberg-style reduced-space cluster selection,
        - switch reduced coordinates if the active cluster changes,
        - reconstruct the state from the selected local affine subspace
  • compute reconstruction error
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Path handling
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
localpod_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.config import GRID_X, GRID_Y, DT, NUM_STEPS


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


plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"],
})
plt.rc('font', size=13)


# ----------------------------------------------------------------------
# NPZ LOADER FOR NEW FILE FORMAT
# ----------------------------------------------------------------------
def load_local_pod_data(filename=None):
    """
    Loads the unified NPZ file produced by stage1_local_pod_offline_cheap.py
    """
    if filename is None:
        filename = os.path.join(localpod_dir, "local_pod_data.npz")
    data = np.load(filename, allow_pickle=True)

    S_shape         = tuple(data["S_shape"])
    u0_raw          = data["u0_list"]
    uc_raw          = data["uc_list"]
    V_raw           = data["V_list"]
    cluster_indices = list(data["cluster_indices"])
    d_const         = np.asarray(data["d_const"], dtype=float)
    g_list          = data["g_list"]   # (K,K) object array
    T_list          = data["T_list"]   # (K,K) object array
    h_list          = data["h_list"]   # (K,K) object array

    # convert u0_list, uc_list, V_list to clean Python lists of float arrays
    u0_list = [np.asarray(u, dtype=float) for u in u0_raw]
    uc_list = [np.asarray(u, dtype=float) for u in uc_raw]
    V_list  = [np.asarray(V, dtype=float) for V in V_raw]

    return (S_shape, u0_list, uc_list, V_list,
            cluster_indices, d_const, g_list, T_list, h_list)


# ----------------------------------------------------------------------
# INITIAL CLUSTER SELECTION (FULL-STATE, ONE-TIME)
# ----------------------------------------------------------------------
def select_initial_cluster(u, uc_list):
    """
    One-shot selection for the first snapshot:
        k0 = argmin_k ||u - u_c,k||^2  in full space.
    (Cost O(KN), but done only once.)
    """
    d2 = [np.linalg.norm(u - uc)**2 for uc in uc_list]
    return int(np.argmin(d2))


# ----------------------------------------------------------------------
# REDUCED-SPACE CLUSTER SELECTION (GRIMBERG)
# ----------------------------------------------------------------------
def select_cluster_reduced(k_current, y_k, d_const, g_list):
    """
    Grimberg-style selection:

        For fixed current cluster k and reduced coords y_k,
        pick l minimizing

            2 g_{k,l}^T y_k + d_const[k,l].

    Parameters
    ----------
    k_current : int
        Index of current cluster k.
    y_k : (n_k,) array
        Reduced coordinates in current cluster basis V_k.
    d_const : (K,K) float array
        d_const[k,l] = ||u0_k - u_c,l||^2.
    g_list : (K,K) object array
        g_list[k,l] = V_k^T (u0_k - u_c,l).

    Returns
    -------
    l_best : int
        Index of selected cluster.
    """
    K = d_const.shape[0]
    scores = np.empty(K, dtype=float)
    for l in range(K):
        g_kl = np.asarray(g_list[k_current, l], dtype=float)
        scores[l] = 2.0 * (g_kl @ y_k) + d_const[k_current, l]
    return int(np.argmin(scores))


# ----------------------------------------------------------------------
# SWITCH COORDINATES BETWEEN CLUSTERS
# ----------------------------------------------------------------------
def switch_coordinates(y_old, k_old, k_new, T_list, h_list):
    """
    y_new = T_{k_new,k_old} y_old + h_{k_new,k_old}
    """
    T = np.asarray(T_list[k_new, k_old], dtype=float)
    h = np.asarray(h_list[k_new, k_old], dtype=float)
    return T @ y_old + h


# ----------------------------------------------------------------------
# LOCAL AFFINE PROJECTION (FOR A GIVEN CLUSTER)
# ----------------------------------------------------------------------
def local_pod_project(u, k, u0_list, V_list):
    """
    Project full state u onto local affine subspace of cluster k:

        u ≈ u0_k + V_k y_k,   with  y_k = V_k^T (u - u0_k).
    """
    u0_k = u0_list[k]
    Vk   = V_list[k]

    y_k   = Vk.T @ (u - u0_k)
    u_rec = u0_k + Vk @ y_k

    return u_rec, y_k


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():

    # ---------------- User settings ----------------
    mu1, mu2 = 4.56, 0.019
    dt, num_steps = DT, NUM_STEPS
    snap_folder = os.path.join(parent_dir, "Results", "param_snaps")
    local_model_file = os.path.join(localpod_dir, "local_pod_data.npz")

    # ---------------- Load the offline model ----------------
    print(f"[ONLINE] Loading local POD model: {local_model_file}")
    (S_shape, u0_list, uc_list, V_list,
     cluster_indices, d_const, g_list, T_list, h_list) = load_local_pod_data(local_model_file)

    K = len(V_list)
    print(f"[ONLINE] Loaded {K} clusters.")

    # ---------------- Load HDM trajectory ----------------
    print(f"[ONLINE] Loading HDM for μ1={mu1:.3f}, μ2={mu2:.3f}")
    N_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    w0 = np.ones(2 * N_cells, dtype=float)

    hdm_snaps = load_or_compute_snaps([mu1, mu2], GRID_X, GRID_Y,
                                      w0, dt, num_steps,
                                      snap_folder=snap_folder)
    hdm_snaps = np.asarray(hdm_snaps, dtype=float)

    N, T = hdm_snaps.shape
    print(f"[ONLINE] HDM snapshots shape = {hdm_snaps.shape}")

    # storage
    local_rec       = np.zeros_like(hdm_snaps)
    y_history       = []
    cluster_history = []

    total_err2  = 0.0
    total_norm2 = 0.0

    t0_global = time.time()

    # ---------------- Initial snapshot: full-space cluster choice ----------------
    u0_snap = hdm_snaps[:, 0]
    k = select_initial_cluster(u0_snap, uc_list)
    print(f"[ONLINE] Initial cluster k0 = {k} / {K-1}")

    # project onto initial cluster
    u_rec_0, y_k = local_pod_project(u0_snap, k, u0_list, V_list)

    local_rec[:, 0] = u_rec_0
    y_history.append(y_k)
    cluster_history.append(k)

    diff0 = u0_snap - u_rec_0
    total_err2  += diff0 @ diff0
    total_norm2 += u0_snap @ u0_snap

    # ---------------- Loop over remaining time steps ----------------
    for t in range(1, T):
        u_t = hdm_snaps[:, t]

        # 1) project current HDM snapshot onto *current* cluster k
        u_rec_k, y_k = local_pod_project(u_t, k, u0_list, V_list)

        # 2) reduced-space cluster selection around current cluster
        l_new = select_cluster_reduced(k, y_k, d_const, g_list)

        if l_new != k:
            # 3) switch coordinates
            y_l = switch_coordinates(y_k, k, l_new, T_list, h_list)

            # reconstruct with new cluster
            u0_l = u0_list[l_new]
            V_l  = V_list[l_new]
            u_rec_t = u0_l + V_l @ y_l

            k = l_new
            y_k = y_l
        else:
            # stay in same cluster
            u_rec_t = u_rec_k

        # 4) store
        local_rec[:, t] = u_rec_t
        y_history.append(y_k)
        cluster_history.append(k)

        # 5) error
        diff = u_t - u_rec_t
        total_err2  += diff @ diff
        total_norm2 += u_t @ u_t

        if (t % 50) == 0 or t == T - 1:
            print(f"[STEP {t+1}/{T}] active cluster = {k}")

    # ---------------- Final error ----------------
    rel_err = np.sqrt(total_err2 / total_norm2)
    total_time = time.time() - t0_global
    avg_time_per_step = total_time / T

    print(f"\n[ONLINE] Local POD relative error = {rel_err:.4e}")
    print(f"[ONLINE] Total reconstruction time = {total_time:.2f} s")
    print(f"[ONLINE] Avg per step = {avg_time_per_step:.4e} s")

    # ---------------- Plot snapshots ----------------
    inds = range(0, T, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    plot_snaps(GRID_X, GRID_Y, hdm_snaps, inds,
               label="HDM", fig_ax=(fig, ax1, ax2),
               color="black", linewidth=3)

    plot_snaps(GRID_X, GRID_Y, local_rec, inds,
               label="Local POD", fig_ax=(fig, ax1, ax2),
               color="#9400D3", linewidth=3)

    ax2.legend(loc="center right", fontsize=16, frameon=True)
    fig.suptitle(
        fr"Local POD Projection   ($\mu_1={mu1:.2f}$, $\mu_2={mu2:.3f}$) "
        fr"  error={100*rel_err:.2f}%",
        fontsize=15
    )

    out_png = os.path.join(
        localpod_dir,
        f"local_pod_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[ONLINE] Saved plot: {out_png}")

    cluster_history_arr = np.asarray(cluster_history, dtype=int)
    cluster_counts = np.bincount(cluster_history_arr, minlength=K)
    num_switches = int(np.sum(cluster_history_arr[1:] != cluster_history_arr[:-1]))
    mode_counts = [int(V.shape[1]) for V in V_list]

    report_path = os.path.join(
        localpod_dir,
        f"stage2_local_pod_projection_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage2_local_pod_projection_cheap.py"),
                    ("mu1", mu1),
                    ("mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("local_model_file", local_model_file),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "offline_model_info",
                [
                    ("num_clusters", K),
                    ("S_shape_offline", S_shape),
                    ("cluster_sizes_after_overlap", [int(len(i)) for i in cluster_indices]),
                    ("retained_modes_per_cluster", mode_counts),
                ],
            ),
            (
                "reconstruction_metrics",
                [
                    ("hdm_snapshot_shape", hdm_snaps.shape),
                    ("relative_reconstruction_error", rel_err),
                    ("error_percent", 100.0 * rel_err),
                    ("initial_cluster", int(cluster_history_arr[0])),
                    ("final_cluster", int(cluster_history_arr[-1])),
                    ("cluster_switches", num_switches),
                    ("cluster_visit_counts", cluster_counts.tolist()),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("total_reconstruction_time", total_time),
                    ("avg_time_per_step", avg_time_per_step),
                ],
            ),
            (
                "outputs",
                [
                    ("projection_plot_png", out_png),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[ONLINE] Saved summary: {report_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
