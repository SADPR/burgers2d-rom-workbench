#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 2: LOCAL QUADRATIC MANIFOLD PROJECTION CHECK

Loads LocalQuadratic/local_qm_data.npz and evaluates projection quality
against HDM snapshots at a target parameter.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
localquadratic_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.config import GRID_X, GRID_Y, W0


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


plt.rcParams.update(
    {
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": ["STIXGeneral"],
    }
)
plt.rc("font", size=13)


def load_local_qm_data(filename=None):
    if filename is None:
        filename = os.path.join(localquadratic_dir, "local_qm_data.npz")

    data = np.load(filename, allow_pickle=True)

    S_shape = tuple(data["S_shape"])
    u0_list = [np.asarray(u, dtype=np.float64) for u in data["u0_list"]]
    uc_list = [np.asarray(u, dtype=np.float64) for u in data["uc_list"]]
    uref_list = [np.asarray(u, dtype=np.float64) for u in data["uref_list"]]
    V_list = [np.asarray(V, dtype=np.float64) for V in data["V_list"]]
    H_list = [np.asarray(H, dtype=np.float64) for H in data["H_list"]]
    cluster_indices = [np.asarray(idx, dtype=int) for idx in data["cluster_indices"]]
    d_const = np.asarray(data["d_const"], dtype=np.float64)
    g_list = data["g_list"]
    T_list = data["T_list"]
    h_list = data["h_list"]
    n_list = (
        np.asarray(data["n_list"], dtype=int).tolist()
        if "n_list" in data.files
        else [int(V.shape[1]) for V in V_list]
    )

    return (
        S_shape,
        u0_list,
        uc_list,
        uref_list,
        V_list,
        H_list,
        n_list,
        cluster_indices,
        d_const,
        g_list,
        T_list,
        h_list,
    )


def build_Q_quadratic_symmetric_vec(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    i_triu, j_triu = np.triu_indices(q.size)
    return q[i_triu] * q[j_triu]


def select_initial_cluster(u, uc_list):
    d2 = [np.linalg.norm(u - uc) ** 2 for uc in uc_list]
    return int(np.argmin(d2))


def select_cluster_reduced(k_current, q_k, d_const, g_list):
    K = d_const.shape[0]
    scores = np.empty(K, dtype=np.float64)
    for l in range(K):
        g_kl = np.asarray(g_list[k_current, l], dtype=np.float64)
        scores[l] = 2.0 * (g_kl @ q_k) + d_const[k_current, l]
    return int(np.argmin(scores))


def switch_coordinates(q_old, k_old, k_new, T_list, h_list):
    T = np.asarray(T_list[k_new, k_old], dtype=np.float64)
    h = np.asarray(h_list[k_new, k_old], dtype=np.float64)
    return T @ q_old + h


def local_qm_project(u, k, u0_list, uref_list, V_list, H_list):
    u0_k = u0_list[k]
    u_ref_k = uref_list[k]
    V_k = V_list[k]
    H_k = H_list[k]

    # Keep linear coordinates aligned with the selector precomputations.
    q_k = V_k.T @ (u - u0_k)
    Q_k = build_Q_quadratic_symmetric_vec(q_k)
    u_rec = u_ref_k + V_k @ q_k + H_k @ Q_k
    return u_rec, q_k


def main():
    # ---------------- user settings ----------------
    mu1, mu2 = 4.56, 0.019
    dt, num_steps = 0.05, 500
    snap_folder = os.path.join(parent_dir, "Results", "param_snaps")
    local_model_file = os.path.join(localquadratic_dir, "local_qm_data.npz")

    print(f"[ONLINE-QM] Loading local QM model: {local_model_file}")
    (
        S_shape,
        u0_list,
        uc_list,
        uref_list,
        V_list,
        H_list,
        n_list,
        cluster_indices,
        d_const,
        g_list,
        T_list,
        h_list,
    ) = load_local_qm_data(local_model_file)

    K = len(V_list)
    print(f"[ONLINE-QM] Loaded {K} clusters.")

    w0 = np.asarray(W0, dtype=np.float64).copy()

    print(f"[ONLINE-QM] Loading HDM for mu1={mu1:.3f}, mu2={mu2:.3f}")
    hdm_snaps = load_or_compute_snaps(
        [mu1, mu2],
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder=snap_folder,
    )
    hdm_snaps = np.asarray(hdm_snaps, dtype=np.float64)
    _, T = hdm_snaps.shape
    print(f"[ONLINE-QM] HDM snapshots shape = {hdm_snaps.shape}")

    local_rec = np.zeros_like(hdm_snaps)
    cluster_history = []
    q_history = []

    total_err2 = 0.0
    total_norm2 = 0.0
    t0_global = time.time()

    u0_snap = hdm_snaps[:, 0]
    k = select_initial_cluster(u0_snap, uc_list)
    print(f"[ONLINE-QM] Initial cluster k0 = {k} / {K - 1}")

    u_rec_0, q_k = local_qm_project(u0_snap, k, u0_list, uref_list, V_list, H_list)
    local_rec[:, 0] = u_rec_0
    cluster_history.append(k)
    q_history.append(q_k)

    diff0 = u0_snap - u_rec_0
    total_err2 += diff0 @ diff0
    total_norm2 += u0_snap @ u0_snap

    for t in range(1, T):
        u_t = hdm_snaps[:, t]
        u_rec_k, q_k = local_qm_project(u_t, k, u0_list, uref_list, V_list, H_list)
        l_new = select_cluster_reduced(k, q_k, d_const, g_list)

        if l_new != k:
            q_l = switch_coordinates(q_k, k, l_new, T_list, h_list)
            u_ref_l = uref_list[l_new]
            V_l = V_list[l_new]
            H_l = H_list[l_new]
            Q_l = build_Q_quadratic_symmetric_vec(q_l)
            u_rec_t = u_ref_l + V_l @ q_l + H_l @ Q_l
            k = l_new
            q_k = q_l
        else:
            u_rec_t = u_rec_k

        local_rec[:, t] = u_rec_t
        cluster_history.append(k)
        q_history.append(q_k)

        diff = u_t - u_rec_t
        total_err2 += diff @ diff
        total_norm2 += u_t @ u_t

        if (t % 50) == 0 or t == T - 1:
            print(f"[STEP {t+1}/{T}] active cluster = {k}")

    rel_err = np.sqrt(total_err2 / total_norm2)
    total_time = time.time() - t0_global
    avg_time_per_step = total_time / T

    print(f"[ONLINE-QM] Local QM relative error = {rel_err:.4e}")
    print(f"[ONLINE-QM] Total reconstruction time = {total_time:.2f} s")
    print(f"[ONLINE-QM] Avg per step = {avg_time_per_step:.4e} s")

    out_npy = os.path.join(
        localquadratic_dir,
        f"local_qm_projection_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(out_npy, local_rec)

    inds = range(0, T, 100)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snaps,
        inds,
        label="HDM",
        fig_ax=(fig, ax1, ax2),
        color="black",
        linewidth=3,
    )
    plot_snaps(
        GRID_X,
        GRID_Y,
        local_rec,
        inds,
        label="Local QPROM",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=3,
    )

    ax2.legend(loc="center right", fontsize=16, frameon=True)
    fig.suptitle(
        rf"Local Quadratic Manifold Projection "
        rf"($\mu_1={mu1:.2f}$, $\mu_2={mu2:.3f}$) "
        rf"error={100 * rel_err:.2f}%",
        fontsize=15,
    )

    out_png = os.path.join(
        localquadratic_dir,
        f"local_qm_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    cluster_history_arr = np.asarray(cluster_history, dtype=int)
    cluster_counts = np.bincount(cluster_history_arr, minlength=K)
    num_switches = int(np.sum(cluster_history_arr[1:] != cluster_history_arr[:-1]))

    report_path = os.path.join(
        localquadratic_dir,
        f"stage2_local_qm_projection_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage2_local_qm_projection.py"),
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
                    ("S_shape_offline", S_shape),
                    ("num_clusters", K),
                    ("cluster_sizes_after_overlap", [int(len(i)) for i in cluster_indices]),
                    ("n_qm_per_cluster", n_list),
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
                    ("projection_snapshots_npy", out_npy),
                    ("projection_plot_png", out_png),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )

    print(f"[ONLINE-QM] Saved snapshots: {out_npy}")
    print(f"[ONLINE-QM] Saved plot: {out_png}")
    print(f"[ONLINE-QM] Saved summary: {report_path}")
    return total_time, rel_err


if __name__ == "__main__":
    main()
