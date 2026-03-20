#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 2: LOCAL POD PER CLUSTER (LOCAL POD-RBF)

Inputs (from stage1):
  - local_rbf_clusters_full.npz

Outputs (inside LocalPOD-RBF):
  - local_rbf_pod_per_cluster.npz
  - stage2_pod_plots/cluster_<k>_sv_loss.png
  - stage2_cluster_ranks.png
  - stage2_pod_summary.txt

POD is computed on centered local snapshots:
    S_k_centered = S_k - u0_k,
where u0_k is the Stage 1 cluster centroid.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps
from burgers.config import GRID_X, GRID_Y


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


def build_local_snapshot_matrix_full(
    cluster_idx,
    mu_per_snapshot,
    t_per_snapshot,
    dt,
    num_steps,
    snap_folder,
    snap_cache,
):
    """
    Build local full-state snapshot matrix S_k for one cluster.
    """
    cluster_idx = np.asarray(cluster_idx, dtype=int)
    m_k = int(cluster_idx.size)
    if m_k == 0:
        raise ValueError("Cluster has zero snapshots.")

    n_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    w0 = np.ones(2 * n_cells, dtype=float)

    mu0 = tuple(mu_per_snapshot[cluster_idx[0], :])
    if mu0 not in snap_cache:
        snap_cache[mu0] = np.asarray(
            load_or_compute_snaps(
                list(mu0),
                GRID_X,
                GRID_Y,
                w0,
                dt,
                num_steps,
                snap_folder=snap_folder,
            ),
            dtype=float,
        )
    snaps0 = snap_cache[mu0]

    n_full, _ = snaps0.shape
    s_k = np.zeros((n_full, m_k), dtype=float)

    mus_k = mu_per_snapshot[cluster_idx, :]
    ts_k = t_per_snapshot[cluster_idx]

    unique_mus, inv_mu = np.unique(mus_k, axis=0, return_inverse=True)

    for i, mu_vec in enumerate(unique_mus):
        mu_tuple = tuple(mu_vec.tolist())

        if mu_tuple not in snap_cache:
            snap_cache[mu_tuple] = np.asarray(
                load_or_compute_snaps(
                    list(mu_tuple),
                    GRID_X,
                    GRID_Y,
                    w0,
                    dt,
                    num_steps,
                    snap_folder=snap_folder,
                ),
                dtype=float,
            )

        snaps_mu = snap_cache[mu_tuple]
        t_full = int(snaps_mu.shape[1])

        idx_local_i = np.where(inv_mu == i)[0]
        for j_local in idx_local_i:
            t_j = int(ts_k[j_local])
            if not (0 <= t_j < t_full):
                raise ValueError(
                    f"Time index t_j={t_j} out of range [0, {t_full - 1}] for mu={mu_tuple}."
                )
            s_k[:, j_local] = snaps_mu[:, t_j]

    return s_k


def n_for_tol_squared(sigmas, eps2):
    """
    Smallest n such that
      1 - sum_{i<=n}(sigma_i^2)/sum_j(sigma_j^2) <= eps2.
    """
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    if sigmas.size == 0:
        return 1

    s2 = sigmas ** 2
    total = float(np.sum(s2))
    if total <= 0.0:
        return 1

    loss2 = 1.0 - np.cumsum(s2) / total
    ok = np.where(loss2 <= float(eps2))[0]
    if ok.size == 0:
        return int(sigmas.size)
    return int(ok[0] + 1)


def plot_squared_energy_loss(sigmas, eps2, cluster_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    if sigmas.size == 0:
        return None

    s2 = sigmas ** 2
    total = float(np.sum(s2))
    if total <= 0.0:
        return None

    loss2 = 1.0 - np.cumsum(s2) / total
    n_needed = n_for_tol_squared(sigmas, eps2)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, loss2.size + 1), loss2, linewidth=2)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Mode index n")
    plt.ylabel("Relative POD loss")
    plt.axvline(
        x=n_needed,
        ymin=0.05,
        ymax=0.95,
        color="r",
        linestyle="--",
        label=f"eps2={eps2:.0e}, n={n_needed}",
    )
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"cluster_{cluster_id}_sv_loss.png")
    plt.savefig(out_png, dpi=220)
    plt.close()
    return out_png


def plot_cluster_ranks(r_list, out_png):
    ranks = np.asarray(r_list, dtype=int)
    xs = np.arange(ranks.size)

    plt.figure(figsize=(8, 4.5))
    plt.bar(xs, ranks, color="#4e79a7")
    plt.xlabel("Cluster index")
    plt.ylabel("Retained POD modes")
    plt.title("Local POD-RBF Stage2 retained rank per cluster")
    plt.xticks(xs)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main(
    clusters_file=os.path.join(script_dir, "local_rbf_clusters_full.npz"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    eps2_pod=1e-6,
    output_file=os.path.join(script_dir, "local_rbf_pod_per_cluster.npz"),
    plot_dir=os.path.join(script_dir, "stage2_pod_plots"),
    rank_plot_file=os.path.join(script_dir, "stage2_cluster_ranks.png"),
    summary_file=os.path.join(script_dir, "stage2_pod_summary.txt"),
):
    if float(eps2_pod) <= 0.0:
        raise ValueError("eps2_pod must be > 0.")

    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    print("\n====================================================")
    print("     STAGE 2: LOCAL POD-RBF POD PER CLUSTER")
    print("====================================================")
    print(f"[STAGE2] clusters_file={clusters_file}")
    print(f"[STAGE2] snap_folder={snap_folder}")
    print(f"[STAGE2] eps2_pod={eps2_pod:.1e}")

    if not os.path.exists(clusters_file):
        raise FileNotFoundError(f"Missing clusters file: {clusters_file}")

    t0_total = time.time()
    data = np.load(clusters_file, allow_pickle=True)

    k_count = int(data["K"])
    s_w_shape = tuple(data["S_w_shape"])
    labels = np.asarray(data["labels"], dtype=int)
    centers_w = np.asarray(data["centers_w"], dtype=float)
    cluster_indices_obj = data["cluster_indices"]
    param_list = np.asarray(data["param_list"], dtype=float)
    mu_per_snapshot = np.asarray(data["mu_per_snapshot"], dtype=float)
    t_per_snapshot = np.asarray(data["t_per_snapshot"], dtype=int)
    time_indices = np.asarray(data["time_indices"], dtype=int)
    time_subsample = int(data["time_subsample"])
    dt = float(data["dt"])
    num_steps = int(data["num_steps"])

    print(f"[STAGE2] Loaded K={k_count} clusters, S_w_shape={s_w_shape}")

    u_list = [None] * k_count
    u0_svd_list = [None] * k_count
    sigma_list = [None] * k_count
    r_list = [0] * k_count
    sv_plot_files = []

    total_snapshots_used = 0
    snap_cache = {}

    elapsed_build_total = 0.0
    elapsed_svd_total = 0.0

    for k in range(k_count):
        print("----------------------------------------------------")
        print(f"[STAGE2] Processing cluster {k}/{k_count - 1}")

        cluster_idx_k = np.asarray(cluster_indices_obj[k], dtype=int)
        m_k = int(cluster_idx_k.size)
        total_snapshots_used += m_k

        print(f"[STAGE2]   snapshots in cluster={m_k}")
        if m_k == 0:
            print("[STAGE2]   empty cluster. Skipping.")
            continue

        t0 = time.time()
        s_k_full = build_local_snapshot_matrix_full(
            cluster_idx=cluster_idx_k,
            mu_per_snapshot=mu_per_snapshot,
            t_per_snapshot=t_per_snapshot,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
            snap_cache=snap_cache,
        )
        elapsed_build = time.time() - t0
        elapsed_build_total += elapsed_build

        print(f"[STAGE2]   S_k_full shape={s_k_full.shape} (built in {elapsed_build:.2f}s)")

        u0_k = np.asarray(centers_w[:, k], dtype=float).reshape(-1)
        if u0_k.size != s_k_full.shape[0]:
            raise ValueError(
                f"Cluster centroid size mismatch for k={k}: "
                f"got {u0_k.size}, expected {s_k_full.shape[0]}."
            )
        s_k_centered = s_k_full - u0_k[:, None]

        t0 = time.time()
        u_k_full, s_k, _ = np.linalg.svd(s_k_centered, full_matrices=False)
        elapsed_svd = time.time() - t0
        elapsed_svd_total += elapsed_svd

        r_full = int(s_k.size)
        r_k = n_for_tol_squared(s_k, eps2_pod)
        r_k = max(1, min(r_k, r_full))

        print(f"[STAGE2]   full_rank={r_full}, retained={r_k}, svd_time={elapsed_svd:.2f}s")

        u_list[k] = np.asarray(u_k_full[:, :r_k], dtype=float)
        u0_svd_list[k] = np.asarray(u0_k, dtype=float)
        sigma_list[k] = np.asarray(s_k, dtype=float)
        r_list[k] = int(r_k)

        plot_file = plot_squared_energy_loss(s_k, eps2_pod, cluster_id=k, out_dir=plot_dir)
        if plot_file is not None:
            sv_plot_files.append(plot_file)

        del s_k_full, s_k_centered, u_k_full

    # Persist outputs
    u_obj = np.empty(k_count, dtype=object)
    u0_obj = np.empty(k_count, dtype=object)
    sigma_obj = np.empty(k_count, dtype=object)
    for k in range(k_count):
        u_obj[k] = u_list[k]
        if u0_svd_list[k] is None:
            u0_obj[k] = np.asarray(centers_w[:, k], dtype=float)
        else:
            u0_obj[k] = u0_svd_list[k]
        sigma_obj[k] = sigma_list[k]

    r_array = np.asarray(r_list, dtype=int)

    np.savez(
        output_file,
        K=k_count,
        U_list=u_obj,
        u0_svd_list=u0_obj,
        sigma_list=sigma_obj,
        r_list=r_array,
        cluster_indices=cluster_indices_obj,
        S_w_shape=np.asarray(s_w_shape, dtype=int),
        centers_w=centers_w,
        labels=labels,
        param_list=param_list,
        mu_per_snapshot=mu_per_snapshot,
        t_per_snapshot=t_per_snapshot,
        time_indices=time_indices,
        time_subsample=time_subsample,
        dt=dt,
        num_steps=num_steps,
        eps2_pod=float(eps2_pod),
        centered_svd=np.asarray(1, dtype=np.int64),
        centering_source=np.asarray("stage1_centers_w"),
    )
    print(f"[STAGE2] Saved POD NPZ: {output_file}")

    plot_cluster_ranks(r_array, rank_plot_file)
    print(f"[STAGE2] Saved rank plot: {rank_plot_file}")

    elapsed_total = time.time() - t0_total

    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage2_local_pod_per_cluster.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("clusters_file", clusters_file),
                    ("snap_folder", snap_folder),
                    ("eps2_pod", float(eps2_pod)),
                    ("center_local_snapshots_for_svd", True),
                    ("centering_source", "stage1_centers_w"),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("time_subsample", time_subsample),
                ],
            ),
            (
                "clusters",
                [
                    ("K", k_count),
                    ("S_w_shape", s_w_shape),
                    ("total_snapshots_used", total_snapshots_used),
                    ("retained_ranks", r_list),
                    ("min_rank", int(np.min(r_array) if r_array.size else 0)),
                    ("max_rank", int(np.max(r_array) if r_array.size else 0)),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_local_matrices", elapsed_build_total),
                    ("svd_total", elapsed_svd_total),
                    ("total", elapsed_total),
                ],
            ),
            (
                "outputs",
                [
                    ("pod_npz", output_file),
                    ("rank_plot_png", rank_plot_file),
                    ("sv_loss_plot_count", len(sv_plot_files)),
                    ("sv_loss_plot_dir", plot_dir),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE2] Saved summary: {summary_file}")
    print("[STAGE2] Done.\n")


if __name__ == "__main__":
    main()
