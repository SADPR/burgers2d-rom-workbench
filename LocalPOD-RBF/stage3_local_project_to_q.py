#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: BUILD LOCAL AFFINE ORIGINS AND CENTERED REDUCED COORDINATES

Inputs (from stage2):
  - local_rbf_pod_per_cluster.npz

Outputs (inside LocalPOD-RBF):
  - local_rbf_q_per_cluster.npz
  - stage3_cluster_snapshot_counts.png
  - stage3_projection_summary.txt
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


def build_u0_and_centered_q(
    u_k,
    cluster_idx_k,
    mu_per_snapshot,
    t_per_snapshot,
    dt,
    num_steps,
    snap_folder,
    snap_cache,
    u0_k_fixed=None,
):
    """
    For one cluster:
      Q_k = U_k^T (W_k - u0_k),
    where u0_k is either:
      - u0_k_fixed (if provided, preferred),
      - the local cluster mean otherwise.
    """
    cluster_idx_k = np.asarray(cluster_idx_k, dtype=int)
    m_k = int(cluster_idx_k.size)
    if m_k == 0:
        raise ValueError("Cluster has zero snapshots.")

    u_k = np.asarray(u_k, dtype=float)
    n_full, _ = u_k.shape

    n_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    w0 = np.ones(2 * n_cells, dtype=float)

    mus_k = mu_per_snapshot[cluster_idx_k, :]
    ts_k = t_per_snapshot[cluster_idx_k]
    unique_mus, inv_mu = np.unique(mus_k, axis=0, return_inverse=True)

    w_k = np.zeros((n_full, m_k), dtype=float)

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
            w_k[:, j_local] = snaps_mu[:, t_j]

    if u0_k_fixed is None:
        u0_k = np.mean(w_k, axis=1)
        u0_source = "cluster_mean"
    else:
        u0_k = np.asarray(u0_k_fixed, dtype=float).reshape(-1)
        if u0_k.size != n_full:
            raise ValueError(
                f"u0_k_fixed size mismatch: got {u0_k.size}, expected {n_full}."
            )
        u0_source = "stage2_u0_svd"
    q_k = u_k.T @ (w_k - u0_k[:, None])
    return u0_k, q_k, m_k, u0_source


def plot_cluster_snapshot_counts(cluster_sizes, out_png):
    sizes = np.asarray(cluster_sizes, dtype=int)
    xs = np.arange(sizes.size)

    plt.figure(figsize=(8, 4.5))
    plt.bar(xs, sizes, color="#59a14f")
    plt.xlabel("Cluster index")
    plt.ylabel("Snapshots used in stage3")
    plt.title("Local POD-RBF Stage3 cluster snapshot counts")
    plt.xticks(xs)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main(
    pod_file=os.path.join(script_dir, "local_rbf_pod_per_cluster.npz"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    output_file=os.path.join(script_dir, "local_rbf_q_per_cluster.npz"),
    counts_plot_file=os.path.join(script_dir, "stage3_cluster_snapshot_counts.png"),
    summary_file=os.path.join(script_dir, "stage3_projection_summary.txt"),
):
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    print("\n====================================================")
    print("     STAGE 3: LOCAL POD-RBF PROJECTION TO Q")
    print("====================================================")
    print(f"[STAGE3] pod_file={pod_file}")
    print(f"[STAGE3] snap_folder={snap_folder}")

    if not os.path.exists(pod_file):
        raise FileNotFoundError(f"Missing stage2 POD file: {pod_file}")

    t0_total = time.time()
    data = np.load(pod_file, allow_pickle=True)

    k_count = int(data["K"])
    u_list = data["U_list"]
    r_list = np.asarray(data["r_list"], dtype=int)
    cluster_indices = data["cluster_indices"]
    mu_per_snapshot = np.asarray(data["mu_per_snapshot"], dtype=float)
    t_per_snapshot = np.asarray(data["t_per_snapshot"], dtype=int)
    dt = float(data["dt"])
    num_steps = int(data["num_steps"])
    u0_svd_list = data["u0_svd_list"] if "u0_svd_list" in data.files else None

    q_list = [None] * k_count
    u0_list = [None] * k_count
    v_list = [None] * k_count
    cluster_sizes = [0] * k_count
    u0_source_list = [None] * k_count

    elapsed_build_total = 0.0
    snap_cache = {}

    for k in range(k_count):
        print("----------------------------------------------------")
        print(f"[STAGE3] Processing cluster {k}/{k_count - 1}")

        u_k = u_list[k]
        if u_k is None:
            print(f"[STAGE3]   U_k is None for cluster {k}. Skipping.")
            continue

        cluster_idx_k = np.asarray(cluster_indices[k], dtype=int)
        m_k = int(cluster_idx_k.size)
        cluster_sizes[k] = m_k
        r_k = int(np.asarray(u_k).shape[1])
        print(f"[STAGE3]   r_k={r_k}, snapshots={m_k}")

        if m_k == 0:
            print("[STAGE3]   empty cluster. Skipping.")
            continue

        u0_k_fixed = None
        if u0_svd_list is not None:
            candidate = u0_svd_list[k]
            if candidate is not None:
                u0_k_fixed = np.asarray(candidate, dtype=float).reshape(-1)

        t0 = time.time()
        u0_k, q_k, _, u0_source = build_u0_and_centered_q(
            u_k=u_k,
            cluster_idx_k=cluster_idx_k,
            mu_per_snapshot=mu_per_snapshot,
            t_per_snapshot=t_per_snapshot,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
            snap_cache=snap_cache,
            u0_k_fixed=u0_k_fixed,
        )
        elapsed = time.time() - t0
        elapsed_build_total += elapsed

        print(
            f"[STAGE3]   Built u0_k shape={u0_k.shape}, "
            f"Q_k shape={q_k.shape}, source={u0_source} in {elapsed:.2f}s"
        )

        u0_list[k] = np.asarray(u0_k, dtype=float)
        q_list[k] = np.asarray(q_k, dtype=float)
        v_list[k] = np.asarray(u_k, dtype=float)
        u0_source_list[k] = u0_source

    # Convert to object arrays safely
    q_obj = np.empty(k_count, dtype=object)
    u0_obj = np.empty(k_count, dtype=object)
    v_obj = np.empty(k_count, dtype=object)

    for k in range(k_count):
        q_obj[k] = q_list[k]
        u0_obj[k] = u0_list[k]
        v_obj[k] = v_list[k]

    np.savez(
        output_file,
        K=k_count,
        u0_list=u0_obj,
        V_list=v_obj,
        Q_list=q_obj,
        cluster_indices=cluster_indices,
        mu_per_snapshot=mu_per_snapshot,
        t_per_snapshot=t_per_snapshot,
        dt=dt,
        num_steps=num_steps,
        r_list=r_list,
        u0_source_per_cluster=np.asarray(u0_source_list, dtype=object),
    )
    print(f"[STAGE3] Saved Q NPZ: {output_file}")

    plot_cluster_snapshot_counts(cluster_sizes, counts_plot_file)
    print(f"[STAGE3] Saved cluster-count plot: {counts_plot_file}")

    elapsed_total = time.time() - t0_total

    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage3_local_project_to_q.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("pod_file", pod_file),
                    ("snap_folder", snap_folder),
                    (
                        "u0_source_policy",
                        "stage2_u0_svd_if_available_else_cluster_mean",
                    ),
                    ("dt", dt),
                    ("num_steps", num_steps),
                ],
            ),
            (
                "clusters",
                [
                    ("K", k_count),
                    ("cluster_snapshot_counts", cluster_sizes),
                    ("r_list", r_list.tolist()),
                    ("u0_source_per_cluster", u0_source_list),
                    ("nonempty_clusters", int(np.sum(np.asarray(cluster_sizes) > 0))),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_u0_and_q_total", elapsed_build_total),
                    ("total", elapsed_total),
                ],
            ),
            (
                "outputs",
                [
                    ("q_npz", output_file),
                    ("cluster_count_png", counts_plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE3] Saved summary: {summary_file}")
    print("[STAGE3] Done.\n")


if __name__ == "__main__":
    main()
