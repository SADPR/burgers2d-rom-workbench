#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 1: CLUSTER FULL SNAPSHOTS [u; v] FOR LOCAL POD-RBF

Build the global full-state snapshot matrix and cluster its columns.
Optionally add overlap between clusters.

Outputs (inside LocalPOD-RBF):
  - local_rbf_clusters_full.npz
  - stage1_cluster_sizes.png
  - stage1_cluster_summary.txt
"""

import os
import sys
import time
from datetime import datetime
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, get_snapshot_params
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


def build_global_snapshot_matrix_full(dt, num_steps, snap_folder, time_subsample=1):
    """
    Build full-state snapshot matrix S_w = [w(mu_1), w(mu_2), ...],
    where each w stacks u and v.

    Returns
    -------
    S_w : ndarray, shape (N_full, M)
    param_list : ndarray, shape (n_params, 2)
    time_indices : ndarray, shape (T_used,)
    mu_per_snapshot : ndarray, shape (M, 2)
    t_per_snapshot : ndarray, shape (M,)
    """
    if int(time_subsample) < 1:
        raise ValueError("time_subsample must be >= 1.")

    param_list = np.asarray(get_snapshot_params(), dtype=float)
    n_params = int(param_list.shape[0])

    print(f"[STAGE1] Using {n_params} parameter points for clustering.")
    print(f"[STAGE1] Time subsample factor = {time_subsample}")

    # Robust w0 size from the configured grid
    n_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    w0 = np.ones(2 * n_cells, dtype=float)

    snaps0 = np.asarray(
        load_or_compute_snaps(
            param_list[0],
            GRID_X,
            GRID_Y,
            w0,
            dt,
            num_steps,
            snap_folder=snap_folder,
        ),
        dtype=float,
    )
    n_full, t_full = snaps0.shape

    if n_full % 2 != 0:
        raise ValueError(f"Expected even full-state size, got {n_full}.")

    time_indices = np.arange(0, t_full, int(time_subsample), dtype=int)
    t_used = int(time_indices.size)
    m_total = int(n_params * t_used)

    print(f"[STAGE1] Full steps per parameter: {t_full}")
    print(f"[STAGE1] Used steps per parameter: {t_used}")
    print(f"[STAGE1] Total snapshots M      : {m_total}")

    s_w = np.zeros((n_full, m_total), dtype=float)
    mu_per_snapshot = np.zeros((m_total, 2), dtype=float)
    t_per_snapshot = np.zeros(m_total, dtype=int)

    col = 0
    t0 = time.time()
    for i, mu in enumerate(param_list):
        print(f"[STAGE1] Loading snapshots for mu={mu.tolist()} ({i + 1}/{n_params})")
        snaps_mu = np.asarray(
            load_or_compute_snaps(
                mu,
                GRID_X,
                GRID_Y,
                w0,
                dt,
                num_steps,
                snap_folder=snap_folder,
            ),
            dtype=float,
        )

        w_mu = snaps_mu[:, time_indices]
        ncols = int(w_mu.shape[1])

        s_w[:, col:col + ncols] = w_mu
        mu_per_snapshot[col:col + ncols, :] = mu
        t_per_snapshot[col:col + ncols] = time_indices
        col += ncols

    if col != m_total:
        raise RuntimeError("Internal mismatch when assembling S_w.")

    print(f"[STAGE1] Global S_w shape = {s_w.shape}")
    print(f"[STAGE1] Assembly time    = {time.time() - t0:.2f} s")

    return s_w, param_list, time_indices, mu_per_snapshot, t_per_snapshot


def cluster_snapshots_kmeans(s_w, n_clusters):
    s_w = np.asarray(s_w, dtype=float)
    _, m = s_w.shape
    print(f"[STAGE1] Running K-means on {m} snapshots...")

    t0 = time.time()
    kmeans = KMeans(
        n_clusters=int(n_clusters),
        random_state=1,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    kmeans.fit(s_w.T)
    print(f"[STAGE1] K-means finished in {time.time() - t0:.2f} s")

    labels = kmeans.labels_.astype(int)
    centers = kmeans.cluster_centers_.T.astype(float)
    return labels, centers


def cluster_snapshots_fuzzy(s_w, n_clusters, m=2.0, error=1e-5, maxiter=1000):
    if fuzz is None:
        raise ImportError(
            "scikit-fuzzy is required for fuzzy clustering. Install: pip install scikit-fuzzy"
        )

    s_w = np.asarray(s_w, dtype=float)
    _, m_snap = s_w.shape
    print(f"[STAGE1] Running fuzzy c-means on {m_snap} snapshots...")

    t0 = time.time()
    cntr, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=s_w,
        c=int(n_clusters),
        m=float(m),
        error=float(error),
        maxiter=int(maxiter),
        init=None,
        seed=1,
    )
    print(f"[STAGE1] Fuzzy c-means finished in {time.time() - t0:.2f} s")

    labels = np.argmax(memberships, axis=0).astype(int)
    centers = cntr.T.astype(float)
    return labels, centers


def build_overlapping_clusters(s_w, labels, centers_w, phi):
    """
    Build overlapping clusters (Algorithm-1 style).
    """
    s_w = np.asarray(s_w, dtype=float)
    _, m = s_w.shape
    k_count = int(centers_w.shape[1])

    sk = [np.where(labels == k)[0] for k in range(k_count)]
    sk_plus = [set(indices.tolist()) for indices in sk]
    neighbors = [set() for _ in range(k_count)]

    # Neighbor map from two nearest centers per snapshot
    for s in range(m):
        ws = s_w[:, s]
        dists = np.linalg.norm(centers_w - ws[:, None], axis=0)
        idx_sorted = np.argsort(dists)
        k = int(idx_sorted[0])
        l = int(idx_sorted[1])
        neighbors[k].add(l)
        neighbors[l].add(k)

    # Add overlap from neighbors
    for k in range(k_count):
        for l in neighbors[k]:
            s_l = sk[l]
            if s_l.size == 0:
                continue

            n_add = int(np.floor(float(phi) * s_l.size))
            if n_add <= 0:
                continue

            s_l_snaps = s_w[:, s_l]
            dists_kl = np.linalg.norm(s_l_snaps - centers_w[:, k][:, None], axis=0)
            chosen = s_l[np.argsort(dists_kl)[:n_add]]
            sk_plus[k].update(chosen.tolist())

    return [np.array(sorted(list(sk_plus[k])), dtype=int) for k in range(k_count)]


def plot_cluster_sizes(primary_sizes, overlap_sizes, out_png):
    k_count = len(primary_sizes)
    xs = np.arange(k_count)

    plt.figure(figsize=(9, 5))
    plt.bar(xs - 0.18, primary_sizes, width=0.36, label="primary", color="#4e79a7")
    plt.bar(xs + 0.18, overlap_sizes, width=0.36, label="with overlap", color="#f28e2b")
    plt.xlabel("Cluster index")
    plt.ylabel("Snapshots")
    plt.title("Local POD-RBF Stage1 cluster sizes")
    plt.xticks(xs)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def main(
    n_clusters=10,
    dt=DT,
    num_steps=NUM_STEPS,
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    clustering_method="kmeans",  # "kmeans" or "fuzzy"
    phi=0.1,
    time_subsample=1,
    output_file=os.path.join(script_dir, "local_rbf_clusters_full.npz"),
    summary_file=os.path.join(script_dir, "stage1_cluster_summary.txt"),
    cluster_plot_file=os.path.join(script_dir, "stage1_cluster_sizes.png"),
):
    if not (0.0 <= float(phi) <= 1.0):
        raise ValueError("phi must be in [0, 1].")
    if int(n_clusters) < 2:
        raise ValueError("n_clusters must be >= 2.")

    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    print("\n====================================================")
    print("     STAGE 1: LOCAL POD-RBF CLUSTERING IN [u;v]")
    print("====================================================")
    print(f"[STAGE1] n_clusters={n_clusters}")
    print(f"[STAGE1] dt={dt}, num_steps={num_steps}")
    print(f"[STAGE1] clustering_method={clustering_method}")
    print(f"[STAGE1] phi={phi}")
    print(f"[STAGE1] time_subsample={time_subsample}")
    print(f"[STAGE1] snap_folder={snap_folder}")

    t0_total = time.time()

    t0 = time.time()
    s_w, param_list, time_indices, mu_per_snapshot, t_per_snapshot = build_global_snapshot_matrix_full(
        dt=dt,
        num_steps=num_steps,
        snap_folder=snap_folder,
        time_subsample=time_subsample,
    )
    elapsed_build = time.time() - t0

    t0 = time.time()
    method = str(clustering_method).strip().lower()
    if method == "kmeans":
        labels, centers_w = cluster_snapshots_kmeans(s_w, n_clusters)
    elif method == "fuzzy":
        labels, centers_w = cluster_snapshots_fuzzy(s_w, n_clusters)
    else:
        raise ValueError("clustering_method must be 'kmeans' or 'fuzzy'.")
    elapsed_cluster = time.time() - t0

    primary_counts = Counter(labels.tolist())
    primary_sizes = [int(primary_counts.get(k, 0)) for k in range(int(n_clusters))]

    if float(phi) > 0.0:
        print(f"[STAGE1] Building overlapping clusters with phi={phi}...")
        cluster_indices = build_overlapping_clusters(s_w, labels, centers_w, phi)
    else:
        cluster_indices = [np.where(labels == k)[0] for k in range(int(n_clusters))]

    overlap_sizes = [int(np.asarray(ids).size) for ids in cluster_indices]

    s_w_shape = np.array(s_w.shape, dtype=int)
    cluster_indices_obj = np.empty(int(n_clusters), dtype=object)
    for k in range(int(n_clusters)):
        cluster_indices_obj[k] = np.asarray(cluster_indices[k], dtype=int)

    np.savez(
        output_file,
        K=int(n_clusters),
        S_w_shape=s_w_shape,
        labels=np.asarray(labels, dtype=int),
        centers_w=np.asarray(centers_w, dtype=float),
        cluster_indices=cluster_indices_obj,
        param_list=np.asarray(param_list, dtype=float),
        mu_per_snapshot=np.asarray(mu_per_snapshot, dtype=float),
        t_per_snapshot=np.asarray(t_per_snapshot, dtype=int),
        time_indices=np.asarray(time_indices, dtype=int),
        time_subsample=int(time_subsample),
        dt=float(dt),
        num_steps=int(num_steps),
    )
    print(f"[STAGE1] Saved clusters NPZ: {output_file}")

    plot_cluster_sizes(primary_sizes, overlap_sizes, cluster_plot_file)
    print(f"[STAGE1] Saved cluster-size plot: {cluster_plot_file}")

    elapsed_total = time.time() - t0_total

    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage1_cluster_snapshots_u.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("n_clusters", int(n_clusters)),
                    ("clustering_method", method),
                    ("phi", float(phi)),
                    ("time_subsample", int(time_subsample)),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "snapshot_matrix",
                [
                    ("S_w_shape", tuple(s_w.shape)),
                    ("num_parameters", int(param_list.shape[0])),
                    ("num_snapshots", int(s_w.shape[1])),
                    ("time_indices_count", int(time_indices.size)),
                ],
            ),
            (
                "clusters",
                [
                    ("primary_sizes", primary_sizes),
                    ("overlap_sizes", overlap_sizes),
                    ("min_overlap_size", int(min(overlap_sizes) if overlap_sizes else 0)),
                    ("max_overlap_size", int(max(overlap_sizes) if overlap_sizes else 0)),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_snapshot_matrix", elapsed_build),
                    ("clustering", elapsed_cluster),
                    ("total", elapsed_total),
                ],
            ),
            (
                "outputs",
                [
                    ("clusters_npz", output_file),
                    ("cluster_sizes_png", cluster_plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE1] Saved summary: {summary_file}")
    print("[STAGE1] Done.\n")


if __name__ == "__main__":
    main()
