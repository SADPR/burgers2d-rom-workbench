#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE 1: OFFLINE LOCAL POD + PRECOMPUTATIONS
Produces a SINGLE file: local_pod_data.npz

Content of local_pod_data.npz:
    - S_shape        : shape of global snapshot matrix S
    - u0_list        : list of local affine origins u0_k  (cluster means)
    - uc_list        : list of cluster centroids u_c,k    (same here)
    - V_list         : list of local POD bases V_k
    - cluster_indices: list of snapshot indices per (possibly overlapping) cluster
    - d_const        : (K,K) array with d_const[k,l] = ||u0_k - u_c,l||^2
    - g_list         : (K,K) object array with g_list[k,l] = V_k^T (u0_k - u_c,l)
    - T_list         : (K,K) object array with T_list[l,k] = V_l^T V_k
    - h_list         : (K,K) object array with h_list[l,k] = V_l^T (u0_k - u0_l)

The NPZ structure is unchanged; only the way clusters are built changes
(k-means vs fuzzy c-means, with optional overlap φ).
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

# fuzzy c-means (optional)
try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

# -------------------------------------------------------------
# IMPORTS AND PATH SETUP
# -------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
localpod_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps
from burgers.core import get_snapshot_params
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


# ================================================================
# POD UTILITY
# ================================================================
def pod_basis(snaps, n_modes=None, tol=None, method="rsvd"):
    """
    Compute POD basis for columns of `snaps`.

    Parameters
    ----------
    snaps : (N,M) array
    n_modes : int or None
    tol : float, used if n_modes is None
    method : 'svd' or 'rsvd'
    """
    snaps = np.asarray(snaps, dtype=float)
    N, M = snaps.shape

    if method == "svd":
        U, s, _ = np.linalg.svd(snaps, full_matrices=False)
    else:
        # randomized SVD on full rank (min(N,M)), we will truncate later
        U, s, _ = randomized_svd(snaps, n_components=min(N, M),
                                 random_state=1)

    if n_modes is None:
        if tol is None:
            raise ValueError("If n_modes is None, a tolerance 'tol' must be provided.")
        s2 = s ** 2
        loss = 1.0 - np.cumsum(s2) / np.sum(s2)
        n_modes = int(np.argmax(loss <= tol)) + 1

    return U[:, :n_modes].astype(float), s[:n_modes].astype(float)


# ================================================================
# GLOBAL SNAPSHOT MATRIX
# ================================================================
def build_global_snapshot_matrix(dt, num_steps, snap_folder, param_list=None):
    """
    Build global snapshot matrix S = [ S(mu1) | S(mu2) | ... ].
    """
    if param_list is None:
        param_list = get_snapshot_params()
    print(f"[OFFLINE] Using {len(param_list)} parameter points.")

    # ------------------------------------------------------------
    # FIX: correct size of initial condition w0
    # ------------------------------------------------------------
    N_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    w0 = np.ones(2 * N_cells, dtype=float)
    # ------------------------------------------------------------

    # 1st snapshot to determine sizes
    S0 = load_or_compute_snaps(
        param_list[0],
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder=snap_folder
    )
    S0 = np.asarray(S0, dtype=float)
    N, T = S0.shape

    S = np.zeros((N, len(param_list) * T), dtype=float)
    col = 0
    for mu in param_list:
        S_mu = load_or_compute_snaps(
            mu,
            GRID_X,
            GRID_Y,
            w0,
            dt,
            num_steps,
            snap_folder=snap_folder
        )
        S_mu = np.asarray(S_mu, dtype=float)
        S[:, col:col + T] = S_mu
        col += T

    print(f"[OFFLINE] Snapshot matrix S: {S.shape}")
    return S



# ================================================================
# CLUSTERING: HARD K-MEANS
# ================================================================
def cluster_snapshots_kmeans(S, n_clusters):
    """
    Hard k-means clustering of columns of S (snapshots).
    Returns
    -------
    labels  : (M,) primary cluster index for each snapshot
    centers : (N, n_clusters) cluster centroids
    """
    S = np.asarray(S, dtype=float)
    M = S.shape[1]
    print(f"[OFFLINE] Running k-means on {M} snapshots...")

    t0 = time.time()
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=1,
                    n_init=10)
    kmeans.fit(S.T)
    print(f"[OFFLINE] k-means finished in {time.time() - t0:.2f}s")

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.T.astype(float)  # shape (N, nc)

    return labels, centers


# ================================================================
# CLUSTERING: FUZZY C-MEANS
# ================================================================
def cluster_snapshots_fuzzy(S, n_clusters, m=2.0, error=1e-5, maxiter=1000):
    """
    Fuzzy c-means clustering of columns of S (snapshots).

    Returns
    -------
    labels  : (M,) primary cluster index (argmax membership)
    centers : (N, n_clusters) fuzzy cluster centers
    """
    if fuzz is None:
        raise ImportError(
            "scikit-fuzzy is required for fuzzy c-means clustering.\n"
            "Install it with: pip install scikit-fuzzy"
        )

    S = np.asarray(S, dtype=float)
    N, M = S.shape
    print(f"[OFFLINE] Running fuzzy c-means on {M} snapshots...")

    t0 = time.time()
    # skfuzzy expects data as (features, samples) = (N, M)
    cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=S,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
        seed=1
    )
    print(f"[OFFLINE] fuzzy c-means finished in {time.time() - t0:.2f}s")

    # cntr: (n_clusters, N)
    centers = cntr.T.astype(float)  # (N, n_clusters)
    # U: (n_clusters, M) membership; primary label = argmax membership
    labels = np.argmax(U, axis=0)

    return labels, centers


# ================================================================
# OVERLAP CONSTRUCTION (Algorithm 1 style)
# ================================================================
def build_overlapping_clusters(S, labels, centers, phi):
    """
    Build overlapping clusters S_k^+ from disjoint labels S_k and
    cluster centers, following Algorithm 1 (Grimberg et al.) in spirit.

    Parameters
    ----------
    S       : (N,M) snapshot matrix
    labels  : (M,) primary cluster index for each snapshot
    centers : (N,K) cluster centroids
    phi     : float in [0,1], fraction of |S_l| to add to neighbor k

    Returns
    -------
    cluster_indices_plus : list of length K
        Each entry is a sorted ndarray of snapshot indices belonging to S_k^+.
    """
    S = np.asarray(S, dtype=float)
    N, M = S.shape
    K = centers.shape[1]

    # Initial disjoint clusters S_k
    Sk = [np.where(labels == k)[0] for k in range(K)]
    Sk_plus = [set(indices.tolist()) for indices in Sk]

    # --- Step 1: build intercluster connectivity neik using closest two centers ---
    neik = [set() for _ in range(K)]

    for s in range(M):
        us = S[:, s]
        # distances to all centers
        dists = np.linalg.norm(centers - us[:, None], axis=0)
        # indices of two closest centers
        idx_sorted = np.argsort(dists)
        k = int(idx_sorted[0])
        l = int(idx_sorted[1])

        neik[k].add(l)
        neik[l].add(k)

    # --- Step 2: add overlap ---
    for k in range(K):
        for l in neik[k]:
            Sl = Sk[l]
            if Sl.size == 0:
                continue

            n_add = int(np.floor(phi * Sl.size))
            if n_add <= 0:
                continue

            # For snapshots belonging to cluster l, find those closest to center k
            Sl_snaps = S[:, Sl]                     # shape (N, |S_l|)
            diff = Sl_snaps - centers[:, k][:, None]
            dists_kl = np.linalg.norm(diff, axis=0)

            idx_sort = np.argsort(dists_kl)
            chosen = Sl[idx_sort[:n_add]]

            Sk_plus[k].update(chosen.tolist())

    # Convert to sorted arrays
    cluster_indices_plus = [np.array(sorted(list(Sk_plus[k])), dtype=int)
                            for k in range(K)]

    return cluster_indices_plus


# ================================================================
# LOCAL POD BUILDER
# ================================================================
def build_local_pod_bases(S, cluster_indices, centers, pod_tol=1e-6, pod_method="rsvd"):
    """
    For each cluster k:
      - use snapshot set S_k^+ given by cluster_indices[k]
      - build local mean u0_k (chosen as the cluster centroid centers[:,k])
      - compute local POD basis V_k from centered snapshots
        using tolerance-based truncation.
    """
    S = np.asarray(S, dtype=float)
    n_clusters = centers.shape[1]

    u0_list = []
    uc_list = []
    V_list = []

    for k in range(n_clusters):
        idx_k = cluster_indices[k]
        S_k = S[:, idx_k]
        print(f"[OFFLINE] Cluster {k}: {S_k.shape[1]} snapshots (after overlap)")

        u_ck = centers[:, k].astype(float)   # centroid of original disjoint cluster
        u0_k = u_ck.copy()                   # affine origin

        S_k_centered = S_k - u0_k[:, None]

        V_k, _ = pod_basis(
            S_k_centered,
            n_modes=None,
            tol=pod_tol,
            method=pod_method,
        )
        print(f"[OFFLINE] Cluster {k}: retained POD modes = {V_k.shape[1]} (tol={pod_tol:.1e})")

        u0_list.append(u0_k.astype(float))
        uc_list.append(u_ck.astype(float))
        V_list.append(V_k.astype(float))

    return u0_list, uc_list, V_list


# ================================================================
# PRECOMPUTATIONS FOR ONLINE PHASE (GRIMBERG-STYLE)
# ================================================================
def precompute_quantities(u0_list, uc_list, V_list):
    """
    Precompute:
      - d_const[k,l] = ||u0_k - u_c,l||^2
      - g_list[k,l]  = V_k^T (u0_k - u_c,l)
      - T_list[l,k]  = V_l^T V_k
      - h_list[l,k]  = V_l^T (u0_k - u0_l)

    All stored as 2D object arrays of shape (K,K).
    """
    K = len(V_list)

    d_const = np.zeros((K, K), dtype=float)
    g_list = np.empty((K, K), dtype=object)
    T_list = np.empty((K, K), dtype=object)
    h_list = np.empty((K, K), dtype=object)

    # distance constants + g vectors
    for k in range(K):
        u0_k = np.asarray(u0_list[k], dtype=float)
        Vk   = np.asarray(V_list[k], dtype=float)

        for l in range(K):
            uc_l = np.asarray(uc_list[l], dtype=float)
            diff = u0_k - uc_l
            d_const[k, l] = float(diff @ diff)
            g_list[k, l]  = Vk.T @ diff   # shape (n_k,)

    # switching operators (note indexing: T_list[l,k] and h_list[l,k])
    for l in range(K):
        Vl   = np.asarray(V_list[l], dtype=float)
        u0_l = np.asarray(u0_list[l], dtype=float)

        for k in range(K):
            Vk   = np.asarray(V_list[k], dtype=float)
            u0_k = np.asarray(u0_list[k], dtype=float)

            T_list[l, k] = Vl.T @ Vk                     # (n_l, n_k)
            h_list[l, k] = Vl.T @ (u0_k - u0_l)          # (n_l,)

    return d_const, g_list, T_list, h_list


# ================================================================
# SAVE EVERYTHING  (unchanged NPZ format)
# ================================================================
def save_npz(filename, **kwargs):
    np.savez(filename, **kwargs)
    print(f"[SAVE] local_pod_data saved to {filename}")


def as_object_array(seq):
    """
    Robust conversion of a Python sequence of arrays/objects into a 1D
    object array. This avoids numpy trying to broadcast nested arrays.
    """
    out = np.empty(len(seq), dtype=object)
    for i, item in enumerate(seq):
        out[i] = item
    return out


# ================================================================
# MAIN
# ================================================================
def main():

    # ---------------- user choices ----------------
    n_clusters = 10
    pod_tol = 1e-6
    pod_method = "rsvd"
    dt = DT
    num_steps = NUM_STEPS
    snap_folder = os.path.join(parent_dir, "Results", "param_snaps")

    # clustering options
    clustering_method = "kmeans"  # "kmeans" or "fuzzy"
    phi = 0.1                     # overlap factor in [0,1]

    print(f"[OFFLINE] Clustering method: {clustering_method}, phi={phi}")
    print(f"[OFFLINE] Local POD truncation: tol={pod_tol:.1e}, method={pod_method}")

    run_start = time.time()
    param_list = get_snapshot_params()

    # 1) build global snapshot matrix
    t0 = time.time()
    S = build_global_snapshot_matrix(
        dt, num_steps, snap_folder, param_list=param_list
    )
    elapsed_snapshots = time.time() - t0

    # 2) clustering (disjoint)
    t0 = time.time()
    if clustering_method.lower() == "kmeans":
        labels, centers = cluster_snapshots_kmeans(S, n_clusters)
    elif clustering_method.lower() == "fuzzy":
        labels, centers = cluster_snapshots_fuzzy(S, n_clusters)
    else:
        raise ValueError("clustering_method must be 'kmeans' or 'fuzzy'.")
    elapsed_clustering = time.time() - t0

    # 3) build (possibly overlapping) snapshot sets
    t0 = time.time()
    if phi > 0.0:
        cluster_indices = build_overlapping_clusters(S, labels, centers, phi)
    else:
        # no overlap: just disjoint clusters
        cluster_indices = [np.where(labels == k)[0] for k in range(n_clusters)]
    elapsed_overlap = time.time() - t0

    # 4) local POD on overlapping sets S_k^+
    t0 = time.time()
    u0_list, uc_list, V_list = build_local_pod_bases(
        S,
        cluster_indices,
        centers,
        pod_tol=pod_tol,
        pod_method=pod_method,
    )
    elapsed_local_pod = time.time() - t0

    # 5) Grimberg-style precomputations
    t0 = time.time()
    d_const, g_list, T_list, h_list = precompute_quantities(
        u0_list, uc_list, V_list
    )
    elapsed_precompute = time.time() - t0

    # 6) save single-file model  (format unchanged)
    t0 = time.time()
    local_model_file = os.path.join(localpod_dir, "local_pod_data.npz")
    save_npz(
        local_model_file,
        S_shape=S.shape,
        u0_list=as_object_array(u0_list),
        uc_list=as_object_array(uc_list),
        V_list=as_object_array(V_list),
        cluster_indices=as_object_array(cluster_indices),
        d_const=d_const,
        g_list=g_list,
        T_list=T_list,
        h_list=h_list,
    )
    elapsed_save = time.time() - t0

    cluster_sizes_disjoint = np.bincount(labels, minlength=n_clusters).astype(int)
    cluster_sizes_overlap = [int(idx.size) for idx in cluster_indices]
    retained_modes = [int(V.shape[1]) for V in V_list]
    total_time = time.time() - run_start

    report_path = os.path.join(localpod_dir, "stage1_local_pod_offline_summary.txt")
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage1_local_pod_offline_cheap.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("n_clusters", n_clusters),
                    ("clustering_method", clustering_method),
                    ("phi_overlap", phi),
                    ("pod_tol", pod_tol),
                    ("pod_method", pod_method),
                    ("num_training_parameters", len(param_list)),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "snapshot_matrix",
                [
                    ("S_shape", S.shape),
                    ("state_size", S.shape[0]),
                    ("total_snapshots", S.shape[1]),
                ],
            ),
            (
                "clusters",
                [
                    ("cluster_sizes_disjoint", cluster_sizes_disjoint.tolist()),
                    ("cluster_sizes_after_overlap", cluster_sizes_overlap),
                    ("retained_modes_per_cluster", retained_modes),
                    ("min_retained_modes", int(np.min(retained_modes))),
                    ("max_retained_modes", int(np.max(retained_modes))),
                    ("avg_retained_modes", float(np.mean(retained_modes))),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_snapshots", elapsed_snapshots),
                    ("clustering", elapsed_clustering),
                    ("overlap_build", elapsed_overlap),
                    ("local_pod", elapsed_local_pod),
                    ("precompute_switching", elapsed_precompute),
                    ("save_model", elapsed_save),
                    ("total", total_time),
                ],
            ),
            (
                "outputs",
                [
                    ("local_pod_model_npz", local_model_file),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[SAVE] stage1 text summary saved to {report_path}")


if __name__ == "__main__":
    main()
