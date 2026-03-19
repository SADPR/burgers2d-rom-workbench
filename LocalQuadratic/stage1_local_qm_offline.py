#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 1: OFFLINE LOCAL QUADRATIC MANIFOLD BUILD

Produces:
  - LocalQuadratic/local_qm_data.npz
  - LocalQuadratic/stage1_local_qm_offline_summary.txt
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
localquadratic_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps
from burgers.core import get_snapshot_params
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


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


def as_object_array(seq):
    out = np.empty(len(seq), dtype=object)
    for i, item in enumerate(seq):
        out[i] = item
    return out


def n_from_tol(svals, pod_tol):
    svals = np.asarray(svals, dtype=np.float64).reshape(-1)
    if svals.size == 0:
        return 1, np.nan, np.nan

    s2 = svals ** 2
    total = float(np.sum(s2))
    if total <= 0.0:
        return 1, 1.0, 0.0

    residual = 1.0 - np.cumsum(s2) / total
    idx = np.where(residual <= pod_tol)[0]
    if idx.size == 0:
        n_keep = int(svals.size)
    else:
        n_keep = int(idx[0] + 1)
    captured = 1.0 - float(residual[n_keep - 1])
    lost = float(residual[n_keep - 1])
    return n_keep, captured, lost


def build_Q_quadratic_symmetric(q_mat):
    n, _ = q_mat.shape
    i_triu, j_triu = np.triu_indices(n)
    return q_mat[i_triu, :] * q_mat[j_triu, :]


def compute_H_ridge(E, Q, alpha):
    m = Q.shape[0]
    A = Q @ Q.T + alpha * np.eye(m, dtype=np.float64)
    B = E @ Q.T
    try:
        H = np.linalg.solve(A.T, B.T).T
    except np.linalg.LinAlgError:
        H = np.linalg.lstsq(A.T, B.T, rcond=None)[0].T
    return H


def build_global_snapshot_matrix(dt, num_steps, snap_folder, param_list=None):
    if param_list is None:
        param_list = get_snapshot_params()
    if len(param_list) == 0:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    print(f"[OFFLINE-QM] Using {len(param_list)} parameter points.")

    w0 = np.asarray(W0, dtype=np.float64).copy()
    S0 = load_or_compute_snaps(
        param_list[0], GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder
    )
    S0 = np.asarray(S0, dtype=np.float64)
    N, T = S0.shape

    S = np.zeros((N, len(param_list) * T), dtype=np.float64)
    col = 0
    for mu in param_list:
        S_mu = load_or_compute_snaps(
            mu, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder
        )
        S[:, col:col + T] = np.asarray(S_mu, dtype=np.float64)
        col += T

    print(f"[OFFLINE-QM] Snapshot matrix S: {S.shape}")
    return S


def cluster_snapshots_kmeans(S, n_clusters):
    S = np.asarray(S, dtype=np.float64)
    print(f"[OFFLINE-QM] Running k-means on {S.shape[1]} snapshots...")

    t0 = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
    kmeans.fit(S.T)
    print(f"[OFFLINE-QM] k-means finished in {time.time() - t0:.2f}s")

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.T.astype(np.float64)
    return labels, centers


def cluster_snapshots_fuzzy(S, n_clusters, m=2.0, error=1e-5, maxiter=1000):
    if fuzz is None:
        raise ImportError(
            "scikit-fuzzy is required for fuzzy c-means clustering.\n"
            "Install it with: pip install scikit-fuzzy"
        )

    S = np.asarray(S, dtype=np.float64)
    print(f"[OFFLINE-QM] Running fuzzy c-means on {S.shape[1]} snapshots...")

    t0 = time.time()
    cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=S,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
        seed=1,
    )
    print(f"[OFFLINE-QM] fuzzy c-means finished in {time.time() - t0:.2f}s")

    centers = cntr.T.astype(np.float64)
    labels = np.argmax(U, axis=0)
    return labels, centers


def build_overlapping_clusters(S, labels, centers, phi):
    S = np.asarray(S, dtype=np.float64)
    _, M = S.shape
    K = centers.shape[1]

    Sk = [np.where(labels == k)[0] for k in range(K)]
    Sk_plus = [set(indices.tolist()) for indices in Sk]

    neik = [set() for _ in range(K)]
    for s in range(M):
        us = S[:, s]
        dists = np.linalg.norm(centers - us[:, None], axis=0)
        idx_sorted = np.argsort(dists)
        k = int(idx_sorted[0])
        l = int(idx_sorted[1])
        neik[k].add(l)
        neik[l].add(k)

    for k in range(K):
        for l in neik[k]:
            Sl = Sk[l]
            if Sl.size == 0:
                continue
            n_add = int(np.floor(phi * Sl.size))
            if n_add <= 0:
                continue

            Sl_snaps = S[:, Sl]
            dists_kl = np.linalg.norm(Sl_snaps - centers[:, k][:, None], axis=0)
            chosen = Sl[np.argsort(dists_kl)[:n_add]]
            Sk_plus[k].update(chosen.tolist())

    return [np.array(sorted(list(Sk_plus[k])), dtype=int) for k in range(K)]


def build_local_qm_bases(
    S,
    cluster_indices,
    centers,
    pod_tol=1e-6,
    zeta_qua=0.1,
    alpha_ridge=1e-4,
    pod_method="svd",
):
    """
    For each cluster k:
      1) build centered local snapshots
      2) compute n_trad from POD tolerance (pod_tol)
      3) compute quadratic dimension n_k with heuristic
      4) fit H_k in u ≈ u_ref_k + V_k q + H_k Q(q)
    """
    S = np.asarray(S, dtype=np.float64)
    n_clusters = centers.shape[1]

    u0_list = []
    uc_list = []
    uref_list = []
    V_list = []
    H_list = []

    n_trad_list = []
    n_list = []
    pod_energy_captured = []
    pod_energy_lost = []

    for k in range(n_clusters):
        idx_k = cluster_indices[k]
        S_k = S[:, idx_k]
        N, Ns_k = S_k.shape
        print(f"[OFFLINE-QM] Cluster {k}: {Ns_k} snapshots (after overlap)")

        u_ck = centers[:, k].astype(np.float64)
        u0_k = u_ck.copy()
        u_ref_k = u0_k.copy()
        S_k_centered = S_k - u_ref_k[:, None]

        t0 = time.time()
        if pod_method == "rsvd":
            U_full_k, s_all_k, _ = randomized_svd(
                S_k_centered,
                n_components=min(S_k_centered.shape),
                random_state=1,
            )
        else:
            U_full_k, s_all_k, _ = np.linalg.svd(S_k_centered, full_matrices=False)
        print(f"[OFFLINE-QM] Cluster {k}: SVD done in {time.time() - t0:.2f}s")

        n_trad_k, captured_k, lost_k = n_from_tol(s_all_k, pod_tol)

        n_qua_raw = (np.sqrt(9.0 + 8.0 * n_trad_k) - 3.0) / 2.0
        n_qua_corr = int(np.floor((1.0 + zeta_qua) * n_qua_raw))
        n_max_ls = int(np.floor((np.sqrt(1.0 + 8.0 * Ns_k) - 1.0) / 2.0))
        n_k = max(1, min(n_qua_corr, n_max_ls, int(s_all_k.size)))

        print(
            f"[OFFLINE-QM] Cluster {k}: n_trad={n_trad_k}, "
            f"n_qua_raw={n_qua_raw:.2f}, n_qua_corr={n_qua_corr}, n_k={n_k}"
        )

        V_k = U_full_k[:, :n_k]
        q_mat_k = V_k.T @ S_k_centered
        S_lin_k = u_ref_k[:, None] + V_k @ q_mat_k
        E_k = S_k - S_lin_k
        Q_k = build_Q_quadratic_symmetric(q_mat_k)
        H_k = compute_H_ridge(E_k, Q_k, alpha=alpha_ridge)

        u0_list.append(u0_k)
        uc_list.append(u_ck)
        uref_list.append(u_ref_k)
        V_list.append(V_k.astype(np.float64))
        H_list.append(H_k.astype(np.float64))

        n_trad_list.append(int(n_trad_k))
        n_list.append(int(n_k))
        pod_energy_captured.append(float(captured_k))
        pod_energy_lost.append(float(lost_k))

    return (
        u0_list,
        uc_list,
        uref_list,
        V_list,
        H_list,
        n_trad_list,
        n_list,
        pod_energy_captured,
        pod_energy_lost,
    )


def precompute_quantities(u0_list, uc_list, V_list):
    K = len(V_list)
    d_const = np.zeros((K, K), dtype=np.float64)
    g_list = np.empty((K, K), dtype=object)
    T_list = np.empty((K, K), dtype=object)
    h_list = np.empty((K, K), dtype=object)

    for k in range(K):
        u0_k = np.asarray(u0_list[k], dtype=np.float64)
        V_k = np.asarray(V_list[k], dtype=np.float64)
        for l in range(K):
            uc_l = np.asarray(uc_list[l], dtype=np.float64)
            diff = u0_k - uc_l
            d_const[k, l] = float(diff @ diff)
            g_list[k, l] = V_k.T @ diff

    for l in range(K):
        V_l = np.asarray(V_list[l], dtype=np.float64)
        u0_l = np.asarray(u0_list[l], dtype=np.float64)
        for k in range(K):
            V_k = np.asarray(V_list[k], dtype=np.float64)
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            T_list[l, k] = V_l.T @ V_k
            h_list[l, k] = V_l.T @ (u0_k - u0_l)

    return d_const, g_list, T_list, h_list


def save_npz(filename, **kwargs):
    np.savez(filename, **kwargs)
    print(f"[SAVE-QM] local_qm_data saved to {filename}")


def main():
    # ---------------- user choices ----------------
    n_clusters = 10
    dt = DT
    num_steps = NUM_STEPS
    snap_folder = os.path.join(parent_dir, "Results", "param_snaps")

    pod_tol = 1e-6
    zeta_qua = 0.1
    alpha_ridge = 1e-4
    pod_method = "svd"  # "svd" or "rsvd"

    clustering_method = "kmeans"  # "kmeans" or "fuzzy"
    phi = 0.1

    os.makedirs(snap_folder, exist_ok=True)

    print(f"[OFFLINE-QM] Clustering method: {clustering_method}, phi={phi}")
    print(
        f"[OFFLINE-QM] Local POD truncation: tol={pod_tol:.1e}, method={pod_method}"
    )

    run_start = time.time()
    param_list = get_snapshot_params()

    t0 = time.time()
    S = build_global_snapshot_matrix(
        dt, num_steps, snap_folder=snap_folder, param_list=param_list
    )
    elapsed_snapshots = time.time() - t0

    t0 = time.time()
    if clustering_method.lower() == "kmeans":
        labels, centers = cluster_snapshots_kmeans(S, n_clusters)
    elif clustering_method.lower() == "fuzzy":
        labels, centers = cluster_snapshots_fuzzy(S, n_clusters)
    else:
        raise ValueError("clustering_method must be 'kmeans' or 'fuzzy'.")
    elapsed_clustering = time.time() - t0

    t0 = time.time()
    if phi > 0.0:
        cluster_indices = build_overlapping_clusters(S, labels, centers, phi)
    else:
        cluster_indices = [np.where(labels == k)[0] for k in range(n_clusters)]
    elapsed_overlap = time.time() - t0

    t0 = time.time()
    (
        u0_list,
        uc_list,
        uref_list,
        V_list,
        H_list,
        n_trad_list,
        n_list,
        pod_energy_captured,
        pod_energy_lost,
    ) = build_local_qm_bases(
        S,
        cluster_indices,
        centers,
        pod_tol=pod_tol,
        zeta_qua=zeta_qua,
        alpha_ridge=alpha_ridge,
        pod_method=pod_method,
    )
    elapsed_local_qm = time.time() - t0

    t0 = time.time()
    d_const, g_list, T_list, h_list = precompute_quantities(u0_list, uc_list, V_list)
    elapsed_precompute = time.time() - t0

    t0 = time.time()
    local_model_file = os.path.join(localquadratic_dir, "local_qm_data.npz")
    save_npz(
        local_model_file,
        S_shape=S.shape,
        u0_list=as_object_array(u0_list),
        uc_list=as_object_array(uc_list),
        uref_list=as_object_array(uref_list),
        V_list=as_object_array(V_list),
        H_list=as_object_array(H_list),
        n_trad_list=np.asarray(n_trad_list, dtype=int),
        n_list=np.asarray(n_list, dtype=int),
        cluster_indices=as_object_array(cluster_indices),
        d_const=d_const,
        g_list=g_list,
        T_list=T_list,
        h_list=h_list,
    )
    elapsed_save = time.time() - t0

    cluster_sizes_disjoint = np.bincount(labels, minlength=n_clusters).astype(int)
    cluster_sizes_overlap = [int(idx.size) for idx in cluster_indices]
    total_time = time.time() - run_start

    report_path = os.path.join(localquadratic_dir, "stage1_local_qm_offline_summary.txt")
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage1_local_qm_offline.py"),
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
                    ("zeta_qua", zeta_qua),
                    ("alpha_ridge", alpha_ridge),
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
                    ("n_trad_per_cluster", n_trad_list),
                    ("n_qm_per_cluster", n_list),
                    ("pod_energy_captured_per_cluster", pod_energy_captured),
                    ("pod_energy_lost_per_cluster", pod_energy_lost),
                    ("min_n_qm", int(np.min(n_list))),
                    ("max_n_qm", int(np.max(n_list))),
                    ("avg_n_qm", float(np.mean(n_list))),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_snapshots", elapsed_snapshots),
                    ("clustering", elapsed_clustering),
                    ("overlap_build", elapsed_overlap),
                    ("local_qm_build", elapsed_local_qm),
                    ("precompute_switching", elapsed_precompute),
                    ("save_model", elapsed_save),
                    ("total", total_time),
                ],
            ),
            (
                "outputs",
                [
                    ("local_qm_model_npz", local_model_file),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[SAVE-QM] stage1 text summary saved to {report_path}")


if __name__ == "__main__":
    main()
