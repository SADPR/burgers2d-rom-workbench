#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 5: LOCAL POD-RBF PROJECTION / DIAGNOSTICS

Local POD–RBF Projection (LOCAL PROM-RBF, no dynamics).

This script tests the *purely data-driven* local POD–RBF reconstruction:

  - Load the all-in-one offline data produced by Stage 4:
        local_pod_rbf_all_offline.npz

  - Load the HDM trajectory for a given (mu1, mu2).

  - For each time step:
        1) Maintain an active cluster index k_t.
           For t = 0, select k_0 as the one whose affine origin u0_k
           is closest to the HDM snapshot u_t in full space:
               k_0 = argmin_k ||u_t - u0_k||^2.
           For t > 0, use Grimberg-style *reduced-space* selection:
               - Compute y_k = V_k^T (u_t - u0_k) in the current cluster k.
               - Update the cluster via:
                     l* = argmin_l ( 2 g_{k,l}^T y_k + d_const[k,l] )

        2) In the selected cluster k_t, represent u_t via local POD–RBF:
               u_t ≈ u0_k + V_k q_k,
           with q_s predicted from q_p via RBF when available.

  - Compute the global relative error over the full trajectory.
  - Plot HDM vs Local POD–RBF snapshots using plot_snaps().

Additionally, this script computes a *reference local POD* reconstruction
(without RBF closure) using the full local basis V_k and the full q_lin, so
we can compare how far the POD–RBF is from the best linear local projection.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PATHS / IMPORTS
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.config import GRID_X, GRID_Y, W0

plt.rcParams.update(
    {
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": ["STIXGeneral"],
    }
)
plt.rc("font", size=13)


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


# ----------------------------------------------------------------------
# RBF kernels (must be consistent with Stage 4)
# ----------------------------------------------------------------------
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)

rbf_kernels = {
    "gaussian": gaussian_rbf,
    "imq": inverse_multiquadric_rbf,
}


# ----------------------------------------------------------------------
# LOAD OFFLINE LOCAL POD–RBF DATA (Stage 4 all-in-one file)
# ----------------------------------------------------------------------
def load_local_pod_rbf_data(filename=os.path.join(script_dir, "local_pod_rbf_all_offline.npz")):
    """
    Load the unified NPZ file produced by Stage 4.

    Expected content:
        - u0_list         : (K,) object array of affine origins u0_k
        - V_list          : (K,) object array of local POD bases V_k
        - cluster_indices : (K,) object array of global snapshot indices
        - n_primary       : int  (same for all clusters)
        - models          : (K,) object array, each entry is the dict
                            returned by train_rbf_for_cluster(...)
        - d_const         : (K,K) float array, d_const[k,l] ≈ ||u0_k - u0_l||^2
        - g_list          : (K,K) object array, g_list[k,l] = V_k^T (u0_k - u0_l)
    """
    data = np.load(filename, allow_pickle=True)

    u0_raw          = data["u0_list"]
    V_raw           = data["V_list"]
    cluster_indices = list(data["cluster_indices"])
    n_primary       = int(data["n_primary"])
    models_raw      = data["models"]

    d_const         = np.asarray(data["d_const"], dtype=float)
    g_raw           = data["g_list"]

    # Convert to clean Python lists / arrays
    u0_list = [np.asarray(u, dtype=float) for u in u0_raw]
    V_list  = [np.asarray(V, dtype=float) for V in V_raw]
    models  = [m for m in models_raw]

    # g_list as nested list g_list[k][l]
    K = d_const.shape[0]
    g_list = [[None for _ in range(K)] for _ in range(K)]
    for k in range(K):
        for l in range(K):
            g_list[k][l] = np.asarray(g_raw[k, l], dtype=float)

    return (
        u0_list,
        V_list,
        cluster_indices,
        n_primary,
        models,
        d_const,
        g_list,
    )


# ----------------------------------------------------------------------
# CLUSTER SELECTION (full-space nearest u0_k)
# ----------------------------------------------------------------------
def select_cluster_by_u0(u, u0_list):
    """
    Pick cluster index as the closest affine origin u0_k in full space:

        k = argmin_k ||u - u0_k||^2

    This is O(KN). Used for the initial snapshot (t=0).
    """
    d2 = [np.linalg.norm(u - u0_k) ** 2 for u0_k in u0_list]
    return int(np.argmin(d2))


# ----------------------------------------------------------------------
# REDUCED-SPACE GRIMBERG SELECTION (first-order rule)
# ----------------------------------------------------------------------
def select_cluster_reduced(k_current, y_k, d_const, g_list):
    """
    Grimberg reduced-space selector (first-order approximation):

        l* = argmin_l ( 2 g_{k,l}^T y_k + d_const[k,l] ),

    where:
      - k_current : active cluster index k,
      - y_k       : reduced coordinates in cluster k (V_k^T (u - u0_k)),
      - d_const   : (K,K) array with d_const[k,l] ≈ ||u0_k - u0_l||^2,
      - g_list    : nested list g_list[k][l] = V_k^T (u0_k - u0_l).

    Returns
    -------
    l_best : int
        Index of best cluster according to this reduced distance rule.
    """
    K = d_const.shape[0]
    y_k = np.asarray(y_k, dtype=float)
    scores = np.empty(K, dtype=float)

    for l in range(K):
        g_kl = g_list[k_current][l]
        scores[l] = 2.0 * (g_kl @ y_k) + d_const[k_current, l]

    return int(np.argmin(scores))


# ----------------------------------------------------------------------
# LOCAL POD–RBF PROJECTION (for a given cluster k)
# ----------------------------------------------------------------------
def local_pod_rbf_project(u, k, u0_list, V_list, models, n_primary):
    """
    Project full state u onto the local POD manifold of cluster k and
    apply the RBF closure on the secondary coordinates (if available).

    Manifold representation:

        u ≈ u0_k + V_k q_k,    with q_k ∈ R^{r_k}

    where the first n_primary components q_p are "primary" and the
    remaining ones q_s are "secondary" and approximated by

        q_s ≈ RBF_k(q_p).

    Parameters
    ----------
    u : (N,) ndarray
        Full HDM state.
    k : int
        Cluster index.
    u0_list, V_list :
        Offline affine origins and bases from Stage 4.
    models :
        List of model dicts for each cluster, as returned by
        train_rbf_for_cluster().
    n_primary : int
        Number of primary modes used as RBF inputs.

    Returns
    -------
    u_rec : (N,) ndarray
        Reconstructed state using POD–RBF in cluster k.
    q_k   : (r_k,) ndarray
        Reduced coordinates used for this reconstruction.
    """
    u = np.asarray(u, dtype=float)
    u0_k = u0_list[k]
    V_k  = V_list[k]
    model_k = models[k]

    # Linear reduced coordinates
    r_k = V_k.shape[1]
    q_lin = V_k.T @ (u - u0_k)  # (r_k,)

    # If no RBF or no secondary part, we just use linear q_lin
    has_rbf     = model_k.get("has_rbf", False)
    n_secondary = int(model_k.get("n_secondary", 0))
    n_total_k   = int(model_k.get("n_total", r_k))

    if (not has_rbf) or (n_secondary <= 0) or (r_k <= n_primary):
        q_k = q_lin
        u_rec = u0_k + V_k @ q_k
        return u_rec, q_k

    # Ensure consistency
    n_total_k   = min(n_total_k, r_k)
    n_secondary = min(n_secondary, n_total_k - n_primary)

    # Split linear coordinates into primary and secondary blocks
    q_p = q_lin[:n_primary]  # (n_primary,)

    # RBF components
    scaler      = model_k["scaler"]
    X_train     = np.asarray(model_k["q_p_train"], dtype=float)  # (n_samples, n_primary)
    W           = np.asarray(model_k["W"], dtype=float)          # (n_samples, n_secondary)
    epsilon     = float(model_k["epsilon"])
    kernel_name = model_k["kernel_name"]

    kernel_func = rbf_kernels[kernel_name]

    # Normalize q_p with the same MinMaxScaler
    q_p_norm = scaler.transform(q_p.reshape(1, -1))[0, :]  # (n_primary,)

    # Distances to training points in normalized space
    dists = np.linalg.norm(X_train - q_p_norm.reshape(1, -1), axis=1)  # (n_samples,)

    # RBF interpolation: q_s_pred ∈ R^{n_secondary}
    phi = kernel_func(dists, epsilon)  # (n_samples,)
    q_s_pred = phi @ W                 # (n_secondary,)

    # Build full reduced coordinates q_k (length r_k)
    q_k = np.zeros(r_k, dtype=float)
    q_k[:n_primary] = q_p
    q_k[n_primary : n_primary + n_secondary] = q_s_pred

    # Reconstruction
    u_rec = u0_k + V_k @ q_k
    return u_rec, q_k


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(
    mu1=4.56,
    mu2=0.019,
    dt=0.05,
    num_steps=500,
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    local_model_file=os.path.join(script_dir, "local_pod_rbf_all_offline.npz"),
    output_dir=script_dir,
):

    os.makedirs(output_dir, exist_ok=True)

    print(f"[ONLINE-RBF] Using μ1={mu1:.3f}, μ2={mu2:.3f}")
    print(f"[ONLINE-RBF] dt={dt}, num_steps={num_steps}")
    print(f"[ONLINE-RBF] Snap folder: {snap_folder}")
    print(f"[ONLINE-RBF] Offline model file: {local_model_file}")

    # ---------------- Load offline local POD–RBF model ----------------
    print(f"\n[ONLINE-RBF] Loading local POD–RBF model...")
    (
        u0_list,
        V_list,
        cluster_indices,
        n_primary,
        models,
        d_const,
        g_list,
    ) = load_local_pod_rbf_data(local_model_file)

    K = len(V_list)
    print(f"[ONLINE-RBF] Loaded {K} clusters. n_primary = {n_primary}\n")

    # ---------------- Load HDM trajectory ----------------
    print(f"[ONLINE-RBF] Loading HDM for μ1={mu1:.3f}, μ2={mu2:.3f}")
    hdm_snaps = load_or_compute_snaps(
        [mu1, mu2],
        GRID_X,
        GRID_Y,
        W0,
        dt,
        num_steps,
        snap_folder=snap_folder,
    )
    hdm_snaps = np.asarray(hdm_snaps, dtype=float)
    N, T = hdm_snaps.shape
    print(f"[ONLINE-RBF] HDM snapshots shape = {hdm_snaps.shape}\n")

    # ---------------- Storage and error accumulators ----------------
    local_rec     = np.zeros_like(hdm_snaps)  # POD–RBF reconstruction
    local_pod_rec = np.zeros_like(hdm_snaps)  # pure local POD (reference)
    cluster_history = np.zeros(T, dtype=int)

    total_err2_rbf = 0.0
    total_err2_pod = 0.0
    total_norm2    = 0.0

    t0_global = time.time()

    # ---------------- Time loop with reduced-space selection ---------
    k_current = None

    for t in range(T):
        u_t = hdm_snaps[:, t]

        # Initial cluster: full-space nearest u0
        if t == 0 or k_current is None:
            k_current = select_cluster_by_u0(u_t, u0_list)

        # Reduced coordinates in the current cluster
        u0_k = u0_list[k_current]
        V_k  = V_list[k_current]
        y_k  = V_k.T @ (u_t - u0_k)

        # One-step Grimberg reduced-space update
        k_new = select_cluster_reduced(k_current, y_k, d_const, g_list)

        if k_new != k_current:
            # Optionally, re-evaluate reduced coordinates in the new cluster
            u0_new = u0_list[k_new]
            V_new  = V_list[k_new]
            y_k_new = V_new.T @ (u_t - u0_new)
            k_current = k_new
            y_k = y_k_new  # (not used later, but conceptually updated)

        k = k_current
        cluster_history[t] = k

        # Project and reconstruct with POD–RBF in selected cluster
        u_rec_t, q_k = local_pod_rbf_project(
            u_t, k, u0_list, V_list, models, n_primary
        )
        local_rec[:, t] = u_rec_t

        # Reference: pure local POD (full basis, no RBF, i.e. q_lin)
        u0_k = u0_list[k]
        V_k  = V_list[k]
        q_lin = V_k.T @ (u_t - u0_k)
        u_pod_t = u0_k + V_k @ q_lin
        local_pod_rec[:, t] = u_pod_t

        # Accumulate errors
        diff_rbf = u_t - u_rec_t
        diff_pod = u_t - u_pod_t
        total_err2_rbf += diff_rbf @ diff_rbf
        total_err2_pod += diff_pod @ diff_pod
        total_norm2    += u_t @ u_t

        if (t % 10) == 0 or t == T - 1:
            print(f"[STEP {t+1}/{T}] active cluster = {k}")

    # ---------------- Final error & timings ----------------
    rel_err_rbf = np.sqrt(total_err2_rbf / total_norm2)
    rel_err_pod = np.sqrt(total_err2_pod / total_norm2)
    total_time = time.time() - t0_global

    print(f"\n[ONLINE-RBF] Local POD–RBF relative error  = {rel_err_rbf:.4e}")
    print(f"[ONLINE-RBF] Local POD (full basis) error = {rel_err_pod:.4e}")
    print(f"[ONLINE-RBF] Total reconstruction time    = {total_time:.2f} s")
    print(f"[ONLINE-RBF] Avg per step                = {total_time / T:.4e} s\n")

    # Cluster usage statistics
    unique_clusters, counts = np.unique(cluster_history, return_counts=True)
    print("[ONLINE-RBF] Cluster usage over trajectory:")
    for c, cnt in zip(unique_clusters, counts):
        frac = 100.0 * cnt / T
        print(f"  cluster {c}: {cnt} steps ({frac:.1f}%)")

    # ---------------- Plot snapshots ----------------
    inds = range(0, T, max(1, T // 5))  # ~5 snapshots along the trajectory
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # HDM
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

    # Local POD–RBF
    plot_snaps(
        GRID_X,
        GRID_Y,
        local_rec,
        inds,
        label="Local POD–RBF",
        fig_ax=(fig, ax1, ax2),
        color="#1f77b4",
        linewidth=3,
    )

    # Local POD (reference, full basis)
    plot_snaps(
        GRID_X,
        GRID_Y,
        local_pod_rec,
        inds,
        label="Local POD (full basis)",
        fig_ax=(fig, ax1, ax2),
        color="#ff7f0e",
        linewidth=2,
    )

    ax2.legend(loc="center right", fontsize=16, frameon=True)
    fig.suptitle(
        rf"Local POD--RBF vs Local POD "
        rf"($\mu_1={mu1:.2f}$, $\mu_2={mu2:.3f}$) "
        rf"err\_RBF={100 * rel_err_rbf:.2f}%, "
        rf"err\_POD={100 * rel_err_pod:.2f}%",
        fontsize=15,
    )

    out_png = os.path.join(
        output_dir,
        f"local_pod_rbf_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[ONLINE-RBF] Saved plot: {out_png}")

    out_rbf_npy = os.path.join(
        output_dir,
        f"local_pod_rbf_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    out_pod_npy = os.path.join(
        output_dir,
        f"local_pod_reference_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    out_cluster_npy = os.path.join(
        output_dir,
        f"local_pod_rbf_cluster_history_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(out_rbf_npy, local_rec)
    np.save(out_pod_npy, local_pod_rec)
    np.save(out_cluster_npy, cluster_history)

    summary_txt = os.path.join(
        output_dir,
        f"stage5_local_pod_rbf_projection_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        summary_txt,
        [
            (
                "configuration",
                [
                    ("mu1", float(mu1)),
                    ("mu2", float(mu2)),
                    ("dt", float(dt)),
                    ("num_steps", int(num_steps)),
                    ("snap_folder", snap_folder),
                    ("offline_model_file", local_model_file),
                    ("n_clusters", int(K)),
                    ("n_primary", int(n_primary)),
                ],
            ),
            (
                "errors",
                [
                    ("rel_err_local_pod_rbf", float(rel_err_rbf)),
                    ("rel_err_local_pod_full_basis", float(rel_err_pod)),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("total_reconstruction", float(total_time)),
                    ("avg_per_step", float(total_time / T)),
                ],
            ),
            (
                "cluster_usage",
                [
                    ("unique_clusters", unique_clusters.tolist()),
                    ("counts", counts.tolist()),
                ],
            ),
            (
                "outputs",
                [
                    ("rbf_reconstruction_npy", out_rbf_npy),
                    ("pod_reference_npy", out_pod_npy),
                    ("cluster_history_npy", out_cluster_npy),
                    ("projection_plot_png", out_png),
                    ("summary_txt", summary_txt),
                ],
            ),
        ],
    )
    print(f"[ONLINE-RBF] Saved summary: {summary_txt}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
