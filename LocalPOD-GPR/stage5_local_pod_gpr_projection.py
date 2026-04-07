#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 5: LOCAL POD-GPR PROJECTION / DIAGNOSTICS

Local POD-GPR Projection (LOCAL PROM-GPR, no dynamics).

This script tests the purely data-driven local POD-GPR reconstruction:

  - Load the all-in-one offline data produced by Stage 4:
        local_pod_gpr_all_offline.npz

  - Load the HDM trajectory for a given (mu1, mu2).

  - For each time step:
        1) Maintain an active cluster index k_t.
           For t = 0, select k_0 as the one whose affine origin u0_k
           is closest to the HDM snapshot u_t in full space:
               k_0 = argmin_k ||u_t - u0_k||^2.
           For t > 0, use Grimberg-style reduced-space selection:
               - Compute y_k = V_k^T (u_t - u0_k) in the current cluster k.
               - Update the cluster via:
                     l* = argmin_l ( 2 g_{k,l}^T y_k + d_const[k,l] )

        2) In the selected cluster k_t, represent u_t via local POD-GPR:
               u_t ~= u0_k + V_k q_k,
           with q_s predicted from q_p via GPR when available.

  - Compute the global relative error over the full trajectory.
  - Plot HDM vs Local POD-GPR snapshots using plot_snaps().

Additionally, this script computes a reference local POD reconstruction
(without GPR closure) using the full local basis V_k and the full q_lin.
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
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS
from burgers.pod_gpr_manifold import decode_gp


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
# LOAD OFFLINE LOCAL POD-GPR DATA (Stage 4 all-in-one file)
# ----------------------------------------------------------------------
def load_local_pod_gpr_data(filename=os.path.join(script_dir, "local_pod_gpr_all_offline.npz")):
    """
    Load the unified NPZ file produced by Stage 4.

    Expected content:
        - u0_list         : (K,) object array of affine origins u0_k
        - V_list          : (K,) object array of local POD bases V_k
        - cluster_indices : (K,) object array of global snapshot indices
        - n_primary       : int  (same for all clusters)
        - models          : (K,) object array, each entry is the dict
                            returned by train_gpr_for_cluster(...)
        - d_const         : (K,K) float array, d_const[k,l] ~= ||u0_k - u0_l||^2
        - g_list          : (K,K) object array, g_list[k,l] = V_k^T (u0_k - u0_l)
    """
    data = np.load(filename, allow_pickle=True)

    u0_raw = data["u0_list"]
    v_raw = data["V_list"]
    cluster_indices = list(data["cluster_indices"])
    n_primary = int(data["n_primary"])
    models_raw = data["models"]

    d_const = np.asarray(data["d_const"], dtype=float)
    g_raw = data["g_list"]

    u0_list = [np.asarray(u, dtype=float) for u in u0_raw]
    v_list = [np.asarray(v, dtype=float) for v in v_raw]
    models = [m for m in models_raw]

    k_count = d_const.shape[0]
    g_list = [[None for _ in range(k_count)] for _ in range(k_count)]
    for k in range(k_count):
        for l in range(k_count):
            g_list[k][l] = np.asarray(g_raw[k, l], dtype=float)

    return (
        u0_list,
        v_list,
        cluster_indices,
        n_primary,
        models,
        d_const,
        g_list,
    )


# ----------------------------------------------------------------------
# CLUSTER SELECTION
# ----------------------------------------------------------------------
def select_cluster_by_u0(u, u0_list):
    """
    Pick cluster index as the closest affine origin u0_k in full space:

        k = argmin_k ||u - u0_k||^2
    """
    d2 = [np.linalg.norm(u - u0_k) ** 2 for u0_k in u0_list]
    return int(np.argmin(d2))


def select_cluster_reduced(k_current, y_k, d_const, g_list):
    """
    Grimberg reduced-space selector (first-order approximation):

        l* = argmin_l ( 2 g_{k,l}^T y_k + d_const[k,l] )
    """
    k_count = d_const.shape[0]
    y_k = np.asarray(y_k, dtype=float)
    scores = np.empty(k_count, dtype=float)

    for l in range(k_count):
        g_kl = g_list[k_current][l]
        scores[l] = 2.0 * (g_kl @ y_k) + d_const[k_current, l]

    return int(np.argmin(scores))


# ----------------------------------------------------------------------
# LOCAL POD-GPR PROJECTION (for a given cluster k)
# ----------------------------------------------------------------------
def _predict_qs_with_gpr(gpr_model, scaler, q_p, use_custom_predict=True):
    x_scaled = scaler.transform(q_p.reshape(1, -1))

    if use_custom_predict:
        has_custom = (
            hasattr(gpr_model, "X_train_")
            and hasattr(gpr_model, "alpha_")
            and hasattr(gpr_model, "kernel_")
        )
        if has_custom:
            k_vec = gpr_model.kernel_(gpr_model.X_train_, x_scaled).ravel()
            q_s = k_vec @ gpr_model.alpha_
            return np.asarray(q_s, dtype=float).reshape(-1)

    q_s = gpr_model.predict(x_scaled)
    return np.asarray(q_s, dtype=float).reshape(-1)


def local_pod_gpr_project(u, k, u0_list, v_list, models, n_primary, use_custom_predict=True):
    """
    Project full state u onto the local POD manifold of cluster k and
    apply the GPR closure on the secondary coordinates (if available).

    Returns
    -------
    u_rec : (N,) ndarray
        Reconstructed state using POD-GPR in cluster k.
    q_k   : (r_k,) ndarray
        Reduced coordinates used for this reconstruction.
    """
    u = np.asarray(u, dtype=float)
    u0_k = u0_list[k]
    v_k = v_list[k]
    model_k = models[k]

    r_k = v_k.shape[1]
    q_lin = v_k.T @ (u - u0_k)

    has_gpr = bool(model_k.get("has_gpr", False))
    n_secondary = int(model_k.get("n_secondary", 0))
    n_total_k = int(model_k.get("n_total", r_k))

    if (not has_gpr) or (n_secondary <= 0) or (r_k <= n_primary):
        q_k = q_lin
        u_rec = u0_k + v_k @ q_k
        return u_rec, q_k

    n_total_k = min(n_total_k, r_k)
    n_secondary = min(n_secondary, max(0, n_total_k - n_primary))
    if n_secondary <= 0:
        q_k = q_lin
        u_rec = u0_k + v_k @ q_k
        return u_rec, q_k

    q_p = q_lin[:n_primary]

    scaler = model_k["scaler"]
    gpr_model = model_k["gpr_model"]

    q_s_pred = _predict_qs_with_gpr(
        gpr_model=gpr_model,
        scaler=scaler,
        q_p=q_p,
        use_custom_predict=use_custom_predict,
    )

    q_k = np.zeros(r_k, dtype=float)
    q_k[:n_primary] = q_p
    q_k[n_primary : n_primary + n_secondary] = q_s_pred[:n_secondary]

    # Reconstruction via common manifold decoder for consistency.
    u_rec = decode_gp(
        q_p=q_p,
        gp_model=gpr_model,
        basis=v_k[:, :n_primary],
        basis2=v_k[:, n_primary : n_primary + n_secondary],
        scaler=scaler,
        u_ref=u0_k,
        use_custom_predict=use_custom_predict,
        echo_level=0,
    )

    return np.asarray(u_rec, dtype=float).reshape(-1), q_k


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(
    mu1=4.56,
    mu2=0.019,
    dt=DT,
    num_steps=NUM_STEPS,
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    local_model_file=os.path.join(script_dir, "local_pod_gpr_all_offline.npz"),
    output_dir=script_dir,
    use_custom_predict=True,
    selector_mode="nonlinear",
):

    os.makedirs(output_dir, exist_ok=True)

    print(f"[ONLINE-GPR] Using mu1={mu1:.3f}, mu2={mu2:.3f}")
    print(f"[ONLINE-GPR] dt={dt}, num_steps={num_steps}")
    print(f"[ONLINE-GPR] Snap folder: {snap_folder}")
    print(f"[ONLINE-GPR] Offline model file: {local_model_file}")
    selector_mode = str(selector_mode).strip().lower()
    if selector_mode not in ("linear", "nonlinear"):
        raise ValueError("selector_mode must be one of: 'linear', 'nonlinear'.")

    # ---------------- Load offline local POD-GPR model ----------------
    print("\n[ONLINE-GPR] Loading local POD-GPR model...")
    (
        u0_list,
        v_list,
        cluster_indices,
        n_primary,
        models,
        d_const,
        g_list,
    ) = load_local_pod_gpr_data(local_model_file)

    k_count = len(v_list)
    print(f"[ONLINE-GPR] Loaded {k_count} clusters. n_primary = {n_primary}\n")
    print(f"[ONLINE-GPR] Cluster selector mode = {selector_mode}")

    # ---------------- Load HDM trajectory ----------------
    print(f"[ONLINE-GPR] Loading HDM for mu1={mu1:.3f}, mu2={mu2:.3f}")
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
    _, t_count = hdm_snaps.shape
    print(f"[ONLINE-GPR] HDM snapshots shape = {hdm_snaps.shape}\n")

    # ---------------- Storage and error accumulators ----------------
    local_rec = np.zeros_like(hdm_snaps)
    local_pod_rec = np.zeros_like(hdm_snaps)
    cluster_history = np.zeros(t_count, dtype=int)

    total_err2_gpr = 0.0
    total_err2_pod = 0.0
    total_norm2 = 0.0

    t0_global = time.time()

    # ---------------- Time loop with reduced-space selection ---------
    k_current = None

    for t in range(t_count):
        u_t = hdm_snaps[:, t]

        # Initial cluster: full-space nearest u0
        if t == 0 or k_current is None:
            k_current = select_cluster_by_u0(u_t, u0_list)

        # Reduced coordinates in the current cluster
        u0_k = u0_list[k_current]
        v_k = v_list[k_current]
        y_k = v_k.T @ (u_t - u0_k)
        if selector_mode == "nonlinear":
            _, y_k = local_pod_gpr_project(
                u=u_t,
                k=k_current,
                u0_list=u0_list,
                v_list=v_list,
                models=models,
                n_primary=n_primary,
                use_custom_predict=use_custom_predict,
            )

        # One-step Grimberg reduced-space update
        k_new = select_cluster_reduced(k_current, y_k, d_const, g_list)

        if k_new != k_current:
            u0_new = u0_list[k_new]
            v_new = v_list[k_new]
            y_k_new = v_new.T @ (u_t - u0_new)
            k_current = k_new
            y_k = y_k_new

        k = k_current
        cluster_history[t] = k

        # Project and reconstruct with POD-GPR in selected cluster
        u_rec_t, _ = local_pod_gpr_project(
            u=u_t,
            k=k,
            u0_list=u0_list,
            v_list=v_list,
            models=models,
            n_primary=n_primary,
            use_custom_predict=use_custom_predict,
        )
        local_rec[:, t] = u_rec_t

        # Reference: pure local POD (full basis, no GPR)
        u0_k = u0_list[k]
        v_k = v_list[k]
        q_lin = v_k.T @ (u_t - u0_k)
        u_pod_t = u0_k + v_k @ q_lin
        local_pod_rec[:, t] = u_pod_t

        # Accumulate errors
        diff_gpr = u_t - u_rec_t
        diff_pod = u_t - u_pod_t
        total_err2_gpr += diff_gpr @ diff_gpr
        total_err2_pod += diff_pod @ diff_pod
        total_norm2 += u_t @ u_t

        if (t % 10) == 0 or t == t_count - 1:
            print(f"[STEP {t+1}/{t_count}] active cluster = {k}")

    # ---------------- Final error & timings ----------------
    rel_err_gpr = np.sqrt(total_err2_gpr / total_norm2)
    rel_err_pod = np.sqrt(total_err2_pod / total_norm2)
    total_time = time.time() - t0_global

    print(f"\n[ONLINE-GPR] Local POD-GPR relative error   = {rel_err_gpr:.4e}")
    print(f"[ONLINE-GPR] Local POD (full basis) error  = {rel_err_pod:.4e}")
    print(f"[ONLINE-GPR] Total reconstruction time     = {total_time:.2f} s")
    print(f"[ONLINE-GPR] Avg per step                 = {total_time / t_count:.4e} s\n")

    # Cluster usage statistics
    unique_clusters, counts = np.unique(cluster_history, return_counts=True)
    print("[ONLINE-GPR] Cluster usage over trajectory:")
    for c, cnt in zip(unique_clusters, counts):
        frac = 100.0 * cnt / t_count
        print(f"  cluster {c}: {cnt} steps ({frac:.1f}%)")

    # ---------------- Plot snapshots ----------------
    inds = range(0, t_count, max(1, t_count // 5))
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
        label="Local POD-GPR",
        fig_ax=(fig, ax1, ax2),
        color="#1f77b4",
        linewidth=3,
    )

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
        rf"Local POD--GPR vs Local POD "
        rf"($\mu_1={mu1:.2f}$, $\mu_2={mu2:.3f}$) "
        rf"err\_GPR={100 * rel_err_gpr:.2f}%, "
        rf"err\_POD={100 * rel_err_pod:.2f}%",
        fontsize=15,
    )

    out_png = os.path.join(
        output_dir,
        f"local_pod_gpr_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[ONLINE-GPR] Saved plot: {out_png}")

    out_gpr_npy = os.path.join(
        output_dir,
        f"local_pod_gpr_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    out_pod_npy = os.path.join(
        output_dir,
        f"local_pod_reference_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    out_cluster_npy = os.path.join(
        output_dir,
        f"local_pod_gpr_cluster_history_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(out_gpr_npy, local_rec)
    np.save(out_pod_npy, local_pod_rec)
    np.save(out_cluster_npy, cluster_history)

    summary_txt = os.path.join(
        output_dir,
        f"stage5_local_pod_gpr_projection_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("n_clusters", int(k_count)),
                    ("n_primary", int(n_primary)),
                    ("use_custom_predict", bool(use_custom_predict)),
                    ("selector_mode", selector_mode),
                ],
            ),
            (
                "errors",
                [
                    ("rel_err_local_pod_gpr", float(rel_err_gpr)),
                    ("rel_err_local_pod_full_basis", float(rel_err_pod)),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("total_reconstruction", float(total_time)),
                    ("avg_per_step", float(total_time / t_count)),
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
                    ("gpr_reconstruction_npy", out_gpr_npy),
                    ("pod_reference_npy", out_pod_npy),
                    ("cluster_history_npy", out_cluster_npy),
                    ("projection_plot_png", out_png),
                    ("summary_txt", summary_txt),
                ],
            ),
        ],
    )
    print(f"[ONLINE-GPR] Saved summary: {summary_txt}")


if __name__ == "__main__":
    main()
