#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TRAIN LOCAL POD-RBF MODELS (GRID SEARCH WITH LAMBDA)

Inputs (from stage3):
  - local_rbf_q_per_cluster.npz

Outputs (inside LocalPOD-RBF):
  - local_pod_rbf_models.pkl
  - local_pod_rbf_switching.npz
  - local_pod_rbf_all_offline.npz
  - stage4_cluster_qs_train_error.png
  - stage4_training_summary.txt
"""

import os
import time
import pickle
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# REPORT HELPERS
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# RBF KERNELS + SEARCH GRIDS
# ---------------------------------------------------------------------
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)


def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)


def matern32_rbf(r, epsilon):
    sqrt3 = np.sqrt(3.0)
    z = sqrt3 * epsilon * r
    return (1.0 + z) * np.exp(-z)


rbf_kernels = {
    "gaussian": gaussian_rbf,
    #"imq": inverse_multiquadric_rbf,
    #"matern32": matern32_rbf,
}

EPSILON_VALUES = np.logspace(np.log10(0.1), np.log10(5.0), 15)
LAMBDA_VALUES = np.array([1e-12, 1e-10], dtype=float)


# ---------------------------------------------------------------------
# NUMERICAL HELPERS
# ---------------------------------------------------------------------
def svd_solve_full_rank(kmat, y, rcond=1e-12):
    """
    Solve kmat * W = y by SVD pseudo-inverse, robust to ill-conditioning.
    """
    u, s, vt = np.linalg.svd(kmat, full_matrices=False)

    if s.size == 0 or s[0] <= 0.0:
        return np.zeros((kmat.shape[1], y.shape[1]), dtype=y.dtype)

    tol = float(rcond) * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return vt.T @ (s_inv[:, None] * (u.T @ y))


def get_duplicate_mask(x, delta):
    """
    Keep one representative among near-duplicate rows of x.
    """
    n_samples = int(x.shape[0])
    keep = np.ones(n_samples, dtype=bool)

    if float(delta) <= 0.0 or n_samples <= 1:
        return keep

    for i in range(n_samples):
        if not keep[i]:
            continue
        xi = x[i]
        for j in range(i + 1, n_samples):
            if keep[j] and np.linalg.norm(xi - x[j]) < float(delta):
                keep[j] = False
    return keep


# ---------------------------------------------------------------------
# PRECOMPUTE CLUSTER SWITCHING QUANTITIES
# ---------------------------------------------------------------------
def precompute_quantities(u0_list, v_list):
    """
    Build reduced switching constants for local models.
    """
    k_count = len(v_list)

    # Find a valid reference shape for fallback construction.
    ref_u0 = None
    ref_v = None
    for k in range(k_count):
        if u0_list[k] is not None and v_list[k] is not None:
            ref_u0 = np.asarray(u0_list[k], dtype=float).reshape(-1)
            ref_v = np.asarray(v_list[k], dtype=float)
            break

    if ref_u0 is None or ref_v is None:
        raise RuntimeError("No valid cluster basis/center found for switching precomputations.")

    n_full = int(ref_u0.size)

    d_const = np.zeros((k_count, k_count), dtype=float)
    g_list = np.empty((k_count, k_count), dtype=object)
    t_list = np.empty((k_count, k_count), dtype=object)
    h_list = np.empty((k_count, k_count), dtype=object)

    for k in range(k_count):
        u0_k = ref_u0 if u0_list[k] is None else np.asarray(u0_list[k], dtype=float).reshape(-1)
        v_k = (
            np.zeros((n_full, 1), dtype=float)
            if v_list[k] is None
            else np.asarray(v_list[k], dtype=float)
        )

        for l in range(k_count):
            u0_l = ref_u0 if u0_list[l] is None else np.asarray(u0_list[l], dtype=float).reshape(-1)
            diff = u0_k - u0_l
            d_const[k, l] = float(diff @ diff)
            g_list[k, l] = v_k.T @ diff

    for l in range(k_count):
        v_l = (
            np.zeros((n_full, 1), dtype=float)
            if v_list[l] is None
            else np.asarray(v_list[l], dtype=float)
        )
        u0_l = ref_u0 if u0_list[l] is None else np.asarray(u0_list[l], dtype=float).reshape(-1)

        for k in range(k_count):
            v_k = (
                np.zeros((n_full, 1), dtype=float)
                if v_list[k] is None
                else np.asarray(v_list[k], dtype=float)
            )
            u0_k = ref_u0 if u0_list[k] is None else np.asarray(u0_list[k], dtype=float).reshape(-1)

            t_list[l, k] = v_l.T @ v_k
            h_list[l, k] = v_l.T @ (u0_k - u0_l)

    return d_const, g_list, t_list, h_list


# ---------------------------------------------------------------------
# TRAIN ONE CLUSTER
# ---------------------------------------------------------------------
def train_rbf_for_cluster(
    k,
    q_k,
    n_primary=4,
    delta_duplicates=0.0,
    n_splits_max=5,
):
    """
    Train one local RBF map q_s(q_p) for cluster k.

    Uses grid search over (kernel, epsilon, lambda) with K-fold CV.
    """
    if q_k is None:
        print(f"[STAGE4]   Cluster {k}: Q_k is None. Skipping.")
        return {
            "has_rbf": False,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    q_k = np.asarray(q_k, dtype=float)
    r_k, m_k = q_k.shape

    if m_k < 3:
        print(f"[STAGE4]   Cluster {k}: only {m_k} snapshots. Skipping RBF.")
        return {
            "has_rbf": False,
            "n_primary": int(n_primary),
            "n_total": int(r_k),
            "n_secondary": 0,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    n_total_k = int(r_k)
    if n_total_k <= int(n_primary):
        print(
            f"[STAGE4]   Cluster {k}: r_k={r_k} <= n_primary={n_primary}. "
            "No secondary block, skipping RBF."
        )
        return {
            "has_rbf": False,
            "n_primary": int(n_primary),
            "n_total": n_total_k,
            "n_secondary": 0,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    n_secondary_k = int(n_total_k - int(n_primary))

    print(
        f"[STAGE4]   Cluster {k}: r_k={r_k}, M_k={m_k}, "
        f"n_primary={n_primary}, n_secondary={n_secondary_k}"
    )

    q_p = q_k[:n_primary, :]
    q_s = q_k[n_primary:n_total_k, :]

    x_raw = q_p.T
    y_raw = q_s.T

    mask = get_duplicate_mask(x_raw, delta_duplicates)
    n_removed = int(x_raw.shape[0] - np.sum(mask))
    if n_removed > 0:
        print(f"[STAGE4]   Cluster {k}: removed {n_removed} near-duplicates.")

    x = x_raw[mask, :]
    y = y_raw[mask, :]

    n_samples = int(x.shape[0])
    if n_samples < 3:
        print(f"[STAGE4]   Cluster {k}: only {n_samples} samples after filtering. Skipping RBF.")
        return {
            "has_rbf": False,
            "n_primary": int(n_primary),
            "n_total": n_total_k,
            "n_secondary": n_secondary_k,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    # Scale primary coordinates
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    x_norm = scaler.fit_transform(x)

    scale_vec = scaler.scale_.copy()
    min_vec = scaler.min_.copy()

    n_splits = min(int(n_splits_max), n_samples)
    if n_splits < 2:
        n_splits = 2

    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(x_norm))

    print(
        f"[STAGE4]   Cluster {k}: {n_splits}-fold CV, "
        f"{len(rbf_kernels)} kernels x {len(EPSILON_VALUES)} eps x {len(LAMBDA_VALUES)} lambdas"
    )

    best_score = np.inf
    best_kernel_name = None
    best_epsilon = None
    best_lambda = None

    for kernel_name, kernel_func in rbf_kernels.items():
        for eps in EPSILON_VALUES:
            for lam in LAMBDA_VALUES:
                fold_rpes = []

                for train_idx, val_idx in splits:
                    x_train = x_norm[train_idx]
                    y_train = y[train_idx]
                    x_val = x_norm[val_idx]
                    y_val = y[val_idx]

                    d_train = np.linalg.norm(
                        x_train[:, np.newaxis, :] - x_train[np.newaxis, :, :],
                        axis=2,
                    )
                    phi_train = kernel_func(d_train, float(eps))
                    phi_train_reg = phi_train + float(lam) * np.eye(phi_train.shape[0], dtype=float)

                    w_cv = svd_solve_full_rank(phi_train_reg, y_train, rcond=1e-12)

                    d_val = np.linalg.norm(
                        x_val[:, np.newaxis, :] - x_train[np.newaxis, :, :],
                        axis=2,
                    )
                    phi_val = kernel_func(d_val, float(eps))
                    y_pred = phi_val @ w_cv

                    denom = np.linalg.norm(y_val, ord="fro") + 1e-14
                    rpe = 100.0 * np.linalg.norm(y_val - y_pred, ord="fro") / denom
                    fold_rpes.append(float(rpe))

                if not fold_rpes:
                    continue

                avg_rpe = float(np.mean(fold_rpes))
                print(
                    f"[STAGE4]     kernel={kernel_name}, eps={eps:.3e}, lam={lam:.1e}, CV RPE={avg_rpe:.3f}%"
                )

                if avg_rpe < best_score:
                    best_score = avg_rpe
                    best_kernel_name = kernel_name
                    best_epsilon = float(eps)
                    best_lambda = float(lam)

    if best_kernel_name is None:
        print(f"[STAGE4]   Cluster {k}: no feasible RBF found.")
        return {
            "has_rbf": False,
            "n_primary": int(n_primary),
            "n_total": n_total_k,
            "n_secondary": n_secondary_k,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    print(
        f"[STAGE4]   Cluster {k}: best kernel={best_kernel_name}, "
        f"eps={best_epsilon:.3e}, lambda={best_lambda:.1e}, CV RPE={best_score:.3f}%"
    )

    # Final fit
    kernel_best = rbf_kernels[best_kernel_name]
    d_full = np.linalg.norm(x_norm[:, np.newaxis, :] - x_norm[np.newaxis, :, :], axis=2)
    phi_full = kernel_best(d_full, best_epsilon)
    phi_full_reg = phi_full + best_lambda * np.eye(phi_full.shape[0], dtype=float)

    w = svd_solve_full_rank(phi_full_reg, y, rcond=1e-12)

    y_fit = phi_full @ w
    train_rel = np.linalg.norm(y_fit - y, ord="fro") / (np.linalg.norm(y, ord="fro") + 1e-14)
    train_rel_pct = 100.0 * float(train_rel)

    print(f"[STAGE4]   Cluster {k}: training relative error in q_s = {train_rel_pct:.3f}%")

    return {
        "has_rbf": True,
        "n_primary": int(n_primary),
        "n_total": n_total_k,
        "n_secondary": n_secondary_k,
        "scaler": scaler,
        "scale": scale_vec,
        "min": min_vec,
        "W": w,
        "q_p_train": x_norm,
        "epsilon": best_epsilon,
        "lambda": best_lambda,
        "kernel_name": best_kernel_name,
        "best_cv_rpe_pct": float(best_score),
        "train_rel_qs": float(train_rel),
        "train_rel_qs_pct": float(train_rel_pct),
        "n_samples_used": int(n_samples),
        "n_removed_duplicates": int(n_removed),
    }


def plot_cluster_train_errors(models, out_png):
    errs = []
    for model in models:
        if not model.get("has_rbf", False):
            errs.append(np.nan)
        else:
            errs.append(float(model.get("train_rel_qs_pct", np.nan)))

    errs_arr = np.asarray(errs, dtype=float)
    xs = np.arange(errs_arr.size)

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(xs, np.nan_to_num(errs_arr, nan=0.0), color="#4e79a7")
    for i, val in enumerate(errs_arr):
        if np.isfinite(val):
            bars[i].set_color("#4e79a7")
        else:
            bars[i].set_color("#9d9d9d")
    plt.xlabel("Cluster index")
    plt.ylabel("Train error in q_s [%]")
    plt.title("Local POD-RBF Stage4 training error per cluster")
    plt.xticks(xs)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(
    q_file=os.path.join(script_dir, "local_rbf_q_per_cluster.npz"),
    model_pkl=os.path.join(script_dir, "local_pod_rbf_models.pkl"),
    switching_npz=os.path.join(script_dir, "local_pod_rbf_switching.npz"),
    all_in_one_npz=os.path.join(script_dir, "local_pod_rbf_all_offline.npz"),
    summary_file=os.path.join(script_dir, "stage4_training_summary.txt"),
    train_error_plot_file=os.path.join(script_dir, "stage4_cluster_qs_train_error.png"),
    n_primary=5,
    delta_duplicates=1e-3,
    n_splits_max=5,
):
    os.makedirs(script_dir, exist_ok=True)

    print("\n==============================================")
    print("   STAGE 4: LOCAL POD-RBF TRAINING           ")
    print("==============================================")
    print(f"[STAGE4] q_file={q_file}")
    print(f"[STAGE4] n_primary={n_primary}")

    if not os.path.exists(q_file):
        raise FileNotFoundError(f"Missing stage3 file: {q_file}")

    t0_total = time.time()
    data = np.load(q_file, allow_pickle=True)

    u0_list = list(data["u0_list"])
    v_list = list(data["V_list"])
    q_list = list(data["Q_list"])
    cluster_indices = list(data["cluster_indices"])
    r_list = np.asarray(data["r_list"], dtype=int)
    dt = float(data["dt"])
    num_steps = int(data["num_steps"])

    k_count = len(q_list)
    print(f"[STAGE4] Found K={k_count} clusters.")

    models = []
    elapsed_train = 0.0

    t0 = time.time()
    for k in range(k_count):
        print(f"\n[STAGE4] === Training cluster {k} ===")
        model_k = train_rbf_for_cluster(
            k=k,
            q_k=q_list[k],
            n_primary=n_primary,
            delta_duplicates=delta_duplicates,
            n_splits_max=n_splits_max,
        )
        models.append(model_k)
    elapsed_train = time.time() - t0

    print(f"\n[STAGE4] Finished training all clusters in {elapsed_train:.2f}s")

    # Save pickle (debug/inspection)
    with open(model_pkl, "wb") as file:
        pickle.dump(
            {
                "models": models,
                "n_primary": int(n_primary),
                "cluster_indices": cluster_indices,
            },
            file,
        )
    print(f"[STAGE4] Saved models PKL: {model_pkl}")

    # Switching precomputations
    d_const, g_list, _, _ = precompute_quantities(u0_list, v_list)

    # Build explicit object arrays to avoid broadcasting issues
    ref_u0 = None
    ref_v = None
    for k in range(k_count):
        if u0_list[k] is not None and v_list[k] is not None:
            ref_u0 = np.asarray(u0_list[k], dtype=float).reshape(-1)
            ref_v = np.asarray(v_list[k], dtype=float)
            break
    if ref_u0 is None or ref_v is None:
        raise RuntimeError("No valid cluster data found in stage3 input.")
    n_full = int(ref_u0.size)

    u0_arr = np.empty(k_count, dtype=object)
    v_arr = np.empty(k_count, dtype=object)
    clust_arr = np.empty(k_count, dtype=object)
    models_arr = np.empty(k_count, dtype=object)
    g_arr = np.empty((k_count, k_count), dtype=object)

    for k in range(k_count):
        u0_arr[k] = (
            ref_u0.copy()
            if u0_list[k] is None
            else np.asarray(u0_list[k], dtype=float).reshape(-1)
        )
        v_arr[k] = (
            np.zeros((n_full, 1), dtype=float)
            if v_list[k] is None
            else np.asarray(v_list[k], dtype=float)
        )
        clust_arr[k] = cluster_indices[k]
        models_arr[k] = models[k]

    for k in range(k_count):
        for l in range(k_count):
            g_arr[k, l] = g_list[k, l]

    np.savez(
        switching_npz,
        u0_list=u0_arr,
        V_list=v_arr,
        cluster_indices=clust_arr,
        n_primary=int(n_primary),
        d_const=np.asarray(d_const, dtype=float),
        g_list=g_arr,
    )
    print(f"[STAGE4] Saved switching NPZ: {switching_npz}")

    np.savez(
        all_in_one_npz,
        u0_list=u0_arr,
        V_list=v_arr,
        cluster_indices=clust_arr,
        n_primary=int(n_primary),
        models=models_arr,
        d_const=np.asarray(d_const, dtype=float),
        g_list=g_arr,
    )
    print(f"[STAGE4] Saved all-in-one NPZ: {all_in_one_npz}")

    plot_cluster_train_errors(models, train_error_plot_file)
    print(f"[STAGE4] Saved training-error plot: {train_error_plot_file}")

    elapsed_total = time.time() - t0_total

    has_rbf_flags = [bool(m.get("has_rbf", False)) for m in models]
    train_errs = [float(m.get("train_rel_qs_pct", np.nan)) for m in models]
    cv_errs = [float(m.get("best_cv_rpe_pct", np.nan)) for m in models]
    best_eps = [m.get("epsilon", None) for m in models]
    best_lam = [m.get("lambda", None) for m in models]
    best_kernel = [m.get("kernel_name", None) for m in models]

    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_local_pod_rbf_training_grid_search.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("q_file", q_file),
                    ("n_primary", int(n_primary)),
                    ("delta_duplicates", float(delta_duplicates)),
                    ("n_splits_max", int(n_splits_max)),
                    ("epsilon_grid_size", int(EPSILON_VALUES.size)),
                    ("lambda_grid_size", int(LAMBDA_VALUES.size)),
                    ("kernels", list(rbf_kernels.keys())),
                    ("dt", dt),
                    ("num_steps", num_steps),
                ],
            ),
            (
                "clusters",
                [
                    ("K", k_count),
                    ("r_list", r_list.tolist()),
                    ("has_rbf", has_rbf_flags),
                    ("best_kernel", best_kernel),
                    ("best_epsilon", best_eps),
                    ("best_lambda", best_lam),
                    ("cv_rpe_percent", cv_errs),
                    ("train_rpe_percent", train_errs),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("training", elapsed_train),
                    ("total", elapsed_total),
                ],
            ),
            (
                "outputs",
                [
                    ("models_pkl", model_pkl),
                    ("switching_npz", switching_npz),
                    ("all_in_one_npz", all_in_one_npz),
                    ("train_error_png", train_error_plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE4] Saved summary: {summary_file}")
    print("[STAGE4] Done.\n")


if __name__ == "__main__":
    main()
