#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TRAIN LOCAL POD-RBF MODELS (BAYESIAN SEARCH)

Inputs (from stage3):
  - local_rbf_q_per_cluster.npz

Outputs (inside LocalPOD-RBF):
  - local_pod_rbf_models.pkl
  - local_pod_rbf_switching.npz
  - local_pod_rbf_all_offline.npz
  - stage4_cluster_qs_train_error_bayesian.png
  - stage4_training_bayesian_summary.txt
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

try:
    from skopt import gp_minimize
    from skopt.space import Real, Categorical
except ImportError as exc:
    gp_minimize = None
    Real = None
    Categorical = None
    SKOPT_IMPORT_ERROR = exc
else:
    SKOPT_IMPORT_ERROR = None


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
# RBF KERNELS
# ---------------------------------------------------------------------
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)


def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)


def matern32_rbf(r, epsilon):
    sqrt3 = np.sqrt(3.0)
    z = sqrt3 * epsilon * r
    return (1.0 + z) * np.exp(-z)


RBF_KERNELS = {
    "gaussian": gaussian_rbf,
    "imq": inverse_multiquadric_rbf,
    "matern32": matern32_rbf,
}


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


def pairwise_distances(a, b=None):
    a = np.asarray(a, dtype=np.float64)
    if b is None:
        b = a
    else:
        b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def cv_relative_error_percent(x, y, folds, kernel, epsilon, lambda_val):
    fold_errors = []

    for train_idx, val_idx in folds:
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        d_train = pairwise_distances(x_train)
        phi_train = kernel(d_train, epsilon)
        phi_train_reg = phi_train + float(lambda_val) * np.eye(phi_train.shape[0], dtype=np.float64)

        w_cv = svd_solve_full_rank(phi_train_reg, y_train, rcond=1e-12)

        d_val = pairwise_distances(x_val, x_train)
        phi_val = kernel(d_val, epsilon)
        y_pred = phi_val @ w_cv

        denom = np.linalg.norm(y_val, ord="fro")
        if denom <= 0.0:
            continue

        rel_err_pct = 100.0 * np.linalg.norm(y_val - y_pred, ord="fro") / denom
        fold_errors.append(rel_err_pct)

    if len(fold_errors) == 0:
        return np.inf

    return float(np.mean(fold_errors))


def _normalize_kernel_candidates(kernel_candidates):
    if isinstance(kernel_candidates, str):
        values = [kernel_candidates]
    else:
        try:
            values = list(kernel_candidates)
        except TypeError:
            values = [kernel_candidates]

    values = [str(v).strip().lower() for v in values if str(v).strip() != ""]
    if len(values) == 0:
        raise ValueError("kernel_candidates must contain at least one kernel name.")

    out = []
    for name in values:
        if name not in RBF_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{name}'. Available: {list(RBF_KERNELS.keys())}"
            )
        out.append(name)

    # Keep order, remove duplicates.
    return list(dict.fromkeys(out))


def resolve_positive_interval(values, value_range, name):
    """
    Return (vmin, vmax, is_fixed).

    - If `values` is provided:
      * size==1 => fixed
      * size>=2 => interval from min/max
    - Else use `value_range`.
    """
    if values is not None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} values are empty.")
        if np.any(arr <= 0.0):
            raise ValueError(f"All values in {name} must be positive.")
        arr = np.unique(arr)
        if arr.size == 1:
            val = float(arr[0])
            return val, val, True
        return float(np.min(arr)), float(np.max(arr)), False

    if value_range is None or len(value_range) != 2:
        raise ValueError(f"{name}_range must contain exactly 2 values.")

    vmin = float(value_range[0])
    vmax = float(value_range[1])
    if vmin <= 0.0 or vmax <= 0.0:
        raise ValueError(f"{name}_range must be strictly positive. Got {value_range}.")
    if vmin > vmax:
        raise ValueError(f"{name}_range must be increasing. Got {value_range}.")
    if np.isclose(vmin, vmax):
        return vmin, vmax, True

    return vmin, vmax, False


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
# TRAIN ONE CLUSTER (BAYESIAN SEARCH)
# ---------------------------------------------------------------------
def train_rbf_for_cluster_bayesian(
    k,
    q_k,
    n_primary=4,
    delta_duplicates=0.0,
    n_splits_max=5,
    kernel_candidates=("gaussian", "imq"),
    epsilon_values=None,
    lambda_values=None,
    epsilon_range=(0.1, 5.0),
    lambda_range=(1e-12, 1e-10),
    n_calls=30,
    n_random_starts=8,
    acq_func="EI",
    random_seed=42,
    verbose=True,
):
    """
    Train one local RBF map q_s(q_p) for cluster k.

    Uses Bayesian optimization over (kernel, epsilon, lambda) with K-fold CV.
    """
    if q_k is None:
        print(f"[STAGE4-BO]   Cluster {k}: Q_k is None. Skipping.")
        return {
            "has_rbf": False,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    kernel_candidates = _normalize_kernel_candidates(kernel_candidates)

    q_k = np.asarray(q_k, dtype=float)
    r_k, m_k = q_k.shape

    if m_k < 3:
        print(f"[STAGE4-BO]   Cluster {k}: only {m_k} snapshots. Skipping RBF.")
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
            f"[STAGE4-BO]   Cluster {k}: r_k={r_k} <= n_primary={n_primary}. "
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
        f"[STAGE4-BO]   Cluster {k}: r_k={r_k}, M_k={m_k}, "
        f"n_primary={n_primary}, n_secondary={n_secondary_k}"
    )

    q_p = q_k[:n_primary, :]
    q_s = q_k[n_primary:n_total_k, :]

    x_raw = q_p.T
    y_raw = q_s.T

    mask = get_duplicate_mask(x_raw, delta_duplicates)
    n_removed = int(x_raw.shape[0] - np.sum(mask))
    if n_removed > 0:
        print(f"[STAGE4-BO]   Cluster {k}: removed {n_removed} near-duplicates.")

    x = x_raw[mask, :]
    y = y_raw[mask, :]

    n_samples = int(x.shape[0])
    if n_samples < 3:
        print(
            f"[STAGE4-BO]   Cluster {k}: only {n_samples} samples after filtering. "
            "Skipping RBF."
        )
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

    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed)).split(x_norm))

    eps_min, eps_max, eps_fixed = resolve_positive_interval(
        values=epsilon_values,
        value_range=epsilon_range,
        name="epsilon",
    )
    lam_min, lam_max, lam_fixed = resolve_positive_interval(
        values=lambda_values,
        value_range=lambda_range,
        name="lambda",
    )

    print(
        f"[STAGE4-BO]   Cluster {k}: {n_splits}-fold CV, "
        f"kernels={kernel_candidates}, "
        f"epsilon={'fixed' if eps_fixed else 'range'}[{eps_min:.3e},{eps_max:.3e}], "
        f"lambda={'fixed' if lam_fixed else 'range'}[{lam_min:.1e},{lam_max:.1e}]"
    )

    dims = []
    dim_names = []

    fixed_kernel_name = None
    if len(kernel_candidates) == 1:
        fixed_kernel_name = kernel_candidates[0]
    else:
        dims.append(Categorical(list(kernel_candidates), name="kernel_name"))
        dim_names.append("kernel_name")

    fixed_epsilon = eps_min if eps_fixed else None
    if not eps_fixed:
        dims.append(Real(np.log(eps_min), np.log(eps_max), name="log_epsilon"))
        dim_names.append("log_epsilon")

    fixed_lambda = lam_min if lam_fixed else None
    if not lam_fixed:
        dims.append(Real(np.log(lam_min), np.log(lam_max), name="log_lambda"))
        dim_names.append("log_lambda")

    eval_history = []

    def evaluate_and_record(kernel_name, epsilon, lambda_val):
        cv_err = cv_relative_error_percent(
            x=x_norm,
            y=y,
            folds=splits,
            kernel=RBF_KERNELS[kernel_name],
            epsilon=float(epsilon),
            lambda_val=float(lambda_val),
        )

        entry = {
            "kernel": str(kernel_name),
            "epsilon": float(epsilon),
            "lambda": float(lambda_val),
            "cv_rel_err_pct": float(cv_err),
        }
        eval_history.append(entry)

        if np.isfinite(cv_err):
            print(
                f"[STAGE4-BO]     cluster={k}, kernel={kernel_name}, "
                f"eps={epsilon:.4e}, lambda={lambda_val:.4e}, CV RPE={cv_err:.4f}%"
            )
        else:
            print(
                f"[STAGE4-BO]     cluster={k}, kernel={kernel_name}, "
                f"eps={epsilon:.4e}, lambda={lambda_val:.4e}, CV RPE=inf"
            )

        return cv_err

    if len(dims) == 0:
        evaluate_and_record(
            kernel_name=fixed_kernel_name,
            epsilon=fixed_epsilon,
            lambda_val=fixed_lambda,
        )
        n_calls_effective = 1
        n_random_starts_effective = 0
    else:
        if gp_minimize is None:
            raise ImportError(
                "scikit-optimize is required for Bayesian Stage 4. "
                "Install it with: pip install scikit-optimize"
            ) from SKOPT_IMPORT_ERROR

        n_calls_effective = int(n_calls)
        if n_calls_effective < 1:
            raise ValueError("n_calls must be >= 1.")

        n_random_starts_effective = int(n_random_starts)
        if n_random_starts_effective < 0:
            raise ValueError("n_random_starts must be >= 0.")
        n_random_starts_effective = min(
            n_random_starts_effective,
            max(0, n_calls_effective - 1),
        )

        def objective(params):
            params_map = dict(zip(dim_names, params))

            kernel_name = fixed_kernel_name
            if kernel_name is None:
                kernel_name = params_map["kernel_name"]

            epsilon = fixed_epsilon
            if epsilon is None:
                epsilon = float(np.exp(params_map["log_epsilon"]))

            lambda_val = fixed_lambda
            if lambda_val is None:
                lambda_val = float(np.exp(params_map["log_lambda"]))

            cv_err = evaluate_and_record(
                kernel_name=kernel_name,
                epsilon=epsilon,
                lambda_val=lambda_val,
            )

            if np.isfinite(cv_err):
                return float(cv_err)
            return 1.0e12

        gp_minimize(
            objective,
            dimensions=dims,
            n_calls=n_calls_effective,
            n_initial_points=n_random_starts_effective,
            acq_func=acq_func,
            random_state=int(random_seed),
            verbose=bool(verbose),
        )

    valid_results = [r for r in eval_history if np.isfinite(r["cv_rel_err_pct"])]
    if len(valid_results) == 0:
        print(f"[STAGE4-BO]   Cluster {k}: no feasible RBF found.")
        return {
            "has_rbf": False,
            "n_primary": int(n_primary),
            "n_total": n_total_k,
            "n_secondary": n_secondary_k,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
            "n_samples_used": int(n_samples),
            "n_removed_duplicates": int(n_removed),
            "search_method": "bayesian_optimization",
            "search_evaluations": int(len(eval_history)),
        }

    valid_results.sort(key=lambda r: r["cv_rel_err_pct"])
    best = valid_results[0]

    best_kernel_name = best["kernel"]
    best_epsilon = float(best["epsilon"])
    best_lambda = float(best["lambda"])
    best_score = float(best["cv_rel_err_pct"])

    print(
        f"[STAGE4-BO]   Cluster {k}: best kernel={best_kernel_name}, "
        f"eps={best_epsilon:.3e}, lambda={best_lambda:.1e}, CV RPE={best_score:.3f}%"
    )

    # Final fit
    kernel_best = RBF_KERNELS[best_kernel_name]
    d_full = pairwise_distances(x_norm)
    phi_full = kernel_best(d_full, best_epsilon)
    phi_full_reg = phi_full + best_lambda * np.eye(phi_full.shape[0], dtype=float)

    w = svd_solve_full_rank(phi_full_reg, y, rcond=1e-12)

    y_fit = phi_full @ w
    train_rel = np.linalg.norm(y_fit - y, ord="fro") / (np.linalg.norm(y, ord="fro") + 1e-14)
    train_rel_pct = 100.0 * float(train_rel)

    print(f"[STAGE4-BO]   Cluster {k}: training relative error in q_s = {train_rel_pct:.3f}%")

    top5 = valid_results[:5]

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
        "search_method": "bayesian_optimization",
        "search_evaluations": int(len(eval_history)),
        "n_calls_effective": int(n_calls_effective),
        "n_random_starts_effective": int(n_random_starts_effective),
        "top5_candidates": top5,
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
    plt.title("Local POD-RBF Stage4 Bayesian training error per cluster")
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
    summary_file=os.path.join(script_dir, "stage4_training_bayesian_summary.txt"),
    train_error_plot_file=os.path.join(script_dir, "stage4_cluster_qs_train_error_bayesian.png"),
    n_primary=5,
    delta_duplicates=1e-3,
    n_splits_max=5,
    kernel_candidates=("gaussian", "imq"),
    epsilon_values=None,
    lambda_values=None,
    epsilon_range=(0.1, 5.0),
    lambda_range=(1e-12, 1e-10),
    n_calls=30,
    n_random_starts=8,
    acq_func="EI",
    random_seed=42,
    verbose=True,
):
    os.makedirs(script_dir, exist_ok=True)

    kernel_candidates = _normalize_kernel_candidates(kernel_candidates)

    print("\n==============================================")
    print("   STAGE 4: LOCAL POD-RBF TRAINING (BO)      ")
    print("==============================================")
    print(f"[STAGE4-BO] q_file={q_file}")
    print(f"[STAGE4-BO] n_primary={n_primary}")

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
    print(f"[STAGE4-BO] Found K={k_count} clusters.")

    models = []

    t0 = time.time()
    for k in range(k_count):
        print(f"\n[STAGE4-BO] === Training cluster {k} ===")
        model_k = train_rbf_for_cluster_bayesian(
            k=k,
            q_k=q_list[k],
            n_primary=n_primary,
            delta_duplicates=delta_duplicates,
            n_splits_max=n_splits_max,
            kernel_candidates=kernel_candidates,
            epsilon_values=epsilon_values,
            lambda_values=lambda_values,
            epsilon_range=epsilon_range,
            lambda_range=lambda_range,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            acq_func=acq_func,
            random_seed=int(random_seed) + int(k),
            verbose=verbose,
        )
        models.append(model_k)
    elapsed_train = time.time() - t0

    print(f"\n[STAGE4-BO] Finished training all clusters in {elapsed_train:.2f}s")

    # Save pickle (debug/inspection)
    with open(model_pkl, "wb") as file:
        pickle.dump(
            {
                "models": models,
                "n_primary": int(n_primary),
                "cluster_indices": cluster_indices,
                "search_method": "bayesian_optimization",
            },
            file,
        )
    print(f"[STAGE4-BO] Saved models PKL: {model_pkl}")

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
    print(f"[STAGE4-BO] Saved switching NPZ: {switching_npz}")

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
    print(f"[STAGE4-BO] Saved all-in-one NPZ: {all_in_one_npz}")

    plot_cluster_train_errors(models, train_error_plot_file)
    print(f"[STAGE4-BO] Saved training-error plot: {train_error_plot_file}")

    elapsed_total = time.time() - t0_total

    has_rbf_flags = [bool(m.get("has_rbf", False)) for m in models]
    train_errs = [float(m.get("train_rel_qs_pct", np.nan)) for m in models]
    cv_errs = [float(m.get("best_cv_rpe_pct", np.nan)) for m in models]
    best_eps = [m.get("epsilon", None) for m in models]
    best_lam = [m.get("lambda", None) for m in models]
    best_kernel = [m.get("kernel_name", None) for m in models]
    search_eval_counts = [int(m.get("search_evaluations", 0)) for m in models]

    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_local_pod_rbf_training_bayesian.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("q_file", q_file),
                    ("n_primary", int(n_primary)),
                    ("delta_duplicates", float(delta_duplicates)),
                    ("n_splits_max", int(n_splits_max)),
                    ("kernels", kernel_candidates),
                    (
                        "epsilon_values",
                        None if epsilon_values is None else np.asarray(epsilon_values, dtype=np.float64).tolist(),
                    ),
                    (
                        "lambda_values",
                        None if lambda_values is None else np.asarray(lambda_values, dtype=np.float64).tolist(),
                    ),
                    ("epsilon_range", [float(epsilon_range[0]), float(epsilon_range[1])]),
                    ("lambda_range", [float(lambda_range[0]), float(lambda_range[1])]),
                    ("n_calls", int(n_calls)),
                    ("n_random_starts", int(n_random_starts)),
                    ("acq_func", acq_func),
                    ("random_seed", int(random_seed)),
                    ("search_method", "bayesian_optimization"),
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
                    ("search_evaluations", search_eval_counts),
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
    print(f"[STAGE4-BO] Saved summary: {summary_file}")
    print("[STAGE4-BO] Done.\n")


if __name__ == "__main__":
    main()
