#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TRAIN LOCAL POD-GPR MODELS

Inputs (from stage3):
  - local_gpr_q_per_cluster.npz

Outputs (inside LocalPOD-GPR):
  - local_pod_gpr_models.pkl
  - local_pod_gpr_switching.npz
  - local_pod_gpr_all_offline.npz
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

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, Product


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
# NUMERICAL HELPERS
# ---------------------------------------------------------------------
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


def _build_kernel(kernel_name, constant_value, constant_bounds, length_scale, length_bounds):
    if kernel_name == "matern15":
        return ConstantKernel(
            constant_value=float(constant_value),
            constant_value_bounds=constant_bounds,
        ) * Matern(
            length_scale=float(length_scale),
            length_scale_bounds=length_bounds,
            nu=1.5,
        )
    if kernel_name == "rbf":
        return ConstantKernel(
            constant_value=float(constant_value),
            constant_value_bounds=constant_bounds,
        ) * RBF(
            length_scale=float(length_scale),
            length_scale_bounds=length_bounds,
        )
    raise ValueError(f"Unsupported kernel_name='{kernel_name}'.")


def _is_analytic_jacobian_compatible(kernel_obj):
    if not isinstance(kernel_obj, Product):
        return False
    if not isinstance(kernel_obj.k1, ConstantKernel):
        return False
    if isinstance(kernel_obj.k2, Matern):
        return float(kernel_obj.k2.nu) == 1.5
    if isinstance(kernel_obj.k2, RBF):
        return True
    return False


def _safe_relative_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.linalg.norm(y_true, ord="fro")
    if denom <= 0.0:
        return np.inf
    return 100.0 * float(np.linalg.norm(y_true - y_pred, ord="fro") / denom)


def _normalize_kernel_candidates(kernel_candidates):
    if isinstance(kernel_candidates, str):
        values = [kernel_candidates]
    else:
        try:
            values = list(kernel_candidates)
        except TypeError:
            values = [kernel_candidates]

    values = [str(v).strip() for v in values if str(v).strip() != ""]
    if len(values) == 0:
        raise ValueError("kernel_candidates must contain at least one kernel name.")
    return values


def _normalize_alpha_values(alpha_values):
    if np.isscalar(alpha_values):
        values = [alpha_values]
    else:
        try:
            values = list(alpha_values)
        except TypeError:
            values = [alpha_values]

    out = []
    for v in values:
        a = float(v)
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"alpha values must be positive finite numbers. Got {v}.")
        out.append(a)

    if len(out) == 0:
        raise ValueError("alpha_values must contain at least one value.")
    return out


# ---------------------------------------------------------------------
# PRECOMPUTE CLUSTER SWITCHING QUANTITIES
# ---------------------------------------------------------------------
def precompute_quantities(u0_list, v_list):
    """
    Build reduced switching constants for local models.
    """
    k_count = len(v_list)

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
def train_gpr_for_cluster(
    k,
    q_k,
    n_primary=5,
    delta_duplicates=0.0,
    n_splits_max=5,
    kernel_candidates=("matern15", "rbf"),
    alpha_values=(1e-10, 1e-8, 1e-6),
    constant_value=1.0,
    length_scale=1.0,
    constant_value_bounds=(1e-3, 1e3),
    length_scale_bounds=(1e-2, 5.0),
    optimize_hyperparameters=True,
    n_restarts_optimizer=1,
    normalize_y=False,
    random_seed=42,
):
    """
    Train one local GPR map q_s(q_p) for cluster k.

    Uses grid search over (kernel_name, alpha) via K-fold CV,
    then fits final GPR on all filtered data.
    """
    if q_k is None:
        print(f"[STAGE4]   Cluster {k}: Q_k is None. Skipping.")
        return {
            "has_gpr": False,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    kernel_candidates = _normalize_kernel_candidates(kernel_candidates)
    alpha_values = _normalize_alpha_values(alpha_values)

    q_k = np.asarray(q_k, dtype=float)
    r_k, m_k = q_k.shape

    if m_k < 3:
        print(f"[STAGE4]   Cluster {k}: only {m_k} snapshots. Skipping GPR.")
        return {
            "has_gpr": False,
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
            "No secondary block, skipping GPR."
        )
        return {
            "has_gpr": False,
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
        print(f"[STAGE4]   Cluster {k}: only {n_samples} samples after filtering. Skipping GPR.")
        return {
            "has_gpr": False,
            "n_primary": int(n_primary),
            "n_total": n_total_k,
            "n_secondary": n_secondary_k,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    # Scale primary coordinates
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    x_norm = scaler.fit_transform(x)

    n_splits = min(int(n_splits_max), n_samples)
    if n_splits < 2:
        n_splits = 2

    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed)).split(x_norm))

    print(
        f"[STAGE4]   Cluster {k}: {n_splits}-fold CV, "
        f"{len(kernel_candidates)} kernels x {len(alpha_values)} alphas"
    )

    best_score = np.inf
    best_kernel_name = None
    best_alpha = None

    for kernel_name in kernel_candidates:
        for alpha in alpha_values:
            fold_rpes = []
            failed = False

            for train_idx, val_idx in splits:
                x_train = x_norm[train_idx]
                y_train = y[train_idx]
                x_val = x_norm[val_idx]
                y_val = y[val_idx]

                try:
                    kernel = _build_kernel(
                        kernel_name=kernel_name,
                        constant_value=constant_value,
                        constant_bounds=constant_value_bounds,
                        length_scale=length_scale,
                        length_bounds=length_scale_bounds,
                    )
                    # Keep CV lightweight and robust.
                    gpr_cv = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=float(alpha),
                        optimizer=None,
                        normalize_y=bool(normalize_y),
                        random_state=int(random_seed),
                    )
                    gpr_cv.fit(x_train, y_train)
                    y_pred = gpr_cv.predict(x_val)
                    rpe = _safe_relative_percent(y_val, y_pred)
                    fold_rpes.append(float(rpe))
                except Exception:
                    failed = True
                    break

            if failed or not fold_rpes:
                avg_rpe = np.inf
            else:
                avg_rpe = float(np.mean(fold_rpes))

            if np.isfinite(avg_rpe):
                print(
                    f"[STAGE4]     kernel={kernel_name}, alpha={alpha:.1e}, CV RPE={avg_rpe:.3f}%"
                )
            else:
                print(
                    f"[STAGE4]     kernel={kernel_name}, alpha={alpha:.1e}, CV RPE=inf (failed)"
                )

            if avg_rpe < best_score:
                best_score = avg_rpe
                best_kernel_name = str(kernel_name)
                best_alpha = float(alpha)

    if best_kernel_name is None:
        print(f"[STAGE4]   Cluster {k}: no feasible GPR found.")
        return {
            "has_gpr": False,
            "n_primary": int(n_primary),
            "n_total": n_total_k,
            "n_secondary": n_secondary_k,
            "best_cv_rpe_pct": np.nan,
            "train_rel_qs_pct": np.nan,
        }

    print(
        f"[STAGE4]   Cluster {k}: best kernel={best_kernel_name}, "
        f"alpha={best_alpha:.1e}, CV RPE={best_score:.3f}%"
    )

    # Final fit on all filtered data
    final_kernel = _build_kernel(
        kernel_name=best_kernel_name,
        constant_value=constant_value,
        constant_bounds=constant_value_bounds,
        length_scale=length_scale,
        length_bounds=length_scale_bounds,
    )

    gpr_model = GaussianProcessRegressor(
        kernel=final_kernel,
        alpha=float(best_alpha),
        optimizer="fmin_l_bfgs_b" if bool(optimize_hyperparameters) else None,
        n_restarts_optimizer=int(n_restarts_optimizer) if bool(optimize_hyperparameters) else 0,
        normalize_y=bool(normalize_y),
        random_state=int(random_seed),
    )
    gpr_model.fit(x_norm, y)

    y_fit = gpr_model.predict(x_norm)
    train_rel_pct = _safe_relative_percent(y, y_fit)

    learned_kernel_obj = getattr(gpr_model, "kernel_", getattr(gpr_model, "kernel", None))
    learned_kernel = str(learned_kernel_obj)
    analytic_ok = _is_analytic_jacobian_compatible(learned_kernel_obj)

    print(f"[STAGE4]   Cluster {k}: training relative error in q_s = {train_rel_pct:.3f}%")

    return {
        "has_gpr": True,
        "n_primary": int(n_primary),
        "n_total": n_total_k,
        "n_secondary": n_secondary_k,
        "scaler": scaler,
        "gpr_model": gpr_model,
        "best_kernel_name": best_kernel_name,
        "best_alpha": best_alpha,
        "best_cv_rpe_pct": float(best_score),
        "train_rel_qs": float(train_rel_pct / 100.0),
        "train_rel_qs_pct": float(train_rel_pct),
        "n_samples_used": int(n_samples),
        "n_removed_duplicates": int(n_removed),
        "optimize_hyperparameters": bool(optimize_hyperparameters),
        "n_restarts_optimizer": int(n_restarts_optimizer) if bool(optimize_hyperparameters) else 0,
        "normalize_y": bool(normalize_y),
        "analytic_jacobian_compatible": bool(analytic_ok),
        "learned_kernel": learned_kernel,
    }


def plot_cluster_train_errors(models, out_png):
    errs = []
    for model in models:
        if not model.get("has_gpr", False):
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
    plt.title("Local POD-GPR Stage4 training error per cluster")
    plt.xticks(xs)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(
    q_file=os.path.join(script_dir, "local_gpr_q_per_cluster.npz"),
    model_pkl=os.path.join(script_dir, "local_pod_gpr_models.pkl"),
    switching_npz=os.path.join(script_dir, "local_pod_gpr_switching.npz"),
    all_in_one_npz=os.path.join(script_dir, "local_pod_gpr_all_offline.npz"),
    summary_file=os.path.join(script_dir, "stage4_training_summary.txt"),
    train_error_plot_file=os.path.join(script_dir, "stage4_cluster_qs_train_error.png"),
    n_primary=5,
    delta_duplicates=1e-3,
    n_splits_max=5,
    kernel_candidates=("matern15", "rbf"),
    alpha_values=(1e-10, 1e-8, 1e-6),
    constant_value=1.0,
    length_scale=1.0,
    constant_value_bounds=(1e-3, 1e3),
    length_scale_bounds=(1e-2, 5.0),
    optimize_hyperparameters=True,
    n_restarts_optimizer=1,
    normalize_y=False,
    random_seed=42,
):
    os.makedirs(script_dir, exist_ok=True)

    kernel_candidates = _normalize_kernel_candidates(kernel_candidates)
    alpha_values = _normalize_alpha_values(alpha_values)

    print("\n==============================================")
    print("   STAGE 4: LOCAL POD-GPR TRAINING           ")
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

    t0 = time.time()
    for k in range(k_count):
        print(f"\n[STAGE4] === Training cluster {k} ===")
        model_k = train_gpr_for_cluster(
            k=k,
            q_k=q_list[k],
            n_primary=n_primary,
            delta_duplicates=delta_duplicates,
            n_splits_max=n_splits_max,
            kernel_candidates=kernel_candidates,
            alpha_values=alpha_values,
            constant_value=constant_value,
            length_scale=length_scale,
            constant_value_bounds=constant_value_bounds,
            length_scale_bounds=length_scale_bounds,
            optimize_hyperparameters=optimize_hyperparameters,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_seed=random_seed,
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

    has_gpr_flags = [bool(m.get("has_gpr", False)) for m in models]
    train_errs = [float(m.get("train_rel_qs_pct", np.nan)) for m in models]
    cv_errs = [float(m.get("best_cv_rpe_pct", np.nan)) for m in models]
    best_alpha = [m.get("best_alpha", None) for m in models]
    best_kernel = [m.get("best_kernel_name", None) for m in models]
    analytic_ok = [m.get("analytic_jacobian_compatible", None) for m in models]

    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_local_pod_gpr_training.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("q_file", q_file),
                    ("n_primary", int(n_primary)),
                    ("delta_duplicates", float(delta_duplicates)),
                    ("n_splits_max", int(n_splits_max)),
                    ("kernel_candidates", list(kernel_candidates)),
                    ("alpha_values", list(alpha_values)),
                    ("constant_value", float(constant_value)),
                    ("length_scale", float(length_scale)),
                    ("constant_value_bounds", list(constant_value_bounds)),
                    ("length_scale_bounds", list(length_scale_bounds)),
                    ("optimize_hyperparameters", bool(optimize_hyperparameters)),
                    ("n_restarts_optimizer", int(n_restarts_optimizer) if bool(optimize_hyperparameters) else 0),
                    ("normalize_y", bool(normalize_y)),
                    ("random_seed", int(random_seed)),
                    ("dt", dt),
                    ("num_steps", num_steps),
                ],
            ),
            (
                "clusters",
                [
                    ("K", k_count),
                    ("r_list", r_list.tolist()),
                    ("has_gpr", has_gpr_flags),
                    ("best_kernel", best_kernel),
                    ("best_alpha", best_alpha),
                    ("cv_rpe_percent", cv_errs),
                    ("train_rpe_percent", train_errs),
                    ("analytic_jacobian_compatible", analytic_ok),
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
                    ("training_error_png", train_error_plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE4] Saved summary: {summary_file}")
    print("[STAGE4] Done.\n")


if __name__ == "__main__":
    main()
