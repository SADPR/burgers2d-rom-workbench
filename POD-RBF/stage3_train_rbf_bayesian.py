#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: TRAIN POD-RBF MAP (BAYESIAN OPTIMIZATION)

Inputs from previous stages (inside POD-RBF):
  - basis.npy
  - q_p.npy
  - q_s.npy

Outputs (inside POD-RBF/pod_rbf_model):
  - scaler.pkl
  - U_p.npy
  - U_s.npy
  - u_ref.npy
  - q_p_normalized.npy
  - q_s.npy
  - rbf_weights.pkl
  - stage3_train_rbf_bayesian_summary.txt

This stage can be rerun multiple times with different Bayesian search
settings without recomputing stage1 or stage2.
"""

import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np
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
# PATHS / IMPORTS
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


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


def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)


def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)


def multiquadric_rbf(r, epsilon):
    return np.sqrt(1.0 + (epsilon * r) ** 2)


def matern32_rbf(r, epsilon):
    sqrt3 = np.sqrt(3.0)
    z = sqrt3 * epsilon * r
    return (1.0 + z) * np.exp(-z)


def pairwise_distances(a, b=None):
    a = np.asarray(a, dtype=np.float64)
    if b is None:
        b = a
    else:
        b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def remove_near_duplicates(x, tol):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and np.linalg.norm(x[i] - x[j]) < tol:
                keep[j] = False
    return keep


def resolve_u_ref(uref_mode, uref_file, stage2_metadata_file, expected_size):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    stage2_use_u_ref = None
    u_ref_vec = None
    u_ref_source = None

    if os.path.exists(stage2_metadata_file):
        meta = np.load(stage2_metadata_file, allow_pickle=True)
        if "use_u_ref" in meta.files:
            stage2_use_u_ref = bool(np.asarray(meta["use_u_ref"]).reshape(-1)[0])
        if "u_ref_used" in meta.files:
            candidate = np.asarray(meta["u_ref_used"], dtype=np.float64).reshape(-1)
            if candidate.size == expected_size:
                u_ref_vec = candidate
                u_ref_source = f"{stage2_metadata_file}:u_ref_used"

    if mode == "off":
        use_u_ref = False
    elif mode == "on":
        use_u_ref = True
    else:
        if stage2_use_u_ref is not None:
            use_u_ref = stage2_use_u_ref
        else:
            use_u_ref = (u_ref_vec is not None) or os.path.exists(uref_file)

    if use_u_ref:
        if u_ref_vec is None:
            if not os.path.exists(uref_file):
                raise FileNotFoundError(
                    "u_ref is required by current settings but file is missing: "
                    f"{uref_file}"
                )
            u_ref_vec = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
            u_ref_source = uref_file
        if u_ref_vec.size != expected_size:
            raise ValueError(
                f"u_ref size mismatch: got {u_ref_vec.size}, expected {expected_size}."
            )
    else:
        u_ref_vec = np.zeros(expected_size, dtype=np.float64)
        u_ref_source = "zeros(off)"

    return bool(use_u_ref), u_ref_vec, u_ref_source


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


def cv_relative_error_percent(x, y, folds, kernel, epsilon, lambda_val):
    fold_errors = []

    for train_idx, val_idx in folds:
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        d_train = pairwise_distances(x_train)
        phi_train = kernel(d_train, epsilon)
        phi_train += np.eye(phi_train.shape[0], dtype=np.float64) * lambda_val

        try:
            w_cv = np.linalg.solve(phi_train, y_train)
        except np.linalg.LinAlgError:
            return np.inf

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


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    q_p_file=os.path.join(script_dir, "q_p.npy"),
    q_s_file=os.path.join(script_dir, "q_s.npy"),
    stage2_metadata_file=os.path.join(script_dir, "stage2_projection_metadata.npz"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    model_dir=os.path.join(script_dir, "pod_rbf_model"),
    report_file=os.path.join(script_dir, "stage3_train_rbf_bayesian_summary.txt"),
    duplicate_tol=1e-3,
    kernel_candidates=("imq", "gaussian", "matern"),
    epsilon_values=None,
    lambda_values=None,
    epsilon_range=(0.5, 10.0),
    lambda_range=(1e-10, 1e-8),
    n_calls=40,
    n_random_starts=10,
    acq_func="EI",
    uref_mode="auto",
    cv_folds=5,
    random_seed=42,
    verbose=True,
):
    for path in (basis_file, q_p_file, q_s_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}. Run stage1/stage2 first.")

    os.makedirs(model_dir, exist_ok=True)

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    q_p = np.asarray(np.load(q_p_file, allow_pickle=False), dtype=np.float64)
    q_s = np.asarray(np.load(q_s_file, allow_pickle=False), dtype=np.float64)

    if q_p.ndim != 2 or q_s.ndim != 2:
        raise ValueError("q_p and q_s must be 2D arrays.")
    if q_p.shape[1] != q_s.shape[1]:
        raise ValueError(
            f"q_p and q_s sample mismatch: {q_p.shape[1]} vs {q_s.shape[1]}."
        )

    n_primary = int(q_p.shape[0])
    n_secondary = int(q_s.shape[0])
    n_samples = int(q_p.shape[1])

    if n_secondary < 1:
        raise ValueError("q_s has zero rows. Increase total_modes in stage2.")
    if basis.shape[1] < n_primary + n_secondary:
        raise ValueError(
            "Basis has insufficient columns for (q_p, q_s): "
            f"basis columns={basis.shape[1]}, required={n_primary + n_secondary}."
        )

    use_u_ref, u_ref_vec, u_ref_source = resolve_u_ref(
        uref_mode=uref_mode,
        uref_file=uref_file,
        stage2_metadata_file=stage2_metadata_file,
        expected_size=basis.shape[0],
    )

    kernel_map = {
        "gaussian": gaussian_rbf,
        "imq": inverse_multiquadric_rbf,
        "multiquadric": multiquadric_rbf,
        "matern": matern32_rbf,
    }

    if isinstance(kernel_candidates, str):
        kernel_candidates = (kernel_candidates,)
    else:
        kernel_candidates = tuple(kernel_candidates)
    if len(kernel_candidates) == 0:
        raise ValueError("kernel_candidates must contain at least one kernel name.")

    for name in kernel_candidates:
        if name not in kernel_map:
            raise ValueError(
                f"Unsupported kernel '{name}'. Available: {list(kernel_map.keys())}"
            )

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

    print("\n====================================================")
    print("   STAGE 3: TRAIN POD-RBF (BAYESIAN OPTIMIZATION)")
    print("====================================================")
    print(f"[STAGE3-BO] n_primary={n_primary}, n_secondary={n_secondary}, n_samples={n_samples}")
    print(f"[STAGE3-BO] kernels={kernel_candidates}")
    print(
        f"[STAGE3-BO] epsilon: {'fixed' if eps_fixed else 'range'} "
        f"[{eps_min:.4e}, {eps_max:.4e}]"
    )
    print(
        f"[STAGE3-BO] lambda: {'fixed' if lam_fixed else 'range'} "
        f"[{lam_min:.4e}, {lam_max:.4e}]"
    )
    print(
        f"[STAGE3-BO] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref_vec):.3e}"
    )

    # Normalize primary coordinates to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_all = scaler.fit_transform(q_p.T)
    y_all = q_s.T

    # Remove near-duplicate points in normalized primary space
    dup_mask = remove_near_duplicates(x_all, duplicate_tol)
    x = x_all[dup_mask]
    y = y_all[dup_mask]

    removed = int(np.sum(~dup_mask))
    print(f"[STAGE3-BO] Removed near duplicates: {removed}")
    print(f"[STAGE3-BO] Training samples after filtering: {x.shape[0]}")

    if x.shape[0] < 2:
        raise RuntimeError(
            "Not enough samples after duplicate filtering. "
            "Decrease duplicate_tol or regenerate projections."
        )

    n_splits = min(int(cv_folds), x.shape[0])
    if n_splits < 2:
        raise RuntimeError("Need at least 2 CV folds for Bayesian search.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = list(kf.split(x))

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
            x=x,
            y=y,
            folds=folds,
            kernel=kernel_map[kernel_name],
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
                f"[STAGE3-BO][EVAL] kernel={kernel_name}, eps={epsilon:.4e}, "
                f"lambda={lambda_val:.4e}, rel_err={cv_err:.4f}%"
            )
        else:
            print(
                f"[STAGE3-BO][EVAL] kernel={kernel_name}, eps={epsilon:.4e}, "
                f"lambda={lambda_val:.4e}, rel_err=inf (failed solve)"
            )

        return cv_err

    t0 = time.time()

    if len(dims) == 0:
        # Fully fixed hyperparameters.
        evaluate_and_record(
            kernel_name=fixed_kernel_name,
            epsilon=fixed_epsilon,
            lambda_val=fixed_lambda,
        )
        n_calls_effective = 1
        n_initial_points_effective = 0
    else:
        if gp_minimize is None:
            raise ImportError(
                "scikit-optimize is required for Bayesian Stage 3. "
                "Install it with: pip install scikit-optimize"
            ) from SKOPT_IMPORT_ERROR

        n_calls_effective = int(n_calls)
        if n_calls_effective < 1:
            raise ValueError("n_calls must be >= 1.")

        n_initial_points_effective = int(n_random_starts)
        if n_initial_points_effective < 0:
            raise ValueError("n_random_starts must be >= 0.")
        n_initial_points_effective = min(
            n_initial_points_effective,
            max(0, n_calls_effective - 1),
        )

        def objective(params):
            params_map = dict(zip(dim_names, params))

            if fixed_kernel_name is None:
                kernel_name = params_map["kernel_name"]
            else:
                kernel_name = fixed_kernel_name

            if fixed_epsilon is None:
                epsilon = float(np.exp(params_map["log_epsilon"]))
            else:
                epsilon = float(fixed_epsilon)

            if fixed_lambda is None:
                lambda_val = float(np.exp(params_map["log_lambda"]))
            else:
                lambda_val = float(fixed_lambda)

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
            n_initial_points=n_initial_points_effective,
            acq_func=acq_func,
            random_state=random_seed,
            verbose=bool(verbose),
        )

    elapsed_search = time.time() - t0

    valid_results = [r for r in eval_history if np.isfinite(r["cv_rel_err_pct"])]
    if len(valid_results) == 0:
        raise RuntimeError("All Bayesian evaluations failed (singular solves).")

    valid_results.sort(key=lambda r: r["cv_rel_err_pct"])
    best = valid_results[0]

    best_kernel_name = best["kernel"]
    best_epsilon = float(best["epsilon"])
    best_lambda = float(best["lambda"])
    best_cv_rel_err = float(best["cv_rel_err_pct"])

    print("[STAGE3-BO] Bayesian search finished")
    print(f"[STAGE3-BO] Best kernel: {best_kernel_name}")
    print(f"[STAGE3-BO] Best epsilon: {best_epsilon:.6e}")
    print(f"[STAGE3-BO] Best lambda: {best_lambda:.6e}")
    print(f"[STAGE3-BO] Best CV relative error: {best_cv_rel_err:.4f}%")

    # Final training with all filtered data
    kernel = kernel_map[best_kernel_name]
    d_all = pairwise_distances(x)
    phi_all = kernel(d_all, best_epsilon)
    phi_all += np.eye(phi_all.shape[0], dtype=np.float64) * best_lambda

    try:
        w_final = np.linalg.solve(phi_all, y)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(phi_all, y, rcond=None)[0]

    # Save model artifacts
    u_p = basis[:, :n_primary]
    u_s = basis[:, n_primary:n_primary + n_secondary]

    np.save(os.path.join(model_dir, "U_p.npy"), u_p)
    np.save(os.path.join(model_dir, "U_s.npy"), u_s)
    np.save(os.path.join(model_dir, "u_ref.npy"), u_ref_vec)
    np.save(os.path.join(model_dir, "q_p_normalized.npy"), x.T)
    np.save(os.path.join(model_dir, "q_s.npy"), y.T)

    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)

    weights_path = os.path.join(model_dir, "rbf_weights.pkl")
    with open(weights_path, "wb") as file:
        pickle.dump(
            {
                "W": w_final,
                "q_p_train": x,
                "q_s_train": y,
                "epsilon": best_epsilon,
                "kernel_name": best_kernel_name,
                "lambda": best_lambda,
                "primary_modes": n_primary,
                "secondary_modes": n_secondary,
                "duplicate_tol": duplicate_tol,
                "use_u_ref": bool(use_u_ref),
                "u_ref_l2_norm": float(np.linalg.norm(u_ref_vec)),
                "u_ref_source": u_ref_source,
                "search_method": "bayesian_optimization",
            },
            file,
        )

    top5 = valid_results[:5]
    top5_summary = [
        {
            "rank": i + 1,
            "kernel": r["kernel"],
            "epsilon": r["epsilon"],
            "lambda": r["lambda"],
            "cv_rel_err_pct": r["cv_rel_err_pct"],
        }
        for i, r in enumerate(top5)
    ]

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage3_train_rbf_bayesian.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("q_p_file", q_p_file),
                    ("q_s_file", q_s_file),
                    (
                        "stage2_metadata_file",
                        stage2_metadata_file if os.path.exists(stage2_metadata_file) else None,
                    ),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref_vec))),
                    ("kernel_candidates", list(kernel_candidates)),
                    ("epsilon_values", None if epsilon_values is None else np.asarray(epsilon_values, dtype=np.float64).tolist()),
                    ("lambda_values", None if lambda_values is None else np.asarray(lambda_values, dtype=np.float64).tolist()),
                    ("epsilon_range", [eps_min, eps_max]),
                    ("lambda_range", [lam_min, lam_max]),
                    ("epsilon_fixed", eps_fixed),
                    ("lambda_fixed", lam_fixed),
                    ("duplicate_tol", duplicate_tol),
                    ("cv_folds", n_splits),
                    ("random_seed", random_seed),
                    ("acq_func", acq_func),
                    ("n_calls", n_calls_effective),
                    ("n_random_starts", n_initial_points_effective),
                ],
            ),
            (
                "data_shapes",
                [
                    ("basis_shape", basis.shape),
                    ("q_p_shape", q_p.shape),
                    ("q_s_shape", q_s.shape),
                    ("x_train_filtered_shape", x.shape),
                    ("y_train_filtered_shape", y.shape),
                    ("W_shape", w_final.shape),
                ],
            ),
            (
                "bayesian_search",
                [
                    ("evaluated_combinations", len(eval_history)),
                    ("valid_combinations", len(valid_results)),
                    ("failed_combinations", len(eval_history) - len(valid_results)),
                    ("best_kernel", best_kernel_name),
                    ("best_epsilon", best_epsilon),
                    ("best_lambda", best_lambda),
                    ("best_cv_relative_error_percent", best_cv_rel_err),
                    ("top5", top5_summary),
                    ("search_time_seconds", elapsed_search),
                ],
            ),
            (
                "outputs",
                [
                    ("model_dir", model_dir),
                    ("scaler_pkl", os.path.join(model_dir, "scaler.pkl")),
                    ("U_p_npy", os.path.join(model_dir, "U_p.npy")),
                    ("U_s_npy", os.path.join(model_dir, "U_s.npy")),
                    ("u_ref_npy", os.path.join(model_dir, "u_ref.npy")),
                    ("q_p_normalized_npy", os.path.join(model_dir, "q_p_normalized.npy")),
                    ("q_s_npy", os.path.join(model_dir, "q_s.npy")),
                    ("weights_pkl", weights_path),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )

    print(f"[STAGE3-BO] Saved model to: {model_dir}")
    print(f"[STAGE3-BO] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
