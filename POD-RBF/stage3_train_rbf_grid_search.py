#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: TRAIN POD-RBF MAP (GRID SEARCH)

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
  - stage3_train_rbf_summary.txt

This stage can be rerun multiple times with different grids without
recomputing stage1 or stage2.
"""

import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

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


def build_positive_grid(values, value_range, n_points, name):
    if values is not None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} grid is empty.")
    else:
        if value_range is None or len(value_range) != 2:
            raise ValueError(f"{name}_range must contain exactly 2 values.")
        vmin, vmax = float(value_range[0]), float(value_range[1])
        if vmin <= 0.0 or vmax <= 0.0 or vmin >= vmax:
            raise ValueError(
                f"{name}_range must be positive and strictly increasing. Got {value_range}."
            )
        arr = np.geomspace(vmin, vmax, int(n_points), dtype=np.float64)

    if np.any(arr <= 0.0):
        raise ValueError(f"All values in {name} grid must be positive.")

    arr = np.unique(arr)
    return arr


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
    report_file=os.path.join(script_dir, "stage3_train_rbf_summary.txt"),
    duplicate_tol=1e-3,
    kernel_candidates=("imq", "gaussian", "matern"),
    epsilon_values=None,
    lambda_values=None,
    epsilon_range=(0.5, 10.0),
    lambda_range=(1e-10, 1e-8),
    n_epsilon=10,
    n_lambda=2,
    uref_mode="auto",
    cv_folds=5,
    random_seed=42,
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

    kernel_candidates = tuple(kernel_candidates)
    for name in kernel_candidates:
        if name not in kernel_map:
            raise ValueError(
                f"Unsupported kernel '{name}'. Available: {list(kernel_map.keys())}"
            )

    eps_grid = build_positive_grid(
        values=epsilon_values,
        value_range=epsilon_range,
        n_points=n_epsilon,
        name="epsilon",
    )
    lam_grid = build_positive_grid(
        values=lambda_values,
        value_range=lambda_range,
        n_points=n_lambda,
        name="lambda",
    )

    print("\n====================================================")
    print("       STAGE 3: TRAIN POD-RBF (GRID SEARCH)")
    print("====================================================")
    print(f"[STAGE3] n_primary={n_primary}, n_secondary={n_secondary}, n_samples={n_samples}")
    print(f"[STAGE3] kernels={kernel_candidates}")
    print(f"[STAGE3] epsilon grid size={eps_grid.size}, lambda grid size={lam_grid.size}")
    print(
        f"[STAGE3] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
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
    print(f"[STAGE3] Removed near duplicates: {removed}")
    print(f"[STAGE3] Training samples after filtering: {x.shape[0]}")

    if x.shape[0] < 2:
        raise RuntimeError(
            "Not enough samples after duplicate filtering. "
            "Decrease duplicate_tol or regenerate projections."
        )

    n_splits = min(int(cv_folds), x.shape[0])
    if n_splits < 2:
        raise RuntimeError("Need at least 2 CV folds for grid search.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = list(kf.split(x))

    total_combinations = len(kernel_candidates) * eps_grid.size * lam_grid.size
    print(f"[STAGE3] Total combinations to evaluate: {total_combinations}")

    t0 = time.time()
    results = []

    for kernel_name in kernel_candidates:
        kernel = kernel_map[kernel_name]
        for epsilon in eps_grid:
            for lambda_val in lam_grid:
                cv_err = cv_relative_error_percent(
                    x=x,
                    y=y,
                    folds=folds,
                    kernel=kernel,
                    epsilon=float(epsilon),
                    lambda_val=float(lambda_val),
                )

                results.append(
                    {
                        "kernel": kernel_name,
                        "epsilon": float(epsilon),
                        "lambda": float(lambda_val),
                        "cv_rel_err_pct": float(cv_err),
                    }
                )

                if np.isfinite(cv_err):
                    print(
                        f"[STAGE3][GRID] kernel={kernel_name}, eps={epsilon:.4e}, "
                        f"lambda={lambda_val:.4e}, rel_err={cv_err:.4f}%"
                    )
                else:
                    print(
                        f"[STAGE3][GRID] kernel={kernel_name}, eps={epsilon:.4e}, "
                        f"lambda={lambda_val:.4e}, rel_err=inf (failed solve)"
                    )

    elapsed_search = time.time() - t0

    valid_results = [r for r in results if np.isfinite(r["cv_rel_err_pct"])]
    if len(valid_results) == 0:
        raise RuntimeError("All grid-search combinations failed (singular solves).")

    valid_results.sort(key=lambda r: r["cv_rel_err_pct"])
    best = valid_results[0]

    best_kernel_name = best["kernel"]
    best_epsilon = float(best["epsilon"])
    best_lambda = float(best["lambda"])
    best_cv_rel_err = float(best["cv_rel_err_pct"])

    print("[STAGE3] Grid search finished")
    print(f"[STAGE3] Best kernel: {best_kernel_name}")
    print(f"[STAGE3] Best epsilon: {best_epsilon:.6e}")
    print(f"[STAGE3] Best lambda: {best_lambda:.6e}")
    print(f"[STAGE3] Best CV relative error: {best_cv_rel_err:.4f}%")

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
                "search_method": "grid_search",
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
                    ("script", "stage3_train_rbf_grid_search.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("q_p_file", q_p_file),
                    ("q_s_file", q_s_file),
                    ("stage2_metadata_file", stage2_metadata_file if os.path.exists(stage2_metadata_file) else None),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref_vec))),
                    ("kernel_candidates", list(kernel_candidates)),
                    ("epsilon_grid", eps_grid.tolist()),
                    ("lambda_grid", lam_grid.tolist()),
                    ("duplicate_tol", duplicate_tol),
                    ("cv_folds", n_splits),
                    ("random_seed", random_seed),
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
                "grid_search",
                [
                    ("total_combinations", total_combinations),
                    ("valid_combinations", len(valid_results)),
                    ("failed_combinations", total_combinations - len(valid_results)),
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
    print(f"[STAGE3] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
