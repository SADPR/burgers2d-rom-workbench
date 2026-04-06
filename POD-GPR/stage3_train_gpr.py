#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: TRAIN POD-GPR MAP

Inputs from previous stages (inside POD-GPR):
  - basis.npy
  - q_p.npy
  - q_s.npy

Outputs (inside POD-GPR/pod_gpr_model):
  - scaler.pkl
  - U_p.npy
  - U_s.npy
  - u_ref.npy
  - q_p_normalized.npy
  - q_s.npy
  - gpr_model.pkl
  - stage3_validation_relative_error.png
  - stage3_train_gpr_summary.txt
"""

import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF


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


def safe_rel_error_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.linalg.norm(y_true, ord="fro")
    if denom <= 0.0:
        return None
    return float(100.0 * np.linalg.norm(y_true - y_pred, ord="fro") / denom)


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
            raise ValueError(f"u_ref size mismatch: got {u_ref_vec.size}, expected {expected_size}.")
    else:
        u_ref_vec = np.zeros(expected_size, dtype=np.float64)
        u_ref_source = "zeros(off)"

    return bool(use_u_ref), u_ref_vec, u_ref_source


def build_kernel(kernel_name, constant_value, constant_bounds, length_scale, length_bounds):
    if kernel_name == "matern15":
        return ConstantKernel(
            constant_value=constant_value, constant_value_bounds=constant_bounds
        ) * Matern(length_scale=length_scale, length_scale_bounds=length_bounds, nu=1.5)
    if kernel_name == "rbf":
        return ConstantKernel(
            constant_value=constant_value, constant_value_bounds=constant_bounds
        ) * RBF(length_scale=length_scale, length_scale_bounds=length_bounds)
    raise ValueError("kernel_name must be one of: 'matern15', 'rbf'.")


def split_train_validation_indices(n_samples, validation_fraction, rng):
    indices = np.arange(n_samples, dtype=np.int64)
    if n_samples < 2 or validation_fraction <= 0.0:
        return indices, np.array([], dtype=np.int64)

    perm = rng.permutation(indices)
    n_val = int(np.floor(validation_fraction * n_samples))
    n_val = max(1, n_val)
    n_val = min(n_val, n_samples - 1)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def save_validation_plot(rel_err_per_sample_pct, out_path):
    rel_err_per_sample_pct = np.asarray(rel_err_per_sample_pct, dtype=np.float64).reshape(-1)
    if rel_err_per_sample_pct.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        np.arange(rel_err_per_sample_pct.size),
        rel_err_per_sample_pct,
        color="tab:red",
        linewidth=1.5,
        marker="o",
        markersize=3.0,
    )
    ax.set_xlabel("Validation sample index")
    ax.set_ylabel("Relative error [%]")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    q_p_file=os.path.join(script_dir, "q_p.npy"),
    q_s_file=os.path.join(script_dir, "q_s.npy"),
    stage2_metadata_file=os.path.join(script_dir, "stage2_projection_metadata.npz"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    model_dir=os.path.join(script_dir, "pod_gpr_model"),
    report_file=os.path.join(script_dir, "stage3_train_gpr_summary.txt"),
    validation_plot_file=os.path.join(script_dir, "stage3_validation_relative_error.png"),
    kernel_name="matern15",
    alpha=1e-8,
    length_scale=1.0,
    constant_value=1.0,
    length_scale_bounds=(1e-2, 5.0),
    constant_value_bounds=(1e-3, 1e3),
    optimize_hyperparameters=True,
    n_restarts_optimizer=1,
    normalize_y=False,
    duplicate_tol=1e-3,
    max_train_samples=None,
    validation_fraction=0.1,
    random_seed=42,
    uref_mode="auto",
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
        raise ValueError(f"q_p and q_s sample mismatch: {q_p.shape[1]} vs {q_s.shape[1]}.")

    n_primary = int(q_p.shape[0])
    n_secondary = int(q_s.shape[0])
    n_samples_raw = int(q_p.shape[1])

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

    print("\n====================================================")
    print("         STAGE 3: TRAIN POD-GPR")
    print("====================================================")
    print(f"[STAGE3] n_primary={n_primary}, n_secondary={n_secondary}, n_samples={n_samples_raw}")
    print(f"[STAGE3] kernel={kernel_name}, alpha={alpha:.3e}")
    print(
        f"[STAGE3] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref_vec):.3e}"
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_all = scaler.fit_transform(q_p.T)
    y_all = q_s.T

    if duplicate_tol is not None and float(duplicate_tol) > 0.0:
        mask = remove_near_duplicates(x_all, float(duplicate_tol))
    else:
        mask = np.ones(x_all.shape[0], dtype=bool)

    x_filtered = x_all[mask]
    y_filtered = y_all[mask]
    duplicates_removed = int(np.sum(~mask))

    rng = np.random.default_rng(int(random_seed))
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
        if max_train_samples < 2:
            raise ValueError("max_train_samples must be >= 2 when provided.")
    if max_train_samples is not None and x_filtered.shape[0] > max_train_samples:
        selected = np.sort(rng.choice(x_filtered.shape[0], size=max_train_samples, replace=False))
        x_used = x_filtered[selected]
        y_used = y_filtered[selected]
    else:
        x_used = x_filtered
        y_used = y_filtered

    if x_used.shape[0] < 2:
        raise RuntimeError(
            "Not enough samples available for GPR training. "
            "Decrease duplicate_tol or increase available snapshots."
        )

    train_idx, val_idx = split_train_validation_indices(
        n_samples=x_used.shape[0],
        validation_fraction=float(validation_fraction),
        rng=rng,
    )
    x_train = x_used[train_idx]
    y_train = y_used[train_idx]

    kernel = build_kernel(
        kernel_name=kernel_name,
        constant_value=float(constant_value),
        constant_bounds=constant_value_bounds,
        length_scale=float(length_scale),
        length_bounds=length_scale_bounds,
    )

    gpr_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(alpha),
        optimizer="fmin_l_bfgs_b" if optimize_hyperparameters else None,
        n_restarts_optimizer=int(n_restarts_optimizer) if optimize_hyperparameters else 0,
        normalize_y=bool(normalize_y),
        random_state=int(random_seed),
    )

    t0 = time.time()
    gpr_model.fit(x_train, y_train)
    elapsed_train = time.time() - t0

    train_pred = gpr_model.predict(x_train)
    train_rel_err = safe_rel_error_percent(y_train, train_pred)

    val_rel_err = None
    val_plot_saved = False
    val_rel_per_sample = np.array([], dtype=np.float64)
    if val_idx.size > 0:
        x_val = x_used[val_idx]
        y_val = y_used[val_idx]
        val_pred = gpr_model.predict(x_val)
        val_rel_err = safe_rel_error_percent(y_val, val_pred)

        denom = np.linalg.norm(y_val, axis=1)
        err = np.linalg.norm(y_val - val_pred, axis=1)
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        val_rel_per_sample = 100.0 * err / safe_denom
        val_plot_saved = save_validation_plot(val_rel_per_sample, validation_plot_file)

    print(f"[STAGE3] Removed near duplicates: {duplicates_removed}")
    print(f"[STAGE3] Samples used for training pipeline: {x_used.shape[0]}")
    print(f"[STAGE3] Train subset size: {x_train.shape[0]}")
    print(f"[STAGE3] Validation subset size: {val_idx.size}")
    print(f"[STAGE3] Training time: {elapsed_train:.3f} s")
    if train_rel_err is not None:
        print(f"[STAGE3] Train relative error: {train_rel_err:.4f}%")
    if val_rel_err is not None:
        print(f"[STAGE3] Validation relative error: {val_rel_err:.4f}%")

    u_p = basis[:, :n_primary]
    u_s = basis[:, n_primary:n_primary + n_secondary]

    np.save(os.path.join(model_dir, "U_p.npy"), u_p)
    np.save(os.path.join(model_dir, "U_s.npy"), u_s)
    np.save(os.path.join(model_dir, "u_ref.npy"), u_ref_vec)
    np.save(os.path.join(model_dir, "q_p_normalized.npy"), x_used.T)
    np.save(os.path.join(model_dir, "q_s.npy"), y_used.T)
    np.savez(
        os.path.join(model_dir, "stage3_training_metadata.npz"),
        train_indices=train_idx,
        validation_indices=val_idx,
        duplicates_removed=np.asarray(duplicates_removed, dtype=np.int64),
        samples_raw=np.asarray(n_samples_raw, dtype=np.int64),
        samples_after_filtering=np.asarray(x_filtered.shape[0], dtype=np.int64),
        samples_used=np.asarray(x_used.shape[0], dtype=np.int64),
    )

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    gpr_path = os.path.join(model_dir, "gpr_model.pkl")
    with open(gpr_path, "wb") as file:
        pickle.dump(gpr_model, file)

    learned_kernel_obj = getattr(gpr_model, "kernel_", getattr(gpr_model, "kernel", None))
    learned_kernel = str(learned_kernel_obj)
    analytic_jacobian_compatible = (
        hasattr(learned_kernel_obj, "k1")
        and hasattr(learned_kernel_obj, "k2")
        and isinstance(learned_kernel_obj.k1, ConstantKernel)
        and (
            (isinstance(learned_kernel_obj.k2, Matern) and float(learned_kernel_obj.k2.nu) == 1.5)
            or isinstance(learned_kernel_obj.k2, RBF)
        )
    )

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage3_train_gpr.py"),
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
                    ("kernel_name_requested", kernel_name),
                    ("alpha", alpha),
                    ("length_scale_initial", length_scale),
                    ("constant_value_initial", constant_value),
                    ("length_scale_bounds", list(length_scale_bounds)),
                    ("constant_value_bounds", list(constant_value_bounds)),
                    ("optimize_hyperparameters", optimize_hyperparameters),
                    ("n_restarts_optimizer", int(n_restarts_optimizer) if optimize_hyperparameters else 0),
                    ("normalize_y", normalize_y),
                    ("duplicate_tol", duplicate_tol),
                    ("max_train_samples", max_train_samples),
                    ("validation_fraction", validation_fraction),
                    ("random_seed", random_seed),
                    ("analytic_jacobian_compatible", analytic_jacobian_compatible),
                ],
            ),
            (
                "data_shapes",
                [
                    ("basis_shape", basis.shape),
                    ("q_p_shape", q_p.shape),
                    ("q_s_shape", q_s.shape),
                    ("x_filtered_shape", x_filtered.shape),
                    ("x_used_shape", x_used.shape),
                    ("x_train_shape", x_train.shape),
                    ("x_validation_shape", (val_idx.size, x_used.shape[1])),
                    ("gpr_alpha_shape", np.asarray(gpr_model.alpha_).shape),
                ],
            ),
            (
                "training_quality",
                [
                    ("train_relative_error_percent", train_rel_err),
                    ("validation_relative_error_percent", val_rel_err),
                    (
                        "validation_relative_error_percent_mean_per_sample",
                        float(np.mean(val_rel_per_sample)) if val_rel_per_sample.size > 0 else None,
                    ),
                    (
                        "validation_relative_error_percent_max_per_sample",
                        float(np.max(val_rel_per_sample)) if val_rel_per_sample.size > 0 else None,
                    ),
                    ("learned_kernel", learned_kernel),
                    ("training_time_seconds", elapsed_train),
                ],
            ),
            (
                "outputs",
                [
                    ("model_dir", model_dir),
                    ("gpr_model_pkl", gpr_path),
                    ("scaler_pkl", scaler_path),
                    ("U_p_npy", os.path.join(model_dir, "U_p.npy")),
                    ("U_s_npy", os.path.join(model_dir, "U_s.npy")),
                    ("u_ref_npy", os.path.join(model_dir, "u_ref.npy")),
                    ("q_p_normalized_npy", os.path.join(model_dir, "q_p_normalized.npy")),
                    ("q_s_npy", os.path.join(model_dir, "q_s.npy")),
                    ("training_metadata_npz", os.path.join(model_dir, "stage3_training_metadata.npz")),
                    ("validation_error_png", validation_plot_file if val_plot_saved else None),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )
    print(f"[STAGE3] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
