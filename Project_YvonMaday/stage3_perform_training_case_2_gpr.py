#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Case-2 GPR closure map: qN_s = N_gpr(mu1, mu2, t)."""

import argparse
import os
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel

try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset
try:
    from stage3_qn_utils import resolve_primary_modes
except ModuleNotFoundError:
    from .stage3_qn_utils import resolve_primary_modes
try:
    from stage3_perform_training_case_2_ann import load_prom_dataset_case2
except ModuleNotFoundError:
    from .stage3_perform_training_case_2_ann import load_prom_dataset_case2
try:
    from stage3_split_utils import split_indices_ecsw_param_time
except ModuleNotFoundError:
    from .stage3_split_utils import split_indices_ecsw_param_time
try:
    from gpr_map_common import (
        apply_scaler,
        build_kernel,
        fit_scaler_stats,
        invert_scaler,
        parse_bounds_csv,
        rel_frob_percent,
        remove_near_duplicates,
        serialize_gpr_model,
    )
except ModuleNotFoundError:
    from .gpr_map_common import (
        apply_scaler,
        build_kernel,
        fit_scaler_stats,
        invert_scaler,
        parse_bounds_csv,
        rel_frob_percent,
        remove_near_duplicates,
        serialize_gpr_model,
    )
try:
    from project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _subsample(idx: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count is None or int(max_count) <= 0 or idx.size <= int(max_count):
        return np.asarray(idx, dtype=np.int64)
    choose = rng.choice(idx, size=int(max_count), replace=False)
    choose.sort()
    return choose.astype(np.int64)


def main(argv=None):
    ensure_layout_dirs()

    parser = argparse.ArgumentParser(
        description="Train Case-2 GPR map qN_s=N_gpr(mu1,mu2,t)."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional explicit dataset directory containing per_mu/ and meta.npy.",
    )
    parser.add_argument("--primary-modes", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="case2_model_gpr.pt")
    parser.add_argument("--summary-name", type=str, default="case2_training_summary_gpr.txt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--val-snap-time-offset",
        type=int,
        default=1,
        help="ECSW-like split: minimum time column considered for validation (>=1).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=1200,
        help="Cap train samples used by GPR fit (full GPR is O(N^3)).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=4000,
        help="Optional cap for validation samples used for metrics.",
    )
    parser.add_argument("--duplicate-tol", type=float, default=0.0)
    parser.add_argument("--x-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")
    parser.add_argument("--y-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")

    parser.add_argument("--kernel-name", choices=("matern15", "rbf"), default="matern15")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-12,
        help="Numerical jitter added to K diagonal. Keep small when using WhiteKernel.",
    )
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--length-scale-bounds", type=str, default="1e-2,5.0")
    parser.add_argument("--constant-value", type=float, default=1.0)
    parser.add_argument("--constant-value-bounds", type=str, default="1e-3,1e3")
    parser.add_argument(
        "--use-white-kernel",
        action="store_true",
        default=True,
        help="Add WhiteKernel(noise_level) and learn observation noise.",
    )
    parser.add_argument(
        "--no-white-kernel",
        action="store_false",
        dest="use_white_kernel",
    )
    parser.add_argument("--white-noise-level", type=float, default=1e-6)
    parser.add_argument("--white-noise-bounds", type=str, default="1e-10,1e0")
    parser.add_argument(
        "--ard",
        action="store_true",
        default=True,
        help="Use ARD kernel length scales (independent scales for mu1, mu2, t).",
    )
    parser.add_argument(
        "--no-ard",
        action="store_false",
        dest="ard",
    )
    parser.add_argument("--optimize-hyperparameters", action="store_true", default=True)
    parser.add_argument("--no-optimize-hyperparameters", action="store_false", dest="optimize_hyperparameters")
    parser.add_argument("--n-restarts-optimizer", type=int, default=2)
    parser.add_argument("--normalize-y", action="store_true", default=False)
    args = parser.parse_args(argv)

    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    if not (0.0 < float(args.val_frac) < 0.5):
        raise ValueError("--val-frac must be in (0, 0.5).")
    if float(args.alpha) <= 0.0:
        raise ValueError("--alpha must be > 0.")
    if int(args.max_train_samples) < 2:
        raise ValueError("--max-train-samples must be >= 2.")
    if int(args.max_val_samples) < 1:
        raise ValueError("--max-val-samples must be >= 1.")

    dataset_backend = str(args.dataset_backend).strip().lower()
    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=args.dataset_ntot,
        expected_backend=dataset_backend,
        requested_dataset_dir=args.dataset_dir,
    )
    n_primary = resolve_primary_modes(args.primary_modes, dataset_meta, dataset_ntot)

    model_name = str(args.model_name).strip()
    if not model_name:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)

    summary_name = str(args.summary_name).strip() or "case2_training_summary_gpr.txt"
    summary_path = os.path.join(STAGE3_DIR, summary_name)

    print(f"[Case2-GPR] dataset_dir = {dataset_dir}")
    print(f"[Case2-GPR] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case2-GPR] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case2-GPR] primary_modes = {n_primary}")

    x_raw, y_raw = load_prom_dataset_case2(dataset_root, primary_modes=n_primary)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)

    n_samples, in_dim = x_raw.shape
    out_dim = y_raw.shape[1]
    if in_dim != 3:
        raise ValueError(f"Case-2 input dim must be 3, got {in_dim}.")

    print(f"[Case2-GPR] Loaded: M={n_samples}, in_dim={in_dim}, out_dim={out_dim}")

    tr_idx, va_idx, split_info = split_indices_ecsw_param_time(
        x_raw,
        val_frac=float(args.val_frac),
        seed=seed,
        snap_time_offset=int(args.val_snap_time_offset),
        ensure_mu_coverage=True,
    )
    print(
        "[Case2-GPR] split = ecsw_param_time_stratified "
        f"(mu_groups={split_info['num_mu_groups']}, "
        f"time_per_mu={split_info['num_time_per_mu']}, "
        f"val_requested={split_info['val_frac_requested']:.4f}, "
        f"val_actual={split_info['val_frac_actual']:.4f})"
    )

    tr_idx_fit = _subsample(tr_idx, int(args.max_train_samples), rng)
    va_idx_eval = _subsample(va_idx, int(args.max_val_samples), rng)

    x_tr_raw = x_raw[tr_idx_fit]
    y_tr_raw = y_raw[tr_idx_fit]
    x_va_raw = x_raw[va_idx_eval]
    y_va_raw = y_raw[va_idx_eval]

    x_stats = fit_scaler_stats(x_tr_raw, args.x_scaling)
    y_stats = fit_scaler_stats(y_tr_raw, args.y_scaling)

    x_tr = apply_scaler(x_tr_raw, x_stats)
    y_tr = apply_scaler(y_tr_raw, y_stats)
    x_va = apply_scaler(x_va_raw, x_stats)
    y_va = apply_scaler(y_va_raw, y_stats)

    keep = remove_near_duplicates(x_tr, float(args.duplicate_tol))
    removed = int(np.sum(~keep))
    x_tr = x_tr[keep]
    y_tr = y_tr[keep]

    if x_tr.shape[0] < 2:
        raise RuntimeError(
            "Not enough train samples after duplicate filtering. Lower --duplicate-tol or increase --max-train-samples."
        )

    length_bounds = parse_bounds_csv(args.length_scale_bounds, "length_scale_bounds")
    constant_bounds = parse_bounds_csv(args.constant_value_bounds, "constant_value_bounds")

    kernel = build_kernel(
        kernel_name=args.kernel_name,
        constant_value=float(args.constant_value),
        constant_bounds=constant_bounds,
        length_scale=float(args.length_scale),
        length_bounds=length_bounds,
        input_dim=int(in_dim),
        ard=bool(args.ard),
    )
    white_bounds = parse_bounds_csv(args.white_noise_bounds, "white_noise_bounds")
    if bool(args.use_white_kernel):
        kernel = kernel + WhiteKernel(
            noise_level=float(args.white_noise_level),
            noise_level_bounds=white_bounds,
        )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(args.alpha),
        optimizer="fmin_l_bfgs_b" if bool(args.optimize_hyperparameters) else None,
        n_restarts_optimizer=int(args.n_restarts_optimizer) if bool(args.optimize_hyperparameters) else 0,
        normalize_y=bool(args.normalize_y),
        random_state=seed,
    )

    print(f"[Case2-GPR] train_used = {x_tr.shape[0]} (removed_duplicates={removed})")
    print(f"[Case2-GPR] val_used = {x_va.shape[0]}")
    print(f"[Case2-GPR] kernel_init = {kernel}")

    t0 = time.time()
    gpr.fit(x_tr, y_tr)
    elapsed = time.time() - t0

    yhat_tr = np.asarray(gpr.predict(x_tr), dtype=np.float64)
    yhat_va = np.asarray(gpr.predict(x_va), dtype=np.float64)

    tr_rel_scaled = rel_frob_percent(y_tr, yhat_tr)
    va_rel_scaled = rel_frob_percent(y_va, yhat_va)
    tr_mse_scaled = float(np.mean((yhat_tr - y_tr) ** 2))
    va_mse_scaled = float(np.mean((yhat_va - y_va) ** 2))

    yhat_tr_raw = invert_scaler(yhat_tr, y_stats)
    yhat_va_raw = invert_scaler(yhat_va, y_stats)
    tr_rel_raw = rel_frob_percent(y_tr_raw, yhat_tr_raw)
    va_rel_raw = rel_frob_percent(y_va_raw, yhat_va_raw)
    tr_mse_raw = float(np.mean((yhat_tr_raw - y_tr_raw) ** 2))
    va_mse_raw = float(np.mean((yhat_va_raw - y_va_raw) ** 2))

    learned_kernel = str(getattr(gpr, "kernel_", gpr.kernel))

    ckpt = {
        "format": "gpr_map",
        "case": "case2",
        "mapping": "qN_s = N_gpr(mu1, mu2, t)",
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "primary_modes": int(n_primary),
        "secondary_modes": int(dataset_ntot - n_primary),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "n_s": int(out_dim),
        "seed": int(seed),
        "x_scaling": str(args.x_scaling),
        "y_scaling": str(args.y_scaling),
        "x_stats": x_stats,
        "y_stats": y_stats,
        "kernel_name": str(args.kernel_name),
        "kernel_ard": bool(args.ard),
        "kernel_init": str(kernel),
        "kernel_learned": learned_kernel,
        "alpha": float(args.alpha),
        "use_white_kernel": bool(args.use_white_kernel),
        "white_noise_level": float(args.white_noise_level),
        "white_noise_bounds": tuple(float(v) for v in white_bounds),
        "length_scale": float(args.length_scale),
        "length_scale_bounds": tuple(float(v) for v in length_bounds),
        "constant_value": float(args.constant_value),
        "constant_value_bounds": tuple(float(v) for v in constant_bounds),
        "optimize_hyperparameters": bool(args.optimize_hyperparameters),
        "n_restarts_optimizer": int(args.n_restarts_optimizer),
        "normalize_y": bool(args.normalize_y),
        "val_frac": float(args.val_frac),
        "val_split": str(split_info["split_mode"]),
        "val_split_snap_time_offset": int(split_info["snap_time_offset"]),
        "max_train_samples": int(args.max_train_samples),
        "max_val_samples": int(args.max_val_samples),
        "duplicate_tol": float(args.duplicate_tol),
        "n_samples_total": int(n_samples),
        "n_samples_train_split": int(tr_idx.size),
        "n_samples_train_fit": int(tr_idx_fit.size),
        "n_samples_train_after_duplicates": int(x_tr.shape[0]),
        "n_samples_val_split": int(va_idx.size),
        "n_samples_val_eval": int(va_idx_eval.size),
        "n_unique_mu_groups": int(split_info["num_mu_groups"]),
        "n_val_mu_groups": None,
        "n_time_per_mu": int(split_info["num_time_per_mu"]),
        "n_candidates_total": int(split_info["num_candidates_total"]),
        "n_selected_total": int(split_info["num_selected_total"]),
        "val_frac_actual": float(split_info["val_frac_actual"]),
        "train_rel_frob_percent_scaled": float(tr_rel_scaled),
        "val_rel_frob_percent_scaled": float(va_rel_scaled),
        "train_mse_scaled": float(tr_mse_scaled),
        "val_mse_scaled": float(va_mse_scaled),
        "train_rel_frob_percent": float(tr_rel_raw),
        "val_rel_frob_percent": float(va_rel_raw),
        "train_mse": float(tr_mse_raw),
        "val_mse": float(va_mse_raw),
        "elapsed_s": float(elapsed),
        "gpr_payload": serialize_gpr_model(gpr),
    }
    import torch

    torch.save(ckpt, model_path)

    write_kv_txt(
        summary_path,
        [
            ("model_name", model_name),
            ("model_path", model_path),
            ("dataset_root", dataset_root),
            ("dataset_dir", dataset_dir),
            ("dataset_ntot", int(dataset_ntot)),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("primary_modes", int(n_primary)),
            ("secondary_modes", int(dataset_ntot - n_primary)),
            ("samples_M", int(n_samples)),
            ("in_dim", int(in_dim)),
            ("n_s", int(out_dim)),
            ("kernel_init", str(kernel)),
            ("kernel_learned", learned_kernel),
            ("alpha", float(args.alpha)),
            ("use_white_kernel", bool(args.use_white_kernel)),
            ("white_noise_level", float(args.white_noise_level)),
            ("white_noise_bounds", tuple(float(v) for v in white_bounds)),
            ("train_rel_frob_percent", float(tr_rel_raw)),
            ("val_rel_frob_percent", float(va_rel_raw)),
            ("train_mse", float(tr_mse_raw)),
            ("val_mse", float(va_mse_raw)),
            ("train_used_after_duplicates", int(x_tr.shape[0])),
            ("val_used", int(x_va.shape[0])),
            ("elapsed_s", float(elapsed)),
        ],
    )

    print(f"[Case2-GPR] Training done in {elapsed:.2f}s")
    print(f"[Case2-GPR] learned_kernel = {learned_kernel}")
    print(f"[Case2-GPR] val_rel_frob_percent = {va_rel_raw:.4f}%")
    print(f"[Case2-GPR] Saved checkpoint: {model_path}")
    print(f"[Case2-GPR] Summary: {summary_path}")


if __name__ == "__main__":
    main()
