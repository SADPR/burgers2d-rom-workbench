#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train full data-driven sparse GPR map: qN = G_sparse_gpr(mu1, mu2, t)."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset
try:
    from stage3_perform_training_rom_data_driven import load_prom_dataset_rom_data_driven
except ModuleNotFoundError:
    from .stage3_perform_training_rom_data_driven import load_prom_dataset_rom_data_driven
try:
    from stage3_split_utils import split_indices_ecsw_param_time
except ModuleNotFoundError:
    from .stage3_split_utils import split_indices_ecsw_param_time
try:
    from gpr_map_common import (
        apply_scaler,
        fit_scaler_stats,
        invert_scaler,
        rel_frob_percent,
        remove_near_duplicates,
    )
except ModuleNotFoundError:
    from .gpr_map_common import (
        apply_scaler,
        fit_scaler_stats,
        invert_scaler,
        rel_frob_percent,
        remove_near_duplicates,
    )
try:
    from stage3_sparse_gpr_common import (
        choose_inducing_points,
        fit_sparse_gp_output,
        predict_sparse_batch,
        resolve_device,
    )
except ModuleNotFoundError:
    from .stage3_sparse_gpr_common import (
        choose_inducing_points,
        fit_sparse_gp_output,
        predict_sparse_batch,
        resolve_device,
    )
try:
    from project_layout import STAGE3_DIR, ensure_layout_dirs, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE3_DIR, ensure_layout_dirs, write_kv_txt

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
        description="Train full data-driven sparse GPR map qN=G_sparse_gpr(mu1,mu2,t)."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional explicit dataset directory containing per_mu/ and meta.npy.",
    )
    parser.add_argument(
        "--validation-dataset-dir",
        type=str,
        default=None,
        help=(
            "Optional external validation dataset directory. If provided, the "
            "training dataset is used only for fitting and this dataset is used "
            "only for validation."
        ),
    )
    parser.add_argument(
        "--stage3-dir",
        type=str,
        default=None,
        help="Optional Stage3 output directory. Default: project_layout.STAGE3_DIR.",
    )
    parser.add_argument("--model-name", type=str, default="rom_data_driven_sparse_gpr_model.pt")
    parser.add_argument("--summary-name", type=str, default="rom_data_driven_sparse_gpr_training_summary.txt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--val-snap-time-offset",
        type=int,
        default=1,
        help="ECSW-like split: minimum time column considered for validation (>=1).",
    )
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--max-val-samples", type=int, default=4000)
    parser.add_argument("--duplicate-tol", type=float, default=0.0)
    parser.add_argument("--x-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")
    parser.add_argument("--y-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")

    parser.add_argument("--num-inducing", type=int, default=451)
    parser.add_argument("--inducing-selection", choices=("random", "kmeans"), default="kmeans")
    parser.add_argument("--kmeans-max-iters", type=int, default=40)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-fit-samples", type=int, default=40000)

    parser.add_argument("--kernel-name", choices=("rbf", "matern15"), default="rbf")
    parser.add_argument("--ard", action="store_true", default=True)
    parser.add_argument("--no-ard", action="store_false", dest="ard")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-noise", type=float, default=1e-6)
    parser.add_argument(
        "--max-noise",
        type=float,
        default=None,
        help="Optional upper bound for the learned Gaussian noise variance.",
    )
    parser.add_argument(
        "--elbo-beta",
        type=float,
        default=1.0,
        help="Weight applied to the KL term in the sparse-GP variational objective.",
    )
    parser.add_argument("--fixed-inducing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    external_validation = args.validation_dataset_dir is not None

    if (not external_validation) and not (0.0 < float(args.val_frac) < 0.5):
        raise ValueError("--val-frac must be in (0, 0.5).")
    if int(args.max_train_samples) < 2:
        raise ValueError("--max-train-samples must be >= 2.")
    if int(args.max_val_samples) < 1:
        raise ValueError("--max-val-samples must be >= 1.")
    if int(args.num_inducing) <= 0:
        raise ValueError("--num-inducing must be > 0.")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be > 0.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0.")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be > 0.")
    if float(args.min_noise) <= 0.0:
        raise ValueError("--min-noise must be > 0.")
    if args.max_noise is not None and float(args.max_noise) <= float(args.min_noise):
        raise ValueError("--max-noise must be larger than --min-noise.")
    if float(args.elbo_beta) <= 0.0:
        raise ValueError("--elbo-beta must be > 0.")

    dataset_backend = str(args.dataset_backend).strip().lower()
    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=args.dataset_ntot,
        expected_backend=dataset_backend,
        requested_dataset_dir=args.dataset_dir,
    )

    stage3_dir = (
        os.path.abspath(os.path.expanduser(str(args.stage3_dir)))
        if args.stage3_dir is not None
        else STAGE3_DIR
    )
    stage3_models_dir = os.path.join(stage3_dir, "models")
    os.makedirs(stage3_models_dir, exist_ok=True)

    model_name = str(args.model_name).strip()
    if not model_name:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = os.path.join(stage3_models_dir, model_name)

    summary_name = str(args.summary_name).strip() or "rom_data_driven_sparse_gpr_training_summary.txt"
    summary_path = os.path.join(stage3_dir, summary_name)

    print(f"[ROM-DataDriven-SparseGPR] dataset_dir = {dataset_dir}")
    print(f"[ROM-DataDriven-SparseGPR] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[ROM-DataDriven-SparseGPR] solve_backend = {dataset_meta.get('solve_backend')}")

    x_raw, y_raw = load_prom_dataset_rom_data_driven(dataset_root)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)

    n_samples, in_dim = x_raw.shape
    out_dim = y_raw.shape[1]
    if in_dim != 3:
        raise ValueError(f"Input dim must be 3 (mu1,mu2,t), got {in_dim}.")
    if int(out_dim) != int(dataset_ntot):
        raise RuntimeError(f"Output dim mismatch: out_dim={out_dim}, dataset_ntot={dataset_ntot}.")
    print(f"[ROM-DataDriven-SparseGPR] Loaded: M={n_samples}, in_dim={in_dim}, out_dim={out_dim}")

    validation_dataset_root = None
    validation_dataset_dir = None
    validation_dataset_meta = None

    if external_validation:
        validation_dataset_root, validation_dataset_ntot, validation_dataset_dir, validation_dataset_meta, _ = resolve_stage3_dataset(
            this_dir=THIS_DIR,
            requested_ntot=args.dataset_ntot,
            expected_backend=dataset_backend,
            requested_dataset_dir=args.validation_dataset_dir,
        )
        if int(validation_dataset_ntot) != int(dataset_ntot):
            raise RuntimeError(
                "External validation ntot mismatch: "
                f"train={dataset_ntot}, val={validation_dataset_ntot}."
            )
        x_val_raw_all, y_val_raw_all = load_prom_dataset_rom_data_driven(validation_dataset_root)
        x_val_raw_all = np.asarray(x_val_raw_all, dtype=np.float64)
        y_val_raw_all = np.asarray(y_val_raw_all, dtype=np.float64)
        if x_val_raw_all.ndim != 2 or x_val_raw_all.shape[1] != in_dim:
            raise RuntimeError(
                f"External validation input shape mismatch: {x_val_raw_all.shape}; expected (*,{in_dim})."
            )
        if y_val_raw_all.ndim != 2 or y_val_raw_all.shape[1] != out_dim:
            raise RuntimeError(
                f"External validation output shape mismatch: {y_val_raw_all.shape}; expected (*,{out_dim})."
            )

        tr_idx = np.arange(n_samples, dtype=np.int64)
        va_idx = np.arange(x_val_raw_all.shape[0], dtype=np.int64)
        tr_idx_fit = _subsample(tr_idx, int(args.max_train_samples), rng)
        va_idx_eval = _subsample(va_idx, int(args.max_val_samples), rng)

        x_tr_raw = x_raw[tr_idx_fit]
        y_tr_raw = y_raw[tr_idx_fit]
        x_va_raw = x_val_raw_all[va_idx_eval]
        y_va_raw = y_val_raw_all[va_idx_eval]

        train_mu_groups = np.unique(np.round(x_raw[:, :2], decimals=12), axis=0)
        val_mu_groups = np.unique(np.round(x_val_raw_all[:, :2], decimals=12), axis=0)
        split_info = {
            "split_mode": "external_dataset",
            "snap_time_offset": -1,
            "num_mu_groups": int(train_mu_groups.shape[0]),
            "num_val_mu_groups": int(val_mu_groups.shape[0]),
            "num_time_per_mu": int(n_samples // max(train_mu_groups.shape[0], 1)),
            "num_candidates_total": int(x_val_raw_all.shape[0]),
            "num_selected_total": int(va_idx_eval.size),
            "val_frac_requested": float("nan"),
            "val_frac_actual": float(x_va_raw.shape[0] / max(x_va_raw.shape[0] + x_tr_raw.shape[0], 1)),
        }
        print(
            "[ROM-DataDriven-SparseGPR] split = external_dataset "
            f"(train_mu_groups={split_info['num_mu_groups']}, "
            f"val_mu_groups={split_info['num_val_mu_groups']}, "
            f"val_used={x_va_raw.shape[0]})"
        )
    else:
        tr_idx, va_idx, split_info = split_indices_ecsw_param_time(
            x_raw,
            val_frac=float(args.val_frac),
            seed=seed,
            snap_time_offset=int(args.val_snap_time_offset),
            ensure_mu_coverage=True,
        )
        print(
            "[ROM-DataDriven-SparseGPR] split = ecsw_param_time_stratified "
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
        raise RuntimeError("Not enough train samples after duplicate filtering.")

    if int(args.num_inducing) > int(x_tr.shape[0]):
        raise RuntimeError(
            f"num_inducing={int(args.num_inducing)} exceeds effective train samples={int(x_tr.shape[0])}."
        )

    dev = resolve_device(args.device)
    print(f"[ROM-DataDriven-SparseGPR] device = {dev}")
    print(f"[ROM-DataDriven-SparseGPR] train_used = {x_tr.shape[0]} (removed_duplicates={removed})")
    print(f"[ROM-DataDriven-SparseGPR] val_used = {x_va.shape[0]}")
    print(
        f"[ROM-DataDriven-SparseGPR] sparse setup: kernel={args.kernel_name}, ard={bool(args.ard)}, "
        f"num_inducing={int(args.num_inducing)}, epochs={int(args.epochs)}, batch={int(args.batch_size)}"
    )

    inducing_init = choose_inducing_points(
        x_tr,
        num_inducing=int(args.num_inducing),
        method=str(args.inducing_selection),
        seed=seed,
        kmeans_max_iters=int(args.kmeans_max_iters),
        kmeans_batch_size=int(args.kmeans_batch_size),
        kmeans_fit_samples=int(args.kmeans_fit_samples),
    )

    all_inducing = np.zeros((out_dim, int(args.num_inducing), in_dim), dtype=np.float64)
    all_alpha = np.zeros((out_dim, int(args.num_inducing)), dtype=np.float64)
    all_ls = np.zeros((out_dim, in_dim), dtype=np.float64)
    all_os = np.zeros((out_dim,), dtype=np.float64)
    all_noise = np.zeros((out_dim,), dtype=np.float64)
    best_val_mse = np.zeros((out_dim,), dtype=np.float64)

    t0 = time.time()
    for j in range(out_dim):
        print(f"\n[ROM-DataDriven-SparseGPR] Training output {j + 1}/{out_dim}")
        stats = fit_sparse_gp_output(
            x_tr,
            y_tr[:, j],
            x_val=x_va,
            y_val=y_va[:, j],
            inducing_init=inducing_init,
            kernel_name=str(args.kernel_name),
            ard=bool(args.ard),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            min_noise=float(args.min_noise),
            max_noise=None if args.max_noise is None else float(args.max_noise),
            elbo_beta=float(args.elbo_beta),
            learn_inducing=(not bool(args.fixed_inducing)),
            device=dev,
            seed=int(seed + 997 * (j + 1)),
            log_every=int(args.log_every),
        )
        all_inducing[j, :, :] = stats["inducing_points"]
        all_alpha[j, :] = stats["alpha"]
        all_ls[j, :] = stats["lengthscales"]
        all_os[j] = float(stats["outputscale"])
        all_noise[j] = float(stats["noise"])
        best_val_mse[j] = float(stats["best_val_mse"])
    elapsed = time.time() - t0

    payload = {
        "model_family": "sparse_gp",
        "kernel_name": str(args.kernel_name),
        "ard": bool(args.ard),
        "inducing_points": all_inducing,
        "alpha": all_alpha,
        "lengthscales": all_ls,
        "outputscales": all_os,
        "noise": all_noise,
    }

    yhat_tr = predict_sparse_batch(x_tr, payload)
    yhat_va = predict_sparse_batch(x_va, payload)

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

    kernel_learned = (
        f"sparse_{str(args.kernel_name)}_ard={bool(args.ard)} "
        f"(num_inducing={int(args.num_inducing)}, per-output hyperparameters)"
    )

    ckpt = {
        "format": "sparse_gpr_map_full",
        "mapping": "qN = G_sparse_gpr(mu1, mu2, t)",
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "validation_dataset_root": validation_dataset_root,
        "validation_dataset_dir": validation_dataset_dir,
        "validation_dataset_backend": None if validation_dataset_meta is None else validation_dataset_meta.get("solve_backend"),
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "n_tot": int(out_dim),
        "seed": int(seed),
        "x_scaling": str(args.x_scaling),
        "y_scaling": str(args.y_scaling),
        "x_stats": x_stats,
        "y_stats": y_stats,
        "kernel_name": str(args.kernel_name),
        "kernel_ard": bool(args.ard),
        "kernel_learned": kernel_learned,
        "sparse_gp_payload": payload,
        "num_inducing": int(args.num_inducing),
        "inducing_selection": str(args.inducing_selection),
        "kmeans_max_iters": int(args.kmeans_max_iters),
        "kmeans_batch_size": int(args.kmeans_batch_size),
        "kmeans_fit_samples": int(args.kmeans_fit_samples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "min_noise": float(args.min_noise),
        "max_noise": None if args.max_noise is None else float(args.max_noise),
        "elbo_beta": float(args.elbo_beta),
        "fixed_inducing": bool(args.fixed_inducing),
        "device": str(dev),
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
        "n_val_mu_groups": split_info.get("num_val_mu_groups", None),
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
        "mean_best_val_mse_per_output": float(np.nanmean(best_val_mse)),
        "elapsed_s": float(elapsed),
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
            ("validation_dataset_root", validation_dataset_root if validation_dataset_root is not None else "internal_split"),
            ("validation_dataset_dir", validation_dataset_dir if validation_dataset_dir is not None else "internal_split"),
            ("dataset_ntot", int(dataset_ntot)),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("samples_M", int(n_samples)),
            ("in_dim", int(in_dim)),
            ("n_tot", int(out_dim)),
            ("kernel_learned", kernel_learned),
            ("num_inducing", int(args.num_inducing)),
            ("inducing_selection", str(args.inducing_selection)),
            ("epochs", int(args.epochs)),
            ("batch_size", int(args.batch_size)),
            ("lr", float(args.lr)),
            ("min_noise", float(args.min_noise)),
            ("max_noise", "none" if args.max_noise is None else float(args.max_noise)),
            ("elbo_beta", float(args.elbo_beta)),
            ("fixed_inducing", bool(args.fixed_inducing)),
            ("val_split", str(split_info["split_mode"])),
            ("train_rel_frob_percent", float(tr_rel_raw)),
            ("val_rel_frob_percent", float(va_rel_raw)),
            ("train_mse", float(tr_mse_raw)),
            ("val_mse", float(va_mse_raw)),
            ("train_used_after_duplicates", int(x_tr.shape[0])),
            ("val_used", int(x_va.shape[0])),
            ("mean_best_val_mse_per_output", float(np.nanmean(best_val_mse))),
            ("elapsed_s", float(elapsed)),
        ],
    )

    print(f"[ROM-DataDriven-SparseGPR] Training done in {elapsed:.2f}s")
    print(f"[ROM-DataDriven-SparseGPR] kernel_learned = {kernel_learned}")
    print(f"[ROM-DataDriven-SparseGPR] val_rel_frob_percent = {va_rel_raw:.4f}%")
    print(f"[ROM-DataDriven-SparseGPR] Saved checkpoint: {model_path}")
    print(f"[ROM-DataDriven-SparseGPR] Summary: {summary_path}")


if __name__ == "__main__":
    main()
