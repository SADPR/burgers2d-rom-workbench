#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Case-2 RBF closure map with grid-search hyperparameter selection.

Case 2 mapping:
    qN_s = N_rbf(mu1, mu2, t)
"""

import argparse
import os
import time
import numpy as np
import torch

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
    from stage3_split_utils import split_indices_holdout_mu_group
except ModuleNotFoundError:
    from .stage3_split_utils import split_indices_holdout_mu_group
try:
    from rbf_map_common import (
        parse_csv_floats,
        parse_csv_strings,
        train_rbf_grid_map,
    )
except ModuleNotFoundError:
    from .rbf_map_common import (
        parse_csv_floats,
        parse_csv_strings,
        train_rbf_grid_map,
    )
try:
    from project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ensure_layout_dirs()
    parser = argparse.ArgumentParser(
        description="Train Case-2 RBF map qN_s=N_rbf(mu1,mu2,t) with grid search."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="hprom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional explicit dataset directory containing per_mu/ and meta.npy.",
    )
    parser.add_argument("--primary-modes", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="case2_model_rbf.pt")
    parser.add_argument("--summary-name", type=str, default="case2_training_summary_rbf.txt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--kernel-candidates", type=str, default="imq,gaussian,matern,multiquadric,linear")
    parser.add_argument("--epsilon-grid", type=str, default="0.25,0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--lambda-grid", type=str, default="1e-10,1e-8,1e-6,1e-4")
    parser.add_argument(
        "--use-ard",
        action="store_true",
        help="Use ARD lengthscales in distance metric (shared across outputs).",
    )
    parser.add_argument(
        "--ard-lengthscale-grid",
        type=str,
        default="1.0",
        help="Grid of ARD scale values. When --use-ard is enabled, all 3D combinations are tested.",
    )
    parser.add_argument(
        "--per-output-lambda",
        action="store_true",
        help="Select lambda separately per output coefficient (shared kernel/epsilon/ARD).",
    )
    parser.add_argument("--x-scaling", choices=("minmax_-1_1", "zscore"), default="minmax_-1_1")
    parser.add_argument("--y-scaling", choices=("minmax_-1_1", "zscore"), default="zscore")
    parser.add_argument("--duplicate-tol", type=float, default=0.0)
    parser.add_argument("--max-centers", type=int, default=1200)
    parser.add_argument(
        "--val-split-mode",
        choices=("ecsw_param_time_stratified", "mu_group_holdout"),
        default="ecsw_param_time_stratified",
        help="Validation strategy used for RBF hyperparameter selection.",
    )
    parser.add_argument(
        "--val-snap-time-offset",
        type=int,
        default=1,
        help="ECSW-like split: minimum time column considered for validation (>=1).",
    )
    parser.add_argument(
        "--val-holdout-mu",
        type=str,
        default="",
        help="For mu_group_holdout mode: holdout parameter pair as 'mu1,mu2'.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print grid-search status every N candidates (<=0 disables progress prints).",
    )
    args = parser.parse_args(argv)

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
    summary_path = os.path.join(STAGE3_DIR, str(args.summary_name))

    kernels = parse_csv_strings(args.kernel_candidates)
    eps_grid = parse_csv_floats(args.epsilon_grid)
    lam_grid = parse_csv_floats(args.lambda_grid)
    ard_grid = parse_csv_floats(args.ard_lengthscale_grid)
    if int(args.max_centers) < 2:
        raise ValueError("--max-centers must be >= 2.")

    print(f"[Case2-RBF] dataset_dir = {dataset_dir}")
    print(f"[Case2-RBF] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case2-RBF] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case2-RBF] primary_modes = {n_primary}")
    print(f"[Case2-RBF] kernels = {kernels}")
    print(f"[Case2-RBF] epsilon_grid = {eps_grid}")
    print(f"[Case2-RBF] lambda_grid = {lam_grid}")
    print(f"[Case2-RBF] use_ard = {bool(args.use_ard)}")
    print(f"[Case2-RBF] ard_lengthscale_grid = {ard_grid}")
    print(f"[Case2-RBF] per_output_lambda = {bool(args.per_output_lambda)}")
    print(f"[Case2-RBF] max_centers = {int(args.max_centers)}")
    print(f"[Case2-RBF] progress_every = {int(args.progress_every)}")

    x_raw, y_raw = load_prom_dataset_case2(dataset_root, primary_modes=n_primary)
    m, in_dim = x_raw.shape
    _, out_dim = y_raw.shape
    print(f"[Case2-RBF] Loaded: M={m}, in_dim={in_dim}, out_dim={out_dim}")
    split_mode = str(args.val_split_mode).strip().lower()
    if split_mode == "ecsw_param_time_stratified":
        tr_idx, va_idx, split_info = split_indices_ecsw_param_time(
            x_raw,
            val_frac=float(args.val_frac),
            seed=int(args.seed),
            snap_time_offset=int(args.val_snap_time_offset),
            ensure_mu_coverage=True,
        )
        print(
            "[Case2-RBF] split = ecsw_param_time_stratified "
            f"(mu_groups={split_info['num_mu_groups']}, "
            f"time_per_mu={split_info['num_time_per_mu']}, "
            f"val_requested={split_info['val_frac_requested']:.4f}, "
            f"val_actual={split_info['val_frac_actual']:.4f})"
        )
    else:
        holdout_mu = None
        if str(args.val_holdout_mu).strip():
            parts = [s.strip() for s in str(args.val_holdout_mu).split(",")]
            if len(parts) != 2:
                raise ValueError("--val-holdout-mu must be 'mu1,mu2'.")
            holdout_mu = (float(parts[0]), float(parts[1]))
        tr_idx, va_idx, split_info = split_indices_holdout_mu_group(
            x_raw,
            holdout_mu=holdout_mu,
            avoid_center_and_corners=True,
        )
        print(
            "[Case2-RBF] split = mu_group_holdout "
            f"(holdout_mu=({split_info['holdout_mu1']:.3f},{split_info['holdout_mu2']:.4f}), "
            f"mu_groups={split_info['num_mu_groups']}, "
            f"val_samples={split_info['num_selected_total']})"
        )

    t0 = time.time()
    fit = train_rbf_grid_map(
        x_raw=x_raw,
        y_raw=y_raw,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        train_indices=tr_idx,
        val_indices=va_idx,
        kernel_candidates=kernels,
        epsilon_grid=eps_grid,
        lambda_grid=lam_grid,
        use_ard=bool(args.use_ard),
        ard_lengthscale_grid=ard_grid,
        per_output_lambda=bool(args.per_output_lambda),
        x_scaling=str(args.x_scaling),
        y_scaling=str(args.y_scaling),
        duplicate_tol=float(args.duplicate_tol),
        max_centers=int(args.max_centers),
        progress_every=int(args.progress_every),
        progress_prefix="[Case2-RBF][grid]",
    )
    elapsed = time.time() - t0

    ckpt = {
        "format": "rbf_map",
        "case": "case2",
        "mapping": "qN_s = N_rbf(mu1, mu2, t)",
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "primary_modes": int(n_primary),
        "secondary_modes": int(dataset_ntot - n_primary),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "seed": int(args.seed),
        "val_split": str(split_info["split_mode"]),
        "val_split_snap_time_offset": int(split_info["snap_time_offset"]),
        "val_frac_actual": float(split_info["val_frac_actual"]),
        "n_unique_mu_groups": int(split_info["num_mu_groups"]),
        "n_time_per_mu": int(split_info["num_time_per_mu"]),
        "n_candidates_total": int(split_info["num_candidates_total"]),
        "n_selected_total": int(split_info["num_selected_total"]),
        **fit,
    }
    torch.save(ckpt, model_path)

    write_kv_txt(
        summary_path,
        [
            ("model_name", model_name),
            ("model_path", model_path),
            ("dataset_root", dataset_root),
            ("dataset_ntot", int(dataset_ntot)),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("primary_modes", int(n_primary)),
            ("secondary_modes", int(dataset_ntot - n_primary)),
            ("samples_M", int(m)),
            ("in_dim", int(in_dim)),
            ("out_dim", int(out_dim)),
            ("kernel_name", fit["kernel_name"]),
            ("epsilon", fit["epsilon"]),
            ("lambda_reg", fit["lambda_reg"]),
            ("lambda_reg_min", fit.get("lambda_reg_min")),
            ("lambda_reg_max", fit.get("lambda_reg_max")),
            ("lambda_reg_unique_count", fit.get("lambda_reg_unique_count")),
            ("per_output_lambda", fit.get("per_output_lambda")),
            ("use_ard", fit.get("use_ard")),
            ("ard_lengthscales", fit.get("ard_lengthscales")),
            ("best_val_rel_frob_percent", fit["best_val_rel_frob_percent"]),
            ("final_fit_rel_frob_percent", fit["final_fit_rel_frob_percent"]),
            ("val_split", str(split_info["split_mode"])),
            ("val_split_snap_time_offset", int(split_info["snap_time_offset"])),
            ("val_frac_requested", float(args.val_frac)),
            ("val_frac_actual", float(split_info["val_frac_actual"])),
            ("val_holdout_mu1", split_info.get("holdout_mu1")),
            ("val_holdout_mu2", split_info.get("holdout_mu2")),
            ("n_unique_mu_groups", int(split_info["num_mu_groups"])),
            ("n_time_per_mu", int(split_info["num_time_per_mu"])),
            ("n_candidates_total", int(split_info["num_candidates_total"])),
            ("n_selected_total", int(split_info["num_selected_total"])),
            ("n_centers_final", fit["n_centers_final"]),
            ("elapsed_s", elapsed),
        ],
    )
    print(f"[Case2-RBF] Training done in {elapsed:.2f}s")
    print(f"[Case2-RBF] Saved checkpoint: {model_path}")
    print(f"[Case2-RBF] Summary: {summary_path}")


if __name__ == "__main__":
    main()
