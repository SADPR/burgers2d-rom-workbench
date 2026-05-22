#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Case-1 RBF closure map with grid-search hyperparameter selection.

Case 1 mapping:
    qN_s = N_rbf(qN_p)
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
    from stage3_perform_training_case_1_ann import load_prom_dataset_case1
except ModuleNotFoundError:
    from .stage3_perform_training_case_1_ann import load_prom_dataset_case1
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
        description="Train Case-1 RBF map qN_s=N_rbf(qN_p) with grid search."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="hprom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument("--primary-modes", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="case1_model_rbf.pt")
    parser.add_argument("--summary-name", type=str, default="case1_training_summary_rbf.txt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--kernel-candidates", type=str, default="imq,gaussian,matern,multiquadric,linear")
    parser.add_argument("--epsilon-grid", type=str, default="0.25,0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--lambda-grid", type=str, default="1e-10,1e-8,1e-6,1e-4")
    parser.add_argument("--x-scaling", choices=("minmax_-1_1", "zscore"), default="minmax_-1_1")
    parser.add_argument("--y-scaling", choices=("minmax_-1_1", "zscore"), default="zscore")
    parser.add_argument("--duplicate-tol", type=float, default=0.0)
    parser.add_argument("--max-centers", type=int, default=1200)
    args = parser.parse_args(argv)

    dataset_backend = str(args.dataset_backend).strip().lower()
    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=args.dataset_ntot,
        expected_backend=dataset_backend,
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
    if int(args.max_centers) < 2:
        raise ValueError("--max-centers must be >= 2.")

    print(f"[Case1-RBF] dataset_dir = {dataset_dir}")
    print(f"[Case1-RBF] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case1-RBF] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case1-RBF] primary_modes = {n_primary}")
    print(f"[Case1-RBF] kernels = {kernels}")
    print(f"[Case1-RBF] epsilon_grid = {eps_grid}")
    print(f"[Case1-RBF] lambda_grid = {lam_grid}")
    print(f"[Case1-RBF] max_centers = {int(args.max_centers)}")

    x_raw, y_raw = load_prom_dataset_case1(dataset_root, primary_modes=n_primary)
    m, in_dim = x_raw.shape
    _, out_dim = y_raw.shape
    print(f"[Case1-RBF] Loaded: M={m}, in_dim={in_dim}, out_dim={out_dim}")

    t0 = time.time()
    fit = train_rbf_grid_map(
        x_raw=x_raw,
        y_raw=y_raw,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        kernel_candidates=kernels,
        epsilon_grid=eps_grid,
        lambda_grid=lam_grid,
        x_scaling=str(args.x_scaling),
        y_scaling=str(args.y_scaling),
        duplicate_tol=float(args.duplicate_tol),
        max_centers=int(args.max_centers),
    )
    elapsed = time.time() - t0

    ckpt = {
        "format": "rbf_map",
        "case": "case1",
        "mapping": "qN_s = N_rbf(qN_p)",
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "primary_modes": int(n_primary),
        "secondary_modes": int(dataset_ntot - n_primary),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "seed": int(args.seed),
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
            ("best_val_rel_frob_percent", fit["best_val_rel_frob_percent"]),
            ("final_fit_rel_frob_percent", fit["final_fit_rel_frob_percent"]),
            ("n_centers_final", fit["n_centers_final"]),
            ("elapsed_s", elapsed),
        ],
    )
    print(f"[Case1-RBF] Training done in {elapsed:.2f}s")
    print(f"[Case1-RBF] Saved checkpoint: {model_path}")
    print(f"[Case1-RBF] Summary: {summary_path}")


if __name__ == "__main__":
    main()

