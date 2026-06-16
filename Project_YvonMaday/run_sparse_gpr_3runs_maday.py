#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Launch sparse-GPR trainings for n_s=131, n_s=141 and n_s=151 in Results_Maday."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


@dataclass(frozen=True)
class RunSpec:
    name: str
    mode: str  # case2 or full
    primary_modes: int
    model_name: str
    summary_name: str


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_default_dataset_dir(root: Path) -> Path:
    candidates = [
        root / "250x250" / "Results" / "Stage2" / "prom_coeff_dataset_ntot151",
        root / "Results" / "Stage2" / "prom_coeff_dataset_ntot151",
    ]
    for d in candidates:
        if (d / "per_mu").is_dir() and (d / "meta.npy").is_file():
            return d
    return candidates[0]


def _validate_dataset_dir(path: Path) -> None:
    if not (path / "per_mu").is_dir():
        raise FileNotFoundError(f"Missing per_mu folder in dataset dir: {path}")
    if not (path / "meta.npy").is_file():
        raise FileNotFoundError(f"Missing meta.npy in dataset dir: {path}")


def _stream_to_console_and_log(cmd: Sequence[str], cwd: Path, log_path: Path, dry_run: bool = False) -> None:
    print("\n[run] " + " ".join(cmd))
    print(f"[log] {log_path}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Command failed (exit={ret}). See log: {log_path}")


def _build_specs() -> List[RunSpec]:
    return [
        RunSpec(
            name="sparse_gpr_ns131",
            mode="case2",
            primary_modes=20,
            model_name="case2_sparse_gpr_mu_t_ns131.pt",
            summary_name="case2_sparse_gpr_mu_t_ns131_summary.txt",
        ),
        RunSpec(
            name="sparse_gpr_ns141",
            mode="case2",
            primary_modes=10,
            model_name="case2_sparse_gpr_mu_t_ns141.pt",
            summary_name="case2_sparse_gpr_mu_t_ns141_summary.txt",
        ),
        RunSpec(
            name="sparse_gpr_ns151",
            mode="full",
            primary_modes=0,
            model_name="rom_data_driven_sparse_gpr_mu_t_ntot151.pt",
            summary_name="rom_data_driven_sparse_gpr_mu_t_ntot151_summary.txt",
        ),
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Launch 3 sparse-GPR trainings (Maday isolated).")
    parser.add_argument("--maday-tag", type=str, default="maday_clean_try04")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=151)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=4008)
    parser.add_argument("--max-val-samples", type=int, default=501)
    parser.add_argument("--x-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")
    parser.add_argument("--y-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")
    parser.add_argument("--duplicate-tol", type=float, default=0.0)

    parser.add_argument("--num-inducing", type=int, default=451)
    parser.add_argument("--inducing-selection", choices=("random", "kmeans"), default="kmeans")
    parser.add_argument("--kmeans-max-iters", type=int, default=40)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-fit-samples", type=int, default=40000)
    parser.add_argument("--kernel-name", choices=("rbf", "matern15"), default="rbf")
    parser.add_argument("--no-ard", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-noise", type=float, default=1e-6)
    parser.add_argument("--fixed-inducing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--log-subdir", type=str, default="logs_sparse_gpr_3runs")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset: sparse_gpr_ns131 sparse_gpr_ns141 sparse_gpr_ns151",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = _project_root()
    case2_wrapper = root / "stage3_perform_training_case_2_sparse_gpr_maday.py"
    full_wrapper = root / "stage3_perform_training_rom_data_driven_sparse_gpr_maday.py"
    if not case2_wrapper.is_file() or not full_wrapper.is_file():
        raise FileNotFoundError("Could not find sparse-GPR Maday wrapper scripts in Project_YvonMaday/.")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else _resolve_default_dataset_dir(root)
    _validate_dataset_dir(dataset_dir)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    log_dir = Path(paths.stage3) / args.log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs()
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise ValueError(f"--only did not match known run names. Got: {args.only}")

    print("[launcher] tag:", paths.tag)
    print("[launcher] dataset_dir:", dataset_dir)
    print("[launcher] stage3 models:", paths.stage3_models)
    print("[launcher] log_dir:", log_dir)

    for spec in specs:
        wrapper = case2_wrapper if spec.mode == "case2" else full_wrapper
        cmd = [
            args.python_exe,
            str(wrapper),
            "--maday-tag",
            paths.tag,
            "--dataset-backend",
            args.dataset_backend,
            "--dataset-ntot",
            str(args.dataset_ntot),
            "--dataset-dir",
            str(dataset_dir),
            "--model-name",
            spec.model_name,
            "--summary-name",
            spec.summary_name,
            "--seed",
            str(args.seed),
            "--val-frac",
            str(args.val_frac),
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-val-samples",
            str(args.max_val_samples),
            "--x-scaling",
            args.x_scaling,
            "--y-scaling",
            args.y_scaling,
            "--duplicate-tol",
            str(args.duplicate_tol),
            "--num-inducing",
            str(args.num_inducing),
            "--inducing-selection",
            args.inducing_selection,
            "--kmeans-max-iters",
            str(args.kmeans_max_iters),
            "--kmeans-batch-size",
            str(args.kmeans_batch_size),
            "--kmeans-fit-samples",
            str(args.kmeans_fit_samples),
            "--kernel-name",
            args.kernel_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--min-noise",
            str(args.min_noise),
            "--device",
            args.device,
            "--log-every",
            str(args.log_every),
        ]
        if args.maday_results_root is not None:
            cmd.extend(["--maday-results-root", str(args.maday_results_root)])
        if args.no_ard:
            cmd.append("--no-ard")
        if args.fixed_inducing:
            cmd.append("--fixed-inducing")
        if spec.mode == "case2":
            cmd.extend(["--primary-modes", str(spec.primary_modes)])

        log_path = log_dir / f"{spec.name}.log"
        _stream_to_console_and_log(cmd, cwd=root, log_path=log_path, dry_run=args.dry_run)

    print("\n[done] Completed requested sparse-GPR runs.")
    print(f"[done] Models folder: {paths.stage3_models}")
    print(f"[done] Logs folder:   {log_dir}")


if __name__ == "__main__":
    main()
