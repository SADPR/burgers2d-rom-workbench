#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Launch 4 Case-2 Stage-3 trainings (ANN/GPR for n=20 and n=10) in Results_Maday.

This script runs, in sequence:
  1) ANN with n=20 (n_s=131)
  2) GPR with n=20 (n_s=131)
  3) ANN with n=10 (n_s=141)
  4) GPR with n=10 (n_s=141)

All outputs are redirected to Results_Maday/<tag>/Stage3/models via the
existing Maday wrapper scripts.
"""

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
    family: str  # ann or gpr
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
    printable = " ".join(cmd)
    print(f"\n[run] {printable}")
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
            name="ann_ns131",
            family="ann",
            primary_modes=20,
            model_name="case2_ann_mu_t_ns131.pt",
            summary_name="case2_ann_mu_t_ns131_summary.txt",
        ),
        RunSpec(
            name="gpr_ns131",
            family="gpr",
            primary_modes=20,
            model_name="case2_gpr_mu_t_ns131.pt",
            summary_name="case2_gpr_mu_t_ns131_summary.txt",
        ),
        RunSpec(
            name="ann_ns141",
            family="ann",
            primary_modes=10,
            model_name="case2_ann_mu_t_ns141.pt",
            summary_name="case2_ann_mu_t_ns141_summary.txt",
        ),
        RunSpec(
            name="gpr_ns141",
            family="gpr",
            primary_modes=10,
            model_name="case2_gpr_mu_t_ns141.pt",
            summary_name="case2_gpr_mu_t_ns141_summary.txt",
        ),
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Launch 4 Case-2 ANN/GPR trainings (Maday isolated).")
    parser.add_argument("--maday-tag", type=str, default="maday_clean_try04")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=151)

    parser.add_argument("--ann-hidden-dims", type=str, default="32,64,128,256,256")
    parser.add_argument("--ann-activation", choices=("elu", "gelu", "silu", "tanh", "relu", "leaky_relu"), default="elu")
    parser.add_argument("--ann-seed", type=int, default=42)

    parser.add_argument("--gpr-seed", type=int, default=42)
    parser.add_argument("--gpr-val-frac", type=float, default=0.1)
    parser.add_argument("--gpr-max-train-samples", type=int, default=1200)
    parser.add_argument("--gpr-max-val-samples", type=int, default=4000)
    parser.add_argument("--gpr-alpha", type=float, default=1e-12)
    parser.add_argument("--gpr-n-restarts-optimizer", type=int, default=2)
    parser.add_argument("--gpr-length-scale-bounds", type=str, default="1e-2,5.0")
    parser.add_argument("--gpr-white-noise-level", type=float, default=1e-6)
    parser.add_argument("--gpr-white-noise-bounds", type=str, default="1e-10,1e0")
    parser.add_argument("--gpr-no-white-kernel", action="store_true")

    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--log-subdir", type=str, default="logs_case2_ann_gpr_4runs")
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset: ann_ns131 gpr_ns131 ann_ns141 gpr_ns141")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = _project_root()
    ann_wrapper = root / "stage3_perform_training_case_2_ann_test_n20_maday.py"
    gpr_wrapper = root / "stage3_perform_training_case_2_gpr_maday.py"

    if not ann_wrapper.is_file() or not gpr_wrapper.is_file():
        raise FileNotFoundError("Could not find ANN/GPR maday wrapper scripts in Project_YvonMaday/.")

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
            raise ValueError(f"--only did not match any known run names. Got: {args.only}")

    print("[launcher] tag:", paths.tag)
    print("[launcher] dataset_dir:", dataset_dir)
    print("[launcher] stage3 models:", paths.stage3_models)
    print("[launcher] log_dir:", log_dir)

    for spec in specs:
        wrapper = ann_wrapper if spec.family == "ann" else gpr_wrapper
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
            "--primary-modes",
            str(spec.primary_modes),
            "--model-name",
            spec.model_name,
            "--summary-name",
            spec.summary_name,
        ]

        if args.maday_results_root is not None:
            cmd.extend(["--maday-results-root", str(args.maday_results_root)])

        if spec.family == "ann":
            cmd.extend(
                [
                    "--hidden-dims",
                    args.ann_hidden_dims,
                    "--activation",
                    args.ann_activation,
                    "--seed",
                    str(args.ann_seed),
                ]
            )
        else:
            cmd.extend(
                [
                    "--seed",
                    str(args.gpr_seed),
                    "--val-frac",
                    str(args.gpr_val_frac),
                    "--max-train-samples",
                    str(args.gpr_max_train_samples),
                    "--max-val-samples",
                    str(args.gpr_max_val_samples),
                    "--alpha",
                    str(args.gpr_alpha),
                    "--n-restarts-optimizer",
                    str(args.gpr_n_restarts_optimizer),
                    "--length-scale-bounds",
                    args.gpr_length_scale_bounds,
                    "--white-noise-level",
                    str(args.gpr_white_noise_level),
                    "--white-noise-bounds",
                    args.gpr_white_noise_bounds,
                ]
            )
            if args.gpr_no_white_kernel:
                cmd.append("--no-white-kernel")

        log_path = log_dir / f"{spec.name}.log"
        _stream_to_console_and_log(cmd, cwd=root, log_path=log_path, dry_run=args.dry_run)

    print("\n[done] Completed requested runs.")
    print(f"[done] Models folder: {paths.stage3_models}")
    print(f"[done] Logs folder:   {log_dir}")


if __name__ == "__main__":
    main()
