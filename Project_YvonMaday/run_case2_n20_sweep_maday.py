#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run Case-2 n=20 sweep in isolated Maday experiment folders."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


@dataclass(frozen=True)
class SweepConfig:
    tag: str
    hidden_dims: str
    activation: str
    lr: float
    weight_decay: float
    dropout: float
    batch_size: int


DEFAULT_CONFIGS: Tuple[SweepConfig, ...] = (
    SweepConfig("c1", "32,64,128,256,256", "elu", 1e-3, 1e-6, 0.00, 128),
    SweepConfig("c2", "64,64,128,256,256", "elu", 8e-4, 1e-6, 0.00, 128),
    SweepConfig("c3", "64,128,128,256,256", "gelu", 6e-4, 1e-5, 0.05, 96),
    SweepConfig("c4", "96,192,192,256", "silu", 7e-4, 2e-5, 0.08, 128),
    SweepConfig("c5", "64,64,64,128,128", "tanh", 1e-3, 1e-6, 0.00, 64),
)

DEFAULT_POINTS: Tuple[Tuple[float, float], ...] = (
    (4.875, 0.0225),
    (4.560, 0.0190),
    (5.190, 0.0260),
)


def _validate_dataset_dir(raw_path: str, expected_backend: str) -> str:
    dataset_dir = str(raw_path).strip()
    if len(dataset_dir) == 0:
        raise ValueError("--dataset-dir is empty.")
    d = Path(dataset_dir).expanduser().resolve()
    per_mu = d / "per_mu"
    meta = d / "meta.npy"
    if not per_mu.is_dir():
        raise FileNotFoundError(f"Missing per_mu folder in --dataset-dir: {d}")
    if not meta.exists():
        raise FileNotFoundError(f"Missing meta.npy in --dataset-dir: {d}")

    import numpy as np

    m = np.load(str(meta), allow_pickle=True).item()
    backend = str(m.get("solve_backend", "")).strip().lower()
    wanted = str(expected_backend).strip().lower()
    if backend != wanted:
        raise ValueError(
            f"--dataset-dir backend mismatch: solve_backend='{backend}', expected '{wanted}'. dataset_dir={d}"
        )
    return str(d)


def _run(cmd: Sequence[str], cwd: Path, unbuffered: bool) -> None:
    print("\n[run] " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    if unbuffered:
        env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _parse_seeds(raw: str) -> List[int]:
    out = []
    for item in raw.split(","):
        txt = item.strip()
        if txt:
            out.append(int(txt))
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def _score_ranking(
    csv_path: Path,
    model_stems: Iterable[str],
    points: Sequence[Tuple[float, float]],
    weights: Sequence[float],
) -> List[Tuple[float, float, float, float, str]]:
    by_model = defaultdict(dict)
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    model_stems = set(model_stems)
    for row in rows:
        model = row["model"]
        if model not in model_stems:
            continue
        key = (round(float(row["mu1"]), 3), round(float(row["mu2"]), 4))
        by_model[model][key] = float(row["rel_frob_percent"])

    p0 = (round(points[0][0], 3), round(points[0][1], 4))
    p1 = (round(points[1][0], 3), round(points[1][1], 4))
    p2 = (round(points[2][0], 3), round(points[2][1], 4))

    rank = []
    for model, vals in by_model.items():
        if p0 in vals and p1 in vals and p2 in vals:
            e0, e1, e2 = vals[p0], vals[p1], vals[p2]
            score = weights[0] * e0 + weights[1] * e1 + weights[2] * e2
            rank.append((score, e0, e1, e2, model))
    rank.sort(key=lambda x: x[0])
    return rank


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Case-2 n=20 sweep in Results_Maday.")
    parser.add_argument("--maday-tag", type=str, default="exp_maday_p2")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=151)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--primary-modes", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="11,23,42,77")
    parser.add_argument("--model-prefix", type=str, default="case2_model_n20_maday")
    parser.add_argument("--reference-source", choices=("linear_runs", "stage2"), default="linear_runs")
    parser.add_argument("--python", type=str, default="python3")
    parser.add_argument("--unbuffered", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--score-weights", type=str, default="0.2,0.4,0.4")
    args = parser.parse_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)

    root = Path(__file__).resolve().parent
    train_script = root / "stage3_perform_training_case_2_ann_test_n20_maday.py"
    eval_script = root / "check_case2_offline_errors.py"
    out_dir = Path(paths.figures) / "offline_case2"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir_validated = None
    if args.dataset_dir is not None:
        dataset_dir_validated = _validate_dataset_dir(args.dataset_dir, args.dataset_backend)

    seeds = _parse_seeds(args.seeds)
    weights = [float(x.strip()) for x in args.score_weights.split(",") if x.strip()]
    if len(weights) != 3:
        raise ValueError("--score-weights must have exactly three values.")

    model_names: List[str] = []
    for cfg in DEFAULT_CONFIGS:
        for seed in seeds:
            model_name = f"{args.model_prefix}_{cfg.tag}_s{seed}.pt"
            model_names.append(model_name)
            if args.skip_train:
                continue
            cmd = [
                args.python,
                str(train_script),
                "--maday-tag",
                paths.tag,
                "--dataset-backend",
                args.dataset_backend,
                "--dataset-ntot",
                str(args.dataset_ntot),
                "--primary-modes",
                str(args.primary_modes),
                "--model-name",
                model_name,
                "--hidden-dims",
                cfg.hidden_dims,
                "--activation",
                cfg.activation,
                "--lr",
                str(cfg.lr),
                "--weight-decay",
                str(cfg.weight_decay),
                "--dropout",
                str(cfg.dropout),
                "--batch-size",
                str(cfg.batch_size),
                "--seed",
                str(seed),
            ]
            if args.maday_results_root is not None:
                cmd.extend(["--maday-results-root", args.maday_results_root])
            if dataset_dir_validated is not None:
                cmd.extend(["--dataset-dir", dataset_dir_validated])
            if args.unbuffered and cmd[0].startswith("python"):
                cmd.insert(1, "-u")
            _run(cmd, cwd=root, unbuffered=args.unbuffered)

    model_paths: List[Path] = []
    for model_name in model_names:
        p = Path(paths.stage3_models) / model_name
        if p.exists():
            model_paths.append(p)
        else:
            print(f"[warn] missing model, skipping: {p}")
    if not model_paths:
        raise RuntimeError("No sweep model checkpoints were found in Results_Maday.")

    eval_cmd = [
        args.python,
        str(eval_script),
        "--reference-source",
        args.reference_source,
        "--output-dir",
        str(out_dir),
    ]
    for mu1, mu2 in DEFAULT_POINTS:
        eval_cmd.extend(["--point", f"{mu1},{mu2}"])
    for model_path in model_paths:
        eval_cmd.extend(["--model-path", str(model_path)])
    if args.unbuffered and eval_cmd[0].startswith("python"):
        eval_cmd.insert(1, "-u")
    _run(eval_cmd, cwd=root, unbuffered=args.unbuffered)

    csv_path = out_dir / f"case2_offline_errors_{args.reference_source}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {csv_path}")

    stems = [Path(m).stem for m in model_names]
    rank = _score_ranking(csv_path, stems, DEFAULT_POINTS, weights)
    rank_path = out_dir / "case2_n20_sweep_ranking.txt"
    with rank_path.open("w", encoding="utf-8") as f:
        f.write("model,score,verif,test1,test2\n")
        for score, ev, e1, e2, model in rank:
            f.write(f"{model},{score:.6f},{ev:.6f},{e1:.6f},{e2:.6f}\n")

    print("\nTop 10 models:")
    for score, ev, e1, e2, model in rank[:10]:
        print(
            f"{model:40s} score={score:8.3f} "
            f"verif={ev:7.3f}% test1={e1:7.3f}% test2={e2:7.3f}%"
        )
    print(f"\nSaved ranking: {rank_path}")


if __name__ == "__main__":
    main()

