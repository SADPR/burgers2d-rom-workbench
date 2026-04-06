#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporary offline diagnostic for Case-2 maps.

Goal:
  Compare offline map prediction q_s(mu, t) against ROM reference q_s_ref
  without running online PROM/HPROM solves.

Reference choices:
  - linear_runs (default): uses Results/Runs/Linear/*/qN.npy (best for off-grid test points).
  - stage2: uses Results/Stage2/prom_coeff_dataset_ntot*/per_mu/*/qN.npy
            (works for points present in Stage-2 dataset).

Usage (from Project_YvonMaday/250x250):
  python3 check_case2_offline_errors.py
  python3 check_case2_offline_errors.py --reference-source stage2
  python3 check_case2_offline_errors.py --model-path ../250x250/Results/Stage3/models/case2_model_n20.pt
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        return self.fc6(x)


class Case2Model(nn.Module):
    """Input: (mu1, mu2, t), output: q_s."""

    def __init__(self, n_s):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, 3)), np.ones((1, 3)))
        self.core = CoreMLP(3, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw


def _parse_points(raw_points: List[str]) -> List[Tuple[float, float]]:
    points = []
    for txt in raw_points:
        parts = [s.strip() for s in txt.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid --point '{txt}', expected 'mu1,mu2'.")
        points.append((float(parts[0]), float(parts[1])))
    return points


def _find_linear_run_qn(base_dir: Path, ntot: int, mu1: float, mu2: float) -> Optional[Path]:
    run_dir = base_dir / "Results" / "Runs" / "Linear"
    if not run_dir.is_dir():
        return None
    prefix = f"linear_prom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot{ntot}"
    cand = run_dir / prefix / "qN.npy"
    if cand.exists():
        return cand
    # fallback tolerant search
    for d in run_dir.glob(f"linear_prom_mu1_*_mu2_*_ntot{ntot}"):
        qn = d / "qN.npy"
        mu = d / "mu.npy"
        if qn.exists() and mu.exists():
            try:
                mu_vec = np.load(mu, allow_pickle=False).reshape(-1)
                if mu_vec.size >= 2 and abs(mu_vec[0] - mu1) < 1e-12 and abs(mu_vec[1] - mu2) < 1e-12:
                    return qn
            except Exception:
                continue
    return None


def _find_linear_run_t(base_dir: Path, ntot: int, mu1: float, mu2: float) -> Optional[Path]:
    run_dir = base_dir / "Results" / "Runs" / "Linear"
    prefix = f"linear_prom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot{ntot}"
    cand = run_dir / prefix / "t.npy"
    if cand.exists():
        return cand
    for d in run_dir.glob(f"linear_prom_mu1_*_mu2_*_ntot{ntot}"):
        t = d / "t.npy"
        mu = d / "mu.npy"
        if t.exists() and mu.exists():
            try:
                mu_vec = np.load(mu, allow_pickle=False).reshape(-1)
                if mu_vec.size >= 2 and abs(mu_vec[0] - mu1) < 1e-12 and abs(mu_vec[1] - mu2) < 1e-12:
                    return t
            except Exception:
                continue
    return None


def _find_stage2_qn(base_dir: Path, ntot: int, mu1: float, mu2: float) -> Optional[Path]:
    d = (
        base_dir
        / "Results"
        / "Stage2"
        / f"prom_coeff_dataset_ntot{ntot}"
        / "per_mu"
        / f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"
        / "qN.npy"
    )
    return d if d.exists() else None


def _find_stage2_t(base_dir: Path, ntot: int, mu1: float, mu2: float) -> Optional[Path]:
    d = (
        base_dir
        / "Results"
        / "Stage2"
        / f"prom_coeff_dataset_ntot{ntot}"
        / "per_mu"
        / f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"
        / "t.npy"
    )
    return d if d.exists() else None


def _load_case2_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    n_s = int(ckpt["n_s"])
    ntot = int(ckpt.get("dataset_ntot"))
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"{ckpt_path}: expected in_dim=3, got {in_dim}.")
    if ntot <= n_s:
        raise ValueError(f"{ckpt_path}: invalid split ntot={ntot}, n_s={n_s}.")

    model = Case2Model(n_s).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ntot, n_s


def _predict_qs(model: nn.Module, mu1: float, mu2: float, t: np.ndarray, device: torch.device) -> np.ndarray:
    x = np.column_stack(
        [
            np.full_like(t, float(mu1), dtype=np.float32),
            np.full_like(t, float(mu2), dtype=np.float32),
            t.astype(np.float32),
        ]
    )
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    # (nt, ns) -> (ns, nt)
    return y.T


def main():
    parser = argparse.ArgumentParser(description="Offline Case-2 map error checker.")
    parser.add_argument(
        "--model-path",
        action="append",
        default=None,
        help="Checkpoint path(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--reference-source",
        choices=("linear_runs", "stage2"),
        default="linear_runs",
        help="Where to read qN reference from.",
    )
    parser.add_argument(
        "--point",
        action="append",
        default=["4.875,0.0225", "4.56,0.019", "5.19,0.026"],
        help="Evaluation point 'mu1,mu2'. Can be passed multiple times.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Figures/offline_case2",
    )
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    base_dir = this_dir
    out_dir = (base_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    points = _parse_points(args.point)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[offline-case2] CUDA requested but not available. Using CPU.")
        device = torch.device("cpu")

    if args.model_path:
        model_paths = [Path(p).expanduser().resolve() for p in args.model_path]
    else:
        defaults = [
            base_dir / "Results" / "Stage3" / "models" / "case2_model.pt",
            base_dir / "Results" / "Stage3" / "models" / "case2_model_n20.pt",
            base_dir
            / "Results_Enrichment"
            / "Stage3"
            / "prom_coeff_dataset_ntot151_enriched_lhs20"
            / "models"
            / "case2_model_enriched.pt",
            base_dir
            / "Results_Enrichment"
            / "Stage3"
            / "prom_coeff_dataset_ntot151_enriched_lhs20"
            / "models"
            / "case2_model_enriched_n20.pt",
        ]
        model_paths = [p.resolve() for p in defaults if p.exists()]
        if not model_paths:
            raise FileNotFoundError("No default Case-2 checkpoints found. Use --model-path.")

    rows = []
    missing_refs = []

    for ckpt_path in model_paths:
        if not ckpt_path.exists():
            print(f"[offline-case2] skip missing model: {ckpt_path}")
            continue

        model, ntot, n_s = _load_case2_model(ckpt_path, device=device)
        n_p = ntot - n_s
        model_tag = ckpt_path.stem

        for (mu1, mu2) in points:
            if args.reference_source == "linear_runs":
                qn_path = _find_linear_run_qn(base_dir, ntot, mu1, mu2)
                t_path = _find_linear_run_t(base_dir, ntot, mu1, mu2)
            else:
                qn_path = _find_stage2_qn(base_dir, ntot, mu1, mu2)
                t_path = _find_stage2_t(base_dir, ntot, mu1, mu2)

            if qn_path is None or t_path is None:
                missing_refs.append((model_tag, mu1, mu2, args.reference_source))
                continue

            qn_ref = np.load(qn_path, allow_pickle=False)
            t = np.load(t_path, allow_pickle=False).reshape(-1)

            if qn_ref.ndim != 2 or qn_ref.shape[0] != ntot:
                raise ValueError(f"Unexpected qN shape at {qn_path}: {qn_ref.shape}, expected ({ntot}, nt).")
            if qn_ref.shape[1] != t.size:
                raise ValueError(
                    f"Time length mismatch for {qn_path}: qN nt={qn_ref.shape[1]} vs t={t.size}."
                )

            q_s_ref = qn_ref[n_p:, :]  # (n_s, nt)
            q_s_pred = _predict_qs(model, mu1, mu2, t, device=device)  # (n_s, nt)

            err = q_s_ref - q_s_pred

            abs_frob = float(np.linalg.norm(err))
            ref_frob = float(np.linalg.norm(q_s_ref))
            rel_frob_pct = 100.0 * abs_frob / (ref_frob + 1e-30)

            abs_mode = np.linalg.norm(err, axis=1)
            ref_mode = np.linalg.norm(q_s_ref, axis=1)
            rel_mode_pct = 100.0 * abs_mode / (ref_mode + 1e-30)

            rows.append(
                {
                    "model": model_tag,
                    "model_path": str(ckpt_path),
                    "reference_source": args.reference_source,
                    "mu1": float(mu1),
                    "mu2": float(mu2),
                    "n_tot": int(ntot),
                    "n_p": int(n_p),
                    "n_s": int(n_s),
                    "nt": int(t.size),
                    "rel_frob_percent": rel_frob_pct,
                    "mean_mode_rel_percent": float(np.mean(rel_mode_pct)),
                    "median_mode_rel_percent": float(np.median(rel_mode_pct)),
                    "p95_mode_rel_percent": float(np.percentile(rel_mode_pct, 95.0)),
                    "max_mode_rel_percent": float(np.max(rel_mode_pct)),
                    "qN_ref_path": str(qn_path),
                }
            )

    csv_path = out_dir / f"case2_offline_errors_{args.reference_source}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_path",
                "reference_source",
                "mu1",
                "mu2",
                "n_tot",
                "n_p",
                "n_s",
                "nt",
                "rel_frob_percent",
                "mean_mode_rel_percent",
                "median_mode_rel_percent",
                "p95_mode_rel_percent",
                "max_mode_rel_percent",
                "qN_ref_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[offline-case2] wrote: {csv_path}")
    if rows:
        rows_sorted = sorted(rows, key=lambda r: (r["model"], r["mu1"], r["mu2"]))
        print("[offline-case2] summary (rel_frob_percent):")
        for r in rows_sorted:
            print(
                "  "
                f"{r['model']:<30s} "
                f"mu=({r['mu1']:.3f},{r['mu2']:.4f}) "
                f"n_p={r['n_p']:>3d} "
                f"relF={r['rel_frob_percent']:.3f}% "
                f"meanMode={r['mean_mode_rel_percent']:.3f}% "
                f"p95Mode={r['p95_mode_rel_percent']:.3f}%"
            )
    else:
        print("[offline-case2] no rows were produced.")

    if missing_refs:
        print("\n[offline-case2] missing references:")
        for model_tag, mu1, mu2, src in missing_refs:
            print(f"  model={model_tag} mu=({mu1:.3f},{mu2:.4f}) source={src}")


if __name__ == "__main__":
    main()

