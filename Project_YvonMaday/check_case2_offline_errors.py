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

Usage (from Project_YvonMaday):
  python3 check_case2_offline_errors.py
  python3 check_case2_offline_errors.py --reference-source stage2
  python3 check_case2_offline_errors.py --model-path Results/Stage3/models/case2_model_n20.pt
  python3 check_case2_offline_errors.py --global-coeff 95 --model-path <m1> --model-path <m2>
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from gpr_map_common import build_torch_case2_gpr_from_ckpt
except ModuleNotFoundError:
    from .gpr_map_common import build_torch_case2_gpr_from_ckpt


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


def _make_activation(name: str):
    key = str(name).strip().lower()
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


class CoreMLPLegacy(nn.Module):
    """Legacy fixed-width MLP with fc1..fc6 keys."""

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


class CoreMLP(nn.Module):
    """Configurable MLP with Sequential net.0, net.2, ... keys."""

    def __init__(self, in_dim, out_dim, hidden_dims, activation="elu", dropout=0.0):
        super().__init__()
        dims = [int(in_dim)] + [int(d) for d in hidden_dims] + [int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_make_activation(activation))
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Case2Model(nn.Module):
    """Input: (mu1, mu2, t), output: q_s."""

    def __init__(self, n_s, hidden_dims=None, activation="elu", dropout=0.0, legacy=False):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, 3)), np.ones((1, 3)))
        if legacy:
            self.core = CoreMLPLegacy(3, n_s)
        else:
            if hidden_dims is None:
                hidden_dims = (32, 64, 128, 256, 256)
            self.core = CoreMLP(3, n_s, hidden_dims=hidden_dims, activation=activation, dropout=dropout)
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
    fmt = str(ckpt.get("format", "")).strip().lower()

    if fmt in ("gpr_map", "gpr_map_full"):
        if fmt == "gpr_map_full":
            ntot = int(ckpt.get("dataset_ntot", ckpt.get("n_tot", ckpt.get("out_dim"))))
            n_s = int(ntot)
        else:
            n_s = int(ckpt.get("out_dim", ckpt.get("n_s")))
            ntot = int(ckpt.get("dataset_ntot"))
        in_dim = int(ckpt.get("in_dim", 3))
        if in_dim != 3:
            raise ValueError(f"{ckpt_path}: expected in_dim=3, got {in_dim}.")
        if ntot < n_s:
            raise ValueError(f"{ckpt_path}: invalid split ntot={ntot}, n_s={n_s}.")
        model = build_torch_case2_gpr_from_ckpt(ckpt).to(device)
        model.eval()
        return model, ntot, n_s

    # ANN Case-2 style checkpoint (predicts only q_s)
    if "n_s" in ckpt:
        n_s = int(ckpt["n_s"])
        ntot = int(ckpt.get("dataset_ntot"))
    # Full data-driven ANN checkpoint (predicts full qN)
    elif "n_tot" in ckpt:
        ntot = int(ckpt.get("n_tot", ckpt.get("dataset_ntot")))
        n_s = int(ntot)
    else:
        raise KeyError(f"{ckpt_path}: unsupported checkpoint schema (missing n_s and n_tot).")

    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"{ckpt_path}: expected in_dim=3, got {in_dim}.")
    if ntot < n_s:
        raise ValueError(f"{ckpt_path}: invalid split ntot={ntot}, n_s={n_s}.")

    sd = ckpt["state_dict"]
    has_legacy = any(k.startswith("core.fc1.") for k in sd.keys())
    if has_legacy:
        model = Case2Model(n_s, legacy=True).to(device)
    else:
        hidden_dims = ckpt.get("hidden_dims", (32, 64, 128, 256, 256))
        activation = ckpt.get("activation", "elu")
        dropout = float(ckpt.get("dropout", 0.0))
        model = Case2Model(
            n_s,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            legacy=False,
        ).to(device)
    model.load_state_dict(sd, strict=True)
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


def _parse_global_coeffs(raw_items: Optional[List[str]]) -> List[int]:
    if raw_items is None:
        return []
    out: List[int] = []
    seen = set()
    for item in raw_items:
        for token in str(item).split(","):
            txt = str(token).strip()
            if not txt:
                continue
            val = int(txt)
            if val < 1:
                raise ValueError(f"Invalid --global-coeff '{item}'. Use 1-based positive indices.")
            if val in seen:
                continue
            seen.add(val)
            out.append(val)
    return out


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
        default=None,
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
    parser.add_argument(
        "--global-coeff",
        action="append",
        default=None,
        help=(
            "1-based global qN coefficient index to compare consistently across models "
            "(e.g. --global-coeff 95). Can be passed multiple times."
        ),
    )
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    base_dir = this_dir
    out_dir = (base_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = args.point if args.point is not None else ["4.875,0.0225", "4.56,0.019", "5.19,0.026"]
    # Deduplicate while preserving order.
    seen = set()
    raw_points_unique = []
    for p in raw_points:
        key = str(p).strip()
        if key in seen:
            continue
        seen.add(key)
        raw_points_unique.append(key)
    points = _parse_points(raw_points_unique)
    global_coeffs = _parse_global_coeffs(args.global_coeff)
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
    coeff_rows = []
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

            abs_coeff = np.linalg.norm(err, axis=1)
            ref_coeff = np.linalg.norm(q_s_ref, axis=1)
            rel_coeff_pct = 100.0 * abs_coeff / (ref_coeff + 1e-30)

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
                    "mean_coeff_rel_percent": float(np.mean(rel_coeff_pct)),
                    "median_coeff_rel_percent": float(np.median(rel_coeff_pct)),
                    "p95_coeff_rel_percent": float(np.percentile(rel_coeff_pct, 95.0)),
                    "max_coeff_rel_percent": float(np.max(rel_coeff_pct)),
                    # Backward-compatible aliases used by older scripts/notebooks.
                    "mean_mode_rel_percent": float(np.mean(rel_coeff_pct)),
                    "median_mode_rel_percent": float(np.median(rel_coeff_pct)),
                    "p95_mode_rel_percent": float(np.percentile(rel_coeff_pct, 95.0)),
                    "max_mode_rel_percent": float(np.max(rel_coeff_pct)),
                    "qN_ref_path": str(qn_path),
                }
            )

            # Optional direct comparison of selected global coefficients.
            for gc in global_coeffs:
                idx0 = int(gc) - 1  # global 0-based index in qN
                if idx0 < n_p or idx0 >= ntot:
                    coeff_rows.append(
                        {
                            "model": model_tag,
                            "model_path": str(ckpt_path),
                            "reference_source": args.reference_source,
                            "mu1": float(mu1),
                            "mu2": float(mu2),
                            "n_tot": int(ntot),
                            "n_p": int(n_p),
                            "n_s": int(n_s),
                            "global_coeff_1based": int(gc),
                            "local_qs_coeff_1based": "",
                            "status": "not_predicted_by_map",
                            "rel_coeff_percent": "",
                            "abs_coeff_l2_error": "",
                            "ref_coeff_l2_norm": "",
                        }
                    )
                    continue

                loc = idx0 - n_p  # 0-based index in q_s
                e = q_s_ref[loc, :] - q_s_pred[loc, :]
                e_abs = float(np.linalg.norm(e))
                r_abs = float(np.linalg.norm(q_s_ref[loc, :]))
                rel = 100.0 * e_abs / (r_abs + 1e-30)
                coeff_rows.append(
                    {
                        "model": model_tag,
                        "model_path": str(ckpt_path),
                        "reference_source": args.reference_source,
                        "mu1": float(mu1),
                        "mu2": float(mu2),
                        "n_tot": int(ntot),
                        "n_p": int(n_p),
                        "n_s": int(n_s),
                        "global_coeff_1based": int(gc),
                        "local_qs_coeff_1based": int(loc + 1),
                        "status": "ok",
                        "rel_coeff_percent": float(rel),
                        "abs_coeff_l2_error": float(e_abs),
                        "ref_coeff_l2_norm": float(r_abs),
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
                "mean_coeff_rel_percent",
                "median_coeff_rel_percent",
                "p95_coeff_rel_percent",
                "max_coeff_rel_percent",
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
                f"meanCoeff={r['mean_coeff_rel_percent']:.3f}% "
                f"p95Coeff={r['p95_coeff_rel_percent']:.3f}%"
            )
    else:
        print("[offline-case2] no rows were produced.")

    if global_coeffs:
        coeff_csv_path = out_dir / f"case2_global_coeff_errors_{args.reference_source}.csv"
        with coeff_csv_path.open("w", newline="") as f:
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
                    "global_coeff_1based",
                    "local_qs_coeff_1based",
                    "status",
                    "rel_coeff_percent",
                    "abs_coeff_l2_error",
                    "ref_coeff_l2_norm",
                ],
            )
            writer.writeheader()
            for row in coeff_rows:
                writer.writerow(row)

        print(f"\n[offline-case2] wrote: {coeff_csv_path}")
        ok_rows = [r for r in coeff_rows if r["status"] == "ok"]
        if ok_rows:
            print("[offline-case2] selected global coefficient summary:")
            ok_rows_sorted = sorted(
                ok_rows,
                key=lambda r: (
                    int(r["global_coeff_1based"]),
                    str(r["model"]),
                    float(r["mu1"]),
                    float(r["mu2"]),
                ),
            )
            for r in ok_rows_sorted:
                print(
                    "  "
                    f"gCoeff={int(r['global_coeff_1based']):>3d} "
                    f"{str(r['model']):<30s} "
                    f"mu=({float(r['mu1']):.3f},{float(r['mu2']):.4f}) "
                    f"n_p={int(r['n_p']):>3d} "
                    f"local_qs={int(r['local_qs_coeff_1based']):>3d} "
                    f"relCoeff={float(r['rel_coeff_percent']):.3f}%"
                )
        not_pred = [r for r in coeff_rows if r["status"] != "ok"]
        if not_pred:
            print("[offline-case2] selected global coefficients not predicted by some models:")
            for r in not_pred:
                print(
                    "  "
                    f"gCoeff={int(r['global_coeff_1based']):>3d} "
                    f"{str(r['model']):<30s} "
                    f"mu=({float(r['mu1']):.3f},{float(r['mu2']):.4f}) "
                    f"n_p={int(r['n_p']):>3d} status={r['status']}"
                )

    if missing_refs:
        print("\n[offline-case2] missing references:")
        for model_tag, mu1, mu2, src in missing_refs:
            print(f"  model={model_tag} mu=({mu1:.3f},{mu2:.4f}) source={src}")


if __name__ == "__main__":
    main()
