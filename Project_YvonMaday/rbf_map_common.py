#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for Case-1/2/3 RBF map training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def parse_csv_floats(txt: str) -> List[float]:
    vals = [s.strip() for s in str(txt).split(",")]
    out: List[float] = []
    for v in vals:
        if not v:
            continue
        out.append(float(v))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def parse_csv_strings(txt: str) -> List[str]:
    vals = [s.strip() for s in str(txt).split(",")]
    out = [v for v in vals if v]
    if not out:
        raise ValueError("Expected at least one string value.")
    return out


def _safe_std(std: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(np.asarray(std, dtype=np.float64), eps)


def fit_scaler_stats(x: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    m = str(mode).strip().lower()
    if m == "zscore":
        return {"mode": m, "mean": x.mean(axis=0), "std": _safe_std(x.std(axis=0))}
    if m == "minmax_-1_1":
        xmin = x.min(axis=0)
        xmax = x.max(axis=0)
        span = np.maximum(xmax - xmin, 1e-12)
        return {"mode": m, "min": xmin, "max": xmax, "span": span}
    raise ValueError(f"Unsupported scaling mode '{mode}'. Use 'zscore' or 'minmax_-1_1'.")


def apply_scaler(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mode = str(stats["mode"]).strip().lower()
    if mode == "zscore":
        return (x - stats["mean"]) / stats["std"]
    if mode == "minmax_-1_1":
        return 2.0 * (x - stats["min"]) / stats["span"] - 1.0
    raise ValueError(f"Unsupported scaler mode '{mode}'.")


def invert_scaler(y_scaled: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    mode = str(stats["mode"]).strip().lower()
    if mode == "zscore":
        return y_scaled * stats["std"] + stats["mean"]
    if mode == "minmax_-1_1":
        return 0.5 * (y_scaled + 1.0) * stats["span"] + stats["min"]
    raise ValueError(f"Unsupported scaler mode '{mode}'.")


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def kernel_matrix_from_dist(d: np.ndarray, kernel_name: str, epsilon: float) -> np.ndarray:
    k = str(kernel_name).strip().lower()
    e = float(epsilon)
    if e <= 0.0:
        raise ValueError(f"epsilon must be positive, got {e}.")

    de = e * np.asarray(d, dtype=np.float64)
    if k == "gaussian":
        return np.exp(-(de ** 2))
    if k == "imq":
        return 1.0 / np.sqrt(1.0 + de ** 2)
    if k == "multiquadric":
        return np.sqrt(1.0 + de ** 2)
    if k == "linear":
        return np.asarray(d, dtype=np.float64)
    if k == "matern":
        z = math.sqrt(3.0) * de
        return (1.0 + z) * np.exp(-z)
    raise ValueError(
        f"Unsupported kernel '{kernel_name}'. Use one of: gaussian, imq, multiquadric, linear, matern."
    )


def _remove_near_duplicates(x: np.ndarray, tol: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if tol <= 0.0:
        return np.ones(x.shape[0], dtype=bool)
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        di = np.linalg.norm(x[(i + 1) :, :] - x[i : i + 1, :], axis=1)
        keep[(i + 1) :][di < tol] = False
    return keep


@dataclass
class RBFGridResult:
    kernel_name: str
    epsilon: float
    lambda_reg: float
    train_rel_frob_percent: float
    val_rel_frob_percent: float
    train_mse: float
    val_mse: float
    n_train_used: int
    n_centers: int


def _fit_rbf_weights_ridge(phi: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lam = float(lambda_reg)
    if lam < 0.0:
        raise ValueError(f"lambda_reg must be >= 0, got {lam}.")
    a = phi.T @ phi
    a.flat[:: a.shape[0] + 1] += lam
    b = phi.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(a, b, rcond=None)[0]
    return w


def _rel_frob_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    den = np.linalg.norm(y_true, ord="fro")
    if den <= 0.0:
        return float("nan")
    return 100.0 * float(np.linalg.norm(y_true - y_pred, ord="fro") / den)


def train_rbf_grid_map(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    kernel_candidates: Sequence[str] = ("imq", "gaussian", "matern"),
    epsilon_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    lambda_grid: Sequence[float] = (1e-10, 1e-8, 1e-6, 1e-4),
    x_scaling: str = "minmax_-1_1",
    y_scaling: str = "zscore",
    duplicate_tol: float = 0.0,
    max_centers: int = 1200,
) -> Dict[str, object]:
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    if x_raw.ndim != 2 or y_raw.ndim != 2:
        raise ValueError(f"x_raw and y_raw must be 2D. Got {x_raw.shape}, {y_raw.shape}.")
    if x_raw.shape[0] != y_raw.shape[0]:
        raise ValueError(f"Sample mismatch: x has {x_raw.shape[0]}, y has {y_raw.shape[0]}.")
    if not (0.0 < float(val_frac) < 0.5):
        raise ValueError(f"val_frac must be in (0,0.5), got {val_frac}.")

    n_samples = x_raw.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n_samples)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if tr_idx.size < 2:
        raise RuntimeError("Not enough training samples after split.")

    x_tr_raw = x_raw[tr_idx]
    y_tr_raw = y_raw[tr_idx]
    x_va_raw = x_raw[val_idx]
    y_va_raw = y_raw[val_idx]

    x_stats = fit_scaler_stats(x_tr_raw, x_scaling)
    y_stats = fit_scaler_stats(y_tr_raw, y_scaling)

    x_tr = apply_scaler(x_tr_raw, x_stats)
    y_tr = apply_scaler(y_tr_raw, y_stats)
    x_va = apply_scaler(x_va_raw, x_stats)
    y_va = apply_scaler(y_va_raw, y_stats)

    keep = _remove_near_duplicates(x_tr, float(duplicate_tol))
    x_tr = x_tr[keep]
    y_tr = y_tr[keep]
    n_train_used = int(x_tr.shape[0])
    if n_train_used < 2:
        raise RuntimeError("Not enough training points after duplicate filtering.")

    n_centers = min(int(max_centers), n_train_used)
    if n_centers < 2:
        raise ValueError(f"max_centers must allow at least 2 centers, got {max_centers}.")
    if n_centers < n_train_used:
        center_idx = rng.choice(n_train_used, size=n_centers, replace=False)
        center_idx.sort()
    else:
        center_idx = np.arange(n_train_used, dtype=np.int64)

    centers = x_tr[center_idx]
    d_tr = pairwise_distances(x_tr, centers)
    d_va = pairwise_distances(x_va, centers)

    grid_results: List[RBFGridResult] = []
    best: Dict[str, object] | None = None

    for kernel_name in kernel_candidates:
        for epsilon in epsilon_grid:
            for lambda_reg in lambda_grid:
                phi_tr = kernel_matrix_from_dist(d_tr, kernel_name, float(epsilon))
                phi_va = kernel_matrix_from_dist(d_va, kernel_name, float(epsilon))
                w = _fit_rbf_weights_ridge(phi_tr, y_tr, float(lambda_reg))

                yhat_tr = phi_tr @ w
                yhat_va = phi_va @ w

                tr_rel = _rel_frob_percent(y_tr, yhat_tr)
                va_rel = _rel_frob_percent(y_va, yhat_va)
                tr_mse = float(np.mean((yhat_tr - y_tr) ** 2))
                va_mse = float(np.mean((yhat_va - y_va) ** 2))

                row = RBFGridResult(
                    kernel_name=str(kernel_name),
                    epsilon=float(epsilon),
                    lambda_reg=float(lambda_reg),
                    train_rel_frob_percent=float(tr_rel),
                    val_rel_frob_percent=float(va_rel),
                    train_mse=tr_mse,
                    val_mse=va_mse,
                    n_train_used=n_train_used,
                    n_centers=n_centers,
                )
                grid_results.append(row)

                if best is None:
                    best = {"row": row, "w": w}
                else:
                    best_row: RBFGridResult = best["row"]
                    if row.val_rel_frob_percent < best_row.val_rel_frob_percent:
                        best = {"row": row, "w": w}

    if best is None:
        raise RuntimeError("RBF grid search produced no candidate.")

    best_row: RBFGridResult = best["row"]

    # Refit final model on all data with chosen hyperparameters.
    x_all_stats = fit_scaler_stats(x_raw, x_scaling)
    y_all_stats = fit_scaler_stats(y_raw, y_scaling)
    x_all = apply_scaler(x_raw, x_all_stats)
    y_all = apply_scaler(y_raw, y_all_stats)

    keep_all = _remove_near_duplicates(x_all, float(duplicate_tol))
    x_all = x_all[keep_all]
    y_all = y_all[keep_all]

    n_all_used = int(x_all.shape[0])
    n_centers_final = min(int(max_centers), n_all_used)
    if n_centers_final < n_all_used:
        center_idx_all = rng.choice(n_all_used, size=n_centers_final, replace=False)
        center_idx_all.sort()
    else:
        center_idx_all = np.arange(n_all_used, dtype=np.int64)

    centers_all = x_all[center_idx_all]
    d_all = pairwise_distances(x_all, centers_all)
    phi_all = kernel_matrix_from_dist(d_all, best_row.kernel_name, best_row.epsilon)
    w_all = _fit_rbf_weights_ridge(phi_all, y_all, best_row.lambda_reg)
    yhat_all = phi_all @ w_all
    fit_rel = _rel_frob_percent(y_all, yhat_all)
    fit_mse = float(np.mean((yhat_all - y_all) ** 2))

    return {
        "x_scaling": str(x_scaling).strip().lower(),
        "y_scaling": str(y_scaling).strip().lower(),
        "x_stats": x_all_stats,
        "y_stats": y_all_stats,
        "centers_norm": centers_all,
        "W": w_all,
        "kernel_name": best_row.kernel_name,
        "epsilon": float(best_row.epsilon),
        "lambda_reg": float(best_row.lambda_reg),
        "train_val_split_seed": int(seed),
        "val_frac": float(val_frac),
        "duplicate_tol": float(duplicate_tol),
        "max_centers": int(max_centers),
        "n_samples_total": int(n_samples),
        "n_samples_train_split": int(n_train_used),
        "n_samples_all_final": int(n_all_used),
        "n_centers_final": int(n_centers_final),
        "best_val_rel_frob_percent": float(best_row.val_rel_frob_percent),
        "best_val_mse": float(best_row.val_mse),
        "final_fit_rel_frob_percent": float(fit_rel),
        "final_fit_mse": float(fit_mse),
        "grid_results": [r.__dict__ for r in grid_results],
    }


class TorchRBFMap(nn.Module):
    """Differentiable RBF map used by online ROM solvers."""

    def __init__(
        self,
        *,
        centers_norm: np.ndarray,
        weights: np.ndarray,
        kernel_name: str,
        epsilon: float,
        x_scaling: str,
        y_scaling: str,
        x_stats: Dict[str, np.ndarray],
        y_stats: Dict[str, np.ndarray],
    ):
        super().__init__()
        centers_norm = np.asarray(centers_norm, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if centers_norm.ndim != 2 or weights.ndim != 2:
            raise ValueError(
                f"centers_norm and weights must be 2D. Got {centers_norm.shape}, {weights.shape}."
            )
        if centers_norm.shape[0] != weights.shape[0]:
            raise ValueError(
                f"Center/weight mismatch: {centers_norm.shape[0]} centers vs {weights.shape[0]} rows in W."
            )

        self.register_buffer("centers_norm", torch.tensor(centers_norm, dtype=torch.float32))
        self.register_buffer("W", torch.tensor(weights, dtype=torch.float32))
        self.kernel_name = str(kernel_name).strip().lower()
        self.epsilon = float(epsilon)
        self.x_scaling = str(x_scaling).strip().lower()
        self.y_scaling = str(y_scaling).strip().lower()

        # Non-trainable anchor parameter so utility code can discover/move device safely.
        self._device_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

        if self.x_scaling == "zscore":
            self.register_buffer("x_mean", torch.tensor(np.asarray(x_stats["mean"], dtype=np.float32)))
            self.register_buffer("x_std", torch.tensor(np.asarray(x_stats["std"], dtype=np.float32)))
            self.register_buffer("x_min", torch.zeros_like(self.x_mean))
            self.register_buffer("x_span", torch.ones_like(self.x_mean))
        elif self.x_scaling == "minmax_-1_1":
            self.register_buffer("x_min", torch.tensor(np.asarray(x_stats["min"], dtype=np.float32)))
            self.register_buffer("x_span", torch.tensor(np.asarray(x_stats["span"], dtype=np.float32)))
            self.register_buffer("x_mean", torch.zeros_like(self.x_min))
            self.register_buffer("x_std", torch.ones_like(self.x_min))
        else:
            raise ValueError(f"Unsupported x_scaling '{self.x_scaling}'.")

        if self.y_scaling == "zscore":
            self.register_buffer("y_mean", torch.tensor(np.asarray(y_stats["mean"], dtype=np.float32)))
            self.register_buffer("y_std", torch.tensor(np.asarray(y_stats["std"], dtype=np.float32)))
            self.register_buffer("y_min", torch.zeros_like(self.y_mean))
            self.register_buffer("y_span", torch.ones_like(self.y_mean))
        elif self.y_scaling == "minmax_-1_1":
            self.register_buffer("y_min", torch.tensor(np.asarray(y_stats["min"], dtype=np.float32)))
            self.register_buffer("y_span", torch.tensor(np.asarray(y_stats["span"], dtype=np.float32)))
            self.register_buffer("y_mean", torch.zeros_like(self.y_min))
            self.register_buffer("y_std", torch.ones_like(self.y_min))
        else:
            raise ValueError(f"Unsupported y_scaling '{self.y_scaling}'.")

    def _scale_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_scaling == "zscore":
            return (x - self.x_mean) / self.x_std
        return 2.0 * (x - self.x_min) / self.x_span - 1.0

    def _unscale_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.y_scaling == "zscore":
            return y * self.y_std + self.y_mean
        return 0.5 * (y + 1.0) * self.y_span + self.y_min

    def _kernel(self, d: torch.Tensor) -> torch.Tensor:
        de = self.epsilon * d
        if self.kernel_name == "gaussian":
            return torch.exp(-(de ** 2))
        if self.kernel_name == "imq":
            return 1.0 / torch.sqrt(1.0 + de ** 2)
        if self.kernel_name == "multiquadric":
            return torch.sqrt(1.0 + de ** 2)
        if self.kernel_name == "linear":
            return d
        if self.kernel_name == "matern":
            z = math.sqrt(3.0) * de
            return (1.0 + z) * torch.exp(-z)
        raise ValueError(f"Unsupported kernel '{self.kernel_name}'.")

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x_raw.ndim == 1:
            x_raw = x_raw.unsqueeze(0)
            squeeze = True
        x = self._scale_x(x_raw)
        d = torch.cdist(x, self.centers_norm, p=2.0)
        phi = self._kernel(d)
        y_scaled = phi @ self.W
        y = self._unscale_y(y_scaled)
        if squeeze:
            y = y.squeeze(0)
        return y


def build_torch_rbf_from_ckpt(ckpt: Dict[str, object]) -> TorchRBFMap:
    return TorchRBFMap(
        centers_norm=np.asarray(ckpt["centers_norm"], dtype=np.float32),
        weights=np.asarray(ckpt["W"], dtype=np.float32),
        kernel_name=str(ckpt["kernel_name"]),
        epsilon=float(ckpt["epsilon"]),
        x_scaling=str(ckpt["x_scaling"]),
        y_scaling=str(ckpt["y_scaling"]),
        x_stats=ckpt["x_stats"],
        y_stats=ckpt["y_stats"],
    )

