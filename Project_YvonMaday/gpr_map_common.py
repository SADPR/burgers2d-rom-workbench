#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for Case-2 GPR map training and inference."""

from __future__ import annotations

import math
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF


def parse_csv_floats(txt: str):
    vals = [s.strip() for s in str(txt).split(",")]
    out = []
    for v in vals:
        if not v:
            continue
        out.append(float(v))
    if not out:
        raise ValueError("Expected at least one numeric value.")
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


def remove_near_duplicates(x: np.ndarray, tol: float) -> np.ndarray:
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


def rel_frob_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    den = np.linalg.norm(y_true, ord="fro")
    if den <= 0.0:
        return float("nan")
    return 100.0 * float(np.linalg.norm(y_true - y_pred, ord="fro") / den)


def parse_bounds_csv(txt: str, name: str) -> Tuple[float, float]:
    vals = parse_csv_floats(txt)
    if len(vals) != 2:
        raise ValueError(f"{name} must contain exactly two floats: min,max")
    lo = float(vals[0])
    hi = float(vals[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi <= 0.0 or lo >= hi:
        raise ValueError(f"Invalid {name}: got ({lo}, {hi})")
    return lo, hi


def build_kernel(
    *,
    kernel_name: str,
    constant_value: float,
    constant_bounds: Tuple[float, float],
    length_scale: float,
    length_bounds: Tuple[float, float],
    input_dim: int = 1,
    ard: bool = False,
):
    kernel_name = str(kernel_name).strip().lower()
    input_dim = int(input_dim)
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}.")

    if bool(ard):
        ls = np.full((input_dim,), float(length_scale), dtype=np.float64)
        lb = np.tile(
            np.asarray([[float(length_bounds[0]), float(length_bounds[1])]], dtype=np.float64),
            (input_dim, 1),
        )
    else:
        ls = float(length_scale)
        lb = (float(length_bounds[0]), float(length_bounds[1]))

    base = ConstantKernel(
        constant_value=float(constant_value),
        constant_value_bounds=(float(constant_bounds[0]), float(constant_bounds[1])),
    )
    if kernel_name == "matern15":
        return base * Matern(
            length_scale=ls,
            length_scale_bounds=lb,
            nu=1.5,
        )
    if kernel_name == "rbf":
        return base * RBF(
            length_scale=ls,
            length_scale_bounds=lb,
        )
    raise ValueError("kernel_name must be one of: 'matern15', 'rbf'.")


def serialize_gpr_model(gpr_model) -> bytes:
    return pickle.dumps(gpr_model, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_gpr_model(payload: bytes):
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"Invalid GPR payload type: {type(payload)}")
    return pickle.loads(payload)


class TorchCase2GPRMap(nn.Module):
    """Torch wrapper around sklearn GPR for Case-2 map q_s(mu1,mu2,t)."""

    def __init__(self, *, gpr_model, x_stats: Dict[str, np.ndarray], y_stats: Dict[str, np.ndarray]):
        super().__init__()
        self.gpr_model = gpr_model
        self.x_scaling = str(x_stats["mode"]).strip().lower()
        self.y_scaling = str(y_stats["mode"]).strip().lower()

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
            raise ValueError(f"Unsupported x_scaling mode: {self.x_scaling}")

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
            raise ValueError(f"Unsupported y_scaling mode: {self.y_scaling}")

    def _scale_x(self, x_np: np.ndarray) -> np.ndarray:
        if self.x_scaling == "zscore":
            return (x_np - self.x_mean.detach().cpu().numpy()[None, :]) / self.x_std.detach().cpu().numpy()[None, :]
        return 2.0 * (x_np - self.x_min.detach().cpu().numpy()[None, :]) / self.x_span.detach().cpu().numpy()[None, :] - 1.0

    def _unscale_y(self, y_np: np.ndarray) -> np.ndarray:
        if self.y_scaling == "zscore":
            return y_np * self.y_std.detach().cpu().numpy()[None, :] + self.y_mean.detach().cpu().numpy()[None, :]
        return 0.5 * (y_np + 1.0) * self.y_span.detach().cpu().numpy()[None, :] + self.y_min.detach().cpu().numpy()[None, :]

    def forward(self, x):
        if torch.is_tensor(x):
            x_t = x
            in_device = x_t.device
            x_np = x_t.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            in_device = torch.device("cpu")
            x_np = np.asarray(x, dtype=np.float64)

        squeeze_out = False
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            squeeze_out = True
        if x_np.ndim != 2 or x_np.shape[1] != 3:
            raise ValueError(f"Case2 GPR input must have shape (N,3). Got {x_np.shape}.")

        x_n = self._scale_x(x_np)
        y_n = self.gpr_model.predict(x_n)
        y_n = np.asarray(y_n, dtype=np.float64)
        if y_n.ndim == 1:
            y_n = y_n[:, None]
        y = self._unscale_y(y_n).astype(np.float32, copy=False)

        out = torch.from_numpy(y)
        if squeeze_out:
            out = out.reshape(-1)
        return out.to(in_device)


def _rbf_ard_kernel(
    x: np.ndarray,
    z: np.ndarray,
    lengthscales: np.ndarray,
    outputscale: float,
) -> np.ndarray:
    ls = np.maximum(np.asarray(lengthscales, dtype=np.float64), 1e-14)
    inv_l2 = 1.0 / (ls * ls)
    x2 = np.sum((x * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True)
    z2 = np.sum((z * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True).T
    cross = (x * inv_l2[None, :]) @ z.T
    sq = np.maximum(x2 + z2 - 2.0 * cross, 0.0)
    return float(outputscale) * np.exp(-0.5 * sq)


def _matern15_ard_kernel(
    x: np.ndarray,
    z: np.ndarray,
    lengthscales: np.ndarray,
    outputscale: float,
) -> np.ndarray:
    ls = np.maximum(np.asarray(lengthscales, dtype=np.float64), 1e-14)
    inv_l2 = 1.0 / (ls * ls)
    x2 = np.sum((x * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True)
    z2 = np.sum((z * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True).T
    cross = (x * inv_l2[None, :]) @ z.T
    sq = np.maximum(x2 + z2 - 2.0 * cross, 0.0)
    r = np.sqrt(3.0 * sq)
    return float(outputscale) * (1.0 + r) * np.exp(-r)


def _kernel_ard_matrix(
    x: np.ndarray,
    z: np.ndarray,
    *,
    lengthscales: np.ndarray,
    outputscale: float,
    kernel_name: str,
) -> np.ndarray:
    key = str(kernel_name).strip().lower()
    if key == "rbf":
        return _rbf_ard_kernel(x, z, lengthscales, outputscale)
    if key == "matern15":
        return _matern15_ard_kernel(x, z, lengthscales, outputscale)
    raise ValueError(f"Unsupported sparse kernel_name='{kernel_name}'.")


class TorchCase2SparseGPRMap(nn.Module):
    """Torch wrapper around exported sparse-GPR map for Case-2 style inputs."""

    def __init__(
        self,
        *,
        sparse_payload: Dict[str, object],
        x_stats: Dict[str, np.ndarray],
        y_stats: Dict[str, np.ndarray],
    ):
        super().__init__()
        self.kernel_name = str(sparse_payload.get("kernel_name", "rbf")).strip().lower()
        self.x_scaling = str(x_stats["mode"]).strip().lower()
        self.y_scaling = str(y_stats["mode"]).strip().lower()

        self.inducing_points = np.asarray(sparse_payload["inducing_points"], dtype=np.float64)
        self.alpha = np.asarray(sparse_payload["alpha"], dtype=np.float64)
        self.lengthscales = np.asarray(sparse_payload["lengthscales"], dtype=np.float64)
        self.outputscales = np.asarray(sparse_payload["outputscales"], dtype=np.float64)

        if self.inducing_points.ndim != 3:
            raise ValueError(f"inducing_points must be (out_dim,m,in_dim), got {self.inducing_points.shape}")
        if self.alpha.ndim != 2:
            raise ValueError(f"alpha must be (out_dim,m), got {self.alpha.shape}")
        if self.lengthscales.ndim != 2:
            raise ValueError(f"lengthscales must be (out_dim,in_dim), got {self.lengthscales.shape}")
        if self.outputscales.ndim != 1:
            raise ValueError(f"outputscales must be (out_dim,), got {self.outputscales.shape}")

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
            raise ValueError(f"Unsupported x_scaling mode: {self.x_scaling}")

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
            raise ValueError(f"Unsupported y_scaling mode: {self.y_scaling}")

    def _scale_x(self, x_np: np.ndarray) -> np.ndarray:
        if self.x_scaling == "zscore":
            return (x_np - self.x_mean.detach().cpu().numpy()[None, :]) / self.x_std.detach().cpu().numpy()[None, :]
        return 2.0 * (x_np - self.x_min.detach().cpu().numpy()[None, :]) / self.x_span.detach().cpu().numpy()[None, :] - 1.0

    def _unscale_y(self, y_np: np.ndarray) -> np.ndarray:
        if self.y_scaling == "zscore":
            return y_np * self.y_std.detach().cpu().numpy()[None, :] + self.y_mean.detach().cpu().numpy()[None, :]
        return 0.5 * (y_np + 1.0) * self.y_span.detach().cpu().numpy()[None, :] + self.y_min.detach().cpu().numpy()[None, :]

    def forward(self, x):
        if torch.is_tensor(x):
            in_device = x.device
            x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            in_device = torch.device("cpu")
            x_np = np.asarray(x, dtype=np.float64)

        squeeze_out = False
        if x_np.ndim == 1:
            x_np = x_np[None, :]
            squeeze_out = True
        if x_np.ndim != 2 or x_np.shape[1] != 3:
            raise ValueError(f"Case2 sparse-GPR input must have shape (N,3). Got {x_np.shape}.")

        x_n = self._scale_x(x_np)
        n_eval = int(x_n.shape[0])
        out_dim = int(self.alpha.shape[0])
        y_n = np.zeros((n_eval, out_dim), dtype=np.float64)
        for j in range(out_dim):
            kxz = _kernel_ard_matrix(
                x_n,
                self.inducing_points[j, :, :],
                lengthscales=self.lengthscales[j, :],
                outputscale=float(self.outputscales[j]),
                kernel_name=self.kernel_name,
            )
            y_n[:, j] = kxz @ self.alpha[j, :]

        y = self._unscale_y(y_n).astype(np.float32, copy=False)
        out = torch.from_numpy(y)
        if squeeze_out:
            out = out.reshape(-1)
        return out.to(in_device)


def build_torch_case2_gpr_from_ckpt(ckpt: Dict[str, object]) -> nn.Module:
    x_stats = ckpt.get("x_stats", None)
    y_stats = ckpt.get("y_stats", None)
    if x_stats is None or y_stats is None:
        raise KeyError("Checkpoint missing x_stats and/or y_stats.")

    fmt = str(ckpt.get("format", "")).strip().lower()
    if fmt in ("sparse_gpr_map", "sparse_gpr_map_full") or ("sparse_gp_payload" in ckpt):
        sparse_payload = ckpt.get("sparse_gp_payload", None)
        if sparse_payload is None:
            raise KeyError("Sparse-GPR checkpoint missing 'sparse_gp_payload'.")
        return TorchCase2SparseGPRMap(
            sparse_payload=sparse_payload,
            x_stats=x_stats,
            y_stats=y_stats,
        )

    payload = ckpt.get("gpr_payload", None)
    if payload is None:
        raise KeyError("Checkpoint missing 'gpr_payload'.")
    gpr_model = deserialize_gpr_model(payload)
    return TorchCase2GPRMap(gpr_model=gpr_model, x_stats=x_stats, y_stats=y_stats)
