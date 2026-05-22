#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared model utilities for non-intrusive POD-DL training/inference."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def parse_hidden_dims(text: str) -> Tuple[int, ...]:
    raw = [tok.strip() for tok in str(text).split(",")]
    vals = tuple(int(v) for v in raw if len(v) > 0)
    if len(vals) == 0:
        raise ValueError("hidden-dims cannot be empty.")
    if any(v < 1 for v in vals):
        raise ValueError(f"hidden-dims must be positive integers, got {vals}.")
    return vals


def _activation_module(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ZScoreScaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ZScoreUnscaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


class MinMaxScaler(nn.Module):
    def __init__(self, x_min: np.ndarray, x_max: np.ndarray, eps: float = 1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > eps, half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center) / self.half_range


class MinMaxUnscaler(nn.Module):
    def __init__(self, x_min: np.ndarray, x_max: np.ndarray, eps: float = 1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > eps, half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.half_range + self.center


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "elu",
) -> nn.Sequential:
    dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(_activation_module(activation))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _build_scaler_pair(scaling: str, stats: dict):
    sc = str(scaling).strip().lower()
    if sc == "zscore":
        return ZScoreScaler(stats["mean"], stats["std"]), ZScoreUnscaler(stats["mean"], stats["std"])
    if sc == "minmax_-1_1":
        return MinMaxScaler(stats["min"], stats["max"]), MinMaxUnscaler(stats["min"], stats["max"])
    raise ValueError(f"Unsupported scaling: {scaling}")


class PODDLDataDrivenModel(nn.Module):
    """
    POD-DL data-driven network:
      z_pred = phi(mu1,mu2,t)
      q_pred = D(z_pred)
    with an encoder E used in training:
      z_enc = E(q_true), q_rec = D(z_enc)
    """

    def __init__(
        self,
        q_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Sequence[int] = (256, 128),
        decoder_hidden_dims: Sequence[int] = (128, 256),
        dynamics_hidden_dims: Sequence[int] = (64, 128, 128),
        activation: str = "elu",
        x_scaling: str = "zscore",
        q_scaling: str = "zscore",
        x_stats: dict | None = None,
        q_stats: dict | None = None,
    ):
        super().__init__()
        q_dim = int(q_dim)
        latent_dim = int(latent_dim)

        if x_stats is None:
            if str(x_scaling).strip().lower() == "minmax_-1_1":
                x_stats = {
                    "min": np.zeros((1, 3), dtype=np.float32),
                    "max": np.ones((1, 3), dtype=np.float32),
                }
            else:
                x_stats = {
                    "mean": np.zeros((1, 3), dtype=np.float32),
                    "std": np.ones((1, 3), dtype=np.float32),
                }
        if q_stats is None:
            if str(q_scaling).strip().lower() == "minmax_-1_1":
                q_stats = {
                    "min": np.zeros((1, q_dim), dtype=np.float32),
                    "max": np.ones((1, q_dim), dtype=np.float32),
                }
            else:
                q_stats = {
                    "mean": np.zeros((1, q_dim), dtype=np.float32),
                    "std": np.ones((1, q_dim), dtype=np.float32),
                }

        self.x_scaling = str(x_scaling).strip().lower()
        self.q_scaling = str(q_scaling).strip().lower()
        self.activation = str(activation).strip().lower()

        self.x_scaler, _ = _build_scaler_pair(self.x_scaling, x_stats)
        self.q_scaler, self.q_unscaler = _build_scaler_pair(self.q_scaling, q_stats)

        self.encoder = build_mlp(
            in_dim=q_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=latent_dim,
            activation=self.activation,
        )
        self.decoder = build_mlp(
            in_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            out_dim=q_dim,
            activation=self.activation,
        )
        self.dynamics = build_mlp(
            in_dim=3,
            hidden_dims=dynamics_hidden_dims,
            out_dim=latent_dim,
            activation=self.activation,
        )

    def encode_q(self, q_raw: torch.Tensor) -> torch.Tensor:
        q_n = self.q_scaler(q_raw)
        return self.encoder(q_n)

    def decode_z(self, z: torch.Tensor) -> torch.Tensor:
        q_n = self.decoder(z)
        return self.q_unscaler(q_n)

    def predict_z_from_x(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_n = self.x_scaler(x_raw)
        return self.dynamics(x_n)

    def predict_q_from_x(self, x_raw: torch.Tensor) -> torch.Tensor:
        z = self.predict_z_from_x(x_raw)
        return self.decode_z(z)

    def reconstruct_q(self, q_raw: torch.Tensor) -> torch.Tensor:
        z = self.encode_q(q_raw)
        return self.decode_z(z)

    def forward(self, x_raw: torch.Tensor, q_raw: torch.Tensor | None = None, return_terms: bool = False):
        z_pred = self.predict_z_from_x(x_raw)
        q_pred = self.decode_z(z_pred)

        if q_raw is None:
            if return_terms:
                return {"q_pred": q_pred, "z_pred": z_pred}
            return q_pred

        z_enc = self.encode_q(q_raw)
        q_rec = self.decode_z(z_enc)
        if return_terms:
            return {"q_pred": q_pred, "z_pred": z_pred, "z_enc": z_enc, "q_rec": q_rec}
        return q_pred, z_pred, z_enc, q_rec
