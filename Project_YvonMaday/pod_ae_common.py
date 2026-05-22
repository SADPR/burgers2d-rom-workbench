#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared PROM-POD-AE model utilities for Project_YvonMaday."""

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
    activation: str = "tanh",
) -> nn.Sequential:
    dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(_activation_module(activation))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PROMPODAEAutoencoder(nn.Module):
    """
    Autoencoder in q-space:
      z = E(scale(q))
      q_hat = unscale(D(z))

    decode_from_latent(z) returns q_hat in raw q-space.
    """

    def __init__(
        self,
        q_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (192, 96, 48),
        scaling: str = "minmax_-1_1",
        activation: str = "tanh",
        q_stats: dict | None = None,
    ):
        super().__init__()
        q_dim = int(q_dim)
        latent_dim = int(latent_dim)

        self.scaling = str(scaling).strip().lower()
        self.activation = str(activation).strip().lower()

        zeros = np.zeros((1, q_dim), dtype=np.float32)
        ones = np.ones((1, q_dim), dtype=np.float32)

        if q_stats is None:
            q_stats = {}

        if self.scaling == "minmax_-1_1":
            q_min = np.asarray(q_stats.get("min", zeros), dtype=np.float32)
            q_max = np.asarray(q_stats.get("max", ones), dtype=np.float32)
            self.scaler = MinMaxScaler(q_min, q_max)
            self.unscaler = MinMaxUnscaler(q_min, q_max)
        elif self.scaling == "zscore":
            q_mean = np.asarray(q_stats.get("mean", zeros), dtype=np.float32)
            q_std = np.asarray(q_stats.get("std", ones), dtype=np.float32)
            self.scaler = ZScoreScaler(q_mean, q_std)
            self.unscaler = ZScoreUnscaler(q_mean, q_std)
        else:
            raise ValueError(f"Unsupported scaling: {scaling}")

        self.encoder = build_mlp(q_dim, hidden_dims, latent_dim, activation=self.activation)
        self.decoder = build_mlp(
            latent_dim,
            tuple(reversed(tuple(int(v) for v in hidden_dims))),
            q_dim,
            activation=self.activation,
        )

    def forward(self, q_raw: torch.Tensor) -> torch.Tensor:
        q_norm = self.scaler(q_raw)
        z = self.encoder(q_norm)
        q_norm_hat = self.decoder(z)
        q_raw_hat = self.unscaler(q_norm_hat)
        return q_raw_hat

    def encode(self, q_raw: torch.Tensor) -> torch.Tensor:
        q_norm = self.scaler(q_raw)
        return self.encoder(q_norm)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        q_norm_hat = self.decoder(z)
        return self.unscaler(q_norm_hat)


def infer_scaling_from_state_dict(state_dict: dict, fallback: str | None = None) -> str:
    keys = set(state_dict.keys())
    if any(k.startswith("scaler.center") for k in keys):
        return "minmax_-1_1"
    if any(k.startswith("scaler.mean") for k in keys):
        return "zscore"
    return str(fallback) if fallback is not None else "minmax_-1_1"


def resolve_activation_from_checkpoint(checkpoint: dict, scaling: str) -> str:
    if "activation" in checkpoint and checkpoint["activation"] is not None:
        return str(checkpoint["activation"])
    # Backward-friendly default.
    return "tanh" if str(scaling).strip().lower() == "minmax_-1_1" else "silu"
