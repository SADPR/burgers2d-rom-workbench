#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared validation split utilities for Stage-3 trainers."""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.ecsw_utils import build_ecsw_snapshot_plan


def split_indices_ecsw_param_time(
    x_raw: np.ndarray,
    *,
    val_frac: float,
    seed: int,
    snap_time_offset: int = 1,
    ensure_mu_coverage: bool = True,
    round_decimals: int = 12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build train/validation split using ECSW-like parameter-time stratified sampling.

    Parameters
    ----------
    x_raw : ndarray, shape (N,3)
        Input rows [mu1, mu2, t].
    val_frac : float
        Requested validation fraction in (0, 0.5).
    seed : int
        Random seed used by the ECSW planner.
    snap_time_offset : int
        Time offset used by ECSW planner (>=1).

    Returns
    -------
    tr_idx, va_idx, info
        Train and validation integer indices, plus split metadata.
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[1] < 3:
        raise ValueError(f"x_raw must have shape (N,>=3), got {x_raw.shape}")
    if not (0.0 < float(val_frac) < 0.5):
        raise ValueError("val_frac must be in (0, 0.5).")

    n_samples = int(x_raw.shape[0])
    all_idx = np.arange(n_samples, dtype=np.int64)

    mu_key = np.round(x_raw[:, :2], decimals=int(round_decimals))
    mu_points, inv = np.unique(mu_key, axis=0, return_inverse=True)
    num_mu = int(mu_points.shape[0])
    if num_mu < 1:
        raise RuntimeError("No mu-groups found in dataset.")

    rows_sorted_by_mu = []
    n_time = None
    for imu in range(num_mu):
        idx = np.flatnonzero(inv == imu).astype(np.int64)
        if idx.size == 0:
            raise RuntimeError(f"Empty mu-group detected at imu={imu}.")

        tvals = np.asarray(x_raw[idx, 2], dtype=np.float64)
        order = np.argsort(tvals, kind="mergesort")
        idx_sorted = idx[order]
        t_sorted = tvals[order]

        t_key = np.round(t_sorted, decimals=int(round_decimals))
        uniq_t, uniq_pos = np.unique(t_key, return_index=True)
        if uniq_t.size != t_key.size:
            keep = np.sort(uniq_pos)
            idx_sorted = idx_sorted[keep]
            t_sorted = t_sorted[keep]

        if n_time is None:
            n_time = int(idx_sorted.size)
        elif int(idx_sorted.size) != int(n_time):
            raise RuntimeError(
                "All mu-groups must have same number of time samples for ECSW split. "
                f"Got {idx_sorted.size} vs expected {n_time} at imu={imu}."
            )

        if n_time < 2:
            raise RuntimeError("Need at least 2 time samples per mu-group for validation split.")

        rows_sorted_by_mu.append(idx_sorted)

    plan = build_ecsw_snapshot_plan(
        num_steps=int(n_time),
        snap_time_offset=max(1, int(snap_time_offset)),
        num_mu=int(num_mu),
        mode="global_param_time_stratified",
        total_snapshots=None,
        total_snapshots_percent=100.0 * float(val_frac),
        mu_points=np.asarray(mu_points, dtype=np.float64),
        random_seed=int(seed),
        ensure_mu_coverage=bool(ensure_mu_coverage),
    )

    selected_by_mu = plan.get("selected_now_cols_by_mu", [])
    if len(selected_by_mu) != num_mu:
        raise RuntimeError(
            f"ECSW plan returned {len(selected_by_mu)} mu buckets, expected {num_mu}."
        )

    va_parts = []
    for imu, cols in enumerate(selected_by_mu):
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if cols.size == 0:
            continue
        cols = cols[(cols >= 0) & (cols < n_time)]
        if cols.size == 0:
            continue
        va_parts.append(rows_sorted_by_mu[imu][cols])

    if va_parts:
        va_idx = np.unique(np.concatenate(va_parts).astype(np.int64))
    else:
        raise RuntimeError("ECSW split produced zero validation samples.")

    is_train = np.ones((n_samples,), dtype=bool)
    is_train[va_idx] = False
    tr_idx = all_idx[is_train]

    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError(
            f"Invalid split sizes: train={tr_idx.size}, val={va_idx.size}."
        )

    info = {
        "split_mode": "ecsw_param_time_stratified",
        "num_mu_groups": int(num_mu),
        "num_time_per_mu": int(n_time),
        "num_candidates_total": int(plan.get("num_candidates_total", 0)),
        "num_selected_total": int(plan.get("num_selected_total", int(va_idx.size))),
        "val_frac_requested": float(val_frac),
        "val_frac_actual": float(va_idx.size / max(1, n_samples)),
        "snap_time_offset": int(max(1, int(snap_time_offset))),
    }
    return tr_idx.astype(np.int64), va_idx.astype(np.int64), info


def _is_corner_mu(mu: np.ndarray, mu_points: np.ndarray) -> bool:
    mu1_vals = np.unique(np.asarray(mu_points[:, 0], dtype=np.float64))
    mu2_vals = np.unique(np.asarray(mu_points[:, 1], dtype=np.float64))
    mu1, mu2 = float(mu[0]), float(mu[1])
    return (mu1 == float(mu1_vals.min()) or mu1 == float(mu1_vals.max())) and (
        mu2 == float(mu2_vals.min()) or mu2 == float(mu2_vals.max())
    )


def _is_center_mu(mu: np.ndarray, mu_points: np.ndarray) -> bool:
    mu1_vals = np.unique(np.asarray(mu_points[:, 0], dtype=np.float64))
    mu2_vals = np.unique(np.asarray(mu_points[:, 1], dtype=np.float64))
    if (mu1_vals.size % 2) == 0 or (mu2_vals.size % 2) == 0:
        return False
    mu1_mid = float(mu1_vals[mu1_vals.size // 2])
    mu2_mid = float(mu2_vals[mu2_vals.size // 2])
    return float(mu[0]) == mu1_mid and float(mu[1]) == mu2_mid


def split_indices_holdout_mu_group(
    x_raw: np.ndarray,
    *,
    holdout_mu: Optional[Tuple[float, float]] = None,
    avoid_center_and_corners: bool = True,
    round_decimals: int = 12,
    match_atol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build train/validation split by holding out one full (mu1,mu2) trajectory.

    If holdout_mu is None, chooses the first available mu-group that is not
    center/corner when avoid_center_and_corners=True.
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[1] < 3:
        raise ValueError(f"x_raw must have shape (N,>=3), got {x_raw.shape}")

    n_samples = int(x_raw.shape[0])
    all_idx = np.arange(n_samples, dtype=np.int64)
    mu_key = np.round(x_raw[:, :2], decimals=int(round_decimals))
    mu_points, inv = np.unique(mu_key, axis=0, return_inverse=True)
    num_mu = int(mu_points.shape[0])
    if num_mu < 2:
        raise RuntimeError("Need at least 2 mu-groups to hold one out.")

    holdout_gid = None
    if holdout_mu is not None:
        target = np.asarray([holdout_mu[0], holdout_mu[1]], dtype=np.float64)
        matches = np.where(
            np.all(
                np.isclose(
                    np.asarray(mu_points, dtype=np.float64),
                    target[None, :],
                    rtol=0.0,
                    atol=float(match_atol),
                ),
                axis=1,
            )
        )[0]
        if matches.size == 0:
            avail = [tuple(np.asarray(v, dtype=np.float64).tolist()) for v in mu_points]
            raise ValueError(
                f"Requested holdout mu={tuple(target.tolist())} not found in dataset mu groups "
                f"(atol={float(match_atol):.1e}). Available={avail}"
            )
        holdout_gid = int(matches[0])
    else:
        for gid, mu in enumerate(mu_points):
            if avoid_center_and_corners and (_is_corner_mu(mu, mu_points) or _is_center_mu(mu, mu_points)):
                continue
            holdout_gid = int(gid)
            break
        if holdout_gid is None:
            raise RuntimeError(
                "Could not auto-select holdout mu-group outside center/corners. "
                "Please pass --val-holdout-mu explicitly."
            )

    holdout_vec = np.asarray(mu_points[holdout_gid], dtype=np.float64)
    if avoid_center_and_corners:
        if _is_corner_mu(holdout_vec, mu_points):
            raise ValueError(
                f"Holdout mu={tuple(holdout_vec.tolist())} is a corner; "
                "set another --val-holdout-mu."
            )
        if _is_center_mu(holdout_vec, mu_points):
            raise ValueError(
                f"Holdout mu={tuple(holdout_vec.tolist())} is the center point; "
                "set another --val-holdout-mu."
            )

    is_val = (inv == holdout_gid)
    va_idx = np.flatnonzero(is_val).astype(np.int64)
    tr_idx = np.flatnonzero(~is_val).astype(np.int64)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError(
            f"Invalid holdout split sizes: train={tr_idx.size}, val={va_idx.size}."
        )

    # Estimate per-mu time count for logging.
    counts = np.bincount(inv, minlength=num_mu).astype(np.int64)
    n_time = int(counts[holdout_gid])
    if np.any(counts != n_time):
        n_time = int(np.median(counts))

    info = {
        "split_mode": "mu_group_holdout",
        "num_mu_groups": int(num_mu),
        "num_time_per_mu": int(n_time),
        "num_candidates_total": int(n_samples),
        "num_selected_total": int(va_idx.size),
        "val_frac_requested": float(va_idx.size / max(1, n_samples)),
        "val_frac_actual": float(va_idx.size / max(1, n_samples)),
        "snap_time_offset": int(0),
        "holdout_mu1": float(holdout_vec[0]),
        "holdout_mu2": float(holdout_vec[1]),
        "holdout_mu_is_corner": bool(_is_corner_mu(holdout_vec, mu_points)),
        "holdout_mu_is_center": bool(_is_center_mu(holdout_vec, mu_points)),
    }
    return tr_idx.astype(np.int64), va_idx.astype(np.int64), info
