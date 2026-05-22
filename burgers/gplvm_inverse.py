# -*- coding: utf-8 -*-
"""
Utilities for robust latent-variable inverse solves used by POD-GPLVM.

Main use cases:
  - Infer z from q_target by solving min_z ||q(z) - q_target||^2.
  - Infer z from full/restricted state by solving min_z ||w(z) - w_target||^2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def latent_box_bounds(
    z_cloud,
    margin_rel=0.20,
    margin_abs=0.25,
):
    """
    Build conservative per-dimension bounds from a latent cloud.

    Parameters
    ----------
    z_cloud : ndarray, shape (n_samples, n_latent)
        Reference latent samples (typically Z_train).
    margin_rel : float
        Relative padding with respect to per-dimension span.
    margin_abs : float
        Absolute minimum padding.
    """
    z_cloud = np.asarray(z_cloud, dtype=np.float64)
    if z_cloud.ndim != 2 or z_cloud.shape[0] < 1:
        raise ValueError("z_cloud must be a 2D array with at least one row.")

    z_min = np.min(z_cloud, axis=0)
    z_max = np.max(z_cloud, axis=0)
    span = z_max - z_min

    rel = float(max(margin_rel, 0.0))
    absv = float(max(margin_abs, 0.0))

    pad = np.maximum(absv, rel * np.where(span > 0.0, span, 1.0))
    lb = z_min - pad
    ub = z_max + pad

    # Keep a strictly valid interval in degenerate dimensions.
    bad = ub <= lb
    if np.any(bad):
        eps = 1e-8 * np.maximum(1.0, np.abs(lb[bad]))
        ub[bad] = lb[bad] + eps

    return lb.astype(np.float64), ub.astype(np.float64)


def nearest_seed_indices(
    target,
    train_points,
    n_starts=5,
):
    """
    Return nearest-neighbor seed indices in Euclidean distance.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    train_points = np.asarray(train_points, dtype=np.float64)

    if train_points.ndim != 2 or train_points.shape[0] < 1:
        raise ValueError("train_points must be a 2D array with at least one row.")
    if train_points.shape[1] != target.size:
        raise ValueError(
            "target/train_points dimension mismatch: "
            f"{target.size} vs {train_points.shape[1]}"
        )

    n = train_points.shape[0]
    k = int(max(1, min(int(n_starts), n)))

    d2 = np.sum((train_points - target[None, :]) ** 2, axis=1)
    if k == n:
        idx = np.argsort(d2)
    else:
        part = np.argpartition(d2, kth=k - 1)[:k]
        idx = part[np.argsort(d2[part])]

    return idx.astype(np.int64), d2[idx]


def _clip_to_bounds(x, lb, ub):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)

    if x.size != lb.size or x.size != ub.size:
        raise ValueError("x/lb/ub size mismatch.")
    if np.any(ub <= lb):
        raise ValueError("Invalid bounds: require ub > lb component-wise.")

    # Stay strictly inside when possible (TRF may adjust anyway, this avoids edge glitches).
    eps = 1e-12 * np.maximum(1.0, np.maximum(np.abs(lb), np.abs(ub)))
    lo = lb + eps
    hi = ub - eps
    mask = hi <= lo
    lo[mask] = lb[mask]
    hi[mask] = ub[mask]
    return np.minimum(np.maximum(x, lo), hi)


def solve_bounded_nls(
    x0,
    residual_func,
    jac_func=None,
    lb=None,
    ub=None,
    prior_center=None,
    prior_weight=0.0,
    max_nfev=200,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-8,
    loss="linear",
    f_scale=1.0,
):
    """
    Solve a bounded nonlinear least-squares inverse problem.

    Minimizes:
        ||r(x)||_2^2 + prior_weight * ||x - prior_center||_2^2

    where r(x) is provided by `residual_func`.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    n = x0.size

    if lb is None:
        lb = np.full(n, -np.inf, dtype=np.float64)
    if ub is None:
        ub = np.full(n, np.inf, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    if lb.size != n or ub.size != n:
        raise ValueError("Bounds size mismatch with x0.")
    if np.any(ub <= lb):
        raise ValueError("Invalid bounds: require ub > lb component-wise.")

    x0 = _clip_to_bounds(x0, lb, ub)

    reg = float(prior_weight)
    if reg < 0.0:
        raise ValueError("prior_weight must be non-negative.")
    use_prior = (prior_center is not None) and (reg > 0.0)
    if use_prior:
        prior_center = np.asarray(prior_center, dtype=np.float64).reshape(-1)
        if prior_center.size != n:
            raise ValueError("prior_center size mismatch with x0.")
        sqrt_reg = float(np.sqrt(reg))
    else:
        prior_center = None
        sqrt_reg = 0.0

    def fun(x):
        r = np.asarray(residual_func(x), dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(r)):
            return np.full(r.shape, 1e30, dtype=np.float64)

        if use_prior:
            rp = sqrt_reg * (x - prior_center)
            return np.concatenate((r, rp))
        return r

    if jac_func is None:
        jac = "2-point"
    else:
        def jac(x):
            J = np.asarray(jac_func(x), dtype=np.float64)
            if J.ndim != 2:
                J = np.atleast_2d(J)
            if use_prior:
                Jp = sqrt_reg * np.eye(n, dtype=np.float64)
                J = np.vstack((J, Jp))
            return J

    result = least_squares(
        fun=fun,
        x0=x0,
        jac=jac,
        bounds=(lb, ub),
        method="trf",
        max_nfev=int(max_nfev),
        ftol=float(ftol),
        xtol=float(xtol),
        gtol=float(gtol),
        x_scale="jac",
        loss=loss,
        f_scale=float(f_scale),
    )

    return np.asarray(result.x, dtype=np.float64), result

