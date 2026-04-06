#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared Stage-3 helpers for loading/splitting reduced coordinates from Stage-2 datasets."""

import os
import numpy as np


def load_qn_from_mu_dir(mu_dir: str) -> np.ndarray:
    """
    Load full reduced coordinates qN with backward compatibility.

    Preferred (new) format:
      - qN.npy

    Legacy fallback:
      - qN_p.npy
      - qN_s.npy
      -> qN = vstack([qN_p, qN_s])

    If fallback is used, this helper also writes qN.npy in-place to migrate
    the folder to the canonical format.
    """
    qn_path = os.path.join(mu_dir, "qN.npy")
    if os.path.exists(qn_path):
        qn = np.asarray(np.load(qn_path, allow_pickle=False), dtype=np.float64)
        if qn.ndim != 2:
            raise ValueError(f"{mu_dir}: qN.npy must be 2D (n_tot,T), got {qn.shape}")
        return qn

    qnp_path = os.path.join(mu_dir, "qN_p.npy")
    qns_path = os.path.join(mu_dir, "qN_s.npy")
    if not (os.path.exists(qnp_path) and os.path.exists(qns_path)):
        raise FileNotFoundError(
            f"{mu_dir}: missing qN.npy and missing legacy pair qN_p.npy/qN_s.npy."
        )

    qnp = np.asarray(np.load(qnp_path, allow_pickle=False), dtype=np.float64)
    qns = np.asarray(np.load(qns_path, allow_pickle=False), dtype=np.float64)
    if qnp.ndim != 2:
        raise ValueError(f"{mu_dir}: qN_p.npy must be 2D (n_p,T), got {qnp.shape}")
    if qns.ndim != 2:
        raise ValueError(f"{mu_dir}: qN_s.npy must be 2D (n_s,T), got {qns.shape}")
    if qnp.shape[1] != qns.shape[1]:
        raise ValueError(
            f"{mu_dir}: legacy split time mismatch qN_p T={qnp.shape[1]} vs qN_s T={qns.shape[1]}"
        )
    qn = np.vstack([qnp, qns])
    # Best-effort migration cache to canonical format.
    try:
        np.save(qn_path, qn)
    except OSError:
        pass
    return qn


def resolve_primary_modes(requested_primary_modes, dataset_meta: dict, n_tot: int, default_primary: int = 10) -> int:
    """
    Resolve training split n (primary modes) from CLI or dataset metadata.
    """
    if requested_primary_modes is None:
        if isinstance(dataset_meta, dict) and dataset_meta.get("primary_modes", None) is not None:
            n = int(dataset_meta.get("primary_modes"))
        elif isinstance(dataset_meta, dict) and dataset_meta.get("default_primary_modes", None) is not None:
            n = int(dataset_meta.get("default_primary_modes"))
        else:
            n = int(default_primary)
    else:
        n = int(requested_primary_modes)

    if not (0 < n < int(n_tot)):
        raise ValueError(f"primary_modes={n} must satisfy 0 < primary_modes < n_tot={n_tot}.")
    return n


def split_qn(qn: np.ndarray, primary_modes: int):
    """
    Split full qN into primary/truncated blocks according to `primary_modes`.
    """
    qn = np.asarray(qn, dtype=np.float64)
    if qn.ndim != 2:
        raise ValueError(f"qN must be 2D (n_tot,T), got {qn.shape}")
    n_tot, _ = qn.shape
    n = int(primary_modes)
    if not (0 < n < n_tot):
        raise ValueError(f"primary_modes={n} must satisfy 0 < primary_modes < n_tot={n_tot}.")
    return qn[:n, :], qn[n:, :]
