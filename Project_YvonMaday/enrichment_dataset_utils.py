#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset discovery helpers for enrichment workflows."""

import os
import re
import numpy as np

try:
    from enrichment_layout import ENRICHMENT_STAGE2_DIR
except ModuleNotFoundError:
    from .enrichment_layout import ENRICHMENT_STAGE2_DIR


def _read_meta(dataset_dir):
    meta_path = os.path.join(dataset_dir, "meta.npy")
    if not os.path.exists(meta_path):
        return {}, meta_path
    meta = np.load(meta_path, allow_pickle=True).item()
    if not isinstance(meta, dict):
        return {}, meta_path
    return meta, meta_path


def resolve_enrichment_dataset(requested_ntot=None, expected_backend="hprom"):
    """
    Resolve the latest enrichment dataset under Results_Enrichment/Stage2.

    Returns:
      per_mu_root, detected_ntot, dataset_dir, meta, meta_path
    """
    if not os.path.isdir(ENRICHMENT_STAGE2_DIR):
        raise FileNotFoundError(f"Missing enrichment stage2 directory: {ENRICHMENT_STAGE2_DIR}")

    pattern = re.compile(r"prom_coeff_dataset_ntot(\d+)_enriched_lhs\d+$")
    candidates = []
    for name in os.listdir(ENRICHMENT_STAGE2_DIR):
        m = pattern.fullmatch(name)
        if m is None:
            continue
        ntot = int(m.group(1))
        if requested_ntot is not None and ntot != int(requested_ntot):
            continue
        dataset_dir = os.path.join(ENRICHMENT_STAGE2_DIR, name)
        per_mu_root = os.path.join(dataset_dir, "per_mu")
        if not os.path.isdir(per_mu_root):
            continue
        candidates.append((os.path.getmtime(dataset_dir), ntot, dataset_dir, per_mu_root))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No enrichment dataset found in {ENRICHMENT_STAGE2_DIR}. "
            "Run stage2_build_enrichment_lhs_qn_dataset.py first."
        )

    candidates.sort(key=lambda x: (x[0], x[1]))
    _, detected_ntot, dataset_dir, per_mu_root = candidates[-1]

    meta, meta_path = _read_meta(dataset_dir)
    if expected_backend is not None:
        backend = str(meta.get("solve_backend", "")).strip().lower()
        wanted = str(expected_backend).strip().lower()
        if backend and backend != wanted:
            raise ValueError(
                f"Enrichment dataset backend mismatch: solve_backend='{backend}', expected '{wanted}'."
            )

    if meta.get("total_modes", None) is not None and int(meta["total_modes"]) != int(detected_ntot):
        raise ValueError(
            f"Enrichment dataset metadata mismatch: total_modes={meta['total_modes']} "
            f"but directory encodes ntot={detected_ntot}."
        )

    return per_mu_root, detected_ntot, dataset_dir, meta, meta_path
