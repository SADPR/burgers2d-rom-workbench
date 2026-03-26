#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared dataset discovery and validation utilities for Stage 3 training scripts.
"""

import os
import re
import numpy as np

try:
    from project_layout import STAGE2_DIR, stage2_dataset_dir
except ModuleNotFoundError:
    from .project_layout import STAGE2_DIR, stage2_dataset_dir


def _read_meta(dataset_dir: str):
    meta_path = os.path.join(dataset_dir, "meta.npy")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing dataset metadata file: {meta_path}\n"
            "Run stage2_build_prom_qn_dataset.py first."
        )

    meta = np.load(meta_path, allow_pickle=True).item()
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid metadata format in {meta_path}: expected dict, got {type(meta)}")

    return meta, meta_path


def resolve_stage3_dataset(this_dir: str, requested_ntot=None, expected_backend="hprom"):
    """
    Return:
      - per_mu_root: <dataset_dir>/per_mu
      - detected_ntot: integer parsed from folder name
      - dataset_dir: preferred <this_dir>/Results/Stage2/prom_coeff_dataset_ntot{detected_ntot}
        with fallback to legacy <this_dir>/prom_coeff_dataset_ntot{detected_ntot}
      - meta: dict loaded from meta.npy
      - meta_path: absolute path to metadata file

    If requested_ntot is None, choose the most recently modified matching dataset
    that contains per_mu/.
    """
    search_roots = [STAGE2_DIR, this_dir]

    if requested_ntot is not None:
        detected_ntot = int(requested_ntot)
        dataset_candidates = [
            stage2_dataset_dir(detected_ntot),
            os.path.join(this_dir, f"prom_coeff_dataset_ntot{detected_ntot}"),
        ]
        dataset_dir = None
        per_mu_root = None
        for cand in dataset_candidates:
            per_mu_cand = os.path.join(cand, "per_mu")
            if os.path.isdir(per_mu_cand):
                dataset_dir = cand
                per_mu_root = per_mu_cand
                break
        if dataset_dir is None:
            checked = "\n".join([f"  - {p}" for p in dataset_candidates])
            raise FileNotFoundError(
                "Missing dataset directory for requested ntot. Checked:\n"
                f"{checked}"
            )
    else:
        candidates = []
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                match = re.fullmatch(r"prom_coeff_dataset_ntot(\d+)", name)
                if match is None:
                    continue
                dataset_dir_i = os.path.join(root, name)
                per_mu_root_i = os.path.join(dataset_dir_i, "per_mu")
                if os.path.isdir(per_mu_root_i):
                    candidates.append(
                        (os.path.getmtime(dataset_dir_i), int(match.group(1)), dataset_dir_i, per_mu_root_i)
                    )

        if len(candidates) == 0:
            roots_msg = "\n".join([f"  - {p}" for p in search_roots])
            raise FileNotFoundError(
                "No dataset folder matching 'prom_coeff_dataset_ntot*/per_mu' found. Checked roots:\n"
                f"{roots_msg}"
            )

        candidates.sort(key=lambda x: (x[0], x[1]))
        _, detected_ntot, dataset_dir, per_mu_root = candidates[-1]

    meta, meta_path = _read_meta(dataset_dir)

    if expected_backend is not None:
        backend = str(meta.get("solve_backend", "")).strip().lower()
        wanted = str(expected_backend).strip().lower()
        if backend != wanted:
            raise ValueError(
                f"Dataset backend mismatch for '{dataset_dir}': solve_backend='{backend}', expected '{wanted}'."
            )

    meta_ntot = meta.get("total_modes")
    if meta_ntot is not None and int(meta_ntot) != int(detected_ntot):
        raise ValueError(
            f"Dataset metadata mismatch in '{meta_path}': total_modes={meta_ntot} "
            f"but directory encodes ntot={detected_ntot}."
        )

    return per_mu_root, detected_ntot, dataset_dir, meta, meta_path
