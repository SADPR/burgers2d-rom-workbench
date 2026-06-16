#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared dataset discovery and validation utilities for Stage 3 training scripts.
"""

import importlib
import json
import os
import re
import sys
import numpy as np

try:
    from project_layout import STAGE2_DIR, stage2_dataset_dir
except ModuleNotFoundError:
    from .project_layout import STAGE2_DIR, stage2_dataset_dir


def _install_numpy_pickle_compat_aliases():
    """Allow NumPy 1.x to unpickle object arrays written by NumPy 2.x."""
    if "numpy._core" in sys.modules:
        return

    numpy_core = importlib.import_module("numpy.core")
    sys.modules["numpy._core"] = numpy_core
    for name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules[f"numpy._core.{name}"] = module


def read_dataset_meta(dataset_dir: str):
    json_path = os.path.join(dataset_dir, "meta.json")
    meta_path = os.path.join(dataset_dir, "meta.npy")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        source_path = json_path
    elif os.path.exists(meta_path):
        try:
            meta = np.load(meta_path, allow_pickle=True).item()
        except ModuleNotFoundError as exc:
            missing_module = str(exc.name or "")
            if not (
                missing_module == "numpy._core"
                or missing_module.startswith("numpy._core.")
            ):
                raise
            _install_numpy_pickle_compat_aliases()
            meta = np.load(meta_path, allow_pickle=True).item()
        source_path = meta_path
    else:
        raise FileNotFoundError(
            "Missing dataset metadata file. Checked:\n"
            f"  - {json_path}\n"
            f"  - {meta_path}\n"
            "Run stage2_build_prom_qn_dataset.py first."
        )

    if not isinstance(meta, dict):
        raise ValueError(
            f"Invalid metadata format in {source_path}: expected dict, got {type(meta)}"
        )

    return meta, source_path


def _read_meta(dataset_dir: str):
    """Backward-compatible private alias."""
    return read_dataset_meta(dataset_dir)


def resolve_stage3_dataset(
    this_dir: str,
    requested_ntot=None,
    expected_backend="hprom",
    requested_dataset_dir=None,
):
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

    if requested_dataset_dir is not None:
        dataset_dir = os.path.abspath(os.path.expanduser(str(requested_dataset_dir)))
        per_mu_root = os.path.join(dataset_dir, "per_mu")
        if not os.path.isdir(per_mu_root):
            raise FileNotFoundError(
                f"Missing per_mu folder in requested dataset directory:\n  - {dataset_dir}"
            )

        meta, meta_path = _read_meta(dataset_dir)
        meta_ntot = meta.get("total_modes")
        if requested_ntot is not None:
            detected_ntot = int(requested_ntot)
        elif meta_ntot is not None:
            detected_ntot = int(meta_ntot)
        else:
            match = re.fullmatch(r"prom_coeff_dataset_ntot(\d+)", os.path.basename(dataset_dir))
            if match is None:
                raise ValueError(
                    "Unable to infer n_tot from requested dataset directory name and metadata "
                    "has no 'total_modes'. Please pass --dataset-ntot explicitly."
                )
            detected_ntot = int(match.group(1))

        if meta_ntot is not None and int(meta_ntot) != int(detected_ntot):
            raise ValueError(
                f"Dataset metadata mismatch in '{meta_path}': total_modes={meta_ntot} "
                f"but resolved ntot={detected_ntot}."
            )
    elif requested_ntot is not None:
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
        meta, meta_path = _read_meta(dataset_dir)
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
