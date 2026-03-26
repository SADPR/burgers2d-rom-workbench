#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train enriched nonlinear maps (Case1/Case2/Case3/DataDriven) from an enrichment dataset.

This script launches the dedicated `*_enriched.py` Stage 3 trainers.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from enrichment_layout import ENRICHMENT_STAGE2_DIR, ENRICHMENT_STAGE3_DIR, ensure_enrichment_dirs
except ModuleNotFoundError:
    from .enrichment_layout import ENRICHMENT_STAGE2_DIR, ENRICHMENT_STAGE3_DIR, ensure_enrichment_dirs
try:
    from project_layout import write_kv_txt
except ModuleNotFoundError:
    from .project_layout import write_kv_txt


THIS_DIR = Path(__file__).resolve().parent


def _latest_enrichment_dataset_dir():
    if not os.path.isdir(ENRICHMENT_STAGE2_DIR):
        raise FileNotFoundError(f"Missing enrichment stage2 directory: {ENRICHMENT_STAGE2_DIR}")

    pattern = re.compile(r"prom_coeff_dataset_ntot\d+_enriched_lhs\d+$")
    candidates = []
    for name in os.listdir(ENRICHMENT_STAGE2_DIR):
        path = os.path.join(ENRICHMENT_STAGE2_DIR, name)
        if os.path.isdir(path) and pattern.fullmatch(name):
            per_mu = os.path.join(path, "per_mu")
            if os.path.isdir(per_mu):
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(
            "No enrichment dataset found. Run stage2_build_enrichment_lhs_qn_dataset.py first."
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_meta(dataset_dir):
    meta_path = os.path.join(dataset_dir, "meta.npy")
    if not os.path.exists(meta_path):
        return {}
    meta = np.load(meta_path, allow_pickle=True).item()
    if not isinstance(meta, dict):
        return {}
    return meta


def main():
    # -----------------------------
    # User settings
    # -----------------------------
    dataset_dir_override = None  # e.g. ".../Results_Enrichment/Stage2/prom_coeff_dataset_ntot..._enriched_lhs..."

    # -----------------------------
    # Resolve dataset + output dir
    # -----------------------------
    ensure_enrichment_dirs()

    dataset_dir = os.path.abspath(dataset_dir_override) if dataset_dir_override else _latest_enrichment_dataset_dir()
    dataset_root = os.path.join(dataset_dir, "per_mu")
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Missing per_mu directory: {dataset_root}")

    meta = _load_meta(dataset_dir)
    dataset_name = os.path.basename(dataset_dir.rstrip(os.sep))
    stage3_out_dir = os.path.join(ENRICHMENT_STAGE3_DIR, dataset_name)
    os.makedirs(stage3_out_dir, exist_ok=True)

    tasks = [
        "stage3_perform_training_case_1_ann_enriched.py",
        "stage3_perform_training_case_2_ann_enriched.py",
        "stage3_perform_training_case_3_ann_enriched.py",
        "stage3_perform_training_rom_data_driven_enriched.py",
    ]

    print(f"[Stage3-Enrich] dataset_dir = {dataset_dir}")
    print(f"[Stage3-Enrich] dataset_root = {dataset_root}")
    print(f"[Stage3-Enrich] stage3_out_dir = {stage3_out_dir}")

    for script_name in tasks:
        script_path = THIS_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing trainer script: {script_path}")

        print(f"[Stage3-Enrich] running {script_name}")
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(THIS_DIR))

    write_kv_txt(
        os.path.join(stage3_out_dir, "stage3_enrichment_run_summary.txt"),
        [
            ("dataset_dir", dataset_dir),
            ("dataset_root", dataset_root),
            ("stage3_out_dir", stage3_out_dir),
            ("dataset_backend", meta.get("solve_backend", "unknown")),
            ("dataset_total_modes", meta.get("total_modes", "unknown")),
            ("dataset_num_traj", meta.get("num_traj", "unknown")),
            ("scripts", ", ".join(tasks)),
        ],
    )

    print("[Stage3-Enrich] done.")
    print(f"[Stage3-Enrich] outputs: {stage3_out_dir}")


if __name__ == "__main__":
    main()
