#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared output layout helpers for Project_YvonMaday."""

import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "Results")

STAGE1_DIR = os.path.join(RESULTS_DIR, "Stage1")
STAGE2_DIR = os.path.join(RESULTS_DIR, "Stage2")
STAGE3_DIR = os.path.join(RESULTS_DIR, "Stage3")
STAGE3_MODELS_DIR = os.path.join(STAGE3_DIR, "models")

RUNS_DIR = os.path.join(RESULTS_DIR, "Runs")
RUNS_LINEAR_DIR = os.path.join(RUNS_DIR, "Linear")
RUNS_CASE1_DIR = os.path.join(RUNS_DIR, "Case1")
RUNS_CASE2_DIR = os.path.join(RUNS_DIR, "Case2")
RUNS_CASE3_DIR = os.path.join(RUNS_DIR, "Case3")
RUNS_ECSW_DIR = os.path.join(RUNS_DIR, "ECSW")
RUNS_DATA_DRIVEN_DIR = os.path.join(RUNS_DIR, "DataDriven")


def ensure_layout_dirs():
    for path in (
        RESULTS_DIR,
        STAGE1_DIR,
        STAGE2_DIR,
        STAGE3_DIR,
        STAGE3_MODELS_DIR,
        RUNS_DIR,
        RUNS_LINEAR_DIR,
        RUNS_CASE1_DIR,
        RUNS_CASE2_DIR,
        RUNS_CASE3_DIR,
        RUNS_ECSW_DIR,
        RUNS_DATA_DRIVEN_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def stage1_artifact_path(name):
    return os.path.join(STAGE1_DIR, name)


def resolve_stage1_artifact(name):
    candidates = [
        stage1_artifact_path(name),
        os.path.join(THIS_DIR, name),  # legacy location
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def stage2_dataset_dir(ntot):
    return os.path.join(STAGE2_DIR, f"prom_coeff_dataset_ntot{int(ntot)}")


def stage2_dataset_candidates():
    roots = [STAGE2_DIR, THIS_DIR]  # legacy fallback
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                yield path


def stage3_model_path(name):
    return os.path.join(STAGE3_MODELS_DIR, name)


def resolve_stage3_model(name):
    candidates = [
        stage3_model_path(name),
        os.path.join(THIS_DIR, name),  # legacy location
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def write_kv_txt(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, value in items:
            f.write(f"{key}: {value}\n")
