#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Isolated layout helpers for Maday proposal experiments.

This module intentionally does not touch the default `Results/Stage*` folders.
All artifacts are written under `Results_Maday/`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_MADAY_DIR = os.path.join(THIS_DIR, "Results_Maday")


def sanitize_tag(tag: str) -> str:
    raw = str(tag).strip()
    if not raw:
        raise ValueError("Experiment tag cannot be empty.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    safe = safe.strip("._-")
    if not safe:
        raise ValueError(f"Invalid experiment tag after sanitization: {tag!r}")
    return safe


@dataclass(frozen=True)
class MadayPaths:
    root: str
    tag: str
    exp_dir: str
    stage1: str
    stage2: str
    stage3: str
    stage3_models: str
    figures: str
    runs: str


def build_paths(tag: str, results_root: str | None = None) -> MadayPaths:
    safe_tag = sanitize_tag(tag)
    root = os.path.abspath(results_root or DEFAULT_RESULTS_MADAY_DIR)
    exp_dir = os.path.join(root, safe_tag)
    stage1 = os.path.join(exp_dir, "Stage1")
    stage2 = os.path.join(exp_dir, "Stage2")
    stage3 = os.path.join(exp_dir, "Stage3")
    stage3_models = os.path.join(stage3, "models")
    figures = os.path.join(exp_dir, "Figures")
    runs = os.path.join(exp_dir, "Runs")
    return MadayPaths(
        root=root,
        tag=safe_tag,
        exp_dir=exp_dir,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        stage3_models=stage3_models,
        figures=figures,
        runs=runs,
    )


def ensure_paths(paths: MadayPaths) -> MadayPaths:
    for d in (paths.root, paths.exp_dir, paths.stage1, paths.stage2, paths.stage3, paths.stage3_models, paths.figures, paths.runs):
        os.makedirs(d, exist_ok=True)
    return paths


def get_paths(tag: str, results_root: str | None = None) -> MadayPaths:
    return ensure_paths(build_paths(tag=tag, results_root=results_root))

