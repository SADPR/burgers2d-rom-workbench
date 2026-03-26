#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Output layout helpers for enrichment workflows (isolated from Results)."""

import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICHMENT_RESULTS_DIR = os.path.join(THIS_DIR, "Results_Enrichment")

ENRICHMENT_STAGE2_DIR = os.path.join(ENRICHMENT_RESULTS_DIR, "Stage2")
ENRICHMENT_STAGE3_DIR = os.path.join(ENRICHMENT_RESULTS_DIR, "Stage3")
ENRICHMENT_STAGE3_MODELS_DIR = os.path.join(ENRICHMENT_STAGE3_DIR, "models")

ENRICHMENT_RUNS_DIR = os.path.join(ENRICHMENT_RESULTS_DIR, "Runs")
ENRICHMENT_RUNS_LINEAR_DIR = os.path.join(ENRICHMENT_RUNS_DIR, "Linear")
ENRICHMENT_RUNS_CASE1_DIR = os.path.join(ENRICHMENT_RUNS_DIR, "Case1")
ENRICHMENT_RUNS_CASE2_DIR = os.path.join(ENRICHMENT_RUNS_DIR, "Case2")
ENRICHMENT_RUNS_CASE3_DIR = os.path.join(ENRICHMENT_RUNS_DIR, "Case3")
ENRICHMENT_RUNS_DATA_DRIVEN_DIR = os.path.join(ENRICHMENT_RUNS_DIR, "DataDriven")
ENRICHMENT_RUNS_ECSW_DIR = os.path.join(ENRICHMENT_RUNS_DIR, "ECSW")


def ensure_enrichment_dirs():
    for path in (
        ENRICHMENT_RESULTS_DIR,
        ENRICHMENT_STAGE2_DIR,
        ENRICHMENT_STAGE3_DIR,
        ENRICHMENT_STAGE3_MODELS_DIR,
        ENRICHMENT_RUNS_DIR,
        ENRICHMENT_RUNS_LINEAR_DIR,
        ENRICHMENT_RUNS_CASE1_DIR,
        ENRICHMENT_RUNS_CASE2_DIR,
        ENRICHMENT_RUNS_CASE3_DIR,
        ENRICHMENT_RUNS_DATA_DRIVEN_DIR,
        ENRICHMENT_RUNS_ECSW_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def enrichment_stage2_dataset_dir(total_modes, lhs_samples):
    return os.path.join(
        ENRICHMENT_STAGE2_DIR,
        f"prom_coeff_dataset_ntot{int(total_modes)}_enriched_lhs{int(lhs_samples)}",
    )

