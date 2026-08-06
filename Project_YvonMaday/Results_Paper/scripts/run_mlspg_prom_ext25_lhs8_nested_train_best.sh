#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
export PAPER_TAG="${PAPER_TAG:-mlspg_prom_enrichment_ext25_lhs8_nested}"
export DATASET_DIR="${DATASET_DIR:-$PAPER_RESULTS_ROOT/$PAPER_TAG/Stage2/prom_coeff_dataset_ntot151_enriched_lhs8}"
export VAL_DATASET_DIR="${VAL_DATASET_DIR:-$PAPER_RESULTS_ROOT/mlspg_prom_main/Stage2/prom_coeff_dataset_ntot151_validation2}"
export CAMPAIGN_LABEL="${CAMPAIGN_LABEL:-ext25-lhs8-nested}"
export EXPECTED_BASE_TRAJ=9
export EXPECTED_INTERIOR_LHS=4
export EXPECTED_EXTERIOR_LHS=4
export EXPECTED_LHS_TRAJ=8
export EXPECTED_TOTAL_TRAJ=17

exec bash "$SCRIPT_DIR/run_mlspg_prom_ext25_lhs36_train_best.sh" "$@"
