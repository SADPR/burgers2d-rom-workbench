#!/usr/bin/env bash
# Train the fixed Euclidean architectures on the 9+8 and/or 9+18 datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LEVEL="${1:-all}"
case "$LEVEL" in
  all|lhs8|lhs18) ;;
  *) echo "Usage: $0 [all|lhs8|lhs18]" >&2; exit 2 ;;
esac

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Results_Paper}"
VALIDATION_DATASET_DIR="${VALIDATION_DATASET_DIR:-$PAPER_RESULTS_ROOT/euclidean_prom_main/Stage2/prom_coeff_dataset_ntot151_validation2}"
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-24}"
FORCE="${FORCE:-0}"

run_level() {
  local level="$1" expected tag dataset
  case "$level" in
    lhs8) expected=17 ;;
    lhs18) expected=27 ;;
  esac
  tag="euclidean_prom_enrichment_${level}"
  dataset="$PAPER_RESULTS_ROOT/$tag/Stage2/prom_coeff_dataset_ntot151_enriched_${level}"
  echo "[euclidean-nested-train] ===== $level: $expected training trajectories ====="
  PAPER_RESULTS_ROOT="$PAPER_RESULTS_ROOT" \
  PAPER_TAG="$tag" \
  DATASET_DIR="$dataset" \
  VALIDATION_DATASET_DIR="$VALIDATION_DATASET_DIR" \
  EXPECTED_TRAIN_TRAJECTORIES="$expected" \
  CAMPAIGN_LABEL="Euclidean nested $level" \
  TRAIN_NUM_THREADS="$TRAIN_NUM_THREADS" \
  FORCE="$FORCE" \
  bash "$SCRIPT_DIR/run_euclidean_prom_main_train_best.sh" all
}

[[ -d "$VALIDATION_DATASET_DIR" ]] || { echo "[error] Missing validation data: $VALIDATION_DATASET_DIR" >&2; exit 1; }
if [[ "$LEVEL" == "all" ]]; then
  run_level lhs8
  run_level lhs18
else
  run_level "$LEVEL"
fi
