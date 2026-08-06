#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
SOURCE_DATASET="${SOURCE_DATASET:-$PAPER_RESULTS_ROOT/mlspg_prom_enrichment_ext25_lhs36/Stage2/prom_coeff_dataset_ntot151_enriched_lhs36}"
OUTPUT_DATASET="${OUTPUT_DATASET:-$PAPER_RESULTS_ROOT/mlspg_prom_enrichment_ext25_lhs8_nested/Stage2/prom_coeff_dataset_ntot151_enriched_lhs8}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

args=(
  --source-dataset "$SOURCE_DATASET"
  --output-dataset "$OUTPUT_DATASET"
  --n-interior 4
  --n-exterior 4
  --subset-label ext25_lhs8_nested
)
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi
if [[ "$PLAN_ONLY" == "1" ]]; then
  args+=(--plan-only)
fi

echo "[prom-lhs8-stage2] source:    $SOURCE_DATASET"
echo "[prom-lhs8-stage2] output:    $OUTPUT_DATASET"
echo "[prom-lhs8-stage2] selection: 9 baseline + 4 nested interior + 4 nested margin"
echo "[prom-lhs8-stage2] force:     $FORCE"
echo "[prom-lhs8-stage2] plan only: $PLAN_ONLY"

python3 -u Results_Paper/build_nested_lhs_subset_dataset.py "${args[@]}"
