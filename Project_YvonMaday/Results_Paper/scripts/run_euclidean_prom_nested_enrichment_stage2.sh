#!/usr/bin/env bash
# Build the two nested Euclidean-POD PROM coefficient datasets: 9+8 and 9+18.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Results_Paper}"
BASE_DATASET="${BASE_DATASET:-$PAPER_RESULTS_ROOT/euclidean_prom_main/Stage2/prom_coeff_dataset_ntot151}"
BASIS="${BASIS:-$PAPER_RESULTS_ROOT/MetricStudy/euclidean/Stage1/basis.npy}"
UREF="${UREF:-$PAPER_RESULTS_ROOT/MetricStudy/euclidean/Stage1/u_ref.npy}"
LOG_ROOT="${LOG_ROOT:-$PAPER_RESULTS_ROOT/euclidean_prom_enrichment_lhs18/logs/stage2}"

PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

export BLIS_NUM_THREADS="$PROM_NUM_THREADS"
export GOTO_NUM_THREADS="$PROM_NUM_THREADS"
export MKL_NUM_THREADS="$PROM_NUM_THREADS"
export OMP_NUM_THREADS="$PROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$PROM_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_RESULTS_ROOT/euclidean_prom_enrichment_lhs18/.mplcache}"

for required in "$BASE_DATASET/meta.json" "$BASIS" "$UREF"; do
  [[ -f "$required" ]] || { echo "[error] Missing required artifact: $required" >&2; exit 1; }
done
mkdir -p "$LOG_ROOT" "$MPLCONFIGDIR"

args=(
  --base-dataset-dir "$BASE_DATASET"
  --results-root "$PAPER_RESULTS_ROOT"
  --basis-path "$BASIS"
  --u-ref-path "$UREF"
  --total-modes 151
  --lhs-seed 42
  --margin-fraction 0.25
  --linear-solver lstsq
  --normal-eq-reg 1e-12
  --max-its 20
  --relnorm-cutoff 1e-5
  --min-delta 1e-2
)
[[ "$FORCE" == "1" ]] && args+=(--force)
[[ "$PLAN_ONLY" == "1" ]] && args+=(--plan-only)

echo "[euclidean-nested-stage2] source:  $BASE_DATASET"
echo "[euclidean-nested-stage2] results: $PAPER_RESULTS_ROOT"
echo "[euclidean-nested-stage2] levels:  baseline 9; nested 9+8; nested 9+18"
echo "[euclidean-nested-stage2] solves:  18 new full-residual PROM trajectories only"
echo "[euclidean-nested-stage2] threads: $PROM_NUM_THREADS"
echo "[euclidean-nested-stage2] force:   $FORCE"
echo "[euclidean-nested-stage2] plan:    $PLAN_ONLY"

log_name="stage2_euclidean_nested_lhs8_lhs18.log"
[[ "$PLAN_ONLY" == "1" ]] && log_name="stage2_euclidean_nested_lhs8_lhs18_plan.log"
python3 -u stage2_build_euclidean_prom_nested_enrichment.py "${args[@]}" 2>&1 | tee "$LOG_ROOT/$log_name"
