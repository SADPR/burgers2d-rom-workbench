#!/usr/bin/env bash
# Solve the 18 frozen enrichment points once with the baseline linear HPROM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Results_Paper}"
BASELINE_ROOT="${BASELINE_EUCLIDEAN_HPROM_ROOT:-$PAPER_RESULTS_ROOT/euclidean_hprom_main}"
BASE_DATASET="${BASE_DATASET:-$BASELINE_ROOT/Stage2/prom_coeff_dataset_ntot151}"
BASIS="${BASIS:-$PAPER_RESULTS_ROOT/MetricStudy/euclidean/Stage1/basis.npy}"
UREF="${UREF:-$PAPER_RESULTS_ROOT/MetricStudy/euclidean/Stage1/u_ref.npy}"
LINEAR_ECSW="${LINEAR_ECSW:-$BASELINE_ROOT/Stage2/ecsw/Linear/ecsw_weights_lspg_ntot151.npy}"
DESIGN="${DESIGN:-$PROJECT_DIR/euclidean_nested_enrichment_design.json}"
LOG_ROOT="${LOG_ROOT:-$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_lhs18/logs/stage2}"

HPROM_NUM_THREADS="${HPROM_NUM_THREADS:-24}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
export BLIS_NUM_THREADS="$HPROM_NUM_THREADS" GOTO_NUM_THREADS="$HPROM_NUM_THREADS"
export MKL_NUM_THREADS="$HPROM_NUM_THREADS" OMP_NUM_THREADS="$HPROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$HPROM_NUM_THREADS" NUMEXPR_NUM_THREADS="$HPROM_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_lhs18/.mplcache}"

for required in "$BASE_DATASET/meta.json" "$BASIS" "$UREF" "$LINEAR_ECSW" "$DESIGN"; do
  [[ -f "$required" ]] || { echo "[error] Missing required artifact: $required" >&2; exit 1; }
done
mkdir -p "$LOG_ROOT" "$MPLCONFIGDIR"

args=(
  --base-dataset-dir "$BASE_DATASET"
  --results-root "$PAPER_RESULTS_ROOT"
  --basis-path "$BASIS"
  --u-ref-path "$UREF"
  --ecsw-weights-path "$LINEAR_ECSW"
  --design-path "$DESIGN"
  --total-modes 151
  --linear-solver lstsq
  --normal-eq-reg 1e-12
  --max-its 20
  --relnorm-cutoff 1e-5
  --min-delta 1e-2
)
[[ "$FORCE" == 1 ]] && args+=(--force)
[[ "$PLAN_ONLY" == 1 ]] && args+=(--plan-only)

echo "[euclidean-hprom-nested-stage2] Python: $PYTHON_BIN"
echo "[euclidean-hprom-nested-stage2] baseline: $BASE_DATASET"
echo "[euclidean-hprom-nested-stage2] linear rule: $LINEAR_ECSW"
echo "[euclidean-hprom-nested-stage2] solves: at most 18; LHS8 and LHS12 are copied subsets"
echo "[euclidean-hprom-nested-stage2] numerical threads: $HPROM_NUM_THREADS"
"$PYTHON_BIN" -u stage2_build_euclidean_hprom_nested_enrichment.py "${args[@]}" \
  2>&1 | tee "$LOG_ROOT/stage2_euclidean_hprom_nested_lhs8_lhs12_lhs18.log"
