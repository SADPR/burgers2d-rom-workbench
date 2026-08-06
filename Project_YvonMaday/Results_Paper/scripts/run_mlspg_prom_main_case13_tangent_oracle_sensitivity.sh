#!/usr/bin/env bash
# Temporary appendix diagnostic for the state-dependent Case-1 and Case-3
# PROM--ANN closures.  This never writes into Runs/PROM production folders.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

CASE="${1:-all}"
case "$CASE" in
  case1|case3|all) ;;
  *)
    echo "Usage: $0 [case1|case3|all]" >&2
    exit 2
    ;;
esac

PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
BASIS_PATH="${BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
U_REF_PATH="${U_REF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/Results_Paper/tmp_case13_tangent_oracle_sensitivity}"
LOG_DIR="${LOG_DIR:-$PROM_ROOT/logs/appendix/case13_tangent_oracle_sensitivity}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

# The native ANN error is inserted as a starred point in addition to this
# common set of controlled tail-error levels.
LEVELS="${LEVELS:-0 1 3 5 10 15 20 30 50}"

export BLIS_NUM_THREADS="$PROM_NUM_THREADS"
export GOTO_NUM_THREADS="$PROM_NUM_THREADS"
export MKL_NUM_THREADS="$PROM_NUM_THREADS"
export OMP_NUM_THREADS="$PROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$PROM_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROM_ROOT/.mplcache}"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

extra_args=()
if [[ "$FORCE" == "1" ]]; then
  extra_args+=(--force)
fi
if [[ "$PLAN_ONLY" == "1" ]]; then
  extra_args+=(--plan-only)
fi

echo "[case13-tangent-oracle] case:          $CASE"
echo "[case13-tangent-oracle] PROM root:     $PROM_ROOT"
echo "[case13-tangent-oracle] output root:   $OUTPUT_ROOT"
echo "[case13-tangent-oracle] levels (%):    $LEVELS"
echo "[case13-tangent-oracle] threads:       $PROM_NUM_THREADS"
echo "[case13-tangent-oracle] device:        $ONLINE_DEVICE"
echo "[case13-tangent-oracle] force:         $FORCE"
echo "[case13-tangent-oracle] plan only:     $PLAN_ONLY"

python3 -u run_case13_tangent_oracle_sensitivity_tmp.py \
  --case "$CASE" \
  --points all \
  --levels $LEVELS \
  --n-primary 10 \
  --n-tot 151 \
  --prom-root "$PROM_ROOT" \
  --basis-path "$BASIS_PATH" \
  --u-ref-path "$U_REF_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --device "$ONLINE_DEVICE" \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --include-ann-level \
  "${extra_args[@]}" \
  2>&1 | tee "$LOG_DIR/${CASE}_tangent_oracle_sensitivity.log"
