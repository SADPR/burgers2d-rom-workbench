#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
BASIS="${BASIS:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
UREF="${UREF:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
OUT="${OUT:-$PROJECT_DIR/Results_Paper/tmp_case2_secondary_sensitivity}"
LOG_DIR="${LOG_DIR:-$PROM_ROOT/logs/online/case2_secondary_sensitivity}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
CLEAR="${CLEAR:-1}"

export BLIS_NUM_THREADS="$PROM_NUM_THREADS"
export GOTO_NUM_THREADS="$PROM_NUM_THREADS"
export MKL_NUM_THREADS="$PROM_NUM_THREADS"
export OMP_NUM_THREADS="$PROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$PROM_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROM_ROOT/.mplcache}"

LEVELS="${LEVELS:-0 1 3 5 10 15 20 30 50 75 100}"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

if [[ "$CLEAR" == "1" ]]; then
  echo "[case2-secondary-100pct] clearing previous temporary diagnostic: $OUT"
  rm -rf "$OUT"
fi

echo "[case2-secondary-100pct] prom root: $PROM_ROOT"
echo "[case2-secondary-100pct] output:    $OUT"
echo "[case2-secondary-100pct] levels:    $LEVELS"
echo "[case2-secondary-100pct] threads:   $PROM_NUM_THREADS"

python3 -u run_case2_secondary_sensitivity_tmp.py \
  --points all \
  --levels $LEVELS \
  --n-primary 10 \
  --n-tot 151 \
  --prom-root "$PROM_ROOT" \
  --basis-path "$BASIS" \
  --u-ref-path "$UREF" \
  --output-root "$OUT" \
  --solver-variant plain \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --include-ann-level \
  --force \
  2>&1 | tee "$LOG_DIR/case2_secondary_sensitivity_100pct.log"
