#!/usr/bin/env bash
# Benchmark only operations needed by the HPROM speed-up table.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

MODE="${1:-all}"
case "$MODE" in hdm|direct|all) ;; *) echo "Usage: $0 [hdm|direct|all]" >&2; exit 2;; esac
PAPER_ROOT="${EUCLIDEAN_HPROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_hprom_main}"
TIMING="$PAPER_ROOT/timing"
THREADS="${TIMING_THREADS:-24}"
DIRECT_REPEATS="${DIRECT_REPEATS:-10}"
HDM_REPEATS="${HDM_REPEATS:-1}"
FORCE_TIMING="${FORCE_TIMING:-0}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export BLIS_NUM_THREADS="$THREADS" GOTO_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"
mkdir -p "$TIMING" "$MPLCONFIGDIR"

run_hdm() {
  if [[ "$FORCE_TIMING" != 1 && -f "$TIMING/hdm/hdm_timing.json" ]]; then
    echo "[skip] Existing fresh-HDM timing summary: $TIMING/hdm/hdm_timing.json"
    return
  fi
  "$PYTHON_BIN" -u benchmark_hdm_reporting_points.py --output "$TIMING/hdm" \
    --threads "$THREADS" --repeats "$HDM_REPEATS" 2>&1 | tee "$TIMING/hdm.log"
}
run_direct() {
  if [[ "$FORCE_TIMING" != 1 && -f "$TIMING/direct_inference_repeat${DIRECT_REPEATS}_summary.txt" ]]; then
    echo "[skip] Existing repeated direct-map timing summary."
    return
  fi
  "$PYTHON_BIN" -u Results_Paper/benchmark_hprom_direct_maps.py \
    --campaign-root "$PAPER_ROOT" --device cpu --threads "$THREADS" \
    --repeats "$DIRECT_REPEATS" --warmup 3 \
    --podnn-model-name master_ann_mu_t_to_qtot_ntot151_best.pt \
    --poddl-model-name pod_dl_data_driven_ntot151_best.pt \
    --summary-path "$TIMING/direct_inference_repeat${DIRECT_REPEATS}_summary.txt" \
    2>&1 | tee "$TIMING/direct_inference.log"
}

[[ "$MODE" == hdm || "$MODE" == all ]] && run_hdm
[[ "$MODE" == direct || "$MODE" == all ]] && run_direct
echo "[euclidean-hprom-timing] completed mode=$MODE output=$TIMING"
