#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"

# This diagnostic is dominated by dense reduced matrix products.
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-24}"
export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"

OUT_DIR="$PROJECT_DIR/Results_Paper/MetricStudy/low_high_transfer"
mkdir -p "$OUT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

python3 -u analyze_case2_transfer_operator.py \
  --basis "Euclidean POD=$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/basis.npy" \
  --basis "LSPG-sensitive POD=$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy" \
  --snap-dir "$PROJECT_DIR/../Results/param_snaps" \
  --point 4.560,0.0190 \
  --point 4.875,0.0225 \
  --point 5.190,0.0260 \
  --n-primary 10 \
  --n-tot 151 \
  --time-start-index 1 \
  --time-stop-index 500 \
  --stride 1 \
  --normal-rcond 1e-12 \
  --output-dir "$OUT_DIR" \
  --progress-every 25 \
  2>&1 | tee "$OUT_DIR/low_high_transfer_build.log"
