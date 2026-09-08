#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/mlspg_prom_main"
export BASIS="$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/basis.npy"
export UREF="$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
export DATASET_DIR="${DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2}"
export LOG_DIR="$PAPER_ROOT/logs/stage2_validation"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"
export PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$PROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$PROM_NUM_THREADS"
export MKL_NUM_THREADS="$PROM_NUM_THREADS"
export BLIS_NUM_THREADS="$PROM_NUM_THREADS"
export GOTO_NUM_THREADS="$PROM_NUM_THREADS"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

echo "[mlspg-prom-validation] output:  $DATASET_DIR"
echo "[mlspg-prom-validation] basis:   $BASIS"
echo "[mlspg-prom-validation] u_ref:   $UREF"
echo "[mlspg-prom-validation] threads: $PROM_NUM_THREADS"
echo "[mlspg-prom-validation] points:"
echo "  Q2 cell midpoint: mu=(4.5625, 0.02625)"
echo "  Q4 cell midpoint: mu=(5.1875, 0.01875)"

python3 -u stage2_build_prom_qn_dataset.py \
  --backend prom \
  --total-modes 151 \
  --basis-path "$BASIS" \
  --u-ref-path "$UREF" \
  --output-dir "$DATASET_DIR" \
  --mu-pair 4.5625 0.02625 \
  --mu-pair 5.1875 0.01875 \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --no-save-rom-snaps \
  --no-plots \
  2>&1 | tee "$LOG_DIR/stage2_prom_qn_ntot151_validation2.log"

grep -E "solve_backend|dataset_dir|basis_path|u_ref_path|total_modes|num_traj|mu_source|mu_list" \
  "$DATASET_DIR/stage2_summary.txt" | tee "$LOG_DIR/stage2_prom_qn_ntot151_validation2_check.txt"
