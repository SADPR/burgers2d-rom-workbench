#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export BASIS="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
export UREF="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
export DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
export ECSW_DIR="$PAPER_ROOT/Stage2/ecsw"
export LOG_DIR="$PAPER_ROOT/logs"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

mkdir -p "$LOG_DIR" "$ECSW_DIR" "$MPLCONFIGDIR"

python3 -u stage2_build_prom_qn_dataset.py \
  --backend hprom \
  --total-modes 151 \
  --basis-path "$BASIS" \
  --u-ref-path "$UREF" \
  --output-dir "$DATASET_DIR" \
  --ecsw-weights-dir "$ECSW_DIR" \
  --rebuild-ecsw \
  --ecsw-snapshot-percent 2.0 \
  --ecsw-num-training-mu 9 \
  --ecsw-snap-time-offset 3 \
  --ecsw-random-seed 42 \
  --ecsw-ensure-mu-coverage \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --no-save-rom-snaps \
  --no-plots \
  2>&1 | tee "$LOG_DIR/stage2_hprom_qn_ntot151.log"

grep -E "solve_backend|dataset_dir|basis_path|u_ref_path|total_modes|n_ecsw_elements|ecsw_residual|ecsw_snapshot_percent|ecsw_num_selected_total" \
  "$DATASET_DIR/stage2_summary.txt" | tee "$LOG_DIR/stage2_hprom_qn_ntot151_check.txt"
