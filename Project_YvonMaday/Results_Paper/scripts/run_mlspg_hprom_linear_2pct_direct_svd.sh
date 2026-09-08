#!/usr/bin/env bash
set -euo pipefail

# Linear HPROM ECSW rebuild for the PROM/HROM paper campaign.
# - 2% snapshot pairs
# - parameter/time-aware global stratification
# - deterministic direct dense SVD before ECM, not the randomized-SVD wrapper
#
# Usage:
#   bash Results_Paper/scripts/run_mlspg_hprom_linear_2pct_direct_svd.sh stage2   # rebuild ECSW + 9 training HPROM qN trajectories
#   bash Results_Paper/scripts/run_mlspg_hprom_linear_2pct_direct_svd.sh 4pts     # run the 4 linear HPROM evaluation points using existing weights
#   bash Results_Paper/scripts/run_mlspg_hprom_linear_2pct_direct_svd.sh all      # stage2 then 4pts

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="${PAPER_TAG:-mlspg_hprom_main}"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export BASIS="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
export UREF="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
export DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
export ECSW_DIR="$PAPER_ROOT/Stage2/ecsw"
export ECSW_WEIGHTS="$ECSW_DIR/ecsw_weights_lspg_ntot151.npy"
export LINEAR_RUNS="$PAPER_ROOT/Runs/Linear"
export LOG_DIR="$PAPER_ROOT/logs/linear_hprom_2pct_direct_svd"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

THREADS="${ROM_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export BLIS_NUM_THREADS="$THREADS"
export GOTO_NUM_THREADS="$THREADS"

mkdir -p "$DATASET_DIR" "$ECSW_DIR" "$LINEAR_RUNS" "$LOG_DIR" "$MPLCONFIGDIR"

if [[ ! -f "$BASIS" ]]; then echo "Missing BASIS: $BASIS" >&2; exit 1; fi
if [[ ! -f "$UREF" ]]; then echo "Missing UREF: $UREF" >&2; exit 1; fi

run_stage2() {
  echo "[linear-hprom-2pct] rebuilding ECSW and Stage2 linear HPROM qN data"
  echo "[linear-hprom-2pct] output root: $PAPER_ROOT"
  echo "[linear-hprom-2pct] basis:       $BASIS"
  echo "[linear-hprom-2pct] u_ref:       $UREF"
  echo "[linear-hprom-2pct] ECSW dir:    $ECSW_DIR"
  echo "[linear-hprom-2pct] threads:     $THREADS"

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
    --ecsw-svd-rel-tol 1e-8 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --no-save-rom-snaps \
    --no-plots \
    2>&1 | tee "$LOG_DIR/stage2_linear_hprom_ntot151_ecsw2pct_direct_svd.log"

  grep -E "solve_backend|dataset_dir|basis_path|u_ref_path|total_modes|n_ecsw_elements|ecsw_residual|ecsw_snapshot_mode|ecsw_snapshot_percent|ecsw_num_selected_total|ecsw_num_selected_per_mu|ecsw_svd_method|ecsw_svd_relative_tolerance|ecsw_weights_path" \
    "$DATASET_DIR/stage2_summary.txt" | tee "$LOG_DIR/stage2_linear_hprom_ntot151_ecsw2pct_direct_svd_check.txt"
}

run_one_linear_hprom() {
  local label="$1"
  local mu1="$2"
  local mu2="$3"
  local tag="mu1_${mu1}_mu2_${mu2}"

  echo "================ ${label}: linear HPROM mu=(${mu1},${mu2}) ================"
  python3 -u run_prom.py \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --ecsw-weights-path "$ECSW_WEIGHTS" \
    --ecsw-svd-rel-tol 1e-8 \
    --output-root "$LINEAR_RUNS" \
    --no-save-rom-snaps \
    2>&1 | tee "$LOG_DIR/linear_hprom_${label}_${tag}.log"
}

run_4pts() {
  if [[ ! -f "$ECSW_WEIGHTS" ]]; then
    echo "Missing ECSW weights: $ECSW_WEIGHTS" >&2
    echo "Run: bash Results_Paper/scripts/run_mlspg_hprom_linear_2pct_direct_svd.sh stage2" >&2
    exit 1
  fi

  run_one_linear_hprom verification       4.875 0.0225
  run_one_linear_hprom offgrid1           4.560 0.0190
  run_one_linear_hprom offgrid2           5.190 0.0260
  run_one_linear_hprom extrapolation20pct 4.000 0.0330

  for f in "$LINEAR_RUNS"/*/summary.txt; do
    echo "==== $(basename "$(dirname "$f")")"
    grep -E "solve_backend_effective|basis_path|ecsw_weights_path|ecsw_weights_source|ecsw_svd_method|ecsw_svd_relative_tolerance|n_ecsw_elements|online_solve_elapsed_s|relative_error_percent|output_dir" "$f"
  done | tee "$LOG_DIR/linear_hprom_4pts_quick_summary.txt"
}

case "$MODE" in
  stage2)
    run_stage2
    ;;
  4pts)
    run_4pts
    ;;
  all)
    run_stage2
    run_4pts
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use one of: stage2, 4pts, all" >&2
    exit 2
    ;;
esac
