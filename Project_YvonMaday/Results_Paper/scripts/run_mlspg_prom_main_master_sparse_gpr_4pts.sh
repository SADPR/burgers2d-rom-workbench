#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"  # all | data | case2

export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_TAG="mlspg_prom_main"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"
export LOG_DIR="$PAPER_ROOT/logs/master_sparse_gpr_4pts"

export PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$PROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$PROM_NUM_THREADS"
export MKL_NUM_THREADS="$PROM_NUM_THREADS"
export BLIS_NUM_THREADS="$PROM_NUM_THREADS"
export GOTO_NUM_THREADS="$PROM_NUM_THREADS"

ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
NO_PLOT="${NO_PLOT:-0}"
N_PRIMARY="${N_PRIMARY:-10}"

SPARSE_GPR_LABEL="${SPARSE_GPR_LABEL:-matern15_ard_m451_fixed}"
SPARSE_GPR_MODEL_PATH="${SPARSE_GPR_MODEL_PATH:-$PAPER_ROOT/Stage3/models/master_sparse_gpr_mu_t_to_qtot_ntot151_${SPARSE_GPR_LABEL}.pt}"

BASIS_PATH="${BASIS_PATH:-$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
U_REF_PATH="${U_REF_PATH:-$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"

DATA_ROOT="$PAPER_ROOT/Runs/DataDriven_MasterSparseGPR"
CASE2_ROOT="$PAPER_ROOT/Runs/PROM/Case2_MasterSparseGPR_NSweep/np${N_PRIMARY}"

mkdir -p "$MPLCONFIGDIR" "$LOG_DIR" "$DATA_ROOT" "$CASE2_ROOT"

if [[ ! -f "$SPARSE_GPR_MODEL_PATH" ]]; then
  echo "Missing sparse-GPR checkpoint: $SPARSE_GPR_MODEL_PATH" >&2
  echo "Run Results_Paper/scripts/run_mlspg_prom_main_train_master_sparse_gpr.sh first." >&2
  exit 1
fi

if [[ ! -f "$BASIS_PATH" ]]; then
  echo "Missing basis: $BASIS_PATH" >&2
  exit 1
fi

if [[ ! -f "$U_REF_PATH" ]]; then
  echo "Missing u_ref: $U_REF_PATH" >&2
  exit 1
fi

plot_args=()
if [[ "$NO_PLOT" == "1" ]]; then
  plot_args+=(--no-plot)
fi

echo "[master-sparse-gpr-4pts] family:    $FAMILY"
echo "[master-sparse-gpr-4pts] model:     $SPARSE_GPR_MODEL_PATH"
echo "[master-sparse-gpr-4pts] basis:     $BASIS_PATH"
echo "[master-sparse-gpr-4pts] u_ref:     $U_REF_PATH"
echo "[master-sparse-gpr-4pts] data root: $DATA_ROOT"
echo "[master-sparse-gpr-4pts] case2 root:$CASE2_ROOT"
echo "[master-sparse-gpr-4pts] n primary: $N_PRIMARY"
echo "[master-sparse-gpr-4pts] device:    $ONLINE_DEVICE"
echo "[master-sparse-gpr-4pts] threads:   $PROM_NUM_THREADS"
echo "[master-sparse-gpr-4pts] force:     $FORCE"

run_point() {
  local label="$1"
  local mu1="$2"
  local mu2="$3"

  echo
  echo "================ ${label}: mu=(${mu1},${mu2}) ================"

  local data_dir="$DATA_ROOT/rom_data_driven_mu1_${mu1}_mu2_${mu2}_ntot151"
  local data_summary="$data_dir/rom_data_driven_summary.txt"
  if [[ "$FAMILY" == "all" || "$FAMILY" == "data" ]]; then
    if [[ "$FORCE" != "1" && -f "$data_summary" ]]; then
      echo "[skip] data-driven sparse-GPR exists: $data_summary"
    else
      echo "[run] Data-driven sparse-GPR: mu=(${mu1},${mu2})"
      python3 -u run_rom_data_driven.py \
        --mu1 "$mu1" \
        --mu2 "$mu2" \
        --total-modes 151 \
        --device "$ONLINE_DEVICE" \
        --model-path "$SPARSE_GPR_MODEL_PATH" \
        --basis-path "$BASIS_PATH" \
        --u-ref-path "$U_REF_PATH" \
        --output-root "$DATA_ROOT" \
        "${plot_args[@]}" \
        2>&1 | tee "$LOG_DIR/data_${label}.log"
    fi
  fi

  local case2_summary="$CASE2_ROOT/case2_prom_ann_master_sparse_gpr_mu1_${mu1}_mu2_${mu2}_n${N_PRIMARY}_ntot151_summary.txt"
  if [[ "$FAMILY" == "all" || "$FAMILY" == "case2" ]]; then
    if [[ "$FORCE" != "1" && -f "$case2_summary" ]]; then
      echo "[skip] Case 2 PROM sparse-GPR exists: $case2_summary"
    else
      echo "[run] Case 2 PROM sparse-GPR: mu=(${mu1},${mu2}), n=${N_PRIMARY}"
      python3 -u run_prom_ann_case_2.py \
        --backend prom \
        --no-ecsw \
        --mu1 "$mu1" \
        --mu2 "$mu2" \
        --device "$ONLINE_DEVICE" \
        --model-path "$SPARSE_GPR_MODEL_PATH" \
        --target-primary-modes "$N_PRIMARY" \
        --basis-path "$BASIS_PATH" \
        --u-ref-path "$U_REF_PATH" \
        --output-root "$CASE2_ROOT" \
        --run-tag-extra master_sparse_gpr \
        --max-its 20 \
        --relnorm-cutoff 1e-5 \
        --min-delta 1e-2 \
        --linear-solver lstsq \
        --normal-eq-reg 1e-12 \
        "${plot_args[@]}" \
        2>&1 | tee "$LOG_DIR/case2_n${N_PRIMARY}_${label}.log"
    fi
  fi
}

run_point verification        4.875 0.0225
run_point offgrid1            4.560 0.0190
run_point offgrid2            5.190 0.0260
run_point extrapolation20pct  4.000 0.0330
