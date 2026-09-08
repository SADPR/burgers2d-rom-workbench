#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
MODEL="$PAPER_ROOT/Stage3/models/pod_dl_data_driven_ntot151_best.pt"
OUT_ROOT="$PAPER_ROOT/Runs/PODDL_Best"
LOG_DIR="$PAPER_ROOT/logs/online/PODDL_Best"
FORCE="${FORCE:-0}"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

python3 "$SCRIPT_DIR/normalize_mlspg_hprom_main_layout.py"

if [[ ! -f "$MODEL" ]]; then
  echo "[error] Missing required file: $MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"

if [[ "$FORCE" == "1" ]]; then
  echo "[clean] FORCE=1: removing previous POD-DL Best outputs."
  rm -rf "$OUT_ROOT"
  mkdir -p "$OUT_ROOT"
fi

export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_point() {
  local mu1="$1"
  local mu2="$2"
  local mu1_tag
  local mu2_tag
  local log_file

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  log_file="$LOG_DIR/pod_dl_best_mu1_${mu1_tag}_mu2_${mu2_tag}.log"

  echo "[run] POD-DL Best prediction for mu=(${mu1}, ${mu2})"
  python3 -u run_pod_dl_data_driven.py \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --device auto \
    --model-path "$MODEL" \
    --output-root "$OUT_ROOT" \
    2>&1 | tee "$log_file"
}

run_point 4.560 0.0190
run_point 4.875 0.0225
run_point 5.190 0.0260

SUMMARY="$LOG_DIR/pod_dl_best_3pts_summary.txt"
{
  echo "[quick-summary] $(date)"
  echo "model: $MODEL"
  for f in "$OUT_ROOT"/*/pod_dl_data_driven_summary.txt; do
    [[ -f "$f" ]] || continue
    echo "==== $(basename "$(dirname "$f")")"
    grep -E "mu_test|model_path|basis_path|u_ref_path|model_ntot|total_modes_used|latent_dim|inference_time_s|relative_error_percent|output_dir" "$f"
  done
} | tee "$SUMMARY"

echo "[done] Outputs: $OUT_ROOT"
echo "[done] Logs:    $LOG_DIR"
echo "[done] Summary: $SUMMARY"
