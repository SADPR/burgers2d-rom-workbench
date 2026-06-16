#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
PAPER_ROOT="$PROJECT_DIR/Results_Paper/$PAPER_TAG"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODEL="$PAPER_ROOT/Stage3/models/data_driven_ann_ntot151_best.pt"
OUT_ROOT="$PAPER_ROOT/Runs/DataDriven_Best"
LOG_DIR="$PAPER_ROOT/logs/online/DataDriven_Best"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

python3 "$SCRIPT_DIR/normalize_mlspg_hprom_main_layout.py"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"

for required in "$BASIS" "$UREF" "$MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required file: $required" >&2
    exit 1
  fi
done

run_point() {
  local mu1="$1"
  local mu2="$2"
  local log_file="$LOG_DIR/data_driven_best_mu1_${mu1}_mu2_${mu2}.log"

  echo "[run] Data-driven Best prediction for mu=(${mu1}, ${mu2})"
  python3 -u run_rom_data_driven.py \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --device auto \
    --model-path "$MODEL" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$OUT_ROOT" \
    2>&1 | tee "$log_file"
}

run_point 4.560 0.0190
run_point 4.875 0.0225
run_point 5.190 0.0260

summary_file="$LOG_DIR/data_driven_best_3pts_quick_summary.txt"
{
  echo "[quick-summary] $(date)"
  echo "model: $MODEL"
  echo "basis: $BASIS"
  echo "u_ref: $UREF"
  for f in "$OUT_ROOT"/*/rom_data_driven_summary.txt; do
    [[ -f "$f" ]] || continue
    echo "==== $(basename "$(dirname "$f")")"
    grep -E "mu_test|model_path|basis_path|u_ref_path|total_modes_used|inference_time_s|relative_error_percent|save_rom_snaps|output_dir" "$f"
  done
} | tee "$summary_file"

echo "[done] Outputs: $OUT_ROOT"
echo "[done] Logs:    $LOG_DIR"
echo "[done] Summary: $summary_file"
