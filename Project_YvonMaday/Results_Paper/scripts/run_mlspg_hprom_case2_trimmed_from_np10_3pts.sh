#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODEL="$PAPER_ROOT/Stage3/models/case2_ann_ntot151_np10_best.pt"

export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

ECSW_PERCENT="${1:-1.0}"
case "$ECSW_PERCENT" in
  1|1.0)
    ECSW_PERCENT="1.0"
    ECSW_TAG="ECSW1pct"
    ;;
  2|2.0)
    ECSW_PERCENT="2.0"
    ECSW_TAG="ECSW2pct"
    ;;
  *)
    echo "[error] ECSW percentage must be 1.0 or 2.0, got: $ECSW_PERCENT" >&2
    exit 2
    ;;
esac

python3 "$SCRIPT_DIR/normalize_mlspg_hprom_main_layout.py"

for required in "$BASIS" "$UREF" "$MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$MPLCONFIGDIR"

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
}

FAMILY_PATH="Case2_TrimmedFromNp10/np20"
OUTPUT_ROOT="$PAPER_ROOT/Runs/$ECSW_TAG/$FAMILY_PATH"
WEIGHTS_DIR="$PAPER_ROOT/ECSW/${ECSW_TAG#ECSW}/$FAMILY_PATH"
LOG_DIR="$PAPER_ROOT/logs/online/$ECSW_TAG/$FAMILY_PATH"
RUNNER="run_prom_ann_case_2.py"
RUN_TAG_EXTRA="trimmed_from_np10"

mkdir -p "$OUTPUT_ROOT" "$WEIGHTS_DIR" "$LOG_DIR"

MODEL_BASE="$(basename "$MODEL" .pt)"
WEIGHTS="$WEIGHTS_DIR/ecsw_weights_ann_case2_${MODEL_BASE}_n20_ntot151.npy"

if [[ -f "$WEIGHTS" ]]; then
  echo "[build] Reusing Case 2 trimmed-from-np10 ECSW ${ECSW_PERCENT}% rule: $WEIGHTS"
else
  echo "[build] Constructing Case 2 trimmed-from-np10 ECSW ${ECSW_PERCENT}% rule once."
  set_threads 24
  python3 -u "$RUNNER" \
    --backend hprom \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device cuda \
    --model-path "$MODEL" \
    --target-primary-modes 20 \
    --drop-first-secondary 10 \
    --run-tag-extra "$RUN_TAG_EXTRA" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$OUTPUT_ROOT" \
    --ecsw-weights-dir "$WEIGHTS_DIR" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --rebuild-ecsw \
    --ecsw-only \
    2>&1 | tee "$LOG_DIR/ecsw_build.log"

  if [[ ! -f "$WEIGHTS" ]]; then
    echo "[error] ECSW build completed without producing: $WEIGHTS" >&2
    exit 1
  fi
fi

run_point() {
  local mu1="$1"
  local mu2="$2"
  local mu1_tag
  local mu2_tag
  local run_tag

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  run_tag="case2_hprom_ann_${RUN_TAG_EXTRA}_mu1_${mu1_tag}_mu2_${mu2_tag}_n20_ntot151"

  if [[ \
    -f "$OUTPUT_ROOT/${run_tag}_summary.txt" && \
    -f "$OUTPUT_ROOT/${run_tag}_snaps.npy" && \
    -f "$OUTPUT_ROOT/${run_tag}_qN.npy" \
  ]]; then
    echo "[skip] Case 2 trimmed-from-np10 already complete at mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] Case 2 trimmed-from-np10 with ECSW ${ECSW_PERCENT}% | mu=(${mu1}, ${mu2})"
  set_threads 1
  python3 -u "$RUNNER" \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device cuda \
    --model-path "$MODEL" \
    --target-primary-modes 20 \
    --drop-first-secondary 10 \
    --run-tag-extra "$RUN_TAG_EXTRA" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$OUTPUT_ROOT" \
    --ecsw-weights-dir "$WEIGHTS_DIR" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$LOG_DIR/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_point 4.560 0.0190
run_point 4.875 0.0225
run_point 5.190 0.0260

SUMMARY="$LOG_DIR/case2_trimmed_from_np10_3pts_summary.txt"
{
  echo "[campaign] MLSPG-sensitive baseline Case 2 diagnostic: n=20 online split using trimmed n=10 checkpoint"
  echo "basis: $BASIS"
  echo "u_ref: $UREF"
  echo "model: $MODEL"
  echo "target_primary_modes: 20"
  echo "drop_first_secondary: 10"
  echo "ECSW_PERCENT: $ECSW_PERCENT"
  echo
  find "$OUTPUT_ROOT" -type f -name "*_summary.txt" -print0 \
    | sort -z \
    | while IFS= read -r -d '' summary; do
        echo "==== ${summary#$PAPER_ROOT/}"
        grep -E \
          "mu_test|model_path|checkpoint_primary_modes|target_primary_modes|drop_first_secondary|trimmed_from_checkpoint|ecsw_snapshot_percent|ecsw_weights_path|ecsw_residual|n_ecsw_elements|ecsw_setup_elapsed_s|online_solve_elapsed_s|relative_error_percent|qN_source|qN_output|snaps_output" \
          "$summary"
      done
} | tee "$SUMMARY"

echo "[done] Outputs: $OUTPUT_ROOT"
echo "[done] ECSW rules: $WEIGHTS_DIR"
echo "[done] Logs:    $LOG_DIR"
echo "[done] Summary: $SUMMARY"
