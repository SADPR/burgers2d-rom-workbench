#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
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

CASE1_MODEL="$PAPER_ROOT/Stage3/models/case1_ann_ntot151_best.pt"
CASE2_NP10_MODEL="$PAPER_ROOT/Stage3/models/case2_ann_ntot151_np10_best.pt"
CASE2_NP20_MODEL="$PAPER_ROOT/Stage3/models/case2_ann_ntot151_np20_best.pt"
CASE3_MODEL="$PAPER_ROOT/Stage3/models/case3_ann_ntot151_best.pt"

for required in \
  "$BASIS" \
  "$UREF" \
  "$CASE1_MODEL" \
  "$CASE2_NP10_MODEL" \
  "$CASE2_NP20_MODEL" \
  "$CASE3_MODEL"
do
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

weights_path() {
  local case_name="$1"
  local model="$2"
  local primary="$3"
  local weights_dir="$4"
  local model_base
  model_base="$(basename "$model" .pt)"
  printf "%s/ecsw_weights_ann_%s_%s_n%s_ntot151.npy" \
    "$weights_dir" "$case_name" "$model_base" "$primary"
}

build_ecsw_once() {
  local label="$1"
  local case_name="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local output_root="$6"
  local weights_dir="$7"
  local log_dir="$8"
  local weights
  weights="$(weights_path "$case_name" "$model" "$primary" "$weights_dir")"

  mkdir -p "$output_root" "$weights_dir" "$log_dir"
  if [[ -f "$weights" ]]; then
    echo "[build] Reusing ${label} ECSW ${ECSW_PERCENT}% rule: $weights"
    return
  fi

  echo "[build] Constructing ${label} ECSW ${ECSW_PERCENT}% rule once."
  set_threads 24
  python3 -u "$runner" \
    --backend hprom \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device cuda \
    --model-path "$model" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --rebuild-ecsw \
    --ecsw-only \
    2>&1 | tee "$log_dir/ecsw_build.log"

  if [[ ! -f "$weights" ]]; then
    echo "[error] ECSW build completed without producing: $weights" >&2
    exit 1
  fi
}

run_point() {
  local label="$1"
  local case_name="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local output_root="$6"
  local weights_dir="$7"
  local log_dir="$8"
  local mu1="$9"
  local mu2="${10}"
  local mu1_tag
  local mu2_tag
  local run_tag

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  run_tag="${case_name}_hprom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n${primary}_ntot151"

  if [[ \
    -f "$output_root/${run_tag}_summary.txt" && \
    -f "$output_root/${run_tag}_snaps.npy" && \
    -f "$output_root/${run_tag}_qN.npy" \
  ]]; then
    echo "[skip] ${label} already complete at mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] ${label} with ECSW ${ECSW_PERCENT}% | mu=(${mu1}, ${mu2})"
  set_threads 1
  python3 -u "$runner" \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device cuda \
    --model-path "$model" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
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
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_family() {
  local label="$1"
  local case_name="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local family_path="$6"
  local output_root="$PAPER_ROOT/Runs/$ECSW_TAG/$family_path"
  local weights_dir="$PAPER_ROOT/ECSW/${ECSW_TAG#ECSW}/$family_path"
  local log_dir="$PAPER_ROOT/logs/online/$ECSW_TAG/$family_path"

  build_ecsw_once \
    "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir"

  run_point "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir" 4.560 0.0190
  run_point "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir" 4.875 0.0225
  run_point "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir" 5.190 0.0260
}

run_family \
  "Case 1 Best" case1 run_prom_ann_case_1.py \
  "$CASE1_MODEL" 10 "Case1_Best"

run_family \
  "Case 2 Best (n_p=10)" case2 run_prom_ann_case_2.py \
  "$CASE2_NP10_MODEL" 10 "Case2_Best/np10"

run_family \
  "Case 2 Best (n_p=20)" case2 run_prom_ann_case_2.py \
  "$CASE2_NP20_MODEL" 20 "Case2_Best/np20"

run_family \
  "Case 3 Best" case3 run_prom_ann_case_3.py \
  "$CASE3_MODEL" 10 "Case3_Best"

SUMMARY="$PAPER_ROOT/logs/online/$ECSW_TAG/all_best_3pts_summary.txt"
mkdir -p "$(dirname "$SUMMARY")"
{
  echo "[campaign] MLSPG-sensitive HPROM best models with ECSW ${ECSW_PERCENT}%"
  echo "basis: $BASIS"
  echo "u_ref: $UREF"
  echo
  find "$PAPER_ROOT/Runs/$ECSW_TAG" -type f -name "*_summary.txt" -print0 \
    | sort -z \
    | while IFS= read -r -d '' summary; do
        echo "==== ${summary#$PAPER_ROOT/}"
        grep -E \
          "mu_test|model_path|ecsw_snapshot_percent|ecsw_weights_path|ecsw_weights_source|ecsw_residual|n_ecsw_elements|ecsw_setup_elapsed_s|online_solve_elapsed_s|relative_error_percent|qN_source|qN_output|snaps_output" \
          "$summary"
      done
} | tee "$SUMMARY"

echo "[done] ECSW ${ECSW_PERCENT}% outputs: $PAPER_ROOT/Runs/$ECSW_TAG"
echo "[done] ECSW ${ECSW_PERCENT}% rules:   $PAPER_ROOT/ECSW/${ECSW_TAG#ECSW}"
echo "[done] Summary:          $SUMMARY"
