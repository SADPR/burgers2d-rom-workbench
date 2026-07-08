#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

campaign="${1:-both}"
case "$campaign" in
  baseline|ext25|both) ;;
  *)
    echo "Usage: $0 [baseline|ext25|both]" >&2
    exit 2
    ;;
esac

BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
ECSW_PERCENT="${ECSW_PERCENT:-1.0}"
ECSW_TAG="ECSW1pct"
ECSW_SNAPSHOT_MODE="${ECSW_SNAPSHOT_MODE:-global_param_time_stratified}"
ECSW_NUM_TRAINING_MU="${ECSW_NUM_TRAINING_MU:-9}"
ECSW_BUILD_THREADS="${ECSW_BUILD_THREADS:-24}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE_ANN="${FORCE_ANN:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$ECSW_PERCENT" in
  1|1.0) ECSW_PERCENT="1.0" ;;
  *)
    echo "[error] This refresh launcher is fixed to 1% ANN ECSW. Got ECSW_PERCENT=$ECSW_PERCENT" >&2
    exit 2
    ;;
esac

case "$ECSW_SNAPSHOT_MODE" in
  strided_per_mu|global_stratified_random|global_param_time_stratified) ;;
  *)
    echo "[error] Unsupported ECSW_SNAPSHOT_MODE=$ECSW_SNAPSHOT_MODE" >&2
    exit 2
    ;;
esac

case "$ONLINE_DEVICE" in
  cpu|cuda|auto) ;;
  *)
    echo "[error] ONLINE_DEVICE must be cpu, cuda, or auto; got: $ONLINE_DEVICE" >&2
    exit 2
    ;;
esac

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] Missing required file: $path" >&2
    exit 1
  fi
}

ann_weights_path() {
  local case_name="$1"
  local model="$2"
  local primary="$3"
  local weights_dir="$4"
  local model_base
  model_base="$(basename "$model" .pt)"
  printf "%s/ecsw_weights_ann_%s_%s_n%s_ntot151.npy" \
    "$weights_dir" "$case_name" "$model_base" "$primary"
}

build_ann_ecsw_once() {
  local label="$1"
  local case_name="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local output_root="$6"
  local weights_dir="$7"
  local log_dir="$8"
  local weights
  weights="$(ann_weights_path "$case_name" "$model" "$primary" "$weights_dir")"

  mkdir -p "$output_root" "$weights_dir" "$log_dir"
  if [[ -f "$weights" ]]; then
    echo "[build] Reusing ${label} ANN ECSW 1% rule: $weights"
    return
  fi

  echo "[build] Constructing ${label} ANN ECSW 1% rule once."
  set_threads "$ECSW_BUILD_THREADS"
  python3 -u "$runner" \
    --backend hprom \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-snapshot-mode "$ECSW_SNAPSHOT_MODE" \
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

run_ann_point() {
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
  local point_label="${11}"
  local mu1_tag
  local mu2_tag
  local run_tag

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  run_tag="${case_name}_hprom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n${primary}_ntot151"

  mkdir -p "$output_root" "$log_dir"
  if [[ \
    -f "$output_root/${run_tag}_summary.txt" && \
    -f "$output_root/${run_tag}_snaps.npy" && \
    -f "$output_root/${run_tag}_qN.npy" \
  ]]; then
    echo "[skip] ${label} already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] ${label} | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$ONLINE_THREADS"
  python3 -u "$runner" \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-snapshot-mode "$ECSW_SNAPSHOT_MODE" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

clean_ann_family() {
  local paper_root="$1"
  local family_path="$2"
  local include_extrapolation="$3"

  rm -rf \
    "$paper_root/Runs/$ECSW_TAG/$family_path" \
    "$paper_root/ECSW/1pct/$family_path" \
    "$paper_root/logs/online/$ECSW_TAG/$family_path"

  if [[ "$include_extrapolation" == "1" ]]; then
    rm -rf \
      "$paper_root/Runs/Extrapolation20pct/$ECSW_TAG/$family_path" \
      "$paper_root/logs/online/Extrapolation20pct/$ECSW_TAG/$family_path"
  fi
}

summarize_ann_campaign() {
  local paper_root="$1"
  local summary="$2"
  mkdir -p "$(dirname "$summary")"
  {
    echo "[summary] ANN HPROM refresh after initialization fix"
    echo "paper_root: $paper_root"
    echo "basis: $BASIS"
    echo "u_ref: $UREF"
    echo "online_device: $ONLINE_DEVICE"
    echo "online_threads: $ONLINE_THREADS"
    echo "ecsw_build_threads: $ECSW_BUILD_THREADS"
    echo "ecsw_snapshot_mode: $ECSW_SNAPSHOT_MODE"
    echo
    find "$paper_root/Runs" -type f -name "*_summary.txt" -print0 \
      | sort -z \
      | while IFS= read -r -d '' f; do
          case "$f" in
            *Case1_Best*|*Case2_Best*|*Case3_Best*)
              echo "==== ${f#$paper_root/}"
              grep -E \
                "mu_test|model_path|basis_path|u_ref_path|solve_backend|target_primary_modes|checkpoint_primary_modes|ecsw_snapshot_percent|ecsw_weights_path|ecsw_residual|n_ecsw_elements|online_solve_elapsed_s|relative_error_percent|qN_output|snaps_output" \
                "$f" || true
              ;;
          esac
        done
  } | tee "$summary"
}

run_baseline() {
  local paper_root="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
  local models_dir="$paper_root/Stage3/models"
  local mpl="$paper_root/.mplcache"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$mpl}"

  local case1_model="$models_dir/case1_ann_ntot151_best.pt"
  local case2_np10_model="$models_dir/case2_ann_ntot151_np10_best.pt"
  local case2_np20_model="$models_dir/case2_ann_ntot151_np20_best.pt"
  local case3_model="$models_dir/case3_ann_ntot151_best.pt"

  local -a families=(
    "Case 1 Best|case1|run_prom_ann_case_1.py|$case1_model|10|Case1_Best"
    "Case 2 Best (n=10)|case2|run_prom_ann_case_2.py|$case2_np10_model|10|Case2_Best/np10"
    "Case 2 Best (n=20)|case2|run_prom_ann_case_2.py|$case2_np20_model|20|Case2_Best/np20"
    "Case 3 Best|case3|run_prom_ann_case_3.py|$case3_model|10|Case3_Best"
  )

  local -a points_in=(
    "4.560 0.0190 offgrid1"
    "4.875 0.0225 verification"
    "5.190 0.0260 offgrid2"
  )
  local -a points_out=(
    "4.000 0.0330 extrapolation20pct"
  )

  echo "[campaign] baseline ANN refresh: $paper_root"
  mkdir -p "$mpl"
  for required in "$BASIS" "$UREF" "$case1_model" "$case2_np10_model" "$case2_np20_model" "$case3_model"; do
    require_file "$required"
  done

  local entry label case_name runner model primary family_path
  if [[ "$FORCE_ANN" == "1" ]]; then
    echo "[clean] baseline FORCE_ANN=1: removing only ANN Case 1/2/3 outputs, logs, and ANN ECSW rules."
    for entry in "${families[@]}"; do
      IFS="|" read -r label case_name runner model primary family_path <<< "$entry"
      clean_ann_family "$paper_root" "$family_path" 1
    done
  fi

  for entry in "${families[@]}"; do
    IFS="|" read -r label case_name runner model primary family_path <<< "$entry"
    local base_output="$paper_root/Runs/$ECSW_TAG/$family_path"
    local weights_dir="$paper_root/ECSW/1pct/$family_path"
    local base_log="$paper_root/logs/online/$ECSW_TAG/$family_path"
    local out_output="$paper_root/Runs/Extrapolation20pct/$ECSW_TAG/$family_path"
    local out_log="$paper_root/logs/online/Extrapolation20pct/$ECSW_TAG/$family_path"

    build_ann_ecsw_once "$label" "$case_name" "$runner" "$model" "$primary" \
      "$base_output" "$weights_dir" "$base_log"

    local point mu1 mu2 point_label
    for point in "${points_in[@]}"; do
      read -r mu1 mu2 point_label <<< "$point"
      run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
        "$base_output" "$weights_dir" "$base_log" "$mu1" "$mu2" "$point_label"
    done
    for point in "${points_out[@]}"; do
      read -r mu1 mu2 point_label <<< "$point"
      run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
        "$out_output" "$weights_dir" "$out_log" "$mu1" "$mu2" "$point_label"
    done
  done

  summarize_ann_campaign "$paper_root" "$paper_root/logs/online/ann_refresh_after_init_fix_summary.txt"
}

run_ext25() {
  local paper_root="$PROJECT_DIR/Results_Paper/mlspg_hprom_enrichment_ext25_lhs36"
  local models_dir="$paper_root/Stage3/models"
  local mpl="$paper_root/.mplcache"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$mpl}"

  local case1_model="$models_dir/case1_ann_ntot151_best.pt"
  local case2_np10_model="$models_dir/case2_ann_ntot151_np10_best.pt"
  local case2_np20_model="$models_dir/case2_ann_ntot151_np20_best.pt"
  local case3_model="$models_dir/case3_ann_ntot151_best.pt"

  local -a families=(
    "Case 1 Best|case1|run_prom_ann_case_1.py|$case1_model|10|Case1_Best"
    "Case 2 Best (n=10)|case2|run_prom_ann_case_2.py|$case2_np10_model|10|Case2_Best/np10"
    "Case 2 Best (n=20)|case2|run_prom_ann_case_2.py|$case2_np20_model|20|Case2_Best/np20"
    "Case 3 Best|case3|run_prom_ann_case_3.py|$case3_model|10|Case3_Best"
  )
  local -a points=(
    "4.875 0.0225 verification"
    "4.560 0.0190 offgrid1"
    "5.190 0.0260 offgrid2"
    "4.000 0.0330 extrapolation20pct"
  )

  echo "[campaign] ext25-lhs36 ANN refresh: $paper_root"
  mkdir -p "$mpl"
  for required in "$BASIS" "$UREF" "$case1_model" "$case2_np10_model" "$case2_np20_model" "$case3_model"; do
    require_file "$required"
  done

  local entry label case_name runner model primary family_path
  if [[ "$FORCE_ANN" == "1" ]]; then
    echo "[clean] ext25 FORCE_ANN=1: removing only ANN Case 1/2/3 outputs, logs, and ANN ECSW rules."
    for entry in "${families[@]}"; do
      IFS="|" read -r label case_name runner model primary family_path <<< "$entry"
      clean_ann_family "$paper_root" "$family_path" 0
    done
  fi

  for entry in "${families[@]}"; do
    IFS="|" read -r label case_name runner model primary family_path <<< "$entry"
    local output_root="$paper_root/Runs/$ECSW_TAG/$family_path"
    local weights_dir="$paper_root/ECSW/1pct/$family_path"
    local log_dir="$paper_root/logs/online/$ECSW_TAG/$family_path"

    build_ann_ecsw_once "$label" "$case_name" "$runner" "$model" "$primary" \
      "$output_root" "$weights_dir" "$log_dir"

    local point mu1 mu2 point_label
    for point in "${points[@]}"; do
      read -r mu1 mu2 point_label <<< "$point"
      run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
        "$output_root" "$weights_dir" "$log_dir" "$mu1" "$mu2" "$point_label"
    done
  done

  summarize_ann_campaign "$paper_root" "$paper_root/logs/online/ann_refresh_after_init_fix_summary.txt"
}

cat <<EOF
[ann-refresh] campaign: $campaign
[ann-refresh] policy: refresh only ANN Case 1, Case 2 n=10, Case 2 n=20, Case 3
[ann-refresh] nonlinear ECSW: rebuilt only when FORCE_ANN=1 or missing
[ann-refresh] linear/POD-AE/POD-NN/POD-DL outputs are untouched
[ann-refresh] basis: $BASIS
[ann-refresh] u_ref: $UREF
[ann-refresh] online_device: $ONLINE_DEVICE
[ann-refresh] online_threads: $ONLINE_THREADS
[ann-refresh] ecsw_build_threads: $ECSW_BUILD_THREADS
[ann-refresh] ecsw_snapshot_mode: $ECSW_SNAPSHOT_MODE
[ann-refresh] force_ann: $FORCE_ANN
EOF

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[ann-refresh] PLAN_ONLY=1; no file checks, cleanup, ECSW build, or online solves were run."
  exit 0
fi

case "$campaign" in
  baseline) run_baseline ;;
  ext25) run_ext25 ;;
  both)
    run_baseline
    run_ext25
    ;;
esac

echo "[ann-refresh] complete"
