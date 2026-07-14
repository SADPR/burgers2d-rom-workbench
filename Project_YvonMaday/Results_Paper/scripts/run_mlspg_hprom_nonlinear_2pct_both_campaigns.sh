#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

CAMPAIGN="${1:-both}"
case "$CAMPAIGN" in
  baseline|ext25|both) ;;
  *)
    echo "Usage: $0 [baseline|ext25|both]" >&2
    exit 2
    ;;
esac

BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
ECSW_PERCENT="${ECSW_PERCENT:-2.0}"
ECSW_TAG="ECSW2pct"
ECSW_DIR_TAG="2pct"
ECSW_NUM_TRAINING_MU="${ECSW_NUM_TRAINING_MU:-9}"
ECSW_SNAPSHOT_MODE="${ECSW_SNAPSHOT_MODE:-global_param_time_stratified}"
ECSW_RANDOM_SEED="${ECSW_RANDOM_SEED:-42}"
ECSW_SVD_REL_TOL="${ECSW_SVD_REL_TOL:-1e-8}"
ECSW_BUILD_THREADS="${ECSW_BUILD_THREADS:-24}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cuda}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$ECSW_PERCENT" in
  2|2.0) ECSW_PERCENT="2.0" ;;
  *)
    echo "[error] This production launcher is fixed to learned-intrusive ECSW_PERCENT=2.0. Got $ECSW_PERCENT" >&2
    exit 2
    ;;
esac

case "$ECSW_SNAPSHOT_MODE" in
  global_param_time_stratified|global_stratified_random|strided_per_mu) ;;
  *)
    echo "[error] Unsupported ECSW_SNAPSHOT_MODE=$ECSW_SNAPSHOT_MODE" >&2
    exit 2
    ;;
esac

case "$ONLINE_DEVICE" in
  cpu|cuda) ;;
  *)
    echo "[error] ONLINE_DEVICE must be cpu or cuda; got: $ONLINE_DEVICE" >&2
    exit 2
    ;;
esac

POINTS=(
  "4.875 0.0225 verification"
  "4.560 0.0190 offgrid1"
  "5.190 0.0260 offgrid2"
  "4.000 0.0330 extrapolation20pct"
)

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

ann_extra_args() {
  local case_name="$1"
  local primary="$2"
  if [[ "$case_name" == "case2" ]]; then
    printf '%s\n' --target-primary-modes "$primary"
  fi
  if [[ "$case_name" == "case3" ]]; then
    printf '%s\n' --ecsw-svd-rel-tol "$ECSW_SVD_REL_TOL"
  fi
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
  local -a extra=()
  weights="$(ann_weights_path "$case_name" "$model" "$primary" "$weights_dir")"
  while IFS= read -r arg; do
    extra+=("$arg")
  done < <(ann_extra_args "$case_name" "$primary")

  mkdir -p "$output_root" "$weights_dir" "$log_dir"
  if [[ -f "$weights" ]]; then
    echo "[build] Reusing ${label} ECSW ${ECSW_PERCENT}% rule: $weights"
    return
  fi

  echo "[build] Constructing ${label} ECSW ${ECSW_PERCENT}% rule once."
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
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    ${extra[@]+"${extra[@]}"} \
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
  local mu1_tag mu2_tag run_tag
  local -a extra=()

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  run_tag="${case_name}_hprom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n${primary}_ntot151"
  while IFS= read -r arg; do
    extra+=("$arg")
  done < <(ann_extra_args "$case_name" "$primary")

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
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    ${extra[@]+"${extra[@]}"} \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

build_podae_ecsw_once() {
  local model="$1"
  local output_root="$2"
  local weights_dir="$3"
  local log_dir="$4"
  local weights="$weights_dir/ecsw_weights_pod_ae_ntot151.npy"

  mkdir -p "$output_root" "$weights_dir" "$log_dir"
  if [[ -f "$weights" ]]; then
    echo "[build] Reusing PROM-POD-AE ECSW ${ECSW_PERCENT}% rule: $weights"
    return
  fi

  echo "[build] Constructing PROM-POD-AE ECSW ${ECSW_PERCENT}% rule once."
  set_threads "$ECSW_BUILD_THREADS"
  python3 -u run_prom_pod_ae.py \
    --backend hprom \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    --rebuild-ecsw \
    --ecsw-only \
    2>&1 | tee "$log_dir/ecsw_build.log"

  if [[ ! -f "$weights" ]]; then
    echo "[error] PROM-POD-AE ECSW build completed without producing: $weights" >&2
    exit 1
  fi
}

run_podae_point() {
  local model="$1"
  local output_root="$2"
  local weights_dir="$3"
  local log_dir="$4"
  local mu1="$5"
  local mu2="$6"
  local point_label="$7"
  local mu1_tag mu2_tag stem latent_dim

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  latent_dim="$(python3 - <<PY
import torch
ck = torch.load("$model", map_location="cpu")
print(int(ck["latent_dim"]))
PY
)"
  stem="podae_hprom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"

  mkdir -p "$output_root" "$log_dir"
  if [[ \
    -f "$output_root/${stem}_summary.txt" && \
    -f "$output_root/${stem}_snaps.npy" && \
    -f "$output_root/${stem}_qN.npy" \
  ]]; then
    echo "[skip] PROM-POD-AE already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] PROM-POD-AE | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$ONLINE_THREADS"
  python3 -u run_prom_pod_ae.py \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

clean_family() {
  local root="$1"
  local family_path="$2"
  local include_baseline_extrap="$3"
  rm -rf \
    "$root/Runs/$ECSW_TAG/$family_path" \
    "$root/ECSW/$ECSW_DIR_TAG/$family_path" \
    "$root/logs/online/$ECSW_TAG/$family_path"
  if [[ "$include_baseline_extrap" == "1" ]]; then
    rm -rf \
      "$root/Runs/Extrapolation20pct/$ECSW_TAG/$family_path" \
      "$root/logs/online/Extrapolation20pct/$ECSW_TAG/$family_path"
  fi
}

run_ann_family_for_campaign() {
  local campaign_name="$1"
  local root="$2"
  local label="$3"
  local case_name="$4"
  local runner="$5"
  local model="$6"
  local primary="$7"
  local family_path="$8"
  local include_baseline_extrap="$9"
  local weights_dir="$root/ECSW/$ECSW_DIR_TAG/$family_path"
  local build_output="$root/Runs/$ECSW_TAG/$family_path"
  local build_logs="$root/logs/online/$ECSW_TAG/$family_path"

  require_file "$model"
  if [[ "$FORCE" == "1" ]]; then
    clean_family "$root" "$family_path" "$include_baseline_extrap"
  fi

  build_ann_ecsw_once "$campaign_name ${label}" "$case_name" "$runner" "$model" "$primary" \
    "$build_output" "$weights_dir" "$build_logs"

  local point mu1 mu2 point_label output_root log_dir
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    if [[ "$include_baseline_extrap" == "1" && "$point_label" == "extrapolation20pct" ]]; then
      output_root="$root/Runs/Extrapolation20pct/$ECSW_TAG/$family_path"
      log_dir="$root/logs/online/Extrapolation20pct/$ECSW_TAG/$family_path"
    else
      output_root="$root/Runs/$ECSW_TAG/$family_path"
      log_dir="$root/logs/online/$ECSW_TAG/$family_path"
    fi
    run_ann_point "$campaign_name ${label}" "$case_name" "$runner" "$model" "$primary" \
      "$output_root" "$weights_dir" "$log_dir" "$mu1" "$mu2" "$point_label"
  done
}

run_podae_family_for_campaign() {
  local campaign_name="$1"
  local root="$2"
  local model="$3"
  local family_path="PODAE_Best"
  local include_baseline_extrap="$4"
  local weights_dir="$root/ECSW/$ECSW_DIR_TAG/$family_path"
  local build_output="$root/Runs/$ECSW_TAG/$family_path"
  local build_logs="$root/logs/online/$ECSW_TAG/$family_path"

  require_file "$model"
  if [[ "$FORCE" == "1" ]]; then
    clean_family "$root" "$family_path" "$include_baseline_extrap"
  fi

  build_podae_ecsw_once "$model" "$build_output" "$weights_dir" "$build_logs"

  local point mu1 mu2 point_label output_root log_dir
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    if [[ "$include_baseline_extrap" == "1" && "$point_label" == "extrapolation20pct" ]]; then
      output_root="$root/Runs/Extrapolation20pct/$ECSW_TAG/$family_path"
      log_dir="$root/logs/online/Extrapolation20pct/$ECSW_TAG/$family_path"
    else
      output_root="$root/Runs/$ECSW_TAG/$family_path"
      log_dir="$root/logs/online/$ECSW_TAG/$family_path"
    fi
    run_podae_point "$model" "$output_root" "$weights_dir" "$log_dir" "$mu1" "$mu2" "$point_label"
  done
}

summarize_campaign() {
  local root="$1"
  local label="$2"
  local summary="$root/logs/online/${ECSW_TAG}_learned_intrusive_4pts_summary.txt"
  mkdir -p "$(dirname "$summary")"
  {
    echo "[summary] $label learned-intrusive HPROM production rerun"
    echo "ecsw_percent: $ECSW_PERCENT"
    echo "ecsw_snapshot_mode: $ECSW_SNAPSHOT_MODE"
    echo "ecsw_random_seed: $ECSW_RANDOM_SEED"
    echo "case3_ecsw_svd_rel_tol: $ECSW_SVD_REL_TOL"
    echo "basis: $BASIS"
    echo "u_ref: $UREF"
    echo
    find "$root/Runs" -type f -name "*_summary.txt" -print0 \
      | sort -z \
      | while IFS= read -r -d '' f; do
          case "$f" in
            *"/$ECSW_TAG/Case1"*|*"/$ECSW_TAG/Case2"*|*"/$ECSW_TAG/Case3"*|*"/$ECSW_TAG/PODAE"*)
              echo "==== ${f#$root/}"
              grep -E "mu_test|model_path|solve_backend|target_primary_modes|checkpoint_primary_modes|ecsw_snapshot_percent|ecsw_snapshot_mode|ecsw_snapshot_random_seed|ecsw_svd_rel_tol|ecsw_weights_path|ecsw_residual|n_ecsw_elements|online_solve_elapsed_s|relative_error_percent|qN_output|snaps_output" "$f" || true
              ;;
          esac
        done
  } | tee "$summary"
  echo "[summary] $summary"
}

run_baseline() {
  local root="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
  local models="$root/Stage3/models"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/.mplcache}"
  mkdir -p "$MPLCONFIGDIR"

  run_ann_family_for_campaign "baseline" "$root" "PROM-ANN Case 1" case1 run_prom_ann_case_1.py \
    "$models/case1_ann_ntot151_best.pt" 10 "Case1_Best" 1
  run_ann_family_for_campaign "baseline" "$root" "Case 2 from POD-NN master map (n=10)" case2 run_prom_ann_case_2.py \
    "$models/data_driven_ann_ntot151_best.pt" 10 "Case2_Master/np10" 1
  run_ann_family_for_campaign "baseline" "$root" "Case 2 from POD-NN master map (n=20)" case2 run_prom_ann_case_2.py \
    "$models/data_driven_ann_ntot151_best.pt" 20 "Case2_Master/np20" 1
  run_ann_family_for_campaign "baseline" "$root" "PROM-ANN Case 3" case3 run_prom_ann_case_3.py \
    "$models/case3_ann_ntot151_best.pt" 10 "Case3_Best" 1
  run_podae_family_for_campaign "baseline" "$root" "$models/prom_pod_ae_ntot151_best.pt" 1
  summarize_campaign "$root" "baseline"
}

run_ext25() {
  local root="$PROJECT_DIR/Results_Paper/mlspg_hprom_enrichment_ext25_lhs36"
  local models="$root/Stage3/models"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/.mplcache}"
  mkdir -p "$MPLCONFIGDIR"

  run_ann_family_for_campaign "ext25-lhs36" "$root" "PROM-ANN Case 1" case1 run_prom_ann_case_1.py \
    "$models/case1_ann_ntot151_gelu_samearch_test.pt" 10 "Case1_GELU_SameArch_Test" 0
  run_ann_family_for_campaign "ext25-lhs36" "$root" "Case 2 from POD-NN master map (n=10)" case2 run_prom_ann_case_2.py \
    "$models/data_driven_ann_ntot151_best.pt" 10 "Case2_Master/np10" 0
  run_ann_family_for_campaign "ext25-lhs36" "$root" "Case 2 from POD-NN master map (n=20)" case2 run_prom_ann_case_2.py \
    "$models/data_driven_ann_ntot151_best.pt" 20 "Case2_Master/np20" 0
  run_ann_family_for_campaign "ext25-lhs36" "$root" "PROM-ANN Case 3" case3 run_prom_ann_case_3.py \
    "$models/case3_ann_ntot151_best.pt" 10 "Case3_Best" 0
  run_podae_family_for_campaign "ext25-lhs36" "$root" "$models/prom_pod_ae_ntot151_best.pt" 0
  summarize_campaign "$root" "ext25-lhs36"
}

cat <<PLAN
[2pct-nonlinear] campaign:          $CAMPAIGN
[2pct-nonlinear] ECSW percent:      $ECSW_PERCENT
[2pct-nonlinear] snapshot mode:     $ECSW_SNAPSHOT_MODE
[2pct-nonlinear] random seed:       $ECSW_RANDOM_SEED
[2pct-nonlinear] Case 3 SVD tol:    $ECSW_SVD_REL_TOL
[2pct-nonlinear] ECSW build threads: $ECSW_BUILD_THREADS
[2pct-nonlinear] online threads:    $ONLINE_THREADS
[2pct-nonlinear] device:            $ONLINE_DEVICE
[2pct-nonlinear] FORCE:             $FORCE
[2pct-nonlinear] runs:              Case 1, Case 2 n=10/n=20 from POD-NN master map, Case 3, PROM-POD-AE
[2pct-nonlinear] skipped:           Linear HPROM, POD-NN-ROM, POD-DL-ROM
PLAN

require_file "$BASIS"
require_file "$UREF"

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[2pct-nonlinear] PLAN_ONLY=1; no ECSW builds or online solves were run."
  exit 0
fi

case "$CAMPAIGN" in
  baseline) run_baseline ;;
  ext25) run_ext25 ;;
  both)
    run_baseline
    run_ext25
    ;;
esac

printf '[done] 2%% learned-intrusive HPROM rerun complete.\n'
