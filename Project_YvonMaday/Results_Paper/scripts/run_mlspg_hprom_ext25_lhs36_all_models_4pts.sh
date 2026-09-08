#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

family="${1:-all}"
case "$family" in
  all|linear|intrusive|rom|case1|case2|case3|pod_ae|data_driven|pod_dl) ;;
  *)
    echo "Usage: $0 [all|linear|intrusive|rom|case1|case2|case3|pod_ae|data_driven|pod_dl]" >&2
    exit 2
    ;;
esac

PAPER_ROOT="${PAPER_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_hprom_enrichment_ext25_lhs36}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_hprom_main}"
BASIS="${BASIS:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
UREF="${UREF:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
MODELS_DIR="$PAPER_ROOT/Stage3/models"

ECSW_PERCENT="${ECSW_PERCENT:-2.0}"
ECSW_TAG="ECSW2pct"
ECSW_DIR_TAG="2pct"
ECSW_NUM_TRAINING_MU="${ECSW_NUM_TRAINING_MU:-9}"
ECSW_SNAPSHOT_MODE="${ECSW_SNAPSHOT_MODE:-global_param_time_stratified}"
ECSW_RANDOM_SEED="${ECSW_RANDOM_SEED:-42}"
ECSW_SVD_REL_TOL="${ECSW_SVD_REL_TOL:-1e-8}"
ECSW_BUILD_THREADS="${ECSW_BUILD_THREADS:-24}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

case "$ECSW_PERCENT" in
  2|2.0) ECSW_PERCENT="2.0" ;;
  *)
    echo "[error] This launcher is fixed to nonlinear ECSW_PERCENT=2.0. Got $ECSW_PERCENT" >&2
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
  cpu|cuda|auto) ;;
  *)
    echo "[error] ONLINE_DEVICE must be cpu, cuda, or auto; got: $ONLINE_DEVICE" >&2
    exit 2
    ;;
esac

LINEAR_ECSW="$BASELINE_ROOT/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy"
CASE1_MODEL="$MODELS_DIR/case1_ann_ntot151_best.pt"
MASTER_MODEL="$MODELS_DIR/data_driven_ann_ntot151_best.pt"
CASE3_MODEL="$MODELS_DIR/case3_ann_ntot151_best.pt"
PODAE_MODEL="$MODELS_DIR/prom_pod_ae_ntot151_best.pt"
PODDL_MODEL="$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"

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

run_ann_family() {
  local label="$1"
  local case_name="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local family_path="$6"
  local output_root="$PAPER_ROOT/Runs/$ECSW_TAG/$family_path"
  local weights_dir="$PAPER_ROOT/ECSW/$ECSW_DIR_TAG/$family_path"
  local log_dir="$PAPER_ROOT/logs/online/$ECSW_TAG/$family_path"

  require_file "$model"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$weights_dir" "$log_dir"
  fi
  build_ann_ecsw_once "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir"

  local point mu1 mu2 point_label
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
      "$output_root" "$weights_dir" "$log_dir" "$mu1" "$mu2" "$point_label"
  done
}

run_linear_family() {
  local output_root="$PAPER_ROOT/Runs/LinearHPROM"
  local log_dir="$PAPER_ROOT/logs/online/LinearHPROM"
  local point mu1 mu2 point_label mu1_tag mu2_tag stem run_dir

  require_file "$LINEAR_ECSW"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$log_dir"
  fi
  mkdir -p "$output_root" "$log_dir"

  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    stem="linear_hprom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
    run_dir="$output_root/$stem"

    if [[ -f "$run_dir/summary.txt" && -f "$run_dir/qN.npy" ]]; then
      echo "[skip] Linear HPROM already complete at ${point_label}: mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] Linear HPROM | ${point_label} | mu=(${mu1}, ${mu2})"
    set_threads "$ONLINE_THREADS"
    python3 -u run_prom.py \
      --backend hprom \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --total-modes 151 \
      --basis-path "$BASIS" \
      --u-ref-path "$UREF" \
      --ecsw-weights-path "$LINEAR_ECSW" \
      --output-root "$output_root" \
      2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  done
}

run_podae_family() {
  local output_root="$PAPER_ROOT/Runs/$ECSW_TAG/PODAE_Best"
  local weights_dir="$PAPER_ROOT/ECSW/$ECSW_DIR_TAG/PODAE_Best"
  local log_dir="$PAPER_ROOT/logs/online/$ECSW_TAG/PODAE_Best"
  local weights="$weights_dir/ecsw_weights_pod_ae_ntot151.npy"
  local latent_dim point mu1 mu2 point_label mu1_tag mu2_tag stem

  require_file "$PODAE_MODEL"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$weights_dir" "$log_dir"
  fi
  mkdir -p "$output_root" "$weights_dir" "$log_dir"

  latent_dim="$(python3 - <<PY
import torch
checkpoint = torch.load("$PODAE_MODEL", map_location="cpu")
print(int(checkpoint["latent_dim"]))
PY
)"

  if [[ ! -f "$weights" ]]; then
    echo "[build] Constructing PROM-POD-AE ECSW ${ECSW_PERCENT}% rule once."
    set_threads "$ECSW_BUILD_THREADS"
    python3 -u run_prom_pod_ae.py \
      --backend hprom \
      --mu1 4.875 \
      --mu2 0.0225 \
      --device "$ONLINE_DEVICE" \
      --model-path "$PODAE_MODEL" \
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
  else
    echo "[build] Reusing PROM-POD-AE ECSW ${ECSW_PERCENT}% rule: $weights"
  fi

  if [[ ! -f "$weights" ]]; then
    echo "[error] PROM-POD-AE ECSW build completed without producing: $weights" >&2
    exit 1
  fi

  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    stem="podae_hprom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"

    if [[ \
      -f "$output_root/${stem}_summary.txt" && \
      -f "$output_root/${stem}_snaps.npy" && \
      -f "$output_root/${stem}_qN.npy" \
    ]]; then
      echo "[skip] PROM-POD-AE already complete at ${point_label}: mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] PROM-POD-AE | ${point_label} | mu=(${mu1}, ${mu2})"
    set_threads "$ONLINE_THREADS"
    python3 -u run_prom_pod_ae.py \
      --backend hprom \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --device "$ONLINE_DEVICE" \
      --model-path "$PODAE_MODEL" \
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
  done
}

run_data_driven_family() {
  local output_root="$PAPER_ROOT/Runs/DataDriven_Best"
  local log_dir="$PAPER_ROOT/logs/online/DataDriven_Best"
  local point mu1 mu2 point_label mu1_tag mu2_tag out_dir

  require_file "$MASTER_MODEL"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$log_dir"
  fi
  mkdir -p "$output_root" "$log_dir"

  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    out_dir="$output_root/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"

    if [[ \
      -f "$out_dir/rom_data_driven_summary.txt" && \
      -f "$out_dir/rom_snaps.npy" && \
      -f "$out_dir/qN.npy" \
    ]]; then
      echo "[skip] POD-NN-ROM already complete at ${point_label}: mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] POD-NN-ROM | ${point_label} | mu=(${mu1}, ${mu2})"
    set_threads "$ONLINE_THREADS"
    python3 -u run_rom_data_driven.py \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --total-modes 151 \
      --device "$ONLINE_DEVICE" \
      --model-path "$MASTER_MODEL" \
      --basis-path "$BASIS" \
      --u-ref-path "$UREF" \
      --output-root "$output_root" \
      2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  done
}

run_poddl_family() {
  local output_root="$PAPER_ROOT/Runs/PODDL_Best"
  local log_dir="$PAPER_ROOT/logs/online/PODDL_Best"
  local latent_dim point mu1 mu2 point_label mu1_tag mu2_tag out_dir

  require_file "$PODDL_MODEL"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$log_dir"
  fi
  mkdir -p "$output_root" "$log_dir"

  latent_dim="$(python3 - <<PY
import torch
checkpoint = torch.load("$PODDL_MODEL", map_location="cpu")
print(int(checkpoint["latent_dim"]))
PY
)"

  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    out_dir="$output_root/pod_dl_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"

    if [[ \
      -f "$out_dir/pod_dl_data_driven_summary.txt" && \
      -f "$out_dir/rom_snaps.npy" && \
      -f "$out_dir/qN.npy" \
    ]]; then
      echo "[skip] POD-DL-ROM already complete at ${point_label}: mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] POD-DL-ROM | ${point_label} | mu=(${mu1}, ${mu2})"
    set_threads "$ONLINE_THREADS"
    python3 -u run_pod_dl_data_driven.py \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --total-modes 151 \
      --device "$ONLINE_DEVICE" \
      --model-path "$PODDL_MODEL" \
      --output-root "$output_root" \
      2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  done
}

summarize() {
  local summary="$PAPER_ROOT/logs/online/ext25_lhs36_all_models_4pts_summary.txt"
  mkdir -p "$(dirname "$summary")"
  {
    echo "[campaign] Extended-domain enriched HPROM/ROM four-point evaluation"
    echo "family: $family"
    echo "basis: $BASIS"
    echo "u_ref: $UREF"
    echo "linear_ecsw: $LINEAR_ECSW"
    echo "nonlinear_ecsw_percent: $ECSW_PERCENT"
    echo "nonlinear_ecsw_snapshot_mode: $ECSW_SNAPSHOT_MODE"
    echo "nonlinear_ecsw_random_seed: $ECSW_RANDOM_SEED"
    echo "case3_ecsw_svd_rel_tol: $ECSW_SVD_REL_TOL"
    echo
    find "$PAPER_ROOT/Runs" -type f \( -name "*summary.txt" -o -name "summary.txt" -o -name "rom_data_driven_summary.txt" -o -name "pod_dl_data_driven_summary.txt" \) -print0 \
      | sort -z \
      | while IFS= read -r -d '' f; do
          echo "==== ${f#$PAPER_ROOT/}"
          grep -E "mu_test|method|model_path|solve_backend|target_primary_modes|checkpoint_primary_modes|basis_path|u_ref_path|latent_dim|total_modes_used|ecsw_snapshot_percent|ecsw_snapshot_mode|ecsw_snapshot_random_seed|ecsw_svd_rel_tol|ecsw_weights_path|ecsw_residual|n_ecsw_elements|online_solve_elapsed_s|inference_time_s|relative_error_percent|qN_output|snaps_output|output_dir" "$f" || true
        done
  } | tee "$summary"
  echo "[summary] $summary"
}

print_plan() {
  cat <<EOF
[ext25-4pts] family:          $family
[ext25-4pts] output root:     $PAPER_ROOT
[ext25-4pts] basis:           $BASIS
[ext25-4pts] u_ref:           $UREF
[ext25-4pts] linear ECSW:     $LINEAR_ECSW
[ext25-4pts] nonlinear ECSW:  ${ECSW_PERCENT}% / $ECSW_SNAPSHOT_MODE / seed=$ECSW_RANDOM_SEED
[ext25-4pts] ECSW build thrs: $ECSW_BUILD_THREADS
[ext25-4pts] online threads:  $ONLINE_THREADS
[ext25-4pts] online device:   $ONLINE_DEVICE
[ext25-4pts] force:           $FORCE
[ext25-4pts] models:
  Case 1:      $CASE1_MODEL
  Case 2:      $MASTER_MODEL  (used for n=10 and n=20)
  Case 3:      $CASE3_MODEL
  PROM-POD-AE: $PODAE_MODEL
  POD-NN-ROM:  $MASTER_MODEL
  POD-DL-ROM:  $PODDL_MODEL
EOF
}

run_selected() {
  case "$1" in
    linear) run_linear_family ;;
    case1) run_ann_family "PROM-ANN Case 1" case1 run_prom_ann_case_1.py "$CASE1_MODEL" 10 "Case1_Best" ;;
    case2)
      run_ann_family "PROM-ANN Case 2 from master map (n=10)" case2 run_prom_ann_case_2.py "$MASTER_MODEL" 10 "Case2_Master/np10"
      run_ann_family "PROM-ANN Case 2 from master map (n=20)" case2 run_prom_ann_case_2.py "$MASTER_MODEL" 20 "Case2_Master/np20"
      ;;
    case3) run_ann_family "PROM-ANN Case 3" case3 run_prom_ann_case_3.py "$CASE3_MODEL" 10 "Case3_Best" ;;
    pod_ae) run_podae_family ;;
    data_driven) run_data_driven_family ;;
    pod_dl) run_poddl_family ;;
    intrusive)
      run_selected case1
      run_selected case2
      run_selected case3
      run_selected pod_ae
      ;;
    rom)
      run_selected data_driven
      run_selected pod_dl
      ;;
    all)
      run_selected linear
      run_selected intrusive
      run_selected rom
      ;;
  esac
}

print_plan
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[ext25-4pts] PLAN_ONLY=1; no file checks, ECSW builds, or online solves were run."
  exit 0
fi

for required in "$BASIS" "$UREF"; do
  require_file "$required"
done
mkdir -p "$PAPER_ROOT/Runs" "$PAPER_ROOT/ECSW/$ECSW_DIR_TAG" "$PAPER_ROOT/logs/online" "$MPLCONFIGDIR"

run_selected "$family"
summarize

echo "[done] Enriched HPROM/ROM four-point evaluation complete."
echo "[done] Runs: $PAPER_ROOT/Runs"
echo "[done] Logs: $PAPER_ROOT/logs/online"
