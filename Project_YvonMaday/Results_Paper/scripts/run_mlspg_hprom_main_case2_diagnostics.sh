#!/usr/bin/env bash
# Temporary HPROM diagnostics mirroring the Case-2 PROM n-sweep and tail test.
# All generated outputs are kept outside production Runs/ and ECSW/ directories.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"
case "$FAMILY" in
  all|nsweep|sensitivity|check) ;;
  *)
    echo "Usage: $0 [all|nsweep|sensitivity|check]" >&2
    exit 2
    ;;
esac

HPROM_ROOT="${HPROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_hprom_main}"
DIAG_ROOT="${DIAG_ROOT:-$PROJECT_DIR/Results_Paper/tmp_case2_hprom_diagnostics}"
BASIS="${BASIS:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
UREF="${UREF:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
MODEL="${MODEL:-$HPROM_ROOT/Stage3/models/data_driven_ann_ntot151_best.pt}"

# n=0 is the direct POD-NN-ROM and n=151 is the existing linear HPROM.
CASE2_N_VALUES="${CASE2_N_VALUES:-3 5 10 20 30 50}"
LEVELS="${LEVELS:-0 1 3 5 10 15 20 30 50}"

ECSW_PERCENT="${ECSW_PERCENT:-2.0}"
ECSW_NUM_TRAINING_MU="${ECSW_NUM_TRAINING_MU:-9}"
ECSW_SNAPSHOT_OFFSET="${ECSW_SNAPSHOT_OFFSET:-3}"
ECSW_SNAPSHOT_MODE="${ECSW_SNAPSHOT_MODE:-global_param_time_stratified}"
ECSW_RANDOM_SEED="${ECSW_RANDOM_SEED:-42}"
ECSW_BUILD_THREADS="${ECSW_BUILD_THREADS:-16}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
REBUILD_ECSW="${REBUILD_ECSW:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

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
  if [[ ! -f "$1" ]]; then
    echo "[error] Missing required file: $1" >&2
    exit 1
  fi
}

mu_tags() {
  printf "%.3f %.4f\n" "$1" "$2"
}

check_references() {
  local item mu1 mu2 label mu1_tag mu2_tag linear_dir direct_dir
  for item in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<<"$item"
    read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
    linear_dir="$HPROM_ROOT/Runs/Linear/linear_hprom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
    direct_dir="$HPROM_ROOT/Runs/DataDriven_Best/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
    require_file "$linear_dir/qN.npy"
    require_file "$linear_dir/summary.txt"
    require_file "$direct_dir/qN.npy"
    require_file "$direct_dir/rom_data_driven_summary.txt"
    echo "[ok] HPROM references at ${label}: linear and n=0 direct map"
  done
}

build_rule() {
  local n="$1"
  local run_root="$DIAG_ROOT/n_sweep/Runs/np${n}"
  local weights_dir="$DIAG_ROOT/n_sweep/ECSW/np${n}"
  local log_dir="$DIAG_ROOT/n_sweep/logs/np${n}"
  local weights="$weights_dir/ecsw_weights_ann_case2_data_driven_ann_ntot151_best_n${n}_ntot151.npy"

  mkdir -p "$run_root" "$weights_dir" "$log_dir"
  if [[ "$REBUILD_ECSW" != "1" && -f "$weights" ]]; then
    echo "[skip] Case 2 n=${n} diagnostic ECSW rule exists: $weights"
    return
  fi

  if [[ "$PLAN_ONLY" == "1" ]]; then
    echo "[plan] Build Case 2 n=${n} diagnostic ECSW rule at $weights"
    return
  fi

  echo "[build] Case 2 n=${n} ECSW ${ECSW_PERCENT}% rule."
  set_threads "$ECSW_BUILD_THREADS"
  if [[ "$REBUILD_ECSW" == "1" ]]; then
    python3 -u run_prom_ann_case_2.py \
    --backend hprom \
    --ecsw-only \
    --rebuild-ecsw \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" \
    --target-primary-modes "$n" \
    --run-tag-extra hprom_nsweep \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$run_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset "$ECSW_SNAPSHOT_OFFSET" \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-snapshot-mode "$ECSW_SNAPSHOT_MODE" \
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    2>&1 | tee "$log_dir/ecsw_build.log"
  else
    python3 -u run_prom_ann_case_2.py \
    --backend hprom \
    --ecsw-only \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" \
    --target-primary-modes "$n" \
    --run-tag-extra hprom_nsweep \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$run_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset "$ECSW_SNAPSHOT_OFFSET" \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-snapshot-mode "$ECSW_SNAPSHOT_MODE" \
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    2>&1 | tee "$log_dir/ecsw_build.log"
  fi
  require_file "$weights"
}

run_nsweep_point() {
  local n="$1"
  local mu1="$2"
  local mu2="$3"
  local label="$4"
  local run_root="$DIAG_ROOT/n_sweep/Runs/np${n}"
  local weights_dir="$DIAG_ROOT/n_sweep/ECSW/np${n}"
  local log_dir="$DIAG_ROOT/n_sweep/logs/np${n}"
  local mu1_tag mu2_tag run_tag
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  run_tag="case2_hprom_ann_hprom_nsweep_mu1_${mu1_tag}_mu2_${mu2_tag}_n${n}_ntot151"

  if [[ "$FORCE" != "1" && -f "$run_root/${run_tag}_summary.txt" && -f "$run_root/${run_tag}_qN.npy" ]]; then
    echo "[skip] HPROM Case 2 n=${n} at ${label}: $run_root/${run_tag}_summary.txt"
    return
  fi
  if [[ "$PLAN_ONLY" == "1" ]]; then
    echo "[plan] HPROM Case 2 n=${n} | ${label} | mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] HPROM Case 2 n=${n} | ${label} | mu=(${mu1}, ${mu2})"
  set_threads "$ONLINE_THREADS"
  python3 -u run_prom_ann_case_2.py \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" \
    --target-primary-modes "$n" \
    --run-tag-extra hprom_nsweep \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$run_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu "$ECSW_NUM_TRAINING_MU" \
    --ecsw-snap-time-offset "$ECSW_SNAPSHOT_OFFSET" \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-snapshot-mode "$ECSW_SNAPSHOT_MODE" \
    --ecsw-random-seed "$ECSW_RANDOM_SEED" \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    --no-save-rom-snaps \
    --no-plot \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_nsweep() {
  local n item mu1 mu2 label
  for n in $CASE2_N_VALUES; do
    if (( n <= 0 || n >= 151 )); then
      echo "[error] Every Case-2 n must be in [1, 150], got ${n}." >&2
      exit 2
    fi
    build_rule "$n"
    for item in "${POINTS[@]}"; do
      read -r mu1 mu2 label <<<"$item"
      run_nsweep_point "$n" "$mu1" "$mu2" "$label"
    done
  done
}

run_sensitivity() {
  local weights="$HPROM_ROOT/ECSW/2pct/Case2_Master/np10/ecsw_weights_ann_case2_data_driven_ann_ntot151_best_n10_ntot151.npy"
  local out="$DIAG_ROOT/secondary_sensitivity_n10"
  local log_dir="$DIAG_ROOT/secondary_sensitivity_n10/logs"
  require_file "$weights"
  mkdir -p "$out" "$log_dir"

  if [[ "$PLAN_ONLY" == "1" ]]; then
    echo "[plan] HPROM Case 2 n=10 tail sensitivity at levels: $LEVELS"
    return
  fi

  echo "[run] HPROM Case 2 n=10 fixed-ECSW tail sensitivity."
  set_threads "$ONLINE_THREADS"
  python3 -u run_case2_hprom_secondary_sensitivity_tmp.py \
    --points all \
    --levels $LEVELS \
    --n-primary 10 \
    --n-tot 151 \
    --hprom-root "$HPROM_ROOT" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --ecsw-weights-path "$weights" \
    --output-root "$out" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    --include-ann-level \
    --force \
    2>&1 | tee "$log_dir/case2_hprom_secondary_sensitivity.log"
}

require_file "$BASIS"
require_file "$UREF"
require_file "$MODEL"
mkdir -p "$DIAG_ROOT" "$HPROM_ROOT/.mplcache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HPROM_ROOT/.mplcache}"

echo "[case2-hprom-diagnostics] family:          $FAMILY"
echo "[case2-hprom-diagnostics] HPROM root:      $HPROM_ROOT"
echo "[case2-hprom-diagnostics] diagnostic root: $DIAG_ROOT"
echo "[case2-hprom-diagnostics] model:           $MODEL"
echo "[case2-hprom-diagnostics] n values:        0 $CASE2_N_VALUES 151"
echo "[case2-hprom-diagnostics] tail levels:     $LEVELS"
echo "[case2-hprom-diagnostics] ECSW:            ${ECSW_PERCENT}% / ${ECSW_SNAPSHOT_MODE} / seed ${ECSW_RANDOM_SEED}"
echo "[case2-hprom-diagnostics] threads:         build=${ECSW_BUILD_THREADS}, online=${ONLINE_THREADS}"
echo "[case2-hprom-diagnostics] device:          $ONLINE_DEVICE"
echo "[case2-hprom-diagnostics] force:           $FORCE"
echo "[case2-hprom-diagnostics] rebuild ECSW:    $REBUILD_ECSW"
echo "[case2-hprom-diagnostics] plan only:       $PLAN_ONLY"

if [[ "$FAMILY" == "all" || "$FAMILY" == "check" ]]; then
  check_references
fi
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[case2-hprom-diagnostics] PLAN_ONLY complete; no rules or online solves were run."
  exit 0
fi
if [[ "$FAMILY" == "all" || "$FAMILY" == "nsweep" ]]; then
  run_nsweep
fi
if [[ "$FAMILY" == "all" || "$FAMILY" == "sensitivity" ]]; then
  run_sensitivity
fi

echo "[case2-hprom-diagnostics] done."
