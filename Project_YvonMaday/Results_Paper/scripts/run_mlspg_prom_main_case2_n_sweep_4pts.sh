#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"
case "$FAMILY" in
  all|data_driven|case2|check_linear) ;;
  *)
    echo "Usage: $0 [all|data_driven|case2|check_linear]" >&2
    exit 2
    ;;
esac

PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODEL="${MODEL:-$PROM_ROOT/Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt}"
LOG_ROOT="$PROM_ROOT/logs/online/case2_master_ann_n_sweep"
CASE2_ROOT="$PROM_ROOT/Runs/PROM/Case2_MasterANN_NSweep"
DATA_ROOT="$PROM_ROOT/Runs/ROM/DataDriven_MasterANN"
LINEAR_ROOT="$PROM_ROOT/Runs/Linear"

# n=0 is the data-driven model; n=151 is the linear PROM reference.
CASE2_N_VALUES="${CASE2_N_VALUES:-3 5 10 20 30 50 100}"

FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"

case "$ONLINE_DEVICE" in
  cpu|cuda|auto) ;;
  *)
    echo "[error] ONLINE_DEVICE must be auto, cpu, or cuda; got: $ONLINE_DEVICE" >&2
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

mu_tags() {
  local mu1="$1"
  local mu2="$2"
  printf "%.3f %.4f\n" "$mu1" "$mu2"
}

should_run_family() {
  local key="$1"
  [[ "$FAMILY" == "all" || "$FAMILY" == "$key" ]]
}

check_linear_point() {
  local mu1="$1"
  local mu2="$2"
  local point_label="$3"
  local mu1_tag mu2_tag out_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out_dir="$LINEAR_ROOT/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
  require_file "$out_dir/summary.txt"
  require_file "$out_dir/qN.npy"
  echo "[ok] n=151 linear PROM reference exists at ${point_label}: $out_dir"
}

run_data_driven_point() {
  local mu1="$1"
  local mu2="$2"
  local point_label="$3"
  local mu1_tag mu2_tag out_dir log_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out_dir="$DATA_ROOT/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
  log_dir="$LOG_ROOT/DataDriven_n0"

  mkdir -p "$DATA_ROOT" "$log_dir"
  if [[ "$FORCE" != "1" && -f "$out_dir/rom_data_driven_summary.txt" && -f "$out_dir/qN.npy" ]]; then
    echo "[skip] n=0 data-driven already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] n=0 data-driven master ANN | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_rom_data_driven.py \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$DATA_ROOT" \
    --no-save-rom-snaps \
    --no-plot \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_case2_point() {
  local n="$1"
  local mu1="$2"
  local mu2="$3"
  local point_label="$4"
  local output_root="$CASE2_ROOT/np${n}"
  local log_dir="$LOG_ROOT/Case2_np${n}"
  local mu1_tag mu2_tag run_tag
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  run_tag="case2_prom_ann_master_qtot_mu1_${mu1_tag}_mu2_${mu2_tag}_n${n}_ntot151"

  mkdir -p "$output_root" "$log_dir"
  if [[ "$FORCE" != "1" && -f "$output_root/${run_tag}_summary.txt" && -f "$output_root/${run_tag}_qN.npy" ]]; then
    echo "[skip] Case 2 n=${n} already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] PROM Case 2 n=${n} master ANN | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_prom_ann_case_2.py \
    --backend prom \
    --no-ecsw \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" \
    --target-primary-modes "$n" \
    --run-tag-extra master_qtot \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    --no-save-rom-snaps \
    --no-plot \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

require_file "$BASIS"
require_file "$UREF"
require_file "$MODEL"

mkdir -p "$PROM_ROOT/.mplcache" "$LOG_ROOT"
export MPLCONFIGDIR="$PROM_ROOT/.mplcache"

echo "[case2-n-sweep] family:       $FAMILY"
echo "[case2-n-sweep] output root:   $PROM_ROOT"
echo "[case2-n-sweep] case2 root:    $CASE2_ROOT"
echo "[case2-n-sweep] model:         $MODEL"
echo "[case2-n-sweep] n values:      0 ${CASE2_N_VALUES} 151"
echo "[case2-n-sweep] basis:         $BASIS"
echo "[case2-n-sweep] u_ref:         $UREF"
echo "[case2-n-sweep] threads:       $PROM_NUM_THREADS"
echo "[case2-n-sweep] device:        $ONLINE_DEVICE"
echo "[case2-n-sweep] force:         $FORCE"
echo "[case2-n-sweep] plan only:     $PLAN_ONLY"

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[case2-n-sweep] PLAN_ONLY complete; no solves were run."
  exit 0
fi

for item in "${POINTS[@]}"; do
  read -r mu1 mu2 point_label <<<"$item"
  if should_run_family check_linear || [[ "$FAMILY" == "all" ]]; then
    check_linear_point "$mu1" "$mu2" "$point_label"
  fi
  if should_run_family data_driven; then
    run_data_driven_point "$mu1" "$mu2" "$point_label"
  fi
done

if should_run_family case2; then
  for n in $CASE2_N_VALUES; do
    if (( n <= 0 || n >= 151 )); then
      echo "[error] Case 2 n must satisfy 1 <= n <= 150; got n=${n}." >&2
      exit 2
    fi
    for item in "${POINTS[@]}"; do
      read -r mu1 mu2 point_label <<<"$item"
      run_case2_point "$n" "$mu1" "$mu2" "$point_label"
    done
  done
fi

echo "[case2-n-sweep] done."
