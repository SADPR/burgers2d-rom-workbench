#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"
case "$FAMILY" in
  all|linear|ann|intrusive|case1|case2_np10|case2_np20|case3|podae) ;;
  *)
    echo "Usage: $0 [all|linear|ann|intrusive|case1|case2_np10|case2_np20|case3|podae]" >&2
    exit 2
    ;;
esac

SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_hprom_main}"
PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODELS="$SOURCE_ROOT/Stage3/models"
LOG_ROOT="$PROM_ROOT/logs/online/PROM"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"

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

mu_tags() {
  local mu1="$1"
  local mu2="$2"
  printf "%.3f %.4f\n" "$mu1" "$mu2"
}

should_run_family() {
  local key="$1"
  case "$FAMILY" in
    all) return 0 ;;
    intrusive) [[ "$key" != "linear" ]] ;;
    ann) [[ "$key" == case1 || "$key" == case2_np10 || "$key" == case2_np20 || "$key" == case3 ]] ;;
    *) [[ "$FAMILY" == "$key" ]] ;;
  esac
}

run_linear_point() {
  local mu1="$1"
  local mu2="$2"
  local point_label="$3"
  local output_root="$PROM_ROOT/Runs/Linear"
  local log_dir="$LOG_ROOT/Linear"
  local mu1_tag mu2_tag out_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out_dir="$output_root/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"

  mkdir -p "$output_root" "$log_dir"
  if [[ "$FORCE" != "1" && -f "$out_dir/summary.txt" ]]; then
    echo "[skip] Linear PROM already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] Linear PROM | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_prom.py \
    --backend prom \
    --no-ecsw \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_ann_point() {
  local label="$1"
  local key="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local family_path="$6"
  local mu1="$7"
  local mu2="$8"
  local point_label="$9"
  local output_root="$PROM_ROOT/Runs/PROM/$family_path"
  local log_dir="$LOG_ROOT/$family_path"
  local case_name run_tag mu1_tag mu2_tag

  case "$key" in
    case1) case_name="case1" ;;
    case2_np10|case2_np20) case_name="case2" ;;
    case3) case_name="case3" ;;
    *) echo "[error] Unsupported ANN key: $key" >&2; exit 2 ;;
  esac

  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  run_tag="${case_name}_prom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n${primary}_ntot151"

  mkdir -p "$output_root" "$log_dir"
  if [[ "$FORCE" != "1" && -f "$output_root/${run_tag}_summary.txt" && -f "$output_root/${run_tag}_snaps.npy" ]]; then
    echo "[skip] ${label} PROM already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] ${label} PROM | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u "$runner" \
    --backend prom \
    --no-ecsw \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_podae_point() {
  local model="$1"
  local mu1="$2"
  local mu2="$3"
  local point_label="$4"
  local output_root="$PROM_ROOT/Runs/PROM/PODAE_Best"
  local log_dir="$LOG_ROOT/PODAE_Best"
  local mu1_tag mu2_tag run_tag
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  run_tag="podae_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz10"

  mkdir -p "$output_root" "$log_dir"
  if [[ "$FORCE" != "1" && -f "$output_root/${run_tag}_summary.txt" && -f "$output_root/${run_tag}_snaps.npy" ]]; then
    echo "[skip] PROM-POD-AE already complete at ${point_label}: mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] PROM-POD-AE | ${point_label} | mu=(${mu1}, ${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_prom_pod_ae.py \
    --backend prom \
    --no-ecsw \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --output-root "$output_root" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_family_points() {
  local key="$1"
  local label="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local family_path="$6"

  if ! should_run_family "$key"; then
    return
  fi
  require_file "$model"
  for item in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<<"$item"
    run_ann_point "$label" "$key" "$runner" "$model" "$primary" "$family_path" "$mu1" "$mu2" "$point_label"
  done
}

require_file "$BASIS"
require_file "$UREF"
require_file "$MODELS/case1_ann_ntot151_best.pt"
require_file "$MODELS/case2_ann_ntot151_np10_best.pt"
require_file "$MODELS/case2_ann_ntot151_np20_best.pt"
require_file "$MODELS/case3_ann_ntot151_best.pt"
require_file "$MODELS/prom_pod_ae_ntot151_best.pt"

mkdir -p "$PROM_ROOT/.mplcache" "$LOG_ROOT"
export MPLCONFIGDIR="$PROM_ROOT/.mplcache"

echo "[mlspg-prom-main] family:       $FAMILY"
echo "[mlspg-prom-main] source root:   $SOURCE_ROOT"
echo "[mlspg-prom-main] output root:   $PROM_ROOT"
echo "[mlspg-prom-main] basis:         $BASIS"
echo "[mlspg-prom-main] u_ref:         $UREF"
echo "[mlspg-prom-main] threads:       $PROM_NUM_THREADS"
echo "[mlspg-prom-main] device:        $ONLINE_DEVICE"
echo "[mlspg-prom-main] force:         $FORCE"
echo "[mlspg-prom-main] plan only:     $PLAN_ONLY"

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[mlspg-prom-main] PLAN_ONLY complete; no PROM solves were run."
  exit 0
fi

if should_run_family linear; then
  for item in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<<"$item"
    run_linear_point "$mu1" "$mu2" "$point_label"
  done
fi

run_family_points \
  case1 "PROM-ANN Case 1" run_prom_ann_case_1.py \
  "$MODELS/case1_ann_ntot151_best.pt" 10 "Case1_Best"

run_family_points \
  case2_np10 "PROM-ANN Case 2 (n=10)" run_prom_ann_case_2.py \
  "$MODELS/case2_ann_ntot151_np10_best.pt" 10 "Case2_Best/np10"

run_family_points \
  case2_np20 "PROM-ANN Case 2 (n=20)" run_prom_ann_case_2.py \
  "$MODELS/case2_ann_ntot151_np20_best.pt" 20 "Case2_Best/np20"

run_family_points \
  case3 "PROM-ANN Case 3" run_prom_ann_case_3.py \
  "$MODELS/case3_ann_ntot151_best.pt" 10 "Case3_Best"

if should_run_family podae; then
  PODAE_MODEL="$MODELS/prom_pod_ae_ntot151_best.pt"
  require_file "$PODAE_MODEL"
  for item in "${POINTS[@]}"; do
    read -r mu1 mu2 point_label <<<"$item"
    run_podae_point "$PODAE_MODEL" "$mu1" "$mu2" "$point_label"
  done
fi

SUMMARY="$LOG_ROOT/mlspg_prom_main_baseline_4pts_summary.txt"
{
  echo "[campaign] MLSPG-sensitive baseline PROM full-residual diagnostic"
  echo "source_hprom_root: $SOURCE_ROOT"
  echo "prom_root: $PROM_ROOT"
  echo "basis: $BASIS"
  echo "u_ref: $UREF"
  echo
  find "$PROM_ROOT/Runs" -type f -name "*summary.txt" -print0 2>/dev/null \
    | sort -z \
    | while IFS= read -r -d '' summary; do
        echo "==== ${summary#$PROM_ROOT/}"
        grep -E \
          "mu_test|solve_backend_effective|use_ecsw|model_path|n_ecsw_elements|online_solve_elapsed_s|elapsed_s|relative_error_percent|qN_output|snaps_output|output_dir|output_root" \
          "$summary" || true
      done
} | tee "$SUMMARY"

echo "[done] PROM outputs: $PROM_ROOT/Runs"
echo "[done] Summary:      $SUMMARY"
