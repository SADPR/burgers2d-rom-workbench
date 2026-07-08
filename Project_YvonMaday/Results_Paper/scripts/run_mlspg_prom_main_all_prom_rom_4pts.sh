#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"
case "$FAMILY" in
  all|prom|nonlinear_prom|rom|linear|case1|case2|case3|podae|data_driven|poddl) ;;
  *)
    echo "Usage: $0 [all|prom|nonlinear_prom|rom|linear|case1|case2|case3|podae|data_driven|poddl]" >&2
    exit 2
    ;;
esac

ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODELS="$ROOT/Stage3/models"

CASE1_MODEL="$MODELS/case1_ann_ntot151_best.pt"
CASE2_MASTER_MODEL="$MODELS/master_ann_mu_t_to_qtot_ntot151_best.pt"
CASE3_MODEL="$MODELS/case3_ann_ntot151_best.pt"
PODAE_MODEL="$MODELS/prom_pod_ae_ntot151_best.pt"
PODDL_MODEL="$MODELS/pod_dl_data_driven_ntot151_best.pt"

PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

case "$ONLINE_DEVICE" in
  auto|cpu|cuda) ;;
  *) echo "[error] ONLINE_DEVICE must be auto, cpu, or cuda; got: $ONLINE_DEVICE" >&2; exit 2 ;;
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

mu_tags() {
  local mu1="$1"
  local mu2="$2"
  printf "%.3f %.4f\n" "$mu1" "$mu2"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] Missing required file: $path" >&2
    exit 1
  fi
}

should_run() {
  local key="$1"
  case "$FAMILY" in
    all) return 0 ;;
    prom) [[ "$key" == linear || "$key" == case1 || "$key" == case2 || "$key" == case3 || "$key" == podae ]] ;;
    nonlinear_prom) [[ "$key" == case1 || "$key" == case2 || "$key" == case3 || "$key" == podae ]] ;;
    rom) [[ "$key" == data_driven || "$key" == poddl ]] ;;
    *) [[ "$FAMILY" == "$key" ]] ;;
  esac
}

skip_if_done() {
  local label="$1"
  local summary="$2"
  local expected_model="${3:-}"
  if [[ "$FORCE" != "1" && -f "$summary" ]]; then
    if [[ -n "$expected_model" ]]; then
      if grep -Fq "model_path: $expected_model" "$summary"; then
        echo "[skip] $label already complete with expected model: $summary"
        return 0
      fi
      echo "[stale] $label summary exists but model_path is not expected; rerunning:"
      echo "        summary:  $summary"
      echo "        expected: $expected_model"
      grep -E '^model_path:' "$summary" | sed 's/^/        found:    /' || true
      return 1
    fi
    echo "[skip] $label already complete: $summary"
    return 0
  fi
  return 1
}

run_linear() {
  local mu1="$1" mu2="$2" point_label="$3" mu1_tag mu2_tag out_dir log_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out_dir="$ROOT/Runs/Linear/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
  log_dir="$ROOT/logs/online/PROM/Linear"
  mkdir -p "$ROOT/Runs/Linear" "$log_dir"
  skip_if_done "Linear PROM | $point_label" "$out_dir/summary.txt" && return

  echo "[run] Linear PROM | $point_label | mu=($mu1,$mu2)"
  python3 -u run_prom.py \
    --backend prom --no-ecsw \
    --mu1 "$mu1" --mu2 "$mu2" \
    --total-modes 151 \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$ROOT/Runs/Linear" \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_ann_case() {
  local key="$1" label="$2" runner="$3" model="$4" output_root="$5" summary_tag="$6" mu1="$7" mu2="$8" point_label="$9"
  local mu1_tag mu2_tag log_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  log_dir="$ROOT/logs/online/PROM/$label"
  mkdir -p "$output_root" "$log_dir"
  skip_if_done "$label | $point_label" "$output_root/${summary_tag}_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_summary.txt" "$model" && return

  echo "[run] $label | $point_label | mu=($mu1,$mu2)"
  python3 -u "$runner" \
    --backend prom --no-ecsw \
    --mu1 "$mu1" --mu2 "$mu2" \
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

run_case2_master() {
  local mu1="$1" mu2="$2" point_label="$3" mu1_tag mu2_tag output_root log_dir summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  output_root="$ROOT/Runs/PROM/Case2_MasterANN/np10"
  log_dir="$ROOT/logs/online/PROM/Case2_MasterANN_np10"
  summary="$output_root/case2_prom_ann_master_qtot_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_summary.txt"
  mkdir -p "$output_root" "$log_dir"
  skip_if_done "Case2 MasterANN n=10 | $point_label" "$summary" "$CASE2_MASTER_MODEL" && return

  echo "[run] Case2 MasterANN n=10 | $point_label | mu=($mu1,$mu2)"
  python3 -u run_prom_ann_case_2.py \
    --backend prom --no-ecsw \
    --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$CASE2_MASTER_MODEL" \
    --target-primary-modes 10 \
    --run-tag-extra master_qtot \
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

run_podae() {
  local mu1="$1" mu2="$2" point_label="$3" mu1_tag mu2_tag output_root log_dir summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  output_root="$ROOT/Runs/PROM/PODAE_Best"
  log_dir="$ROOT/logs/online/PROM/PODAE_Best"
  summary="$output_root/podae_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz10_summary.txt"
  mkdir -p "$output_root" "$log_dir"
  skip_if_done "PROM-POD-AE | $point_label" "$summary" "$PODAE_MODEL" && return

  echo "[run] PROM-POD-AE | $point_label | mu=($mu1,$mu2)"
  python3 -u run_prom_pod_ae.py \
    --backend prom --no-ecsw \
    --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" \
    --model-path "$PODAE_MODEL" \
    --output-root "$output_root" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_data_driven() {
  local mu1="$1" mu2="$2" point_label="$3" mu1_tag mu2_tag output_root log_dir summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  output_root="$ROOT/Runs/ROM/DataDriven_MasterANN"
  log_dir="$ROOT/logs/online/ROM/DataDriven_MasterANN"
  summary="$output_root/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/rom_data_driven_summary.txt"
  mkdir -p "$output_root" "$log_dir"
  skip_if_done "Data-driven ROM | $point_label" "$summary" "$CASE2_MASTER_MODEL" && return

  echo "[run] Data-driven ROM | $point_label | mu=($mu1,$mu2)"
  python3 -u run_rom_data_driven.py \
    --mu1 "$mu1" --mu2 "$mu2" \
    --total-modes 151 \
    --device "$ONLINE_DEVICE" \
    --model-path "$CASE2_MASTER_MODEL" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_poddl() {
  local mu1="$1" mu2="$2" point_label="$3" mu1_tag mu2_tag output_root log_dir summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  output_root="$ROOT/Runs/ROM/PODDL_Best"
  log_dir="$ROOT/logs/online/ROM/PODDL_Best"
  summary="$output_root/pod_dl_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz10/pod_dl_data_driven_summary.txt"
  mkdir -p "$output_root" "$log_dir"
  skip_if_done "POD-DL-ROM | $point_label" "$summary" "$PODDL_MODEL" && return

  echo "[run] POD-DL-ROM | $point_label | mu=($mu1,$mu2)"
  python3 -u run_pod_dl_data_driven.py \
    --mu1 "$mu1" --mu2 "$mu2" \
    --total-modes 151 \
    --device "$ONLINE_DEVICE" \
    --model-path "$PODDL_MODEL" \
    --output-root "$output_root" \
    2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

require_file "$BASIS"
require_file "$UREF"
require_file "$CASE1_MODEL"
require_file "$CASE2_MASTER_MODEL"
require_file "$CASE3_MODEL"
require_file "$PODAE_MODEL"
require_file "$PODDL_MODEL"

mkdir -p "$ROOT/.mplcache"
export MPLCONFIGDIR="$ROOT/.mplcache"
set_threads "$PROM_NUM_THREADS"

echo "[prom-rom-4pts] family:  $FAMILY"
echo "[prom-rom-4pts] root:    $ROOT"
echo "[prom-rom-4pts] threads: $PROM_NUM_THREADS"
echo "[prom-rom-4pts] device:  $ONLINE_DEVICE"
echo "[prom-rom-4pts] force:   $FORCE"

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[prom-rom-4pts] PLAN_ONLY complete."
  exit 0
fi

for item in "${POINTS[@]}"; do
  read -r mu1 mu2 point_label <<<"$item"
  echo
  echo "================ $point_label: mu=($mu1,$mu2) ================"

  should_run linear && run_linear "$mu1" "$mu2" "$point_label"
  should_run case1 && run_ann_case case1 Case1_Best run_prom_ann_case_1.py "$CASE1_MODEL" "$ROOT/Runs/PROM/Case1_Best" case1_prom_ann "$mu1" "$mu2" "$point_label"
  should_run case2 && run_case2_master "$mu1" "$mu2" "$point_label"
  should_run case3 && run_ann_case case3 Case3_Best run_prom_ann_case_3.py "$CASE3_MODEL" "$ROOT/Runs/PROM/Case3_Best" case3_prom_ann "$mu1" "$mu2" "$point_label"
  should_run podae && run_podae "$mu1" "$mu2" "$point_label"
  should_run data_driven && run_data_driven "$mu1" "$mu2" "$point_label"
  should_run poddl && run_poddl "$mu1" "$mu2" "$point_label"
done

echo
echo "[prom-rom-4pts] done."
echo "[prom-rom-4pts] outputs: $ROOT/Runs"
