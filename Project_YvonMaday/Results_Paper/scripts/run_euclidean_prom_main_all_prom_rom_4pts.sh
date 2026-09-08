#!/usr/bin/env bash
# Run the Euclidean-POD full-residual PROM and ROM models at the four paper points.
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

ROOT="${EUCLIDEAN_PROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_prom_main}"
BASIS="${EUCLIDEAN_BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/basis.npy}"
UREF="${EUCLIDEAN_UREF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/u_ref.npy}"
MODELS="$ROOT/Stage3/models"

CASE1_MODEL="$MODELS/case1_ann_ntot151_best.pt"
MASTER_MODEL="$MODELS/master_ann_mu_t_to_qtot_ntot151_best.pt"
CASE3_MODEL="$MODELS/case3_ann_ntot151_best.pt"
PODAE_MODEL="$MODELS/prom_pod_ae_ntot151_best.pt"
PODDL_MODEL="$MODELS/pod_dl_data_driven_ntot151_best.pt"

PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
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

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[error] Missing required file: $path" >&2; exit 1; }
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

mu_tags() {
  printf "%.3f %.4f\n" "$1" "$2"
}

skip_if_done() {
  local label="$1"
  local summary="$2"
  local model="$3"
  if [[ "$FORCE" != "1" && -f "$summary" ]] && grep -Fq "model_path: $model" "$summary"; then
    echo "[skip] $label: $summary"
    return 0
  fi
  return 1
}

run_linear() {
  local mu1="$1" mu2="$2" label="$3" out log summary
  local mu1_tag mu2_tag
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/Linear"
  log="$ROOT/logs/online/PROM/Linear/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  [[ "$FORCE" == "1" || ! -f "$summary" ]] || { echo "[skip] Linear PROM | $label"; return; }
  echo "[run] Linear PROM | $label | mu=($mu1,$mu2)"
  python3 -u run_prom.py --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 \
    --basis-path "$BASIS" --u-ref-path "$UREF" --output-root "$out" 2>&1 | tee "$log"
}

run_case1() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/PROM/Case1_Best"
  log="$ROOT/logs/online/PROM/Case1_Best/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/case1_prom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  skip_if_done "PROM-ANN Case 1 | $label" "$summary" "$CASE1_MODEL" && return
  echo "[run] PROM-ANN Case 1 | $label | mu=($mu1,$mu2)"
  python3 -u run_prom_ann_case_1.py --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" --model-path "$CASE1_MODEL" --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$out" --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --linear-solver lstsq --normal-eq-reg 1e-12 2>&1 | tee "$log"
}

run_case2() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/PROM/Case2_MasterANN/np10"
  log="$ROOT/logs/online/PROM/Case2_MasterANN_np10/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/case2_prom_ann_master_qtot_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  skip_if_done "PROM-ANN Case 2 n=10 | $label" "$summary" "$MASTER_MODEL" && return
  echo "[run] PROM-ANN Case 2 n=10 | $label | mu=($mu1,$mu2)"
  python3 -u run_prom_ann_case_2.py --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" --model-path "$MASTER_MODEL" --target-primary-modes 10 --run-tag-extra master_qtot \
    --basis-path "$BASIS" --u-ref-path "$UREF" --output-root "$out" \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 --linear-solver lstsq --normal-eq-reg 1e-12 \
    2>&1 | tee "$log"
}

run_case3() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/PROM/Case3_Best"
  log="$ROOT/logs/online/PROM/Case3_Best/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/case3_prom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  skip_if_done "PROM-ANN Case 3 | $label" "$summary" "$CASE3_MODEL" && return
  echo "[run] PROM-ANN Case 3 | $label | mu=($mu1,$mu2)"
  python3 -u run_prom_ann_case_3.py --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" --model-path "$CASE3_MODEL" --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$out" --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --linear-solver lstsq --normal-eq-reg 1e-12 2>&1 | tee "$log"
}

run_podae() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/PROM/PODAE_Best"
  log="$ROOT/logs/online/PROM/PODAE_Best/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/podae_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz10_summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  skip_if_done "PROM-POD-AE | $label" "$summary" "$PODAE_MODEL" && return
  echo "[run] PROM-POD-AE | $label | mu=($mu1,$mu2)"
  python3 -u run_prom_pod_ae.py --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" --model-path "$PODAE_MODEL" --output-root "$out" \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 --linear-solver lstsq --normal-eq-reg 1e-12 \
    2>&1 | tee "$log"
}

run_data_driven() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/ROM/DataDriven_MasterANN"
  log="$ROOT/logs/online/ROM/DataDriven_MasterANN/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/rom_data_driven_summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  skip_if_done "POD-NN-ROM | $label" "$summary" "$MASTER_MODEL" && return
  echo "[run] POD-NN-ROM | $label | mu=($mu1,$mu2)"
  python3 -u run_rom_data_driven.py --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 \
    --device "$ONLINE_DEVICE" --model-path "$MASTER_MODEL" --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$out" 2>&1 | tee "$log"
}

run_poddl() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$ROOT/Runs/ROM/PODDL_Best"
  log="$ROOT/logs/online/ROM/PODDL_Best/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/pod_dl_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz10/pod_dl_data_driven_summary.txt"
  mkdir -p "$out" "$(dirname "$log")"
  skip_if_done "POD-DL-ROM | $label" "$summary" "$PODDL_MODEL" && return
  echo "[run] POD-DL-ROM | $label | mu=($mu1,$mu2)"
  python3 -u run_pod_dl_data_driven.py --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 \
    --device "$ONLINE_DEVICE" --model-path "$PODDL_MODEL" --output-root "$out" 2>&1 | tee "$log"
}

require_file "$BASIS"
require_file "$UREF"
should_run case1 && require_file "$CASE1_MODEL"
should_run case2 && require_file "$MASTER_MODEL"
should_run case3 && require_file "$CASE3_MODEL"
should_run podae && require_file "$PODAE_MODEL"
should_run data_driven && require_file "$MASTER_MODEL"
should_run poddl && require_file "$PODDL_MODEL"

mkdir -p "$ROOT/.mplcache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplcache}"
set_threads "$PROM_NUM_THREADS"

echo "[euclidean-prom-4pts] family:  $FAMILY"
echo "[euclidean-prom-4pts] root:    $ROOT"
echo "[euclidean-prom-4pts] basis:   $BASIS"
echo "[euclidean-prom-4pts] threads: $PROM_NUM_THREADS"
echo "[euclidean-prom-4pts] device:  $ONLINE_DEVICE"
echo "[euclidean-prom-4pts] force:   $FORCE"
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[euclidean-prom-4pts] PLAN_ONLY=1; no runs were executed."
  exit 0
fi

for item in "${POINTS[@]}"; do
  read -r mu1 mu2 label <<<"$item"
  echo "================ $label: mu=($mu1,$mu2) ================"
  should_run linear && run_linear "$mu1" "$mu2" "$label"
  should_run case1 && run_case1 "$mu1" "$mu2" "$label"
  should_run case2 && run_case2 "$mu1" "$mu2" "$label"
  should_run case3 && run_case3 "$mu1" "$mu2" "$label"
  should_run podae && run_podae "$mu1" "$mu2" "$label"
  should_run data_driven && run_data_driven "$mu1" "$mu2" "$label"
  should_run poddl && run_poddl "$mu1" "$mu2" "$label"
done

echo "[euclidean-prom-4pts] done. Outputs: $ROOT/Runs"
