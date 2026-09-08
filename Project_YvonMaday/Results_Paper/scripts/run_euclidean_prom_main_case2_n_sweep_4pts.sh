#!/usr/bin/env bash
set -euo pipefail

# The endpoints n=0 (POD-NN-ROM), n=10, and n=151 (linear PROM) already
# exist in the Euclidean four-point campaign. This launcher adds only the
# missing intrusive Case-2 dimensions without duplicating those trajectories.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"
case "$FAMILY" in
  all|case2|check) ;;
  *)
    echo "Usage: $0 [all|case2|check]" >&2
    exit 2
    ;;
esac

ROOT="${EUCLIDEAN_PROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_prom_main}"
BASIS="${EUCLIDEAN_BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/basis.npy}"
UREF="${EUCLIDEAN_UREF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/u_ref.npy}"
MODEL="${MASTER_ANN_MODEL:-$ROOT/Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt}"
RUN_ROOT="$ROOT/Runs/PROM/Case2_MasterANN_NSweep"
LOG_ROOT="$ROOT/logs/online/PROM/Case2_MasterANN_NSweep"

# n=0, 10, and 151 are reused from the completed four-point campaign.
CASE2_N_VALUES="${CASE2_N_VALUES:-20 30 50 100}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

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
  local mu1="$1" mu2="$2"
  printf "%.3f %.4f\n" "$mu1" "$mu2"
}

check_existing_endpoints() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")

  require_file "$ROOT/Runs/Linear/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/summary.txt"
  require_file "$ROOT/Runs/Linear/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/qN.npy"
  require_file "$ROOT/Runs/PROM/Case2_MasterANN/np10/case2_prom_ann_master_qtot_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_summary.txt"
  require_file "$ROOT/Runs/PROM/Case2_MasterANN/np10/case2_prom_ann_master_qtot_mu1_${mu1_tag}_mu2_${mu2_tag}_n10_ntot151_qN.npy"
  require_file "$ROOT/Runs/ROM/DataDriven_MasterANN/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/rom_data_driven_summary.txt"
  require_file "$ROOT/Runs/ROM/DataDriven_MasterANN/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/qN.npy"
  echo "[ok] Existing n=0,10,151 endpoints | ${label}"
}

run_case2_point() {
  local n="$1" mu1="$2" mu2="$3" label="$4" mu1_tag mu2_tag out log summary
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out="$RUN_ROOT/np${n}"
  log="$LOG_ROOT/np${n}/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  summary="$out/case2_prom_ann_master_qtot_mu1_${mu1_tag}_mu2_${mu2_tag}_n${n}_ntot151_summary.txt"

  mkdir -p "$out" "$(dirname "$log")"
  if [[ "$FORCE" != "1" && -f "$summary" && -f "${summary%_summary.txt}_qN.npy" ]]; then
    echo "[skip] Case 2 n=${n} | ${label}"
    return
  fi

  echo "[run] Euclidean PROM-ANN Case 2 n=${n} | ${label} | mu=(${mu1},${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_prom_ann_case_2.py \
    --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" \
    --device "$ONLINE_DEVICE" --model-path "$MODEL" --target-primary-modes "$n" \
    --run-tag-extra master_qtot --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$out" --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --linear-solver lstsq --normal-eq-reg 1e-12 --no-save-rom-snaps --no-plot \
    2>&1 | tee "$log"
}

require_file "$BASIS"
require_file "$UREF"
require_file "$MODEL"

echo "[euclidean-case2-n-sweep] root:       $ROOT"
echo "[euclidean-case2-n-sweep] model:      $MODEL"
echo "[euclidean-case2-n-sweep] new n:      $CASE2_N_VALUES"
echo "[euclidean-case2-n-sweep] reused n:   0, 10, 151"
echo "[euclidean-case2-n-sweep] threads:    $PROM_NUM_THREADS"
echo "[euclidean-case2-n-sweep] device:     $ONLINE_DEVICE"
echo "[euclidean-case2-n-sweep] force:      $FORCE"
echo "[euclidean-case2-n-sweep] plan only:  $PLAN_ONLY"

for item in "${POINTS[@]}"; do
  read -r mu1 mu2 label <<<"$item"
  check_existing_endpoints "$mu1" "$mu2" "$label"
done

if [[ "$PLAN_ONLY" == "1" || "$FAMILY" == "check" ]]; then
  echo "[euclidean-case2-n-sweep] endpoint check complete; no solves were run."
  exit 0
fi

for n in $CASE2_N_VALUES; do
  if (( n <= 10 || n >= 151 )); then
    echo "[error] New Case-2 dimensions must satisfy 11 <= n <= 150; got n=${n}." >&2
    exit 2
  fi
  for item in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<<"$item"
    run_case2_point "$n" "$mu1" "$mu2" "$label"
  done
done

echo "[euclidean-case2-n-sweep] done."
