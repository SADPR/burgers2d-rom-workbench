#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

THREADS="${ROM_NUM_THREADS:-24}"
export BLIS_NUM_THREADS="$THREADS"
export GOTO_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

PLAN_ONLY="${PLAN_ONLY:-0}"
FORCE="${FORCE:-0}"

BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/u_ref.npy"
RUN_ROOT="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Runs"
LINEAR_ROOT="$RUN_ROOT/Linear"
PLAIN_ROOT="$RUN_ROOT/Case2_Plain_Oracle_PROMOnly_Legacy/np10"
PG_ROOT="$RUN_ROOT/Case2_PG_Oracle_PROMOnly_Legacy/np10"
LOG_ROOT="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/logs/case2_oracle_promonly_legacy"
export MPLCONFIGDIR="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/.mplcache"

POINTS=(
  "4.875 0.0225 verification"
  "4.560 0.0190 offgrid1"
  "5.190 0.0260 offgrid2"
  "4.000 0.0330 extrapolation20pct"
)

for required in "$BASIS" "$UREF"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$LINEAR_ROOT" "$PLAIN_ROOT" "$PG_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

echo "[euclidean-case2-oracle] basis:       $BASIS"
echo "[euclidean-case2-oracle] u_ref:       $UREF"
echo "[euclidean-case2-oracle] linear root: $LINEAR_ROOT"
echo "[euclidean-case2-oracle] plain root:  $PLAIN_ROOT"
echo "[euclidean-case2-oracle] PG root:     $PG_ROOT"
echo "[euclidean-case2-oracle] logs:        $LOG_ROOT"
echo "[euclidean-case2-oracle] threads:     $THREADS"
echo "[euclidean-case2-oracle] force:       $FORCE"
echo "[euclidean-case2-oracle] plan only:   $PLAN_ONLY"

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  if [[ "$PLAN_ONLY" == "1" ]]; then
    printf '[plan] '
    printf '%q ' "$@"
    printf '2>&1 | tee %q\n' "$log_file"
  else
    "$@" 2>&1 | tee "$log_file"
  fi
}

have_linear_outputs() {
  local run_dir="$1"
  [[ -f "$run_dir/summary.txt" && -f "$run_dir/qN.npy" && -f "$run_dir/rom_snaps.npy" ]]
}

have_oracle_outputs() {
  local root="$1"
  local stem="$2"
  [[ -f "$root/${stem}_summary.txt" && -f "$root/${stem}_qN.npy" && -f "$root/${stem}_snaps.npy" ]]
}

run_linear_if_needed() {
  local mu1="$1"
  local mu2="$2"
  local linear_dir="$LINEAR_ROOT/linear_prom_mu1_${mu1}_mu2_${mu2}_ntot151"
  local log_file="$LOG_ROOT/linear_prom_mu1_${mu1}_mu2_${mu2}.log"

  if [[ "$FORCE" != "1" ]] && have_linear_outputs "$linear_dir"; then
    echo "[skip] Linear PROM exists: $linear_dir"
    return
  fi

  echo "[run] Linear PROM: mu=($mu1,$mu2)"
  run_with_log "$log_file" \
    python3 -u run_prom.py \
      --backend prom \
      --no-ecsw \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --total-modes 151 \
      --basis-path "$BASIS" \
      --u-ref-path "$UREF" \
      --output-root "$LINEAR_ROOT"
}

run_oracle_if_needed() {
  local variant="$1"
  local out_root="$2"
  local tag="$3"
  local mu1="$4"
  local mu2="$5"

  local linear_dir="$LINEAR_ROOT/linear_prom_mu1_${mu1}_mu2_${mu2}_ntot151"
  local stem="${tag}_mu1_${mu1}_mu2_${mu2}_n10_ntot151_basis_pert0.00pct"
  local log_file="$LOG_ROOT/${tag}_mu1_${mu1}_mu2_${mu2}.log"

  if [[ "$PLAN_ONLY" != "1" ]] && ! have_linear_outputs "$linear_dir"; then
    echo "[error] Missing linear PROM outputs before oracle run: $linear_dir" >&2
    exit 1
  fi

  if [[ "$FORCE" != "1" ]] && have_oracle_outputs "$out_root" "$stem"; then
    echo "[skip] ${variant} oracle exists: $out_root/${stem}_summary.txt"
    return
  fi

  echo "[run] Case 2 ${variant} oracle PROM: mu=($mu1,$mu2)"
  run_with_log "$log_file" \
    python3 -u run_case2_pg_oracle_tmp.py \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --n-primary 10 \
      --n-tot 151 \
      --work-dir "$PROJECT_DIR" \
      --linear-run-dir "$linear_dir" \
      --basis-path "$BASIS" \
      --u-ref-path "$UREF" \
      --qbar-perturb-percent 0.0 \
      --qbar-perturb-seed 0 \
      --output-dir "$out_root" \
      --run-tag-prefix "$tag" \
      --solver-variant "$variant" \
      --oracle-mode legacy \
      --max-its 20 \
      --relnorm-cutoff 1e-5 \
      --min-delta 1e-2 \
      --linear-solver lstsq \
      --normal-eq-reg 1e-12
}

for point in "${POINTS[@]}"; do
  read -r mu1 mu2 label <<< "$point"
  echo
  echo "================ $label: mu=($mu1,$mu2) ================"
  run_linear_if_needed "$mu1" "$mu2"
  run_oracle_if_needed "plain" "$PLAIN_ROOT" "euclidean_plain_oracle_promonly_legacy" "$mu1" "$mu2"
  run_oracle_if_needed "pg" "$PG_ROOT" "euclidean_pg_oracle_promonly_legacy" "$mu1" "$mu2"
done

echo
echo "[done] Euclidean Case 2 oracle-only diagnostic complete."
echo "[done] Linear: $LINEAR_ROOT"
echo "[done] Plain oracle: $PLAIN_ROOT"
echo "[done] PG oracle: $PG_ROOT"
