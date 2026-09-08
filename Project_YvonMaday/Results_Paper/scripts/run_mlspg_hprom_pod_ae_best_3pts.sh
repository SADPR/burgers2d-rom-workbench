#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
MODEL="$PAPER_ROOT/Stage3/models/prom_pod_ae_ntot151_best.pt"
OUT_ROOT="$PAPER_ROOT/Runs/ECSW1pct/PODAE_Best"
WEIGHTS_DIR="$PAPER_ROOT/ECSW/1pct/PODAE_Best"
LOG_DIR="$PAPER_ROOT/logs/online/ECSW1pct/PODAE_Best"
FORCE="${FORCE:-0}"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

python3 "$SCRIPT_DIR/normalize_mlspg_hprom_main_layout.py"

for required in "$MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT" "$WEIGHTS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

if [[ "$FORCE" == "1" ]]; then
  echo "[clean] FORCE=1: removing previous POD-AE ECSW1pct outputs and weights."
  rm -rf "$OUT_ROOT" "$WEIGHTS_DIR"
  mkdir -p "$OUT_ROOT" "$WEIGHTS_DIR"
fi

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
}

WEIGHTS="$WEIGHTS_DIR/ecsw_weights_pod_ae_ntot151.npy"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "[build] Constructing POD-AE ECSW 1% rule once."
  set_threads 24
  python3 -u run_prom_pod_ae.py \
    --backend hprom \
    --mu1 4.875 \
    --mu2 0.0225 \
    --device cuda \
    --model-path "$MODEL" \
    --output-root "$OUT_ROOT" \
    --ecsw-weights-dir "$WEIGHTS_DIR" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent 1.0 \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --rebuild-ecsw \
    --ecsw-only \
    2>&1 | tee "$LOG_DIR/ecsw_build.log"
else
  echo "[build] Reusing POD-AE ECSW 1% rule: $WEIGHTS"
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[error] ECSW build completed without producing: $WEIGHTS" >&2
  exit 1
fi

run_point() {
  local mu1="$1"
  local mu2="$2"
  local mu1_tag
  local mu2_tag
  local latent_dim
  local stem
  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  latent_dim="$(python3 - <<PY
import torch
ck = torch.load("$MODEL", map_location="cpu")
print(int(ck["latent_dim"]))
PY
)"
  stem="podae_hprom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"

  if [[ -f "$OUT_ROOT/${stem}_summary.txt" && -f "$OUT_ROOT/${stem}_snaps.npy" && -f "$OUT_ROOT/${stem}_qN.npy" ]]; then
    echo "[skip] POD-AE already complete at mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] POD-AE Best online with ECSW 1% | mu=(${mu1}, ${mu2})"
  set_threads 1
  python3 -u run_prom_pod_ae.py \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device cuda \
    --model-path "$MODEL" \
    --output-root "$OUT_ROOT" \
    --ecsw-weights-dir "$WEIGHTS_DIR" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent 1.0 \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$LOG_DIR/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_point 4.560 0.0190
run_point 4.875 0.0225
run_point 5.190 0.0260

SUMMARY="$LOG_DIR/pod_ae_best_3pts_summary.txt"
{
  echo "[quick-summary] $(date)"
  echo "model: $MODEL"
  echo "weights: $WEIGHTS"
  for f in "$OUT_ROOT"/*_summary.txt; do
    [[ -f "$f" ]] || continue
    echo "==== $(basename "$f")"
    grep -E "mu_test|model_path|basis_path|u_ref_path|q_dim|latent_dim|solve_backend|n_ecsw_elements|ecsw_setup_elapsed_s|online_solve_elapsed_s|relative_error_percent|snaps_output|qN_output" "$f"
  done
} | tee "$SUMMARY"

echo "[done] Outputs: $OUT_ROOT"
echo "[done] Logs:    $LOG_DIR"
echo "[done] Summary: $SUMMARY"
