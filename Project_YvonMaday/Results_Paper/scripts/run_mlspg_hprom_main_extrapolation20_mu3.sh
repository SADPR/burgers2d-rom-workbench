#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="${PAPER_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_hprom_main}"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODELS_DIR="$PAPER_ROOT/Stage3/models"
ECSW_TAG="ECSW1pct"
ECSW_PERCENT="1.0"
ONLINE_DEVICE="${ONLINE_DEVICE:-cuda}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

# 20% beyond the upper-left corner of the training box:
# mu1 = 4.25 - 0.20*(5.50 - 4.25) = 4.000
# mu2 = 0.030 + 0.20*(0.030 - 0.015) = 0.0330
MU1_OUT="${MU1_OUT:-4.000}"
MU2_OUT="${MU2_OUT:-0.0330}"
MU_TAG="mu1_$(printf "%.3f" "$MU1_OUT")_mu2_$(printf "%.4f" "$MU2_OUT")"

OUT_ROOT="$PAPER_ROOT/Runs/Extrapolation20pct"
LOG_ROOT="$PAPER_ROOT/logs/online/Extrapolation20pct"
LINEAR_ECSW="$PAPER_ROOT/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy"

CASE1_MODEL="$MODELS_DIR/case1_ann_ntot151_best.pt"
CASE2_NP10_MODEL="$MODELS_DIR/case2_ann_ntot151_np10_best.pt"
CASE2_NP20_MODEL="$MODELS_DIR/case2_ann_ntot151_np20_best.pt"
CASE3_MODEL="$MODELS_DIR/case3_ann_ntot151_best.pt"
PODAE_MODEL="$MODELS_DIR/prom_pod_ae_ntot151_best.pt"
DATA_DRIVEN_MODEL="$MODELS_DIR/data_driven_ann_ntot151_best.pt"
PODDL_MODEL="$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"

CASE1_WEIGHTS="$PAPER_ROOT/ECSW/1pct/Case1_Best/ecsw_weights_ann_case1_case1_ann_ntot151_best_n10_ntot151.npy"
CASE2_NP10_WEIGHTS="$PAPER_ROOT/ECSW/1pct/Case2_Best/np10/ecsw_weights_ann_case2_case2_ann_ntot151_np10_best_n10_ntot151.npy"
CASE2_NP20_WEIGHTS="$PAPER_ROOT/ECSW/1pct/Case2_Best/np20/ecsw_weights_ann_case2_case2_ann_ntot151_np20_best_n20_ntot151.npy"
CASE3_WEIGHTS="$PAPER_ROOT/ECSW/1pct/Case3_Best/ecsw_weights_ann_case3_case3_ann_ntot151_best_n10_ntot151.npy"
PODAE_WEIGHTS="$PAPER_ROOT/ECSW/1pct/PODAE_Best/ecsw_weights_pod_ae_ntot151.npy"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

case "$ONLINE_DEVICE" in
  cpu|cuda|auto) ;;
  *)
    echo "[error] ONLINE_DEVICE must be cpu, cuda, or auto; got: $ONLINE_DEVICE" >&2
    exit 2
    ;;
esac

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

for required in \
  "$BASIS" \
  "$UREF" \
  "$LINEAR_ECSW" \
  "$CASE1_MODEL" \
  "$CASE2_NP10_MODEL" \
  "$CASE2_NP20_MODEL" \
  "$CASE3_MODEL" \
  "$PODAE_MODEL" \
  "$DATA_DRIVEN_MODEL" \
  "$PODDL_MODEL" \
  "$CASE1_WEIGHTS" \
  "$CASE2_NP10_WEIGHTS" \
  "$CASE2_NP20_WEIGHTS" \
  "$CASE3_WEIGHTS" \
  "$PODAE_WEIGHTS"
do
  require_file "$required"
done

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

if [[ "$FORCE" == "1" ]]; then
  echo "[clean] FORCE=1: removing previous extrapolation-20% outputs and logs only."
  rm -rf "$OUT_ROOT" "$LOG_ROOT"
  mkdir -p "$OUT_ROOT" "$LOG_ROOT"
fi

cat <<EOF
[campaign] Baseline MLSPG-sensitive extrapolation test
[campaign] point: mu^(3) = (${MU1_OUT}, ${MU2_OUT})
[campaign] definition: 20% beyond upper-left training corner
[campaign] output root: $OUT_ROOT
[campaign] logs: $LOG_ROOT
[campaign] ECSW policy: reuse existing baseline ECSW rules only; no rebuild
[campaign] online threads: $ONLINE_THREADS
[campaign] device: $ONLINE_DEVICE
EOF

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[campaign] PLAN_ONLY=1; no solves were run."
  exit 0
fi

run_linear() {
  local output_root="$OUT_ROOT/Linear"
  local log_dir="$LOG_ROOT/Linear"
  local stem="linear_hprom_${MU_TAG}_ntot151"
  mkdir -p "$output_root" "$log_dir"
  if [[ -f "$output_root/$stem/summary.txt" && -f "$output_root/$stem/qN.npy" ]]; then
    echo "[skip] Linear HPROM already complete for mu^(3)."
    return
  fi
  echo "[run] Linear HPROM | mu=(${MU1_OUT}, ${MU2_OUT})"
  set_threads "$ONLINE_THREADS"
  python3 -u run_prom.py \
    --backend hprom \
    --mu1 "$MU1_OUT" \
    --mu2 "$MU2_OUT" \
    --total-modes 151 \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --ecsw-weights-path "$LINEAR_ECSW" \
    --output-root "$output_root" \
    2>&1 | tee "$log_dir/${MU_TAG}.log"
}

run_ann() {
  local label="$1"
  local case_name="$2"
  local runner="$3"
  local model="$4"
  local primary="$5"
  local family_path="$6"
  local weights_dir="$PAPER_ROOT/ECSW/1pct/$family_path"
  local output_root="$OUT_ROOT/$ECSW_TAG/$family_path"
  local log_dir="$LOG_ROOT/$ECSW_TAG/$family_path"
  local stem="${case_name}_hprom_ann_${MU_TAG}_n${primary}_ntot151"
  mkdir -p "$output_root" "$log_dir"
  if [[ -f "$output_root/${stem}_summary.txt" && -f "$output_root/${stem}_qN.npy" ]]; then
    echo "[skip] $label already complete for mu^(3)."
    return
  fi
  echo "[run] $label | mu=(${MU1_OUT}, ${MU2_OUT})"
  set_threads "$ONLINE_THREADS"
  python3 -u "$runner" \
    --backend hprom \
    --mu1 "$MU1_OUT" \
    --mu2 "$MU2_OUT" \
    --device "$ONLINE_DEVICE" \
    --model-path "$model" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/${MU_TAG}.log"
}

run_podae() {
  local output_root="$OUT_ROOT/$ECSW_TAG/PODAE_Best"
  local weights_dir="$PAPER_ROOT/ECSW/1pct/PODAE_Best"
  local log_dir="$LOG_ROOT/$ECSW_TAG/PODAE_Best"
  local latent_dim
  latent_dim="$(python3 - <<PY
import torch
ck = torch.load("$PODAE_MODEL", map_location="cpu")
print(int(ck["latent_dim"]))
PY
)"
  local stem="podae_hprom_${MU_TAG}_ntot151_nz${latent_dim}"
  mkdir -p "$output_root" "$log_dir"
  if [[ -f "$output_root/${stem}_summary.txt" && -f "$output_root/${stem}_qN.npy" ]]; then
    echo "[skip] PROM-POD-AE already complete for mu^(3)."
    return
  fi
  echo "[run] PROM-POD-AE | mu=(${MU1_OUT}, ${MU2_OUT})"
  set_threads "$ONLINE_THREADS"
  python3 -u run_prom_pod_ae.py \
    --backend hprom \
    --mu1 "$MU1_OUT" \
    --mu2 "$MU2_OUT" \
    --device "$ONLINE_DEVICE" \
    --model-path "$PODAE_MODEL" \
    --output-root "$output_root" \
    --ecsw-weights-dir "$weights_dir" \
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$log_dir/${MU_TAG}.log"
}

run_data_driven() {
  local output_root="$OUT_ROOT/DataDriven_Best"
  local log_dir="$LOG_ROOT/DataDriven_Best"
  local run_dir="$output_root/rom_data_driven_${MU_TAG}_ntot151"
  mkdir -p "$output_root" "$log_dir"
  if [[ -f "$run_dir/rom_data_driven_summary.txt" && -f "$run_dir/qN.npy" ]]; then
    echo "[skip] POD-NN-ROM already complete for mu^(3)."
    return
  fi
  echo "[run] POD-NN-ROM | mu=(${MU1_OUT}, ${MU2_OUT})"
  set_threads "$ONLINE_THREADS"
  python3 -u run_rom_data_driven.py \
    --mu1 "$MU1_OUT" \
    --mu2 "$MU2_OUT" \
    --total-modes 151 \
    --device "$ONLINE_DEVICE" \
    --model-path "$DATA_DRIVEN_MODEL" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$output_root" \
    2>&1 | tee "$log_dir/${MU_TAG}.log"
}

run_poddl() {
  local output_root="$OUT_ROOT/PODDL_Best"
  local log_dir="$LOG_ROOT/PODDL_Best"
  local run_dir
  mkdir -p "$output_root" "$log_dir"
  run_dir="$(find "$output_root" -maxdepth 1 -type d -name "pod_dl_data_driven_${MU_TAG}_ntot151_nz*" | head -n 1 || true)"
  if [[ -n "$run_dir" && -f "$run_dir/pod_dl_data_driven_summary.txt" && -f "$run_dir/qN.npy" ]]; then
    echo "[skip] POD-DL-ROM already complete for mu^(3)."
    return
  fi
  echo "[run] POD-DL-ROM | mu=(${MU1_OUT}, ${MU2_OUT})"
  set_threads "$ONLINE_THREADS"
  python3 -u run_pod_dl_data_driven.py \
    --mu1 "$MU1_OUT" \
    --mu2 "$MU2_OUT" \
    --total-modes 151 \
    --device "$ONLINE_DEVICE" \
    --model-path "$PODDL_MODEL" \
    --output-root "$output_root" \
    2>&1 | tee "$log_dir/${MU_TAG}.log"
}

run_linear
run_ann "PROM-ANN Case 1" case1 run_prom_ann_case_1.py "$CASE1_MODEL" 10 "Case1_Best"
run_ann "PROM-ANN Case 2 (n=10)" case2 run_prom_ann_case_2.py "$CASE2_NP10_MODEL" 10 "Case2_Best/np10"
run_ann "PROM-ANN Case 2 (n=20)" case2 run_prom_ann_case_2.py "$CASE2_NP20_MODEL" 20 "Case2_Best/np20"
run_ann "PROM-ANN Case 3" case3 run_prom_ann_case_3.py "$CASE3_MODEL" 10 "Case3_Best"
run_podae
run_data_driven
run_poddl

SUMMARY="$LOG_ROOT/${MU_TAG}_summary.txt"
{
  echo "[summary] Baseline extrapolation-20% test"
  echo "mu3: [${MU1_OUT}, ${MU2_OUT}]"
  echo "basis: $BASIS"
  echo "u_ref: $UREF"
  echo "linear_ecsw: $LINEAR_ECSW"
  echo
  find "$OUT_ROOT" -type f \( -name "*_summary.txt" -o -name "summary.txt" -o -name "rom_data_driven_summary.txt" -o -name "pod_dl_data_driven_summary.txt" \) -print0 \
    | sort -z \
    | while IFS= read -r -d '' summary; do
        echo "==== ${summary#$OUT_ROOT/}"
        grep -E \
          "mu_test|model_path|basis_path|u_ref_path|solve_backend|total_modes_used|q_dim|latent_dim|n_ecsw_elements|ecsw_weights_path|ecsw_weights_source|online_solve_elapsed_s|inference_time_s|relative_error_percent|qN_output|snaps_output|output_dir" \
          "$summary" || true
      done
} | tee "$SUMMARY"

echo "[done] Outputs: $OUT_ROOT"
echo "[done] Logs:    $LOG_ROOT"
echo "[done] Summary: $SUMMARY"
