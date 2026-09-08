#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="${PAPER_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_hprom_enrichment}"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MODELS_DIR="$PAPER_ROOT/Stage3/models"
ECSW_PERCENT="1.0"
ECSW_TAG="ECSW1pct"
ONLINE_DEVICE="${ONLINE_DEVICE:-cuda}"
ECSW_BUILD_THREADS="${ECSW_BUILD_THREADS:-24}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

case "$ONLINE_DEVICE" in
  cpu|cuda) ;;
  *)
    echo "[error] ONLINE_DEVICE must be cpu or cuda, got: $ONLINE_DEVICE" >&2
    exit 2
    ;;
esac

CASE1_MODEL="$MODELS_DIR/case1_ann_ntot151_best.pt"
CASE2_NP10_MODEL="$MODELS_DIR/case2_ann_ntot151_np10_best.pt"
CASE2_NP20_MODEL="$MODELS_DIR/case2_ann_ntot151_np20_best.pt"
CASE3_MODEL="$MODELS_DIR/case3_ann_ntot151_best.pt"
PODAE_MODEL="$MODELS_DIR/prom_pod_ae_ntot151_best.pt"
DATA_DRIVEN_MODEL="$MODELS_DIR/data_driven_ann_ntot151_best.pt"
PODDL_MODEL="$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"

for required in \
  "$BASIS" \
  "$UREF" \
  "$CASE1_MODEL" \
  "$CASE2_NP10_MODEL" \
  "$CASE2_NP20_MODEL" \
  "$CASE3_MODEL" \
  "$PODAE_MODEL" \
  "$DATA_DRIVEN_MODEL" \
  "$PODDL_MODEL"
do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$PAPER_ROOT/Runs" "$PAPER_ROOT/ECSW" "$PAPER_ROOT/logs/online" "$MPLCONFIGDIR"

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
}

print_plan() {
  cat <<EOF
[campaign] Enriched MLSPG-sensitive online campaign
[campaign] execution: strictly serial, one family and one parameter at a time
[campaign] evaluation order:
  1. verification mu^(v) = (4.875, 0.0225)
  2. off-grid    mu^(1) = (4.560, 0.0190)
  3. off-grid    mu^(2) = (5.190, 0.0260)
[campaign] intrusive ECSW: case-specific 1%, built once and reused for 3 points
[campaign] ECSW training trajectories: same 9 structured trajectories as baseline
[campaign] ECSW build threads: $ECSW_BUILD_THREADS
[campaign] online threads: $ONLINE_THREADS
[campaign] output root: $PAPER_ROOT
[campaign] models:
  Case 1:       $CASE1_MODEL
  Case 2 n=10: $CASE2_NP10_MODEL
  Case 2 n=20: $CASE2_NP20_MODEL
  Case 3:       $CASE3_MODEL
  PROM-POD-AE:  $PODAE_MODEL
  POD-NN-ROM:   $DATA_DRIVEN_MODEL
  POD-DL-ROM:   $PODDL_MODEL
EOF
}

print_plan
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[campaign] PLAN_ONLY=1; no ECSW construction or online solve was run."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  echo "[clean] FORCE=1: removing previous enriched online outputs, online logs, and case-specific ECSW rules."
  rm -rf \
    "$PAPER_ROOT/Runs/$ECSW_TAG" \
    "$PAPER_ROOT/Runs/DataDriven_Best" \
    "$PAPER_ROOT/Runs/PODDL_Best" \
    "$PAPER_ROOT/ECSW/1pct" \
    "$PAPER_ROOT/logs/online"
  mkdir -p "$PAPER_ROOT/logs/online"
fi

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
  weights="$(ann_weights_path "$case_name" "$model" "$primary" "$weights_dir")"

  mkdir -p "$output_root" "$weights_dir" "$log_dir"
  if [[ -f "$weights" ]]; then
    echo "[build] Reusing enriched ${label} ECSW 1% rule: $weights"
    return
  fi

  echo "[build] Constructing enriched ${label} ECSW 1% rule once."
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
    --ecsw-num-training-mu 9 \
    --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECSW_PERCENT" \
    --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage \
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
  local mu1_tag
  local mu2_tag
  local run_tag

  mu1_tag="$(printf "%.3f" "$mu1")"
  mu2_tag="$(printf "%.4f" "$mu2")"
  run_tag="${case_name}_hprom_ann_mu1_${mu1_tag}_mu2_${mu2_tag}_n${primary}_ntot151"

  if [[ \
    -f "$output_root/${run_tag}_summary.txt" && \
    -f "$output_root/${run_tag}_snaps.npy" && \
    -f "$output_root/${run_tag}_qN.npy" \
  ]]; then
    echo "[skip] Enriched ${label} already complete at mu=(${mu1}, ${mu2})"
    return
  fi

  echo "[run] Enriched ${label} | mu=(${mu1}, ${mu2})"
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
  local weights_dir="$PAPER_ROOT/ECSW/1pct/$family_path"
  local log_dir="$PAPER_ROOT/logs/online/$ECSW_TAG/$family_path"

  build_ann_ecsw_once \
    "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir"

  run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir" 4.875 0.0225
  run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir" 4.560 0.0190
  run_ann_point "$label" "$case_name" "$runner" "$model" "$primary" \
    "$output_root" "$weights_dir" "$log_dir" 5.190 0.0260
}

run_podae_family() {
  local output_root="$PAPER_ROOT/Runs/$ECSW_TAG/PODAE_Best"
  local weights_dir="$PAPER_ROOT/ECSW/1pct/PODAE_Best"
  local log_dir="$PAPER_ROOT/logs/online/$ECSW_TAG/PODAE_Best"
  local weights="$weights_dir/ecsw_weights_pod_ae_ntot151.npy"
  local latent_dim

  mkdir -p "$output_root" "$weights_dir" "$log_dir"
  latent_dim="$(python3 - <<PY
import torch
checkpoint = torch.load("$PODAE_MODEL", map_location="cpu")
print(int(checkpoint["latent_dim"]))
PY
)"

  if [[ ! -f "$weights" ]]; then
    echo "[build] Constructing enriched PROM-POD-AE ECSW 1% rule once."
    set_threads "$ECSW_BUILD_THREADS"
    python3 -u run_prom_pod_ae.py \
      --backend hprom \
      --mu1 4.875 \
      --mu2 0.0225 \
      --device "$ONLINE_DEVICE" \
      --model-path "$PODAE_MODEL" \
      --output-root "$output_root" \
      --ecsw-weights-dir "$weights_dir" \
      --ecsw-num-training-mu 9 \
      --ecsw-snap-time-offset 3 \
      --ecsw-snapshot-percent "$ECSW_PERCENT" \
      --ecsw-random-seed 42 \
      --ecsw-ensure-mu-coverage \
      --rebuild-ecsw \
      --ecsw-only \
      2>&1 | tee "$log_dir/ecsw_build.log"
  else
    echo "[build] Reusing enriched PROM-POD-AE ECSW 1% rule: $weights"
  fi

  if [[ ! -f "$weights" ]]; then
    echo "[error] POD-AE ECSW build completed without producing: $weights" >&2
    exit 1
  fi

  local mu1
  local mu2
  local mu1_tag
  local mu2_tag
  local stem
  for point in "4.875 0.0225" "4.560 0.0190" "5.190 0.0260"; do
    read -r mu1 mu2 <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    stem="podae_hprom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"

    if [[ \
      -f "$output_root/${stem}_summary.txt" && \
      -f "$output_root/${stem}_snaps.npy" && \
      -f "$output_root/${stem}_qN.npy" \
    ]]; then
      echo "[skip] Enriched PROM-POD-AE already complete at mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] Enriched PROM-POD-AE | mu=(${mu1}, ${mu2})"
    set_threads "$ONLINE_THREADS"
    python3 -u run_prom_pod_ae.py \
      --backend hprom \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
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
      2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  done
}

run_data_driven_family() {
  local output_root="$PAPER_ROOT/Runs/DataDriven_Best"
  local log_dir="$PAPER_ROOT/logs/online/DataDriven_Best"
  local mu1
  local mu2
  local mu1_tag
  local mu2_tag
  local out_dir

  mkdir -p "$output_root" "$log_dir"
  for point in "4.875 0.0225" "4.560 0.0190" "5.190 0.0260"; do
    read -r mu1 mu2 <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    out_dir="$output_root/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"

    if [[ \
      -f "$out_dir/rom_data_driven_summary.txt" && \
      -f "$out_dir/rom_snaps.npy" && \
      -f "$out_dir/qN.npy" \
    ]]; then
      echo "[skip] Enriched POD-NN-ROM already complete at mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] Enriched POD-NN-ROM | mu=(${mu1}, ${mu2})"
    set_threads "$ONLINE_THREADS"
    python3 -u run_rom_data_driven.py \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --total-modes 151 \
      --device "$ONLINE_DEVICE" \
      --model-path "$DATA_DRIVEN_MODEL" \
      --basis-path "$BASIS" \
      --u-ref-path "$UREF" \
      --output-root "$output_root" \
      2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  done
}

run_poddl_family() {
  local output_root="$PAPER_ROOT/Runs/PODDL_Best"
  local log_dir="$PAPER_ROOT/logs/online/PODDL_Best"
  local latent_dim
  local mu1
  local mu2
  local mu1_tag
  local mu2_tag
  local out_dir

  mkdir -p "$output_root" "$log_dir"
  latent_dim="$(python3 - <<PY
import torch
checkpoint = torch.load("$PODDL_MODEL", map_location="cpu")
print(int(checkpoint["latent_dim"]))
PY
)"

  for point in "4.875 0.0225" "4.560 0.0190" "5.190 0.0260"; do
    read -r mu1 mu2 <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    out_dir="$output_root/pod_dl_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"

    if [[ \
      -f "$out_dir/pod_dl_data_driven_summary.txt" && \
      -f "$out_dir/rom_snaps.npy" && \
      -f "$out_dir/qN.npy" \
    ]]; then
      echo "[skip] Enriched POD-DL-ROM already complete at mu=(${mu1}, ${mu2})"
      continue
    fi

    echo "[run] Enriched POD-DL-ROM | mu=(${mu1}, ${mu2})"
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

echo "[family 1/7] PROM-ANN Case 1"
run_ann_family \
  "Case 1 Best" case1 run_prom_ann_case_1.py \
  "$CASE1_MODEL" 10 "Case1_Best"

echo "[family 2/7] PROM-ANN Case 2, n=10"
run_ann_family \
  "Case 2 Best (n=10)" case2 run_prom_ann_case_2.py \
  "$CASE2_NP10_MODEL" 10 "Case2_Best/np10"

echo "[family 3/7] PROM-ANN Case 2, n=20"
run_ann_family \
  "Case 2 Best (n=20)" case2 run_prom_ann_case_2.py \
  "$CASE2_NP20_MODEL" 20 "Case2_Best/np20"

echo "[family 4/7] PROM-ANN Case 3"
run_ann_family \
  "Case 3 Best" case3 run_prom_ann_case_3.py \
  "$CASE3_MODEL" 10 "Case3_Best"

echo "[family 5/7] PROM-POD-AE"
run_podae_family

echo "[family 6/7] POD-NN-ROM"
run_data_driven_family

echo "[family 7/7] POD-DL-ROM"
run_poddl_family

SUMMARY="$PAPER_ROOT/logs/online/enrichment_all_best_3pts_summary.txt"
{
  echo "[campaign] Enriched MLSPG-sensitive best models"
  echo "execution: serial"
  echo "evaluation_order: verification, off-grid-1, off-grid-2"
  echo "basis: $BASIS"
  echo "u_ref: $UREF"
  echo "intrusive_ecsw_snapshot_percent: $ECSW_PERCENT"
  echo "intrusive_ecsw_num_training_mu: 9"
  echo
  find "$PAPER_ROOT/Runs" -type f -name "*summary.txt" -print0 \
    | sort -z \
    | while IFS= read -r -d '' summary; do
        echo "==== ${summary#$PAPER_ROOT/}"
        grep -E \
          "mu_test|method|model_path|basis_path|u_ref_path|latent_dim|solve_backend|ecsw_snapshot_percent|ecsw_weights_path|ecsw_residual|n_ecsw_elements|online_solve_elapsed_s|inference_time_s|relative_error_percent|qN_source|qN_output|snaps_output|output_dir" \
          "$summary" || true
      done
} | tee "$SUMMARY"

echo "[done] Enriched online campaign completed serially."
echo "[done] Runs:    $PAPER_ROOT/Runs"
echo "[done] ECSW:    $PAPER_ROOT/ECSW/1pct"
echo "[done] Logs:    $PAPER_ROOT/logs/online"
echo "[done] Summary: $SUMMARY"
