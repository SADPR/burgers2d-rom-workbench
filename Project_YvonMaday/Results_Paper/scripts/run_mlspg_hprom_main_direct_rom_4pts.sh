#!/usr/bin/env bash
# Evaluate the two baseline HPROM-trained, non-intrusive maps at all four paper points.
set -euo pipefail

family="${1:-all}"
case "$family" in
  all|data_driven|pod_dl) ;;
  *)
    echo "Usage: $0 [all|data_driven|pod_dl]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
MODELS_DIR="$PAPER_ROOT/Stage3/models"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
MASTER_MODEL="$MODELS_DIR/data_driven_ann_ntot151_best.pt"
PODDL_MODEL="$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"

ONLINE_THREADS="${ONLINE_THREADS:-1}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

POINTS=(
  "4.875 0.0225 verification"
  "4.560 0.0190 offgrid1"
  "5.190 0.0260 offgrid2"
  "4.000 0.0330 extrapolation20pct"
)

set_threads() {
  export BLIS_NUM_THREADS="$ONLINE_THREADS"
  export GOTO_NUM_THREADS="$ONLINE_THREADS"
  export MKL_NUM_THREADS="$ONLINE_THREADS"
  export OMP_NUM_THREADS="$ONLINE_THREADS"
  export OPENBLAS_NUM_THREADS="$ONLINE_THREADS"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] Missing required file: $1" >&2
    exit 1
  fi
}

should_run() {
  local key="$1"
  [[ "$family" == "all" || "$family" == "$key" ]]
}

run_data_driven() {
  local output_root="$PAPER_ROOT/Runs/DataDriven_Best"
  local log_dir="$PAPER_ROOT/logs/online/DataDriven_Best"
  local point mu1 mu2 label mu1_tag mu2_tag out_dir

  require_file "$MASTER_MODEL"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$log_dir"
  fi
  mkdir -p "$output_root" "$log_dir" "$MPLCONFIGDIR"

  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    out_dir="$output_root/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
    if [[ "$FORCE" != "1" && -f "$out_dir/rom_data_driven_summary.txt" && -f "$out_dir/qN.npy" ]]; then
      echo "[skip] POD-NN-ROM exists: $out_dir"
      continue
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
      echo "[plan] POD-NN-ROM | $label | mu=($mu1, $mu2)"
      continue
    fi
    echo "[run] POD-NN-ROM | $label | mu=($mu1, $mu2)"
    set_threads
    python3 -u run_rom_data_driven.py \
      --mu1 "$mu1" \
      --mu2 "$mu2" \
      --total-modes 151 \
      --device "$ONLINE_DEVICE" \
      --model-path "$MASTER_MODEL" \
      --basis-path "$BASIS" \
      --u-ref-path "$UREF" \
      --output-root "$output_root" \
      2>&1 | tee "$log_dir/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
  done
}

run_poddl() {
  local output_root="$PAPER_ROOT/Runs/PODDL_Best"
  local log_dir="$PAPER_ROOT/logs/online/PODDL_Best"
  local latent_dim point mu1 mu2 label mu1_tag mu2_tag out_dir

  require_file "$PODDL_MODEL"
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$output_root" "$log_dir"
  fi
  mkdir -p "$output_root" "$log_dir" "$MPLCONFIGDIR"
  latent_dim="$(python3 - <<PY
import torch
checkpoint = torch.load("$PODDL_MODEL", map_location="cpu", weights_only=False)
print(int(checkpoint["latent_dim"]))
PY
)"

  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<< "$point"
    mu1_tag="$(printf "%.3f" "$mu1")"
    mu2_tag="$(printf "%.4f" "$mu2")"
    out_dir="$output_root/pod_dl_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151_nz${latent_dim}"
    if [[ "$FORCE" != "1" && -f "$out_dir/pod_dl_data_driven_summary.txt" && -f "$out_dir/qN.npy" ]]; then
      echo "[skip] POD-DL-ROM exists: $out_dir"
      continue
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
      echo "[plan] POD-DL-ROM | $label | mu=($mu1, $mu2)"
      continue
    fi
    echo "[run] POD-DL-ROM | $label | mu=($mu1, $mu2)"
    set_threads
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

require_file "$BASIS"
require_file "$UREF"

echo "[hprom-main-direct-rom] family:  $family"
echo "[hprom-main-direct-rom] root:    $PAPER_ROOT"
echo "[hprom-main-direct-rom] device:  $ONLINE_DEVICE"
echo "[hprom-main-direct-rom] threads: $ONLINE_THREADS"
echo "[hprom-main-direct-rom] force:   $FORCE"
echo "[hprom-main-direct-rom] plan:    $PLAN_ONLY"

should_run data_driven && run_data_driven
should_run pod_dl && run_poddl
