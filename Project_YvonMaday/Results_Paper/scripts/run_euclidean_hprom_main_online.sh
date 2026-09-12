#!/usr/bin/env bash
# Build closure-specific ECM rules and evaluate all Euclidean HPROM families.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

STAGE="${1:-all}"
FAMILY="${2:-all}"
case "$STAGE" in rules|reporting|direct|all) ;; *) echo "Usage: $0 [rules|reporting|direct|all] [all|case1|case2|case3|pod_ae|podnn|poddl]" >&2; exit 2;; esac
case "$FAMILY" in all|case1|case2|case3|pod_ae|podnn|poddl) ;; *) echo "Unknown family: $FAMILY" >&2; exit 2;; esac

PAPER_ROOT="${EUCLIDEAN_HPROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_hprom_main}"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1/u_ref.npy"
MODELS="$PAPER_ROOT/Stage3/models"
ECSW_ROOT="$PAPER_ROOT/Stage4/ecm"
RUNS="$PAPER_ROOT/Runs"
LOGS="$PAPER_ROOT/logs/online"

CASE1_MODEL="$MODELS/case1_ann_ntot151_best.pt"
MASTER_MODEL="$MODELS/master_ann_mu_t_to_qtot_ntot151_best.pt"
CASE3_MODEL="$MODELS/case3_ann_ntot151_best.pt"
PODAE_MODEL="$MODELS/prom_pod_ae_ntot151_best.pt"
PODDL_MODEL="$MODELS/pod_dl_data_driven_ntot151_best.pt"

RULE_THREADS="${RULE_THREADS:-24}"
ONLINE_THREADS="${ONLINE_THREADS:-24}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
ECM_PERCENT="${ECM_PERCENT:-2.0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

POINTS=(
  "4.875 0.0225 verification"
  "4.560 0.0190 offgrid1"
  "5.190 0.0260 offgrid2"
  "4.000 0.0330 extrapolation20pct"
)

set_threads() {
  export BLIS_NUM_THREADS="$1" GOTO_NUM_THREADS="$1" MKL_NUM_THREADS="$1"
  export OMP_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1" NUMEXPR_NUM_THREADS="$1"
}

selected() { [[ "$FAMILY" == all || "$FAMILY" == "$1" ]]; }
require_file() { [[ -f "$1" ]] || { echo "[error] Missing required file: $1" >&2; exit 1; }; }

common_ecm_args() {
  printf '%s\n' \
    --backend hprom --device "$ONLINE_DEVICE" \
    --ecsw-num-training-mu 9 --ecsw-snap-time-offset 3 \
    --ecsw-snapshot-percent "$ECM_PERCENT" --ecsw-random-seed 42 \
    --ecsw-ensure-mu-coverage
}

build_rule() {
  local family="$1" runner="$2" model="$3" output="$4" weights="$5"
  require_file "$model"
  mkdir -p "$output" "$weights" "$LOGS/$family"
  if [[ "$FORCE" != 1 ]] && find "$weights" -maxdepth 1 -type f -name '*.npy' -print -quit | grep -q .; then
    echo "[skip] Closure-specific ECM already exists for $family."
    return
  fi
  if [[ "$PLAN_ONLY" == 1 ]]; then echo "[plan] Build closure-specific ECM: $family"; return; fi
  echo "[rule] Building or checking closure-specific ECM for $family."
  set_threads "$RULE_THREADS"
  local args
  mapfile -t args < <(common_ecm_args)
  local basis_args=()
  local method_args=()
  if [[ "$runner" != run_prom_pod_ae.py ]]; then
    basis_args=(--basis-path "$BASIS" --u-ref-path "$UREF")
  fi
  if [[ "$runner" == run_prom_ann_case_2.py ]]; then
    method_args=(--target-primary-modes 10)
  fi
  "$PYTHON_BIN" -u "$runner" "${args[@]}" "${basis_args[@]}" "${method_args[@]}" \
    --mu1 4.875 --mu2 0.0225 --model-path "$model" \
    --output-root "$output" --ecsw-weights-dir "$weights" \
    --rebuild-ecsw --ecsw-only \
    2>&1 | tee "$LOGS/$family/ecm_build.log"
  find "$weights" -maxdepth 1 -type f -name '*.npy' -print -quit | grep -q . || {
    echo "[error] ECM build did not create a weight file for $family." >&2; exit 1;
  }
}

run_intrusive_point() {
  local family="$1" runner="$2" model="$3" output="$4" weights="$5"
  local mu1="$6" mu2="$7" label="$8" mu1_tag mu2_tag
  mu1_tag="$(printf '%.3f' "$mu1")"; mu2_tag="$(printf '%.4f' "$mu2")"
  if [[ "$FORCE" != 1 ]] && find "$output" -maxdepth 1 -type f \
      -name "*hprom*mu1_${mu1_tag}_mu2_${mu2_tag}*_summary.txt" -print -quit | grep -q .; then
    echo "[skip] $family $label is complete."
    return
  fi
  if [[ "$PLAN_ONLY" == 1 ]]; then echo "[plan] $family $label mu=($mu1,$mu2)"; return; fi
  echo "[run] $family $label mu=($mu1,$mu2)."
  set_threads "$ONLINE_THREADS"
  local args
  mapfile -t args < <(common_ecm_args)
  local basis_args=()
  local method_args=()
  if [[ "$runner" != run_prom_pod_ae.py ]]; then
    basis_args=(--basis-path "$BASIS" --u-ref-path "$UREF")
  fi
  if [[ "$runner" == run_prom_ann_case_2.py ]]; then
    method_args=(--target-primary-modes 10)
  fi
  "$PYTHON_BIN" -u "$runner" "${args[@]}" "${basis_args[@]}" "${method_args[@]}" \
    --mu1 "$mu1" --mu2 "$mu2" --model-path "$model" \
    --output-root "$output" --ecsw-weights-dir "$weights" \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --linear-solver lstsq --normal-eq-reg 1e-12 \
    2>&1 | tee "$LOGS/$family/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_intrusive_family() {
  local family="$1" runner="$2" model="$3"
  local output="$RUNS/$family" weights="$ECSW_ROOT/$family" point mu1 mu2 label
  require_file "$model"; mkdir -p "$output" "$weights" "$LOGS/$family"
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<< "$point"
    run_intrusive_point "$family" "$runner" "$model" "$output" "$weights" "$mu1" "$mu2" "$label"
  done
}

run_direct_family() {
  local family="$1" runner="$2" model="$3" summary_name="$4" prefix="$5"
  local output="$RUNS/$family" point mu1 mu2 label mu1_tag mu2_tag folder
  require_file "$model"; mkdir -p "$output" "$LOGS/$family"
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<< "$point"
    mu1_tag="$(printf '%.3f' "$mu1")"; mu2_tag="$(printf '%.4f' "$mu2")"
    folder="$output/${prefix}_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
    if [[ "$family" == PODDL_ROM ]]; then folder="${folder}_nz10"; fi
    if [[ "$FORCE" != 1 && -f "$folder/$summary_name" && -f "$folder/qN.npy" ]]; then
      echo "[skip] $family $label is complete."
      continue
    fi
    if [[ "$PLAN_ONLY" == 1 ]]; then echo "[plan] $family $label mu=($mu1,$mu2)"; continue; fi
    echo "[run] $family $label mu=($mu1,$mu2)."
    set_threads "$ONLINE_THREADS"
    if [[ "$family" == PODNN_ROM ]]; then
      "$PYTHON_BIN" -u "$runner" --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 \
        --device "$ONLINE_DEVICE" --model-path "$model" --basis-path "$BASIS" \
        --u-ref-path "$UREF" --output-root "$output" \
        2>&1 | tee "$LOGS/$family/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
    else
      "$PYTHON_BIN" -u "$runner" --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 \
        --device "$ONLINE_DEVICE" --model-path "$model" --output-root "$output" \
        2>&1 | tee "$LOGS/$family/mu1_${mu1_tag}_mu2_${mu2_tag}.log"
    fi
  done
}

require_file "$BASIS"; require_file "$UREF"
mkdir -p "$ECSW_ROOT" "$RUNS" "$LOGS" "$MPLCONFIGDIR"
echo "[euclidean-hprom-online] stage=$STAGE family=$FAMILY root=$PAPER_ROOT"
echo "[euclidean-hprom-online] ECM=$ECM_PERCENT% rule_threads=$RULE_THREADS online_threads=$ONLINE_THREADS device=$ONLINE_DEVICE"

if [[ "$PLAN_ONLY" != 1 ]]; then
  for model in "$CASE1_MODEL" "$MASTER_MODEL" "$CASE3_MODEL" "$PODAE_MODEL" "$PODDL_MODEL"; do
    require_file "$model"
  done
  "$PYTHON_BIN" - "$CASE1_MODEL" "$MASTER_MODEL" "$CASE3_MODEL" "$PODAE_MODEL" "$PODDL_MODEL" <<'PY'
import sys
import torch
for path in sys.argv[1:]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backend = str(checkpoint.get("dataset_backend", "")).lower()
    validation = str(checkpoint.get("validation_dataset_backend", "")).lower()
    if backend != "hprom" or validation != "hprom":
        raise SystemExit(
            f"Refusing non-HPROM-consistent checkpoint {path}: "
            f"training={backend!r}, validation={validation!r}"
        )
print("All learned checkpoints declare HPROM training and HPROM external validation.")
PY
fi

if [[ "$STAGE" == rules || "$STAGE" == all ]]; then
  selected case1 && build_rule HPROM_ANN_C1 run_prom_ann_case_1.py "$CASE1_MODEL" "$RUNS/HPROM_ANN_C1" "$ECSW_ROOT/HPROM_ANN_C1"
  selected case2 && build_rule HPROM_ANN_C2 run_prom_ann_case_2.py "$MASTER_MODEL" "$RUNS/HPROM_ANN_C2" "$ECSW_ROOT/HPROM_ANN_C2"
  selected case3 && build_rule HPROM_ANN_C3 run_prom_ann_case_3.py "$CASE3_MODEL" "$RUNS/HPROM_ANN_C3" "$ECSW_ROOT/HPROM_ANN_C3"
  selected pod_ae && build_rule HPROM_POD_AE run_prom_pod_ae.py "$PODAE_MODEL" "$RUNS/HPROM_POD_AE" "$ECSW_ROOT/HPROM_POD_AE"
fi

if [[ "$STAGE" == reporting || "$STAGE" == all ]]; then
  selected case1 && run_intrusive_family HPROM_ANN_C1 run_prom_ann_case_1.py "$CASE1_MODEL"
  selected case2 && run_intrusive_family HPROM_ANN_C2 run_prom_ann_case_2.py "$MASTER_MODEL"
  selected case3 && run_intrusive_family HPROM_ANN_C3 run_prom_ann_case_3.py "$CASE3_MODEL"
  selected pod_ae && run_intrusive_family HPROM_POD_AE run_prom_pod_ae.py "$PODAE_MODEL"
fi

if [[ "$STAGE" == direct || "$STAGE" == all ]]; then
  selected podnn && run_direct_family PODNN_ROM run_rom_data_driven.py "$MASTER_MODEL" rom_data_driven_summary.txt rom_data_driven
  selected poddl && run_direct_family PODDL_ROM run_pod_dl_data_driven.py "$PODDL_MODEL" pod_dl_data_driven_summary.txt pod_dl_data_driven
fi

echo "[euclidean-hprom-online] completed stage=$STAGE family=$FAMILY"
