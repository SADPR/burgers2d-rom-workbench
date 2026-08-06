#!/usr/bin/env bash
# Build a clean Euclidean-POD PROM campaign for a Case-2 tail-error diagnostic.
# The campaign intentionally contains no HPROM/ECM step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

STAGE="${1:-all}"
case "$STAGE" in
  all|train9|val2|train|online|diagnostic) ;;
  *)
    echo "Usage: $0 [all|train9|val2|train|online|diagnostic]" >&2
    exit 2
    ;;
esac

# Keep this study separate from the legacy MetricStudy/euclidean oracle checks
# and from the LSPG-sensitive PROM campaign.
PAPER_ROOT="$PROJECT_DIR/Results_Paper"
PROM_ROOT="${EUCLIDEAN_PROM_ROOT:-$PAPER_ROOT/euclidean_prom_main}"
BASIS="$PAPER_ROOT/MetricStudy/euclidean/Stage1/basis.npy"
UREF="$PAPER_ROOT/MetricStudy/euclidean/Stage1/u_ref.npy"
TRAIN_DATASET="$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151"
VAL_DATASET="$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2"
STAGE3_DIR="$PROM_ROOT/Stage3"
MODEL="$STAGE3_DIR/models/master_ann_mu_t_to_qtot_ntot151_best.pt"
LINEAR_ROOT="$PROM_ROOT/Runs/Linear"
DATA_ROOT="$PROM_ROOT/Runs/ROM/DataDriven_MasterANN"
DIAGNOSTIC_ROOT="${EUCLIDEAN_DIAGNOSTIC_ROOT:-$PAPER_ROOT/tmp_euclidean_case2_secondary_sensitivity}"
LOG_ROOT="$PROM_ROOT/logs/euclidean_case2_tail_sensitivity"

FORCE="${FORCE:-0}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-$PROM_NUM_THREADS}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
LEVELS="${LEVELS:-0 1 3 5 10 15 20 30 50 75 100}"

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
  printf "%.3f %.4f\\n" "$1" "$2"
}

should_run() {
  [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

run_train9() {
  if [[ "$FORCE" != "1" && -f "$TRAIN_DATASET/stage2_summary.txt" ]]; then
    echo "[skip] nine-parameter Euclidean training dataset exists: $TRAIN_DATASET"
    return
  fi
  echo "[run] Euclidean linear-PROM trajectories for the baseline 3x3 training grid."
  set_threads "$PROM_NUM_THREADS"
  python3 -u stage2_build_prom_qn_dataset.py \
    --backend prom --total-modes 151 \
    --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-dir "$TRAIN_DATASET" \
    --linear-solver lstsq --normal-eq-reg 1e-12 \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --no-save-rom-snaps --no-plots \
    2>&1 | tee "$LOG_ROOT/stage2_train9.log"
}

run_val2() {
  if [[ "$FORCE" != "1" && -f "$VAL_DATASET/stage2_summary.txt" ]]; then
    echo "[skip] two-point Euclidean external-validation dataset exists: $VAL_DATASET"
    return
  fi
  echo "[run] Euclidean linear-PROM trajectories at the two held-out parameter-validation points."
  set_threads "$PROM_NUM_THREADS"
  python3 -u stage2_build_prom_qn_dataset.py \
    --backend prom --total-modes 151 \
    --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-dir "$VAL_DATASET" \
    --mu-pair 4.5625 0.02625 \
    --mu-pair 5.1875 0.01875 \
    --linear-solver lstsq --normal-eq-reg 1e-12 \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --no-save-rom-snaps --no-plots \
    2>&1 | tee "$LOG_ROOT/stage2_val2.log"
}

run_train() {
  require_file "$TRAIN_DATASET/stage2_summary.txt"
  require_file "$VAL_DATASET/stage2_summary.txt"
  if [[ "$FORCE" != "1" && -f "$MODEL" && -f "$STAGE3_DIR/master_ann_mu_t_to_qtot_ntot151_best_summary.txt" ]]; then
    echo "[skip] Euclidean master POD-NN map exists: $MODEL"
    return
  fi
  echo "[run] Master POD-NN map (mu,t)->q_tot with external two-parameter validation."
  set_threads "$TRAIN_NUM_THREADS"
  python3 -u stage3_perform_training_rom_data_driven_maday.py \
    --maday-results-root "$PAPER_ROOT" --maday-tag euclidean_prom_main \
    --dataset-backend prom --dataset-ntot 151 \
    --dataset-dir "$TRAIN_DATASET" --validation-dataset-dir "$VAL_DATASET" \
    --model-name master_ann_mu_t_to_qtot_ntot151_best.pt \
    --summary-name master_ann_mu_t_to_qtot_ntot151_best_summary.txt \
    --hidden-dims 256,512,512,256 --activation elu \
    --loss-function mse --loss-space raw \
    --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
    --epochs 7000 --patience 300 \
    --lr-scheduler-factor 0.5 --lr-scheduler-patience 60 --lr-scheduler-min-lr 1e-6 \
    --clip-grad 1.0 --seed 42 \
    2>&1 | tee "$LOG_ROOT/stage3_master_ann.log"
}

run_linear_point() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out_dir="$LINEAR_ROOT/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
  if [[ "$FORCE" != "1" && -f "$out_dir/qN.npy" && -f "$out_dir/rom_snaps.npy" ]]; then
    echo "[skip] Euclidean linear PROM at ${label}: $out_dir"
    return
  fi
  echo "[run] Euclidean linear PROM | ${label} | mu=(${mu1},${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_prom.py \
    --backend prom --no-ecsw --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 \
    --basis-path "$BASIS" --u-ref-path "$UREF" --output-root "$LINEAR_ROOT" \
    --no-plot \
    2>&1 | tee "$LOG_ROOT/linear_mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_data_driven_point() {
  local mu1="$1" mu2="$2" label="$3" mu1_tag mu2_tag out_dir
  read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
  out_dir="$DATA_ROOT/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151"
  if [[ "$FORCE" != "1" && -f "$out_dir/qN.npy" && -f "$out_dir/rom_data_driven_summary.txt" ]]; then
    echo "[skip] Euclidean direct POD-NN map at ${label}: $out_dir"
    return
  fi
  echo "[run] Euclidean direct POD-NN map | ${label} | mu=(${mu1},${mu2})"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_rom_data_driven.py \
    --mu1 "$mu1" --mu2 "$mu2" --total-modes 151 --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$DATA_ROOT" --no-save-rom-snaps --no-plot \
    2>&1 | tee "$LOG_ROOT/data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}.log"
}

run_online() {
  require_file "$MODEL"
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<<"$point"
    run_linear_point "$mu1" "$mu2" "$label"
    run_data_driven_point "$mu1" "$mu2" "$label"
  done
}

run_diagnostic() {
  for point in "${POINTS[@]}"; do
    read -r mu1 mu2 label <<<"$point"
    read -r mu1_tag mu2_tag < <(mu_tags "$mu1" "$mu2")
    require_file "$LINEAR_ROOT/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/qN.npy"
    require_file "$LINEAR_ROOT/linear_prom_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/rom_snaps.npy"
    require_file "$DATA_ROOT/rom_data_driven_mu1_${mu1_tag}_mu2_${mu2_tag}_ntot151/qN.npy"
  done
  echo "[run] Case-2 Euclidean tail perturbation diagnostic | n=10 | levels: $LEVELS"
  set_threads "$PROM_NUM_THREADS"
  python3 -u run_case2_secondary_sensitivity_tmp.py \
    --points all --levels "$LEVELS" --n-primary 10 --n-tot 151 \
    --prom-root "$PROM_ROOT" --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$DIAGNOSTIC_ROOT" \
    --solver-variant plain --linear-solver lstsq --normal-eq-reg 1e-12 \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --include-ann-level --force \
    2>&1 | tee "$LOG_ROOT/case2_tail_sensitivity.log"
}

require_file "$BASIS"
require_file "$UREF"
mkdir -p "$PROM_ROOT/.mplcache" "$LOG_ROOT" "$LINEAR_ROOT" "$DATA_ROOT"
export MPLCONFIGDIR="$PROM_ROOT/.mplcache"

echo "[euclidean-case2-tail] stage:          $STAGE"
echo "[euclidean-case2-tail] campaign root:  $PROM_ROOT"
echo "[euclidean-case2-tail] basis:          $BASIS"
echo "[euclidean-case2-tail] validation:     (4.5625,0.02625), (5.1875,0.01875)"
echo "[euclidean-case2-tail] model:          (256,512,512,256), ELU, raw MSE"
echo "[euclidean-case2-tail] diagnostic:     $DIAGNOSTIC_ROOT"
echo "[euclidean-case2-tail] levels (%):     $LEVELS"
echo "[euclidean-case2-tail] force:          $FORCE"

should_run train9 && run_train9
should_run val2 && run_val2
should_run train && run_train
should_run online && run_online
should_run diagnostic && run_diagnostic

echo "[euclidean-case2-tail] done."
