#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_TAG="mlspg_prom_main"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"

export TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151}"
export VAL_DATASET_DIR="${VAL_DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2}"
export LOG_DIR="$PAPER_ROOT/logs/stage3_master_dense_gpr"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

export TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"

DENSE_GPR_LABEL="${DENSE_GPR_LABEL:-matern15_ard_all_alpha1e10_nowhite}"
DENSE_GPR_KERNEL="${DENSE_GPR_KERNEL:-matern15}"
DENSE_GPR_ALPHA="${DENSE_GPR_ALPHA:-1e-10}"
DENSE_GPR_LENGTH_SCALE="${DENSE_GPR_LENGTH_SCALE:-1.0}"
DENSE_GPR_LENGTH_BOUNDS="${DENSE_GPR_LENGTH_BOUNDS:-1e-3,1e1}"
DENSE_GPR_CONSTANT_VALUE="${DENSE_GPR_CONSTANT_VALUE:-1.0}"
DENSE_GPR_CONSTANT_BOUNDS="${DENSE_GPR_CONSTANT_BOUNDS:-1e-3,1e3}"
DENSE_GPR_USE_WHITE="${DENSE_GPR_USE_WHITE:-0}"
DENSE_GPR_WHITE_NOISE="${DENSE_GPR_WHITE_NOISE:-1e-8}"
DENSE_GPR_WHITE_BOUNDS="${DENSE_GPR_WHITE_BOUNDS:-1e-12,1e-3}"
DENSE_GPR_OPTIMIZE="${DENSE_GPR_OPTIMIZE:-1}"
DENSE_GPR_N_RESTARTS="${DENSE_GPR_N_RESTARTS:-0}"
DENSE_GPR_MAX_TRAIN_SAMPLES="${DENSE_GPR_MAX_TRAIN_SAMPLES:-1000000}"
DENSE_GPR_MAX_VAL_SAMPLES="${DENSE_GPR_MAX_VAL_SAMPLES:-1000000}"
DENSE_GPR_SEED="${DENSE_GPR_SEED:-42}"

MODEL_NAME="master_dense_gpr_mu_t_to_qtot_ntot151_${DENSE_GPR_LABEL}.pt"
SUMMARY_NAME="master_dense_gpr_mu_t_to_qtot_ntot151_${DENSE_GPR_LABEL}_summary.txt"
LOG_FILE="$LOG_DIR/train_master_dense_gpr_${DENSE_GPR_LABEL}.log"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR" "$PAPER_ROOT/Stage3/models"

if [[ ! -d "$TRAIN_DATASET_DIR/per_mu" ]]; then
  echo "Missing training per_mu dataset: $TRAIN_DATASET_DIR" >&2
  exit 1
fi

if [[ ! -d "$VAL_DATASET_DIR/per_mu" ]]; then
  echo "Missing validation per_mu dataset: $VAL_DATASET_DIR" >&2
  echo "Run Results_Paper/scripts/run_mlspg_prom_main_validation_2pts.sh first." >&2
  exit 1
fi

white_args=(--no-white-kernel)
if [[ "$DENSE_GPR_USE_WHITE" == "1" ]]; then
  white_args=(--use-white-kernel)
fi

optimizer_args=(--optimize-hyperparameters)
if [[ "$DENSE_GPR_OPTIMIZE" == "0" ]]; then
  optimizer_args=(--no-optimize-hyperparameters)
fi

echo "[master-dense-gpr] training dataset:   $TRAIN_DATASET_DIR"
echo "[master-dense-gpr] validation dataset: $VAL_DATASET_DIR"
echo "[master-dense-gpr] output root:        $PAPER_ROOT/Stage3"
echo "[master-dense-gpr] model:              $MODEL_NAME"
echo "[master-dense-gpr] kernel:             $DENSE_GPR_KERNEL"
echo "[master-dense-gpr] alpha:              $DENSE_GPR_ALPHA"
echo "[master-dense-gpr] white kernel:       $DENSE_GPR_USE_WHITE"
echo "[master-dense-gpr] optimize:           $DENSE_GPR_OPTIMIZE"
echo "[master-dense-gpr] restarts:           $DENSE_GPR_N_RESTARTS"
echo "[master-dense-gpr] threads:            $TRAIN_NUM_THREADS"

python3 -u stage3_perform_training_rom_data_driven_gpr.py \
  --stage3-dir "$PAPER_ROOT/Stage3" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$TRAIN_DATASET_DIR" \
  --validation-dataset-dir "$VAL_DATASET_DIR" \
  --model-name "$MODEL_NAME" \
  --summary-name "$SUMMARY_NAME" \
  --x-scaling zscore \
  --y-scaling zscore \
  --kernel-name "$DENSE_GPR_KERNEL" \
  --ard \
  --alpha "$DENSE_GPR_ALPHA" \
  --length-scale "$DENSE_GPR_LENGTH_SCALE" \
  --length-scale-bounds "$DENSE_GPR_LENGTH_BOUNDS" \
  --constant-value "$DENSE_GPR_CONSTANT_VALUE" \
  --constant-value-bounds "$DENSE_GPR_CONSTANT_BOUNDS" \
  --white-noise-level "$DENSE_GPR_WHITE_NOISE" \
  --white-noise-bounds "$DENSE_GPR_WHITE_BOUNDS" \
  "${white_args[@]}" \
  "${optimizer_args[@]}" \
  --n-restarts-optimizer "$DENSE_GPR_N_RESTARTS" \
  --max-train-samples "$DENSE_GPR_MAX_TRAIN_SAMPLES" \
  --max-val-samples "$DENSE_GPR_MAX_VAL_SAMPLES" \
  --seed "$DENSE_GPR_SEED" \
  2>&1 | tee "$LOG_FILE"

grep -E "model_path|dataset_dir|validation_dataset_dir|n_tot|kernel_init|kernel_learned|alpha|use_white_kernel|optimize_hyperparameters|n_restarts_optimizer|max_train_samples|val_split|train_rel_frob_percent|val_rel_frob_percent|elapsed_s" \
  "$PAPER_ROOT/Stage3/$SUMMARY_NAME" \
  | tee "$LOG_DIR/train_master_dense_gpr_${DENSE_GPR_LABEL}_check.txt"
