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
export LOG_DIR="$PAPER_ROOT/logs/stage3_master_sparse_gpr"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

export TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"

SPARSE_GPR_LABEL="${SPARSE_GPR_LABEL:-matern15_ard_m451_fixed}"
SPARSE_GPR_KERNEL="${SPARSE_GPR_KERNEL:-matern15}"
SPARSE_GPR_NUM_INDUCING="${SPARSE_GPR_NUM_INDUCING:-451}"
SPARSE_GPR_EPOCHS="${SPARSE_GPR_EPOCHS:-160}"
SPARSE_GPR_BATCH_SIZE="${SPARSE_GPR_BATCH_SIZE:-2048}"
SPARSE_GPR_LR="${SPARSE_GPR_LR:-3e-2}"
SPARSE_GPR_WEIGHT_DECAY="${SPARSE_GPR_WEIGHT_DECAY:-0.0}"
SPARSE_GPR_MIN_NOISE="${SPARSE_GPR_MIN_NOISE:-1e-6}"
SPARSE_GPR_MAX_NOISE="${SPARSE_GPR_MAX_NOISE:-1.0}"
SPARSE_GPR_ELBO_BETA="${SPARSE_GPR_ELBO_BETA:-1.0}"
SPARSE_GPR_DEVICE="${SPARSE_GPR_DEVICE:-auto}"
SPARSE_GPR_SEED="${SPARSE_GPR_SEED:-42}"
SPARSE_GPR_FIXED_INDUCING="${SPARSE_GPR_FIXED_INDUCING:-1}"
SPARSE_GPR_MAX_TRAIN_SAMPLES="${SPARSE_GPR_MAX_TRAIN_SAMPLES:-1000000}"
SPARSE_GPR_MAX_VAL_SAMPLES="${SPARSE_GPR_MAX_VAL_SAMPLES:-1000000}"
SPARSE_GPR_LOG_EVERY="${SPARSE_GPR_LOG_EVERY:-20}"

MODEL_NAME="master_sparse_gpr_mu_t_to_qtot_ntot151_${SPARSE_GPR_LABEL}.pt"
SUMMARY_NAME="master_sparse_gpr_mu_t_to_qtot_ntot151_${SPARSE_GPR_LABEL}_summary.txt"
LOG_FILE="$LOG_DIR/train_master_sparse_gpr_${SPARSE_GPR_LABEL}.log"

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

fixed_inducing_args=()
if [[ "$SPARSE_GPR_FIXED_INDUCING" == "1" ]]; then
  fixed_inducing_args+=(--fixed-inducing)
fi

echo "[master-sparse-gpr] training dataset:   $TRAIN_DATASET_DIR"
echo "[master-sparse-gpr] validation dataset: $VAL_DATASET_DIR"
echo "[master-sparse-gpr] output root:        $PAPER_ROOT/Stage3"
echo "[master-sparse-gpr] model:              $MODEL_NAME"
echo "[master-sparse-gpr] kernel:             $SPARSE_GPR_KERNEL"
echo "[master-sparse-gpr] inducing:           $SPARSE_GPR_NUM_INDUCING"
echo "[master-sparse-gpr] epochs:             $SPARSE_GPR_EPOCHS"
echo "[master-sparse-gpr] fixed inducing:     $SPARSE_GPR_FIXED_INDUCING"
echo "[master-sparse-gpr] device:             $SPARSE_GPR_DEVICE"
echo "[master-sparse-gpr] threads:            $TRAIN_NUM_THREADS"

python3 -u stage3_perform_training_rom_data_driven_sparse_gpr.py \
  --stage3-dir "$PAPER_ROOT/Stage3" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$TRAIN_DATASET_DIR" \
  --validation-dataset-dir "$VAL_DATASET_DIR" \
  --model-name "$MODEL_NAME" \
  --summary-name "$SUMMARY_NAME" \
  --x-scaling zscore \
  --y-scaling zscore \
  --num-inducing "$SPARSE_GPR_NUM_INDUCING" \
  --inducing-selection kmeans \
  --kernel-name "$SPARSE_GPR_KERNEL" \
  --ard \
  --epochs "$SPARSE_GPR_EPOCHS" \
  --batch-size "$SPARSE_GPR_BATCH_SIZE" \
  --lr "$SPARSE_GPR_LR" \
  --weight-decay "$SPARSE_GPR_WEIGHT_DECAY" \
  --min-noise "$SPARSE_GPR_MIN_NOISE" \
  --max-noise "$SPARSE_GPR_MAX_NOISE" \
  --elbo-beta "$SPARSE_GPR_ELBO_BETA" \
  --device "$SPARSE_GPR_DEVICE" \
  --seed "$SPARSE_GPR_SEED" \
  --max-train-samples "$SPARSE_GPR_MAX_TRAIN_SAMPLES" \
  --max-val-samples "$SPARSE_GPR_MAX_VAL_SAMPLES" \
  --log-every "$SPARSE_GPR_LOG_EVERY" \
  "${fixed_inducing_args[@]}" \
  2>&1 | tee "$LOG_FILE"

grep -E "model_path|dataset_dir|validation_dataset_dir|n_tot|kernel_learned|num_inducing|epochs|batch_size|lr|min_noise|max_noise|elbo_beta|fixed_inducing|val_split|train_rel_frob_percent|val_rel_frob_percent|elapsed_s" \
  "$PAPER_ROOT/Stage3/$SUMMARY_NAME" \
  | tee "$LOG_DIR/train_master_sparse_gpr_${SPARSE_GPR_LABEL}_check.txt"
