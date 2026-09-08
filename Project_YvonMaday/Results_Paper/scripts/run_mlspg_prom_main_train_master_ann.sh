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
export LOG_DIR="$PAPER_ROOT/logs/stage3_master_ann"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"
export TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-16}"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"

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

echo "[master-ann] training dataset:   $TRAIN_DATASET_DIR"
echo "[master-ann] validation dataset: $VAL_DATASET_DIR"
echo "[master-ann] output root:        $PAPER_ROOT/Stage3"
echo "[master-ann] threads:            $TRAIN_NUM_THREADS"

python3 -u stage3_perform_training_rom_data_driven_maday.py \
  --maday-results-root "$PAPER_RESULTS_ROOT" \
  --maday-tag "$PAPER_TAG" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$TRAIN_DATASET_DIR" \
  --validation-dataset-dir "$VAL_DATASET_DIR" \
  --model-name master_ann_mu_t_to_qtot_ntot151.pt \
  --summary-name master_ann_mu_t_to_qtot_ntot151_summary.txt \
  --hidden-dims 32,64,128,256,256 \
  --activation elu \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-6 \
  --epochs 6000 \
  --patience 250 \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 60 \
  --lr-scheduler-min-lr 1e-6 \
  --clip-grad 1.0 \
  --seed 42 \
  2>&1 | tee "$LOG_DIR/train_master_ann_mu_t_to_qtot_ntot151.log"

grep -E "dataset_dir|validation_dataset_dir|samples_M|train_samples|val_samples|n_tot|hidden_dims|activation|best_val_mse|train_rel_frob_percent|val_rel_frob_percent|val_split|trainable_parameters" \
  "$PAPER_ROOT/Stage3/master_ann_mu_t_to_qtot_ntot151_summary.txt" \
  | tee "$LOG_DIR/train_master_ann_mu_t_to_qtot_ntot151_check.txt"
