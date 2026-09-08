#!/usr/bin/env bash
set -euo pipefail

# Build an OOF tail-error basis and test the residual-corrected Case--2 PROM.
# Existing Case--2 results are never reused or overwritten by this diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

ACTION="${1:-all}"
case "$ACTION" in
  build|test|all) ;;
  *) echo "Usage: $0 {build|test|all}" >&2; exit 2 ;;
esac

PAPER_ROOT="${PAPER_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151}"
VALIDATION_DATASET_DIR="${VALIDATION_DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2}"
STAGE3_DIR="${STAGE3_DIR:-$PAPER_ROOT/Stage3}"
MODEL_PATH="${MODEL_PATH:-$STAGE3_DIR/models/master_ann_mu_t_to_qtot_ntot151_best.pt}"
TAIL_BASIS_DIR="${TAIL_BASIS_DIR:-$STAGE3_DIR/case2_tail_correction_oof_ntot151_np10}"
TAIL_BASIS_PATH="${TAIL_BASIS_PATH:-$TAIL_BASIS_DIR/tail_correction_basis.npy}"
RUN_ROOT="${RUN_ROOT:-$PAPER_ROOT/Runs/PROM/Case2_TailCorrectedOOF/np10}"
BASIS_PATH="${BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
U_REF_PATH="${U_REF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
RANKS="${RANKS:-0 1 3 5 10}"
POINTS="${POINTS:-4.875,0.0225 4.560,0.0190 5.190,0.0260 4.000,0.0330}"
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-16}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"
OOF_FOLDS="${OOF_FOLDS:-3}"
WEIGHTING="${WEIGHTING:-plain}"
if [[ "$WEIGHTING" == "plain" ]]; then
  LOG_SUFFIX="${LOG_SUFFIX:-}"
else
  LOG_SUFFIX="${LOG_SUFFIX:-_$WEIGHTING}"
fi

export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"
mkdir -p "$TAIL_BASIS_DIR" "$RUN_ROOT" "$PAPER_ROOT/logs/case2_tail_corrected" "$MPLCONFIGDIR"

if [[ ! -d "$TRAIN_DATASET_DIR/per_mu" || ! -d "$VALIDATION_DATASET_DIR/per_mu" ]]; then
  echo "Missing training or validation dataset per_mu directory." >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" || ! -f "$BASIS_PATH" || ! -f "$U_REF_PATH" ]]; then
  echo "Missing master model, basis, or reference state." >&2
  exit 1
fi

echo "[case2-tail-corrected] action:       $ACTION"
echo "[case2-tail-corrected] dataset:      $TRAIN_DATASET_DIR"
echo "[case2-tail-corrected] validation:   $VALIDATION_DATASET_DIR"
echo "[case2-tail-corrected] master ANN:   $MODEL_PATH"
echo "[case2-tail-corrected] OOF basis:    $TAIL_BASIS_PATH"
echo "[case2-tail-corrected] ranks:        $RANKS"
echo "[case2-tail-corrected] points:       $POINTS"
echo "[case2-tail-corrected] threads/dev:  $TRAIN_NUM_THREADS/$ONLINE_DEVICE"
echo "[case2-tail-corrected] force:        $FORCE"

if [[ "$ACTION" == "build" || "$ACTION" == "all" ]]; then
  build_args=(--weighting "$WEIGHTING")
  [[ "$FORCE" == "1" ]] && build_args+=(--force)
  [[ "$WEIGHTING" == "observability" ]] && build_args+=(--u-ref-path "$U_REF_PATH")
  python3 -u stage3_build_case2_tail_correction_basis.py \
    --dataset-dir "$TRAIN_DATASET_DIR" \
    --validation-dataset-dir "$VALIDATION_DATASET_DIR" \
    --stage3-dir "$STAGE3_DIR" \
    --basis-path "$BASIS_PATH" \
    --ntot 151 \
    --n-primary 10 \
    --folds "$OOF_FOLDS" \
    --hidden-dims 256,512,512,256 \
    --activation elu \
    --epochs 7000 \
    --patience 300 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-6 \
    --seed 42 \
    --device "$ONLINE_DEVICE" \
    --num-threads "$TRAIN_NUM_THREADS" \
    --output-dir "$TAIL_BASIS_DIR" \
    "${build_args[@]}" \
    2>&1 | tee "$PAPER_ROOT/logs/case2_tail_corrected/build_oof_basis${LOG_SUFFIX}.log"
fi

if [[ "$ACTION" == "test" || "$ACTION" == "all" ]]; then
  if [[ ! -f "$TAIL_BASIS_PATH" ]]; then
    echo "Missing OOF tail-correction basis: $TAIL_BASIS_PATH" >&2
    exit 1
  fi
  for rank in $RANKS; do
    test_args=()
    [[ "$FORCE" == "1" ]] && test_args+=(--force)
    python3 -u run_prom_ann_case_2_tail_corrected.py \
      --model-path "$MODEL_PATH" \
      --correction-basis-path "$TAIL_BASIS_PATH" \
      --basis-path "$BASIS_PATH" \
      --u-ref-path "$U_REF_PATH" \
      --output-root "$RUN_ROOT/r${rank}" \
      --rank "$rank" \
      --n-primary 10 \
      --points $POINTS \
      --device "$ONLINE_DEVICE" \
      --max-its 20 \
      --relnorm-cutoff 1e-5 \
      --min-delta 1e-2 \
      "${test_args[@]}" \
      2>&1 | tee "$PAPER_ROOT/logs/case2_tail_corrected/test_r${rank}${LOG_SUFFIX}.log"
  done
fi

echo "[case2-tail-corrected] done."
