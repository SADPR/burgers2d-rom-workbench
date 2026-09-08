#!/usr/bin/env bash
# Train and evaluate the exact solver-aware Case--2 objective proposed by Maday.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

ACTION="${1:-all}"
case "$ACTION" in
  audit|train|online|all) ;;
  *)
    echo "Usage: $0 [audit|train|online|all]" >&2
    exit 2
    ;;
esac

PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
STAGE3_DIR="$PROM_ROOT/Stage3"
TRAIN_DIR="${TRAIN_DATASET_DIR:-$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151}"
VAL_DIR="${VAL_DATASET_DIR:-$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2}"
BASIS="${BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
UREF="${U_REF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
MASTER_MODEL="${MASTER_MODEL:-$STAGE3_DIR/models/master_ann_mu_t_to_qtot_ntot151_best.pt}"
MODEL_NAME="${MODEL_NAME:-case2_online_response_master_qtot_ntot151_np10_best.pt}"
MODEL="$STAGE3_DIR/models/$MODEL_NAME"
LOG_DIR="$PROM_ROOT/logs/case2_online_response"
ONLINE_ROOT="$PROM_ROOT/Runs/PROM/Case2_OnlineResponse/np10"

TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-16}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
EPOCHS="${EPOCHS:-6}"
VALIDATION_EVERY="${VALIDATION_EVERY:-1}"
PATIENCE="${PATIENCE:-3}"
LR="${LR:-5e-6}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-0}"
CASE2_MAX_ITS="${CASE2_MAX_ITS:-20}"
CASE2_RELNORM_CUTOFF="${CASE2_RELNORM_CUTOFF:-1e-5}"
CASE2_MIN_DELTA="${CASE2_MIN_DELTA:-1e-2}"

export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"
export MPLCONFIGDIR="$PROM_ROOT/.mplcache"
mkdir -p "$LOG_DIR" "$MPLCONFIGDIR" "$STAGE3_DIR/models" "$ONLINE_ROOT"

for path in "$TRAIN_DIR/per_mu" "$VAL_DIR/per_mu" "$BASIS" "$UREF" "$MASTER_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "[error] Missing required input: $path" >&2
    exit 1
  fi
done

echo "[case2-online-response] action:       $ACTION"
echo "[case2-online-response] train/val:    $TRAIN_DIR | $VAL_DIR"
echo "[case2-online-response] source model: $MASTER_MODEL"
echo "[case2-online-response] output model: $MODEL"
echo "[case2-online-response] epochs:       $EPOCHS"
echo "[case2-online-response] threads:      $TRAIN_NUM_THREADS"
echo "[case2-online-response] Case-2 solve: max_its=$CASE2_MAX_ITS relnorm=$CASE2_RELNORM_CUTOFF min_delta=$CASE2_MIN_DELTA"
echo "[case2-online-response] force:        $FORCE"
echo "[case2-online-response] resume:       $RESUME"

run_audit() {
  python3 -u stage3_perform_training_case_2_online_response.py \
    --dataset-dir "$TRAIN_DIR" \
    --validation-dataset-dir "$VAL_DIR" \
    --stage3-dir "$STAGE3_DIR" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --initialize-from-master "$MASTER_MODEL" \
    --model-name "$MODEL_NAME" \
    --n-primary 10 \
    --num-threads "$TRAIN_NUM_THREADS" \
    --max-its "$CASE2_MAX_ITS" \
    --relnorm-cutoff "$CASE2_RELNORM_CUTOFF" \
    --min-delta "$CASE2_MIN_DELTA" \
    --audit-only \
    2>&1 | tee "$LOG_DIR/directional_audit.log"
}

run_train() {
  local force_args=()
  local resume_args=()
  if [[ "$FORCE" == "1" ]]; then
    force_args+=(--force)
  fi
  if [[ "$RESUME" == "1" ]]; then
    resume_args+=(--resume)
  fi
  python3 -u stage3_perform_training_case_2_online_response.py \
    --dataset-dir "$TRAIN_DIR" \
    --validation-dataset-dir "$VAL_DIR" \
    --stage3-dir "$STAGE3_DIR" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --initialize-from-master "$MASTER_MODEL" \
    --model-name "$MODEL_NAME" \
    --n-primary 10 \
    --alpha-primary auto \
    --epochs "$EPOCHS" \
    --validation-every "$VALIDATION_EVERY" \
    --patience "$PATIENCE" \
    --lr "$LR" \
    --num-threads "$TRAIN_NUM_THREADS" \
    --max-its "$CASE2_MAX_ITS" \
    --relnorm-cutoff "$CASE2_RELNORM_CUTOFF" \
    --min-delta "$CASE2_MIN_DELTA" \
    "${resume_args[@]}" \
    "${force_args[@]}" \
    2>&1 | tee "$LOG_DIR/train.log"
}

run_online_point() {
  local mu1="$1"
  local mu2="$2"
  local tag1 tag2
  tag1=$(printf '%.3f' "$mu1")
  tag2=$(printf '%.4f' "$mu2")
  local run_tag="case2_prom_ann_online_response_mu1_${tag1}_mu2_${tag2}_n10_ntot151"
  if [[ "$FORCE" != "1" && -f "$ONLINE_ROOT/${run_tag}_summary.txt" ]]; then
    echo "[skip] online response Case 2: mu=($mu1,$mu2)"
    return
  fi
  python3 -u run_prom_ann_case_2.py \
    --backend prom --no-ecsw \
    --mu1 "$mu1" --mu2 "$mu2" --device "$ONLINE_DEVICE" \
    --model-path "$MODEL" --target-primary-modes 10 \
    --run-tag-extra online_response \
    --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-root "$ONLINE_ROOT" \
    --max-its "$CASE2_MAX_ITS" --relnorm-cutoff "$CASE2_RELNORM_CUTOFF" --min-delta "$CASE2_MIN_DELTA" \
    --linear-solver lstsq --normal-eq-reg 1e-12 \
    --no-save-rom-snaps --no-plot \
    2>&1 | tee "$LOG_DIR/online_mu1_${tag1}_mu2_${tag2}.log"
}

if [[ "$ACTION" == "audit" || "$ACTION" == "all" ]]; then
  run_audit
fi
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
  run_train
fi
if [[ "$ACTION" == "online" || "$ACTION" == "all" ]]; then
  if [[ ! -f "$MODEL" ]]; then
    echo "[error] Missing trained solver-aware model: $MODEL" >&2
    exit 1
  fi
  run_online_point 4.875 0.0225
  run_online_point 4.560 0.0190
  run_online_point 5.190 0.0260
  run_online_point 4.000 0.0330
fi

echo "[case2-online-response] done."
