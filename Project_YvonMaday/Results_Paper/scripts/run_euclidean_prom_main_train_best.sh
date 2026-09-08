#!/usr/bin/env bash
# Train the fixed Euclidean-POD PROM models with external parameter validation.
# Case 2 and POD-NN-ROM intentionally share one master map mu,t -> q_tot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

FAMILY="${1:-all}"
case "$FAMILY" in
  all|case1|case2|case3|data_driven|pod_ae|pod_dl) ;;
  *)
    echo "Usage: $0 [all|case1|case2|case3|data_driven|pod_ae|pod_dl]" >&2
    exit 2
    ;;
esac

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Results_Paper}"
PAPER_TAG="${PAPER_TAG:-euclidean_prom_main}"
PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
DATASET_DIR="${DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151}"
VALIDATION_DATASET_DIR="${VALIDATION_DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2}"
STAGE3_DIR="$PAPER_ROOT/Stage3"
MODELS_DIR="$STAGE3_DIR/models"
LOG_ROOT="$PAPER_ROOT/logs/train_best_external_validation"

TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-24}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
TRAIN_SMOKE_TEST="${TRAIN_SMOKE_TEST:-0}"

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
}

selected_epochs() {
  if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then echo 1; else echo "$1"; fi
}

selected_patience() {
  if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then echo 1; else echo "$1"; fi
}

selected_pretrain() {
  if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then echo 0; else echo "$1"; fi
}

check_dataset() {
  DATASET_DIR="$DATASET_DIR" VALIDATION_DATASET_DIR="$VALIDATION_DATASET_DIR" python3 - <<'PY'
import os
from pathlib import Path

import numpy as np

from stage3_dataset_utils import read_dataset_meta

expected = (
    ("training", Path(os.environ["DATASET_DIR"]), 9),
    ("external validation", Path(os.environ["VALIDATION_DATASET_DIR"]), 2),
)
for label, dataset, expected_mu in expected:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Missing {label} dataset directory: {dataset}")
    meta, meta_path = read_dataset_meta(dataset)
    if str(meta.get("solve_backend", "")).lower() != "prom":
        raise SystemExit(f"{label}: expected solve_backend=prom, got {meta.get('solve_backend')!r}.")
    if int(meta.get("total_modes", -1)) != 151:
        raise SystemExit(f"{label}: expected total_modes=151.")
    mu_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())
    if len(mu_dirs) != expected_mu:
        raise SystemExit(f"{label}: expected {expected_mu} trajectories, found {len(mu_dirs)}.")
    for mu_dir in mu_dirs:
        qn = np.load(mu_dir / "qN.npy", mmap_mode="r", allow_pickle=False)
        if qn.shape != (151, 501) or not np.isfinite(qn[:, :1]).all():
            raise SystemExit(f"{label}: invalid qN in {mu_dir}: {qn.shape}.")
    print(f"[euclidean-prom-train-check] {label}: {dataset}")
    print(f"[euclidean-prom-train-check] metadata: {meta_path}")
    print(f"[euclidean-prom-train-check] trajectories: {len(mu_dirs)} x 501")
PY
}

run_logged() {
  local model_path="$1"
  local summary_path="$2"
  local log_path="$3"
  shift 3

  if [[ "$FORCE" == "1" ]]; then
    rm -f "$model_path" "$summary_path" "$log_path"
  fi
  if [[ -f "$model_path" && -f "$summary_path" ]]; then
    echo "[skip] Existing trained model: $model_path"
    return
  fi
  mkdir -p "$(dirname "$model_path")" "$(dirname "$summary_path")" "$(dirname "$log_path")"
  "$@" 2>&1 | tee "$log_path"
  [[ -f "$model_path" ]] || { echo "[error] Missing model after training: $model_path" >&2; exit 1; }
  [[ -f "$summary_path" ]] || { echo "[error] Missing summary after training: $summary_path" >&2; exit 1; }
}

train_case1() {
  local model="$MODELS_DIR/case1_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/case1_ann_ntot151_best_summary.txt"
  local log="$LOG_ROOT/case1/case1_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train Euclidean PROM-ANN Case 1 | wide SiLU | n=10"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_case_1_ann_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend prom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --validation-dataset-dir "$VALIDATION_DATASET_DIR" --primary-modes 10 \
      --model-name "case1_ann_ntot151_best.pt" --summary-name "case1_ann_ntot151_best_summary.txt" \
      --seed 42 --hidden-dims "256,512,512,256" --activation silu \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_master_map() {
  local model="$MODELS_DIR/master_ann_mu_t_to_qtot_ntot151_best.pt"
  local summary="$STAGE3_DIR/master_ann_mu_t_to_qtot_ntot151_best_summary.txt"
  local log="$LOG_ROOT/master_ann/master_ann_mu_t_to_qtot_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train Euclidean master map for PROM-ANN Case 2 and POD-NN-ROM | wide SiLU"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_rom_data_driven_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend prom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --validation-dataset-dir "$VALIDATION_DATASET_DIR" \
      --model-name "master_ann_mu_t_to_qtot_ntot151_best.pt" \
      --summary-name "master_ann_mu_t_to_qtot_ntot151_best_summary.txt" \
      --seed 42 --hidden-dims "256,512,512,256" --activation silu \
      --loss-function mse --loss-space raw --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_case3() {
  local model="$MODELS_DIR/case3_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/case3_ann_ntot151_best_summary.txt"
  local log="$LOG_ROOT/case3/case3_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train Euclidean PROM-ANN Case 3 | wide SiLU | n=10"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_case_3_ann_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend prom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --validation-dataset-dir "$VALIDATION_DATASET_DIR" --primary-modes 10 \
      --model-name "case3_ann_ntot151_best.pt" --summary-name "case3_ann_ntot151_best_summary.txt" \
      --seed 42 --hidden-dims "256,512,512,256" --activation silu \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_pod_ae() {
  local model="$MODELS_DIR/prom_pod_ae_ntot151_best.pt"
  local summary="$STAGE3_DIR/prom_pod_ae_ntot151_best_summary.txt"
  local log="$LOG_ROOT/pod_ae/prom_pod_ae_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6500)"
  patience="$(selected_patience 300)"
  echo "==== Train Euclidean PROM-POD-AE | latent 10 | GELU | z-score"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_prom_pod_ae.py \
      --dataset-backend prom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --validation-dataset-dir "$VALIDATION_DATASET_DIR" \
      --stage3-dir "$STAGE3_DIR" --models-dir "$MODELS_DIR" \
      --model-name "prom_pod_ae_ntot151_best.pt" --summary-name "prom_pod_ae_ntot151_best_summary.txt" \
      --seed 42 --latent-dim 10 --hidden-dims "512,256,128" --activation gelu --scaling zscore \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_pod_dl() {
  local model="$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"
  local summary="$STAGE3_DIR/pod_dl_data_driven_ntot151_best_summary.txt"
  local log="$LOG_ROOT/pod_dl/pod_dl_data_driven_ntot151_best.log"
  local epochs patience pretrain
  epochs="$(selected_epochs 7000)"
  patience="$(selected_patience 320)"
  pretrain="$(selected_pretrain 300)"
  echo "==== Train Euclidean POD-DL-ROM | latent 10 | SiLU"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_pod_dl_data_driven.py \
      --dataset-backend prom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --validation-dataset-dir "$VALIDATION_DATASET_DIR" \
      --stage3-dir "$STAGE3_DIR" --models-dir "$MODELS_DIR" \
      --model-name "pod_dl_data_driven_ntot151_best.pt" --summary-name "pod_dl_data_driven_ntot151_best_summary.txt" \
      --seed 42 --latent-dim 10 --encoder-hidden-dims "512,256" --decoder-hidden-dims "256,512" \
      --dynamics-hidden-dims "256,512,512,256" --activation silu --x-scaling zscore --q-scaling zscore \
      --omega-data 1.0 --omega-latent 0.03 --omega-recon 0.01 --pretrain-epochs "$pretrain" \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

print_plan() {
  cat <<EOF
[euclidean-prom-train] selected-architecture training with external parameter validation
[euclidean-prom-train] campaign:     $PAPER_ROOT
[euclidean-prom-train] training:     $DATASET_DIR
[euclidean-prom-train] validation:   $VALIDATION_DATASET_DIR
[euclidean-prom-train] stage3:       $STAGE3_DIR
[euclidean-prom-train] logs:         $LOG_ROOT
[euclidean-prom-train] family:       $FAMILY
[euclidean-prom-train] threads:      $TRAIN_NUM_THREADS
[euclidean-prom-train] force:        $FORCE
[euclidean-prom-train] validation:   2 external parameter trajectories (1002 rows)
[euclidean-prom-train] architectures:
  PROM-ANN Case 1:  q_p -> (256,512,512,256) SiLU -> q_s, n=10
  PROM-ANN Case 2:  shared master map (mu,t) -> (256,512,512,256) SiLU -> q_tot
  PROM-ANN Case 3:  [q_p,mu,t] -> (256,512,512,256) SiLU -> q_s, n=10
  PROM-POD-AE:       151 -> (512,256,128) -> 10 -> mirror, GELU, z-score
  POD-DL-ROM:        latent=10, SiLU, encoder=(512,256), decoder=(256,512), dynamics=(256,512,512,256)
EOF
}

run_family() {
  case "$1" in
    case1) train_case1 ;;
    case2|data_driven) train_master_map ;;
    case3) train_case3 ;;
    pod_ae) train_pod_ae ;;
    pod_dl) train_pod_dl ;;
  esac
}

mkdir -p "$MODELS_DIR" "$LOG_ROOT" "$PAPER_ROOT/.mplcache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"
set_threads "$TRAIN_NUM_THREADS"

print_plan
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[euclidean-prom-train] PLAN_ONLY=1; no checks or training were run."
  exit 0
fi

check_dataset
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[euclidean-prom-train] CHECK_ONLY=1; datasets are complete and no training was run."
  exit 0
fi

if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then
  echo "[euclidean-prom-train] TRAIN_SMOKE_TEST=1: one epoch, no POD-DL pretraining."
fi

if [[ "$FAMILY" == "all" ]]; then
  for requested in case1 case2 case3 pod_ae pod_dl; do
    run_family "$requested"
  done
else
  run_family "$FAMILY"
fi

echo "[euclidean-prom-train] completed family=$FAMILY"
