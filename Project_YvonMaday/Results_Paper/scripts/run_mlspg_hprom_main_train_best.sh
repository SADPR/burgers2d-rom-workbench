#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
export PAPER_TAG="${PAPER_TAG:-mlspg_hprom_main}"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="${DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151}"
export STAGE3_DIR="$PAPER_ROOT/Stage3"
export MODELS_DIR="$STAGE3_DIR/models"
export LOG_ROOT="$PAPER_ROOT/logs/train_best_baseline"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"

TRAIN_SMOKE_TEST="${TRAIN_SMOKE_TEST:-0}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
family="${1:-all}"

case "$family" in
  all|case1|case2|case3|data_driven|pod_ae|pod_dl) ;;
  *)
    echo "Usage: $0 [all|case1|case2|case3|data_driven|pod_ae|pod_dl]" >&2
    exit 2
    ;;
esac

mkdir -p "$MODELS_DIR" "$LOG_ROOT" "$MPLCONFIGDIR"

selected_epochs() {
  if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then echo 1; else echo "$1"; fi
}

selected_patience() {
  if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then echo 1; else echo "$1"; fi
}

selected_pretrain() {
  if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then echo 0; else echo "$1"; fi
}

print_plan() {
  cat <<EOF
[hprom-baseline-train] selected-architecture training, no sweep
[hprom-baseline-train] paper root:  $PAPER_ROOT
[hprom-baseline-train] dataset:     $DATASET_DIR
[hprom-baseline-train] stage3 dir:  $STAGE3_DIR
[hprom-baseline-train] models dir:  $MODELS_DIR
[hprom-baseline-train] logs dir:    $LOG_ROOT
[hprom-baseline-train] family:      $family
[hprom-baseline-train] threads:     $TRAIN_NUM_THREADS
[hprom-baseline-train] force:       $FORCE
[hprom-baseline-train] selected architectures:
  Case 1:       wide SiLU, n=10, hidden=(256,512,512,256)
  Case 2:       uses the POD-NN-ROM map mu,t -> q_tot; no separate Case 2 ANN is trained
  Case 3:       wide SiLU, n=10, hidden=(256,512,512,256)
  POD-NN-ROM:   wide SiLU, hidden=(256,512,512,256)
  PROM-POD-AE:  latent=10, GELU, z-score, hidden=(512,256,128)
  POD-DL-ROM:   latent=10, SiLU, encoder=(512,256), decoder=(256,512), dynamics=(256,512,512,256)
EOF
}

check_dataset() {
  python3 - <<'PY'
import os
from pathlib import Path

import numpy as np

from stage3_dataset_utils import read_dataset_meta

dataset = Path(os.environ["DATASET_DIR"]).expanduser().resolve()
if not dataset.is_dir():
    raise SystemExit(f"Missing HPROM baseline dataset directory: {dataset}")

meta, meta_path = read_dataset_meta(dataset)
solve_backend = str(meta.get("solve_backend", "")).lower()
total_modes = int(meta.get("total_modes", -1))
num_traj = int(meta.get("num_traj", -1))

if solve_backend != "hprom":
    raise SystemExit(f"Expected solve_backend=hprom, got {meta.get('solve_backend')!r}.")
if total_modes != 151:
    raise SystemExit(f"Expected total_modes=151, got {total_modes}.")
if num_traj != 9:
    raise SystemExit(f"Expected 9 baseline training trajectories, got {num_traj}.")

per_mu = dataset / "per_mu"
mu_dirs = sorted(path for path in per_mu.iterdir() if path.is_dir())
if len(mu_dirs) != 9:
    raise SystemExit(f"Expected 9 per_mu directories, found {len(mu_dirs)} in {per_mu}.")

for mu_dir in mu_dirs:
    qn_path = mu_dir / "qN.npy"
    if not qn_path.is_file():
        raise SystemExit(f"Missing qN target: {qn_path}")
    qn = np.load(qn_path, allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid qN in {mu_dir}: shape={qn.shape}")

print("[hprom-baseline-train-check] dataset:", dataset)
print("[hprom-baseline-train-check] metadata:", meta_path)
print("[hprom-baseline-train-check] trajectories:", len(mu_dirs), "(9 x 501 = 4509 rows)")
print("[hprom-baseline-train-check] solve_backend:", solve_backend)
print("[hprom-baseline-train-check] total_modes:", total_modes)
print("[hprom-baseline-train-check] ecsw_weights_path:", meta.get("ecsw_weights_path"))
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
  if [[ ! -f "$model_path" ]]; then
    echo "[error] Training finished without model: $model_path" >&2
    exit 1
  fi
  if [[ ! -f "$summary_path" ]]; then
    echo "[error] Training finished without summary: $summary_path" >&2
    exit 1
  fi
}

train_case1() {
  local log_dir="$LOG_ROOT/case1"
  local model="$MODELS_DIR/case1_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/case1_ann_ntot151_best_summary.txt"
  local log="$log_dir/case1_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train HPROM baseline Case 1: wide SiLU"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_case_1_ann_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --primary-modes 10 \
      --model-name "case1_ann_ntot151_best.pt" \
      --summary-name "case1_ann_ntot151_best_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --hidden-dims "256,512,512,256" --activation silu \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_case3() {
  local log_dir="$LOG_ROOT/case3"
  local model="$MODELS_DIR/case3_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/case3_ann_ntot151_best_summary.txt"
  local log="$log_dir/case3_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train HPROM baseline Case 3: wide SiLU"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_case_3_ann_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --primary-modes 10 \
      --model-name "case3_ann_ntot151_best.pt" \
      --summary-name "case3_ann_ntot151_best_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --hidden-dims "256,512,512,256" --activation silu \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_data_driven() {
  local log_dir="$LOG_ROOT/data_driven"
  local model="$MODELS_DIR/data_driven_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/data_driven_ann_ntot151_best_summary.txt"
  local log="$log_dir/data_driven_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train HPROM baseline POD-NN-ROM: wide SiLU"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_rom_data_driven_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --model-name "data_driven_ann_ntot151_best.pt" \
      --summary-name "data_driven_ann_ntot151_best_summary.txt" \
      --hidden-dims "256,512,512,256" --activation silu \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      --seed 42
}

train_pod_ae() {
  local log_dir="$LOG_ROOT/pod_ae"
  local model="$MODELS_DIR/prom_pod_ae_ntot151_best.pt"
  local summary="$STAGE3_DIR/prom_pod_ae_ntot151_best_summary.txt"
  local log="$log_dir/prom_pod_ae_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6500)"
  patience="$(selected_patience 300)"
  echo "==== Train HPROM baseline PROM-POD-AE: latent-10 GELU z-score"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_prom_pod_ae.py \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --stage3-dir "$STAGE3_DIR" --models-dir "$MODELS_DIR" \
      --model-name "prom_pod_ae_ntot151_best.pt" \
      --summary-name "prom_pod_ae_ntot151_best_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --latent-dim 10 --hidden-dims "512,256,128" \
      --activation gelu --scaling zscore \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

train_pod_dl() {
  local log_dir="$LOG_ROOT/pod_dl"
  local model="$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"
  local summary="$STAGE3_DIR/pod_dl_data_driven_ntot151_best_summary.txt"
  local log="$log_dir/pod_dl_data_driven_ntot151_best.log"
  local epochs patience pretrain
  epochs="$(selected_epochs 7000)"
  patience="$(selected_patience 320)"
  pretrain="$(selected_pretrain 300)"
  echo "==== Train HPROM baseline POD-DL-ROM: latent-10 SiLU"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_pod_dl_data_driven.py \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --stage3-dir "$STAGE3_DIR" --models-dir "$MODELS_DIR" \
      --model-name "pod_dl_data_driven_ntot151_best.pt" \
      --summary-name "pod_dl_data_driven_ntot151_best_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --latent-dim 10 \
      --encoder-hidden-dims "512,256" --decoder-hidden-dims "256,512" \
      --dynamics-hidden-dims "256,512,512,256" --activation silu \
      --x-scaling zscore --q-scaling zscore \
      --omega-data 1.0 --omega-latent 0.03 --omega-recon 0.01 \
      --pretrain-epochs "$pretrain" \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6
}

run_family() {
  case "$1" in
    case1) train_case1 ;;
    case2)
      echo "==== Case 2 uses the POD-NN-ROM master map; training data_driven instead."
      train_data_driven
      ;;
    case3) train_case3 ;;
    data_driven) train_data_driven ;;
    pod_ae) train_pod_ae ;;
    pod_dl) train_pod_dl ;;
  esac
}

print_plan
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[hprom-baseline-train] PLAN_ONLY=1; no dataset check or training was run."
  exit 0
fi

check_dataset

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[hprom-baseline-train] CHECK_ONLY=1; dataset is complete and no training was run."
  exit 0
fi

if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then
  echo "[hprom-baseline-train] TRAIN_SMOKE_TEST=1: using one epoch and no POD-DL pretrain."
fi

if [[ "$family" == "all" ]]; then
  for requested in case1 case3 data_driven pod_ae pod_dl; do
    run_family "$requested"
  done
else
  run_family "$family"
fi

echo "[hprom-baseline-train] Completed selected training for family=$family"
