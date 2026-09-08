#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
export PAPER_TAG="${PAPER_TAG:-mlspg_hprom_enrichment_ext25_lhs36}"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="${DATASET_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151_enriched_lhs36}"
export VAL_DATASET_DIR="${VAL_DATASET_DIR:-$PAPER_RESULTS_ROOT/mlspg_hprom_main/Stage2/prom_coeff_dataset_ntot151_validation2}"
export STAGE3_DIR="$PAPER_ROOT/Stage3"
export MODELS_DIR="$STAGE3_DIR/models"
export LOG_ROOT="$PAPER_ROOT/logs/train_best"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"

TRAIN_SMOKE_TEST="${TRAIN_SMOKE_TEST:-0}"
TRAIN_EXECUTION="${TRAIN_EXECUTION:-sequential}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
family="${1:-all}"

case "$family" in
  all|case1|case2|case3|data_driven|pod_ae|pod_dl) ;;
  *)
    echo "Usage: $0 [all|case1|case2|case3|data_driven|pod_ae|pod_dl]" >&2
    echo "Note: case2 trains the master POD-NN map used as the Case-2 tail source." >&2
    exit 2
    ;;
esac
case "$TRAIN_EXECUTION" in
  sequential|parallel) ;;
  *)
    echo "[error] TRAIN_EXECUTION must be sequential or parallel." >&2
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
[ext25-lhs36-train] selected-architecture training, no sweep
[ext25-lhs36-train] paper root:  $PAPER_ROOT
[ext25-lhs36-train] dataset:     $DATASET_DIR
[ext25-lhs36-train] validation:  $VAL_DATASET_DIR
[ext25-lhs36-train] stage3 dir:  $STAGE3_DIR
[ext25-lhs36-train] models dir:  $MODELS_DIR
[ext25-lhs36-train] logs dir:    $LOG_ROOT
[ext25-lhs36-train] execution:   $TRAIN_EXECUTION
[ext25-lhs36-train] family:      $family
[ext25-lhs36-train] threads:     $TRAIN_NUM_THREADS
[ext25-lhs36-train] selected architectures:
  Case 1:       C02 wide SiLU, n=10, hidden=(256,512,512,256)
  Case 2:       uses the POD-NN-ROM map mu,t -> q_tot; no separate Case 2 ANN is trained
  Case 3:       C02 wide SiLU, n=10, hidden=(256,512,512,256)
  POD-NN-ROM:   A10 wide SiLU, hidden=(256,512,512,256)
  PROM-POD-AE:  PAE06, latent=10, GELU, z-score, hidden=(512,256,128)
  POD-DL-ROM:   PDL07, latent=10, SiLU, low latent regularization
EOF
}

check_dataset() {
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np

project = Path.cwd().resolve()
dataset = Path(os.environ["DATASET_DIR"]).expanduser().resolve()
val_dataset = Path(os.environ["VAL_DATASET_DIR"]).expanduser().resolve()
meta_path = dataset / "meta.json"
if not meta_path.is_file():
    raise SystemExit(f"Missing extended enrichment metadata: {meta_path}\nRun Stage 2 first and wait for completion.")
meta = json.loads(meta_path.read_text())

def localize(path_like):
    path = Path(path_like).expanduser()
    if path.exists():
        return path.resolve()
    marker = "/Project_YvonMaday/"
    text = str(path)
    if marker in text:
        candidate = project / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate.resolve()
    return path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ecsw = localize(meta["ecsw_weights_path"])
if not ecsw.is_file():
    raise SystemExit(f"Cannot resolve the fixed baseline linear ECSW file: {ecsw}")
checks = [
    (meta.get("solve_backend") == "hprom", "Expected solve_backend=hprom."),
    (meta.get("num_traj") == 45, f"Expected 45 trajectories, got {meta.get('num_traj')}."),
    (meta.get("num_base_traj_copied") == 9, "Expected 9 copied baseline trajectories."),
    (meta.get("num_lhs_traj") == 36, "Expected 36 LHS enrichment trajectories."),
    (meta.get("num_interior_lhs_traj") == 18, "Expected 18 interior LHS trajectories."),
    (meta.get("num_exterior_lhs_traj") == 18, "Expected 18 exterior LHS trajectories."),
    (meta.get("coefficient_storage") == "direct_solver_qN_only", "Expected direct solver-side qN targets."),
    (meta.get("ecsw_weights_copied") is False, "ECSW must not be copied."),
    (meta.get("ecsw_weights_rebuilt") is False, "ECSW must not be rebuilt."),
    (abs(float(meta.get("margin_fraction", -1.0)) - 0.25) < 1e-14, "Expected margin_fraction=0.25."),
]
for ok, msg in checks:
    if not ok:
        raise SystemExit(msg)
if sha256(ecsw) != meta.get("ecsw_weights_sha256"):
    raise SystemExit("The fixed baseline linear ECSW checksum does not match metadata.")

mu_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())
if len(mu_dirs) != 45:
    raise SystemExit(f"Expected 45 per_mu directories, found {len(mu_dirs)}.")
for mu_dir in mu_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid qN in {mu_dir}: shape={qn.shape}.")

val_meta_path = val_dataset / "meta.json"
if not val_meta_path.is_file():
    raise SystemExit(f"Missing validation metadata: {val_meta_path}")
val_meta = json.loads(val_meta_path.read_text())
if val_meta.get("solve_backend") != "hprom":
    raise SystemExit(f"Expected validation solve_backend=hprom, found {val_meta.get('solve_backend')}.")
val_dirs = sorted(path for path in (val_dataset / "per_mu").iterdir() if path.is_dir())
if len(val_dirs) != 2:
    raise SystemExit(f"Expected 2 validation trajectories, found {len(val_dirs)} in {val_dataset}.")
for mu_dir in val_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid validation qN in {mu_dir}: shape={qn.shape}.")

print("[ext25-lhs36-train-check] dataset:", dataset)
print("[ext25-lhs36-train-check] trajectories:", len(mu_dirs), "(45 x 501 = 22545 rows)")
print("[ext25-lhs36-train-check] validation trajectories:", len(val_dirs), "(2 x 501 = 1002 rows)")
print("[ext25-lhs36-train-check] direct qN:", meta["coefficient_storage"])
print("[ext25-lhs36-train-check] fixed linear ECSW:", ecsw)
print("[ext25-lhs36-train-check] ECSW SHA-256:", meta["ecsw_weights_sha256"])
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
  echo "==== Train ext25-lhs36 Case 1: C02 wide SiLU"
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

train_master_ann() {
  local log_dir="$LOG_ROOT/data_driven"
  local model="$MODELS_DIR/data_driven_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/data_driven_ann_ntot151_best_summary.txt"
  local log="$log_dir/data_driven_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train ext25-lhs36 POD-NN-ROM master map for Case 2/Data-driven"
  run_logged "$model" "$summary" "$log" \
    python3 -u stage3_perform_training_rom_data_driven_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --validation-dataset-dir "$VAL_DATASET_DIR" \
      --model-name "data_driven_ann_ntot151_best.pt" \
      --summary-name "data_driven_ann_ntot151_best_summary.txt" \
      --hidden-dims "256,512,512,256" --activation silu \
      --batch-size 128 --lr 5e-4 --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      --seed 42
}

train_case3() {
  local log_dir="$LOG_ROOT/case3"
  local model="$MODELS_DIR/case3_ann_ntot151_best.pt"
  local summary="$STAGE3_DIR/case3_ann_ntot151_best_summary.txt"
  local log="$log_dir/case3_ann_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6000)"
  patience="$(selected_patience 220)"
  echo "==== Train ext25-lhs36 Case 3: C02 wide SiLU"
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

train_pod_ae() {
  local log_dir="$LOG_ROOT/pod_ae"
  local model="$MODELS_DIR/prom_pod_ae_ntot151_best.pt"
  local summary="$STAGE3_DIR/prom_pod_ae_ntot151_best_summary.txt"
  local log="$log_dir/prom_pod_ae_ntot151_best.log"
  local epochs patience
  epochs="$(selected_epochs 6500)"
  patience="$(selected_patience 300)"
  echo "==== Train ext25-lhs36 PROM-POD-AE: PAE06 latent-10 GELU z-score wide"
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
  echo "==== Train ext25-lhs36 POD-DL-ROM: PDL07 latent-10 SiLU low-latent-reg"
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
    case2|data_driven) train_master_ann ;;
    case3) train_case3 ;;
    pod_ae) train_pod_ae ;;
    pod_dl) train_pod_dl ;;
  esac
}

print_plan
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[ext25-lhs36-train] PLAN_ONLY=1; no dataset check or training was run."
  exit 0
fi

check_dataset

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[ext25-lhs36-train] CHECK_ONLY=1; dataset is complete and no training was run."
  exit 0
fi

if [[ "$TRAIN_SMOKE_TEST" == "1" ]]; then
  echo "[ext25-lhs36-train] TRAIN_SMOKE_TEST=1: using one epoch and no POD-DL pretrain."
fi

if [[ "$family" == "all" ]]; then
  families=(case1 data_driven case3 pod_ae pod_dl)
  if [[ "$TRAIN_EXECUTION" == "parallel" ]]; then
    echo "[ext25-lhs36-train] Running selected trainings concurrently. Use only with adequate resources."
    pids=()
    for requested in "${families[@]}"; do
      run_family "$requested" &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    if [[ "$status" -ne 0 ]]; then
      echo "[error] At least one selected training failed." >&2
      exit "$status"
    fi
  else
    for requested in "${families[@]}"; do
      run_family "$requested"
    done
  fi
else
  run_family "$family"
fi

echo "[ext25-lhs36-train] Completed selected training for family=$family"
