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
export LOG_DIR="$PAPER_ROOT/logs/stage3_master_ann_sweep"
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
  exit 1
fi

echo "[master-ann-sweep] training dataset:   $TRAIN_DATASET_DIR"
echo "[master-ann-sweep] validation dataset: $VAL_DATASET_DIR"
echo "[master-ann-sweep] output root:        $PAPER_ROOT/Stage3"
echo "[master-ann-sweep] threads:            $TRAIN_NUM_THREADS"

run_cfg() {
  local label="$1"
  local hidden="$2"
  local activation="$3"
  local loss_function="$4"
  local loss_space="$5"
  local batch="$6"
  local lr="$7"
  local wd="$8"
  local dropout="$9"
  local epochs="${10}"
  local patience="${11}"

  echo "==== master ANN sweep: ${label}"
  echo "hidden=${hidden} activation=${activation} loss=${loss_function}/${loss_space} batch=${batch} lr=${lr} wd=${wd} dropout=${dropout}"

  python3 -u stage3_perform_training_rom_data_driven_maday.py \
    --maday-results-root "$PAPER_RESULTS_ROOT" \
    --maday-tag "$PAPER_TAG" \
    --dataset-backend prom \
    --dataset-ntot 151 \
    --dataset-dir "$TRAIN_DATASET_DIR" \
    --validation-dataset-dir "$VAL_DATASET_DIR" \
    --model-name "master_ann_mu_t_to_qtot_ntot151_${label}.pt" \
    --summary-name "master_ann_mu_t_to_qtot_ntot151_${label}_summary.txt" \
    --hidden-dims "$hidden" \
    --activation "$activation" \
    --loss-function "$loss_function" \
    --loss-space "$loss_space" \
    --batch-size "$batch" \
    --lr "$lr" \
    --weight-decay "$wd" \
    --dropout "$dropout" \
    --epochs "$epochs" \
    --patience "$patience" \
    --lr-scheduler-factor 0.5 \
    --lr-scheduler-patience 60 \
    --lr-scheduler-min-lr 1e-6 \
    --clip-grad 1.0 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/${label}.log"
}

# Current baseline from the first master ANN run.
run_cfg M00_current_elu_raw_mse        "32,64,128,256,256"  elu  mse       raw        128 1e-3 1e-6 0.00 6000 250

# A10-style wide architecture from the previous sweep winner.
run_cfg M01_A10_silu_raw_mse          "256,512,512,256"    silu mse       raw        128 5e-4 1e-6 0.00 7000 300
run_cfg M02_A10_elu_raw_mse           "256,512,512,256"    elu  mse       raw        128 5e-4 1e-6 0.00 7000 300
run_cfg M03_A10_gelu_raw_mse          "256,512,512,256"    gelu mse       raw        128 5e-4 1e-6 0.00 7000 300

# Equalize coefficient importance by applying the loss in normalized q-space.
run_cfg M04_A10_silu_norm_mse         "256,512,512,256"    silu mse       normalized 128 5e-4 1e-6 0.00 7000 300
run_cfg M05_A10_silu_norm_smoothl1    "256,512,512,256"    silu smooth_l1 normalized 128 5e-4 1e-6 0.00 7000 300

python3 - <<'PY'
from pathlib import Path
import csv
import os
import shutil

stage3 = Path(os.environ["PAPER_ROOT"]) / "Stage3"
models = stage3 / "models"
log_dir = Path(os.environ["LOG_DIR"])
rows = []

for p in sorted(stage3.glob("master_ann_mu_t_to_qtot_ntot151_M*_summary.txt")):
    data = {}
    for line in p.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    label = p.name.replace("master_ann_mu_t_to_qtot_ntot151_", "").replace("_summary.txt", "")
    rows.append({
        "label": label,
        "val_rel_frob_percent": data.get("val_rel_frob_percent", ""),
        "train_rel_frob_percent": data.get("train_rel_frob_percent", ""),
        "best_val_loss": data.get("best_val_loss", data.get("best_val_mse", "")),
        "hidden_dims": data.get("hidden_dims", ""),
        "activation": data.get("activation", ""),
        "loss_function": data.get("loss_function", ""),
        "loss_space": data.get("loss_space", ""),
        "batch_size": data.get("batch_size", ""),
        "lr": data.get("lr", ""),
        "weight_decay": data.get("weight_decay", ""),
        "dropout": data.get("dropout", ""),
        "epochs_ran": data.get("epochs_ran", ""),
        "trainable_parameters": data.get("trainable_parameters", ""),
        "model_path": data.get("model_path", ""),
    })

def score(row):
    try:
        return float(row["val_rel_frob_percent"])
    except Exception:
        try:
            return float(row["best_val_loss"])
        except Exception:
            return 1e99

rows.sort(key=score)
out = log_dir / "master_ann_sweep_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0].keys()) if rows else ["label"]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("[master-ann-sweep-summary]", out)
for row in rows:
    print(
        row["label"],
        "val_rel_frob_percent=", row["val_rel_frob_percent"],
        "best_val_loss=", row["best_val_loss"],
        "loss=", f'{row["loss_function"]}/{row["loss_space"]}',
    )

if rows:
    winner = rows[0]
    source_model = models / f"master_ann_mu_t_to_qtot_ntot151_{winner['label']}.pt"
    source_summary = stage3 / f"master_ann_mu_t_to_qtot_ntot151_{winner['label']}_summary.txt"
    target_model = models / "master_ann_mu_t_to_qtot_ntot151_best.pt"
    target_summary = stage3 / "master_ann_mu_t_to_qtot_ntot151_best_summary.txt"
    shutil.copy2(source_model, target_model)
    text = source_summary.read_text(errors="replace")
    text = text.replace(source_model.name, target_model.name)
    text = text.rstrip() + f"\n\nsweep_winner_label: {winner['label']}\n"
    target_summary.write_text(text)
    print("[master-ann-sweep-winner]", winner["label"])
    print("[master-ann-sweep-model]", target_model)
PY
