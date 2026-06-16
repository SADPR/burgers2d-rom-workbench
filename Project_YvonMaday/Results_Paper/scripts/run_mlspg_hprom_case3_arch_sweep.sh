#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
export STAGE3_DIR="$PAPER_ROOT/Stage3"
export MODELS_DIR="$STAGE3_DIR/models"
export LOG_DIR="$PAPER_ROOT/logs/case3_arch_sweep"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

mkdir -p "$MODELS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

cleanup_failed_candidates() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    rm -f \
      "$MODELS_DIR"/case3_ann_ntot151_C*.pt \
      "$STAGE3_DIR"/case3_ann_ntot151_C*_summary.txt
    echo "[case3-sweep] Failed; incomplete candidate checkpoints were removed." >&2
  fi
  exit "$status"
}
trap cleanup_failed_candidates EXIT

if [[ ! -f "$DATASET_DIR/meta.npy" && ! -f "$DATASET_DIR/meta.json" ]]; then
  echo "[error] Missing Stage-2 dataset metadata in: $DATASET_DIR" >&2
  exit 1
fi

python3 - <<'PY'
import os
from stage3_dataset_utils import read_dataset_meta

p = os.environ["DATASET_DIR"]
meta, meta_path = read_dataset_meta(p)
print("[case3-sweep-check] dataset_dir:", p)
print("[case3-sweep-check] metadata:", meta_path)
print("[case3-sweep-check] solve_backend:", meta.get("solve_backend"))
print("[case3-sweep-check] total_modes:", meta.get("total_modes"))
print("[case3-sweep-check] basis_path:", meta.get("basis_path"))
if str(meta.get("solve_backend")).lower() != "hprom":
    raise SystemExit("Case-3 architecture sweep requires the HPROM dataset.")
if int(meta.get("total_modes")) != 151:
    raise SystemExit("Expected total_modes=151.")
PY

# Remove only stale Case-3 artifacts from this paper campaign.
rm -f \
  "$MODELS_DIR"/case3_ann_ntot151_C*.pt \
  "$MODELS_DIR"/case3_ann_ntot151_best.pt \
  "$MODELS_DIR"/case3_hprom_ann_n10_ntot151.pt \
  "$STAGE3_DIR"/case3_ann_ntot151_C*_summary.txt \
  "$STAGE3_DIR"/case3_ann_ntot151_best_summary.txt \
  "$STAGE3_DIR"/case3_hprom_ann_n10_ntot151_summary.txt \
  "$LOG_DIR"/*.log \
  "$LOG_DIR"/case3_arch_sweep_summary.csv

run_cfg() {
  local label="$1"
  local hidden="$2"
  local activation="$3"
  local batch="$4"
  local lr="$5"
  local epochs="$6"
  local patience="$7"

  echo "==== Case-3 architecture sweep: ${label}"
  echo "hidden=${hidden} activation=${activation} batch=${batch} lr=${lr}"

  python3 -u stage3_perform_training_case_3_ann_maday.py \
    --maday-results-root "$PAPER_RESULTS_ROOT" \
    --maday-tag "$PAPER_TAG" \
    --dataset-backend hprom \
    --dataset-ntot 151 \
    --dataset-dir "$DATASET_DIR" \
    --primary-modes 10 \
    --model-name "case3_ann_ntot151_${label}.pt" \
    --summary-name "case3_ann_ntot151_${label}_summary.txt" \
    --seed 42 \
    --val-frac 0.1 \
    --hidden-dims "$hidden" \
    --activation "$activation" \
    --batch-size "$batch" \
    --lr "$lr" \
    --weight-decay 1e-6 \
    --dropout 0.0 \
    --epochs "$epochs" \
    --patience "$patience" \
    --lr-scheduler-factor 0.5 \
    --lr-scheduler-patience 50 \
    --lr-scheduler-min-lr 1e-6 \
    2>&1 | tee "$LOG_DIR/${label}.log"
}

# Case 3: [q_p in R^10, mu1, mu2, t] -> q_s in R^141.
run_cfg C00_legacy_elu       "32,64,128,256,256" "elu"  128 1e-3 5000 180
run_cfg C01_medium_silu      "128,256,256,128"    "silu" 128 5e-4 6000 220
run_cfg C02_wide_silu        "256,512,512,256"    "silu" 128 5e-4 6000 220
run_cfg C03_wide_elu         "256,512,512,256"    "elu"  128 5e-4 6000 220

python3 - <<'PY'
from pathlib import Path
import csv
import os
import shutil

stage3 = Path(os.environ["STAGE3_DIR"])
models = Path(os.environ["MODELS_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
rows = []

for summary in sorted(stage3.glob("case3_ann_ntot151_C*_summary.txt")):
    data = {}
    for line in summary.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    label = summary.name.replace("case3_ann_ntot151_", "").replace("_summary.txt", "")
    rows.append(
        {
            "label": label,
            "val_rel_frob_percent": data.get("val_rel_frob_percent", ""),
            "train_rel_frob_percent": data.get("train_rel_frob_percent", ""),
            "best_val_mse": data.get("best_val_mse", ""),
            "epochs_ran": data.get("epochs_ran", ""),
            "hidden_dims": data.get("hidden_dims", ""),
            "activation": data.get("activation", ""),
            "batch_size": data.get("batch_size", ""),
            "lr": data.get("lr", ""),
            "trainable_parameters": data.get("trainable_parameters", ""),
            "candidate_model": str(models / f"case3_ann_ntot151_{label}.pt"),
        }
    )

if not rows:
    raise SystemExit("No Case-3 sweep summaries found.")

def score(row):
    try:
        return float(row["val_rel_frob_percent"])
    except Exception:
        return float("inf")

rows.sort(key=score)
winner = rows[0]
winner_label = winner["label"]

csv_path = log_dir / "case3_arch_sweep_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

canonical_model = models / "case3_ann_ntot151_best.pt"
canonical_summary = stage3 / "case3_ann_ntot151_best_summary.txt"
canonical_log = log_dir / "case3_ann_ntot151_best.log"

winner_model = models / f"case3_ann_ntot151_{winner_label}.pt"
winner_summary = stage3 / f"case3_ann_ntot151_{winner_label}_summary.txt"
winner_log = log_dir / f"{winner_label}.log"
shutil.move(winner_model, canonical_model)
shutil.move(winner_summary, canonical_summary)
shutil.move(winner_log, canonical_log)

text = canonical_summary.read_text()
text = text.replace(
    f"model_name: case3_ann_ntot151_{winner_label}.pt",
    "model_name: case3_ann_ntot151_best.pt",
)
for line in text.splitlines():
    if line.startswith("model_path:"):
        text = text.replace(line, f"model_path: {canonical_model}")
        break
text += f"\nsweep_winner_label: {winner_label}\n"
canonical_summary.write_text(text)

for path in models.glob("case3_ann_ntot151_C*.pt"):
    path.unlink()
for path in stage3.glob("case3_ann_ntot151_C*_summary.txt"):
    path.unlink()
for path in log_dir.glob("C*.log"):
    path.unlink()

print("[case3-sweep-summary]", csv_path)
for row in rows:
    print(
        " ",
        row["label"],
        "val_rel_frob_percent=",
        row["val_rel_frob_percent"],
        "best_val_mse=",
        row["best_val_mse"],
    )
print("[case3-sweep-winner]", winner_label)
print("[case3-sweep-model]", canonical_model)
print("[case3-sweep-summary-file]", canonical_summary)
PY

trap - EXIT
echo "[done] Only the selected Case-3 checkpoint is retained:"
echo "       $MODELS_DIR/case3_ann_ntot151_best.pt"
