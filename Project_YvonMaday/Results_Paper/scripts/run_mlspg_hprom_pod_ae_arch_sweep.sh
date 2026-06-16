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
export LOG_DIR="$PAPER_ROOT/logs/pod_ae_arch_sweep"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

mkdir -p "$MODELS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

cleanup_failed_candidates() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    rm -f \
      "$MODELS_DIR"/prom_pod_ae_ntot151_PAE*.pt \
      "$STAGE3_DIR"/prom_pod_ae_ntot151_PAE*_summary.txt
    echo "[pod-ae-sweep] Failed; incomplete candidate checkpoints were removed." >&2
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
print("[pod-ae-sweep-check] dataset_dir:", p)
print("[pod-ae-sweep-check] metadata:", meta_path)
print("[pod-ae-sweep-check] solve_backend:", meta.get("solve_backend"))
print("[pod-ae-sweep-check] total_modes:", meta.get("total_modes"))
print("[pod-ae-sweep-check] basis_path:", meta.get("basis_path"))
if str(meta.get("solve_backend")).lower() != "hprom":
    raise SystemExit("POD-AE architecture sweep requires the HPROM dataset.")
if int(meta.get("total_modes")) != 151:
    raise SystemExit("Expected total_modes=151.")
PY

# Remove only stale POD-AE sweep artifacts from this paper campaign.
rm -f \
  "$MODELS_DIR"/prom_pod_ae_ntot151_PAE*.pt \
  "$MODELS_DIR"/prom_pod_ae_ntot151_best.pt \
  "$STAGE3_DIR"/prom_pod_ae_ntot151_PAE*_summary.txt \
  "$STAGE3_DIR"/prom_pod_ae_ntot151_best_summary.txt \
  "$LOG_DIR"/*.log \
  "$LOG_DIR"/pod_ae_arch_sweep_summary.csv

run_cfg() {
  local label="$1"
  local latent="$2"
  local hidden="$3"
  local activation="$4"
  local scaling="$5"
  local batch="$6"
  local lr="$7"
  local epochs="$8"
  local patience="$9"

  echo "==== POD-AE architecture sweep: ${label}"
  echo "latent=${latent} hidden=${hidden} activation=${activation} scaling=${scaling} batch=${batch} lr=${lr}"

  python3 -u stage3_perform_training_prom_pod_ae.py \
    --dataset-backend hprom \
    --dataset-ntot 151 \
    --dataset-dir "$DATASET_DIR" \
    --stage3-dir "$STAGE3_DIR" \
    --models-dir "$MODELS_DIR" \
    --model-name "prom_pod_ae_ntot151_${label}.pt" \
    --summary-name "prom_pod_ae_ntot151_${label}_summary.txt" \
    --seed 42 \
    --val-frac 0.1 \
    --latent-dim "$latent" \
    --hidden-dims "$hidden" \
    --activation "$activation" \
    --scaling "$scaling" \
    --batch-size "$batch" \
    --lr "$lr" \
    --weight-decay 1e-6 \
    --epochs "$epochs" \
    --patience "$patience" \
    --lr-scheduler-factor 0.5 \
    --lr-scheduler-patience 50 \
    --lr-scheduler-min-lr 1e-6 \
    2>&1 | tee "$LOG_DIR/${label}.log"
}

# Controlled POD-AE sweep in q_N space, q_N in R^151 -> z -> q_N.
# The latent dimension is fixed to 10 for all candidates. The metric is
# validation relative Frobenius error in coefficient space.
run_cfg PAE00_l10_tanh_minmax_small   10 "192,96,48"        "tanh" "minmax_-1_1" 256 1e-3 4500 220
run_cfg PAE01_l10_silu_zscore_medium  10 "256,128,64"       "silu" "zscore"      128 5e-4 5500 260
run_cfg PAE02_l10_elu_zscore_medium   10 "256,128,64"       "elu"  "zscore"      128 5e-4 5500 260
run_cfg PAE03_l10_gelu_zscore_medium  10 "256,128,64"       "gelu" "zscore"      128 5e-4 5500 260
run_cfg PAE04_l10_silu_zscore_wide    10 "512,256,128"      "silu" "zscore"      128 5e-4 6500 300
run_cfg PAE05_l10_elu_zscore_wide     10 "512,256,128"      "elu"  "zscore"      128 5e-4 6500 300
run_cfg PAE06_l10_gelu_zscore_wide    10 "512,256,128"      "gelu" "zscore"      128 5e-4 6500 300
run_cfg PAE07_l10_silu_minmax_wide    10 "512,256,128"      "silu" "minmax_-1_1" 128 5e-4 6500 300

python3 - <<'PY'
from pathlib import Path
import csv
import os
import shutil

stage3 = Path(os.environ["STAGE3_DIR"])
models = Path(os.environ["MODELS_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
rows = []

for summary in sorted(stage3.glob("prom_pod_ae_ntot151_PAE*_summary.txt")):
    data = {}
    for line in summary.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    label = summary.name.replace("prom_pod_ae_ntot151_", "").replace("_summary.txt", "")
    rows.append(
        {
            "label": label,
            "val_rel_frob_percent": data.get("val_rel_frob_percent", ""),
            "train_rel_frob_percent": data.get("train_rel_frob_percent", ""),
            "best_val_mse": data.get("best_val_mse", ""),
            "epochs_ran": data.get("epochs_ran", ""),
            "latent_dim": data.get("latent_dim", ""),
            "hidden_dims": data.get("hidden_dims", ""),
            "activation": data.get("activation", ""),
            "scaling": data.get("scaling", ""),
            "batch_size": data.get("batch_size", ""),
            "lr": data.get("lr", ""),
            "trainable_parameters": data.get("trainable_parameters", ""),
            "candidate_model": str(models / f"prom_pod_ae_ntot151_{label}.pt"),
        }
    )

if not rows:
    raise SystemExit("No POD-AE sweep summaries found.")

def score(row):
    try:
        return float(row["val_rel_frob_percent"])
    except Exception:
        return float("inf")

rows.sort(key=score)
winner = rows[0]
winner_label = winner["label"]

csv_path = log_dir / "pod_ae_arch_sweep_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

canonical_model = models / "prom_pod_ae_ntot151_best.pt"
canonical_summary = stage3 / "prom_pod_ae_ntot151_best_summary.txt"
canonical_log = log_dir / "prom_pod_ae_ntot151_best.log"

winner_model = models / f"prom_pod_ae_ntot151_{winner_label}.pt"
winner_summary = stage3 / f"prom_pod_ae_ntot151_{winner_label}_summary.txt"
winner_log = log_dir / f"{winner_label}.log"
shutil.move(winner_model, canonical_model)
shutil.move(winner_summary, canonical_summary)
shutil.move(winner_log, canonical_log)

text = canonical_summary.read_text()
text = text.replace(
    f"model_name: prom_pod_ae_ntot151_{winner_label}.pt",
    "model_name: prom_pod_ae_ntot151_best.pt",
)
for line in text.splitlines():
    if line.startswith("model_path:"):
        text = text.replace(line, f"model_path: {canonical_model}")
        break
text += f"\nsweep_winner_label: {winner_label}\n"
canonical_summary.write_text(text)

for path in models.glob("prom_pod_ae_ntot151_PAE*.pt"):
    path.unlink()
for path in stage3.glob("prom_pod_ae_ntot151_PAE*_summary.txt"):
    path.unlink()
for path in log_dir.glob("PAE*.log"):
    path.unlink()

print("[pod-ae-sweep-summary]", csv_path)
for row in rows:
    print(
        " ",
        row["label"],
        "val_rel_frob_percent=",
        row["val_rel_frob_percent"],
        "best_val_mse=",
        row["best_val_mse"],
    )
print("[pod-ae-sweep-winner]", winner_label)
print("[pod-ae-sweep-model]", canonical_model)
print("[pod-ae-sweep-summary-file]", canonical_summary)
PY

trap - EXIT
echo "[done] Only the selected POD-AE checkpoint is retained:"
echo "       $MODELS_DIR/prom_pod_ae_ntot151_best.pt"
