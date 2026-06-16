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
export LOG_DIR="$PAPER_ROOT/logs/pod_dl_arch_sweep"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

mkdir -p "$MODELS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

cleanup_failed_candidates() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    rm -f \
      "$MODELS_DIR"/pod_dl_data_driven_ntot151_PDL*.pt \
      "$STAGE3_DIR"/pod_dl_data_driven_ntot151_PDL*_summary.txt
    echo "[pod-dl-sweep] Failed; incomplete candidate checkpoints were removed." >&2
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
print("[pod-dl-sweep-check] dataset_dir:", p)
print("[pod-dl-sweep-check] metadata:", meta_path)
print("[pod-dl-sweep-check] solve_backend:", meta.get("solve_backend"))
print("[pod-dl-sweep-check] total_modes:", meta.get("total_modes"))
print("[pod-dl-sweep-check] basis_path:", meta.get("basis_path"))
if str(meta.get("solve_backend")).lower() != "hprom":
    raise SystemExit("POD-DL architecture sweep requires the HPROM dataset.")
if int(meta.get("total_modes")) != 151:
    raise SystemExit("Expected total_modes=151.")
PY

# Remove only stale POD-DL sweep artifacts from this paper campaign.
rm -f \
  "$MODELS_DIR"/pod_dl_data_driven_ntot151_PDL*.pt \
  "$MODELS_DIR"/pod_dl_data_driven_ntot151_best.pt \
  "$STAGE3_DIR"/pod_dl_data_driven_ntot151_PDL*_summary.txt \
  "$STAGE3_DIR"/pod_dl_data_driven_ntot151_best_summary.txt \
  "$LOG_DIR"/*.log \
  "$LOG_DIR"/pod_dl_arch_sweep_summary.csv

run_cfg() {
  local label="$1"
  local latent="$2"
  local enc="$3"
  local dec="$4"
  local dyn="$5"
  local activation="$6"
  local x_scaling="$7"
  local q_scaling="$8"
  local omega_latent="$9"
  local omega_recon="${10}"
  local pretrain="${11}"
  local batch="${12}"
  local lr="${13}"
  local epochs="${14}"
  local patience="${15}"

  echo "==== POD-DL architecture sweep: ${label}"
  echo "latent=${latent} enc=${enc} dec=${dec} dyn=${dyn}"
  echo "activation=${activation} x_scaling=${x_scaling} q_scaling=${q_scaling}"
  echo "omega_latent=${omega_latent} omega_recon=${omega_recon} pretrain=${pretrain} batch=${batch} lr=${lr}"

  python3 -u stage3_perform_training_pod_dl_data_driven.py \
    --dataset-backend hprom \
    --dataset-ntot 151 \
    --dataset-dir "$DATASET_DIR" \
    --stage3-dir "$STAGE3_DIR" \
    --models-dir "$MODELS_DIR" \
    --model-name "pod_dl_data_driven_ntot151_${label}.pt" \
    --summary-name "pod_dl_data_driven_ntot151_${label}_summary.txt" \
    --seed 42 \
    --val-frac 0.1 \
    --latent-dim "$latent" \
    --encoder-hidden-dims "$enc" \
    --decoder-hidden-dims "$dec" \
    --dynamics-hidden-dims "$dyn" \
    --activation "$activation" \
    --x-scaling "$x_scaling" \
    --q-scaling "$q_scaling" \
    --omega-data 1.0 \
    --omega-latent "$omega_latent" \
    --omega-recon "$omega_recon" \
    --pretrain-epochs "$pretrain" \
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

# POD-DL-ROM sweep: q_N in R^151 -> encoder z, and (mu,t) -> dynamics z -> decoder q_N.
# The latent dimension is fixed to 10 for all candidates. Winner is selected by
# validation relative Frobenius error of q_N prediction.
run_cfg PDL00_l10_legacy_elu          10 "256,128"     "128,256"     "64,128,128"          "elu"  "zscore" "zscore" 0.10 0.00 0   256 1e-3 5000 220
run_cfg PDL01_l10_silu_balanced       10 "256,128"     "128,256"     "256,512,256"         "silu" "zscore" "zscore" 0.05 0.01 250 128 5e-4 6000 260
run_cfg PDL02_l10_elu_balanced        10 "256,128"     "128,256"     "256,512,256"         "elu"  "zscore" "zscore" 0.05 0.01 250 128 5e-4 6000 260
run_cfg PDL03_l10_silu_wide           10 "512,256"     "256,512"     "256,512,512,256"     "silu" "zscore" "zscore" 0.05 0.01 300 128 5e-4 7000 320
run_cfg PDL04_l10_elu_wide            10 "512,256"     "256,512"     "256,512,512,256"     "elu"  "zscore" "zscore" 0.05 0.01 300 128 5e-4 7000 320
run_cfg PDL05_l10_gelu_wide           10 "512,256"     "256,512"     "256,512,512,256"     "gelu" "zscore" "zscore" 0.05 0.01 300 128 5e-4 7000 320
run_cfg PDL06_l10_silu_paper_loss     10 "512,256"     "256,512"     "256,512,512,256"     "silu" "zscore" "zscore" 0.10 0.00 0   128 5e-4 7000 320
run_cfg PDL07_l10_silu_lowlatreg      10 "512,256"     "256,512"     "256,512,512,256"     "silu" "zscore" "zscore" 0.03 0.01 300 128 5e-4 7000 320

python3 - <<'PY'
from pathlib import Path
import csv
import os
import shutil

stage3 = Path(os.environ["STAGE3_DIR"])
models = Path(os.environ["MODELS_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
rows = []

for summary in sorted(stage3.glob("pod_dl_data_driven_ntot151_PDL*_summary.txt")):
    data = {}
    for line in summary.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    label = summary.name.replace("pod_dl_data_driven_ntot151_", "").replace("_summary.txt", "")
    rows.append(
        {
            "label": label,
            "val_rel_frob_percent": data.get("val_rel_frob_percent", ""),
            "train_rel_frob_percent": data.get("train_rel_frob_percent", ""),
            "best_val_total": data.get("best_val_total", ""),
            "epochs_ran": data.get("epochs_ran", ""),
            "latent_dim": data.get("latent_dim", ""),
            "encoder_hidden_dims": data.get("encoder_hidden_dims", ""),
            "decoder_hidden_dims": data.get("decoder_hidden_dims", ""),
            "dynamics_hidden_dims": data.get("dynamics_hidden_dims", ""),
            "activation": data.get("activation", ""),
            "x_scaling": data.get("x_scaling", ""),
            "q_scaling": data.get("q_scaling", ""),
            "omega_latent": data.get("omega_latent", ""),
            "omega_recon": data.get("omega_recon", ""),
            "pretrain_epochs": data.get("pretrain_epochs", ""),
            "batch_size": data.get("batch_size", ""),
            "lr": data.get("lr", ""),
            "trainable_parameters": data.get("trainable_parameters", ""),
            "candidate_model": str(models / f"pod_dl_data_driven_ntot151_{label}.pt"),
        }
    )

if not rows:
    raise SystemExit("No POD-DL sweep summaries found.")

def score(row):
    try:
        return float(row["val_rel_frob_percent"])
    except Exception:
        return float("inf")

rows.sort(key=score)
winner = rows[0]
winner_label = winner["label"]

csv_path = log_dir / "pod_dl_arch_sweep_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

canonical_model = models / "pod_dl_data_driven_ntot151_best.pt"
canonical_summary = stage3 / "pod_dl_data_driven_ntot151_best_summary.txt"
canonical_log = log_dir / "pod_dl_data_driven_ntot151_best.log"

winner_model = models / f"pod_dl_data_driven_ntot151_{winner_label}.pt"
winner_summary = stage3 / f"pod_dl_data_driven_ntot151_{winner_label}_summary.txt"
winner_log = log_dir / f"{winner_label}.log"
shutil.move(winner_model, canonical_model)
shutil.move(winner_summary, canonical_summary)
shutil.move(winner_log, canonical_log)

text = canonical_summary.read_text()
text = text.replace(
    f"model_name: pod_dl_data_driven_ntot151_{winner_label}.pt",
    "model_name: pod_dl_data_driven_ntot151_best.pt",
)
for line in text.splitlines():
    if line.startswith("model_path:"):
        text = text.replace(line, f"model_path: {canonical_model}")
        break
text += f"\nsweep_winner_label: {winner_label}\n"
canonical_summary.write_text(text)

for path in models.glob("pod_dl_data_driven_ntot151_PDL*.pt"):
    path.unlink()
for path in stage3.glob("pod_dl_data_driven_ntot151_PDL*_summary.txt"):
    path.unlink()
for path in log_dir.glob("PDL*.log"):
    path.unlink()

print("[pod-dl-sweep-summary]", csv_path)
for row in rows:
    print(
        " ",
        row["label"],
        "val_rel_frob_percent=",
        row["val_rel_frob_percent"],
        "best_val_total=",
        row["best_val_total"],
    )
print("[pod-dl-sweep-winner]", winner_label)
print("[pod-dl-sweep-model]", canonical_model)
print("[pod-dl-sweep-summary-file]", canonical_summary)
PY

trap - EXIT
echo "[done] Only the selected POD-DL checkpoint is retained:"
echo "       $MODELS_DIR/pod_dl_data_driven_ntot151_best.pt"
