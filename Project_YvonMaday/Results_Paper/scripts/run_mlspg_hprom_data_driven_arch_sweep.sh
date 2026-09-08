#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
export LOG_DIR="$PAPER_ROOT/logs/data_driven_arch_sweep"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

if [[ ! -f "$DATASET_DIR/meta.npy" && ! -f "$DATASET_DIR/meta.json" ]]; then
  echo "Missing Stage-2 dataset metadata in: $DATASET_DIR" >&2
  exit 1
fi

python3 - <<'PY'
import os
from stage3_dataset_utils import read_dataset_meta
p = os.environ['DATASET_DIR']
meta, meta_path = read_dataset_meta(p)
print('[sweep-check] dataset_dir:', p)
print('[sweep-check] metadata:', meta_path)
print('[sweep-check] solve_backend:', meta.get('solve_backend'))
print('[sweep-check] total_modes:', meta.get('total_modes'))
if str(meta.get('solve_backend')).lower() != 'hprom':
    raise SystemExit('Architecture sweep requires the HPROM dataset.')
if int(meta.get('total_modes')) != 151:
    raise SystemExit('Expected total_modes=151.')
PY

run_cfg() {
  local label="$1"
  local hidden="$2"
  local activation="$3"
  local batch="$4"
  local lr="$5"
  local wd="$6"
  local dropout="$7"
  local epochs="$8"
  local patience="$9"

  echo "==== data-driven arch sweep: ${label}"
  echo "hidden=${hidden} activation=${activation} batch=${batch} lr=${lr} wd=${wd} dropout=${dropout} epochs=${epochs} patience=${patience}"

  python3 -u stage3_perform_training_rom_data_driven_maday.py \
    --maday-results-root "$PAPER_RESULTS_ROOT" \
    --maday-tag "$PAPER_TAG" \
    --dataset-backend hprom \
    --dataset-ntot 151 \
    --dataset-dir "$DATASET_DIR" \
    --model-name "rom_data_driven_ann_mu_t_ntot151_${label}.pt" \
    --summary-name "rom_data_driven_ann_mu_t_ntot151_${label}_summary.txt" \
    --hidden-dims "$hidden" \
    --activation "$activation" \
    --batch-size "$batch" \
    --lr "$lr" \
    --weight-decay "$wd" \
    --dropout "$dropout" \
    --epochs "$epochs" \
    --patience "$patience" \
    --lr-scheduler-factor 0.5 \
    --lr-scheduler-patience 50 \
    --lr-scheduler-min-lr 1e-6 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/${label}.log"
}

# Sweep logic for 3 -> 151 with 4509 samples:
# - include compact networks to test overfitting control;
# - include wider/deeper networks because output dimension is large;
# - include batch 64 for more stochastic updates and batch 128/256 for stability;
# - use lower lr for wide models to avoid unstable validation oscillations.
run_cfg A00_current_b128_lr1e3        "32,64,128,256,256"       elu 128 1e-3 1e-6 0.00 5000 160
run_cfg A01_compact64_b64_lr1e3       "64,64,64,64"             elu  64 1e-3 1e-6 0.00 5000 160
run_cfg A02_compact128_b64_lr1e3      "128,128,128"             elu  64 1e-3 1e-6 0.00 5000 160
run_cfg A03_bottleneck_b64_lr1e3      "64,128,256,128,64"       elu  64 1e-3 1e-6 0.00 5000 180
run_cfg A04_medium_b128_lr1e3         "128,256,256,128"         elu 128 1e-3 1e-6 0.00 5000 180
run_cfg A05_wide_b128_lr5e4           "256,512,512,256"         elu 128 5e-4 1e-6 0.00 6000 220
run_cfg A06_wide_b256_lr5e4           "256,512,512,256"         elu 256 5e-4 1e-6 0.00 6000 220
run_cfg A07_deep_wide_b128_lr5e4      "128,256,512,512,256,128" elu 128 5e-4 1e-6 0.00 6000 240
run_cfg A08_deep_wide_reg_b128_lr5e4  "128,256,512,512,256,128" elu 128 5e-4 1e-5 0.02 6000 240
run_cfg A09_silu_medium_b128_lr1e3    "128,256,256,128"        silu 128 1e-3 1e-6 0.00 5000 180
run_cfg A10_silu_wide_b128_lr5e4      "256,512,512,256"        silu 128 5e-4 1e-6 0.00 6000 220
run_cfg A11_small_b64_lr1e3           "32,64,128"               elu  64 1e-3 1e-6 0.00 5000 160

python3 - <<'PY'
from pathlib import Path
import re, csv, os, shutil
root = Path(os.environ['PAPER_ROOT']) / 'Stage3'
out = Path(os.environ['LOG_DIR']) / 'data_driven_arch_sweep_summary.csv'
rows = []
for p in sorted(root.glob('rom_data_driven_ann_mu_t_ntot151_*_summary.txt')):
    d = {}
    for line in p.read_text().splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        d[k.strip()] = v.strip()
    label = p.name.replace('rom_data_driven_ann_mu_t_ntot151_', '').replace('_summary.txt', '')
    row = {
        'label': label,
        'best_val_mse': d.get('best_val_mse', ''),
        'val_rel_frob_percent': d.get('val_rel_frob_percent', ''),
        'train_rel_frob_percent': d.get('train_rel_frob_percent', ''),
        'epochs_ran': d.get('epochs_ran', ''),
        'hidden_dims': d.get('hidden_dims', ''),
        'activation': d.get('activation', ''),
        'batch_size': d.get('batch_size', ''),
        'lr': d.get('lr', ''),
        'weight_decay': d.get('weight_decay', ''),
        'dropout': d.get('dropout', ''),
        'trainable_parameters': d.get('trainable_parameters', ''),
        'model_path': d.get('model_path', ''),
    }
    rows.append(row)

def score(row):
    try:
        return float(row['val_rel_frob_percent'])
    except Exception:
        try:
            return float(row['best_val_mse'])
        except Exception:
            return 1e99
rows.sort(key=score)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['label'])
    w.writeheader()
    w.writerows(rows)
print('[sweep-summary]', out)
for r in rows[:5]:
    print(r['label'], 'val_rel_frob_percent=', r['val_rel_frob_percent'], 'best_val_mse=', r['best_val_mse'])
if rows:
    winner = rows[0]
    source_model = root / 'models' / f"rom_data_driven_ann_mu_t_ntot151_{winner['label']}.pt"
    target_model = root / 'models' / 'data_driven_ann_ntot151_best.pt'
    source_summary = root / f"rom_data_driven_ann_mu_t_ntot151_{winner['label']}_summary.txt"
    target_summary = root / 'data_driven_ann_ntot151_best_summary.txt'
    shutil.copy2(source_model, target_model)
    text = source_summary.read_text(errors='replace')
    text = text.replace(source_model.name, target_model.name)
    text = text.rstrip() + f"\n\nsweep_winner_label: {winner['label']}\n"
    target_summary.write_text(text)
    print(f'[sweep-winner] {winner["label"]}')
    print(f'[sweep-model] {target_model}')
PY
