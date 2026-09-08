#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
export LOG_DIR="$PAPER_ROOT/logs/case2_arch_sweep"
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
print('[case2-sweep-check] dataset_dir:', p)
print('[case2-sweep-check] metadata:', meta_path)
print('[case2-sweep-check] solve_backend:', meta.get('solve_backend'))
print('[case2-sweep-check] total_modes:', meta.get('total_modes'))
print('[case2-sweep-check] basis_path:', meta.get('basis_path'))
if str(meta.get('solve_backend')).lower() != 'hprom':
    raise SystemExit('Case-2 architecture sweep requires the HPROM dataset.')
if int(meta.get('total_modes')) != 151:
    raise SystemExit('Expected total_modes=151.')
PY

run_cfg() {
  local label="$1"
  local primary="$2"
  local hidden="$3"
  local activation="$4"
  local batch="$5"
  local lr="$6"
  local wd="$7"
  local dropout="$8"
  local epochs="$9"
  local patience="${10}"

  local full_label="np${primary}_${label}"
  echo "==== Case-2 arch sweep: ${full_label}"
  echo "primary=${primary} hidden=${hidden} activation=${activation} batch=${batch} lr=${lr} wd=${wd} dropout=${dropout} epochs=${epochs} patience=${patience}"

  python3 -u stage3_perform_training_case_2_ann_test_n20_maday.py \
    --maday-results-root "$PAPER_RESULTS_ROOT" \
    --maday-tag "$PAPER_TAG" \
    --dataset-backend hprom \
    --dataset-ntot 151 \
    --dataset-dir "$DATASET_DIR" \
    --primary-modes "$primary" \
    --model-name "case2_ann_ntot151_${full_label}.pt" \
    --summary-name "case2_ann_ntot151_${full_label}_summary.txt" \
    --val-split-mode row \
    --val-frac 0.1 \
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
    2>&1 | tee "$LOG_DIR/${full_label}.log"
}

# Case-2 map: (mu1, mu2, t) -> q_{n+1:151}, with n in {10,20}.
# The sweep is intentionally compact: it tests the old baseline, the A10-style winner,
# activation effects, and one deeper/wider alternative without creating a huge campaign.
for primary in 10 20; do
  run_cfg B00_current_b128_lr1e3       "$primary" "32,64,128,256,256"        elu  128 1e-3 1e-6 0.00 5000 160
  run_cfg B01_A10_like_b128_lr5e4     "$primary" "256,512,512,256"          silu 128 5e-4 1e-6 0.00 6000 220
  run_cfg B02_A10_elu_b128_lr5e4      "$primary" "256,512,512,256"          elu  128 5e-4 1e-6 0.00 6000 220
  run_cfg B03_medium_silu_b128_lr1e3  "$primary" "128,256,256,128"          silu 128 1e-3 1e-6 0.00 5000 180
  run_cfg B04_deep_silu_b128_lr5e4    "$primary" "128,256,512,512,256,128"  silu 128 5e-4 1e-6 0.00 6000 240
done

python3 - <<'PY'
from pathlib import Path
import csv, os, shutil
root = Path(os.environ['PAPER_ROOT']) / 'Stage3'
out = Path(os.environ['LOG_DIR']) / 'case2_arch_sweep_summary.csv'
rows = []
for p in sorted(root.glob('case2_ann_ntot151_np*_summary.txt')):
    d = {}
    for line in p.read_text(errors='replace').splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        d[k.strip()] = v.strip()
    label = p.name.replace('case2_ann_ntot151_', '').replace('_summary.txt', '')
    row = {
        'label': label,
        'primary_modes': d.get('primary_modes', ''),
        'secondary_modes': d.get('secondary_modes', ''),
        'n_s': d.get('n_s', ''),
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
rows.sort(key=lambda r: (int(r['primary_modes']) if str(r['primary_modes']).isdigit() else 999, score(r)))
out.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0].keys()) if rows else ['label']
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print('[case2-sweep-summary]', out)
for primary in ('10', '20'):
    subset = [r for r in rows if str(r['primary_modes']) == primary]
    if not subset:
        continue
    print(f'[case2-sweep-summary] best primary={primary}')
    for r in subset[:3]:
        print(' ', r['label'], 'val_rel_frob_percent=', r['val_rel_frob_percent'], 'best_val_mse=', r['best_val_mse'])
    winner = subset[0]
    source_model = root / 'models' / f"case2_ann_ntot151_{winner['label']}.pt"
    target_model = root / 'models' / f'case2_ann_ntot151_np{primary}_best.pt'
    source_summary = root / f"case2_ann_ntot151_{winner['label']}_summary.txt"
    target_summary = root / f'case2_ann_ntot151_np{primary}_best_summary.txt'
    shutil.copy2(source_model, target_model)
    text = source_summary.read_text(errors='replace')
    text = text.replace(source_model.name, target_model.name)
    text = text.rstrip() + f"\n\nsweep_winner_label: {winner['label'].split('_', 1)[1]}\n"
    target_summary.write_text(text)
    print(f'[case2-sweep-winner] primary={primary} label={winner["label"]}')
    print(f'[case2-sweep-model] {target_model}')
PY
