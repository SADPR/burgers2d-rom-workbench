#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
export LOG_DIR="$PAPER_ROOT/logs"
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
print('[stage3-check] dataset_dir:', p)
print('[stage3-check] metadata:', meta_path)
print('[stage3-check] solve_backend:', meta.get('solve_backend'))
print('[stage3-check] total_modes:', meta.get('total_modes'))
print('[stage3-check] basis_path:', meta.get('basis_path'))
print('[stage3-check] ecsw_weights_path:', meta.get('ecsw_weights_path'))
if str(meta.get('solve_backend')).lower() != 'hprom':
    raise SystemExit('Stage-3 training requires an HPROM dataset here.')
if int(meta.get('total_modes')) != 151:
    raise SystemExit('Expected total_modes=151.')
PY

# Case 1 is selected separately by:
#   Results_Paper/scripts/run_mlspg_hprom_case1_arch_sweep.sh
# This avoids overwriting the retained sweep winner.

python3 -u stage3_perform_training_case_2_ann_test_n20_maday.py \
  --maday-results-root "$PAPER_RESULTS_ROOT" \
  --maday-tag "$PAPER_TAG" \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --dataset-dir "$DATASET_DIR" \
  --primary-modes 10 \
  --model-name case2_hprom_ann_mu_t_n10_ntot151.pt \
  --summary-name case2_hprom_ann_mu_t_n10_ntot151_summary.txt \
  --hidden-dims 32,64,128,256,256 \
  --activation elu \
  --seed 42 \
  2>&1 | tee "$LOG_DIR/train_case2_hprom_ann_mu_t_n10_ntot151.log"

# Case 3 is selected separately by:
#   Results_Paper/scripts/run_mlspg_hprom_case3_arch_sweep.sh
# This avoids overwriting the retained sweep winner.

python3 -u stage3_perform_training_rom_data_driven_maday.py \
  --maday-results-root "$PAPER_RESULTS_ROOT" \
  --maday-tag "$PAPER_TAG" \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --dataset-dir "$DATASET_DIR" \
  --model-name rom_data_driven_ann_mu_t_ntot151.pt \
  2>&1 | tee "$LOG_DIR/train_data_driven_hprom_ann_ntot151.log"

for f in "$PAPER_ROOT"/Stage3/*_summary.txt; do
  echo "==== $(basename "$f")"
  grep -E "dataset_backend|dataset_dir|primary_modes|secondary_modes|n_s|n_tot|best_val_mse|epochs_ran|architecture|optimizer|lr|trainable_parameters" "$f" || true
done | tee "$LOG_DIR/stage3_ann_training_quick_summary.txt"
