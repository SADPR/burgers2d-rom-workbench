#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
PAPER_ROOT="${PAPER_RESULTS_ROOT}/mlspg_hprom_main"
OUT_DIR="${OUT_DIR:-$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2}"
LOG_DIR="${LOG_DIR:-$PAPER_ROOT/logs/stage2_validation}"
BASIS_PATH="${BASIS_PATH:-$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
U_REF_PATH="${U_REF_PATH:-$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
ECSW_WEIGHTS_DIR="${ECSW_WEIGHTS_DIR:-$PAPER_ROOT/Stage2/ecsw}"
ROM_NUM_THREADS="${ROM_NUM_THREADS:-1}"
FORCE="${FORCE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

export BLIS_NUM_THREADS="$ROM_NUM_THREADS"
export GOTO_NUM_THREADS="$ROM_NUM_THREADS"
export MKL_NUM_THREADS="$ROM_NUM_THREADS"
export OMP_NUM_THREADS="$ROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$ROM_NUM_THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

cat <<EOF
[hprom-validation2] output:       $OUT_DIR
[hprom-validation2] basis:        $BASIS_PATH
[hprom-validation2] u_ref:        $U_REF_PATH
[hprom-validation2] ECSW dir:     $ECSW_WEIGHTS_DIR
[hprom-validation2] threads:      $ROM_NUM_THREADS
[hprom-validation2] force:        $FORCE
[hprom-validation2] validation mu:
  - (4.5625, 0.02625)
  - (5.1875, 0.01875)
EOF

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[hprom-validation2] PLAN_ONLY=1; no run was launched."
  exit 0
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$OUT_DIR"
fi

if [[ -f "$OUT_DIR/meta.json" && "$FORCE" != "1" ]]; then
  echo "[skip] Existing validation dataset: $OUT_DIR"
  exit 0
fi

python3 -u stage2_build_prom_qn_dataset.py \
  --backend hprom \
  --total-modes 151 \
  --basis-path "$BASIS_PATH" \
  --u-ref-path "$U_REF_PATH" \
  --output-dir "$OUT_DIR" \
  --ecsw-weights-dir "$ECSW_WEIGHTS_DIR" \
  --no-rebuild-ecsw \
  --mu-pair 4.5625 0.02625 \
  --mu-pair 5.1875 0.01875 \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --no-save-rom-snaps \
  --no-plots \
  2>&1 | tee "$LOG_DIR/stage2_hprom_qn_ntot151_validation2.log"

OUT_DIR_FOR_CHECK="$OUT_DIR" python3 - <<'PY'
from pathlib import Path
import json
import os
import numpy as np

out = Path(os.environ["OUT_DIR_FOR_CHECK"]).expanduser().resolve()
meta = json.loads((out / "meta.json").read_text())
mu_dirs = sorted((out / "per_mu").glob("mu1_*"))
print("[hprom-validation2-check] solve_backend:", meta.get("solve_backend"))
print("[hprom-validation2-check] num_traj:", meta.get("num_traj"))
print("[hprom-validation2-check] rebuild_ecsw_weights:", meta.get("rebuild_ecsw_weights"))
print("[hprom-validation2-check] ecsw_weights_path:", meta.get("ecsw_weights_path"))
print("[hprom-validation2-check] n_ecsw_elements:", meta.get("n_ecsw_elements"))
for mu_dir in mu_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    print("[hprom-validation2-check]", mu_dir.name, qn.shape)
if meta.get("solve_backend") != "hprom" or len(mu_dirs) != 2:
    raise SystemExit("Invalid HPROM validation dataset.")
PY
