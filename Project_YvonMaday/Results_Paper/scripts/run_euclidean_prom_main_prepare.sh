#!/usr/bin/env bash
# Prepare the reproducible Euclidean-POD, full-residual PROM campaign.
# This script deliberately never touches MetricStudy/euclidean/Stage1 or mlspg_*.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

STAGE="${1:-all}"
case "$STAGE" in
  all|check|clean|train9|val2) ;;
  *)
    echo "Usage: $0 [all|check|clean|train9|val2]" >&2
    exit 2
    ;;
esac

PAPER_ROOT="$PROJECT_DIR/Results_Paper"
PROM_ROOT="${EUCLIDEAN_PROM_ROOT:-$PAPER_ROOT/euclidean_prom_main}"
METRIC_ROOT="$PAPER_ROOT/MetricStudy/euclidean/Stage1"
BASIS="$METRIC_ROOT/basis.npy"
UREF="$METRIC_ROOT/u_ref.npy"
TRAIN_DATASET="$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151"
VAL_DATASET="$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151_validation2"
LOG_ROOT="$PROM_ROOT/logs/prepare"
MANIFEST="$PROM_ROOT/campaign_manifest.txt"

FORCE="${FORCE:-0}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
ALLOW_CLEAN="${ALLOW_CLEAN:-0}"

TRAIN_MU=(
  "4.25 0.015"
  "4.25 0.0225"
  "4.25 0.03"
  "4.875 0.015"
  "4.875 0.0225"
  "4.875 0.03"
  "5.5 0.015"
  "5.5 0.0225"
  "5.5 0.03"
)
VAL_MU=(
  "4.5625 0.02625"
  "5.1875 0.01875"
)

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] Missing required file: $path" >&2
    exit 1
  fi
}

metric_check() {
  require_file "$BASIS"
  require_file "$UREF"
  python3 - "$BASIS" "$UREF" <<'PY'
from pathlib import Path
import sys
import numpy as np

basis_path = Path(sys.argv[1])
uref_path = Path(sys.argv[2])
basis = np.load(basis_path, mmap_mode="r", allow_pickle=False)
uref = np.load(uref_path, mmap_mode="r", allow_pickle=False)
if basis.shape != (125000, 151):
    raise SystemExit(f"Expected Euclidean basis shape (125000, 151), got {basis.shape}.")
if uref.shape != (125000,):
    raise SystemExit(f"Expected Euclidean reference shape (125000,), got {uref.shape}.")
if not np.isfinite(basis[:, :1]).all() or not np.isfinite(uref).all():
    raise SystemExit("Euclidean basis or reference contains non-finite values.")
print(f"[euclidean-prom-prepare] basis: {basis_path}")
print(f"[euclidean-prom-prepare] basis shape: {basis.shape}")
print(f"[euclidean-prom-prepare] u_ref: {uref_path}")
PY
}

write_manifest() {
  mkdir -p "$PROM_ROOT"
  cat > "$MANIFEST" <<EOF
campaign: Euclidean POD full-residual PROM
metric: Euclidean POD
basis_path: $BASIS
u_ref_path: $UREF
total_modes: 151
basis_construction: direct SVD; centered snapshots; energy_lost <= 1e-4
solver: full-residual LSPG/PROM
linear_solver: lstsq
normal_eq_reg: 1e-12
max_its: 20
relnorm_cutoff: 1e-5
min_delta: 1e-2
training_mu: ${TRAIN_MU[*]}
parameter_validation_mu: ${VAL_MU[*]}
evaluation_mu: (4.875,0.0225), (4.560,0.0190), (5.190,0.0260), (4.000,0.0330)
random_seed: 42
EOF
}

clean_campaign() {
  if [[ "$ALLOW_CLEAN" != "1" ]]; then
    echo "[error] Refusing cleanup without ALLOW_CLEAN=1." >&2
    exit 2
  fi
  if [[ "$PROM_ROOT" != "$PAPER_ROOT/euclidean_prom_main" ]]; then
    echo "[error] Refusing cleanup outside the default euclidean_prom_main root: $PROM_ROOT" >&2
    exit 2
  fi
  echo "[clean] Removing only generated Euclidean PROM campaign outputs under: $PROM_ROOT"
  rm -rf "$PROM_ROOT/Stage2" "$PROM_ROOT/Stage3" "$PROM_ROOT/Runs" \
    "$PROM_ROOT/logs" "$PROM_ROOT/.mplcache" "$MANIFEST"
}

dataset_complete() {
  local dataset="$1"
  local expected_trajectories="$2"
  [[ -f "$dataset/stage2_summary.txt" ]] || return 1
  python3 - "$dataset" "$expected_trajectories" <<'PY'
from pathlib import Path
import sys
import numpy as np

from stage3_dataset_utils import read_dataset_meta

dataset = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
meta, _ = read_dataset_meta(dataset)
if str(meta.get("solve_backend", "")).lower() != "prom":
    raise SystemExit("Expected solve_backend=prom.")
if int(meta.get("total_modes", -1)) != 151:
    raise SystemExit("Expected total_modes=151.")
mu_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())
if len(mu_dirs) != expected:
    raise SystemExit(f"Expected {expected} parameter trajectories, found {len(mu_dirs)}.")
for mu_dir in mu_dirs:
    qn = np.load(mu_dir / "qN.npy", mmap_mode="r", allow_pickle=False)
    if qn.shape != (151, 501):
        raise SystemExit(f"Invalid qN shape in {mu_dir}: {qn.shape}.")
print(f"[euclidean-prom-prepare] complete dataset: {dataset} ({len(mu_dirs)} trajectories)")
PY
}

build_train9() {
  if [[ "$FORCE" != "1" ]] && dataset_complete "$TRAIN_DATASET" 9; then
    echo "[skip] Euclidean training dataset is already complete: $TRAIN_DATASET"
    return
  fi
  echo "[run] Building the nine Euclidean linear-PROM training trajectories."
  set_threads "$PROM_NUM_THREADS"
  python3 -u stage2_build_prom_qn_dataset.py \
    --backend prom --total-modes 151 \
    --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-dir "$TRAIN_DATASET" \
    --linear-solver lstsq --normal-eq-reg 1e-12 \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --no-save-rom-snaps --no-plots \
    2>&1 | tee "$LOG_ROOT/stage2_train9.log"
  dataset_complete "$TRAIN_DATASET" 9
}

build_val2() {
  if [[ "$FORCE" != "1" ]] && dataset_complete "$VAL_DATASET" 2; then
    echo "[skip] Euclidean external-validation dataset is already complete: $VAL_DATASET"
    return
  fi
  echo "[run] Building the two Euclidean external parameter-validation trajectories."
  set_threads "$PROM_NUM_THREADS"
  python3 -u stage2_build_prom_qn_dataset.py \
    --backend prom --total-modes 151 \
    --basis-path "$BASIS" --u-ref-path "$UREF" \
    --output-dir "$VAL_DATASET" \
    --mu-pair 4.5625 0.02625 \
    --mu-pair 5.1875 0.01875 \
    --linear-solver lstsq --normal-eq-reg 1e-12 \
    --max-its 20 --relnorm-cutoff 1e-5 --min-delta 1e-2 \
    --no-save-rom-snaps --no-plots \
    2>&1 | tee "$LOG_ROOT/stage2_val2.log"
  dataset_complete "$VAL_DATASET" 2
}

mkdir -p "$PROM_ROOT" "$LOG_ROOT" "$PROM_ROOT/.mplcache"
export MPLCONFIGDIR="$PROM_ROOT/.mplcache"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[euclidean-prom-prepare] stage:       $STAGE"
echo "[euclidean-prom-prepare] campaign:    $PROM_ROOT"
echo "[euclidean-prom-prepare] metric root: $METRIC_ROOT"
echo "[euclidean-prom-prepare] threads:     $PROM_NUM_THREADS"
echo "[euclidean-prom-prepare] force:       $FORCE"

case "$STAGE" in
  check)
    metric_check
    write_manifest
    ;;
  clean)
    clean_campaign
    ;;
  train9)
    metric_check
    write_manifest
    build_train9
    ;;
  val2)
    metric_check
    write_manifest
    build_val2
    ;;
  all)
    metric_check
    write_manifest
    build_train9
    build_val2
    ;;
esac

echo "[euclidean-prom-prepare] done."
