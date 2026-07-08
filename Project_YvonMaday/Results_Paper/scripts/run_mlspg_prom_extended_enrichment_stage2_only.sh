#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
export PROM_ROOT="${PROM_ROOT:-$PAPER_RESULTS_ROOT/mlspg_prom_main}"
export EXT_ROOT="${EXT_ROOT:-$PAPER_RESULTS_ROOT/mlspg_prom_enrichment_ext25_lhs36}"
export BASE_DATASET="${BASE_DATASET:-$PROM_ROOT/Stage2/prom_coeff_dataset_ntot151}"
export DATASET_DIR="${DATASET_DIR:-$EXT_ROOT/Stage2/prom_coeff_dataset_ntot151_enriched_lhs36}"
export BASIS="${BASIS:-$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
export UREF="${UREF:-$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
export LOG_DIR="${LOG_DIR:-$EXT_ROOT/logs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EXT_ROOT/.mplcache}"

PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
export BLIS_NUM_THREADS="$PROM_NUM_THREADS"
export GOTO_NUM_THREADS="$PROM_NUM_THREADS"
export MKL_NUM_THREADS="$PROM_NUM_THREADS"
export OMP_NUM_THREADS="$PROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$PROM_NUM_THREADS"

INTERIOR_SAMPLES="${INTERIOR_SAMPLES:-18}"
EXTERIOR_SAMPLES="${EXTERIOR_SAMPLES:-18}"
LHS_SEED="${LHS_SEED:-42}"
MARGIN_FRACTION="${MARGIN_FRACTION:-0.25}"
PLAN_ONLY="${PLAN_ONLY:-0}"
REGENERATE_DESIGN="${REGENERATE_DESIGN:-0}"

# If the old HPROM design exists, reuse it exactly. Otherwise the Python
# builder regenerates the same deterministic design from the seed.
DEFAULT_DESIGN_SOURCE="$PAPER_RESULTS_ROOT/mlspg_hprom_enrichment_ext25_lhs36/Stage2/prom_coeff_dataset_ntot151_enriched_lhs36"
DESIGN_SOURCE_DIR="${DESIGN_SOURCE_DIR:-}"
if [[ -z "$DESIGN_SOURCE_DIR" && -f "$DEFAULT_DESIGN_SOURCE/lhs_mu.npy" ]]; then
  DESIGN_SOURCE_DIR="$DEFAULT_DESIGN_SOURCE"
fi

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

for required in "$BASE_DATASET/meta.json" "$BASIS" "$UREF"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required artifact: $required" >&2
    exit 1
  fi
done

echo "[prom-extended-enrichment-stage2] base dataset: $BASE_DATASET"
echo "[prom-extended-enrichment-stage2] output:       $DATASET_DIR"
echo "[prom-extended-enrichment-stage2] design source:${DESIGN_SOURCE_DIR:- generated from seed}"
echo "[prom-extended-enrichment-stage2] design:       ${INTERIOR_SAMPLES} interior + ${EXTERIOR_SAMPLES} exterior, margin=${MARGIN_FRACTION}, seed=${LHS_SEED}"
echo "[prom-extended-enrichment-stage2] threads:      $PROM_NUM_THREADS"
echo "[prom-extended-enrichment-stage2] plan only:    $PLAN_ONLY"

args=(
  --base-dataset-dir "$BASE_DATASET"
  --output-dir "$DATASET_DIR"
  --basis-path "$BASIS"
  --u-ref-path "$UREF"
  --total-modes 151
  --interior-samples "$INTERIOR_SAMPLES"
  --exterior-samples "$EXTERIOR_SAMPLES"
  --lhs-seed "$LHS_SEED"
  --margin-fraction "$MARGIN_FRACTION"
  --exclude-mu 4.875 0.0225
  --exclude-mu 4.560 0.0190
  --exclude-mu 5.190 0.0260
  --exclude-mu 4.000 0.0330
  --linear-solver lstsq
  --normal-eq-reg 1e-12
  --max-its 20
  --relnorm-cutoff 1e-5
  --min-delta 1e-2
)

if [[ -n "$DESIGN_SOURCE_DIR" ]]; then
  args+=(--design-source-dir "$DESIGN_SOURCE_DIR")
fi
if [[ "$PLAN_ONLY" == "1" ]]; then
  args+=(--plan-only)
fi
if [[ "$REGENERATE_DESIGN" == "1" ]]; then
  args+=(--regenerate-design)
fi

log_name="stage2_prom_ext25_lhs36.log"
if [[ "$PLAN_ONLY" == "1" ]]; then
  log_name="stage2_prom_ext25_lhs36_plan.log"
fi

python3 -u stage2_build_prom_extended_enrichment_qn_dataset.py "${args[@]}" \
  2>&1 | tee "$LOG_DIR/$log_name"

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[prom-extended-enrichment-stage2] PLAN_ONLY=1, skipping qN completeness check."
  exit 0
fi

python3 - <<'PY' | tee "$LOG_DIR/stage2_prom_ext25_lhs36_check.txt"
import json
from pathlib import Path
import os

import numpy as np

dataset = Path(os.environ["DATASET_DIR"]).resolve()
meta = json.loads((dataset / "meta.json").read_text())
mu_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())

if len(mu_dirs) != 45:
    raise SystemExit(f"Expected 45 trajectories, found {len(mu_dirs)}.")
if meta.get("solve_backend") != "prom":
    raise SystemExit(f"Expected solve_backend=prom, found {meta.get('solve_backend')}.")
if meta.get("num_base_traj_copied") != 9:
    raise SystemExit("Expected 9 copied baseline trajectories.")
if meta.get("num_interior_lhs_traj") != 18 or meta.get("num_exterior_lhs_traj") != 18:
    raise SystemExit("Expected 18 interior and 18 exterior LHS trajectories.")
if meta.get("num_lhs_traj") != 36 or meta.get("num_traj") != 45:
    raise SystemExit("Unexpected trajectory counts in metadata.")

for mu_dir in mu_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid qN in {mu_dir}: shape={qn.shape}")

print("[check] dataset:", dataset)
print("[check] trajectories:", len(mu_dirs))
print("[check] baseline trajectories:", meta["num_base_traj_copied"])
print("[check] interior LHS trajectories:", meta["num_interior_lhs_traj"])
print("[check] exterior LHS trajectories:", meta["num_exterior_lhs_traj"])
print("[check] direct qN:", meta["coefficient_storage"])
print("[check] reused design source:", meta["reused_design_source"])
print("[check] design source:", meta["design_source_dir"])
PY
