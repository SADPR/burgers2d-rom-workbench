#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export BASE_TAG="${BASE_TAG:-mlspg_hprom_main}"
export EXT_TAG="${EXT_TAG:-mlspg_hprom_enrichment_ext25_lhs36}"
export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
export BASE_ROOT="$PAPER_RESULTS_ROOT/$BASE_TAG"
export EXT_ROOT="$PAPER_RESULTS_ROOT/$EXT_TAG"
export BASE_DATASET="$BASE_ROOT/Stage2/prom_coeff_dataset_ntot151"
export DATASET_DIR="${DATASET_DIR:-$EXT_ROOT/Stage2/prom_coeff_dataset_ntot151_enriched_lhs36}"
export BASIS="$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/basis.npy"
export UREF="$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
export LINEAR_ECSW="$BASE_ROOT/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy"
export LOG_DIR="$EXT_ROOT/logs"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EXT_ROOT/.mplcache}"

ROM_NUM_THREADS="${ROM_NUM_THREADS:-24}"
export BLIS_NUM_THREADS="$ROM_NUM_THREADS"
export GOTO_NUM_THREADS="$ROM_NUM_THREADS"
export MKL_NUM_THREADS="$ROM_NUM_THREADS"
export OMP_NUM_THREADS="$ROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$ROM_NUM_THREADS"

INTERIOR_SAMPLES="${INTERIOR_SAMPLES:-18}"
EXTERIOR_SAMPLES="${EXTERIOR_SAMPLES:-18}"
LHS_SEED="${LHS_SEED:-42}"
MARGIN_FRACTION="${MARGIN_FRACTION:-0.25}"
PLAN_ONLY="${PLAN_ONLY:-0}"
REGENERATE_DESIGN="${REGENERATE_DESIGN:-0}"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

for required in "$BASE_DATASET/meta.npy" "$BASIS" "$UREF" "$LINEAR_ECSW"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required baseline artifact: $required" >&2
    exit 1
  fi
done

if [[ "$INTERIOR_SAMPLES" -ne 18 || "$EXTERIOR_SAMPLES" -ne 18 ]]; then
  echo "[warning] This paper script is calibrated for 18 interior + 18 exterior points." >&2
fi

echo "[extended-enrichment-stage2] Output root: $EXT_ROOT"
echo "[extended-enrichment-stage2] Dataset:     $DATASET_DIR"
echo "[extended-enrichment-stage2] Reusing the exact baseline linear ECSW file:"
echo "  $LINEAR_ECSW"
echo "[extended-enrichment-stage2] No ECSW rule will be built or copied."
echo "[extended-enrichment-stage2] Design: $INTERIOR_SAMPLES interior + $EXTERIOR_SAMPLES exterior LHS, margin_fraction=$MARGIN_FRACTION, seed=$LHS_SEED"

args=(
  --base-dataset-dir "$BASE_DATASET"
  --output-dir "$DATASET_DIR"
  --basis-path "$BASIS"
  --u-ref-path "$UREF"
  --ecsw-weights-path "$LINEAR_ECSW"
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

if [[ "$PLAN_ONLY" == "1" ]]; then
  args+=(--plan-only)
fi
if [[ "$REGENERATE_DESIGN" == "1" ]]; then
  args+=(--regenerate-design)
fi

log_name="stage2_ext25_lhs36.log"
if [[ "$PLAN_ONLY" == "1" ]]; then
  log_name="stage2_ext25_lhs36_plan.log"
fi

python3 -u stage2_build_paper_extended_enrichment_qn_dataset.py "${args[@]}" \
  2>&1 | tee "$LOG_DIR/$log_name"

if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "[extended-enrichment-stage2] PLAN_ONLY=1, skipping qN completeness check."
  exit 0
fi

python3 - <<'PY' | tee "$LOG_DIR/stage2_ext25_lhs36_check.txt"
import hashlib
import json
import os
from pathlib import Path

import numpy as np

dataset = Path(os.environ["DATASET_DIR"]).resolve()
expected_ecsw = Path(os.environ["LINEAR_ECSW"]).resolve()
meta = json.loads((dataset / "meta.json").read_text())

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected_total = 45
actual_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())
if len(actual_dirs) != expected_total:
    raise SystemExit(f"Expected {expected_total} trajectories, found {len(actual_dirs)}.")
if Path(meta["ecsw_weights_path"]).resolve() != expected_ecsw:
    raise SystemExit("Extended enrichment metadata does not reference the exact baseline ECSW file.")
if meta.get("ecsw_weights_copied") is not False or meta.get("ecsw_weights_rebuilt") is not False:
    raise SystemExit("Invalid ECSW provenance flags in extended enrichment metadata.")
if meta["ecsw_weights_sha256"] != sha256(expected_ecsw):
    raise SystemExit("ECSW SHA-256 mismatch.")
if meta.get("num_base_traj_copied") != 9:
    raise SystemExit("Expected 9 copied baseline trajectories.")
if meta.get("num_interior_lhs_traj") != 18 or meta.get("num_exterior_lhs_traj") != 18:
    raise SystemExit("Expected 18 interior and 18 exterior LHS trajectories.")
if meta.get("num_lhs_traj") != 36 or meta.get("num_traj") != expected_total:
    raise SystemExit("Unexpected trajectory counts in metadata.")
if abs(float(meta.get("margin_fraction")) - 0.25) > 1e-14:
    raise SystemExit("Expected margin_fraction=0.25.")
for mu_dir in actual_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid qN in {mu_dir}: shape={qn.shape}.")

print("[check] trajectories:", len(actual_dirs))
print("[check] baseline trajectories:", meta["num_base_traj_copied"])
print("[check] interior LHS trajectories:", meta["num_interior_lhs_traj"])
print("[check] exterior LHS trajectories:", meta["num_exterior_lhs_traj"])
print("[check] direct solver qN:", meta["coefficient_storage"])
print("[check] fixed linear ECSW:", expected_ecsw)
print("[check] fixed linear ECSW SHA-256:", meta["ecsw_weights_sha256"])
print("[check] ECSW copied:", meta["ecsw_weights_copied"])
print("[check] ECSW rebuilt:", meta["ecsw_weights_rebuilt"])
PY
