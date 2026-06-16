#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export BASE_TAG="mlspg_hprom_main"
export ENRICHMENT_TAG="mlspg_hprom_enrichment"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export BASE_ROOT="$PAPER_RESULTS_ROOT/$BASE_TAG"
export ENRICHMENT_ROOT="$PAPER_RESULTS_ROOT/$ENRICHMENT_TAG"
export BASE_DATASET="$BASE_ROOT/Stage2/prom_coeff_dataset_ntot151"
export DATASET_DIR="$ENRICHMENT_ROOT/Stage2/prom_coeff_dataset_ntot151_enriched_lhs20"
export BASIS="$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/basis.npy"
export UREF="$PAPER_RESULTS_ROOT/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
export LINEAR_ECSW="$BASE_ROOT/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy"
export LOG_DIR="$ENRICHMENT_ROOT/logs"
export MPLCONFIGDIR="$ENRICHMENT_ROOT/.mplcache"

ROM_NUM_THREADS="${ROM_NUM_THREADS:-24}"
export BLIS_NUM_THREADS="$ROM_NUM_THREADS"
export GOTO_NUM_THREADS="$ROM_NUM_THREADS"
export MKL_NUM_THREADS="$ROM_NUM_THREADS"
export OMP_NUM_THREADS="$ROM_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$ROM_NUM_THREADS"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"

for required in "$BASE_DATASET/meta.npy" "$BASIS" "$UREF" "$LINEAR_ECSW"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required baseline artifact: $required" >&2
    exit 1
  fi
done

echo "[enrichment-stage2] Reusing the exact baseline linear ECSW file:"
echo "  $LINEAR_ECSW"
echo "[enrichment-stage2] No ECSW rule will be built or copied."

python3 -u stage2_build_paper_enrichment_lhs_qn_dataset.py \
  --base-dataset-dir "$BASE_DATASET" \
  --output-dir "$DATASET_DIR" \
  --basis-path "$BASIS" \
  --u-ref-path "$UREF" \
  --ecsw-weights-path "$LINEAR_ECSW" \
  --total-modes 151 \
  --lhs-samples 20 \
  --lhs-seed 42 \
  --exclude-mu 4.875 0.0225 \
  --exclude-mu 4.560 0.0190 \
  --exclude-mu 5.190 0.0260 \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  2>&1 | tee "$LOG_DIR/stage2_enrichment_lhs20.log"

python3 - <<'PY' | tee "$LOG_DIR/stage2_enrichment_lhs20_check.txt"
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

actual_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())
if len(actual_dirs) != 29:
    raise SystemExit(f"Expected 29 trajectories, found {len(actual_dirs)}.")
if Path(meta["ecsw_weights_path"]).resolve() != expected_ecsw:
    raise SystemExit("Enrichment metadata does not reference the exact baseline ECSW file.")
if meta.get("ecsw_weights_copied") is not False or meta.get("ecsw_weights_rebuilt") is not False:
    raise SystemExit("Invalid ECSW provenance flags in enrichment metadata.")
if meta["ecsw_weights_sha256"] != sha256(expected_ecsw):
    raise SystemExit("ECSW SHA-256 mismatch.")
for mu_dir in actual_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid qN in {mu_dir}: shape={qn.shape}.")

print("[check] trajectories:", len(actual_dirs))
print("[check] baseline trajectories:", meta["num_base_traj_copied"])
print("[check] LHS trajectories:", meta["num_lhs_traj"])
print("[check] direct solver qN:", meta["coefficient_storage"])
print("[check] fixed linear ECSW:", expected_ecsw)
print("[check] fixed linear ECSW SHA-256:", meta["ecsw_weights_sha256"])
print("[check] ECSW copied:", meta["ecsw_weights_copied"])
print("[check] ECSW rebuilt:", meta["ecsw_weights_rebuilt"])
PY
