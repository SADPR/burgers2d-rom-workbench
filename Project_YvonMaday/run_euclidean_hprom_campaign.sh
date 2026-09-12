#!/usr/bin/env bash
# Resumable end-to-end Euclidean HPROM campaign. Run with bash in an existing allocation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
case "$STAGE" in check|prepare|train|rules|b3|online|timing|summary|all) ;;
  *) echo "Usage: $0 [check|prepare|train|rules|b3|online|timing|summary|all]" >&2; exit 2;; esac

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHON_BIN
PAPER_ROOT="$PWD/Results_Paper/euclidean_hprom_main"
export EUCLIDEAN_HPROM_ROOT="${EUCLIDEAN_HPROM_ROOT:-$PAPER_ROOT}"
THREADS="${HPROM_THREADS:-24}"
export PROM_NUM_THREADS="$THREADS" TRAIN_NUM_THREADS="$THREADS" RULE_THREADS="$THREADS"
export ONLINE_THREADS="$THREADS" CASE2_B3_THREADS="$THREADS" TIMING_THREADS="$THREADS"
export ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
export MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$EUCLIDEAN_HPROM_ROOT/.mplcache}"
mkdir -p "$EUCLIDEAN_HPROM_ROOT/logs" "$MPLCONFIGDIR"

check() {
  if [[ -f euclidean_hprom_upload.sha256 ]]; then
    (cd .. && sha256sum --check Project_YvonMaday/euclidean_hprom_upload.sha256)
  fi
  "$PYTHON_BIN" - <<'PY'
import os, socket, sys
from pathlib import Path
import numpy, scipy, torch
host = socket.gethostname().split('.')[0]
if host.startswith('login'):
    raise SystemExit('Run the numerical campaign on an allocated compute node, not a login node.')
affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
print(f"Python: {sys.executable}")
print(f"Host: {socket.gethostname()} | CPU affinity: {affinity}")
print(f"numpy={numpy.__version__} scipy={scipy.__version__} torch={torch.__version__}")

# Refuse to create new HDM data silently. These are the nine existing training
# trajectories plus the four reporting trajectories used for accuracy scoring.
root = Path.cwd().parent
snapshots = root / "Results/param_snaps"
parameters = [
    *((a, b) for a in (4.25, 4.875, 5.5) for b in (0.015, 0.0225, 0.03)),
    (4.875, 0.0225), (4.56, 0.019), (5.19, 0.026), (4.0, 0.033),
]
for mu1, mu2 in parameters:
    path = snapshots / f"mu1_{mu1}+mu2_{mu2}.npy"
    if not path.is_file():
        raise SystemExit(f"Missing existing HDM snapshot database entry: {path}")
    shape = numpy.load(path, mmap_mode="r", allow_pickle=False).shape
    if shape != (125000, 501):
        raise SystemExit(f"Invalid HDM snapshot shape at {path}: {shape}")
print("Existing HDM database: nine training and four reporting trajectories OK")
PY
  bash Results_Paper/scripts/run_euclidean_hprom_main_prepare.sh check
}
prepare() { bash Results_Paper/scripts/run_euclidean_hprom_main_prepare.sh all; }
train() { bash Results_Paper/scripts/run_euclidean_hprom_main_train_best.sh all; }
rules() { bash Results_Paper/scripts/run_euclidean_hprom_main_online.sh rules all; }
b3() { bash Results_Paper/scripts/run_euclidean_hprom_case2_b3.sh all; }
online() {
  bash Results_Paper/scripts/run_euclidean_hprom_main_online.sh reporting all
  bash Results_Paper/scripts/run_euclidean_hprom_main_online.sh direct all
}
timing() { bash Results_Paper/scripts/benchmark_euclidean_hprom.sh all; }
summary() {
  "$PYTHON_BIN" -u summarize_euclidean_hprom_campaign.py \
    --campaign-root "$EUCLIDEAN_HPROM_ROOT"
}

case "$STAGE" in
  check) check ;;
  prepare) prepare ;;
  train) train ;;
  rules) rules ;;
  b3) b3 ;;
  online) online ;;
  timing) timing ;;
  summary) summary ;;
  all)
    check
    prepare
    train
    rules
    b3
    online
    timing
    summary
    ;;
esac
echo "[euclidean-hprom-campaign] completed stage=$STAGE root=$EUCLIDEAN_HPROM_ROOT"
