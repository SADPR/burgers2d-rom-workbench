#!/usr/bin/env bash
# Resumable end-to-end Euclidean HPROM campaign. Run with bash in an existing allocation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
case "$STAGE" in check|prepare|train|rules|b3|online|timing|retime|retime-finish|summary|all) ;;
  *) echo "Usage: $0 [check|prepare|train|rules|b3|online|timing|retime|retime-finish|summary|all]" >&2; exit 2;; esac

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHON_BIN
PAPER_ROOT="$PWD/Results_Paper/euclidean_hprom_main"
export EUCLIDEAN_HPROM_ROOT="${EUCLIDEAN_HPROM_ROOT:-$PAPER_ROOT}"
OFFLINE_THREADS="${HPROM_THREADS:-24}"
BASELINE_THREADS="${HPROM_BASELINE_THREADS:-24}"
LEARNED_ONLINE_THREADS="${HPROM_LEARNED_THREADS:-1}"
export PROM_NUM_THREADS="${PROM_NUM_THREADS:-$OFFLINE_THREADS}"
export TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-$OFFLINE_THREADS}"
export RULE_THREADS="${RULE_THREADS:-$OFFLINE_THREADS}"
export CASE2_B3_RULE_THREADS="${CASE2_B3_RULE_THREADS:-$OFFLINE_THREADS}"
export LINEAR_REPORT_THREADS="${LINEAR_REPORT_THREADS:-$BASELINE_THREADS}"
export ONLINE_THREADS="${ONLINE_THREADS:-$LEARNED_ONLINE_THREADS}"
export CASE2_B3_ONLINE_THREADS="${CASE2_B3_ONLINE_THREADS:-$LEARNED_ONLINE_THREADS}"
export HDM_TIMING_THREADS="${HDM_TIMING_THREADS:-$BASELINE_THREADS}"
export DIRECT_TIMING_THREADS="${DIRECT_TIMING_THREADS:-$LEARNED_ONLINE_THREADS}"
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
write_timing_protocol() {
  local hdm_threads="$1" linear_threads="$2" learned_threads="$3"
  local b3_threads="$4" direct_threads="$5"
  mkdir -p "$EUCLIDEAN_HPROM_ROOT/timing"
  "$PYTHON_BIN" - "$EUCLIDEAN_HPROM_ROOT/timing/online_thread_protocol.json" \
    "$hdm_threads" "$linear_threads" "$learned_threads" "$b3_threads" \
    "$direct_threads" "$OFFLINE_THREADS" <<'PY'
import datetime
import json
import os
import socket
import sys

(
    path,
    hdm_threads,
    linear_threads,
    learned_threads,
    b3_threads,
    direct_threads,
    offline_threads,
) = sys.argv[1:]
payload = {
    "protocol": "method-appropriate thread-count online comparison",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "hdm_threads": int(hdm_threads),
    "linear_hprom_threads": int(linear_threads),
    "learned_intrusive_online_threads": int(learned_threads),
    "case2_b3_online_threads": int(b3_threads),
    "direct_inference_threads": int(direct_threads),
    "offline_training_threads": int(offline_threads),
    "ecm_construction_threads": int(offline_threads),
    "reused_artifacts": [
        "Euclidean basis",
        "trained checkpoints",
        "frozen ECM rules",
        "24-thread HDM timing",
        "24-thread linear-HPROM reporting runs",
    ],
    "overwritten_artifacts": ["learned reporting-point online runs", "direct-map timing"],
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
print(f"Online timing protocol: {path}")
PY
}
validate_existing_baselines() {
  "$PYTHON_BIN" - "$EUCLIDEAN_HPROM_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
hdm_path = root / "timing/hdm/hdm_timing.json"
if not hdm_path.is_file():
    raise SystemExit(f"Missing existing HDM timing: {hdm_path}")
hdm = json.loads(hdm_path.read_text(encoding="utf-8"))
if int(hdm.get("threads", -1)) != 24:
    raise SystemExit(f"Existing HDM timing did not use 24 threads: {hdm_path}")

points = ((4.875, 0.0225), (4.560, 0.0190), (5.190, 0.0260), (4.000, 0.0330))
for mu1, mu2 in points:
    tag = f"linear_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"
    summary = root / "Runs/Linear" / tag / "summary.txt"
    if not summary.is_file():
        raise SystemExit(f"Missing existing linear-HPROM run: {summary}")

log_path = root / "logs/campaign_driver.log"
if not log_path.is_file():
    raise SystemExit(f"Missing campaign log needed to verify the linear-HPROM thread count: {log_path}")
log = log_path.read_text(encoding="utf-8", errors="replace")
thread_recorded = (
    re.search(r"\[euclidean-hprom-prepare\] threads:\s+24\b", log)
    or re.search(r"linear online threads:\s+24\b", log)
)
if not thread_recorded:
    raise SystemExit("The campaign log does not certify a 24-thread linear-HPROM run.")
print("Existing baselines verified: HDM=24 threads; linear HPROM=24 threads.")
PY
}
validate_existing_learned_retime() {
  "$PYTHON_BIN" - "$EUCLIDEAN_HPROM_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = sorted((root / "logs").glob("retime*.log"))
start = "[euclidean-hprom-online] ECM=2.0% rule_threads=24 online_threads=1 device=cpu"
done = "[euclidean-hprom-online] completed stage=reporting family=all"
certified = False
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    begin = text.rfind(start)
    if begin >= 0 and text.find(done, begin) >= 0:
        certified = True
        print(f"Existing one-thread learned HPROM runs certified by: {path}")
        break
if not certified:
    raise SystemExit("No retime log certifies completion of all conventional learned HPROMs with one thread.")

specs = {
    "HPROM_ANN_C1": "case1_hprom_ann*_summary.txt",
    "HPROM_ANN_C2": "case2_hprom_ann*_summary.txt",
    "HPROM_ANN_C3": "case3_hprom_ann*_summary.txt",
    "HPROM_POD_AE": "podae_hprom*_summary.txt",
}
for family, pattern in specs.items():
    files = list((root / "Runs" / family).glob(pattern))
    if len(files) != 4:
        raise SystemExit(f"Expected four completed {family} summaries, found {len(files)}.")
PY
}
complete_retime() {
  local baseline_threads=24 learned_threads=1
  OVERWRITE_REPORTING=1 CASE2_B3_ONLINE_THREADS="$learned_threads" \
    bash Results_Paper/scripts/run_euclidean_hprom_case2_b3.sh reporting
  FORCE_TIMING=1 DIRECT_TIMING_THREADS="$learned_threads" \
    bash Results_Paper/scripts/benchmark_euclidean_hprom.sh direct
  write_timing_protocol \
    "$baseline_threads" "$baseline_threads" "$learned_threads" \
    "$learned_threads" "$learned_threads"
  summary
}
retime() {
  # The baselines use the full allocation. Small learned online solves are
  # single-threaded to avoid nested BLAS/PyTorch overhead.
  local baseline_threads=24 learned_threads=1
  echo "[euclidean-hprom-retime] Reusing basis, checkpoints, and frozen ECM rules."
  echo "[euclidean-hprom-retime] Reusing HDM and linear HPROM measured with $baseline_threads threads."
  echo "[euclidean-hprom-retime] Learned HPROMs and direct maps: $learned_threads thread."
  validate_existing_baselines
  FORCE=1 ONLINE_THREADS="$learned_threads" \
    bash Results_Paper/scripts/run_euclidean_hprom_main_online.sh reporting all
  complete_retime
}
finish_retime() {
  validate_existing_baselines
  validate_existing_learned_retime
  complete_retime
}

case "$STAGE" in
  check) check ;;
  prepare) prepare ;;
  train) train ;;
  rules) rules ;;
  b3) b3 ;;
  online) online ;;
  timing) timing ;;
  retime) retime ;;
  retime-finish) finish_retime ;;
  summary) summary ;;
  all)
    check
    prepare
    train
    rules
    b3
    online
    timing
    write_timing_protocol \
      "$HDM_TIMING_THREADS" "$LINEAR_REPORT_THREADS" "$ONLINE_THREADS" \
      "$CASE2_B3_ONLINE_THREADS" "$DIRECT_TIMING_THREADS"
    summary
    ;;
esac
echo "[euclidean-hprom-campaign] completed stage=$STAGE root=$EUCLIDEAN_HPROM_ROOT"
