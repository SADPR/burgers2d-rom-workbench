#!/usr/bin/env bash
# Run directly in an existing interactive allocation. No sbatch or srun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SLURM_JOB_ID:?Run this script inside your allocated Sherlock compute node}"
NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
if [[ -n "$NODES" ]] && command -v scontrol >/dev/null 2>&1; then
  if ! scontrol show hostnames "$NODES" | grep -Fxq "$(hostname -s)"; then
    echo "This host is not in your allocation. Enter the allocated compute node first." >&2
    exit 1
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/case2-mpl-${SLURM_JOB_ID}"

sha256sum --check Project_YvonMaday/case2_sherlock_inputs.sha256
echo "Using the active Python environment: $PYTHON_BIN"
# pytest is a development dependency, not a solver dependency. Never install
# packages or change the active environment to run this benchmark.
if "$PYTHON_BIN" -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("pytest") is None)'; then
  "$PYTHON_BIN" -m pytest -q tests/test_case2_residual_correction.py \
    tests/test_ecm_consistency.py tests/test_case2_sherlock_benchmark.py
else
  echo "pytest is not installed: remote unit tests skipped (not marked as passed)."
fi
"$PYTHON_BIN" -c 'import Project_YvonMaday.run_case2_local_corrections; import Project_YvonMaday.run_case2_local_reference_benchmarks; print("Solver imports: OK")'

mkdir -p Project_YvonMaday/Results_Paper
OUT="$(mktemp -d "$PWD/Project_YvonMaday/Results_Paper/euclidean_case2_sherlock_${SLURM_JOB_ID}_XXXXXX")"
echo "Results: $OUT"
echo "Running 16 trajectories sequentially on $(hostname), using 24 numerical threads."
echo "Four methods x four parameters x one measurement. No full-trajectory warmups."
"$PYTHON_BIN" -u Project_YvonMaday/run_case2_sherlock_benchmark.py \
  --threads 24 --repetitions 1 --warmups 0 \
  --methods case2_known_initial adaptive3_known_initial case1_known_initial case3_known_initial \
  --output "$OUT" 2>&1 | tee "$OUT/benchmark.log"
