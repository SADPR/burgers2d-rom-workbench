#!/usr/bin/env bash
# Paired implementation benchmark inside the user's existing allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
OUTPUT="$PWD/Project_YvonMaday/Results_Paper/euclidean_b3_reuse_sherlock"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/b3-reuse-mpl}"
echo "Eight trajectories: original and predictor reuse at each of four parameters."
echo "One numerical thread, one measurement per variant/point, no trajectory warmups."
echo "Results: $OUTPUT"
"$PYTHON_BIN" -u Project_YvonMaday/benchmark_case2_b3_reuse.py --output "$OUTPUT"
