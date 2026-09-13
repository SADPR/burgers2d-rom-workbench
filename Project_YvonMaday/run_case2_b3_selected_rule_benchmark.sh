#!/usr/bin/env bash
# Execute with bash, not source, inside an allocated node with myenv active.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/b3-budget-mpl-${USER}}"
OUTPUT="${B3_BUDGET_OUTPUT:-Project_YvonMaday/Results_Paper/euclidean_b3_budget_sherlock}"
printf 'Eight trajectories: two frozen B3 rules x four parameters, one thread each.\n'
printf 'No training, no rule fitting, no HDM solves, no trajectory warmups.\n'
"$PYTHON_BIN" -u Project_YvonMaday/benchmark_case2_b3_selected_rule.py \
    --study-root Project_YvonMaday/Results_Paper/euclidean_b3_budget_local \
    --output "$OUTPUT"
