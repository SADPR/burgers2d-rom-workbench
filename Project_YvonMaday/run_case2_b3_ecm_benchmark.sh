#!/usr/bin/env bash
# Run with bash inside an existing allocation and active Python environment.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BUNDLE="$PROJECT_DIR/Results_Paper/euclidean_b3_ecm_deployment"
if [[ -n "${B3_ECM_BENCHMARK_OUTPUT:-}" ]]; then
    OUTPUT="$B3_ECM_BENCHMARK_OUTPUT"
else
    OUTPUT="$(mktemp -d "$PROJECT_DIR/Results_Paper/euclidean_b3_ecm_sherlock_${SLURM_JOB_ID:-manual}_XXXXXX")"
fi
mkdir -p "$OUTPUT"
printf 'Results: %s\n' "$OUTPUT"
"$PYTHON_BIN" -u Project_YvonMaday/benchmark_case2_b3_ecm.py \
    --bundle "$BUNDLE" --output "$OUTPUT" 2>&1 | tee "$OUTPUT/benchmark.log"
