#!/usr/bin/env bash
# Build, validate, and report Case 2+B3 for one enriched Case 2 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LEVEL="${1:?Usage: $0 lhs8|lhs12|lhs18}"
case "$LEVEL" in lhs8|lhs12|lhs18) ;; *) echo "Unknown level: $LEVEL" >&2; exit 2 ;; esac
export EUCLIDEAN_HPROM_ROOT="$PROJECT_DIR/Results_Paper/euclidean_hprom_enrichment_${LEVEL}"
export BASELINE_EUCLIDEAN_HPROM_ROOT="$PROJECT_DIR/Results_Paper/euclidean_hprom_main"
export CASE2_B3_RULE_THREADS="${CASE2_B3_RULE_THREADS:-24}"
export CASE2_B3_ONLINE_THREADS="${CASE2_B3_ONLINE_THREADS:-1}"

bash "$SCRIPT_DIR/run_euclidean_hprom_case2_b3.sh" all
