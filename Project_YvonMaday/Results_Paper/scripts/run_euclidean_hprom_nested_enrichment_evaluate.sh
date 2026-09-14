#!/usr/bin/env bash
# Build the family-specific rule (when needed) and evaluate four reporting points.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LEVEL="${1:?Usage: $0 lhs8|lhs12|lhs18 case1|case2|case3|pod_ae|pod_dl}"
FAMILY="${2:?Usage: $0 lhs8|lhs12|lhs18 case1|case2|case3|pod_ae|pod_dl}"
case "$LEVEL" in lhs8|lhs12|lhs18) ;; *) echo "Unknown level: $LEVEL" >&2; exit 2 ;; esac
case "$FAMILY" in case1|case2|case3|pod_ae|pod_dl) ;; *) echo "Unknown family: $FAMILY" >&2; exit 2 ;; esac

export EUCLIDEAN_HPROM_ROOT="${EUCLIDEAN_HPROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_hprom_enrichment_${LEVEL}}"
export RULE_THREADS="${RULE_THREADS:-24}"
export ONLINE_THREADS="${ONLINE_THREADS:-1}"
export ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
export FORCE="${FORCE:-0}"

case "$FAMILY" in
  case1|case3|pod_ae)
    bash "$SCRIPT_DIR/run_euclidean_hprom_main_online.sh" all "$FAMILY"
    ;;
  case2)
    bash "$SCRIPT_DIR/run_euclidean_hprom_main_online.sh" all case2
    bash "$SCRIPT_DIR/run_euclidean_hprom_main_online.sh" direct podnn
    ;;
  pod_dl)
    bash "$SCRIPT_DIR/run_euclidean_hprom_main_online.sh" direct poddl
    ;;
esac
