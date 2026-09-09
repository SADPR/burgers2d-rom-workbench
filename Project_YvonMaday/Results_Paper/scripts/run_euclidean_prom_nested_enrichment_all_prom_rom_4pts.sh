#!/usr/bin/env bash
# Evaluate all PROM and direct-ROM models for the two nested Euclidean campaigns.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LEVEL="${1:-all}"
case "$LEVEL" in
  all|lhs8|lhs18) ;;
  *) echo "Usage: $0 [all|lhs8|lhs18]" >&2; exit 2 ;;
esac

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Results_Paper}"
PROM_NUM_THREADS="${PROM_NUM_THREADS:-24}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"

run_level() {
  local level="$1" root
  root="$PAPER_RESULTS_ROOT/euclidean_prom_enrichment_${level}"
  [[ -d "$root/Stage3/models" ]] || { echo "[error] Missing trained models: $root/Stage3/models" >&2; exit 1; }
  echo "[euclidean-nested-online] ===== $level ====="
  EUCLIDEAN_PROM_ROOT="$root" \
  PROM_NUM_THREADS="$PROM_NUM_THREADS" \
  ONLINE_DEVICE="$ONLINE_DEVICE" \
  FORCE="$FORCE" \
  bash "$SCRIPT_DIR/run_euclidean_prom_main_all_prom_rom_4pts.sh" all
}

if [[ "$LEVEL" == "all" ]]; then
  run_level lhs8
  run_level lhs18
else
  run_level "$LEVEL"
fi
