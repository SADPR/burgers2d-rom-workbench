#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_enrichment_ext25_lhs36}"
export PROM_NUM_THREADS="${PROM_NUM_THREADS:-16}"
export ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
export FORCE="${FORCE:-0}"
export PLAN_ONLY="${PLAN_ONLY:-0}"

family="${1:-all}"
case "$family" in
  all|nonlinear_prom|rom|case1|case2|case3|podae|data_driven|poddl) ;;
  *)
    echo "Usage: $0 [all|nonlinear_prom|rom|case1|case2|case3|podae|data_driven|poddl]" >&2
    echo "This wrapper intentionally does not run the linear PROM, to avoid duplicated linear outputs." >&2
    exit 2
    ;;
esac

echo "[prom-ext25-4pts] root:    $PROM_ROOT"
echo "[prom-ext25-4pts] family:  $family"
echo "[prom-ext25-4pts] threads: $PROM_NUM_THREADS"
echo "[prom-ext25-4pts] device:  $ONLINE_DEVICE"
echo "[prom-ext25-4pts] force:   $FORCE"

if [[ "$family" == "all" ]]; then
  bash Results_Paper/scripts/run_mlspg_prom_main_all_prom_rom_4pts.sh nonlinear_prom
  bash Results_Paper/scripts/run_mlspg_prom_main_all_prom_rom_4pts.sh rom
else
  bash Results_Paper/scripts/run_mlspg_prom_main_all_prom_rom_4pts.sh "$family"
fi
