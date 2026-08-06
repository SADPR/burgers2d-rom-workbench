#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PROM_ROOT="${PROM_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_enrichment_ext25_lhs8_nested}"
if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  echo "[prom-lhs8-4pts] root: $PROM_ROOT"
  echo "[prom-lhs8-4pts] PLAN_ONLY=1; no online solves were run."
  exit 0
fi

exec bash "$SCRIPT_DIR/run_mlspg_prom_ext25_lhs36_all_prom_rom_4pts.sh" "$@"
