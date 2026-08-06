#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

stage="${1:-all}"
case "$stage" in
  all|stage2|train|online) ;;
  *)
    echo "Usage: $0 [all|stage2|train|online]" >&2
    exit 2
    ;;
esac

echo "[prom-lhs8-campaign] stage: $stage"
echo "[prom-lhs8-campaign] design: 9 baseline + 4 interior + 4 margin = 17 trajectories"
echo "[prom-lhs8-campaign] source: existing fixed-seed 18+18 LHS PROM data"

if [[ "$stage" == "all" || "$stage" == "stage2" ]]; then
  bash "$SCRIPT_DIR/build_mlspg_prom_ext25_lhs8_nested_stage2.sh"
fi
if [[ "$stage" == "all" || "$stage" == "train" ]]; then
  bash "$SCRIPT_DIR/run_mlspg_prom_ext25_lhs8_nested_train_best.sh" all
fi
if [[ "$stage" == "all" || "$stage" == "online" ]]; then
  bash "$SCRIPT_DIR/run_mlspg_prom_ext25_lhs8_nested_all_prom_rom_4pts.sh" all
fi
