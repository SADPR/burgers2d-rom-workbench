#!/usr/bin/env bash
# Execute the reproducible Euclidean 9+8 and 9+18 PROM enrichment campaign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE="${1:-all}"
case "$STAGE" in
  all|stage2|train|online) ;;
  *) echo "Usage: $0 [all|stage2|train|online]" >&2; exit 2 ;;
esac

echo "[euclidean-nested-campaign] stage: $STAGE"
echo "[euclidean-nested-campaign] levels: 9+8 (4 interior + 4 margin), 9+18 (9 interior + 9 margin)"
echo "[euclidean-nested-campaign] no 9+36 PROM dataset or solve is created."

if [[ "$STAGE" == "all" || "$STAGE" == "stage2" ]]; then
  bash "$SCRIPT_DIR/run_euclidean_prom_nested_enrichment_stage2.sh"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  bash "$SCRIPT_DIR/run_euclidean_prom_nested_enrichment_train_best.sh" all
fi
if [[ "$STAGE" == "all" || "$STAGE" == "online" ]]; then
  bash "$SCRIPT_DIR/run_euclidean_prom_nested_enrichment_all_prom_rom_4pts.sh" all
fi
