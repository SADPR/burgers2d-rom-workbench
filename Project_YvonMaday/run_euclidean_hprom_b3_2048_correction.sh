#!/usr/bin/env bash
# Correct only enriched B3 rules; preserve every existing 4096 result.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export CASE2_B3_DRAWS=2048
export CASE2_B3_REUSE_PREDICTOR=1
export CASE2_B3_RULE_THREADS="${CASE2_B3_RULE_THREADS:-24}"
export CASE2_B3_ONLINE_THREADS="${CASE2_B3_ONLINE_THREADS:-1}"
export OVERWRITE_REPORTING=0
# Pin the cubature teacher and scoring references to the existing baseline.
export CASE2_TRAINING_DATASET="$PROJECT_DIR/Results_Paper/euclidean_hprom_main/Stage2/prom_coeff_dataset_ntot151"
export CASE2_REFERENCE_CAMPAIGN_ROOT="$PROJECT_DIR/Results_Paper/euclidean_hprom_main"

if (( $# > 0 )); then
  levels=("$@")
else
  levels=(lhs8 lhs12 lhs18)
fi
for level in "${levels[@]}"; do
  case "$level" in lhs8|lhs12|lhs18) ;;
    *) echo "Usage: bash $0 [lhs8 lhs12 lhs18]" >&2; exit 2 ;;
  esac
done

printf 'B3 only: 2048 draws, seed 418, predictor reuse enabled.\n'
printf 'No ANN training, no new training trajectories, no HDM solves, no other models.\n'
printf 'Per enrichment: refit weights, audit, 2 validation + 4 reporting trajectories.\n'
printf 'Offline threads=%s; online threads=%s. Completed identical runs are skipped.\n' \
  "$CASE2_B3_RULE_THREADS" "$CASE2_B3_ONLINE_THREADS"

failed=0
for level in "${levels[@]}"; do
  printf '\n=== B3 2048: %s ===\n' "$level"
  # Keep the inner fail-fast script outside a Bash conditional; otherwise Bash
  # disables its errexit behavior inside functions used by that conditional.
  set +e
  bash "$PROJECT_DIR/Results_Paper/scripts/run_euclidean_hprom_nested_enrichment_b3.sh" "$level"
  status=$?
  set -e
  if (( status != 0 )); then
    printf '\nB3 %s stopped (code %s). Other enrichments will still be attempted.\n' "$level" "$status" >&2
    failed=1
  fi
done
exit "$failed"
