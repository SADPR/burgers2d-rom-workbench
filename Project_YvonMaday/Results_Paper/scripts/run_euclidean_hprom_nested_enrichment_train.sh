#!/usr/bin/env bash
# Train one fixed model family at one nested HPROM enrichment level.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LEVEL="${1:?Usage: $0 lhs8|lhs12|lhs18 case1|case2|case3|pod_ae|pod_dl}"
FAMILY="${2:?Usage: $0 lhs8|lhs12|lhs18 case1|case2|case3|pod_ae|pod_dl}"
case "$LEVEL" in lhs8) EXPECTED=17 ;; lhs12) EXPECTED=21 ;; lhs18) EXPECTED=27 ;; *) echo "Unknown level: $LEVEL" >&2; exit 2 ;; esac
case "$FAMILY" in case1|case2|case3|pod_ae|pod_dl) ;; *) echo "Unknown family: $FAMILY" >&2; exit 2 ;; esac

PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Results_Paper}"
TAG="euclidean_hprom_enrichment_${LEVEL}"
DATASET="$PAPER_RESULTS_ROOT/$TAG/Stage2/prom_coeff_dataset_ntot151_enriched_${LEVEL}"
VALIDATION="${VALIDATION_DATASET_DIR:-$PAPER_RESULTS_ROOT/euclidean_hprom_main/Stage2/prom_coeff_dataset_ntot151_validation2}"

PAPER_RESULTS_ROOT="$PAPER_RESULTS_ROOT" \
PAPER_TAG="$TAG" \
DATASET_DIR="$DATASET" \
VALIDATION_DATASET_DIR="$VALIDATION" \
EXPECTED_TRAIN_TRAJECTORIES="$EXPECTED" \
CAMPAIGN_LABEL="Euclidean HPROM nested $LEVEL" \
TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-24}" \
FORCE="${FORCE:-0}" \
bash "$SCRIPT_DIR/run_euclidean_hprom_main_train_best.sh" "$FAMILY"
