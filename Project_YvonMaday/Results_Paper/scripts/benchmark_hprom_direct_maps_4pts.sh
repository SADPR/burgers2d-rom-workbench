#!/usr/bin/env bash
set -euo pipefail

# Times only the already-trained POD--NN--ROM and POD--DL--ROM maps.  It never
# invokes an HDM/PROM/HPROM solve and never writes under Runs/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

CAMPAIGN="${1:-all}"
case "$CAMPAIGN" in
  all|baseline|enriched) ;;
  *)
    echo "Usage: $0 [all|baseline|enriched]" >&2
    exit 2
    ;;
esac

ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
ONLINE_THREADS="${ONLINE_THREADS:-1}"
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mlspg_hprom_direct_timing_mpl}"
mkdir -p "$MPLCONFIGDIR"

case "$ONLINE_DEVICE" in
  cpu|cuda) ;;
  *)
    echo "[error] ONLINE_DEVICE must be cpu or cuda, got: $ONLINE_DEVICE" >&2
    exit 2
    ;;
esac

set_threads() {
  local count="$1"
  export BLIS_NUM_THREADS="$count"
  export GOTO_NUM_THREADS="$count"
  export MKL_NUM_THREADS="$count"
  export OMP_NUM_THREADS="$count"
  export OPENBLAS_NUM_THREADS="$count"
  export VECLIB_MAXIMUM_THREADS="$count"
}

run_campaign() {
  local label="$1"
  local root="$2"
  local summary="$root/timing/direct_inference_repeat${REPEATS}_summary.txt"
  mkdir -p "$(dirname "$summary")"

  echo "[direct-timing] campaign: $label"
  echo "[direct-timing] root:     $root"
  echo "[direct-timing] device:   $ONLINE_DEVICE"
  echo "[direct-timing] threads:  $ONLINE_THREADS"
  echo "[direct-timing] repeats:  $REPEATS"
  echo "[direct-timing] warmup:   $WARMUP"
  echo "[direct-timing] output:   $summary"

  python3 -u Results_Paper/benchmark_hprom_direct_maps.py \
    --campaign-root "$root" \
    --device "$ONLINE_DEVICE" \
    --threads "$ONLINE_THREADS" \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --summary-path "$summary"
}

set_threads "$ONLINE_THREADS"

if [[ "$CAMPAIGN" == "all" || "$CAMPAIGN" == "baseline" ]]; then
  run_campaign "baseline" "$PROJECT_DIR/Results_Paper/mlspg_hprom_main"
fi
if [[ "$CAMPAIGN" == "all" || "$CAMPAIGN" == "enriched" ]]; then
  run_campaign "enriched" "$PROJECT_DIR/Results_Paper/mlspg_hprom_enrichment_ext25_lhs36"
fi
