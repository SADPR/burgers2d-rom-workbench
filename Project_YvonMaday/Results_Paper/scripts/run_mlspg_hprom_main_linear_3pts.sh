#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_TAG="mlspg_hprom_main"
export PAPER_RESULTS_ROOT="$PWD/Results_Paper"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export BASIS="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
export UREF="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
export ECSW_WEIGHTS="$PAPER_ROOT/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy"
export LINEAR_RUNS="$PAPER_ROOT/Runs/Linear"
export LOG_DIR="$PAPER_ROOT/logs"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

mkdir -p "$LINEAR_RUNS" "$LOG_DIR" "$MPLCONFIGDIR"

if [[ ! -f "$BASIS" ]]; then echo "Missing BASIS: $BASIS" >&2; exit 1; fi
if [[ ! -f "$UREF" ]]; then echo "Missing UREF: $UREF" >&2; exit 1; fi
if [[ ! -f "$ECSW_WEIGHTS" ]]; then echo "Missing ECSW_WEIGHTS: $ECSW_WEIGHTS" >&2; exit 1; fi

run_one() {
  local mu1="$1"
  local mu2="$2"
  local tag="mu1_${mu1}_mu2_${mu2}"
  echo "==== Linear HPROM at mu=(${mu1}, ${mu2})"
  python3 -u run_prom.py \
    --backend hprom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --ecsw-weights-path "$ECSW_WEIGHTS" \
    --output-root "$LINEAR_RUNS" \
    --no-save-rom-snaps \
    2>&1 | tee "$LOG_DIR/linear_hprom_${tag}.log"
}

# Two off-grid test points and one in-grid verification point.
run_one 4.560 0.0190
run_one 4.875 0.0225
run_one 5.190 0.0260

for f in "$LINEAR_RUNS"/*/summary.txt; do
  echo "==== $(dirname "$f" | xargs basename)"
  grep -E "solve_backend_effective|basis_path|ecsw_weights_path|ecsw_weights_source|n_ecsw_elements|online_solve_elapsed_s|relative_error_percent|output_dir" "$f"
done | tee "$LOG_DIR/linear_hprom_3pts_quick_summary.txt"
