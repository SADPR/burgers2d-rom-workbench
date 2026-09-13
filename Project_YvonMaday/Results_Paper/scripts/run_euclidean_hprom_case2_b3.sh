#!/usr/bin/env bash
# Construct, validate, freeze, and report Euclidean HPROM Case 2+B3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

STAGE="${1:-all}"
case "$STAGE" in rule|validation|reporting|all) ;; *) echo "Usage: $0 [rule|validation|reporting|all]" >&2; exit 2;; esac

PAPER_ROOT="${EUCLIDEAN_HPROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_hprom_main}"
export CASE2_CAMPAIGN_ROOT="$PAPER_ROOT"
export CASE2_BASIS_DIR="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1"
export CASE2_MODEL_PATH="$PAPER_ROOT/Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt"
export CASE2_HYPER_OUTPUT="$PAPER_ROOT/Stage4/case2_b3"
OUT="$CASE2_HYPER_OUTPUT"
RULE="$OUT/positive_fit4096/weights.npy"
RULE_THREADS="${CASE2_B3_RULE_THREADS:-${CASE2_B3_THREADS:-24}}"
ONLINE_THREADS="${CASE2_B3_ONLINE_THREADS:-1}"

export MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"
mkdir -p "$OUT" "$MPLCONFIGDIR" "$PAPER_ROOT/logs/case2_b3"

require_file() { [[ -f "$1" ]] || { echo "[error] Missing required file: $1" >&2; exit 1; }; }
set_threads() {
  export OMP_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1" MKL_NUM_THREADS="$1"
  export BLIS_NUM_THREADS="$1" GOTO_NUM_THREADS="$1" NUMEXPR_NUM_THREADS="$1"
}
require_file "$CASE2_MODEL_PATH"
require_file "$CASE2_BASIS_DIR/basis.npy"
require_file "$CASE2_BASIS_DIR/u_ref.npy"

build_rule() {
  set_threads "$RULE_THREADS"
  "$PYTHON_BIN" -u build_case2_leverage_rule.py --draws 4096 --output "$OUT"
  "$PYTHON_BIN" -u fit_case2_positive_cubature.py --output "$OUT" --draws 4096 \
    --threads "$RULE_THREADS" --maxiter 1500 \
    2>&1 | tee "$PAPER_ROOT/logs/case2_b3/positive_fit4096.log"
  require_file "$RULE"
}

validate_rule() {
  require_file "$RULE"
  set_threads "$RULE_THREADS"
  "$PYTHON_BIN" -u audit_case2_hyperreduction.py --output "$OUT" --rule-file "$RULE" --threads "$RULE_THREADS" \
    2>&1 | tee "$PAPER_ROOT/logs/case2_b3/operator_audit.log"
  set_threads "$ONLINE_THREADS"
  "$PYTHON_BIN" -u run_case2_hyperreduction.py validation --output "$OUT" --rule-file "$RULE" \
    --threads "$ONLINE_THREADS" --methods hprom3 \
    2>&1 | tee "$PAPER_ROOT/logs/case2_b3/validation_rollout.log"
  "$PYTHON_BIN" -u validate_case2_hyper_rule.py --output "$OUT" --rule-file "$RULE" \
    2>&1 | tee "$PAPER_ROOT/logs/case2_b3/selection.log"
}

report_rule() {
  require_file "$OUT/selection.json"
  "$PYTHON_BIN" - "$OUT/selection.json" <<'PY'
import json, sys
selection = json.load(open(sys.argv[1]))
if not selection.get("accepted", False):
    raise SystemExit("The held-out validation protocol did not accept this B3 rule")
if selection.get("selection_uses_reporting_points", True):
    raise SystemExit("Invalid selection manifest: reporting data were used")
PY
  set_threads "$ONLINE_THREADS"
  "$PYTHON_BIN" -u run_case2_hyperreduction.py reporting --output "$OUT" --rule-file "$RULE" \
    --threads "$ONLINE_THREADS" --methods hprom3 \
    2>&1 | tee "$PAPER_ROOT/logs/case2_b3/reporting_rollout.log"
}

case "$STAGE" in
  rule) build_rule ;;
  validation) validate_rule ;;
  reporting) report_rule ;;
  all) build_rule; validate_rule; report_rule ;;
esac
echo "[euclidean-hprom-case2-b3] completed stage=$STAGE output=$OUT rule_threads=$RULE_THREADS online_threads=$ONLINE_THREADS"
