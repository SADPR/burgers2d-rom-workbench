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
BASELINE_HPROM_ROOT="${BASELINE_EUCLIDEAN_HPROM_ROOT:-$PROJECT_DIR/Results_Paper/euclidean_hprom_main}"
export CASE2_CAMPAIGN_ROOT="$PAPER_ROOT"
export CASE2_BASIS_DIR="$PROJECT_DIR/Results_Paper/MetricStudy/euclidean/Stage1"
export CASE2_MODEL_PATH="$PAPER_ROOT/Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt"
export CASE2_HYPER_OUTPUT="$PAPER_ROOT/Stage4/case2_b3"
export CASE2_TRAINING_DATASET="${CASE2_TRAINING_DATASET:-$BASELINE_HPROM_ROOT/Stage2/prom_coeff_dataset_ntot151}"
export CASE2_REFERENCE_CAMPAIGN_ROOT="${CASE2_REFERENCE_CAMPAIGN_ROOT:-$BASELINE_HPROM_ROOT}"
OUT="$CASE2_HYPER_OUTPUT"
DRAWS="${CASE2_B3_DRAWS:-2048}"
if [[ ! "$DRAWS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] CASE2_B3_DRAWS must be a positive integer" >&2
  exit 2
fi
RULE_TAG="positive_fit${DRAWS}"
RULE="$OUT/$RULE_TAG/weights.npy"
SELECTION="$OUT/$RULE_TAG/selection.json"
LOG_DIR="$PAPER_ROOT/logs/case2_b3/$RULE_TAG"
RULE_THREADS="${CASE2_B3_RULE_THREADS:-${CASE2_B3_THREADS:-24}}"
ONLINE_THREADS="${CASE2_B3_ONLINE_THREADS:-1}"
REUSE_PREDICTOR="${CASE2_B3_REUSE_PREDICTOR:-1}"
OVERWRITE_REPORTING="${OVERWRITE_REPORTING:-0}"
if [[ ! "$RULE_THREADS" =~ ^[1-9][0-9]*$ || ! "$ONLINE_THREADS" =~ ^[1-9][0-9]*$ ||
      ! "$REUSE_PREDICTOR" =~ ^[01]$ || ! "$OVERWRITE_REPORTING" =~ ^[01]$ ]]; then
  echo "[error] Threads must be positive integers; reuse/overwrite must be 0 or 1" >&2
  exit 2
fi

export MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"
mkdir -p "$OUT" "$MPLCONFIGDIR" "$LOG_DIR"

require_file() { [[ -f "$1" ]] || { echo "[error] Missing required file: $1" >&2; exit 1; }; }
set_threads() {
  export OMP_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1" MKL_NUM_THREADS="$1"
  export BLIS_NUM_THREADS="$1" GOTO_NUM_THREADS="$1" NUMEXPR_NUM_THREADS="$1"
}
require_file "$CASE2_MODEL_PATH"
require_file "$CASE2_BASIS_DIR/basis.npy"
require_file "$CASE2_BASIS_DIR/u_ref.npy"
require_file "$CASE2_TRAINING_DATASET/meta.json"

build_rule() {
  set_threads "$RULE_THREADS"
  "$PYTHON_BIN" -u build_case2_leverage_rule.py --draws "$DRAWS" --seed 418 --output "$OUT"
  "$PYTHON_BIN" -u fit_case2_positive_cubature.py --output "$OUT" --draws "$DRAWS" \
    --threads "$RULE_THREADS" --maxiter 1500 \
    2>&1 | tee "$LOG_DIR/fit.log"
  require_file "$RULE"
}

validate_rule() {
  require_file "$RULE"
  set_threads "$RULE_THREADS"
  "$PYTHON_BIN" -u audit_case2_hyperreduction.py --output "$OUT" --rule-file "$RULE" --threads "$RULE_THREADS" \
    2>&1 | tee "$LOG_DIR/operator_audit.log"
  set_threads "$ONLINE_THREADS"
  local command=(
    "$PYTHON_BIN" -u run_case2_hyperreduction.py validation --output "$OUT" --rule-file "$RULE"
    --threads "$ONLINE_THREADS" --methods hprom3
  )
  if [[ "$REUSE_PREDICTOR" == 1 ]]; then command+=(--reuse-predictor); fi
  "${command[@]}" 2>&1 | tee "$LOG_DIR/validation_rollout.log"
  "$PYTHON_BIN" -u validate_case2_hyper_rule.py --output "$OUT" --rule-file "$RULE" \
    --selection-file "$SELECTION" 2>&1 | tee "$LOG_DIR/selection.log"
}

report_rule() {
  require_file "$SELECTION"
  "$PYTHON_BIN" - "$SELECTION" "$RULE" "$PROJECT_DIR/.." <<'PY'
import hashlib, json, os, sys
from pathlib import Path
selection = json.load(open(sys.argv[1]))
if not selection.get("accepted", False):
    raise SystemExit("The held-out validation protocol did not accept this B3 rule")
if selection.get("selection_uses_reporting_points", True):
    raise SystemExit("Invalid selection manifest: reporting data were used")
rule = Path(sys.argv[2]).resolve()
digest = hashlib.sha256(rule.read_bytes()).hexdigest()
if Path(selection["rule_file"]).resolve() != rule or selection.get("weights_sha256") != digest:
    raise SystemExit("Selection manifest does not match the current B3 weights")
root = Path(sys.argv[3]).resolve()
basis = Path(os.environ["CASE2_BASIS_DIR"])
paths = [Path(os.environ["CASE2_MODEL_PATH"]), basis / "basis.npy", basis / "u_ref.npy",
         Path(os.environ["CASE2_TRAINING_DATASET"]) / "meta.json"]
inputs = {str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if selection.get("inputs") != inputs:
    raise SystemExit("The model, basis or training metadata changed after validation")
PY
  set_threads "$ONLINE_THREADS"
  local command=(
    "$PYTHON_BIN" -u run_case2_hyperreduction.py reporting
    --output "$OUT" --rule-file "$RULE"
    --threads "$ONLINE_THREADS" --methods hprom3
  )
  if [[ "$REUSE_PREDICTOR" == 1 ]]; then command+=(--reuse-predictor); fi
  if [[ "$OVERWRITE_REPORTING" == 1 ]]; then
    command+=(--overwrite-existing)
  fi
  "${command[@]}" \
    2>&1 | tee "$LOG_DIR/reporting_rollout.log"
}

case "$STAGE" in
  rule) build_rule ;;
  validation) validate_rule ;;
  reporting) report_rule ;;
  all) build_rule; validate_rule; report_rule ;;
esac
echo "[euclidean-hprom-case2-b3] completed stage=$STAGE output=$OUT draws=$DRAWS reuse_predictor=$REUSE_PREDICTOR rule_threads=$RULE_THREADS online_threads=$ONLINE_THREADS"
