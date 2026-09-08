#!/usr/bin/env bash
set -euo pipefail

# Evaluate the GLS (precision-weighted) Case--2 PROM solve at the standard
# four points, across a small (rank, ridge_fraction) grid. No offline
# retraining: the Sigma_tail factor is built on the fly from the OOF tail
# errors already saved by stage3_build_case2_tail_correction_basis.py.
# Existing Case--2 results are never reused or overwritten by this diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="${PAPER_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
MODEL_PATH="${MODEL_PATH:-$PAPER_ROOT/Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt}"
BASIS_PATH="${BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
U_REF_PATH="${U_REF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
OOF_ERRORS_PATH="${OOF_ERRORS_PATH:-$PAPER_ROOT/Stage3/case2_tail_correction_oof_ntot151_np10/oof_tail_errors.npy}"
RUN_ROOT="${RUN_ROOT:-$PAPER_ROOT/Runs/PROM/Case2_GLS/np10}"
RANKS="${RANKS:-10 30}"
RIDGE_FRACTIONS="${RIDGE_FRACTIONS:-0.01 0.1}"
POINTS="${POINTS:-4.875,0.0225 4.560,0.0190 5.190,0.0260 4.000,0.0330}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"

mkdir -p "$RUN_ROOT" "$PAPER_ROOT/logs/case2_gls"

if [[ ! -f "$MODEL_PATH" || ! -f "$BASIS_PATH" || ! -f "$U_REF_PATH" || ! -f "$OOF_ERRORS_PATH" ]]; then
  echo "Missing master model, basis, reference state, or OOF error matrix." >&2
  exit 1
fi

echo "[case2-gls] model:   $MODEL_PATH"
echo "[case2-gls] ranks:   $RANKS"
echo "[case2-gls] ridges:  $RIDGE_FRACTIONS"
echo "[case2-gls] points:  $POINTS"
echo "[case2-gls] device:  $ONLINE_DEVICE"

for rank in $RANKS; do
  for ridge in $RIDGE_FRACTIONS; do
    args=()
    [[ "$FORCE" == "1" ]] && args+=(--force)
    tag="r${rank}_ridge${ridge}"
    python3 -u run_prom_ann_case_2_gls.py \
      --model-path "$MODEL_PATH" \
      --basis-path "$BASIS_PATH" \
      --u-ref-path "$U_REF_PATH" \
      --oof-errors-path "$OOF_ERRORS_PATH" \
      --output-root "$RUN_ROOT/$tag" \
      --n-primary 10 \
      --rank "$rank" \
      --ridge-fraction "$ridge" \
      --points $POINTS \
      --device "$ONLINE_DEVICE" \
      --max-its 20 \
      --relnorm-cutoff 1e-5 \
      --min-delta 1e-2 \
      "${args[@]}" \
      2>&1 | tee "$PAPER_ROOT/logs/case2_gls/test_${tag}.log"
  done
done

echo "[case2-gls] done."
