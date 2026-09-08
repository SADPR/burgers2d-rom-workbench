#!/usr/bin/env bash
set -euo pipefail

# Evaluate the robust (IRLS) Case--2 PROM solve at the standard four points.
# No offline data is built or changed: same master ANN, same basis, same
# tail-injection mechanism as the production Case--2 run. Only the online
# Gauss-Newton normal equations change (see the module docstring in
# burgers/pod_ann_manifold.py:inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_robust).
# Existing Case--2 results are never reused or overwritten by this diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PAPER_ROOT="${PAPER_ROOT:-$PROJECT_DIR/Results_Paper/mlspg_prom_main}"
MODEL_PATH="${MODEL_PATH:-$PAPER_ROOT/Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt}"
BASIS_PATH="${BASIS_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy}"
U_REF_PATH="${U_REF_PATH:-$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy}"
RUN_ROOT="${RUN_ROOT:-$PAPER_ROOT/Runs/PROM/Case2_Robust/np10}"
KERNELS="${KERNELS:-none huber geman_mcclure}"
HUBER_DELTA="${HUBER_DELTA:-1.345}"
POINTS="${POINTS:-4.875,0.0225 4.560,0.0190 5.190,0.0260 4.000,0.0330}"
ONLINE_DEVICE="${ONLINE_DEVICE:-cpu}"
FORCE="${FORCE:-0}"

mkdir -p "$RUN_ROOT" "$PAPER_ROOT/logs/case2_robust"

if [[ ! -f "$MODEL_PATH" || ! -f "$BASIS_PATH" || ! -f "$U_REF_PATH" ]]; then
  echo "Missing master model, basis, or reference state." >&2
  exit 1
fi

echo "[case2-robust] model:   $MODEL_PATH"
echo "[case2-robust] kernels: $KERNELS"
echo "[case2-robust] points:  $POINTS"
echo "[case2-robust] device:  $ONLINE_DEVICE"

for kernel in $KERNELS; do
  args=()
  [[ "$FORCE" == "1" ]] && args+=(--force)
  python3 -u run_prom_ann_case_2_robust.py \
    --model-path "$MODEL_PATH" \
    --basis-path "$BASIS_PATH" \
    --u-ref-path "$U_REF_PATH" \
    --output-root "$RUN_ROOT/$kernel" \
    --n-primary 10 \
    --robust-kernel "$kernel" \
    --huber-delta "$HUBER_DELTA" \
    --points $POINTS \
    --device "$ONLINE_DEVICE" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    "${args[@]}" \
    2>&1 | tee "$PAPER_ROOT/logs/case2_robust/test_${kernel}.log"
done

echo "[case2-robust] done."
