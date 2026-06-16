#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# The ANN/Jacobian online path is slower under BLAS oversubscription.
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TAG="mlspg_prom_probe"
PAPER_ROOT="$PROJECT_DIR/Results_Paper/$TAG"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
MODEL_NAME="case2_prom_ann_ntot151_np10_B01_A10_like_b128_lr5e4.pt"
MODEL="$PAPER_ROOT/Stage3/models/$MODEL_NAME"
CASE2_ROOT="$PAPER_ROOT/Runs/Case2_B01_PROM"
LOG_DIR="$PAPER_ROOT/logs/case2_B01_prom_online_1thread"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

for required in \
  "$BASIS" \
  "$UREF" \
  "$DATASET_DIR/stage2_summary.txt" \
  "$MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing preparation output: $required" >&2
    echo "[error] Run run_mlspg_prom_case2_prepare_24threads.sh first." >&2
    exit 1
  fi
done

if ! grep -q "^coordinate_source: solver_coordinates$" "$DATASET_DIR/stage2_summary.txt"; then
  echo "[error] Stage-2 dataset does not contain direct solver coordinates." >&2
  exit 1
fi

echo "[threads] PROM-ANN online evaluation uses one BLAS/OpenMP thread."
echo "[clean] Replacing only the previous Case-2 PROM-ANN online outputs."
rm -rf "$CASE2_ROOT" "$LOG_DIR"
mkdir -p "$CASE2_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"

points=(
  "4.560 0.0190"
  "4.875 0.0225"
  "5.190 0.0260"
)

for point in "${points[@]}"; do
  read -r mu1 mu2 <<< "$point"
  echo "[run] Case-2 PROM-ANN | mu=(${mu1}, ${mu2})"
  python3 -u run_prom_ann_case_2.py \
    --backend prom \
    --no-ecsw \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --device auto \
    --model-path "$MODEL" \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$CASE2_ROOT" \
    --max-its 20 \
    --relnorm-cutoff 1e-5 \
    --min-delta 1e-2 \
    --linear-solver lstsq \
    --normal-eq-reg 1e-12 \
    2>&1 | tee "$LOG_DIR/case2_prom_B01_n10_mu1_${mu1}_mu2_${mu2}.log"
done

summary_file="$LOG_DIR/prom_case2_online_1thread_summary.txt"
{
  echo "thread_count: 1"
  echo "coordinate_source: solver_coordinates"
  echo "dataset: $DATASET_DIR"
  echo "model: $MODEL"
  echo "case2_root: $CASE2_ROOT"
  for f in "$CASE2_ROOT"/*_summary.txt; do
    [[ -f "$f" ]] || continue
    echo "---- $(basename "$f")"
    grep -E "mu_test|solve_backend_effective|relative_error_percent|qN_source|qN_output|snaps_output" "$f"
  done
} | tee "$summary_file"

echo "[done] One-thread PROM-ANN online evaluation completed."
echo "[done] Summary: $summary_file"
