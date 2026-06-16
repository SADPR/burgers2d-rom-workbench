#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# PROM residual/Jacobian assembly and reduced solves benefit from threaded BLAS.
export BLIS_NUM_THREADS=24
export GOTO_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

TAG="mlspg_prom_probe"
PAPER_ROOT="$PROJECT_DIR/Results_Paper/$TAG"
BASIS="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PROJECT_DIR/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
DATASET_DIR="$PAPER_ROOT/Stage2/prom_coeff_dataset_ntot151"
MODEL_NAME="case2_prom_ann_ntot151_np10_B01_A10_like_b128_lr5e4.pt"
MODEL="$PAPER_ROOT/Stage3/models/$MODEL_NAME"
MODEL_SUMMARY="case2_prom_ann_ntot151_np10_B01_A10_like_b128_lr5e4_summary.txt"
LINEAR_ROOT="$PAPER_ROOT/Runs/Linear_PROM"
LOG_DIR="$PAPER_ROOT/logs"
export MPLCONFIGDIR="$PAPER_ROOT/.mplcache"

for required in "$BASIS" "$UREF"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required file: $required" >&2
    exit 1
  fi
done

echo "[threads] PROM preparation uses 24 BLAS/OpenMP threads."
echo "[clean] Removing the previous isolated PROM probe preparation outputs."
rm -rf "$PAPER_ROOT/Stage2" "$PAPER_ROOT/Stage3" "$PAPER_ROOT/Runs" "$LOG_DIR"
mkdir -p "$PAPER_ROOT/Stage3/models" "$LINEAR_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"

echo "[1/3] Building PROM Stage-2 data from solver-side coordinates."
python3 -u stage2_build_prom_qn_dataset.py \
  --backend prom \
  --total-modes 151 \
  --basis-path "$BASIS" \
  --u-ref-path "$UREF" \
  --output-dir "$DATASET_DIR" \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --no-save-rom-snaps \
  --no-plots \
  2>&1 | tee "$LOG_DIR/stage2_prom_solver_qn_ntot151.log"

grep -q "^coordinate_source: solver_coordinates$" "$DATASET_DIR/stage2_summary.txt"
grep -E "solve_backend|dataset_dir|basis_path|u_ref_path|total_modes|coordinate_(recovery|source)" \
  "$DATASET_DIR/stage2_summary.txt" | tee "$LOG_DIR/stage2_prom_solver_qn_ntot151_check.txt"

echo "[2/3] Training Case-2 PROM ANN: (mu1, mu2, t) -> q_11:151."
python3 -u stage3_perform_training_case_2_ann_test_n20_maday.py \
  --maday-tag "$TAG" \
  --maday-results-root "$PROJECT_DIR/Results_Paper" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$DATASET_DIR" \
  --primary-modes 10 \
  --model-name "$MODEL_NAME" \
  --summary-name "$MODEL_SUMMARY" \
  --seed 42 \
  --val-frac 0.1 \
  --val-split-mode row \
  --hidden-dims 256,512,512,256 \
  --activation silu \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-6 \
  --dropout 0.0 \
  --epochs 6000 \
  --patience 220 \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 50 \
  --lr-scheduler-min-lr 1e-6 \
  2>&1 | tee "$LOG_DIR/train_case2_prom_B01_n10_solver_qn.log"

if [[ ! -f "$MODEL" ]]; then
  echo "[error] Training did not create the expected model: $MODEL" >&2
  exit 1
fi

points=(
  "4.560 0.0190"
  "4.875 0.0225"
  "5.190 0.0260"
)

echo "[3/3] Running direct linear PROM references at the three points."
for point in "${points[@]}"; do
  read -r mu1 mu2 <<< "$point"
  python3 -u run_prom.py \
    --backend prom \
    --mu1 "$mu1" \
    --mu2 "$mu2" \
    --total-modes 151 \
    --no-ecsw \
    --basis-path "$BASIS" \
    --u-ref-path "$UREF" \
    --output-root "$LINEAR_ROOT" \
    2>&1 | tee "$LOG_DIR/linear_prom_mu1_${mu1}_mu2_${mu2}.log"
done

summary_file="$LOG_DIR/prom_case2_prepare_24threads_summary.txt"
{
  echo "thread_count: 24"
  echo "coordinate_source: solver_coordinates"
  echo "dataset: $DATASET_DIR"
  echo "model: $MODEL"
  echo "linear_root: $LINEAR_ROOT"
  echo
  echo "=== Stage 2 ==="
  grep -E "solve_backend|total_modes|coordinate_(recovery|source)" "$DATASET_DIR/stage2_summary.txt"
  echo
  echo "=== Model ==="
  grep -E "dataset_backend|primary_modes|secondary_modes|hidden_dims|activation|batch_size|lr:|val_rel_frob_percent" \
    "$PAPER_ROOT/Stage3/$MODEL_SUMMARY"
  echo
  echo "=== Linear PROM ==="
  find "$LINEAR_ROOT" -type f -name "summary.txt" -print0 |
    sort -z |
    while IFS= read -r -d '' f; do
      echo "---- $(basename "$(dirname "$f")")"
      grep -E "solve_backend_effective|qN_source|relative_error_percent|save_rom_snaps" "$f"
    done
} | tee "$summary_file"

echo "[done] PROM preparation completed."
echo "[done] Next run the one-thread online script."
echo "[done] Summary: $summary_file"
