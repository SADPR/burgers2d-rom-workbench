#!/bin/bash
# Rebuild the cubature of the three GLOBAL ROMs and re-measure them.
#
#   sbatch final_table_global.sbatch
#
# One array task per row. Each task writes only into FinalTable/<row>/, so the
# tasks are independent of each other and of the repository's Results/.
# The POD bases, the quadratic coefficient matrix and the trained GPR models
# are reused untouched -- the randomized SVD here is unseeded, so rebuilding a
# basis would silently change every number downstream.
#
# Memory is sized by the ECSW training matrix: the global linear needs
# n_red x n_snaps x n_cells = 96 x 90 x 62500 x 8 B = 4.3 GB, plus the
# cubature's working copies.

#SBATCH --job-name=ft-global
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=ft-global-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

ROWS=("HPROM (global)" "HQPROM (global)" "HPROM-GPR (global)")
ROW="${ROWS[$SLURM_ARRAY_TASK_ID]}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NT="${SLURM_CPUS_PER_TASK:-24}"
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ft-mpl-${SLURM_JOB_ID}"

echo "=== $ROW | $NT threads | $(date) ==="
srun --cpu-bind=cores "$PYTHON_BIN" -u rerun_table_timings.py \
    --only "$ROW" --rebuild-ecsw --repeats 3
echo "=== done $(date) ==="
