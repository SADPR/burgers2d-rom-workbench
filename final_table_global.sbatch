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
#SBATCH --job-name=ft-global
#SBATCH --partition=cfarhat
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=ft-global-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

ROWS=("HPROM (global)" "HQPROM (global)" "HPROM-GPR (global)")
ROW="${ROWS[$SLURM_ARRAY_TASK_ID]}"

module load python/3.9.0
source ~/myenv/bin/activate

export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
export MPLBACKEND=Agg

echo "=== $ROW | 24 threads | $(date) ==="
python3 -u rerun_table_timings.py \
    --only "$ROW" --rebuild-ecsw --repeats 3
echo "=== done $(date) ==="
