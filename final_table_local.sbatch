#!/bin/bash
# Rebuild the cubature of the nine LOCAL ROMs and re-measure them.
#
#   sbatch final_table_local.sbatch
#
# One array task per row, each confined to FinalTable/<row>/. The local POD
# bases, quadratic manifolds, GPR models and clustering are reused untouched;
# only the cubature is rebuilt, with the current assembly code.
#
#SBATCH --job-name=ft-local
#SBATCH --partition=cfarhat
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=ft-local-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

ROWS=("Local HPROM (Nc=3)"     "Local HPROM (Nc=5)"     "Local HPROM (Nc=8)"
      "Local HQPROM (Nc=3)"    "Local HQPROM (Nc=5)"    "Local HQPROM (Nc=8)"
      "Local HPROM-GPR (Nc=3)" "Local HPROM-GPR (Nc=5)" "Local HPROM-GPR (Nc=8)")
ROW="${ROWS[$SLURM_ARRAY_TASK_ID]}"

module load python/3.9.0
source ~/myenv/bin/activate

export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
export MPLBACKEND=Agg

echo "=== $ROW | 24 threads | $(date) ==="
python3 -u rerun_table_timings.py \
    --only "$ROW" --rebuild-ecsw --repeats 3
echo "=== done $(date) ==="
