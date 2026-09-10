#!/bin/bash
# Rebuild the cubature of the nine LOCAL ROMs and re-measure them.
#
#   sbatch final_table_local.sbatch
#
# One array task per row, each confined to FinalTable/<row>/. The local POD
# bases, quadratic manifolds, GPR models and clustering are reused untouched;
# only the cubature is rebuilt, with the current assembly code.
#
# The largest local ECSW training matrix seen so far is 4320 x 62500 (2.2 GB)
# and the slowest took 4928 s, so 32 GB and 8 h leave ample headroom.

#SBATCH --job-name=ft-local
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=ft-local-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

ROWS=("Local HPROM (Nc=3)"     "Local HPROM (Nc=5)"     "Local HPROM (Nc=8)"
      "Local HQPROM (Nc=3)"    "Local HQPROM (Nc=5)"    "Local HQPROM (Nc=8)"
      "Local HPROM-GPR (Nc=3)" "Local HPROM-GPR (Nc=5)" "Local HPROM-GPR (Nc=8)")
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
