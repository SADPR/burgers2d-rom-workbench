#!/bin/bash
#SBATCH --job-name=case2-krylov3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=case2-krylov3-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated experiment root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/case2-mpl-${SLURM_JOB_ID}"
OUT="$PWD/Project_YvonMaday/Results_Paper/euclidean_case2_sherlock_${SLURM_JOB_ID}"
sha256sum --check Project_YvonMaday/case2_sherlock_inputs.sha256
srun --cpu-bind=cores "$PYTHON_BIN" -u Project_YvonMaday/run_case2_sherlock_benchmark.py \
    --threads 24 --repetitions 1 --warmups 0 --output "$OUT"
