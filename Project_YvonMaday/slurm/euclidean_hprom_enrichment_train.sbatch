#!/usr/bin/env bash
# One model-training job: one full Sherlock node for a 24-thread run.
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
: "${PYTHON_BIN:?PYTHON_BIN must identify the active Sherlock environment}"

export TRAIN_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS" OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS" BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS" NUMEXPR_NUM_THREADS="$TRAIN_NUM_THREADS"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-hprom-train-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

bash Project_YvonMaday/Results_Paper/scripts/run_euclidean_hprom_nested_enrichment_train.sh "$1" "$2"
