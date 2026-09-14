#!/usr/bin/env bash
# One enriched Case-2+B3 rule/validation/reporting job: one full node.
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
: "${PYTHON_BIN:?PYTHON_BIN must identify the active Sherlock environment}"

export CASE2_B3_RULE_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export CASE2_B3_ONLINE_THREADS=1
export OMP_NUM_THREADS="$CASE2_B3_RULE_THREADS" OPENBLAS_NUM_THREADS="$CASE2_B3_RULE_THREADS"
export MKL_NUM_THREADS="$CASE2_B3_RULE_THREADS" BLIS_NUM_THREADS="$CASE2_B3_RULE_THREADS"
export GOTO_NUM_THREADS="$CASE2_B3_RULE_THREADS" NUMEXPR_NUM_THREADS="$CASE2_B3_RULE_THREADS"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-hprom-b3-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

bash Project_YvonMaday/Results_Paper/scripts/run_euclidean_hprom_nested_enrichment_b3.sh "$1"
