#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-maday_clean_try01}"
PERT="${2:-1.0}"
SEED="${3:-11}"

export MPLCONFIGDIR="${ROOT}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

OUTROOT="${ROOT}/Results_Maday/${TAG}/Case2_oracle_pert${PERT}_seed${SEED}"
mkdir -p "${OUTROOT}"

MUS=(
  "4.875 0.0225"
  "4.560 0.0190"
  "5.190 0.0260"
)

run_case () {
  local label="$1"
  local mu1="$2"
  local mu2="$3"
  local linear_dir="$4"
  local basis="$5"
  local uref="$6"
  local outdir="$7"

  python3 -u "${ROOT}/run_case2_pg_oracle_tmp.py" \
    --mu1 "${mu1}" --mu2 "${mu2}" \
    --n-primary 10 --n-tot 151 \
    --linear-run-dir "${linear_dir}" \
    --basis-path "${basis}" \
    --u-ref-path "${uref}" \
    --qbar-perturb-percent "${PERT}" \
    --qbar-perturb-seed "${SEED}" \
    --output-dir "${outdir}" \
    --run-tag-prefix "${label}" \
    2>&1 | tee "${outdir}/${label}_mu1_${mu1}_mu2_${mu2}.log"
}

for pair in "${MUS[@]}"; do
  read -r MU1 MU2 <<< "${pair}"
  CASE_DIR="${OUTROOT}/mu1_${MU1}_mu2_${MU2}"
  mkdir -p "${CASE_DIR}"

  BASE_NAME="linear_prom_mu1_${MU1}_mu2_${MU2}_ntot151"

  run_case "euclid" "${MU1}" "${MU2}" \
    "${ROOT}/Results_Maday/${TAG}/Runs/Linear_tol/euclid_after_fix/${BASE_NAME}" \
    "${ROOT}/Results_Maday/${TAG}/Stage1_euclid/basis.npy" \
    "${ROOT}/Results_Maday/${TAG}/Stage1_euclid/u_ref.npy" \
    "${CASE_DIR}"

  run_case "weighted" "${MU1}" "${MU2}" \
    "${ROOT}/Results_Maday/${TAG}/Runs/Linear_tol/weighted_after_fix/${BASE_NAME}" \
    "${ROOT}/Results_Maday/${TAG}/Stage1/basis_weighted.npy" \
    "${ROOT}/Results_Maday/${TAG}/Stage1/u_ref_weighted.npy" \
    "${CASE_DIR}"

  run_case "corrected" "${MU1}" "${MU2}" \
    "${ROOT}/Results_Maday/${TAG}/Runs/Linear_tol/corrected_after_fix/${BASE_NAME}" \
    "${ROOT}/Results_Maday/${TAG}/Stage1/basis_corrected_p2_n10_Aavg.npy" \
    "${ROOT}/Results_Maday/${TAG}/Stage1/u_ref_weighted.npy" \
    "${CASE_DIR}"
done

echo "[DONE] Oracle suite finished at: ${OUTROOT}"
