#!/usr/bin/env bash
# Submit training and dependent evaluation jobs after Stage 2 has completed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PROJECT_DIR/Project_YvonMaday/Results_Paper}"
SUBMIT_B3="${SUBMIT_B3:-1}"

command -v sbatch >/dev/null || { echo "[error] sbatch is unavailable." >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "[error] Python is not executable: $PYTHON_BIN" >&2; exit 1; }

levels=(lhs8 lhs12 lhs18)
families=(case1 case2 case3 pod_ae pod_dl)
declare -A expected=( [lhs8]=17 [lhs12]=21 [lhs18]=27 )
declare -A train_jobs
declare -A eval_jobs
declare -A b3_jobs

for level in "${levels[@]}"; do
  dataset="$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_${level}/Stage2/prom_coeff_dataset_ntot151_enriched_${level}"
  "$PYTHON_BIN" - "$dataset" "${expected[$level]}" "$level" <<'PY'
import json
import pathlib
import sys

dataset = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
level = sys.argv[3]
meta_path = dataset / "meta.json"
if not meta_path.is_file():
    raise SystemExit(f"Missing completed Stage-2 metadata: {meta_path}")
meta = json.loads(meta_path.read_text())
count = len([path for path in (dataset / "per_mu").iterdir() if path.is_dir()])
if meta.get("solve_backend") != "hprom" or meta.get("subset_level") != level:
    raise SystemExit(f"Wrong Stage-2 provenance in {meta_path}")
if int(meta.get("num_traj", -1)) != expected or count != expected:
    raise SystemExit(f"Incomplete {level}: metadata={meta.get('num_traj')}, directories={count}, expected={expected}")
print(f"Stage 2 verified: {level}, {count} HPROM trajectories")
PY
done

mkdir -p "$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_submit_logs"
SBATCH_EXPORT="ALL,PYTHON_BIN=$PYTHON_BIN,PAPER_RESULTS_ROOT=$PAPER_RESULTS_ROOT"

for level in "${levels[@]}"; do
  root="$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_${level}"
  mkdir -p "$root/logs/slurm"
  for family in "${families[@]}"; do
    key="${level}_${family}"
    train_jobs[$key]="$(sbatch --parsable \
      --job-name="ehtr_${level}_${family}" \
      --output="$root/logs/slurm/train_${family}_%j.out" \
      --error="$root/logs/slurm/train_${family}_%j.err" \
      --export="$SBATCH_EXPORT" \
      Project_YvonMaday/slurm/euclidean_hprom_enrichment_train.sbatch "$level" "$family")"
    echo "submitted train $level $family: ${train_jobs[$key]}"
  done
done

for level in "${levels[@]}"; do
  root="$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_${level}"
  for family in "${families[@]}"; do
    key="${level}_${family}"
    eval_jobs[$key]="$(sbatch --parsable \
      --dependency="afterok:${train_jobs[$key]}" \
      --job-name="ehev_${level}_${family}" \
      --output="$root/logs/slurm/eval_${family}_%j.out" \
      --error="$root/logs/slurm/eval_${family}_%j.err" \
      --export="$SBATCH_EXPORT" \
      Project_YvonMaday/slurm/euclidean_hprom_enrichment_evaluate.sbatch "$level" "$family")"
    echo "submitted evaluate $level $family after ${train_jobs[$key]}: ${eval_jobs[$key]}"
  done
done

if [[ "$SUBMIT_B3" == 1 ]]; then
  for level in "${levels[@]}"; do
    root="$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_${level}"
    key="${level}_case2"
    b3_jobs[$level]="$(sbatch --parsable \
      --dependency="afterok:${train_jobs[$key]}" \
      --job-name="ehb3_${level}" \
      --output="$root/logs/slurm/b3_%j.out" \
      --error="$root/logs/slurm/b3_%j.err" \
      --export="$SBATCH_EXPORT" \
      Project_YvonMaday/slurm/euclidean_hprom_enrichment_b3.sbatch "$level")"
    echo "submitted B3 $level after ${train_jobs[$key]}: ${b3_jobs[$level]}"
  done
fi

summary="$PAPER_RESULTS_ROOT/euclidean_hprom_enrichment_submit_logs/submission_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "python=$PYTHON_BIN"
  echo "all jobs: cfarhat, 1 exclusive node, 24 tasks per node, 24:00:00"
  echo "training and ECM construction: 24 threads; learned online trajectories: 1 thread"
  echo "B3: enabled=$SUBMIT_B3; rule construction=24 threads, online=1 thread"
  for key in "${!train_jobs[@]}"; do echo "train $key ${train_jobs[$key]}"; done | sort
  for key in "${!eval_jobs[@]}"; do echo "evaluate $key ${eval_jobs[$key]}"; done | sort
  for key in "${!b3_jobs[@]}"; do echo "b3 $key ${b3_jobs[$key]}"; done | sort
} | tee "$summary"
echo "Submission manifest: $summary"
