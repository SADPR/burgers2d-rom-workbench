#!/usr/bin/env bash
# Run with bash inside an existing compute-node allocation and active myenv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SLURM_JOB_ID:?Run inside your allocated Sherlock compute node}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/case2-hyper-mpl-${SLURM_JOB_ID}"

sha256sum --check Project_YvonMaday/case2_hyper_inputs.sha256
"$PYTHON_BIN" - <<'PY'
import os
import socket
import Project_YvonMaday.run_case2_hyperreduction
affinity = os.sched_getaffinity(0)
if len(affinity) < 24:
    raise RuntimeError(f"Only {len(affinity)} logical CPUs available; this protocol requires 24")
if socket.gethostname().split('.')[0].startswith('login'):
    raise RuntimeError("Run on the allocated compute node")
print(f"Active Python and solver imports OK; CPU affinity permits {len(affinity)} CPUs")
PY

OUT="$(mktemp -d "$PWD/Project_YvonMaday/Results_Paper/euclidean_case2_hyper_sherlock_${SLURM_JOB_ID}_XXXXXX")"
export CASE2_HYPER_RUN_DIR="$OUT"
"$PYTHON_BIN" - <<'PY'
import json
import os
import platform
import socket
import sys
from pathlib import Path
import numpy
import scipy
import torch
from threadpoolctl import threadpool_info
result = dict(hostname=socket.gethostname(), job=os.environ['SLURM_JOB_ID'],
              cpu_affinity=sorted(os.sched_getaffinity(0)), python=sys.executable,
              platform=platform.platform(), numpy=numpy.__version__, scipy=scipy.__version__,
              torch=str(torch.__version__), threadpools=threadpool_info(),
              numerical_threads=24, repetitions=1, warmups=0, total_trajectories=8)
(Path(os.environ['CASE2_HYPER_RUN_DIR'])/'environment.json').write_text(json.dumps(result, indent=2)+'\n')
PY

echo "Results: $OUT"
echo "Eight trajectories: two HPROM methods x four parameters x one run, 24 numerical threads."
"$PYTHON_BIN" -u Project_YvonMaday/run_case2_hyperreduction.py reporting \
  --output "$OUT" --threads 24 --methods hprom0 hprom3 \
  --rule-file "$PWD/Project_YvonMaday/Results_Paper/euclidean_case2_hyperreduction/positive_fit4096/weights.npy" \
  2>&1 | tee "$OUT/benchmark.log"

"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path
root = Path(os.environ['CASE2_HYPER_RUN_DIR'])
rows = []
for path in sorted(root.glob('reporting_*/summary.json')):
    data = json.loads(path.read_text())
    rows.append(dict(method=data['config']['method'], mu1=data['config']['mu'][0],
                     mu2=data['config']['mu'][1], state_error_percent=data['state_error_percent'],
                     coefficient_error_percent=data['coefficient_error_percent'],
                     online_seconds=data['online_seconds'], sampled_cells=data['sampled_cells'],
                     stencil_cells=data['stencil_cells']))
if len(rows) != 8:
    raise RuntimeError(f"Expected eight completed runs; found {len(rows)}")
with (root/'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
lines = ['One run per point; times use 24 numerical threads.']
for method in ['hprom0', 'hprom3']:
    inside = [row for row in rows if row['method'] == method and row['mu1'] != 4.0]
    lines.append(f"{method}: in-domain mean online time = {sum(r['online_seconds'] for r in inside)/3:.6f} s")
(root/'comparison.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo "Complete: $OUT"
