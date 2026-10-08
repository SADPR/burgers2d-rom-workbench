#!/usr/bin/env python3
"""Measure one fresh production FOM trajectory at each comparison parameter."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
POINTS = [(4.56, .019), (4.75, .020), (5.19, .026)]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def worker(args):
    import numpy as np
    from threadpoolctl import threadpool_info
    from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0
    from burgers.core import inviscid_burgers_implicit2D

    pools = threadpool_info()
    if any(pool['num_threads'] != args.threads for pool in pools):
        raise RuntimeError(f'Unexpected numerical-library thread configuration: {pools}')
    start = time.perf_counter()
    states = inviscid_burgers_implicit2D(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, [args.mu1, args.mu2],
    )
    seconds = time.perf_counter() - start
    # Output and trajectory comparisons are outside the timed region.
    if states.shape != (125000, 501) or not np.all(np.isfinite(states)):
        raise RuntimeError('Invalid FOM trajectory')
    cache = REPO/'Results/param_snaps'/f'mu1_{args.mu1}+mu2_{args.mu2}.npy'
    reference = np.load(cache, mmap_mode='r')
    difference = float(np.linalg.norm(states-reference)/np.linalg.norm(reference))
    report = dict(mu1=args.mu1, mu2=args.mu2, fom_seconds=seconds,
                  num_steps=NUM_STEPS, full_state_size=states.shape[0],
                  relative_difference_to_saved_fom=difference,
                  final_state_min=float(states[:, -1].min()),
                  final_state_max=float(states[:, -1].max()),
                  native_thread_pools=pools,
                  measured_region='production inviscid_burgers_implicit2D: operator setup and 500 steps',
                  excluded='Python startup, imports, output files, and reference comparison',
                  max_newton_iterations=100, relative_residual_tolerance=1e-12)
    Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
    print(f'COMPLETED {(args.mu1, args.mu2)}: {seconds:.6f} seconds; saved-FOM difference {difference:.3e}', flush=True)


def benchmark(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    started = datetime.now(timezone.utc).isoformat()
    hashes = {str(path.relative_to(REPO)): digest(path) for path in
              (REPO/'burgers/core.py', REPO/'burgers/gauss_newton.py', REPO/'burgers/config.py')}
    records = []
    for a, b in POINTS:
        stem = f'fom_mu1_{a:.2f}_mu2_{b:.3f}'
        report = out/f'{stem}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--mu1', str(a), '--mu2', str(b), '--report', str(report),
                   '--threads', str(args.threads)]
        print(f'Starting fresh FOM trajectory at {(a,b)}', flush=True)
        with (out/f'{stem}.log').open('w') as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        row = json.loads(report.read_text())
        log_text = (out/f'{stem}.log').read_text()
        converged_steps = len(re.findall(r'^\d+: [0-9.eE+-]+$', log_text, flags=re.M))
        if converged_steps != 500:
            raise RuntimeError(f'Only {converged_steps}/500 FOM steps reported convergence')
        row['converged_steps'] = converged_steps
        report.write_text(json.dumps(row, indent=2)+'\n')
        records.append({key: row[key] for key in
                        ('mu1', 'mu2', 'fom_seconds', 'relative_difference_to_saved_fom')})
        with (out/'runs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        print(f'Completed {(a,b)}: {row["fom_seconds"]:.6f} seconds', flush=True)
    average = sum(row['fom_seconds'] for row in records)/len(records)
    (out/'provenance.json').write_text(json.dumps(dict(
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        threads=args.threads, repeats_per_parameter=1, mean_fom_seconds=average,
        input_sha256=hashes, measured_region='production FOM function including setup and time stepping',
    ), indent=2)+'\n')
    print(f'Completed all three FOM trajectories; mean {average:.6f} seconds. Reports: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--output-dir', default='Results/FOM_benchmark_20261004')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mu1', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--mu2', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--report', help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args) if args.worker else benchmark(args)
