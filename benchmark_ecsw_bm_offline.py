#!/usr/bin/env python3
"""Measure fresh classical and by-mode ECSW training without replacing saved models."""
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
WEIGHTS = {
    'ecsw': 'ecsw_weights_lspg_ecsw_tol1e-10_offset1_snap5pct.npy',
    'bm_ecsw': 'ecsw_weights_lspg_bm_ecsw_tol1e-10_offset1_snap5pct.npy',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_report(path):
    return dict(line.split(': ', 1) for line in path.read_text().splitlines() if ': ' in line)


def benchmark(args):
    out = Path(args.output_dir).resolve()
    cache = REPO / 'Results/param_snaps'
    for mu1 in (4.25, 4.875, 5.5):
        for mu2 in (.015, .0225, .03):
            if not (cache / f'mu1_{mu1}+mu2_{mu2}.npy').is_file():
                raise FileNotFoundError(f'Missing cached training trajectory: {(mu1, mu2)}')
    if not (cache / 'mu1_4.56+mu2_0.019.npy').is_file():
        raise FileNotFoundError('Missing cached online reference')
    inputs = [REPO / 'POD' / name for name in ('basis.npy', 'sigma.npy', 'u_ref.npy', *WEIGHTS.values())]
    hashes = {str(p.relative_to(REPO)): digest(p) for p in inputs}
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    pool_command = [sys.executable, '-c',
                    'import run_hprom, json; from threadpoolctl import threadpool_info; print(json.dumps(threadpool_info()))']
    pools = json.loads(subprocess.check_output(pool_command, cwd=REPO, env=env, text=True))
    if any(p['num_threads'] != args.threads for p in pools):
        raise RuntimeError(f'Unexpected native thread counts: {pools}')
    cpu_model = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name'))
    provenance = {
        'started_local': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'cpu_model': cpu_model, 'native_threads': args.threads, 'native_thread_pools': pools,
        'repetitions_per_method': 1, 'method_order': list(WEIGHTS),
        'basis_size': 96, 'training_pairs': 225, 'random_seed': 42,
        'snapshot_percent': 5., 'time_offset': 1, 'relative_training_tolerance': 1e-5,
        'max_cells': None, 'bm_candidate_score': 'signed_sum',
        'measured_field': 'run_hprom.main ecsw_time_seconds',
        'measured_region': 'configuration selection and sampling plot, cached HDM loading, projected training-matrix assembly, greedy NNLS fitting',
        'excluded': 'HDM generation, POD construction/loading, final report-fit recomputation, saved-model output, online integration, trajectory error and plots',
        'fresh_matrix_and_weights': True, 'inputs_sha256': hashes,
        'sources_sha256': {str(p.relative_to(REPO)): digest(p) for p in (
            REPO / 'run_hprom.py', REPO / 'burgers/ecsw_utils.py',
            REPO / 'burgers/linear_manifold.py', REPO / 'burgers/gauss_newton.py')},
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'threadpoolctl')},
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    records = []
    for method in WEIGHTS:
        run_dir = out / method
        run_dir.mkdir()
        pod = run_dir / 'POD'
        pod.mkdir()
        for name in ('basis.npy', 'sigma.npy', 'u_ref.npy'):
            (pod / name).symlink_to(REPO / 'POD' / name)
        (run_dir / 'param_snaps').symlink_to(cache, target_is_directory=True)
        command = [sys.executable, '-u', str(REPO / 'run_hprom.py'),
                   '--compute-ecsw', '--selection-method', method, '--num-modes', '96',
                   '--snap-time-offset', '1', '--ecsw-snapshot-percent', '5',
                   '--ecsw-random-seed', '42', '--ecsw-tol-squared', '1e-10',
                   '--linear-solver', 'lstsq', '--bm-jacobian', 'gauss_newton',
                   '--bm-candidate-score', 'signed_sum',
                   '--pod-dir', str(pod), '--results-dir', str(run_dir)]
        (run_dir / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        (out / 'status.json').write_text(json.dumps({'completed': False, 'running_method': method}) + '\n')
        print(f'Starting fresh {method} training; isolated outputs: {run_dir}', flush=True)
        wall_start = time.perf_counter()
        with (run_dir / 'runner.log').open('x') as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall_seconds = time.perf_counter() - wall_start
        reports = list(run_dir.glob('*summary*.txt'))
        if len(reports) != 1:
            raise RuntimeError(f'Expected one report in {run_dir}')
        report = read_report(reports[0])
        if report['snapshot_selected_total'] != '225' or report['basis_size'] != '96':
            raise RuntimeError('Unmatched training inputs')
        if report['ecsw_tolerance_met'] != 'True':
            raise RuntimeError('Training did not attain its prescribed tolerance')
        import numpy as np
        from burgers.config import GRID_X, GRID_Y
        from burgers.ecsw_utils import generate_augmented_mesh
        weight_path = Path(report['ecsw_weights_npy'])
        weights = np.load(weight_path)
        cells = np.flatnonzero(np.any(weights > 0, axis=1) if weights.ndim == 2 else weights > 0)
        augmented = generate_augmented_mesh(GRID_X, GRID_Y, cells)
        records.append({
            'method': method, 'training_seconds': float(report['ecsw_time_seconds']),
            'cells': cells.size, 'augmented_cells': augmented.size,
            'positive_weights': np.count_nonzero(weights),
            'training_relative_error': float(report['ecsw_residual']),
            'whole_command_seconds': wall_seconds,
            'report': str(reports[0].relative_to(out)),
            'weights_sha256': digest(weight_path),
            'baseline_weights_sha256': hashes[str((REPO / 'POD' / WEIGHTS[method]).relative_to(REPO))],
        })
        with (out / 'runs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Completed {method}: {records[-1]['training_seconds']:.3f}s offline; {cells.size} cells", flush=True)
    if any(digest(p) != hashes[str(p.relative_to(REPO))] for p in inputs):
        raise RuntimeError('An existing model input changed')
    provenance['completed_local'] = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    ratio = records[0]['training_seconds'] / records[1]['training_seconds']
    lines = ['Fresh ECSW/BM-ECSW offline timings: one run per method, 20 native threads.',
             'Same 96-mode basis, 225 projected state pairs, seed 42 and relative tolerance 1e-5.',
             'Cached training HDM trajectories and the existing POD basis are used; matrices and weights are rebuilt.',
             'Times use the production training timer; the following online run and output processing are excluded.', '']
    for row in records:
        lines.append(f"{row['method']}: {row['training_seconds']:.6f} s ({row['training_seconds']/60:.6f} min), "
                     f"{row['cells']} cells, {row['augmented_cells']} augmented, training error {row['training_relative_error']:.9e}")
    lines.extend(['', f'Classical training time / BM training time: {ratio:.6f}.',
                  'Single-run estimates; original models and online benchmark results are preserved.'])
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    (out / 'status.json').write_text(json.dumps({'completed': True, 'completed_methods': 2}) + '\n')
    print(f'Offline benchmark completed: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--output-dir', default=str(REPO / 'Results/ECSW_BM_offline_benchmark_20261005'))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    benchmark(args)
