#!/usr/bin/env python3
"""Train an independent PSD patch rule from zero, then compare LSPG HPROMs.

The original mesh is partitioned into 2x2 candidate patches. Selection uses
all original-mesh PSD correlations, with adaptive-rank factor refits. Existing
ECSW and BM models enter only the subsequent online comparison.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent


def train(args):
    import numpy as np
    from threadpoolctl import threadpool_info
    from burgers.config import GRID_X, GRID_Y, DT, NUM_STEPS
    from burgers.core import get_snapshot_params, get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
    from burgers.ecsw_utils import build_ecsw_snapshot_plan, generate_augmented_mesh
    from burgers.patch_metric import regular_mesh_groups, greedy_patch_metrics, PatchResidualMetric
    from run_residual_metric_experiment import load_model, save_json
    out = Path(args.output_dir)
    started = time.perf_counter()
    basis, ref = load_model()
    nc = basis.shape[0]//2
    parameters = get_snapshot_params()
    plan = build_ecsw_snapshot_plan(num_steps=NUM_STEPS, snap_time_offset=1,
        num_mu=len(parameters), mode='global_param_time_stratified',
        total_snapshots_percent=5., random_seed=42, ensure_mu_coverage=True, mu_points=parameters)
    count = plan['num_selected_total']
    if count != 225:
        raise ValueError('Expected the matched 225-configuration training plan.')
    save_json(out/'training_plan.json', dict(parameters=parameters, **plan))
    cache = out/'training_data'
    cache.mkdir(exist_ok=False)
    residuals = np.lib.format.open_memmap(cache/'residuals.npy', mode='w+', dtype=np.float64,
                                         shape=(count, nc, 2))
    tangents = np.lib.format.open_memmap(cache/'tangents.npy', mode='w+', dtype=np.float64,
                                       shape=(count, nc, 2, basis.shape[1]))
    target = np.zeros((count, basis.shape[1]))
    D_x, D_y, JD_x, JD_y, eye = get_ops(GRID_X, GRID_Y)
    configuration = 0
    for mu, now in zip(parameters, plan['selected_now_cols_by_mu']):
        now = np.asarray(now, dtype=int)
        snapshots = np.load(REPO/'Results/param_snaps'/f'mu1_{mu[0]}+mu2_{mu[1]}.npy', mmap_mode='r')
        coords = np.linalg.lstsq(basis, snapshots[:, np.concatenate((now, now-1))]-ref[:, None], rcond=None)[0]
        for j in range(len(now)):
            current = ref+basis @ coords[:, j]
            previous = ref+basis @ coords[:, j+len(now)]
            r = inviscid_burgers_res2D(current, GRID_X, GRID_Y, DT, previous, mu, D_x, D_y)
            W = inviscid_burgers_exact_jac2D(current, DT, JD_x, JD_y, eye) @ basis
            residuals[configuration, :, 0], residuals[configuration, :, 1] = r[:nc], r[nc:]
            tangents[configuration, :, 0], tangents[configuration, :, 1] = W[:nc], W[nc:]
            target[configuration] = W.T @ r
            recovered = np.einsum('ean,ea->n', tangents[configuration], residuals[configuration])
            if not np.allclose(recovered, target[configuration], rtol=1e-10, atol=1e-10):
                raise RuntimeError('Local blocks do not reproduce the full projected residual.')
            configuration += 1
        save_json(out/'status.json', dict(stage='assembly', configurations=configuration,
            total_configurations=count, assembly_seconds=time.perf_counter()-started,
            threads=args.threads, initialization='no selected patches or trained weights'))
        print(f'Assembly: {configuration}/{count} configurations', flush=True)
    residuals.flush()
    tangents.flush()
    np.save(cache/'target.npy', target.reshape(-1))
    assembly_seconds = time.perf_counter()-started
    shape = (len(GRID_Y)-1, len(GRID_X)-1)
    groups = regular_mesh_groups(shape, side=2)
    if len({len(g) for g in groups}) != 1:
        raise ValueError('This first batched experiment requires an even grid.')
    save_json(out/'candidate_groups.json', dict(mesh_shape=shape, patch_shape=[2, 2],
        candidates=len(groups), definition='all disjoint 2x2 blocks of the original mesh'))
    def progress(row):
        visible=dict(row)
        if 'selection_iteration' in visible:
            visible['factor_iterations']=visible.get('iteration',0)
            visible['iteration']=visible['selection_iteration']
        else:
            visible['selection_iteration']=visible.get('iteration',0)
        save_json(out/'status.json', dict(**visible, threads=args.threads, tolerance=args.tolerance,
            assembly_seconds=assembly_seconds,
            initialization='empty support; independent original-mesh patch selection'))
        with (out/'training_history.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if row['stage'] != 'factor_fit':
            print(json.dumps(row), flush=True)
    def checkpoint(active_groups, factors, row):
        cells = np.sort(np.concatenate(active_groups))
        metric = PatchResidualMetric(cells, [g.copy() for g in active_groups], factors)
        temporary = out/'patch_metric.tmp.npz'
        metric.save(temporary)
        temporary.replace(out/'patch_metric.npz')
        save_json(out/'checkpoint.json', row)
    metric, report = greedy_patch_metrics(residuals, tangents, target.reshape(-1), groups,
        relative_tolerance=args.tolerance, initial_rank=2, kkt_tolerance=1e-8,
        max_seconds=args.selection_seconds, progress=progress, checkpoint=checkpoint)
    metric.save(out/'patch_metric.npz')
    report.update(assembly_seconds=assembly_seconds, offline_seconds=time.perf_counter()-started,
        augmented_cells=len(generate_augmented_mesh(GRID_X, GRID_Y, metric.selected_cells)),
        transformed_residual_rows=metric.output_rows, modes=basis.shape[1],
        native_thread_pools=threadpool_info(), threads=args.threads)
    # Matrix properties at three training configurations, without integration.
    checks = []
    for c in (0, count//2, count-1):
        local_W = tangents[c, metric.selected_cells]
        rows = np.concatenate((local_W[:, 0], local_W[:, 1]))
        transformed = metric.apply(rows)
        singular_values = np.linalg.svd(transformed, compute_uv=False)
        threshold = max(transformed.shape)*np.finfo(float).eps*singular_values[0]
        K = transformed.T @ transformed
        checks.append(dict(configuration=c, transformed_rank=int(np.sum(singular_values > threshold)),
            smallest_singular_value=float(singular_values[-1]),
            correction_asymmetry=float(np.linalg.norm(K-K.T)/max(np.linalg.norm(K), 1e-300)),
            smallest_correction_eigenvalue=float(np.linalg.eigvalsh(K)[0])))
    report['matrix_checks'] = checks
    save_json(out/'training.json', report)
    save_json(out/'status.json', dict(stage='trained', **{k:v for k, v in report.items() if k != 'history'}))
    print(f"Training: {report['cells']} cells; error={report['relative_training_error']:.9e}; {report['stop_reason']}", flush=True)


def online(args):
    import contextlib
    import numpy as np
    from burgers.patch_metric import PatchResidualMetric
    from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS
    from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
    from run_residual_metric_experiment import load_model, save_json
    out = Path(args.output_dir)
    basis, ref = load_model()
    metric = PatchResidualMetric.load(out/'patch_metric.npz')
    weights = np.zeros(basis.shape[0]//2)
    weights[metric.selected_cells] = 1.
    stem = f'patch_psd_mu1_{args.mu1}_mu2_{args.mu2}'
    with (out/f'{stem}.log').open('w') as log, contextlib.redirect_stdout(log):
        started = time.perf_counter()
        coords, stats = inviscid_burgers_implicit2D_LSPG_ecsw(GRID_X, GRID_Y, weights,
            W0, DT, NUM_STEPS, [args.mu1, args.mu2], basis, u_ref=ref,
            max_its=20, relnorm_cutoff=1e-5, min_delta=.01, linear_solver='lstsq',
            residual_transform=metric)
        seconds = time.perf_counter()-started
    if not np.isfinite(coords).all():
        raise RuntimeError('Nonfinite patch-HPROM trajectory.')
    np.save(out/f'{stem}_coordinates.npy', coords)
    truth = np.load(REPO/'Results/param_snaps'/f'mu1_{args.mu1}+mu2_{args.mu2}.npy', mmap_mode='r')
    squared_difference, squared_truth, state_errors = 0., 0., []
    for start in range(0, coords.shape[1], 25):
        stop = min(start+25, coords.shape[1])
        reconstructed = ref[:, None]+basis @ coords[:, start:stop]
        difference = reconstructed-truth[:, start:stop]
        squared_difference += float(np.sum(difference*difference))
        squared_truth += float(np.sum(truth[:, start:stop]**2))
        state_errors.extend(np.linalg.norm(difference, axis=0)/np.linalg.norm(truth[:, start:stop], axis=0))
    np.save(out/f'{stem}_trajectory_errors.npy', state_errors)
    training = json.loads((out/'training.json').read_text())
    row = dict(method='patch_psd', mu1=args.mu1, mu2=args.mu2, cells=metric.selected_cells.size,
        groups=len(metric.groups), augmented_cells=training['augmented_cells'],
        online_seconds=seconds, error_percent=100*np.sqrt(squared_difference/squared_truth),
        max_snapshot_error_percent=100*max(state_errors), iterations=int(stats[0]),
        jacobian_seconds=stats[1], residual_seconds=stats[2], linear_solve_seconds=stats[3],
        training_tolerance_met=training['tolerance_met'])
    save_json(out/f'{stem}.json', row)
    print(json.dumps(row), flush=True)


def experiment(args):
    from run_residual_metric_experiment import save_json, file_hash, PARAMETERS
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(args.output_dir).resolve() if args.output_dir else REPO/'Results'/f'ECSW_patch_PSD_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(out/'mplcache')
    sources = ['POD/basis.npy', 'POD/u_ref.npy', 'burgers/patch_metric.py',
               'burgers/linear_manifold.py', 'burgers/gauss_newton.py', Path(__file__).name,
               'burgers/config.py', 'burgers/core.py', 'burgers/ecsw_utils.py']
    metadata = dict(started_utc=stamp, threads=args.threads, tolerance=args.tolerance,
        selection_time_budget_seconds=args.selection_seconds,
        initialization='from zero; existing cubature supports and weights are not read during training',
        candidates='all original-mesh disjoint 2x2 cell patches',
        refit='adaptive-rank PSD factors; original-row error and convex KKT checks',
        sources={name:file_hash(REPO/name) for name in sources})
    save_json(out/'provenance.json', metadata)
    save_json(out/'status.json', dict(stage='starting', **metadata))
    copies = out/'source'
    copies.mkdir()
    for name in sources[2:]:
        shutil.copyfile(REPO/name, copies/Path(name).name)
    base = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(out),
            '--threads', str(args.threads), '--tolerance', str(args.tolerance),
            '--selection-seconds', str(args.selection_seconds)]
    with (out/'training.log').open('w') as log:
        result = subprocess.run(base+['--stage', 'train'], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        previous = json.loads((out/'status.json').read_text())
        save_json(out/'status.json', dict(stage='failed', exit_code=result.returncode,
                                         last_progress=previous, log=str(out/'training.log')))
        raise RuntimeError(f'Patch training failed; see {out}/training.log')
    report = json.loads((out/'training.json').read_text())
    rank_ok = all(row['transformed_rank'] == report['modes'] for row in report['matrix_checks'])
    if not report['tolerance_met'] or not rank_ok:
        reason = report['stop_reason'] if not report['tolerance_met'] else 'rank-deficient online correction'
        lines = ['# Independent PSD patch training: incomplete', '',
            f"Relative training error: {report['relative_training_error']:.9e}; requested tolerance: {args.tolerance:.1e}.",
            f"Cells: {report['cells']}; groups: {report['groups']}; stop reason: {reason}.",
            'Online comparison was not run: training must attain tolerance and sampled checks must have full reduced rank.',
            'This execution does not establish whether the proposed method succeeds or fails online.']
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        save_json(out/'status.json', dict(stage='training_incomplete', tolerance_met=report['tolerance_met'],
            cells=report['cells'], groups=report['groups'],
            relative_training_error=report['relative_training_error'], tolerance=args.tolerance,
            stop_reason=reason, online_comparison_run=False, full_reduced_rank=rank_ok,
            summary=str(out/'summary.md')))
        print(f'Training incomplete: {out}; {reason}', flush=True)
        return
    records = []
    with (out/'online_worker.log').open('w') as log:
        for mu1, mu2 in PARAMETERS:
            for method in ('patch_psd', 'ecsw', 'bm_ecsw'):
                save_json(out/'status.json', dict(stage='online', method=method, mu1=mu1, mu2=mu2,
                    completed_runs=len(records), total_runs=9, threads=args.threads))
                if method == 'patch_psd':
                    command = base+['--stage', 'online', '--mu1', str(mu1), '--mu2', str(mu2)]
                else:
                    command = [sys.executable, str(REPO/'run_residual_metric_experiment.py'),
                        '--stage', 'online', '--output-dir', str(out), '--threads', str(args.threads),
                        '--method', method, '--mu1', str(mu1), '--mu2', str(mu2)]
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = json.loads((out/f'{method}_mu1_{mu1}_mu2_{mu2}.json').read_text())
                records.append(row)
                save_json(out/'online_results.json', records)
    fields = ['method', 'mu1', 'mu2', 'cells', 'augmented_cells', 'online_seconds', 'error_percent',
              'max_snapshot_error_percent', 'iterations']
    with (out/'online_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    report = json.loads((out/'training.json').read_text())
    lines = ['# Independent adaptive-rank PSD patch experiment', '',
        'Training starts with no selected cells or patches. Every original-mesh 2x2 candidate is scored.',
        'All active PSD factors are refitted; directions are not frozen. Selection requires numerical KKT convergence of the active fit.', '',
        f"Training error: {report['relative_training_error']:.9e}; tolerance attained: {report['tolerance_met']}.",
        f"Stop reason: {report['stop_reason']}.",
        f"Sampled cells: {report['cells']}; patches: {report['groups']}; augmented cells: {report['augmented_cells']}.",
        f"Assembly: {report['assembly_seconds']:.3f} s; selection/refits: {report['training_seconds']:.3f} s.", '',
        '| Parameter | Method | Cells | Online (s) | Trajectory error (%) |',
        '|---|---|---:|---:|---:|']
    for row in records:
        lines.append(f"| ({row['mu1']}, {row['mu2']}) | {row['method']} | {row['cells']} | {row['online_seconds']:.3f} | {row['error_percent']:.6f} |")
    lines += ['', 'If training stopped early, patch online results evaluate an incomplete rule; they are not a tolerance-matched comparison.',
        'Single-run timings with ten or fewer native threads; concurrent AERO-F work can affect cost.',
        'The convex KKT checks apply to active groups only; they do not certify minimum mesh cardinality.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'status.json', dict(stage='complete', summary=str(out/'summary.md'),
        cells=report['cells'], relative_training_error=report['relative_training_error'], tolerance_met=report['tolerance_met']))
    print(f'Completed: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir')
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--selection-seconds', type=float, default=1800.)
    parser.add_argument('--stage', choices=['train', 'online'])
    parser.add_argument('--mu1', type=float)
    parser.add_argument('--mu2', type=float)
    args = parser.parse_args()
    if not 1 <= args.threads <= 10:
        parser.error('Use 1 to 10 threads to leave capacity for the concurrent AERO-F work.')
    if args.stage == 'train':
        train(args)
    elif args.stage == 'online':
        online(args)
    else:
        experiment(args)


if __name__ == '__main__':
    main()
