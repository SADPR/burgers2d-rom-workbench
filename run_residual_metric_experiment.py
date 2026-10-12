#!/usr/bin/env python3
"""Train PSD residual-block cubature and compare three global LSPG HPROMs.

Default: 96 modes, 225 pairs, relative training tolerance 1e-5, 10 threads.
Each online measurement uses a fresh process and the existing 500-step case.
No existing weights, results, or manuscript files are overwritten.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
PARAMETERS = [(4.56, .019), (4.75, .020), (5.19, .026)]
BASELINES = {
    'ecsw': 'ecsw_weights_lspg_ecsw_tol1e-10_offset1_snap5pct.npy',
    'bm_ecsw': 'ecsw_weights_lspg_bm_ecsw_tol1e-10_offset1_snap5pct.npy',
}


def save_json(path, value):
    def convert(x):
        return x.tolist() if hasattr(x, 'tolist') else str(x)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=convert)+'\n')
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_model():
    import numpy as np
    basis = np.load(REPO/'POD/basis.npy')
    ref = np.load(REPO/'POD/u_ref.npy')
    if basis.shape != (125000, 96):
        raise ValueError(f'Unexpected matched-comparison basis: {basis.shape}')
    return basis, ref


def train(args):
    import numpy as np
    from threadpoolctl import threadpool_info
    from burgers.config import GRID_X, GRID_Y, DT, NUM_STEPS
    from burgers.core import (get_snapshot_params, get_ops,
                              inviscid_burgers_res2D, inviscid_burgers_exact_jac2D)
    from burgers.ecsw_utils import build_ecsw_snapshot_plan, generate_augmented_mesh
    from burgers.residual_metric import greedy_psd_metric_weights
    out = Path(args.output_dir)
    started = time.perf_counter()
    basis, ref = load_model()
    parameters = get_snapshot_params()
    plan = build_ecsw_snapshot_plan(num_steps=NUM_STEPS, snap_time_offset=1,
        num_mu=len(parameters), mode='global_param_time_stratified',
        total_snapshots_percent=5., random_seed=42, ensure_mu_coverage=True,
        mu_points=parameters)
    if plan['num_selected_total'] != 225:
        raise ValueError('Expected the original 225-pair training plan.')
    save_json(out/'training_plan.json', {'parameters': parameters, **plan})
    data = out/'training_matrices'
    data.mkdir(exist_ok=False)
    shape = (plan['num_selected_total'] * basis.shape[1], basis.shape[0]//2)
    blocks = [np.lib.format.open_memmap(data/f'G{name}.npy', mode='w+',
              dtype=np.float64, shape=shape, fortran_order=True) for name in ('11', '12', '22')]
    D_x, D_y, JD_x, JD_y, eye = get_ops(GRID_X, GRID_Y)
    ncell = shape[1]
    target = np.zeros(shape[0])
    offset = 0
    for mu, now in zip(parameters, plan['selected_now_cols_by_mu']):
        now = np.asarray(now, dtype=int)
        path = REPO/'Results/param_snaps'/f'mu1_{mu[0]}+mu2_{mu[1]}.npy'
        snaps = np.load(path, mmap_mode='r')
        columns = np.concatenate((now, now-1))
        # Same least-squares projection as the production training builder,
        # solving the multiple right-hand sides together.
        coords = np.linalg.lstsq(basis, snaps[:, columns] - ref[:, None], rcond=None)[0]
        for j in range(len(now)):
            current = ref + basis @ coords[:, j]
            previous = ref + basis @ coords[:, j+len(now)]
            residual = inviscid_burgers_res2D(current, GRID_X, GRID_Y, DT,
                                             previous, mu, D_x, D_y)
            W = inviscid_burgers_exact_jac2D(current, DT, JD_x, JD_y, eye) @ basis
            rows = slice(offset, offset+basis.shape[1])
            ru, rv = residual[:ncell], residual[ncell:]
            Wu, Wv = W[:ncell], W[ncell:]
            blocks[0][rows] = (ru[:, None]*Wu).T
            blocks[1][rows] = (rv[:, None]*Wu + ru[:, None]*Wv).T
            blocks[2][rows] = (rv[:, None]*Wv).T
            target[rows] = W.T @ residual
            offset += basis.shape[1]
        print(f'[assembly] mu={mu}, pairs={offset//basis.shape[1]}/225', flush=True)
        save_json(out/'status.json', {'stage':'assembly', 'pairs':offset//basis.shape[1],
            'total_pairs':225, 'threads':args.threads, 'tolerance':args.tolerance})
    for block in blocks:
        block.flush()
    assembly_seconds = time.perf_counter()-started
    # Classical A_e = xi_e I is contained in this residual-space family.
    identity_target = blocks[0].sum(axis=1) + blocks[2].sum(axis=1)
    identity_relative_error = float(np.linalg.norm(identity_target-target)/np.linalg.norm(target))
    if identity_relative_error > 1e-12:
        raise RuntimeError('The matrix-weighted assembly does not recover full LSPG.')
    scalar_weights = np.load(REPO/'POD'/BASELINES['ecsw'])
    classic_prediction = blocks[0] @ scalar_weights + blocks[2] @ scalar_weights
    baseline_training_error = float(np.linalg.norm(classic_prediction-target)/np.linalg.norm(target))
    np.save(out/'target.npy', target)
    print(f'[assembly] {assembly_seconds:.2f}s, identity check={identity_relative_error:.3e}, '
          f'existing classical training error={baseline_training_error:.9e}', flush=True)
    def progress(row):
        save_json(out/'status.json', {'stage':'selection', 'threads':args.threads,
            'tolerance':args.tolerance, **row})
        with (out/'selection_history.jsonl').open('a') as history:
            history.write(json.dumps(row)+'\n')
    def checkpoint(selected, scales, coefficients, target_norm):
        temp = out/'checkpoint.tmp.npz'
        np.savez(temp, selected_cells=selected, column_scales=scales,
                 cone_coordinates=coefficients, target_norm=target_norm,
                 joint_refits=len(selected))
        temp.replace(out/'checkpoint.npz')
    matrices, report = greedy_psd_metric_weights(blocks, target,
        relative_tolerance=args.tolerance, progress=progress, threads=args.threads,
        checkpoint=checkpoint)
    np.save(out/'residual_metric_weights.npy', matrices)
    selected = np.flatnonzero(np.trace(matrices, axis1=1, axis2=2)>0)
    eigs = np.linalg.eigvalsh(matrices[selected])
    report.update(assembly_seconds=assembly_seconds,
        offline_seconds=time.perf_counter()-started,
        augmented_cells=len(generate_augmented_mesh(GRID_X, GRID_Y, selected)),
        min_eigenvalue=float(eigs.min()) if eigs.size else 0.,
        off_diagonal_cells=int(np.count_nonzero(matrices[:,0,1])),
        identity_relative_error=identity_relative_error,
        existing_classical_training_error=baseline_training_error,
        native_thread_pools=threadpool_info(), training_shape=shape,
        projection='least-squares, batched multiple right-hand sides',
        greedy_rule='maximum PSD correlation; one cell added before every full joint SOC refit')
    save_json(out/'training.json', report)
    save_json(out/'status.json', {'stage':'trained', **{k:v for k,v in report.items() if k!='history'}})
    print(f'[training complete] {report["cells"]} cells, '
          f'error={report["relative_training_error"]:.9e}, met={report["tolerance_met"]}', flush=True)


def online(args):
    import contextlib
    import numpy as np
    from threadpoolctl import threadpool_info
    from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS
    from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
    from burgers.ecsw_utils import generate_augmented_mesh
    out = Path(args.output_dir)
    basis, ref = load_model()
    method = args.method
    weights = np.load(out/'residual_metric_weights.npy' if method=='psd_metric'
                      else REPO/'POD'/BASELINES[method])
    support = (np.trace(weights, axis1=1, axis2=2)>0 if weights.ndim==3
               else np.any(weights>0, axis=1) if weights.ndim==2 else weights>0)
    mu = [args.mu1, args.mu2]
    stem = f'{method}_mu1_{args.mu1}_mu2_{args.mu2}'
    with (out/f'{stem}.log').open('w') as log, contextlib.redirect_stdout(log):
        started = time.perf_counter()
        coords, stats = inviscid_burgers_implicit2D_LSPG_ecsw(
            GRID_X, GRID_Y, weights, W0, DT, NUM_STEPS, mu, basis,
            u_ref=ref, max_its=20, relnorm_cutoff=1e-5, min_delta=1e-2,
            linear_solver='lstsq', bm_jacobian='gauss_newton')
        seconds = time.perf_counter()-started
    if not np.all(np.isfinite(coords)):
        raise RuntimeError(f'Nonfinite online trajectory: {stem}')
    np.save(out/f'{stem}_coordinates.npy', coords)
    decoded = ref[:,None]+basis@coords
    hdm = np.load(REPO/'Results/param_snaps'/f'mu1_{args.mu1}+mu2_{args.mu2}.npy', mmap_mode='r')
    relative_error = float(np.linalg.norm(decoded-hdm)/np.linalg.norm(hdm))
    trajectory_errors = np.linalg.norm(decoded-hdm,axis=0)/np.linalg.norm(hdm,axis=0)
    np.save(out/f'{stem}_trajectory_errors.npy', trajectory_errors)
    row = dict(method=method, mu1=args.mu1, mu2=args.mu2,
        cells=int(np.count_nonzero(support)),
        augmented_cells=len(generate_augmented_mesh(GRID_X,GRID_Y,np.flatnonzero(support))),
        online_seconds=seconds, error_percent=100*relative_error,
        max_snapshot_error_percent=100*float(trajectory_errors.max()),
        iterations=int(stats[0]), jacobian_seconds=stats[1], residual_seconds=stats[2],
        linear_solve_seconds=stats[3], native_thread_pools=threadpool_info())
    if method=='bm_ecsw':
        row.update(projected_tolerance_not_met_steps=stats[4], stop_reasons=stats[7])
    save_json(out/f'{stem}.json', row)
    print(f'[online] {stem}: {seconds:.6f}s, error={row["error_percent"]:.8f}%', flush=True)


def experiment(args):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(args.output_dir).resolve() if args.output_dir else REPO/'Results'/f'ECSW_BM_PSD_metric_{stamp}'
    out.mkdir(exist_ok=False, parents=True)
    save_json(out/'status.json', {'stage':'starting', 'threads':args.threads,
        'tolerance':args.tolerance, 'training':'from zero; one full joint refit per added cell'})
    sources = [REPO/'POD'/name for name in ('basis.npy','u_ref.npy',*BASELINES.values())]
    hashes = {str(p.relative_to(REPO)):file_hash(p) for p in sources}
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = str(args.threads)
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(out/'mplcache')
    provenance = dict(started_utc=stamp, requested_threads=args.threads,
        tolerance=args.tolerance, input_sha256=hashes,
        hypothesis='PSD residual-space metrics guarantee a common sampled least-squares objective',
        training='from zero; 96 modes, 225 projected consecutive pairs; seed 42',
        online_protocol='one fresh-process run per method and parameter; 500 steps; same timer as production',
        timing_limit='single-run timings are indicative, not a repeatability study')
    save_json(out/'provenance.json', provenance)
    source_dir=out/'source'
    source_dir.mkdir()
    for name in ('burgers/residual_metric.py','burgers/gauss_newton.py',
                 'burgers/linear_manifold.py','run_residual_metric_experiment.py'):
        shutil.copyfile(REPO/name,source_dir/Path(name).name)
    provenance['versions']={name:importlib.metadata.version(name)
                            for name in ('numpy','scipy','clarabel','threadpoolctl')}
    provenance['training_refit']='all three entries free under PSD; Clarabel; full joint refit after every added cell'
    save_json(out/'provenance.json',provenance)
    print(f'Experiment directory: {out}', flush=True)
    base = [sys.executable, str(Path(__file__).resolve()), '--output-dir',str(out),
            '--threads',str(args.threads),'--tolerance',str(args.tolerance)]
    with (out/'training.log').open('w') as log:
        result = subprocess.run(base+['--stage','train'],cwd=REPO,env=env,
                                stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        last_progress = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        save_json(out/'status.json', {'stage':'failed','exit_code':result.returncode,
            'last_progress':last_progress,'log':str(out/'training.log')})
        raise RuntimeError(f'Training failed; see {out / "training.log"}')
    records = []
    for index,(mu1,mu2) in enumerate(PARAMETERS):
        methods=['ecsw','psd_metric','bm_ecsw']
        methods=methods[index:]+methods[:index]
        for method in methods:
            subprocess.run(base+['--stage','online','--method',method,
                '--mu1',str(mu1),'--mu2',str(mu2)],cwd=REPO,env=env,check=True)
            row=json.loads((out/f'{method}_mu1_{mu1}_mu2_{mu2}.json').read_text())
            records.append(row)
            save_json(out/'online_results.json',records)
    for p in sources:
        if file_hash(p)!=hashes[str(p.relative_to(REPO))]:
            raise RuntimeError(f'Existing model input changed: {p}')
    report=json.loads((out/'training.json').read_text())
    fields=['method','mu1','mu2','cells','augmented_cells','online_seconds','error_percent','max_snapshot_error_percent','iterations']
    with (out/'online_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    lines=['# PSD residual-metric experiment','',
        'A fixed positive semidefinite 2 x 2 matrix weights each selected cell residual.',
        'Training selects cells by PSD directional correlations and jointly refits all three symmetric matrix entries under PSD constraints.',
        'All selected matrices are jointly refitted after every added cell.',
        'This is a separate residual-space cubature variant, not the existing modal BM-ECSW rule.','',
        f'Relative training error: {report["relative_training_error"]:.9e}; tolerance {args.tolerance:.1e}; attained: {report["tolerance_met"]}.',
        f'Sampled cells: {report["cells"]}; augmented cells: {report["augmented_cells"]}; free coefficients: {report["free_coefficients"]}.',
        f'Assembly: {report["assembly_seconds"]:.3f} s; selection: {report["selection_seconds"]:.3f} s; conic refits: {report["conic_refit_seconds"]:.3f} s; offline total: {report["offline_seconds"]:.3f} s.','',
        '| Parameter | HPROM | Cells | Augmented cells | Online (s) | Trajectory error (%) |',
        '|---|---|---:|---:|---:|---:|']
    labels={'ecsw':'ECSW','psd_metric':'PSD residual metric','bm_ecsw':'BM-ECSW'}
    for row in records:
        lines.append(f'| ({row["mu1"]}, {row["mu2"]}) | {labels[row["method"]]} | {row["cells"]} | {row["augmented_cells"]} | {row["online_seconds"]:.6f} | {row["error_percent"]:.8f} |')
    lines += ['', f'One fresh-process run per configuration with {args.threads} native threads; actual pools are recorded.',
        'Online time includes mesh setup, initial projection and time integration; it excludes loading, decoding, error evaluation and output.',
        'Classical and PSD-metric HPROMs use the same weighted-residual stopping and least-squares solver. BM retains its existing projected-equation stopping and step control.',
        'Offline assembly batches the original least-squares projections and vectorizes the local products, so its timing is not directly comparable to the older assembly implementation.',
        'The potential and symmetric positive semidefinite Gauss-Newton matrix follow from construction. They do not guarantee smaller trajectory error, fewer cells, or global nonlinear convergence.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    if not args.keep_training:
        shutil.rmtree(out/'training_matrices')
    provenance['completed_utc']=datetime.now(timezone.utc).isoformat()
    save_json(out/'provenance.json',provenance)
    save_json(out/'status.json',{'stage':'complete','summary':str(out/'summary.md')})
    print(f'Completed: {out}/summary.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir')
    parser.add_argument('--threads',type=int,default=10)
    parser.add_argument('--tolerance',type=float,default=1e-5)
    parser.add_argument('--keep-training',action='store_true')
    parser.add_argument('--stage',choices=['train','online'])
    parser.add_argument('--method',choices=['ecsw','bm_ecsw','psd_metric'])
    parser.add_argument('--mu1',type=float)
    parser.add_argument('--mu2',type=float)
    args=parser.parse_args()
    if args.threads<1 or not 0<args.tolerance<1:
        parser.error('Require positive threads and tolerance in (0,1).')
    if args.stage=='train':train(args)
    elif args.stage=='online':online(args)
    else:experiment(args)


if __name__=='__main__':
    main()
