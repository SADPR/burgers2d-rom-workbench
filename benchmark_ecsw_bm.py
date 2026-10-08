#!/usr/bin/env python3
"""Repeat the matched-training classical ECSW/BM-ECSW online comparison.

Each measured run uses the production run_hprom.main in a fresh process.
Runs are sequential and method order alternates. Both rules were trained on
225 consecutive pairs; the benchmark loads their existing weights.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent
PARAMETERS = [(4.56, .019), (4.75, .020), (5.19, .026)]
WEIGHTS = {
    'ecsw': 'ecsw_weights_lspg_ecsw_tol1e-10_offset1_snap5pct.npy',
    'bm_ecsw': 'ecsw_weights_lspg_bm_ecsw_tol1e-10_offset1_snap5pct.npy',
}
CELLS = {'ecsw': 2492, 'bm_ecsw': 300}


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_report(path):
    return dict(line.split(': ', 1) for line in path.read_text().splitlines()
                if ': ' in line)


def worker(args):
    from run_hprom import main
    from threadpoolctl import threadpool_info

    info = threadpool_info()
    if any(pool['num_threads'] != args.threads for pool in info):
        raise RuntimeError(f'Unexpected native thread configuration: {info}')
    online_s, error = main(
        mu1=args.mu1, mu2=args.mu2, compute_ecsw=False,
        pod_dir=str(REPO/'POD'), results_dir=args.run_dir,
        selection_method=args.method, snap_time_offset=1,
        ecsw_snapshot_percent=5., ecsw_random_seed=42,
        ecsw_tol_squared=1e-10, linear_solver='lstsq', bm_max_its=20,
        bm_jacobian=args.bm_jacobian, bm_min_delta=1e-2,
    )
    (Path(args.run_dir)/'worker.json').write_text(json.dumps({
        'online_seconds': online_s, 'error_percent': error,
        'native_thread_pools': info, 'bm_jacobian': args.bm_jacobian,
    }, indent=2)+'\n')


def write_results(out, records, repeats, bm_jacobian):
    with (out/'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Classical ECSW versus BM-ECSW: repeated online timings', '',
             f'{repeats} fresh-process runs per method and parameter. Sequential execution; 20 CPU threads by default (actual pools recorded per run).',
             'Matched training: 225 consecutive pairs, nine parameters, 96 POD modes, relative training tolerance 1e-5.',
             f'Classical: 2492 cells, weighted Gauss-Newton with lstsq. BM: 300 cells, {bm_jacobian} matrix, maximum 20 iterations. Both enable plateau stopping at 1e-2, on their respective norm.',
             'Times cover the production HPROM function: mesh/operator setup, initial projection and 500 time steps. They exclude training, Python startup, model loading, full-field decoding, HDM comparison, plotting and output files.', '',
             '| Parameters | Method | Run 1 (s) | Run 2 (s) | Run 3 (s) | Mean (s) | Sample SD (s) |',
             '|---|---|---:|---:|---:|---:|---:|']
    stats = []
    for mu1,mu2 in PARAMETERS:
        for method in WEIGHTS:
            rows = sorted((r for r in records if r['mu1']==mu1 and r['mu2']==mu2 and r['method']==method), key=lambda r:r['repeat'])
            if len(rows) != repeats:
                continue
            times = [r['online_seconds'] for r in rows]
            avg = statistics.mean(times)
            sd = statistics.stdev(times) if len(times)>1 else 0.
            label = 'ECSW' if method=='ecsw' else 'BM-ECSW'
            values = ' | '.join(f'{x:.6f}' for x in times)
            # The requested default has exactly three runs; CSV supports any repeat count.
            if repeats == 3:
                lines.append(f'| ({mu1}, {mu2:.3f}) | {label} | {values} | {avg:.6f} | {sd:.6f} |')
            stats.append(dict(mu1=mu1,mu2=mu2,method=method,mean_seconds=avg,
                              median_seconds=statistics.median(times),sample_sd_seconds=sd,
                              min_seconds=min(times),max_seconds=max(times)))
    if stats:
        with (out/'statistics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    if len(stats)==6:
        lines += ['', '| Parameters | Mean ECSW / mean BM-ECSW |', '|---|---:|']
        for mu1,mu2 in PARAMETERS:
            classic=next(r for r in stats if r['mu1']==mu1 and r['mu2']==mu2 and r['method']=='ecsw')
            bm=next(r for r in stats if r['mu1']==mu1 and r['mu2']==mu2 and r['method']=='bm_ecsw')
            lines.append(f"| ({mu1}, {mu2:.3f}) | {classic['mean_seconds']/bm['mean_seconds']:.3f}x |")
        bmrows=[r for r in records if r['method']=='bm_ecsw']
        lines += ['', f"BM projected-equation tolerance met: {sum(500-r['nonconverged_steps'] for r in bmrows)}/{500*len(bmrows)} steps.",
                  'These measurements compare the implemented solvers and rules. The classical and BM stopping criteria differ, as documented in their production summaries.',
                  'Three runs characterize observed repeatability on this machine; no statistical significance claim is made.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def benchmark(args):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(args.output_dir) if args.output_dir else REPO/'Results'/f'ECSW_BM_benchmark_{stamp}'
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [REPO/'POD'/fn for fn in WEIGHTS.values()]
    sources += [REPO/'POD'/fn for fn in ('basis.npy','sigma.npy','u_ref.npy')]
    hashes = {str(p.relative_to(REPO)):file_hash(p) for p in sources}
    cache = REPO/'Results'/'param_snaps'
    for mu1,mu2 in PARAMETERS:
        if not (cache/f'mu1_{mu1}+mu2_{mu2}.npy').exists():
            raise FileNotFoundError(f'Missing cached HDM snapshots for {(mu1,mu2)}')
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]=str(args.threads)
    model = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    provenance = {
        'started_utc': stamp, 'cpu_model':model, 'logical_cpus':os.cpu_count(),
        'requested_native_threads':args.threads,'repeats':args.repeats,
        'versions':{package:importlib.metadata.version(package) for package in ('numpy','scipy','threadpoolctl')},
        'input_sha256':hashes,'method_order':'rotating parameter order; alternating method order in each pair',
        'warmup_runs':0,'measured_region':'run_hprom.main total_hprom_time_seconds',
        'bm_jacobian':args.bm_jacobian,'bm_plateau_tolerance':1e-2,
    }
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    records=[]
    for repeat in range(1,args.repeats+1):
        order = PARAMETERS[repeat%3:] + PARAMETERS[:repeat%3]
        for index,(mu1,mu2) in enumerate(order):
            methods = list(WEIGHTS)
            if (repeat+index)%2:
                methods.reverse()
            for method in methods:
                run_dir=out/f'{method}_mu1_{mu1}_mu2_{mu2:.3f}_repeat{repeat}'
                run_dir.mkdir()
                (run_dir/'param_snaps').symlink_to(cache,target_is_directory=True)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',
                     '--method',method,'--mu1',str(mu1),'--mu2',str(mu2),
                     '--threads',str(args.threads),'--run-dir',str(run_dir),
                     '--bm-jacobian',args.bm_jacobian]
                print(f'Run {len(records)+1}/{6*args.repeats}: {method}, mu=({mu1},{mu2}), repeat={repeat}',flush=True)
                start=time.perf_counter()
                with (run_dir/'runner.log').open('w') as log:
                    subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                wall=time.perf_counter()-start
                files=list(run_dir.glob('*summary*.txt'))
                if len(files)!=1:
                    raise RuntimeError(f'Expected one summary in {run_dir}: {files}')
                report=read_report(files[0]); measurement=json.loads((run_dir/'worker.json').read_text())
                cells=int(report['num_nonzero_weights'])
                if cells!=CELLS[method] or report['snap_time_offset']!='1' or report['compute_ecsw']!='False':
                    raise RuntimeError(f'Unexpected comparison configuration: {report}')
                not_met=0 if method=='ecsw' else int(report['bm_projected_tolerance_not_met_steps'])
                if not_met:
                    raise RuntimeError(f'BM failed to converge in {not_met} steps')
                records.append(dict(sequence=len(records)+1,repeat=repeat,method=method,mu1=mu1,mu2=mu2,
                                    cells=cells,online_seconds=measurement['online_seconds'],wall_seconds=wall,
                                    error_percent=measurement['error_percent'],nonconverged_steps=not_met,
                                    iterations=int(report['gn_iterations_total'] if method=='ecsw' else report['bm_nonlinear_iterations_total']),
                                    bm_jacobian=args.bm_jacobian if method=='bm_ecsw' else '',
                                    plateau_steps=int(report['bm_plateau_steps']) if method=='bm_ecsw' else 0,
                                    jacobian_seconds=float(report['jacobian_time_seconds']),
                                    residual_seconds=float(report['residual_time_seconds']),
                                    linear_solve_seconds=float(report['linear_solve_time_seconds']),
                                    report=str(files[0].relative_to(out))))
                write_results(out,records,args.repeats,args.bm_jacobian)
                print(f"  Online: {measurement['online_seconds']:.6f}s; error: {measurement['error_percent']:.8f}%",flush=True)
    if any(file_hash(p)!=hashes[str(p.relative_to(REPO))] for p in sources):
        raise RuntimeError('A model input changed during the benchmark')
    provenance['completed_utc']=datetime.now(timezone.utc).isoformat()
    provenance['completed_runs']=len(records)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Completed: {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--threads',type=int,default=20)
    parser.add_argument('--output-dir')
    parser.add_argument('--bm-jacobian', choices=('exact','gauss_newton'), default='gauss_newton')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--method',choices=tuple(WEIGHTS),help=argparse.SUPPRESS)
    parser.add_argument('--mu1',type=float,help=argparse.SUPPRESS)
    parser.add_argument('--mu2',type=float,help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.repeats<1 or args.threads<1:
        parser.error('repeats and threads must be positive')
    if args.worker:
        if args.method is None or args.mu1 is None or args.mu2 is None or args.run_dir is None:
            parser.error('worker requires method, mu1, mu2 and run-dir')
        worker(args)
    else:
        benchmark(args)


if __name__=='__main__':
    main()
