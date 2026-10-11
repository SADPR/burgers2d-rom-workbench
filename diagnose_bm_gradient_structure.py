#!/usr/bin/env python3
"""Test common-objective compatibility of saved Burgers cubatures.

The diagnostic differentiates the projected vector exactly, including its
residual-weighted second derivatives. It performs no training or integration.
Every trajectory evaluation holds the saved previous state fixed. Common-state
probes use the cached full PROM at mu=(4.56,0.019), steps 1,5,20.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from burgers.config import DT, GRID_X, GRID_Y
from burgers.core import get_ops
from burgers.gauss_newton import assemble_bm_lspg_system_2D
from diagnose_bm_ecsw import RULES, SampledSystem, digest
from diagnose_bm_stability import PARAMETERS, save_json, save_csv

RULE_NAMES=('classical_90_offset3','classical_225_offset1','bm_90_offset1','bm_225_offset1')
BASE=Path('Results/BM_ECSW_stability_20261009')
EXTRA=Path('Results/BM_ECSW_stability_classical90_20261009')
LABELS={'classical_90_offset3':'HPROM (ECSW), Nc=90',
        'classical_225_offset1':'HPROM (ECSW), Nc=225',
        'bm_90_offset1':'HPROM (BM-ECSW), Nc=90',
        'bm_225_offset1':'HPROM (BM-ECSW), Nc=225'}


def evaluate(system,q,previous):
    system.previous=system.reference+system.basis@previous
    state=system.reference+system.basis@q
    residual=system.residual(state)
    tangent=system.jac(state)@system.basis
    F,K=assemble_bm_lspg_system_2D(residual,tangent,system.basis,
        system.weights,system.jdx,system.jdy,DT,jacobian_kind='exact')
    _,A=assemble_bm_lspg_system_2D(residual,tangent,system.basis,
        system.weights,system.jdx,system.jdy,DT,jacobian_kind='gauss_newton')
    if not np.isfinite(K).all():
        raise ValueError('Nonfinite exact projected Jacobian.')
    skew=K-K.T
    i,j=np.unravel_index(np.argmax(np.abs(skew)),skew.shape)
    norm=np.linalg.norm(K)
    stats=dict(exact_asymmetry=float(np.linalg.norm(skew)/norm),
        correction_asymmetry=float(np.linalg.norm(A-A.T)/np.linalg.norm(A)),
        second_derivative_relative=float(np.linalg.norm(K-A)/norm),
        largest_cross_difference=float(skew[i,j]),i=int(i+1),j=int(j+1),
        derivative_i_j=float(K[i,j]),derivative_j_i=float(K[j,i]))
    return F,K,A,stats


def finite_difference_check(system,q,K,stats,context):
    """Check both columns and their crossed entries with the previous state fixed."""
    i,j=stats['i']-1,stats['j']-1
    cols={}
    errors=[]
    for k in (i,j):
        eps=1e-4*max(1.,abs(q[k]))
        delta=np.zeros_like(q)
        delta[k]=eps
        plus=system.evaluate(q+delta,False)[0]
        minus=system.evaluate(q-delta,False)[0]
        col=(plus-minus)/(2*eps)
        cols[k]=col
        errors.append(np.linalg.norm(col-K[:,k])/np.linalg.norm(K[:,k]))
    crossed=float(cols[j][i]-cols[i][j])
    return dict(**context,i=int(i+1),j=int(j+1),
        exact_difference=float(K[i,j]-K[j,i]),
        finite_difference_difference=crossed,
        finite_derivative_i_j=float(cols[j][i]),finite_derivative_j_i=float(cols[i][j]),
        column_relative_error=float(max(errors)),
        crossed_error_scaled=float(abs(crossed-(K[i,j]-K[j,i]))/np.linalg.norm(K)))


def summarize(records,names):
    summary=[]
    for name in names:
        rows=[r for r in records if r['rule']==name]
        values=[r['exact_asymmetry'] for r in rows]
        summary.append(dict(rule=name,evaluations=len(rows),
            exact_asymmetry_min=float(min(values)),
            exact_asymmetry_median=float(np.median(values)),
            exact_asymmetry_max=float(max(values)),
            nonsymmetric_evaluations=int(sum(x>1e-10 for x in values))))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='Results/BM_ECSW_gradient_structure_20261009')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    basis=np.load('POD/basis.npy')
    reference=np.load('POD/u_ref.npy')
    weights={name:np.load(Path('POD')/RULES[name]) for name in RULE_NAMES}
    baseline=json.loads((BASE/'provenance.json').read_text())
    supplemental=json.loads((EXTRA/'provenance.json').read_text())
    for key in ('POD/basis.npy','POD/u_ref.npy','burgers/gauss_newton.py','burgers/linear_manifold.py'):
        actual=digest(key)
        if baseline['sources'][key]!=actual or supplemental['sources'][key]!=actual:
            raise ValueError(f'Saved trajectories use a different source: {key}')
    all_meta={**baseline['weights'],**supplemental['weights']}
    for name in RULE_NAMES:
        if digest(Path('POD')/RULES[name])!=all_meta[name]['sha256']:
            raise ValueError(f'Cubature weights changed: {name}')
    ops=get_ops(GRID_X,GRID_Y)
    provenance=dict(parameters=PARAMETERS,dt=DT,modes=basis.shape[1],
        methods=all_meta,
        threads=args.threads,
        sources={key:digest(key) for key in ('POD/basis.npy','POD/u_ref.npy',
             'burgers/gauss_newton.py','burgers/linear_manifold.py',__file__)},
        common_states='Saved full PROM, mu=(4.56,.019), predictor and computed state at steps1,5,20; identical coordinates/history for every rule.',
        trajectory_states='All 500 computed states of every saved production trajectory, with its own previous state held fixed.',
        derivative='Exact derivative of the projected vector, including residual-weighted second derivatives; correction approximation measured separately.',
        finite_differences='Central differences of the directly evaluated projected vector, two columns at the strongest exact crossed mismatch; epsilon=1e-4*max(1,abs(q_k)).',
        symmetry_count_threshold=1e-10,
        limitations='Nonsymmetry excludes a local smooth scalar objective. It does not establish that loss of an objective alone causes trajectory failure.',
        trajectories={})
    with threadpool_limits(limits=args.threads):
        provenance['native_threads']=threadpool_info()
        save_json(out/'provenance.json',provenance)
        systems={name:SampledSystem(weights[name],basis,reference,ops,PARAMETERS[0]) for name in RULE_NAMES}
        prom=np.load(BASE/'probe_PROM_coordinates.npy')
        common,checks,first_matrices=[],[],{}
        for step in (1,5,20):
            for location,q in (('predictor',prom[:,step-1]),('PROM_state',prom[:,step])):
                for name,system in systems.items():
                    F,K,A,stats=evaluate(system,q,prom[:,step-1])
                    context=dict(rule=name,step=step,location=location,
                        mu1=PARAMETERS[0][0],mu2=PARAMETERS[0][1])
                    common.append(dict(**context,**stats))
                    if step==1 and location=='predictor':
                        first_matrices[name+'_exact']=K
                        first_matrices[name+'_correction']=A
                        first_matrices[name+'_projected_vector']=F
                        checks.append(finite_difference_check(system,q,K,stats,context))
        np.savez(out/'first_common_matrices.npz',**first_matrices)
        save_csv(out/'common_states.csv',common)
        print('Common-state tests: '+json.dumps(summarize(common,RULE_NAMES)),flush=True)
        trajectory=[]
        for mu in PARAMETERS:
            for name in RULE_NAMES:
                folder=EXTRA if name=='classical_90_offset3' else BASE
                tag=f'{name}_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}'
                path=folder/f'{tag}_coordinates.npy'
                coords=np.load(path)
                if coords.shape!=(basis.shape[1],501):
                    raise ValueError(f'Incomplete trajectory: {path}')
                provenance['trajectories'][tag]=dict(path=str(path),sha256=digest(path))
                system=SampledSystem(weights[name],basis,reference,ops,mu)
                for step in range(1,501):
                    _,K,_,stats=evaluate(system,coords[:,step],coords[:,step-1])
                    context=dict(rule=name,step=step,location='online_state',mu1=mu[0],mu2=mu[1])
                    trajectory.append(dict(**context,**stats))
                    if step in (1,100,500):
                        checks.append(finite_difference_check(system,coords[:,step],K,stats,context))
                    if step%100==0:
                        save_json(out/'status.json',dict(stage='saved_trajectory_checks',
                            rule=name,mu=mu,step=step,evaluations=len(trajectory),total=6000))
                print(f'Finished {tag}',flush=True)
                save_csv(out/'trajectory_states.csv',trajectory)
        save_csv(out/'finite_difference_checks.csv',checks)
        common_summary=summarize(common,RULE_NAMES)
        trajectory_summary=summarize(trajectory,RULE_NAMES)
        summary=dict(common_states=common_summary,trajectory_states=trajectory_summary,
            max_finite_difference_column_error=max(r['column_relative_error'] for r in checks),
            max_finite_difference_crossed_error_scaled=max(r['crossed_error_scaled'] for r in checks),
            derivative_checks=len(checks))
        save_json(out/'summary.json',summary)
        save_json(out/'provenance.json',provenance)
        lines=['Burgers: common-objective compatibility of saved ECSW and BM rules','',
            'Exact Jacobians include the residual-weighted Hessian terms.',
            'No retraining or new integration. One native thread by default.',
            'A nonzero K-K^T directly violates the crossed-derivative compatibility condition.',
            'Asymmetry = ||K-K^T||F/||K||F; counts use threshold1e-10.','',
            'Common full-PROM states (first test parameter; steps1,5,20; predictor and computed state):']
        for row in common_summary:
            lines.append(f"{LABELS[row['rule']]}: median={row['exact_asymmetry_median']:.8g}, range=[{row['exact_asymmetry_min']:.8g},{row['exact_asymmetry_max']:.8g}], nonsymmetric={row['nonsymmetric_evaluations']}/{row['evaluations']}")
        lines+=['','Computed online states (all three parameters;500 steps each):']
        for row in trajectory_summary:
            lines.append(f"{LABELS[row['rule']]}: median={row['exact_asymmetry_median']:.8g}, range=[{row['exact_asymmetry_min']:.8g},{row['exact_asymmetry_max']:.8g}], nonsymmetric={row['nonsymmetric_evaluations']}/{row['evaluations']}")
        lines+=['',f"Maximum finite-difference column error: {summary['max_finite_difference_column_error']:.8g}",
            f"Maximum crossed-difference error / ||K||F: {summary['max_finite_difference_crossed_error_scaled']:.8g}",
            '','Reproduce from repository root using a NEW output directory:',
            'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python diagnose_bm_gradient_structure.py --threads 1 --output-dir Results/BM_ECSW_gradient_structure_new',
            '','Exact-Jacobian nonsymmetry excludes a common smooth objective near tested states.',
            'It does not prove that nonsymmetry or absence of an objective alone causes instability.','']
        (out/'summary.txt').write_text('\n'.join(lines))
        save_json(out/'status.json',dict(stage='complete',common_evaluations=len(common),
            trajectory_evaluations=len(trajectory),finite_difference_checks=len(checks)))
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
