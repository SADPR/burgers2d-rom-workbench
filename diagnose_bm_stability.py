#!/usr/bin/env python3
"""Reproduce Burgers BM stability diagnostics without changing trained rules.

Production trajectories use the existing classical GN / BM GN-type solvers.
Exact derivatives are used ONLY to diagnose the implicit time-step map at
converged roots with a common previous PROM state, not for these trajectories.
"""
import argparse
import csv
import inspect
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import burgers.gauss_newton as gn
import burgers.linear_manifold as lm
from burgers.config import GRID_X, GRID_Y, W0, DT
from burgers.core import get_ops
from diagnose_bm_ecsw import SampledSystem, RULES, digest

NAMES = ('classical_225_offset1', 'bm_90_offset1', 'bm_225_offset1')
PARAMETERS = ((4.56, .019), (4.75, .020), (5.19, .026))


def save_json(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def save_csv(path, data):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def matrix_metrics(K):
    with threadpool_limits(limits=1):
        eig = np.linalg.eigvalsh((K + K.T) * .5)
        singular = np.linalg.svd(K, compute_uv=False)
        return dict(sym_min=float(eig[0]), sym_max=float(eig[-1]),
                    sym_ratio=float(eig[0] / eig[-1]),
                    min_real_eigenvalue=float(np.linalg.eigvals(K).real.min()),
                    asymmetry=float(np.linalg.norm(K-K.T) / np.linalg.norm(K)),
                    condition=float(singular[0] / singular[-1]))


def production_run(name, weights, mu, basis, reference, out, steps):
    """Record every correction matrix; callbacks leave solver values unchanged."""
    matrices, stops = [], []
    context = {'step': 0, 'evaluation': 0}
    original_bm = lm.newton_BM_ECSW_2D
    original_classical = lm.gauss_newton_ECSW_2D
    original_assemble = gn.assemble_bm_lspg_system_2D

    def record(K):
        matrices.append(dict(step=context['step'], evaluation=context['evaluation'],
                             **matrix_metrics(K)))
        context['evaluation'] += 1

    def assemble(*args, **kwargs):
        F, K = original_assemble(*args, **kwargs)
        if K is not None:
            record(K)
        return F, K

    def bm(*args, **kwargs):
        context['step'] += 1
        context['evaluation'] = 0
        result = original_bm(*args, **kwargs)
        stops.append(dict(step=context['step'], reason=result[5],
                          final_ratio=float(result[4]), evaluations=len(result[1])))
        return result

    def classical(*args, **kwargs):
        context['step'] += 1
        context['evaluation'] = 0
        bound = inspect.signature(original_classical).bind(*args, **kwargs)
        values = bound.arguments
        original_jac = values['jac']

        def jac(state):
            J = original_jac(state)
            A = J @ values['basis']
            record(A.T @ (values['sample_weights'][:, None] * A))
            return J

        values['jac'] = jac
        result = original_classical(**values)
        norms = result[1]
        ratio = norms[-1] / max(norms[0], np.finfo(float).tiny)
        reason = ('sampled_tolerance' if ratio < 1e-5 else
                  'plateau' if len(norms) < 20 else 'max_its')
        stops.append(dict(step=context['step'], reason=reason,
                          final_ratio=float(ratio), evaluations=len(norms)))
        return result

    tag = f'{name}_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}'
    lm.newton_BM_ECSW_2D = bm
    lm.gauss_newton_ECSW_2D = classical
    gn.assemble_bm_lspg_system_2D = assemble
    try:
        with (out / f'{tag}.log').open('w') as log, redirect_stdout(log):
            coordinates, stats = lm.inviscid_burgers_implicit2D_LSPG_ecsw(
                GRID_X, GRID_Y, weights, W0, DT, steps, mu, basis, u_ref=reference,
                max_its=20, relnorm_cutoff=1e-5, min_delta=.01,
                linear_solver='lstsq', bm_jacobian='gauss_newton')
    finally:
        lm.newton_BM_ECSW_2D = original_bm
        lm.gauss_newton_ECSW_2D = original_classical
        gn.assemble_bm_lspg_system_2D = original_assemble
    np.save(out / f'{tag}_coordinates.npy', coordinates)
    save_csv(out / f'{tag}_matrices.csv', matrices)
    save_csv(out / f'{tag}_stops.csv', stops)
    hdm_path = Path(f'Results/param_snaps/mu1_{mu[0]}+mu2_{mu[1]}.npy')
    hdm = np.load(hdm_path, mmap_mode='r')
    errors = []
    # Stream the HDM comparison; do not allocate another full trajectory.
    for start in range(0, steps+1, 25):
        stop = min(start+25, steps+1)
        target = hdm[:, start:stop]
        errors.extend((np.linalg.norm(reference[:, None] + basis @ coordinates[:, start:stop]
                                      - target, axis=0) / np.linalg.norm(target, axis=0)).tolist())
    np.save(out / f'{tag}_errors.npy', errors)
    predictors = [r for r in matrices if r['evaluation'] == 0]
    summary = dict(rule=name, mu=list(mu), steps=steps,
        finite=bool(np.isfinite(coordinates).all()), max_error_percent=100*max(errors),
        stop_reasons=dict(Counter(r['reason'] for r in stops)),
        matrix_evaluations=len(matrices),
        indefinite_evaluations=sum(r['sym_ratio'] < -1e-12 for r in matrices),
        nonpositive_real_evaluations=sum(r['min_real_eigenvalue'] <= 0 for r in matrices),
        predictor_indefinite_steps=sum(r['sym_ratio'] < -1e-12 for r in predictors),
        predictor_sym_ratio_median=float(np.median([r['sym_ratio'] for r in predictors])),
        predictor_condition_median=float(np.median([r['condition'] for r in predictors])),
        all_sym_ratio_min=min(r['sym_ratio'] for r in matrices),
        all_sym_ratio_max=max(r['sym_ratio'] for r in matrices),
        hdm_file=str(hdm_path), hdm_sha256=digest(hdm_path))
    save_json(out / f'{tag}.json', summary)
    return summary


def solve_root(system, initial):
    return gn.newton_BM_ECSW_2D(system.residual, system.jac, system.basis, initial,
        system.weights, system.jdx, system.jdy, DT, max_its=40,
        relnorm_cutoff=1e-11, u_ref=system.reference, jacobian_kind='exact', min_delta=None)


def probe_maps(basis, reference, weights, out):
    """Implicit function derivative T=-K^{-1}B, at tightly converged roots."""
    mu = PARAMETERS[0]
    with (out/'probe_solver.log').open('w') as log, redirect_stdout(log):
        states, coordinates, _ = lm.inviscid_burgers_implicit2D_LSPG(
            GRID_X, GRID_Y, W0, DT, 20, mu, basis,
            u_ref=reference, return_red_coords=True, linear_solver='lstsq')
        np.save(out/'probe_PROM_coordinates.npy', coordinates)
        ops = get_ops(GRID_X, GRID_Y)
        systems = {'PROM': SampledSystem(np.ones(basis.shape[0]//2), basis, reference, ops, mu)}
        systems.update({name: SampledSystem(weights[name], basis, reference, ops, mu) for name in NAMES})
        rows = []
        checks = []
        for step in (1, 5, 20):
            root_ref = None
            for name, system in systems.items():
                system.previous = states[system.cols, step-1].copy()
                initial = coordinates[:, step-1]
                previous_base = system.previous.copy()
                state0 = system.reference + system.basis @ initial
                _, Jhat0 = gn.assemble_bm_lspg_system_2D(system.residual(state0),
                    system.jac(state0) @ system.basis, system.basis, system.weights,
                    system.jdx, system.jdy, DT, jacobian_kind='gauss_newton')
                solution = solve_root(system, initial)
                q = solution[0]
                F, K = system.evaluate(q)
                current = system.reference + system.basis @ q
                A = system.jac(current) @ system.basis
                basis_rows = system.basis[np.searchsorted(system.cols, system.rows)]
                Ap = system.jac(system.previous) @ system.basis - 2*basis_rows
                Au, Av = np.split(A, 2)
                Apu, Apv = np.split(Ap, 2)
                B = (system.weights*Au).T @ Apu + (system.weights*Av).T @ Apv
                with threadpool_limits(limits=1):
                    T = -np.linalg.solve(K, B)
                    _, singular, right = np.linalg.svd(T)
                    map_radius = np.abs(np.linalg.eigvals(T)).max()
                if name == 'PROM':
                    root_ref = q
                error = np.linalg.norm(basis @ (q-root_ref)) / np.linalg.norm(reference + basis @ root_ref)
                rows.append(dict(rule=name, step=step, root_converged=bool(solution[3]),
                    root_ratio=float(solution[4]), root_norm=float(np.linalg.norm(F)),
                    error_vs_PROM_percent=float(100*error), amplification=float(singular[0]),
                    spectral_radius=float(map_radius), **matrix_metrics(Jhat0)))
                # Independently validate current/history derivatives, plus the map
                # by actually resolving perturbed previous-state equations.
                if step == 1:
                    d = right[0]
                    eps = 1e-3
                    finite_K = (system.evaluate(q+eps*d, False)[0] - system.evaluate(q-eps*d, False)[0])/(2*eps)
                    system.previous = previous_base + eps*(system.basis @ d)
                    fplus = system.evaluate(q, False)[0]
                    plus = solve_root(system, q)
                    system.previous = previous_base - eps*(system.basis @ d)
                    fminus = system.evaluate(q, False)[0]
                    minus = solve_root(system, q)
                    system.previous = previous_base
                    finite_B = (fplus-fminus)/(2*eps)
                    finite_T = (plus[0]-minus[0])/(2*eps)
                    checks.append(dict(rule=name, epsilon=eps,
                        current_derivative_error=float(np.linalg.norm(finite_K-K@d)/np.linalg.norm(K@d)),
                        history_derivative_error=float(np.linalg.norm(finite_B-B@d)/np.linalg.norm(B@d)),
                        map_derivative_error=float(np.linalg.norm(finite_T-T@d)/np.linalg.norm(T@d)),
                        predicted_amplification=float(singular[0]), measured_amplification=float(np.linalg.norm(finite_T)),
                        plus_converged=bool(plus[3]), minus_converged=bool(minus[3]),
                        plus_ratio=float(plus[4]), minus_ratio=float(minus[4])))
                save_csv(out/'probes.csv', rows)
                if checks:
                    save_csv(out/'derivative_checks.csv', checks)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='Results/BM_ECSW_stability_20261009')
    parser.add_argument('--threads', type=int, default=10)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    basis = np.load('POD/basis.npy')
    reference = np.load('POD/u_ref.npy')
    weights = {name: np.load(Path('POD')/RULES[name]) for name in NAMES}
    with threadpool_limits(limits=args.threads):
        save_json(out/'provenance.json', dict(parameters=PARAMETERS, dt=DT, steps=500,
            modes=basis.shape[1], threads=threadpool_info(),
            weights={name:dict(path=str(Path('POD')/RULES[name]),
                sha256=digest(Path('POD')/RULES[name]),
                cells=int(np.count_nonzero(np.any(w>0, axis=1) if w.ndim==2 else w>0)))
                     for name,w in weights.items()},
            sources={path:digest(path) for path in ('POD/basis.npy','POD/u_ref.npy',
                     'burgers/gauss_newton.py','burgers/linear_manifold.py',__file__)},
            online='Existing classical GN and BM GN-type; max20; relative1e-5; plateau.01; no retraining.',
            matrix_sampling='BM: every derivative evaluation, including stopping iterates. Classical: each solved correction. Predictor is first of each step.',
            probes='Tightly converged implicit-equation roots with common previous PROM states at steps1,5,20. Exact derivatives ONLY for diagnosing time-step maps.',
            limits='Training count also changes selected cells/weights. Amplification>1 is local sensitivity, not by itself an asymptotic instability proof.'))
        runs = []
        for mu in PARAMETERS:
            for name in NAMES:
                save_json(out/'status.json', dict(stage='online', rule=name, mu=mu, completed=len(runs), total=9))
                result = production_run(name, weights[name], mu, basis, reference, out, 500)
                runs.append(result)
                save_json(out/'runs.json', runs)
                print(json.dumps(result), flush=True)
        save_json(out/'status.json', dict(stage='common_state_probes', online_completed=9))
        probe_maps(basis, reference, weights, out)
        save_json(out/'status.json', dict(stage='complete', online_completed=9, probe_steps=[1,5,20]))


if __name__ == '__main__':
    main()
