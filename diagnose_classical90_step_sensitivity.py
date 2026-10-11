#!/usr/bin/env python3
"""Add saved classical ECSW (Nc=90) to the existing one-step diagnostics.

Reuses saved PROM coordinates; performs no training or trajectory integration.
The exact solves are diagnostic and do not alter the production solver.
"""
import argparse
import csv
import json
from contextlib import redirect_stdout
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from burgers.config import GRID_X, GRID_Y, DT
from burgers.core import get_ops
import burgers.gauss_newton as gn
from diagnose_bm_ecsw import SampledSystem, RULES, digest
from diagnose_bm_stability import solve_root, matrix_metrics, save_csv, save_json


def measure(system, initial, root_ref, full_basis, full_reference):
    state0 = system.reference + system.basis @ initial
    _, approx = gn.assemble_bm_lspg_system_2D(
        system.residual(state0), system.jac(state0) @ system.basis,
        system.basis, system.weights, system.jdx, system.jdy, DT,
        jacobian_kind='gauss_newton')
    solution = solve_root(system, initial)
    q = solution[0]
    if not solution[3] or solution[4] >= 1e-11:
        raise RuntimeError(f'Projected equations did not reach the common tolerance: {solution[4]}')
    F, exact = system.evaluate(q)
    A = system.jac(system.reference + system.basis @ q) @ system.basis
    basis_rows = system.basis[np.searchsorted(system.cols, system.rows)]
    Ap = system.jac(system.previous) @ system.basis - 2*basis_rows
    Au, Av = np.split(A, 2)
    Apu, Apv = np.split(Ap, 2)
    history = (system.weights*Au).T @ Apu + (system.weights*Av).T @ Apv
    transition = -np.linalg.solve(exact, history)
    amplification = float(np.linalg.svd(transition, compute_uv=False)[0])
    error = 100*np.linalg.norm(full_basis @ (q-root_ref)) / np.linalg.norm(full_reference + full_basis @ root_ref)
    row = dict(root_converged=bool(solution[3]), root_ratio=float(solution[4]),
               root_norm=float(np.linalg.norm(F)), error_vs_PROM_percent=float(error),
               amplification=amplification,
               spectral_radius=float(np.abs(np.linalg.eigvals(transition)).max()),
               **matrix_metrics(approx))
    return row, q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-results', default='Results/BM_ECSW_stability_20261009')
    parser.add_argument('--classical-results', default='Results/BM_ECSW_stability_classical90_20261009')
    parser.add_argument('--output-dir', default='Results/BM_ECSW_stability_classical90_steps_20261010')
    args = parser.parse_args()
    base, old_classical, out = map(Path, (args.base_results, args.classical_results, args.output_dir))
    meta = json.loads((base/'provenance.json').read_text())
    for name in ('POD/basis.npy', 'POD/u_ref.npy'):
        if digest(name) != meta['sources'][name]:
            raise RuntimeError(f'Saved reference differs: {name}')
    rule = 'classical_90_offset3'
    weights_path = Path('POD')/RULES[rule]
    old_weights = json.loads((old_classical/'provenance.json').read_text())['weights'][rule]
    if digest(weights_path) != old_weights['sha256']:
        raise RuntimeError('The saved classical rule has changed.')
    with (base/'probes.csv').open() as f:
        old_probes = list(csv.DictReader(f))
    out.mkdir(parents=True, exist_ok=False)
    source_names = ('POD/basis.npy', 'POD/u_ref.npy', 'burgers/gauss_newton.py',
                    'burgers/linear_manifold.py', 'burgers/core.py',
                    'diagnose_bm_ecsw.py', 'diagnose_bm_stability.py')
    save_json(out/'provenance.json', dict(parameter=[4.56, .019], dt=DT,
        steps=[1,5,20], threads=1, modes=96, base_results=str(base),
        reference_coordinates=dict(path=str(base/'probe_PROM_coordinates.npy'),
                                   sha256=digest(base/'probe_PROM_coordinates.npy')),
        weights={rule:old_weights}, sources={p:digest(p) for p in source_names},
        script_sha256=digest(__file__),
        protocol='Saved PROM previous states, diagnostic exact solves at relative residual <1e-11; correction matrices at shared predictors; exact step-map sensitivity at each solved state.',
        control='Reproduce the existing classical Nc=225 first-step row before adding the new row.',
        limits='No retraining, no trajectory integration, no change to production solvers.'))
    basis, reference = np.load('POD/basis.npy'), np.load('POD/u_ref.npy')
    coordinates = np.load(base/'probe_PROM_coordinates.npy')
    weights = np.load(weights_path)
    ops, mu = get_ops(GRID_X, GRID_Y), [4.56,.019]
    full = SampledSystem(np.ones(basis.shape[0]//2), basis, reference, ops, mu)
    system = SampledSystem(weights, basis, reference, ops, mu)
    control = SampledSystem(np.load(Path('POD')/RULES['classical_225_offset1']),
                            basis, reference, ops, mu)
    rows, roots, reference_roots = [], [], []
    t0 = time.monotonic()
    for step in (1,5,20):
        save_json(out/'status.json', dict(stage='one_step', step=step,
            completed=len(rows), total=3, seconds=time.monotonic()-t0))
        initial = coordinates[:,step-1]
        previous = reference + basis @ initial
        full.previous = previous[full.cols].copy()
        system.previous = previous[system.cols].copy()
        with (out/f'step_{step}_solver.log').open('w') as log, redirect_stdout(log):
            ref_solution = solve_root(full, initial)
            if not ref_solution[3] or ref_solution[4] >= 1e-11:
                raise RuntimeError(f'PROM solve did not attain the diagnostic tolerance: {ref_solution[4]}')
            root_ref = ref_solution[0]
            if step == 1:
                control.previous = previous[control.cols].copy()
                check, _ = measure(control, initial, root_ref, basis, reference)
                saved = next(r for r in old_probes if r['rule']=='classical_225_offset1' and r['step']=='1')
                for key in ('error_vs_PROM_percent', 'amplification', 'sym_ratio', 'condition'):
                    if not np.isclose(check[key],float(saved[key]), rtol=1e-6, atol=1e-9):
                        raise RuntimeError(f'Existing first-step row was not reproduced: {key}: {check[key]} vs {saved[key]}')
                save_json(out/'existing_row_reproduction.json', check)
            measured, root = measure(system, initial, root_ref, basis, reference)
        rows.append(dict(rule=rule, step=step, **measured))
        roots.append(root)
        reference_roots.append(root_ref)
        save_csv(out/'probes.csv', rows)
        np.save(out/'classical90_root_coordinates.npy', np.column_stack(roots))
        np.save(out/'PROM_root_coordinates.npy', np.column_stack(reference_roots))
        print(json.dumps(rows[-1]), flush=True)
    save_json(out/'status.json', dict(stage='complete', probe_steps=[1,5,20],
        completed=3, total=3, seconds=time.monotonic()-t0))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
