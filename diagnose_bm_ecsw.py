#!/usr/bin/env python3
"""Compare existing cubature rules on the same full-mesh LSPG trajectory.

No weights are retrained or modified. The reference PROM is run from scratch.
One-step solves use the reference previous state, isolating local behavior
from accumulated trajectory errors. All Jacobians use the production kernel.
"""
import argparse
import csv
import hashlib
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_info

from burgers.config import GRID_X, GRID_Y, W0, DT
from burgers.core import (
    EcswJacobianAssembler, get_ops, inviscid_burgers_exact_jac2D,
    inviscid_burgers_res2D, inviscid_burgers_res2D_ecsw,
)
from burgers.ecsw_utils import generate_augmented_mesh
from burgers.gauss_newton import assemble_bm_lspg_system_2D, newton_BM_ECSW_2D
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG

RULES = {
    'bm_90_offset3': 'ecsw_weights_lspg_bm_ecsw_tol1e-10.npy',
    'bm_90_offset1': 'ecsw_weights_lspg_bm_ecsw_tol1e-10_offset1.npy',
    'bm_225_offset1': 'ecsw_weights_lspg_bm_ecsw_tol1e-10_offset1_snap5pct.npy',
    'classical_90_offset3': 'ecsw_weights_lspg_ecsw_tol1e-10.npy',
    'classical_225_offset1': 'ecsw_weights_lspg_ecsw_tol1e-10_offset1_snap5pct.npy',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class SampledSystem:
    def __init__(self, weights, basis, reference, ops, mu):
        _, _, jdx, jdy, eye = ops
        nc = basis.shape[0] // 2
        self.cells = np.flatnonzero(np.any(weights > 0, axis=1)
                                   if weights.ndim == 2 else weights > 0)
        self.augmented = generate_augmented_mesh(GRID_X, GRID_Y, self.cells)
        self.rows = np.concatenate((self.cells, nc + self.cells))
        self.cols = np.concatenate((self.augmented, nc + self.augmented))
        self.basis = np.ascontiguousarray(basis[self.cols])
        self.reference = reference[self.cols]
        self.weights = weights[self.cells]
        if weights.ndim == 1:
            self.weights = np.repeat(self.weights[:, None], basis.shape[1], axis=1)
        self.jdx = jdx.tocsr()[self.cells, :][:, self.augmented]
        self.jdy = jdy.tocsr()[self.cells, :][:, self.augmented]
        self.jac = EcswJacobianAssembler(
            DT, self.jdx, self.jdy, eye[self.rows, :][:, self.cols], self.augmented,
        )
        self.mu = mu
        self.previous = None

    def residual(self, state):
        return inviscid_burgers_res2D_ecsw(
            state, GRID_X, GRID_Y, DT, self.previous, self.mu,
            self.jdx, self.jdy, self.cells, self.augmented,
        )

    def evaluate(self, y, derivative=True):
        state = self.reference + self.basis @ y
        return assemble_bm_lspg_system_2D(
            self.residual(state), self.jac(state) @ self.basis,
            self.basis, self.weights, self.jdx, self.jdy, DT, derivative,
        )

    def solve(self, y0, budget):
        return newton_BM_ECSW_2D(
            self.residual, self.jac, self.basis, y0, self.weights,
            self.jdx, self.jdy, DT, max_its=budget, u_ref=self.reference,
        )


def matrix_stats(K):
    sv = np.linalg.svd(K, compute_uv=False)
    return {
        'condition': float(sv[0] / sv[-1]),
        'sigma_min': float(sv[-1]),
        'sigma_max': float(sv[0]),
        'asymmetry_relative': float(np.linalg.norm(K-K.T) / np.linalg.norm(K)),
    }


def write_csv(path, records):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_diagnostic_summary(output_dir):
    """Summarize saved measurements without rerunning solvers or training."""
    out = Path(output_dir)
    meta = json.loads((out/'provenance.json').read_text())
    equations = list(csv.DictReader((out/'equations.csv').open()))
    solves = list(csv.DictReader((out/'one_step_solves.csv').open()))
    propagation = list(csv.DictReader((out/'propagation.csv').open()))
    checks = list(csv.DictReader((out/'propagation_checks.csv').open()))
    lines = [
        '# BM-ECSW training diagnosis', '',
        f"Parameter: {meta['mu']}. Fresh full-mesh PROM reference: {meta['reference_steps']} steps.",
        'Same 96-mode affine POD space, time step, initial state and existing trained weights for every rule.', '',
        '## First-step comparison', '',
        'Each solve starts with the same previous PROM state. Errors below are relative to the PROM, not the HDM.', '',
        '| Rule | Cells | State error vs PROM (%) | Predictor Jacobian condition | Maximum local amplification |',
        '|---|---:|---:|---:|---:|',
    ]
    names = {'bm_90_offset3':'BM, 90 pairs, offset 3',
             'bm_90_offset1':'BM, 90 pairs, offset 1',
             'bm_225_offset1':'BM, 225 pairs, offset 1',
             'classical_90_offset3':'Classical, 90 pairs, offset 3',
             'classical_225_offset1':'Classical, 225 pairs, offset 1'}
    for name, label in names.items():
        eq = next(r for r in equations if r['step']=='1' and r['rule']==name and r['location']=='predictor')
        sol = next(r for r in solves if r['step']=='1' and r['rule']==name and r['solver']=='newton20')
        prop = next(r for r in propagation if r['step']=='1' and r['rule']==name)
        cells = meta['weight_files'][name]['cells']
        lines.append(f"| {label} | {cells} | {100*float(sol['state_error_vs_PROM']):.6g} | {float(eq['condition']):.6g} | {float(prop['map_max_amplification']):.6g} |")
    full = next(r for r in propagation if r['step']=='1' and r['rule']=='full_PROM')
    lines += ['', f"The full PROM's maximum local amplification at this step is {float(full['map_max_amplification']):.8g}.", '',
              '## Nonlinear solves from the shared reference trajectory', '',
              '| Rule | Converged one-step probes (20 iterations) | Largest state error vs PROM (%) |',
              '|---|---:|---:|']
    for name,label in names.items():
        rows = [r for r in solves if r['rule']==name and r['solver']=='newton20']
        count = sum(r['converged']=='True' for r in rows)
        largest = max(100*float(r['state_error_vs_PROM']) for r in rows)
        lines.append(f'| {label} | {count}/{len(rows)} | {largest:.6g} |')
    lines += ['', f"Probed steps: {meta['evaluated_steps']}.", '',
              '## What these measurements establish', '',
              '- The 90-pair BM rules already solve to displaced states in the first step, before accumulated trajectory error.',
              '- A converged BM equation does not imply a small full-PROM equation or a state close to the PROM.',
              '- The 90-pair systems have more sensitive Jacobians and substantially different local time-step maps.',
              '- The 225-pair rule follows the PROM much more closely in these probes.',
              '- Increasing snapshots changes the selected cells and weights too. These experiments do not establish a universal minimum snapshot count.',
              '- They do not measure energy conservation or prove that loss of a common potential caused the failures.', '',
              '## Meaning of local amplification', '',
              'For q(y_n,y_previous)=0, K=dq/dy_n and B=dq/dy_previous, the local map derivative is T=-K^{-1} B.',
              'The maximum amplification is the largest singular value of T. It describes infinitesimal previous-coordinate perturbations near the local root.',
              'One value above one is not a proof of instability over a varying trajectory; the full PROM can also have transient amplification.', '',
              '## Validation', '',
              '- Sampled equations and Jacobians were compared with independently sampled full-mesh residual/JV contributions at every probe.',
              f"- Maximum directional current-state Jacobian finite-difference error: {max(meta['directional_finite_difference_errors'].values()):.3e}.",
              '- The previous-state derivative is also checked against full-mesh finite differences in tests/test_bm_lspg_jacobian.py.', '',
              '| Rule | Predicted amplification | Measured amplification after resolving perturbed steps | Relative derivative error |',
              '|---|---:|---:|---:|']
    for row in checks:
        lines.append(f"| {names[row['rule']]} | {float(row['predicted_amplification']):.8g} | {float(row['measured_amplification']):.8g} | {float(row['derivative_error_relative']):.3e} |")
    lines += ['', '## Limits and reproducibility', '',
              '- One evaluation parameter is diagnosed here. Existing online comparison reports contain the three evaluation cases for the 225-pair rule.',
              '- The reference uses the production full PROM with its existing residual/plateau stopping criterion; it is not an exact stationarity solve.',
              '- Classical weights use Newton only for this equation-level diagnostic. Production classical HPROM uses its existing square-root-weighted Gauss-Newton solver.',
              '- No training rule, tolerance or production iteration budget was changed. Input file hashes and BLAS thread counts are in provenance.json.', '',
              '```bash',
              f"OPENBLAS_NUM_THREADS=20 OMP_NUM_THREADS=20 .venv/bin/python diagnose_bm_ecsw.py --reference-steps {meta['reference_steps']} --mu1 {meta['mu'][0]} --mu2 {meta['mu'][1]} --output-dir {out}",
              '```', '']
    (out/'summary.md').write_text('\n'.join(lines))
    if meta['trajectory_errors_vs_PROM']:
        fig, ax = plt.subplots(figsize=(9,5))
        for name, errors in meta['trajectory_errors_vs_PROM'].items():
            times = DT*np.arange(len(errors))
            ax.semilogy(times[1:], 100*np.asarray(errors[1:]), label=names[name])
        ax.set(xlabel='Time', ylabel='Relative state error vs PROM (%)',
               title='Existing BM trajectories compared with a fresh full-mesh PROM')
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out/'trajectory_errors_vs_PROM.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-steps', type=int, default=50)
    parser.add_argument('--mu1', type=float, default=4.56)
    parser.add_argument('--mu2', type=float, default=.019)
    parser.add_argument('--output-dir', default='Results/BM_ECSW_diagnostics')
    args = parser.parse_args()
    if args.reference_steps < 3:
        parser.error('--reference-steps must be at least 3')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    basis = np.load('POD/basis.npy')
    reference = np.load('POD/u_ref.npy')
    mu = [args.mu1, args.mu2]
    ops = get_ops(GRID_X, GRID_Y)
    dx, dy, jdx, jdy, eye = ops
    full_weights = np.broadcast_to(np.ones((basis.shape[0]//2, 1)),
                                   (basis.shape[0]//2, basis.shape[1]))
    weights = {name: np.load(Path('POD') / fn) for name, fn in RULES.items()}
    systems = {name: SampledSystem(w, basis, reference, ops, mu)
               for name, w in weights.items()}

    print(f'Running fresh full-mesh PROM reference: {args.reference_steps} steps, mu={mu}', flush=True)
    t0 = time.time()
    with open(out / 'reference_solver.log', 'w') as log, redirect_stdout(log):
        states, coords, stats = inviscid_burgers_implicit2D_LSPG(
            GRID_X, GRID_Y, W0, DT, args.reference_steps, mu, basis,
            u_ref=reference, return_red_coords=True, linear_solver='lstsq',
        )
    reference_seconds = time.time()-t0
    print(f'Reference finished in {reference_seconds:.1f}s', flush=True)
    np.save(out / 'reference_coordinates.npy', coords)
    steps = sorted({n for n in (1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 150, 200, 300, 400, args.reference_steps)
                    if n <= args.reference_steps})
    records, solves, propagation, propagation_checks = [], [], [], []
    checks = {}
    trajectory_errors = {}

    def full_evaluate(y, previous, derivative=True):
        state = reference + basis @ y
        r = inviscid_burgers_res2D(state, GRID_X, GRID_Y, DT, previous, mu, dx, dy)
        tangent = inviscid_burgers_exact_jac2D(state, DT, jdx, jdy, eye) @ basis
        q, K = assemble_bm_lspg_system_2D(
            r, tangent, basis, full_weights, jdx, jdy, DT, derivative,
        )
        return q, K, r, tangent

    for n in steps:
        previous = states[:, n-1]
        y0 = coords[:, n-1]
        q0, K0, r0, A0 = full_evaluate(y0, previous)
        scale = np.linalg.norm(q0)
        qref, Kref, _, Aref = full_evaluate(coords[:, n], previous)
        Aprevious = inviscid_burgers_exact_jac2D(previous, DT, jdx, jdy, eye) @ basis - 2*basis
        Bref = Aref.T @ Aprevious
        Tref = -np.linalg.solve(Kref, Bref)
        propagation.append(dict(step=n, rule='full_PROM',
                                map_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(Tref)))),
                                map_max_amplification=float(np.linalg.norm(Tref, 2)),
                                map_error_relative=0.))
        for location, y in (('predictor', y0), ('reference_solution', coords[:, n])):
            if location == 'predictor':
                qf, Kf = q0, K0
                r, A = r0, A0
            else:
                qf, Kf, r, A = full_evaluate(y, previous)
            full_stats = matrix_stats(Kf)
            records.append(dict(step=n, time=n*DT, location=location, rule='full_PROM',
                                equation_error_scaled=0., equation_norm_scaled=float(np.linalg.norm(qf)/scale),
                                jacobian_error_relative=0., newton_step_error_relative=0.,
                                **full_stats))
            full_step = np.linalg.solve(Kf, -qf) if location == 'predictor' else None
            for name, system in systems.items():
                system.previous = previous[system.cols]
                q, K = assemble_bm_lspg_system_2D(
                    r[system.rows], A[system.rows], system.basis, system.weights,
                    system.jdx, system.jdy, DT,
                )
                sampled_q, sampled_K = system.evaluate(y)
                np.testing.assert_allclose(q, sampled_q, rtol=1e-8, atol=1e-9)
                np.testing.assert_allclose(K, sampled_K, rtol=1e-8, atol=1e-9)
                step_error = np.nan
                if full_step is not None:
                    step_error = np.linalg.norm(np.linalg.solve(K, -q)-full_step)/np.linalg.norm(full_step)
                records.append(dict(step=n, time=n*DT, location=location, rule=name,
                                    equation_error_scaled=float(np.linalg.norm(q-qf)/scale),
                                    equation_norm_scaled=float(np.linalg.norm(q)/scale),
                                    jacobian_error_relative=float(np.linalg.norm(K-Kf)/np.linalg.norm(Kf)),
                                    newton_step_error_relative=float(step_error), **matrix_stats(K)))
                if n == 1 and location == 'predictor':
                    direction = np.random.default_rng(42).normal(size=basis.shape[1])
                    direction /= np.linalg.norm(direction)
                    h = 1e-3
                    numerical = (system.evaluate(y+h*direction, False)[0]
                                 -system.evaluate(y-h*direction, False)[0])/(2*h)
                    checks[name] = float(np.linalg.norm(numerical-K@direction)/np.linalg.norm(K@direction))

        for name, system in systems.items():
            # Every rule starts from the SAME previous PROM state.
            result = system.solve(y0, 20)
            # Derivative of the converged one-step map with respect to the
            # previous reduced state: T = - (dq/dy)^-1 (dq/dy_previous).
            _, Ksol = system.evaluate(result[0])
            Acurrent = system.jac(system.reference + system.basis @ result[0]) @ system.basis
            Ap = Aprevious[system.rows]
            Acu, Acv = np.split(Acurrent, 2)
            Apu, Apv = np.split(Ap, 2)
            B = (system.weights*Acu).T @ Apu + (system.weights*Acv).T @ Apv
            transition = -np.linalg.solve(Ksol, B)
            propagation.append(dict(step=n, rule=name,
                                    map_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(transition)))),
                                    map_max_amplification=float(np.linalg.norm(transition, 2)),
                                    map_error_relative=float(np.linalg.norm(transition-Tref)/np.linalg.norm(Tref))))
            if n == 1 and result[3]:
                # Validate the largest local amplification by perturbing the
                # previous state and actually resolving the nonlinear step.
                _, singular, right = np.linalg.svd(transition)
                direction = right[0]
                eps = 1e-3
                previous_base = system.previous.copy()
                perturbation = eps * (system.basis @ direction)
                system.previous = previous_base + perturbation
                plus = system.solve(result[0], 20)
                system.previous = previous_base - perturbation
                minus = system.solve(result[0], 20)
                system.previous = previous_base
                measured = (plus[0]-minus[0])/(2*eps)
                predicted = transition @ direction
                propagation_checks.append(dict(rule=name, epsilon=eps,
                    plus_converged=bool(plus[3]), minus_converged=bool(minus[3]),
                    predicted_amplification=float(singular[0]),
                    measured_amplification=float(np.linalg.norm(measured)),
                    derivative_error_relative=float(np.linalg.norm(measured-predicted)/np.linalg.norm(predicted))))
            attempts = [('newton20', result[0], result[3], result[4], result[5], len(result[1]))]
            if name.startswith('bm_90') and not result[3]:
                extra = system.solve(y0, 100)
                attempts.append(('newton100', extra[0], extra[3], extra[4], extra[5], len(extra[1])))
                if n <= 3:
                    qinit = np.linalg.norm(system.evaluate(y0, False)[0])
                    opt = least_squares(
                        lambda y: system.evaluate(y, False)[0], y0,
                        jac=lambda y: system.evaluate(y)[1], max_nfev=400,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, x_scale='jac',
                    )
                    ratio = np.linalg.norm(opt.fun) / qinit
                    attempts.append(('scipy_least_squares400', opt.x, ratio < 1e-5,
                                     ratio, f'status{opt.status}', opt.nfev))
            for solver, y, converged, ratio, reason, nit in attempts:
                wf = reference + basis @ y
                qfull, _, _, _ = full_evaluate(y, previous, False)
                solves.append(dict(step=n, rule=name, solver=solver, converged=bool(converged),
                                   final_projected_ratio=float(ratio), stop_reason=reason, iterations=int(nit),
                                   state_error_vs_PROM=float(np.linalg.norm(wf-states[:,n])/np.linalg.norm(states[:,n])),
                                   coordinate_error_vs_PROM=float(np.linalg.norm(y-coords[:,n])),
                                   full_equation_norm_scaled=float(np.linalg.norm(qfull)/scale),
                                   min_state=float(wf.min()), max_state=float(wf.max())))
        write_csv(out/'equations.csv', records)
        write_csv(out/'one_step_solves.csv', solves)
        write_csv(out/'propagation.csv', propagation)
        if propagation_checks:
            write_csv(out/'propagation_checks.csv', propagation_checks)
        print(f'Diagnosed reference step {n}/{args.reference_steps}', flush=True)

    for name in ('bm_90_offset3', 'bm_90_offset1', 'bm_225_offset1'):
        suffix = {'bm_90_offset3': '', 'bm_90_offset1': '_offset1',
                  'bm_225_offset1': '_offset1_snap5pct'}[name]
        path = Path('Results') / f'hprom_bm_ecsw_tol1e-10{suffix}_iter20_snaps_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}.npy'
        if path.exists():
            trajectory = np.load(path, mmap_mode='r')
            err = np.linalg.norm(trajectory[:,:args.reference_steps+1]-states, axis=0)/np.linalg.norm(states,axis=0)
            trajectory_errors[name] = err.tolist()

    provenance = {
        'mu': mu, 'dt': DT, 'reference_steps': args.reference_steps,
        'reference_solver': 'production full-mesh LSPG Gauss-Newton; lstsq; max20; plateau1e-2',
        'reference_stats': [float(x) for x in stats],
        'reference_seconds': reference_seconds, 'total_seconds': time.time()-t0, 'evaluated_steps': steps,
        'equation_error_normalization': 'norm of full PROM equation at same-step predictor; avoids dividing by near-zero solution residual',
        'weight_files': {name: {'path': str(Path('POD')/fn), 'sha256': digest(Path('POD')/fn),
                               'cells': int(systems[name].cells.size)} for name,fn in RULES.items()},
        'basis_sha256': digest('POD/basis.npy'), 'reference_sha256': digest('POD/u_ref.npy'),
        'directional_finite_difference_errors': checks,
        'trajectory_errors_vs_PROM': trajectory_errors,
        'blas': threadpool_info(),
        'limitations': ['One evaluation parameter; first reference_steps steps only.',
                        'Newton solves for scalar classical weights are diagnostic; the production classical method uses Gauss-Newton.',
                        'Training count alone is not isolated: greedy selection changes cells and weights as well.',
                        'Failure of a nonlinear solver does not prove that the equations have no root.'],
    }
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in RULES:
        rows = [row for row in records if row['rule']==name and row['location']=='predictor']
        times = [row['time'] for row in rows]
        for ax, key in zip(axes, ('equation_error_scaled','jacobian_error_relative','condition')):
            ax.semilogy(times, [max(row[key],1e-16) for row in rows], '.-', label=name)
    for ax, title in zip(axes, ('Equation error / full predictor norm', 'Relative Jacobian error', 'Jacobian condition number')):
        ax.set(title=title, xlabel='Time')
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/'equations_and_jacobians.png', dpi=180)
    plt.close(fig)
    write_diagnostic_summary(out)
    print(f'Completed. Reports: {out}', flush=True)


if __name__ == '__main__':
    main()
