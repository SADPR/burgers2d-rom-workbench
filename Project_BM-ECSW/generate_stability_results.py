#!/usr/bin/env python3
"""Generate Appendix A.4 Burgers tables/figure from saved diagnostics only."""
import csv
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from threadpoolctl import threadpool_limits


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_step_table(out, document, additional_results=()):
    """Read saved one-step probes and write Table 6 without rerunning solvers."""
    probes = read_csv(out/'probes.csv')
    base_meta = json.loads((out/'provenance.json').read_text())
    for path in additional_results:
        folder = Path(path)
        if json.loads((folder/'status.json').read_text())['stage'] != 'complete':
            raise RuntimeError(f'Additional one-step probes have not completed: {folder}')
        meta = json.loads((folder/'provenance.json').read_text())
        for name in ('POD/basis.npy', 'POD/u_ref.npy'):
            if meta['sources'][name] != base_meta['sources'][name]:
                raise ValueError(f'Additional probes use a different reference: {name}')
        extra = read_csv(folder/'probes.csv')
        existing = {(r['rule'], r['step']) for r in probes}
        if any((r['rule'], r['step']) in existing for r in extra):
            raise ValueError(f'Duplicate one-step probe in {folder}')
        probes.extend(extra)
    if any(r['root_converged'] != 'True' or float(r['root_ratio']) >= 1e-11 for r in probes):
        raise ValueError('One-step equations did not reach the common residual tolerance.')
    models = [('PROM', 'PROM'),
              ('classical_90_offset3', 'HPROM (ECSW), $N_c=90$'),
              ('classical_225_offset1', 'HPROM (ECSW), $N_c=225$'),
              ('bm_90_offset1', 'HPROM (BM-ECSW), $N_c=90$'),
              ('bm_225_offset1', 'HPROM (BM-ECSW), $N_c=225$')]
    rows = []
    for name, label in models:
        candidates = [r for r in probes if r['rule']==name and r['step']=='1']
        if not candidates and name == 'classical_90_offset3':
            continue
        if len(candidates) != 1:
            raise ValueError(f'Expected one first-step probe: {name}')
        r = candidates[0]
        condition = f"{float(r['condition']):,.2f}".replace(',', r'\,')
        error = float(r['error_vs_PROM_percent'])
        error_text = '0' if error == 0 else f'{error:.3f}'
        if 0 < error < .001:
            mantissa, exponent = f'{error:.2e}'.split('e')
            error_text = rf'${mantissa}\times10^{{{int(exponent)}}}$'
        rows.append(f"{label} & ${float(r['sym_ratio']):+.3f}$ & {condition} & {error_text}\\\\")
    table = '\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n'
    table += r'Model & $\lambda_{\min}/\lambda_{\max}$ & $\kappa_2(\widehat{\matr{J}}_n)$ & State error [\%]\\'+'\n\\midrule\n'
    table += '\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (document/'tables/burgers_step_sensitivity.tex').write_text(table)
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='Results/BM_ECSW_stability_20261009')
    parser.add_argument('--document', default='Project_BM-ECSW')
    parser.add_argument('--additional-results', action='append', default=[],
                        help='Additional saved online comparisons to include in Table 5.')
    parser.add_argument('--additional-probes', action='append', default=[],
                        help='Additional saved one-step comparisons to include in Table 6.')
    parser.add_argument('--only-step-table', action='store_true',
                        help='Regenerate only Table 6, preserving the other tables and figures.')
    args = parser.parse_args()
    out, document = Path(args.results), Path(args.document)
    if json.loads((out/'status.json').read_text())['stage'] != 'complete':
        raise RuntimeError('Diagnostics have not completed.')
    manifest = out/'table_inputs.json'
    inputs = json.loads(manifest.read_text()) if manifest.exists() else {}
    additional_probes = args.additional_probes or inputs.get('additional_probe_results', [])
    if args.additional_probes:
        inputs['additional_probe_results'] = args.additional_probes
    if args.only_step_table:
        probes = write_step_table(out, document, additional_probes)
        if args.additional_probes:
            manifest.write_text(json.dumps(inputs, indent=2)+'\n')
        print(f'Table 6 regenerated from {len(probes)} saved one-step probes.')
        return
    runs = json.loads((out/'runs.json').read_text())
    provenance = json.loads((out/'provenance.json').read_text())
    sources = {run['rule']:out for run in runs}
    supplements = args.additional_results
    if not supplements and manifest.exists():
        supplements = inputs.get('additional_results', [])
    for path in supplements:
        folder = Path(path)
        if json.loads((folder/'status.json').read_text())['stage'] != 'complete':
            raise RuntimeError(f'Additional diagnostics have not completed: {folder}')
        extra = json.loads((folder/'runs.json').read_text())
        if any(run['rule'] in sources for run in extra):
            raise ValueError(f'Duplicate rule in additional results: {folder}')
        extra_meta = json.loads((folder/'provenance.json').read_text())
        for name in ('POD/basis.npy', 'POD/u_ref.npy', 'burgers/gauss_newton.py', 'burgers/linear_manifold.py'):
            if extra_meta['sources'][name] != provenance['sources'][name]:
                raise ValueError(f'Basis or solver differs in additional results: {name}')
        sources.update({run['rule']:folder for run in extra})
        provenance['weights'].update(extra_meta['weights'])
        runs.extend(extra)
    if supplements or args.additional_probes:
        if supplements:
            inputs['additional_results'] = supplements
        manifest.write_text(json.dumps(inputs, indent=2)+'\n')
    norms = {}
    for run in runs:
        mu = run['mu']
        key = tuple(mu)
        tag = f"{run['rule']}_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}"
        if key not in norms:
            hdm = np.load(run['hdm_file'], mmap_mode='r')
            norms[key] = np.einsum('ij,ij->j', hdm, hdm)
        err = np.load(sources[run['rule']]/f'{tag}_errors.npy')
        run['trajectory_error_percent'] = float(100*np.sqrt(np.sum(err**2*norms[key])/np.sum(norms[key])))
    (out/'trajectory_summary.json').write_text(json.dumps(runs, indent=2)+'\n')
    rules = ('classical_225_offset1', 'bm_90_offset1', 'bm_225_offset1')
    table_rules = [('classical_225_offset1',225,'HPROM (ECSW)'),
                   ('bm_90_offset1',90,'HPROM (BM-ECSW)'),
                   ('bm_225_offset1',225,'HPROM (BM-ECSW)')]
    if 'classical_90_offset3' in sources:
        table_rules.insert(0,('classical_90_offset3',90,'HPROM (ECSW)'))
    rows = []
    for rule, count, label in table_rules:
        records = [r for r in runs if r['rule'] == rule]
        matrices = []
        for r in records:
            mu = r['mu']
            tag = f"{rule}_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}"
            matrices += [m for m in read_csv(sources[rule]/f'{tag}_matrices.csv') if m['evaluation']=='0']
        if len(matrices) != 1500 or len(records) != 3:
            raise ValueError(f'Table 5 requires three complete 500-step runs: {rule}')
        ratio = np.median([float(m['sym_ratio']) for m in matrices])
        negative = sum(float(m['sym_ratio']) < -1e-12 for m in matrices)
        cells = provenance['weights'][rule]['cells']
        error = max(r['trajectory_error_percent'] for r in records)
        cellstr = f'{cells:,}'.replace(',', r'\,')
        negstr = f'{negative:,}'.replace(',', r'\,')
        errstr = f'{error:,.3f}'.replace(',', r'\,')
        rows.append(f'{label} & {count} & {cellstr} & ${ratio:+.3f}$ & ${negstr}/1\\,500$ & {errstr}\\\\')
    table = r'''\begin{tabular}{@{}lrrrrr@{}}
\toprule
Model & $N_c$ & $n_e$ & Median $\lambda_{\min}/\lambda_{\max}$ & Indefinite & $\mathbb{RE}_{2,\vect{u}}^{\max}$ [\%]\\
\midrule
'''+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
'''
    (document/'tables/burgers_stability.tex').write_text(table)
    probes = write_step_table(out, document, additional_probes)
    plt.rcParams.update({'font.family':'serif', 'font.size':11, 'mathtext.fontset':'cm'})
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.05))
    plot_styles = {
        ('HPROM (ECSW)', 90): ('black', ':'),
        ('HPROM (ECSW)', 225): ('black', '-'),
        ('HPROM (BM-ECSW)', 90): ('#cf2424', '-'),
        ('HPROM (BM-ECSW)', 225): ('#1465b5', '--'),
    }
    for name, count, model in table_rules:
        color, style = plot_styles[model, count]
        label = f'{model}, $N_c={count}$'
        tag = f'{name}_mu1_4.56_mu2_0.019'
        errors = np.load(sources[name]/f'{tag}_errors.npy')
        matrices = [m for m in read_csv(sources[name]/f'{tag}_matrices.csv') if m['evaluation']=='0']
        mark = (dict(marker='o', markevery=50, markersize=4,
                     markerfacecolor='none', markeredgewidth=.9, zorder=4)
                if model == 'HPROM (ECSW)' and count == 90 else {})
        axes[0].semilogy(np.arange(len(errors))*.05, errors*100, color=color, linestyle=style, lw=2, alpha=.83, label=label, **mark)
        axes[1].plot([int(m['step'])*.05 for m in matrices], [float(m['sym_ratio']) for m in matrices],
                     color=color, linestyle=style, lw=2, alpha=.83, **mark)
    axes[0].set(ylabel='Instantaneous relative state error [%]', title='(a) Error against the HDM')
    axes[1].set(ylabel=r'$\lambda_{\min}/\lambda_{\max}$', title=r'(b) Symmetric part of $\widehat{\mathbf{J}}_n$')
    axes[1].axhline(0, color='.5', lw=.8)
    for ax in axes:
        ax.set(xlabel='Time', xlim=(0,25))
        ax.grid(alpha=.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0,.14,1,1))
    fig.savefig(document/'figures/burgers_bm_stability.pdf', bbox_inches='tight')
    fig.savefig('/tmp/burgers_bm_stability.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    checks = read_csv(out/'derivative_checks.csv')
    assert all(r['root_converged']=='True' for r in probes)
    assert max(float(r['map_derivative_error']) for r in checks) < 2e-10
    lines = ['Burgers BM-ECSW stability diagnostics', '',
        'No retraining. Saved 90-pair and 225-pair rules.',
        f'{len(runs)} fresh trajectories: {len(table_rules)} rules, three parameters, 500 steps each.',
        '96 POD coordinates; dt=0.05; at most 10 native threads.',
        'Production classical GN / BM GN-type with max20, relative1e-5, plateau0.01.',
        'Timing is instrumented and is NOT a speed benchmark.', '',
        'rule | mu | trajectory error (%) | indefinite predictors | stopping counts']
    for r in runs:
        lines.append(f"{r['rule']} | {r['mu']} | {r['trajectory_error_percent']:.10g} | "
                     f"{r['predictor_indefinite_steps']}/500 | {r['stop_reasons']}")
    lines += ['', 'Common previous-state probes: steps1,5,20 at mu=(4.56,0.019).',
        ('The additional classical 90-pair rule is included in the online and one-step comparisons.'
         if any(r['rule']=='classical_90_offset3' for r in probes) else
         'The additional classical 90-pair rule is included in Table 5 online measurements only.'),
        'These locate roots with exact derivatives for diagnostics only.',
        f'All {len(probes)} roots attained relative projected residual <1e-11.',
        'First-step results: rule | state error vs PROM (%) | ||T||2 | sym ratio | cond2(Jhat)']
    for r in probes:
        if r['step']=='1':
            lines.append(' | '.join(r[k] for k in ('rule','error_vs_PROM_percent','amplification','sym_ratio','condition')))
    lines += ['', 'Derivative validation (central differences, epsilon=1e-3):']
    for key in ('current_derivative_error','history_derivative_error','map_derivative_error'):
        lines.append(f'Maximum relative {key}: {max(float(r[key]) for r in checks):.6e}')
    lines += ['The perturbed re-solves reached relative residuals of order 1e-10.',
        'They stopped at numerical precision before the requested 1e-11 relative threshold',
        '(relative to a much smaller initial residual); derivative_checks.csv retains these false flags.',
        'The resolved central differences agree with the predicted map to <2e-10 relative error.', '',
        'Interpretation and limits:',
        '- Training accuracy does not ensure accurate online equations.',
        '- The 90-pair rule has indefinite symmetric parts AND eigenvalues with negative real parts.',
        '- This differs from the AERO-F example with positive real eigenvalue parts.',
        '- Selection, weights, and training coverage change together; no isolated causal attribution.',
        '- Jhat is a nonlinear correction approximation, not the physical-time perturbation operator.',
        '- T=-J_n^{-1}B_n uses exact derivatives at a nonsingular implicit root.',
        '- Local amplification>1 is not by itself a proof of long-time instability.',
        '- 225 pairs are not a universal threshold or a gradient-structure guarantee.', '',
        'Reproduce from repository root (use a NEW output directory):',
        'OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 .venv/bin/python diagnose_bm_stability.py --threads 10 --output-dir Results/BM_ECSW_stability_new',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Project_BM-ECSW/generate_stability_results.py --results Results/BM_ECSW_stability_new',
        'Input hashes, solver-source hashes, and protocol are in provenance.json.', '']
    (out/'summary.txt').write_text('\n'.join(lines))
    print(json.dumps([{k:r[k] for k in ('rule','mu','trajectory_error_percent','stop_reasons','nonpositive_real_evaluations','matrix_evaluations')} for r in runs], indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
