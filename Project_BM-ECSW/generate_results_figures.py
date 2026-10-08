#!/usr/bin/env python3
"""Generate implementation-note figures and tables from saved ECSW/BM data."""
import csv
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from burgers.config import GRID_X, GRID_Y
from burgers.ecsw_utils import generate_augmented_mesh

OUT = Path(__file__).resolve().parent / 'figures'
TABLES = OUT.parent / 'tables'
BENCH = REPO / 'Results/ECSW_BM_GN_benchmark_20261005'
FOM_BENCH = REPO / 'Results/FOM_benchmark_20261004'
OFFLINE_BENCH = REPO / 'Results/ECSW_BM_offline_benchmark_20261005'
POINTS = [(4.56, .019), (4.75, .020), (5.19, .026)]
RULES = [
    ('ecsw', 'HPROM (ECSW)', 'ecsw_weights_lspg_ecsw_tol1e-10_offset1_snap5pct.npy'),
    ('bm_ecsw', 'HPROM (BM-ECSW)', 'ecsw_weights_lspg_bm_ecsw_tol1e-10_offset1_snap5pct.npy'),
]


def latex_number(value, decimals=None):
    """Format quantities with a thin space between groups of three digits."""
    spec = ',d' if decimals is None else f',.{decimals}f'
    return format(value, spec).replace(',', r'\,')


def run_rows(runs, method, mu1, mu2):
    return sorted((r for r in runs if r['method'] == method
                   and float(r['mu1']) == mu1 and float(r['mu2']) == mu2),
                  key=lambda r: int(r['repeat']))


def write_table(name, columns, header, rows):
    content = [rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}', r'\toprule',
               header + r'\\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']
    (TABLES / name).write_text('\n'.join(content) + '\n')


def trained_rules():
    comparison = (REPO / 'Results/bm_ecsw_lspg_comparison.txt').read_text()
    norms = {}
    training_labels = {'ecsw': 'Classical ECSW', 'bm_ecsw': 'BM-ECSW'}
    for method, label, _ in RULES:
        match = re.search(re.escape(training_labels[method]) + r':.*?training norm error ([0-9.eE+-]+)',
                          comparison, flags=re.S)
        if match is None:
            raise ValueError(f'Missing training residual for {label}')
        norms[method] = float(match.group(1))
    rules = []
    for method, label, filename in RULES:
        weights = np.load(REPO / 'POD' / filename)
        positive = np.any(weights > 0, axis=1) if weights.ndim == 2 else weights > 0
        cells = np.flatnonzero(positive)
        augmented = generate_augmented_mesh(GRID_X, GRID_Y, cells)
        rules.append((method, label, cells, augmented, np.count_nonzero(weights), norms[method]))
    return rules


def sampled_meshes(rules):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), sharex=True, sharey=True)
    cmap = ListedColormap(['white', '#c5d5e8', '#195487'])
    for ax, (_, label, cells, augmented, _, _) in zip(axes, rules):
        mesh = np.zeros((GRID_Y.size - 1) * (GRID_X.size - 1), dtype=int)
        mesh[augmented] = 1
        mesh[cells] = 2
        ax.imshow(mesh.reshape(GRID_Y.size - 1, GRID_X.size - 1), origin='lower',
                  extent=(0, 100, 0, 100), cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        ax.set(title=f'{label}\n${latex_number(cells.size)}$ weighted; '
                 f'${latex_number(augmented.size)}$ augmented', xlabel='$x$')
    axes[0].set_ylabel('$y$')
    fig.legend(handles=[Patch(color='#195487', label='Weighted cells'),
                        Patch(color='#c5d5e8', label='Additional stencil cells')],
               loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUT / 'paper_sampled_cells.pdf', bbox_inches='tight')
    plt.close(fig)


def solution_slices(runs):
    paths = [REPO / 'Results/param_snaps/mu1_4.56+mu2_0.019.npy']
    for method in ('ecsw', 'bm_ecsw'):
        record = run_rows(runs, method, 4.56, .019)[0]
        report = dict(line.split(': ', 1) for line in
                      (BENCH / record['report']).read_text().splitlines()
                      if ': ' in line)
        path = Path(report['hprom_snapshots_npy'])
        paths.append(path if path.is_absolute() else REPO / path)
    trajectories = [np.load(path, mmap_mode='r') for path in paths]
    labels = ['HDM', 'HPROM (ECSW)', 'HPROM (BM-ECSW)']
    styles = [('black', '-'), ('#245782', '--'), ('#c46b16', ':')]
    x = .5 * (GRID_X[1:] + GRID_X[:-1])
    y = .5 * (GRID_Y[1:] + GRID_Y[:-1])
    row = int(np.argmin(np.abs(y - 50.2)))
    indices = row * x.size + np.arange(x.size)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, time, step in zip(axes, (5, 15, 25), (100, 300, 500)):
        for data, label, (color, style) in zip(trajectories, labels, styles):
            ax.plot(x, data[indices, step], linestyle=style, color=color, label=label, linewidth=1.5)
        ax.set(title=f'$t={time}$', xlabel='$x$', xlim=(0, 100))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('$u_x$')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(OUT / 'paper_solution_slices.pdf', bbox_inches='tight')
    plt.close(fig)


def timing_plot(runs, rules):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(3)
    width = .32
    means, sds = {}, {}
    labels = {method: f'{label}: ${latex_number(cells.size)}$ cells'
              for method, label, cells, _, _, _ in rules}
    for method, shift, color in [
        ('ecsw', -width/2, '#245782'),
        ('bm_ecsw', width/2, '#238561'),
    ]:
        values = [np.array([float(r['online_seconds']) for r in run_rows(runs, method, *point)])
                  for point in POINTS]
        means[method] = np.array([v.mean() for v in values])
        sds[method] = np.array([v.std(ddof=1) for v in values])
        ax.bar(x + shift, means[method], width, color=color, label=labels[method], alpha=.88,
               yerr=sds[method], capsize=4, error_kw={'elinewidth': 1.2})
        for idx, vals in enumerate(values):
            ax.scatter(idx + shift + np.linspace(-.055, .055, len(vals)), vals,
                       color='black', s=16, zorder=4)
    ax.set_xticks(x, [f'({a:.2f}, {b:.3f})' for a, b in POINTS])
    ax.set(xlabel='Evaluation parameters $(\\mu_1,\\mu_2)$',
           ylabel='Online time for 500 steps [s]',
           ylim=(0, 1.2 * max(float(row['online_seconds']) for row in runs)))
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / 'online_timings.pdf', bbox_inches='tight')
    plt.close(fig)


def cubature_table(rules):
    with (OFFLINE_BENCH / 'runs.csv').open() as stream:
        measurements = {row['method']: row for row in csv.DictReader(stream)}
    if set(measurements) != {method for method, _, _ in RULES}:
        raise ValueError('Both fresh offline measurements must finish before generating Table 1.')
    rows = []
    for method, label, cells, augmented, _, error in rules:
        measurement = measurements[method]
        if int(measurement['cells']) != cells.size or int(measurement['augmented_cells']) != augmented.size:
            raise ValueError(f'Offline measurement does not match the published mesh: {method}')
        seconds = float(measurement['training_seconds'])
        mantissa, exponent = f'{error:.3e}'.split('e')
        rows.append(f'{label} & 96 & {latex_number(cells.size)} & {latex_number(augmented.size)} & '
                    + rf'${mantissa}\times10^{{{int(exponent)}}}$'
                    + f' & {latex_number(seconds/60, 2)}' + r'\\')
    write_table('paper_cubature.tex', 'lrrrrr',
                r'Computational model & $n$ & $n_e$ & $n_e^+$ & $\eta$ & \shortstack{Offline training\\time (min)}', rows)
    classical = float(measurements['ecsw']['training_seconds'])
    bm = float(measurements['bm_ecsw']['training_seconds'])
    metrics = [
        rf'\newcommand{{\ECSWOfflineMinutes}}{{{latex_number(classical/60, 2)}}}',
        rf'\newcommand{{\BMOfflineMinutes}}{{{latex_number(bm/60, 2)}}}',
        rf'\newcommand{{\OfflineTrainingRatio}}{{{latex_number(classical/bm, 2)}}}',
    ]
    (TABLES / 'offline_metrics.tex').write_text('\n'.join(metrics) + '\n')


def data_tables(runs, rules):
    model_names = {'ecsw': 'HPROM (ECSW)', 'bm_ecsw': 'HPROM (BM-ECSW)'}
    with (FOM_BENCH/'runs.csv').open() as stream:
        fom_rows = list(csv.DictReader(stream))
    fom_times = {(float(row['mu1']), float(row['mu2'])): float(row['fom_seconds'])
                 for row in fom_rows}
    if set(fom_times) != set(POINTS):
        raise ValueError('All three fresh FOM runs must finish before generating speedups.')
    cubature_table(rules)

    performance = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
        r'Parameter & \multicolumn{2}{c}{Global Euclidean error (\%)} & \multicolumn{3}{c}{Wall-clock time (min)} & \multicolumn{2}{c}{Speedup factor}\\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}',
        r'$(\mu_1,\mu_2)$ & \shortstack{HPROM\\(ECSW)} & \shortstack{HPROM\\(BM-ECSW)} & FOM & \shortstack{HPROM\\(ECSW)} & \shortstack{HPROM\\(BM-ECSW)} & \shortstack{HPROM\\(ECSW)} & \shortstack{HPROM\\(BM-ECSW)}\\', r'\midrule']
    timing_rows = []
    means_by_method = {method: [] for method, _, _ in RULES}
    errors_by_method = {method: [] for method, _, _ in RULES}
    for a, b in POINTS:
        means, errors = {}, {}
        for method, _, _ in RULES:
            records = run_rows(runs, method, a, b)
            values = np.array([float(r['online_seconds']) for r in records])
            means[method] = values.mean()
            errors[method] = float(records[0]['error_percent'])
            means_by_method[method].append(means[method])
            errors_by_method[method].append(errors[method])
            nums = ' & '.join(f'{latex_number(v, 3)}' for v in values)
            timing_rows.append(f'$({a:.2f},{b:.3f})$ & {model_names[method]} & {nums} & '
                               f'{latex_number(values.mean(), 3)} & {latex_number(values.std(ddof=1), 3)}' + r'\\')
        timing_rows.append(r'\addlinespace')
        t_fom = fom_times[a, b]
        performance.append(f'$({a:.2f},{b:.3f})$ & {latex_number(errors["ecsw"], 4)} & {latex_number(errors["bm_ecsw"], 4)} & '
                           f'{latex_number(t_fom/60, 3)} & {latex_number(means["ecsw"]/60, 5)} & {latex_number(means["bm_ecsw"]/60, 5)} & '
                           f'{latex_number(t_fom/means["ecsw"], 2)} & {latex_number(t_fom/means["bm_ecsw"], 2)}' + r'\\')
    performance += [r'\bottomrule', r'\end{tabular}']
    (TABLES/'paper_performance.tex').write_text('\n'.join(performance)+'\n')
    mean_times = {method: float(np.mean(values)) for method, values in means_by_method.items()}
    t_fom_mean = float(np.mean(list(fom_times.values())))
    rows = [f'FOM & {latex_number(125000)} & {latex_number(62500)} & '
            f'{latex_number(62500)} & -- & {latex_number(t_fom_mean/60, 3)} & --' + r'\\']
    for method, _, cells, augmented, _, _ in rules:
        rows.append(f'{model_names[method]} & 96 & {latex_number(cells.size)} & {latex_number(augmented.size)} & '
                    f'{latex_number(max(errors_by_method[method]), 4)} & {latex_number(mean_times[method]/60, 5)} & '
                    f'{latex_number(t_fom_mean/mean_times[method], 2)}' + r'\\')
    write_table('paper_summary.tex', 'lrrrrrr',
                r'Computational model & \shortstack{$n$\\($N$ for FOM)} & \shortstack{$n_e$\\($N_e$ for FOM)} & $n_e^+$ & $\mathbb{RE}_{2,\vect{u}}^{\max}$ (\%) & \shortstack{Wall-clock time\\(min)} & \shortstack{Speedup\\factor}', rows)
    write_table('timings.tex', 'llrrrrr',
                'Parameter & Computational model & Run 1 & Run 2 & Run 3 & Mean & SD', timing_rows)
    rows = [f'$({a:.2f},{b:.3f})$ & {latex_number(fom_times[a,b], 3)} & {latex_number(fom_times[a,b]/60, 3)}' + r'\\'
            for a,b in POINTS]
    rows.append(f'Mean & {latex_number(t_fom_mean, 3)} & {latex_number(t_fom_mean/60, 3)}' + r'\\')
    write_table('fom_timings.tex', 'lrr', 'Parameter & Wall-clock time (s) & Wall-clock time (min)', rows)
    prose = [rf'\newcommand{{\FOMMeanMinutes}}{{{latex_number(t_fom_mean/60, 3)}}}',
             rf'\newcommand{{\ECSWFOMSpeedup}}{{{latex_number(t_fom_mean/mean_times["ecsw"], 2)}}}',
             rf'\newcommand{{\BMFOMSpeedup}}{{{latex_number(t_fom_mean/mean_times["bm_ecsw"], 2)}}}']
    bm_rows = [row for row in runs if row['method'] == 'bm_ecsw']
    prose += [
        rf'\newcommand{{\ECSWMinSeconds}}{{{latex_number(min(means_by_method["ecsw"]), 2)}}}',
        rf'\newcommand{{\ECSWMaxSeconds}}{{{latex_number(max(means_by_method["ecsw"]), 2)}}}',
        rf'\newcommand{{\BMMinSeconds}}{{{latex_number(min(means_by_method["bm_ecsw"]), 2)}}}',
        rf'\newcommand{{\BMMaxSeconds}}{{{latex_number(max(means_by_method["bm_ecsw"]), 2)}}}',
        rf'\newcommand{{\ECSWMaxGlobalError}}{{{latex_number(max(errors_by_method["ecsw"]), 4)}}}',
        rf'\newcommand{{\BMMaxGlobalError}}{{{latex_number(max(errors_by_method["bm_ecsw"]), 4)}}}',
        rf'\newcommand{{\BMLargestErrorIncrease}}{{{latex_number(max(b-a for a,b in zip(errors_by_method["ecsw"], errors_by_method["bm_ecsw"])), 4)}}}',
        rf'\newcommand{{\BMMinIterationsTotal}}{{{latex_number(min(int(row["iterations"]) for row in bm_rows))}}}',
        rf'\newcommand{{\BMMaxIterationsTotal}}{{{latex_number(max(int(row["iterations"]) for row in bm_rows))}}}',
        rf'\newcommand{{\BMConvergedSteps}}{{{latex_number(sum(500-int(row["nonconverged_steps"]) for row in bm_rows))}}}',
    ]
    (TABLES/'fom_metrics.tex').write_text('\n'.join(prose)+'\n')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    with (BENCH / 'runs.csv').open() as stream:
        runs = list(csv.DictReader(stream))
    if len(runs) != 18 or any(len(run_rows(runs, method, *point)) != 3
                             for method, _, _ in RULES for point in POINTS):
        raise ValueError('The matched benchmark must contain three runs per method and parameter.')
    if any(row.get('bm_jacobian') != 'gauss_newton'
           for row in runs if row['method'] == 'bm_ecsw'):
        raise ValueError('BM figures require the Gauss-Newton-type benchmark.')
    rules = trained_rules()
    sampled_meshes(rules)
    solution_slices(runs)
    timing_plot(runs, rules)
    data_tables(runs, rules)
    print('Implementation-note figures and tables generated from saved matched-comparison data.')
