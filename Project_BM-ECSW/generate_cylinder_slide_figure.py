#!/usr/bin/env python3
"""Draw the cylinder slide's coordinate norm from saved trajectories.

Reuses the data selection of simulations/plot_bmecsw_divergence.py.
No trajectories, weights, or matrix diagnostics are recomputed.
"""
from pathlib import Path
import argparse
import importlib.util
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-dir', type=Path,
                        default=Path('/home/sares/aero-f-fpc-rom-workbench/simulations'))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'figures' / 'aerof_bm_coordinate_norm_slides.pdf')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('cylinder_figure_source',
                                                 args.sim_dir / 'plot_bmecsw_divergence.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    plt.rcParams.update({
        'text.usetex': shutil.which('latex') is not None,
        'font.family': 'serif', 'font.size': 13,
        'axes.labelsize': 16, 'xtick.labelsize': 13,
        'ytick.labelsize': 13, 'legend.fontsize': 11.5,
    })
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for label, _, online, _, color, style in source.CASES:
        if label == 'HDM':
            coordinates = np.loadtxt(args.sim_dir / 'run.offline.param.01' / 'nonlinearrom'
                                    / 'cluster0' / 'state.coords', delimiter=',', skiprows=1)[3 * 751:4 * 751]
            time = 0.2 * np.arange(len(coordinates))
            norm = np.linalg.norm(coordinates, axis=1)
        else:
            time, norm = source.reduced_norm(online)
        if label.startswith('HPROM (ECSW)'):
            color = '#2ca02c'
        visible = time <= source.T_MAX
        ax.semilogy(time[visible], norm[visible], style, color=color,
                    linewidth=2.2 if label == 'HDM' else 1.9, label=label)
        if online and time[-1] < source.T_MAX - 1e-6:
            ax.plot(time[-1], norm[-1], 'x', color=color, markersize=9, markeredgewidth=2.3)
    ax.set_xlabel(r'$t$', labelpad=1)
    ax.set_ylabel(r'$\|\mathbf{q}(t)\|_2$', labelpad=3)
    ax.set_xlim(0, source.T_MAX)
    ax.set_yticks([1e-1, 1, 10, 100, 1e3])
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
              labelspacing=0.25, borderpad=0.35, handlelength=2.5)
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.24, top=0.96)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
