#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build two PROM comparison figures for the 250x250 campaign:
1) Baseline PROM
2) Enriched PROM

Each figure contains three points (two off-grid + one verification), and overlays:
HDM (black), Case 1 (red), Case 2 (blue), Case 3 (green), Data-driven (orange).
"""

import glob
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import make_2D_grid


def set_latex_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.1,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
    })


def _find_one(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern:\n{pattern}")
    return matches[0]


def _find_hdm_snap(hdm_dirs, mu1, mu2):
    for hdm_dir in hdm_dirs:
        if not os.path.isdir(hdm_dir):
            continue
        candidates = [
            os.path.join(hdm_dir, f"mu1_{mu1:g}+mu2_{mu2:g}.npy"),
            os.path.join(hdm_dir, f"mu1_{mu1:.2f}+mu2_{mu2:.3f}.npy"),
            os.path.join(hdm_dir, f"mu1_{mu1:.3f}+mu2_{mu2:.4f}.npy"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        try:
            return _find_one(os.path.join(hdm_dir, f"mu1_{mu1:.2f}*mu2_{mu2:.3f}*.npy"))
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"HDM snapshot not found for mu=({mu1},{mu2}) in: {hdm_dirs}")


def _case_snap(results_root, case_id, mu1, mu2, backend, enriched=False):
    if backend not in {"prom", "hprom"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if enriched:
        pat = os.path.join(
            results_root,
            f"Runs/Case{case_id}/case{case_id}_{backend}_ann_enriched_mu1_{mu1:.3f}_mu2_{mu2:.4f}_n*_ntot*_snaps.npy",
        )
    else:
        pat = os.path.join(
            results_root,
            f"Runs/Case{case_id}/case{case_id}_{backend}_ann_mu1_{mu1:.3f}_mu2_{mu2:.4f}_n*_ntot*_snaps.npy",
        )
    return _find_one(pat)


def _dd_snap(results_root, mu1, mu2, enriched=False):
    if enriched:
        pat = os.path.join(
            results_root,
            f"Runs/DataDriven/rom_data_driven_enriched_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot*/rom_snaps.npy",
        )
    else:
        pat = os.path.join(
            results_root,
            f"Runs/DataDriven/rom_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot*/rom_snaps.npy",
        )
    return _find_one(pat)


def _load_all_snaps(results_root, hdm_dirs, mu1, mu2, backend, enriched=False):
    paths = {
        "HDM": _find_hdm_snap(hdm_dirs, mu1, mu2),
        "Case 1": _case_snap(results_root, 1, mu1, mu2, backend=backend, enriched=enriched),
        "Case 2": _case_snap(results_root, 2, mu1, mu2, backend=backend, enriched=enriched),
        "Case 3": _case_snap(results_root, 3, mu1, mu2, backend=backend, enriched=enriched),
        "Data-driven": _dd_snap(results_root, mu1, mu2, enriched=enriched),
    }
    snaps = {name: np.load(path) for name, path in paths.items()}
    ref_shape = snaps["HDM"].shape
    for name, arr in snaps.items():
        if arr.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for {name} at mu=({mu1},{mu2}). "
                f"Expected {ref_shape}, got {arr.shape}."
            )
    return snaps


def _plot_group(output_path, title, results_root, hdm_dirs, mu_list, backend):
    # 250x250 mesh
    grid_x, grid_y = make_2D_grid(0, 100, 0, 100, 250, 250)
    x = 0.5 * (grid_x[1:] + grid_x[:-1])
    y = 0.5 * (grid_y[1:] + grid_y[:-1])
    nx = x.size
    ny = y.size
    mid_x = nx // 2
    mid_y = ny // 2

    # Keep enough temporal context while reducing clutter in multi-model overlays.
    steps = [0, 125, 250, 375, 500]
    final_step = steps[-1]
    common_alpha = 0.80

    colors = {
        "HDM": "black",
        "Case 1": "red",
        "Case 2": "blue",
        "Case 3": "green",
        "Data-driven": "orange",
    }

    nrows = len(mu_list)
    fig, axs = plt.subplots(nrows, 2, figsize=(14, 4.2 * nrows), constrained_layout=False)
    if nrows == 1:
        axs = np.array([axs])

    for row, (mu1, mu2, enriched) in enumerate(mu_list):
        snaps = _load_all_snaps(
            results_root,
            hdm_dirs,
            mu1,
            mu2,
            backend=backend,
            enriched=enriched,
        )

        for model_name in ["HDM", "Case 1", "Case 2", "Case 3", "Data-driven"]:
            arr = snaps[model_name]
            for ind in steps:
                is_final = ind == final_step
                label = model_name if is_final else None
                snap_u = arr[: nx * ny, ind].reshape(ny, nx)

                if is_final:
                    alpha = common_alpha
                    linewidth = 2.2
                    linestyle = "-"
                    zorder = 5
                else:
                    alpha = common_alpha
                    linewidth = 0.95
                    linestyle = "--"
                    zorder = 2

                axs[row, 0].plot(
                    x,
                    snap_u[mid_y, :],
                    color=colors[model_name],
                    alpha=alpha,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                    label=label,
                )
                axs[row, 1].plot(
                    y,
                    snap_u[:, mid_x],
                    color=colors[model_name],
                    alpha=alpha,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                    label=label,
                )

        axs[row, 0].set_title(rf"$\mu=({mu1:.2f},{mu2:.4f})$: $u_x(x,y_{{mid}})$")
        axs[row, 1].set_title(rf"$\mu=({mu1:.2f},{mu2:.4f})$: $u_x(x_{{mid}},y)$")
        axs[row, 0].set_xlabel(r"$x$")
        axs[row, 1].set_xlabel(r"$y$")
        axs[row, 0].set_ylabel(r"$u_x$")
        axs[row, 1].set_ylabel(r"$u_x$")
        axs[row, 0].grid(True)
        axs[row, 1].grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.035, fontsize=16)
    fig.text(
        0.5,
        0.012,
        r"Dashed curves: intermediate times; solid curves: final time $t=25$.",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.94])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    set_latex_plot_style()

    hdm_dirs = [
        os.path.join(THIS_DIR, "param_snaps"),
        os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
    ]
    baseline_root = os.path.join(THIS_DIR, "Results")
    enriched_root = os.path.join(THIS_DIR, "Results_Enrichment")
    out_dir = os.path.join(THIS_DIR, "Figures")
    os.makedirs(out_dir, exist_ok=True)

    test_points = [
        (4.875, 0.0225),
        (4.56, 0.0190),
        (5.19, 0.0260),
    ]

    outputs = []

    backend = "prom"
    baseline_out = os.path.join(out_dir, f"baseline_{backend}_hdm_vs_all_models.png")
    enriched_out = os.path.join(out_dir, f"enriched_{backend}_hdm_vs_all_models.png")

    _plot_group(
        output_path=baseline_out,
        title="Baseline (9-point training, PROM online): HDM vs Case1/Case2/Case3/Data-driven",
        results_root=baseline_root,
        hdm_dirs=hdm_dirs,
        mu_list=[(mu1, mu2, False) for (mu1, mu2) in test_points],
        backend=backend,
    )
    _plot_group(
        output_path=enriched_out,
        title="Enriched (9+20 training, PROM online): HDM vs Case1/Case2/Case3/Data-driven",
        results_root=enriched_root,
        hdm_dirs=hdm_dirs,
        mu_list=[(mu1, mu2, True) for (mu1, mu2) in test_points],
        backend=backend,
    )
    outputs.extend((baseline_out, enriched_out))

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
