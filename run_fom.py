#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the full-order model (HDM) for the 2D inviscid Burgers problem,
save the snapshots, and generate HDM slice plots in LaTeX-style format.
"""

import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from burgers.core import (
    load_or_compute_snaps,
    plot_snaps,
)
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    """
    Configure matplotlib to use a LaTeX-like academic style.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "figure.figsize": (12, 8),
    })


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def main(mu1=4.56, mu2=0.019, save_snaps=True, save_plot=True):
    """
    Run the HDM, save snapshots, and save HDM plot.

    Parameters
    ----------
    mu1 : float
        First problem parameter.
    mu2 : float
        Second problem parameter.
    save_snaps : bool
        Whether to save the computed HDM snapshots.
    save_plot : bool
        Whether to save the HDM slice plot.

    Returns
    -------
    elapsed_time : float
        Wall-clock time for the HDM snapshot acquisition.
    hdm_snaps : ndarray
        HDM snapshot matrix of shape (N_dofs, num_steps+1).
    """

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    results_dir = "Results"
    snap_folder = os.path.join(results_dir, "param_snaps")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot style
    # ------------------------------------------------------------------
    set_latex_plot_style()

    # ------------------------------------------------------------------
    # Time-stepping, grid, and initial condition from config
    # ------------------------------------------------------------------
    dt = DT
    num_steps = NUM_STEPS
    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    mu_rom = [mu1, mu2]
    snaps_path = None
    fig_path = None

    # ------------------------------------------------------------------
    # HDM computation / loading
    # ------------------------------------------------------------------
    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(
        mu=mu_rom,
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        snap_folder=snap_folder,
    )
    elapsed_time = time.time() - t0

    print(f"Elapsed HDM time: {elapsed_time:.3e} seconds")

    # ------------------------------------------------------------------
    # Save HDM snapshots
    # ------------------------------------------------------------------
    if save_snaps:
        snaps_path = os.path.join(
            results_dir,
            f"hdm_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy"
        )
        np.save(snaps_path, hdm_snaps)
        print(f"HDM snapshots saved to: {snaps_path}")

    # ------------------------------------------------------------------
    # Plot HDM slices only
    # ------------------------------------------------------------------
    if save_plot:
        snaps_to_plot = range(0, num_steps + 1, 100)

        fig, ax1, ax2 = plot_snaps(
            grid_x,
            grid_y,
            hdm_snaps,
            snaps_to_plot,
            label="HDM",
            color="black",
            linewidth=2.8,
            linestyle="solid",
        )

        title_str = rf"$\mu_1 = {mu_rom[0]:.2f}, \mu_2 = {mu_rom[1]:.3f}$"
        fig.suptitle(title_str, y=0.98)

        ax1.legend(loc="best", frameon=True)
        ax2.legend(loc="best", frameon=True)

        plt.tight_layout()

        fig_path = os.path.join(
            results_dir,
            f"hdm_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"HDM plot saved to: {fig_path}")

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(
        results_dir,
        f"fom_summary_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("mu1", mu_rom[0]),
                    ("mu2", mu_rom[1]),
                ],
            ),
            (
                "configuration",
                [
                    ("save_snaps", save_snaps),
                    ("save_plot", save_plot),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "discretization",
                [
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("num_cells_x", num_cells_x),
                    ("num_cells_y", num_cells_y),
                    ("full_state_size", w0.size),
                ],
            ),
            (
                "fom_timing",
                [
                    ("total_hdm_time_seconds", elapsed_time),
                    ("avg_hdm_time_per_step_seconds", elapsed_time / num_steps),
                ],
            ),
            (
                "snapshot_stats",
                [
                    ("snapshot_shape", hdm_snaps.shape),
                    ("snapshot_l2_norm", np.linalg.norm(hdm_snaps)),
                    ("snapshot_min", np.min(hdm_snaps)),
                    ("snapshot_max", np.max(hdm_snaps)),
                ],
            ),
            (
                "outputs",
                [
                    ("hdm_snapshots_npy", snaps_path),
                    ("hdm_plot_png", fig_path),
                ],
            ),
        ],
    )
    print(f"FOM text summary saved to: {report_path}")

    return elapsed_time, hdm_snaps


if __name__ == "__main__":
    main()
