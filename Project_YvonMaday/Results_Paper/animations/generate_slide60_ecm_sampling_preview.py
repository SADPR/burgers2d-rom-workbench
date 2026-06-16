#!/usr/bin/env python3
"""Generate the Slide 60 parameter-time-aware ECM sampling visual."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burgers.ecsw_utils import build_ecsw_snapshot_plan


OUTPUT = HERE / "previews" / "slide60_ecm_parameter_time_sampling.png"


def main() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
            "font.size": 16,
        }
    )

    mu1_values = np.linspace(4.25, 5.50, 3)
    mu2_values = np.linspace(0.015, 0.030, 3)
    mu_points = np.asarray(
        [[mu1, mu2] for mu1 in mu1_values for mu2 in mu2_values],
        dtype=float,
    )

    dt = 0.05
    plan = build_ecsw_snapshot_plan(
        num_steps=500,
        snap_time_offset=3,
        num_mu=mu_points.shape[0],
        mode="global_param_time_stratified",
        total_snapshots=None,
        total_snapshots_percent=1.0,
        mu_points=mu_points,
        random_seed=42,
        ensure_mu_coverage=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    t_min = dt * plan["candidate_now_cols"][0]
    t_max = dt * plan["candidate_now_cols"][-1]
    for imu, (mu1, mu2) in enumerate(mu_points):
        ax.plot(
            [mu1, mu1],
            [mu2, mu2],
            [t_min, t_max],
            color="#aeb5bd",
            linewidth=4.5,
            alpha=0.48,
            solid_capstyle="round",
            label=(
                r"\text{Candidate parameter--time samples}"
                if imu == 0
                else None
            ),
        )

        selected_cols = np.asarray(plan["selected_now_cols_by_mu"][imu], dtype=int)
        ax.scatter(
            np.full(selected_cols.size, mu1),
            np.full(selected_cols.size, mu2),
            dt * selected_cols,
            marker="x",
            s=72,
            linewidths=2.1,
            color="#d62728",
            depthshade=False,
            label=(
                r"\text{Selected for ECM training }(1\%)"
                if imu == 0
                else None
            ),
        )

    ax.set_xlabel(r"$\mu_1$", labelpad=10)
    ax.set_ylabel(r"$\mu_2$", labelpad=12)
    ax.set_zlabel(r"$t$", labelpad=8)
    ax.set_xlim(4.18, 5.57)
    ax.set_ylim(0.014, 0.031)
    ax.set_zlim(0.0, 25.0)
    ax.set_xticks(mu1_values)
    ax.set_yticks(mu2_values)
    ax.set_zticks(np.arange(0.0, 25.1, 5.0))
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.45)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        frameon=True,
        fontsize=13,
        handlelength=2.2,
    )

    fig.subplots_adjust(left=0.01, right=0.97, bottom=0.02, top=0.98)
    fig.savefig(
        OUTPUT,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    print(OUTPUT)
    print(f"selected_per_parameter={plan['num_selected_per_mu']}")
    print(f"selected_total={plan['num_selected_total']}")


if __name__ == "__main__":
    main()
