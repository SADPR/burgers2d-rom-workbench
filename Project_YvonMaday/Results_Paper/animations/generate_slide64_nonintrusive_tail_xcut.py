#!/usr/bin/env python3
"""Generate a static final-time x-cut comparison for the slide after 63."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate_burgers_presentation_assets as assets


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "slide64_nonintrusive_tail_xcut_mu2_tfinal.png"

POINT = assets.POINTS[2]
TIME_INDEX = assets.NT - 1


def load_xcut_data() -> dict[str, np.ndarray]:
    data = {
        "HDM": assets.snapshot_cut_data(assets.hdm_path(POINT))[0][:, TIME_INDEX],
        "PROM-POD-AE": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["podae"], POINT)
        )[0][:, TIME_INDEX],
        "POD-NN-ROM": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["podnn"], POINT)
        )[0][:, TIME_INDEX],
        "POD-DL-ROM": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["poddl"], POINT)
        )[0][:, TIME_INDEX],
    }
    return {key: np.asarray(value, dtype=float) for key, value in data.items()}


def main() -> None:
    assets.configure_style()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    data = load_xcut_data()
    ymax = max(float(np.max(values)) for values in data.values())
    ymin = min(float(np.min(values)) for values in data.values())
    ypad = 0.04 * (ymax - ymin)

    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    styles = {
        "HDM": dict(color=assets.COLORS["HDM"], linewidth=3.0, alpha=0.82, zorder=2),
        "PROM-POD-AE": dict(
            color=assets.COLORS["PROM-POD-AE"], linewidth=2.8, alpha=0.82, zorder=3
        ),
        "POD-NN-ROM": dict(
            color=assets.COLORS["POD-NN-ROM"], linewidth=2.8, alpha=0.82, zorder=3
        ),
        "POD-DL-ROM": dict(
            color=assets.COLORS["POD-DL-ROM"], linewidth=2.8, alpha=0.82, zorder=3
        ),
    }

    for label in ("HDM", "PROM-POD-AE", "POD-NN-ROM", "POD-DL-ROM"):
        ax.plot(assets.X, data[label], label=label, **styles[label])

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u_x$")
    ax.set_title(
        rf"$u_x(x,y={assets.Y[assets.MID_Y]:.1f},t)$"
        "\n"
        rf"$\mu_1={POINT.mu1:.2f},\quad \mu_2={POINT.mu2:.3f},\quad t={TIME_INDEX * assets.DT:.2f}$",
        pad=12,
    )
    ax.grid(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=True,
        ncol=4,
        columnspacing=1.2,
        handlelength=2.8,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.24, top=0.84)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
