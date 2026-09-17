#!/usr/bin/env python3
"""Generate the representative 2D Burgers HDM figure used in the paper."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


PAPER = Path(__file__).resolve().parent
REPOSITORY = PAPER.parents[1]
SNAPSHOTS = (
    REPOSITORY
    / "Project_YvonMaday"
    / "Results"
    / "param_snaps"
    / "mu1_4.56+mu2_0.019.npy"
)
OUTPUT = PAPER / "Figures" / "euclidean_hprom"

NX = NY = 250
NXY = NX * NY
DT = 0.05
TIME_INDEX = 150
SURFACE_STRIDE = 3
Z_LIMITS = (0.0, 5.0)
RED = "#D62728"
BLUE = "#1F77B4"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
        }
    )


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    snapshots = np.load(SNAPSHOTS, mmap_mode="r", allow_pickle=False)
    if snapshots.shape != (2 * NXY, 501):
        raise ValueError(f"Unexpected HDM trajectory shape: {snapshots.shape}")
    field = np.asarray(snapshots[:NXY, TIME_INDEX]).reshape(NY, NX)

    x = (np.arange(NX) + 0.5) * (100.0 / NX)
    y = (np.arange(NY) + 0.5) * (100.0 / NY)
    xx, yy = np.meshgrid(x, y)
    mid_x, mid_y = NX // 2, NY // 2

    figure = plt.figure(figsize=(7.25, 3.65))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.58, 1.0),
        left=0.015,
        right=0.985,
        bottom=0.14,
        top=0.94,
        wspace=0.22,
        hspace=0.58,
    )
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_x = figure.add_subplot(grid[0, 1])
    axis_y = figure.add_subplot(grid[1, 1], sharey=axis_x)

    norm = Normalize(*Z_LIMITS)
    axis_3d.plot_surface(
        xx[::SURFACE_STRIDE, ::SURFACE_STRIDE],
        yy[::SURFACE_STRIDE, ::SURFACE_STRIDE],
        field[::SURFACE_STRIDE, ::SURFACE_STRIDE],
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )

    z_span = np.asarray(Z_LIMITS)
    plane_x, plane_z = np.meshgrid(x, z_span)
    axis_3d.plot_surface(
        plane_x,
        np.full_like(plane_x, y[mid_y]),
        plane_z,
        color=RED,
        alpha=0.14,
        shade=False,
        rasterized=True,
    )
    plane_y, plane_z = np.meshgrid(y, z_span)
    axis_3d.plot_surface(
        np.full_like(plane_y, x[mid_x]),
        plane_y,
        plane_z,
        color=BLUE,
        alpha=0.14,
        shade=False,
        rasterized=True,
    )
    axis_3d.plot(
        x,
        np.full(NX, y[mid_y]),
        field[mid_y, :],
        color=RED,
        linewidth=1.8,
    )
    axis_3d.plot(
        np.full(NY, x[mid_x]),
        y,
        field[:, mid_x],
        color=BLUE,
        linewidth=1.8,
    )
    axis_3d.set(
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        zlim=Z_LIMITS,
        xlabel=r"$x$",
        ylabel=r"$y$",
        zlabel=r"$u_x(x,y,t)$",
        title=r"(a) HDM field $u_x(x,y,t)$",
    )
    axis_3d.view_init(elev=24, azim=-55)
    axis_3d.set_box_aspect((1.15, 1.0, 0.76), zoom=1.02)
    axis_3d.tick_params(pad=0.5)
    for pane in (axis_3d.xaxis.pane, axis_3d.yaxis.pane, axis_3d.zaxis.pane):
        pane.set_alpha(0.04)

    axis_x.plot(x, field[mid_y, :], color=RED, linewidth=1.7)
    axis_x.set_title(r"(b) Horizontal centerline: $y=50.2$")
    axis_x.set(
        xlim=(0.0, 100.0),
        ylim=Z_LIMITS,
        xlabel=r"$x$",
        ylabel=r"$u_x$",
    )

    axis_y.plot(y, field[:, mid_x], color=BLUE, linewidth=1.7)
    axis_y.set_title(r"(c) Vertical centerline: $x=50.2$")
    axis_y.set(xlim=(0.0, 100.0), ylim=Z_LIMITS, xlabel=r"$y$", ylabel=r"$u_x$")

    for axis in (axis_x, axis_y):
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.set_yticks((0, 1, 2, 3, 4, 5))
        axis.grid(True)

    pdf = OUTPUT / "burgers_problem_3d.pdf"
    png = OUTPUT / "burgers_problem_3d.png"
    figure.savefig(pdf, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
