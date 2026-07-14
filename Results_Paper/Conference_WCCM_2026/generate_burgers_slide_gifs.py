#!/usr/bin/env python3
"""Render the animated centerline-cut visual for the 2D Burgers slides."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize


ANIMATIONS = Path(
    "/home/kratos/burgers2d-rom-workbench/Project_YvonMaday/Results_Paper/animations"
)
sys.path.insert(0, str(ANIMATIONS))
import generate_burgers_presentation_assets as assets

OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def main(output: Path) -> None:
    assets.configure_style()
    point = assets.POINTS[1]
    frame_ids = np.arange(0, assets.NT, 10)
    snaps = assets.load_npy(assets.hdm_path(point))
    stride = 3
    norm = Normalize(vmin=0.0, vmax=5.6)
    xx, yy = np.meshgrid(assets.X, assets.Y)
    xx_coarse = xx[::stride, ::stride]
    yy_coarse = yy[::stride, ::stride]

    fig = plt.figure(figsize=(8.4, 8.4))
    ax3 = fig.add_axes([0.04, 0.35, 0.92, 0.54], projection="3d")
    ax3.set(
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        zlim=(0.0, 5.6),
        xlabel=r"$x$",
        ylabel=r"$y$",
        zlabel=r"$u_x(x,y,t)$",
    )
    ax3.view_init(elev=25, azim=-53)
    ax3.set_box_aspect((1.15, 1.0, 0.82), zoom=1.04)

    z_span = np.linspace(0.0, 5.6, 2)
    plane_x, plane_z = np.meshgrid(assets.X, z_span)
    ax3.plot_surface(
        plane_x,
        np.full_like(plane_x, assets.Y[assets.MID_Y]),
        plane_z,
        color="#c62828",
        alpha=0.19,
        shade=False,
    )
    plane_y, plane_z = np.meshgrid(assets.Y, z_span)
    ax3.plot_surface(
        np.full_like(plane_y, assets.X[assets.MID_X]),
        plane_y,
        plane_z,
        color="#1565c0",
        alpha=0.19,
        shade=False,
    )

    field0 = np.asarray(snaps[: assets.NXY, frame_ids[0]]).reshape(assets.NY, assets.NX)
    surface = [
        ax3.plot_surface(
            xx_coarse,
            yy_coarse,
            field0[::stride, ::stride],
            cmap="viridis",
            norm=norm,
            linewidth=0,
            antialiased=True,
            alpha=0.94,
        )
    ]
    xcut3d, = ax3.plot(
        assets.X,
        np.full(assets.NX, assets.Y[assets.MID_Y]),
        field0[assets.MID_Y, :],
        color="#c62828",
        linewidth=3.2,
        label=rf"$x$-cut: $u_x(x,y={assets.Y[assets.MID_Y]:.1f})$",
    )
    ycut3d, = ax3.plot(
        np.full(assets.NY, assets.X[assets.MID_X]),
        assets.Y,
        field0[:, assets.MID_X],
        color="#1565c0",
        linewidth=3.2,
        label=rf"$y$-cut: $u_x(x={assets.X[assets.MID_X]:.1f},y)$",
    )
    ax3.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
    title = fig.suptitle(
        rf"\textbf{{Centerline cut planes}}"
        "\n"
        rf"$\mu_1={point.mu1:.2f},\quad \mu_2={point.mu2:.3f},\quad t=0.00$",
        x=0.5,
        y=0.98,
        fontsize=15,
    )

    ax_x = fig.add_axes([0.085, 0.075, 0.39, 0.20])
    xcut2d, = ax_x.plot(assets.X, field0[assets.MID_Y, :], color="#c62828", linewidth=2.8)
    ax_x.set(xlim=(0.0, 100.0), ylim=(0.0, 5.6), xlabel=r"$x$", ylabel=r"$u_x$")
    ax_x.set_title(
        rf"\textbf{{Horizontal cut: }}$y={assets.Y[assets.MID_Y]:.1f}$",
        color="#8e1b1b",
    )
    ax_x.grid(True)

    ax_y = fig.add_axes([0.56, 0.075, 0.39, 0.20])
    ycut2d, = ax_y.plot(assets.Y, field0[:, assets.MID_X], color="#1565c0", linewidth=2.8)
    ax_y.set(xlim=(0.0, 100.0), ylim=(0.0, 5.6), xlabel=r"$y$", ylabel=r"$u_x$")
    ax_y.set_title(
        rf"\textbf{{Vertical cut: }}$x={assets.X[assets.MID_X]:.1f}$",
        color="#0d47a1",
    )
    ax_y.grid(True)

    def update(frame_index: int):
        tidx = int(frame_ids[frame_index])
        field = np.asarray(snaps[: assets.NXY, tidx]).reshape(assets.NY, assets.NX)
        surface[0].remove()
        surface[0] = ax3.plot_surface(
            xx_coarse,
            yy_coarse,
            field[::stride, ::stride],
            cmap="viridis",
            norm=norm,
            linewidth=0,
            antialiased=True,
            alpha=0.94,
        )
        xcut3d.set_data_3d(
            assets.X,
            np.full(assets.NX, assets.Y[assets.MID_Y]),
            field[assets.MID_Y, :],
        )
        ycut3d.set_data_3d(
            np.full(assets.NY, assets.X[assets.MID_X]),
            assets.Y,
            field[:, assets.MID_X],
        )
        xcut2d.set_ydata(field[assets.MID_Y, :])
        ycut2d.set_ydata(field[:, assets.MID_X])
        title.set_text(
            rf"\textbf{{Centerline cut planes}}"
            "\n"
            rf"$\mu_1={point.mu1:.2f},\quad \mu_2={point.mu2:.3f},\quad t={tidx * assets.DT:.2f}$"
        )
        return surface[0], xcut3d, ycut3d, xcut2d, ycut2d, title

    movie = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=90, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=10), dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    output_directory = Path(sys.argv[1]) if len(sys.argv) == 2 else OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    assets.hdm_3d_animation(output_directory / "burgers_hdm_3d.gif", assets.POINTS[1])
    main(output_directory / "burgers_cutplane_explanation.gif")
