#!/usr/bin/env python3
"""Build the WCCM Burgers global-vs-local animation assets (N_c=3)."""

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


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
WORKBENCH = PAPER.parent
RESULTS = WORKBENCH / "Results"
OUTPUT = HERE / "outputs"
PREVIEWS = HERE / "previews"

DT = 0.05
NFRAMES = 501
FRAME_IDS = np.arange(0, NFRAMES, 20)
POINTS = ((4.56, 0.019), (4.75, 0.020), (5.19, 0.026))

COLORS = {
    "hdm": "#1a1a1a",
    "global": "#6a6a6a",
    "linear": "#2468b4",
    "quadratic": "#7c4fa3",
    "gpr": "#138a5b",
    "red": "#c62828",
}

# Values are the N_c=3 headline campaign reported in Results_Paper/main.tex.
METRICS = {
    "hprom": {
        "label": "Global HPROM",
        "short": "HPROM",
        "avg_error": 1.030,
        "max_error": 1.096,
        "speedup": 16.841,
        "nq": "96",
        "secondary": "--",
        "ne": "4,824",
    },
    "hqprom": {
        "label": "Global HQPROM",
        "short": "HQPROM",
        "avg_error": 0.871,
        "max_error": 0.933,
        "speedup": 14.084,
        "nq": "39",
        "secondary": r"quadratic: $39^2$",
        "ne": "3,505",
    },
    "hprom_gpr": {
        "label": "Global HPROM-GPR",
        "short": "HPROM-GPR",
        "avg_error": 0.691,
        "max_error": 0.845,
        "speedup": 33.698,
        "nq": "20",
        "secondary": r"$\bar n=131$",
        "ne": "1,801",
    },
    "local_hprom": {
        "label": r"Local HPROM ($N_c=3$)",
        "short": "Local HPROM",
        "avg_error": 0.827,
        "max_error": 0.854,
        "speedup": 33.168,
        "nq": "35--48",
        "secondary": "--",
        "ne": "3,649",
    },
    "local_hqprom": {
        "label": r"Local HQPROM ($N_c=3$)",
        "short": "Local HQPROM",
        "avg_error": 0.716,
        "max_error": 0.844,
        "speedup": 40.259,
        "nq": "11--15",
        "secondary": r"quadratic: $11^2$--$15^2$",
        "ne": "1,164",
    },
    "local_hprom_gpr": {
        "label": r"Local HPROM-GPR ($N_c=3$)",
        "short": "Local HPROM-GPR",
        "avg_error": 0.510,
        "max_error": 0.640,
        "speedup": 40.841,
        "nq": "10",
        "secondary": r"$\bar n=60$--$97$",
        "ne": "541",
    },
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def point_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.2f}_mu2_{mu2:.3f}"


def trajectory_path(key: str, mu1: float, mu2: float) -> Path:
    return RESULTS / f"{key}_snaps_{point_tag(mu1, mu2)}.npy"


def load_trajectory(key: str, mu1: float, mu2: float) -> np.ndarray:
    path = trajectory_path(key, mu1, mu2)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def grid_from_trajectory(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    nxy = trajectory.shape[0] // 2
    nx = int(round(np.sqrt(nxy)))
    if nx * nx != nxy:
        raise ValueError(f"Expected a square u_x grid, found {trajectory.shape}.")
    ny = nx
    x = (np.arange(nx) + 0.5) * (100.0 / nx)
    y = x.copy()
    idx_x = (ny // 2) * nx + np.arange(nx)
    idx_y = np.arange(ny) * nx + (nx // 2)
    return x, y, idx_x, nx, ny


def sampled_cuts(key: str, mu1: float, mu2: float) -> tuple[np.ndarray, np.ndarray]:
    trajectory = load_trajectory(key, mu1, mu2)
    _, _, idx_x, nx, ny = grid_from_trajectory(trajectory)
    idx_y = np.arange(ny) * nx + (nx // 2)
    return (
        np.asarray(trajectory[idx_x, :][:, FRAME_IDS]),
        np.asarray(trajectory[idx_y, :][:, FRAME_IDS]),
    )


def all_cuts(keys: tuple[str, ...]) -> dict[str, dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]]:
    return {
        key: {point: sampled_cuts(key, *point) for point in POINTS}
        for key in keys
    }


def save_preview(gif: Path, destination: Path, frame_index: int | None = None) -> None:
    from PIL import Image

    image = Image.open(gif)
    index = image.n_frames // 2 if frame_index is None else frame_index
    image.seek(index)
    image.convert("RGB").save(destination)


def parameter_domain_plot(output: Path) -> None:
    configure_style()
    train_mu1 = np.array([4.25, 4.875, 5.50])
    train_mu2 = np.array([0.015, 0.0225, 0.030])
    train = np.array([(a, b) for a in train_mu1 for b in train_mu2])

    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbf8")
    ax.scatter(train[:, 0], train[:, 1], s=210, c="black", marker="o", zorder=3, label=r"Baseline $3\times3$ grid")
    tests = np.asarray(POINTS)
    ax.scatter(
        tests[:, 0], tests[:, 1], s=300, c=COLORS["red"], marker="*",
        edgecolors="white", linewidths=0.9, zorder=5, label="Test points",
    )
    offsets = ((11, 7), (11, 7), (11, 7))
    for i, ((mu1, mu2), offset) in enumerate(zip(POINTS, offsets), start=1):
        ax.annotate(
            rf"$\mu^{{({i})}}$", (mu1, mu2), xytext=offset,
            textcoords="offset points", color="#9f2424", fontsize=16,
        )
    ax.set_title("Baseline training set in parameter space", fontsize=20, pad=10)
    ax.set_xlabel(r"$\mu_1$", fontsize=17)
    ax.set_ylabel(r"$\mu_2$", fontsize=17)
    ax.set_xlim(3.72, 6.03)
    ax.set_ylim(0.0088, 0.0372)
    ax.grid(True)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=True, fontsize=14,
    )
    fig.subplots_adjust(left=0.14, right=0.96, top=0.91, bottom=0.19)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


def hdm_cutplane_gif(output: Path) -> None:
    """Animate the current-study HDM surface and both centerline cuts."""
    configure_style()
    mu1, mu2 = POINTS[0]
    hdm = load_trajectory("hdm", mu1, mu2)
    x, y, _, nx, ny = grid_from_trajectory(hdm)
    xx, yy = np.meshgrid(x, y)
    mid_x, mid_y = nx // 2, ny // 2
    stride = 3
    norm = Normalize(vmin=0.0, vmax=6.6)

    fig = plt.figure(figsize=(8.4, 8.4))
    ax3 = fig.add_axes([0.04, 0.35, 0.92, 0.54], projection="3d")
    ax3.set(xlim=(0, 100), ylim=(0, 100), zlim=(0, 6.6), xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$u_x(x,y,t)$")
    ax3.view_init(elev=25, azim=-53)
    ax3.set_box_aspect((1.15, 1.0, 0.82), zoom=1.04)

    z_span = np.linspace(0.0, 6.6, 2)
    px, pz = np.meshgrid(x, z_span)
    ax3.plot_surface(px, np.full_like(px, y[mid_y]), pz, color=COLORS["red"], alpha=0.19, shade=False)
    py, pz = np.meshgrid(y, z_span)
    ax3.plot_surface(np.full_like(py, x[mid_x]), py, pz, color="#1565c0", alpha=0.19, shade=False)

    def field(frame: int) -> np.ndarray:
        return np.asarray(hdm[: nx * ny, frame]).reshape(ny, nx)

    first = field(int(FRAME_IDS[0]))
    surface = [ax3.plot_surface(xx[::stride, ::stride], yy[::stride, ::stride], first[::stride, ::stride], cmap="viridis", norm=norm, linewidth=0, antialiased=True, alpha=0.94)]
    x3, = ax3.plot(x, np.full(nx, y[mid_y]), first[mid_y, :], color=COLORS["red"], linewidth=3.0, label=rf"$x$-cut: $y={y[mid_y]:.1f}$")
    y3, = ax3.plot(np.full(ny, x[mid_x]), y, first[:, mid_x], color="#1565c0", linewidth=3.0, label=rf"$y$-cut: $x={x[mid_x]:.1f}$")
    ax3.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
    title = fig.suptitle("HDM centerline cut planes\n" + rf"$\mu^{{(1)}}=({mu1:.2f},{mu2:.3f}),\quad t=0.00$", y=0.98, fontsize=15)

    ax_x = fig.add_axes([0.085, 0.075, 0.39, 0.20])
    x2, = ax_x.plot(x, first[mid_y, :], color=COLORS["red"], linewidth=2.6)
    ax_x.set(xlim=(0, 100), ylim=(0, 6.6), xlabel=r"$x$", ylabel=r"$u_x$")
    ax_x.set_title(rf"Horizontal cut: $y={y[mid_y]:.1f}$", color="#8e1b1b")
    ax_x.grid(True)
    ax_y = fig.add_axes([0.56, 0.075, 0.39, 0.20])
    y2, = ax_y.plot(y, first[:, mid_x], color="#1565c0", linewidth=2.6)
    ax_y.set(xlim=(0, 100), ylim=(0, 6.6), xlabel=r"$y$", ylabel=r"$u_x$")
    ax_y.set_title(rf"Vertical cut: $x={x[mid_x]:.1f}$", color="#0d47a1")
    ax_y.grid(True)

    def update(frame_index: int):
        tidx = int(FRAME_IDS[frame_index])
        value = field(tidx)
        surface[0].remove()
        surface[0] = ax3.plot_surface(xx[::stride, ::stride], yy[::stride, ::stride], value[::stride, ::stride], cmap="viridis", norm=norm, linewidth=0, antialiased=True, alpha=0.94)
        x3.set_data_3d(x, np.full(nx, y[mid_y]), value[mid_y, :])
        y3.set_data_3d(np.full(ny, x[mid_x]), y, value[:, mid_x])
        x2.set_ydata(value[mid_y, :])
        y2.set_ydata(value[:, mid_x])
        title.set_text("HDM centerline cut planes\n" + rf"$\mu^{{(1)}}=({mu1:.2f},{mu2:.3f}),\quad t={tidx * DT:.2f}$")
        return surface[0], x3, y3, x2, y2, title

    animation.FuncAnimation(fig, update, frames=len(FRAME_IDS), interval=110, blit=False).save(output, writer=animation.PillowWriter(fps=9), dpi=105)
    plt.close(fig)


def global_local_gif_legacy(output: Path, global_key: str, local_key: str, local_color: str) -> None:
    configure_style()
    cuts = all_cuts(("hdm", global_key, local_key))
    first_hdm = load_trajectory("hdm", *POINTS[0])
    x, y, _, _, _ = grid_from_trajectory(first_hdm)
    global_metrics = METRICS[global_key]
    local_metrics = METRICS[local_key]

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 9.0), squeeze=False)
    artists = []
    for row, point in enumerate(POINTS):
        for col, grid in enumerate((x, y)):
            ax = axes[row, col]
            ax.set(xlim=(0, 100), ylim=(0, 6.6), ylabel=r"$u_x$")
            ax.set_xlabel(r"$x$" if col == 0 else r"$y$")
            ax.grid(True)
            cut_name = rf"$u_x(x,y={y[len(y)//2]:.1f},t)$" if col == 0 else rf"$u_x(x={x[len(x)//2]:.1f},y,t)$"
            ax.set_title(rf"$\mu^{{({row + 1})}}=({point[0]:.2f},{point[1]:.3f})$: {cut_name}")
            hdm_line, = ax.plot(grid, np.zeros_like(grid), color=COLORS["hdm"], linewidth=2.7, label="HDM")
            global_line, = ax.plot(grid, np.zeros_like(grid), color=COLORS["global"], linewidth=2.0, linestyle="--", label=global_metrics["short"])
            local_line, = ax.plot(grid, np.zeros_like(grid), color=local_color, linewidth=2.25, label=local_metrics["short"])
            artists.append((hdm_line, global_line, local_line, point, col))

    fig.legend(handles=[artists[0][0], artists[0][1], artists[0][2]], loc="upper center", bbox_to_anchor=(0.5, 0.950), ncol=3, frameon=True)
    fig.suptitle(
        f"{global_metrics['short']} versus {local_metrics['short']} ($N_c=3$)",
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    # Keep the two headline figures visible in every GIF frame.  These cards
    # deliberately use the same colours as the corresponding solution curves.
    global_card = fig.text(
        0.28,
        0.865,
        rf"$\mathbf{{GLOBAL}}$" + "\n" + rf"mean $\mathbb{{RE}}_2$: ${global_metrics['avg_error']:.3f}\%$  |  speedup: ${global_metrics['speedup']:.1f}\times$",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=COLORS["global"],
        bbox={"boxstyle": "round,pad=0.50", "facecolor": "#f0f0f0", "edgecolor": COLORS["global"], "linewidth": 1.4},
    )
    local_card = fig.text(
        0.72,
        0.865,
        rf"$\mathbf{{LOCAL}}$ ($N_c=3$)" + "\n" + rf"mean $\mathbb{{RE}}_2$: ${local_metrics['avg_error']:.3f}\%$  |  speedup: ${local_metrics['speedup']:.1f}\times$",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=local_color,
        bbox={"boxstyle": "round,pad=0.50", "facecolor": "#f7fbff", "edgecolor": local_color, "linewidth": 1.6},
    )
    improvement = fig.text(
        0.5,
        0.030,
        rf"local: ${(1.0 - local_metrics['avg_error'] / global_metrics['avg_error']) * 100.0:.0f}\%$ lower mean error  |  ${local_metrics['speedup'] / global_metrics['speedup']:.1f}\times$ faster",
        ha="center",
        fontsize=11.6,
        fontweight="bold",
        color=local_color,
    )
    time_stamp = fig.text(0.965, 0.030, r"$t=0.00$", ha="right", fontsize=11.2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.785, bottom=0.085, hspace=0.52, wspace=0.20)

    def update(frame_index: int):
        tidx = int(FRAME_IDS[frame_index])
        changed = []
        for hdm_line, global_line, local_line, point, col in artists:
            hdm_line.set_ydata(cuts["hdm"][point][col][:, frame_index])
            global_line.set_ydata(cuts[global_key][point][col][:, frame_index])
            local_line.set_ydata(cuts[local_key][point][col][:, frame_index])
            changed.extend((hdm_line, global_line, local_line))
        time_stamp.set_text(rf"$t={tidx * DT:.2f}$")
        return [*changed, global_card, local_card, improvement, time_stamp]

    animation.FuncAnimation(fig, update, frames=len(FRAME_IDS), interval=110, blit=False).save(output, writer=animation.PillowWriter(fps=9), dpi=105)
    plt.close(fig)


def global_local_gif(output: Path, global_key: str, local_key: str, local_color: str) -> None:
    """Animate one spatial cut with the global/local story kept separate."""
    configure_style()
    cuts = all_cuts(("hdm", global_key, local_key))
    first_hdm = load_trajectory("hdm", *POINTS[0])
    x, y, _, _, _ = grid_from_trajectory(first_hdm)
    global_metrics = METRICS[global_key]
    local_metrics = METRICS[local_key]
    improvement_error = (
        1.0 - local_metrics["avg_error"] / global_metrics["avg_error"]
    ) * 100.0
    improvement_speed = local_metrics["speedup"] / global_metrics["speedup"]

    fig = plt.figure(figsize=(11.4, 7.6))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(0.38, 1.0),
        left=0.020,
        right=0.990,
        top=0.790,
        bottom=0.090,
        hspace=0.55,
        wspace=0.13,
    )
    summary_axis = fig.add_subplot(grid[:, 0])
    cut_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]
    summary_axis.set_axis_off()

    artists = []
    for row, (axis, point) in enumerate(zip(cut_axes, POINTS)):
        axis.set(
            xlim=(0, 100),
            ylim=(0, 6.6),
            ylabel=r"$u_x$",
            xlabel=r"$x$",
        )
        axis.grid(True)
        axis.set_title(
            rf"$\mu^{{({row + 1})}}=({point[0]:.2f},{point[1]:.3f})$: "
            rf"$u_x(x,y={y[len(y)//2]:.1f},t)$"
        )
        hdm_line, = axis.plot(
            x, np.zeros_like(x), color=COLORS["hdm"], linewidth=2.7, label="HDM"
        )
        global_line, = axis.plot(
            x,
            np.zeros_like(x),
            color=COLORS["global"],
            linewidth=2.0,
            linestyle="--",
            label=global_metrics["short"],
        )
        local_line, = axis.plot(
            x,
            np.zeros_like(x),
            color=local_color,
            linewidth=2.25,
            label=local_metrics["short"],
        )
        artists.append((hdm_line, global_line, local_line, point))

    fig.suptitle(
        "{} versus {} ($N_c=3$)".format(
            global_metrics["short"], local_metrics["short"]
        ),
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        handles=[artists[0][0], artists[0][1], artists[0][2]],
        loc="upper center",
        bbox_to_anchor=(0.69, 0.947),
        ncol=3,
        frameon=True,
    )

    summary_axis.text(
        0.5,
        0.77,
        "GLOBAL\n{}\n\nmean error: {:.3f}%\nspeedup: {:.1f}x".format(
            global_metrics["short"],
            global_metrics["avg_error"],
            global_metrics["speedup"],
        ),
        ha="center",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=COLORS["global"],
        bbox={
            "boxstyle": "round,pad=0.60",
            "facecolor": "#f2f2f2",
            "edgecolor": COLORS["global"],
            "linewidth": 1.45,
        },
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.5,
        0.42,
        "LOCAL ($N_c=3$)\n{}\n\nmean error: {:.3f}%\nspeedup: {:.1f}x".format(
            local_metrics["short"],
            local_metrics["avg_error"],
            local_metrics["speedup"],
        ),
        ha="center",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=local_color,
        bbox={
            "boxstyle": "round,pad=0.60",
            "facecolor": "#f7fbff",
            "edgecolor": local_color,
            "linewidth": 1.65,
        },
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.5,
        0.14,
        "LOCAL ADVANTAGE\n{:.0f}% lower mean error\n{:.1f}x faster".format(
            improvement_error, improvement_speed
        ),
        ha="center",
        va="center",
        fontsize=11.3,
        fontweight="bold",
        color=local_color,
        transform=summary_axis.transAxes,
    )
    time_stamp = fig.text(0.967, 0.030, r"$t=0.00$", ha="right", fontsize=11.2)

    def update(frame_index: int):
        tidx = int(FRAME_IDS[frame_index])
        changed = []
        for hdm_line, global_line, local_line, point in artists:
            hdm_line.set_ydata(cuts["hdm"][point][0][:, frame_index])
            global_line.set_ydata(cuts[global_key][point][0][:, frame_index])
            local_line.set_ydata(cuts[local_key][point][0][:, frame_index])
            changed.extend((hdm_line, global_line, local_line))
        time_stamp.set_text(rf"$t={tidx * DT:.2f}$")
        return [*changed, time_stamp]

    animation.FuncAnimation(
        fig, update, frames=len(FRAME_IDS), interval=110, blit=False
    ).save(output, writer=animation.PillowWriter(fps=9), dpi=115)
    plt.close(fig)


def global_local_single_point_gif(
    output: Path,
    global_key: str,
    local_key: str,
    local_color: str,
    point_index: int,
) -> None:
    """Animate one selected parameter point for a global/local pair."""
    configure_style()
    point = POINTS[point_index]
    cuts = all_cuts(("hdm", global_key, local_key))
    first_hdm = load_trajectory("hdm", *point)
    x, y, _, _, _ = grid_from_trajectory(first_hdm)
    global_metrics = METRICS[global_key]
    local_metrics = METRICS[local_key]
    improvement_error = (
        1.0 - local_metrics["avg_error"] / global_metrics["avg_error"]
    ) * 100.0
    improvement_speed = local_metrics["speedup"] / global_metrics["speedup"]

    fig = plt.figure(figsize=(11.4, 5.8))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.38, 1.0),
        left=0.020,
        right=0.990,
        top=0.750,
        bottom=0.135,
        wspace=0.13,
    )
    summary_axis = fig.add_subplot(grid[0, 0])
    axis = fig.add_subplot(grid[0, 1])
    summary_axis.set_axis_off()

    axis.set(
        xlim=(0, 100),
        ylim=(0, 6.6),
        ylabel=r"$u_x$",
        xlabel=r"$x$",
    )
    axis.grid(True)
    axis.set_title(
        rf"$\mu^{{({point_index + 1})}}=({point[0]:.2f},{point[1]:.3f})$: "
        rf"$u_x(x,y={y[len(y)//2]:.1f},t)$",
        pad=10,
    )
    hdm_line, = axis.plot(
        x, np.zeros_like(x), color=COLORS["hdm"], linewidth=2.8, label="HDM"
    )
    global_line, = axis.plot(
        x,
        np.zeros_like(x),
        color=COLORS["global"],
        linewidth=2.15,
        linestyle="--",
        label=global_metrics["short"],
    )
    local_line, = axis.plot(
        x,
        np.zeros_like(x),
        color=local_color,
        linewidth=2.4,
        label=local_metrics["short"],
    )

    fig.suptitle(
        "{} versus {} ($N_c=3$)".format(
            global_metrics["short"], local_metrics["short"]
        ),
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        handles=[hdm_line, global_line, local_line],
        loc="upper center",
        bbox_to_anchor=(0.69, 0.915),
        ncol=3,
        frameon=True,
    )
    summary_axis.text(
        0.5,
        0.72,
        "GLOBAL\n{}\n\nmean error: {:.3f}%\nspeedup: {:.1f}x".format(
            global_metrics["short"],
            global_metrics["avg_error"],
            global_metrics["speedup"],
        ),
        ha="center",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=COLORS["global"],
        bbox={
            "boxstyle": "round,pad=0.60",
            "facecolor": "#f2f2f2",
            "edgecolor": COLORS["global"],
            "linewidth": 1.45,
        },
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.5,
        0.37,
        "LOCAL ($N_c=3$)\n{}\n\nmean error: {:.3f}%\nspeedup: {:.1f}x".format(
            local_metrics["short"],
            local_metrics["avg_error"],
            local_metrics["speedup"],
        ),
        ha="center",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=local_color,
        bbox={
            "boxstyle": "round,pad=0.60",
            "facecolor": "#f7fbff",
            "edgecolor": local_color,
            "linewidth": 1.65,
        },
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.5,
        0.075,
        "LOCAL ADVANTAGE\n{:.0f}% lower mean error | {:.1f}x faster".format(
            improvement_error, improvement_speed
        ),
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=local_color,
        transform=summary_axis.transAxes,
    )
    time_stamp = fig.text(0.967, 0.055, r"$t=0.00$", ha="right", fontsize=11.2)

    def update(frame_index: int):
        tidx = int(FRAME_IDS[frame_index])
        hdm_line.set_ydata(cuts["hdm"][point][0][:, frame_index])
        global_line.set_ydata(cuts[global_key][point][0][:, frame_index])
        local_line.set_ydata(cuts[local_key][point][0][:, frame_index])
        time_stamp.set_text(rf"$t={tidx * DT:.2f}$")
        return [hdm_line, global_line, local_line, time_stamp]

    animation.FuncAnimation(
        fig, update, frames=len(FRAME_IDS), interval=110, blit=False
    ).save(output, writer=animation.PillowWriter(fps=9), dpi=115)
    plt.close(fig)


def local_family_gif(output: Path) -> None:
    configure_style()
    point = POINTS[1]
    keys = ("local_hprom", "local_hqprom", "local_hprom_gpr")
    colors = (COLORS["linear"], COLORS["quadratic"], COLORS["gpr"])
    cuts = all_cuts(("hdm", *keys))
    hdm = load_trajectory("hdm", *point)
    x, y, _, _, _ = grid_from_trajectory(hdm)

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 9.0), squeeze=False)
    artists = []
    for row, (key, color) in enumerate(zip(keys, colors)):
        metric = METRICS[key]
        for col, grid in enumerate((x, y)):
            ax = axes[row, col]
            ax.set(xlim=(0, 100), ylim=(0, 6.6), ylabel=r"$u_x$")
            ax.set_xlabel(r"$x$" if col == 0 else r"$y$")
            ax.grid(True)
            if col == 0:
                ax.text(0.02, 0.93, rf"{metric['short']}: $\mathbb{{RE}}_2={metric['avg_error']:.3f}\%$, ${metric['speedup']:.1f}\times$", transform=ax.transAxes, va="top", fontsize=9.2)
            hdm_line, = ax.plot(grid, np.zeros_like(grid), color=COLORS["hdm"], linewidth=2.7, label="HDM")
            model_line, = ax.plot(grid, np.zeros_like(grid), color=color, linewidth=2.25, label=metric["short"])
            artists.append((hdm_line, model_line, key, col))
    axes[0, 0].set_title(rf"Horizontal cut: $u_x(x,y={y[len(y)//2]:.1f},t)$")
    axes[0, 1].set_title(rf"Vertical cut: $u_x(x={x[len(x)//2]:.1f},y,t)$")
    fig.legend(handles=[artists[0][0], artists[0][1], artists[2][1], artists[4][1]], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=True)
    fig.suptitle(rf"Local model family at $\mu^{{(2)}}=({point[0]:.2f},{point[1]:.3f})$ ($N_c=3$)", y=0.985, fontsize=16, fontweight="bold")
    footer = fig.text(0.5, 0.014, r"$t=0.00$", ha="center", fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.07, hspace=0.47, wspace=0.20)

    def update(frame_index: int):
        tidx = int(FRAME_IDS[frame_index])
        changed = []
        for hdm_line, model_line, key, col in artists:
            hdm_line.set_ydata(cuts["hdm"][point][col][:, frame_index])
            model_line.set_ydata(cuts[key][point][col][:, frame_index])
            changed.extend((hdm_line, model_line))
        footer.set_text(rf"$t={tidx * DT:.2f}$")
        return [*changed, footer]

    animation.FuncAnimation(fig, update, frames=len(FRAME_IDS), interval=110, blit=False).save(output, writer=animation.PillowWriter(fps=9), dpi=105)
    plt.close(fig)


def accuracy_cost_scatter_plot(output: Path) -> None:
    """Render the static slide plot; the configuration table stays in main.tex."""
    # This is deliberately local to this figure so that the GIF assets retain their
    # previous rendering configuration.  It also makes this plot directly usable
    # beside the LaTeX table in the WCCM slide deck.
    with plt.rc_context():
        configure_style()
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        fig, ax = plt.subplots(figsize=(8.6, 6.25))
        ax.set(
            xscale="log",
            xlim=(12, 72),
            ylim=(0.40, 1.14),
            xlabel=r"Speedup relative to the HDM",
            ylabel=r"Mean relative state error (\%)",
        )
        ax.grid(True, which="both")
        ax.set_axisbelow(True)

        entries = (
            ("Global HPROM", "hprom", COLORS["linear"], "o", (6, 10), "left"),
            ("Global HQPROM", "hqprom", COLORS["quadratic"], "o", (6, -17), "left"),
            ("Global HPROM--GPR", "hprom_gpr", COLORS["gpr"], "o", (-7, 8), "right"),
            (r"Local HPROM ($N_c=3$)", "local_hprom", COLORS["linear"], "D", (7, 10), "left"),
            (r"Local HQPROM ($N_c=3$)", "local_hqprom", COLORS["quadratic"], "D", (7, -17), "left"),
            (r"Local HPROM--GPR ($N_c=3$)", "local_hprom_gpr", COLORS["gpr"], "D", (-7, 8), "right"),
        )
        for label, key, color, marker, offset, horizontal_alignment in entries:
            metric = METRICS[key]
            ax.scatter(
                metric["speedup"],
                metric["avg_error"],
                marker=marker,
                s=82,
                color=color,
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
            )
            ax.annotate(
                label,
                (metric["speedup"], metric["avg_error"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9.3,
                color=color,
                ha=horizontal_alignment,
                va="center",
            )

        ax.plot([], [], marker="o", markersize=7, linestyle="None", color="#333333", label="Global model")
        ax.plot([], [], marker="D", markersize=7, linestyle="None", color="#333333", label=r"Local model ($N_c=3$)")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#aaaaaa")
        ax.set_title(r"Global--local accuracy--cost summary", fontsize=15, pad=12)
        fig.tight_layout()
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)


def accuracy_cost_summary_merged_plot(output: Path) -> None:
    """Build a slide-format, top--bottom accuracy/cost and table summary."""
    table_path = OUTPUT / "global_local_performance_table.png"
    if not table_path.exists():
        raise FileNotFoundError(f"Render the LaTeX table first: {table_path}")

    with plt.rc_context():
        configure_style()
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        fig = plt.figure(figsize=(16.0, 9.0))
        ax = fig.add_axes([0.09, 0.54, 0.82, 0.37])
        ax.set(
            xscale="log",
            xlim=(12, 72),
            ylim=(0.40, 1.14),
            xlabel=r"Speedup relative to the HDM",
            ylabel=r"Mean relative state error (\%)",
        )
        ax.grid(True, which="both")
        ax.set_axisbelow(True)
        entries = (
            ("Global HPROM", "hprom", COLORS["linear"], "o", (6, 10), "left"),
            ("Global HQPROM", "hqprom", COLORS["quadratic"], "o", (6, -17), "left"),
            ("Global HPROM--GPR", "hprom_gpr", COLORS["gpr"], "o", (-7, 8), "right"),
            (r"Local HPROM ($N_c=3$)", "local_hprom", COLORS["linear"], "D", (7, 10), "left"),
            (r"Local HQPROM ($N_c=3$)", "local_hqprom", COLORS["quadratic"], "D", (7, -17), "left"),
            (r"Local HPROM--GPR ($N_c=3$)", "local_hprom_gpr", COLORS["gpr"], "D", (-7, 8), "right"),
        )
        for label, key, color, marker, offset, horizontal_alignment in entries:
            metric = METRICS[key]
            ax.scatter(
                metric["speedup"],
                metric["avg_error"],
                marker=marker,
                s=92,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                label,
                (metric["speedup"], metric["avg_error"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=10.5,
                color=color,
                ha=horizontal_alignment,
                va="center",
            )
        ax.plot([], [], marker="o", markersize=7, linestyle="None", color="#333333", label="Global model")
        ax.plot([], [], marker="D", markersize=7, linestyle="None", color="#333333", label=r"Local model ($N_c=3$)")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#aaaaaa")
        ax.set_title(r"Global--local accuracy--cost summary", fontsize=17, pad=10)

        # The axes ratio is chosen to preserve the table PNG's native aspect
        # ratio, so the LaTeX table remains undistorted and spans the same width
        # as the plot above.
        table_image = plt.imread(table_path)
        table_width = 0.82
        table_height = table_width * (table_image.shape[0] / table_image.shape[1]) * (16.0 / 9.0)
        table_ax = fig.add_axes([0.09, 0.065, table_width, table_height])
        table_ax.imshow(table_image, aspect="auto")
        table_ax.axis("off")
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)


def mesh_montage(output: Path, files: tuple[str, str, str], labels: tuple[str, str, str], title: str) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    for ax, filename, label in zip(axes, files, labels):
        image = plt.imread(PAPER / filename)
        ax.imshow(image)
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.03, wspace=0.03)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    parameter_domain_plot(OUTPUT / "parameter_domain_test_points.png")
    hdm_cutplane_gif(OUTPUT / "hdm_centerline_cuts_mu1.gif")
    global_local_gif(OUTPUT / "global_vs_local_hprom.gif", "hprom", "local_hprom", COLORS["linear"])
    global_local_gif(OUTPUT / "global_vs_local_hqprom.gif", "hqprom", "local_hqprom", COLORS["quadratic"])
    global_local_gif(OUTPUT / "global_vs_local_hprom_gpr.gif", "hprom_gpr", "local_hprom_gpr", COLORS["gpr"])
    local_family_gif(OUTPUT / "local_hprom_hqprom_gpr_mu2.gif")
    accuracy_cost_scatter_plot(OUTPUT / "global_local_accuracy_cost_summary.png")
    accuracy_cost_summary_merged_plot(OUTPUT / "global_local_accuracy_cost_summary_merged.png")
    mesh_montage(
        OUTPUT / "appendix_global_ecsw_meshes.png",
        ("hprom_reduced_mesh.png", "hqprom_reduced_mesh.png", "hprom_gpr_reduced_mesh.png"),
        ("HPROM ($N_e=4{,}824$)", "HQPROM ($N_e=3{,}505$)", "HPROM-GPR ($N_e=1{,}801$)"),
        "Global ECSW reduced meshes",
    )
    mesh_montage(
        OUTPUT / "appendix_local_ecsw_meshes.png",
        ("local_hprom_reduced_mesh.png", "local_hqprom_reduced_mesh.png", "local_hprom_gpr_reduced_mesh.png"),
        ("Local HPROM ($N_e=3{,}649$)", "Local HQPROM ($N_e=1{,}164$)", "Local HPROM-GPR ($N_e=901$)"),
        r"Local ECSW reduced meshes ($N_c=3$)",
    )
    for gif in OUTPUT.glob("*.gif"):
        save_preview(gif, PREVIEWS / f"{gif.stem}_mid.png")
        print(f"saved {gif}")


if __name__ == "__main__":
    main()
