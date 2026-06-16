#!/usr/bin/env python3
"""Generate presentation-oriented Burgers figures and animations.

The assets are deliberately organized around the conference narrative:

1. introduce the HDM solution,
2. explain the centerline cuts used in later comparisons,
3. reveal the baseline and enriched parameter sets,
4. compare model families without putting every method on every frame.

All solution arrays are memory mapped because each trajectory is roughly
500 MB. Only the two centerline cuts needed by the presentation are read.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
PROJECT = PAPER.parent
OUTPUT = HERE / "outputs"

MAIN = PAPER / "mlspg_hprom_main"
ENRICHED = PAPER / "mlspg_hprom_enrichment"
METRIC = PAPER / "MetricStudy" / "lspg_sensitive" / "Stage1"
ENRICHED_DATASET = (
    ENRICHED / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs20"
)

NX = 250
NY = 250
NXY = NX * NY
NT = 501
DT = 0.05
NTOT = 151

X = np.linspace(0.2, 99.8, NX)
Y = np.linspace(0.2, 99.8, NY)
MID_X = NX // 2
MID_Y = NY // 2
IDX_X_CUT = MID_Y * NX + np.arange(NX)
IDX_Y_CUT = np.arange(NY) * NX + MID_X

DOMAIN = (4.25, 5.50, 0.015, 0.030)
EVALUATION_POINTS = [
    ("v", 4.875, 0.0225, "verification"),
    ("1", 4.560, 0.0190, "off-grid"),
    ("2", 5.190, 0.0260, "off-grid"),
]

COLORS = {
    "HDM": "#111111",
    "Linear HPROM": "#777777",
    "PROM-ANN Case 1": "#2676c8",
    "PROM-ANN Case 2": "#00a6b2",
    "PROM-ANN Case 3": "#228b45",
    "PROM-POD-AE": "#7b4ab5",
    "POD-NN-ROM": "#e67e22",
    "POD-DL-ROM": "#d94f8a",
}


@dataclass(frozen=True)
class Point:
    tag: str
    mu1: float
    mu2: float
    kind: str

    @property
    def path_tag(self) -> str:
        return f"mu1_{self.mu1:.3f}_mu2_{self.mu2:.4f}"

    @property
    def title(self) -> str:
        return (
            rf"$\mu^{{({self.tag})}}=({self.mu1:.3f},{self.mu2:.4f})$"
            f"  [{self.kind}]"
        )


POINTS = [Point(*values) for values in EVALUATION_POINTS]


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    color: str
    family_path: str | None = None
    file_prefix: str | None = None
    n_primary: int | None = None
    data_driven: bool = False
    pod_ae: bool = False
    pod_dl: bool = False


MODELS = {
    "case1": Model(
        "case1",
        "PROM-ANN Case 1",
        COLORS["PROM-ANN Case 1"],
        family_path="Case1_Best",
        file_prefix="case1_hprom_ann",
        n_primary=10,
    ),
    "case2": Model(
        "case2",
        "PROM-ANN Case 2",
        COLORS["PROM-ANN Case 2"],
        family_path="Case2_Best/np10",
        file_prefix="case2_hprom_ann",
        n_primary=10,
    ),
    "case3": Model(
        "case3",
        "PROM-ANN Case 3",
        COLORS["PROM-ANN Case 3"],
        family_path="Case3_Best",
        file_prefix="case3_hprom_ann",
        n_primary=10,
    ),
    "podae": Model(
        "podae",
        "PROM-POD-AE",
        COLORS["PROM-POD-AE"],
        pod_ae=True,
    ),
    "podnn": Model(
        "podnn",
        "POD-NN-ROM",
        COLORS["POD-NN-ROM"],
        data_driven=True,
    ),
    "poddl": Model(
        "poddl",
        "POD-DL-ROM",
        COLORS["POD-DL-ROM"],
        pod_dl=True,
    ),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
        }
    )


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def hdm_path(point: Point) -> Path:
    names = [
        f"mu1_{point.mu1:g}+mu2_{point.mu2:g}.npy",
        f"mu1_{point.mu1:.3f}+mu2_{point.mu2:.4f}.npy",
    ]
    for root in (PROJECT / "Results" / "param_snaps", PROJECT / "250x250" / "param_snaps"):
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
        matches = sorted(root.glob(f"mu1_{point.mu1:g}+mu2_*.npy"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"HDM snapshots not found for {point}")


def linear_q_path(point: Point) -> Path:
    return (
        MAIN
        / "Runs"
        / "Linear"
        / f"linear_hprom_{point.path_tag}_ntot151"
        / "qN.npy"
    )


def model_snaps_path(root: Path, model: Model, point: Point) -> Path:
    if model.data_driven:
        return (
            root
            / "Runs"
            / "DataDriven_Best"
            / f"rom_data_driven_{point.path_tag}_ntot151"
            / "rom_snaps.npy"
        )
    if model.pod_dl:
        return (
            root
            / "Runs"
            / "PODDL_Best"
            / f"pod_dl_data_driven_{point.path_tag}_ntot151_nz10"
            / "rom_snaps.npy"
        )
    if model.pod_ae:
        return (
            root
            / "Runs"
            / "ECSW1pct"
            / "PODAE_Best"
            / f"podae_hprom_{point.path_tag}_ntot151_nz10_snaps.npy"
        )
    if model.family_path is None or model.file_prefix is None or model.n_primary is None:
        raise ValueError(model)
    return (
        root
        / "Runs"
        / "ECSW1pct"
        / model.family_path
        / (
            f"{model.file_prefix}_{point.path_tag}_n{model.n_primary}"
            "_ntot151_snaps.npy"
        )
    )


def baseline_points() -> np.ndarray:
    path = ENRICHED_DATASET / "baseline_mu.npy"
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if values.shape != (9, 2):
        raise ValueError(f"Unexpected baseline parameter shape: {values.shape}")
    return values


def lhs_points() -> np.ndarray:
    path = ENRICHED_DATASET / "lhs_mu.npy"
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if values.shape != (20, 2):
        raise ValueError(f"Unexpected LHS parameter shape: {values.shape}")
    return values


def draw_domain(ax: plt.Axes) -> None:
    mu1_min, mu1_max, mu2_min, mu2_max = DOMAIN
    ax.set_xlim(mu1_min - 0.05, mu1_max + 0.05)
    ax.set_ylim(mu2_min - 0.0007, mu2_max + 0.0007)
    ax.set_xlabel(r"$\mu_1$  (left boundary value)")
    ax.set_ylabel(r"$\mu_2$  (source growth rate)")
    ax.set_facecolor("#fbfbf8")
    ax.grid(True)


def parameter_figure(stage: str, output: Path) -> None:
    base = baseline_points()
    lhs = lhs_points()

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    draw_domain(ax)
    ax.scatter(
        base[:, 0],
        base[:, 1],
        s=105,
        marker="o",
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.6,
        label="9 HDM training parameters",
        zorder=4,
    )

    if stage in {"evaluation", "enriched"}:
        for point in POINTS:
            color = "#b71c1c" if point.kind == "verification" else "#d35400"
            marker = "*" if point.kind == "verification" else "X"
            label = None
            if point.tag == "v":
                label = "verification point"
            elif point.tag == "1":
                label = "2 off-grid test points"
            ax.scatter(
                point.mu1,
                point.mu2,
                s=215 if marker == "*" else 135,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=label,
                zorder=6,
            )
            offsets = {"v": (10, -24), "1": (10, 10), "2": (10, 10)}
            ax.annotate(
                rf"$\mu^{{({point.tag})}}$",
                (point.mu1, point.mu2),
                xytext=offsets[point.tag],
                textcoords="offset points",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    if stage == "enriched":
        ax.scatter(
            lhs[:, 0],
            lhs[:, 1],
            s=72,
            marker="D",
            facecolor="#3f8fc5",
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            label="20 LHS linear-HPROM trajectories",
            zorder=3,
        )

    if stage == "training":
        title = "Baseline HDM training set"
        subtitle = r"$3\times3$ parameter grid $\times$ 501 states = 4509 HDM snapshots"
    elif stage == "evaluation":
        title = "Fixed training and evaluation protocol"
        subtitle = "The verification point is the center training parameter; tests are off-grid."
    elif stage == "enriched":
        title = "Coefficient-data enrichment without additional HDM solves"
        subtitle = r"$9$ HDM trajectories $+$ $20$ linear-HPROM trajectories $=29\times501$ samples"
    else:
        raise ValueError(stage)

    ax.set_title(title, fontweight="bold", pad=25)
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#333333",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3 if stage == "enriched" else 2,
        frameon=True,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.23)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def parameter_sequence_gif(paths: list[Path], output: Path) -> None:
    frames: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
        frames.extend([image.copy()] * 4)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        disposal=2,
    )


def create_hdm_3d_scene(
    field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    norm: Normalize,
    point: Point,
    time_index: int,
) -> tuple[plt.Figure, plt.Axes, list, plt.Text]:
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_axes([0.035, 0.045, 0.93, 0.81], projection="3d")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_zlim(0.0, 5.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$u_x(x,y,t)$")
    ax.view_init(elev=25, azim=-53)
    ax.set_box_aspect((1.15, 1.0, 0.82), zoom=1.04)

    surface = [
        ax.plot_surface(
            xx,
            yy,
            field,
            cmap="viridis",
            norm=norm,
            linewidth=0,
            antialiased=True,
            rcount=yy.shape[0],
            ccount=xx.shape[1],
        )
    ]
    title = fig.suptitle(
        rf"\textbf{{HDM trajectory}}"
        "\n"
        rf"$\mu_1={point.mu1:.2f},\quad "
        rf"\mu_2={point.mu2:.3f},\quad t={time_index * DT:.2f}$",
        x=0.5,
        y=0.975,
        fontsize=15,
    )
    return fig, ax, surface, title


def hdm_3d_preview(output: Path, point: Point, time_index: int) -> None:
    snaps = load_npy(hdm_path(point))
    stride = 3
    xx, yy = np.meshgrid(X[::stride], Y[::stride])
    norm = Normalize(vmin=0.0, vmax=5.6)
    field = np.asarray(snaps[:NXY, time_index]).reshape(NY, NX)[::stride, ::stride]
    fig, _, _, _ = create_hdm_3d_scene(field, xx, yy, norm, point, time_index)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def hdm_3d_animation(output: Path, point: Point) -> None:
    snaps = load_npy(hdm_path(point))
    frame_ids = np.arange(0, NT, 10)
    stride = 3
    xx, yy = np.meshgrid(X[::stride], Y[::stride])
    norm = Normalize(vmin=0.0, vmax=5.6)
    field0 = np.asarray(snaps[:NXY, frame_ids[0]]).reshape(NY, NX)[::stride, ::stride]
    fig, ax, surface, title = create_hdm_3d_scene(
        field0,
        xx,
        yy,
        norm,
        point,
        int(frame_ids[0]),
    )

    def update(frame_index: int):
        tidx = int(frame_ids[frame_index])
        field = np.asarray(snaps[:NXY, tidx]).reshape(NY, NX)[::stride, ::stride]
        surface[0].remove()
        surface[0] = ax.plot_surface(
            xx,
            yy,
            field,
            cmap="viridis",
            norm=norm,
            linewidth=0,
            antialiased=True,
            rcount=yy.shape[0],
            ccount=xx.shape[1],
        )
        title.set_text(
            rf"\textbf{{HDM trajectory}}"
            "\n"
            rf"$\mu_1={point.mu1:.2f},\quad "
            rf"\mu_2={point.mu2:.3f},\quad t={tidx * DT:.2f}$"
        )
        return surface[0], title

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=90,
        blit=False,
    )
    movie.save(output, writer=animation.PillowWriter(fps=10), dpi=120)
    plt.close(fig)


def cutplane_explanation(output: Path, point: Point, time_index: int = 250) -> None:
    snaps = load_npy(hdm_path(point))
    field = np.asarray(snaps[:NXY, time_index]).reshape(NY, NX)
    xx, yy = np.meshgrid(X, Y)
    stride = 3

    fig = plt.figure(figsize=(8.4, 8.4))
    ax3 = fig.add_axes([0.04, 0.35, 0.92, 0.54], projection="3d")
    ax3.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        field[::stride, ::stride],
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )

    zmin = 0.0
    zmax = 5.6
    z_span = np.linspace(zmin, zmax, 2)

    plane_x, plane_z = np.meshgrid(X, z_span)
    plane_y = np.full_like(plane_x, Y[MID_Y])
    ax3.plot_surface(plane_x, plane_y, plane_z, color="#c62828", alpha=0.19, shade=False)

    plane_y2, plane_z2 = np.meshgrid(Y, z_span)
    plane_x2 = np.full_like(plane_y2, X[MID_X])
    ax3.plot_surface(plane_x2, plane_y2, plane_z2, color="#1565c0", alpha=0.19, shade=False)

    ax3.plot(
        X,
        np.full(NX, Y[MID_Y]),
        field[MID_Y, :],
        color="#c62828",
        linewidth=3.2,
        label=rf"$x$-cut: $u_x(x,y={Y[MID_Y]:.1f})$",
    )
    ax3.plot(
        np.full(NY, X[MID_X]),
        Y,
        field[:, MID_X],
        color="#1565c0",
        linewidth=3.2,
        label=rf"$y$-cut: $u_x(x={X[MID_X]:.1f},y)$",
    )
    ax3.set_xlim(0.0, 100.0)
    ax3.set_ylim(0.0, 100.0)
    ax3.set_zlim(zmin, zmax)
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$y$")
    ax3.set_zlabel(r"$u_x(x,y,t)$")
    ax3.view_init(elev=25, azim=-53)
    ax3.set_box_aspect((1.15, 1.0, 0.82), zoom=1.04)

    fig.suptitle(
        rf"\textbf{{Centerline cut planes}}"
        "\n"
        rf"$\mu_1={point.mu1:.2f},\quad "
        rf"\mu_2={point.mu2:.3f},\quad t={time_index * DT:.2f}$",
        x=0.5,
        y=0.98,
        fontsize=15,
    )
    ax3.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
    )

    ax_x = fig.add_axes([0.085, 0.075, 0.39, 0.20])
    ax_x.plot(X, field[MID_Y, :], color="#c62828", linewidth=2.8)
    ax_x.set_xlabel(r"$x$")
    ax_x.set_ylabel(r"$u_x$")
    ax_x.set_title(
        rf"\textbf{{Horizontal cut: }}$y={Y[MID_Y]:.1f}$",
        color="#8e1b1b",
    )
    ax_x.set_xlim(0.0, 100.0)
    ax_x.set_ylim(zmin, zmax)
    ax_x.grid(True)

    ax_y = fig.add_axes([0.56, 0.075, 0.39, 0.20])
    ax_y.plot(Y, field[:, MID_X], color="#1565c0", linewidth=2.8)
    ax_y.set_xlabel(r"$y$")
    ax_y.set_ylabel(r"$u_x$")
    ax_y.set_title(
        rf"\textbf{{Vertical cut: }}$x={X[MID_X]:.1f}$",
        color="#0d47a1",
    )
    ax_y.set_xlim(0.0, 100.0)
    ax_y.set_ylim(zmin, zmax)
    ax_y.grid(True)

    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def load_linear_cut_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    basis = load_npy(METRIC / "basis.npy")
    u_ref = np.asarray(load_npy(METRIC / "u_ref.npy"), dtype=float).reshape(-1)
    vx = np.asarray(basis[IDX_X_CUT, :NTOT], dtype=float)
    vy = np.asarray(basis[IDX_Y_CUT, :NTOT], dtype=float)
    ux_ref = u_ref[IDX_X_CUT]
    uy_ref = u_ref[IDX_Y_CUT]

    lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for point in POINTS:
        q = np.asarray(load_npy(linear_q_path(point)), dtype=float)
        lines[point.tag] = (
            ux_ref[:, None] + vx @ q,
            uy_ref[:, None] + vy @ q,
        )
    return lines


def snapshot_cut_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    snaps = load_npy(path)
    return snaps[IDX_X_CUT, :], snaps[IDX_Y_CUT, :]


def comparison_axes(
    rows: int,
    title: str,
    points: list[Point],
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(rows, 2, figsize=(12.8, 3.15 * rows), squeeze=False)
    for row, point in enumerate(points):
        for col, (grid, cut) in enumerate(((X, "x"), (Y, "y"))):
            ax = axes[row, col]
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, 5.6)
            ax.set_xlabel(r"$x$" if cut == "x" else r"$y$")
            ax.set_ylabel(r"$u_x$")
            ax.grid(True)
            cut_text = (
                rf"$u_x(x,y={Y[MID_Y]:.1f})$"
                if cut == "x"
                else rf"$u_x(x={X[MID_X]:.1f},y)$"
            )
            ax.set_title(f"{point.title}: {cut_text}")
    fig.suptitle(title, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.10, hspace=0.46, wspace=0.22)
    return fig, axes


def get_legend_handles(legend: plt.Legend) -> list[Line2D]:
    """Return legend handles across supported Matplotlib versions."""
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = legend.legendHandles
    return handles


def linear_vs_hdm_gif(output: Path) -> None:
    linear = load_linear_cut_data()
    hdm = {point.tag: snapshot_cut_data(hdm_path(point)) for point in POINTS}
    frame_ids = np.arange(0, NT, 10)

    fig, axes = comparison_axes(
        len(POINTS),
        "HDM and linear HPROM",
        POINTS,
    )
    artists: list[tuple[Line2D, Line2D]] = []
    for row, point in enumerate(POINTS):
        row_artists = []
        for col, grid in enumerate((X, Y)):
            h_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                color=COLORS["HDM"],
                linewidth=3.0,
                label="HDM",
            )
            l_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                color="#d62728",
                linewidth=2.3,
                linestyle="--",
                label="Linear HPROM",
            )
            row_artists.append((h_line, l_line))
        artists.extend(row_artists)
    time_text = fig.text(0.5, 0.025, "", ha="center", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="upper right", frameon=True)

    def update(frame_index: int):
        tidx = int(frame_ids[frame_index])
        flat_index = 0
        changed: list[Line2D] = []
        for point in POINTS:
            for col in range(2):
                h_line, l_line = artists[flat_index]
                h_line.set_ydata(hdm[point.tag][col][:, tidx])
                l_line.set_ydata(linear[point.tag][col][:, tidx])
                changed.extend((h_line, l_line))
                flat_index += 1
        time_text.set_text(rf"$t={tidx * DT:.2f}$")
        return [*changed, time_text]

    movie = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=90, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=10), dpi=105)
    plt.close(fig)


def gallery_gif(
    output: Path,
    title: str,
    model_keys: list[str],
    root: Path,
    points: list[Point],
    include_linear: bool,
) -> None:
    hdm = {point.tag: snapshot_cut_data(hdm_path(point)) for point in points}
    linear = load_linear_cut_data() if include_linear else {}
    model_lines = {
        key: {
            point.tag: snapshot_cut_data(model_snaps_path(root, MODELS[key], point))
            for point in points
        }
        for key in model_keys
    }

    time_ids = np.arange(0, NT, 20)
    hold_frames = 5
    schedule: list[tuple[str, int]] = []
    for key in model_keys:
        schedule.extend((key, int(tidx)) for tidx in time_ids)
        schedule.extend((key, NT - 1) for _ in range(hold_frames))

    fig, axes = comparison_axes(len(points), title, points)
    artists: list[tuple[Line2D, Line2D | None, Line2D]] = []
    for row, point in enumerate(points):
        for col, grid in enumerate((X, Y)):
            h_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                color=COLORS["HDM"],
                linewidth=3.0,
            )
            linear_line = None
            if include_linear:
                linear_line, = axes[row, col].plot(
                    grid,
                    np.zeros_like(grid),
                    color=COLORS["Linear HPROM"],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
            model_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                linewidth=2.6,
            )
            artists.append((h_line, linear_line, model_line))

    legend_handles = [
        Line2D([0], [0], color=COLORS["HDM"], linewidth=3.0, label="HDM"),
    ]
    if include_linear:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["Linear HPROM"],
                linewidth=1.8,
                linestyle="--",
                label="Linear HPROM",
            )
        )
    active_handle = Line2D([0], [0], color="#000000", linewidth=2.6, label="")
    legend_handles.append(active_handle)
    legend = axes[0, 0].legend(handles=legend_handles, loc="upper right", frameon=True)
    footer = fig.text(0.5, 0.025, "", ha="center", fontsize=12, fontweight="bold")

    def update(frame_index: int):
        key, tidx = schedule[frame_index]
        model = MODELS[key]
        flat_index = 0
        changed: list[Line2D] = []
        for point in points:
            for col in range(2):
                h_line, linear_line, model_line = artists[flat_index]
                h_line.set_ydata(hdm[point.tag][col][:, tidx])
                if linear_line is not None:
                    linear_line.set_ydata(linear[point.tag][col][:, tidx])
                    changed.append(linear_line)
                model_line.set_ydata(model_lines[key][point.tag][col][:, tidx])
                model_line.set_color(model.color)
                changed.extend((h_line, model_line))
                flat_index += 1

        legend_text = legend.get_texts()[-1]
        legend_text.set_text(model.label)
        get_legend_handles(legend)[-1].set_color(model.color)
        footer.set_text(rf"{model.label}   |   $t={tidx * DT:.2f}$")
        return [*changed, footer]

    movie = animation.FuncAnimation(fig, update, frames=len(schedule), interval=110, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=9), dpi=100)
    plt.close(fig)


def enrichment_before_after_gif(output: Path) -> None:
    points = [POINTS[1], POINTS[2]]
    model_keys = ["case2", "podae", "podnn", "poddl"]
    hdm = {point.tag: snapshot_cut_data(hdm_path(point)) for point in points}
    baseline = {
        key: {
            point.tag: snapshot_cut_data(model_snaps_path(MAIN, MODELS[key], point))
            for point in points
        }
        for key in model_keys
    }
    enriched = {
        key: {
            point.tag: snapshot_cut_data(model_snaps_path(ENRICHED, MODELS[key], point))
            for point in points
        }
        for key in model_keys
    }

    time_ids = np.arange(0, NT, 20)
    hold_frames = 5
    schedule: list[tuple[str, int]] = []
    for key in model_keys:
        schedule.extend((key, int(tidx)) for tidx in time_ids)
        schedule.extend((key, NT - 1) for _ in range(hold_frames))

    fig, axes = comparison_axes(
        len(points),
        "Effect of 20 linear-HPROM enrichment trajectories at off-grid points",
        points,
    )
    artists: list[tuple[Line2D, Line2D, Line2D]] = []
    for row, point in enumerate(points):
        for col, grid in enumerate((X, Y)):
            h_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                color=COLORS["HDM"],
                linewidth=3.0,
            )
            b_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                linewidth=2.0,
                linestyle="--",
                alpha=0.85,
            )
            e_line, = axes[row, col].plot(
                grid,
                np.zeros_like(grid),
                linewidth=2.7,
            )
            artists.append((h_line, b_line, e_line))

    legend_handles = [
        Line2D([0], [0], color=COLORS["HDM"], linewidth=3.0, label="HDM"),
        Line2D([0], [0], color="#555555", linewidth=2.0, linestyle="--", label="baseline training"),
        Line2D([0], [0], color="#555555", linewidth=2.7, label="enriched training"),
    ]
    legend = axes[0, 0].legend(handles=legend_handles, loc="upper right", frameon=True)
    footer = fig.text(0.5, 0.025, "", ha="center", fontsize=12, fontweight="bold")

    def update(frame_index: int):
        key, tidx = schedule[frame_index]
        model = MODELS[key]
        flat_index = 0
        changed: list[Line2D] = []
        for point in points:
            for col in range(2):
                h_line, b_line, e_line = artists[flat_index]
                h_line.set_ydata(hdm[point.tag][col][:, tidx])
                b_line.set_ydata(baseline[key][point.tag][col][:, tidx])
                e_line.set_ydata(enriched[key][point.tag][col][:, tidx])
                b_line.set_color(model.color)
                e_line.set_color(model.color)
                changed.extend((h_line, b_line, e_line))
                flat_index += 1

        legend.get_texts()[1].set_text(f"{model.label}: baseline")
        legend.get_texts()[2].set_text(f"{model.label}: enriched")
        handles = get_legend_handles(legend)
        handles[1].set_color(model.color)
        handles[2].set_color(model.color)
        footer.set_text(rf"{model.label}   |   $t={tidx * DT:.2f}$")
        return [*changed, footer]

    movie = animation.FuncAnimation(fig, update, frames=len(schedule), interval=110, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=9), dpi=105)
    plt.close(fig)


def write_manifest(output: Path) -> None:
    rows = [
        ("parameter_training_only.png", "Baseline 3x3 HDM training grid."),
        (
            "parameter_training_evaluation.png",
            "Baseline grid, center verification point, and two off-grid tests.",
        ),
        (
            "parameter_training_evaluation_lhs.png",
            "Baseline/evaluation points plus 20 linear-HPROM LHS trajectories.",
        ),
        (
            "parameter_domain_sequence.gif",
            "Progressive reveal of the three parameter-domain figures.",
        ),
        ("burgers_hdm_3d.gif", "HDM u_x surface evolving at off-grid point mu^(1)."),
        (
            "burgers_cutplane_explanation.png",
            "Static 3D surface with the x- and y-center cuts used in comparisons.",
        ),
        (
            "linear_vs_hdm.gif",
            "HDM versus linear HPROM at all three evaluation points.",
        ),
        (
            "intrusive_model_gallery.gif",
            "One intrusive nonlinear model at a time versus HDM and linear HPROM.",
        ),
        (
            "nonintrusive_model_gallery.gif",
            "One non-intrusive model at a time versus HDM.",
        ),
        (
            "enrichment_before_after.gif",
            "Baseline versus enriched predictions at the two off-grid points.",
        ),
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["asset", "purpose"])
        writer.writerows(rows)


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    training = OUTPUT / "parameter_training_only.png"
    evaluation = OUTPUT / "parameter_training_evaluation.png"
    enriched = OUTPUT / "parameter_training_evaluation_lhs.png"

    parameter_figure("training", training)
    parameter_figure("evaluation", evaluation)
    parameter_figure("enriched", enriched)
    parameter_sequence_gif(
        [training, evaluation, enriched],
        OUTPUT / "parameter_domain_sequence.gif",
    )

    hdm_3d_animation(OUTPUT / "burgers_hdm_3d.gif", POINTS[1])
    cutplane_explanation(
        OUTPUT / "burgers_cutplane_explanation.png",
        POINTS[1],
        time_index=250,
    )
    linear_vs_hdm_gif(OUTPUT / "linear_vs_hdm.gif")
    gallery_gif(
        OUTPUT / "intrusive_model_gallery.gif",
        "Intrusive nonlinear HPROMs: one model at a time",
        ["case1", "case2", "case3", "podae"],
        MAIN,
        POINTS,
        include_linear=True,
    )
    gallery_gif(
        OUTPUT / "nonintrusive_model_gallery.gif",
        "Non-intrusive baselines: direct coordinate prediction",
        ["podnn", "poddl"],
        MAIN,
        POINTS,
        include_linear=False,
    )
    enrichment_before_after_gif(OUTPUT / "enrichment_before_after.gif")
    write_manifest(OUTPUT / "asset_manifest.csv")

    for path in sorted(OUTPUT.iterdir()):
        if path.is_file():
            print(path.relative_to(HERE))


if __name__ == "__main__":
    main()
