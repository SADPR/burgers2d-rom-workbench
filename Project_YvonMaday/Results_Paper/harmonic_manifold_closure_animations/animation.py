import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.unicode_minus": False,
    }
)

ELEVATION = 15
AZIMUTH = 225


@dataclass(frozen=True)
class AnimationSpec:
    title: str
    output_name: str
    surface_color: str
    curve_color: str
    approximation_label: str
    manifold_label: str
    error: float
    show_linear_mapping: bool = False
    curve_linestyle: str = "-"


def render_manifold_animation(data, surface, approximation, spec):
    output_dir = HERE / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / spec.output_name

    surface_x, surface_y, surface_z = surface
    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title(spec.title, fontsize=21, pad=12)
    ax.set_xlabel(r"$u_1$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$u_2$", fontsize=16, labelpad=8)
    ax.set_zlabel(r"$u_3$", fontsize=16, labelpad=7)
    ax.tick_params(labelsize=10)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect((2.0, 1.8, 1.0))

    all_x = np.concatenate((surface_x.ravel(), data.u[:, 0]))
    all_y = np.concatenate((surface_y.ravel(), data.u[:, 1]))
    all_z = np.concatenate((surface_z.ravel(), data.u[:, 2]))
    ax.set_xlim(all_x.min() - 0.08, all_x.max() + 0.08)
    ax.set_ylim(all_y.min() - 0.08, all_y.max() + 0.08)
    ax.set_zlim(all_z.min() - 0.08, all_z.max() + 0.08)

    true_curve, = ax.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=2.8,
        color="black",
        label=r"\textit{trajectory} $\mathbf{u}(t)$",
    )
    linear_curve, = ax.plot(
        [],
        [],
        [],
        color="gray",
        linewidth=1.7,
    )
    manifold = ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        color=spec.surface_color,
        alpha=0.0,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    wire = ax.plot_wireframe(
        surface_x,
        surface_y,
        surface_z,
        rstride=3,
        cstride=3,
        color="gray",
        linewidth=0.35,
        alpha=0.0,
    )
    reconstructed_curve, = ax.plot(
        [],
        [],
        [],
        color=spec.curve_color,
        linewidth=3.0,
        linestyle=spec.curve_linestyle,
    )
    lift, = ax.plot(
        [],
        [],
        [],
        color="crimson",
        linewidth=3.2,
        alpha=0.9,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="none",
            label=r"\textit{trajectory} $\mathbf{u}(t)$",
        )
    ]
    if spec.show_linear_mapping:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="gray",
                linewidth=1.7,
                label=(
                    r"linear approximation"
                    "\n"
                    r"\hspace{1em}"
                    r"$(\mathbf{u}_{\mathrm{ref}}"
                    r"+\mathbf{V}\mathbf{q})$"
                ),
            )
        )
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                color=spec.curve_color,
                linewidth=2.5,
                linestyle=spec.curve_linestyle,
                label=spec.approximation_label,
            ),
            Patch(
                facecolor=spec.surface_color,
                alpha=0.3,
                edgecolor="gray",
                label=spec.manifold_label,
            ),
        ]
    )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.93),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=True,
        framealpha=0.65,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        rf"relative reconstruction error: ${100.0 * spec.error:.2f}\%$",
        ha="center",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)

    curve_steps = np.unique(
        np.linspace(1, len(data.u), 120, dtype=int)
    )
    n_true = len(curve_steps)
    n_linear = len(curve_steps) if spec.show_linear_mapping else 0
    n_fade = 18
    n_reconstruction = len(curve_steps)
    n_hold = 24
    n_rotation = 144
    total_frames = (
        n_true
        + n_linear
        + n_fade
        + n_reconstruction
        + n_hold
        + n_rotation
    )

    def update(frame):
        if frame < n_true:
            end = curve_steps[frame]
            true_curve.set_data(data.u[:end, 0], data.u[:end, 1])
            true_curve.set_3d_properties(data.u[:end, 2])
        else:
            true_curve.set_data(data.u[:, 0], data.u[:, 1])
            true_curve.set_3d_properties(data.u[:, 2])

        linear_start = n_true
        fade_start = linear_start + n_linear
        reconstruction_start = fade_start + n_fade
        hold_start = reconstruction_start + n_reconstruction
        rotation_start = hold_start + n_hold

        if spec.show_linear_mapping:
            if linear_start <= frame < fade_start:
                local_frame = frame - linear_start
                end = curve_steps[local_frame]
                linear_curve.set_data(
                    data.u_linear[:end, 0], data.u_linear[:end, 1]
                )
                linear_curve.set_3d_properties(data.u_linear[:end, 2])
            elif frame >= fade_start:
                linear_curve.set_data(
                    data.u_linear[:, 0], data.u_linear[:, 1]
                )
                linear_curve.set_3d_properties(data.u_linear[:, 2])

        if fade_start <= frame < reconstruction_start:
            alpha = (frame - fade_start + 1) / n_fade
            manifold.set_alpha(0.18 * alpha)
            wire.set_alpha(0.35 * alpha)
        elif frame >= reconstruction_start:
            manifold.set_alpha(0.18)
            wire.set_alpha(0.35)

        if reconstruction_start <= frame < hold_start:
            local_frame = frame - reconstruction_start
            end = curve_steps[local_frame]
            reconstructed_curve.set_data(
                approximation[:end, 0], approximation[:end, 1]
            )
            reconstructed_curve.set_3d_properties(approximation[:end, 2])

            index = end - 1
            if spec.show_linear_mapping:
                lift.set_data(
                    [data.u_linear[index, 0], approximation[index, 0]],
                    [data.u_linear[index, 1], approximation[index, 1]],
                )
                lift.set_3d_properties(
                    [data.u_linear[index, 2], approximation[index, 2]]
                )
        elif frame >= hold_start:
            reconstructed_curve.set_data(
                approximation[:, 0], approximation[:, 1]
            )
            reconstructed_curve.set_3d_properties(approximation[:, 2])

        if frame >= rotation_start:
            angle = AZIMUTH + 360.0 * (frame - rotation_start) / n_rotation
            ax.view_init(elev=ELEVATION, azim=angle)

        return (
            true_curve,
            linear_curve,
            reconstructed_curve,
            lift,
            manifold,
            wire,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000.0 / 24.0,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=24), dpi=100)
    plt.close(fig)
    print(output_path)
