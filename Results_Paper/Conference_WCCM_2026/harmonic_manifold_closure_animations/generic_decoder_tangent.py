"""Animate a generic decoder manifold and its local tangent space."""

from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from animation import AZIMUTH, ELEVATION, plt
from benchmark import build_benchmark


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"


def decoder(q):
    q = np.asarray(q)
    q_1 = q[..., 0]
    q_2 = q[..., 1]
    u_3 = (
        0.40 * (q_1**2 - q_2**2)
        + 0.06 * (3.0 * q_2 - 4.0 * q_2**3)
    )
    return np.stack((q_1, q_2, u_3), axis=-1)


def tangent(q):
    q_1, q_2 = q
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.80 * q_1, -0.80 * q_2 + 0.18 - 0.72 * q_2**2],
        ]
    )


def tangent_patch(q, half_width=0.25):
    offsets = np.array(
        [
            [-half_width, -half_width],
            [half_width, -half_width],
            [half_width, half_width],
            [-half_width, half_width],
        ]
    )
    state = decoder(q)
    basis = tangent(q)
    return state + offsets @ basis.T


def build_figure():
    data = build_benchmark()
    q_axis = np.linspace(-1.12, 1.12, 60)
    Q_1, Q_2 = np.meshgrid(q_axis, q_axis)
    grid_q = np.stack((Q_1, Q_2), axis=-1)
    surface = decoder(grid_q)

    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        r"\textbf{Decoder manifold and local tangent space}",
        fontsize=20,
        pad=12,
    )
    ax.set_xlabel(r"$u_1$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$u_2$", fontsize=16, labelpad=8)
    ax.set_zlabel(r"$u_3$", fontsize=16, labelpad=7)
    ax.tick_params(labelsize=10)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect((2.0, 1.8, 1.0))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-0.65, 0.65)

    manifold = ax.plot_surface(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        color="lightskyblue",
        alpha=0.0,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    wire = ax.plot_wireframe(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        rstride=4,
        cstride=4,
        color="slategray",
        linewidth=0.35,
        alpha=0.0,
    )
    trajectory, = ax.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=2.8,
        color="black",
    )
    selected_point, = ax.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=8,
        color="crimson",
        zorder=10,
    )
    tangent_q1, = ax.plot(
        [],
        [],
        [],
        color="royalblue",
        linewidth=4.0,
    )
    tangent_q2, = ax.plot(
        [],
        [],
        [],
        color="darkorange",
        linewidth=4.0,
    )

    initial_patch = tangent_patch(data.harmonic_q[0])
    tangent_plane = Poly3DCollection(
        [initial_patch],
        facecolor="gold",
        edgecolor="darkgoldenrod",
        linewidth=0.8,
        alpha=0.0,
    )
    ax.add_collection3d(tangent_plane)

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="none",
                label=r"trajectory $\mathbf{u}(t)$",
            ),
            Patch(
                facecolor="lightskyblue",
                edgecolor="slategray",
                alpha=0.35,
                label=r"trial manifold $\mathcal M$",
            ),
            Line2D(
                [],
                [],
                color="crimson",
                marker="o",
                linestyle="none",
                label=r"$\widetilde{\mathbf u}=\mathcal D(\mathbf q)$",
            ),
            Patch(
                facecolor="gold",
                edgecolor="darkgoldenrod",
                alpha=0.45,
                label=r"local tangent space",
            ),
            Line2D(
                [],
                [],
                color="royalblue",
                linewidth=3.0,
                label=r"$\partial\mathcal D/\partial q_1$",
            ),
            Line2D(
                [],
                [],
                color="darkorange",
                linewidth=3.0,
                label=r"$\partial\mathcal D/\partial q_2$",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.97, 0.93),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=True,
        framealpha=0.72,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.060,
        r"$\mathbf T(\mathbf q)=\dfrac{\partial\mathcal D}{\partial\mathbf q}"
        r"=\left["
        r"\dfrac{\partial\mathcal D}{\partial q_1}\ "
        r"\dfrac{\partial\mathcal D}{\partial q_2}"
        r"\right]$",
        ha="center",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.91)

    return (
        data,
        fig,
        ax,
        manifold,
        wire,
        trajectory,
        selected_point,
        tangent_plane,
        tangent_q1,
        tangent_q2,
    )


def set_local_geometry(
    data,
    index,
    alpha,
    selected_point,
    tangent_plane,
    tangent_q1,
    tangent_q2,
):
    q = data.harmonic_q[index]
    state = decoder(q)
    basis = tangent(q)
    vector_scale = 0.43

    selected_point.set_data([state[0]], [state[1]])
    selected_point.set_3d_properties([state[2]])
    selected_point.set_alpha(alpha)

    patch = tangent_patch(q)
    tangent_plane.set_verts([patch])
    tangent_plane.set_alpha(0.30 * alpha)

    endpoint_1 = state + vector_scale * basis[:, 0]
    tangent_q1.set_data(
        [state[0], endpoint_1[0]],
        [state[1], endpoint_1[1]],
    )
    tangent_q1.set_3d_properties([state[2], endpoint_1[2]])
    tangent_q1.set_alpha(alpha)

    endpoint_2 = state + vector_scale * basis[:, 1]
    tangent_q2.set_data(
        [state[0], endpoint_2[0]],
        [state[1], endpoint_2[1]],
    )
    tangent_q2.set_3d_properties([state[2], endpoint_2[2]])
    tangent_q2.set_alpha(alpha)


def render_static():
    elements = build_figure()
    (
        data,
        fig,
        _,
        manifold,
        wire,
        trajectory,
        selected_point,
        tangent_plane,
        tangent_q1,
        tangent_q2,
    ) = elements

    manifold.set_alpha(0.20)
    wire.set_alpha(0.32)
    trajectory.set_data(data.u[:, 0], data.u[:, 1])
    trajectory.set_3d_properties(data.u[:, 2])
    set_local_geometry(
        data,
        43,
        1.0,
        selected_point,
        tangent_plane,
        tangent_q1,
        tangent_q2,
    )

    for suffix in ("png", "pdf", "svg"):
        output_path = OUTPUT_DIR / f"generic_decoder_tangent.{suffix}"
        fig.savefig(
            output_path,
            dpi=220 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        print(output_path)
    plt.close(fig)


def render_gif():
    elements = build_figure()
    (
        data,
        fig,
        ax,
        manifold,
        wire,
        trajectory,
        selected_point,
        tangent_plane,
        tangent_q1,
        tangent_q2,
    ) = elements

    curve_steps = np.unique(np.linspace(1, len(data.u), 110, dtype=int))
    moving_indices = np.linspace(
        0,
        len(data.u) - 2,
        180,
        dtype=int,
    )
    n_trajectory = len(curve_steps)
    n_manifold_fade = 20
    n_tangent_fade = 18
    n_move = len(moving_indices)
    n_hold = 24
    n_rotation = 120
    total_frames = (
        n_trajectory
        + n_manifold_fade
        + n_tangent_fade
        + n_move
        + n_hold
        + n_rotation
    )

    def update(frame):
        manifold_start = n_trajectory
        tangent_start = manifold_start + n_manifold_fade
        move_start = tangent_start + n_tangent_fade
        hold_start = move_start + n_move
        rotation_start = hold_start + n_hold

        if frame < n_trajectory:
            end = curve_steps[frame]
            trajectory.set_data(data.u[:end, 0], data.u[:end, 1])
            trajectory.set_3d_properties(data.u[:end, 2])
        else:
            trajectory.set_data(data.u[:, 0], data.u[:, 1])
            trajectory.set_3d_properties(data.u[:, 2])

        if manifold_start <= frame < tangent_start:
            alpha = (frame - manifold_start + 1) / n_manifold_fade
            manifold.set_alpha(0.20 * alpha)
            wire.set_alpha(0.32 * alpha)
        elif frame >= tangent_start:
            manifold.set_alpha(0.20)
            wire.set_alpha(0.32)

        if tangent_start <= frame < move_start:
            alpha = (frame - tangent_start + 1) / n_tangent_fade
            set_local_geometry(
                data,
                0,
                alpha,
                selected_point,
                tangent_plane,
                tangent_q1,
                tangent_q2,
            )
        elif move_start <= frame < hold_start:
            index = moving_indices[frame - move_start]
            set_local_geometry(
                data,
                index,
                1.0,
                selected_point,
                tangent_plane,
                tangent_q1,
                tangent_q2,
            )
        elif frame >= hold_start:
            set_local_geometry(
                data,
                moving_indices[-1],
                1.0,
                selected_point,
                tangent_plane,
                tangent_q1,
                tangent_q2,
            )

        if frame >= rotation_start:
            angle = AZIMUTH + 360.0 * (
                frame - rotation_start
            ) / n_rotation
            ax.view_init(elev=ELEVATION, azim=angle)

        return (
            manifold,
            wire,
            trajectory,
            selected_point,
            tangent_plane,
            tangent_q1,
            tangent_q2,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000.0 / 24.0,
        blit=False,
    )
    output_path = OUTPUT_DIR / "generic_decoder_tangent.gif"
    animation.save(output_path, writer=PillowWriter(fps=24), dpi=100)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    render_static()
    render_gif()
