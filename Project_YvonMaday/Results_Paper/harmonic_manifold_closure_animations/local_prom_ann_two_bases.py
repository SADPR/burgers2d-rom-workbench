import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from animation import AZIMUTH, ELEVATION, HERE, plt


def base_closure(q_1, q_2):
    return 0.18 * (q_1**2 - 0.55 * q_2**2)


def closure_1(q_1, q_2):
    bump = 0.48 * np.exp(
        -((q_1 + 0.42) ** 2 / 0.11 + (q_2 - 0.62) ** 2 / 0.050)
    )
    return base_closure(q_1, q_2) + bump


def closure_2(q_1, q_2):
    dip = 0.40 * np.exp(
        -((q_1 - 0.45) ** 2 / 0.12 + (q_2 + 0.58) ** 2 / 0.055)
    )
    return base_closure(q_1, q_2) - dip


def build_surface(closure, q_2_min, q_2_max, n=58):
    q_1 = np.linspace(-1.18, 1.18, n)
    q_2 = np.linspace(q_2_min, q_2_max, n)
    Q_1, Q_2 = np.meshgrid(q_1, q_2)
    Z = closure(Q_1, Q_2)
    return Q_1, Q_2, Z


# Deliberately simple two-chart trajectory. The first half in time lives on
# local closure 1; the second half lives on local closure 2. This is a visual
# future-work sketch, not a quantitative benchmark result.
n_samples = 241
t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
q_1 = np.cos(t)
q_2 = 0.86 * np.sin(t)
first_half = t < np.pi
second_half = ~first_half

u = np.empty((n_samples, 3))
u[:, 0] = q_1
u[:, 1] = q_2
u[first_half, 2] = closure_1(q_1[first_half], q_2[first_half])
u[second_half, 2] = closure_2(q_1[second_half], q_2[second_half])

local_1 = u[first_half]
local_2 = u[second_half]
switched = np.vstack((local_1, local_2))

surface_1 = build_surface(closure_1, -0.05, 1.08)
surface_2 = build_surface(closure_2, -1.08, 0.05)

output_dir = HERE / "outputs"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "local_prom_ann_two_bases.gif"
preview_path = output_dir / "local_prom_ann_two_bases_preview.png"

fig = plt.figure(figsize=(8.6, 6.8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(
    r"\textbf{Two local PROM--ANN closure manifolds}",
    fontsize=19,
    pad=12,
)
ax.set_xlabel(r"$u_1$", fontsize=16, labelpad=8)
ax.set_ylabel(r"$u_2$", fontsize=16, labelpad=8)
ax.set_zlabel(r"$u_3$", fontsize=16, labelpad=7)
ax.tick_params(labelsize=10)
ax.view_init(elev=ELEVATION, azim=AZIMUTH)
ax.set_box_aspect((2.0, 1.8, 1.0))
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.25, 1.25)
ax.set_zlim(-0.55, 0.70)

trajectory, = ax.plot(
    [],
    [],
    [],
    linestyle="none",
    marker="o",
    markersize=2.8,
    color="black",
)
local_curve_1, = ax.plot(
    [],
    [],
    [],
    color="mediumorchid",
    linestyle="--",
    linewidth=2.5,
)
local_curve_2, = ax.plot(
    [],
    [],
    color="mediumseagreen",
    linestyle="--",
    linewidth=2.5,
)
switched_curve, = ax.plot(
    [],
    [],
    color="crimson",
    linewidth=3.2,
)

surface_artist_1 = ax.plot_surface(
    *surface_1,
    color="plum",
    alpha=0.0,
    linewidth=0,
    antialiased=True,
    shade=True,
)
surface_artist_2 = ax.plot_surface(
    *surface_2,
    color="palegreen",
    alpha=0.0,
    linewidth=0,
    antialiased=True,
    shade=True,
)

legend_handles = [
    Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="none",
        label=r"\textit{trajectory} $\mathbf{u}(t)$",
    ),
    Line2D(
        [],
        [],
        color="mediumorchid",
        linestyle="--",
        linewidth=2.5,
        label=(
            r"local PROM--ANN chart 1"
            "\n"
            r"\hspace{1em}"
            r"$(\mathbf{u}_{\mathrm{ref}}^{(1)}"
            r"+\mathbf{V}^{(1)}\mathbf{q}^{(1)}"
            r"+\overline{\mathbf{V}}^{(1)}"
            r"\mathcal{N}_1(\mathbf{q}^{(1)}))$"
        ),
    ),
    Line2D(
        [],
        [],
        color="mediumseagreen",
        linestyle="--",
        linewidth=2.5,
        label=(
            r"local PROM--ANN chart 2"
            "\n"
            r"\hspace{1em}"
            r"$(\mathbf{u}_{\mathrm{ref}}^{(2)}"
            r"+\mathbf{V}^{(2)}\mathbf{q}^{(2)}"
            r"+\overline{\mathbf{V}}^{(2)}"
            r"\mathcal{N}_2(\mathbf{q}^{(2)}))$"
        ),
    ),
    Line2D(
        [],
        [],
        color="crimson",
        linewidth=3.2,
        label=r"\textit{piecewise nonlinear approximation}",
    ),
    Patch(
        facecolor="plum",
        edgecolor="none",
        alpha=0.45,
        label=r"local closure manifold 1",
    ),
    Patch(
        facecolor="palegreen",
        edgecolor="none",
        alpha=0.45,
        label=r"local closure manifold 2",
    ),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.93),
    bbox_transform=fig.transFigure,
    frameon=True,
    fancybox=True,
    framealpha=0.72,
    fontsize=8.7,
)
fig.text(
    0.5,
    0.025,
    r"simple time split: $0\leq t<\pi$ uses chart 1, $\pi\leq t<2\pi$ uses chart 2",
    ha="center",
    fontsize=12.5,
)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)

steps_all = np.unique(np.linspace(1, n_samples, 120, dtype=int))
steps_1 = np.unique(np.linspace(1, len(local_1), 70, dtype=int))
steps_2 = np.unique(np.linspace(1, len(local_2), 70, dtype=int))
n_trajectory = len(steps_all)
n_fade = 18
n_local_1 = len(steps_1)
n_local_2 = len(steps_2)
n_switch = len(steps_all)
n_hold = 24
n_spin = 144

fade_1_start = n_trajectory
local_1_start = fade_1_start + n_fade
fade_2_start = local_1_start + n_local_1
local_2_start = fade_2_start + n_fade
switch_start = local_2_start + n_local_2
hold_start = switch_start + n_switch
spin_start = hold_start + n_hold
total_frames = spin_start + n_spin


def set_curve(line, points, count):
    line.set_data(points[:count, 0], points[:count, 1])
    line.set_3d_properties(points[:count, 2])


def reset_animation_state():
    for line in (
        trajectory,
        local_curve_1,
        local_curve_2,
        switched_curve,
    ):
        line.set_data([], [])
        line.set_3d_properties([])
    surface_artist_1.set_alpha(0.0)
    surface_artist_2.set_alpha(0.0)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    return (
        trajectory,
        local_curve_1,
        local_curve_2,
        switched_curve,
        surface_artist_1,
        surface_artist_2,
    )


def update(frame):
    if frame < n_trajectory:
        set_curve(trajectory, u, steps_all[frame])
    else:
        set_curve(trajectory, u, n_samples)

    if fade_1_start <= frame < local_1_start:
        alpha = (frame - fade_1_start + 1) / n_fade
        surface_artist_1.set_alpha(0.28 * alpha)
    elif frame >= local_1_start:
        surface_artist_1.set_alpha(0.28)

    if local_1_start <= frame < fade_2_start:
        set_curve(local_curve_1, local_1, steps_1[frame - local_1_start])
    elif frame >= fade_2_start:
        set_curve(local_curve_1, local_1, len(local_1))

    if fade_2_start <= frame < local_2_start:
        alpha = (frame - fade_2_start + 1) / n_fade
        surface_artist_2.set_alpha(0.28 * alpha)
    elif frame >= local_2_start:
        surface_artist_2.set_alpha(0.28)

    if local_2_start <= frame < switch_start:
        set_curve(local_curve_2, local_2, steps_2[frame - local_2_start])
    elif frame >= switch_start:
        set_curve(local_curve_2, local_2, len(local_2))

    if switch_start <= frame < hold_start:
        set_curve(switched_curve, switched, steps_all[frame - switch_start])
    elif frame >= hold_start:
        set_curve(switched_curve, switched, n_samples)

    if frame >= spin_start:
        fraction = (frame - spin_start) / max(1, n_spin - 1)
        ax.view_init(elev=ELEVATION, azim=AZIMUTH + 360.0 * fraction)

    return (
        trajectory,
        local_curve_1,
        local_curve_2,
        switched_curve,
        surface_artist_1,
        surface_artist_2,
    )


reset_animation_state()
animation = FuncAnimation(
    fig,
    update,
    init_func=reset_animation_state,
    frames=total_frames,
    interval=1000.0 / 24.0,
    blit=False,
)
animation.save(output_path, writer=PillowWriter(fps=24), dpi=100)

update(hold_start)
fig.savefig(preview_path, dpi=150)

plt.close(fig)
print(output_path)
print(preview_path)
