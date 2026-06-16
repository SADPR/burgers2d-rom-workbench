import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from animation import AZIMUTH, ELEVATION, HERE, plt
from benchmark import build_benchmark, relative_error


def circular_slice(values, start, length):
    indices = np.arange(start, start + length) % len(values)
    return values[indices], indices


def local_linear_model(segment):
    u_ref = segment[0].copy()
    shifted = segment - u_ref
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    V = vt[:2].T
    q = shifted @ V
    reconstruction = u_ref + q @ V.T
    return u_ref, V, q, reconstruction


data = build_benchmark()
u = data.u[:-1]
n_samples = len(u)
n_segments = 4
overlap = 0.15
segment_length = int(np.ceil((1.0 + overlap) * n_samples / n_segments))
half_length = segment_length // 2
centres = np.linspace(0, n_samples, n_segments, endpoint=False, dtype=int)
colors = ("skyblue", "palegreen", "plum", "lightcoral")

local_models = []
blend_sum = np.zeros_like(u)
blend_count = np.zeros((n_samples, 1))

for centre, color in zip(centres, colors):
    segment, indices = circular_slice(
        u, centre - half_length, segment_length
    )
    u_ref, V, q, reconstruction = local_linear_model(segment)

    padding = 0.18
    q_1 = np.linspace(
        q[:, 0].min() - padding, q[:, 0].max() + padding, 24
    )
    q_2 = np.linspace(
        q[:, 1].min() - padding, q[:, 1].max() + padding, 24
    )
    Q_1, Q_2 = np.meshgrid(q_1, q_2)
    grid_q = np.column_stack((Q_1.ravel(), Q_2.ravel()))
    plane = u_ref + grid_q @ V.T

    local_models.append(
        {
            "indices": indices,
            "reconstruction": reconstruction,
            "surface": (
                plane[:, 0].reshape(Q_1.shape),
                plane[:, 1].reshape(Q_1.shape),
                plane[:, 2].reshape(Q_1.shape),
            ),
            "color": color,
        }
    )
    blend_sum[indices] += reconstruction
    blend_count[indices] += 1.0

u_piecewise = blend_sum / np.clip(blend_count, 1.0, None)
u_closed = np.vstack((u, u[0]))
u_piecewise_closed = np.vstack((u_piecewise, u_piecewise[0]))

fig = plt.figure(figsize=(8.6, 6.8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(r"\textbf{Piecewise linear manifolds}", fontsize=21, pad=12)
ax.set_xlabel(r"$u_1$", fontsize=16, labelpad=8)
ax.set_ylabel(r"$u_2$", fontsize=16, labelpad=8)
ax.set_zlabel(r"$u_3$", fontsize=16, labelpad=7)
ax.tick_params(labelsize=10)
ax.view_init(elev=ELEVATION, azim=AZIMUTH)
ax.set_box_aspect((2.0, 1.8, 1.0))
ax.set_xlim(u[:, 0].min() - 0.25, u[:, 0].max() + 0.25)
ax.set_ylim(u[:, 1].min() - 0.25, u[:, 1].max() + 0.25)
ax.set_zlim(u[:, 2].min() - 0.18, u[:, 2].max() + 0.18)

trajectory, = ax.plot(
    [],
    [],
    [],
    linestyle="none",
    marker="o",
    markersize=2.8,
    color="black",
)
local_curve, = ax.plot(
    [],
    [],
    [],
    linestyle="--",
    linewidth=2.2,
    color="darkorange",
)
piecewise_curve, = ax.plot(
    [],
    [],
    [],
    linewidth=2.8,
    color="darkorange",
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
        color="darkorange",
        linestyle="--",
        linewidth=2.2,
        label=(
            r"local linear approximation"
            "\n"
            r"\hspace{1em}"
            r"$(\mathbf{u}_{\mathrm{ref}}^{(i)}"
            r"+\mathbf{V}^{(i)}\mathbf{q}^{(i)})$"
        ),
    ),
    Line2D(
        [],
        [],
        color="darkorange",
        linewidth=2.8,
        label=r"\textit{piecewise linear approximation}",
    ),
]
for index, model in enumerate(local_models, start=1):
    legend_handles.append(
        Patch(
            facecolor=model["color"],
            edgecolor="gray",
            alpha=0.3,
            label=rf"linear manifold ${index}$",
        )
    )
ax.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.94),
    bbox_transform=fig.transFigure,
    frameon=True,
    fancybox=True,
    framealpha=0.65,
    fontsize=10,
)
piecewise_error = relative_error(
    u_closed, u_piecewise_closed, data.u_ref
)
fig.text(
    0.5,
    0.025,
    rf"relative reconstruction error: ${100.0 * piecewise_error:.2f}\%$",
    ha="center",
    fontsize=13,
)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)

n_trajectory = 120
n_surface_fade = 12
n_local_draw = 55
n_surface_out = 8
n_final_surface = 12
n_piecewise_draw = 120
n_hold = 20
n_spin = 144

phases = []
frame = 0
phases.append(("trajectory", None, frame, frame + n_trajectory))
frame += n_trajectory
for segment_index in range(n_segments):
    phases.append(
        ("surface_in", segment_index, frame, frame + n_surface_fade)
    )
    frame += n_surface_fade
    phases.append(
        ("local_draw", segment_index, frame, frame + n_local_draw)
    )
    frame += n_local_draw
    phases.append(
        ("surface_out", segment_index, frame, frame + n_surface_out)
    )
    frame += n_surface_out
for segment_index in range(n_segments):
    phases.append(
        ("final_surface", segment_index, frame, frame + n_final_surface)
    )
    frame += n_final_surface
phases.append(("piecewise_draw", None, frame, frame + n_piecewise_draw))
frame += n_piecewise_draw
phases.append(("hold", None, frame, frame + n_hold))
frame += n_hold
phases.append(("spin", None, frame, frame + n_spin))
frame += n_spin
total_frames = frame

temporary_surface = None
final_surfaces = [None] * n_segments


def current_phase(frame_number):
    for phase in phases:
        if phase[2] <= frame_number < phase[3]:
            return phase
    return phases[-1]


def update(frame_number):
    global temporary_surface
    kind, segment_index, start, end = current_phase(frame_number)

    if kind == "trajectory":
        count = int(
            np.ceil((frame_number - start + 1) * len(u_closed) / (end - start))
        )
        trajectory.set_data(u_closed[:count, 0], u_closed[:count, 1])
        trajectory.set_3d_properties(u_closed[:count, 2])

    elif kind == "surface_in":
        model = local_models[segment_index]
        if frame_number == start:
            if temporary_surface is not None:
                temporary_surface.remove()
            temporary_surface = ax.plot_surface(
                *model["surface"],
                color=model["color"],
                edgecolor="gray",
                linewidth=0.3,
                alpha=0.0,
            )
            local_curve.set_data([], [])
            local_curve.set_3d_properties([])
        temporary_surface.set_alpha(
            0.18 * (frame_number - start + 1) / (end - start)
        )

    elif kind == "local_draw":
        model = local_models[segment_index]
        count = int(
            np.ceil(
                (frame_number - start + 1)
                * len(model["reconstruction"])
                / (end - start)
            )
        )
        reconstruction = model["reconstruction"]
        local_curve.set_data(
            reconstruction[:count, 0], reconstruction[:count, 1]
        )
        local_curve.set_3d_properties(reconstruction[:count, 2])

    elif kind == "surface_out":
        alpha = 1.0 - (frame_number - start + 1) / (end - start)
        temporary_surface.set_alpha(0.18 * alpha)
        if frame_number == end - 1:
            temporary_surface.remove()
            temporary_surface = None
            local_curve.set_data([], [])
            local_curve.set_3d_properties([])

    elif kind == "final_surface":
        model = local_models[segment_index]
        if final_surfaces[segment_index] is None:
            final_surfaces[segment_index] = ax.plot_surface(
                *model["surface"],
                color=model["color"],
                edgecolor="gray",
                linewidth=0.3,
                alpha=0.0,
            )
        final_surfaces[segment_index].set_alpha(
            0.18 * (frame_number - start + 1) / (end - start)
        )

    elif kind == "piecewise_draw":
        count = int(
            np.ceil(
                (frame_number - start + 1)
                * len(u_piecewise_closed)
                / (end - start)
            )
        )
        piecewise_curve.set_data(
            u_piecewise_closed[:count, 0],
            u_piecewise_closed[:count, 1],
        )
        piecewise_curve.set_3d_properties(u_piecewise_closed[:count, 2])

    elif kind == "spin":
        fraction = (frame_number - start) / max(1, end - start - 1)
        ax.view_init(
            elev=ELEVATION,
            azim=AZIMUTH + 360.0 * fraction,
        )

    return trajectory, local_curve, piecewise_curve


animation = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=1000.0 / 24.0,
    blit=False,
)
output = HERE / "outputs" / "piecewise_linear_manifold.gif"
output.parent.mkdir(exist_ok=True)
animation.save(output, writer=PillowWriter(fps=24), dpi=100)
plt.close(fig)
print(output)
