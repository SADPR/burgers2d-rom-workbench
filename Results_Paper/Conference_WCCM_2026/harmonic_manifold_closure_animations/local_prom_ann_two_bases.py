import argparse
from dataclasses import dataclass

import numpy as np

from animation import AZIMUTH, ELEVATION, HERE, plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DOMAIN = (-1.05, 1.05)
N_LINEAR_CHARTS = 6
ASSETS = ("local_linear", "global_ann_rbf_gpr", "local_nonlinear")


@dataclass(frozen=True)
class ManifoldData:
    q: np.ndarray
    u: np.ndarray
    Q_1: np.ndarray
    Q_2: np.ndarray
    Z: np.ndarray
    upper_surface: tuple[np.ndarray, np.ndarray, np.ndarray]
    lower_surface: tuple[np.ndarray, np.ndarray, np.ndarray]
    global_gpr_surface: tuple[np.ndarray, np.ndarray, np.ndarray]
    global_gpr_curve: np.ndarray
    local_linear_curve: np.ndarray
    local_nonlinear_curve: np.ndarray
    local_linear_patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    train_q: np.ndarray
    train_z: np.ndarray
    local_linear_error: float
    global_gpr_error: float
    local_nonlinear_error: float


def gaussian(q_1, q_2, center_1, center_2, amplitude, width_1, width_2):
    return amplitude * np.exp(
        -0.5
        * (
            ((q_1 - center_1) / width_1) ** 2
            + ((q_2 - center_2) / width_2) ** 2
        )
    )


def backbone(q_1, q_2):
    return (
        0.10 * (q_1**2 - 0.55 * q_2**2)
        - 0.06 * q_1 * q_2
        + 0.055 * np.sin(1.5 * np.pi * q_1) * np.cos(0.7 * np.pi * q_2)
    )


def upper_features(q_1, q_2):
    return (
        gaussian(q_1, q_2, -0.68, 0.54, 0.62, 0.16, 0.17)
        - gaussian(q_1, q_2, -0.10, 0.78, 0.34, 0.18, 0.13)
        + gaussian(q_1, q_2, 0.62, 0.33, 0.47, 0.13, 0.20)
        + gaussian(q_1, q_2, 0.78, 0.76, 0.27, 0.11, 0.11)
    )


def lower_features(q_1, q_2):
    return (
        -gaussian(q_1, q_2, -0.62, -0.55, 0.54, 0.15, 0.18)
        + gaussian(q_1, q_2, 0.36, -0.68, 0.51, 0.17, 0.15)
        - gaussian(q_1, q_2, 0.78, -0.20, 0.30, 0.12, 0.19)
        + gaussian(q_1, q_2, -0.12, -0.18, 0.20, 0.20, 0.14)
    )


def upper_weight(q_2, width=0.13):
    return 0.5 * (1.0 + np.tanh(q_2 / width))


def local_closure(q_1, q_2):
    weight = upper_weight(q_2)
    return (
        backbone(q_1, q_2)
        + weight * upper_features(q_1, q_2)
        + (1.0 - weight) * lower_features(q_1, q_2)
    )


def upper_closure(q_1, q_2):
    return local_closure(q_1, q_2)


def lower_closure(q_1, q_2):
    return local_closure(q_1, q_2)


def build_surface(closure, q_2_min, q_2_max, n=72):
    q_1 = np.linspace(DOMAIN[0], DOMAIN[1], n)
    q_2 = np.linspace(q_2_min, q_2_max, n)
    Q_1, Q_2 = np.meshgrid(q_1, q_2)
    Z = closure(Q_1, Q_2)
    return Q_1, Q_2, Z


def build_global_surface(n=74):
    q_1 = np.linspace(DOMAIN[0], DOMAIN[1], n)
    q_2 = np.linspace(DOMAIN[0], DOMAIN[1], n)
    Q_1, Q_2 = np.meshgrid(q_1, q_2)
    Z = local_closure(Q_1, Q_2)
    return Q_1, Q_2, Z


def relative_error(reference, approximation):
    reference_centered = reference - reference.mean(axis=0)
    return np.linalg.norm(reference - approximation) / np.linalg.norm(reference_centered)


def rbf_kernel(left, right, length_scale):
    left = np.asarray(left)
    right = np.asarray(right)
    squared_distance = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * squared_distance / length_scale**2)


def global_gpr_predict(train_q, train_z, evaluation_q, length_scale=0.52, noise=5.0e-3):
    mean_z = train_z.mean()
    centered_z = train_z - mean_z
    covariance = rbf_kernel(train_q, train_q, length_scale)
    covariance = covariance + noise * np.eye(len(train_q))
    weights = np.linalg.solve(covariance, centered_z)
    cross_covariance = rbf_kernel(evaluation_q, train_q, length_scale)
    return mean_z + cross_covariance @ weights


def cyclic_indices(center, half_width, n_points):
    return np.arange(center - half_width, center + half_width + 1) % n_points


def fit_rank_two_pod_plane(points):
    u_ref = points.mean(axis=0)
    shifted = points - u_ref
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    return u_ref, vt[:2].T


def reconstruct_on_plane(points, u_ref, basis):
    local_q = (points - u_ref) @ basis
    return u_ref + local_q @ basis.T


def build_pod_patch(points, u_ref, basis):
    local_q = (points - u_ref) @ basis
    q_min = local_q.min(axis=0)
    q_max = local_q.max(axis=0)
    q_center = 0.5 * (q_min + q_max)
    half_side = 0.58 * np.max(q_max - q_min)
    half_side = max(half_side, 0.07)

    q_1 = np.linspace(q_center[0] - half_side, q_center[0] + half_side, 9)
    q_2 = np.linspace(q_center[1] - half_side, q_center[1] + half_side, 9)
    Q_1, Q_2 = np.meshgrid(q_1, q_2)
    grid_q = np.column_stack((Q_1.ravel(), Q_2.ravel()))
    plane = u_ref + grid_q @ basis.T
    return tuple(plane[:, component].reshape(Q_1.shape) for component in range(3))


def build_local_linear_pod(u):
    n_points = len(u)
    center_indices = np.linspace(0, n_points, N_LINEAR_CHARTS, endpoint=False, dtype=int)
    fit_half_width = max(9, int(np.ceil(0.55 * n_points / N_LINEAR_CHARTS)))

    models = []
    patches = []
    for center in center_indices:
        indices = cyclic_indices(center, fit_half_width, n_points)
        points = u[indices]
        u_ref, basis = fit_rank_two_pod_plane(points)
        models.append((center, u_ref, basis))
        patches.append(build_pod_patch(points, u_ref, basis))

    sample_indices = np.arange(n_points)
    distances = np.array(
        [
            np.minimum(
                np.abs(sample_indices - center),
                n_points - np.abs(sample_indices - center),
            )
            for center, _, _ in models
        ]
    )
    assignments = distances.argmin(axis=0)
    reconstruction = np.empty_like(u)
    for model_index, (_, u_ref, basis) in enumerate(models):
        mask = assignments == model_index
        reconstruction[mask] = reconstruct_on_plane(u[mask], u_ref, basis)

    return patches, reconstruction


def build_manifold_data(n_samples=289):
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    q_1 = 0.84 * np.cos(theta) + 0.13 * np.cos(3.0 * theta + 0.25)
    q_2 = 0.78 * np.sin(theta) + 0.12 * np.sin(2.0 * theta - 0.45)
    q = np.column_stack((q_1, q_2))
    z = local_closure(q_1, q_2)
    u = np.column_stack((q_1, q_2, z))

    Q_1, Q_2, Z = build_global_surface()
    upper_surface = build_surface(upper_closure, -0.04, DOMAIN[1])
    lower_surface = build_surface(lower_closure, DOMAIN[0], 0.04)

    train_indices = np.arange(0, n_samples, 8)
    train_q = q[train_indices]
    train_z = z[train_indices]
    global_gpr_z = global_gpr_predict(train_q, train_z, q)
    global_gpr_curve = np.column_stack((q_1, q_2, global_gpr_z))
    grid_q = np.column_stack((Q_1.ravel(), Q_2.ravel()))
    global_gpr_surface_z = global_gpr_predict(train_q, train_z, grid_q).reshape(Q_1.shape)

    local_linear_patches, local_linear_curve = build_local_linear_pod(u)
    local_nonlinear_curve = u.copy()

    return ManifoldData(
        q=q,
        u=u,
        Q_1=Q_1,
        Q_2=Q_2,
        Z=Z,
        upper_surface=upper_surface,
        lower_surface=lower_surface,
        global_gpr_surface=(Q_1, Q_2, global_gpr_surface_z),
        global_gpr_curve=global_gpr_curve,
        local_linear_curve=local_linear_curve,
        local_nonlinear_curve=local_nonlinear_curve,
        local_linear_patches=local_linear_patches,
        train_q=train_q,
        train_z=train_z,
        local_linear_error=relative_error(u, local_linear_curve),
        global_gpr_error=relative_error(u, global_gpr_curve),
        local_nonlinear_error=relative_error(u, local_nonlinear_curve),
    )


def axis_limits(data):
    z_values = [
        data.Z.ravel(),
        data.global_gpr_surface[2].ravel(),
        data.local_linear_curve[:, 2],
        data.global_gpr_curve[:, 2],
    ]
    for patch in data.local_linear_patches:
        z_values.append(patch[2].ravel())
    z = np.concatenate(z_values)
    return (
        (DOMAIN[0] - 0.10, DOMAIN[1] + 0.10),
        (DOMAIN[0] - 0.10, DOMAIN[1] + 0.10),
        (z.min() - 0.12, z.max() + 0.12),
    )


def configure_axis(ax, title, limits):
    ax.set_title(title, fontsize=21, pad=12)
    ax.set_xlabel(r"$q_1$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$q_2$", fontsize=16, labelpad=8)
    ax.set_zlabel(r"$u_3$", fontsize=16, labelpad=7)
    ax.tick_params(labelsize=10)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect((2.0, 1.8, 1.0))
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


def set_curve(line, points, count):
    line.set_data(points[:count, 0], points[:count, 1])
    line.set_3d_properties(points[:count, 2])


def add_true_wireframe(ax, data, alpha=0.0):
    return ax.plot_wireframe(
        data.Q_1,
        data.Q_2,
        data.Z,
        rstride=7,
        cstride=7,
        color="0.45",
        linewidth=0.35,
        alpha=alpha,
    )


def set_fade_alpha(fade_artists, scale=1.0):
    for artist, target_alpha in fade_artists:
        artist.set_visible(scale > 0.0)
        artist.set_alpha(target_alpha * scale)


def add_footer(fig, error):
    fig.text(
        0.5,
        0.025,
        rf"relative reconstruction error: ${100.0 * error:.2f}\%$",
        ha="center",
        fontsize=13,
    )


def local_linear_legend():
    return [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="none",
            label=r"\textit{trajectory} $\mathbf{u}(t)$",
        ),
        Patch(
            facecolor="lightsalmon",
            edgecolor="sienna",
            alpha=0.35,
            label=rf"${N_LINEAR_CHARTS}$ local rank-2 affine charts",
        ),
        Line2D(
            [],
            [],
            color="darkorange",
            linestyle="--",
            linewidth=2.5,
            label=(
                r"piecewise local POD"
                "\n"
                r"\hspace{1em}"
                r"$(\mathbf{u}_{\mathrm{ref}}^{(i)}"
                r"+\mathbf{V}^{(i)}\mathbf{q}^{(i)})$"
            ),
        ),
    ]


def global_ann_rbf_gpr_legend():
    return [
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
            color="royalblue",
            marker="o",
            linestyle="none",
            label=r"closure training samples",
        ),
        Patch(
            facecolor="plum",
            edgecolor="mediumvioletred",
            alpha=0.35,
            label=r"global ANN/RBF/GPR surface",
        ),
        Line2D(
            [],
            [],
            color="crimson",
            linestyle="--",
            linewidth=2.5,
            label=(
                r"global ANN/RBF/GPR closure"
                "\n"
                r"\hspace{1em}"
                r"$(\mathbf{u}_{\mathrm{ref}}+\mathbf{V}\mathbf{q}"
                r"+\overline{\mathbf{V}}\,\mathcal{N}(\mathbf{q}))$"
            ),
        ),
    ]


def local_nonlinear_legend():
    return [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="none",
            label=r"\textit{trajectory} $\mathbf{u}(t)$",
        ),
        Patch(
            facecolor="plum",
            edgecolor="mediumorchid",
            alpha=0.38,
            label=r"local ANN/RBF/GPR chart 1",
        ),
        Patch(
            facecolor="palegreen",
            edgecolor="seagreen",
            alpha=0.38,
            label=r"local ANN/RBF/GPR chart 2",
        ),
        Line2D(
            [],
            [],
            color="crimson",
            linewidth=3.0,
            label=(
                r"piecewise ANN/RBF/GPR closure"
                "\n"
                r"\hspace{1em}"
                r"$(\mathbf{u}_{\mathrm{ref}}^{(i)}"
                r"+\mathbf{V}^{(i)}\mathbf{q}^{(i)}"
                r"+\overline{\mathbf{V}}^{(i)}"
                r"\mathcal{N}_i(q_1,q_2))$"
            ),
        ),
    ]


def build_asset_figure(data, asset):
    limits = axis_limits(data)
    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    if asset == "local_linear":
        configure_axis(
            ax,
            rf"\textbf{{Local linear POD: {N_LINEAR_CHARTS} rank-2 charts}}",
            limits,
        )
        wire = add_true_wireframe(ax, data)
        fade_artists = [(wire, 0.18)]
        for patch in data.local_linear_patches:
            surface = ax.plot_surface(
                *patch,
                color="lightsalmon",
                edgecolor="sienna",
                linewidth=0.25,
                alpha=0.0,
                shade=True,
            )
            fade_artists.append((surface, 0.30))
        approximation = data.local_linear_curve
        approximation_color = "darkorange"
        approximation_style = "--"
        approximation_width = 2.8
        legend_handles = local_linear_legend()
        legend_fontsize = 9.4
        error = data.local_linear_error

    elif asset == "global_ann_rbf_gpr":
        configure_axis(ax, r"\textbf{Global ANN/RBF/GPR closure}", limits)
        wire = add_true_wireframe(ax, data)
        surface = ax.plot_surface(
            *data.global_gpr_surface,
            color="plum",
            edgecolor="mediumvioletred",
            linewidth=0.16,
            alpha=0.0,
            shade=True,
        )
        training_points = ax.scatter(
            data.train_q[:, 0],
            data.train_q[:, 1],
            data.train_z,
            s=18,
            color="royalblue",
            alpha=0.0,
            depthshade=False,
        )
        fade_artists = [(wire, 0.16), (surface, 0.32), (training_points, 1.0)]
        approximation = data.global_gpr_curve
        approximation_color = "crimson"
        approximation_style = "--"
        approximation_width = 2.8
        legend_handles = global_ann_rbf_gpr_legend()
        legend_fontsize = 8.9
        error = data.global_gpr_error

    elif asset == "local_nonlinear":
        configure_axis(ax, r"\textbf{Two local ANN/RBF/GPR closure charts}", limits)
        wire = add_true_wireframe(ax, data)
        lower_surface = ax.plot_surface(
            *data.lower_surface,
            color="palegreen",
            edgecolor="seagreen",
            linewidth=0.16,
            alpha=0.0,
            shade=True,
        )
        upper_surface = ax.plot_surface(
            *data.upper_surface,
            color="plum",
            edgecolor="mediumorchid",
            linewidth=0.16,
            alpha=0.0,
            shade=True,
        )
        fade_artists = [
            (wire, 0.12),
            (lower_surface, 0.36),
            (upper_surface, 0.36),
        ]
        approximation = data.local_nonlinear_curve
        approximation_color = "crimson"
        approximation_style = "-"
        approximation_width = 3.2
        legend_handles = local_nonlinear_legend()
        legend_fontsize = 8.8
        error = data.local_nonlinear_error

    else:
        raise ValueError(f"unknown asset: {asset}")

    true_curve, = ax.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=2.8,
        color="black",
    )
    approximation_curve, = ax.plot(
        [],
        [],
        [],
        color=approximation_color,
        linestyle=approximation_style,
        linewidth=approximation_width,
    )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.93),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=True,
        framealpha=0.72,
        fontsize=legend_fontsize,
    )
    add_footer(fig, error)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)
    set_fade_alpha(fade_artists, 0.0)

    return fig, ax, {
        "true_curve": true_curve,
        "approximation_curve": approximation_curve,
        "approximation": approximation,
        "fade_artists": fade_artists,
    }


def reset_asset(ax, artists):
    artists["true_curve"].set_data([], [])
    artists["true_curve"].set_3d_properties([])
    artists["approximation_curve"].set_data([], [])
    artists["approximation_curve"].set_3d_properties([])
    set_fade_alpha(artists["fade_artists"], 0.0)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    return [
        artists["true_curve"],
        artists["approximation_curve"],
        *[artist for artist, _ in artists["fade_artists"]],
    ]


def set_final_asset_state(data, artists):
    set_curve(artists["true_curve"], data.u, len(data.u))
    set_curve(
        artists["approximation_curve"],
        artists["approximation"],
        len(artists["approximation"]),
    )
    set_fade_alpha(artists["fade_artists"], 1.0)


def save_asset_png(data, asset, path):
    fig, _, artists = build_asset_figure(data, asset)
    set_final_asset_state(data, artists)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_asset_gif(
    data,
    asset,
    path,
    *,
    full_trajectory_at_start=False,
    rotation_laps=1,
):
    fig, ax, artists = build_asset_figure(data, asset)

    curve_steps = np.unique(np.linspace(1, len(data.u), 120, dtype=int))
    n_true = 0 if full_trajectory_at_start else len(curve_steps)
    n_fade = 18
    n_reconstruction = len(curve_steps)
    n_hold = 24
    n_rotation = 144 * rotation_laps
    fade_start = n_true
    reconstruction_start = fade_start + n_fade
    hold_start = reconstruction_start + n_reconstruction
    rotation_start = hold_start + n_hold
    total_frames = rotation_start + n_rotation

    def init():
        return reset_asset(ax, artists)

    def update(frame):
        if frame < n_true:
            set_curve(artists["true_curve"], data.u, curve_steps[frame])
        else:
            set_curve(artists["true_curve"], data.u, len(data.u))

        if fade_start <= frame < reconstruction_start:
            alpha = (frame - fade_start + 1) / n_fade
            set_fade_alpha(artists["fade_artists"], alpha)
        elif frame >= reconstruction_start:
            set_fade_alpha(artists["fade_artists"], 1.0)

        if reconstruction_start <= frame < hold_start:
            local_frame = frame - reconstruction_start
            set_curve(
                artists["approximation_curve"],
                artists["approximation"],
                curve_steps[local_frame],
            )
        elif frame >= hold_start:
            set_curve(
                artists["approximation_curve"],
                artists["approximation"],
                len(artists["approximation"]),
            )

        if frame >= rotation_start:
            fraction = (frame - rotation_start) / max(1, n_rotation - 1)
            ax.view_init(
                elev=ELEVATION,
                azim=AZIMUTH + 360.0 * rotation_laps * fraction,
            )

        return [
            artists["true_curve"],
            artists["approximation_curve"],
            *[artist for artist, _ in artists["fade_artists"]],
        ]

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=1000.0 / 24.0,
        blit=False,
    )
    animation.save(path, writer=PillowWriter(fps=24), dpi=100)
    plt.close(fig)


def output_paths(output_dir):
    return {
        "local_linear": (
            output_dir / "local_prom_ann_two_bases_local_linear.png",
            output_dir / "local_prom_ann_two_bases_local_linear.gif",
        ),
        "global_ann_rbf_gpr": (
            output_dir / "local_prom_ann_two_bases_global_ann_rbf_gpr.png",
            output_dir / "local_prom_ann_two_bases_global_ann_rbf_gpr.gif",
        ),
        "local_nonlinear": (
            output_dir / "local_prom_ann_two_bases_local_nonlinear.png",
            output_dir / "local_prom_ann_two_bases_local_nonlinear.gif",
        ),
    }


def full_trajectory_two_laps_output_paths(output_dir):
    return {
        "local_linear": output_dir
        / "local_prom_ann_two_bases_local_linear_full_trajectory_two_laps.gif",
        "global_ann_rbf_gpr": output_dir
        / "local_prom_ann_two_bases_global_ann_rbf_gpr_full_trajectory_two_laps.gif",
        "local_nonlinear": output_dir
        / "local_prom_ann_two_bases_local_nonlinear_full_trajectory_two_laps.gif",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render separate local-linear, global-GPR, and local-nonlinear assets."
    )
    parser.add_argument(
        "--png-only",
        "--preview-only",
        dest="png_only",
        action="store_true",
        help="write the three PNG files without rendering GIFs",
    )
    parser.add_argument(
        "--full-trajectory-two-laps",
        action="store_true",
        help=(
            "write parallel GIFs with the full trajectory visible from the first "
            "frame and two final camera laps"
        ),
    )
    args = parser.parse_args()

    if args.png_only and args.full_trajectory_two_laps:
        parser.error("--png-only cannot be combined with --full-trajectory-two-laps")

    output_dir = HERE / "outputs"
    output_dir.mkdir(exist_ok=True)
    data = build_manifold_data()
    paths = output_paths(output_dir)

    if args.full_trajectory_two_laps:
        variant_paths = full_trajectory_two_laps_output_paths(output_dir)
        for asset in ASSETS:
            gif_path = variant_paths[asset]
            save_asset_gif(
                data,
                asset,
                gif_path,
                full_trajectory_at_start=True,
                rotation_laps=2,
            )
            print(gif_path)
        return

    for asset in ASSETS:
        png_path, gif_path = paths[asset]
        save_asset_png(data, asset, png_path)
        print(png_path)
        if not args.png_only:
            save_asset_gif(data, asset, gif_path)
            print(gif_path)


if __name__ == "__main__":
    main()
