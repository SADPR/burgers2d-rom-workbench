from dataclasses import dataclass

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from sklearn.kernel_ridge import KernelRidge

from animation import AZIMUTH, ELEVATION, HERE, plt
from benchmark import build_benchmark, relative_error


@dataclass(frozen=True)
class CaseSpec:
    case: int
    title: str
    map_symbol: str
    map_arguments: str
    output_name: str
    inset_title: str
    input_kind: str


def one_mode_split(data):
    shifted = data.u - data.u_ref
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    V = vt[:1].T
    V_bar = vt[1:].T
    q = shifted @ V
    q_bar = shifted @ V_bar
    u_linear = data.u_ref + q @ V.T
    return V, V_bar, q, q_bar, u_linear


def multiplicity_pair(q, q_bar):
    n_samples = len(q) - 1
    best = None
    for first in range(n_samples):
        for second in range(first + 20, n_samples):
            q_distance = abs(q[first, 0] - q[second, 0])
            if q_distance > 5.0e-3:
                continue
            bar_distance = np.linalg.norm(q_bar[first] - q_bar[second])
            if best is None or bar_distance > best[0]:
                best = (bar_distance, first, second)
    if best is None:
        raise RuntimeError("Could not identify a multiplicity pair.")
    return best[1], best[2]


def closure_inputs(spec, data, q):
    periodic_time = np.column_stack((np.cos(data.t), np.sin(data.t)))
    if spec.input_kind == "state":
        return q
    if spec.input_kind == "parameter_time":
        # The illustrative trajectory has a fixed parameter. The periodic
        # encoding represents the active time part of (mu, t).
        return periodic_time
    if spec.input_kind == "hybrid":
        return np.column_stack((q, periodic_time))
    raise ValueError(f"Unknown input kind: {spec.input_kind}")


def fit_closure(inputs, q_bar):
    model = KernelRidge(
        kernel="rbf",
        gamma=3.0,
        alpha=1.0e-8,
    )
    model.fit(inputs[:-1], q_bar[:-1])
    return model, model.predict(inputs)


def render_case(spec, data, V, V_bar, q, q_bar, u_linear):
    inputs = closure_inputs(spec, data, q)
    model, q_bar_prediction = fit_closure(inputs, q_bar)
    approximation = (
        data.u_ref
        + q @ V.T
        + q_bar_prediction @ V_bar.T
    )
    error = relative_error(data.u, approximation, data.u_ref)
    first, second = multiplicity_pair(q, q_bar)

    output_dir = HERE / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / spec.output_name

    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(spec.title, fontsize=19, pad=12)
    ax.set_xlabel(r"$u_1$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$u_2$", fontsize=16, labelpad=8)
    ax.set_zlabel(r"$u_3$", fontsize=16, labelpad=7)
    ax.tick_params(labelsize=10)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect((2.0, 1.8, 1.0))

    all_points = np.vstack((data.u, u_linear, approximation))
    ax.set_xlim(all_points[:, 0].min() - 0.12, all_points[:, 0].max() + 0.12)
    ax.set_ylim(all_points[:, 1].min() - 0.12, all_points[:, 1].max() + 0.12)
    ax.set_zlim(all_points[:, 2].min() - 0.12, all_points[:, 2].max() + 0.12)

    line_coordinate = np.linspace(
        q[:, 0].min() - 0.18,
        q[:, 0].max() + 0.18,
        100,
    )
    pod_line_points = data.u_ref + line_coordinate[:, np.newaxis] @ V.T
    ax.plot(
        pod_line_points[:, 0],
        pod_line_points[:, 1],
        pod_line_points[:, 2],
        color="dodgerblue",
        linewidth=4.0,
        alpha=0.65,
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
    linear_curve, = ax.plot(
        [],
        [],
        [],
        color="gray",
        linewidth=1.8,
    )
    closure_curve, = ax.plot(
        [],
        [],
        [],
        color="mediumvioletred",
        linestyle="--",
        linewidth=2.8,
    )
    lift, = ax.plot(
        [],
        [],
        [],
        color="crimson",
        linewidth=3.2,
        alpha=0.9,
    )

    common_q = 0.5 * (q[first, 0] + q[second, 0])
    common_projection = data.u_ref + common_q * V[:, 0]
    branch_colors = ("darkorange", "purple")
    ax.scatter(
        [data.u[first, 0], data.u[second, 0]],
        [data.u[first, 1], data.u[second, 1]],
        [data.u[first, 2], data.u[second, 2]],
        color=branch_colors,
        s=48,
        depthshade=False,
        zorder=8,
    )
    ax.scatter(
        [common_projection[0]],
        [common_projection[1]],
        [common_projection[2]],
        color="crimson",
        marker="x",
        s=70,
        linewidth=2.2,
        depthshade=False,
        zorder=9,
    )
    for index, color in zip((first, second), branch_colors):
        ax.plot(
            [common_projection[0], data.u[index, 0]],
            [common_projection[1], data.u[index, 1]],
            [common_projection[2], data.u[index, 2]],
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
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
            color="dodgerblue",
            linewidth=4.0,
            label=r"one-mode POD line $\mathbf{u}_{\mathrm{ref}}+\mathbf{V}q$",
        ),
        Line2D(
            [],
            [],
            color="mediumvioletred",
            linestyle="--",
            linewidth=2.8,
            label=(
                rf"Case {spec.case} ANN/RBF/GPR approximation"
                "\n"
                r"\hspace{1em}"
                rf"$(\mathbf{{u}}_{{\mathrm{{ref}}}}+\mathbf{{V}}q"
                rf"+\overline{{\mathbf{{V}}}}\,"
                rf"\mathcal{{{spec.map_symbol}}}({spec.map_arguments}))$"
            ),
        ),
        Line2D(
            [],
            [],
            color="crimson",
            marker="x",
            linestyle="none",
            markersize=8,
            label=r"same retained coordinate $q$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.93),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=True,
        framealpha=0.68,
        fontsize=9.5,
    )

    inset = fig.add_axes([0.09, 0.13, 0.31, 0.21])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.90))
    if spec.input_kind == "state":
        inset_x = q[:, 0]
        inset.set_xlabel(r"$q$", fontsize=10)
        inset.scatter(
            inset_x,
            q_bar[:, 0],
            color="black",
            s=7,
            alpha=0.55,
        )
        order = np.argsort(inset_x)
        prediction_order = order
    else:
        inset_x = data.t
        inset.set_xlabel(r"$t$ (fixed $\boldsymbol{\mu}$)", fontsize=10)
        inset.plot(
            inset_x,
            q_bar[:, 0],
            color="black",
            linewidth=1.2,
            alpha=0.55,
        )
        prediction_order = np.arange(len(inset_x))

    predicted_map, = inset.plot(
        [],
        [],
        color="mediumvioletred",
        linewidth=2.0,
    )
    inset.scatter(
        [inset_x[first], inset_x[second]],
        [q_bar[first, 0], q_bar[second, 0]],
        color=branch_colors,
        s=26,
        zorder=5,
    )
    inset.set_ylabel(r"$\bar q_1$", fontsize=10)
    inset.set_title(spec.inset_title, fontsize=10)
    inset.tick_params(labelsize=8)

    fig.text(
        0.5,
        0.025,
        (
            rf"$\dim(\mathbf{{V}})=1,\ "
            rf"\dim(\overline{{\mathbf{{V}}}})=2$"
            rf"\qquad relative reconstruction error: "
            rf"${100.0 * error:.2f}\%$"
        ),
        ha="center",
        fontsize=12.5,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)

    curve_steps = np.unique(
        np.linspace(1, len(data.u), 120, dtype=int)
    )
    n_trajectory = len(curve_steps)
    n_linear = len(curve_steps)
    n_closure = len(curve_steps)
    n_hold = 24
    n_spin = 144
    linear_start = n_trajectory
    closure_start = linear_start + n_linear
    hold_start = closure_start + n_closure
    spin_start = hold_start + n_hold
    total_frames = spin_start + n_spin

    def update(frame):
        if frame < n_trajectory:
            end = curve_steps[frame]
            trajectory.set_data(data.u[:end, 0], data.u[:end, 1])
            trajectory.set_3d_properties(data.u[:end, 2])
        else:
            trajectory.set_data(data.u[:, 0], data.u[:, 1])
            trajectory.set_3d_properties(data.u[:, 2])

        if linear_start <= frame < closure_start:
            end = curve_steps[frame - linear_start]
            linear_curve.set_data(
                u_linear[:end, 0], u_linear[:end, 1]
            )
            linear_curve.set_3d_properties(u_linear[:end, 2])
        elif frame >= closure_start:
            linear_curve.set_data(u_linear[:, 0], u_linear[:, 1])
            linear_curve.set_3d_properties(u_linear[:, 2])

        if closure_start <= frame < hold_start:
            end = curve_steps[frame - closure_start]
            closure_curve.set_data(
                approximation[:end, 0], approximation[:end, 1]
            )
            closure_curve.set_3d_properties(approximation[:end, 2])

            visible = prediction_order[:end]
            predicted_map.set_data(
                inset_x[visible],
                q_bar_prediction[visible, 0],
            )

            index = end - 1
            lift.set_data(
                [u_linear[index, 0], approximation[index, 0]],
                [u_linear[index, 1], approximation[index, 1]],
            )
            lift.set_3d_properties(
                [u_linear[index, 2], approximation[index, 2]]
            )
        elif frame >= hold_start:
            closure_curve.set_data(
                approximation[:, 0], approximation[:, 1]
            )
            closure_curve.set_3d_properties(approximation[:, 2])
            predicted_map.set_data(
                inset_x[prediction_order],
                q_bar_prediction[prediction_order, 0],
            )

        if frame >= spin_start:
            fraction = (frame - spin_start) / max(1, n_spin - 1)
            ax.view_init(
                elev=ELEVATION,
                azim=AZIMUTH + 360.0 * fraction,
            )

        return trajectory, linear_curve, closure_curve, lift, predicted_map

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


CASES = {
    1: CaseSpec(
            case=1,
            title=(
                r"\textbf{Case 1: state-conditioned closure "
                r"(ANN/RBF/GPR)}"
            ),
            map_symbol="N",
            map_arguments=r"q",
            output_name="case1_ann_rbf_gpr_state.gif",
            inset_title=r"non-injective map: same $q$, different $\bar q_1$",
            input_kind="state",
        ),
    2: CaseSpec(
            case=2,
            title=(
                r"\textbf{Case 2: parameter--time closure "
                r"(ANN/RBF/GPR)}"
            ),
            map_symbol="M",
            map_arguments=r"\boldsymbol{\mu},t",
            output_name="case2_ann_rbf_gpr_parameter_time.gif",
            inset_title=r"$(\boldsymbol{\mu},t)$ identifies the branch",
            input_kind="parameter_time",
        ),
    3: CaseSpec(
            case=3,
            title=(
                r"\textbf{Case 3: hybrid closure "
                r"(ANN/RBF/GPR)}"
            ),
            map_symbol="H",
            map_arguments=r"q,\boldsymbol{\mu},t",
            output_name="case3_ann_rbf_gpr_hybrid.gif",
            inset_title=r"hybrid input: state plus branch information",
            input_kind="hybrid",
        ),
}


def render_case_number(case_number):
    data = build_benchmark()
    V, V_bar, q, q_bar, u_linear = one_mode_split(data)
    render_case(
        CASES[case_number],
        data,
        V,
        V_bar,
        q,
        q_bar,
        u_linear,
    )


def main():
    for case_number in CASES:
        render_case_number(case_number)


if __name__ == "__main__":
    main()
