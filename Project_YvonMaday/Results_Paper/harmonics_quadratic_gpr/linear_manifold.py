import numpy as np

from animation import AnimationSpec, render_manifold_animation
from benchmark import (
    build_benchmark,
    latent_grid,
    lift_surface,
    relative_error,
)


data = build_benchmark()
q_1, q_2 = latent_grid(data)
surface = lift_surface(data, q_1, q_2, np.zeros_like(q_1))

spec = AnimationSpec(
    title=r"\textbf{Linear manifold}",
    output_name="linear_manifold.gif",
    surface_color="lightsteelblue",
    curve_color="dodgerblue",
    approximation_label=(
        r"linear approximation"
        "\n"
        r"\hspace{1em}"
        r"$(\mathbf{u}_{\mathrm{ref}}+\mathbf{V}\mathbf{q})$"
    ),
    manifold_label=r"linear manifold",
    error=relative_error(data.u, data.u_linear, data.u_ref),
)

render_manifold_animation(data, surface, data.u_linear, spec)
