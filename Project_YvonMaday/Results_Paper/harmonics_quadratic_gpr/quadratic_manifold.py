import numpy as np

from animation import AnimationSpec, render_manifold_animation
from benchmark import (
    build_benchmark,
    latent_grid,
    relative_error,
)


data = build_benchmark()
q_1, q_2 = latent_grid(data)
surface_features = np.stack(
    (q_1**2, q_1 * q_2, q_2**2),
    axis=-1,
)
grid_q = np.column_stack((q_1.ravel(), q_2.ravel()))
quadratic_surface = (
    data.u_ref
    + grid_q @ data.V.T
    + surface_features.reshape(-1, 3) @ data.H.T
).reshape(q_1.shape + (3,))
surface = (
    quadratic_surface[:, :, 0],
    quadratic_surface[:, :, 1],
    quadratic_surface[:, :, 2],
)

spec = AnimationSpec(
    title=r"\textbf{Quadratic manifold}",
    output_name="quadratic_manifold.gif",
    surface_color="plum",
    curve_color="mediumvioletred",
    approximation_label=(
        r"quadratic approximation"
        "\n"
        r"\hspace{1em}"
        r"$(\mathbf{u}_{\mathrm{ref}}+\mathbf{V}\mathbf{q}"
        r"+\mathbf{H}\mathbf{h}_2(\mathbf{q}))$"
    ),
    manifold_label=r"quadratic manifold",
    error=relative_error(data.u, data.u_quadratic, data.u_ref),
    show_linear_mapping=True,
    curve_linestyle="--",
)

render_manifold_animation(data, surface, data.u_quadratic, spec)
