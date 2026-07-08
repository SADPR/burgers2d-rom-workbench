import numpy as np
from scipy.interpolate import RBFInterpolator

from animation import AnimationSpec, render_manifold_animation
from benchmark import (
    build_benchmark,
    latent_grid,
    lift_surface,
    relative_error,
)


data = build_benchmark()

# One RBF realization is used only to draw the generic learned closure. The
# presentation treats ANN, RBF, and GPR as interchangeable realizations of F.
closure_model = RBFInterpolator(
    data.q[:-1],
    data.closure_coordinate[:-1, np.newaxis],
    kernel="thin_plate_spline",
    smoothing=1.0e-10,
)
closure_prediction = closure_model(data.q).ravel()
u_nonlinear = (
    data.u_ref
    + data.q @ data.V.T
    + closure_prediction[:, np.newaxis] @ data.V_bar.T
)

q_1, q_2 = latent_grid(data)
grid_q = np.column_stack((q_1.ravel(), q_2.ravel()))
surface_coordinate = closure_model(grid_q).reshape(q_1.shape)
surface = lift_surface(data, q_1, q_2, surface_coordinate)

spec = AnimationSpec(
    title=r"\textbf{General nonlinear closure (ANN/RBF/GPR)}",
    output_name="general_ann_rbf_gpr_closure.gif",
    surface_color="orchid",
    curve_color="mediumvioletred",
    approximation_label=(
        r"nonlinear closure approximation"
        "\n"
        r"\hspace{1em}"
        r"$(\mathbf{u}_{\mathrm{ref}}+\mathbf{V}\mathbf{q}"
        r"+\overline{\mathbf{V}}\,"
        r"\mathcal{F}(\mathbf{q},\boldsymbol{\mu},t))$"
    ),
    manifold_label=r"nonlinear closure manifold",
    error=relative_error(data.u, u_nonlinear, data.u_ref),
    show_linear_mapping=True,
    curve_linestyle="--",
)

render_manifold_animation(data, surface, u_nonlinear, spec)
