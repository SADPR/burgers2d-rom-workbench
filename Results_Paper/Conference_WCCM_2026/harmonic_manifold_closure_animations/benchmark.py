from dataclasses import dataclass

import numpy as np


A_QUADRATIC = 0.40
EPSILON_THIRD = 0.06


@dataclass(frozen=True)
class BenchmarkData:
    t: np.ndarray
    harmonic_q: np.ndarray
    q: np.ndarray
    u_ref: np.ndarray
    V: np.ndarray
    V_bar: np.ndarray
    nonlinear_coordinate: np.ndarray
    closure_coordinate: np.ndarray
    u: np.ndarray
    u_linear: np.ndarray
    u_quadratic: np.ndarray
    H: np.ndarray


def quadratic_features(q):
    q = np.asarray(q)
    return np.column_stack(
        (
            q[:, 0] ** 2,
            q[:, 0] * q[:, 1],
            q[:, 1] ** 2,
        )
    )


def build_benchmark(
    n_samples=241,
    quadratic_amplitude=A_QUADRATIC,
    epsilon=EPSILON_THIRD,
):
    t = np.linspace(0.0, 2.0 * np.pi, n_samples)
    harmonic_q = np.column_stack((np.cos(t), np.sin(t)))
    nonlinear_coordinate = (
        quadratic_amplitude
        * (harmonic_q[:, 0] ** 2 - harmonic_q[:, 1] ** 2)
        + epsilon * np.sin(3.0 * t)
    )
    u = np.column_stack(
        (
            np.cos(t),
            np.sin(t),
            nonlinear_coordinate,
        )
    )

    # Match the original animations: use the initial condition as affine
    # reference, then compute a rank-two POD basis from shifted snapshots.
    u_ref = u[0].copy()
    shifted_u = u - u_ref
    _, _, vt = np.linalg.svd(shifted_u, full_matrices=False)
    V = vt[:2].T
    V_bar = vt[2:3].T
    q = shifted_u @ V
    closure_coordinate = (shifted_u @ V_bar).ravel()
    u_linear = u_ref + q @ V.T

    features = quadratic_features(q)
    residual = u - u_linear
    coefficients, _, _, _ = np.linalg.lstsq(
        features, residual, rcond=None
    )
    H = coefficients.T
    u_quadratic = u_linear + features @ H.T

    return BenchmarkData(
        t=t,
        harmonic_q=harmonic_q,
        q=q,
        u_ref=u_ref,
        V=V,
        V_bar=V_bar,
        nonlinear_coordinate=nonlinear_coordinate,
        closure_coordinate=closure_coordinate,
        u=u,
        u_linear=u_linear,
        u_quadratic=u_quadratic,
        H=H,
    )


def latent_grid(data, n_points=55, padding=0.15):
    q_1 = np.linspace(
        data.q[:, 0].min() - padding,
        data.q[:, 0].max() + padding,
        n_points,
    )
    q_2 = np.linspace(
        data.q[:, 1].min() - padding,
        data.q[:, 1].max() + padding,
        n_points,
    )
    return np.meshgrid(q_1, q_2)


def lift_surface(data, q_1, q_2, closure_coordinate):
    grid_q = np.column_stack((q_1.ravel(), q_2.ravel()))
    surface = (
        data.u_ref
        + grid_q @ data.V.T
        + closure_coordinate.ravel()[:, np.newaxis] @ data.V_bar.T
    )
    return tuple(
        surface[:, component].reshape(q_1.shape)
        for component in range(3)
    )


def relative_error(reference, approximation, u_ref):
    numerator = np.linalg.norm(reference - approximation)
    denominator = np.linalg.norm(reference - u_ref)
    return numerator / denominator
