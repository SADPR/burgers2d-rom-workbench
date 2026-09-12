"""Residual-adaptive tail spaces for Euclidean Case-2 LSPG.

The space is frozen within each time step. This is not the derivative of a
state-dependent decoder, and it must not be updated inside its Newton solve.
"""

import numpy as np
from scipy.linalg import qr


def residual_krylov_space(jacobian, primary, secondary, residual, rank):
    """Return B spanning K_rank(C.T C, C.T r), without assembling J @ Vs.

    A = J Vp, P = I - Q Q.T, C = P J Vs, where Q spans range(A).
    Eliminating the primary increment leaves min_b ||P r + C b||_2.
    A small Krylov space restricts that secondary correction, rather than
    selecting a fixed subset of POD coordinates. The coefficient metric is
    Euclidean. For cubature, J and r must both carry sqrt(weights); Vp/Vs
    are then stencil restrictions of the original basis, without renormalizing
    them. Full-state POD orthonormality is checked by the experiment runner.
    """
    if rank < 0 or rank > secondary.shape[1]:
        raise ValueError("rank must be between zero and the tail dimension")
    if rank == 0:
        return np.empty((secondary.shape[1], 0))
    a = jacobian @ primary
    q, r = qr(a, mode="economic", check_finite=False)
    if np.linalg.matrix_rank(r) < a.shape[1]:
        raise ValueError("The primary Jacobian is rank deficient")

    def project(v):
        return v - q @ (q.T @ v)

    def adjoint(v):
        return secondary.T @ (jacobian.T @ v)

    vector = adjoint(project(residual))
    initial = np.linalg.norm(vector)
    if initial <= 1e-14 * max(1.0, np.linalg.norm(residual)):
        return np.empty((secondary.shape[1], 0))
    columns = []
    for _ in range(rank):
        raw_norm = np.linalg.norm(vector)
        if columns:
            b = np.column_stack(columns)
            # Twice-reorthogonalized Arnoldi avoids spurious directions near
            # Krylov breakdown (important when J is close to the identity).
            for _ in range(2):
                vector -= b @ (b.T @ vector)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12 * max(raw_norm, np.finfo(float).tiny):
            break
        columns.append(vector / norm)
        if len(columns) < rank:
            vector = adjoint(project(jacobian @ (secondary @ columns[-1])))
    return np.column_stack(columns) if columns else np.empty((secondary.shape[1], 0))


def solve_affine_step(offset, tangent, initial, residual, jacobian,
                      max_its=20, min_delta=1e-2, relnorm_cutoff=1e-5):
    """Original Case-2 Gauss-Newton stopping rule on a frozen affine space."""
    z = initial.copy()
    state = offset + tangent @ z
    previous_norm = None
    first_norm = None
    for iteration in range(max_its):
        f = residual(state)
        norm = np.linalg.norm(f)
        if not np.isfinite(norm):
            raise FloatingPointError("Non-finite residual")
        if first_norm is None:
            first_norm = norm + 1e-30
        if norm / first_norm < relnorm_cutoff:
            break
        if previous_norm is not None:
            if abs(previous_norm - norm) / (previous_norm + 1e-30) < min_delta:
                break
        previous_norm = norm
        update = np.linalg.lstsq(jacobian(state) @ tangent, -f, rcond=None)[0]
        z += update
        state = offset + tangent @ z
    return z, state, iteration + 1, float(np.linalg.norm(residual(state)))
