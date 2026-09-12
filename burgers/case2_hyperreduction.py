"""Positive cubature for the residual-adaptive, affine Case-2 solve.

The rows of the residual and Jacobian carry sqrt(xi), including during
construction of the Krylov space. Only stencil states are reconstructed.
"""

import numpy as np
from scipy import sparse

from burgers.core import (
    get_ops, inviscid_burgers_res2D_ecsw, inviscid_burgers_exact_jac2D_ecsw,
)
from burgers.ecsw_utils import generate_augmented_mesh
from burgers.case2_residual_correction import residual_krylov_space, solve_affine_step


class SampledBurgers:
    """An exact restriction of the full discrete residual, with positive weights."""

    def __init__(self, grid_x, grid_y, dt, weights):
        self.grid_x, self.grid_y, self.dt = grid_x, grid_y, dt
        self.ncells = (len(grid_x) - 1) * (len(grid_y) - 1)
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (self.ncells,) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError("Expected finite nonnegative cell weights with positive support")
        self.samples = np.flatnonzero(weights > 0)
        self.augmented = generate_augmented_mesh(grid_x, grid_y, self.samples)
        self.state_indices = np.r_[self.augmented, self.ncells + self.augmented]
        self.residual_indices = np.r_[self.samples, self.ncells + self.samples]
        self.sqrt_weights = np.sqrt(np.tile(weights[self.samples], 2))
        self.overlap = np.isin(self.augmented, self.samples)
        _, _, dx, dy, _ = get_ops(grid_x, grid_y)
        self.dx = dx.tocsr()[self.samples][:, self.augmented]
        self.dy = dy.tocsr()[self.samples][:, self.augmented]
        identity = sparse.eye(self.ncells, format="csr")[self.samples][:, self.augmented]
        self.identity = sparse.block_diag((identity, identity), format="csr")

    def step_operators(self, previous, mu):
        dx = np.diff(self.grid_x)
        cols = self.samples % dx.size
        centers = .5 * (self.grid_x[1:] + self.grid_x[:-1])
        lbc = np.where(cols == 0, .5 * self.dt * mu[0]**2 / dx[0], 0.)
        source = self.dt * .02 * np.exp(mu[1] * centers[cols])

        def residual(state):
            raw = inviscid_burgers_res2D_ecsw(
                state, self.grid_x, self.grid_y, self.dt, previous, mu,
                self.dx, self.dy, self.samples, self.augmented,
                lbc=lbc, src=source, overlap=self.overlap)
            return self.sqrt_weights * raw

        def jacobian(state):
            raw = inviscid_burgers_exact_jac2D_ecsw(
                state, self.dt, self.dx, self.dy, self.identity,
                self.samples, self.augmented)
            return raw.multiply(self.sqrt_weights[:, None]).tocsr()

        return residual, jacobian


def sampled_affine_step(mesh, primary, secondary, offset, previous, mu,
                        previous_primary, rank, **solver_options):
    """Freeze B_r at the predictor, then solve the single weighted objective.

    primary/secondary are restrictions of the full Euclidean POD basis. They
    must NOT be reorthonormalized on the stencil: that would change the meaning
    of the coefficients and the Euclidean secondary-coordinate Krylov metric.
    """
    residual, jacobian = mesh.step_operators(previous, mu)
    candidate = offset + primary @ previous_primary
    b = residual_krylov_space(jacobian(candidate), primary, secondary,
                             residual(candidate), rank)
    tangent = np.asfortranarray(np.column_stack((primary, secondary @ b)))
    initial = np.r_[previous_primary, np.zeros(b.shape[1])]
    z, state, iterations, norm = solve_affine_step(
        offset, tangent, initial, residual, jacobian, **solver_options)
    return z, b, state, iterations, norm


def cell_moments(jv, residual, tangent_coordinates):
    """Entity contributions to V.T J.T r, T.T J.T J T, and ||r||^2.

    Both velocity residuals in a cell share one weight. The complete gradient
    keeps information about all secondary coordinates; the Gram block trains
    the primary and full-residual adaptive directions at the training anchor.
    Each block is scaled by its Frobenius norm before compression.
    """
    ncells = residual.size // 2
    gradient = (jv[:ncells] * residual[:ncells, None]
                + jv[ncells:] * residual[ncells:, None])
    jt = jv @ tangent_coordinates
    i, j = np.triu_indices(jt.shape[1])
    gram = jt[:ncells, i] * jt[:ncells, j] + jt[ncells:, i] * jt[ncells:, j]
    # sqrt(2) gives upper-triangular coordinates the symmetric Frobenius norm.
    gram[:, i != j] *= np.sqrt(2.)
    energy = (residual[:ncells]**2 + residual[ncells:]**2)[:, None]
    return [block / max(np.linalg.norm(block), np.finfo(float).tiny)
            for block in (gradient, gram, energy)]
