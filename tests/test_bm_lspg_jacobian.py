"""Regression checks for the mode index in the BM-LSPG Jacobian."""

import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from burgers.core import (
    EcswJacobianAssembler,
    get_ops,
    inviscid_burgers_exact_jac2D,
    inviscid_burgers_res2D,
    inviscid_burgers_res2D_ecsw,
    make_2D_grid,
)
from burgers.ecsw_utils import generate_augmented_mesh
from burgers.gauss_newton import newton_BM_ECSW_2D


class BmLspgJacobianTests(unittest.TestCase):
    def solve_linear(self, weights, rhs):
        A = np.array([[1., 2.], [2., -1.], [.5, 1.5], [-1., .25]])
        basis = np.eye(4, 2)
        jacobian = np.column_stack((A, np.zeros((4, 2))))
        zero = sp.csr_matrix((2, 2))
        return newton_BM_ECSW_2D(
            func=lambda w: A @ w[:2] - rhs,
            jac=lambda w: jacobian,
            basis=basis,
            y0=np.zeros(2),
            sample_weights=weights,
            JDx=zero,
            JDy=zero,
            dt=.05,
            max_its=1,
            relnorm_cutoff=1e-12,
        )

    def test_distinct_mode_weights_solve_linear_system_in_one_update(self):
        A = np.array([[1., 2.], [2., -1.], [.5, 1.5], [-1., .25]])
        expected = np.array([.7, -1.2])
        result = self.solve_linear(
            weights=np.array([[2., .5], [.25, 3.]]),
            rhs=A @ expected,
        )
        np.testing.assert_allclose(result[0], expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(result[3])
        self.assertLess(result[4], 1e-12)

    def test_shared_weights_agree_with_square_root_weighted_least_squares(self):
        A = np.array([[1., 2.], [2., -1.], [.5, 1.5], [-1., .25]])
        rhs = np.array([2., -.3, 1.7, 4.])
        cell_weights = np.array([2., 3.])
        sqrt_rows = np.sqrt(np.tile(cell_weights, 2))
        expected = np.linalg.lstsq(
            sqrt_rows[:, None] * A, sqrt_rows * rhs, rcond=None
        )[0]
        result = self.solve_linear(
            np.repeat(cell_weights[:, None], 2, axis=1), rhs
        )
        np.testing.assert_allclose(result[0], expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(result[3])

    def test_sampled_burgers_jacobian_matches_full_equation_finite_differences(self):
        rng = np.random.default_rng(729)
        grid_x, grid_y = make_2D_grid(0., 4., 0., 4., 4, 4)
        Dx, Dy, JDx, JDy, eye = get_ops(grid_x, grid_y)
        n_cells, n_modes, dt = 16, 3, .2
        basis = np.linalg.qr(rng.normal(size=(2 * n_cells, n_modes)))[0]
        reference = 1.5 + rng.uniform(0., .3, 2 * n_cells)
        y0 = np.array([.4, -.3, .2])
        previous = reference + basis @ np.array([-.1, .2, -.4])
        mu = [1.7, .02]
        sample = np.array([0, 2, 5, 7, 10])
        augmented = generate_augmented_mesh(grid_x, grid_y, sample)
        full_rows = np.concatenate((sample, n_cells + sample))
        full_cols = np.concatenate((augmented, n_cells + augmented))
        weights = rng.uniform(.1, 3., (sample.size, n_modes))
        dx_sample = JDx.tocsr()[sample, :][:, augmented]
        dy_sample = JDy.tocsr()[sample, :][:, augmented]
        assembler = EcswJacobianAssembler(
            dt, dx_sample, dy_sample, eye[full_rows, :][:, full_cols], augmented
        )

        def sampled_residual(w):
            return inviscid_burgers_res2D_ecsw(
                w, grid_x, grid_y, dt, previous[full_cols], mu,
                dx_sample, dy_sample, sample, augmented,
            )

        # Build q independently on the full mesh, then sample its entities.
        def full_projected_equation(y, previous_state=previous):
            w = reference + basis @ y
            residual = inviscid_burgers_res2D(
                w, grid_x, grid_y, dt, previous_state, mu, Dx, Dy
            )
            tangent = inviscid_burgers_exact_jac2D(w, dt, JDx, JDy, eye) @ basis
            q = np.zeros(n_modes)
            for i in range(n_modes):
                for row, cell in enumerate(sample):
                    q[i] += weights[row, i] * (
                        tangent[cell, i] * residual[cell]
                        + tangent[n_cells + cell, i] * residual[n_cells + cell]
                    )
            return q

        captured = []
        original_solve = np.linalg.solve

        def record_newton_matrix(matrix, rhs):
            captured.append((matrix.copy(), rhs.copy()))
            return original_solve(matrix, rhs)

        with mock.patch("burgers.gauss_newton.np.linalg.solve", side_effect=record_newton_matrix):
            newton_BM_ECSW_2D(
                sampled_residual, assembler, basis[full_cols, :], y0,
                weights, dx_sample, dy_sample, dt,
                u_ref=reference[full_cols], max_its=1,
            )
        self.assertTrue(captured)
        matrix, rhs = captured[0]
        np.testing.assert_allclose(-rhs, full_projected_equation(y0), atol=1e-12)
        eps = 1e-4
        numerical = np.column_stack([
            (
                full_projected_equation(y0 + eps * d)
                - full_projected_equation(y0 - eps * d)
            ) / (2 * eps)
            for d in np.eye(n_modes)
        ])
        self.assertGreater(np.linalg.norm(numerical - numerical.T), 1e-3)
        np.testing.assert_allclose(matrix, numerical, rtol=1e-7, atol=1e-8)

        # q also depends on the previous state through R. For trapezoidal
        # Burgers, dR/dw_previous = J(w_previous) - 2 I.
        current_tangent = inviscid_burgers_exact_jac2D(
            reference + basis @ y0, dt, JDx, JDy, eye
        ) @ basis
        previous_tangent = inviscid_burgers_exact_jac2D(
            previous, dt, JDx, JDy, eye
        ) @ basis - 2 * basis
        Au, Av = np.split(current_tangent[full_rows], 2)
        Pu, Pv = np.split(previous_tangent[full_rows], 2)
        previous_derivative = (weights * Au).T @ Pu + (weights * Av).T @ Pv
        numerical_previous = np.column_stack([
            (full_projected_equation(y0, previous + eps * basis @ d)
             - full_projected_equation(y0, previous - eps * basis @ d)) / (2 * eps)
            for d in np.eye(n_modes)
        ])
        np.testing.assert_allclose(previous_derivative, numerical_previous,
                                   rtol=1e-7, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
