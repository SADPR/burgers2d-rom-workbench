#!/usr/bin/env python3
"""Focused regression tests for nonlinear ECM objective consistency."""

import unittest
from unittest import mock

import numpy as np
import torch

from burgers.gauss_newton import (
    _sqrt_ecm_weights,
    gauss_newton_pod_ann_ecsw,
    gauss_newton_pod_gp_ecsw,
    gauss_newton_pod_rbf_ecsw,
    gauss_newton_poddl_ecsw,
)
import burgers.pod_ann_manifold as pod_ann
import burgers.pod_dl_manifold as pod_dl
import burgers.pod_gplvm_manifold as pod_gplvm
import burgers.pod_gpr_manifold as pod_gpr
import burgers.pod_rbf_manifold as pod_rbf
import burgers.quadratic_manifold as pod_qm


def _unit_jacobian(*_args):
    return np.eye(4, dtype=np.float64)


def _recording_residual(record):
    def residual(w, _grid_x, _grid_y, _dt, wp, *_args):
        record.append(np.asarray(wp, dtype=np.float64).copy())
        return np.asarray(w, dtype=np.float64) - np.asarray(wp, dtype=np.float64)

    return residual


class _Case2Tail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x.reshape(-1)[-1:] + 0.0 * self.anchor


class _IdentityLatent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode(self, q):
        return q.reshape(-1)[:1] + 0.0 * self.anchor

    def decode_from_latent(self, z):
        return z.reshape(-1)[:1] + 0.0 * self.anchor


class EcmWeightingTests(unittest.TestCase):
    def _expected_weighted_solution(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, -1.0, 5.0, 2.0])
        xi = np.array([1.0, 9.0, 1.0, 9.0])
        return a, b, xi, float(np.dot(xi * a, b) / np.dot(xi * a, a))

    def test_ann_ecm_uses_square_root_weights(self):
        a, b, _, expected = self._expected_weighted_solution()

        def residual(w):
            return a * float(np.asarray(w).reshape(-1)[0]) - b

        y, _, _ = gauss_newton_pod_ann_ecsw(
            func=residual,
            jac=lambda _w: a[:, None],
            y0=torch.tensor([0.0]),
            decode=lambda y, with_grad=False: y.reshape(1),
            jacfwdfunc=lambda _y: torch.ones((1, 1)),
            sample_inds=np.array([0, 1]),
            augmented_sample=np.array([0, 1]),
            weight=np.array([1.0, 9.0]),
            max_its=3,
            relnorm_cutoff=1e-12,
            min_delta=0.0,
        )
        self.assertAlmostEqual(float(y.item()), expected, places=6)

    def test_podae_ecm_uses_square_root_weights(self):
        a, b, _, expected = self._expected_weighted_solution()

        def residual(w):
            return a * float(np.asarray(w).reshape(-1)[0]) - b

        z, _, _ = gauss_newton_poddl_ecsw(
            func=residual,
            jac=lambda _w: a[:, None],
            z0=torch.tensor([0.0]),
            decode=lambda z, with_grad=False: z.reshape(1),
            jac_u_z=lambda _z: torch.ones((1, 1)),
            sample_inds=np.array([0, 1]),
            augmented_sample=np.array([0, 1]),
            weight=np.array([1.0, 9.0]),
            max_its=3,
            relnorm_cutoff=1e-12,
            min_delta=0.0,
        )
        self.assertAlmostEqual(float(z.item()), expected, places=6)

    def test_rbf_ecm_uses_square_root_weights(self):
        a, b, _, expected = self._expected_weighted_solution()

        def residual(y):
            return a * float(np.asarray(y).reshape(-1)[0]) - b

        y, _, _ = gauss_newton_pod_rbf_ecsw(
            func=residual,
            jac=lambda _y: a[:, None],
            y0=np.array([0.0]),
            decode_rbf=lambda y: np.asarray(y).reshape(1),
            jac_rbf=lambda _y: np.ones((1, 1)),
            sample_inds=np.array([0, 1]),
            augmented_sample=np.array([0, 1]),
            weights=np.array([1.0, 9.0]),
            max_its=3,
            relnorm_cutoff=1e-12,
            min_delta=0.0,
            linear_solver="lstsq",
        )
        self.assertAlmostEqual(float(y[0]), expected, places=6)

    def test_gp_ecm_uses_square_root_weights(self):
        a, b, _, expected = self._expected_weighted_solution()

        def residual(y):
            return a * float(np.asarray(y).reshape(-1)[0]) - b

        y, _, _ = gauss_newton_pod_gp_ecsw(
            func=residual,
            jac=lambda _y: a[:, None],
            y0=np.array([0.0]),
            decode_gp=lambda y: np.asarray(y).reshape(1),
            jac_gp=lambda _y: np.ones((1, 1)),
            sample_inds=np.array([0, 1]),
            augmented_sample=np.array([0, 1]),
            weights=np.array([1.0, 9.0]),
            max_its=3,
            relnorm_cutoff=1e-12,
            min_delta=0.0,
            linear_solver="lstsq",
        )
        self.assertAlmostEqual(float(y[0]), expected, places=6)

    def test_ecm_weights_must_be_nonnegative(self):
        with self.assertRaises(ValueError):
            _sqrt_ecm_weights([1.0, -1.0])


class EcmPredecessorProjectionTests(unittest.TestCase):
    def setUp(self):
        self.ops = (None, None, None, None, None)
        self.basis_one = np.array([[1.0], [0.0], [0.0], [0.0]])
        self.basis_two = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        )
        self.basis_tail = np.array([[0.0], [1.0], [0.0], [0.0]])

    def test_case1_projects_the_predecessor(self):
        record = []

        def approx(y):
            return torch.tensor(self.basis_two, dtype=y.dtype, device=y.device) @ y

        with mock.patch.object(pod_ann, "get_ops", return_value=self.ops):
            pod_ann.compute_ECSW_training_matrix_2D_pod_ann(
                snaps=np.array([[3.0], [4.0], [7.0], [8.0]]),
                prev_snaps=np.array([[5.0], [6.0], [70.0], [80.0]]),
                basis=self.basis_two,
                approx=approx,
                jacfwdfunc=lambda _y: torch.tensor(self.basis_two, dtype=torch.float32),
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
            )

        np.testing.assert_allclose(record[0], [5.0, 6.0, 0.0, 0.0])

    def test_case2_projects_the_immediate_predecessor_at_previous_time(self):
        record = []
        model = _Case2Tail()

        with mock.patch.object(pod_ann, "get_ops", return_value=self.ops):
            pod_ann.compute_ECSW_training_matrix_2D_pod_ann_case2(
                snaps=np.array([[3.0], [2.0], [4.0], [5.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                t_samples=np.array([2.0]),
                basis=self.basis_one,
                basis2=self.basis_tail,
                ann_model=model,
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
            )

        np.testing.assert_allclose(record[0], [5.0, 1.5, 0.0, 0.0])

    def test_case3_projects_the_immediate_predecessor_at_previous_time(self):
        record = []
        model = _Case2Tail()

        with mock.patch.object(pod_ann, "get_ops", return_value=self.ops):
            pod_ann.compute_ECSW_training_matrix_2D_pod_ann_case3(
                snaps=np.array([[3.0], [2.0], [4.0], [5.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                t_samples=np.array([2.0]),
                basis=self.basis_one,
                basis2=self.basis_tail,
                ann_model=model,
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
                projection_max_its=4,
            )

        np.testing.assert_allclose(record[0], [5.0, 1.5, 0.0, 0.0], atol=1e-6)

    def test_podae_projects_the_predecessor(self):
        record = []
        model = _IdentityLatent()

        with mock.patch.object(pod_dl, "get_ops", return_value=self.ops):
            pod_dl.compute_ECSW_training_matrix_2D_pod_dl(
                snaps=np.array([[3.0], [4.0], [5.0], [6.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                basis=self.basis_one,
                pod_dl_model=model,
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
            )

        np.testing.assert_allclose(record[0], [5.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_rbf_projects_the_predecessor(self):
        record = []

        with (
            mock.patch.object(pod_rbf, "get_ops", return_value=self.ops),
            mock.patch.object(
                pod_rbf,
                "decode_rbf_global",
                side_effect=lambda q, *_args, **_kwargs: self.basis_one @ np.asarray(q).reshape(-1),
            ),
            mock.patch.object(pod_rbf, "jac_rbf_global", return_value=self.basis_one),
        ):
            pod_rbf.compute_ECSW_training_matrix_2D_rbf_global(
                snaps=np.array([[3.0], [4.0], [5.0], [6.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                basis=self.basis_one,
                basis2=self.basis_tail,
                W_global=None,
                q_p_train=None,
                q_s_train=None,
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
                scaler=None,
                epsilon=1.0,
            )

        np.testing.assert_allclose(record[0], [5.0, 0.0, 0.0, 0.0])

    def test_gpr_projects_the_predecessor(self):
        record = []

        with (
            mock.patch.object(pod_gpr, "get_ops", return_value=self.ops),
            mock.patch.object(pod_gpr, "_get_gp_runtime_cache", return_value=None),
            mock.patch.object(pod_gpr, "_get_scaler_affine_cache", return_value=None),
            mock.patch.object(
                pod_gpr,
                "decode_gp",
                side_effect=lambda q_p, **_kwargs: self.basis_one @ np.asarray(q_p).reshape(-1),
            ),
            mock.patch.object(pod_gpr, "jac_gp", return_value=self.basis_one),
        ):
            pod_gpr.compute_ECSW_training_matrix_2D_gpr(
                snaps=np.array([[3.0], [4.0], [5.0], [6.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                basis=self.basis_one,
                basis2=self.basis_tail,
                gp_model=object(),
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
                scaler=None,
            )

        np.testing.assert_allclose(record[0], [5.0, 0.0, 0.0, 0.0])

    def test_gplvm_projects_the_predecessor(self):
        record = []

        with (
            mock.patch.object(pod_gplvm, "get_ops", return_value=self.ops),
            mock.patch.object(
                pod_gplvm,
                "_initial_latent_from_state",
                side_effect=lambda state, *_args, **_kwargs: np.asarray(state[:1], dtype=np.float64),
            ),
            mock.patch.object(
                pod_gplvm,
                "decode_gplvm",
                side_effect=lambda z, *_args, **_kwargs: self.basis_one @ np.asarray(z).reshape(-1),
            ),
            mock.patch.object(pod_gplvm, "jac_gplvm", return_value=self.basis_one),
        ):
            pod_gplvm.compute_ECSW_training_matrix_2D_gplvm(
                snaps=np.array([[3.0], [4.0], [5.0], [6.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                basis_q=self.basis_one,
                gplvm_model={"Z_train": np.array([[0.0], [1.0]])},
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
                inverse_method="gauss_newton",
            )

        np.testing.assert_allclose(record[0], [5.0, 0.0, 0.0, 0.0])

    def test_quadratic_manifold_projects_the_predecessor(self):
        record = []

        with mock.patch.object(pod_qm, "get_ops", return_value=self.ops):
            pod_qm.compute_ECSW_training_matrix_2D_qm(
                snaps=np.array([[3.0], [4.0], [5.0], [6.0]]),
                prev_snaps=np.array([[5.0], [99.0], [8.0], [9.0]]),
                V=self.basis_one,
                H=np.zeros((4, 1), dtype=np.float64),
                u_ref=np.zeros(4),
                res=_recording_residual(record),
                jac=_unit_jacobian,
                grid_x=None,
                grid_y=None,
                dt=0.5,
                mu=(1.0, 2.0),
                max_gn_its=2,
            )

        np.testing.assert_allclose(record[0], [5.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
