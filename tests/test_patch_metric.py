"""Independent derivative, optimality and online checks of PSD patch factors."""
import unittest
import numpy as np
from burgers.patch_metric import (svec, smat, FactorFitProblem, fit_adaptive_rank,
    PatchResidualMetric, assemble_group_design, regular_mesh_groups, patch_correlations,
    greedy_patch_metrics)
from burgers.gauss_newton import gauss_newton_ECSW_2D


class PatchMetricTests(unittest.TestCase):
    def test_factor_gradient_matches_centered_differences(self):
        rng = np.random.default_rng(37)
        G = rng.normal(size=(25, 9))
        b = rng.normal(size=25)
        problem = FactorFitProblem(G, b, [3, 2], gram=G.T @ G)
        x = rng.normal(size=3*2+2*1)
        _, gradient = problem.objective_gradient(x, [2, 1])
        finite = []
        for k in range(x.size):
            delta = np.zeros(x.size)
            delta[k] = 1e-6
            finite.append((problem.objective_gradient(x+delta, [2, 1])[0]
                          -problem.objective_gradient(x-delta, [2, 1])[0])/2e-6)
        np.testing.assert_allclose(gradient, finite, rtol=1e-6, atol=1e-7)

    def test_qr_preserves_objective_and_factor_gradient(self):
        rng=np.random.default_rng(71)
        G=rng.normal(size=(25,9))
        b=rng.normal(size=25)
        ranks=[2,1]
        x=rng.normal(size=8)
        original=FactorFitProblem(G,b,[3,2])
        stable=FactorFitProblem(G,b,[3,2],gram=G.T @ G)
        stable.enable_qr()
        stable.center_objective(stable.unpack(x,ranks))
        perturbed=x+rng.normal(size=8)*.01
        value,gradient=stable.objective_gradient(perturbed,ranks)
        expected,expected_gradient=original.objective_gradient(perturbed,ranks)
        self.assertAlmostEqual(value+stable.center_value,expected,places=12)
        np.testing.assert_allclose(gradient,expected_gradient,rtol=1e-12,atol=1e-12)
        finite=[]
        for i in range(x.size):
            delta=np.eye(x.size)[i]*1e-6
            finite.append((stable.objective_gradient(perturbed+delta,ranks)[0]
                          -stable.objective_gradient(perturbed-delta,ranks)[0])/2e-6)
        np.testing.assert_allclose(gradient,finite,rtol=1e-6,atol=1e-7)

    def test_rank_growth_recovers_analytical_psd_projection(self):
        target = np.array([[2., .5, 0.], [.5, -1., .3], [0., .3, 1.]])
        values, vectors = np.linalg.eigh(target)
        expected = (vectors*np.maximum(values, 0.)) @ vectors.T
        factors, report = fit_adaptive_rank(np.eye(6), svec(target), [3],
            initial_rank=1, max_seconds=15., kkt_tolerance=1e-8,
            stage_iterations=1500, gram=np.eye(6))
        actual = factors[0] @ factors[0].T
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertGreaterEqual(factors[0].shape[1], 2)
        self.assertTrue(report['numerical_kkt_satisfied'])
        self.assertFalse(report['tolerance_met'])

    def test_square_rank_deficient_factor_can_escape_a_saddle(self):
        # A square factor can have unused/null columns. Counting columns as
        # rank previously blocked a necessary PSD descent direction.
        target = np.diag([1., 2., 3.])
        initial = np.diag([1., 0., 0.])
        factors, report = fit_adaptive_rank(np.eye(6), svec(target), [3],
            initial_factors=[initial], max_seconds=15., gram=np.eye(6))
        np.testing.assert_allclose(factors[0] @ factors[0].T, target, atol=2e-5)
        self.assertTrue(report['tolerance_met'])
        self.assertLessEqual(factors[0].shape[1], 3)

    def test_design_prediction_and_candidate_correlations(self):
        rng = np.random.default_rng(42)
        R = rng.normal(size=(5, 8, 2))
        W = rng.normal(size=(5, 8, 2, 3))
        groups = regular_mesh_groups((2, 4))
        G, dimensions = assemble_group_design(R, W, np.arange(8), groups)
        factors = [rng.normal(size=(d, 2)) for d in dimensions]
        matrices = [L @ L.T for L in factors]
        predicted = G @ np.concatenate([svec(A) for A in matrices])
        direct = np.zeros((5, 3))
        for cells, A in zip(groups, matrices):
            for c in range(5):
                direct[c] += W[c, cells].reshape(8, 3).T @ A @ R[c, cells].ravel()
        np.testing.assert_allclose(predicted, direct.ravel(), atol=2e-13)
        h = rng.normal(size=15)
        correlations = patch_correlations(R, W, h, groups)
        gradient = G.T @ h
        for p, S in enumerate(correlations):
            np.testing.assert_allclose(svec(S), gradient[p*36:(p+1)*36], atol=2e-13)

    def test_common_objective_and_symmetric_online_matrix(self):
        rng = np.random.default_rng(45)
        groups = [np.array([2, 0]), np.array([3, 1])]
        Ls = [rng.normal(size=(4, 3)) for _ in groups]
        metric = PatchResidualMetric(np.arange(4), groups, Ls)
        A = rng.normal(size=(8, 2))
        offset = rng.normal(size=8)
        q = np.array([.8, -.4])
        def residual(x):
            r = A @ x-offset
            r[1] += x[0]*x[1]
            return r
        tangent = A.copy()
        tangent[1] += [q[1], q[0]]
        transformed_W = metric.apply(tangent)
        gradient = transformed_W.T @ metric.apply(residual(q))
        numerical = []
        for i in range(2):
            delta = np.eye(2)[i]*1e-6
            numerical.append((np.linalg.norm(metric.apply(residual(q+delta)))**2
                             -np.linalg.norm(metric.apply(residual(q-delta)))**2)/4e-6)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-8, atol=1e-8)
        K = transformed_W.T @ transformed_W
        np.testing.assert_allclose(K, K.T, atol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(K)[0], 0.)

    def test_patch_identity_matches_existing_classical_solver(self):
        rng = np.random.default_rng(40)
        A = rng.normal(size=(8, 2))
        b = rng.normal(size=8)
        arguments = dict(func=lambda q:A @ q-b, jac=lambda q:A,
            basis=np.eye(2), y0=np.zeros(2), sample_inds=np.arange(4),
            augmented_sample=np.arange(4), sample_weights=np.ones(8))
        classical = gauss_newton_ECSW_2D(**arguments)[0]
        metric = PatchResidualMetric(np.arange(4), [np.arange(4)], [np.eye(8)])
        transformed = gauss_newton_ECSW_2D(**arguments, residual_transform=metric)[0]
        np.testing.assert_allclose(transformed, classical, atol=1e-13)

    def test_independent_selection_starts_empty_and_reaches_tolerance(self):
        rng = np.random.default_rng(23)
        R = rng.normal(size=(5, 8, 2))
        W = rng.normal(size=(5, 8, 2, 3))
        b = np.einsum('cean,cea->cn', W, R).ravel()
        groups = regular_mesh_groups((2, 4))
        progress = []
        metric, report = greedy_patch_metrics(R, W, b, groups, max_seconds=30., progress=progress.append)
        self.assertEqual(progress[0]['iteration'], 0)
        self.assertEqual(progress[0]['cells'], 0)
        self.assertTrue(report['tolerance_met'], report['stop_reason'])
        G, _ = assemble_group_design(R, W, np.arange(8), metric.groups)
        prediction = G @ np.concatenate([svec(L @ L.T) for L in metric.factors])
        self.assertLessEqual(np.linalg.norm(prediction-b)/np.linalg.norm(b), 1e-5)


if __name__ == '__main__':
    unittest.main()
