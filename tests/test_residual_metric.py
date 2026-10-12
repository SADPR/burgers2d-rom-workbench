"""Independent analytical checks of full-PSD residual-metric refits."""
import importlib.util
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.optimize import nnls
from burgers.residual_metric import (project_soc_blocks, refit_psd_matrices,
                                     greedy_psd_metric_weights)


@unittest.skipUnless(importlib.util.find_spec('clarabel'), 'optional Clarabel dependency')
class FullPsdRefitTests(unittest.TestCase):
    def test_identity_training_has_analytical_projection_solution(self):
        # Includes interior, polar-cone and boundary solutions.
        target = np.array([[3.,.2,-.7],[-3.,.2,.7],[-1.,2.,.3],[.2,1.5,-2.]])
        x, _, report = refit_psd_matrices(np.eye(target.size),target.ravel(),
                                         np.zeros(target.size),threads=1)
        expected = []
        for t,z,b in target:
            radius = np.hypot(z,b)
            if t>=radius: expected.append([t,z,b])
            elif t<=-radius: expected.append([0.,0.,0.])
            else:
                s=(t+radius)/2
                expected.append([s,s*z/radius,s*b/radius])
        np.testing.assert_allclose(x.reshape(-1,3),expected,atol=1e-7)
        self.assertEqual(report['status'],'Solved')

    def test_refit_rotates_rank_one_initial_matrix(self):
        # Initial A=[[1,0],[0,0]], optimum A=[[1,.7],[.7,2]].
        initial=np.array([.5,.5,0.])
        target=np.array([1.5,-.5,.7])
        x,residual,_=refit_psd_matrices(np.eye(3),target,initial,threads=1)
        np.testing.assert_allclose(x,target,atol=1e-7)
        self.assertLess(np.linalg.norm(residual),1e-7)

    def test_full_refit_improves_over_fixed_rank_one_directions(self):
        rng=np.random.default_rng(7)
        ncell=5
        training=rng.normal(size=(40,3*ncell))
        target_coeff=np.tile([1.5,-.5,-.7],ncell)
        target=training@target_coeff
        angles=rng.uniform(-np.pi,np.pi,ncell)
        atoms=np.zeros((3*ncell,ncell))
        for e,angle in enumerate(angles):
            atoms[3*e:3*e+3,e]=[.5,.5*np.cos(2*angle),.5*np.sin(2*angle)]
        restricted,_=nnls(training@atoms,target)
        initial=atoms@restricted
        restricted_error=np.linalg.norm(target-training@initial)
        _,residual,report=refit_psd_matrices(training,target,initial,threads=1)
        self.assertGreater(restricted_error,1.)
        self.assertLess(np.linalg.norm(residual),1e-6)
        self.assertEqual(report['status'],'Solved')

    def test_small_cell_selection_reaches_original_residual_tolerance(self):
        rng=np.random.default_rng(42)
        blocks=[rng.normal(size=(12,6)) for _ in range(3)]
        target=blocks[0].sum(axis=1)+blocks[2].sum(axis=1)
        weights,report=greedy_psd_metric_weights(blocks,target,relative_tolerance=1e-5,threads=1)
        self.assertTrue(report['tolerance_met'])
        self.assertGreaterEqual(np.linalg.eigvalsh(weights).min(),-1e-10)
        fitted=(blocks[0]@weights[:,0,0]+blocks[1]@weights[:,0,1]+blocks[2]@weights[:,1,1])
        self.assertLessEqual(np.linalg.norm(fitted-target)/np.linalg.norm(target),1e-5)


    def test_every_added_cell_gets_a_full_refit_and_checkpoint(self):
        rng = np.random.default_rng(42)
        blocks = [rng.normal(size=(12, 8)) for _ in range(3)]
        target = blocks[0].sum(axis=1) + blocks[2].sum(axis=1)
        saved, progress, column_counts = [], [], []
        def checkpoint(selected, scales, coefficients, norm_b):
            saved.append((selected.copy(), coefficients.copy()))
        def refit(training, *args, **kwargs):
            column_counts.append(training.shape[1])
            return refit_psd_matrices(training, *args, **kwargs)
        with patch('burgers.residual_metric.refit_psd_matrices', side_effect=refit):
            weights, report = greedy_psd_metric_weights(
                blocks, target, relative_tolerance=1e-5, threads=1,
                checkpoint=checkpoint, progress=progress.append)
        count = report['selected_cells']
        self.assertTrue(report['tolerance_met'])
        self.assertEqual(column_counts, list(range(3, 3*count+1, 3)))
        self.assertEqual(report['joint_refits'], count)
        self.assertEqual(len(saved), count)
        self.assertEqual(len(progress), count)
        self.assertGreaterEqual(np.linalg.eigvalsh(weights).min(), -1e-10)

    def test_failed_status_retries_the_same_problem_without_relaxing_tolerances(self):
        import clarabel
        original = clarabel.DefaultSolver
        calls = []
        def factory(P, q, A, center, cones, settings):
            calls.append((P.toarray(), center.copy(), settings.tol_gap_abs,
                          settings.tol_feas, settings.reduced_tol_feas))
            solver = original(P, q, A, center, cones, settings)
            def solve():
                solution = solver.solve()
                if len(calls) == 1:
                    # Preserve a real solver iterate, but force the failure
                    # branch so this check does not depend on platform roundoff.
                    fields = ('x', 'iterations', 'solve_time', 'r_prim', 'r_dual',
                              'obj_val', 'obj_val_dual')
                    return SimpleNamespace(status='InsufficientProgress',
                        **{name:getattr(solution, name) for name in fields})
                return solution
            return SimpleNamespace(solve=solve, get_info=solver.get_info)
        target = np.array([1.5, -.5, .7])
        with patch('clarabel.DefaultSolver', side_effect=factory):
            x, residual, report = refit_psd_matrices(
                np.eye(3), target, np.zeros(3), threads=1)
        np.testing.assert_allclose(x, target, atol=1e-7)
        self.assertLess(np.linalg.norm(residual), 1e-7)
        self.assertEqual(report['retry_count'], 1)
        self.assertEqual(report['attempts'][0]['status'], 'InsufficientProgress')
        self.assertIn(report['status'], ('Solved', 'AlmostSolved'))
        np.testing.assert_array_equal(calls[0][0], calls[1][0])
        self.assertEqual(calls[0][2:], calls[1][2:])
        self.assertGreater(np.linalg.norm(calls[1][1]), 0.)

    def test_failed_solver_status_is_never_accepted(self):
        import clarabel
        original = clarabel.DefaultSolver
        calls = []
        def factory(*args):
            calls.append(args[-1].direct_solve_method)
            solver = original(*args)
            def solve():
                solution = solver.solve()
                fields = ('x', 'iterations', 'solve_time', 'r_prim', 'r_dual',
                          'obj_val', 'obj_val_dual')
                return SimpleNamespace(status='InsufficientProgress',
                    **{name:getattr(solution, name) for name in fields})
            return SimpleNamespace(solve=solve, get_info=solver.get_info)
        with patch('clarabel.DefaultSolver', side_effect=factory):
            with self.assertRaisesRegex(RuntimeError, 'after equivalent retries'):
                refit_psd_matrices(np.eye(3), np.array([1.5, -.5, .7]),
                                   np.zeros(3), threads=1)
        self.assertEqual(calls, ['faer', 'faer', 'qdldl'])


if __name__=='__main__':
    unittest.main()
