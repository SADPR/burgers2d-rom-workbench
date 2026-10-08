"""Exact training-pair counts for comparisons across time offsets."""
import unittest
import numpy as np

from burgers.core import get_snapshot_params
from burgers.ecsw_utils import build_ecsw_snapshot_plan


class SnapshotCountTests(unittest.TestCase):
    def plan(self, offset, count=None, percent=None):
        return build_ecsw_snapshot_plan(
            num_steps=500, snap_time_offset=offset, num_mu=9,
            mode='global_param_time_stratified', total_snapshots=count,
            total_snapshots_percent=percent, mu_points=get_snapshot_params(),
            random_seed=42,
        )

    def test_exact_225_pairs_for_both_offsets(self):
        for offset in (1, 3):
            with self.subTest(offset=offset):
                plan = self.plan(offset, count=225)
                self.assertEqual(plan['num_selected_total'], 225)
                self.assertEqual(plan['num_selected_per_mu'], [25]*9)
                for selected in plan['selected_now_cols_by_mu']:
                    self.assertTrue(np.all(selected >= offset))
                    self.assertTrue(np.all(selected < 500))
                    self.assertEqual(np.unique(selected).size, selected.size)

    def test_count_matches_existing_225_consecutive_pair_plan(self):
        exact = self.plan(1, count=225)
        percentage = self.plan(1, percent=5.)
        for a, b in zip(exact['selected_now_cols_by_mu'],
                        percentage['selected_now_cols_by_mu']):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
