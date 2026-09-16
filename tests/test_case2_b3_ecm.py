import numpy as np
import pytest

from Project_YvonMaday.study_case2_b3_ecm import (
    rank_for_tolerance, range_extension, squared_norm, ecm_rule,
)


def test_rank_counts_unrepresented_energy():
    # A perfect compressed SVD still leaves energy outside its range.
    with pytest.raises(ValueError, match="Range"):
        rank_for_tolerance(np.array([90., 5.]), 100., .1)
    rank, error = rank_for_tolerance(np.array([90., 5., 4.5]), 100., .1)
    assert rank == 3
    assert error == pytest.approx(np.sqrt(.005))


def test_randomized_range_error_matches_direct_projection():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(60, 8)) @ rng.normal(size=(8, 40))
    first, a = range_extension(matrix, np.empty((60, 0)), 4, 12)
    second, b = range_extension(matrix, first, 4, 12)
    q, projected = np.column_stack((first, second)), np.vstack((a, b))
    np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(q @ projected, matrix, atol=1e-11)
    assert abs(squared_norm(matrix)-squared_norm(projected)) < 1e-8


def test_ecm_integrates_basis_and_constant_with_positive_weights(tmp_path):
    x = np.linspace(-1, 1, 35)
    basis = np.linalg.qr(np.column_stack((x, x**2, x**3)))[0]
    weights, error = ecm_rule(basis, 1e-10, tmp_path / "ecm.log")
    assert (weights >= 0).all()
    assert np.count_nonzero(weights) <= 4
    assert error < 1e-9
    np.testing.assert_allclose(weights.sum(), len(x), atol=1e-8)
    np.testing.assert_allclose(basis.T @ weights, basis.sum(axis=0), atol=1e-8)
