import numpy as np
import pytest

from Project_YvonMaday.study_case2_b3_cubature_budget import nested_matrix, audit_checks


def test_nested_moments_match_direct_assembly_and_gradient():
    rng = np.random.default_rng(23)
    moments = rng.normal(size=(20, 9))
    old = np.array([2., 0, 3., 4., 1., 0, 2., 7., 5.])
    new = np.array([0., 0, 5., 1., 0., 0, 0., 2., 0.])
    parent = moments[:, old > 0] * old[old > 0]
    result = nested_matrix(parent, old, new)
    direct = moments[:, new > 0] * new[new > 0]
    np.testing.assert_allclose(result, direct, atol=1e-14)
    x, target = rng.random(3), moments.sum(axis=1)
    np.testing.assert_allclose(result.T @ (result @ x - target),
                               direct.T @ (direct @ x - target), atol=1e-12)
    new[1] = 1
    with pytest.raises(ValueError, match="subset"):
        nested_matrix(parent, old, new)


def test_operator_audit_enforces_each_original_bound():
    record = dict(Gram_eigenvalue_min=.98, Gram_eigenvalue_max=1.02,
                  gradient_relative_error=.01, full_true_linearized_residual=2.,
                  sampled_true_linearized_residual=2.01)
    assert audit_checks([record.copy() for _ in range(14)])[1]
    for key, bad in (("Gram_eigenvalue_min", .79), ("Gram_eigenvalue_max", 1.21),
                     ("gradient_relative_error", .051), ("sampled_true_linearized_residual", 2.11)):
        records = [record.copy() for _ in range(14)]
        records[-1][key] = bad
        assert not audit_checks(records)[1]
    records = [record.copy() for _ in range(14)]
    records[-1]["Gram_eigenvalue_min"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        audit_checks(records)
