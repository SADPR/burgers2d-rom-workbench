"""Numerical and presentation contracts for the HPROM paper asset generator."""

import importlib
from pathlib import Path

import numpy as np


def assets(monkeypatch):
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    monkeypatch.syspath_prepend(str(paper))
    return importlib.import_module("generate_euclidean_hprom_paper_assets")


def test_orthogonal_state_errors_match_explicit_reconstruction(monkeypatch):
    module = assets(monkeypatch)
    rng = np.random.default_rng(141)
    basis = np.linalg.qr(rng.normal(size=(31, 13)))[0]
    uref = rng.normal(size=(31, 1))
    truth = rng.normal(size=(31, 17))
    q = rng.normal(size=(13, 17))
    projected = basis.T @ (truth - uref)
    remainder = truth - uref - basis @ projected
    trajectory, history = module.state_errors(
        q, projected, np.sum(remainder**2, axis=0), np.sum(truth**2, axis=0)
    )
    direct = truth - uref - basis @ q
    np.testing.assert_allclose(trajectory, 100 * np.linalg.norm(direct) / np.linalg.norm(truth))
    np.testing.assert_allclose(history, 100 * np.linalg.norm(direct, axis=0) / np.linalg.norm(truth, axis=0))


def test_log_limits_include_all_positive_values(monkeypatch):
    module = assets(monkeypatch)
    values = [np.array([0, -.5, .0031, .24, 1551.3, np.nan, np.inf])]
    lower, upper = module.limits(values)
    assert 0 < lower < .0031
    assert upper > 1551.3


def test_table_budget_and_intrusive_separations(monkeypatch, tmp_path):
    module = assets(monkeypatch)
    monkeypatch.setattr(module, "TABLES", tmp_path)
    module.table("test.tex", "lrr|rr", "Model & Mean & Extrap & Mean & Extrap",
                 [["HPROM--POD--AE", "1", "2", "3", "4"], None,
                  ["POD--NN--ROM", "5", "6", "7", "8"]])
    text = (tmp_path / "test.tex").read_text()
    assert r"\begin{tabular}{lrr|rr}" in text
    assert "4 \\\\\n\\midrule\nPOD--NN--ROM" in text
    assert module.METHODS == ("linear", "case1", "case2", "b3", "case3", "podae", "podnn", "poddl")
