"""Numerical and presentation contracts for the HPROM paper asset generator."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


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


def test_enriched_b3_uses_2048_and_never_falls_back(monkeypatch, tmp_path):
    module = assets(monkeypatch)
    monkeypatch.setattr(module, "CAMPAIGNS", {"lhs8": tmp_path})
    rule_dir = tmp_path / "Stage4/case2_b3/positive_fit2048"
    rule_dir.mkdir(parents=True)
    (rule_dir / "selection.json").write_text(json.dumps({"rule_file": str(rule_dir / "weights.npy")}))
    np.save(rule_dir / "weights.npy", np.ones(62500))
    folder = rule_dir.parent / "reporting_positive_fit2048_hprom3_p0_steps500"
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps({"state_error_percent": .44}))
    np.save(folder / "qN.npy", np.ones((151,501)))
    old = rule_dir.parent / "reporting_positive_fit4096_hprom3_p0_steps500"
    old.mkdir()
    (old / "summary.json").write_text("4096 must not be read")
    record = module.load_record("lhs8", "b3", 0)
    assert record.weights_path == rule_dir / "weights.npy"
    assert record.qpath == folder / "qN.npy"
    (folder / "qN.npy").unlink()
    with pytest.raises(FileNotFoundError):
        module.load_record("lhs8", "b3", 0)


def test_baseline_b3_resolves_frozen_selected_deployment(monkeypatch, tmp_path):
    module = assets(monkeypatch)
    local, remote = tmp_path / "local", tmp_path / "sherlock"
    monkeypatch.setattr(module, "B3_BUDGET_LOCAL", local)
    monkeypatch.setattr(module, "B3_BUDGET_SHERLOCK", remote)
    rule_dir = local / "draws2048"
    rule_dir.mkdir(parents=True)
    remote.mkdir()
    chosen = dict(draws=2048, accepted=True, min_gram=.85, max_gram=1.09,
                  max_gradient_error=.03, max_linearized_ratio=1.01,
                  weights_sha256="frozen-weights", validation_error_ratios=[1.02,1.03])
    selection = dict(selected_draws=2048, smaller_rule_accepted=True, uses_reporting_points=False,
                     candidates=[chosen], thresholds={"max_validation_error_ratio": 1.05})
    (local / "selection.json").write_text(json.dumps(selection))
    environment = dict(selection=selection, threads=1, reuse_predictor=True,
                       selected_weights_sha256="frozen-weights", inputs={"master": "frozen-master"})
    (remote / "environment.json").write_text(json.dumps(environment))
    manifest, rule, audit, _ = module.b3_provenance("baseline")
    assert rule == rule_dir / "weights.npy"
    assert audit == rule_dir / "validation_operator_audit.json"
    assert manifest["accepted"] and manifest["checks"]["minimum_Gram_eigenvalue"] == .85
    environment["reuse_predictor"] = False
    (remote / "environment.json").write_text(json.dumps(environment))
    with pytest.raises(ValueError, match="protocol mismatch"):
        module.b3_provenance("baseline")
