"""Numerical and presentation contracts for the HPROM paper asset generator."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_enrichment_state_scale_preserves_shared_limits_and_outlier(monkeypatch):
    module = assets(monkeypatch)
    records = {}
    for level in module.LEVELS:
        for method in module.ENRICHMENT_METHODS:
            for point in range(4):
                value = 10.836 if (level, method, point) == ("lhs8", "podnn", 3) else .5
                records[level, method, point] = SimpleNamespace(state=value, coefficient=value)
    for point in range(4):
        records["baseline", "b3", point] = SimpleNamespace(
            state=.844 if point == 3 else .454, coefficient=.5)
    figures = []
    monkeypatch.setattr(module, "save", lambda fig, name: figures.append((fig, name)))
    module.enrichment_comparison_figure(records, "state")
    fig, name = figures.pop()
    try:
        assert name == "enrichment_state"
        assert all(ax.get_yscale() == "symlog" for ax in fig.axes)
        np.testing.assert_allclose(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
        assert fig.axes[0].get_ylim()[0] == 0
        assert fig.axes[0].get_ylim()[1] > 10.836
        assert any(np.isclose(patch.get_height(), 10.836) for patch in fig.axes[1].patches)
        for ax, expected in zip(fig.axes, (.454, .844)):
            np.testing.assert_allclose(ax.get_xticks()[1:4], [1, 2, 3])
            reference = next(line for line in ax.lines if line.get_label() == "Linear HPROM")
            assert all(reference.get_zorder() > patch.get_zorder() for patch in ax.patches)
            b3_bars = [patch for patch in ax.patches
                       if np.isclose(patch.get_x() + patch.get_width()/2, 2)]
            assert len(b3_bars) == 1
            np.testing.assert_allclose(b3_bars[0].get_height(), expected)
            np.testing.assert_allclose(b3_bars[0].get_width(), .18)
            np.testing.assert_allclose(b3_bars[0].get_facecolor(), ax.patches[0].get_facecolor())
            np.testing.assert_allclose(b3_bars[0].get_edgecolor(), ax.patches[0].get_edgecolor())
        assert r"Case 2+$\mathbf B_3$ (9)" not in fig.axes[0].get_legend_handles_labels()[1]
    finally:
        module.plt.close(fig)


def test_baseline_history_figure_uses_shared_log_limits(monkeypatch):
    module = assets(monkeypatch)
    records = {}
    values = np.full(501, .4)
    for point in range(4):
        for method in ("linear", "case1", "case2", "b3", "case3"):
            history = values.copy()
            if point == 2 and method == "case2":
                history[0] = 9.0
            records["baseline", method, point] = SimpleNamespace(history=history)
    figures = []
    monkeypatch.setattr(module, "save", lambda fig, name: figures.append((fig, name)))
    module.baseline_history_figure(records)
    fig, name = figures.pop()
    try:
        assert name == "b3_histories"
        assert all(ax.get_yscale() == "log" for ax in fig.axes)
        assert all(ax.get_ylim() == fig.axes[0].get_ylim() for ax in fig.axes)
        assert fig.axes[0].get_ylim()[0] < .4
        assert fig.axes[0].get_ylim()[1] > 9.0
    finally:
        module.plt.close(fig)
    records["baseline", "linear", 0].history[0] = 0.0
    with pytest.raises(ValueError, match="finite positive values"):
        module.baseline_history_figure(records)


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


def test_online_size_counts_residual_unknowns(monkeypatch):
    module = assets(monkeypatch)
    assert module.baseline_online_dimension("linear") == "151"
    assert module.baseline_online_dimension("b3") == "13"
    for method in ("case1", "case2", "case3", "podae"):
        assert module.baseline_online_dimension(method) == "10"
    for method in ("podnn", "poddl"):
        assert module.baseline_online_dimension(method) == "--"
    assert module.BASELINE_NAMES["case2"] == "HPROM--ANN Case 2"


def test_podnn_training_row_repeats_case2_master_map():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    lines = (paper / "tables/euclidean_hprom/training_baseline.tex").read_text().splitlines()
    case2 = next(line for line in lines if line.startswith("HPROM--ANN Case 2 "))
    podnn = next(line for line in lines if line.startswith("POD--NN--ROM &"))
    assert case2.split(" & ")[2:] == podnn.split(" & ")[2:]


def test_autoencoder_names_start_on_encoder_rows(monkeypatch):
    module = assets(monkeypatch)
    details = {}
    for method in module.NETWORKS:
        details["baseline", method] = {
            "trainable_parameters": "123",
            "train_rel_frob_percent": "1.0",
            "val_rel_frob_percent": "2.0",
            "hidden_dims": "(512, 256, 128)",
            "encoder_hidden_dims": "(512, 256)",
            "decoder_hidden_dims": "(256, 512)",
            "dynamics_hidden_dims": "(256, 512, 512, 256)",
        }
    rows = module.training_baseline_rows(details)
    for method in ("podae", "poddl"):
        first = next(i for i, row in enumerate(rows) if row and row[0] == module.NAMES[method])
        assert rows[first][1] == ("10" if method == "podae" else "--")
        assert rows[first][2].startswith(r"$\mathcal E$:")
        assert rows[first + 1][0] == ""
        assert rows[first + 1][2].startswith(r"$\mathcal D$:")
    poddl = next(i for i, row in enumerate(rows) if row and row[0] == module.NAMES["poddl"])
    assert rows[poddl + 2][2].startswith(r"$\mathcal G_z$:")


def test_model_tables_have_online_size_as_the_only_dimension_column():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    names = ("training_baseline", "training_enrichment", "baseline_state", "baseline_coefficient",
             "timing_baseline", "enrichment_state", "enrichment_coefficient", "cubature",
             "lhs8_state", "lhs8_coefficient", "lhs12_state", "lhs12_coefficient",
             "lhs18_state", "lhs18_coefficient")
    files = [paper / "tables/euclidean_hprom" / f"{name}.tex" for name in names]
    prom = ("euclidean_case2_rank_validation", "euclidean_case2_adaptive_state",
            "euclidean_case2_adaptive_coefficient", "euclidean_prom_case2_n_sweep_state_errors")
    files += [paper / "tables/euclidean_prom_only" / f"{name}.tex" for name in prom]
    for path in files:
        text = path.read_text()
        header = next(line.split("Model &", 1)[1] for line in text.splitlines() if "Model &" in line)
        assert header.startswith(" Online size &"), path
        assert all(symbol not in header for symbol in (r"$n$", r"$\bar n$", r"$r$", r"$n_z$", r"$d$")), path
        for line in text.splitlines():
            if line.startswith(("HPROM--", "PROM--", "POD--", "Linear ", "HDM &")):
                size = line.split(" & ")[1].replace("{,}", "")
                assert size == "--" if line.startswith("POD--") else size.isdigit(), (path, line)


def test_enrichment_tables_match_baseline_dimension_and_model_order():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    directory = paper / "tables/euclidean_hprom"
    training = (directory / "training_enrichment.tex").read_text().splitlines()
    case2 = next(line for line in training if line.startswith("HPROM--ANN Case 2 &"))
    podnn = next(line for line in training if line.startswith("POD--NN--ROM &"))
    assert case2.split(" & ")[2:] == podnn.split(" & ")[2:]
    assert training[training.index(podnn) - 1] == r"\midrule"
    for name in ("enrichment_state", "enrichment_coefficient"):
        text = (directory / f"{name}.tex").read_text()
        assert r"Model & Online size & Mean" in text
        assert r"HPROM--ANN Case 2 & 10 &" in text
        assert r"HPROM--POD--AE & 10 &" in text
        assert r"POD--DL--ROM & -- &" in text
    state = (directory / "enrichment_state.tex").read_text()
    b3 = next(line for line in state.splitlines()
              if line.startswith(r"HPROM--ANN Case 2+$\mathbf B_3$ &"))
    assert b3 == (r"HPROM--ANN Case 2+$\mathbf B_3$ & 13 & 0.454 & 0.844 & "
                  r"-- & -- & -- & -- & -- & -- \\")
    assert state.index(b3) > state.index("HPROM--ANN Case 2 &")
    assert state.index(b3) < state.index("HPROM--ANN Case 3 &")
    linear = next(line for line in state.splitlines()
                  if line.startswith("Linear HPROM &"))
    assert linear == (r"Linear HPROM & 151 & 0.441 & 0.850 & "
                      r"\multicolumn{2}{c|}{unchanged} & "
                      r"\multicolumn{2}{c|}{unchanged} & "
                      r"\multicolumn{2}{c}{unchanged} \\")
    assert r"Case 2+$\mathbf B_3$" not in (directory / "enrichment_coefficient.tex").read_text()


def test_prom_appendix_rank_and_reporting_tables_show_equal_online_size():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    directory = paper / "tables/euclidean_prom_only"
    rank = (directory / "euclidean_case2_rank_validation.tex").read_text()
    assert r"Model & Online size &" in rank
    assert r"PROM--ANN Case 2 + $\mathbf B_{3}$ & 13" in rank
    assert r"PROM--ANN Case 2 & 13" in rank
    for name, mean in (("euclidean_case2_adaptive_state", "0.447"),
                       ("euclidean_case2_adaptive_coefficient", "0.215")):
        text = (directory / f"{name}.tex").read_text()
        assert r"Model & Online size &" in text
        assert r"PROM--ANN Case 2+$\mathbf B_3$ & 13" in text
        assert r"PROM--ANN Case 2 & 13" in text
        adaptive = next(line for line in text.splitlines()
                        if line.startswith(r"PROM--ANN Case 2+$\mathbf B_3$ &"))
        assert f" & {mean} & " in adaptive
        assert "($n=" not in text
        assert text.index("PROM--ANN Case 1 &") < text.index("PROM--ANN Case 2 &")
        assert text.index(r"PROM--ANN Case 2+$\mathbf B_3$ &") < text.index("PROM--ANN Case 3 &")


def test_prom_sweep_distinguishes_formal_endpoints_from_online_solves():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    text = (paper / "tables/euclidean_prom_only/euclidean_prom_case2_n_sweep_state_errors.tex").read_text()
    assert r"Model & Online size &" in text
    assert r"POD--NN--ROM & -- &" in text
    assert r"PROM--ANN Case 2 & 10 &" in text
    assert r"Linear PROM & 151 &" in text
    assert text.index("Linear PROM &") < text.index("PROM--ANN Case 2 &") < text.index("POD--NN--ROM &")
    lines = text.splitlines()
    podnn_line = next(i for i, line in enumerate(lines) if line.startswith("POD--NN--ROM &"))
    assert lines[podnn_line - 1] == r"\midrule"


def test_hprom_appendix_includes_all_three_enrichment_budgets():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    manuscript = (paper / "manuscript_paper_euclidean.tex").read_text()
    labels = [f"tab:hprom-lhs{budget}-state" for budget in (8, 12, 18)]
    assert all(label in manuscript for label in labels)
    assert [manuscript.index(r"\label{" + label + "}") for label in labels] == sorted(
        manuscript.index(r"\label{" + label + "}") for label in labels)
    directory = paper / "tables/euclidean_hprom"
    model_orders = []
    for budget in (8, 12, 18):
        text = (directory / f"lhs{budget}_state.tex").read_text()
        assert r"Model & Online size &" in text
        assert r"HPROM--ANN Case 2 & 10 &" in text
        assert r"HPROM--ANN Case 2+$\mathbf B_3$ & 13 & -- & -- & -- & -- & --" in text
        assert r"HPROM--POD--AE & 10 &" in text
        assert r"POD--DL--ROM & -- &" in text
        model_orders.append([line.split(" & ")[0] for line in text.splitlines()
                             if line.startswith(("Linear HPROM &", "HPROM--", "POD--"))])
    assert model_orders[0] == model_orders[1] == model_orders[2]
    baseline = (directory / "baseline_state.tex").read_text()
    baseline_order = [line.split(" & ")[0] for line in baseline.splitlines()
                      if line.startswith(("Linear HPROM &", "HPROM--", "POD--"))]
    assert model_orders[0] == baseline_order
    training = (directory / "training_enrichment.tex").read_text()
    assert training.count("Train (\\%)") == 4
    assert training.count("Val (\\%)") == 4


def test_appendix_state_tables_share_columns_and_precision():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    paths = [paper / "tables/euclidean_prom_only" / f"{name}.tex" for name in (
        "euclidean_case2_adaptive_state", "euclidean_prom_case2_n_sweep_state_errors")]
    paths += [paper / "tables/euclidean_hprom" / f"lhs{budget}_state.tex"
              for budget in (8, 12, 18)]
    header = (r"Model & Online size & $\mu^{(v)}$ & $\mu^{(1)}$ & "
              r"$\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\")
    for path in paths:
        lines = path.read_text().splitlines()
        assert lines[:3] == [r"\begin{tabular}{lr|rrrr|r}", r"\toprule", header]
        for line in lines:
            if not line.startswith(("Linear ", "PROM--", "HPROM--", "POD--")):
                continue
            values = [cell.rstrip(" \\") for cell in line.split(" & ")[2:]]
            assert len(values) == 5
            assert all(value == "--" or (value.count(".") == 1 and
                       len(value.rsplit(".", 1)[1]) == 3) for value in values)
        if any(line.startswith("POD--NN--ROM &") for line in lines):
            direct = next(i for i, line in enumerate(lines)
                          if line.startswith("POD--NN--ROM &"))
            assert lines[direct - 1] == r"\midrule"
    tail = (paper / "tables/euclidean_prom_only/euclidean_prom_case2_tail_primary_errors.tex").read_text()
    assert tail.startswith(r"\begin{tabular}{l|rrrr}")
    assert "Imposed tail error (\\%)" in tail


def test_timing_table_uses_online_size_and_element_counts():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    text = (paper / "tables/euclidean_hprom/timing_baseline.tex").read_text()
    assert r"Model & Online size & No. elements" in text
    assert "Wall-clock time (s) & Speedup factor" in text
    assert r"HDM & 125{,}000 & 62{,}500" in text
    assert r"HPROM--ANN Case 2+$\mathbf B_3$ & 13 & 839" in text
    assert r"HPROM--POD--AE & 10 & 901" in text
    assert r"POD--NN--ROM & -- & --" in text
    assert "Threads" not in text
    for model in ("POD--NN--ROM", "POD--DL--ROM"):
        row = next(line for line in text.splitlines() if line.startswith(model))
        assert r"$^\dagger$" in row.split(" & ")[-2]
        assert r"$^\dagger$" in row.split(" & ")[-1]
        assert "{,}" in row.split(" & ")[-1]


def test_online_coefficient_tables_are_excluded_from_the_manuscript():
    paper = Path(__file__).resolve().parents[1] / "Project_YvonMaday/Results_Paper"
    manuscript = (paper / "manuscript_paper_euclidean.tex").read_text()
    main, appendix = manuscript.split(r"\appendix", 1)
    assert "e_q(" not in main
    assert r"\label{eq:coefficient-error}" in appendix
    assert appendix.count(r"\label{fig:adaptive-rank-validation}") == 1
    for label in ("online-coefficients", "hprom-enrichment-coefficients",
                  "adaptive-rank-validation", "adaptive-coefficients"):
        assert rf"\label{{tab:{label}}}" not in manuscript
    for name in ("baseline_coefficient", "enrichment_coefficient",
                 "euclidean_case2_rank_validation", "euclidean_case2_adaptive_coefficient"):
        assert f"{name}.tex" not in manuscript
    for name in ("training_baseline", "baseline_state", "enrichment_state", "timing_baseline"):
        assert f"{name}.tex" in main
    assert "training_enrichment.tex" not in main
    assert "training_enrichment.tex" in appendix


def test_enriched_b3_is_not_reported_without_its_own_ecm_rule(monkeypatch, tmp_path):
    module = assets(monkeypatch)
    monkeypatch.setattr(module, "CAMPAIGNS", {"lhs8": tmp_path})
    with pytest.raises(ValueError, match="ECM rules"):
        module.load_record("lhs8", "b3", 0)
    with pytest.raises(ValueError, match="validated ECM"):
        module.b3_provenance("lhs8")


def test_baseline_b3_resolves_frozen_ecm_deployment(monkeypatch, tmp_path):
    module = assets(monkeypatch)
    local, remote, bundle = (tmp_path / name for name in ("local", "sherlock", "bundle"))
    monkeypatch.setattr(module, "B3_ECM_LOCAL", local)
    monkeypatch.setattr(module, "B3_ECM_SHERLOCK", remote)
    monkeypatch.setattr(module, "B3_ECM_DEPLOYMENT", bundle)
    rule_dir = local / "ecm_tol0.01"
    rule_dir.mkdir(parents=True)
    remote.mkdir()
    bundle.mkdir()
    np.save(bundle / "ecm_weights.npy", np.ones(4))
    chosen = dict(tolerance=.01, sampled_cells=4, accepted=True, min_gram=.85, max_gram=1.09,
                  max_gradient_error=.03, max_linearized_ratio=1.01,
                  weights_sha256=module.digest(bundle / "ecm_weights.npy"),
                  validation_coefficient_ratios=[1.02,1.03])
    selection = dict(accepted=True, selection_uses_reporting_points=False,
                     candidates=[chosen], protocol={"inputs": {"master": "frozen-master"},
                                                    "maximum_validation_coefficient_ratio": 1.05})
    (bundle / "selection.json").write_text(json.dumps(selection))
    protocol = dict(threads=1, reuse_predictor=True, sampled_cells={"ecm": 4})
    (remote / "protocol.json").write_text(json.dumps(protocol))
    manifest, rule, audit, _ = module.b3_provenance("baseline")
    assert rule == bundle / "ecm_weights.npy"
    assert audit == bundle / "validation_operator_audit.json"
    assert manifest["accepted"] and manifest["checks"]["minimum_Gram_eigenvalue"] == .85
    protocol["reuse_predictor"] = False
    (remote / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol mismatch"):
        module.b3_provenance("baseline")
