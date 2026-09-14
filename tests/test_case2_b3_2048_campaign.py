import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from Project_YvonMaday import validate_case2_hyper_rule as validator


PROJECT = Path(__file__).resolve().parents[1] / "Project_YvonMaday"
SCRIPTS = PROJECT / "Results_Paper/scripts"


def prepare_validation(root):
    rule = root / "positive_fit2048/weights.npy"
    rule.parent.mkdir(parents=True)
    np.save(rule, np.ones(3))
    digest = hashlib.sha256(rule.read_bytes()).hexdigest()
    inputs = {"basis": "basis-hash", "master": "enriched-master-hash"}
    records = [dict(point=p, step=k, Gram_eigenvalue_min=.98,
                    Gram_eigenvalue_max=1.02, gradient_relative_error=.01,
                    sampled_true_linearized_residual=2.01, full_true_linearized_residual=2.)
               for p in range(2) for k in (3, 15, 75, 175, 275, 425, 500)]
    audit = rule.parent / "validation_operator_audit.json"
    audit.write_text(json.dumps(dict(inputs=inputs, weights_sha256=digest, records=records)))
    for point in range(2):
        folder = root / f"validation_positive_fit2048_hprom3_p{point}_steps500"
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(dict(
            config=dict(inputs=inputs, weights_sha256=digest),
            coefficient_error_percent=.05, online_seconds=8., state_error_percent=.44)))
        old = root / f"validation_positive_fit4096_hprom3_p{point}_steps500"
        old.mkdir()
        (old / "summary.json").write_text("Old 4096 results must not be read")
    (root / "selection.json").write_text("Old 4096 selection must not be changed")
    return rule, audit


def select(monkeypatch, root, rule):
    target = rule.parent / "selection.json"
    monkeypatch.setattr(sys, "argv", ["validate", "--output", str(root), "--rule-file",
                                      str(rule), "--selection-file", str(target)])
    validator.main()
    return json.loads(target.read_text())


def test_selection_is_rule_specific_and_preserves_4096(monkeypatch, tmp_path):
    rule, _ = prepare_validation(tmp_path)
    selected = select(monkeypatch, tmp_path, rule)
    assert selected["accepted"] and not selected["selection_uses_reporting_points"]
    assert selected["weights_sha256"] == hashlib.sha256(rule.read_bytes()).hexdigest()
    assert all("positive_fit2048" in r["summary"] for r in selected["held_out_validation"])
    assert (tmp_path / "selection.json").read_text() == "Old 4096 selection must not be changed"


@pytest.mark.parametrize("fault", ["weights", "inputs", "coverage", "threshold"])
def test_selection_rejects_stale_or_failed_validation(monkeypatch, tmp_path, fault):
    rule, audit_path = prepare_validation(tmp_path)
    audit = json.loads(audit_path.read_text())
    if fault == "weights":
        np.save(rule, np.full(3, 2.))
    elif fault == "inputs":
        audit["inputs"]["master"] = "wrong-master"
    elif fault == "coverage":
        audit["records"][-1] = audit["records"][0]
    else:
        audit["records"][-1]["gradient_relative_error"] = .051
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(SystemExit):
        select(monkeypatch, tmp_path, rule)
    target = rule.parent / "selection.json"
    assert not target.exists() or not json.loads(target.read_text())["accepted"]


def test_evaluation_propagates_predictor_reuse_and_resumes(monkeypatch, tmp_path):
    from Project_YvonMaday import run_case2_hyperreduction as hyper

    rule = tmp_path / "weights.npy"
    np.save(rule, np.ones(3))
    args = SimpleNamespace(stage="reporting", methods=["hprom3"], rule_file=rule,
                           output=tmp_path, point_indices=[0], steps=1, threads=1,
                           overwrite_existing=False, reuse_predictor=True)
    calls = []

    def rollout(*args, **kwargs):
        calls.append(kwargs)
        return np.zeros((151, 2)), dict(online_seconds=1.)

    monkeypatch.setattr(hyper, "SampledBurgers", lambda *a: SimpleNamespace(samples=[0], augmented=[0]))
    monkeypatch.setattr(hyper, "hyper_rollout", rollout)
    monkeypatch.setattr(hyper, "reference_path", lambda *a: rule)
    monkeypatch.setattr(hyper, "score", lambda *a: dict(state_error_percent=.44))
    hyper.evaluate(args, None, None, None, {"master": "hash"})
    hyper.evaluate(args, None, None, None, {"master": "hash"})
    assert calls == [{"reuse_predictor": True}]
    record, = tmp_path.glob("reporting_*_hprom3_p0_steps1/summary.json")
    assert json.loads(record.read_text())["config"]["reuse_predictor"] is True
    args.reuse_predictor = False
    with pytest.raises(ValueError, match="Run configuration changed"):
        hyper.evaluate(args, None, None, None, {"master": "hash"})


def mock_campaign(root):
    project = root / "Project_YvonMaday"
    scripts = project / "Results_Paper/scripts"
    scripts.mkdir(parents=True)
    for name in ("run_euclidean_hprom_case2_b3.sh", "run_euclidean_hprom_nested_enrichment_b3.sh"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    shutil.copyfile(PROJECT / "run_euclidean_hprom_b3_2048_correction.sh",
                    project / "run_euclidean_hprom_b3_2048_correction.sh")
    for relative in ("MetricStudy/euclidean/Stage1/basis.npy", "MetricStudy/euclidean/Stage1/u_ref.npy",
                     "euclidean_hprom_main/Stage2/prom_coeff_dataset_ntot151/meta.json"):
        path = project / "Results_Paper" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing input")
    for level in (8, 12, 18):
        path = project / f"Results_Paper/euclidean_hprom_enrichment_lhs{level}"
        model = path / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"existing ANN")
        old = path / "Stage4/case2_b3/positive_fit4096/weights.npy"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"existing 4096 weights")
        (old.parent.parent / "selection.json").write_bytes(b"existing 4096 selection")
    stub = root / "python_stub"
    stub.write_text(f"#!{sys.executable}\n" + '''import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == "-u": args = args[1:]
with open(os.environ["TRACE"], "a") as stream:
    stream.write(json.dumps(dict(args=args, threads=os.environ["OMP_NUM_THREADS"],
                                model=os.environ["CASE2_MODEL_PATH"])) + "\\n")
if args[0] == "-":
    sys.argv = args
    exec(compile(sys.stdin.read(), "inline", "exec"))
    raise SystemExit(0)
if args[0] == os.environ.get("FAIL_SCRIPT"):
    raise SystemExit(9)
def value(flag): return args[args.index(flag)+1]
out = pathlib.Path(value("--output"))
if args[0] == "fit_case2_positive_cubature.py":
    rule = out / ("positive_fit" + value("--draws")) / "weights.npy"
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_bytes(b"new fitted 2048 weights")
elif args[0] == "validate_case2_hyper_rule.py":
    rule = pathlib.Path(value("--rule-file"))
    root = pathlib.Path.cwd().parent
    basis = pathlib.Path(os.environ["CASE2_BASIS_DIR"])
    paths = [pathlib.Path(os.environ["CASE2_MODEL_PATH"]), basis / "basis.npy", basis / "u_ref.npy",
             pathlib.Path(os.environ["CASE2_TRAINING_DATASET"]) / "meta.json"]
    inputs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    pathlib.Path(value("--selection-file")).write_text(json.dumps(dict(
        accepted=True, selection_uses_reporting_points=False, rule_file=str(rule.resolve()),
        weights_sha256=hashlib.sha256(rule.read_bytes()).hexdigest(), inputs=inputs)))
''')
    stub.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(stub), TRACE=str(root / "trace.jsonl"),
               CASE2_B3_DRAWS="4096", CASE2_B3_REUSE_PREDICTOR="0")
    return project, env


@pytest.mark.parametrize("failure", [None, "fit_case2_positive_cubature.py", "validate_case2_hyper_rule.py"])
def test_correction_runs_only_requested_b3_and_stops_failed_levels(tmp_path, failure):
    project, env = mock_campaign(tmp_path)
    if failure:
        env["FAIL_SCRIPT"] = failure
    result = subprocess.run(["bash", str(project / "run_euclidean_hprom_b3_2048_correction.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == (1 if failure else 0), result.stderr
    calls = [json.loads(line) for line in Path(env["TRACE"]).read_text().splitlines()]
    builds = [c for c in calls if c["args"][0] == "build_case2_leverage_rule.py"]
    assert len(builds) == 3
    assert all(c["args"][c["args"].index("--draws")+1] == "2048" for c in builds)
    assert all(c["args"][c["args"].index("--seed")+1] == "418" for c in builds)
    assert all(c["threads"] == "24" for c in builds)
    rollouts = [c for c in calls if c["args"][0] == "run_case2_hyperreduction.py"]
    assert len(rollouts) == (6 if not failure else 3 if failure.startswith("validate") else 0)
    assert all("--reuse-predictor" in c["args"] and c["threads"] == "1" for c in rollouts)
    assert all("positive_fit2048" in c["args"][c["args"].index("--rule-file")+1] for c in rollouts)
    assert all("--overwrite-existing" not in c["args"] for c in rollouts)
    if failure:
        assert not any(c["args"][1] == "reporting" for c in rollouts)
    for old in project.glob("Results_Paper/euclidean_hprom_enrichment_lhs*/Stage4/case2_b3/positive_fit4096/weights.npy"):
        assert old.read_bytes() == b"existing 4096 weights"
        assert (old.parent.parent / "selection.json").read_bytes() == b"existing 4096 selection"


@pytest.mark.parametrize("changed", ["model", "weights"])
def test_reporting_gate_rechecks_validated_files(tmp_path, changed):
    project, env = mock_campaign(tmp_path)
    result = subprocess.run(["bash", str(project / "run_euclidean_hprom_b3_2048_correction.sh"), "lhs8"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    root = project / "Results_Paper/euclidean_hprom_enrichment_lhs8"
    target = root / ("Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt" if changed == "model"
                     else "Stage4/case2_b3/positive_fit2048/weights.npy")
    target.write_bytes(b"changed after validation")
    Path(env["TRACE"]).write_text("")
    env.update(EUCLIDEAN_HPROM_ROOT=str(root), CASE2_B3_DRAWS="2048", OMP_NUM_THREADS="1")
    result = subprocess.run(["bash", str(project / "Results_Paper/scripts/run_euclidean_hprom_case2_b3.sh"),
                             "reporting"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert ("changed after validation" if changed == "model" else "current B3 weights") in result.stderr
    calls = [json.loads(line) for line in Path(env["TRACE"]).read_text().splitlines()]
    assert all(c["args"][0] != "run_case2_hyperreduction.py" for c in calls)
