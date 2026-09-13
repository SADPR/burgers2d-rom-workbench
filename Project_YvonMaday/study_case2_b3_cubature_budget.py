#!/usr/bin/env python3
"""Nested B3 cubature budgets, selected using two validation parameters only."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUDGETS = (3072, 2048)
THRESHOLDS = dict(min_gram=.8, max_gram=1.2, max_gradient_error=.05,
                  max_linearized_ratio=1.05, max_validation_error_ratio=1.05)


def nested_matrix(matrix, parent_prior, child_prior):
    """Restrict M diag(parent_prior) and replace its column scaling exactly."""
    if (not np.isfinite(parent_prior).all() or not np.isfinite(child_prior).all()
            or np.any(parent_prior < 0) or np.any(child_prior < 0)
            or parent_prior.shape != child_prior.shape):
        raise ValueError("Expected compatible finite nonnegative priors.")
    parent = np.flatnonzero(parent_prior > 0)
    child = np.flatnonzero(child_prior > 0)
    if matrix.ndim != 2 or matrix.shape[1] != len(parent) or not len(child):
        raise ValueError("Invalid moment matrix or empty child support.")
    if np.any(parent_prior[child] == 0):
        raise ValueError("Child support is not a subset of the parent support.")
    indices = np.searchsorted(parent, child)
    return np.ascontiguousarray(
        matrix[:, indices] * (child_prior[child] / parent_prior[child])[None, :])


def audit_checks(records):
    if len(records) != 14:
        raise ValueError("Expected fourteen validation operator records.")
    values = [[r[k] for k in (
        "Gram_eigenvalue_min", "Gram_eigenvalue_max", "gradient_relative_error",
        "sampled_true_linearized_residual", "full_true_linearized_residual")]
        for r in records]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite operator audit.")
    checks = dict(
        min_gram=min(r["Gram_eigenvalue_min"] for r in records),
        max_gram=max(r["Gram_eigenvalue_max"] for r in records),
        max_gradient_error=max(r["gradient_relative_error"] for r in records),
        max_linearized_ratio=max(
            r["sampled_true_linearized_residual"] /
            max(r["full_true_linearized_residual"], 1e-30) for r in records))
    passed = (np.isfinite(list(checks.values())).all()
              and checks["min_gram"] >= THRESHOLDS["min_gram"]
              and all(checks[k] <= THRESHOLDS[k] for k in (
                  "max_gram", "max_gradient_error", "max_linearized_ratio")))
    return checks, bool(passed)


def fit_rule(folder, matrix, target, prior, runner, maxiter=1500):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists():
        return json.loads((folder / "fit.json").read_text())
    support = np.flatnonzero(prior > 0)
    start = time.perf_counter()

    def objective(x):
        residual = matrix @ x - target
        return .5 * float(residual @ residual), matrix.T @ residual

    iterations = [0]

    def progress(x):
        iterations[0] += 1
        if iterations[0] % 100 == 0:
            print(f"{folder.name}: iteration={iterations[0]} "
                  f"objective={objective(x)[0]:.7g}", flush=True)

    result = minimize(objective, np.ones(len(support)), jac=True,
                      method="L-BFGS-B", bounds=[(0., None)] * len(support),
                      callback=progress,
                      options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxcor=20))
    weights = np.zeros_like(prior)
    weights[support] = prior[support] * result.x
    np.save(folder / "weights.npy", weights)
    mesh = runner.SampledBurgers(runner.GRID_X, runner.GRID_Y, runner.DT, weights)
    gradient = objective(result.x)[1]
    projected = np.where(result.x > 0, gradient, np.minimum(gradient, 0))
    info = dict(
        candidates=len(support), sampled_cells=len(mesh.samples),
        stencil_cells=len(mesh.augmented), objective=float(result.fun),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),
        projected_gradient_inf=float(np.max(np.abs(projected))),
        fit_seconds=time.perf_counter() - start,
        weights_sha256=runner.sha256(folder / "weights.npy"))
    runner.atomic_json(folder / "fit.json", info)
    print(json.dumps(info), flush=True)
    return info


def rollout_point(folder, stage, point, rule, inputs, runner):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{stage}_p{point}.json"
    if target.exists():
        return json.loads(target.read_text())
    model, v, uref, _ = inputs
    mu = (runner.VALIDATION if stage == "validation" else runner.REPORTING)[point]
    mesh = runner.SampledBurgers(runner.GRID_X, runner.GRID_Y, runner.DT, np.load(rule))
    profile = {}
    q, stats = runner.hyper_rollout(model, v, uref, mu, mesh, 3, 500,
                                    reuse_predictor=True, profile=profile)
    metrics = runner.score(q, v, uref, mu, runner.reference_path(mu, stage))
    record = dict(point=point, mu=list(mu), **stats, **metrics,
                  profile=profile, weights_sha256=runner.sha256(rule), threads=1)
    if not np.isfinite(q).all() or not np.isfinite(record["coefficient_error_percent"]):
        raise FloatingPointError("Non-finite validation/reporting trajectory.")
    np.save(folder / f"{stage}_p{point}_qN.npy", q)
    runner.atomic_json(target, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path,
                        default=ROOT / "Project_YvonMaday/Results_Paper/euclidean_hprom_main")
    parser.add_argument("--offline-threads", type=int, default=2)
    args = parser.parse_args()
    if args.offline_threads < 1:
        parser.error("offline threads must be positive")
    campaign = args.campaign_root.resolve()
    output = args.output.resolve()
    os.environ["CASE2_CAMPAIGN_ROOT"] = str(campaign)
    os.environ["CASE2_MODEL_PATH"] = str(
        campaign / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt")
    os.environ["CASE2_BASIS_DIR"] = str(
        ROOT / "Project_YvonMaday/Results_Paper/MetricStudy/euclidean/Stage1")
    import torch
    from threadpoolctl import threadpool_limits
    from Project_YvonMaday import run_case2_hyperreduction as runner

    output.mkdir(parents=True, exist_ok=True)
    parent = campaign / "Stage4/case2_b3"
    cache = parent / "positive_fit4096"
    original_prior_file = parent / "leverage4096/weights.npy"
    original_prior = np.load(original_prior_file)
    target = np.load(cache / "target.npy")
    source = np.load(cache / "matrix.npy", mmap_mode="r")
    torch.set_num_threads(args.offline_threads)
    with threadpool_limits(args.offline_threads):
        inputs = runner.load_inputs()
        model, v, uref, fingerprints = inputs
        cached = json.loads((cache / "config.json").read_text())
        if cached["inputs"] != fingerprints:
            raise ValueError("Cached moment inputs differ from current inputs.")
        actual_sources = {str(p.relative_to(ROOT)): runner.sha256(p / "qN.npy")
                          for p in runner.training_paths()}
        if cached["training_sources"] != actual_sources:
            raise ValueError("The nine training trajectories differ from the cache.")
        if cached["initial_rule_sha256"] != runner.sha256(original_prior_file):
            raise ValueError("Cached moment column weights differ.")
        protocol = dict(
            budgets=list(BUDGETS), parent_draws=4096, seed=418,
            thresholds=THRESHOLDS, maxiter=1500, rank=3, reuse_predictor=True,
            inputs=fingerprints, training_sources=actual_sources,
            matrix_sha256=runner.sha256(cache / "matrix.npy"),
            target_sha256=runner.sha256(cache / "target.npy"),
            parent_weights_sha256=runner.sha256(cache / "weights.npy"),
            validation_sources={str(runner.reference_path(mu, "validation")):
                                runner.sha256(runner.reference_path(mu, "validation"))
                                for mu in runner.VALIDATION},
            selection="fewest retained cells among rules passing all validation checks",
            reporting_used_for_selection=False)
        path = output / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Study protocol changed; use a separate output.")
        runner.atomic_json(path, protocol)

        # Verify the cached scaling with freshly generated moments at the first
        # training anchor; the other anchors are protected by source hashes.
        support = np.flatnonzero(original_prior > 0)
        _, _, blocks = next(runner.moments(model, v, uref, [cached["training_steps"][0]]))
        cursor = 0
        reproduction = []
        for block in blocks:
            b = block.sum(axis=0)
            scale = max(np.linalg.norm(b), 1e-12)
            direct = (block[support] * original_prior[support, None]).T / scale
            parent_block = source[cursor:cursor + len(b)]
            matrix_delta = float(np.linalg.norm(parent_block - direct) /
                                 max(np.linalg.norm(parent_block), 1e-30))
            target_delta = float(np.linalg.norm(target[cursor:cursor + len(b)] - b / scale) /
                                 max(np.linalg.norm(b / scale), 1e-30))
            reproduction.append(dict(rows=len(b), relative_matrix_delta=matrix_delta,
                                     relative_target_delta=target_delta))
            # Float32 ANN inference is not bitwise portable across BLAS/Torch
            # environments. Keep the original cached moments for every budget;
            # this is only a cross-machine compatibility check, not a fit tolerance.
            if max(matrix_delta, target_delta) > 1e-4:
                raise ValueError("Cached moments failed cross-machine compatibility.")
            cursor += len(b)
        del blocks, block, direct
        runner.atomic_json(output / "cache_reproduction.json",
                           dict(relative_tolerance=1e-4, blocks=reproduction))
        print("Cached moment compatibility: " + json.dumps(reproduction), flush=True)

        n = v.shape[0] // 2
        importance = np.sum(v[:n]**2 + v[n:]**2, axis=1)
        probabilities = .9 * importance / importance.sum() + .1 / n
        samples = np.random.default_rng(418).choice(n, size=4096, p=probabilities)
        regenerated = np.bincount(samples, minlength=n) / (4096 * probabilities)
        np.testing.assert_allclose(regenerated, original_prior, rtol=1e-12, atol=1e-13)
        for budget in BUDGETS:
            prior = np.bincount(samples[:budget], minlength=n) / (budget * probabilities)
            folder = output / f"draws{budget}"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "prior.npy", prior)
            matrix = nested_matrix(source, original_prior, prior)
            fit_rule(folder, matrix, target, prior, runner)
            del matrix

    torch.set_num_threads(1)
    with threadpool_limits(1):
        baseline_validation = [
            rollout_point(output / "baseline", "validation", p,
                          cache / "weights.npy", inputs, runner) for p in range(2)]

    candidates = []
    for budget in BUDGETS:
        folder = output / f"draws{budget}"
        rule = folder / "weights.npy"
        audit_file = folder / "validation_operator_audit.json"
        if not audit_file.exists():
            with (folder / "audit.log").open("w") as stream:
                subprocess.run([
                    sys.executable, str(ROOT / "Project_YvonMaday/audit_case2_hyperreduction.py"),
                    "--output", str(output), "--rule-file", str(rule),
                    "--threads", str(args.offline_threads)],
                    env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        audit = json.loads(audit_file.read_text())
        if audit["weights_sha256"] != runner.sha256(rule) or audit["inputs"] != fingerprints:
            raise ValueError("Audit belongs to different inputs or weights.")
        checks, operators_pass = audit_checks(audit["records"])
        with threadpool_limits(1):
            validation = [rollout_point(folder, "validation", p, rule, inputs, runner)
                          for p in range(2)]
        ratios = [r["coefficient_error_percent"] / ref["coefficient_error_percent"]
                  for r, ref in zip(validation, baseline_validation)]
        passed = bool(operators_pass and np.isfinite(ratios).all()
                      and max(ratios) <= THRESHOLDS["max_validation_error_ratio"])
        info = json.loads((folder / "fit.json").read_text())
        candidate = dict(draws=budget, **info, **checks, validation_error_ratios=ratios,
                         validation_coefficient_errors=[
                             r["coefficient_error_percent"] for r in validation],
                         accepted=passed)
        candidates.append(candidate)
        runner.atomic_json(output / "validation_comparison.json", candidates)
        print("VALIDATION " + json.dumps(candidate), flush=True)

    accepted = [c for c in candidates if c["accepted"]]
    selected = min(accepted, key=lambda c: (c["sampled_cells"], c["draws"])) if accepted else None
    selection = dict(selected_draws=selected["draws"] if selected else 4096,
                     smaller_rule_accepted=bool(selected), uses_reporting_points=False,
                     thresholds=THRESHOLDS, candidates=candidates)
    # Freeze this decision before accessing any reporting trajectories.
    runner.atomic_json(output / "selection.json", selection)
    if not selected:
        print("Neither smaller rule passed validation. Retain the 3532-cell rule.", flush=True)
        return

    reporting = []
    rule = output / f"draws{selected['draws']}/weights.npy"
    with threadpool_limits(1):
        for p in range(4):
            # Same optimized solver for both rules; reporting cannot change selection.
            order = ("baseline", "selected") if p % 2 == 0 else ("selected", "baseline")
            results = {}
            for name in order:
                results[name] = rollout_point(
                    output / name, "reporting", p,
                    cache / "weights.npy" if name == "baseline" else rule, inputs, runner)
            a, b = results["baseline"], results["selected"]
            reporting.append(dict(
                point=p, baseline_cells=3532, selected_cells=selected["sampled_cells"],
                baseline_state_error_percent=a.get("state_error_percent"),
                selected_state_error_percent=b.get("state_error_percent"),
                baseline_coefficient_error_percent=a["coefficient_error_percent"],
                selected_coefficient_error_percent=b["coefficient_error_percent"],
                baseline_seconds=a["online_seconds"], selected_seconds=b["online_seconds"],
                baseline_iterations=a["iterations"], selected_iterations=b["iterations"]))
            runner.atomic_json(output / "reporting_comparison.json", reporting)
    with (output / "reporting_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reporting[0]))
        writer.writeheader()
        writer.writerows(reporting)
    print("REPORTING " + json.dumps(reporting, indent=2), flush=True)


if __name__ == "__main__":
    main()
