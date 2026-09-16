#!/usr/bin/env python3
"""Full-mesh ECM experiment for B3; no changes to accepted manuscript rules.

Compression uses an adaptive randomized range and a measured Frobenius error.
ECM sees all mesh cells, not a preselected importance-sampling support.
Validation alone selects a rule; reporting follows a frozen selection.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def squared_norm(matrix, batch=1024):
    return sum(float(np.einsum("ij,ij->", matrix[:, j:j+batch],
                              matrix[:, j:j+batch]))
               for j in range(0, matrix.shape[1], batch))


def rank_for_tolerance(eigenvalues, total_squared, tolerance):
    """Include projection loss outside Q, not merely the compressed SVD tail."""
    remaining = np.maximum(0., total_squared - np.cumsum(eigenvalues))
    admissible = np.flatnonzero(remaining <= tolerance**2 * total_squared)
    if not len(admissible):
        raise ValueError("Range does not meet the requested full-matrix tolerance")
    rank = int(admissible[0] + 1)
    return rank, float(np.sqrt(remaining[rank-1] / total_squared))


def range_extension(matrix, q, width, seed):
    """Block randomized range with one power iteration and reorthogonalization."""
    def orthogonalize(y):
        if q.shape[1]:
            for _ in range(2):
                y -= q @ (q.T @ y)
        return la.qr(y, mode="economic", check_finite=False)[0]

    rng = np.random.default_rng(seed + q.shape[1])
    y = orthogonalize(matrix @ rng.standard_normal((matrix.shape[1], width)))
    z = la.qr(matrix.T @ y, mode="economic", check_finite=False)[0]
    y = orthogonalize(matrix @ z)
    return y, y.T @ matrix


def ecm_rule(basis, tolerance, log):
    from burgers.empirical_cubature_method import EmpiricalCubatureMethod

    # The constant is explicitly part of the orthonormal integration basis.
    # Avoid the legacy selector's division by a vanishing constant complement.
    constant = np.ones((basis.shape[0], 1)) / np.sqrt(basis.shape[0])
    augmented = la.qr(np.column_stack((constant, basis)), mode="economic",
                      check_finite=False)[0]
    selector = EmpiricalCubatureMethod(ECM_tolerance=tolerance)
    selector.SetUp(augmented, constrain_sum_of_weights=False)
    with log.open("w") as stream, contextlib.redirect_stdout(stream):
        selector.Run()
    support = np.atleast_1d(selector.z).astype(int)
    values = np.asarray(selector.w).reshape(-1)
    if len(np.unique(support)) != len(support) or not np.isfinite(values).all():
        raise ValueError("Invalid ECM support or weights")
    if np.any(values <= 0):
        raise ValueError("ECM returned nonpositive retained weights")
    weights = np.zeros(len(basis))
    weights[support] = values
    target = augmented.sum(axis=0)
    error = float(la.norm(augmented.T @ weights - target) / la.norm(target))
    if error > max(10 * tolerance, 1e-7):
        raise ValueError(f"ECM compressed integration error {error} exceeds tolerance")
    return weights, error


def moment_errors(matrix, target, weights, blocks):
    error = matrix.T @ weights - target
    stats = {}
    for kind in sorted({b["kind"] for b in blocks}):
        records = [b for b in blocks if b["kind"] == kind]
        numerator = sum(float(error[b["start"]:b["stop"]] @
                              error[b["start"]:b["stop"]]) for b in records)
        denominator = sum(float(target[b["start"]:b["stop"]] @
                                target[b["start"]:b["stop"]]) for b in records)
        stats[kind] = float(np.sqrt(numerator / max(denominator, 1e-30)))
    return stats


def build_moments(output, inputs, runner, cache):
    model, v, uref, fingerprints = inputs
    source = json.loads((cache / "config.json").read_text())
    sources = {str(p.relative_to(ROOT)): runner.sha256(p / "qN.npy")
               for p in runner.training_paths()}
    if source["inputs"] != fingerprints or source["training_sources"] != sources:
        raise ValueError("Accepted moment cache uses different training inputs")
    protocol = dict(inputs=fingerprints, training_sources=sources,
                    training_steps=source["training_steps"],
                    candidate_cells="all mesh cells", adaptive_rank=3,
                    scaling="same dynamic and mass block scaling as positive_fit4096")
    manifest = output / "moments.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["protocol"] != protocol:
            raise ValueError("Moment inputs changed")
        return previous
    ncells, ntot = v.shape[0] // 2, v.shape[1]
    # Three adaptive directions are checked at every anchor before assembly.
    count = 9 * len(source["training_steps"])
    ncolumns = count * (ntot + 13 * 14 // 2 + 1) + ntot * (ntot + 1) // 2
    matrix = np.lib.format.open_memmap(output / "moments.npy", mode="w+",
                                      dtype="float64", shape=(ncells, ncolumns),
                                      fortran_order=True)
    blocks, cursor = [], 0
    parent_prior = np.load(cache.parent / "leverage4096/weights.npy")
    if runner.sha256(cache.parent / "leverage4096/weights.npy") != source["initial_rule_sha256"]:
        raise ValueError("Parent moment scaling changed")
    support = np.flatnonzero(parent_prior)
    cached = np.load(cache / "matrix.npy", mmap_mode="r")
    discrepancies = []
    start = time.perf_counter()
    for tag, step, values in runner.moments(model, v, uref, source["training_steps"]):
        if [a.shape[1] for a in values] != [ntot, 91, 1]:
            raise ValueError("Unexpected offline adaptive rank")
        for kind, value in zip(("gradient", "adaptive_Gram", "residual_energy"), values):
            value /= max(la.norm(value.sum(axis=0)), 1e-12)
            stop = cursor + value.shape[1]
            matrix[:, cursor:stop] = value
            direct = (value[support] * parent_prior[support, None]).T
            prior = cached[cursor:stop]
            discrepancy = float(la.norm(direct-prior) / max(la.norm(prior), 1e-30))
            if discrepancy > 1e-4:
                raise ValueError(f"Moment cache reproduction failed: {discrepancy}")
            discrepancies.append(discrepancy)
            blocks.append(dict(kind=kind, start=cursor, stop=stop, parameter=tag, step=step))
            cursor = stop
        print(f"moments {tag} step={step} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    i, j = np.triu_indices(ntot)
    scale = np.sqrt(3 * count / ntot)
    start_mass = cursor
    for first in range(0, len(i), 128):
        ii, jj = i[first:first+128], j[first:first+128]
        gram = v[:ncells, ii]*v[:ncells, jj] + v[ncells:, ii]*v[ncells:, jj]
        gram[:, ii != jj] *= np.sqrt(2.)
        gram *= scale
        matrix[:, cursor:cursor+len(ii)] = gram
        cursor += len(ii)
    blocks.append(dict(kind="mass", start=start_mass, stop=cursor))
    if cursor != ncolumns:
        raise ValueError("Moment matrix size mismatch")
    matrix.flush()
    target = matrix.sum(axis=0)
    cached_target = np.load(cache / "target.npy")
    delta = float(la.norm(target-cached_target)/la.norm(cached_target))
    if delta > 1e-4:
        raise ValueError("Full target differs from the accepted fit")
    np.save(output / "target.npy", target)
    info = dict(protocol=protocol, shape=list(matrix.shape), blocks=blocks,
                maximum_cache_relative_error=max(discrepancies), target_cache_error=delta,
                total_squared=squared_norm(matrix), elapsed_seconds=time.perf_counter()-start)
    runner.atomic_json(manifest, info)
    return info


def compress(output, matrix, info, tolerance, args, runner):
    directory = output / "compression"
    directory.mkdir(exist_ok=True)
    manifest = directory / "progress.json"
    q = np.empty((matrix.shape[0], 0))
    projected = np.empty((0, matrix.shape[1]))
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta["seed"] != args.seed:
            raise ValueError("Compression seed changed")
        q = np.load(directory / "q.npy")
        projected = np.load(directory / "projected.npy")
        if q.shape[1] != meta["rank"] or projected.shape[0] != meta["rank"]:
            raise ValueError("Incomplete compression checkpoint")
    total = info["total_squared"]
    remaining = max(0., total-squared_norm(projected))
    while remaining > (.7*tolerance)**2 * total:
        width = min(args.block_size, args.max_rank-q.shape[1],
                    min(matrix.shape)-q.shape[1])
        if width <= 0:
            raise RuntimeError("Compression memory/rank cap reached; increase --max-rank to resume")
        started = time.perf_counter()
        new, coefficients = range_extension(matrix, q, width, args.seed)
        q = np.column_stack((q, new))
        projected = np.vstack((projected, coefficients))
        remaining = max(0., total-squared_norm(projected))
        np.save(directory / "q.npy", q)
        np.save(directory / "projected.npy", projected)
        runner.atomic_json(manifest, dict(rank=q.shape[1], seed=args.seed,
                           relative_projection_error=float(np.sqrt(remaining/total))))
        print(f"compression rank={q.shape[1]} relative_error={np.sqrt(remaining/total):.6g} "
              f"block_seconds={time.perf_counter()-started:.1f}", flush=True)
    eigenvalues, vectors = la.eigh(projected @ projected.T, check_finite=False)
    eigenvalues, vectors = np.maximum(eigenvalues[::-1], 0), vectors[:, ::-1]
    rank, error = rank_for_tolerance(eigenvalues, total, tolerance)
    basis = q @ vectors[:, :rank]
    # This additional direct check includes the complete, uncompressed matrix.
    measured = np.sqrt(max(0., total-squared_norm(basis.T @ matrix))/total)
    if measured > tolerance * (1+1e-7):
        raise ValueError("Full-matrix compression check failed")
    return basis, dict(rank=rank, range_rank=q.shape[1], relative_frobenius_error=float(measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paper = ROOT / "Project_YvonMaday/Results_Paper"
    parser.add_argument("--output", type=Path, default=paper / "euclidean_b3_ecm_local")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[.1, .05, .02])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-rank", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=418)
    parser.add_argument("--extend-rejected", action="store_true",
                        help="Append stricter tolerances before reporting, archiving the rejected protocol")
    parser.add_argument("--stage", choices=["moments", "all"], default="all")
    args = parser.parse_args()
    if (args.threads < 1 or args.block_size < 1 or args.max_rank < 1
            or any(not 0 < t < 1 for t in args.tolerances)):
        parser.error("Invalid threads, rank limit, or tolerances")
    campaign = paper / "euclidean_hprom_main"
    os.environ["CASE2_CAMPAIGN_ROOT"] = str(campaign)
    os.environ["CASE2_MODEL_PATH"] = str(campaign / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt")
    os.environ["CASE2_BASIS_DIR"] = str(paper / "MetricStudy/euclidean/Stage1")
    os.environ["CASE2_TRAINING_DATASET"] = str(campaign / "Stage2/prom_coeff_dataset_ntot151")
    os.environ["CASE2_REFERENCE_CAMPAIGN_ROOT"] = str(campaign)
    import torch
    from threadpoolctl import threadpool_limits
    from Project_YvonMaday import run_case2_hyperreduction as runner
    from Project_YvonMaday.study_case2_b3_cubature_budget import audit_checks, rollout_point

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cache = campaign / "Stage4/case2_b3/positive_fit4096"
    comparator = paper / "euclidean_b3_budget_local/draws2048/weights.npy"
    with threadpool_limits(args.threads):
        inputs = runner.load_inputs()
        info = build_moments(output, inputs, runner, cache)
    if args.stage == "moments":
        return
    matrix = np.load(output / "moments.npy", mmap_mode="r")
    target = np.load(output / "target.npy")
    protocol = dict(inputs=inputs[3], comparator_sha256=runner.sha256(comparator),
                    tolerances=sorted(set(args.tolerances), reverse=True), seed=args.seed,
                    maximum_validation_coefficient_ratio=1.05,
                    selection_uses_reporting_points=False, ecm_tolerance=1e-8)
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        old = json.loads(protocol_file.read_text())
        selection_path = output / "selection.json"
        previous = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        old_settings = {k: v for k, v in old.items() if k != "tolerances"}
        new_settings = {k: v for k, v in protocol.items() if k != "tolerances"}
        if (not args.extend_rejected or old_settings != new_settings
                or protocol["tolerances"][:len(old["tolerances"])] != old["tolerances"]
                or previous.get("accepted") is not False
                or (output / "reporting_comparison.json").exists()
                or (output / "selected").exists()):
            raise ValueError("Study protocol changed; use a separate output")
        archive = output / f"rejected_protocol_{len(old['tolerances'])}.json"
        if archive.exists():
            raise ValueError("Protocol archive already exists")
        runner.atomic_json(archive, dict(protocol=old, selection=previous,
                           reason="Stricter compression after validation rejection; no reporting used"))
    runner.atomic_json(protocol_file, protocol)
    candidates = []
    selected = None
    for tolerance in protocol["tolerances"]:
        folder = output / f"ecm_tol{tolerance:g}"
        folder.mkdir(exist_ok=True)
        rule = folder / "weights.npy"
        with threadpool_limits(args.threads):
            if not (folder / "fit.json").exists():
                started = time.perf_counter()
                basis, compression = compress(output, matrix, info, tolerance, args, runner)
                print(f"ECM tolerance={tolerance} rank={basis.shape[1]}", flush=True)
                weights, compressed_error = ecm_rule(basis, 1e-8, folder / "ecm.log")
                del basis
                np.save(rule, weights)
                mesh = runner.SampledBurgers(runner.GRID_X, runner.GRID_Y, runner.DT, weights)
                runner.atomic_json(folder / "fit.json", dict(
                    **compression, tolerance=tolerance, compressed_error=compressed_error,
                    sampled_cells=len(mesh.samples), stencil_cells=len(mesh.augmented),
                    moment_relative_errors=moment_errors(matrix, target, weights, info["blocks"]),
                    weights_sha256=runner.sha256(rule), local_offline_seconds=time.perf_counter()-started))
        if not (folder / "validation_operator_audit.json").exists():
            with (folder / "audit.log").open("w") as stream:
                subprocess.run([sys.executable, str(ROOT / "Project_YvonMaday/audit_case2_hyperreduction.py"),
                                "--output", str(output), "--rule-file", str(rule),
                                "--threads", str(args.threads)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        audit = json.loads((folder / "validation_operator_audit.json").read_text())
        fit = json.loads((folder / "fit.json").read_text())
        if audit["weights_sha256"] != runner.sha256(rule) or audit["inputs"] != inputs[3]:
            raise ValueError("Audit provenance mismatch")
        checks, passed = audit_checks(audit["records"])
        candidate = dict(**fit, **checks, operator_pass=passed, accepted=False)
        if passed:
            torch.set_num_threads(1)
            with threadpool_limits(1):
                reference = [rollout_point(output / "reference2048", "validation", p,
                             comparator, inputs, runner) for p in range(2)]
                validation = [rollout_point(folder, "validation", p, rule, inputs, runner)
                              for p in range(2)]
            ratios = [v["coefficient_error_percent"]/r["coefficient_error_percent"]
                      for v, r in zip(validation, reference)]
            candidate.update(validation_coefficient_ratios=ratios,
                             accepted=bool(np.isfinite(ratios).all() and max(ratios) <= 1.05))
            torch.set_num_threads(args.threads)
        candidates.append(candidate)
        runner.atomic_json(output / "validation_comparison.json", candidates)
        print("VALIDATION " + json.dumps(candidate), flush=True)
        if candidate["accepted"]:
            selected = rule
            break
    selection = dict(accepted=selected is not None, selection_uses_reporting_points=False,
                     rule_file=str(selected) if selected else None,
                     weights_sha256=runner.sha256(selected) if selected else None,
                     candidates=candidates, protocol=protocol)
    # This decision is written before any reporting trajectory is run.
    runner.atomic_json(output / "selection.json", selection)
    if selected is None:
        print("No ECM rule passed validation. Reporting points were not used.", flush=True)
        return
    torch.set_num_threads(1)
    reporting = []
    with threadpool_limits(1):
        for point in range(4):
            record = {}
            for name, rule in (("reference2048", comparator), ("selected", selected)):
                record[name] = rollout_point(output / name, "reporting", point, rule, inputs, runner)
            reporting.append(record)
            runner.atomic_json(output / "reporting_comparison.json", reporting)
    print("Complete. Local runtimes are diagnostic, not Sherlock timing results.", flush=True)


if __name__ == "__main__":
    main()
