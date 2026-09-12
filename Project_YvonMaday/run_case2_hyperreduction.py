#!/usr/bin/env python3
"""Train and validate positive ECM for residual-adaptive Euclidean Case 2.

Only the nine method-matched baseline linear reduced trajectories enter
cubature training (PROM by default, or HPROM through the campaign override).
Validation and reporting data are loaded separately and only for scoring.
Raw moment blocks are streamed through a reproducible randomized SVD; an
explicit full-moment audit is reported after positive cubature selection.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/case2-hyper-mpl")

import numpy as np
import scipy.linalg as la
import torch
from threadpoolctl import threadpool_limits

from burgers.case2_hyperreduction import SampledBurgers, sampled_affine_step, cell_moments
from burgers.case2_residual_correction import residual_krylov_space
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.core import get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
from Project_YvonMaday.run_case2_local_corrections import (
    PAPER, VALIDATION, REPORTING, DT, NUM_STEPS, GRID_X, GRID_Y, W0,
    predict, score, rollout, sha256, atomic_json,
)
from Project_YvonMaday.run_prom_ann_case_2 import _load_case2_model


def _configured_path(variable, default):
    return Path(os.environ.get(variable, default)).expanduser().resolve()


# These defaults preserve the original PROM experiment.  Environment overrides
# make the identical algebra operate on HPROM-consistent coefficient data.
CAMPAIGN = _configured_path("CASE2_CAMPAIGN_ROOT", PAPER / "euclidean_prom_main")
BASIS = _configured_path("CASE2_BASIS_DIR", PAPER / "MetricStudy/euclidean/Stage1")
MODEL = _configured_path(
    "CASE2_MODEL_PATH",
    CAMPAIGN / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt",
)
TRAINING = CAMPAIGN / "Stage2/prom_coeff_dataset_ntot151/per_mu"
DEFAULT_OUTPUT = _configured_path(
    "CASE2_HYPER_OUTPUT", PAPER / "euclidean_case2_hyperreduction"
)


def reference_path(mu, stage):
    """Return the method-matched full linear trajectory used only for scoring."""
    if stage == "validation":
        parent = CAMPAIGN / "Stage2/prom_coeff_dataset_ntot151_validation2/per_mu"
    else:
        parent = CAMPAIGN / "Runs/Linear"
    if not parent.is_dir():
        raise FileNotFoundError(f"Missing linear-reference directory: {parent}")
    for path in parent.iterdir():
        mu_path = path / "mu.npy"
        q_path = path / "qN.npy"
        if mu_path.exists() and q_path.exists() and np.allclose(
            np.load(mu_path).reshape(-1), mu, rtol=0, atol=1e-10
        ):
            return q_path
    raise FileNotFoundError(f"No linear reference in {parent} for mu={mu}")


def load_inputs():
    model, ntot, kind, checkpoint = _load_case2_model(MODEL, "cpu")
    v = np.asfortranarray(np.load(BASIS / "basis.npy"))
    uref = np.load(BASIS / "u_ref.npy").reshape(-1)
    if ntot != 151 or kind != "q_tot":
        raise ValueError("Expected the baseline 151-coordinate master ANN")
    if not np.allclose(v.T @ v, np.eye(ntot), rtol=0, atol=1e-10):
        raise ValueError("Expected the Euclidean-orthonormal POD basis")
    metadata_path = CAMPAIGN / "Stage2/prom_coeff_dataset_ntot151/meta.json"
    metadata = json.loads(metadata_path.read_text())
    dataset_backend = str(metadata.get("solve_backend", "")).lower()
    checkpoint_backend = str(checkpoint.get("dataset_backend", "")).lower()
    if dataset_backend not in ("prom", "hprom") or checkpoint_backend != dataset_backend:
        raise ValueError(
            "Case-2 model and coefficient dataset use different backends: "
            f"checkpoint={checkpoint_backend!r}, dataset={dataset_backend!r}"
        )
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for p in (
        MODEL, BASIS / "basis.npy", BASIS / "u_ref.npy", metadata_path)}
    return model, v, uref, fingerprints


def training_paths():
    paths = sorted(p for p in TRAINING.iterdir() if (p / "qN.npy").exists())
    expected = {(a, b) for a in (4.25, 4.875, 5.5) for b in (.015, .0225, .03)}
    actual = {tuple(np.load(p / "mu.npy").reshape(-1)) for p in paths}
    if len(paths) != 9 or actual != expected:
        raise ValueError("Cubature training must use exactly the baseline 3x3 parameters")
    return paths


def moments(model, v, uref, times):
    dx, dy, jdx, jdy, eye = get_ops(GRID_X, GRID_Y)
    vp, vs = v[:, :10], v[:, 10:]
    for path in training_paths():
        mu = np.load(path / "mu.npy").reshape(-1)
        teacher = np.load(path / "qN.npy")
        predictions = predict(model, mu, NUM_STEPS)
        for k in times:
            # A deployment predictor with a method-matched linear predecessor.
            # No HDM states, additional simulations, or reporting labels enter.
            previous = uref + v @ teacher[:, k - 1]
            predictor = uref + vp @ teacher[:10, k - 1] + vs @ predictions[10:, k]
            residual = inviscid_burgers_res2D(predictor, GRID_X, GRID_Y, DT, previous, mu, dx, dy)
            jacobian = inviscid_burgers_exact_jac2D(predictor, DT, jdx, jdy, eye)
            b = residual_krylov_space(jacobian, vp, vs, residual, 3)
            transform = np.zeros((151, 10 + b.shape[1]))
            transform[:10, :10] = np.eye(10)
            transform[10:, 10:] = b
            yield path.name, k, cell_moments(jacobian @ v, residual, transform)


def train(args, model, v, uref, fingerprints):
    folder = args.output / f"rule_s{args.sketch_rank}"
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(inputs=fingerprints, sketch_rank=args.sketch_rank,
                  seed=args.seed, training_steps=args.training_steps,
                  nprimary=10, adaptive_rank=3, dt=DT,
                  moment_types=["full_coordinate_gradient", "adaptive_Gram", "residual_energy"],
                  training_sources={str(p.relative_to(ROOT)): sha256(p / "qN.npy")
                                    for p in training_paths()},
                  implementation={str(p.relative_to(ROOT)): sha256(p) for p in (
                      Path(__file__), ROOT / "burgers/case2_hyperreduction.py",
                      ROOT / "burgers/case2_residual_correction.py")})
    config_path = folder / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Rule configuration changed; use a separate output folder: {folder}")
        if (folder / "summary.json").exists():
            print(f"Completed rule, skipping: {folder}", flush=True)
            return
    else:
        atomic_json(config_path, config)
    start = time.perf_counter()
    ncells = v.shape[0] // 2
    sketch_file = folder / "entity_range.npy"
    if sketch_file.exists():
        q = np.load(sketch_file)
    else:
        rng = np.random.default_rng(args.seed)
        sketch = np.zeros((ncells, args.sketch_rank + 20))
        for tag, k, blocks in moments(model, v, uref, args.training_steps):
            for block in blocks:
                sketch += block @ rng.standard_normal((block.shape[1], sketch.shape[1]))
            print(f"range: {tag} step={k}, elapsed={time.perf_counter()-start:.1f}s", flush=True)
        q, _ = la.qr(sketch, mode="economic", check_finite=False)
        np.save(sketch_file, q)
    compressed_file = folder / "compressed_covariance.npy"
    if compressed_file.exists():
        covariance = np.load(compressed_file)
    else:
        covariance = np.zeros((q.shape[1], q.shape[1]))
        for tag, k, blocks in moments(model, v, uref, args.training_steps):
            for block in blocks:
                compressed = q.T @ block
                covariance += compressed @ compressed.T
            print(f"compression: {tag} step={k}", flush=True)
        np.save(compressed_file, covariance)
    values, vectors = la.eigh(covariance)
    order = np.argsort(values)[::-1][:args.sketch_rank]
    basis = np.ascontiguousarray(q @ vectors[:, order])
    weights_file = folder / "weights.npy"
    if weights_file.exists():
        weights = np.load(weights_file)
    else:
        # Include the constant in the orthonormal basis explicitly, avoiding
        # a nearly zero constant-complement vector inside the ECM class.
        augmented, _ = la.qr(np.column_stack((np.ones(ncells)/np.sqrt(ncells), basis)),
                            mode="economic", check_finite=False)
        selector = EmpiricalCubatureMethod(ECM_tolerance=1e-10)
        selector.SetUp(augmented, constrain_sum_of_weights=False)
        with (folder / "ecm.log").open("w") as stream, contextlib.redirect_stdout(stream):
            selector.Run()
        weights = np.zeros(ncells)
        weights[np.asarray(selector.z).reshape(-1)] = np.asarray(selector.w).reshape(-1)
        if (not np.isfinite(weights).all() or np.any(weights < 0)
                or abs(weights.sum()/ncells - 1) > 1e-6):
            raise ValueError("ECM returned an invalid positive, constant-exact rule")
        np.save(weights_file, weights)
    numerator = np.zeros(3)
    denominator = np.zeros(3)
    worst = np.zeros(3)
    for _, _, blocks in moments(model, v, uref, args.training_steps):
        for i, block in enumerate(blocks):
            target = block.sum(axis=0)
            error = weights @ block - target
            numerator[i] += error @ error
            denominator[i] += target @ target
            worst[i] = max(worst[i], np.linalg.norm(error)/max(np.linalg.norm(target), 1e-30))
    mesh = SampledBurgers(GRID_X, GRID_Y, DT, weights)
    summary = dict(config=config, sampled_cells=len(mesh.samples), stencil_cells=len(mesh.augmented),
                   moment_relative_errors=np.sqrt(numerator/denominator).tolist(),
                   worst_anchor_relative_errors=worst.tolist(),
                   weights_sum=float(weights.sum()), min_positive_weight=float(weights[weights > 0].min()),
                   offline_seconds=time.perf_counter()-start,
                   weights_sha256=sha256(weights_file))
    atomic_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def hyper_rollout(model, v, uref, mu, mesh, rank, steps):
    idx = mesh.state_indices
    vp, vs = np.asfortranarray(v[idx, :10]), np.asfortranarray(v[idx, 10:])
    # Fixed deployment data; the full-dimensional operations stop here.
    qinitial = v.T @ (W0 - uref)
    local_ref = uref[idx]
    start = time.perf_counter()
    predictions = predict(model, mu, steps)
    offsets = np.asfortranarray(local_ref[:, None] + vs @ predictions[10:])
    q = predictions.copy()
    q[:, 0] = qinitial
    primary = qinitial[:10].copy()
    current = local_ref + v[idx] @ qinitial
    iterations, ranks, norms = [], [], []
    for k in range(1, steps + 1):
        z, b, current, count, norm = sampled_affine_step(
            mesh, vp, vs, offsets[:, k], current, mu, primary, rank)
        primary = z[:10]
        q[:10, k] = primary
        q[10:, k] += b @ z[10:]
        iterations.append(count)
        ranks.append(b.shape[1])
        norms.append(norm)
        if k % 100 == 0 or k == steps:
            print(f"HPROM B{rank} mu={mu}, step={k}, elapsed={time.perf_counter()-start:.1f}s", flush=True)
    return q, dict(online_seconds=time.perf_counter()-start,
                   iterations=int(sum(iterations)), min_rank=int(min(ranks)), max_rank=int(max(ranks)),
                   mean_weighted_residual_norm=float(np.mean(norms)))


def evaluate(args, model, v, uref, fingerprints):
    points = VALIDATION if args.stage == "validation" else REPORTING
    for kind in args.methods:
        mesh = None
        if kind.startswith("hprom"):
            weights_file = args.rule_file or args.output / f"rule_s{args.sketch_rank}" / "weights.npy"
            mesh = SampledBurgers(GRID_X, GRID_Y, DT, np.load(weights_file))
        for index in args.point_indices if args.point_indices is not None else range(len(points)):
            mu = points[index]
            rule_tag = args.rule_file.parent.name if args.rule_file else f"s{args.sketch_rank}"
            folder = args.output / f"{args.stage}_{rule_tag}_{kind}_p{index}_steps{args.steps}"
            folder.mkdir(parents=True, exist_ok=True)
            config = dict(inputs=fingerprints, mu=list(mu), method=kind, steps=args.steps,
                          initialization="known_linear", threads=args.threads,
                          weights_sha256=sha256(weights_file) if mesh else None,
                          implementation={str(p.relative_to(ROOT)): sha256(p) for p in (
                              Path(__file__), ROOT / "burgers/case2_hyperreduction.py",
                              ROOT / "burgers/case2_residual_correction.py")})
            target = folder / "summary.json"
            if target.exists():
                if json.loads(target.read_text())["config"] != config:
                    raise ValueError(f"Run configuration changed: {folder}")
                print(f"Completed run, skipping: {folder.name}", flush=True)
                continue
            if mesh:
                q, stats = hyper_rollout(model, v, uref, mu, mesh, int(kind[-1]), args.steps)
                stats.update(sampled_cells=len(mesh.samples), stencil_cells=len(mesh.augmented))
            else:
                q, stats = rollout(model, v, uref, mu, "baseline" if kind == "prom0" else "krylov3",
                                   args.steps, 10, initialization="linear")
            np.save(folder / "qN.npy", q)
            metrics = score(q, v, uref, mu, reference_path(mu, args.stage))
            atomic_json(target, dict(config=config, **stats, **metrics))
            print(json.dumps(dict(method=kind, point=index, **stats, **metrics)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validation", "reporting"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sketch-rank", type=int, default=256)
    parser.add_argument("--rule-file", type=Path, help="Use an explicitly supplied positive cell rule")
    parser.add_argument("--seed", type=int, default=418)
    parser.add_argument("--training-steps", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100, 200, 350, 500])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--methods", nargs="+", choices=("hprom0", "hprom3", "prom0", "prom3"), default=["hprom0", "hprom3"])
    parser.add_argument("--point-indices", type=int, nargs="+")
    args = parser.parse_args()
    if not (1 <= args.steps <= 500 and args.threads > 0 and 1 <= args.sketch_rank <= 2000
            and all(1 <= k <= 500 for k in args.training_steps)):
        parser.error("Invalid steps, thread count or sketch rank")
    if args.point_indices is not None and not all(0 <= i < (2 if args.stage == "validation" else 4) for i in args.point_indices):
        parser.error("Invalid point index")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        inputs = load_inputs()
        if args.stage == "train":
            train(args, *inputs)
        else:
            evaluate(args, *inputs)


if __name__ == "__main__":
    main()
