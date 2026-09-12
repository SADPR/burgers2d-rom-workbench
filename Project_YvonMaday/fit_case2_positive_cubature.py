#!/usr/bin/env python3
"""Nonnegative moment fitting on a trial-basis importance-sampling support.

This is a convex nonnegative least-squares problem. In addition to the dynamic
moments, it fits the complete Euclidean V_tot.T V_tot matrix, so that secondary
directions are explicitly represented. L-BFGS-B uses exact objective gradients;
the resulting quadrature accuracy is measured independently of its exit code.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/case2-hyper-mpl")

import numpy as np
from scipy.optimize import minimize
import torch
from threadpoolctl import threadpool_limits

from burgers.case2_hyperreduction import SampledBurgers
from Project_YvonMaday.run_case2_hyperreduction import load_inputs, moments, DEFAULT_OUTPUT, training_paths
from Project_YvonMaday.run_case2_local_corrections import DT, GRID_X, GRID_Y, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--training-steps", nargs="+", type=int, default=[1, 2, 5, 10, 25, 50, 100, 200, 350, 500])
    args = parser.parse_args()
    if args.maxiter < 1 or args.threads < 1 or not all(1 <= k <= 500 for k in args.training_steps):
        parser.error("Invalid iteration count, threads or training steps")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        model, v, uref, fingerprints = load_inputs()
        initial_file = args.output / f"leverage{args.draws}" / "weights.npy"
        initial = np.load(initial_file)
        support = np.flatnonzero(initial > 0)
        prior = initial[support]
        folder = args.output / f"positive_fit{args.draws}"
        folder.mkdir(parents=True, exist_ok=True)
        config = dict(inputs=fingerprints, initial_rule_sha256=sha256(initial_file),
                      training_steps=args.training_steps, maxiter=args.maxiter,
                      training_sources={str(p.relative_to(ROOT)): sha256(p / "qN.npy") for p in training_paths()},
                      implementation={str(p.relative_to(ROOT)): sha256(p) for p in (
                          Path(__file__), ROOT / "burgers/case2_hyperreduction.py",
                          ROOT / "Project_YvonMaday/run_case2_hyperreduction.py")})
        if (folder / "config.json").exists():
            if json.loads((folder / "config.json").read_text()) != config:
                raise ValueError("Different fitting configuration; use a separate output root")
            if (folder / "summary.json").exists():
                print(f"Complete, skipping {folder}")
                return
        else:
            atomic_json(folder / "config.json", config)
        start = time.perf_counter()
        if (folder / "matrix.npy").exists() and (folder / "target.npy").exists():
            matrix = np.load(folder / "matrix.npy")
            target = np.load(folder / "target.npy")
        else:
            matrices, targets = [], []
            for tag, k, blocks in moments(model, v, uref, args.training_steps):
                for block in blocks:
                    target = block.sum(axis=0)
                    scale = max(np.linalg.norm(target), 1e-12)
                    matrices.append(np.ascontiguousarray((block[support] * prior[:, None]).T / scale))
                    targets.append(target / scale)
                print(f"moments: {tag}, step={k}, elapsed={time.perf_counter()-start:.1f}s", flush=True)
            ncells = v.shape[0] // 2
            i, j = np.triu_indices(151)
            top, bottom = v[support], v[ncells + support]
            gram = top[:, i]*top[:, j] + bottom[:, i]*bottom[:, j]
            gram[:, i != j] *= np.sqrt(2.)
            target = np.eye(151)[i, j]
            # Repeat the shared mass condition once per dynamic block in the
            # averaged least-squares objective, without materializing copies.
            scale = np.sqrt(3*9*len(args.training_steps)) / np.sqrt(151.)
            matrices.append(np.ascontiguousarray((gram * prior[:, None]).T * scale))
            targets.append(target * scale)
            matrix, target = np.vstack(matrices), np.concatenate(targets)
            del matrices, targets, gram
            np.save(folder / "matrix.npy", matrix)
            np.save(folder / "target.npy", target)

        def objective(x):
            residual = matrix @ x - target
            return .5 * float(residual @ residual), matrix.T @ residual

        evaluations = [0]

        def progress(x):
            evaluations[0] += 1
            if evaluations[0] % 50 == 0:
                print(f"NNLS iteration={evaluations[0]}, objective={objective(x)[0]:.6g}, elapsed={time.perf_counter()-start:.1f}s", flush=True)

        result = minimize(objective, np.ones(support.size), jac=True, method="L-BFGS-B",
                          bounds=[(0., None)]*support.size, callback=progress,
                          options=dict(maxiter=args.maxiter, ftol=1e-12, gtol=1e-8, maxcor=20))
        weights = np.zeros(v.shape[0] // 2)
        weights[support] = prior * result.x
        np.save(folder / "weights.npy", weights)
        mesh = SampledBurgers(GRID_X, GRID_Y, DT, weights)
        weighted_v = v[mesh.residual_indices] * mesh.sqrt_weights[:, None]
        eig = np.linalg.eigvalsh(weighted_v.T @ weighted_v)
        gradient = objective(result.x)[1]
        projected_gradient = np.where(result.x > 0, gradient, np.minimum(gradient, 0))
        summary = dict(config=config, method="nonnegative dynamic and complete mass moment fitting",
                       optimizer_success=bool(result.success), optimizer_message=str(result.message),
                       optimizer_iterations=int(result.nit), projected_gradient_inf=float(np.max(np.abs(projected_gradient))),
                       objective_before=float(objective(np.ones(support.size))[0]), objective_after=float(result.fun),
                       sampled_cells=len(mesh.samples), stencil_cells=len(mesh.augmented),
                       mass_eigenvalue_min=float(eig[0]), mass_eigenvalue_max=float(eig[-1]),
                       offline_seconds=time.perf_counter()-start, weights_sha256=sha256(folder / "weights.npy"))
        atomic_json(folder / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
