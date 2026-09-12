#!/usr/bin/env python3
"""Positive importance cubature control based only on the Euclidean trial basis.

For cell e, ell_e = ||V_tot[e,:]||^2 + ||V_tot[Ncells+e,:]||^2.
p_e = .9 ell_e / ntot + .1 / Ncells; xi_e = count_e / (draws p_e).
For any fixed integrand this estimator is unbiased. Including every basis
coordinate in ell_e protects directions invisible to a primary-only rule.
This is importance sampling, not empirical cubature moment fitting.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from burgers.case2_hyperreduction import SampledBurgers
from Project_YvonMaday.run_case2_local_corrections import GRID_X, GRID_Y, DT, atomic_json, sha256
from Project_YvonMaday.run_case2_hyperreduction import BASIS, DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--seed", type=int, default=418)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(n < 1 for n in args.draws):
        parser.error("draws must be positive")
    with threadpool_limits(2):
        v = np.load(BASIS / "basis.npy")
        n = v.shape[0] // 2
        np.testing.assert_allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-10)
        leverage = np.sum(v[:n]**2 + v[n:]**2, axis=1)
        probabilities = .9 * leverage / leverage.sum() + .1/n
        rng = np.random.default_rng(args.seed)
        samples = rng.choice(n, size=max(args.draws), p=probabilities)
        for draws in args.draws:
            folder = args.output / f"leverage{draws}"
            folder.mkdir(parents=True, exist_ok=True)
            weights = np.bincount(samples[:draws], minlength=n) / (draws*probabilities)
            target = folder / "weights.npy"
            if target.exists():
                np.testing.assert_array_equal(np.load(target), weights)
            else:
                np.save(target, weights)
            mesh = SampledBurgers(GRID_X, GRID_Y, DT, weights)
            weighted_v = v[mesh.residual_indices] * mesh.sqrt_weights[:, None]
            eig = np.linalg.eigvalsh(weighted_v.T @ weighted_v)
            summary = dict(method="positive trial-basis leverage importance cubature", draws=draws,
                           seed=args.seed, leverage_fraction=.9, sampled_cells=len(mesh.samples),
                           stencil_cells=len(mesh.augmented), mass_eigenvalue_min=float(eig[0]),
                           mass_eigenvalue_max=float(eig[-1]), weights_sum=float(weights.sum()),
                           basis_sha256=sha256(BASIS / "basis.npy"), weights_sha256=sha256(target))
            atomic_json(folder / "summary.json", summary)
            print(summary, flush=True)


if __name__ == "__main__":
    main()
