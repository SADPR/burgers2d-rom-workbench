#!/usr/bin/env python3
"""Audit cubature sensitivities on the two held-out regression parameters."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.linalg as la
import torch
from threadpoolctl import threadpool_limits

from burgers.case2_hyperreduction import SampledBurgers
from burgers.case2_residual_correction import residual_krylov_space
from burgers.core import get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
from Project_YvonMaday.run_case2_hyperreduction import (
    load_inputs, reference_path, DEFAULT_OUTPUT,
)
from Project_YvonMaday.run_case2_local_corrections import (
    VALIDATION, DT, GRID_X, GRID_Y, predict, atomic_json, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sketch-rank", type=int, default=256)
    parser.add_argument("--rule-file", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        model, v, uref, fingerprints = load_inputs()
        weights_file = args.rule_file or args.output / f"rule_s{args.sketch_rank}" / "weights.npy"
        mesh = SampledBurgers(GRID_X, GRID_Y, DT, np.load(weights_file))
        vp, vs = v[:, :10], v[:, 10:]
        local_v = v[mesh.state_indices]
        dx, dy, jdx, jdy, eye = get_ops(GRID_X, GRID_Y)
        records = []
        for point, mu in enumerate(VALIDATION):
            teacher = np.load(reference_path(mu, "validation"))
            predictions = predict(model, mu, 500)
            for k in [3, 15, 75, 175, 275, 425, 500]:
                previous = uref + v @ teacher[:, k-1]
                candidate = uref + vp @ teacher[:10, k-1] + vs @ predictions[10:, k]
                residual = inviscid_burgers_res2D(candidate, GRID_X, GRID_Y, DT, previous, mu, dx, dy)
                jacobian = inviscid_burgers_exact_jac2D(candidate, DT, jdx, jdy, eye)
                a = jacobian @ v
                f, j = mesh.step_operators(previous[mesh.state_indices], mu)
                sampled_r = f(candidate[mesh.state_indices])
                sampled_j = j(candidate[mesh.state_indices])
                sampled_a = sampled_j @ local_v
                spectrum = la.eigvalsh(sampled_a.T @ sampled_a, a.T @ a)
                record = dict(point=point, mu=list(mu), step=k,
                              Gram_eigenvalue_min=float(spectrum[0]),
                              Gram_eigenvalue_max=float(spectrum[-1]),
                              gradient_relative_error=float(np.linalg.norm(sampled_a.T @ sampled_r-a.T @ residual)
                                                            / np.linalg.norm(a.T @ residual)))
                for label, b in (
                    ("full", residual_krylov_space(jacobian, vp, vs, residual, 3)),
                    ("sampled", residual_krylov_space(sampled_j, local_v[:, :10], local_v[:, 10:], sampled_r, 3))):
                    transform = np.zeros((151, 10+b.shape[1]))
                    transform[:10, :10] = np.eye(10)
                    transform[10:, 10:] = b
                    solve_a, solve_r = (a, residual) if label == "full" else (sampled_a, sampled_r)
                    update = transform @ np.linalg.lstsq(solve_a @ transform, -solve_r, rcond=None)[0]
                    record[label + "_true_linearized_residual"] = float(np.linalg.norm(residual + a @ update))
                records.append(record)
                print(json.dumps(record), flush=True)
        atomic_json(weights_file.parent / "validation_operator_audit.json",
                    dict(inputs=fingerprints, weights_sha256=sha256(weights_file), records=records))


if __name__ == "__main__":
    main()
