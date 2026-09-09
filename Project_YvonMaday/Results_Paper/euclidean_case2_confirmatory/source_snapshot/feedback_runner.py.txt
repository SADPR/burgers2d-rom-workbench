#!/usr/bin/env python3
"""Test a ridge-fitted affine innovation feedback using the existing master ANN.

    tail = M_s(mu,t) + K (q - M_p(mu,t))
    tangent = Vp + Vs K

K is fitted only on nine training PROM trajectories, never on validation or
reporting data. This changes the trial manifold but adds no online unknowns.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from Project_YvonMaday.run_case2_local_corrections import (
    PAPER, CAMPAIGN, BASIS, MODEL, VALIDATION, REPORTING, predict, rollout,
    reference, score, sha256, atomic_json, _load_case2_model,
)


def fit_feedback(primary_errors, tail_errors, ridge):
    """Minimize ||Es-K Ep||_F^2 + lambda ||K||_F^2.

    lambda = ridge * trace(Ep Ep.T)/n, so ridge is dimensionless. The
    unregularized result uses a minimum-norm SVD least-squares solution.
    """
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if ridge == 0:
        return np.linalg.lstsq(primary_errors.T, tail_errors.T, rcond=None)[0].T
    gram = primary_errors @ primary_errors.T
    lam = ridge * np.trace(gram) / gram.shape[0]
    if lam <= 0:
        return np.zeros((tail_errors.shape[0], primary_errors.shape[0]))
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), primary_errors @ tail_errors.T).T


def error_tables(model, dataset):
    tables = []
    for path in sorted(dataset.iterdir()):
        teacher = np.load(path / "qN.npy")
        mu = np.load(path / "mu.npy").reshape(-1)
        tables.append((teacher - predict(model, mu, teacher.shape[1]-1))[:, 1:])
    return np.hstack(tables)


class FeedbackPrediction(torch.nn.Module):
    def __init__(self, model, gain):
        super().__init__()
        self.model = model
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float64))

    def forward(self, x):
        original = self.model(x).to(torch.float64)
        return torch.cat((original[..., :10], original[..., 10:] - original[..., :10] @ self.gain.T), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "reporting"), default="validation")
    parser.add_argument("--ridges", type=float, nargs="+", default=[0, 0.01, 1])
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PAPER / "euclidean_case2_affine_feedback")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    model, _, _, _ = _load_case2_model(MODEL, "cpu")
    train = error_tables(model, CAMPAIGN / "Stage2/prom_coeff_dataset_ntot151/per_mu")
    validation = error_tables(model, CAMPAIGN / "Stage2/prom_coeff_dataset_ntot151_validation2/per_mu")
    v = np.asfortranarray(np.load(BASIS / "basis.npy"))
    uref = np.load(BASIS / "u_ref.npy").reshape(-1)
    if not np.allclose(v.T @ v, np.eye(151), rtol=0, atol=1e-10):
        raise ValueError("Only the Euclidean campaign is supported")
    fits = {}
    gains = {}
    for ridge in args.ridges:
        gain = fit_feedback(train[:10], train[10:], ridge)
        gains[ridge] = gain
        np.save(args.output / f"gain_ridge{ridge:g}.npy", gain)
        fits[str(ridge)] = {
            "gain_spectral_norm": float(np.linalg.norm(gain, 2)),
            "train_tail_error_ratio": float(np.linalg.norm(train[10:] - gain @ train[:10]) / np.linalg.norm(train[10:])),
            "validation_teacher_primary_tail_error_ratio": float(np.linalg.norm(validation[10:] - gain @ validation[:10]) / np.linalg.norm(validation[10:])),
        }
    atomic_json(args.output / "fit_summary.json", fits)
    print(json.dumps(fits, indent=2), flush=True)
    if args.fit_only:
        return
    points = VALIDATION if args.stage == "validation" else REPORTING
    for index, mu in enumerate(points):
        for ridge in args.ridges:
            gain = gains[ridge]
            folder = args.output / f"{args.stage}_{index}_feedback{ridge:g}"
            folder.mkdir(exist_ok=True)
            config = {"method": "affine_feedback", "mu": mu, "ridge": ridge, "steps": 500,
                      "stage": args.stage, "threads": args.threads,
                      "model_sha256": sha256(MODEL), "basis_sha256": sha256(BASIS / "basis.npy"),
                      "gain_sha256": sha256(args.output / f"gain_ridge{ridge:g}.npy"),
                      "runner_sha256": sha256(__file__),
                      "rollout_sha256": sha256(ROOT / "Project_YvonMaday/run_case2_local_corrections.py"),
                      "initialization": "least-squares fit of initial condition on the feedback manifold"}
            summary = folder / "summary.json"
            if summary.exists():
                old = json.loads(summary.read_text())
                if json.dumps(old["config"], sort_keys=True) != json.dumps(config, sort_keys=True):
                    raise RuntimeError(f"Configuration changed: {folder}")
                print(f"Completed, skipping {folder.name}", flush=True)
                continue
            adapted_basis = np.asfortranarray(np.column_stack((v[:, :10] + v[:, 10:] @ gain, v[:, 10:])))
            q, stats = rollout(FeedbackPrediction(model, gain), adapted_basis, uref, mu,
                               "baseline", 500, 10, progress=folder / "progress.json")
            q[10:] += gain @ q[:10]
            np.save(folder / "qN.npy", q)
            result = {"config": config, **stats, **score(q, v, uref, mu, reference(mu, args.stage))}
            atomic_json(summary, result)
            print(json.dumps({k: val for k, val in result.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=int(sys.argv[sys.argv.index("--threads")+1]) if "--threads" in sys.argv else 2):
        main()
