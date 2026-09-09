#!/usr/bin/env python3
"""Local reference timing/accuracy checks without overwriting campaign runs."""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from Project_YvonMaday.run_case2_local_corrections import (
    PAPER, CAMPAIGN, BASIS, MODEL, REPORTING, VALIDATION, atomic_json, sha256,
    reference, score, rollout, _load_case2_model,
)
from Project_YvonMaday.run_prom_ann_case_1 import _load_case1_model, ANNVectorWrapper as Case1Vector
from Project_YvonMaday.run_prom_ann_case_3 import _load_case3_model, ANNVectorWrapper as Case3Vector
from burgers.config import GRID_X, GRID_Y, DT, W0
from burgers.pod_ann_manifold import (
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case3,
    _build_ann_decoder_jacobian,
)
from burgers.core import get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
from burgers.gauss_newton import gauss_newton_pod_ann


def known_initial_nonlinear_rollout(model, v, uref, mu, case, steps=500):
    """Production nonlinear LSPG update with the prescribed initial predecessor.

    Keeps the native float32 decoder, tangent, GN update and tolerances.
    Only initial q and the initial predecessor differ from the native runner.
    """
    start = time.perf_counter()
    vp = torch.tensor(v[:, :10], dtype=torch.float32)
    vs = torch.tensor(v[:, 10:], dtype=torch.float32)
    ref_t = torch.tensor(uref, dtype=torch.float32)
    q = np.zeros((151, steps + 1))
    q[:, 0] = np.linalg.lstsq(v, W0 - uref, rcond=None)[0]
    previous = uref + v @ q[:, 0]
    primary = torch.tensor(q[:10, 0], dtype=torch.float32)
    dx, dy, jdx, jdy, eye = get_ops(GRID_X, GRID_Y)
    iterations = 0
    for step in range(1, steps + 1):
        condition = torch.tensor([*mu, DT*step], dtype=torch.float32)

        def closure(y):
            return model(y if case == 1 else torch.cat((y, condition)))

        def decode(y, with_grad=True):
            with torch.set_grad_enabled(with_grad):
                return ref_t + vp @ y + vs @ closure(y)

        primary, norms, _ = gauss_newton_pod_ann(
            func=lambda w: inviscid_burgers_res2D(w, GRID_X, GRID_Y, DT, previous, mu, dx, dy),
            jac=lambda w: inviscid_burgers_exact_jac2D(w, DT, jdx, jdy, eye),
            y0=primary, decode=decode,
            jacfwdfunc=_build_ann_decoder_jacobian(vp, vs, closure),
            max_its=20, relnorm_cutoff=1e-5, min_delta=1e-2,
            u_ref=uref, linear_solver="lstsq", normal_eq_reg=1e-12)
        iterations += len(norms)
        with torch.no_grad():
            previous = decode(primary, with_grad=False).numpy().copy()
            q[:10, step] = primary.numpy()
            q[10:, step] = closure(primary).numpy()
        if step % 50 == 0:
            print(f"case{case}_known_initial mu={mu} step={step}/{steps} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    return q, {"online_seconds": time.perf_counter()-start, "iterations": iterations}


class InitialOnlyPrediction(torch.nn.Module):
    """Remove ANN predictions after t=0, retaining matched initialization."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if float(x[0, -1]) == 0.0:
            return self.model(x)
        return torch.zeros((x.shape[0], 151), dtype=x.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "reporting"), default="reporting")
    parser.add_argument("--point-indices", nargs="+", type=int, default=[1])
    parser.add_argument("--methods", nargs="+", choices=("case1", "case3", "case1_known_initial", "case3_known_initial", "case2_n20", "krylov3_zero"), default=["case1", "case3"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=PAPER / "euclidean_case2_local_references")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    v = np.asfortranarray(np.load(BASIS / "basis.npy"))
    uref = np.load(BASIS / "u_ref.npy").reshape(-1)
    points = VALIDATION if args.stage == "validation" else REPORTING
    for index in args.point_indices:
        mu = points[index]
        for method in args.methods:
            if method in ("case1", "case1_known_initial"):
                path = CAMPAIGN / "Stage3/models/case1_ann_ntot151_best.pt"
                model, _, _ = _load_case1_model(path, "cpu")
                model = Case1Vector(model)
                solver = inviscid_burgers_implicit2D_LSPG_pod_ann_2D
            elif method in ("case3", "case3_known_initial"):
                path = CAMPAIGN / "Stage3/models/case3_ann_ntot151_best.pt"
                model, *_ = _load_case3_model(path, "cpu")
                model = Case3Vector(model)
                solver = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case3
            else:
                path = MODEL
                model, *_ = _load_case2_model(path, "cpu")
            for repetition in range(args.repeat):
                folder = args.output / f"{args.stage}_{index}_{method}_rep{repetition}"
                folder.mkdir(exist_ok=True)
                config = {"method": method, "stage": args.stage, "mu": mu, "steps": 500,
                          "threads": args.threads, "model_sha256": sha256(path),
                          "basis_sha256": sha256(BASIS / "basis.npy"), "runner_sha256": sha256(__file__),
                          "production_solver_sha256": sha256(ROOT / "burgers/pod_ann_manifold.py")}
                summary = folder / "summary.json"
                if summary.exists():
                    old = json.loads(summary.read_text())
                    if json.dumps(old["config"], sort_keys=True) != json.dumps(config, sort_keys=True):
                        raise RuntimeError(f"Configuration changed: {folder}")
                    print(f"Completed, skipping {folder.name}", flush=True)
                    continue
                with (folder / "solver.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log):
                    if method.endswith("_known_initial"):
                        q, stats = known_initial_nonlinear_rollout(model, v, uref, mu,
                                                                  1 if method.startswith("case1") else 3)
                    elif method in ("case2_n20", "krylov3_zero"):
                        predictor = InitialOnlyPrediction(model) if method == "krylov3_zero" else model
                        variant = "krylov3" if method == "krylov3_zero" else "baseline"
                        dimension = 10 if method == "krylov3_zero" else 20
                        q, stats = rollout(predictor, v, uref, mu, variant, 500, dimension,
                                           progress=folder / "progress.json")
                    else:
                        start = time.perf_counter()
                        snaps, primary, native_stats = solver(
                            GRID_X, GRID_Y, W0, DT, 500, mu, model, None,
                            v[:, :10], v[:, 10:], u_ref=uref, min_delta=1e-2,
                            return_red_coords=True)
                        elapsed = time.perf_counter() - start
                        inputs = primary.T
                        if method == "case3":
                            inputs = np.column_stack((inputs, np.tile(mu, (501, 1)), DT*np.arange(501)))
                        with torch.no_grad():
                            tail = np.asarray([model(torch.tensor(x, dtype=torch.float32)).numpy()
                                               for x in inputs], dtype=np.float64).T
                        q = np.vstack((primary, tail))
                        del snaps
                        stats = {"online_seconds": elapsed, "iterations": int(native_stats[0])}
                np.save(folder / "qN.npy", q)
                result = {"config": config, **stats, **score(q, v, uref, mu, reference(mu, args.stage))}
                atomic_json(summary, result)
                print(folder.name, json.dumps({k: val for k, val in result.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=int(sys.argv[sys.argv.index("--threads")+1]) if "--threads" in sys.argv else 2):
        main()
