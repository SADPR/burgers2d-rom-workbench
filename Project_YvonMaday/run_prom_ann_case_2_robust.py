#!/usr/bin/env python3
"""Evaluate the robust (IRLS) Case--2 PROM solve at selected parameters.

Same master ANN, basis, and tail-injection mechanism as the production
Case--2 solve; the only difference is the Gauss-Newton normal equations used
online (see ``inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_robust``).
``--robust-kernel none`` reproduces the plain Case--2 solve exactly, which is
useful as an in-harness sanity check against the production numbers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0
from burgers.core import load_or_compute_snaps
from burgers.pod_ann_manifold import inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_robust
from project_layout import write_kv_txt
from run_prom_ann_case_2 import _load_basis_and_reference, _load_case2_model, _safe_mu_tag


class MasterTail(nn.Module):
    """Expose only the secondary tail of a full q_tot master ANN."""

    def __init__(self, master: nn.Module, n_primary: int):
        super().__init__()
        self.master = master
        self.n_primary = int(n_primary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.master(x)[..., self.n_primary :]


def _parse_point(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Point must have form mu1,mu2; got {value!r}.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid point {value!r}.") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--basis-path", required=True, type=Path)
    parser.add_argument("--u-ref-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--points", type=_parse_point, nargs="+", required=True)
    parser.add_argument(
        "--robust-kernel",
        choices=("none", "huber", "geman_mcclure", "welsch"),
        default="huber",
        help="'none' reproduces the plain Case--2 LSPG solve exactly.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.345)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1.0e-5)
    parser.add_argument("--min-delta", type=float, default=1.0e-2)
    parser.add_argument("--save-snaps", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    requested_device = args.device
    device_name = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    args.output_root.mkdir(parents=True, exist_ok=True)

    master, output_dim, output_kind, checkpoint = _load_case2_model(str(args.model_path), device=device)
    if output_kind != "q_tot":
        raise ValueError("The robust-Case2 runner requires a full q_tot master ANN checkpoint.")
    ntot = int(checkpoint.get("n_tot", output_dim))
    if output_dim != ntot or not 1 <= args.n_primary < ntot:
        raise ValueError("Invalid master-ANN or primary/tail mode split.")
    _, primary_basis, secondary_basis, u_ref, basis_path, uref_path = _load_basis_and_reference(
        ntot,
        args.n_primary,
        basis_path_override=str(args.basis_path),
        uref_path_override=str(args.u_ref_path),
    )
    tail_model = MasterTail(master, args.n_primary).to(device)
    tail_model.eval()

    print(f"[Case2-Robust] model = {args.model_path}")
    print(f"[Case2-Robust] basis = {basis_path}")
    print(f"[Case2-Robust] n_primary = {args.n_primary}")
    print(f"[Case2-Robust] robust kernel = {args.robust_kernel} (huber_delta={args.huber_delta})")
    print(f"[Case2-Robust] device = {device}")

    snap_folder = THIS_DIR / "Results" / "param_snaps"
    snap_folder.mkdir(parents=True, exist_ok=True)
    for point in args.points:
        tag = _safe_mu_tag(point)
        run_tag = f"case2_prom_ann_robust_{args.robust_kernel}_{tag}_n{args.n_primary}_ntot{ntot}"
        summary_path = args.output_root / f"{run_tag}_summary.txt"
        qn_path = args.output_root / f"{run_tag}_qN.npy"
        snaps_path = args.output_root / f"{run_tag}_snaps.npy"
        if summary_path.exists() and not args.force:
            print(f"[Case2-Robust] skip existing {tag}; use --force to rerun.")
            continue

        hdm = load_or_compute_snaps(
            mu=point,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=W0,
            dt=DT,
            num_steps=NUM_STEPS,
            snap_folder=str(snap_folder),
        )
        started = time.time()
        snaps, red_coords, stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_robust(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=W0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=point,
            ann_model=tail_model,
            ref=None,
            basis=primary_basis,
            basis2=secondary_basis,
            u_ref=u_ref,
            max_its=args.max_its,
            relnorm_cutoff=args.relnorm_cutoff,
            min_delta=args.min_delta,
            robust_kernel=args.robust_kernel,
            huber_delta=args.huber_delta,
            return_red_coords=True,
        )
        elapsed = time.time() - started
        relative_error = 100.0 * np.linalg.norm(hdm - snaps) / max(np.linalg.norm(hdm), 1.0e-30)
        np.save(qn_path, red_coords)
        if args.save_snaps:
            np.save(snaps_path, snaps)
        else:
            snaps_path = Path("not_saved")

        num_its, jac_time, res_time, ls_time = stats
        write_kv_txt(
            str(summary_path),
            [
                ("mu_test", point),
                ("model_path", str(args.model_path)),
                ("basis_path", basis_path),
                ("u_ref_path", uref_path),
                ("n_primary", args.n_primary),
                ("robust_kernel", args.robust_kernel),
                ("huber_delta", args.huber_delta),
                ("tangent", "[V_primary] (unchanged online unknowns)"),
                ("solve_backend", "prom"),
                ("online_solve_elapsed_s", elapsed),
                ("num_iterations", num_its),
                ("jac_time_s", jac_time),
                ("res_time_s", res_time),
                ("ls_time_s", ls_time),
                ("relative_error_percent", relative_error),
                ("qN_source", "solver_primary_only_robust_normal_equations"),
                ("qN_output", str(qn_path)),
                ("snaps_output", str(snaps_path)),
            ],
        )
        print(
            f"[Case2-Robust] mu={point}: state error={relative_error:.4f}% "
            f"kernel={args.robust_kernel} elapsed={elapsed:.2f}s"
        )
        print(f"[Case2-Robust] summary: {summary_path}")


if __name__ == "__main__":
    main()
