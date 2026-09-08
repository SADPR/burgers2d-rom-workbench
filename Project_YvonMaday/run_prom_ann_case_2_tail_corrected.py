#!/usr/bin/env python3
"""Evaluate the low-rank tail-corrected Case--2 PROM at selected parameters."""

from __future__ import annotations

import argparse
import os
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
from burgers.pod_ann_manifold import inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_tail_corrected
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


def _load_correction_basis(path: Path, secondary_basis: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    full_basis = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    n_secondary = int(secondary_basis.shape[1])
    if full_basis.ndim != 2 or full_basis.shape[0] != n_secondary:
        raise ValueError(
            f"Correction basis must have shape ({n_secondary}, rmax), got {full_basis.shape}."
        )
    if not 0 <= rank <= full_basis.shape[1]:
        raise ValueError(f"Rank must lie in [0, {full_basis.shape[1]}], got {rank}.")
    basis = full_basis[:, :rank].copy()
    state_orthonorm_error = 0.0
    if rank:
        gram = secondary_basis.T @ secondary_basis
        state_orthonorm_error = float(np.linalg.norm(basis.T @ gram @ basis - np.eye(rank), ord=np.inf))
        if state_orthonorm_error > 1.0e-8:
            raise ValueError(
                "Correction basis is not orthonormal in the tail-induced state norm: "
                f"error {state_orthonorm_error:.3e}."
            )
    return basis, state_orthonorm_error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--correction-basis-path", required=True, type=Path)
    parser.add_argument("--basis-path", required=True, type=Path)
    parser.add_argument("--u-ref-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--points", type=_parse_point, nargs="+", required=True)
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
        raise ValueError("The tail-corrected runner requires a full q_tot master ANN checkpoint.")
    ntot = int(checkpoint.get("n_tot", output_dim))
    if output_dim != ntot or not 1 <= args.n_primary < ntot:
        raise ValueError("Invalid master-ANN or primary/tail mode split.")
    _, primary_basis, secondary_basis, u_ref, basis_path, uref_path = _load_basis_and_reference(
        ntot,
        args.n_primary,
        basis_path_override=str(args.basis_path),
        uref_path_override=str(args.u_ref_path),
    )
    correction_basis, state_orthonorm_error = _load_correction_basis(
        args.correction_basis_path, secondary_basis, args.rank
    )
    tail_model = MasterTail(master, args.n_primary).to(device)
    tail_model.eval()

    print(f"[Case2-TailCorrected] model = {args.model_path}")
    print(f"[Case2-TailCorrected] basis = {basis_path}")
    print(f"[Case2-TailCorrected] correction basis = {args.correction_basis_path}")
    print(f"[Case2-TailCorrected] n/rank = {args.n_primary}/{args.rank}")
    print(f"[Case2-TailCorrected] ||B^T(Vbar^T Vbar)B-I||_inf = {state_orthonorm_error:.3e}")
    print(f"[Case2-TailCorrected] device = {device}")

    snap_folder = THIS_DIR / "Results" / "param_snaps"
    snap_folder.mkdir(parents=True, exist_ok=True)
    for point in args.points:
        tag = _safe_mu_tag(point)
        run_tag = f"case2_prom_ann_tailcorr_{tag}_n{args.n_primary}_r{args.rank}_ntot{ntot}"
        summary_path = args.output_root / f"{run_tag}_summary.txt"
        qn_path = args.output_root / f"{run_tag}_qN.npy"
        snaps_path = args.output_root / f"{run_tag}_snaps.npy"
        if summary_path.exists() and not args.force:
            print(f"[Case2-TailCorrected] skip existing {tag}; use --force to rerun.")
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
        snaps, q_primary, eta, ann_tail, stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_tail_corrected(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=W0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=point,
            ann_model=tail_model,
            correction_basis=correction_basis,
            ref=None,
            basis=primary_basis,
            basis2=secondary_basis,
            u_ref=u_ref,
            max_its=args.max_its,
            relnorm_cutoff=args.relnorm_cutoff,
            min_delta=args.min_delta,
        )
        elapsed = time.time() - started
        corrected_tail = ann_tail + correction_basis @ eta
        qtot = np.vstack((q_primary, corrected_tail))
        relative_error = 100.0 * np.linalg.norm(hdm - snaps) / max(np.linalg.norm(hdm), 1.0e-30)
        np.save(qn_path, qtot)
        if args.save_snaps:
            np.save(snaps_path, snaps)
        else:
            snaps_path = Path("not_saved")

        num_its, jac_time, res_time, ls_time = stats
        correction_norm = float(np.linalg.norm(correction_basis @ eta) / max(np.linalg.norm(ann_tail), 1.0e-30))
        write_kv_txt(
            str(summary_path),
            [
                ("mu_test", point),
                ("model_path", str(args.model_path)),
                ("basis_path", basis_path),
                ("u_ref_path", uref_path),
                ("correction_basis_path", str(args.correction_basis_path)),
                ("n_primary", args.n_primary),
                ("correction_rank", args.rank),
                ("online_unknown_dimension", args.n_primary + args.rank),
                ("tangent", "[V_primary, V_secondary B_r]"),
                ("tail_state_orthonorm_inf_error", state_orthonorm_error),
                ("solve_backend", "prom"),
                ("online_solve_elapsed_s", elapsed),
                ("num_iterations", num_its),
                ("jac_time_s", jac_time),
                ("res_time_s", res_time),
                ("ls_time_s", ls_time),
                ("relative_error_percent", relative_error),
                ("tail_correction_rel_norm", correction_norm),
                ("qN_source", "solver_primary_plus_residual_solved_low_rank_tail_correction"),
                ("qN_output", str(qn_path)),
                ("snaps_output", str(snaps_path)),
            ],
        )
        print(
            f"[Case2-TailCorrected] mu={point}: state error={relative_error:.4f}% "
            f"tail-correction norm={100.0 * correction_norm:.3f}% elapsed={elapsed:.2f}s"
        )
        print(f"[Case2-TailCorrected] summary: {summary_path}")


if __name__ == "__main__":
    main()
