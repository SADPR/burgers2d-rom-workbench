#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-1 weighted POD builder for Maday experiments.

Writes all artifacts to `Results_Maday/<tag>/Stage1/` and keeps baseline untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import get_snapshot_params, load_or_compute_snaps, plot_singular_value_decay, podsize
from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


def _choose_snap_folder() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
        os.path.join(PROJECT_ROOT, "param_snaps"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _build_snapshot_matrix(mu_samples, snap_folder, dt, num_steps):
    first = np.asarray(
        load_or_compute_snaps(mu_samples[0], GRID_X, GRID_Y, W0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )
    n_dofs, n_time = first.shape
    snaps = np.zeros((n_dofs, n_time * len(mu_samples)), dtype=np.float64)
    snaps[:, :n_time] = first
    col = n_time
    for i, mu in enumerate(mu_samples[1:], start=2):
        print(f"[MADAY-STAGE1] loading snapshots {i}/{len(mu_samples)} for mu={mu}")
        cur = np.asarray(
            load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        if cur.shape != (n_dofs, n_time):
            raise RuntimeError(f"Snapshot shape mismatch for mu={mu}: got {cur.shape}, expected {(n_dofs, n_time)}")
        snaps[:, col:col + n_time] = cur
        col += n_time
    return snaps


def _cell_area_weights() -> np.ndarray:
    dx = np.asarray(GRID_X[1:] - GRID_X[:-1], dtype=np.float64)
    dy = np.asarray(GRID_Y[1:] - GRID_Y[:-1], dtype=np.float64)
    area = np.outer(dy, dx).reshape(-1)
    return np.concatenate([area, area], axis=0)


def _load_weight_diag(kind: str, weight_file: str | None, n_dofs: int) -> np.ndarray:
    mode = str(kind).strip().lower()
    if mode == "identity":
        w = np.ones((n_dofs,), dtype=np.float64)
    elif mode in ("cell_area", "mblock"):
        w = _cell_area_weights()
    elif mode == "diag_file":
        if not weight_file:
            raise ValueError("--weight-file is required when --weighting=diag_file")
        w = np.asarray(np.load(os.path.abspath(os.path.expanduser(weight_file)), allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"Unsupported weighting mode: {kind}")

    if w.size != n_dofs:
        raise ValueError(f"Weight size mismatch: got {w.size}, expected {n_dofs}")
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("All weights must be finite and strictly positive.")
    return w


def _weighted_pod(snaps: np.ndarray, wdiag: np.ndarray, num_modes: int | None, pod_tol: float | None, center: bool):
    snaps = np.asarray(snaps, dtype=np.float64)
    wdiag = np.asarray(wdiag, dtype=np.float64).reshape(-1)
    n_dofs = snaps.shape[0]
    if wdiag.size != n_dofs:
        raise ValueError("Weight dimension mismatch in weighted POD.")

    if center:
        u_ref = np.mean(snaps, axis=1)
        snaps_c = snaps - u_ref[:, None]
    else:
        u_ref = np.zeros((n_dofs,), dtype=np.float64)
        snaps_c = snaps

    sqrt_w = np.sqrt(wdiag)
    snaps_w = sqrt_w[:, None] * snaps_c
    uw, sigma, _ = np.linalg.svd(snaps_w, full_matrices=False)

    if num_modes is not None:
        n_keep = int(num_modes)
    elif pod_tol is not None:
        n_keep = podsize(sigma, energy_thresh=1.0 - float(pod_tol))
    else:
        n_keep = sigma.size
    if n_keep < 1 or n_keep > uw.shape[1]:
        raise ValueError(f"Invalid n_keep={n_keep}, available={uw.shape[1]}")

    basis = uw[:, :n_keep] / sqrt_w[:, None]
    gram = basis.T @ (wdiag[:, None] * basis)
    ortho_err = float(np.linalg.norm(gram - np.eye(n_keep), ord="fro"))

    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    info = {
        "n_keep": int(n_keep),
        "n_available": int(sigma.size),
        "energy_captured": float(energy[n_keep - 1]),
        "energy_lost": float(1.0 - energy[n_keep - 1]),
        "centered": bool(center),
        "weighted_orthonorm_fro_err": ortho_err,
    }
    return basis, sigma, u_ref, info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build weighted POD basis for Maday experiments.")
    parser.add_argument("--maday-tag", type=str, default="exp_maday_p2")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--snap-folder", type=str, default=None)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--pod-tol", type=float, default=1e-6, help="Energy loss tolerance.")
    parser.add_argument("--center", action="store_true", default=True)
    parser.add_argument("--no-center", action="store_false", dest="center")
    parser.add_argument(
        "--weighting",
        choices=("identity", "cell_area", "mblock", "diag_file"),
        default="cell_area",
        help="`mblock` is an explicit alias of `cell_area` for Proposal 1 naming.",
    )
    parser.add_argument("--weight-file", type=str, default=None)
    parser.add_argument("--test-mu", type=str, default="4.56,0.019")
    parser.add_argument("--basis-name", type=str, default="basis_weighted.npy")
    parser.add_argument("--sigma-name", type=str, default="sigma_weighted.npy")
    parser.add_argument("--uref-name", type=str, default="u_ref_weighted.npy")
    parser.add_argument("--weights-name", type=str, default="weights_diag.npy")
    parser.add_argument("--meta-name", type=str, default="stage1_lspg_weighted_pod_metadata.npz")
    parser.add_argument("--summary-name", type=str, default="stage1_lspg_weighted_pod_summary.txt")
    parser.add_argument("--decay-plot-name", type=str, default="stage1_lspg_weighted_pod_decay.png")
    args = parser.parse_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    stage1_dir = paths.stage1
    os.makedirs(stage1_dir, exist_ok=True)

    snap_folder = str(args.snap_folder).strip() if args.snap_folder else _choose_snap_folder()
    os.makedirs(snap_folder, exist_ok=True)

    mu_samples = get_snapshot_params(
        mu1_range=MU1_RANGE,
        mu2_range=MU2_RANGE,
        samples_per_mu=SAMPLES_PER_MU,
    )
    if len(mu_samples) == 0:
        raise RuntimeError("No training parameters returned by get_snapshot_params.")

    print(f"[MADAY-STAGE1] tag={paths.tag}")
    print(f"[MADAY-STAGE1] output_dir={stage1_dir}")
    print(f"[MADAY-STAGE1] snap_folder={snap_folder}")
    print(f"[MADAY-STAGE1] weighting={args.weighting}")

    t0 = time.time()
    snaps = _build_snapshot_matrix(mu_samples, snap_folder=snap_folder, dt=float(args.dt), num_steps=int(args.num_steps))
    t_snap = time.time() - t0

    wdiag = _load_weight_diag(args.weighting, args.weight_file, n_dofs=snaps.shape[0])
    basis, sigma, u_ref, info = _weighted_pod(
        snaps=snaps,
        wdiag=wdiag,
        num_modes=args.num_modes,
        pod_tol=args.pod_tol,
        center=bool(args.center),
    )

    basis_file = os.path.join(stage1_dir, str(args.basis_name))
    sigma_file = os.path.join(stage1_dir, str(args.sigma_name))
    uref_file = os.path.join(stage1_dir, str(args.uref_name))
    wdiag_file = os.path.join(stage1_dir, str(args.weights_name))
    meta_file = os.path.join(stage1_dir, str(args.meta_name))
    summary_file = os.path.join(stage1_dir, str(args.summary_name))
    decay_plot = os.path.join(stage1_dir, str(args.decay_plot_name))

    np.save(basis_file, basis)
    np.save(sigma_file, sigma)
    np.save(uref_file, u_ref)
    np.save(wdiag_file, wdiag)
    np.savez(
        meta_file,
        n_keep=np.asarray(info["n_keep"], dtype=np.int64),
        n_available=np.asarray(info["n_available"], dtype=np.int64),
        energy_captured=np.asarray(info["energy_captured"], dtype=np.float64),
        energy_lost=np.asarray(info["energy_lost"], dtype=np.float64),
        centered=np.asarray(int(info["centered"]), dtype=np.int64),
        weighted_orthonorm_fro_err=np.asarray(info["weighted_orthonorm_fro_err"], dtype=np.float64),
        weighting=np.asarray(str(args.weighting)),
        tag=np.asarray(paths.tag),
        timestamp=np.asarray(datetime.now().isoformat(timespec="seconds")),
    )

    plot_singular_value_decay(
        sigma,
        out_path=decay_plot,
        max_modes=min(1000, sigma.size),
        label=f"Weighted POD ({args.weighting})",
        title="Weighted POD residual energy decay",
        use_latex=True,
    )

    mu_tokens = [s.strip() for s in str(args.test_mu).split(",")]
    if len(mu_tokens) != 2:
        raise ValueError("--test-mu must be 'mu1,mu2'")
    test_mu = [float(mu_tokens[0]), float(mu_tokens[1])]
    hdm_snap = np.asarray(
        load_or_compute_snaps(test_mu, GRID_X, GRID_Y, W0, float(args.dt), int(args.num_steps), snap_folder=snap_folder),
        dtype=np.float64,
    )
    q = basis.T @ (wdiag[:, None] * (hdm_snap - u_ref[:, None]))
    rec = u_ref[:, None] + basis @ q
    rel_err = float(np.linalg.norm(hdm_snap - rec) / (np.linalg.norm(hdm_snap) + 1e-30))

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"tag: {paths.tag}\n")
        f.write(f"script: Project_YvonMaday/stage1_lspg_weighted_pod.py\n")
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"snap_folder: {snap_folder}\n")
        f.write(f"weighting: {args.weighting}\n")
        f.write(f"n_keep: {info['n_keep']}\n")
        f.write(f"n_available: {info['n_available']}\n")
        f.write(f"energy_captured: {info['energy_captured']:.8e}\n")
        f.write(f"energy_lost: {info['energy_lost']:.8e}\n")
        f.write(f"weighted_orthonorm_fro_err: {info['weighted_orthonorm_fro_err']:.8e}\n")
        f.write(f"reconstruction_rel_err_mu_test: {rel_err:.8e}\n")
        f.write(f"snapshot_assembly_time_s: {t_snap:.6f}\n")

    print(f"[MADAY-STAGE1] saved basis: {basis_file}")
    print(f"[MADAY-STAGE1] saved summary: {summary_file}")
    print(f"[MADAY-STAGE1] reconstruction rel error @mu={test_mu}: {rel_err:.6e}")


if __name__ == "__main__":
    main()
