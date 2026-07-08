#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: build PROM-solved qN dataset.

For each training parameter mu, this script runs an n_tot-dimensional ROM
(default: HPROM/ECSW-LSPG) and stores:
- mu.npy
- t.npy
- qN.npy     (full reduced coordinates, shape n_tot x (num_steps+1))
- rom_stats.npy
- prom_stats.npy or hprom_stats.npy (backend-specific alias)
- rom_snaps.npy  (optional reconstructed full snapshots)
- hdm_vs_prom.png or hdm_vs_hprom.png (optional)
"""

import os
import sys
import time
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import (
    get_snapshot_params,
    load_or_compute_snaps,
    plot_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
)
from burgers.linear_manifold import (
    inviscid_burgers_implicit2D_LSPG,
    inviscid_burgers_implicit2D_LSPG_ecsw,
    compute_ECSW_training_matrix_2D,
)
from burgers.ecsw_utils import build_ecsw_snapshot_plan, direct_left_singular_vectors
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU
try:
    from project_layout import STAGE1_DIR, STAGE2_DIR, ensure_layout_dirs, stage2_dataset_dir, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE1_DIR, STAGE2_DIR, ensure_layout_dirs, stage2_dataset_dir, write_kv_txt


def set_latex_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "figure.figsize": (12, 8),
    })


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _time_grid(dt, num_steps):
    return dt * np.arange(num_steps + 1, dtype=np.float64)


def _select_snap_folder(project_root):
    candidates = [
        os.path.join(project_root, "Results", "param_snaps"),
        os.path.join(project_root, "param_snaps"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _load_pod_artifacts(requested_total_modes=None, basis_override=None, uref_override=None):
    # Prefer Stage1 outputs under Results, then fallback to legacy root files.
    if basis_override is not None or uref_override is not None:
        if basis_override is None or uref_override is None:
            raise ValueError("--basis-path and --u-ref-path must be provided together.")

        basis_path = os.path.abspath(os.path.expanduser(str(basis_override)))
        uref_path = os.path.abspath(os.path.expanduser(str(uref_override)))
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Missing basis file: {basis_path}")
        if not os.path.exists(uref_path):
            raise FileNotFoundError(f"Missing reference-state file: {uref_path}")

        basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)

        if basis.ndim != 2:
            raise ValueError(f"basis.npy at '{basis_path}' must be 2D, got shape {basis.shape}.")
        if u_ref.size != basis.shape[0]:
            raise ValueError(
                f"u_ref size mismatch: u_ref has {u_ref.size}, basis has {basis.shape[0]} rows."
            )

        n_available = int(basis.shape[1])
        if requested_total_modes is None:
            total_modes = n_available
        else:
            total_modes = int(requested_total_modes)
            if total_modes < 1 or total_modes > n_available:
                raise ValueError(
                    f"basis.npy at '{basis_path}' has {n_available} modes, "
                    f"but total_modes={total_modes} is requested."
                )
        pod_dir = os.path.dirname(basis_path)
        return basis[:, :total_modes], u_ref, basis_path, uref_path, pod_dir, total_modes, n_available

    pod_candidates = [STAGE1_DIR, THIS_DIR]

    for pod_dir in pod_candidates:
        basis_path = os.path.join(pod_dir, "basis.npy")
        uref_path = os.path.join(pod_dir, "u_ref.npy")

        if not (os.path.exists(basis_path) and os.path.exists(uref_path)):
            continue

        basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)

        if basis.ndim != 2:
            raise ValueError(f"basis.npy at '{basis_path}' must be 2D, got shape {basis.shape}.")
        if u_ref.size != basis.shape[0]:
            raise ValueError(
                f"u_ref size mismatch in '{pod_dir}': u_ref has {u_ref.size}, "
                f"basis has {basis.shape[0]} rows."
            )

        n_available = int(basis.shape[1])
        if requested_total_modes is None:
            total_modes = n_available
        else:
            total_modes = int(requested_total_modes)
            if total_modes < 1 or total_modes > n_available:
                raise ValueError(
                    f"basis.npy at '{basis_path}' has {n_available} modes, "
                    f"but total_modes={total_modes} is requested."
                )

        return basis[:, :total_modes], u_ref, basis_path, uref_path, pod_dir, total_modes, n_available

    checked = "\n".join([f"  - {p}" for p in pod_candidates])
    raise FileNotFoundError(
        "Could not find basis.npy + u_ref.npy in any expected POD directory:\n"
        f"{checked}"
    )


def _compute_ecsw_weights(
    basis,
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu_samples,
    snap_folder,
    snap_time_offset=3,
    snapshot_percent=2.0,
    snapshot_random_seed=42,
    ensure_mu_coverage=True,
    svd_relative_tolerance=1e-8,
):
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")

    snapshot_percent = float(snapshot_percent)
    if not np.isfinite(snapshot_percent) or snapshot_percent <= 0.0:
        raise ValueError("snapshot_percent must be a finite value > 0.")

    ecsw_plan = build_ecsw_snapshot_plan(
        num_steps=num_steps,
        snap_time_offset=snap_time_offset,
        num_mu=len(mu_samples),
        mode="global_param_time_stratified",
        total_snapshots=None,
        total_snapshots_percent=snapshot_percent,
        mu_points=mu_samples,
        random_seed=int(snapshot_random_seed),
        ensure_mu_coverage=bool(ensure_mu_coverage),
    )

    Clist = []
    t0 = time.time()

    for imu, mu in enumerate(mu_samples):
        mu_snaps = load_or_compute_snaps(
            mu=mu,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
        )

        now_cols = np.asarray(ecsw_plan["selected_now_cols_by_mu"][imu], dtype=int)
        if now_cols.size == 0:
            continue

        prev_cols = now_cols - snap_time_offset
        snaps_now = mu_snaps[:, now_cols]
        snaps_prev = mu_snaps[:, prev_cols]

        if snaps_now.shape[1] != snaps_prev.shape[1]:
            raise RuntimeError(
                "ECSW snapshot alignment failed: "
                f"snaps_now has {snaps_now.shape[1]} columns, "
                f"snaps_prev has {snaps_prev.shape[1]} columns."
            )
        if snaps_now.shape[1] == 0:
            continue

        Ci = compute_ECSW_training_matrix_2D(
            snaps_now,
            snaps_prev,
            basis,
            inviscid_burgers_res2D,
            inviscid_burgers_exact_jac2D,
            grid_x,
            grid_y,
            dt,
            mu,
        )
        Clist.append(Ci)

    if not Clist:
        raise RuntimeError(
            "ECSW training produced zero columns for all mu samples. "
            "Increase ecsw_snapshot_percent or adjust snap_time_offset."
        )

    C = np.vstack(Clist)
    C_ecm = np.ascontiguousarray(C, dtype=np.float64)
    b = np.ascontiguousarray(C_ecm.sum(axis=1), dtype=np.float64)

    u = direct_left_singular_vectors(
        C_ecm.T,
        relative_tolerance=float(svd_relative_tolerance),
    )
    if u.shape[1] == 0:
        raise RuntimeError("Direct SVD produced an empty ECSW basis.")

    selector = EmpiricalCubatureMethod()
    selector.SetUp(
        u,
        InitialCandidatesSet=None,
        constrain_sum_of_weights=True,
        constrain_conditions=False,
    )
    selector.Run()

    num_cells = (grid_x.size - 1) * (grid_y.size - 1)
    weights = np.zeros(num_cells, dtype=np.float64)
    weights[selector.z] = selector.w

    elapsed = time.time() - t0
    denom = np.linalg.norm(b)
    rel_res = float(np.linalg.norm(C_ecm @ weights - b) / denom) if denom > 0.0 else np.nan

    return weights, rel_res, elapsed, ecsw_plan


def _load_or_build_ecsw_weights(
    total_modes,
    basis,
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu_samples,
    snap_folder,
    rebuild_weights=True,
    snap_time_offset=3,
    snapshot_percent=2.0,
    snapshot_random_seed=42,
    ensure_mu_coverage=True,
    svd_relative_tolerance=1e-8,
    weights_dir=None,
):
    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)

    weights_root = os.path.abspath(os.path.expanduser(str(weights_dir))) if weights_dir is not None else STAGE2_DIR
    os.makedirs(weights_root, exist_ok=True)
    preferred = os.path.join(weights_root, f"ecsw_weights_lspg_ntot{total_modes}.npy")
    if (not rebuild_weights) and os.path.exists(preferred):
        weights = np.asarray(np.load(preferred, allow_pickle=False), dtype=np.float64).reshape(-1)
        if weights.size != expected_num_cells:
            raise ValueError(
                f"Local ECSW weights size mismatch at '{preferred}': "
                f"got {weights.size}, expected {expected_num_cells}."
            )
        return (
            weights,
            preferred,
            "loaded_local",
            np.nan,
            int(np.sum(weights > 0.0)),
            None,
        )

    weights, rel_res, _, ecsw_plan = _compute_ecsw_weights(
        basis=basis,
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu_samples=mu_samples,
        snap_folder=snap_folder,
        snap_time_offset=snap_time_offset,
        snapshot_percent=snapshot_percent,
        snapshot_random_seed=snapshot_random_seed,
        ensure_mu_coverage=ensure_mu_coverage,
        svd_relative_tolerance=svd_relative_tolerance,
    )

    np.save(preferred, weights)
    n_ecsw = int(np.sum(weights > 0.0))
    return weights, preferred, "computed", rel_res, n_ecsw, ecsw_plan


def main(argv=None):
    # -----------------------------
    # User settings
    # -----------------------------
    total_modes = None
    solve_backend = "prom"
    save_rom_snaps = True
    make_plots = True
    max_its = 20
    relnorm_cutoff = 1e-5
    min_delta = 1e-2
    linear_solver = "lstsq"
    normal_eq_reg = 1e-12
    rebuild_ecsw_weights = True
    ecsw_snapshot_percent = 2.0
    ecsw_random_seed = 42
    ecsw_ensure_mu_coverage = True
    ecsw_snap_time_offset = 3
    ecsw_num_training_mu = 9
    ecsw_svd_rel_tol = 1e-8

    parser = argparse.ArgumentParser(
        description="Build Stage-2 qN dataset with selectable PROM/HPROM backend."
    )
    parser.add_argument("--backend", choices=("prom", "hprom"), default=solve_backend)
    parser.add_argument("--total-modes", type=int, default=total_modes)
    parser.add_argument(
        "--basis-path",
        type=str,
        default=None,
        help="Optional basis override. Must be passed together with --u-ref-path.",
    )
    parser.add_argument(
        "--u-ref-path",
        type=str,
        default=None,
        help="Optional reference-state override. Must be passed together with --basis-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit Stage-2 dataset directory. Default: Results/Stage2/prom_coeff_dataset_ntot{n}.",
    )
    parser.add_argument(
        "--ecsw-weights-dir",
        type=str,
        default=None,
        help="Optional directory where ECSW weights are stored/loaded. Default: output dataset directory when --output-dir is used, otherwise Results/Stage2.",
    )
    parser.add_argument(
        "--mu-pair",
        nargs=2,
        type=float,
        action="append",
        metavar=("MU1", "MU2"),
        default=None,
        help=(
            "Optional parameter pair to solve. Can be passed multiple times. "
            "If omitted, the full configured training grid is used."
        ),
    )
    parser.add_argument("--no-save-rom-snaps", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-its", type=int, default=max_its)
    parser.add_argument("--relnorm-cutoff", type=float, default=relnorm_cutoff)
    parser.add_argument("--min-delta", type=float, default=min_delta)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default=linear_solver)
    parser.add_argument("--normal-eq-reg", type=float, default=normal_eq_reg)
    parser.add_argument("--ecsw-snapshot-percent", type=float, default=ecsw_snapshot_percent)
    parser.add_argument("--ecsw-random-seed", type=int, default=ecsw_random_seed)
    parser.add_argument("--ecsw-snap-time-offset", type=int, default=ecsw_snap_time_offset)
    parser.add_argument("--ecsw-num-training-mu", type=int, default=ecsw_num_training_mu)
    parser.add_argument(
        "--ecsw-svd-rel-tol",
        type=float,
        default=ecsw_svd_rel_tol,
        help="Relative truncation tolerance for the deterministic direct SVD used before ECM.",
    )
    parser.add_argument(
        "--ecsw-ensure-mu-coverage",
        dest="ecsw_ensure_mu_coverage",
        action="store_true",
    )
    parser.add_argument(
        "--ecsw-no-ensure-mu-coverage",
        dest="ecsw_ensure_mu_coverage",
        action="store_false",
    )
    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument("--rebuild-ecsw", dest="rebuild_ecsw", action="store_true")
    rebuild_group.add_argument("--no-rebuild-ecsw", dest="rebuild_ecsw", action="store_false")
    parser.set_defaults(
        rebuild_ecsw=rebuild_ecsw_weights,
        ecsw_ensure_mu_coverage=ecsw_ensure_mu_coverage,
    )
    args = parser.parse_args(argv)

    solve_backend = str(args.backend).strip().lower()
    total_modes = args.total_modes
    save_rom_snaps = not bool(args.no_save_rom_snaps)
    make_plots = not bool(args.no_plots)
    max_its = int(args.max_its)
    relnorm_cutoff = float(args.relnorm_cutoff)
    min_delta = float(args.min_delta)
    linear_solver = str(args.linear_solver).strip().lower()
    normal_eq_reg = float(args.normal_eq_reg)
    rebuild_ecsw_weights = bool(args.rebuild_ecsw)
    ecsw_snapshot_percent = float(args.ecsw_snapshot_percent)
    ecsw_random_seed = int(args.ecsw_random_seed)
    ecsw_ensure_mu_coverage = bool(args.ecsw_ensure_mu_coverage)
    ecsw_snap_time_offset = int(args.ecsw_snap_time_offset)
    ecsw_num_training_mu = int(args.ecsw_num_training_mu)
    ecsw_svd_rel_tol = float(args.ecsw_svd_rel_tol)

    set_latex_plot_style()
    ensure_layout_dirs()

    if solve_backend not in ("prom", "hprom"):
        raise ValueError("solve_backend must be 'prom' or 'hprom'.")

    snap_folder = _select_snap_folder(PROJECT_ROOT)
    os.makedirs(snap_folder, exist_ok=True)

    Vtot, u_ref, basis_path, uref_path, pod_dir, total_modes, n_available = _load_pod_artifacts(
        total_modes,
        basis_override=args.basis_path,
        uref_override=args.u_ref_path,
    )
    w0 = np.asarray(W0, dtype=np.float64).copy()

    out_dir = (
        os.path.abspath(os.path.expanduser(str(args.output_dir)))
        if args.output_dir is not None
        else stage2_dataset_dir(total_modes)
    )
    per_mu_dir = os.path.join(out_dir, "per_mu")
    os.makedirs(per_mu_dir, exist_ok=True)

    if Vtot.shape[0] != w0.size:
        raise ValueError(
            f"Basis/state mismatch: basis has {Vtot.shape[0]} rows, "
            f"but W0 has size {w0.size}. Check grid/config consistency."
        )

    if args.mu_pair:
        mu_list = [np.asarray(pair, dtype=np.float64) for pair in args.mu_pair]
        mu_source = "custom_mu_pair"
    else:
        mu_list = get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        )
        mu_source = "configured_training_grid"
    if len(mu_list) == 0:
        raise RuntimeError("No parameter points were provided or generated.")

    ecsw_weights = None
    ecsw_weights_path = None
    ecsw_weights_source = None
    ecsw_residual = np.nan
    n_ecsw_elements = None
    ecsw_plan = None

    if solve_backend == "hprom":
        ecsw_num_training_mu = max(1, min(int(ecsw_num_training_mu), len(mu_list)))
        ecsw_mu_samples = mu_list[:ecsw_num_training_mu]
        ecsw_weights, ecsw_weights_path, ecsw_weights_source, ecsw_residual, n_ecsw_elements, ecsw_plan = _load_or_build_ecsw_weights(
            total_modes=total_modes,
            basis=Vtot,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu_samples=ecsw_mu_samples,
            snap_folder=snap_folder,
            rebuild_weights=rebuild_ecsw_weights,
            snap_time_offset=ecsw_snap_time_offset,
            snapshot_percent=ecsw_snapshot_percent,
            snapshot_random_seed=ecsw_random_seed,
            ensure_mu_coverage=ecsw_ensure_mu_coverage,
            svd_relative_tolerance=ecsw_svd_rel_tol,
            weights_dir=(
                args.ecsw_weights_dir
                if args.ecsw_weights_dir is not None
                else (out_dir if args.output_dir is not None else None)
            ),
        )

    print(f"[ROM-QN] solve_backend: {solve_backend}")
    print(f"[ROM-QN] POD directory: {pod_dir}")
    print(f"[ROM-QN] Loaded basis: {basis_path} (available={n_available}, using={total_modes})")
    print(f"[ROM-QN] Loaded u_ref: {uref_path}")
    print(f"[ROM-QN] Output dir:   {out_dir}")
    print(f"[ROM-QN] snap_folder:  {snap_folder}")
    print(f"[ROM-QN] save_rom_snaps={save_rom_snaps} | make_plots={make_plots}")
    if solve_backend == "hprom":
        print(f"[ROM-QN] ECSW weights: {ecsw_weights_path} ({ecsw_weights_source})")
        print(f"[ROM-QN] N_e = {n_ecsw_elements}")
        print(f"[ROM-QN] ECSW training trajectories used = {ecsw_num_training_mu}")
        print("[ROM-QN] ECSW snapshot mode = global_param_time_stratified")
        print(f"[ROM-QN] ECSW snapshot percent = {ecsw_snapshot_percent:.3f}")
        print(f"[ROM-QN] ECSW random seed = {ecsw_random_seed}")
        print(f"[ROM-QN] ECSW ensure mu coverage = {ecsw_ensure_mu_coverage}")
        print("[ROM-QN] ECSW SVD method = direct_dense_svd")
        print(f"[ROM-QN] ECSW SVD relative tolerance = {ecsw_svd_rel_tol:.3e}")
        if ecsw_plan is not None:
            print(
                f"[ROM-QN] ECSW selected {ecsw_plan['num_selected_total']} / "
                f"{ecsw_plan['num_candidates_total']} candidate snapshot pairs"
            )
            print(f"[ROM-QN] ECSW selected per mu: {ecsw_plan['num_selected_per_mu']}")
        print(f"[ROM-QN] rebuild_ecsw_weights = {rebuild_ecsw_weights}")

    t_ref = _time_grid(DT, NUM_STEPS)
    plot_steps = list(range(0, NUM_STEPS + 1, 100))
    if NUM_STEPS not in plot_steps:
        plot_steps.append(NUM_STEPS)

    # -----------------------------
    # Run ROM for each mu
    # -----------------------------
    for traj_id, mu in enumerate(mu_list, start=1):
        tag = _safe_mu_tag(mu)
        mu_dir = os.path.join(per_mu_dir, tag)
        os.makedirs(mu_dir, exist_ok=True)

        print(f"\n[ROM-QN] [{traj_id}/{len(mu_list)}] {solve_backend.upper()} solve for {tag}")
        t0 = time.time()

        if solve_backend == "prom":
            rom_snaps, qN, rom_stats = inviscid_burgers_implicit2D_LSPG(
                grid_x=GRID_X,
                grid_y=GRID_Y,
                w0=w0,
                dt=DT,
                num_steps=NUM_STEPS,
                mu=mu,
                basis=Vtot,
                u_ref=u_ref,
                max_its=max_its,
                relnorm_cutoff=relnorm_cutoff,
                min_delta=min_delta,
                linear_solver=linear_solver,
                normal_eq_reg=normal_eq_reg,
                return_red_coords=True,
            )
        else:
            qN, rom_stats = inviscid_burgers_implicit2D_LSPG_ecsw(
                grid_x=GRID_X,
                grid_y=GRID_Y,
                weights=ecsw_weights,
                w0=w0,
                dt=DT,
                num_steps=NUM_STEPS,
                mu=mu,
                basis=Vtot,
                u_ref=u_ref,
                max_its=max_its,
                relnorm_cutoff=relnorm_cutoff,
                min_delta=min_delta,
                linear_solver=linear_solver,
                normal_eq_reg=normal_eq_reg,
            )
            rom_snaps = u_ref[:, None] + Vtot @ qN

        if qN.ndim != 2:
            raise RuntimeError(f"Unexpected qN shape: {qN.shape}")

        reconstructed = u_ref[:, None] + Vtot @ qN
        state_scale = max(float(np.linalg.norm(rom_snaps)), np.finfo(np.float64).eps)
        coordinate_state_rel_error = float(
            np.linalg.norm(reconstructed - rom_snaps) / state_scale
        )
        if coordinate_state_rel_error > 1e-10:
            raise RuntimeError(
                "Solver coordinates are inconsistent with the returned ROM state: "
                f"relative reconstruction error={coordinate_state_rel_error:.3e}."
            )

        n_dofs, n_time = rom_snaps.shape
        t_vec = t_ref if len(t_ref) == n_time else DT * np.arange(n_time, dtype=np.float64)

        np.save(os.path.join(mu_dir, "mu.npy"), np.asarray(mu, dtype=np.float64))
        np.save(os.path.join(mu_dir, "t.npy"), t_vec)
        np.save(os.path.join(mu_dir, "qN.npy"), qN)
        np.save(os.path.join(mu_dir, "rom_stats.npy"), np.asarray(rom_stats, dtype=np.float64))

        if solve_backend == "prom":
            np.save(os.path.join(mu_dir, "prom_stats.npy"), np.asarray(rom_stats, dtype=np.float64))
        else:
            np.save(os.path.join(mu_dir, "hprom_stats.npy"), np.asarray(rom_stats, dtype=np.float64))

        if save_rom_snaps:
            np.save(os.path.join(mu_dir, "rom_snaps.npy"), rom_snaps)

        if make_plots:
            hdm_snaps = load_or_compute_snaps(
                mu=mu,
                grid_x=GRID_X,
                grid_y=GRID_Y,
                w0=w0,
                dt=DT,
                num_steps=NUM_STEPS,
                snap_folder=snap_folder,
            )

            fig, ax1, ax2 = plot_snaps(
                GRID_X,
                GRID_Y,
                hdm_snaps,
                plot_steps,
                label="HDM",
                color="black",
                linewidth=2.8,
                linestyle="solid",
            )
            plot_snaps(
                GRID_X,
                GRID_Y,
                rom_snaps,
                plot_steps,
                label="PROM" if solve_backend == "prom" else "HPROM",
                fig_ax=(fig, ax1, ax2),
                color="blue",
                linewidth=1.8,
                linestyle="solid",
            )
            ax1.legend()
            ax2.legend()
            plt.tight_layout()

            plot_name = "hdm_vs_prom.png" if solve_backend == "prom" else "hdm_vs_hprom.png"
            plot_path = os.path.join(mu_dir, plot_name)
            plt.savefig(plot_path, dpi=200)
            plt.close(fig)

        elapsed = time.time() - t0
        print(f"[ROM-QN] saved: {mu_dir}")
        print(
            "[ROM-QN] solver-coordinate state consistency = "
            f"{coordinate_state_rel_error:.3e}"
        )
        print(f"[ROM-QN] shape={n_dofs}x{n_time} | elapsed={elapsed:.2f} s")

    meta = {
        "solve_backend": solve_backend,
        "total_modes": int(total_modes),
        "n_available_modes": int(n_available),
        "coefficient_storage": "qN_only",
        "coordinate_recovery": "solver_coordinates",
        "coordinate_source": "solver_coordinates",
        "num_traj": int(len(mu_list)),
        "mu_source": mu_source,
        "mu_list": [[float(mu[0]), float(mu[1])] for mu in mu_list],
        "dt": float(DT),
        "num_steps": int(NUM_STEPS),
        "basis_path": basis_path,
        "u_ref_path": uref_path,
        "pod_dir": pod_dir,
        "save_rom_snaps": bool(save_rom_snaps),
        "make_plots": bool(make_plots),
        "snap_folder": snap_folder,
        "linear_solver": linear_solver,
        "normal_eq_reg": float(normal_eq_reg),
        "max_its": int(max_its),
        "relnorm_cutoff": float(relnorm_cutoff),
        "min_delta": float(min_delta),
        "state_size": int(Vtot.shape[0]),
        "reduced_size": int(Vtot.shape[1]),
        "ecsw_weights_path": ecsw_weights_path,
        "ecsw_weights_source": ecsw_weights_source,
        "ecsw_residual": float(ecsw_residual) if np.isfinite(ecsw_residual) else np.nan,
        "n_ecsw_elements": None if n_ecsw_elements is None else int(n_ecsw_elements),
        "ecsw_snap_time_offset": int(ecsw_snap_time_offset),
        "ecsw_num_training_mu": int(ecsw_num_training_mu),
        "ecsw_snapshot_mode": "global_param_time_stratified",
        "ecsw_snapshot_percent": float(ecsw_snapshot_percent),
        "ecsw_random_seed": int(ecsw_random_seed),
        "ecsw_ensure_mu_coverage": bool(ecsw_ensure_mu_coverage),
        "ecsw_svd_method": "direct_dense_svd",
        "ecsw_svd_relative_tolerance": float(ecsw_svd_rel_tol),
        "ecsw_num_candidates_total": (
            None if ecsw_plan is None else int(ecsw_plan["num_candidates_total"])
        ),
        "ecsw_num_selected_total": (
            None if ecsw_plan is None else int(ecsw_plan["num_selected_total"])
        ),
        "ecsw_num_selected_per_mu": (
            None if ecsw_plan is None else [int(v) for v in ecsw_plan["num_selected_per_mu"]]
        ),
        "rebuild_ecsw_weights": bool(rebuild_ecsw_weights),
    }
    np.save(os.path.join(out_dir, "meta.npy"), meta, allow_pickle=True)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, allow_nan=True)

    summary_path = os.path.join(out_dir, "stage2_summary.txt")
    write_kv_txt(
        summary_path,
        [
            ("solve_backend", solve_backend),
            ("dataset_dir", out_dir),
            ("per_mu_dir", per_mu_dir),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path),
            ("total_modes", total_modes),
            ("coefficient_storage", "qN_only"),
            (
                "coordinate_recovery",
                "solver_coordinates",
            ),
            ("coordinate_source", "solver_coordinates"),
            ("num_traj", len(mu_list)),
            ("mu_source", mu_source),
            ("mu_list", [[float(mu[0]), float(mu[1])] for mu in mu_list]),
            ("ecsw_num_training_mu", ecsw_num_training_mu),
            ("ecsw_snapshot_mode", "global_param_time_stratified"),
            ("ecsw_snapshot_percent", ecsw_snapshot_percent),
            ("ecsw_random_seed", ecsw_random_seed),
            ("ecsw_ensure_mu_coverage", ecsw_ensure_mu_coverage),
            ("ecsw_svd_method", "direct_dense_svd"),
            ("ecsw_svd_relative_tolerance", ecsw_svd_rel_tol),
            ("ecsw_snap_time_offset", ecsw_snap_time_offset),
            (
                "ecsw_num_candidates_total",
                None if ecsw_plan is None else int(ecsw_plan["num_candidates_total"]),
            ),
            (
                "ecsw_num_selected_total",
                None if ecsw_plan is None else int(ecsw_plan["num_selected_total"]),
            ),
            (
                "ecsw_num_selected_per_mu",
                None if ecsw_plan is None else [int(v) for v in ecsw_plan["num_selected_per_mu"]],
            ),
            ("ecsw_weights_path", ecsw_weights_path),
            ("ecsw_weights_source", ecsw_weights_source),
            ("ecsw_residual", ecsw_residual),
            ("n_ecsw_elements", n_ecsw_elements),
        ],
    )

    print("\n[ROM-QN] done.")
    print(f"[ROM-QN] per-parameter outputs under: {per_mu_dir}")
    print(f"[ROM-QN] summary: {summary_path}")


if __name__ == "__main__":
    main()
