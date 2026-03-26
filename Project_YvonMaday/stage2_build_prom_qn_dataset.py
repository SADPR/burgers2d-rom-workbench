#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: build PROM-solved qN dataset.

For each training parameter mu, this script runs an n_tot-dimensional ROM
(default: HPROM/ECSW-LSPG) and stores:
- mu.npy
- t.npy
- qN_p.npy   (primary block, first n modes)
- qN_s.npy   (secondary block, remaining n_tot-n modes)
- rom_stats.npy
- prom_stats.npy or hprom_stats.npy (backend-specific alias)
- rom_snaps.npy  (optional reconstructed full snapshots)
- hdm_vs_prom.png or hdm_vs_hprom.png (optional)
"""

import os
import sys
import time

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
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
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


def _load_pod_artifacts(requested_total_modes=None):
    # Prefer Stage1 outputs under Results, then fallback to legacy root files.
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
    snap_sample_factor=10,
    snap_time_offset=3,
):
    if snap_sample_factor < 1:
        raise ValueError("snap_sample_factor must be >= 1.")
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")

    Clist = []
    t0 = time.time()

    for mu in mu_samples:
        mu_snaps = load_or_compute_snaps(
            mu=mu,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
        )

        start_col = snap_time_offset
        stop_col = num_steps
        snaps_now = mu_snaps[:, start_col:stop_col:snap_sample_factor]
        snaps_prev = mu_snaps[:, 0:stop_col - snap_time_offset:snap_sample_factor]

        if snaps_now.shape[1] != snaps_prev.shape[1]:
            raise RuntimeError(
                "ECSW snapshot alignment failed: "
                f"snaps_now has {snaps_now.shape[1]} columns, "
                f"snaps_prev has {snaps_prev.shape[1]} columns."
            )
        if snaps_now.shape[1] == 0:
            raise RuntimeError(
                "ECSW training produced zero columns. "
                "Adjust snap_time_offset or snap_sample_factor."
            )

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

    C = np.vstack(Clist)
    C_ecm = np.ascontiguousarray(C, dtype=np.float64)
    b = np.ascontiguousarray(C_ecm.sum(axis=1), dtype=np.float64)

    rsvd = RandomizedSingularValueDecomposition()
    u, _, _, _ = rsvd.Calculate(C_ecm.T, 1e-8)

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

    return weights, rel_res, elapsed


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
    snap_sample_factor=10,
    snap_time_offset=3,
):
    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)

    os.makedirs(STAGE2_DIR, exist_ok=True)
    preferred = os.path.join(STAGE2_DIR, f"ecsw_weights_lspg_ntot{total_modes}.npy")
    if (not rebuild_weights) and os.path.exists(preferred):
        weights = np.asarray(np.load(preferred, allow_pickle=False), dtype=np.float64).reshape(-1)
        if weights.size != expected_num_cells:
            raise ValueError(
                f"Local ECSW weights size mismatch at '{preferred}': "
                f"got {weights.size}, expected {expected_num_cells}."
            )
        return weights, preferred, "loaded_local", np.nan, int(np.sum(weights > 0.0))

    weights, rel_res, _ = _compute_ecsw_weights(
        basis=basis,
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu_samples=mu_samples,
        snap_folder=snap_folder,
        snap_sample_factor=snap_sample_factor,
        snap_time_offset=snap_time_offset,
    )

    np.save(preferred, weights)
    n_ecsw = int(np.sum(weights > 0.0))
    return weights, preferred, "computed", rel_res, n_ecsw


def main():
    # -----------------------------
    # User settings
    # -----------------------------
    total_modes = None
    primary_modes = 10

    # backend in {"prom", "hprom"}
    solve_backend = "prom"

    save_rom_snaps = True
    make_plots = True
    max_its = 20
    relnorm_cutoff = 1e-5
    min_delta = 1e-2
    linear_solver = "lstsq"
    normal_eq_reg = 1e-12

    # HPROM/ECSW settings
    rebuild_ecsw_weights = True
    ecsw_snap_sample_factor = 10
    ecsw_snap_time_offset = 3
    ecsw_num_training_mu = 1

    set_latex_plot_style()
    ensure_layout_dirs()

    if solve_backend not in ("prom", "hprom"):
        raise ValueError("solve_backend must be 'prom' or 'hprom'.")

    snap_folder = _select_snap_folder(PROJECT_ROOT)
    os.makedirs(snap_folder, exist_ok=True)

    Vtot, u_ref, basis_path, uref_path, pod_dir, total_modes, n_available = _load_pod_artifacts(total_modes)
    w0 = np.asarray(W0, dtype=np.float64).copy()

    if not (total_modes > primary_modes):
        raise ValueError(
            f"primary_modes={primary_modes} must be smaller than total_modes={total_modes}. "
            "Either lower primary_modes or increase retained POD modes in stage1_pod.py."
        )

    out_dir = stage2_dataset_dir(total_modes)
    per_mu_dir = os.path.join(out_dir, "per_mu")
    os.makedirs(per_mu_dir, exist_ok=True)

    if Vtot.shape[0] != w0.size:
        raise ValueError(
            f"Basis/state mismatch: basis has {Vtot.shape[0]} rows, "
            f"but W0 has size {w0.size}. Check grid/config consistency."
        )

    mu_list = get_snapshot_params(
        mu1_range=MU1_RANGE,
        mu2_range=MU2_RANGE,
        samples_per_mu=SAMPLES_PER_MU,
    )
    if len(mu_list) == 0:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    ecsw_weights = None
    ecsw_weights_path = None
    ecsw_weights_source = None
    ecsw_residual = np.nan
    n_ecsw_elements = None

    if solve_backend == "hprom":
        ecsw_num_training_mu = max(1, min(int(ecsw_num_training_mu), len(mu_list)))
        ecsw_mu_samples = mu_list[:ecsw_num_training_mu]
        ecsw_weights, ecsw_weights_path, ecsw_weights_source, ecsw_residual, n_ecsw_elements = _load_or_build_ecsw_weights(
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
            snap_sample_factor=ecsw_snap_sample_factor,
            snap_time_offset=ecsw_snap_time_offset,
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
            rom_snaps, rom_stats = inviscid_burgers_implicit2D_LSPG(
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
            )
            qN = Vtot.T @ (rom_snaps - u_ref[:, None])
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

        n_dofs, n_time = rom_snaps.shape
        t_vec = t_ref if len(t_ref) == n_time else DT * np.arange(n_time, dtype=np.float64)

        qN_p = qN[:primary_modes, :]
        qN_s = qN[primary_modes:total_modes, :]

        np.save(os.path.join(mu_dir, "mu.npy"), np.asarray(mu, dtype=np.float64))
        np.save(os.path.join(mu_dir, "t.npy"), t_vec)
        np.save(os.path.join(mu_dir, "qN_p.npy"), qN_p)
        np.save(os.path.join(mu_dir, "qN_s.npy"), qN_s)
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
        print(f"[ROM-QN] shape={n_dofs}x{n_time} | elapsed={elapsed:.2f} s")

    meta = {
        "solve_backend": solve_backend,
        "total_modes": int(total_modes),
        "n_available_modes": int(n_available),
        "primary_modes": int(primary_modes),
        "secondary_modes": int(total_modes - primary_modes),
        "num_traj": int(len(mu_list)),
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
        "ecsw_snap_sample_factor": int(ecsw_snap_sample_factor),
        "ecsw_snap_time_offset": int(ecsw_snap_time_offset),
        "ecsw_num_training_mu": int(ecsw_num_training_mu),
        "rebuild_ecsw_weights": bool(rebuild_ecsw_weights),
    }
    np.save(os.path.join(out_dir, "meta.npy"), meta, allow_pickle=True)

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
            ("primary_modes", primary_modes),
            ("num_traj", len(mu_list)),
            ("ecsw_num_training_mu", ecsw_num_training_mu),
            ("ecsw_snap_sample_factor", ecsw_snap_sample_factor),
            ("ecsw_snap_time_offset", ecsw_snap_time_offset),
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
