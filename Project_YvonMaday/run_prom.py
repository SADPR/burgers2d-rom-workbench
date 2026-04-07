#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified non-ANN single-parameter runner with selectable backend.

Backends:
- solve_backend='prom'  -> full LSPG solve
- solve_backend='hprom' -> ECSW hyper-reduced LSPG solve

By default, for HPROM this script first tries to reuse ECSW weights created in
Stage 2 (same ntot), and only falls back to local/recomputed weights if needed.
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU
from burgers.core import (
    get_snapshot_params,
    load_or_compute_snaps,
    plot_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
)
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.linear_manifold import (
    compute_ECSW_training_matrix_2D,
    inviscid_burgers_implicit2D_LSPG,
    inviscid_burgers_implicit2D_LSPG_ecsw,
)
from burgers.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

try:
    from project_layout import (
        STAGE2_DIR,
        RUNS_ECSW_DIR,
        RUNS_LINEAR_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        write_kv_txt,
    )
except ModuleNotFoundError:
    from .project_layout import (
        STAGE2_DIR,
        RUNS_ECSW_DIR,
        RUNS_LINEAR_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        write_kv_txt,
    )


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


def _select_snap_folder(project_root):
    candidates = [
        os.path.join(project_root, "Results", "param_snaps"),
        os.path.join(project_root, "param_snaps"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _load_basis_and_reference(total_modes=None):
    basis_path = resolve_stage1_artifact("basis.npy")
    uref_path = resolve_stage1_artifact("u_ref.npy")

    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Missing basis file: {basis_path}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got shape {basis.shape}")

    n_available = int(basis.shape[1])
    if total_modes is None:
        total_modes = n_available
    else:
        total_modes = int(total_modes)
        if total_modes < 1 or total_modes > n_available:
            raise ValueError(
                f"Requested total_modes={total_modes}, but basis has {n_available} modes."
            )

    Vtot = basis[:, :total_modes]

    if os.path.exists(uref_path):
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(Vtot.shape[0], dtype=np.float64)

    if u_ref.size != Vtot.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {Vtot.shape[0]} from basis rows."
        )

    return Vtot, u_ref, basis_path, uref_path, total_modes, n_available


def _resolve_stage2_ecsw_weights_path(total_modes):
    dataset_candidates = [
        os.path.join(STAGE2_DIR, f"prom_coeff_dataset_ntot{int(total_modes)}"),
        os.path.join(THIS_DIR, f"prom_coeff_dataset_ntot{int(total_modes)}"),
    ]

    for dataset_dir in dataset_candidates:
        meta_path = os.path.join(dataset_dir, "meta.npy")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = np.load(meta_path, allow_pickle=True).item()
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        wpath = meta.get("ecsw_weights_path", None)
        if isinstance(wpath, str) and len(wpath) > 0:
            if os.path.exists(wpath):
                return wpath
            cand = os.path.join(dataset_dir, wpath)
            if os.path.exists(cand):
                return cand

    direct_candidates = [
        os.path.join(STAGE2_DIR, f"ecsw_weights_lspg_ntot{int(total_modes)}.npy"),
        os.path.join(THIS_DIR, f"ecsw_weights_lspg_ntot{int(total_modes)}.npy"),
    ]
    for path in direct_candidates:
        if os.path.exists(path):
            return path
    return None


def _compute_ecsw_weights(
    basis,
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu_samples,
    snap_folder,
    snap_sample_factor=50,
    snap_time_offset=3,
):
    if snap_sample_factor < 1:
        raise ValueError("snap_sample_factor must be >= 1.")
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")

    clist = []
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

        stop_col = num_steps
        snaps_now = mu_snaps[:, snap_time_offset:stop_col:snap_sample_factor]
        snaps_prev = mu_snaps[:, 0:stop_col - snap_time_offset:snap_sample_factor]

        if snaps_now.shape[1] != snaps_prev.shape[1]:
            raise RuntimeError(
                "ECSW snapshot alignment failed: "
                f"snaps_now has {snaps_now.shape[1]} columns, snaps_prev has {snaps_prev.shape[1]} columns."
            )
        if snaps_now.shape[1] == 0:
            raise RuntimeError(
                "ECSW training produced zero columns. Adjust snap_time_offset or snap_sample_factor."
            )

        ci = compute_ECSW_training_matrix_2D(
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
        clist.append(ci)

    C = np.vstack(clist)
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
    stage2_weights_path=None,
    rebuild_weights=False,
    snap_sample_factor=50,
    snap_time_offset=3,
):
    os.makedirs(RUNS_ECSW_DIR, exist_ok=True)
    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)
    local_weights_path = os.path.join(RUNS_ECSW_DIR, f"ecsw_weights_linear_ntot{total_modes}.npy")

    def _load_checked(path):
        w = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if w.size != expected_num_cells:
            raise ValueError(
                f"ECSW weights size mismatch at '{path}': got {w.size}, expected {expected_num_cells}."
            )
        return w

    if not rebuild_weights:
        if stage2_weights_path is not None and os.path.exists(stage2_weights_path):
            weights = _load_checked(stage2_weights_path)
            return weights, stage2_weights_path, "loaded_stage2", np.nan, int(np.sum(weights > 0.0))

        if os.path.exists(local_weights_path):
            weights = _load_checked(local_weights_path)
            return weights, local_weights_path, "loaded_local", np.nan, int(np.sum(weights > 0.0))

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

    np.save(local_weights_path, weights)
    return weights, local_weights_path, "computed", rel_res, int(np.sum(weights > 0.0))


def main(
    mu_test=(4.56, 0.019),
    solve_backend="hprom",
    use_ecsw=True,
    total_modes=None,
    rebuild_ecsw_weights=False,
    ecsw_snap_sample_factor=50,
    ecsw_snap_time_offset=3,
    ecsw_num_training_mu=9,
    use_stage2_ecsw_weights=True,
    save_rom_snaps=True,
    make_plots=True,
    compute_hdm_error=True,
    output_root_dir=None,
):
    mu_test = [float(mu_test[0]), float(mu_test[1])]

    max_its = 20
    relnorm_cutoff = 1e-5
    min_delta = 1e-2
    linear_solver = "lstsq"
    normal_eq_reg = 1e-12

    if make_plots:
        set_latex_plot_style()
    ensure_layout_dirs()
    if output_root_dir is None:
        output_root_dir = RUNS_LINEAR_DIR
    else:
        output_root_dir = os.path.abspath(output_root_dir)
    os.makedirs(output_root_dir, exist_ok=True)

    solve_backend = str(solve_backend).strip().lower()
    if solve_backend not in ("prom", "hprom"):
        raise ValueError("solve_backend must be 'prom' or 'hprom'.")

    effective_backend = solve_backend
    if solve_backend == "hprom" and not use_ecsw:
        print("[Linear] solve_backend='hprom' with use_ecsw=False -> PROM solve.")
        effective_backend = "prom"
    if solve_backend == "prom" and use_ecsw:
        print("[Linear] use_ecsw=True ignored because solve_backend='prom'.")

    Vtot, u_ref, basis_path, uref_path, total_modes, n_available = _load_basis_and_reference(total_modes)

    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != Vtot.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {Vtot.shape[0]} from basis rows."
        )

    snap_folder = _select_snap_folder(PROJECT_ROOT)
    os.makedirs(snap_folder, exist_ok=True)

    print(f"[Linear] solve_backend(requested) = {solve_backend}")
    print(f"[Linear] solve_backend(effective) = {effective_backend}")
    print(f"[Linear] use_ecsw = {use_ecsw}")
    print(f"[Linear] use_stage2_ecsw_weights = {use_stage2_ecsw_weights}")
    print(f"[Linear] basis = {basis_path} (available={n_available}, using={total_modes})")
    print(f"[Linear] u_ref = {uref_path if os.path.exists(uref_path) else 'zeros'}")
    print(f"[Linear] snap_folder = {snap_folder}")

    weights = None
    weights_path = None
    weights_source = None
    ecsw_residual = np.nan
    n_ecsw_elements = None
    ecsw_setup_elapsed = 0.0
    online_solve_elapsed = np.nan

    if effective_backend == "hprom":
        stage2_weights_path = _resolve_stage2_ecsw_weights_path(total_modes) if use_stage2_ecsw_weights else None

        mu_train_candidates = get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        )
        ecsw_num_training_mu = max(1, min(int(ecsw_num_training_mu), len(mu_train_candidates)))
        mu_train_list = mu_train_candidates[:ecsw_num_training_mu]

        t_ecsw0 = time.time()
        weights, weights_path, weights_source, ecsw_residual, n_ecsw_elements = _load_or_build_ecsw_weights(
            total_modes=total_modes,
            basis=Vtot,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu_samples=mu_train_list,
            snap_folder=snap_folder,
            stage2_weights_path=stage2_weights_path,
            rebuild_weights=rebuild_ecsw_weights,
            snap_sample_factor=ecsw_snap_sample_factor,
            snap_time_offset=ecsw_snap_time_offset,
        )
        ecsw_setup_elapsed = time.time() - t_ecsw0

        print(f"[Linear] Stage2 ECSW candidate: {stage2_weights_path}")
        print(f"[Linear] ECSW weights: {weights_path} ({weights_source})")
        print(f"[Linear] ECSW training trajectories used = {ecsw_num_training_mu}")
        print(f"[Linear] N_e = {n_ecsw_elements}")
        print(f"[Linear] ECSW residual = {ecsw_residual}")

    t0 = time.time()
    if effective_backend == "prom":
        rom_snaps, rom_times = inviscid_burgers_implicit2D_LSPG(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu_test,
            basis=Vtot,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        qN = Vtot.T @ (rom_snaps - u_ref[:, None])
        online_solve_elapsed = time.time() - t0
    else:
        qN, rom_times = inviscid_burgers_implicit2D_LSPG_ecsw(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            weights=weights,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu_test,
            basis=Vtot,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        online_solve_elapsed = time.time() - t0
        rom_snaps = u_ref[:, None] + Vtot @ qN

    elapsed = online_solve_elapsed
    num_its, jac_time, res_time, ls_time = rom_times

    if qN.ndim != 2:
        raise RuntimeError(f"Unexpected qN shape: {qN.shape}")

    t_vec = DT * np.arange(qN.shape[1], dtype=np.float64)

    backend_tag = "hprom" if effective_backend == "hprom" else "prom"
    tag = _safe_mu_tag(mu_test)
    run_tag = f"linear_{backend_tag}_{tag}_ntot{total_modes}"
    out_dir = os.path.join(output_root_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "mu.npy"), np.asarray(mu_test, dtype=np.float64))
    np.save(os.path.join(out_dir, "t.npy"), t_vec)
    np.save(os.path.join(out_dir, "qN.npy"), qN)
    np.save(os.path.join(out_dir, "rom_stats.npy"), np.asarray(rom_times, dtype=np.float64))
    if save_rom_snaps:
        np.save(os.path.join(out_dir, "rom_snaps.npy"), rom_snaps)

    hdm_snaps = None
    rel_err = np.nan
    need_hdm = bool(make_plots or compute_hdm_error)
    if need_hdm:
        hdm_snaps = load_or_compute_snaps(
            mu=mu_test,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            snap_folder=snap_folder,
        )
        rel_err = 100.0 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)

    out_png = os.path.join(out_dir, "hdm_vs_rom.png")
    if make_plots:
        if hdm_snaps is None:
            raise RuntimeError("HDM snapshots are required for plotting but were not loaded.")
        plot_steps = list(range(0, NUM_STEPS + 1, 100))
        if NUM_STEPS not in plot_steps:
            plot_steps.append(NUM_STEPS)

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
            label="HPROM" if effective_backend == "hprom" else "PROM",
            fig_ax=(fig, ax1, ax2),
            color="blue",
            linewidth=1.8,
            linestyle="solid",
        )
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    summary_txt = os.path.join(out_dir, "summary.txt")
    write_kv_txt(
        summary_txt,
        [
            ("mu_test", mu_test),
            ("solve_backend_requested", solve_backend),
            ("solve_backend_effective", effective_backend),
            ("use_ecsw", use_ecsw),
            ("use_stage2_ecsw_weights", use_stage2_ecsw_weights),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path if os.path.exists(uref_path) else "zeros"),
            ("total_modes", total_modes),
            ("save_rom_snaps", bool(save_rom_snaps)),
            ("make_plots", bool(make_plots)),
            ("compute_hdm_error", bool(compute_hdm_error)),
            ("ecsw_num_training_mu", ecsw_num_training_mu),
            ("ecsw_snap_sample_factor", ecsw_snap_sample_factor),
            ("ecsw_snap_time_offset", ecsw_snap_time_offset),
            ("ecsw_weights_path", weights_path if effective_backend == "hprom" else "N/A"),
            ("ecsw_weights_source", weights_source),
            ("ecsw_residual", ecsw_residual),
            ("n_ecsw_elements", n_ecsw_elements),
            ("ecsw_setup_elapsed_s", ecsw_setup_elapsed),
            ("online_solve_elapsed_s", online_solve_elapsed),
            ("elapsed_s", elapsed),
            ("num_iterations", num_its),
            ("jac_time_s", jac_time),
            ("res_time_s", res_time),
            ("ls_time_s", ls_time),
            ("relative_error_percent", rel_err),
            ("output_dir", out_dir),
            ("output_root_dir", output_root_dir),
            ("snap_folder", snap_folder),
        ],
    )

    print(f"[Linear] ecsw_setup_elapsed = {ecsw_setup_elapsed:.3e} s")
    print(f"[Linear] online_solve_elapsed = {online_solve_elapsed:.3e} s")
    print(f"[Linear] elapsed = {elapsed:.3e} s")
    print(f"[Linear] its={num_its} | jac={jac_time:.3e} | res={res_time:.3e} | ls={ls_time:.3e}")
    if np.isfinite(rel_err):
        print(f"[Linear] relative error vs HDM: {rel_err:.2f}%")
    else:
        print("[Linear] relative error vs HDM: not computed")
    print(f"[Linear] outputs: {out_dir}")
    print(f"[Linear] summary: {summary_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified linear PROM/HPROM single-parameter runner.")
    parser.add_argument("--backend", choices=["prom", "hprom"], default="hprom", help="ROM backend")
    parser.add_argument("--mu1", type=float, default=4.56, help="First parameter value")
    parser.add_argument("--mu2", type=float, default=0.019, help="Second parameter value")
    parser.add_argument("--total-modes", type=int, default=None, help="Total reduced modes (n_tot)")
    parser.add_argument("--rebuild-ecsw", action="store_true", help="Force ECSW recomputation")
    parser.add_argument("--ecsw-num-training-mu", type=int, default=9, help="Number of mu trajectories for ECSW training")
    parser.add_argument("--ecsw-snap-sample-factor", type=int, default=50, help="Snapshot stride for ECSW training")
    parser.add_argument("--ecsw-snap-time-offset", type=int, default=3, help="Snapshot start offset for ECSW training")
    parser.add_argument("--no-ecsw", action="store_true", help="Disable ECSW even if backend='hprom'")
    parser.add_argument("--no-stage2-ecsw", action="store_true", help="Do not reuse Stage2 ECSW weights")
    parser.add_argument("--no-save-rom-snaps", action="store_true", help="Do not save rom_snaps.npy")
    parser.add_argument("--no-plot", action="store_true", help="Disable HDM-vs-ROM plotting")
    parser.add_argument("--no-hdm-error", action="store_true", help="Skip HDM error computation when plotting is disabled")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional custom output root directory (default: Results/Runs/Linear).",
    )
    args = parser.parse_args()

    main(
        mu_test=(args.mu1, args.mu2),
        solve_backend=args.backend,
        use_ecsw=not args.no_ecsw,
        total_modes=args.total_modes,
        rebuild_ecsw_weights=args.rebuild_ecsw,
        ecsw_snap_sample_factor=args.ecsw_snap_sample_factor,
        ecsw_snap_time_offset=args.ecsw_snap_time_offset,
        ecsw_num_training_mu=args.ecsw_num_training_mu,
        use_stage2_ecsw_weights=not args.no_stage2_ecsw,
        save_rom_snaps=not args.no_save_rom_snaps,
        make_plots=not args.no_plot,
        compute_hdm_error=not args.no_hdm_error,
        output_root_dir=args.output_root,
    )
