#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TMP sanity check for Case 2 Petrov-Galerkin (PG) with oracle secondary modes.

Purpose
-------
Use qbar(t) from the linear n_tot-PROM trajectory as an oracle map in Case 2 PG:

    w(y,t) = u_ref + V y + Vbar qbar_oracle(t)

This is intended only for verification/debug checks and writes outputs with "tmp" tags.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
REPO_DIR = PROJECT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from burgers.config import DT, NUM_STEPS  # noqa: E402
from burgers.core import load_or_compute_snaps, plot_snaps  # noqa: E402
from burgers.pod_ann_manifold import (  # noqa: E402
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin,
)


def _looks_like_work_dir(path):
    p = Path(path)
    return (p / "Results" / "Stage1" / "basis.npy").exists() and (p / "Results" / "Runs" / "Linear").exists()


def _resolve_work_dir(work_dir_arg):
    if work_dir_arg:
        wd = Path(work_dir_arg).expanduser().resolve()
        if not wd.exists():
            raise FileNotFoundError(
                f"--work-dir does not exist: {wd}"
            )
        return wd

    candidates = [
        THIS_DIR,
        THIS_DIR / "250x250",
        THIS_DIR.parent,
        THIS_DIR.parent / "250x250",
    ]
    for c in candidates:
        if _looks_like_work_dir(c):
            return c.resolve()

    checked = "\n".join(f"  - {c.resolve()}" for c in candidates)
    raise FileNotFoundError(
        "Could not auto-detect workspace directory for this tmp check.\n"
        "Pass --work-dir explicitly. Checked:\n"
        f"{checked}"
    )


def _resolve_snap_folder_tmp(work_dir):
    work_dir = Path(work_dir).resolve()

    # 1) Prefer the folder recorded during Stage-1 POD (most faithful to production runs).
    meta_path = work_dir / "Results" / "Stage1" / "stage1_pod_metadata.npz"
    if meta_path.exists():
        try:
            meta = np.load(meta_path, allow_pickle=False)
            if "snap_folder" in meta.files:
                snap_folder = Path(str(meta["snap_folder"].item())).expanduser().resolve()
                if snap_folder.exists():
                    return snap_folder, "stage1_metadata"
        except Exception:
            pass

    # 2) Match production run logic: repo-level Results/param_snaps.
    repo_snap_folder = REPO_DIR / "Results" / "param_snaps"
    if repo_snap_folder.exists():
        return repo_snap_folder.resolve(), "repo_results_param_snaps"

    # 3) Additional practical fallbacks.
    candidates = [
        work_dir / "Results" / "param_snaps",
        work_dir / "param_snaps",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve(), "fallback_existing"

    # 4) Default creation path (same style as production scripts).
    repo_snap_folder.mkdir(parents=True, exist_ok=True)
    return repo_snap_folder.resolve(), "repo_results_param_snaps_created"


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _set_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.2,
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "figure.figsize": (12, 8),
        }
    )


def _write_kv_txt(path, kv_pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in kv_pairs:
            f.write(f"{k}: {v}\n")


def _load_hdm_snaps_tmp(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder):
    mu1 = mu[0]
    mu2 = mu[1]
    candidates = [
        snap_folder / f"mu1_{mu1}+mu2_{mu2}.npy",
        snap_folder / f"mu1_{mu1:.3f}+mu2_{mu2:.4f}.npy",
        snap_folder / f"mu1_{mu1:.2f}+mu2_{mu2:.3f}.npy",
    ]
    for p in candidates:
        if p.exists():
            return np.asarray(np.load(p, allow_pickle=False), dtype=np.float64), p

    hdm = load_or_compute_snaps(
        mu=mu,
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        snap_folder=str(snap_folder),
    )
    return np.asarray(hdm, dtype=np.float64), None


def _infer_square_grid_tmp(n_state, xlim=(0.0, 100.0), ylim=(0.0, 100.0)):
    n_cells = int(n_state // 2)
    n_side = int(round(np.sqrt(n_cells)))
    if n_side * n_side != n_cells:
        raise ValueError(
            f"Cannot infer square grid from state size {n_state}: n_cells={n_cells} is not a square."
        )
    nx = n_side
    ny = n_side
    grid_x = np.linspace(float(xlim[0]), float(xlim[1]), nx + 1, dtype=np.float64)
    grid_y = np.linspace(float(ylim[0]), float(ylim[1]), ny + 1, dtype=np.float64)
    return grid_x, grid_y, nx, ny


class OracleCase2FromLinearQNTmp(nn.Module):
    """
    Temporary oracle map qbar(mu,t) for Case 2, built from linear PROM qN(t).

    Input:  x = [mu1, mu2, t]   (mu ignored; t used for lookup)
    Output: qbar_oracle(t)
    """

    def __init__(self, qbar_table, dt, t0=0.0):
        super().__init__()
        # Keep oracle coefficients in float64 so the 0% oracle check is not
        # polluted by float32 roundoff when comparing against linear qN.
        qb = np.asarray(qbar_table, dtype=np.float64)
        if qb.ndim != 2:
            raise ValueError(f"qbar_table must be 2D, got shape {qb.shape}")
        self.register_buffer("qbar_table", torch.from_numpy(qb))  # (n_s, n_t)
        self.dt = float(dt)
        self.t0 = float(t0)
        self.n_t = int(qb.shape[1])

    def _time_to_index(self, t_tensor):
        idx = torch.round((t_tensor - self.t0) / self.dt).long()
        return torch.clamp(idx, min=0, max=self.n_t - 1)

    def forward(self, x_raw):
        x = x_raw
        if x.ndim == 1:
            idx = self._time_to_index(x[2])
            return self.qbar_table[:, idx]

        if x.ndim == 2:
            idx = self._time_to_index(x[:, 2])
            return self.qbar_table.index_select(1, idx).T

        raise ValueError(f"Unsupported input shape for oracle model: {tuple(x.shape)}")


def _load_basis_and_reference(work_dir, n_tot, basis_path_override=None, uref_path_override=None):
    if basis_path_override:
        basis_path = Path(basis_path_override).expanduser().resolve()
    else:
        basis_path = work_dir / "Results" / "Stage1" / "basis.npy"
    if uref_path_override:
        uref_path = Path(uref_path_override).expanduser().resolve()
    else:
        uref_path = work_dir / "Results" / "Stage1" / "u_ref.npy"

    if not basis_path.exists():
        raise FileNotFoundError(f"Missing basis: {basis_path}")
    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got shape {basis.shape}")
    if basis.shape[1] < n_tot:
        raise ValueError(f"basis has {basis.shape[1]} modes, requested n_tot={n_tot}")

    if uref_path.exists():
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(basis.shape[0], dtype=np.float64)

    if u_ref.size != basis.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {basis.shape[0]} from basis rows"
        )

    return basis[:, :n_tot], u_ref, basis_path, uref_path


def _build_perturbed_qbar(qbar_oracle, perturb_percent, perturb_seed):
    qbar_oracle = np.asarray(qbar_oracle, dtype=np.float64)
    p = float(perturb_percent)
    if p <= 0.0:
        return qbar_oracle.copy(), np.zeros_like(qbar_oracle)

    rng = np.random.default_rng(int(perturb_seed))
    noise = rng.standard_normal(size=qbar_oracle.shape)
    den = np.linalg.norm(noise)
    if den <= 0.0:
        raise RuntimeError("Random perturbation noise has zero norm.")
    target_abs = (p / 100.0) * np.linalg.norm(qbar_oracle)
    delta = noise * (target_abs / den)
    return qbar_oracle + delta, delta


def _solve_case2_pg_oracle(
    *,
    solver_variant,
    oracle_model,
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    v,
    vbar,
    u_ref,
    max_its,
    relnorm_cutoff,
    min_delta,
    linear_solver,
    normal_eq_reg,
    y_init_table=None,
    wp_table=None,
):
    t0 = time.time()
    if str(solver_variant).strip().lower() == "plain":
        snaps, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu,
            ann_model=oracle_model,
            ref=None,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            max_its=int(max_its),
            relnorm_cutoff=float(relnorm_cutoff),
            min_delta=float(min_delta),
            y_init_table=y_init_table,
            wp_table=wp_table,
        )
    else:
        snaps, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin(
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu,
            ann_model=oracle_model,
            ref=None,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            max_its=int(max_its),
            relnorm_cutoff=float(relnorm_cutoff),
            min_delta=float(min_delta),
            linear_solver=str(linear_solver),
            normal_eq_reg=float(normal_eq_reg),
            y_init_table=y_init_table,
            wp_table=wp_table,
        )
    elapsed = time.time() - t0
    return snaps, rom_times, elapsed


def _align_time_series(*arrays):
    """Trim 2D (state x time) arrays to a shared time length."""
    if len(arrays) == 0:
        raise ValueError("No arrays provided for alignment.")
    min_t = min(int(np.asarray(a).shape[1]) for a in arrays)
    return [np.asarray(a)[:, :min_t] for a in arrays], min_t


def _project_snaps_to_basis_ls(basis, u_ref, snaps):
    """
    Least-squares reduced coordinates for possibly non-orthonormal bases.
    """
    basis = np.asarray(basis, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    snaps = np.asarray(snaps, dtype=np.float64)
    if snaps.ndim != 2:
        raise ValueError(f"snaps must be 2D, got shape {snaps.shape}")
    return np.linalg.lstsq(basis, snaps - u_ref[:, None], rcond=None)[0]


def _cross_coupling_fro_norm(v, vbar):
    v = np.asarray(v, dtype=np.float64)
    vbar = np.asarray(vbar, dtype=np.float64)
    return float(np.linalg.norm(v.T @ vbar, ord="fro"))


def _validate_oracle_mode(mode):
    m = str(mode).strip().lower()
    if m == "legacy":
        return "secondary_oracle_only"
    raise ValueError(
        f"Unsupported oracle mode: {mode}. This diagnostic intentionally only "
        "supports 'legacy': prescribe oracle secondary coefficients and let "
        "Gauss-Newton recover the primary coordinates."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TMP check: Case 2 PG using oracle qbar(t) from linear PROM qN."
    )
    parser.add_argument("--mu1", type=float, default=4.875)
    parser.add_argument("--mu2", type=float, default=0.0225)
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--n-tot", type=int, default=151)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1e-12)
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help=(
            "Workspace root containing Results/Stage1 and Results/Runs. "
            "If omitted, auto-detected."
        ),
    )
    parser.add_argument(
        "--linear-run-dir",
        type=str,
        default=None,
        help="Optional override for linear run directory containing qN.npy and rom_snaps.npy.",
    )
    parser.add_argument(
        "--basis-path",
        type=str,
        default=None,
        help="Optional basis path override (e.g. Results_Maday/.../basis_weighted.npy).",
    )
    parser.add_argument(
        "--u-ref-path",
        type=str,
        default=None,
        help="Optional u_ref path override (must match basis rows).",
    )
    parser.add_argument(
        "--qbar-perturb-percent",
        type=float,
        default=0.0,
        help="Relative perturbation level for oracle qbar in percent of ||qbar||_F.",
    )
    parser.add_argument(
        "--qbar-perturb-seed",
        type=int,
        default=42,
        help="RNG seed for qbar perturbation noise.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--run-tag-prefix",
        type=str,
        default="tmp_case2_pg_oracle_prom",
        help="Output tag prefix.",
    )
    parser.add_argument(
        "--solver-variant",
        choices=("pg", "plain"),
        default="pg",
        help="`pg`: enriched residual testing (current default), `plain`: standard Case2 LSPG.",
    )
    parser.add_argument(
        "--oracle-mode",
        choices=("legacy",),
        default="legacy",
        help=(
            "Only supported mode for this diagnostic: prescribe oracle secondary "
            "coefficients and let the solver recover the primary coordinates."
        ),
    )
    args = parser.parse_args(argv)

    _set_plot_style()

    mu = [float(args.mu1), float(args.mu2)]
    n_p = int(args.n_primary)
    n_tot = int(args.n_tot)
    dt = float(args.dt)
    num_steps = int(args.num_steps)
    mu_tag = _safe_mu_tag(mu)
    work_dir = _resolve_work_dir(args.work_dir)

    if args.linear_run_dir:
        linear_dir = Path(args.linear_run_dir).resolve()
    else:
        linear_dir = work_dir / "Results" / "Runs" / "Linear" / f"linear_prom_{mu_tag}_ntot{n_tot}"
    qn_path = linear_dir / "qN.npy"
    lin_snaps_path = linear_dir / "rom_snaps.npy"
    if not qn_path.exists():
        raise FileNotFoundError(f"Missing linear qN file: {qn_path}")
    if not lin_snaps_path.exists():
        raise FileNotFoundError(f"Missing linear rom_snaps file: {lin_snaps_path}")

    qn_linear_raw = np.asarray(np.load(qn_path, allow_pickle=False), dtype=np.float64)
    lin_snaps = np.asarray(np.load(lin_snaps_path, allow_pickle=False), dtype=np.float64)
    if qn_linear_raw.ndim != 2:
        raise ValueError(f"linear qN must be 2D, got shape {qn_linear_raw.shape}")
    if qn_linear_raw.shape[0] < n_tot:
        raise ValueError(f"linear qN has {qn_linear_raw.shape[0]} modes, requested n_tot={n_tot}")
    if n_p < 1 or n_p >= n_tot:
        raise ValueError(f"Invalid split n_primary={n_p}, n_tot={n_tot}")

    basis, u_ref, basis_path, uref_path = _load_basis_and_reference(
        work_dir,
        n_tot,
        basis_path_override=args.basis_path,
        uref_path_override=args.u_ref_path,
    )
    n_t_lin = min(int(qn_linear_raw.shape[1]), int(lin_snaps.shape[1]))
    qn_linear = qn_linear_raw[:n_tot, :n_t_lin]
    lin_snaps = lin_snaps[:, :n_t_lin]
    lin_qn_recon_rel_err = (
        np.linalg.norm(u_ref[:, None] + basis @ qn_linear - lin_snaps) / (np.linalg.norm(lin_snaps) + 1e-30)
    )
    qn_source = "qN_file"
    if lin_qn_recon_rel_err > 1e-8:
        print(
            "[TMP-Case2-PG-Oracle] warning: linear qN is inconsistent with "
            f"(basis, u_ref, rom_snaps), rel recon error = {lin_qn_recon_rel_err:.3e}. "
            "Using LS-projected coordinates from rom_snaps for oracle construction."
        )
        qn_linear = _project_snaps_to_basis_ls(basis, u_ref, lin_snaps)
        lin_qn_recon_rel_err = (
            np.linalg.norm(u_ref[:, None] + basis @ qn_linear - lin_snaps) / (np.linalg.norm(lin_snaps) + 1e-30)
        )
        qn_source = "ls_from_rom_snaps"

    v = basis[:, :n_p]
    vbar = basis[:, n_p:n_tot]

    qbar_oracle = qn_linear[n_p:n_tot, :]
    if qbar_oracle.shape[0] != vbar.shape[1]:
        raise ValueError(
            f"Oracle qbar size mismatch: qbar has {qbar_oracle.shape[0]}, expected {vbar.shape[1]}"
        )
    oracle_model = OracleCase2FromLinearQNTmp(qbar_table=qbar_oracle, dt=dt)

    w0 = np.asarray(lin_snaps[:, 0], dtype=np.float64).reshape(-1)
    if w0.size != basis.shape[0]:
        raise ValueError(f"W0 size mismatch: got {w0.size}, expected {basis.shape[0]}")
    grid_x, grid_y, nx, ny = _infer_square_grid_tmp(w0.size)

    cross_vt_vbar_fro = _cross_coupling_fro_norm(v, vbar)
    oracle_mode_reason = _validate_oracle_mode(args.oracle_mode)
    print(
        "[TMP-Case2-PG-Oracle] oracle_mode="
        f"{args.oracle_mode} (reason={oracle_mode_reason}, "
        f"||V^T Vbar||_F={cross_vt_vbar_fro:.6e})"
    )

    snap_folder, snap_folder_source = _resolve_snap_folder_tmp(work_dir)
    hdm_snaps, hdm_source = _load_hdm_snaps_tmp(
        mu=mu,
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        snap_folder=snap_folder,
    )

    pg_oracle_snaps, rom_times, online_solve_elapsed = _solve_case2_pg_oracle(
        solver_variant=args.solver_variant,
        oracle_model=oracle_model,
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu=mu,
        v=v,
        vbar=vbar,
        u_ref=u_ref,
        max_its=args.max_its,
        relnorm_cutoff=args.relnorm_cutoff,
        min_delta=args.min_delta,
        linear_solver=args.linear_solver,
        normal_eq_reg=args.normal_eq_reg,
        y_init_table=None,
        wp_table=None,
    )

    # Align references and ROM trajectory to a common time window.
    (aligned, n_t_used) = _align_time_series(hdm_snaps, lin_snaps, pg_oracle_snaps, qn_linear[:n_tot, :])
    hdm_cmp, lin_cmp, pg_cmp, q_lin = aligned
    if (
        hdm_snaps.shape[1] != n_t_used
        or lin_snaps.shape[1] != n_t_used
        or pg_oracle_snaps.shape[1] != n_t_used
        or qn_linear.shape[1] != n_t_used
    ):
        print(
            "[TMP-Case2-PG-Oracle] warning: time-length mismatch detected; "
            f"using common window n_t={n_t_used}."
        )

    # Trajectory-level errors.
    rel_err_hdm = 100.0 * np.linalg.norm(hdm_cmp - pg_cmp) / np.linalg.norm(hdm_cmp)
    rel_err_vs_linear = 100.0 * np.linalg.norm(lin_cmp - pg_cmp) / np.linalg.norm(lin_cmp)

    # Reduced-coordinate consistency checks against linear qN.
    q_pg = _project_snaps_to_basis_ls(basis, u_ref, pg_cmp)
    rel_q_primary = 100.0 * np.linalg.norm(q_pg[:n_p, :] - q_lin[:n_p, :]) / np.linalg.norm(q_lin[:n_p, :])
    rel_q_secondary = (
        100.0
        * np.linalg.norm(q_pg[n_p:, :] - q_lin[n_p:, :])
        / np.linalg.norm(q_lin[n_p:, :])
    )

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = work_dir / "Results" / "Runs" / "Case2"
    out_dir.mkdir(parents=True, exist_ok=True)
    basis_tag = Path(basis_path).stem
    pert_pct = float(args.qbar_perturb_percent)
    run_tag = f"{args.run_tag_prefix}_{mu_tag}_n{n_p}_ntot{n_tot}_{basis_tag}_pert{pert_pct:.2f}pct"
    out_snaps = out_dir / f"{run_tag}_snaps.npy"
    out_qn = out_dir / f"{run_tag}_qN.npy"
    out_pert_snaps = out_dir / f"{run_tag}_snaps_pert.npy"
    out_pert_qn = out_dir / f"{run_tag}_qN_pert.npy"
    out_png = out_dir / f"{run_tag}_hdm_vs_linear_vs_pg_oracle.png"
    out_txt = out_dir / f"{run_tag}_summary.txt"

    np.save(out_snaps, pg_oracle_snaps)
    np.save(out_qn, q_pg)

    pert_enabled = pert_pct > 0.0
    rel_err_hdm_pert = np.nan
    rel_err_vs_linear_pert = np.nan
    rel_q_primary_pert = np.nan
    rel_q_secondary_pert = np.nan
    contamination_gain_primary = np.nan
    contamination_gain_state = np.nan
    delta_qbar_abs = 0.0
    delta_qbar_rel_percent = 0.0
    pert_its = np.nan
    pert_jac = np.nan
    pert_res = np.nan
    pert_ls = np.nan
    pert_elapsed = np.nan
    pg_pert_snaps = None
    pg_pert_cmp = None
    q_pg_pert = None
    if pert_enabled:
        qbar_pert, delta_qbar = _build_perturbed_qbar(
            qbar_oracle=qbar_oracle[:, :n_t_used],
            perturb_percent=pert_pct,
            perturb_seed=args.qbar_perturb_seed,
        )
        oracle_model_pert = OracleCase2FromLinearQNTmp(qbar_table=qbar_pert, dt=dt)
        pg_pert_snaps, pert_times, pert_elapsed = _solve_case2_pg_oracle(
            solver_variant=args.solver_variant,
            oracle_model=oracle_model_pert,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu,
            v=v,
            vbar=vbar,
            u_ref=u_ref,
            max_its=args.max_its,
            relnorm_cutoff=args.relnorm_cutoff,
            min_delta=args.min_delta,
            linear_solver=args.linear_solver,
            normal_eq_reg=args.normal_eq_reg,
            y_init_table=None,
            wp_table=None,
        )
        pert_its, pert_jac, pert_res, pert_ls = pert_times
        pg_pert_cmp = np.asarray(pg_pert_snaps, dtype=np.float64)[:, :n_t_used]
        rel_err_hdm_pert = 100.0 * np.linalg.norm(hdm_cmp - pg_pert_cmp) / np.linalg.norm(hdm_cmp)
        rel_err_vs_linear_pert = 100.0 * np.linalg.norm(lin_cmp - pg_pert_cmp) / np.linalg.norm(lin_cmp)

        q_pg_pert = _project_snaps_to_basis_ls(basis, u_ref, pg_pert_cmp)
        rel_q_primary_pert = 100.0 * np.linalg.norm(q_pg_pert[:n_p, :] - q_lin[:n_p, :]) / np.linalg.norm(q_lin[:n_p, :])
        rel_q_secondary_pert = 100.0 * np.linalg.norm(q_pg_pert[n_p:, :] - q_lin[n_p:, :]) / np.linalg.norm(q_lin[n_p:, :])

        delta_qbar_abs = float(np.linalg.norm(delta_qbar))
        delta_qbar_rel_percent = 100.0 * delta_qbar_abs / (np.linalg.norm(qbar_oracle[:, :n_t_used]) + 1e-30)
        delta_q_primary_abs = float(np.linalg.norm(q_pg_pert[:n_p, :] - q_pg[:n_p, :]))
        delta_state_abs = float(np.linalg.norm(pg_pert_cmp - pg_cmp))
        contamination_gain_primary = delta_q_primary_abs / (delta_qbar_abs + 1e-30)
        contamination_gain_state = delta_state_abs / (delta_qbar_abs + 1e-30)

        np.save(out_pert_snaps, pg_pert_snaps)
        np.save(out_pert_qn, q_pg_pert)

    plot_last = n_t_used - 1
    plot_steps = list(range(0, plot_last + 1, 100))
    if plot_last not in plot_steps:
        plot_steps.append(plot_last)

    fig, ax1, ax2 = plot_snaps(
        grid_x,
        grid_y,
        hdm_cmp,
        plot_steps,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        grid_x,
        grid_y,
        lin_cmp,
        plot_steps,
        label="Linear PROM (n_tot)",
        fig_ax=(fig, ax1, ax2),
        color="0.35",
        linewidth=1.8,
        linestyle="--",
    )
    plot_snaps(
        grid_x,
        grid_y,
        pg_cmp,
        plot_steps,
        label="Case 2 PG Oracle TMP",
        fig_ax=(fig, ax1, ax2),
        color="tab:blue",
        linewidth=2.0,
        linestyle="solid",
    )
    if pert_enabled and pg_pert_cmp is not None:
        plot_snaps(
            grid_x,
            grid_y,
            pg_pert_cmp,
            plot_steps,
            label=f"Case 2 PG Oracle + perturb ({pert_pct:.2f}%)",
            fig_ax=(fig, ax1, ax2),
            color="tab:red",
            linewidth=2.0,
            linestyle="-.",
        )
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    num_its, jac_time, res_time, ls_time = rom_times
    _write_kv_txt(
        out_txt,
        [
            ("solver_variant", f"tmp_case2_oracle_{args.solver_variant}"),
            ("mu_test", mu),
            ("grid_nx", nx),
            ("grid_ny", ny),
            ("n_primary", n_p),
            ("n_tot", n_tot),
            ("n_time_used", int(n_t_used)),
            ("work_dir", str(work_dir)),
            ("snap_folder", str(snap_folder)),
            ("snap_folder_source", snap_folder_source),
            ("linear_run_dir", str(linear_dir)),
            ("hdm_source_path", str(hdm_source) if hdm_source is not None else "computed_via_load_or_compute_snaps"),
            ("basis_path", str(basis_path)),
            ("u_ref_path", str(uref_path)),
            ("linear_qn_source", qn_source),
            ("linear_qn_reconstruction_rel_error", lin_qn_recon_rel_err),
            ("oracle_mode_requested", str(args.oracle_mode)),
            ("oracle_mode_effective", "legacy"),
            ("oracle_mode_reason", oracle_mode_reason),
            ("cross_vt_vbar_fro", cross_vt_vbar_fro),
            ("qbar_perturb_percent_requested", pert_pct),
            ("qbar_perturb_seed", int(args.qbar_perturb_seed)),
            ("qbar_perturb_rel_percent_actual", delta_qbar_rel_percent),
            ("online_solve_elapsed_s", online_solve_elapsed),
            ("num_iterations", num_its),
            ("jac_time_s", jac_time),
            ("res_time_s", res_time),
            ("ls_time_s", ls_time),
            ("relative_error_percent_vs_hdm", rel_err_hdm),
            ("relative_error_percent_vs_linear_prom", rel_err_vs_linear),
            ("relative_q_primary_error_percent_vs_linear", rel_q_primary),
            ("relative_q_secondary_error_percent_vs_linear", rel_q_secondary),
            ("online_solve_elapsed_pert_s", pert_elapsed),
            ("num_iterations_pert", pert_its),
            ("jac_time_pert_s", pert_jac),
            ("res_time_pert_s", pert_res),
            ("ls_time_pert_s", pert_ls),
            ("relative_error_percent_vs_hdm_pert", rel_err_hdm_pert),
            ("relative_error_percent_vs_linear_prom_pert", rel_err_vs_linear_pert),
            ("relative_q_primary_error_percent_vs_linear_pert", rel_q_primary_pert),
            ("relative_q_secondary_error_percent_vs_linear_pert", rel_q_secondary_pert),
            ("contamination_gain_primary_dq_over_dqbar", contamination_gain_primary),
            ("contamination_gain_state_du_over_dqbar", contamination_gain_state),
            ("snaps_output", str(out_snaps)),
            ("qN_output", str(out_qn)),
            ("snaps_pert_output", str(out_pert_snaps) if pert_enabled else "N/A"),
            ("qN_pert_output", str(out_pert_qn) if pert_enabled else "N/A"),
            ("plot_output", str(out_png)),
        ],
    )

    print(f"[TMP-Case2-PG-Oracle] relative error vs HDM: {rel_err_hdm:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] relative error vs linear PROM: {rel_err_vs_linear:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] rel q_primary vs linear: {rel_q_primary:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] rel q_secondary vs linear: {rel_q_secondary:.3f}%")
    if pert_enabled:
        print(f"[TMP-Case2-PG-Oracle] perturb actual rel ||dqbar||/||qbar||: {delta_qbar_rel_percent:.3f}%")
        print(f"[TMP-Case2-PG-Oracle] contamination gain primary: {contamination_gain_primary:.6e}")
        print(f"[TMP-Case2-PG-Oracle] contamination gain state:   {contamination_gain_state:.6e}")
        print(f"[TMP-Case2-PG-Oracle] rel q_primary vs linear (pert): {rel_q_primary_pert:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] saved summary: {out_txt}")
    print(f"[TMP-Case2-PG-Oracle] saved plot:    {out_png}")


if __name__ == "__main__":
    main()
