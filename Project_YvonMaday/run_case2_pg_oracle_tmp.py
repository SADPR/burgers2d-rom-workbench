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
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin,
)


def _looks_like_work_dir(path):
    p = Path(path)
    return (p / "Results" / "Stage1" / "basis.npy").exists() and (p / "Results" / "Runs" / "Linear").exists()


def _resolve_work_dir(work_dir_arg):
    if work_dir_arg:
        wd = Path(work_dir_arg).expanduser().resolve()
        if not _looks_like_work_dir(wd):
            raise FileNotFoundError(
                f"--work-dir does not look valid: {wd}\n"
                "Expected at least: Results/Stage1/basis.npy and Results/Runs/Linear"
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
        qb = np.asarray(qbar_table, dtype=np.float32)
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


def _load_basis_and_reference(work_dir, n_tot):
    basis_path = work_dir / "Results" / "Stage1" / "basis.npy"
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

    qn_linear = np.asarray(np.load(qn_path, allow_pickle=False), dtype=np.float64)
    lin_snaps = np.asarray(np.load(lin_snaps_path, allow_pickle=False), dtype=np.float64)
    if qn_linear.ndim != 2:
        raise ValueError(f"linear qN must be 2D, got shape {qn_linear.shape}")
    if qn_linear.shape[0] < n_tot:
        raise ValueError(f"linear qN has {qn_linear.shape[0]} modes, requested n_tot={n_tot}")
    if n_p < 1 or n_p >= n_tot:
        raise ValueError(f"Invalid split n_primary={n_p}, n_tot={n_tot}")

    basis, u_ref, basis_path, uref_path = _load_basis_and_reference(work_dir, n_tot)
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

    t0 = time.time()
    pg_oracle_snaps, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin(
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
        max_its=int(args.max_its),
        relnorm_cutoff=float(args.relnorm_cutoff),
        min_delta=float(args.min_delta),
        linear_solver=str(args.linear_solver),
        normal_eq_reg=float(args.normal_eq_reg),
    )
    online_solve_elapsed = time.time() - t0

    # Trajectory-level errors.
    rel_err_hdm = 100.0 * np.linalg.norm(hdm_snaps - pg_oracle_snaps) / np.linalg.norm(hdm_snaps)
    rel_err_vs_linear = 100.0 * np.linalg.norm(lin_snaps - pg_oracle_snaps) / np.linalg.norm(lin_snaps)

    # Reduced-coordinate consistency checks against linear qN.
    q_pg = basis.T @ (pg_oracle_snaps - u_ref[:, None])
    q_lin = qn_linear[:n_tot, :]
    rel_q_primary = 100.0 * np.linalg.norm(q_pg[:n_p, :] - q_lin[:n_p, :]) / np.linalg.norm(q_lin[:n_p, :])
    rel_q_secondary = (
        100.0
        * np.linalg.norm(q_pg[n_p:, :] - q_lin[n_p:, :])
        / np.linalg.norm(q_lin[n_p:, :])
    )

    out_dir = work_dir / "Results" / "Runs" / "Case2"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"tmp_case2_pg_oracle_prom_{mu_tag}_n{n_p}_ntot{n_tot}"
    out_snaps = out_dir / f"{run_tag}_snaps.npy"
    out_qn = out_dir / f"{run_tag}_qN.npy"
    out_png = out_dir / f"{run_tag}_hdm_vs_linear_vs_pg_oracle.png"
    out_txt = out_dir / f"{run_tag}_summary.txt"

    np.save(out_snaps, pg_oracle_snaps)
    np.save(out_qn, q_pg)

    plot_steps = list(range(0, num_steps + 1, 100))
    if num_steps not in plot_steps:
        plot_steps.append(num_steps)

    fig, ax1, ax2 = plot_snaps(
        grid_x,
        grid_y,
        hdm_snaps,
        plot_steps,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        grid_x,
        grid_y,
        lin_snaps,
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
        pg_oracle_snaps,
        plot_steps,
        label="Case 2 PG Oracle TMP",
        fig_ax=(fig, ax1, ax2),
        color="tab:blue",
        linewidth=2.0,
        linestyle="solid",
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
            ("solver_variant", "tmp_case2_pg_oracle_check"),
            ("mu_test", mu),
            ("grid_nx", nx),
            ("grid_ny", ny),
            ("n_primary", n_p),
            ("n_tot", n_tot),
            ("work_dir", str(work_dir)),
            ("snap_folder", str(snap_folder)),
            ("snap_folder_source", snap_folder_source),
            ("linear_run_dir", str(linear_dir)),
            ("hdm_source_path", str(hdm_source) if hdm_source is not None else "computed_via_load_or_compute_snaps"),
            ("basis_path", str(basis_path)),
            ("u_ref_path", str(uref_path)),
            ("online_solve_elapsed_s", online_solve_elapsed),
            ("num_iterations", num_its),
            ("jac_time_s", jac_time),
            ("res_time_s", res_time),
            ("ls_time_s", ls_time),
            ("relative_error_percent_vs_hdm", rel_err_hdm),
            ("relative_error_percent_vs_linear_prom", rel_err_vs_linear),
            ("relative_q_primary_error_percent_vs_linear", rel_q_primary),
            ("relative_q_secondary_error_percent_vs_linear", rel_q_secondary),
            ("snaps_output", str(out_snaps)),
            ("qN_output", str(out_qn)),
            ("plot_output", str(out_png)),
        ],
    )

    print(f"[TMP-Case2-PG-Oracle] relative error vs HDM: {rel_err_hdm:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] relative error vs linear PROM: {rel_err_vs_linear:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] rel q_primary vs linear: {rel_q_primary:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] rel q_secondary vs linear: {rel_q_secondary:.3f}%")
    print(f"[TMP-Case2-PG-Oracle] saved summary: {out_txt}")
    print(f"[TMP-Case2-PG-Oracle] saved plot:    {out_png}")


if __name__ == "__main__":
    main()
