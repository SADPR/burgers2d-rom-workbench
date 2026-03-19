#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 1: OFFLINE QUADRATIC MANIFOLD BUILD

Builds a single global quadratic manifold

    u(q) = u_ref + V q + H Q(q),

where Q(q) contains symmetric monomials q_i q_j (i <= j).

Dimension selection:
  1) Compute traditional linear POD size n_trad from tolerance pod_tol.
  2) Apply quadratic heuristic on n_trad to get n.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
quadratic_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(quadratic_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_singular_value_decay
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS
from burgers.core import get_snapshot_params


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def build_Q_symmetric_matrix(q_mat):
    """
    Build symmetric quadratic coordinates in matrix form:
      q_mat: (n, Ns) -> Q: (n(n+1)/2, Ns)
    """
    n, _ = q_mat.shape
    i_triu, j_triu = np.triu_indices(n)
    return q_mat[i_triu, :] * q_mat[j_triu, :]


def pod_rank_from_tolerance(singular_values, pod_tol):
    svals = np.asarray(singular_values, dtype=np.float64)
    if svals.size == 0:
        return 0, np.nan, np.nan

    s2 = svals ** 2
    total = float(np.sum(s2))
    if total <= 0.0:
        return 1, 1.0, 0.0

    residual = 1.0 - np.cumsum(s2) / total
    idx = np.where(residual <= pod_tol)[0]
    if idx.size == 0:
        n_keep = svals.size
    else:
        n_keep = int(idx[0] + 1)

    captured = 1.0 - float(residual[n_keep - 1])
    lost = float(residual[n_keep - 1])
    return n_keep, captured, lost


def compute_H_ridge(E, Q, ridge_alpha):
    """
    Solve
        min_H ||E - H Q||_F^2 + alpha ||H||_F^2
    in closed form:
        H = (E Q^T) (Q Q^T + alpha I)^{-1}
    """
    m = Q.shape[0]
    A = Q @ Q.T
    A = A + ridge_alpha * np.eye(m, dtype=np.float64)
    B = E @ Q.T

    try:
        H = np.linalg.solve(A.T, B.T).T
    except np.linalg.LinAlgError:
        H = np.linalg.lstsq(A.T, B.T, rcond=None)[0].T
    return H


def main(
    pod_tol=1e-6,
    zeta_qua=0.10,
    ridge_alpha=1e-4,
    dt=DT,
    num_steps=NUM_STEPS,
    save_sv_plot=True,
):
    results_dir = os.path.join(parent_dir, "Results")
    snap_folder = os.path.join(results_dir, "param_snaps")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    t_start = time.time()
    param_list = get_snapshot_params()
    if len(param_list) == 0:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    w0 = np.asarray(W0, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Build snapshot matrix
    # ------------------------------------------------------------------
    t0 = time.time()
    S0 = load_or_compute_snaps(
        param_list[0],
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder=snap_folder,
    )
    N, T = S0.shape
    Ns_total = len(param_list) * T
    S = np.zeros((N, Ns_total), dtype=np.float64)

    col = 0
    for mu in param_list:
        S_mu = load_or_compute_snaps(
            mu,
            GRID_X,
            GRID_Y,
            w0,
            dt,
            num_steps,
            snap_folder=snap_folder,
        )
        S[:, col:col + T] = S_mu
        col += T
    elapsed_snapshots = time.time() - t0

    # ------------------------------------------------------------------
    # POD and heuristic for n
    # ------------------------------------------------------------------
    t0 = time.time()
    u_ref = np.mean(S, axis=1)
    S_centered = S - u_ref[:, None]

    U_full, s_all, _ = np.linalg.svd(S_centered, full_matrices=False)
    n_trad, energy_captured, energy_lost = pod_rank_from_tolerance(s_all, pod_tol)

    n_qua_raw = (np.sqrt(9.0 + 8.0 * n_trad) - 3.0) / 2.0
    n_qua_corr = int(np.floor((1.0 + zeta_qua) * n_qua_raw))
    n_max_ls = int(np.floor((np.sqrt(1.0 + 8.0 * Ns_total) - 1.0) / 2.0))
    n = max(1, min(n_qua_corr, n_max_ls, int(s_all.size)))

    V = U_full[:, :n]
    sigma = s_all[:n]
    elapsed_pod = time.time() - t0

    # ------------------------------------------------------------------
    # Build Q and compute H
    # ------------------------------------------------------------------
    t0 = time.time()
    q_mat = V.T @ S_centered
    S_lin = V @ q_mat + u_ref[:, None]
    E = S - S_lin
    Q = build_Q_symmetric_matrix(q_mat)
    elapsed_q_build = time.time() - t0

    t0 = time.time()
    H = compute_H_ridge(E, Q, ridge_alpha=ridge_alpha)
    elapsed_h = time.time() - t0

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    V_path = os.path.join(quadratic_dir, "qm_V.npy")
    H_path = os.path.join(quadratic_dir, "qm_H.npy")
    uref_path = os.path.join(quadratic_dir, "qm_uref.npy")
    sigma_path = os.path.join(quadratic_dir, "qm_sigma.npy")
    metadata_path = os.path.join(quadratic_dir, "qm_metadata.npz")
    sv_plot_path = os.path.join(quadratic_dir, "qm_singular_value_decay.png")

    np.save(V_path, V)
    np.save(H_path, H)
    np.save(uref_path, u_ref)
    np.save(sigma_path, sigma)
    np.savez(
        metadata_path,
        pod_tol=np.float64(pod_tol),
        zeta_qua=np.float64(zeta_qua),
        ridge_alpha=np.float64(ridge_alpha),
        n_trad=np.int64(n_trad),
        n_qua_raw=np.float64(n_qua_raw),
        n_qua_corr=np.int64(n_qua_corr),
        n_max_ls=np.int64(n_max_ls),
        n_final=np.int64(n),
        N=np.int64(N),
        T=np.int64(T),
        Ns_total=np.int64(Ns_total),
        num_training_params=np.int64(len(param_list)),
        energy_captured=np.float64(energy_captured),
        energy_lost=np.float64(energy_lost),
    )

    if save_sv_plot and s_all.size > 0:
        plot_singular_value_decay(
            s_all,
            out_path=sv_plot_path,
            max_modes=min(1000, s_all.size),
            label="Training snapshots",
            title="Quadratic stage1 POD residual energy",
            use_latex=True,
        )
    else:
        sv_plot_path = None

    elapsed_total = time.time() - t_start

    report_path = os.path.join(quadratic_dir, "stage1_quadratic_offline_summary.txt")
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage1_quadratic_offline.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("pod_tol", pod_tol),
                    ("zeta_qua", zeta_qua),
                    ("ridge_alpha", ridge_alpha),
                    ("snap_folder", snap_folder),
                ],
            ),
            (
                "training_data",
                [
                    ("num_training_params", len(param_list)),
                    ("snapshot_shape", S.shape),
                    ("state_size", N),
                    ("snapshots_per_param", T),
                    ("total_snapshots", Ns_total),
                ],
            ),
            (
                "heuristic_dimensioning",
                [
                    ("n_trad_from_tol", n_trad),
                    ("n_qua_raw", n_qua_raw),
                    ("n_qua_corr", n_qua_corr),
                    ("n_max_ls", n_max_ls),
                    ("n_final", n),
                    ("pod_energy_captured_at_n_trad", energy_captured),
                    ("pod_energy_lost_at_n_trad", energy_lost),
                ],
            ),
            (
                "manifold_shapes",
                [
                    ("V_shape", V.shape),
                    ("H_shape", H.shape),
                    ("u_ref_shape", u_ref.shape),
                    ("Q_shape", Q.shape),
                    ("u_ref_l2_norm", np.linalg.norm(u_ref)),
                    ("linear_residual_l2_norm", np.linalg.norm(E)),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("build_snapshots", elapsed_snapshots),
                    ("pod_and_heuristic", elapsed_pod),
                    ("build_q_and_linear_residual", elapsed_q_build),
                    ("compute_H_ridge", elapsed_h),
                    ("total", elapsed_total),
                ],
            ),
            (
                "outputs",
                [
                    ("qm_V_npy", V_path),
                    ("qm_H_npy", H_path),
                    ("qm_uref_npy", uref_path),
                    ("qm_sigma_npy", sigma_path),
                    ("qm_metadata_npz", metadata_path),
                    ("singular_value_plot_png", sv_plot_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )

    print(f"[STAGE1] Saved manifold files to: {quadratic_dir}")
    print(f"[STAGE1] n_trad={n_trad}, heuristic n={n}")
    print(f"[STAGE1] Summary saved to: {report_path}")
    return n


if __name__ == "__main__":
    main()
