#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage-1 Proposal 1 (enriched metric) POD builder for Maday experiments.

Implements (in method-of-snapshots form):

    W_eff = eps * M_block + sum_s omega_s J_s^T P_s J_s

and builds POD modes from the weighted correlation matrix:

    C = S^T W_eff S

This keeps all outputs isolated in Results_Maday/<tag>/Stage1.
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

from burgers.core import get_snapshot_params, load_or_compute_snaps, get_ops, inviscid_burgers_exact_jac2D, plot_singular_value_decay, podsize
from burgers.ecsw_utils import build_ecsw_snapshot_plan
from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


def _parse_range(arg: str | None, default: tuple[float, float]) -> tuple[float, float]:
    if arg is None:
        return float(default[0]), float(default[1])
    parts = [x.strip() for x in str(arg).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{arg}'. Expected 'low,high'.")
    lo = float(parts[0])
    hi = float(parts[1])
    if lo > hi:
        raise ValueError(f"Invalid range '{arg}': low > high.")
    return lo, hi


def _choose_snap_folder() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
        os.path.join(PROJECT_ROOT, "param_snaps"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _cell_area_weights() -> np.ndarray:
    dx = np.asarray(GRID_X[1:] - GRID_X[:-1], dtype=np.float64)
    dy = np.asarray(GRID_Y[1:] - GRID_Y[:-1], dtype=np.float64)
    area = np.outer(dy, dx).reshape(-1)
    return np.concatenate([area, area], axis=0)


def _load_diag_vector(path: str, expected_size: int, *, label: str) -> np.ndarray:
    vec = np.asarray(np.load(os.path.abspath(os.path.expanduser(path)), allow_pickle=False), dtype=np.float64).reshape(-1)
    if vec.size != expected_size:
        raise ValueError(f"{label} size mismatch: got {vec.size}, expected {expected_size}")
    if np.any(~np.isfinite(vec)):
        raise ValueError(f"{label} contains non-finite entries.")
    return vec


def _select_pairs(
    *,
    num_mu: int,
    num_times: int,
    time_stride: int,
    max_items: int | None,
    seed: int,
) -> tuple[dict[int, list[int]], int, int]:
    stride = max(1, int(time_stride))
    candidates = [(imu, it) for imu in range(num_mu) for it in range(0, num_times, stride)]
    n_cand = len(candidates)
    if n_cand < 1:
        raise RuntimeError("No candidate (mu,time) pairs produced.")

    if max_items is None or int(max_items) >= n_cand:
        pick = np.arange(n_cand, dtype=np.int64)
    else:
        k = max(1, int(max_items))
        rng = np.random.default_rng(int(seed))
        pick = np.sort(rng.choice(n_cand, size=k, replace=False))

    select_map: dict[int, list[int]] = {}
    for idx in pick.tolist():
        imu, it = candidates[int(idx)]
        if imu not in select_map:
            select_map[imu] = []
        select_map[imu].append(int(it))
    for imu in select_map:
        select_map[imu] = sorted(set(select_map[imu]))
    return select_map, int(pick.size), int(n_cand)


def _select_pairs_ecsw(
    *,
    num_steps: int,
    mu_samples: list[list[float]],
    mode: str,
    total_percent: float | None,
    total_count: int | None,
    time_offset: int,
    random_seed: int,
) -> tuple[dict[int, list[int]], int, int]:
    mode_map = {
        "ecsw_global_stratified": "global_stratified_random",
        "ecsw_param_time_stratified": "global_param_time_stratified",
    }
    m = str(mode).strip().lower()
    if m not in mode_map:
        raise ValueError(f"Unsupported ECSW-like selection mode: {mode}")
    plan_mode = mode_map[m]
    if total_percent is None and total_count is None:
        raise ValueError(
            f"{mode} requires --*-percent or --*-total to define sampling budget."
        )

    # ECSW plan uses candidate times arange(offset, num_steps),
    # so pass (num_steps+1) to allow selecting the last snapshot index.
    plan = build_ecsw_snapshot_plan(
        num_steps=int(num_steps) + 1,
        snap_time_offset=max(1, int(time_offset)),
        num_mu=int(len(mu_samples)),
        mode=plan_mode,
        total_snapshots=(None if total_count is None else int(total_count)),
        total_snapshots_percent=(None if total_percent is None else float(total_percent)),
        mu_points=np.asarray(mu_samples, dtype=np.float64),
        random_seed=int(random_seed),
        ensure_mu_coverage=True,
    )
    selected_by_mu = plan.get("selected_now_cols_by_mu", [])
    if len(selected_by_mu) != len(mu_samples):
        raise RuntimeError(
            f"ECSW-like plan produced {len(selected_by_mu)} mu buckets, expected {len(mu_samples)}."
        )

    select_map: dict[int, list[int]] = {}
    for imu, arr in enumerate(selected_by_mu):
        cols = np.asarray(arr, dtype=np.int64).reshape(-1)
        cols = cols[(cols >= 0) & (cols <= int(num_steps))]
        if cols.size > 0:
            select_map[imu] = sorted(set(int(c) for c in cols.tolist()))
    n_used = int(sum(len(v) for v in select_map.values()))
    n_cand = int(plan.get("num_candidates_total", 0))
    if n_used < 1:
        raise RuntimeError("ECSW-like selection produced zero samples.")
    return select_map, n_used, n_cand


def _assemble_snapshot_matrix(
    *,
    mu_samples: list[list[float]],
    select_map: dict[int, list[int]],
    snap_folder: str,
    dt: float,
    num_steps: int,
) -> np.ndarray:
    blocks = []
    for imu, mu in enumerate(mu_samples):
        if imu not in select_map:
            continue
        s_mu = np.asarray(
            load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        cols = np.asarray(select_map[imu], dtype=np.int64)
        if np.any(cols < 0) or np.any(cols >= s_mu.shape[1]):
            raise ValueError(f"Invalid selected time indices for mu index {imu}: {cols}")
        blocks.append(s_mu[:, cols])
    if len(blocks) == 0:
        raise RuntimeError("No snapshots selected for Proposal 1 Weff POD.")
    return np.concatenate(blocks, axis=1)


def _time_weights(*, num_steps: int, dt: float, mode: str) -> np.ndarray:
    n_time = int(num_steps) + 1
    m = str(mode).strip().lower()
    if m == "uniform":
        return np.ones((n_time,), dtype=np.float64)
    if m == "trapezoid":
        w = np.full((n_time,), float(dt), dtype=np.float64)
        if n_time >= 1:
            w[0] *= 0.5
        if n_time >= 2:
            w[-1] *= 0.5
        return w
    raise ValueError(f"Unsupported time-weighting mode: {mode}")


def _mu_weights(*, num_mu: int, mode: str) -> np.ndarray:
    if int(num_mu) < 1:
        raise ValueError("num_mu must be >= 1 for mu weights.")
    m = str(mode).strip().lower()
    if m == "uniform":
        return np.full((num_mu,), 1.0 / float(num_mu), dtype=np.float64)
    raise ValueError(f"Unsupported mu-weighting mode: {mode}")


def _build_weff_correlation(
    *,
    S: np.ndarray,
    mu_weights: np.ndarray,
    time_weights: np.ndarray,
    mu_samples: list[list[float]],
    metric_select_map: dict[int, list[int]],
    snap_folder: str,
    dt: float,
    num_steps: int,
    p_diag: np.ndarray | None,
    progress_every: int = 0,
) -> tuple[np.ndarray, int, float]:
    C_j = np.zeros((S.shape[1], S.shape[1]), dtype=np.float64)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(GRID_X, GRID_Y)
    used = 0
    wsum = 0.0
    total_selected = int(sum(len(v) for v in metric_select_map.values()))
    t_start = time.time()
    for imu, mu in enumerate(mu_samples):
        times = metric_select_map.get(imu, None)
        if not times:
            continue
        s_mu = np.asarray(
            load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        for it in times:
            w = s_mu[:, int(it)]
            J = inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)
            JS = np.asarray(J @ S, dtype=np.float64)
            alpha = float(mu_weights[int(imu)] * time_weights[int(it)])
            if p_diag is None:
                C_j += alpha * (JS.T @ JS)
            else:
                C_j += alpha * (JS.T @ (p_diag[:, None] * JS))
            used += 1
            wsum += alpha
            if progress_every > 0 and (used % progress_every == 0 or used == total_selected):
                elapsed = max(time.time() - t_start, 1e-12)
                rate = used / elapsed
                rem = max(total_selected - used, 0)
                eta = rem / max(rate, 1e-12)
                print(
                    "[MADAY-P1-WEFF] Cj progress "
                    f"{used}/{total_selected} | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )

    if used > 0 and wsum > 0.0:
        C_j *= 1.0 / float(wsum)
    C_j = 0.5 * (C_j + C_j.T)
    return C_j, used, wsum


def _build_weff_gram_on_basis(
    *,
    basis: np.ndarray,
    mu_weights: np.ndarray,
    time_weights: np.ndarray,
    mu_samples: list[list[float]],
    metric_select_map: dict[int, list[int]],
    snap_folder: str,
    dt: float,
    num_steps: int,
    p_diag: np.ndarray | None,
    progress_every: int = 0,
) -> tuple[np.ndarray, int, float]:
    G_j = np.zeros((basis.shape[1], basis.shape[1]), dtype=np.float64)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(GRID_X, GRID_Y)
    used = 0
    wsum = 0.0
    total_selected = int(sum(len(v) for v in metric_select_map.values()))
    t_start = time.time()
    for imu, mu in enumerate(mu_samples):
        times = metric_select_map.get(imu, None)
        if not times:
            continue
        s_mu = np.asarray(
            load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        for it in times:
            w = s_mu[:, int(it)]
            J = inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)
            JB = np.asarray(J @ basis, dtype=np.float64)
            alpha = float(mu_weights[int(imu)] * time_weights[int(it)])
            if p_diag is None:
                G_j += alpha * (JB.T @ JB)
            else:
                G_j += alpha * (JB.T @ (p_diag[:, None] * JB))
            used += 1
            wsum += alpha
            if progress_every > 0 and (used % progress_every == 0 or used == total_selected):
                elapsed = max(time.time() - t_start, 1e-12)
                rate = used / elapsed
                rem = max(total_selected - used, 0)
                eta = rem / max(rate, 1e-12)
                print(
                    "[MADAY-P1-WEFF] Gram progress "
                    f"{used}/{total_selected} | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )

    if used > 0 and wsum > 0.0:
        G_j *= 1.0 / float(wsum)
    G_j = 0.5 * (G_j + G_j.T)
    return G_j, used, wsum


def _choose_n_keep(evals: np.ndarray, num_modes: int | None, pod_tol: float | None) -> int:
    if evals.size < 1:
        raise RuntimeError("No positive eigenvalues in Weff snapshot correlation.")
    if num_modes is not None:
        n_keep = int(num_modes)
    elif pod_tol is not None:
        sigma = np.sqrt(np.maximum(evals, 0.0))
        n_keep = int(podsize(sigma, energy_thresh=1.0 - float(pod_tol)))
    else:
        n_keep = int(evals.size)
    if n_keep < 1 or n_keep > evals.size:
        raise ValueError(f"Invalid n_keep={n_keep}; available={evals.size}")
    return int(n_keep)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build Proposal-1 Weff POD basis for Maday experiments.")
    parser.add_argument("--maday-tag", type=str, default="exp_maday_p1_weff")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--snap-folder", type=str, default=None)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--mu1-range", type=str, default=None, help="Override mu1 range as 'low,high'.")
    parser.add_argument("--mu2-range", type=str, default=None, help="Override mu2 range as 'low,high'.")
    parser.add_argument("--samples-per-mu", type=int, default=SAMPLES_PER_MU)
    parser.add_argument("--max-mu-samples", type=int, default=None)
    parser.add_argument("--snapshot-time-stride", type=int, default=10)
    parser.add_argument("--max-snapshot-columns", type=int, default=None)
    parser.add_argument("--snapshot-sample-seed", type=int, default=11)
    parser.add_argument(
        "--snapshot-select-mode",
        choices=("strided", "ecsw_global_stratified", "ecsw_param_time_stratified"),
        default="strided",
    )
    parser.add_argument("--snapshot-percent", type=float, default=None)
    parser.add_argument("--snapshot-total", type=int, default=None)
    parser.add_argument("--snapshot-time-offset", type=int, default=1)
    parser.add_argument("--metric-time-stride", type=int, default=25)
    parser.add_argument("--max-metric-samples", type=int, default=120)
    parser.add_argument("--metric-sample-seed", type=int, default=42)
    parser.add_argument(
        "--metric-select-mode",
        choices=("strided", "ecsw_global_stratified", "ecsw_param_time_stratified"),
        default="strided",
    )
    parser.add_argument("--metric-percent", type=float, default=None)
    parser.add_argument("--metric-total", type=int, default=None)
    parser.add_argument("--metric-time-offset", type=int, default=1)
    parser.add_argument("--metric-time-weighting", choices=("uniform", "trapezoid"), default="uniform")
    parser.add_argument("--metric-mu-weighting", choices=("uniform",), default="uniform")
    parser.add_argument(
        "--purist-full",
        action="store_true",
        default=False,
        help="Use all snapshots and all metric samples (stride=1, no max caps).",
    )
    parser.add_argument("--eps-mode", choices=("fixed", "trace_ratio"), default="fixed")
    parser.add_argument("--eps-mblock", type=float, default=1e-8)
    parser.add_argument(
        "--eps-ratio",
        type=float,
        default=1e-10,
        help="Eta in eps = eta * trace(C_j)/trace(C_m) when --eps-mode=trace_ratio.",
    )
    parser.add_argument("--lspg-p-source", choices=("identity", "diag_file"), default="identity")
    parser.add_argument("--lspg-p-diag-file", type=str, default=None)
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--pod-tol", type=float, default=1e-4)
    parser.add_argument("--center", action="store_true", default=True)
    parser.add_argument("--no-center", action="store_false", dest="center")
    parser.add_argument("--skip-gram-check", action="store_true", default=False)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N metric samples during Cj/Gram assembly; <=0 disables progress prints.",
    )
    parser.add_argument("--test-mu", type=str, default="4.56,0.019")
    parser.add_argument("--basis-name", type=str, default="basis_proposal1_weff.npy")
    parser.add_argument("--sigma-name", type=str, default="sigma_proposal1_weff.npy")
    parser.add_argument("--uref-name", type=str, default="u_ref_proposal1_weff.npy")
    parser.add_argument("--meta-name", type=str, default="stage1_proposal1_weff_metadata.npz")
    parser.add_argument("--summary-name", type=str, default="stage1_proposal1_weff_summary.txt")
    parser.add_argument("--decay-plot-name", type=str, default="stage1_proposal1_weff_decay.png")
    args = parser.parse_args(argv)

    if str(args.eps_mode).strip().lower() == "fixed":
        if float(args.eps_mblock) <= 0.0:
            raise ValueError("--eps-mblock must be strictly positive when --eps-mode=fixed.")
    else:
        if float(args.eps_ratio) <= 0.0:
            raise ValueError("--eps-ratio must be strictly positive when --eps-mode=trace_ratio.")
    if int(args.num_steps) < 1:
        raise ValueError("--num-steps must be >= 1")

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    stage1_dir = paths.stage1
    os.makedirs(stage1_dir, exist_ok=True)

    snap_folder = str(args.snap_folder).strip() if args.snap_folder else _choose_snap_folder()
    os.makedirs(snap_folder, exist_ok=True)

    mu1_range = _parse_range(args.mu1_range, MU1_RANGE)
    mu2_range = _parse_range(args.mu2_range, MU2_RANGE)
    mu_samples = get_snapshot_params(
        mu1_range=mu1_range,
        mu2_range=mu2_range,
        samples_per_mu=int(args.samples_per_mu),
    )
    if args.max_mu_samples is not None and int(args.max_mu_samples) > 0:
        mu_samples = mu_samples[: int(args.max_mu_samples)]
    if len(mu_samples) < 1:
        raise RuntimeError("No mu samples available.")

    snapshot_select_mode = str(args.snapshot_select_mode).strip().lower()
    metric_select_mode = str(args.metric_select_mode).strip().lower()
    snap_stride = int(args.snapshot_time_stride)
    metric_stride = int(args.metric_time_stride)
    max_snapshot_columns = int(args.max_snapshot_columns) if args.max_snapshot_columns is not None else None
    max_metric_samples = int(args.max_metric_samples) if args.max_metric_samples is not None else None
    if bool(args.purist_full):
        snap_stride = 1
        metric_stride = 1
        max_snapshot_columns = None
        max_metric_samples = None
        snapshot_select_mode = "strided"
        metric_select_mode = "strided"

    n_time = int(args.num_steps) + 1
    if snapshot_select_mode == "strided":
        snap_select_map, n_snap_used, n_snap_cand = _select_pairs(
            num_mu=len(mu_samples),
            num_times=n_time,
            time_stride=int(snap_stride),
            max_items=max_snapshot_columns,
            seed=int(args.snapshot_sample_seed),
        )
    else:
        snap_total = int(args.snapshot_total) if args.snapshot_total is not None else max_snapshot_columns
        snap_pct = float(args.snapshot_percent) if args.snapshot_percent is not None else None
        snap_select_map, n_snap_used, n_snap_cand = _select_pairs_ecsw(
            num_steps=int(args.num_steps),
            mu_samples=mu_samples,
            mode=snapshot_select_mode,
            total_percent=snap_pct,
            total_count=snap_total,
            time_offset=int(args.snapshot_time_offset),
            random_seed=int(args.snapshot_sample_seed),
        )

    if metric_select_mode == "strided":
        metric_select_map, n_metric_used, n_metric_cand = _select_pairs(
            num_mu=len(mu_samples),
            num_times=n_time,
            time_stride=int(metric_stride),
            max_items=max_metric_samples,
            seed=int(args.metric_sample_seed),
        )
    else:
        metric_total = int(args.metric_total) if args.metric_total is not None else max_metric_samples
        metric_pct = float(args.metric_percent) if args.metric_percent is not None else None
        metric_select_map, n_metric_used, n_metric_cand = _select_pairs_ecsw(
            num_steps=int(args.num_steps),
            mu_samples=mu_samples,
            mode=metric_select_mode,
            total_percent=metric_pct,
            total_count=metric_total,
            time_offset=int(args.metric_time_offset),
            random_seed=int(args.metric_sample_seed),
        )

    print(f"[MADAY-P1-WEFF] tag={paths.tag}")
    print(f"[MADAY-P1-WEFF] output_dir={stage1_dir}")
    print(f"[MADAY-P1-WEFF] snap_folder={snap_folder}")
    print(f"[MADAY-P1-WEFF] eps_mode={str(args.eps_mode).strip().lower()}")
    if str(args.eps_mode).strip().lower() == "fixed":
        print(f"[MADAY-P1-WEFF] eps_mblock={float(args.eps_mblock):.3e}")
    else:
        print(f"[MADAY-P1-WEFF] eps_ratio={float(args.eps_ratio):.3e}")
    print(f"[MADAY-P1-WEFF] purist_full={bool(args.purist_full)}")
    print(f"[MADAY-P1-WEFF] snapshot_select_mode={snapshot_select_mode}")
    print(f"[MADAY-P1-WEFF] metric_select_mode={metric_select_mode}")
    print(f"[MADAY-P1-WEFF] snapshot columns used: {n_snap_used} / {n_snap_cand}")
    print(f"[MADAY-P1-WEFF] metric samples used:   {n_metric_used} / {n_metric_cand}")

    t0 = time.time()
    S = _assemble_snapshot_matrix(
        mu_samples=mu_samples,
        select_map=snap_select_map,
        snap_folder=snap_folder,
        dt=float(args.dt),
        num_steps=int(args.num_steps),
    )
    t_snap = time.time() - t0

    n_dofs, n_cols = S.shape
    if bool(args.center):
        u_ref = np.mean(S, axis=1)
        S = S - u_ref[:, None]
    else:
        u_ref = np.zeros((n_dofs,), dtype=np.float64)

    m_diag = _cell_area_weights()
    if m_diag.size != n_dofs:
        raise ValueError(f"M_block diag size mismatch: got {m_diag.size}, expected {n_dofs}")

    p_mode = str(args.lspg_p_source).strip().lower()
    if p_mode == "identity":
        p_diag = None
    else:
        if not args.lspg_p_diag_file:
            raise ValueError("--lspg-p-diag-file is required when --lspg-p-source=diag_file")
        p_diag = _load_diag_vector(args.lspg_p_diag_file, n_dofs, label="P diagonal")
        if np.any(p_diag <= 0.0):
            raise ValueError("P diagonal must be strictly positive.")

    metric_mu_weights = _mu_weights(num_mu=len(mu_samples), mode=str(args.metric_mu_weighting))
    metric_time_weights = _time_weights(num_steps=int(args.num_steps), dt=float(args.dt), mode=str(args.metric_time_weighting))

    C_m = S.T @ (m_diag[:, None] * S)
    t0 = time.time()
    C_j, metric_used, metric_wsum = _build_weff_correlation(
        S=S,
        mu_weights=metric_mu_weights,
        time_weights=metric_time_weights,
        mu_samples=mu_samples,
        metric_select_map=metric_select_map,
        snap_folder=snap_folder,
        dt=float(args.dt),
        num_steps=int(args.num_steps),
        p_diag=p_diag,
        progress_every=max(0, int(args.progress_every)),
    )
    t_metric = time.time() - t0

    eps_mode = str(args.eps_mode).strip().lower()
    if eps_mode == "fixed":
        eps_eff = float(args.eps_mblock)
    else:
        tr_m = float(np.trace(C_m))
        tr_j = float(np.trace(C_j))
        eps_eff = float(args.eps_ratio) * (tr_j / max(tr_m, 1e-30))
    C = eps_eff * C_m + C_j
    C = 0.5 * (C + C.T)

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    pos = evals > max(1e-14, 1e-14 * float(evals[0] if evals.size > 0 else 1.0))
    evals = evals[pos]
    evecs = evecs[:, pos]
    n_keep = _choose_n_keep(evals, num_modes=args.num_modes, pod_tol=args.pod_tol)

    lam = evals[:n_keep]
    z = evecs[:, :n_keep]
    basis = S @ (z / np.sqrt(lam)[None, :])
    sigma = np.sqrt(np.maximum(lam, 0.0))

    gram_err = np.nan
    if not bool(args.skip_gram_check):
        G_j, gram_samples, gram_wsum = _build_weff_gram_on_basis(
            basis=basis,
            mu_weights=metric_mu_weights,
            time_weights=metric_time_weights,
            mu_samples=mu_samples,
            metric_select_map=metric_select_map,
            snap_folder=snap_folder,
            dt=float(args.dt),
            num_steps=int(args.num_steps),
            p_diag=p_diag,
            progress_every=max(0, int(args.progress_every)),
        )
        G_m = basis.T @ (m_diag[:, None] * basis)
        G = eps_eff * G_m + G_j
        G = 0.5 * (G + G.T)
        gram_err = float(np.linalg.norm(G - np.eye(n_keep), ord="fro"))
    else:
        gram_samples = 0
        gram_wsum = 0.0

    mu_tokens = [s.strip() for s in str(args.test_mu).split(",")]
    if len(mu_tokens) != 2:
        raise ValueError("--test-mu must be 'mu1,mu2'")
    test_mu = [float(mu_tokens[0]), float(mu_tokens[1])]
    hdm = np.asarray(
        load_or_compute_snaps(test_mu, GRID_X, GRID_Y, W0, float(args.dt), int(args.num_steps), snap_folder=snap_folder),
        dtype=np.float64,
    )
    q_ls = np.linalg.lstsq(basis, hdm - u_ref[:, None], rcond=None)[0]
    rec = u_ref[:, None] + basis @ q_ls
    rel_err = float(np.linalg.norm(hdm - rec) / (np.linalg.norm(hdm) + 1e-30))

    basis_file = os.path.join(stage1_dir, str(args.basis_name))
    sigma_file = os.path.join(stage1_dir, str(args.sigma_name))
    uref_file = os.path.join(stage1_dir, str(args.uref_name))
    meta_file = os.path.join(stage1_dir, str(args.meta_name))
    summary_file = os.path.join(stage1_dir, str(args.summary_name))
    decay_plot = os.path.join(stage1_dir, str(args.decay_plot_name))

    np.save(basis_file, basis)
    np.save(sigma_file, sigma)
    np.save(uref_file, u_ref)
    np.savez(
        meta_file,
        tag=np.asarray(paths.tag),
        timestamp=np.asarray(datetime.now().isoformat(timespec="seconds")),
        n_keep=np.asarray(int(n_keep), dtype=np.int64),
        n_available=np.asarray(int(evals.size), dtype=np.int64),
        n_snapshot_cols=np.asarray(int(n_cols), dtype=np.int64),
        snapshot_columns_used=np.asarray(int(n_snap_used), dtype=np.int64),
        metric_samples_used=np.asarray(int(metric_used), dtype=np.int64),
        metric_weight_sum=np.asarray(float(metric_wsum), dtype=np.float64),
        centered=np.asarray(int(bool(args.center)), dtype=np.int64),
        eps_mode=np.asarray(str(eps_mode)),
        eps_mblock=np.asarray(float(args.eps_mblock), dtype=np.float64),
        eps_ratio=np.asarray(float(args.eps_ratio), dtype=np.float64),
        eps_effective=np.asarray(float(eps_eff), dtype=np.float64),
        metric_time_weighting=np.asarray(str(args.metric_time_weighting)),
        metric_mu_weighting=np.asarray(str(args.metric_mu_weighting)),
        purist_full=np.asarray(int(bool(args.purist_full)), dtype=np.int64),
        lspg_p_source=np.asarray(str(p_mode)),
        gram_err=np.asarray(float(gram_err), dtype=np.float64),
        snap_folder=np.asarray(str(snap_folder)),
    )

    plot_singular_value_decay(
        sigma,
        out_path=decay_plot,
        max_modes=min(1000, sigma.size),
        label="Proposal 1 (W_eff) POD",
        title="Proposal 1 W_eff POD residual energy decay",
        use_latex=True,
    )

    energy = np.cumsum(evals) / np.sum(evals)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"tag: {paths.tag}\n")
        f.write("script: Project_YvonMaday/stage1_lspg_proposal1_weff_pod.py\n")
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"snap_folder: {snap_folder}\n")
        f.write(f"mu_samples: {len(mu_samples)}\n")
        f.write(f"snapshot_columns_used: {n_snap_used}\n")
        f.write(f"snapshot_columns_candidates: {n_snap_cand}\n")
        f.write(f"metric_samples_used: {metric_used}\n")
        f.write(f"metric_samples_candidates: {n_metric_cand}\n")
        f.write(f"purist_full: {bool(args.purist_full)}\n")
        f.write(f"snapshot_time_stride_effective: {int(snap_stride)}\n")
        f.write(f"metric_time_stride_effective: {int(metric_stride)}\n")
        f.write(f"snapshot_select_mode: {snapshot_select_mode}\n")
        f.write(f"metric_select_mode: {metric_select_mode}\n")
        f.write(f"snapshot_percent: {args.snapshot_percent}\n")
        f.write(f"metric_percent: {args.metric_percent}\n")
        f.write(f"metric_time_weighting: {str(args.metric_time_weighting)}\n")
        f.write(f"metric_mu_weighting: {str(args.metric_mu_weighting)}\n")
        f.write(f"eps_mode: {eps_mode}\n")
        f.write(f"eps_mblock: {float(args.eps_mblock):.8e}\n")
        f.write(f"eps_ratio: {float(args.eps_ratio):.8e}\n")
        f.write(f"eps_effective: {float(eps_eff):.8e}\n")
        f.write(f"lspg_p_source: {p_mode}\n")
        f.write(f"n_keep: {n_keep}\n")
        f.write(f"n_available: {evals.size}\n")
        f.write(f"energy_captured: {energy[n_keep - 1]:.8e}\n")
        f.write(f"energy_lost: {1.0 - energy[n_keep - 1]:.8e}\n")
        f.write(f"weff_orthonorm_fro_err: {gram_err}\n")
        f.write(f"reconstruction_rel_err_mu_test: {rel_err:.8e}\n")
        f.write(f"snapshot_assembly_time_s: {t_snap:.6f}\n")
        f.write(f"weff_correlation_time_s: {t_metric:.6f}\n")
        f.write(f"gram_check_samples_used: {gram_samples}\n")
        f.write(f"metric_weight_sum: {metric_wsum:.8e}\n")
        f.write(f"gram_weight_sum: {gram_wsum:.8e}\n")

    print(f"[MADAY-P1-WEFF] saved basis:   {basis_file}")
    print(f"[MADAY-P1-WEFF] saved summary: {summary_file}")
    print(f"[MADAY-P1-WEFF] reconstruction rel error @mu={test_mu}: {rel_err:.6e}")


if __name__ == "__main__":
    main()
