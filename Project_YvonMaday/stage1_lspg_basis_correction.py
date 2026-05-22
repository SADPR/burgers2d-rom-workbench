#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Apply Maday Proposal-2/3 basis correction on an existing Stage-1 basis.

This script is non-intrusive to baseline outputs:
it only reads/writes under `Results_Maday/<tag>/Stage1/`.

Metric options:
- `identity` / `diag_file`: block matrices from V^T M V, V^T M Vbar, ...
- `lspg_avg`: block matrices from averaged LSPG sensitivity metric
    A = sum_s w_s J_s^T P J_s
  without assembling full A (uses reduced blocks directly).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU
from burgers.core import (
    get_snapshot_params,
    get_ops,
    inviscid_burgers_exact_jac2D,
    load_or_compute_snaps,
)
from burgers.ecsw_utils import build_ecsw_snapshot_plan

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


def _parse_range(arg: str, default: tuple[float, float]) -> tuple[float, float]:
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


def _load_diag_vector(path: str, expected_size: int, *, label: str) -> np.ndarray:
    vec = np.asarray(np.load(os.path.abspath(os.path.expanduser(path)), allow_pickle=False), dtype=np.float64).reshape(-1)
    if vec.size != expected_size:
        raise ValueError(f"{label} size mismatch: got {vec.size}, expected {expected_size}")
    if np.any(~np.isfinite(vec)):
        raise ValueError(f"{label} contains non-finite entries.")
    return vec


def _load_diag_metric(source: str, metric_file: str | None, n_dofs: int) -> np.ndarray:
    src = str(source).strip().lower()
    if src == "identity":
        w = np.ones((n_dofs,), dtype=np.float64)
    elif src == "diag_file":
        if not metric_file:
            raise ValueError("--metric-file is required when --metric-source=diag_file")
        w = _load_diag_vector(metric_file, n_dofs, label="metric diagonal")
    else:
        raise ValueError(f"Unsupported diagonal metric source: {source}")
    if np.any(w <= 0.0):
        raise ValueError("Diagonal metric entries must be strictly positive.")
    return w


def _weighted_block(Va: np.ndarray, wdiag: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    return Va.T @ (wdiag[:, None] * Vb)


def _resolve_snap_folder(snap_folder_arg: str | None) -> str:
    if snap_folder_arg:
        return os.path.abspath(os.path.expanduser(str(snap_folder_arg)))
    candidates = [
        os.path.join(PROJECT_ROOT, "Results", "param_snaps"),
        os.path.join(PROJECT_ROOT, "param_snaps"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


def _select_sample_indices(
    num_candidates: int,
    max_samples: int | None,
    sample_seed: int,
) -> np.ndarray:
    if num_candidates < 1:
        raise RuntimeError("No candidate LSPG samples were generated.")

    if max_samples is None or int(max_samples) >= num_candidates:
        return np.arange(num_candidates, dtype=np.int64)

    k = max(1, int(max_samples))
    rng = np.random.default_rng(int(sample_seed))
    idx = rng.choice(num_candidates, size=k, replace=False)
    idx.sort()
    return idx.astype(np.int64)


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


def _select_mu_time_samples_ecsw(
    *,
    num_steps: int,
    mu_samples: list[list[float]],
    mode: str,
    total_percent: float | None,
    total_count: int | None,
    time_offset: int,
    sample_seed: int,
) -> tuple[list[np.ndarray], int]:
    mode_map = {
        "ecsw_global_stratified": "global_stratified_random",
        "ecsw_param_time_stratified": "global_param_time_stratified",
    }
    m = str(mode).strip().lower()
    if m not in mode_map:
        raise ValueError(f"Unsupported sample-select mode: {mode}")
    if total_percent is None and total_count is None:
        raise ValueError(
            f"{mode} requires --sample-percent or --sample-total to define sampling budget."
        )

    plan = build_ecsw_snapshot_plan(
        num_steps=int(num_steps) + 1,
        snap_time_offset=max(1, int(time_offset)),
        num_mu=int(len(mu_samples)),
        mode=mode_map[m],
        total_snapshots=(None if total_count is None else int(total_count)),
        total_snapshots_percent=(None if total_percent is None else float(total_percent)),
        mu_points=np.asarray(mu_samples, dtype=np.float64),
        random_seed=int(sample_seed),
        ensure_mu_coverage=True,
    )
    selected = plan.get("selected_now_cols_by_mu", [])
    if len(selected) != len(mu_samples):
        raise RuntimeError(
            f"ECSW-like sample plan produced {len(selected)} mu buckets, expected {len(mu_samples)}."
        )
    selected_by_mu: list[np.ndarray] = []
    for arr in selected:
        cols = np.asarray(arr, dtype=np.int64).reshape(-1)
        cols = cols[(cols >= 0) & (cols <= int(num_steps))]
        selected_by_mu.append(np.sort(np.unique(cols)))
    n_candidates = int(plan.get("num_candidates_total", 0))
    return selected_by_mu, n_candidates


def _build_lspg_avg_blocks(
    *,
    V: np.ndarray,
    Vbar: np.ndarray,
    dt: float,
    num_steps: int,
    snap_folder: str,
    mu_samples: list[list[float]],
    time_stride: int,
    max_samples: int | None,
    sample_seed: int,
    sample_select_mode: str,
    sample_percent: float | None,
    sample_total: int | None,
    sample_time_offset: int,
    time_weighting: str,
    mu_weighting: str,
    p_source: str,
    p_diag_file: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n_dofs = V.shape[0]
    n = V.shape[1]
    nbar = Vbar.shape[1]

    stride = max(1, int(time_stride))
    select_mode = str(sample_select_mode).strip().lower()
    dt = float(dt)
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError("--num-steps must be >= 1")

    p_mode = str(p_source).strip().lower()
    if p_mode not in ("identity", "diag_file"):
        raise ValueError(f"Unsupported --lspg-p-source: {p_source}")
    if p_mode == "identity":
        p_diag = None
    else:
        if not p_diag_file:
            raise ValueError("--lspg-p-diag-file is required when --lspg-p-source=diag_file")
        p_diag = _load_diag_vector(p_diag_file, n_dofs, label="P diagonal")
        if np.any(p_diag <= 0.0):
            raise ValueError("P diagonal must be strictly positive.")

    # Count candidate (mu, time) states first, then build selected-by-mu sets.
    n_time = num_steps + 1
    if select_mode == "strided":
        candidate_times = np.arange(0, n_time, stride, dtype=np.int64)
        num_candidates = len(mu_samples) * candidate_times.size
        sample_ids = _select_sample_indices(num_candidates, max_samples=max_samples, sample_seed=sample_seed)
        selected_by_mu = [list() for _ in range(len(mu_samples))]
        n_local = int(candidate_times.size)
        for gid in sample_ids.tolist():
            imu = int(gid) // n_local
            j = int(gid) % n_local
            if 0 <= imu < len(selected_by_mu):
                selected_by_mu[imu].append(int(candidate_times[j]))
        selected_by_mu = [np.asarray(sorted(set(lst)), dtype=np.int64) for lst in selected_by_mu]
    else:
        candidate_times = None
        total_count = int(sample_total) if sample_total is not None else (int(max_samples) if max_samples is not None else None)
        selected_by_mu, num_candidates = _select_mu_time_samples_ecsw(
            num_steps=num_steps,
            mu_samples=mu_samples,
            mode=select_mode,
            total_percent=sample_percent,
            total_count=total_count,
            time_offset=int(sample_time_offset),
            sample_seed=int(sample_seed),
        )
    tw = _time_weights(num_steps=num_steps, dt=dt, mode=time_weighting)
    mw = _mu_weights(num_mu=len(mu_samples), mode=mu_weighting)

    H_LL = np.zeros((n, n), dtype=np.float64)
    H_LH = np.zeros((n, nbar), dtype=np.float64)
    H_HH = np.zeros((nbar, nbar), dtype=np.float64)
    H_HL = np.zeros((nbar, n), dtype=np.float64)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(GRID_X, GRID_Y)
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)

    processed = 0
    wsum = 0.0
    for imu, mu in enumerate(mu_samples):
        snaps = np.asarray(
            load_or_compute_snaps(mu, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        if snaps.shape[0] != n_dofs:
            raise ValueError(
                f"Snapshot DOF mismatch for mu={mu}: got {snaps.shape[0]}, expected {n_dofs}."
            )
        if snaps.shape[1] != n_time:
            raise ValueError(
                f"Snapshot time-length mismatch for mu={mu}: got {snaps.shape[1]}, expected {n_time}."
            )

        times_iter = selected_by_mu[imu]

        for it in times_iter:
            w = snaps[:, int(it)]
            J = inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)
            alpha = float(mw[int(imu)] * tw[int(it)])

            JV = np.asarray(J @ V, dtype=np.float64)
            JH = np.asarray(J @ Vbar, dtype=np.float64)
            if p_diag is None:
                H_LL += alpha * (JV.T @ JV)
                H_LH += alpha * (JV.T @ JH)
                H_HH += alpha * (JH.T @ JH)
                H_HL += alpha * (JH.T @ JV)
            else:
                H_LL += alpha * (JV.T @ (p_diag[:, None] * JV))
                H_LH += alpha * (JV.T @ (p_diag[:, None] * JH))
                H_HH += alpha * (JH.T @ (p_diag[:, None] * JH))
                H_HL += alpha * (JH.T @ (p_diag[:, None] * JV))

            processed += 1
            wsum += alpha

    if processed == 0:
        raise RuntimeError("No LSPG samples were processed (processed=0).")

    if wsum <= 0.0:
        raise RuntimeError("Weighted LSPG average has non-positive total weight.")
    wgt = 1.0 / float(wsum)
    H_LL *= wgt
    H_LH *= wgt
    H_HH *= wgt
    H_HL *= wgt

    if candidate_times is not None:
        num_time_candidates = int(candidate_times.size)
    else:
        # In ECSW-like sampling modes there is no single global candidate_times array.
        num_time_candidates = int(max((arr.size for arr in selected_by_mu), default=0))

    info = {
        "metric_source_effective": "lspg_avg",
        "lspg_p_source": p_mode,
        "num_mu_samples": int(len(mu_samples)),
        "num_time_candidates": int(num_time_candidates),
        "num_candidates_total": int(num_candidates),
        "num_samples_used": int(processed),
        "sample_weight_sum": float(wsum),
        "sample_select_mode": str(select_mode),
        "time_stride": int(stride),
        "time_weighting": str(time_weighting),
        "mu_weighting": str(mu_weighting),
        "dt": float(dt),
        "num_steps": int(num_steps),
        "snap_folder": str(snap_folder),
    }
    return H_LL, H_LH, H_HH, H_HL, info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply low/high basis correction in a selected metric.")
    parser.add_argument("--maday-tag", type=str, default="exp_maday_p2")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--stage1-dir", type=str, default=None, help="Optional override for Stage1 directory.")
    parser.add_argument("--basis-file", type=str, default="basis_weighted.npy")
    parser.add_argument("--primary-modes", type=int, required=True, help="Low/retained mode count n.")
    parser.add_argument("--proposal", choices=("high", "low"), default="high", help="'high' = Proposal 2, 'low' = Proposal 3.")
    parser.add_argument("--metric-source", choices=("identity", "diag_file", "lspg_avg"), default="diag_file")
    parser.add_argument("--metric-file", type=str, default=None, help="Required for metric-source=diag_file.")
    parser.add_argument("--regularization", type=float, default=1e-12)
    parser.add_argument("--output-stem", type=str, default=None)

    # Options only used when --metric-source=lspg_avg
    parser.add_argument("--snap-folder", type=str, default=None, help="Override snapshot folder for lspg_avg.")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--mu1-range", type=str, default=None, help="Override mu1 range as 'low,high'.")
    parser.add_argument("--mu2-range", type=str, default=None, help="Override mu2 range as 'low,high'.")
    parser.add_argument("--samples-per-mu", type=int, default=SAMPLES_PER_MU)
    parser.add_argument("--max-mu-samples", type=int, default=None, help="Optional cap on number of mu points used.")
    parser.add_argument("--time-stride", type=int, default=25, help="Use one state every `time-stride` steps.")
    parser.add_argument("--max-samples", type=int, default=240, help="Maximum sampled (mu,time) states for A average.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-select-mode",
        choices=("strided", "ecsw_global_stratified", "ecsw_param_time_stratified"),
        default="strided",
    )
    parser.add_argument("--sample-percent", type=float, default=None)
    parser.add_argument("--sample-total", type=int, default=None)
    parser.add_argument("--sample-time-offset", type=int, default=1)
    parser.add_argument("--time-weighting", choices=("uniform", "trapezoid"), default="uniform")
    parser.add_argument("--mu-weighting", choices=("uniform",), default="uniform")
    parser.add_argument(
        "--use-all-samples",
        action="store_true",
        default=False,
        help="Use all (mu,time) states for lspg_avg (time-stride=1, no max-samples cap).",
    )
    parser.add_argument("--lspg-p-source", choices=("identity", "diag_file"), default="identity")
    parser.add_argument("--lspg-p-diag-file", type=str, default=None)
    args = parser.parse_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    stage1_dir = os.path.abspath(args.stage1_dir) if args.stage1_dir else paths.stage1
    os.makedirs(stage1_dir, exist_ok=True)

    basis_path = args.basis_file
    if not os.path.isabs(basis_path):
        basis_path = os.path.join(stage1_dir, basis_path)
    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Basis file not found: {basis_path}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"Basis must be 2D, got {basis.shape}")
    n_dofs, n_tot = basis.shape

    n = int(args.primary_modes)
    if not (0 < n < n_tot):
        raise ValueError(f"primary-modes must satisfy 0 < n < n_tot ({n_tot}), got {n}")

    V = basis[:, :n]
    Vbar = basis[:, n:]
    nbar = Vbar.shape[1]

    metric_info = {"metric_source_effective": str(args.metric_source).strip().lower()}
    msrc = str(args.metric_source).strip().lower()
    if msrc in ("identity", "diag_file"):
        wdiag = _load_diag_metric(msrc, args.metric_file, n_dofs=n_dofs)
        H_LL = _weighted_block(V, wdiag, V)
        H_LH = _weighted_block(V, wdiag, Vbar)
        H_HH = _weighted_block(Vbar, wdiag, Vbar)
        H_HL = _weighted_block(Vbar, wdiag, V)
        metric_info["diag_metric_min"] = float(np.min(wdiag))
        metric_info["diag_metric_max"] = float(np.max(wdiag))
    else:
        mu1_range = _parse_range(args.mu1_range, MU1_RANGE)
        mu2_range = _parse_range(args.mu2_range, MU2_RANGE)
        mu_samples = get_snapshot_params(
            mu1_range=mu1_range,
            mu2_range=mu2_range,
            samples_per_mu=int(args.samples_per_mu),
        )
        if args.max_mu_samples is not None and int(args.max_mu_samples) > 0:
            mu_samples = mu_samples[: int(args.max_mu_samples)]
        snap_folder = _resolve_snap_folder(args.snap_folder)
        os.makedirs(snap_folder, exist_ok=True)

        time_stride = int(args.time_stride)
        max_samples = int(args.max_samples) if args.max_samples is not None else None
        if bool(args.use_all_samples):
            time_stride = 1
            max_samples = None

        H_LL, H_LH, H_HH, H_HL, info = _build_lspg_avg_blocks(
            V=V,
            Vbar=Vbar,
            dt=float(args.dt),
            num_steps=int(args.num_steps),
            snap_folder=snap_folder,
            mu_samples=mu_samples,
            time_stride=time_stride,
            max_samples=max_samples,
            sample_seed=int(args.sample_seed),
            sample_select_mode=str(args.sample_select_mode),
            sample_percent=(None if args.sample_percent is None else float(args.sample_percent)),
            sample_total=(None if args.sample_total is None else int(args.sample_total)),
            sample_time_offset=int(args.sample_time_offset),
            time_weighting=str(args.time_weighting),
            mu_weighting=str(args.mu_weighting),
            p_source=str(args.lspg_p_source),
            p_diag_file=args.lspg_p_diag_file,
        )
        metric_info.update(info)
        metric_info["use_all_samples"] = bool(args.use_all_samples)

    reg = float(args.regularization)
    if reg < 0.0:
        raise ValueError("--regularization must be >= 0")

    if args.proposal == "high":
        solve_mat = H_LL + reg * np.eye(H_LL.shape[0], dtype=np.float64)
        K = np.linalg.solve(solve_mat, H_LH)
        Vbar_tilde = Vbar - V @ K
        basis_corr = np.concatenate([V, Vbar_tilde], axis=1)
        cross_before = H_LH
        cross_after = H_LL @ K - H_LH
        map_name = "K_high_to_low_shift"
        map_mat = K
    else:
        solve_mat = H_HH + reg * np.eye(H_HH.shape[0], dtype=np.float64)
        L = np.linalg.solve(solve_mat, H_HL)
        V_tilde = V - Vbar @ L
        basis_corr = np.concatenate([V_tilde, Vbar], axis=1)
        cross_before = H_HL
        cross_after = H_HH @ L - H_HL
        map_name = "L_low_to_high_shift"
        map_mat = L

    stem = args.output_stem
    if not stem:
        stem = f"basis_corrected_p{2 if args.proposal == 'high' else 3}_n{n}"

    basis_out = os.path.join(stage1_dir, f"{stem}.npy")
    map_out = os.path.join(stage1_dir, f"{stem}_{map_name}.npy")
    meta_out = os.path.join(stage1_dir, f"{stem}_metadata.npz")
    summary_out = os.path.join(stage1_dir, f"{stem}_summary.txt")

    np.save(basis_out, basis_corr)
    np.save(map_out, map_mat)
    np.savez(
        meta_out,
        tag=np.asarray(paths.tag),
        timestamp=np.asarray(datetime.now().isoformat(timespec="seconds")),
        proposal=np.asarray(args.proposal),
        n_primary=np.asarray(n, dtype=np.int64),
        n_secondary=np.asarray(nbar, dtype=np.int64),
        metric_source=np.asarray(msrc),
        regularization=np.asarray(reg, dtype=np.float64),
        cross_before_fro=np.asarray(np.linalg.norm(cross_before, ord="fro"), dtype=np.float64),
        cross_after_fro=np.asarray(np.linalg.norm(cross_after, ord="fro"), dtype=np.float64),
        lspg_num_samples=np.asarray(int(metric_info.get("num_samples_used", -1)), dtype=np.int64),
    )

    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(f"tag: {paths.tag}\n")
        f.write("script: Project_YvonMaday/stage1_lspg_basis_correction.py\n")
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"proposal: {args.proposal}\n")
        f.write(f"basis_in: {basis_path}\n")
        f.write(f"basis_out: {basis_out}\n")
        f.write(f"mapping_out: {map_out}\n")
        f.write(f"metric_source: {msrc}\n")
        f.write(f"n_primary: {n}\n")
        f.write(f"n_secondary: {nbar}\n")
        f.write(f"cross_before_fro: {np.linalg.norm(cross_before, ord='fro'):.8e}\n")
        f.write(f"cross_after_fro: {np.linalg.norm(cross_after, ord='fro'):.8e}\n")
        for k in sorted(metric_info.keys()):
            f.write(f"{k}: {metric_info[k]}\n")

    print(f"[MADAY-CORR] saved corrected basis: {basis_out}")
    print(f"[MADAY-CORR] saved mapping: {map_out}")
    print(
        "[MADAY-CORR] cross Fro norm "
        f"before={np.linalg.norm(cross_before, ord='fro'):.6e}, "
        f"after={np.linalg.norm(cross_after, ord='fro'):.6e}"
    )
    if msrc == "lspg_avg":
        print(
            "[MADAY-CORR] lspg_avg samples="
            f"{metric_info.get('num_samples_used')} / {metric_info.get('num_candidates_total')}"
        )


if __name__ == "__main__":
    main()
