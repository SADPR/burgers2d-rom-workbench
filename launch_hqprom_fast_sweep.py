#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast global HQPROM screening sweep (stage1+stage2 style, no online HQPROM solve).

What it does:
- Builds training snapshot matrix once.
- Computes POD/SVD once.
- Sweeps (zeta_qua, ridge_alpha) by fitting H on the training set.
- Evaluates quadratic-manifold reconstruction error on test points.
- Writes resume-safe CSV summary.

Optional:
- Save each candidate manifold (qm_V.npy, qm_H.npy, qm_uref.npy, qm_sigma.npy,
  qm_metadata.npz) so you can run full run_hqprom.py later only for finalists.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from burgers.core import load_or_compute_snaps, get_snapshot_params
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS
from Quadratic.stage1_quadratic_offline import (
    build_Q_symmetric_matrix,
    pod_rank_from_tolerance,
)


def parse_float_list(text: str) -> list[float]:
    vals = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def parse_points(text: str) -> list[tuple[float, float]]:
    pts = []
    for group in str(text).split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [p.strip() for p in group.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid point '{group}'. Expected format: mu1,mu2;mu1,mu2"
            )
        pts.append((float(parts[0]), float(parts[1])))
    if not pts:
        raise ValueError("Expected at least one test point.")
    return pts


def ensure_csv(csv_path: Path) -> None:
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "tag",
                "zeta_qua",
                "alpha_ridge",
                "n_trad",
                "n_final",
                "err_4.56_0.019",
                "err_4.75_0.020",
                "err_5.19_0.026",
                "max_err_percent",
                "h_fro_norm",
                "status",
                "note",
            ]
        )


def load_done_tags(csv_path: Path) -> set[str]:
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = str(row.get("status", "ok")).strip().lower()
            tag = str(row.get("tag", "")).strip()
            if tag and status == "ok":
                done.add(tag)
    return done


def append_row(csv_path: Path, row: list[object]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def format_tag(zeta_qua: float, alpha_ridge: float) -> str:
    return f"z{zeta_qua:g}_a{alpha_ridge:.0e}".replace("+", "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast global HQPROM screening sweep via quadratic-manifold "
            "reconstruction (no full HQPROM online solve)."
        )
    )
    parser.add_argument("--pod-tol", type=float, default=1e-4)
    parser.add_argument(
        "--zetas",
        type=str,
        default="2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9",
        help="Comma-separated zeta_qua values.",
    )
    parser.add_argument(
        "--ridge-alphas",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,3e2,1e3,3e3,1e4,3e4,1e5",
        help="Comma-separated ridge alpha values.",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="4.56,0.019;4.75,0.020;5.19,0.026",
        help="Semicolon-separated mu points: mu1,mu2;mu1,mu2",
    )
    parser.add_argument(
        "--center-mode",
        type=str,
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--snap-folder", type=str, default="Results/param_snaps")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="QuadraticSweep/hqprom_fast_screen",
        help="Where summary.csv and optional candidate folders are written.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help=(
            "Save qm_* files for each candidate under <root-dir>/<tag>/ "
            "so you can run full run_hqprom.py later only for finalists."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run all candidates even if already marked as status=ok in summary.csv.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print full traceback on failed candidates.",
    )
    return parser


def solve_H_from_AB(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    m = A.shape[0]
    A_reg = A + float(alpha) * np.eye(m, dtype=np.float64)
    try:
        H = np.linalg.solve(A_reg.T, B.T).T
    except np.linalg.LinAlgError:
        H = np.linalg.lstsq(A_reg.T, B.T, rcond=None)[0].T
    return H


def main() -> None:
    args = build_parser().parse_args()

    if args.num_steps < 1:
        raise ValueError("--num-steps must be >= 1")

    zetas = parse_float_list(args.zetas)
    ridge_alphas = parse_float_list(args.ridge_alphas)
    points = parse_points(args.points)

    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    ensure_csv(csv_path)
    done = set() if args.force else load_done_tags(csv_path)

    param_list = get_snapshot_params()
    if len(param_list) == 0:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    w0 = np.asarray(W0, dtype=np.float64).copy()

    print("[fast-sweep] Building training snapshot matrix S once...", flush=True)
    S0 = load_or_compute_snaps(
        param_list[0], GRID_X, GRID_Y, w0, args.dt, args.num_steps, snap_folder=args.snap_folder
    )
    N, T = S0.shape
    Ns_total = len(param_list) * T
    S = np.zeros((N, Ns_total), dtype=np.float64)

    col = 0
    for mu in param_list:
        S_mu = load_or_compute_snaps(
            mu, GRID_X, GRID_Y, w0, args.dt, args.num_steps, snap_folder=args.snap_folder
        )
        S[:, col:col + T] = S_mu
        col += T

    print(f"[fast-sweep] S shape: {S.shape}", flush=True)

    use_u_ref = args.center_mode == "on"
    if use_u_ref:
        u_ref = np.mean(S, axis=1)
        S -= u_ref[:, None]  # in-place centering, reuse memory
        u_ref_source = "mean(training_snapshots)"
    else:
        u_ref = np.zeros(N, dtype=np.float64)
        u_ref_source = "zeros(off)"

    print("[fast-sweep] Computing one SVD...", flush=True)
    U_full, s_all, _ = np.linalg.svd(S, full_matrices=False)

    n_trad, energy_captured, energy_lost = pod_rank_from_tolerance(s_all, args.pod_tol)
    n_qua_raw = (np.sqrt(9.0 + 8.0 * n_trad) - 3.0) / 2.0
    n_max_ls = int(np.floor((np.sqrt(1.0 + 8.0 * Ns_total) - 1.0) / 2.0))

    print(
        f"[fast-sweep] n_trad={n_trad}, n_qua_raw={n_qua_raw:.4f}, n_max_ls={n_max_ls}, "
        f"energy_lost={energy_lost:.3e}",
        flush=True,
    )

    print("[fast-sweep] Loading test snapshots once...", flush=True)
    test_data = {}
    for (mu1, mu2) in points:
        X = load_or_compute_snaps(
            [mu1, mu2], GRID_X, GRID_Y, w0, args.dt, args.num_steps, snap_folder=args.snap_folder
        )
        Xc = X - u_ref[:, None]
        Xnorm = np.linalg.norm(X)
        test_data[(mu1, mu2)] = (Xc, Xnorm)

    # Map each candidate tag -> (zeta, alpha, n)
    candidates = []
    for z in zetas:
        n_qua_corr = int(np.floor((1.0 + float(z)) * n_qua_raw))
        n_final = max(1, min(n_qua_corr, n_max_ls, int(s_all.size)))
        for a in ridge_alphas:
            tag = format_tag(float(z), float(a))
            candidates.append((tag, float(z), float(a), int(n_final), int(n_qua_corr)))

    # Group by n to avoid recomputing expensive intermediates
    by_n = {}
    for item in candidates:
        by_n.setdefault(item[3], []).append(item)

    n_total = len(candidates)
    done_count = 0

    for n, items in sorted(by_n.items(), key=lambda kv: kv[0]):
        active_items = [it for it in items if (it[0] not in done)]
        if not active_items:
            done_count += len(items)
            continue

        print(f"\n[fast-sweep] Preparing shared matrices for n={n}...", flush=True)
        V = U_full[:, :n]
        sigma = s_all[:n]

        # Training reduced coordinates / quadratic terms
        q_train = V.T @ S                    # (n, Ns)
        Q_train = build_Q_symmetric_matrix(q_train)  # (m, Ns)

        # Build A and B without explicitly forming E to reduce memory:
        # E = S - V q_train
        # B = E Q^T = S Q^T - V (q_train Q^T)
        A = Q_train @ Q_train.T
        SQT = S @ Q_train.T
        qQT = q_train @ Q_train.T
        B = SQT - V @ qQT

        # Precompute test reduced coordinates for this n
        test_proj = {}
        for pt, (Xc, Xnorm) in test_data.items():
            q_t = V.T @ Xc
            Q_t = build_Q_symmetric_matrix(q_t)
            test_proj[pt] = (Xc, Xnorm, q_t, Q_t)

        for (tag, zeta_qua, alpha_ridge, n_final, n_qua_corr) in items:
            if tag in done:
                done_count += 1
                print(f"[skip {done_count}/{n_total}] {tag}", flush=True)
                continue

            print(f"[candidate {done_count + 1}/{n_total}] {tag}", flush=True)
            try:
                H = solve_H_from_AB(A, B, alpha_ridge)

                errs = []
                for (mu1, mu2) in points:
                    Xc, Xnorm, q_t, Q_t = test_proj[(mu1, mu2)]
                    Xc_rec = V @ q_t + H @ Q_t
                    if Xnorm > 0.0:
                        err = 100.0 * np.linalg.norm(Xc - Xc_rec) / Xnorm
                    else:
                        err = np.nan
                    errs.append(float(err))

                max_err = float(np.nanmax(errs))
                h_norm = float(np.linalg.norm(H))

                if args.save_models:
                    cdir = root / tag
                    cdir.mkdir(parents=True, exist_ok=True)
                    np.save(cdir / "qm_V.npy", V)
                    np.save(cdir / "qm_H.npy", H)
                    np.save(cdir / "qm_uref.npy", u_ref)
                    np.save(cdir / "qm_sigma.npy", sigma)
                    np.savez(
                        cdir / "qm_metadata.npz",
                        pod_tol=np.float64(args.pod_tol),
                        zeta_qua=np.float64(zeta_qua),
                        ridge_alpha=np.float64(alpha_ridge),
                        center_mode=np.str_(args.center_mode),
                        use_u_ref=np.bool_(use_u_ref),
                        u_ref_source=np.str_(u_ref_source),
                        n_trad=np.int64(n_trad),
                        n_qua_raw=np.float64(n_qua_raw),
                        n_qua_corr=np.int64(n_qua_corr),
                        n_max_ls=np.int64(n_max_ls),
                        n_final=np.int64(n_final),
                        N=np.int64(N),
                        T=np.int64(T),
                        Ns_total=np.int64(Ns_total),
                        num_training_params=np.int64(len(param_list)),
                        energy_captured=np.float64(energy_captured),
                        energy_lost=np.float64(energy_lost),
                    )

                row = [
                    tag,
                    zeta_qua,
                    alpha_ridge,
                    int(n_trad),
                    int(n_final),
                    errs[0] if len(errs) > 0 else np.nan,
                    errs[1] if len(errs) > 1 else np.nan,
                    errs[2] if len(errs) > 2 else np.nan,
                    max_err,
                    h_norm,
                    "ok",
                    "",
                ]
                append_row(csv_path, row)
                done_count += 1
                print(
                    f"[ok] {tag} -> n={n_final}, errs={errs}, max_err={max_err:.6f}%",
                    flush=True,
                )

            except Exception as exc:
                row = [
                    tag,
                    zeta_qua,
                    alpha_ridge,
                    int(n_trad),
                    int(n_final),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                ]
                append_row(csv_path, row)
                done_count += 1
                print(f"[fail] {tag}: {type(exc).__name__}: {exc}", flush=True)
                if args.traceback:
                    traceback.print_exc()

    print(f"\nDone. Summary: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
