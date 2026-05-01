#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch a local HQPROM sweep without editing LocalQuadratic/stage1_local_qm_offline.py.

This script builds S and clusters once, then sweeps (zeta_qua, alpha_ridge),
builds candidate local quadratic models, runs run_local_hqprom on test points,
and writes a resume-safe CSV summary.
"""

import argparse
import csv
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import runpy


REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from burgers.core import get_snapshot_params
from burgers.config import DT, NUM_STEPS
import run_local_hqprom


def parse_float_list(text):
    vals = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def parse_points(text):
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


def ensure_csv(csv_path):
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "tag",
                "zeta_qua",
                "alpha_ridge",
                "min_n_qm",
                "max_n_qm",
                "avg_n_qm",
                "err_4.56_0.019",
                "err_4.75_0.020",
                "err_5.19_0.026",
                "max_err_percent",
                "status",
                "note",
            ]
        )


def load_done_tags(csv_path):
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


def append_row(csv_path, row):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def format_tag(zeta_qua, alpha_ridge, q_normalization_mode="off"):
    tag = f"z{zeta_qua:g}_a{alpha_ridge:.0e}".replace("+", "")
    mode = str(q_normalization_mode).strip().lower()
    if mode != "off":
        tag += f"_q{mode}"
    return tag


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep local HQPROM over (zeta_qua, alpha_ridge) candidates.",
    )
    parser.add_argument("--pod-tol", type=float, default=1e-4)
    parser.add_argument(
        "--zetas",
        type=str,
        default="0.2,0.5,0.8,1.0,1.2,1.5,2.0,3.0",
        help="Comma-separated zeta_qua values.",
    )
    parser.add_argument(
        "--ridge-alphas",
        type=str,
        default="1e-4,1,1e4",
        help="Comma-separated ridge alpha values.",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="4.56,0.019;4.75,0.020;5.19,0.026",
        help="Semicolon-separated mu points: mu1,mu2;mu1,mu2",
    )

    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument(
        "--clustering-method",
        type=str,
        choices=("kmeans", "fuzzy"),
        default="kmeans",
    )
    parser.add_argument("--phi", type=float, default=0.1)
    parser.add_argument(
        "--pod-method",
        type=str,
        choices=("svd", "rsvd"),
        default="svd",
    )
    parser.add_argument(
        "--q-normalization-mode",
        type=str,
        choices=("off", "std"),
        default="off",
        help="Reduced-coordinate normalization mode used during local quadratic fitting.",
    )
    parser.add_argument(
        "--q-normalization-eps",
        type=float,
        default=1e-12,
        help="Lower bound for q std scales in normalization.",
    )
    parser.add_argument(
        "--selector-mode",
        type=str,
        choices=("linear", "quadratic"),
        default="quadratic",
    )

    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--snap-folder", type=str, default="Results/param_snaps")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="LocalQuadraticSweep/local_hqprom_sweep_auto",
        help="Where candidate folders and summary.csv are written.",
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
    parser.add_argument(
        "--only-reference-case",
        action="store_true",
        help="Run only the reference candidate zeta=1.2, alpha=1e4.",
    )
    parser.add_argument(
        "--traditional-paths",
        action="store_true",
        help=(
            "Use traditional shared files: "
            "LocalQuadratic/local_qm_data.npz and Results/local_hqprom_ecsw_weights.npy."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be >= 2.")
    if args.phi < 0.0:
        raise ValueError("--phi must be >= 0.")

    if args.only_reference_case:
        zetas = [1.2]
        ridge_alphas = [1e4]
    else:
        zetas = parse_float_list(args.zetas)
        ridge_alphas = parse_float_list(args.ridge_alphas)
    points = parse_points(args.points)

    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    ensure_csv(csv_path)

    done = set() if args.force else load_done_tags(csv_path)

    mod = runpy.run_path(str(REPO_DIR / "LocalQuadratic" / "stage1_local_qm_offline.py"))
    build_global_snapshot_matrix = mod["build_global_snapshot_matrix"]
    cluster_snapshots_kmeans = mod["cluster_snapshots_kmeans"]
    cluster_snapshots_fuzzy = mod["cluster_snapshots_fuzzy"]
    build_overlapping_clusters = mod["build_overlapping_clusters"]
    build_local_qm_bases = mod["build_local_qm_bases"]
    precompute_quantities = mod["precompute_quantities"]
    as_object_array = mod["as_object_array"]

    print("[sweep] Building global snapshot matrix once...", flush=True)
    S = build_global_snapshot_matrix(
        args.dt,
        args.num_steps,
        snap_folder=args.snap_folder,
        param_list=get_snapshot_params(),
    )

    print(f"[sweep] Clustering once with method={args.clustering_method}...", flush=True)
    if args.clustering_method == "kmeans":
        labels, centers = cluster_snapshots_kmeans(S, args.n_clusters)
    else:
        labels, centers = cluster_snapshots_fuzzy(S, args.n_clusters)

    if args.phi > 0.0:
        cluster_indices = build_overlapping_clusters(S, labels, centers, args.phi)
    else:
        cluster_indices = [np.where(labels == k)[0] for k in range(args.n_clusters)]

    n_total = len(zetas) * len(ridge_alphas)
    idx = 0
    for zeta_qua in zetas:
        for alpha_ridge in ridge_alphas:
            idx += 1
            tag = format_tag(zeta_qua, alpha_ridge, args.q_normalization_mode)
            if tag in done:
                print(f"[skip {idx}/{n_total}] {tag}", flush=True)
                continue

            out = root / tag
            out.mkdir(parents=True, exist_ok=True)
            print(f"\\n[candidate {idx}/{n_total}] {tag}", flush=True)

            try:
                (
                    u0_list,
                    uc_list,
                    uref_list,
                    V_list,
                    H_list,
                    n_trad_list,
                    n_list,
                    _,
                    _,
                ) = build_local_qm_bases(
                    S,
                    cluster_indices,
                    centers,
                    pod_tol=float(args.pod_tol),
                    zeta_qua=float(zeta_qua),
                    alpha_ridge=float(alpha_ridge),
                    pod_method=args.pod_method,
                    q_normalization_mode=args.q_normalization_mode,
                    q_normalization_eps=float(args.q_normalization_eps),
                )

                d_const, g_list, m_list, T_list, h_list = precompute_quantities(
                    u0_list, uc_list, V_list, H_list
                )

                if args.traditional_paths:
                    model_file = REPO_DIR / "LocalQuadratic" / "local_qm_data.npz"
                else:
                    model_file = out / "local_qm_data.npz"
                np.savez(
                    model_file,
                    S_shape=S.shape,
                    u0_list=as_object_array(u0_list),
                    uc_list=as_object_array(uc_list),
                    uref_list=as_object_array(uref_list),
                    V_list=as_object_array(V_list),
                    H_list=as_object_array(H_list),
                    n_trad_list=np.asarray(n_trad_list, dtype=int),
                    n_list=np.asarray(n_list, dtype=int),
                    cluster_indices=as_object_array(cluster_indices),
                    d_const=d_const,
                    g_list=g_list,
                    m_list=m_list,
                    T_list=T_list,
                    h_list=h_list,
                )

                if args.traditional_paths:
                    weights_file = REPO_DIR / "Results" / "local_hqprom_ecsw_weights.npy"
                else:
                    weights_file = out / "local_hqprom_ecsw_weights.npy"
                errs = []
                for i, (mu1, mu2) in enumerate(points):
                    _, err = run_local_hqprom.main(
                        mu1=mu1,
                        mu2=mu2,
                        local_model_file=str(model_file),
                        weights_file=str(weights_file),
                        compute_ecsw=(i == 0),
                        selector_mode=args.selector_mode,
                        dt=args.dt,
                        num_steps=args.num_steps,
                    )
                    errs.append(float(err))

                    # Archive per-point text summary into candidate folder so
                    # later candidates do not overwrite evidence in Results/.
                    src_summary = REPO_DIR / "Results" / (
                        f"local_hqprom_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt"
                    )
                    if src_summary.exists():
                        dst_summary = out / src_summary.name
                        shutil.copy2(src_summary, dst_summary)

                max_err = float(max(errs))
                row = [
                    tag,
                    float(zeta_qua),
                    float(alpha_ridge),
                    int(np.min(n_list)),
                    int(np.max(n_list)),
                    float(np.mean(n_list)),
                    errs[0] if len(errs) > 0 else np.nan,
                    errs[1] if len(errs) > 1 else np.nan,
                    errs[2] if len(errs) > 2 else np.nan,
                    max_err,
                    "ok",
                    f"q_norm={args.q_normalization_mode}",
                ]
                append_row(csv_path, row)
                print(
                    f"[ok] {tag} -> max_err={max_err:.6f}% | "
                    f"n_qm[min/max/avg]={row[3]}/{row[4]}/{row[5]:.2f}",
                    flush=True,
                )

            except Exception as exc:
                row = [
                    tag,
                    float(zeta_qua),
                    float(alpha_ridge),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                ]
                append_row(csv_path, row)
                print(f"[fail] {tag}: {type(exc).__name__}: {exc}", flush=True)
                if args.traceback:
                    traceback.print_exc()

    print(f"\\nDone. Summary: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
