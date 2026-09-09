#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch a local (affine) HPROM sweep without editing
LocalPOD/stage1_local_pod_offline_cheap.py.

This script builds S and clusters once, then sweeps pod_tol, builds candidate
local POD models, runs run_local_hprom on test points, and writes a
resume-safe CSV summary.

It is the linear counterpart of launch_local_hqprom_sweep.py: there the
accuracy knob is (zeta_qua, alpha_ridge) for the quadratic term, here it is
simply pod_tol, which sets the retained modes per cluster.
"""

import argparse
import csv
import runpy
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from burgers.config import DT, NUM_STEPS
from burgers.core import get_snapshot_params

import run_local_hprom

CSV_FIELDS = (
    "pod_tol",
    "min_modes",
    "max_modes",
    "avg_modes",
    "err_4.56_0.019",
    "err_4.75_0.020",
    "err_5.19_0.026",
    "max_err_percent",
    "status",
    "note",
)


def parse_float_list(text):
    values = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok:
            values.append(float(tok))
    if not values:
        raise ValueError("Empty numeric list.")
    return values


def parse_points(text):
    points = []
    for group in str(text).split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [p.strip() for p in group.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid point group {group!r}; expected mu1,mu2.")
        points.append((float(parts[0]), float(parts[1])))
    if not points:
        raise ValueError("No evaluation points given.")
    return points


def format_tag(pod_tol):
    return f"tol{pod_tol:.1e}"


def ensure_csv(csv_path):
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_FIELDS)


def load_done_tags(csv_path):
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status", "")).strip() == "ok":
                tol = str(row.get("pod_tol", "")).strip()
                if tol:
                    done.add(format_tag(float(tol)))
    return done


def append_row(csv_path, row):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep local (affine) HPROM over pod_tol candidates.",
    )
    parser.add_argument(
        "--pod-tols",
        type=str,
        default="3e-3,2e-3,1.5e-3,1e-3,7e-4,5e-4",
        help="Comma-separated pod_tol values (the accuracy knob).",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="4.56,0.019;4.75,0.020;5.19,0.026",
        help="Semicolon-separated mu points: mu1,mu2;mu1,mu2",
    )
    parser.add_argument("--n-clusters", type=int, default=3)
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
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--snap-folder", type=str, default="Results/param_snaps")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="LocalPODSweep/auto",
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
    return parser


def main():
    args = build_parser().parse_args()

    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be >= 2.")
    if args.phi < 0.0:
        raise ValueError("--phi must be >= 0.")

    pod_tols = parse_float_list(args.pod_tols)
    points = parse_points(args.points)

    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    ensure_csv(csv_path)

    done = set() if args.force else load_done_tags(csv_path)

    mod = runpy.run_path(
        str(REPO_DIR / "LocalPOD" / "stage1_local_pod_offline_cheap.py")
    )
    build_global_snapshot_matrix = mod["build_global_snapshot_matrix"]
    cluster_snapshots_kmeans = mod["cluster_snapshots_kmeans"]
    cluster_snapshots_fuzzy = mod["cluster_snapshots_fuzzy"]
    build_overlapping_clusters = mod["build_overlapping_clusters"]
    build_local_pod_bases = mod["build_local_pod_bases"]
    precompute_quantities = mod["precompute_quantities"]
    as_object_array = mod["as_object_array"]

    print("[sweep] Building global snapshot matrix once...", flush=True)
    S = build_global_snapshot_matrix(
        args.dt,
        args.num_steps,
        snap_folder=args.snap_folder,
        param_list=get_snapshot_params(),
    )

    print(
        f"[sweep] Clustering once with method={args.clustering_method}, "
        f"n_clusters={args.n_clusters}...",
        flush=True,
    )
    if args.clustering_method == "kmeans":
        labels, centers = cluster_snapshots_kmeans(S, args.n_clusters)
    else:
        labels, centers = cluster_snapshots_fuzzy(S, args.n_clusters)

    if args.phi > 0.0:
        cluster_indices = build_overlapping_clusters(S, labels, centers, args.phi)
    else:
        cluster_indices = [
            np.where(labels == k)[0] for k in range(args.n_clusters)
        ]

    n_total = len(pod_tols)
    for idx, pod_tol in enumerate(pod_tols, start=1):
        tag = format_tag(pod_tol)
        if tag in done:
            print(f"[skip {idx}/{n_total}] {tag}", flush=True)
            continue

        out = root / tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[candidate {idx}/{n_total}] {tag}", flush=True)

        try:
            u0_list, uc_list, V_list = build_local_pod_bases(
                S,
                cluster_indices,
                centers,
                pod_tol=float(pod_tol),
                pod_method=args.pod_method,
            )

            d_const, g_list, T_list, h_list = precompute_quantities(
                u0_list, uc_list, V_list
            )

            n_list = [int(V.shape[1]) for V in V_list]

            model_file = out / "local_pod_data.npz"
            np.savez(
                model_file,
                S_shape=S.shape,
                u0_list=as_object_array(u0_list),
                uc_list=as_object_array(uc_list),
                V_list=as_object_array(V_list),
                cluster_indices=as_object_array(cluster_indices),
                d_const=d_const,
                g_list=g_list,
                T_list=T_list,
                h_list=h_list,
            )

            weights_file = out / "local_hprom_ecsw_weights.npy"
            errs = []
            for i, (mu1, mu2) in enumerate(points):
                _, err = run_local_hprom.main(
                    mu1=mu1,
                    mu2=mu2,
                    local_model_file=str(model_file),
                    weights_file=str(weights_file),
                    compute_ecsw=(i == 0),
                    dt=args.dt,
                    num_steps=args.num_steps,
                )
                errs.append(float(err))

                # Archive the per-point text summary into the candidate folder
                # so later candidates do not overwrite evidence in Results/.
                src_summary = REPO_DIR / "Results" / (
                    f"local_hprom_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt"
                )
                if src_summary.exists():
                    shutil.copy2(src_summary, out / src_summary.name)

            max_err = float(max(errs))
            append_row(
                csv_path,
                [
                    float(pod_tol),
                    int(np.min(n_list)),
                    int(np.max(n_list)),
                    float(np.mean(n_list)),
                    errs[0] if len(errs) > 0 else np.nan,
                    errs[1] if len(errs) > 1 else np.nan,
                    errs[2] if len(errs) > 2 else np.nan,
                    max_err,
                    "ok",
                    f"nc={args.n_clusters}",
                ],
            )
            print(
                f"[ok] {tag} -> max_err={max_err:.6f}% | "
                f"modes[min/max/avg]={np.min(n_list)}/{np.max(n_list)}/"
                f"{np.mean(n_list):.2f}",
                flush=True,
            )

        except Exception as exc:
            append_row(
                csv_path,
                [
                    float(pod_tol), "", "", "", "", "", "", "",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                ],
            )
            print(f"[fail] {tag}: {type(exc).__name__}: {exc}", flush=True)
            if args.traceback:
                traceback.print_exc()

    print(f"\nDone. Summary: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
