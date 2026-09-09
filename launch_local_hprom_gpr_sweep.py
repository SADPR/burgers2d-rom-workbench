#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch a local POD-GPR HPROM sweep over n_primary.

Stages 1-3 (clustering, local POD, projection to q) depend only on
(n_clusters, eps2_pod) and are therefore built ONCE per invocation into a
shared `<root>/base/` folder. Stage 4 (GPR training) and the online solve
depend on n_primary and are re-run per candidate into `<root>/nprimary<N>/`.

This mirrors the structure of the manual nc=3 sweep that produced
LocalPOD-GPRSweep/nc3_nprimary{8,9,10}, where every stage-4 summary points
back to a single shared nc3_baseline/q_per_cluster.npz.

Counterpart of launch_local_hqprom_sweep.py (knob: zeta_qua/alpha_ridge) and
launch_local_hprom_sweep.py (knob: pod_tol). Here the knob is n_primary: the
number of coordinates the LSPG solve keeps, with the GPR predicting the
remaining ones of each cluster's retained rank.
"""

import argparse
import csv
import runpy
import shutil
import sys
import traceback
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from burgers.config import DT, NUM_STEPS

import run_local_hprom_gpr

CSV_FIELDS = (
    "n_primary",
    "retained_ranks",
    "err_4.56_0.019",
    "err_4.75_0.020",
    "err_5.19_0.026",
    "max_err_percent",
    "num_nonzero_weights",
    "status",
    "note",
)


def parse_int_list(text):
    values = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok:
            values.append(int(tok))
    if not values:
        raise ValueError("Empty n_primary list.")
    return values


def parse_points(text):
    points = []
    for group in str(text).split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [p.strip() for p in group.split(",")]
        if len(parts) != 2:
            raise ValueError("Invalid point group %r; expected mu1,mu2." % group)
        points.append((float(parts[0]), float(parts[1])))
    if not points:
        raise ValueError("No evaluation points given.")
    return points


def ensure_csv(path):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_FIELDS)


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status", "")).strip() == "ok":
                n = str(row.get("n_primary", "")).strip()
                if n:
                    done.add(int(float(n)))
    return done


def append_row(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def read_key(path, key):
    if not Path(path).exists():
        return None
    for line in open(path):
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    return None


def build_parser():
    p = argparse.ArgumentParser(
        description="Sweep local POD-GPR HPROM over n_primary candidates.",
    )
    p.add_argument(
        "--n-primaries",
        type=str,
        default="6,7,8,9",
        help="Comma-separated n_primary values (the accuracy knob).",
    )
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument(
        "--points",
        type=str,
        default="4.56,0.019;4.75,0.020;5.19,0.026",
    )
    p.add_argument(
        "--eps2-pod",
        type=float,
        default=1e-4,
        help="Local POD tolerance for stage 2 (sets the retained rank pool).",
    )
    p.add_argument(
        "--clustering-method",
        type=str,
        choices=("kmeans", "fuzzy"),
        default="kmeans",
    )
    p.add_argument("--phi", type=float, default=0.1)
    p.add_argument("--time-subsample", type=int, default=1)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--num-steps", type=int, default=NUM_STEPS)
    p.add_argument("--snap-folder", type=str, default="Results/param_snaps")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--root-dir",
        type=str,
        default="LocalPOD-GPRSweep/auto",
    )
    p.add_argument(
        "--reuse-base",
        type=str,
        default=None,
        help=(
            "Reuse an existing stages 1-3 folder (must contain "
            "q_per_cluster.npz), e.g. LocalPOD-GPRSweep/nc3_baseline. "
            "Skips stages 1-3 entirely."
        ),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--traceback", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be >= 2.")
    n_primaries = parse_int_list(args.n_primaries)
    points = parse_points(args.points)

    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "summary.csv"
    ensure_csv(csv_path)
    done = set() if args.force else load_done(csv_path)

    gpr_dir = REPO_DIR / "LocalPOD-GPR"
    stage1 = runpy.run_path(str(gpr_dir / "stage1_cluster_snapshots_u.py"))
    stage2 = runpy.run_path(str(gpr_dir / "stage2_local_pod_per_cluster.py"))
    stage3 = runpy.run_path(str(gpr_dir / "stage3_local_project_to_q.py"))
    stage4 = runpy.run_path(str(gpr_dir / "stage4_local_pod_gpr_training.py"))

    # ---- stages 1-3: shared across all n_primary candidates -------------
    if args.reuse_base:
        base = Path(args.reuse_base)
        q_file = base / "q_per_cluster.npz"
        if not q_file.exists():
            raise FileNotFoundError("No q_per_cluster.npz under %s" % base)
        print("[sweep] Reusing stages 1-3 from %s" % base, flush=True)
    else:
        base = root / "base"
        base.mkdir(parents=True, exist_ok=True)
        clusters_file = base / "clusters.npz"
        pod_file = base / "pod_per_cluster.npz"
        q_file = base / "q_per_cluster.npz"

        print("[sweep] stage1: clustering (n_clusters=%d)..." % args.n_clusters, flush=True)
        stage1["main"](
            n_clusters=args.n_clusters,
            dt=args.dt,
            num_steps=args.num_steps,
            snap_folder=args.snap_folder,
            clustering_method=args.clustering_method,
            phi=args.phi,
            time_subsample=args.time_subsample,
            output_file=str(clusters_file),
            summary_file=str(base / "stage1_summary.txt"),
            cluster_plot_file=str(base / "stage1_sizes.png"),
        )

        print("[sweep] stage2: local POD (eps2_pod=%g)..." % args.eps2_pod, flush=True)
        stage2["main"](
            clusters_file=str(clusters_file),
            snap_folder=args.snap_folder,
            eps2_pod=args.eps2_pod,
            output_file=str(pod_file),
            plot_dir=str(base / "stage2_plots"),
            rank_plot_file=str(base / "stage2_ranks.png"),
            summary_file=str(base / "stage2_summary.txt"),
        )

        print("[sweep] stage3: projection to q...", flush=True)
        stage3["main"](
            pod_file=str(pod_file),
            snap_folder=args.snap_folder,
            output_file=str(q_file),
            counts_plot_file=str(base / "stage3_counts.png"),
            summary_file=str(base / "stage3_summary.txt"),
        )

    ranks = read_key(base / "stage2_summary.txt", "retained_ranks") or "--"
    print("[sweep] retained_ranks = %s" % ranks, flush=True)

    # ---- stage 4 + online: per n_primary --------------------------------
    for idx, n_primary in enumerate(n_primaries, start=1):
        if n_primary in done:
            print("[skip %d/%d] n_primary=%d" % (idx, len(n_primaries), n_primary), flush=True)
            continue

        out = root / ("nprimary%d" % n_primary)
        out.mkdir(parents=True, exist_ok=True)
        print("\n[candidate %d/%d] n_primary=%d" % (idx, len(n_primaries), n_primary), flush=True)

        try:
            stage4["main"](
                q_file=str(q_file),
                model_pkl=str(out / "models.pkl"),
                switching_npz=str(out / "switching.npz"),
                all_in_one_npz=str(out / "all_offline.npz"),
                summary_file=str(out / "stage4_summary.txt"),
                train_error_plot_file=str(out / "stage4_train_err.png"),
                n_primary=int(n_primary),
                random_seed=args.random_seed,
            )

            weights_file = out / "ecsw_weights.npy"
            errs = []
            for i, (mu1, mu2) in enumerate(points):
                _, err = run_local_hprom_gpr.main(
                    mu1=mu1,
                    mu2=mu2,
                    local_model_file=str(out / "all_offline.npz"),
                    weights_file=str(weights_file),
                    compute_ecsw=(i == 0),
                    dt=args.dt,
                    num_steps=args.num_steps,
                )
                errs.append(float(err))

                src = REPO_DIR / "Results" / (
                    "local_hprom_gpr_summary_mu1_%.2f_mu2_%.3f.txt" % (mu1, mu2)
                )
                if src.exists():
                    shutil.copy2(src, out / src.name)

            ne = read_key(
                out / ("local_hprom_gpr_summary_mu1_%.2f_mu2_%.3f.txt" % points[0]),
                "num_nonzero_weights",
            )
            append_row(csv_path, [
                int(n_primary), ranks,
                errs[0] if len(errs) > 0 else "",
                errs[1] if len(errs) > 1 else "",
                errs[2] if len(errs) > 2 else "",
                float(max(errs)), ne or "",
                "ok", "nc=%d eps2=%g" % (args.n_clusters, args.eps2_pod),
            ])
            print("[ok] n_primary=%d -> max_err=%.6f%% (Ne=%s)"
                  % (n_primary, max(errs), ne), flush=True)

        except Exception as exc:
            append_row(csv_path, [
                int(n_primary), ranks, "", "", "", "", "",
                "fail", "%s: %s" % (type(exc).__name__, exc),
            ])
            print("[fail] n_primary=%d: %s: %s" % (n_primary, type(exc).__name__, exc), flush=True)
            if args.traceback:
                traceback.print_exc()

    print("\nDone. Summary: %s" % csv_path, flush=True)


if __name__ == "__main__":
    main()
