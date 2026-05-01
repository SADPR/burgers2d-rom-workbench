#!/usr/bin/env python3
"""Run all HPROM-family online tests (no training / no ECSW recomputation).

This launcher runs:
  - HPROM (linear POD)
  - HQPROM (global quadratic manifold)
  - HPROM-GPR (global POD-GPR manifold)
  - HPROM-DL (global POD-DL manifold)
  - Local HPROM
  - Local HQPROM
  - Local HPROM-GPR

for a list of test points, with compute_ecsw=False for every method.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import traceback
from pathlib import Path


DEFAULT_POINTS = "4.56,0.019;4.75,0.020;5.19,0.026"


def parse_points(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid point '{chunk}'. Use format: mu1,mu2;mu1,mu2;..."
            )
        points.append((float(parts[0]), float(parts[1])))
    if not points:
        raise ValueError("No test points provided.")
    return points


def parse_method_filter(text: str | None) -> set[str] | None:
    if text is None:
        return None
    vals = {tok.strip().lower() for tok in text.split(",") if tok.strip()}
    return vals if vals else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all HPROM-family online tests without ECSW recomputation.",
    )
    p.add_argument("--points", default=DEFAULT_POINTS)
    p.add_argument(
        "--methods",
        default=None,
        help=(
            "Optional comma-separated subset of method keys. "
            "Keys: hprom,hqprom,hprom_gpr,hprom_dl,local_hprom,local_hqprom,local_hprom_gpr"
        ),
    )
    p.add_argument("--results-dir", default="Results")
    p.add_argument("--csv-out", default="Results/hprom_suite_no_training_runs.csv")

    p.add_argument("--pod-dir", default="POD")
    p.add_argument("--qm-dir", default="Quadratic")
    p.add_argument("--gpr-model-dir", default="POD-GPR/pod_gpr_model")
    p.add_argument("--dl-model-dir", default="POD-DL/pod_dl_model")
    p.add_argument("--local-pod-model", default="LocalPOD/local_pod_data.npz")
    p.add_argument("--local-qm-model", default="LocalQuadratic/local_qm_data.npz")
    p.add_argument(
        "--local-gpr-model", default="LocalPOD-GPR/local_pod_gpr_all_offline.npz"
    )

    p.add_argument("--hqprom-weights", default="Results/hqprom_ecsw_weights.npy")
    p.add_argument(
        "--hprom-gpr-weights", default="POD-GPR/pod_gpr_model/ecsw_weights_gpr.npy"
    )
    p.add_argument(
        "--hprom-dl-weights", default="POD-DL/pod_dl_model/ecsw_weights_dl.npy"
    )
    p.add_argument("--local-hprom-weights", default="Results/local_hprom_ecsw_weights.npy")
    p.add_argument(
        "--local-hqprom-weights", default="Results/local_hqprom_ecsw_weights.npy"
    )
    p.add_argument(
        "--local-gpr-weights", default="Results/local_hprom_gpr_ecsw_weights.npy"
    )

    p.add_argument(
        "--local-hqprom-selector",
        default="quadratic",
        choices=("linear", "quadratic"),
    )
    p.add_argument(
        "--local-gpr-selector",
        default="nonlinear",
        choices=("linear", "nonlinear"),
    )
    p.add_argument(
        "--local-gpr-jacobian",
        default="auto",
        choices=("auto", "analytic", "forward_fd", "central_fd"),
    )
    p.add_argument(
        "--local-gpr-verbose",
        dest="local_gpr_verbose",
        action="store_true",
        default=True,
        help="Enable verbose local HPROM-GPR online output (default: True).",
    )
    p.add_argument(
        "--local-gpr-quiet",
        dest="local_gpr_verbose",
        action="store_false",
        help="Disable verbose local HPROM-GPR online output.",
    )
    p.add_argument(
        "--dl-solver-threads",
        type=int,
        default=1,
        help="Threads for HPROM-DL reduced solver (default: 1).",
    )
    p.add_argument(
        "--traceback",
        action="store_true",
        help="Print full traceback when a method fails.",
    )
    return p


def _fmt(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "nan"
    return f"{x:.6f}"


def runner_hprom(args, mu1: float, mu2: float):
    import run_hprom

    return run_hprom.main(
        mu1=mu1,
        mu2=mu2,
        compute_ecsw=False,
        pod_dir=args.pod_dir,
        results_dir=args.results_dir,
    )


def runner_hqprom(args, mu1: float, mu2: float):
    import run_hqprom

    return run_hqprom.main(
        mu1=mu1,
        mu2=mu2,
        qm_dir=args.qm_dir,
        weights_file=args.hqprom_weights,
        compute_ecsw=False,
    )


def runner_hprom_gpr(args, mu1: float, mu2: float):
    import run_hprom_gpr

    return run_hprom_gpr.main(
        mu1=mu1,
        mu2=mu2,
        model_dir=args.gpr_model_dir,
        weights_file=args.hprom_gpr_weights,
        compute_ecsw=False,
    )


def runner_hprom_dl(args, mu1: float, mu2: float):
    import run_hprom_dl

    return run_hprom_dl.main(
        mu1=mu1,
        mu2=mu2,
        model_dir=args.dl_model_dir,
        weights_file=args.hprom_dl_weights,
        compute_ecsw=False,
        solver_threads=args.dl_solver_threads,
    )


def runner_local_hprom(args, mu1: float, mu2: float):
    import run_local_hprom

    return run_local_hprom.main(
        mu1=mu1,
        mu2=mu2,
        local_model_file=args.local_pod_model,
        weights_file=args.local_hprom_weights,
        compute_ecsw=False,
    )


def runner_local_hqprom(args, mu1: float, mu2: float):
    import run_local_hqprom

    return run_local_hqprom.main(
        mu1=mu1,
        mu2=mu2,
        local_model_file=args.local_qm_model,
        weights_file=args.local_hqprom_weights,
        compute_ecsw=False,
        selector_mode=args.local_hqprom_selector,
    )


def runner_local_hprom_gpr(args, mu1: float, mu2: float):
    import run_local_hprom_gpr

    return run_local_hprom_gpr.main(
        mu1=mu1,
        mu2=mu2,
        local_model_file=args.local_gpr_model,
        weights_file=args.local_gpr_weights,
        compute_ecsw=False,
        selector_mode=args.local_gpr_selector,
        jacobian_mode=args.local_gpr_jacobian,
        verbose=bool(args.local_gpr_verbose),
    )


def main() -> int:
    args = build_parser().parse_args()
    points = parse_points(args.points)
    method_filter = parse_method_filter(args.methods)

    methods = [
        ("hprom", "HPROM", runner_hprom),
        ("hqprom", "HQPROM", runner_hqprom),
        ("hprom_gpr", "HPROM-GPR", runner_hprom_gpr),
        ("hprom_dl", "HPROM-DL", runner_hprom_dl),
        ("local_hprom", "Local HPROM", runner_local_hprom),
        ("local_hqprom", "Local HQPROM", runner_local_hqprom),
        ("local_hprom_gpr", "Local HPROM-GPR", runner_local_hprom_gpr),
    ]
    if method_filter is not None:
        methods = [m for m in methods if m[0] in method_filter]
        if not methods:
            raise ValueError("No valid methods selected after --methods filter.")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    print("====================================================")
    print(" HPROM SUITE (NO TRAINING / NO ECSW RECOMPUTE)")
    print("====================================================")
    print(f"Points: {points}")
    print(f"Methods: {[m[0] for m in methods]}")

    for key, label, fn in methods:
        print(f"\n[{label}]")
        for mu1, mu2 in points:
            print(f"  -> mu=({mu1:.2f}, {mu2:.3f})", flush=True)
            status = "ok"
            note = ""
            elapsed = math.nan
            rel_err = math.nan
            try:
                elapsed, rel_err = fn(args, mu1, mu2)
                elapsed = float(elapsed)
                rel_err = float(rel_err)
                print(
                    f"     ok: time={_fmt(elapsed)} s, rel_err={_fmt(rel_err)} %",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                status = "fail"
                note = f"{type(exc).__name__}: {exc}"
                print(f"     fail: {note}", flush=True)
                if args.traceback:
                    traceback.print_exc()

            rows.append(
                [
                    key,
                    label,
                    mu1,
                    mu2,
                    elapsed,
                    rel_err,
                    status,
                    note,
                ]
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method_key",
                "method_label",
                "mu1",
                "mu2",
                "time_seconds",
                "relative_error_percent",
                "status",
                "note",
            ]
        )
        w.writerows(rows)

    print(f"\nSaved run log CSV: {csv_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
