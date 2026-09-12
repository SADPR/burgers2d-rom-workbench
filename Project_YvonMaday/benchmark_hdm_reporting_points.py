#!/usr/bin/env python3
"""Time fresh HDM solves at the three in-domain reporting parameters."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS
from burgers.core import inviscid_burgers_implicit2D


POINTS = (
    ("verification", 4.875, 0.0225),
    ("offgrid1", 4.560, 0.0190),
    ("offgrid2", 5.190, 0.0260),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    with threadpool_limits(args.threads):
        for repeat in range(1, args.repeats + 1):
            for label, mu1, mu2 in POINTS:
                print(f"[hdm-timing] repeat={repeat} point={label} mu=({mu1},{mu2})", flush=True)
                start = time.perf_counter()
                snapshots = inviscid_burgers_implicit2D(
                    GRID_X, GRID_Y, np.asarray(W0).copy(), DT, NUM_STEPS,
                    (mu1, mu2), max_its=100, relnorm_cutoff=1e-12,
                )
                elapsed = time.perf_counter() - start
                if snapshots.shape != (W0.size, NUM_STEPS + 1) or not np.isfinite(snapshots[:, -1]).all():
                    raise RuntimeError(f"Invalid HDM trajectory for {label}: {snapshots.shape}")
                rows.append({"repeat": repeat, "point": label, "mu1": mu1, "mu2": mu2,
                             "online_seconds": elapsed})
                del snapshots
                print(f"[hdm-timing] {label}: {elapsed:.6f} s", flush=True)

    csv_path = args.output / "hdm_timing.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    values = np.asarray([row["online_seconds"] for row in rows])
    summary = {
        "method": "HDM",
        "operation": "fresh full-order solve; excludes file I/O",
        "threads": args.threads,
        "repeats": args.repeats,
        "mean_in_domain_seconds": float(values.mean()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    (args.output / "hdm_timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
