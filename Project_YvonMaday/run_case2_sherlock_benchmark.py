#!/usr/bin/env python3
"""Run the selected Euclidean experiment sequentially on one allocated node.

The default is four methods, four parameters, one measured trajectory each,
24 numerical threads, and no full-trajectory warmups: 16 solves in total.
Timing boundaries remain those of the local runs.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Project_YvonMaday"
METHODS = {
    "case2_ann": ("baseline", "ann"),
    "adaptive3_ann": ("krylov3", "ann"),
    "case2_known_initial": ("baseline", "linear"),
    "adaptive3_known_initial": ("krylov3", "linear"),
    "case1_known_initial": None,
    "case3_known_initial": None,
}
DEFAULT_METHODS = ("case2_known_initial", "adaptive3_known_initial",
                   "case1_known_initial", "case3_known_initial")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def command(method, folder, threads):
    common = ["--stage", "reporting", "--point-indices", "0", "1", "2", "3",
              "--threads", str(threads), "--output", str(folder)]
    if METHODS[method] is None:
        return [sys.executable, "-u", str(PROJECT / "run_case2_local_reference_benchmarks.py"),
                "--methods", method, *common]
    variant, initial = METHODS[method]
    return [sys.executable, "-u", str(PROJECT / "run_case2_local_corrections.py"),
            "--methods", variant, "--initialization", initial, *common]


def summarize(output, repetitions, methods=DEFAULT_METHODS, warmups=0):
    import numpy as np

    rows = []
    for method in methods:
        for point in range(4):
            samples = []
            for rep in range(1, repetitions + 1):
                paths = list((output / f"rep{rep:02d}" / method).glob(
                    f"reporting_{point}_*/summary.json"))
                if len(paths) == 1:
                    samples.append(json.loads(paths[0].read_text()))
            if not samples:
                continue
            times = [s["online_seconds"] for s in samples]
            errors = [s["state_error_percent"] for s in samples]
            rows.append(dict(method=method, point=point, repetitions=len(samples),
                             mean_seconds=float(np.mean(times)),
                             std_seconds=float(np.std(times, ddof=1)) if len(times) > 1 else "",
                             min_seconds=min(times), max_seconds=max(times),
                             mean_state_error_percent=float(np.mean(errors)),
                             state_error_range=max(errors)-min(errors)))
    if not rows:
        return
    with (output / "timing_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    text = ["Euclidean Case-2 Sherlock benchmark", "",
            "Times: full 500-step online rollout, including initialization and adaptive directions.",
            "Excludes model/data loading, accuracy scoring and output-array saving.",
            f"Full-trajectory warmups per method/point: {warmups} (excluded from statistics).",
            "Standard deviation: sample standard deviation; blank when only one sample exists.",
            "With one trajectory per point, timing variability is not estimated.",
            "No HDM speed-up claimed: no matched HDM timing is measured by this job.",
            "Points 0,1,2: in-domain; point 3: extrapolation.", "",
            "method | point | repetitions | mean seconds | std seconds | state error (%)"]
    for row in rows:
        text.append(f"{row['method']} | {row['point']} | {row['repetitions']} | "
                    f"{row['mean_seconds']:.6f} | {row['std_seconds']} | "
                    f"{row['mean_state_error_percent']:.9f}")
    text.extend(["", "In-domain mean time (equal weights on points 0,1,2):"])
    for method in methods:
        group = [row for row in rows if row["method"] == method and row["point"] < 3]
        if len(group) == 3 and all(row["repetitions"] == repetitions for row in group):
            text.append(f"{method}: {np.mean([r['mean_seconds'] for r in group]):.6f} s")
    complete = len(rows) == 4*len(methods) and all(r["repetitions"] == repetitions for r in rows)
    text.extend(["", f"Status: {'COMPLETE' if complete else 'PARTIAL'}"])
    (output / "timing_summary.txt").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--warmups", type=int, choices=(0, 1), default=0)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.threads < 1:
        parser.error("repetitions and threads must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must not contain duplicates")
    trajectory_count = (args.repetitions + args.warmups)*len(args.methods)*4
    if args.dry_run:
        for method in args.methods:
            print(" ".join(command(method, args.output / "rep01" / method, args.threads)))
        print(f"{trajectory_count} full trajectories; threads={args.threads}; warmups={args.warmups}")
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this benchmark inside your allocated compute node, not on a login node")

    import numpy as np
    import scipy
    import torch
    from threadpoolctl import threadpool_info
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT))
    from Project_YvonMaday.run_case2_local_corrections import BASIS, MODEL, CAMPAIGN, REPORTING, reference

    required = [BASIS / "basis.npy", BASIS / "u_ref.npy", MODEL,
                CAMPAIGN / "Stage3/models/case1_ann_ntot151_best.pt",
                CAMPAIGN / "Stage3/models/case3_ann_ntot151_best.pt"]
    for mu in REPORTING:
        required.append(reference(mu, "reporting"))
        required.append(PROJECT / f"Results/param_snaps/mu1_{mu[0]}+mu2_{mu[1]}.npy")
    # Hash source files as well as inputs, so a resumed timing campaign cannot
    # silently mix implementations, machines, libraries or numerical data.
    required.extend(sorted((ROOT / "burgers").glob("*.py")))
    required.extend(sorted(PROJECT.glob("*.py")))
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in required}
    provenance = {"hostname": socket.gethostname(), "platform": platform.platform(),
                  "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                  "torch": str(torch.__version__), "threadpools": threadpool_info(),
                  "threads": args.threads, "repetitions": args.repetitions,
                  "methods": args.methods, "warmups": args.warmups,
                  "total_trajectories": trajectory_count,
                  "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "cpu_affinity": sorted(os.sched_getaffinity(0)), "sha256": fingerprints}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "environment.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise RuntimeError("Different job, environment or inputs: use a fresh output directory")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    for name, cmd in [("hardware.txt", ["lscpu"]), ("packages.txt", [sys.executable, "-m", "pip", "freeze"])]:
        with (args.output / name).open("w") as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    print(f"Protocol: {trajectory_count} trajectories, {args.threads} numerical threads, "
          f"{args.repetitions} measured run(s) per point, {args.warmups} warmup(s).", flush=True)
    print(f"CPU affinity permits {len(os.sched_getaffinity(0))} logical CPUs.", flush=True)
    for rep in range(0 if args.warmups else 1, args.repetitions+1):
        methods = list(args.methods)
        random.Random(7300+rep).shuffle(methods)
        for method in methods:
            folder = args.output / ("warmup" if rep == 0 else f"rep{rep:02d}") / method
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Starting {folder.relative_to(args.output)}", flush=True)
            with (folder / "run.log").open("a") as stream:
                subprocess.run(command(method, folder, args.threads), cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            summarize(args.output, args.repetitions, args.methods, args.warmups)
    print(f"Complete: {args.output / 'timing_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
