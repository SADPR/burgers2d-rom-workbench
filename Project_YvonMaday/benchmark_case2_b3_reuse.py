#!/usr/bin/env python3
"""Compare original B3 and predictor reuse on the same frozen rule and CPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path,
                        default=ROOT / "Project_YvonMaday/Results_Paper/euclidean_hprom_main")
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    output = args.output.resolve()
    # Set before importing numerical libraries. The active Python is retained.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CASE2_CAMPAIGN_ROOT"] = str(campaign)
    os.environ["CASE2_MODEL_PATH"] = str(
        campaign / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt")
    os.environ["CASE2_BASIS_DIR"] = str(
        ROOT / "Project_YvonMaday/Results_Paper/MetricStudy/euclidean/Stage1")

    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from Project_YvonMaday import run_case2_hyperreduction as runner

    if socket.gethostname().split(".")[0].startswith("login"):
        parser.error("Run on an allocated compute node.")
    if output.exists():
        parser.error(f"Output already exists; choose a fresh comparison folder: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    rule = campaign / "Stage4/case2_b3/positive_fit4096/weights.npy"
    rows = []
    with threadpool_limits(1):
        model, v, uref, fingerprints = runner.load_inputs()
        mesh = runner.SampledBurgers(runner.GRID_X, runner.GRID_Y, runner.DT, np.load(rule))
        if len(mesh.samples) != 3532:
            raise ValueError("This experiment requires the frozen 3532-cell rule.")
        for point in range(4):
            checkpoint = campaign / (
                f"Stage4/case2_b3/reporting_positive_fit4096_hprom3_p{point}_steps500")
            saved = json.loads((checkpoint / "summary.json").read_text())["config"]
            if (saved["inputs"] != fingerprints
                    or saved["weights_sha256"] != runner.sha256(rule)
                    or saved["steps"] != 500 or saved["method"] != "hprom3"
                    or saved["initialization"] != "known_linear"):
                raise ValueError(f"Frozen checkpoint inputs or method differ: {checkpoint}")
            if not (checkpoint / "qN.npy").is_file():
                raise FileNotFoundError(checkpoint / "qN.npy")
        config = dict(
            threads=1, hostname=socket.gethostname(), python=sys.executable,
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            threadpools=threadpool_info(), inputs=fingerprints,
            weights_sha256=runner.sha256(rule), sampled_cells=len(mesh.samples),
            steps=500, rank=3, repeats=1, full_trajectory_warmups=0,
            implementation={str(path.relative_to(ROOT)): runner.sha256(path) for path in (
                Path(__file__), ROOT / "burgers/case2_hyperreduction.py",
                ROOT / "burgers/case2_residual_correction.py", Path(runner.__file__))},
        )
        runner.atomic_json(output / "config.json", config)
        for point, mu in enumerate(runner.REPORTING):
            results = {}
            # Alternate order to avoid always giving one variant the first run.
            order = (False, True) if point % 2 == 0 else (True, False)
            for reuse in order:
                name = "reuse" if reuse else "original"
                profile = {}
                print(f"Starting point={point} variant={name}", flush=True)
                q, stats = runner.hyper_rollout(
                    model, v, uref, mu, mesh, 3, 500,
                    reuse_predictor=reuse, profile=profile)
                results[name] = (q, stats, profile)
                np.save(output / f"p{point}_{name}_qN.npy", q)
                runner.atomic_json(output / f"p{point}_{name}.json",
                                   dict(**stats, profile=profile))

            old, old_stats, old_profile = results["original"]
            new, new_stats, new_profile = results["reuse"]
            relative_delta = float(np.linalg.norm(new - old) / np.linalg.norm(old))
            maximum_delta = float(np.max(np.abs(new - old)))
            equivalent = (
                relative_delta <= 1e-9 and maximum_delta <= 1e-6
                and old_stats["iterations"] == new_stats["iterations"]
                and old_stats["min_rank"] == new_stats["min_rank"]
                and old_stats["max_rank"] == new_stats["max_rank"])
            # Accuracy scoring and file I/O are outside both online timers.
            teacher = runner.reference_path(mu, "reporting")
            old_metrics = runner.score(old, v, uref, mu, teacher)
            new_metrics = runner.score(new, v, uref, mu, teacher)
            checkpoint = campaign / (
                f"Stage4/case2_b3/reporting_positive_fit4096_hprom3_p{point}_steps500/qN.npy")
            checkpoint_q = np.load(checkpoint)
            row = dict(
                point=point, mu1=mu[0], mu2=mu[1], equivalent=equivalent,
                original_seconds=old_stats["online_seconds"],
                reuse_seconds=new_stats["online_seconds"],
                speedup=old_stats["online_seconds"] / new_stats["online_seconds"],
                original_iterations=old_stats["iterations"],
                reuse_iterations=new_stats["iterations"],
                relative_q_difference=relative_delta, max_abs_q_difference=maximum_delta,
                original_vs_checkpoint_relative_q=float(
                    np.linalg.norm(old - checkpoint_q) / np.linalg.norm(checkpoint_q)),
                original_state_error_percent=old_metrics.get("state_error_percent"),
                reuse_state_error_percent=new_metrics.get("state_error_percent"),
                **{f"original_{key}": value for key, value in old_profile.items()},
                **{f"reuse_{key}": value for key, value in new_profile.items()},
            )
            rows.append(row)
            runner.atomic_json(output / "comparison.json", rows)
            print(json.dumps(row, indent=2), flush=True)
            if not equivalent:
                raise RuntimeError("Equivalence check failed; inspect comparison.json.")

    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original_mean = float(np.mean([row["original_seconds"] for row in rows[:3]]))
    reuse_mean = float(np.mean([row["reuse_seconds"] for row in rows[:3]]))
    lines = [
        "Paired B3 comparison: 4 points x 2 variants x 1 measurement; 1 thread.",
        "Timings include lightweight phase instrumentation in both variants.",
        "No timing variability estimate from a single measurement per point.",
        f"Original in-domain mean: {original_mean:.6f} s",
        f"Reuse in-domain mean: {reuse_mean:.6f} s",
        f"Speedup: {original_mean / reuse_mean:.6f}",
        f"Time reduction: {100 * (1 - reuse_mean / original_mean):.3f}%",
        "Equivalence: PASS for all four trajectories.",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
