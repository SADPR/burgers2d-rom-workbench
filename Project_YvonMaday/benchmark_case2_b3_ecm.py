#!/usr/bin/env python3
"""Paired deployment of two frozen B3 rules on one allocated Sherlock node."""

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    host = socket.gethostname()
    if host.split(".")[0].startswith("login") and not args.check_only:
        parser.error("Run on an allocated compute node")
    bundle, output = args.bundle.resolve(), args.output.resolve()
    paper = ROOT / "Project_YvonMaday/Results_Paper"
    campaign = paper / "euclidean_hprom_main"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CASE2_CAMPAIGN_ROOT"] = str(campaign)
    os.environ["CASE2_MODEL_PATH"] = str(campaign / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt")
    os.environ["CASE2_BASIS_DIR"] = str(paper / "MetricStudy/euclidean/Stage1")
    os.environ["CASE2_TRAINING_DATASET"] = str(campaign / "Stage2/prom_coeff_dataset_ntot151")
    os.environ["CASE2_REFERENCE_CAMPAIGN_ROOT"] = str(campaign)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from Project_YvonMaday import run_case2_hyperreduction as runner
    from Project_YvonMaday.study_case2_b3_cubature_budget import rollout_point

    manifest = json.loads((bundle / "manifest.json").read_text())
    if not manifest["accepted"] or manifest["selection_uses_reporting_points"]:
        raise ValueError("Bundle does not contain a validation-selected rule")
    for relative, expected in manifest["files"].items():
        if runner.sha256(bundle / relative) != expected:
            raise ValueError(f"Bundle file changed: {relative}")
    for relative, expected in manifest["solver_sources"].items():
        if runner.sha256(ROOT / relative) != expected:
            raise ValueError(f"Solver differs from the local study: {relative}")
    torch.set_num_threads(1)
    with threadpool_limits(1):
        inputs = runner.load_inputs()
        if inputs[3] != manifest["inputs"]:
            raise ValueError("Model, basis or training metadata differs from the local study")
        for mu in runner.REPORTING:
            runner.reference_path(mu, "reporting")
            filename = f"mu1_{mu[0]}+mu2_{mu[1]}.npy"
            if not any((directory / filename).is_file() for directory in
                       (ROOT / "Results/param_snaps", ROOT / "Project_YvonMaday/Results/param_snaps")):
                raise FileNotFoundError(f"Missing HDM scoring snapshots: {filename}")
        if args.check_only:
            print("Bundle hashes, solver sources, ANN, basis, references and HDM files: OK")
            return
        rules = {name: bundle / f"{name}_weights.npy" for name in ("reference2048", "ecm")}
        counts = {name: int(np.count_nonzero(np.load(path))) for name, path in rules.items()}
        protocol = dict(hostname=host, threads=1, measurements_per_point=1,
                        full_trajectory_warmups=0, initialization="known_linear",
                        reuse_predictor=True, inputs=inputs[3],
                        bundle_sha256=runner.sha256(bundle / "manifest.json"),
                        sampled_cells=counts,
                        references={str(runner.reference_path(mu, "reporting").relative_to(ROOT)):
                                    runner.sha256(runner.reference_path(mu, "reporting"))
                                    for mu in runner.REPORTING})
        output.mkdir(parents=True, exist_ok=True)
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing benchmark uses a different host, inputs or protocol")
        runner.atomic_json(protocol_path, protocol)
        if not (output / "environment.json").exists():
            runner.atomic_json(output / "environment.json", dict(
                python=sys.executable, numpy=np.__version__, torch=torch.__version__,
                threadpools=threadpool_info(), recorded_unix_time=time.time(),
                cpu_affinity=sorted(os.sched_getaffinity(0))))
        print("Eight trajectories: two frozen B3 rules x four parameters; one thread.", flush=True)
        print("No training, rule fitting, HDM solve or full-trajectory warmup.", flush=True)
        rows = []
        for point in range(4):
            records = {}
            order = ("reference2048", "ecm") if point % 2 == 0 else ("ecm", "reference2048")
            for name in order:
                result_path = output / name / f"reporting_p{point}.json"
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if previous["weights_sha256"] != runner.sha256(rules[name]):
                        raise ValueError("Cached trajectory has different weights")
                print(f"Starting {name}, point={point}", flush=True)
                records[name] = rollout_point(output / name, "reporting", point,
                                              rules[name], inputs, runner)
            row = dict(point=point, mu1=runner.REPORTING[point][0], mu2=runner.REPORTING[point][1])
            for name, record in records.items():
                for metric in ("state_error_percent", "coefficient_error_percent",
                               "online_seconds", "iterations"):
                    if metric not in record or not np.isfinite(record[metric]):
                        raise ValueError(f"Missing or invalid {metric}; check downloaded HDM snapshots")
                    row[f"{name}_{metric}"] = record[metric]
            rows.append(row)
            runner.atomic_json(output / "comparison.json", rows)
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        old = float(np.mean([r["reference2048_online_seconds"] for r in rows[:3]]))
        new = float(np.mean([r["ecm_online_seconds"] for r in rows[:3]]))
        summary = (f"Host: {host}; one numerical thread.\n"
                   "One measurement per point; no timing variability estimate.\n"
                   f"Retained cells: {counts['reference2048']} -> {counts['ecm']}\n"
                   f"Reference B3 in-domain mean: {old:.6f} s\n"
                   f"ECM B3 in-domain mean: {new:.6f} s\n"
                   f"Reference/ECM time ratio: {old/new:.6f}\n"
                   "This ratio is not an HDM speed-up.\n")
        (output / "summary.txt").write_text(summary)
        print(summary, flush=True)


if __name__ == "__main__":
    main()
