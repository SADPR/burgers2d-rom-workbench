#!/usr/bin/env python3
"""Time a validation-selected B3 rule against the original rule on one host."""

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
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = ROOT / "Project_YvonMaday/Results_Paper/euclidean_hprom_main"
    study = args.study_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Comparison already exists: {output}")
    if socket.gethostname().split(".")[0].startswith("login"):
        parser.error("Use an allocated compute node.")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CASE2_CAMPAIGN_ROOT"] = str(campaign)
    os.environ["CASE2_MODEL_PATH"] = str(
        campaign / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt")
    os.environ["CASE2_BASIS_DIR"] = str(
        ROOT / "Project_YvonMaday/Results_Paper/MetricStudy/euclidean/Stage1")
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from Project_YvonMaday import run_case2_hyperreduction as runner
    from Project_YvonMaday.study_case2_b3_cubature_budget import rollout_point

    selection = json.loads((study / "selection.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    if selection["uses_reporting_points"] or not selection["smaller_rule_accepted"]:
        raise ValueError("No valid frozen selection.")
    chosen, = [c for c in selection["candidates"]
               if c["draws"] == selection["selected_draws"] and c["accepted"]]
    rule = study / f"draws{chosen['draws']}/weights.npy"
    baseline = campaign / "Stage4/case2_b3/positive_fit4096/weights.npy"
    if (runner.sha256(rule) != chosen["weights_sha256"]
            or runner.sha256(baseline) != protocol["parent_weights_sha256"]):
        raise ValueError("Cubature weights differ from the frozen study.")
    baseline_cells = int(np.count_nonzero(np.load(baseline) > 0))
    if int(np.count_nonzero(np.load(rule) > 0)) != chosen["sampled_cells"]:
        raise ValueError("Selected cell count differs from the frozen study.")

    torch.set_num_threads(1)
    rows = []
    with threadpool_limits(1):
        inputs = runner.load_inputs()
        if inputs[3] != protocol["inputs"]:
            raise ValueError("Basis/model/data metadata differ from the selected study.")
        output.mkdir(parents=True)
        runner.atomic_json(output / "environment.json", dict(
            hostname=socket.gethostname(), python=sys.executable, threads=1,
            threadpools=threadpool_info(), inputs=inputs[3],
            parent_weights_sha256=runner.sha256(baseline),
            selected_weights_sha256=runner.sha256(rule),
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            measurements_per_point=1, full_trajectory_warmups=0,
            reuse_predictor=True, selection=selection,
            implementation={str(p.relative_to(ROOT)): runner.sha256(p) for p in (
                Path(__file__), Path(runner.__file__),
                ROOT / "Project_YvonMaday/study_case2_b3_cubature_budget.py",
                ROOT / "burgers/case2_hyperreduction.py",
                ROOT / "burgers/case2_residual_correction.py")}))
        for point in range(4):
            results = {}
            order = ("baseline", "selected") if point % 2 == 0 else ("selected", "baseline")
            for name in order:
                print(f"Starting {name}, point={point}, threads=1", flush=True)
                results[name] = rollout_point(
                    output / name, "reporting", point,
                    baseline if name == "baseline" else rule, inputs, runner)
            a, b = results["baseline"], results["selected"]
            rows.append(dict(
                point=point, baseline_cells=baseline_cells, selected_cells=chosen["sampled_cells"],
                baseline_state_error_percent=a.get("state_error_percent"),
                selected_state_error_percent=b.get("state_error_percent"),
                baseline_coefficient_error_percent=a["coefficient_error_percent"],
                selected_coefficient_error_percent=b["coefficient_error_percent"],
                baseline_iterations=a["iterations"], selected_iterations=b["iterations"],
                baseline_seconds=a["online_seconds"], selected_seconds=b["online_seconds"]))
            runner.atomic_json(output / "comparison.json", rows)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    a = float(np.mean([r["baseline_seconds"] for r in rows[:3]]))
    b = float(np.mean([r["selected_seconds"] for r in rows[:3]]))
    text = "\n".join([
        "B3 with predictor reuse in both variants; 1 thread.",
        "Four points x two rules x one measurement; no timing variability estimate.",
        f"Retained cells: {baseline_cells} -> {chosen['sampled_cells']}",
        f"Baseline in-domain mean: {a:.6f} s",
        f"Selected in-domain mean: {b:.6f} s",
        f"Speed ratio: {a / b:.6f}",
        f"Time reduction: {100 * (1 - b / a):.3f}%",
    ])
    (output / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
