#!/usr/bin/env python3
"""Create a standalone report from the frozen-rule hyperreduction experiment."""

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/case2-hyper-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Project_YvonMaday.run_case2_hyperreduction import DEFAULT_OUTPUT
from Project_YvonMaday.run_case2_local_corrections import PAPER, sha256


def main():
    root = DEFAULT_OUTPUT
    selection = json.loads((root / "selection.json").read_text())
    rule_path = root / selection["rule"]
    if sha256(rule_path) != selection["weights_sha256"]:
        raise ValueError("The selected rule has changed")
    audit = json.loads((PAPER / "tables/euclidean_prom_only/euclidean_case2_adaptive_audit.json").read_text())
    for filename, expected in audit["fingerprints"].items():
        if sha256(ROOT / filename) != expected:
            raise ValueError(f"Stored PROM reference inputs differ: {filename}")
    source = PAPER / "tables/euclidean_prom_only/euclidean_case2_adaptive_metrics.csv"
    with source.open() as stream:
        previous = list(csv.DictReader(stream))
    records = []
    methods = [("case2_known_initial", "PROM Case 2"), ("adaptive3_known_initial", "PROM Case 2 + B3")]
    for key, label in methods:
        for row in previous:
            if row["method"] == key:
                records.append(dict(model=label, point=int(row["point"]),
                                    state_error_percent=float(row["state_error_percent"]),
                                    coefficient_error_percent=float(row["coefficient_error_percent"]),
                                    local_online_seconds=None))
    for method, label in [("hprom0", "HPROM Case 2"), ("hprom3", "HPROM Case 2 + B3")]:
        for point in range(4):
            folder = root / f"reporting_positive_fit4096_{method}_p{point}_steps500"
            data = json.loads((folder / "summary.json").read_text())
            if data["config"]["weights_sha256"] != selection["weights_sha256"]:
                raise ValueError("Reporting run used a different rule")
            records.append(dict(model=label, point=point, state_error_percent=data["state_error_percent"],
                                coefficient_error_percent=data["coefficient_error_percent"],
                                local_online_seconds=data["online_seconds"]))
    with (root / "reporting_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    labels = ["PROM Case 2", "HPROM Case 2", "PROM Case 2 + B3", "HPROM Case 2 + B3"]
    lines = ["# Local Case-2 hyperreduction experiment", "", "## State error against the HDM (%)", "",
             "| Model | Verification | In-domain 1 | In-domain 2 | In-domain mean | Extrapolation |",
             "|---|---:|---:|---:|---:|---:|"]
    for metric, heading in [("state_error_percent", None),
                            ("coefficient_error_percent", "Coefficient error against the full linear PROM (%)")]:
        if heading:
            lines += ["", f"## {heading}", "",
                      "| Model | Verification | In-domain 1 | In-domain 2 | In-domain mean | Extrapolation |",
                      "|---|---:|---:|---:|---:|---:|"]
        for label in labels:
            values = [r[metric] for r in records if r["model"] == label]
            lines.append("| " + label + " | " + " | ".join(f"{x:.4f}" for x in
                         [*values[:3], np.mean(values[:3]), values[3]]) + " |")
    rule = json.loads((rule_path.parent / "summary.json").read_text())
    lines += ["", "## Protocol and limits", "",
              f"- Positive support: {rule['sampled_cells']} / 62500 cells; stencil: {rule['stencil_cells']} / 62500 cells.",
              "- Same baseline 151-mode Euclidean basis, master ANN, 10 primary coordinates and known linear initial state.",
              "- Cubature fitted from the nine existing training trajectories; selected using two separate validation parameters.",
              "- Verification is the center of the original training grid; the other three reporting parameters are excluded from fitting and selection.",
              "- The four reporting trajectories were evaluated only after the rule was frozen in selection.json.",
              "- PROM accuracies come from the previously audited Sherlock run with identical basis/model fingerprints.",
              "- All coefficient errors use the full linear PROM reference; this measures the combined closure and sampling discrepancy.",
              "- Local online timings are exploratory single runs with two numerical threads. No cross-machine speed-up is inferred.",
              "- The first compressed ECM rule (257 cells) failed validation; the successful rule uses nonnegative dynamic and full mass moment fitting on an importance-sampling support.",
              "- The moment optimizer reached its iteration limit. Its output is positive and passes independent operator/trajectory audits; optimal convergence is not claimed.",
              "", "## Exploratory HPROM online times (seconds)", "",
              "| Model | Verification | In-domain 1 | In-domain 2 | In-domain mean | Extrapolation |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in ["HPROM Case 2", "HPROM Case 2 + B3"]:
        values = [r["local_online_seconds"] for r in records if r["model"] == label]
        lines.append("| " + label + " | " + " | ".join(f"{x:.3f}" for x in
                     [*values[:3], np.mean(values[:3]), values[3]]) + " |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#1f77b4", "#9ecae1", "#2ca02c", "#a1d99b"]
    x = np.arange(4)
    for axis, metric, title in zip(axes, ["state_error_percent", "coefficient_error_percent"],
                                  ["State error against HDM", "Coefficient error against linear PROM"]):
        for i, label in enumerate(labels):
            values = [r[metric] for r in records if r["model"] == label]
            axis.bar(x + (i-1.5)*.19, values, width=.18, color=colors[i], edgecolor=".25",
                     linewidth=.5, hatch="//" if "HPROM" in label else "", label=label)
        if metric == "state_error_percent":
            axis.plot(x, audit["linear_state_errors"], "ko-", lw=1, ms=3, label="Linear PROM")
        axis.set_xticks(x, ["Verification", "In-domain 1", "In-domain 2", "Extrapolation"])
        axis.set_title(title)
        axis.set_ylabel(r"Relative error (\%)")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .85))
    fig.savefig(root / "reporting_comparison.png", dpi=220)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
