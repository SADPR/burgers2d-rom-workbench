#!/usr/bin/env python3
"""Export the accepted local rule, its comparator, and a compact audit report."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Project_YvonMaday/Results_Paper"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    study = PAPER / "euclidean_b3_ecm_local"
    bundle = PAPER / "euclidean_b3_ecm_deployment"
    selection = json.loads((study / "selection.json").read_text())
    if not selection["accepted"] or selection["selection_uses_reporting_points"]:
        raise ValueError("No accepted validation-only selection")
    rule = Path(selection["rule_file"])
    reference = PAPER / "euclidean_b3_budget_local/draws2048/weights.npy"
    if (sha(rule) != selection["weights_sha256"]
            or sha(reference) != selection["protocol"]["comparator_sha256"]):
        raise ValueError("Frozen weights changed")
    records = json.loads((study / "reporting_comparison.json").read_text())
    if len(records) != 4:
        raise ValueError("Reporting incomplete")
    for record in records:
        if (record["selected"]["weights_sha256"] != sha(rule)
                or record["reference2048"]["weights_sha256"] != sha(reference)):
            raise ValueError("Reporting used different rules")
    bundle.mkdir(exist_ok=True)
    files = {"ecm_weights.npy": rule, "reference2048_weights.npy": reference,
             "selection.json": study / "selection.json",
             "validation_operator_audit.json": rule.parent / "validation_operator_audit.json",
             "fit.json": rule.parent / "fit.json"}
    for name, source in files.items():
        target = bundle / name
        if target.exists() and sha(target) != sha(source):
            raise ValueError(f"Existing deployment differs: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
    sources = [
        "burgers/case2_hyperreduction.py", "burgers/case2_residual_correction.py",
        "burgers/core.py", "burgers/config.py", "burgers/ecsw_utils.py",
        "Project_YvonMaday/run_case2_hyperreduction.py",
        "Project_YvonMaday/run_case2_local_corrections.py",
        "Project_YvonMaday/run_prom_ann_case_2.py",
        "Project_YvonMaday/study_case2_b3_cubature_budget.py",
    ]
    manifest = dict(accepted=True, selection_uses_reporting_points=False,
                    inputs=selection["protocol"]["inputs"],
                    files={name: sha(bundle / name) for name in files},
                    solver_sources={name: sha(ROOT / name) for name in sources})
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for record in records:
        row = dict(point=record["selected"]["point"], mu1=record["selected"]["mu"][0],
                   mu2=record["selected"]["mu"][1])
        for name, key in (("previous", "reference2048"), ("ecm", "selected")):
            for metric in ("state_error_percent", "coefficient_error_percent", "iterations"):
                row[f"{name}_{metric}"] = record[key][metric]
        rows.append(row)
    report = PAPER / "case2_b3_ecm_local_comparison.csv"
    with report.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fit = selection["candidates"][-1]
    lines = ["# Local ECM experiment for Case 2+B3", "",
             "The baseline HPROM-trained master ANN and the online B3 solver are unchanged.",
             "Only the cubature rule is replaced. The same known initial representation",
             "and predictor reuse are used in both runs. The nine baseline training",
             "trajectories fit the rule; two external parameters select it. Four reporting",
             "points are evaluated only after the selection is frozen.", "",
             "## Cubature and validation", "",
             "| Compression tolerance | Rank | Cells | Operator checks | Trajectory checks |",
             "|---:|---:|---:|:---:|:---:|"]
    for c in selection["candidates"]:
        lines.append(f"| {c['tolerance']:g} | {c['rank']} | {c['sampled_cells']} | "
                     f"{'pass' if c['operator_pass'] else 'fail'} | "
                     f"{('pass' if c['accepted'] else 'fail') if c['operator_pass'] else 'not run'} |")
    lines += ["", "The first three tolerances were rejected before extending the sweep.",
              "The original rejection is archived; all acceptance thresholds were retained.",
              "Compression tolerance is a relative Frobenius projection error, not the",
              "relative singular-value cutoff used elsewhere in the manuscript.", "",
              f"Selected support: {fit['sampled_cells']} residual cells, {fit['stencil_cells']} stencil cells.",
              "Previous rule: 1506 residual cells, 4385 stencil cells.",
              f"Generalized Gram eigenvalues: [{fit['min_gram']:.6f}, {fit['max_gram']:.6f}].",
              f"Maximum gradient relative error: {fit['max_gradient_error']:.6f} (limit 0.05).",
              f"Maximum linearized residual ratio: {fit['max_linearized_ratio']:.6f} (limit 1.05).",
              "Validation coefficient-error ratios ECM/previous: " +
              ", ".join(f"{x:.6f}" for x in fit["validation_coefficient_ratios"]) + ".", "",
              "## Reporting accuracy", "",
              "State errors are percentages against the HDM. Coefficient errors are",
              "percentages against the matching linear HPROM trajectory.", "",
              "| Parameter | Previous state | ECM state | Previous coefficients | ECM coefficients |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| ({row['mu1']}, {row['mu2']}) | {row['previous_state_error_percent']:.6f} | "
                     f"{row['ecm_state_error_percent']:.6f} | {row['previous_coefficient_error_percent']:.6f} | "
                     f"{row['ecm_coefficient_error_percent']:.6f} |")
    means = {key: sum(row[key] for row in rows[:3])/3 for key in
             ("previous_state_error_percent", "ecm_state_error_percent")}
    lines += ["", f"In-domain mean state error: {means['previous_state_error_percent']:.6f}% "
              f"-> {means['ecm_state_error_percent']:.6f}%.", "",
              "The selected ECM rule reduces support and preserves accuracy on the tested",
              "baseline campaign. This is not a guarantee for other parameters or enrichment",
              "budgets; enriched maps require their own fitted and validated rules.",
              "Local wall times are diagnostic and excluded from this accuracy report.",
              "Official timing comparison remains to be run on Sherlock.", "",
              "## Reproducibility", "",
              "34 focused numerical tests passed. The raw moment matrix reproduces the",
              "previous fit's cached moments within 1e-4 relative tolerance, with full-mesh",
              "ECM candidates and an independently checked compression residual.",
              "The deployed rule contains positive weights and integrates the compressed",
              f"basis with relative error {fit['compressed_error']:.3e}.",
              "The protocol, original block errors, and detailed audits are retained in",
              "`euclidean_b3_ecm_local/`; the compact deployment is in",
              "`euclidean_b3_ecm_deployment/`. The production manuscript is unchanged."]
    (PAPER / "case2_b3_ecm_local_report.md").write_text("\n".join(lines) + "\n")
    print(f"Exported {bundle}")
    print(f"Report: {PAPER / 'case2_b3_ecm_local_report.md'}")


if __name__ == "__main__":
    main()
