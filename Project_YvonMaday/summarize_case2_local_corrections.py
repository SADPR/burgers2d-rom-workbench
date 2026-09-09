#!/usr/bin/env python3
"""Rebuild the local experiment report from completed, separately saved runs."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from Project_YvonMaday.run_case2_local_corrections import PAPER, CAMPAIGN, REPORTING, atomic_json, sha256

OUTPUT = PAPER / "euclidean_case2_confirmatory"
DISPLAY = {
    "baseline": "Case 2, original initialization",
    "baseline_known_initial": "Case 2, known initial state",
    "case1": "Case 1, original initialization",
    "case3": "Case 3, original initialization",
    "case1_known_initial": "Case 1, known initial state",
    "case3_known_initial": "Case 3, known initial state",
    "global3": "Case 2 + fixed SVD, r=3",
    "global10": "Case 2 + fixed SVD, r=10",
    "ann_tangent3": "Case 2 + ANN derivative directions",
    "feedback0": "Affine feedback, unregularized",
    "feedback1": "Affine feedback, ridge=1",
    "krylov1": "Case 2 + residual-adaptive r=1",
    "krylov3": "Case 2 + residual-adaptive r=3",
    "krylov5": "Case 2 + residual-adaptive r=5",
    "krylov3_known_initial": "Case 2 + residual-adaptive r=3 + known initial state",
    "krylov3_zero": "Residual-adaptive r=3, no ANN after t=0",
    "case2_n20": "Case 2, n=20",
}
ROOTS = [PAPER / name for name in (
    "euclidean_case2_local_corrections", "euclidean_case2_affine_feedback",
    "euclidean_case2_local_references", "euclidean_case2_known_initial",
    "euclidean_case2_confirmatory", "euclidean_case2_matched_initial_references")]


def all_results():
    results = []
    for root in ROOTS:
        for path in sorted(root.glob("*/summary.json")):
            d = json.loads(path.read_text())
            d["path"] = str(path)
            method = d["config"]["method"]
            if method == "affine_feedback":
                method = f"feedback{d['config']['ridge']:g}"
            if d["config"].get("initialization") == "linear":
                method += "_known_initial"
            d["method_key"] = method
            results.append(d)
    return results


def original_errors(patterns):
    values = []
    for mu in REPORTING:
        tag = f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"
        matches = list(CAMPAIGN.glob(patterns.format(tag=tag)))
        if len(matches) != 1:
            raise ValueError(f"Expected one summary for {tag}: {matches}")
        fields = dict(line.split(":", 1) for line in matches[0].read_text().splitlines() if ":" in line)
        values.append(float(fields["relative_error_percent"]))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Record the rank decision from validation only")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = all_results()
    validation = {}
    for d in data:
        if d["config"]["stage"] == "validation":
            validation.setdefault(d["method_key"], []).append(d)
    if args.freeze:
        candidates = {key: float(np.mean([d["coefficient_error_percent"] for d in validation[key]]))
                      for key in ("krylov1", "krylov3", "krylov5") if len(validation.get(key, [])) == 2}
        if len(candidates) != 3:
            raise RuntimeError("Both validation parameters must be complete for each candidate rank")
        selected = min(candidates, key=candidates.get)
        payload = {"selection_metric": "mean full-trajectory coefficient error vs linear PROM at the two validation parameters",
                   "candidate_values_percent": candidates, "selected_method": selected,
                   "reporting_variants": ["original ANN initial representation", "known initial condition projected onto the full Euclidean trial space"],
                   "reporting_policy": "four existing manuscript parameters; no tuning on these results",
                   "validation_sources_sha256": {d["path"]: sha256(d["path"]) for key in candidates for d in validation[key]},
                   "initial_condition_ablation": "compare baseline and selected correction at both validation points before reporting"}
        destination = OUTPUT / "selection.json"
        if destination.exists() and json.loads(destination.read_text()) != payload:
            raise RuntimeError("Refusing to change the frozen selection")
        atomic_json(destination, payload)
        print(json.dumps(payload, indent=2))
        return

    lines = ["# Local Euclidean Case-2 experiments", "",
             "All errors are percentages. Validation coefficient errors and reporting HDM state errors are different metrics.", "",
             "## Full-trajectory validation", "",
             "Two existing ANN-validation parameters; 500 free-running time steps, no reference predecessors.", "",
             "| Method | Validation 1 | Validation 2 | Mean coefficient error | Mean local time (s) |",
             "|---|---:|---:|---:|---:|"]
    validation_rows = []
    for method, records in sorted(validation.items()):
        records = sorted(records, key=lambda d: d["config"]["mu"])
        if len(records) != 2:
            continue
        errors = [d["coefficient_error_percent"] for d in records]
        seconds = float(np.mean([d["online_seconds"] for d in records]))
        validation_rows.append([method, *errors, float(np.mean(errors)), seconds])
        lines.append(f"| {DISPLAY.get(method, method)} | {errors[0]:.4f} | {errors[1]:.4f} | {np.mean(errors):.4f} | {seconds:.1f} |")
    lines += ["", "Times include online direction construction, where applicable. They are local CPU observations, not publication-grade repeated benchmarks.", "",
              "## Reporting state errors against HDM", "",
              "Original reference rows come from unchanged Euclidean campaign summaries. New rows require all four points to be complete.", "",
              "| Method | Verification | Off-grid 1 | Off-grid 2 | In-domain mean | Extrapolation |",
              "|---|---:|---:|---:|---:|---:|"]
    originals = {
        "Linear PROM (151)": "Runs/Linear/linear_prom_{tag}_ntot151/summary.txt",
        "PROM-ANN Case 1": "Runs/PROM/Case1_Best/*{tag}*summary.txt",
        "PROM-ANN Case 2 (10)": "Runs/PROM/Case2_MasterANN/np10/*{tag}*summary.txt",
        "PROM-ANN Case 2 (20)": "Runs/PROM/Case2_MasterANN_NSweep/np20/*{tag}*summary.txt",
        "PROM-ANN Case 3": "Runs/PROM/Case3_Best/*{tag}*summary.txt",
    }
    state_rows = {label: original_errors(pattern) for label, pattern in originals.items()}
    reporting = {}
    for d in data:
        if d["config"]["stage"] == "reporting" and "state_error_percent" in d:
            reporting.setdefault(d["method_key"], {})[tuple(d["config"]["mu"])] = d
    for method, records in sorted(reporting.items()):
        if all(mu in records for mu in REPORTING):
            state_rows[method] = [records[mu]["state_error_percent"] for mu in REPORTING]
    for label, values in state_rows.items():
        lines.append(f"| {DISPLAY.get(label, label)} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {np.mean(values[:3]):.4f} | {values[3]:.4f} |")
    baseline_check = []
    for d in data:
        if d["config"]["stage"] == "reporting" and d["method_key"] == "baseline":
            mu = d["config"]["mu"]
            tag = f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"
            old = next((CAMPAIGN / "Runs/PROM/Case2_MasterANN/np10").glob(f"*{tag}*qN.npy"))
            saved = np.load(Path(d["path"]).parent / "qN.npy")
            reference_q = np.load(old)
            baseline_check.append({"mu": mu, "max_absolute_coefficient_difference": float(np.max(np.abs(saved-reference_q))),
                                   "relative_frobenius_difference": float(np.linalg.norm(saved-reference_q)/np.linalg.norm(reference_q))})
    if baseline_check:
        atomic_json(OUTPUT / "baseline_reproduction.json", baseline_check)
        largest = max(item["relative_frobenius_difference"] for item in baseline_check)
        lines += ["", f"Baseline reproduction: maximum relative coefficient-trajectory difference from the downloaded Case-2 run is {largest:.3e} ({len(baseline_check)} completed points)."]

    lines += ["", "## Matched local timing check", "",
              "Reporting point (4.56, 0.019), two CPU threads. Repeated rows report the mean and population standard deviation; single observations are labeled n=1.", "",
              "| Method | Repetitions | Mean seconds | Standard deviation |", "|---|---:|---:|---:|"]
    timing = {}
    for d in data:
        if d["config"]["stage"] == "reporting" and np.allclose(d["config"]["mu"], REPORTING[1], rtol=0, atol=1e-12):
            timing.setdefault(d["method_key"], []).append(d["online_seconds"])
    for method, values in sorted(timing.items()):
        spread = f"{np.std(values):.2f}" if len(values) > 1 else "not estimated"
        lines.append(f"| {DISPLAY.get(method, method)} | {len(values)} | {np.mean(values):.2f} | {spread} |")
    lines += ["", "## Scope and limitations", "",
              "- The ANN, its architecture and its nine-parameter training data are unchanged.",
              "- The selected rank is determined only from validation; see selection.json.",
              "- Three adaptive amplitudes add to the ten primary unknowns, but building their directions accesses all 141 tail modes and the full residual/Jacobian.",
              "- The adaptive basis is frozen throughout each time-step nonlinear solve. The method changes the online trial space, not just ANN training.",
              "- The known-initial variant uses the full linear PROM projection of the prescribed initial state. Subsequent states are not teacher forced.",
              "- The zero-tail ablation removes ANN predictions after t=0 but retains the original ANN initial state.",
              "- These are single-checkpoint, single-discretization results, not evidence of universal superiority or novelty.",
              "- The 151-mode linear PROM is a reference approximation, not a mathematical lower bound on HDM state error.",
              "- Neither the manuscript nor MLSPG/HPROM results were modified.", ""]
    (OUTPUT / "results.md").write_text("\n".join(lines))

    findings = ["# Findings from the local Euclidean tests", "",
                "The useful change is an online residual-adaptive tail correction, not the previously tested fixed global B_r.", "",
                "At each time step, build three tail patterns from the current residual and its Jacobian, freeze those patterns, and solve for their amplitudes together with the ten primary coordinates. The ANN architecture, weights and nine-parameter training set are unchanged.", "",
                "The initial-state variant uses only the prescribed initial condition projected onto the existing 151-mode space. No reference trajectory is fed into an online solve.", "",
                "## What helped on validation", ""]
    for key in ("baseline", "krylov1", "krylov3", "krylov3_known_initial", "case1_known_initial", "case3_known_initial"):
        records = validation.get(key, [])
        if len(records) == 2:
            mean = np.mean([d["coefficient_error_percent"] for d in records])
            findings.append(f"- {DISPLAY[key]}: {mean:.4f}% mean coefficient-trajectory error against the linear PROM.")
    findings += ["", "## What did not justify adoption", "",
                 "- The fixed SVD correction reduced validation error only slightly, even with ten extra amplitudes.",
                 "- ANN parameter/time derivative directions and constant affine primary-innovation feedback worsened the validation error.",
                 "- Increasing the adaptive rank from three to five did not improve either validation trajectory with the original initialization.",
                 "- Changing initialization alone did not improve mean validation error; the strong result requires the residual-adaptive correction.",
                 "- Removing the ANN after t=0 substantially degraded the adaptive method. The result is not explained by physics correction alone.", "",
                 "## Reporting errors against HDM", ""]
    for key in ("PROM-ANN Case 2 (10)", "krylov3", "krylov3_known_initial", "case1_known_initial", "case3_known_initial", "Linear PROM (151)"):
        if key in state_rows:
            values = state_rows[key]
            findings.append(f"- {DISPLAY.get(key,key)}: in-domain mean {np.mean(values[:3]):.4f}%; extrapolation {values[3]:.4f}%.")
    findings += ["", "## Why the combination matters", "",
                 "Residual minimization advances the solution from the previous computed state. If the ANN supplies an inaccurate initial state, a small residual can still describe the evolution of that inaccurate initial condition. The known-initial-state treatment removes this source; the adaptive tail correction addresses the new injection errors at subsequent time steps. The ablation results are consistent with this explanation, rather than showing that a smaller residual automatically implies a smaller HDM error.", "",
                 "## Cost and interpretation", "",
                 "This changes the online PROM. It solves 13 amplitudes per time step, but direction construction uses all 141 tail modes and the full residual/Jacobian. It is not claimed to have the original Case-2 cost or to be hyper-reduced.", "",
                 "The first local timings place the adaptive method near the cost of Cases 1 and 3 and above unmodified Case 2. The timing table in results.md separates repeated measurements from single observations; CPU scheduling and frequency were not controlled.", "",
                 "The appropriate conclusion is a promising, validated accuracy improvement for this Euclidean campaign, not universal superiority. All four reporting parameters are pre-existing manuscript points, not a new independent generalization benchmark.", "",
                 "See results.md, the CSV tables, the saved coefficient trajectories and [the method notes](../../CASE2_LOCAL_CORRECTIONS.md) for the experiment definitions and provenance.", ""]
    (OUTPUT / "findings.md").write_text("\n".join(findings))
    with (OUTPUT / "validation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "validation1_coefficient_percent", "validation2_coefficient_percent", "mean_coefficient_percent", "mean_seconds"])
        writer.writerows(validation_rows)
    with (OUTPUT / "state_errors.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "verification", "offgrid1", "offgrid2", "in_domain_mean", "extrapolation"])
        writer.writerows([label, *values[:3], np.mean(values[:3]), values[3]] for label, values in state_rows.items())
    atomic_json(OUTPUT / "result_sources.json", {d["path"]: sha256(d["path"]) for d in data})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": False, "font.family": "STIXGeneral", "font.size": 11})
    chosen = ["Linear PROM (151)", "PROM-ANN Case 1", "PROM-ANN Case 2 (10)", "PROM-ANN Case 3", "krylov3", "krylov3_known_initial"]
    labels = {"Linear PROM (151)": "Linear PROM", "PROM-ANN Case 1": "Case 1", "PROM-ANN Case 2 (10)": "Case 2",
              "PROM-ANN Case 3": "Case 3", "krylov3": "Case 2 + adaptive r=3", "krylov3_known_initial": "Adaptive r=3 + known initial state"}
    colors = ["black", "#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, title, index in zip(axes, ("In-domain mean", "Extrapolatory point"), (None, 3)):
        available = [key for key in chosen if key in state_rows]
        for j, key in enumerate(available):
            value = np.mean(state_rows[key][:3]) if index is None else state_rows[key][index]
            ax.bar(j, value, color=colors[chosen.index(key)], width=.72)
            ax.annotate(f"{value:.3f}", (j, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(len(available)), [labels[key] for key in available], rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.14)
    axes[0].set_ylabel("State error against HDM (%)")
    fig.savefig(OUTPUT / "state_comparison.png", dpi=180)
    fig.savefig(OUTPUT / "state_comparison.pdf")
    plt.close(fig)

    matched = ["Linear PROM (151)", "case1_known_initial", "baseline_known_initial",
               "case3_known_initial", "krylov3_known_initial"]
    if all(key in state_rows for key in matched):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained", sharey=True)
        short = ["Linear PROM", "PROM-ANN C1", "PROM-ANN C2", "PROM-ANN C3", "PROM-ANN C2 + adaptive r=3"]
        for ax, title, index in zip(axes, ("In-domain mean", "Extrapolatory point"), (None, 3)):
            for j, key in enumerate(matched):
                value = np.mean(state_rows[key][:3]) if index is None else state_rows[key][index]
                ax.bar(j, value, color=colors[j], width=.72)
                ax.annotate(f"{value:.3f}", (j, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
            ax.set_xticks(range(len(matched)), short, rotation=30, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            ax.margins(y=.14)
        axes[0].set_ylabel("State error against HDM (%)")
        fig.suptitle("Matched known initial state; identical Euclidean trial basis")
        fig.savefig(OUTPUT / "matched_initial_state_comparison.png", dpi=180)
        fig.savefig(OUTPUT / "matched_initial_state_comparison.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, mu in zip(axes.flat, REPORTING):
        teacher = np.load(CAMPAIGN / f"Runs/Linear/linear_prom_mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}_ntot151/qN.npy")
        normalization = np.linalg.norm(teacher, axis=0)
        for key, color in (("baseline", "#17becf"), ("baseline_known_initial", "#7f7f7f"),
                           ("krylov3", "#ff7f0e"), ("krylov3_known_initial", "#9467bd")):
            record = reporting.get(key, {}).get(mu)
            if record is None:
                continue
            q = np.load(Path(record["path"]).parent / "qN.npy")
            error = 100 * np.linalg.norm(q-teacher, axis=0) / np.maximum(normalization, 1e-30)
            ax.plot(0.05*np.arange(error.size), error, label=key.replace("_", " "), color=color)
        ax.set_title(f"mu = ({mu[0]:g}, {mu[1]:g})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Coefficient error vs linear PROM (%)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="outside upper center", ncol=2)
    fig.savefig(OUTPUT / "trajectory_errors.png", dpi=180)
    fig.savefig(OUTPUT / "trajectory_errors.pdf")
    plt.close(fig)
    print(OUTPUT / "results.md")


if __name__ == "__main__":
    main()
