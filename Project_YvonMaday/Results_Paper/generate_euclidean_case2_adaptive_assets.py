#!/usr/bin/env python3
"""Audit saved Sherlock trajectories and generate the adaptive Case-2 results."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_euclidean_prom_only_assets import POINTS, BASIS_PATH, U_REF_PATH
from manuscript_plot_style import METHOD_COLORS

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parents[1]
SOURCE = PAPER / "euclidean_case2_sherlock_42621312_1Dptdf"
BASE = PAPER / "euclidean_prom_main"
FIGURES = PAPER / "Figures/euclidean_prom_only"
TABLES = PAPER / "tables/euclidean_prom_only"
METHODS = (
    ("case2_known_initial", "PROM--ANN Case 2", METHOD_COLORS["case2_n10"]),
    ("adaptive3_known_initial", r"PROM--ANN Case 2 + $\mathbf{B}_3$", "#8C564B"),
    ("case1_known_initial", "PROM--ANN Case 1", METHOD_COLORS["case1"]),
    ("case3_known_initial", "PROM--ANN Case 3", METHOD_COLORS["case3"]),
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_table(name, header, rows, columns):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule", header + r" \\", r"\midrule"]
    for label, values in rows:
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (TABLES / name).write_text("\n".join(lines) + "\n")


def common_log_limits(axes):
    values = np.concatenate([line.get_ydata() for ax in axes for line in ax.lines
                             if len(line.get_ydata()) == 151])
    positive = values[np.isfinite(values) & (values > 0)]
    limits = (10 ** np.floor(np.log10(positive.min())),
              10 ** np.ceil(np.log10(positive.max())))
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(*limits)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    env = json.loads((SOURCE / "environment.json").read_text())
    if (env["threads"], env["repetitions"], env["total_trajectories"]) != (24, 1, 16):
        raise ValueError("Unexpected Sherlock measurement protocol")
    fingerprints = {}

    def check_input(path):
        key = str(path.relative_to(REPO))
        actual = digest(path)
        if actual != env["sha256"][key]:
            raise ValueError(f"Input differs from the Sherlock campaign: {path}")
        fingerprints[key] = actual

    models = BASE / "Stage3/models"
    for path in (BASIS_PATH, U_REF_PATH,
                 models / "master_ann_mu_t_to_qtot_ntot151_best.pt",
                 models / "case1_ann_ntot151_best.pt",
                 models / "case3_ann_ntot151_best.pt"):
        check_input(path)
    basis = np.load(BASIS_PATH)
    uref = np.load(U_REF_PATH).reshape(-1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(151), atol=1e-10)
    timing_rows = list(csv.DictReader((SOURCE / "timing_summary.csv").open()))
    metrics, histories, coefficients, linear_errors = [], {}, {}, []
    for index, point in enumerate(POINTS):
        tag = f"mu1_{point.mu1:.3f}_mu2_{point.mu2:.4f}"
        reference = BASE / f"Runs/Linear/linear_prom_{tag}_ntot151/qN.npy"
        hdm_path = REPO / "Project_YvonMaday/Results/param_snaps" / point.hdm_file
        check_input(reference)
        check_input(hdm_path)
        teacher = np.load(reference)
        hdm = np.load(hdm_path, mmap_mode="r")
        trajectories = {"linear": teacher}
        summaries = {}
        for method, _, _ in METHODS:
            matches = list((SOURCE / "rep01" / method).glob(f"reporting_{index}_*/summary.json"))
            if len(matches) != 1:
                raise ValueError(f"Expected one complete result: {method}, point {index}")
            summary_path = matches[0]
            fingerprints[str(summary_path.relative_to(REPO))] = digest(summary_path)
            summaries[method] = json.loads(summary_path.read_text())
            qpath = summary_path.parent / "qN.npy"
            fingerprints[str(qpath.relative_to(REPO))] = digest(qpath)
            q = np.load(qpath)
            if q.shape != (151, 501) or not np.isfinite(q).all():
                raise ValueError(f"Invalid trajectory: {qpath}")
            np.testing.assert_allclose(q[:, 0], teacher[:, 0], atol=1e-9, rtol=0)
            trajectories[method] = q
        squared = {key: np.empty(501) for key in trajectories}
        truth_squared = np.empty(501)
        for first in range(0, 501, 32):
            sl = slice(first, min(first + 32, 501))
            truth = np.asarray(hdm[:, sl])
            truth_squared[sl] = np.sum(truth ** 2, axis=0)
            for key, q in trajectories.items():
                difference = uref[:, None] + basis @ q[:, sl] - truth
                squared[key][sl] = np.sum(difference ** 2, axis=0)
        for key, values in squared.items():
            histories[index, key] = 100 * np.sqrt(values / truth_squared)
        linear_errors.append(100 * np.sqrt(squared["linear"].sum() / truth_squared.sum()))
        for method, _, _ in METHODS:
            summary = summaries[method]
            state_error = 100 * np.sqrt(squared[method].sum() / truth_squared.sum())
            q = trajectories[method]
            coefficient_error = 100 * np.linalg.norm(q - teacher) / np.linalg.norm(teacher)
            np.testing.assert_allclose(state_error, summary["state_error_percent"], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(coefficient_error, summary["coefficient_error_percent"], rtol=1e-10)
            timing = [row for row in timing_rows if row["method"] == method and int(row["point"]) == index]
            if len(timing) != 1:
                raise ValueError("Incomplete timing summary")
            np.testing.assert_allclose(float(timing[0]["mean_seconds"]), summary["online_seconds"], rtol=1e-10)
            absolute = np.linalg.norm(q - teacher, axis=1)
            coefficients[index, method] = (absolute, 100 * absolute / np.linalg.norm(teacher, axis=1))
            metrics.append(dict(method=method, point=index, state_error_percent=state_error,
                                coefficient_error_percent=coefficient_error,
                                online_seconds=summary["online_seconds"],
                                direction_seconds=summary.get("direction_seconds", 0.0)))

    def values(method, key):
        return [row[key] for row in metrics if row["method"] == method]

    header = r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$"
    for key, filename in (("state_error_percent", "state"), ("coefficient_error_percent", "coefficient")):
        reference_values = linear_errors if filename == "state" else [0.] * 4
        rows = [("Linear PROM (151 modes)", [f"{x:.3f}" for x in (*reference_values[:3], np.mean(reference_values[:3]), reference_values[3])])]
        for method, label, _ in METHODS:
            v = values(method, key)
            rows.append((label, [f"{x:.3f}" for x in (*v[:3], np.mean(v[:3]), v[3])]))
        write_table(f"euclidean_case2_adaptive_{filename}.tex", header, rows, "lrrrr|r")
    time_rows = []
    for method, label, _ in METHODS:
        v = values(method, "online_seconds")
        time_rows.append((label, [f"{x:.1f}" for x in (*v[:3], np.mean(v[:3]), v[3])]))
    write_table("euclidean_case2_adaptive_times.tex", header, time_rows, "lrrrr|r")

    time = .05 * np.arange(501)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat):
        point = POINTS[index]
        ax.plot(time, histories[index, "linear"], color=METHOD_COLORS["linear"], label="Linear PROM", lw=1.4)
        for method, label, color in METHODS:
            ax.plot(time, histories[index, method], color=color, label=label, lw=1.2)
        ax.set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$")
        ax.grid(alpha=.25)
        ax.set_xlim(0, 25)
    maximum = max(v.max() for v in histories.values())
    axes[0, 0].set_ylim(0, 1.08 * maximum)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"instantaneous state error (\%)")
    for ax in axes[-1]:
        ax.set_xlabel(r"time $t$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(FIGURES / "euclidean_case2_adaptive_state_histories.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True)
    for index, point in enumerate(POINTS):
        for method, label, color in METHODS:
            for row in range(2):
                axes[row, index].plot(np.arange(1, 152), coefficients[index, method][row],
                                      color=color, label=label, lw=1.)
        axes[0, index].set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$", fontsize=9)
        for ax in axes[:, index]:
            ax.axvline(10, color=".4", ls=":", lw=.8)
            ax.grid(which="both", alpha=.2)
            ax.set_xlim(1, 151)
        axes[1, index].set_xlabel("coefficient index")
    common_log_limits(axes[0])
    common_log_limits(axes[1])
    axes[0, 0].set_ylabel(r"$\|q_{N,i}-q_{N,i}^{\mathrm{lin}}\|_2$")
    axes[1, 0].set_ylabel(r"relative coefficient error (\%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FIGURES / "euclidean_case2_adaptive_coefficients.png", dpi=220)
    plt.close(fig)
    with (TABLES / "euclidean_case2_adaptive_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    audit = dict(source=str(SOURCE.relative_to(REPO)), hostname=env["hostname"],
                 job=env["slurm_job_id"], threads=24, repetitions=1,
                 linear_state_errors=linear_errors, fingerprints=fingerprints,
                 verification="All 16 state and coefficient errors recomputed from saved qN; common initial coordinates checked.")
    (TABLES / "euclidean_case2_adaptive_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(audit["verification"])
    for method, _, _ in METHODS:
        print(method, "mean state error:", np.mean(values(method, "state_error_percent")[:3]),
              "mean time:", np.mean(values(method, "online_seconds")[:3]))


if __name__ == "__main__":
    main()
