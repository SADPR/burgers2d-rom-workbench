#!/usr/bin/env python3
"""Generate PROM-only rank and equal-dimension assets from frozen results."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parent
STUDY = PAPER / "euclidean_case2_prom_rank_study"
FIGURE = PAPER / "Figures/euclidean_prom_only/euclidean_case2_rank_validation.png"
TABLE_RANK = PAPER / "tables/euclidean_prom_only/euclidean_case2_rank_validation.tex"
TABLE_STATE = PAPER / "tables/euclidean_prom_only/euclidean_case2_adaptive_state.tex"
TABLE_COEFF = PAPER / "tables/euclidean_prom_only/euclidean_case2_adaptive_coefficient.tex"
AUDIT = PAPER / "tables/euclidean_prom_only/euclidean_case2_rank_assets_audit.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def grouped(rows, key):
    return {value: [row for row in rows if row[key] == value]
            for value in dict.fromkeys(row[key] for row in rows)}


def rank_assets(rows, controls):
    by_rank = grouped(rows, "rank")
    controls_by_rank = grouped(controls, "rank")
    baseline_residual = np.mean([row["mean_residual_norm"] for row in by_rank[0]])
    lines = [
        r"\begin{tabular}{rr rrr rrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{PROM--ANN Case 2 + $\mathbf B_r$, $n=10$} & "
        r"\multicolumn{3}{c}{PROM--ANN Case 2, $n=10+r$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"$r$ & Unknowns & $e_q^{\mathrm{val},1}$ & $e_q^{\mathrm{val},2}$ & Mean & "
        r"$e_q^{\mathrm{val},1}$ & $e_q^{\mathrm{val},2}$ & Mean \\",
        r"\midrule",
    ]
    ranks, unknowns = [], []
    adaptive_means, control_means = [], []
    adaptive_residual_ratios, control_residual_ratios = [], []
    for rank, values in by_rank.items():
        values = sorted(values, key=lambda row: row["point"])
        paired = sorted(controls_by_rank[rank], key=lambda row: row["point"])
        adaptive_errors = [row["coefficient_error_percent"] for row in values]
        control_errors = [row["coefficient_error_percent"] for row in paired]
        adaptive_mean = float(np.mean(adaptive_errors))
        control_mean = float(np.mean(control_errors))
        adaptive_ratio = float(np.mean([row["mean_residual_norm"] for row in values]) /
                               baseline_residual)
        control_ratio = float(np.mean([row["mean_residual_norm"] for row in paired]) /
                              baseline_residual)
        lines.append(
            f"{rank} & {10 + rank} & {adaptive_errors[0]:.3f} & "
            f"{adaptive_errors[1]:.3f} & {adaptive_mean:.3f} & "
            f"{control_errors[0]:.3f} & {control_errors[1]:.3f} & "
            f"{control_mean:.3f} \\\\"
        )
        ranks.append(rank)
        unknowns.append(10 + rank)
        adaptive_means.append(adaptive_mean)
        control_means.append(control_mean)
        adaptive_residual_ratios.append(adaptive_ratio)
        control_residual_ratios.append(control_ratio)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TABLE_RANK.write_text("\n".join(lines) + "\n")

    plt.rcParams.update({
        "font.family": "serif", "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    axes[0].plot(unknowns, adaptive_means, color="#1F77B4", marker="o",
                 label=r"PROM--ANN Case 2 + $\mathbf B_r$ ($n=10$)")
    axes[0].plot(unknowns, control_means, color="#FF7F0E", marker="s",
                 label=r"PROM--ANN Case 2 ($n=10+r$)")
    axes[0].scatter([13], [adaptive_means[ranks.index(3)]], marker="*", s=130,
                    color="#D62728", edgecolor="#111111", linewidth=.5,
                    zorder=5, label=r"Frozen $r=3$")
    axes[0].set_xlabel(r"online unknowns $d=10+r$")
    axes[0].set_ylabel(r"mean validation coefficient-history error (\%)")
    axes[0].set_xticks(unknowns)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=.25)
    axes[0].legend(frameon=True)

    axes[1].plot(unknowns, adaptive_residual_ratios, color="#1F77B4", marker="o",
                 label=r"PROM--ANN Case 2 + $\mathbf B_r$ ($n=10$)")
    axes[1].plot(unknowns, control_residual_ratios, color="#FF7F0E", marker="s",
                 label=r"PROM--ANN Case 2 ($n=10+r$)")
    axes[1].scatter([13], [adaptive_residual_ratios[ranks.index(3)]], marker="*",
                    s=130,
                    color="#D62728", edgecolor="#111111", linewidth=.5, zorder=5)
    axes[1].set_xlabel(r"online unknowns $d=10+r$")
    axes[1].set_ylabel(r"mean converged-residual ratio")
    axes[1].set_xticks(unknowns)
    axes[1].grid(True, alpha=.25)
    axes[1].legend(frameon=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=240, bbox_inches="tight")
    plt.close(fig)


def reporting_tables(rows):
    by_model = grouped(rows, "model")
    order = ("Case 2, n=10", "Case 2, n=13", "Case 2 + B3, n=10")
    labels = {
        "Case 2, n=10": r"PROM--ANN Case 2 ($n=10$)",
        "Case 2, n=13": r"PROM--ANN Case 2 ($n=13$)",
        "Case 2 + B3, n=10": r"PROM--ANN Case 2 + $\mathbf B_3$ ($n=10$)",
    }

    existing_state = {
        "Linear PROM (151 modes)": (0.410, 0.450, 0.463, 0.441, 0.850),
        "PROM--ANN Case 1": (0.418, 0.543, 0.736, 0.566, 1.673),
        "PROM--ANN Case 3": (0.408, 0.512, 0.563, 0.494, 1.521),
    }
    existing_coeff = {
        "Linear PROM (151 modes)": (0., 0., 0., 0., 0.),
        "PROM--ANN Case 1": (0.326, 0.749, 1.393, 0.823, 3.303),
        "PROM--ANN Case 3": (0.225, 0.712, 0.799, 0.579, 2.644),
    }

    def calculated(model, field):
        values = sorted(by_model[model], key=lambda row: row["point"])
        errors = [row[field] for row in values]
        return (*errors[:3], float(np.mean(errors[:3])), errors[3])

    def table(path, field, fixed):
        lines = [r"\begin{tabular}{lrrrr|r}", r"\toprule",
                 r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
                 r"\midrule"]
        values = {model: calculated(model, field) for model in order}
        display = [
            ("Linear PROM (151 modes)", fixed["Linear PROM (151 modes)"]),
            (labels["Case 2, n=10"], values["Case 2, n=10"]),
            (labels["Case 2, n=13"], values["Case 2, n=13"]),
            (labels["Case 2 + B3, n=10"], values["Case 2 + B3, n=10"]),
            ("PROM--ANN Case 1", fixed["PROM--ANN Case 1"]),
            ("PROM--ANN Case 3", fixed["PROM--ANN Case 3"]),
        ]
        for label, row in display:
            lines.append(label + " & " + " & ".join(f"{value:.3f}" for value in row) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        path.write_text("\n".join(lines) + "\n")

    table(TABLE_STATE, "state_error_percent", existing_state)
    table(TABLE_COEFF, "coefficient_error_percent", existing_coeff)


def main():
    rank_source = STUDY / "rank_validation.json"
    control_source = STUDY / "equal_dimension_validation.json"
    reporting_source = STUDY / "matched_reporting.json"
    protocol = STUDY / "protocol.json"
    control_protocol = STUDY / "equal_dimension_protocol.json"
    ranks = json.loads(rank_source.read_text())
    controls = json.loads(control_source.read_text())
    reporting = json.loads(reporting_source.read_text())
    rank_assets(ranks, controls)
    reporting_tables(reporting)
    AUDIT.write_text(json.dumps({
        "sources": {str(path.relative_to(PAPER)): digest(path)
                    for path in (rank_source, control_source, reporting_source,
                                  protocol, control_protocol)},
        "outputs": {str(path.relative_to(PAPER)): digest(path)
                    for path in (FIGURE, TABLE_RANK, TABLE_STATE, TABLE_COEFF)},
        "timings_included": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
