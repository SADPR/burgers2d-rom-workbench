#!/usr/bin/env python3
"""Generate the Euclidean-POD PROM enrichment tables and figures.

The four compared campaigns use the same Euclidean basis, model
architectures, validation trajectories, and online evaluation points.  Only
the PROM coefficient-training trajectories change: 9, 9+8, 9+12, and 9+18.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate_prom_enrichment_assets as shared


PAPER = Path(__file__).resolve().parent
BASE = PAPER / "euclidean_prom_main"
EARLY = PAPER / "euclidean_prom_enrichment_lhs8"
MIDDLE = PAPER / "euclidean_prom_enrichment_lhs12"
INTERMEDIATE = PAPER / "euclidean_prom_enrichment_lhs18"
FIG_DIR = PAPER / "Figures" / "euclidean_prom_only"
TAB_DIR = PAPER / "tables" / "euclidean_prom_only"
BASIS_PATH = PAPER / "MetricStudy" / "euclidean" / "Stage1" / "basis.npy"
U_REF_PATH = PAPER / "MetricStudy" / "euclidean" / "Stage1" / "u_ref.npy"

EARLY_MANIFEST = (
    EARLY
    / "Stage2"
    / "prom_coeff_dataset_ntot151_enriched_lhs8"
    / "parameter_manifest.csv"
)
MIDDLE_MANIFEST = (
    MIDDLE
    / "Stage2"
    / "prom_coeff_dataset_ntot151_enriched_lhs12"
    / "parameter_manifest.csv"
)
INTERMEDIATE_MANIFEST = (
    INTERMEDIATE
    / "Stage2"
    / "prom_coeff_dataset_ntot151_enriched_lhs18"
    / "parameter_manifest.csv"
)

CAMPAIGNS = (
    ("Baseline (9 trajectories)", "baseline 9", BASE, "#9ecae9", "#376795"),
    ("Nested (9+8 trajectories)", "nested 9+8", EARLY, "#fdd0a2", "#e6550d"),
    ("Nested (9+12 trajectories)", "nested 9+12", MIDDLE, "#9dd9d2", "#258f83"),
    (
        "Nested (9+18 trajectories)",
        "nested 9+18",
        INTERMEDIATE,
        "#a1d99b",
        "#2b7a2b",
    ),
)


def configure_shared_generator() -> None:
    """Point the established plotting implementation at Euclidean results."""

    shared.BASE = BASE
    shared.EARLY = EARLY
    shared.MID = MIDDLE
    shared.ENR = INTERMEDIATE
    shared.FIG_DIR = FIG_DIR
    shared.TAB_DIR = TAB_DIR
    shared.BASIS_PATH = BASIS_PATH
    shared.UREF_PATH = U_REF_PATH
    shared.EARLY_PARAMETER_MANIFEST = EARLY_MANIFEST
    shared.MID_PARAMETER_MANIFEST = MIDDLE_MANIFEST
    shared.PARAMETER_MANIFEST = INTERMEDIATE_MANIFEST
    shared.CAMPAIGNS = CAMPAIGNS
    shared.POINTS = (
        shared.Point(
            "verification",
            r"$\mu^{(v)}$",
            "verification",
            4.875,
            0.0225,
            "mu1_4.875+mu2_0.0225.npy",
        ),
        shared.Point(
            "offgrid1",
            r"$\mu^{(1)}$",
            "off-grid 1",
            4.560,
            0.0190,
            "mu1_4.56+mu2_0.019.npy",
        ),
        shared.Point(
            "offgrid2",
            r"$\mu^{(2)}$",
            "off-grid 2",
            5.190,
            0.0260,
            "mu1_5.19+mu2_0.026.npy",
        ),
        shared.Point(
            "extrapolation20pct",
            r"$\mu^{(3)}$",
            "extrapolation",
            4.000,
            0.0330,
            "mu1_4.0+mu2_0.033.npy",
        ),
    )
    shared._RECOVERED_Q.clear()
    shared._PROJ_CACHE = None


def read_manifest(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline: list[tuple[float, float]] = []
    interior: list[tuple[float, float]] = []
    margin: list[tuple[float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            point = (float(row["mu1"]), float(row["mu2"]))
            if row["role"] == "baseline_training":
                baseline.append(point)
            elif row["role"] == "lhs_enrichment":
                target = interior if row["region"] == "interior_original_box" else margin
                target.append(point)
    return np.asarray(baseline), np.asarray(interior), np.asarray(margin)


def verify_nested_designs() -> None:
    """Require identical baselines and strict 8/12/18 subset nesting."""

    manifests = [read_manifest(path) for path in
                 (EARLY_MANIFEST, MIDDLE_MANIFEST, INTERMEDIATE_MANIFEST)]
    baseline_sets = [{tuple(point) for point in values[0]} for values in manifests]
    if not baseline_sets[0] == baseline_sets[1] == baseline_sets[2]:
        raise ValueError("The enrichment campaigns do not share one baseline grid")
    enrichment_sets = [
        {tuple(point) for point in np.vstack((interior, margin))}
        for _baseline, interior, margin in manifests
    ]
    if tuple(map(len, enrichment_sets)) != (8, 12, 18):
        raise ValueError("Unexpected enrichment budgets")
    if not enrichment_sets[0] < enrichment_sets[1] < enrichment_sets[2]:
        raise ValueError("The 9+8, 9+12, and 9+18 designs are not strictly nested")


def plot_sampling_figures() -> list[Path]:
    baseline, early_interior, early_margin = read_manifest(EARLY_MANIFEST)
    _, middle_interior, middle_margin = read_manifest(MIDDLE_MANIFEST)
    _, intermediate_interior, intermediate_margin = read_manifest(INTERMEDIATE_MANIFEST)
    validation = np.asarray(shared.PARAMETER_VALIDATION_POINTS)
    configurations = (
        (
            "Baseline training and parameter validation",
            np.empty((0, 2)),
            np.empty((0, 2)),
            "prom_enrichment_sampling_baseline.png",
            "#e6550d",
            "#d7301f",
        ),
        (
            "Nested 9+8 enrichment and parameter validation",
            early_interior,
            early_margin,
            "prom_enrichment_sampling_early.png",
            "#e6550d",
            "#d7301f",
        ),
        (
            "Nested 9+12 enrichment and parameter validation",
            middle_interior,
            middle_margin,
            "prom_enrichment_sampling_middle.png",
            "#258f83",
            "#176b62",
        ),
        (
            "Nested 9+18 enrichment and parameter validation",
            intermediate_interior,
            intermediate_margin,
            "prom_enrichment_sampling_intermediate.png",
            "#2b7bba",
            "#1b9e77",
        ),
    )
    outputs: list[Path] = []
    for title, interior, margin, filename, interior_color, margin_color in configurations:
        figure, axis = plt.subplots(figsize=(7.3, 7.8))
        shared._setup_parameter_axis(axis, title)
        axis.scatter(
            baseline[:, 0],
            baseline[:, 1],
            s=105,
            color="black",
            edgecolors="black",
            linewidths=0.8,
            label=r"Baseline $3\times3$ grid",
            zorder=4,
        )
        if len(interior):
            axis.scatter(
                interior[:, 0],
                interior[:, 1],
                s=58,
                color=interior_color,
                alpha=0.90,
                label=f"{len(interior)} nested interior LHS points",
                zorder=3,
            )
            axis.scatter(
                margin[:, 0],
                margin[:, 1],
                s=62,
                color=margin_color,
                alpha=0.90,
                label=f"{len(margin)} nested margin LHS points",
                zorder=3,
            )
        axis.scatter(
            validation[:, 0],
            validation[:, 1],
            s=72,
            marker="D",
            color="#d18b20",
            edgecolors="#70420c",
            linewidths=0.8,
            label="Held-out parameter-validation points",
            zorder=6,
        )
        shared._plot_parameter_validation(axis)
        shared._plot_parameter_evaluations(axis)
        output = FIG_DIR / filename
        shared._finish_parameter_figure(figure, axis, output, ncol=3)
        outputs.append(output)
    return outputs


def plot_comparison_bar(metric: str) -> Path:
    """Plot the four-level comparison without hiding the large 9+8 outlier."""

    if metric not in {"state", "coefficient"}:
        raise ValueError(metric)
    values_by_campaign: dict[str, tuple[list[float], list[float]]] = {}
    for _label, _plot_label, model_kind in shared.METHODS:
        for campaign_label, _legend, root, _fill, _edge in CAMPAIGNS:
            means, extrapolations = values_by_campaign.setdefault(campaign_label, ([], []))
            if metric == "state":
                values = [shared.state_error(root, model_kind, point) for point in shared.POINTS]
            else:
                values = [shared.rel_q_error(root, model_kind, point) for point in shared.POINTS]
            means.append(float(np.mean(values[:3])))
            extrapolations.append(values[3])

    x_locations = np.arange(len(shared.ENRICHMENT_BAR_LABELS))
    width = 0.18
    offsets = width * (np.arange(len(CAMPAIGNS)) - 0.5 * (len(CAMPAIGNS) - 1))
    all_values = [
        value
        for means_and_extrapolations in values_by_campaign.values()
        for values in means_and_extrapolations
        for value in values
    ]
    if metric == "state":
        linear = [shared.state_error(BASE, "linear", point) for point in shared.POINTS]
        all_values.extend((float(np.mean(linear[:3])), linear[3]))
    common_ylim = (0.8 * min(all_values), 1.25 * max(all_values))

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    titles = (
        r"in-domain mean: $\mu^{(v)},\mu^{(1)},\mu^{(2)}$",
        r"extrapolatory point $\mu^{(3)}$",
    )
    for axis, value_index, title in zip(axes, (0, 1), titles):
        for offset, (campaign_label, legend, _root, fill, edge) in zip(offsets, CAMPAIGNS):
            axis.bar(
                x_locations + offset,
                values_by_campaign[campaign_label][value_index],
                width,
                label=legend,
                color=fill,
                edgecolor=edge,
            )
        if metric == "state":
            linear = [shared.state_error(BASE, "linear", point) for point in shared.POINTS]
            reference = float(np.mean(linear[:3])) if value_index == 0 else linear[3]
            axis.axhline(
                reference,
                color="black",
                linestyle="-",
                linewidth=1.1,
                label="linear PROM" if axis is axes[0] else None,
            )
        axis.axvline(
            shared.INTRUSIVE_DIRECT_SPLIT,
            color="#5a5a5a",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
        )
        axis.set_yscale("log")
        axis.set_ylim(*common_ylim)
        axis.set_title(title)
        axis.set_xticks(x_locations)
        axis.set_xticklabels(shared.ENRICHMENT_BAR_LABELS, rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", which="both", alpha=0.3)
    ylabel = (
        r"state relative error vs HDM (\%)"
        if metric == "state"
        else r"relative coefficient error vs linear PROM (\%)"
    )
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=True, fontsize=8)
    figure.tight_layout()
    filename = (
        "prom_enrichment_state_error_comparison.png"
        if metric == "state"
        else "prom_enrichment_coeff_error_comparison.png"
    )
    output = FIG_DIR / filename
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_case2_coefficient_curves() -> Path:
    """Compare Case-2 errors with one shared range that contains every curve."""

    x_locations = np.arange(1, shared.NTOT + 1)
    curves: dict[tuple[str, str], np.ndarray] = {}
    positive_min = np.inf
    positive_max = 0.0
    for point in shared.POINTS:
        reference = shared.online_q(BASE, "linear", point)
        denominator = np.maximum(np.linalg.norm(reference, axis=1), 1.0e-14)
        for _campaign_label, legend, root, _fill, _edge in CAMPAIGNS:
            coordinates = shared.online_q(root, "case2", point)
            curve = 100.0 * np.linalg.norm(coordinates - reference, axis=1) / denominator
            curves[(point.key, legend)] = curve
            positive = curve[curve > 0.0]
            positive_min = min(positive_min, float(np.min(positive)))
            positive_max = max(positive_max, float(np.max(positive)))
    lower = 10.0 ** np.floor(np.log10(positive_min))
    upper = 10.0 ** np.ceil(np.log10(positive_max))

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.3), sharex=True, sharey=True)
    for axis, point in zip(axes.ravel(), shared.POINTS):
        for _campaign_label, legend, _root, _fill, color in CAMPAIGNS:
            axis.semilogy(
                x_locations,
                curves[(point.key, legend)],
                color=color,
                linewidth=1.6,
                alpha=0.90,
                label=legend,
            )
        axis.axvline(10, color="#333333", linewidth=1.0, linestyle=":", alpha=0.65)
        axis.set_title(f"{point.tex}: $\\mu=({point.mu1:.3f},{point.mu2:.4f})$")
        axis.set_xlabel("coefficient index")
        axis.set_ylabel(r"relative coefficient error (\%)")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_ylim(lower, upper)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output = FIG_DIR / "prom_enrichment_case2_coeff_rel_errors.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def campaign_headers() -> str:
    return " & ".join(
        rf"\multicolumn{{2}}{{{'c|' if index < len(CAMPAIGNS) - 1 else 'c'}}}{{{label.replace(' trajectories', '')}}}"
        for index, (label, _legend, _root, _fill, _edge) in enumerate(CAMPAIGNS)
    )


def write_training_table() -> Path:
    output = TAB_DIR / "euclidean_prom_enrichment_training_comparison.tex"
    column_spec = "l" + "rr|" * (len(CAMPAIGNS) - 1) + "rr"
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "& " + campaign_headers() + " \\\\",
        "Model & "
        + " & ".join([r"train $e_q$ (\%) & val. $e_q$ (\%)"] * len(CAMPAIGNS))
        + " \\\\",
        r"\midrule",
    ]
    seen: set[str] = set()
    for label, _plot_label, kind in shared.METHODS:
        if kind == "podnn" or kind in seen:
            continue
        seen.add(kind)
        if kind == "case2":
            label = "Master POD--NN--ROM (Case 2 tail source)"
        values: list[float] = []
        for _name, _legend, root, _fill, _edge in CAMPAIGNS:
            summary = shared.read_kv(shared.stage3_summary(root, kind))
            values.extend(
                (
                    float(summary["train_rel_frob_percent"]),
                    float(summary["val_rel_frob_percent"]),
                )
            )
        lines.append(label + " & " + " & ".join(shared.fmt(value) for value in values) + " \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    output.write_text("\n".join(lines) + "\n")
    return output


def write_online_table(kind: str) -> Path:
    if kind not in {"state", "coefficient"}:
        raise ValueError(kind)
    output = TAB_DIR / f"euclidean_prom_enrichment_online_{kind}_comparison.tex"
    column_spec = "l" + "rr|" * (len(CAMPAIGNS) - 1) + "rr"
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "& " + campaign_headers() + " \\\\",
        "Model & "
        + " & ".join([r"\shortstack{in-domain\\mean} & $\mu^{(3)}$"] * len(CAMPAIGNS))
        + " \\\\",
        r"\midrule",
    ]
    entries = [("Linear PROM", "linear"), *((label, model_kind) for label, _, model_kind in shared.METHODS)]
    for index, (label, model_kind) in enumerate(entries):
        if index == 5:
            lines.append(r"\midrule")
        values: list[float] = []
        for _name, _legend, root, _fill, _edge in CAMPAIGNS:
            if kind == "state":
                point_values = [shared.state_error(root, model_kind, point) for point in shared.POINTS]
            elif model_kind == "linear":
                point_values = [0.0] * len(shared.POINTS)
            else:
                point_values = [shared.rel_q_error(root, model_kind, point) for point in shared.POINTS]
            values.extend((float(np.mean(point_values[:3])), point_values[3]))
        lines.append(label + " & " + " & ".join(shared.fmt(value) for value in values) + " \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    output.write_text("\n".join(lines) + "\n")
    return output


def write_dataset_table() -> Path:
    output = TAB_DIR / "euclidean_prom_enrichment_dataset_summary.tex"
    base_meta = json.loads((BASE / "Stage2" / "prom_coeff_dataset_ntot151" / "meta.json").read_text())
    early_meta = json.loads(
        (EARLY / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs8" / "meta.json").read_text()
    )
    middle_meta = json.loads(
        (MIDDLE / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs12" / "meta.json").read_text()
    )
    intermediate_meta = json.loads(
        (
            INTERMEDIATE
            / "Stage2"
            / "prom_coeff_dataset_ntot151_enriched_lhs18"
            / "meta.json"
        ).read_text()
    )
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Quantity & Baseline & $9+8$ & $9+12$ & $9+18$ \\",
        r"\midrule",
        f"Baseline grid trajectories & {base_meta['num_traj']} & {early_meta['num_base_traj_copied']} & {middle_meta['num_base_traj_copied']} & {intermediate_meta['num_base_traj_copied']} \\\\",
        f"Interior LHS trajectories & 0 & {early_meta['num_interior_lhs_traj']} & {middle_meta['num_interior_lhs_traj']} & {intermediate_meta['num_interior_lhs_traj']} \\\\",
        f"Margin LHS trajectories & 0 & {early_meta['num_exterior_lhs_traj']} & {middle_meta['num_exterior_lhs_traj']} & {intermediate_meta['num_exterior_lhs_traj']} \\\\",
        f"Total training trajectories & {base_meta['num_traj']} & {early_meta['num_traj']} & {middle_meta['num_traj']} & {intermediate_meta['num_traj']} \\\\",
        f"Training samples & {base_meta['num_traj'] * 501} & {early_meta['num_traj'] * 501} & {middle_meta['num_traj'] * 501} & {intermediate_meta['num_traj'] * 501} \\\\",
        "Held-out parameter-validation trajectories & 2 & 2 & 2 & 2 \\\\",
        "Held-out parameter-validation samples & 1002 & 1002 & 1002 & 1002 \\\\",
        f"LHS seed & -- & {early_meta['lhs_seed']} & {middle_meta['lhs_seed']} & {intermediate_meta['lhs_seed']} \\\\",
        f"Margin fraction & -- & {early_meta['margin_fraction']} & {middle_meta['margin_fraction']} & {intermediate_meta['margin_fraction']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    configure_shared_generator()
    required = (
        BASE, EARLY, MIDDLE, INTERMEDIATE, BASIS_PATH, U_REF_PATH,
        EARLY_MANIFEST, MIDDLE_MANIFEST, INTERMEDIATE_MANIFEST,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    verify_nested_designs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    tables = (
        write_dataset_table(),
        write_training_table(),
        write_online_table("state"),
        write_online_table("coefficient"),
    )
    figures = (
        *plot_sampling_figures(),
        plot_comparison_bar("state"),
        plot_comparison_bar("coefficient"),
        plot_case2_coefficient_curves(),
        shared.plot_enriched_solution_overlay(),
        shared.plot_enriched_coeff_error_diagnostics(),
        *shared.plot_enriched_coefficient_heatmaps(),
    )
    print("[euclidean-prom-enrichment-assets] tables:")
    for path in tables:
        print(f"  {path}")
    print("[euclidean-prom-enrichment-assets] figures:")
    for path in figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
