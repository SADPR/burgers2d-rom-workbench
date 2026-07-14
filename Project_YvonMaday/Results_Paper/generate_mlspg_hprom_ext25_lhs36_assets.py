#!/usr/bin/env python3
"""Generate manuscript assets for the ext25-lhs36 MLSPG-sensitive campaign.

This expanded enrichment campaign keeps the same MLSPG-sensitive basis and the
same fixed linear Stage-2 ECSW weights as the baseline campaign.  It augments
the 9 baseline trajectories with 36 linear-HPROM trajectories: 18 inside the
original parameter box and 18 in a 25% expanded margin.  The four evaluation
points are excluded from the enrichment set.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manuscript_plot_style import METHOD_COLORS

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MAIN = ROOT / "mlspg_hprom_main"
EXT = ROOT / "mlspg_hprom_enrichment_ext25_lhs36"
METRIC = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1"
STAGE2 = EXT / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs36"
FIG_BASE = ROOT / "Figures"
FIG_DIR = FIG_BASE / "mlspg_hprom_enrichment_ext25_lhs36"
COEFF_DIR = FIG_DIR / "coeff_errors"
TABLE_DIR = ROOT / "tables"

NX = 250
NY = 250
NTOT = 151
FULL_ELEMENTS = 62500
HDM_REFERENCE_TIME_S = 7.37437560e02
MAIN_ECSW_TAG = "ECSW2pct"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.9,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.7,
    }
)

POINTS = [
    {
        "label": "$\\bm\\mu^{(v)}$",
        "name": "verification",
        "mu1": 4.875,
        "mu2": 0.0225,
        "tag": "mu1_4.875_mu2_0.0225",
        "hfile": "mu1_4.875+mu2_0.0225.npy",
        "short": "v",
    },
    {
        "label": "$\\bm\\mu^{(1)}$",
        "name": "off-grid",
        "mu1": 4.560,
        "mu2": 0.0190,
        "tag": "mu1_4.560_mu2_0.0190",
        "hfile": "mu1_4.56+mu2_0.019.npy",
        "short": "1",
    },
    {
        "label": "$\\bm\\mu^{(2)}$",
        "name": "off-grid",
        "mu1": 5.190,
        "mu2": 0.0260,
        "tag": "mu1_5.190_mu2_0.0260",
        "hfile": "mu1_5.19+mu2_0.026.npy",
        "short": "2",
    },
    {
        "label": "$\\bm\\mu^{(3)}$",
        "name": "20\\% extrapolation",
        "mu1": 4.000,
        "mu2": 0.0330,
        "tag": "mu1_4.000_mu2_0.0330",
        "hfile": "mu1_4.0+mu2_0.033.npy",
        "short": "3",
    },
]
POINTS_WITH_HDM = [p for p in POINTS if p["hfile"] is not None]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    table_label: str
    color: str
    family_path: str | None = None
    file_prefix: str | None = None
    n_primary_for_file: int | None = None
    n_primary: int | None = None
    n_secondary: int | None = None
    coeff_split: int | None = None
    kind: str = "ann"  # ann, linear, data, podae, poddl


MODELS = [
    ModelSpec("linear", "Linear HPROM", "Linear HPROM", METHOD_COLORS["linear"], kind="linear"),
    ModelSpec("case1", "PROM-ANN Case 1", "PROM-ANN Case 1", METHOD_COLORS["case1"], "Case1_GELU_SameArch_Test", "case1_hprom_ann", 10, 10, 141, 10),
    ModelSpec("case2_n10", "PROM-ANN Case 2 ($n=10$)", "PROM-ANN Case 2 ($n=10$)", METHOD_COLORS["case2_n10"], "Case2_Best/np10", "case2_hprom_ann", 10, 10, 141, 10),
    ModelSpec("case2_n20", "PROM-ANN Case 2 ($n=20$)", "PROM-ANN Case 2 ($n=20$)", METHOD_COLORS["case2_n20"], "Case2_Best/np20", "case2_hprom_ann", 20, 20, 131, 20),
    ModelSpec("case3", "PROM-ANN Case 3", "PROM-ANN Case 3", METHOD_COLORS["case3"], "Case3_Best", "case3_hprom_ann", 10, 10, 141, 10),
    ModelSpec("podae", "PROM-POD-AE ($n_z=10$)", "PROM-POD-AE", METHOD_COLORS["podae"], n_primary=10, kind="podae"),
    ModelSpec("podnn", "POD-NN-ROM", "POD-NN-ROM", METHOD_COLORS["podnn"], kind="data"),
    ModelSpec("poddl", "POD-DL-ROM ($n_z=10$)", "POD-DL-ROM", METHOD_COLORS["poddl"], n_primary=10, kind="poddl"),
]


def ensure_dirs() -> None:
    for path in (FIG_DIR, COEFF_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fmt(x: float | int | None, nd: int = 3) -> str:
    if x is None:
        return "--"
    try:
        xf = float(x)
    except Exception:
        return "--"
    if not np.isfinite(xf):
        return "--"
    return f"{xf:.{nd}f}"


def fmt_signed(x: float | None, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{nd}f}"


def mu_s(mu1: float, mu2: float) -> tuple[str, str]:
    return f"{mu1:.3f}", f"{mu2:.4f}"


def is_mu3(mu1: float, mu2: float) -> bool:
    return abs(mu1 - 4.0) < 1e-12 and abs(mu2 - 0.033) < 1e-12


def run_root(root: Path, mu1: float, mu2: float) -> Path:
    if root == MAIN and is_mu3(mu1, mu2):
        return MAIN / "Runs" / "Extrapolation20pct"
    return root / "Runs"


def linear_dir(root: Path, mu1: float, mu2: float) -> Path:
    s1, s2 = mu_s(mu1, mu2)
    if root == EXT:
        return EXT / "Runs" / "LinearHPROM" / f"linear_hprom_mu1_{s1}_mu2_{s2}_ntot151"
    if root == MAIN and is_mu3(mu1, mu2):
        return MAIN / "Runs" / "Extrapolation20pct" / "Linear" / f"linear_hprom_mu1_{s1}_mu2_{s2}_ntot151"
    return MAIN / "Runs" / "Linear" / f"linear_hprom_mu1_{s1}_mu2_{s2}_ntot151"


def ann_stem(spec: ModelSpec, mu1: float, mu2: float) -> str:
    if spec.file_prefix is None or spec.n_primary_for_file is None:
        raise ValueError(spec.key)
    s1, s2 = mu_s(mu1, mu2)
    return f"{spec.file_prefix}_mu1_{s1}_mu2_{s2}_n{spec.n_primary_for_file}_ntot151"


def ann_family_path(root: Path, spec: ModelSpec) -> str:
    if root == MAIN and spec.key == "case1":
        return "Case1_Best"
    return str(spec.family_path)


def model_ecsw_tag(spec: ModelSpec) -> str:
    return MAIN_ECSW_TAG


def podae_stem(mu1: float, mu2: float) -> str:
    s1, s2 = mu_s(mu1, mu2)
    return f"podae_hprom_mu1_{s1}_mu2_{s2}_ntot151_nz10"


def data_dir(root: Path, mu1: float, mu2: float) -> Path:
    s1, s2 = mu_s(mu1, mu2)
    return run_root(root, mu1, mu2) / "DataDriven_Best" / f"rom_data_driven_mu1_{s1}_mu2_{s2}_ntot151"


def poddl_dir(root: Path, mu1: float, mu2: float) -> Path:
    s1, s2 = mu_s(mu1, mu2)
    return run_root(root, mu1, mu2) / "PODDL_Best" / f"pod_dl_data_driven_mu1_{s1}_mu2_{s2}_ntot151_nz10"


def summary_path(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> Path:
    if spec.kind == "linear":
        return linear_dir(root, mu1, mu2) / "summary.txt"
    if spec.kind == "data":
        return data_dir(root, mu1, mu2) / "rom_data_driven_summary.txt"
    if spec.kind == "poddl":
        return poddl_dir(root, mu1, mu2) / "pod_dl_data_driven_summary.txt"
    if spec.kind == "podae":
        return run_root(root, mu1, mu2) / MAIN_ECSW_TAG / "PODAE_Best" / f"{podae_stem(mu1, mu2)}_summary.txt"
    return run_root(root, mu1, mu2) / model_ecsw_tag(spec) / ann_family_path(root, spec) / f"{ann_stem(spec, mu1, mu2)}_summary.txt"


def q_path(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> Path:
    if spec.kind == "linear":
        return linear_dir(root, mu1, mu2) / "qN.npy"
    if spec.kind == "data":
        return data_dir(root, mu1, mu2) / "qN.npy"
    if spec.kind == "poddl":
        return poddl_dir(root, mu1, mu2) / "qN.npy"
    if spec.kind == "podae":
        return run_root(root, mu1, mu2) / MAIN_ECSW_TAG / "PODAE_Best" / f"{podae_stem(mu1, mu2)}_qN.npy"
    return run_root(root, mu1, mu2) / model_ecsw_tag(spec) / ann_family_path(root, spec) / f"{ann_stem(spec, mu1, mu2)}_qN.npy"


def snaps_path(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> Path:
    if spec.kind == "linear":
        return linear_dir(root, mu1, mu2) / "rom_snaps.npy"
    if spec.kind == "data":
        return data_dir(root, mu1, mu2) / "rom_snaps.npy"
    if spec.kind == "poddl":
        return poddl_dir(root, mu1, mu2) / "rom_snaps.npy"
    if spec.kind == "podae":
        return run_root(root, mu1, mu2) / MAIN_ECSW_TAG / "PODAE_Best" / f"{podae_stem(mu1, mu2)}_snaps.npy"
    return run_root(root, mu1, mu2) / model_ecsw_tag(spec) / ann_family_path(root, spec) / f"{ann_stem(spec, mu1, mu2)}_snaps.npy"


def read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def ffloat(summary: dict[str, str], key: str) -> float:
    return float(summary[key])


def spec_error(root: Path, spec: ModelSpec, p: dict[str, object]) -> float:
    return ffloat(read_summary(summary_path(root, spec, float(p["mu1"]), float(p["mu2"]))), "relative_error_percent")


def spec_time(root: Path, spec: ModelSpec, p: dict[str, object]) -> float:
    d = read_summary(summary_path(root, spec, float(p["mu1"]), float(p["mu2"])))
    if spec.kind in {"data", "poddl"}:
        return ffloat(d, "inference_time_s")
    return ffloat(d, "online_solve_elapsed_s")


def spec_ne(root: Path, spec: ModelSpec) -> int | None:
    if spec.kind in {"data", "poddl"}:
        return None
    d = read_summary(summary_path(root, spec, POINTS[0]["mu1"], POINTS[0]["mu2"]))
    return int(d["n_ecsw_elements"])


def load_q(root: Path, spec: ModelSpec, p: dict[str, object]) -> np.ndarray:
    path = q_path(root, spec, float(p["mu1"]), float(p["mu2"]))
    if not path.exists():
        raise FileNotFoundError(path)
    q = np.load(path, allow_pickle=False)
    if q.shape != (NTOT, 501):
        raise ValueError(f"Unexpected qN shape {q.shape}: {path}")
    return np.asarray(q, dtype=np.float64)


def hdm_path(hfile: str) -> Path:
    candidates = [
        PROJECT / "Results" / "param_snaps" / hfile,
        PROJECT / "250x250" / "param_snaps" / hfile,
        PROJECT.parent / "Results" / "param_snaps" / hfile,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(hfile)


def point_title(p: dict[str, object]) -> str:
    short = str(p["short"])
    mu1 = float(p["mu1"])
    mu2 = float(p["mu2"])
    if short == "v":
        return rf"$\mu^{{(v)}}=({mu1:.3f},{mu2:.4f})$\quad\textbf{{verification}}"
    if short == "3":
        return rf"$\mu^{{(3)}}=({mu1:.3f},{mu2:.4f})$\quad\textit{{20\% extrapolation}}"
    return rf"$\mu^{{({short})}}=({mu1:.3f},{mu2:.4f})$\quad\textit{{off-grid}}"


def point_title_compact(p: dict[str, object]) -> str:
    short = str(p["short"])
    mu1 = float(p["mu1"])
    mu2 = float(p["mu2"])
    return rf"$\mu^{{({short})}}$" + "\n" + rf"$({mu1:.3f},{mu2:.4f})$"


def axis_limits(*sets: np.ndarray, pad_fraction: float = 0.12) -> tuple[tuple[float, float], tuple[float, float]]:
    arrays = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in sets if np.asarray(s).size]
    pts = np.vstack(arrays)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    xpad = pad_fraction * max(xmax - xmin, 1e-12)
    ypad = pad_fraction * max(ymax - ymin, 1e-12)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def add_domain_rect(ax: plt.Axes, x0: float, x1: float, y0: float, y1: float, *, linestyle: str, color: str, label: str) -> None:
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.3, label=label)


PARAM_XLIM = (3.72, 6.03)
PARAM_YLIM = (0.0088, 0.0372)
PARAM_FIGSIZE = (7.8, 7.8)
PARAM_DPI = 240


def add_eval_labels(ax: plt.Axes, eval_pts: list[dict[str, object]], *, include_legend: bool = True) -> None:
    offsets = {"v": (10, -8), "1": (8, 5), "2": (8, 5), "3": (8, -1)}
    align = {"v": ("left", "top"), "1": ("left", "bottom"), "2": ("left", "bottom"), "3": ("left", "center")}
    for p in eval_pts:
        short = str(p["short"])
        label = "Evaluation points" if include_legend and short == "v" else None
        ax.scatter(p["mu1"], p["mu2"], s=185, marker="*", color="#c62828", edgecolors="white", linewidths=0.75, zorder=6, label=label)
        suffix = "(v)" if short == "v" else f"({short})"
        ha, va = align[short]
        ax.annotate(
            rf"$\mu^{{{suffix}}}$",
            (float(p["mu1"]), float(p["mu2"])),
            xytext=offsets[short],
            textcoords="offset points",
            fontsize=12,
            color="#7f1111",
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.12", "fc": "#fbfbf7", "ec": "none", "alpha": 0.86},
            zorder=7,
        )


def setup_parameter_axis(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#fbfbf7")
    ax.set_xlim(*PARAM_XLIM)
    ax.set_ylim(*PARAM_YLIM)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title(title)
    ax.grid(True)


def finish_parameter_figure(fig: plt.Figure, ax: plt.Axes, out: Path, *, ncol: int) -> None:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.030),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=True,
    )
    fig.subplots_adjust(left=0.11, right=0.965, bottom=0.205, top=0.925)
    fig.savefig(out, dpi=PARAM_DPI)
    plt.close(fig)
    print(f"[figure] {out}")


def make_parameter_figures() -> None:
    baseline = []
    lhs_int = []
    lhs_ext = []
    with (STAGE2 / "parameter_manifest.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = (float(row["mu1"]), float(row["mu2"]))
            if row["role"] == "baseline_training":
                baseline.append(pt)
            elif row["role"] == "lhs_enrichment" and row["region"] == "interior_original_box":
                lhs_int.append(pt)
            elif row["role"] == "lhs_enrichment":
                lhs_ext.append(pt)
    baseline_arr = np.asarray(baseline)
    lhs_int_arr = np.asarray(lhs_int)
    lhs_ext_arr = np.asarray(lhs_ext)

    # Baseline-only figure, using the same window as the expanded figure.
    fig, ax = plt.subplots(figsize=PARAM_FIGSIZE)
    setup_parameter_axis(ax, "Baseline training set in parameter space")
    ax.scatter(baseline_arr[:, 0], baseline_arr[:, 1], s=125, color="black", edgecolors="black", linewidths=0.8, label=r"Baseline $3\times3$ grid", zorder=4)
    out = FIG_BASE / "parameter_domain_training_only.png"
    finish_parameter_figure(fig, ax, out, ncol=1)

    # Baseline plus the original three evaluation points, retained for the non-enriched campaign.
    fig, ax = plt.subplots(figsize=PARAM_FIGSIZE)
    setup_parameter_axis(ax, "Baseline training set in parameter space")
    ax.scatter(baseline_arr[:, 0], baseline_arr[:, 1], s=125, color="black", edgecolors="black", linewidths=0.8, label=r"Baseline $3\times3$ grid", zorder=4)
    add_eval_labels(ax, POINTS[:3])
    out = FIG_BASE / "parameter_domain_sampling_points.png"
    finish_parameter_figure(fig, ax, out, ncol=2)

    # Baseline plus all evaluation points, including the extrapolatory stress-test point.
    fig, ax = plt.subplots(figsize=PARAM_FIGSIZE)
    setup_parameter_axis(ax, "Baseline training set in parameter space")
    ax.scatter(baseline_arr[:, 0], baseline_arr[:, 1], s=125, color="black", edgecolors="black", linewidths=0.8, label=r"Baseline $3\times3$ grid", zorder=4)
    add_eval_labels(ax, POINTS)
    out = FIG_BASE / "parameter_domain_sampling_points_with_mu3.png"
    finish_parameter_figure(fig, ax, out, ncol=2)

    # Expanded enrichment figure with the same format and axes.
    fig, ax = plt.subplots(figsize=PARAM_FIGSIZE)
    setup_parameter_axis(ax, "Expanded enriched training set in parameter space")
    ax.scatter(baseline_arr[:, 0], baseline_arr[:, 1], s=125, color="black", edgecolors="black", linewidths=0.8, label=r"Baseline $3\times3$ grid", zorder=4)
    ax.scatter(lhs_int_arr[:, 0], lhs_int_arr[:, 1], s=62, color="#2b7bba", alpha=0.88, label="18 interior LHS HPROM points", zorder=3)
    ax.scatter(lhs_ext_arr[:, 0], lhs_ext_arr[:, 1], s=68, color="#1b9e77", alpha=0.88, label="18 margin LHS HPROM points", zorder=3)
    add_eval_labels(ax, POINTS)
    out = FIG_DIR / "parameter_domain_extended_enrichment_points.png"
    finish_parameter_figure(fig, ax, out, ncol=2)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[csv] {path}")


def stage3_rows() -> list[dict[str, object]]:
    specs = [
        ("PROM-ANN Case 1", EXT / "Stage3" / "case1_ann_ntot151_gelu_samearch_test_summary.txt", r"$\mathbf q\in\mathbb R^{10}\mapsto\bar{\mathbf q}\in\mathbb R^{141}$", "z-score; GELU"),
        ("PROM-ANN Case 2 ($n=10$)", EXT / "Stage3" / "case2_ann_ntot151_np10_best_summary.txt", r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\bar{\mathbf q}\in\mathbb R^{141}$", "z-score; SiLU"),
        ("PROM-ANN Case 2 ($n=20$)", EXT / "Stage3" / "case2_ann_ntot151_np20_best_summary.txt", r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\bar{\mathbf q}\in\mathbb R^{131}$", "z-score; SiLU"),
        ("PROM-ANN Case 3", EXT / "Stage3" / "case3_ann_ntot151_best_summary.txt", r"$(\mathbf q,\mu_1,\mu_2,t)\in\mathbb R^{13}\mapsto\bar{\mathbf q}\in\mathbb R^{141}$", "z-score; SiLU"),
        ("PROM-POD-AE", EXT / "Stage3" / "prom_pod_ae_ntot151_best_summary.txt", r"$\mathbf q_N\in\mathbb R^{151}\mapsto\mathbf z\in\mathbb R^{10}\mapsto\widehat{\mathbf q}_N$", "z-score; GELU"),
        ("POD-NN-ROM", EXT / "Stage3" / "data_driven_ann_ntot151_best_summary.txt", r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\mathbf q_N\in\mathbb R^{151}$", "z-score; SiLU"),
        ("POD-DL-ROM", EXT / "Stage3" / "pod_dl_data_driven_ntot151_best_summary.txt", r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\mathbf z\in\mathbb R^{10}\mapsto\widehat{\mathbf q}_N$", "z-score; SiLU"),
    ]
    rows = []
    for method, path, learned_map, norm in specs:
        d = read_summary(path)
        if method == "PROM-POD-AE":
            arch = rf"Encoder {d['hidden_dims']}; decoder reverse; $n_z={d['latent_dim']}$"
        elif method == "POD-DL-ROM":
            arch = rf"Encoder {d['encoder_hidden_dims']}; decoder {d['decoder_hidden_dims']}; dynamics {d['dynamics_hidden_dims']}; $n_z={d['latent_dim']}$"
        else:
            arch = rf"MLP hidden widths {d['hidden_dims']}"
        rows.append(
            {
                "method": method,
                "learned_map": learned_map,
                "architecture": arch,
                "normalization_activation": norm,
                "val_rel": float(d["val_rel_frob_percent"]),
                "trainable_parameters": int(d["trainable_parameters"]),
            }
        )
    return rows


def make_training_table() -> None:
    rows = stage3_rows()
    fields = ["method", "learned_map", "architecture", "normalization_activation", "val_rel", "trainable_parameters"]
    write_csv(TABLE_DIR / "mlspg_hprom_ext25_lhs36_training_winners.csv", rows, fields)
    tex = TABLE_DIR / "mlspg_hprom_ext25_lhs36_training_winners.tex"
    with tex.open("w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Stage--3 network architectures used in the expanded enriched campaign. The dataset contains the baseline 9 trajectories plus 36 fixed-linear-HPROM enrichment trajectories.}\n")
        f.write("\\label{tab:mlspg-hprom-ext25-lhs36-training}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lp{0.28\\textwidth}p{0.28\\textwidth}lrr}\n\\toprule\n")
        f.write("Model & Learned map & Network architecture & Normalization / activation & Val. rel. Frobenius (\\%) & Trainable params \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            nparams = f"{int(row['trainable_parameters']):,}".replace(",", "\\,")
            f.write(f"{row['method']} & {row['learned_map']} & {row['architecture']} & {row['normalization_activation']} & {fmt(row['val_rel'], 3)} & {nparams} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    print(f"[tex] {tex}")


def make_error_table() -> None:
    fields = ["point", "mu1", "mu2", "linear_hprom", "prom_ann_case1", "prom_ann_case2_n10", "prom_ann_case2_n20", "prom_ann_case3", "prom_pod_ae", "pod_nn_rom", "pod_dl_rom"]
    rows = []
    for p in POINTS:
        row = {"point": p["label"], "mu1": p["mu1"], "mu2": p["mu2"]}
        for spec, field in zip(MODELS, fields[3:]):
            row[field] = spec_error(EXT, spec, p)
        rows.append(row)
    write_csv(TABLE_DIR / "mlspg_hprom_ext25_lhs36_errors.csv", rows, fields)
    tex = TABLE_DIR / "mlspg_hprom_ext25_lhs36_errors.tex"
    with tex.open("w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Expanded enriched MLSPG-sensitive campaign: relative trajectory errors (\\%) with respect to HDM. The learned models are trained with the baseline 9 HPROM trajectories plus 36 additional fixed-linear-HPROM trajectories sampled in the original and 25\\% expanded parameter boxes.}\n")
        f.write("\\label{tab:mlspg-hprom-ext25-lhs36-errors}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccccc}\n\\toprule\n")
        f.write("Point & Linear HPROM & PROM-ANN Case 1 & PROM-ANN Case 2 ($n=10$) & PROM-ANN Case 2 ($n=20$) & PROM-ANN Case 3 & PROM-POD-AE & POD-NN-ROM & POD-DL-ROM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['point']} & {fmt(row['linear_hprom'])} & {fmt(row['prom_ann_case1'])} & {fmt(row['prom_ann_case2_n10'])} & {fmt(row['prom_ann_case2_n20'])} & {fmt(row['prom_ann_case3'])} & {fmt(row['prom_pod_ae'])} & {fmt(row['pod_nn_rom'])} & {fmt(row['pod_dl_rom'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    print(f"[tex] {tex}")


def mean(vals: list[float]) -> float:
    return float(np.mean(vals))


def make_comparison_table() -> None:
    fields = ["method", "baseline_mean_original3", "ext36_mean_original3", "delta_original3", "baseline_mu3", "ext36_mu3", "delta_mu3"]
    rows = []
    original = POINTS[:3]
    mu3 = POINTS[3]
    for spec in MODELS:
        b_orig = [spec_error(MAIN, spec, p) for p in original]
        e_orig = [spec_error(EXT, spec, p) for p in original]
        b_mu3 = spec_error(MAIN, spec, mu3)
        e_mu3 = spec_error(EXT, spec, mu3)
        rows.append(
            {
                "method": spec.table_label,
                "baseline_mean_original3": mean(b_orig),
                "ext36_mean_original3": mean(e_orig),
                "delta_original3": mean(e_orig) - mean(b_orig),
                "baseline_mu3": b_mu3,
                "ext36_mu3": e_mu3,
                "delta_mu3": e_mu3 - b_mu3,
            }
        )
    write_csv(TABLE_DIR / "mlspg_hprom_ext25_lhs36_vs_current_errors.csv", rows, fields)
    tex = TABLE_DIR / "mlspg_hprom_ext25_lhs36_vs_current_errors.tex"
    with tex.open("w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Effect of the expanded 36-trajectory enrichment relative to the non-enriched baseline. Errors are relative trajectory errors (\\%) versus HDM. Negative changes indicate improvement after enrichment. The first three columns average over $\\bm\\mu^{(v)}$, $\\bm\\mu^{(1)}$, and $\\bm\\mu^{(2)}$; the last three isolate the extrapolatory point $\\bm\\mu^{(3)}$.}\n")
        f.write("\\label{tab:mlspg-hprom-ext25-lhs36-vs-current}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Method & Baseline mean (3 pts.) & Ext. enrich. mean (3 pts.) & Change & Baseline $\\bm\\mu^{(3)}$ & Ext. enrich. $\\bm\\mu^{(3)}$ & Change \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['method']} & {fmt(row['baseline_mean_original3'])} & {fmt(row['ext36_mean_original3'])} & {fmt_signed(row['delta_original3'])} & {fmt(row['baseline_mu3'])} & {fmt(row['ext36_mu3'])} & {fmt_signed(row['delta_mu3'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    print(f"[tex] {tex}")


def make_hyperreduction_table() -> None:
    fields = ["method", "online_dim", "n_s", "n_e", "mean_online_time_s", "mean_speedup_vs_hdm"]
    rows = []
    for spec in MODELS:
        times = [spec_time(EXT, spec, p) for p in POINTS]
        tmean = mean(times)
        if spec.kind in {"linear", "data"}:
            dim = 151
        elif spec.kind in {"podae", "poddl"}:
            dim = 10
        else:
            dim = spec.n_primary
        rows.append(
            {
                "method": spec.table_label,
                "online_dim": dim,
                "n_s": None if spec.kind in {"linear", "data", "podae", "poddl"} else spec.n_secondary,
                "n_e": spec_ne(EXT, spec),
                "mean_online_time_s": tmean,
                "mean_speedup_vs_hdm": HDM_REFERENCE_TIME_S / tmean,
            }
        )
    write_csv(TABLE_DIR / "mlspg_hprom_ext25_lhs36_hyperreduction.csv", rows, fields)
    tex = TABLE_DIR / "mlspg_hprom_ext25_lhs36_hyperreduction.tex"
    with tex.open("w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\caption{Expanded enriched MLSPG-sensitive campaign: online dimensions, ECSW mesh sizes, average online timings over the four evaluation points, and speedups with respect to the HDM mean time $t_{\\mathrm{HDM}}=737.44$ s. Learned intrusive models use selected 2\\% ECSW rules, the linear HPROM uses the fixed Stage--2 ECSW rule, and non-intrusive models use direct network inference with no ECSW.}\n")
        f.write("\\label{tab:mlspg-hprom-ext25-lhs36-hyperreduction}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Method & Online/latent dim. & $\\bar n$ & $N_e$ & Mean online time (s) & Mean speedup vs HDM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            ns = "--" if row["n_s"] is None else str(row["n_s"])
            ne = "--" if row["n_e"] is None else str(row["n_e"])
            f.write(f"{row['method']} & {row['online_dim']} & {ns} & {ne} & {fmt(row['mean_online_time_s'], 4)} & {fmt(row['mean_speedup_vs_hdm'], 1)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    print(f"[tex] {tex}")


def state_lines_from_snaps(snaps: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    ux = snaps[:FULL_ELEMENTS, tidx]
    return np.asarray(ux[idx_x], dtype=np.float64), np.asarray(ux[idx_y], dtype=np.float64)


def make_solution_overlay() -> None:
    idx_x = (NY // 2) * NX + np.arange(NX)
    idx_y = np.arange(NY) * NX + (NX // 2)
    xgrid = np.linspace(0.0, 100.0, NX)
    ygrid = np.linspace(0.0, 100.0, NY)
    time_ids = [120, 300, 500]

    fig, axes = plt.subplots(len(POINTS_WITH_HDM), 2, figsize=(16.0, 3.55 * len(POINTS_WITH_HDM)), sharex=False)
    for r, p in enumerate(POINTS_WITH_HDM):
        hdm = np.load(hdm_path(str(p["hfile"])), mmap_mode="r", allow_pickle=False)
        snaps_by_model = {spec.key: np.load(snaps_path(EXT, spec, p["mu1"], p["mu2"]), mmap_mode="r", allow_pickle=False) for spec in MODELS}
        for c, (ax, grid, idx, cut_label) in enumerate(
            [
                (axes[r, 0], xgrid, idx_x, r"$u_x(x,y_{\mathrm{mid}})$"),
                (axes[r, 1], ygrid, idx_y, r"$u_x(x_{\mathrm{mid}},y)$"),
            ]
        ):
            for tidx in time_ids[:-1]:
                ax.plot(grid, np.asarray(hdm[idx, tidx]), color="black", linestyle="--", linewidth=1.20, alpha=0.43)
            ax.plot(grid, np.asarray(hdm[idx, time_ids[-1]]), color="black", linestyle="-", linewidth=2.9, alpha=0.96, label="HDM" if r == 0 and c == 0 else None)
            for spec in MODELS:
                snaps = snaps_by_model[spec.key]
                for tidx in time_ids[:-1]:
                    xl, yl = state_lines_from_snaps(snaps, idx_x, idx_y, tidx)
                    ax.plot(grid, xl if c == 0 else yl, color=spec.color, linestyle="--", linewidth=1.0, alpha=0.35)
                xl, yl = state_lines_from_snaps(snaps, idx_x, idx_y, time_ids[-1])
                ax.plot(grid, xl if c == 0 else yl, color=spec.color, linestyle="-", linewidth=2.0, alpha=0.88, label=spec.label if r == 0 and c == 0 else None)
            ax.set_title(point_title(p) + f": {cut_label}")
            ax.set_xlabel("$x$" if c == 0 else "$y$")
            ax.set_ylabel("$u_x$")
            ax.grid(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.012))
    fig.suptitle("Expanded enriched MLSPG-sensitive campaign: solution cut-plane overlays", y=1.055)
    fig.text(0.5, 0.012, r"Dashed: intermediate times; solid: final time.", ha="center", fontsize=10.5)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    out = FIG_DIR / "mlspg_hprom_ext25_lhs36_solution_overlays.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def coeff_errors() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for p in POINTS:
        qref = load_q(EXT, MODELS[0], p)
        ref_norm = np.maximum(np.linalg.norm(qref, axis=1), 1e-14)
        for spec in MODELS[1:]:
            q = load_q(EXT, spec, p)
            err = q - qref
            out[(str(p["short"]), spec.key)] = {
                "abs_curve": np.linalg.norm(err, axis=1),
                "rel_curve": np.linalg.norm(err, axis=1) / ref_norm,
                "abs_heat": np.abs(err),
                "rel_heat": np.abs(err) / ref_norm[:, None],
            }
    return out


def make_coeff_curve(errors: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    x = np.arange(1, NTOT + 1)
    fig, axes = plt.subplots(2, len(POINTS), figsize=(18.0, 8.2), sharex=True)
    for c, p in enumerate(POINTS):
        axa, axr = axes[0, c], axes[1, c]
        short = str(p["short"])
        for spec in MODELS[1:]:
            e = errors[(short, spec.key)]
            label = spec.label if c == 0 else None
            axa.semilogy(x, e["abs_curve"] + 1e-14, color=spec.color, linewidth=1.9, alpha=0.88, label=label)
            axr.semilogy(x, e["rel_curve"] + 1e-14, color=spec.color, linewidth=1.9, alpha=0.88, label=label)
        for ax in (axa, axr):
            ax.axvline(10.5, color="0.30", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.axvline(20.5, color="0.30", linestyle=":", linewidth=1.0, alpha=0.65)
            ax.grid(True, which="major")
            ax.set_xlim(1, NTOT)
        axa.set_title(point_title(p))
        axr.set_xlabel(r"Coefficient index $i$")
    axes[0, 0].set_ylabel(r"$\|e_i\|_2$")
    axes[1, 0].set_ylabel(r"$\|e_i\|_2 / \|q_i^{\mathrm{ref}}\|_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Expanded enriched campaign: coefficient errors vs fixed linear HPROM reference", y=1.075)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.4, h_pad=1.0)
    out = COEFF_DIR / "mlspg_hprom_ext25_lhs36_coeff_abs_rel_all_points.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def make_heatmap(errors: dict[tuple[str, str], dict[str, np.ndarray]], kind: str) -> None:
    specs = MODELS[1:]
    fig, axes = plt.subplots(len(specs), len(POINTS), figsize=(20.0, 2.35 * len(specs) + 1.4), sharex=True, sharey=True)
    vals = [errors[(str(p["short"]), spec.key)][kind] for p in POINTS for spec in specs]
    vmax = float(np.nanpercentile(np.concatenate([v.ravel() for v in vals]), 99.0))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    im = None
    for r, spec in enumerate(specs):
        for c, p in enumerate(POINTS):
            ax = axes[r, c]
            img = errors[(str(p["short"]), spec.key)][kind]
            im = ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=[0.0, 25.0, 1, NTOT], vmin=0.0, vmax=vmax, cmap="viridis")
            if spec.coeff_split:
                ax.axhline(spec.coeff_split + 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            if r == 0:
                ax.set_title(point_title_compact(p), fontsize=11, pad=5)
            if c == 0:
                ax.annotate(spec.label, xy=(-0.10, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=12, annotation_clip=False)
            if r == len(specs) - 1:
                ax.set_xlabel(r"Time $t$")
            ax.grid(False)
    fig.subplots_adjust(left=0.20, right=0.89, bottom=0.055, top=0.895, wspace=0.16, hspace=0.26)
    fig.supylabel(r"Coefficient index $i$", x=0.035, fontsize=14)
    cax = fig.add_axes([0.91, 0.14, 0.022, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    if kind == "abs_heat":
        cbar.set_label(r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|$")
        title = "Expanded enriched campaign: absolute coefficient error heatmaps"
        out = COEFF_DIR / "mlspg_hprom_ext25_lhs36_coeff_abs_heatmaps.png"
    else:
        cbar.set_label(r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|/\|q_i^{\mathrm{ref}}\|_2$")
        title = "Expanded enriched campaign: relative coefficient error heatmaps"
        out = COEFF_DIR / "mlspg_hprom_ext25_lhs36_coeff_rel_heatmaps.png"
    fig.suptitle(title, y=0.975)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")


def validate_inputs() -> None:
    missing = []
    for p in POINTS:
        for spec in MODELS:
            for root in (EXT,):
                if not summary_path(root, spec, p["mu1"], p["mu2"]).exists():
                    missing.append(str(summary_path(root, spec, p["mu1"], p["mu2"])))
                if not q_path(root, spec, p["mu1"], p["mu2"]).exists():
                    missing.append(str(q_path(root, spec, p["mu1"], p["mu2"])))
        for spec in MODELS:
            if not summary_path(MAIN, spec, p["mu1"], p["mu2"]).exists():
                missing.append(str(summary_path(MAIN, spec, p["mu1"], p["mu2"])))
    if missing:
        raise FileNotFoundError("Missing expected files:\n" + "\n".join(missing))
    print("[check] required summaries and qN files found")


def main() -> None:
    ensure_dirs()
    validate_inputs()
    make_parameter_figures()
    make_training_table()
    make_error_table()
    make_comparison_table()
    make_hyperreduction_table()
    make_solution_overlay()
    e = coeff_errors()
    make_coeff_curve(e)
    make_heatmap(e, "abs_heat")
    make_heatmap(e, "rel_heat")


if __name__ == "__main__":
    main()
