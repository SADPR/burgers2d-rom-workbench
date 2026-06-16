#!/usr/bin/env python3
"""Generate manuscript assets for the enriched MLSPG-sensitive HPROM campaign.

The enriched campaign uses the same MLSPG-sensitive basis and the same fixed
linear Stage-2 ECSW weights as the non-enriched campaign.  Only the training
set for the learned maps changes: baseline 9 trajectories plus 20 linear-HPROM
LHS trajectories.  Coefficient diagnostics in this script use the solver-side
qN files directly and deliberately fail if any expected qN file is missing.
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

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MAIN = ROOT / "mlspg_hprom_main"
ENRICH = ROOT / "mlspg_hprom_enrichment"
METRIC = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1"
STAGE2 = ENRICH / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs20"
FIG_DIR = ROOT / "Figures" / "mlspg_hprom_enrichment"
COEFF_DIR = FIG_DIR / "coeff_errors"
TABLE_DIR = ROOT / "tables"

NX = 250
NY = 250
NTOT = 151
FULL_ELEMENTS = 62500
HDM_REFERENCE_TIME_S = 7.37437560e02

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.9,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.7,
    }
)

POINTS = [
    (
        "$\\bm\\mu^{(v)}$ \\textbf{(verification)}",
        4.875,
        0.0225,
        "mu1_4.875_mu2_0.0225",
        "mu1_4.875+mu2_0.0225.npy",
        "v",
    ),
    (
        "$\\bm\\mu^{(1)}$ (off-grid)",
        4.560,
        0.0190,
        "mu1_4.560_mu2_0.0190",
        "mu1_4.56+mu2_0.019.npy",
        "1",
    ),
    (
        "$\\bm\\mu^{(2)}$ (off-grid)",
        5.190,
        0.0260,
        "mu1_5.190_mu2_0.0260",
        "mu1_5.19+mu2_0.026.npy",
        "2",
    ),
]


def parameter_plot_limits(*point_sets: np.ndarray, pad_fraction: float = 0.20) -> tuple[tuple[float, float], tuple[float, float]]:
    arrays = [np.asarray(points, dtype=np.float64).reshape(-1, 2) for points in point_sets if np.asarray(points).size]
    if not arrays:
        raise ValueError("At least one non-empty point set is required to define parameter-plot limits.")
    pts = np.vstack(arrays)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    xspan = max(xmax - xmin, 1.0e-12)
    yspan = max(ymax - ymin, 1.0e-12)
    xpad = pad_fraction * xspan
    ypad = pad_fraction * yspan
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def style_parameter_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_box_aspect(1)


def point_plot_title(tag: str, mu1: float, mu2: float) -> str:
    if tag == "mu1_4.875_mu2_0.0225":
        return rf"$\mu^{{(v)}}=({mu1:.3f},{mu2:.4f})$\quad\textbf{{verification}}"
    index = "1" if tag == "mu1_4.560_mu2_0.0190" else "2"
    return rf"$\mu^{{({index})}}=({mu1:.3f},{mu2:.4f})$\quad\textit{{off-grid}}"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    table_label: str
    color: str
    linestyle: str
    alpha: float
    linewidth: float
    family_path: str | None = None
    file_prefix: str | None = None
    n_primary_for_file: int | None = None
    n_primary: int | None = None
    n_secondary: int | None = None
    coeff_split: int | None = None
    is_linear: bool = False
    is_data_driven: bool = False
    is_pod_ae: bool = False
    is_pod_dl: bool = False


MODELS = [
    ModelSpec(
        key="linear",
        label="Linear HPROM",
        table_label="Linear HPROM",
        color="tab:red",
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        is_linear=True,
    ),
    ModelSpec(
        key="case1",
        label="PROM-ANN Case 1",
        table_label="PROM-ANN Case 1",
        color="tab:blue",
        linestyle="-",
        alpha=0.88,
        linewidth=2.0,
        family_path="Case1_Best",
        file_prefix="case1_hprom_ann",
        n_primary_for_file=10,
        n_primary=10,
        n_secondary=141,
        coeff_split=10,
    ),
    ModelSpec(
        key="case2_n10",
        label="PROM-ANN Case 2 ($n=10$)",
        table_label="PROM-ANN Case 2 ($n=10$)",
        color="tab:cyan",
        linestyle="-",
        alpha=0.92,
        linewidth=2.0,
        family_path="Case2_Best/np10",
        file_prefix="case2_hprom_ann",
        n_primary_for_file=10,
        n_primary=10,
        n_secondary=141,
        coeff_split=10,
    ),
    ModelSpec(
        key="case2_n20",
        label="PROM-ANN Case 2 ($n=20$)",
        table_label="PROM-ANN Case 2 ($n=20$)",
        color="tab:brown",
        linestyle="-",
        alpha=0.92,
        linewidth=2.0,
        family_path="Case2_Best/np20",
        file_prefix="case2_hprom_ann",
        n_primary_for_file=20,
        n_primary=20,
        n_secondary=131,
        coeff_split=20,
    ),
    ModelSpec(
        key="case3",
        label="PROM-ANN Case 3",
        table_label="PROM-ANN Case 3",
        color="tab:green",
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        family_path="Case3_Best",
        file_prefix="case3_hprom_ann",
        n_primary_for_file=10,
        n_primary=10,
        n_secondary=141,
        coeff_split=10,
    ),
    ModelSpec(
        key="pod_ae_best",
        label="PROM-POD-AE ($n_z=10$)",
        table_label="PROM-POD-AE",
        color="tab:purple",
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        n_primary=10,
        is_pod_ae=True,
    ),
    ModelSpec(
        key="pod_nn_best",
        label="POD-NN-ROM",
        table_label="POD-NN-ROM",
        color="tab:orange",
        linestyle="-",
        alpha=0.92,
        linewidth=2.0,
        is_data_driven=True,
    ),
    ModelSpec(
        key="pod_dl_best",
        label="POD-DL-ROM ($n_z=10$)",
        table_label="POD-DL-ROM",
        color="tab:pink",
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        n_primary=10,
        is_pod_dl=True,
    ),
]


def ensure_dirs() -> None:
    for d in (FIG_DIR, COEFF_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def read_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text(errors="ignore").splitlines():
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def ffloat(d: dict[str, str], key: str) -> float:
    return float(d[key])


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


def tex_escape_texttt(s: str) -> str:
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def linear_dir(mu1: float, mu2: float) -> Path:
    return MAIN / "Runs" / "Linear" / f"linear_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"


def data_driven_dir(root: Path, mu1: float, mu2: float) -> Path:
    return root / "Runs" / "DataDriven_Best" / f"rom_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"


def pod_dl_dir(root: Path, mu1: float, mu2: float) -> Path:
    return root / "Runs" / "PODDL_Best" / f"pod_dl_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151_nz10"


def pod_ae_stem(mu1: float, mu2: float) -> str:
    return f"podae_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151_nz10"


def run_stem(spec: ModelSpec, mu1: float, mu2: float) -> str:
    if spec.file_prefix is None or spec.n_primary_for_file is None:
        raise ValueError(spec.key)
    return f"{spec.file_prefix}_mu1_{mu1:.3f}_mu2_{mu2:.4f}_n{spec.n_primary_for_file}_ntot151"


def model_run_dir(root: Path, spec: ModelSpec) -> Path:
    if spec.family_path is None:
        raise ValueError(spec.key)
    return root / "Runs" / "ECSW1pct" / spec.family_path


def model_summary_path(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> Path:
    return model_run_dir(root, spec) / f"{run_stem(spec, mu1, mu2)}_summary.txt"


def model_q_path(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> Path:
    if spec.is_linear:
        return linear_dir(mu1, mu2) / "qN.npy"
    if spec.is_data_driven:
        return data_driven_dir(root, mu1, mu2) / "qN.npy"
    if spec.is_pod_ae:
        return root / "Runs" / "ECSW1pct" / "PODAE_Best" / f"{pod_ae_stem(mu1, mu2)}_qN.npy"
    if spec.is_pod_dl:
        return pod_dl_dir(root, mu1, mu2) / "qN.npy"
    return model_run_dir(root, spec) / f"{run_stem(spec, mu1, mu2)}_qN.npy"


def model_summary(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> dict[str, str]:
    if spec.is_linear:
        return read_summary(linear_dir(mu1, mu2) / "summary.txt")
    if spec.is_data_driven:
        return read_summary(data_driven_dir(root, mu1, mu2) / "rom_data_driven_summary.txt")
    if spec.is_pod_ae:
        return read_summary(root / "Runs" / "ECSW1pct" / "PODAE_Best" / f"{pod_ae_stem(mu1, mu2)}_summary.txt")
    if spec.is_pod_dl:
        return read_summary(pod_dl_dir(root, mu1, mu2) / "pod_dl_data_driven_summary.txt")
    return read_summary(model_summary_path(root, spec, mu1, mu2))


def spec_error(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> float:
    return ffloat(model_summary(root, spec, mu1, mu2), "relative_error_percent")


def spec_time(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> float:
    d = model_summary(root, spec, mu1, mu2)
    if spec.is_data_driven or spec.is_pod_dl:
        return ffloat(d, "inference_time_s")
    return ffloat(d, "online_solve_elapsed_s")


def spec_ne(root: Path, spec: ModelSpec) -> int | None:
    if spec.is_data_driven or spec.is_pod_dl:
        return None
    if spec.is_linear:
        return int(model_summary(root, spec, POINTS[0][1], POINTS[0][2])["n_ecsw_elements"])
    return int(model_summary(root, spec, POINTS[0][1], POINTS[0][2])["n_ecsw_elements"])


def load_q(root: Path, spec: ModelSpec, mu1: float, mu2: float) -> np.ndarray:
    p = model_q_path(root, spec, mu1, mu2)
    if not p.exists():
        raise FileNotFoundError(f"Missing direct solver qN: {p}")
    q = np.load(p, allow_pickle=False)
    if q.shape != (NTOT, 501):
        raise ValueError(f"Unexpected qN shape for {p}: {q.shape}")
    return np.asarray(q, dtype=np.float64)


def hdm_path(hdm_file: str) -> Path:
    for base in (
        PROJECT / "Results" / "param_snaps",
        PROJECT / "250x250" / "param_snaps",
        PROJECT.parent / "Results" / "param_snaps",
    ):
        p = base / hdm_file
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find HDM snapshot file {hdm_file}")


def state_lines_from_q(
    q: np.ndarray,
    V: np.ndarray,
    u_ref: np.ndarray,
    idx_xline: np.ndarray,
    idx_yline: np.ndarray,
    tidx: int,
) -> tuple[np.ndarray, np.ndarray]:
    qt = q[:, tidx]
    ux = u_ref[:FULL_ELEMENTS]
    xline = ux[idx_xline] + V[idx_xline, :] @ qt
    yline = ux[idx_yline] + V[idx_yline, :] @ qt
    return xline, yline


def make_sampling_figure() -> Path:
    manifest = STAGE2 / "parameter_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = []
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    baseline = [(float(r["mu1"]), float(r["mu2"])) for r in rows if r["role"].startswith("baseline")]
    lhs = [(float(r["mu1"]), float(r["mu2"])) for r in rows if r["role"] == "lhs_enrichment"]
    eval_pts = [(mu1, mu2, tag) for _, mu1, mu2, _, _, tag in POINTS]
    label_offsets = {"v": (36, -18), "1": (30, 18), "2": (30, 18)}
    label_va = {"v": "top", "1": "bottom", "2": "bottom"}

    baseline_arr = np.asarray(baseline, dtype=np.float64)
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    eval_arr = np.asarray([(mu1, mu2) for mu1, mu2, _ in eval_pts], dtype=np.float64)
    xlim, ylim = parameter_plot_limits(baseline_arr, lhs_arr, eval_arr)

    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.set_facecolor("#fbfbf7")
    ax.scatter([x for x, _ in baseline], [y for _, y in baseline], s=78, facecolors="black", edgecolors="black", linewidths=1.4, label="Baseline $3\\times3$ grid")
    ax.scatter([x for x, _ in lhs], [y for _, y in lhs], s=54, color="#2b7bba", alpha=0.86, label="20 LHS HPROM enrichments")
    for mu1, mu2, tag in eval_pts:
        label = "Evaluation points" if tag == "v" else None
        ax.scatter(mu1, mu2, s=170, marker="*", color="#c62828", edgecolors="white", linewidths=0.7, zorder=5, label=label)
        suffix = "(v)" if tag == "v" else f"({tag})"
        ax.annotate(
            rf"$\mu^{{{suffix}}}$",
            (mu1, mu2),
            xytext=label_offsets[tag],
            textcoords="offset points",
            fontsize=12,
            color="#7f1111",
            ha="left",
            va=label_va[tag],
            arrowprops={
                "arrowstyle": "-",
                "color": "#7f1111",
                "lw": 0.8,
                "shrinkA": 2,
                "shrinkB": 5,
            },
            bbox={"boxstyle": "round,pad=0.12", "fc": "#fbfbf7", "ec": "none", "alpha": 0.86},
            zorder=6,
        )
    style_parameter_axis(ax, xlim, ylim)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Enriched training set in parameter space")
    ax.grid(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    out = FIG_DIR / "parameter_domain_enrichment_points.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_solution_overlay(V: np.ndarray, u_ref: np.ndarray) -> Path:
    idx_xline = (NY // 2) * NX + np.arange(NX)
    idx_yline = np.arange(NY) * NX + (NX // 2)
    xgrid = np.linspace(0.0, 100.0, NX)
    ygrid = np.linspace(0.0, 100.0, NY)
    time_ids = [120, 300, 500]

    fig, axes = plt.subplots(len(POINTS), 2, figsize=(16.0, 10.8), sharex=False)
    for r, (_, mu1, mu2, tag, hfile, _) in enumerate(POINTS):
        hdm = np.load(hdm_path(hfile), mmap_mode="r", allow_pickle=False)
        q_by_model = {spec.key: load_q(MAIN if spec.is_linear else ENRICH, spec, mu1, mu2) for spec in MODELS}
        for c, (ax, grid, idx, cut_label) in enumerate(
            [
                (axes[r, 0], xgrid, idx_xline, r"$u_x(x,y_{\mathrm{mid}})$"),
                (axes[r, 1], ygrid, idx_yline, r"$u_x(x_{\mathrm{mid}},y)$"),
            ]
        ):
            for tidx in time_ids[:-1]:
                ax.plot(grid, np.asarray(hdm[idx, tidx]), color="black", linestyle="--", linewidth=1.20, alpha=0.43)
            ax.plot(grid, np.asarray(hdm[idx, time_ids[-1]]), color="black", linestyle="-", linewidth=2.9, alpha=0.96, label="HDM" if r == 0 and c == 0 else None)
            for spec in MODELS:
                q = q_by_model[spec.key]
                for tidx in time_ids[:-1]:
                    xline, yline = state_lines_from_q(q, V, u_ref, idx_xline, idx_yline, tidx)
                    line = xline if c == 0 else yline
                    ax.plot(grid, line, color=spec.color, linestyle="--", linewidth=1.0, alpha=0.35)
                xline, yline = state_lines_from_q(q, V, u_ref, idx_xline, idx_yline, time_ids[-1])
                line = xline if c == 0 else yline
                ax.plot(
                    grid,
                    line,
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=spec.linewidth,
                    alpha=spec.alpha,
                    label=spec.label if r == 0 and c == 0 else None,
                )
            ax.set_title(point_plot_title(tag, mu1, mu2) + f": {cut_label}")
            ax.set_xlabel("$x$" if c == 0 else "$y$")
            ax.set_ylabel("$u_x$")
            ax.grid(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.012))
    fig.suptitle("Enriched MLSPG-sensitive campaign: solution cut-plane overlays", y=1.055)
    fig.text(0.5, 0.012, "Dashed: intermediate times; solid: final time. ROM curves are reconstructed from saved solver-side qN.", ha="center", fontsize=10.5)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    out = FIG_DIR / "mlspg_hprom_enrichment_solution_overlays.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def compute_coeff_errors() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    errors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for _, mu1, mu2, tag, _, _ in POINTS:
        qref = load_q(MAIN, MODELS[0], mu1, mu2)
        ref_norm = np.maximum(np.linalg.norm(qref, axis=1), 1e-14)
        for spec in MODELS[1:]:
            q = load_q(ENRICH, spec, mu1, mu2)
            e = q - qref
            abs_curve = np.linalg.norm(e, axis=1)
            rel_curve = 100.0 * abs_curve / ref_norm
            abs_heat = np.abs(e)
            rel_heat = 100.0 * abs_heat / ref_norm[:, None]
            errors[(tag, spec.key)] = {
                "abs_curve": abs_curve,
                "rel_curve": rel_curve,
                "abs_heat": abs_heat,
                "rel_heat": rel_heat,
            }
    return errors


def make_coeff_curve_figure(errors: dict[tuple[str, str], dict[str, np.ndarray]]) -> Path:
    x = np.arange(1, NTOT + 1)
    fig, axes = plt.subplots(2, len(POINTS), figsize=(16.0, 8.2), sharex=True)
    for c, (_, mu1, mu2, tag, _, _) in enumerate(POINTS):
        ax_abs, ax_rel = axes[0, c], axes[1, c]
        for spec in MODELS[1:]:
            d = errors[(tag, spec.key)]
            label = spec.label if c == 0 else None
            ax_abs.semilogy(x, d["abs_curve"] + 1e-14, color=spec.color, linestyle=spec.linestyle, linewidth=2.0, alpha=spec.alpha, label=label)
            ax_rel.semilogy(x, d["rel_curve"] / 100.0 + 1e-14, color=spec.color, linestyle=spec.linestyle, linewidth=2.0, alpha=spec.alpha, label=label)
        for ax in (ax_abs, ax_rel):
            ax.axvline(10.5, color="0.30", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.axvline(20.5, color="0.30", linestyle=":", linewidth=1.0, alpha=0.65)
            ax.grid(True, which="major")
            ax.set_xlim(1, NTOT)
        ax_abs.set_title(point_plot_title(tag, mu1, mu2))
        ax_rel.set_xlabel(r"Coefficient index $i$")
    axes[0, 0].set_ylabel(r"$\|e_i\|_2$")
    axes[1, 0].set_ylabel(r"$\|e_i\|_2 / \|q_i^{\mathrm{ref}}\|_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Enriched campaign: coefficient errors vs fixed linear HPROM reference", y=1.075)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.6, h_pad=1.0)
    out = COEFF_DIR / "mlspg_hprom_enrichment_coeff_abs_rel_all_points.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_heatmap_grid(errors: dict[tuple[str, str], dict[str, np.ndarray]], kind: str) -> Path:
    assert kind in {"abs_heat", "rel_heat"}
    specs = MODELS[1:]
    fig, axes = plt.subplots(len(specs), len(POINTS), figsize=(16.0, 2.35 * len(specs) + 1.2), sharex=True, sharey=True)
    values = []
    for _, _, _, tag, _, _ in POINTS:
        for spec in specs:
            image = errors[(tag, spec.key)][kind]
            values.append(image / 100.0 if kind == "rel_heat" else image)
    flat = np.concatenate([v.ravel() for v in values])
    vmax = float(np.nanpercentile(flat, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = None
    for r, spec in enumerate(specs):
        for c, (_, mu1, mu2, tag, _, _) in enumerate(POINTS):
            ax = axes[r, c]
            image = errors[(tag, spec.key)][kind]
            image = image / 100.0 if kind == "rel_heat" else image
            im = ax.imshow(image, origin="lower", aspect="auto", interpolation="nearest", extent=[0.0, 25.0, 1, NTOT], vmin=0.0, vmax=vmax, cmap="viridis")
            if spec.coeff_split:
                ax.axhline(spec.coeff_split + 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            if r == 0:
                ax.set_title(point_plot_title(tag, mu1, mu2))
            if c == 0:
                ax.annotate(spec.label, xy=(-0.10, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=12, annotation_clip=False)
            if r == len(specs) - 1:
                ax.set_xlabel(r"Time $t$")
            ax.grid(False)
    fig.subplots_adjust(left=0.22, right=0.89, bottom=0.055, top=0.93, wspace=0.08, hspace=0.24)
    fig.supylabel(r"Coefficient index $i$", x=0.035, fontsize=14)
    cax = fig.add_axes([0.91, 0.14, 0.022, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|$" if kind == "abs_heat" else r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|/\|q_i^{\mathrm{ref}}\|_2$"
    )
    fig.suptitle("Enriched campaign: absolute coefficient error heatmaps" if kind == "abs_heat" else "Enriched campaign: relative coefficient error heatmaps", y=0.965)
    out = COEFF_DIR / ("mlspg_hprom_enrichment_coeff_abs_heatmaps.png" if kind == "abs_heat" else "mlspg_hprom_enrichment_coeff_rel_heatmaps.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[csv] {path}")


def make_error_table() -> None:
    fields = ["point", "mu1", "mu2", "linear_hprom", "prom_ann_case1", "prom_ann_case2_n10", "prom_ann_case2_n20", "prom_ann_case3", "prom_pod_ae", "pod_nn_rom", "pod_dl_rom"]
    rows: list[dict[str, object]] = []
    spec_by_key = {s.key: s for s in MODELS}
    for point_label, mu1, mu2, _, _, _ in POINTS:
        rows.append(
            {
                "point": point_label,
                "mu1": mu1,
                "mu2": mu2,
                "linear_hprom": spec_error(MAIN, spec_by_key["linear"], mu1, mu2),
                "prom_ann_case1": spec_error(ENRICH, spec_by_key["case1"], mu1, mu2),
                "prom_ann_case2_n10": spec_error(ENRICH, spec_by_key["case2_n10"], mu1, mu2),
                "prom_ann_case2_n20": spec_error(ENRICH, spec_by_key["case2_n20"], mu1, mu2),
                "prom_ann_case3": spec_error(ENRICH, spec_by_key["case3"], mu1, mu2),
                "prom_pod_ae": spec_error(ENRICH, spec_by_key["pod_ae_best"], mu1, mu2),
                "pod_nn_rom": spec_error(ENRICH, spec_by_key["pod_nn_best"], mu1, mu2),
                "pod_dl_rom": spec_error(ENRICH, spec_by_key["pod_dl_best"], mu1, mu2),
            }
        )
    write_csv(TABLE_DIR / "mlspg_hprom_enrichment_errors.csv", rows, fields)
    tex_path = TABLE_DIR / "mlspg_hprom_enrichment_errors.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Enriched MLSPG-sensitive campaign: relative trajectory errors (\\%) with respect to HDM. The learned models are trained with the baseline 9 HPROM trajectories plus 20 LHS HPROM trajectories. Intrusive learned models use 1\\% ECSW rules; the linear HPROM row is the fixed reference from the non-enriched campaign.}\n")
        f.write("\\label{tab:mlspg-hprom-enrichment-errors}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("Point & Linear HPROM & PROM-ANN Case 1 & PROM-ANN Case 2 ($n=10$) & PROM-ANN Case 2 ($n=20$) & PROM-ANN Case 3 & PROM-POD-AE & POD-NN-ROM & POD-DL-ROM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['point']} & {fmt(row['linear_hprom'])} & {fmt(row['prom_ann_case1'])} & {fmt(row['prom_ann_case2_n10'])} & {fmt(row['prom_ann_case2_n20'])} & {fmt(row['prom_ann_case3'])} & {fmt(row['prom_pod_ae'])} & {fmt(row['pod_nn_rom'])} & {fmt(row['pod_dl_rom'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")


def mean(xs: list[float]) -> float:
    return float(np.mean(xs))


def make_comparison_table() -> None:
    fields = ["method", "baseline_mean_error", "enriched_mean_error", "delta_mean", "baseline_offgrid_mean", "enriched_offgrid_mean", "delta_offgrid_mean"]
    rows: list[dict[str, object]] = []
    for spec in MODELS:
        b_root = MAIN
        e_root = MAIN if spec.is_linear else ENRICH
        b = [spec_error(b_root, spec, mu1, mu2) for _, mu1, mu2, _, _, _ in POINTS]
        e = [spec_error(e_root, spec, mu1, mu2) for _, mu1, mu2, _, _, _ in POINTS]
        rows.append(
            {
                "method": spec.table_label,
                "baseline_mean_error": mean(b),
                "enriched_mean_error": mean(e),
                "delta_mean": mean(e) - mean(b),
                "baseline_offgrid_mean": mean(b[1:]),
                "enriched_offgrid_mean": mean(e[1:]),
                "delta_offgrid_mean": mean(e[1:]) - mean(b[1:]),
            }
        )
    write_csv(TABLE_DIR / "mlspg_hprom_enrichment_vs_current_errors.csv", rows, fields)
    tex_path = TABLE_DIR / "mlspg_hprom_enrichment_vs_current_errors.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Effect of adding 20 HPROM-generated LHS trajectories to the Stage--3 training set. Errors are relative trajectory errors (\\%) versus HDM. Negative changes indicate improvement after enrichment.}\n")
        f.write("\\label{tab:mlspg-hprom-enrichment-vs-current}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Baseline mean & Enriched mean & Change & Baseline off-grid mean & Enriched off-grid mean & Change \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['method']} & {fmt(row['baseline_mean_error'])} & {fmt(row['enriched_mean_error'])} & {fmt_signed(row['delta_mean'])} & {fmt(row['baseline_offgrid_mean'])} & {fmt(row['enriched_offgrid_mean'])} & {fmt_signed(row['delta_offgrid_mean'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")


def make_hyperreduction_table() -> None:
    fields = ["method", "online_dim", "n_s", "n_e", "mean_online_time_s", "mean_speedup_vs_hdm"]
    rows: list[dict[str, object]] = []
    for spec in MODELS:
        root = MAIN if spec.is_linear else ENRICH
        times = [spec_time(root, spec, mu1, mu2) for _, mu1, mu2, _, _, _ in POINTS]
        tmean = mean(times)
        if spec.is_linear or spec.is_data_driven:
            online_dim = 151
        elif spec.is_pod_ae or spec.is_pod_dl:
            online_dim = 10
        else:
            online_dim = spec.n_primary
        rows.append(
            {
                "method": spec.table_label,
                "online_dim": online_dim,
                "n_s": None if (spec.is_linear or spec.is_data_driven or spec.is_pod_ae or spec.is_pod_dl) else spec.n_secondary,
                "n_e": spec_ne(root, spec),
                "mean_online_time_s": tmean,
                "mean_speedup_vs_hdm": HDM_REFERENCE_TIME_S / tmean,
            }
        )
    write_csv(TABLE_DIR / "mlspg_hprom_enrichment_hyperreduction.csv", rows, fields)
    tex_path = TABLE_DIR / "mlspg_hprom_enrichment_hyperreduction.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Enriched MLSPG-sensitive campaign: online dimensions, ECSW mesh sizes, average online timings over the three evaluation points, and speedups with respect to the HDM mean time $t_{\\mathrm{HDM}}=737.44$ s. The linear HPROM row uses the fixed Stage--2 ECSW rule from the non-enriched campaign; non-intrusive models use direct network inference and no ECSW.}\n")
        f.write("\\label{tab:mlspg-hprom-enrichment-hyperreduction}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Online/latent dim. & $n_s$ & $n_e$ & Mean online time (s) & Mean speedup vs HDM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            ns = "--" if row["n_s"] is None else str(row["n_s"])
            ne = "--" if row["n_e"] is None else str(row["n_e"])
            f.write(f"{row['method']} & {row['online_dim']} & {ns} & {ne} & {fmt(row['mean_online_time_s'], 4)} & {fmt(row['mean_speedup_vs_hdm'], 1)} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")


def stage3_summary_rows() -> list[dict[str, object]]:
    files = [
        (
            "PROM-ANN Case 1",
            ENRICH / "Stage3" / "case1_ann_ntot151_best_summary.txt",
            r"$\mathbf q\in\mathbb R^{10}\mapsto\bar{\mathbf q}\in\mathbb R^{141}$",
            "z-score; SiLU",
        ),
        (
            "PROM-ANN Case 2 ($n=10$)",
            ENRICH / "Stage3" / "case2_ann_ntot151_np10_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\bar{\mathbf q}\in\mathbb R^{141}$",
            "z-score; SiLU",
        ),
        (
            "PROM-ANN Case 2 ($n=20$)",
            ENRICH / "Stage3" / "case2_ann_ntot151_np20_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\bar{\mathbf q}\in\mathbb R^{131}$",
            "z-score; SiLU",
        ),
        (
            "PROM-ANN Case 3",
            ENRICH / "Stage3" / "case3_ann_ntot151_best_summary.txt",
            r"$(\mathbf q,\mu_1,\mu_2,t)\in\mathbb R^{13}\mapsto\bar{\mathbf q}\in\mathbb R^{141}$",
            "z-score; SiLU",
        ),
        (
            "PROM-POD-AE",
            ENRICH / "Stage3" / "prom_pod_ae_ntot151_best_summary.txt",
            r"$\mathbf q_N\in\mathbb R^{151}\mapsto\mathbf z\in\mathbb R^{10}\mapsto\widehat{\mathbf q}_N$",
            "z-score; GELU",
        ),
        (
            "POD-NN-ROM",
            ENRICH / "Stage3" / "data_driven_ann_ntot151_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\mathbf q_N\in\mathbb R^{151}$",
            "z-score; SiLU",
        ),
        (
            "POD-DL-ROM",
            ENRICH / "Stage3" / "pod_dl_data_driven_ntot151_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\mathbf z\in\mathbb R^{10}\mapsto\widehat{\mathbf q}_N$",
            "z-score; SiLU",
        ),
    ]
    rows = []
    for method, path, learned_map, norm_activation in files:
        d = read_summary(path)
        if method == "PROM-POD-AE":
            hidden = d["hidden_dims"]
            arch = rf"Encoder {hidden}; decoder reverse; $n_z={d['latent_dim']}$"
        elif method == "POD-DL-ROM":
            arch = (
                rf"Encoder {d['encoder_hidden_dims']}; decoder {d['decoder_hidden_dims']}; "
                rf"dynamics {d['dynamics_hidden_dims']}; $n_z={d['latent_dim']}$"
            )
        else:
            arch = rf"MLP hidden widths {d['hidden_dims']}"
        rows.append(
            {
                "method": method,
                "learned_map": learned_map,
                "architecture": arch,
                "normalization_activation": norm_activation,
                "val_rel": float(d["val_rel_frob_percent"]),
                "trainable_parameters": int(d["trainable_parameters"]),
            }
        )
    return rows


def make_training_table() -> None:
    rows = stage3_summary_rows()
    fields = ["method", "learned_map", "architecture", "normalization_activation", "val_rel", "trainable_parameters"]
    write_csv(TABLE_DIR / "mlspg_hprom_enrichment_training_winners.csv", rows, fields)
    tex_path = TABLE_DIR / "mlspg_hprom_enrichment_training_winners.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Stage--3 network architectures used in the enriched campaign. The table reports the learned map, hidden widths or latent dimensions, normalization/activation choices, validation relative Frobenius error on the enriched HPROM coefficient dataset, and the number of trainable parameters.}\n")
        f.write("\\label{tab:mlspg-hprom-enrichment-training}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lp{0.28\\textwidth}p{0.28\\textwidth}lrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Learned map & Network architecture & Normalization / activation & Val. rel. Frobenius (\\%) & Trainable params \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            nparams = f"{int(row['trainable_parameters']):,}".replace(",", "\\,")
            f.write(
                f"{row['method']} & {row['learned_map']} & {row['architecture']} & "
                f"{row['normalization_activation']} & {fmt(row['val_rel'], 3)} & {nparams} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")


def validate_direct_q() -> None:
    missing = []
    for _, mu1, mu2, _, _, _ in POINTS:
        for spec in MODELS:
            root = MAIN if spec.is_linear else ENRICH
            p = model_q_path(root, spec, mu1, mu2)
            if not p.exists():
                missing.append(str(p))
                continue
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            if arr.shape != (NTOT, 501):
                raise ValueError(f"Unexpected shape {arr.shape}: {p}")
    if missing:
        raise FileNotFoundError("Missing direct qN files:\n" + "\n".join(missing))
    print("[check] direct qN files complete for linear reference and enriched models")


def main() -> None:
    ensure_dirs()
    validate_direct_q()
    V = np.load(METRIC / "basis.npy", mmap_mode="r", allow_pickle=False)
    u_ref = np.load(METRIC / "u_ref.npy", mmap_mode="r", allow_pickle=False)
    make_sampling_figure()
    make_training_table()
    make_error_table()
    make_comparison_table()
    make_hyperreduction_table()
    make_solution_overlay(V, u_ref)
    errors = compute_coeff_errors()
    make_coeff_curve_figure(errors)
    make_heatmap_grid(errors, "abs_heat")
    make_heatmap_grid(errors, "rel_heat")


if __name__ == "__main__":
    main()
