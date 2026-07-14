#!/usr/bin/env python3
"""Generate PROM baseline-vs-enriched manuscript assets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manuscript_plot_style import (
    COEFF_ABS_HEAT_VMAX,
    COEFF_ABS_YLIM,
    COEFF_REL_PERCENT_HEAT_VMAX,
    COEFF_REL_PERCENT_YLIM,
    HDM_COLOR,
    METHOD_COLORS,
    STATE_CUTPLANE_YLIM,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = HERE / "mlspg_prom_main"
ENR = HERE / "mlspg_prom_enrichment_ext25_lhs36"
FIG_DIR = HERE / "Figures" / "prom_only"
TAB_DIR = HERE / "tables" / "prom_only"
BASIS_PATH = HERE / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy"
UREF_PATH = HERE / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy"
NTOT = 151

BASELINE_FILL = "#9ecae9"
BASELINE_EDGE = "#376795"
ENRICHED_FILL = "#a1d99b"
ENRICHED_EDGE = "#2b7a2b"

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

@dataclass(frozen=True)
class Point:
    key: str
    tex: str
    short: str
    mu1: float
    mu2: float
    hdm_file: str

POINTS = (
    Point("verification", r"$\mu^{(v)}$", "validation", 4.875, 0.0225, "mu1_4.875+mu2_0.0225.npy"),
    Point("offgrid1", r"$\mu^{(1)}$", "off-grid 1", 4.560, 0.0190, "mu1_4.56+mu2_0.019.npy"),
    Point("offgrid2", r"$\mu^{(2)}$", "off-grid 2", 5.190, 0.0260, "mu1_5.19+mu2_0.026.npy"),
    Point("extrapolation20pct", r"$\mu^{(3)}$", "extrapolation", 4.000, 0.0330, "mu1_4.0+mu2_0.033.npy"),
)

METHODS = (
    ("PROM--ANN Case 1", "PROM-ANN C1", "case1"),
    ("PROM--ANN Case 2 ($n=10$)", "PROM-ANN C2", "case2"),
    ("PROM--ANN Case 3", "PROM-ANN C3", "case3"),
    ("PROM--POD--AE ($n_z=10$)", "PROM-POD-AE", "podae"),
    ("POD--NN--ROM", "POD-NN-ROM", "podnn"),
    ("POD--DL--ROM ($n_z=10$)", "POD-DL-ROM", "poddl"),
)

COLORS = {
    "Case 1": METHOD_COLORS["case1"],
    "Case 2": METHOD_COLORS["case2_n10"],
    "Case 3": METHOD_COLORS["case3"],
    "POD-AE": METHOD_COLORS["podae"],
    "POD-NN-ROM": METHOD_COLORS["podnn"],
    "POD-DL-ROM": METHOD_COLORS["poddl"],
}

_RECOVERED_Q: dict[Path, np.ndarray] = {}
_PROJ_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def mu_tag(p: Point) -> str:
    return f"mu1_{p.mu1:.3f}_mu2_{p.mu2:.4f}"


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def fmt(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def fmt_time(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if abs(x) < 1.0:
        return f"{x:.3f}"
    return f"{x:.1f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def stage3_summary(root: Path, kind: str) -> Path:
    return {
        "case1": root / "Stage3" / "case1_ann_ntot151_best_summary.txt",
        "case2": root / "Stage3" / "master_ann_mu_t_to_qtot_ntot151_best_summary.txt",
        "case3": root / "Stage3" / "case3_ann_ntot151_best_summary.txt",
        "podae": root / "Stage3" / "prom_pod_ae_ntot151_best_summary.txt",
        "podnn": root / "Stage3" / "master_ann_mu_t_to_qtot_ntot151_best_summary.txt",
        "poddl": root / "Stage3" / "pod_dl_data_driven_ntot151_best_summary.txt",
    }[kind]


def summary_and_q(root: Path, kind: str, p: Point) -> tuple[Path, Path | None, Path | None]:
    mt = mu_tag(p)
    if kind == "linear":
        d = BASE / "Runs" / "Linear" / f"linear_prom_{mt}_ntot151"
        return d / "summary.txt", d / "rom_snaps.npy", d / "qN.npy"
    if kind == "case1":
        d = root / "Runs" / "PROM" / "Case1_Best"
        stem = f"case1_prom_ann_{mt}_n10_ntot151"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", None
    if kind == "case2":
        d = root / "Runs" / "PROM" / "Case2_MasterANN" / "np10"
        stem = f"case2_prom_ann_master_qtot_{mt}_n10_ntot151"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", d / f"{stem}_qN.npy"
    if kind == "case3":
        d = root / "Runs" / "PROM" / "Case3_Best"
        stem = f"case3_prom_ann_{mt}_n10_ntot151"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", None
    if kind == "podae":
        d = root / "Runs" / "PROM" / "PODAE_Best"
        stem = f"podae_prom_{mt}_ntot151_nz10"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", d / f"{stem}_qN.npy"
    if kind == "podnn":
        d = root / "Runs" / "ROM" / "DataDriven_MasterANN" / f"rom_data_driven_{mt}_ntot151"
        return d / "rom_data_driven_summary.txt", d / "rom_snaps.npy", d / "qN.npy"
    if kind == "poddl":
        d = root / "Runs" / "ROM" / "PODDL_Best" / f"pod_dl_data_driven_{mt}_ntot151_nz10"
        return d / "pod_dl_data_driven_summary.txt", d / "rom_snaps.npy", d / "qN.npy"
    raise KeyError(kind)


def projection_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _PROJ_CACHE
    if _PROJ_CACHE is None:
        V = np.asarray(np.load(BASIS_PATH, allow_pickle=False), dtype=np.float64)[:, :NTOT]
        uref = np.asarray(np.load(UREF_PATH, allow_pickle=False), dtype=np.float64).reshape(-1)
        _PROJ_CACHE = (V, V.T @ V, V.T @ uref)
    return _PROJ_CACHE


def recover_q(snaps_path: Path) -> np.ndarray:
    snaps_path = snaps_path.resolve()
    if snaps_path in _RECOVERED_Q:
        return _RECOVERED_Q[snaps_path]
    V, gram, vtu = projection_cache()
    snaps = np.load(snaps_path, mmap_mode="r")
    rhs = V.T @ np.asarray(snaps, dtype=np.float64) - vtu[:, None]
    q = np.linalg.solve(gram, rhs)
    _RECOVERED_Q[snaps_path] = q
    return q


def online_q(root: Path, kind: str, p: Point) -> np.ndarray:
    summary, snaps, qpath = summary_and_q(root, kind, p)
    if not summary.exists():
        raise FileNotFoundError(summary)
    if qpath is not None and qpath.exists():
        return np.asarray(np.load(qpath, allow_pickle=False), dtype=np.float64)
    if snaps is not None and snaps.exists():
        return recover_q(snaps)
    raise FileNotFoundError(f"No qN/snaps for {root} {kind} {p.key}")


def rel_q_error(root: Path, kind: str, p: Point) -> float:
    qref = online_q(BASE, "linear", p)
    q = online_q(root, kind, p)
    return 100.0 * float(np.linalg.norm(q - qref) / np.linalg.norm(qref))


def hdm_path(p: Point) -> Path:
    for candidate in (
        REPO / "Results" / "param_snaps" / p.hdm_file,
        HERE.parent / "Results" / "param_snaps" / p.hdm_file,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing HDM trajectory for {p.key}")


def _cut_indices(state_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = state_size // 2
    side = int(round(np.sqrt(n)))
    if 2 * n != state_size or side * side != n:
        raise ValueError(f"Cannot infer the square state grid from {state_size} entries")
    return (side // 2) * side + np.arange(side), np.arange(side) * side + side // 2


def state_cut_lines_from_snaps(path: Path, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    snaps = np.load(path, mmap_mode="r")
    ix, iy = _cut_indices(snaps.shape[0])
    return np.asarray(snaps[ix, tidx]), np.asarray(snaps[iy, tidx])


def state_cut_lines_from_q(q_path: Path, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.load(q_path, mmap_mode="r")
    if q.shape[0] != NTOT:
        if q.shape[1] != NTOT:
            raise ValueError(f"Unexpected coefficient trajectory shape: {q.shape}")
        q = q.T
    V = np.load(BASIS_PATH, mmap_mode="r")[:, :NTOT]
    u_ref = np.load(UREF_PATH, mmap_mode="r").reshape(-1)
    ix, iy = _cut_indices(u_ref.size)
    q_t = np.asarray(q[:, tidx], dtype=np.float64)
    return u_ref[ix] + V[ix] @ q_t, u_ref[iy] + V[iy] @ q_t


def state_cut_lines(root: Path, kind: str, p: Point, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    _summary, snaps, qpath = summary_and_q(root, kind, p)
    if snaps is not None and snaps.exists():
        return state_cut_lines_from_snaps(snaps, tidx)
    if qpath is not None and qpath.exists():
        return state_cut_lines_from_q(qpath, tidx)
    raise FileNotFoundError(f"Missing online state trajectory for {kind} at {p.key}")


def plot_enriched_solution_overlay() -> Path:
    labels = {
        "linear": "Linear PROM",
        "case1": "PROM-ANN C1",
        "case2": "PROM-ANN C2",
        "case3": "PROM-ANN C3",
        "podae": "PROM-POD-AE",
        "podnn": "POD-NN-ROM",
        "poddl": "POD-DL-ROM",
    }
    colors = {
        "linear": METHOD_COLORS["linear"],
        "case1": METHOD_COLORS["case1"],
        "case2": METHOD_COLORS["case2_n10"],
        "case3": METHOD_COLORS["case3"],
        "podae": METHOD_COLORS["podae"],
        "podnn": METHOD_COLORS["podnn"],
        "poddl": METHOD_COLORS["poddl"],
    }
    order = ("linear", "case1", "case2", "case3", "podae", "podnn", "poddl")
    time_ids = (120, 300, 500)
    figure, axes = plt.subplots(len(POINTS), 2, figsize=(12.8, 13.0))
    for row, p in enumerate(POINTS):
        hdm = hdm_path(p)
        final_lines = state_cut_lines_from_snaps(hdm, time_ids[-1])
        grids = (np.linspace(0.0, 100.0, final_lines[0].size), np.linspace(0.0, 100.0, final_lines[1].size))
        for column, (ax, grid, cut_label) in enumerate(zip(axes[row], grids, (r"$u_x(x,y_{\mathrm{mid}})$", r"$u_x(x_{\mathrm{mid}},y)$"))):
            for tidx in time_ids[:-1]:
                ax.plot(grid, state_cut_lines_from_snaps(hdm, tidx)[column], color=HDM_COLOR, lw=0.85, alpha=0.20)
            ax.plot(grid, final_lines[column], color=HDM_COLOR, lw=2.4, label="HDM" if row == 0 and column == 0 else None)
            for kind in order:
                for tidx in time_ids[:-1]:
                    ax.plot(grid, state_cut_lines(ENR, kind, p, tidx)[column], color=colors[kind], lw=0.8, alpha=0.20)
                ax.plot(grid, state_cut_lines(ENR, kind, p, time_ids[-1])[column], color=colors[kind], lw=1.7, alpha=0.96, label=labels[kind] if row == 0 and column == 0 else None)
            ax.set_title(f"{p.tex}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$: {p.short}: {cut_label}")
            ax.set_xlabel(r"$x$" if column == 0 else r"$y$")
            ax.set_ylabel(r"$u_x$")
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(*STATE_CUTPLANE_YLIM)
            ax.grid(True, alpha=0.25)
    handles, names = axes.ravel()[0].get_legend_handles_labels()
    by_name = dict(zip(names, handles))
    ordered = ["HDM", *(labels[kind] for kind in order)]
    figure.legend([by_name[name] for name in ordered if name in by_name], [name for name in ordered if name in by_name], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle("Enriched PROM campaign: solution cut-plane overlays", y=0.995)
    figure.text(0.5, 0.012, "Fainter solid curves: intermediate times; opaque solid curves: final time.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.93))
    out = FIG_DIR / "prom_enriched_solution_overlays.png"
    figure.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return out


def enriched_coefficient_errors() -> tuple[tuple[str, str, str], dict[tuple[str, str], dict[str, np.ndarray]]]:
    methods = (
        ("case1", "PROM-ANN C1", METHOD_COLORS["case1"]),
        ("case2", "PROM-ANN C2", METHOD_COLORS["case2_n10"]),
        ("case3", "PROM-ANN C3", METHOD_COLORS["case3"]),
        ("podae", "PROM-POD-AE", METHOD_COLORS["podae"]),
        ("podnn", "POD-NN-ROM", METHOD_COLORS["podnn"]),
        ("poddl", "POD-DL-ROM", METHOD_COLORS["poddl"]),
    )
    errors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for p in POINTS:
        q_ref = online_q(ENR, "linear", p)
        denom = np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
        for key, _label, _color in methods:
            error = online_q(ENR, key, p) - q_ref
            errors[(p.key, key)] = {
                "abs_curve": np.linalg.norm(error, axis=1),
                "rel_curve": 100.0 * np.linalg.norm(error, axis=1) / denom,
                "abs_heat": np.abs(error),
                "rel_heat": 100.0 * np.abs(error) / denom[:, None],
            }
    return methods, errors


def plot_enriched_coeff_error_diagnostics() -> Path:
    methods, errors = enriched_coefficient_errors()
    figure, axes = plt.subplots(2, len(POINTS), figsize=(16.2, 7.1), sharex=True)
    x = np.arange(1, NTOT + 1)
    for column, p in enumerate(POINTS):
        absolute, relative = axes[0, column], axes[1, column]
        for key, label, color in methods:
            error = errors[(p.key, key)]
            for ax, value in ((absolute, error["abs_curve"]), (relative, error["rel_curve"])):
                ax.semilogy(x, np.maximum(value, 1.0e-14), color=color, lw=1.75, alpha=0.96, label=label if ax is absolute else None)
        for ax in (absolute, relative):
            ax.axvline(10, color="#333333", lw=1.0, ls=":", alpha=0.72)
            ax.grid(True, which="both", alpha=0.22)
            ax.set_xlim(1, NTOT)
        absolute.set_title(f"{p.tex}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        absolute.set_ylim(*COEFF_ABS_YLIM)
        relative.set_ylim(*COEFF_REL_PERCENT_YLIM)
        relative.set_xlabel("coefficient index")
    axes[0, 0].set_ylabel(r"$\\|q_i-q_i^{\\mathrm{ref}}\\|_2$")
    axes[1, 0].set_ylabel(r"relative coefficient error (\%)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    by_name = dict(zip(names, handles))
    figure.legend([by_name[label] for _, label, _ in methods], [label for _, label, _ in methods], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.05, h_pad=0.8)
    out = FIG_DIR / "prom_enriched_coeff_abs_rel_errors.png"
    figure.savefig(out, dpi=220)
    plt.close(figure)
    return out


def plot_enriched_coefficient_heatmaps() -> list[Path]:
    methods, errors = enriched_coefficient_errors()
    outputs: list[Path] = []
    for kind, vmax, colorbar_label, stem in (
        ("abs_heat", COEFF_ABS_HEAT_VMAX, r"$|q_i-q_i^{\\mathrm{ref}}|$", "abs"),
        ("rel_heat", COEFF_REL_PERCENT_HEAT_VMAX, r"relative coefficient error (\%)", "rel"),
    ):
        figure, axes = plt.subplots(len(methods), len(POINTS), figsize=(15.6, 10.9), sharex=True, sharey=True)
        image = None
        for row, (key, label, _color) in enumerate(methods):
            for column, p in enumerate(POINTS):
                ax = axes[row, column]
                image = ax.imshow(errors[(p.key, key)][kind], origin="lower", aspect="auto", interpolation="nearest", extent=(0.0, 25.0, 1.0, float(NTOT)), cmap="viridis", vmin=0.0, vmax=vmax)
                ax.axhline(10.5, color="white", linestyle=":", linewidth=0.75, alpha=0.82)
                if row == 0:
                    ax.set_title(f"{p.tex}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$", pad=5)
                if column == 0:
                    ax.set_ylabel(label)
                if row == len(methods) - 1:
                    ax.set_xlabel("time")
                ax.grid(False)
        figure.subplots_adjust(left=0.17, right=0.88, bottom=0.07, top=0.93, wspace=0.15, hspace=0.22)
        figure.supylabel("coefficient index", x=0.045)
        color_axis = figure.add_axes([0.905, 0.15, 0.017, 0.68])
        figure.colorbar(image, cax=color_axis).set_label(colorbar_label)
        out = FIG_DIR / f"prom_enriched_coeff_{stem}_heatmaps.png"
        figure.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(figure)
        outputs.append(out)
    return outputs


def state_error(root: Path, kind: str, p: Point) -> float:
    summary, _, _ = summary_and_q(root, kind, p)
    kv = read_kv(summary)
    return float(kv["relative_error_percent"])


def elapsed(root: Path, kind: str, p: Point) -> float:
    summary, _, _ = summary_and_q(root, kind, p)
    kv = read_kv(summary)
    for key in ("online_solve_elapsed_s", "elapsed_s", "online_inference_elapsed_s", "inference_time_s"):
        if key in kv:
            return float(kv[key])
    return float("nan")


def write_training_comparison() -> Path:
    out = TAB_DIR / "prom_enrichment_training_comparison.tex"
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for label, _, kind in METHODS:
        if kind == "podnn":
            continue
        key = kind
        if key in seen:
            continue
        seen.add(key)
        kb = read_kv(stage3_summary(BASE, kind))
        ke = read_kv(stage3_summary(ENR, kind))
        train_b = float(kb.get("train_rel_frob_percent", "nan"))
        val_b = float(kb.get("val_rel_frob_percent", "nan"))
        train_e = float(ke.get("train_rel_frob_percent", "nan"))
        val_e = float(ke.get("val_rel_frob_percent", "nan"))
        if kind == "case2":
            label = "Master POD--NN--ROM (Case 2 tail source)"
        rows.append((label, train_b, val_b, train_e, val_e, float(100.0 * (val_b - val_e) / val_b) if val_b else float("nan")))
    with out.open("w") as f:
        f.write("\\begin{tabular}{lrr|rrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Base train $e_q$ (\\%) & Base val. $e_q$ (\\%) & Enriched train $e_q$ (\\%) & Enriched val. $e_q$ (\\%) & Val. reduction (\\%) \\\\ \n")
        f.write("\\midrule\n")
        for label, tb, vb, te, ve, red in rows:
            f.write(f"{label} & {fmt(tb)} & {fmt(vb)} & {fmt(te)} & {fmt(ve)} & {fmt(red,1)} \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def write_online_state_comparison() -> Path:
    out = TAB_DIR / "prom_enrichment_online_state_comparison.tex"
    with out.open("w") as f:
        f.write("\\begin{tabular}{lrrrrr|rrrrr}\n")
        f.write("\\toprule\n")
        f.write(r"& \multicolumn{5}{c|}{Baseline (9 trajectories)} & \multicolumn{5}{c}{Enriched (9+36 trajectories)} \\" + "\n")
        f.write(r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\" + "\n")
        f.write("\\midrule\n")
        entries = [("Linear PROM", "linear"), *[(label, kind) for label, _, kind in METHODS]]
        for index, (label, kind) in enumerate(entries):
            if index == 5:
                f.write("\\midrule\n")
            base = [state_error(BASE, kind, p) for p in POINTS]
            enriched = [state_error(ENR, kind, p) for p in POINTS]
            base_fmt = [*base[:3], float(np.mean(base[:3])), base[3]]
            enr_fmt = [*enriched[:3], float(np.mean(enriched[:3])), enriched[3]]
            f.write(f"{label} & " + " & ".join(fmt(v) for v in base_fmt + enr_fmt) + " \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def write_coeff_comparison() -> Path:
    out = TAB_DIR / "prom_enrichment_online_coeff_comparison.tex"
    with out.open("w") as f:
        f.write("\\begin{tabular}{lrrrrr|rrrrr}\n")
        f.write("\\toprule\n")
        f.write(r"& \multicolumn{5}{c|}{Baseline (9 trajectories)} & \multicolumn{5}{c}{Enriched (9+36 trajectories)} \\" + "\n")
        f.write(r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\" + "\n")
        f.write("\\midrule\n")
        entries = [("Linear PROM", "linear"), *[(label, kind) for label, _, kind in METHODS]]
        for index, (label, kind) in enumerate(entries):
            if index == 5:
                f.write("\\midrule\n")
            base = [0.0] * len(POINTS) if kind == "linear" else [rel_q_error(BASE, kind, p) for p in POINTS]
            enriched = [0.0] * len(POINTS) if kind == "linear" else [rel_q_error(ENR, kind, p) for p in POINTS]
            base_fmt = [*base[:3], float(np.mean(base[:3])), base[3]]
            enr_fmt = [*enriched[:3], float(np.mean(enriched[:3])), enriched[3]]
            f.write(f"{label} & " + " & ".join(fmt(v) for v in base_fmt + enr_fmt) + " \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def copy_sampling_figures() -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = ENR / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs36"
    pairs = (
        (src_dir / "stage2_sampling_points_baseline.png", FIG_DIR / "prom_enrichment_sampling_baseline.png"),
        (src_dir / "stage2_sampling_points.png", FIG_DIR / "prom_enrichment_sampling_points.png"),
    )
    outputs = []
    for src, dst in pairs:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def plot_state_bar() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["C1", "C2 $n=10$", "C3", "POD-AE", "POD-NN", "POD-DL"]
    base_mean, enr_mean, base_mu3, enr_mu3 = [], [], [], []
    for _, _, kind in METHODS:
        b = [state_error(BASE, kind, p) for p in POINTS]
        e = [state_error(ENR, kind, p) for p in POINTS]
        base_mean.append(np.mean(b[:3])); enr_mean.append(np.mean(e[:3])); base_mu3.append(b[3]); enr_mu3.append(e[3])
    lin = [state_error(BASE, "linear", p) for p in POINTS]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    for ax, bvals, evals, title, lin_val in (
        (axes[0], base_mean, enr_mean, r"in-domain mean: $\mu^{(v)},\mu^{(1)},\mu^{(2)}$", np.mean(lin[:3])),
        (axes[1], base_mu3, enr_mu3, r"extrapolatory point $\mu^{(3)}$", lin[3]),
    ):
        ax.bar(x - width/2, bvals, width, label="baseline 9", color=BASELINE_FILL, edgecolor=BASELINE_EDGE)
        ax.bar(x + width/2, evals, width, label="enriched 9+36", color=ENRICHED_FILL, edgecolor=ENRICHED_EDGE)
        ax.axhline(lin_val, color="black", linestyle="-", linewidth=1.1, label="linear PROM" if ax is axes[0] else None)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"state relative error vs HDM (\%)")
    axes[0].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "prom_enrichment_state_error_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_coeff_bar() -> Path:
    labels = ["C1", "C2 $n=10$", "C3", "POD-AE", "POD-NN", "POD-DL"]
    base_mean, enr_mean, base_mu3, enr_mu3 = [], [], [], []
    for _, _, kind in METHODS:
        b = [rel_q_error(BASE, kind, p) for p in POINTS]
        e = [rel_q_error(ENR, kind, p) for p in POINTS]
        base_mean.append(np.mean(b[:3])); enr_mean.append(np.mean(e[:3])); base_mu3.append(b[3]); enr_mu3.append(e[3])
    x = np.arange(len(labels)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    for ax, bvals, evals, title in (
        (axes[0], base_mean, enr_mean, r"in-domain mean coefficient error"),
        (axes[1], base_mu3, enr_mu3, r"extrapolatory coefficient error"),
    ):
        ax.bar(x - width/2, bvals, width, label="baseline 9", color=BASELINE_FILL, edgecolor=BASELINE_EDGE)
        ax.bar(x + width/2, evals, width, label="enriched 9+36", color=ENRICHED_FILL, edgecolor=ENRICHED_EDGE)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"relative coefficient error vs linear PROM (\%)")
    axes[0].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "prom_enrichment_coeff_error_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_case2_coeff_curves() -> Path:
    # Case 2 exposes how coefficient-data coverage affects an injected ANN tail.
    x = np.arange(1, NTOT + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.3), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), POINTS):
        qref = online_q(BASE, "linear", p)
        for root, label, color in ((BASE, "baseline 9", BASELINE_EDGE), (ENR, "enriched 9+36", ENRICHED_EDGE)):
            q = online_q(root, "case2", p)
            denom = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
            rel = 100.0 * np.linalg.norm(q - qref, axis=1) / denom
            ax.semilogy(x, rel, color=color, lw=1.6, alpha=0.85, label=label)
        ax.axvline(10, color="#333333", linewidth=1.0, linestyle=":", alpha=0.65)
        ax.set_title(f"{p.tex}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.set_xlabel("coefficient index")
        ax.set_ylabel(r"relative coefficient error (\%)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(1e-4, 2e2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "prom_enrichment_case2_coeff_rel_errors.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_dataset_macros() -> Path:
    out = TAB_DIR / "prom_enrichment_dataset_summary.tex"
    meta = json.loads((ENR / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs36" / "meta.json").read_text())
    with out.open("w") as f:
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Quantity & Value \\\\ \n")
        f.write("\\midrule\n")
        f.write(f"Baseline PROM trajectories & {meta['num_base_traj_copied']} \\\\ \n")
        f.write(f"Interior LHS trajectories & {meta['num_interior_lhs_traj']} \\\\ \n")
        f.write(f"Exterior LHS trajectories & {meta['num_exterior_lhs_traj']} \\\\ \n")
        f.write(f"Total training trajectories & {meta['num_traj']} \\\\ \n")
        f.write(f"Snapshots per trajectory & 501 \\\\ \n")
        f.write(f"Training samples & {meta['num_traj'] * 501} \\\\ \n")
        f.write(f"LHS seed & {meta['lhs_seed']} \\\\ \n")
        f.write(f"Margin fraction & {meta['margin_fraction']} \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tables = [
        write_dataset_macros(),
        write_training_comparison(),
        write_online_state_comparison(),
        write_coeff_comparison(),
    ]
    figures = [
        *copy_sampling_figures(),
        plot_state_bar(),
        plot_enriched_solution_overlay(),
        plot_coeff_bar(),
        plot_enriched_coeff_error_diagnostics(),
        *plot_enriched_coefficient_heatmaps(),
        plot_case2_coeff_curves(),
    ]
    print("[prom-enrichment-assets] tables:")
    for t in tables:
        print(f"  {t}")
    print("[prom-enrichment-assets] figures:")
    for f in figures:
        print(f"  {f}")


if __name__ == "__main__":
    main()
