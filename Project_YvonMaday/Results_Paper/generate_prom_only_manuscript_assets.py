#!/usr/bin/env python3
"""Generate PROM-only tables and figures for manuscript_prom.tex.

This script reads existing PROM-first outputs. It does not run solvers or modify
manuscript.tex. It deliberately writes to PROM-only figure/table folders.
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    METHOD_LINE_STYLES,
    STATE_CUTPLANE_YLIM,
)

SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
REPO = PAPER.parents[1]
PROM = PAPER / "mlspg_prom_main"
RUNS = PROM / "Runs"
STAGE3 = PROM / "Stage3"
FIG_DIR = PAPER / "Figures" / "prom_only"
TAB_DIR = PAPER / "tables" / "prom_only"
DIAG = PAPER / "Prom_MasterANN_Diagnostic"
BASIS_PATH = PAPER / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy"
U_REF_PATH = PAPER / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy"
NTOT = 151

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

@dataclass(frozen=True)
class Point:
    key: str
    label: str
    mu1: float
    mu2: float
    hdm_file: str

POINTS = (
    Point("verification", r"$\mu^{(v)}$", 4.875, 0.0225, "mu1_4.875+mu2_0.0225.npy"),
    Point("offgrid1", r"$\mu^{(1)}$", 4.560, 0.0190, "mu1_4.56+mu2_0.019.npy"),
    Point("offgrid2", r"$\mu^{(2)}$", 5.190, 0.0260, "mu1_5.19+mu2_0.026.npy"),
    Point("extrapolation20pct", r"$\mu^{(3)}$", 4.000, 0.0330, "mu1_4.0+mu2_0.033.npy"),
)

COLORS = {
    "HDM": HDM_COLOR,
    "Linear PROM": METHOD_COLORS["linear"],
    "PROM-ANN C1": METHOD_COLORS["case1"],
    "PROM-ANN C2": METHOD_COLORS["case2_n10"],
    "PROM-ANN C2 n20": METHOD_COLORS["case2_n20"],
    "PROM-ANN C3": METHOD_COLORS["case3"],
    "PROM-POD-AE": METHOD_COLORS["podae"],
    "POD-NN-ROM": METHOD_COLORS["podnn"],
    "POD-DL-ROM": METHOD_COLORS["poddl"],
}

LINE_STYLES = {
    "Linear PROM": METHOD_LINE_STYLES["linear"],
    "PROM-ANN C1": METHOD_LINE_STYLES["case1"],
    "PROM-ANN C2": METHOD_LINE_STYLES["case2_n10"],
    "PROM-ANN C2 n20": METHOD_LINE_STYLES["case2_n20"],
    "PROM-ANN C3": METHOD_LINE_STYLES["case3"],
    "PROM-POD-AE": METHOD_LINE_STYLES["podae"],
    "POD-NN-ROM": METHOD_LINE_STYLES["podnn"],
    "POD-DL-ROM": METHOD_LINE_STYLES["poddl"],
}

EXPECTED_MODELS = {
    "PROM-ANN C1": STAGE3 / "models" / "case1_ann_ntot151_best.pt",
    "PROM-ANN C2": STAGE3 / "models" / "master_ann_mu_t_to_qtot_ntot151_best.pt",
    "PROM-ANN C2 n20": STAGE3 / "models" / "master_ann_mu_t_to_qtot_ntot151_best.pt",
    "PROM-ANN C3": STAGE3 / "models" / "case3_ann_ntot151_best.pt",
    "PROM-POD-AE": STAGE3 / "models" / "prom_pod_ae_ntot151_best.pt",
    "POD-NN-ROM": STAGE3 / "models" / "master_ann_mu_t_to_qtot_ntot151_best.pt",
    "POD-DL-ROM": STAGE3 / "models" / "pod_dl_data_driven_ntot151_best.pt",
}


def mu_tag(p: Point) -> str:
    return f"mu1_{p.mu1:.3f}_mu2_{p.mu2:.4f}"


def read_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def tex_escape(s: object) -> str:
    txt = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        txt = txt.replace(a, b)
    return txt


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def summary_and_snaps(method: str, p: Point) -> tuple[Path, Path | None, Path | None]:
    mt = mu_tag(p)
    if method == "Linear PROM":
        d = RUNS / "Linear" / f"linear_prom_{mt}_ntot151"
        return d / "summary.txt", d / "rom_snaps.npy", d / "qN.npy"
    if method == "PROM-ANN C1":
        d = RUNS / "PROM" / "Case1_Best"
        stem = f"case1_prom_ann_{mt}_n10_ntot151"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", None
    if method == "PROM-ANN C2":
        d = RUNS / "PROM" / "Case2_MasterANN" / "np10"
        stem = f"case2_prom_ann_master_qtot_{mt}_n10_ntot151"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", d / f"{stem}_qN.npy"
    if method == "PROM-ANN C2 n20":
        d = RUNS / "PROM" / "Case2_MasterANN_NSweep" / "np20"
        stem = f"case2_prom_ann_master_qtot_{mt}_n20_ntot151"
        return d / f"{stem}_summary.txt", None, d / f"{stem}_qN.npy"
    if method == "PROM-ANN C3":
        d = RUNS / "PROM" / "Case3_Best"
        stem = f"case3_prom_ann_{mt}_n10_ntot151"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", None
    if method == "PROM-POD-AE":
        d = RUNS / "PROM" / "PODAE_Best"
        stem = f"podae_prom_{mt}_ntot151_nz10"
        return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", d / f"{stem}_qN.npy"
    if method == "POD-NN-ROM":
        d = RUNS / "ROM" / "DataDriven_MasterANN" / f"rom_data_driven_{mt}_ntot151"
        return d / "rom_data_driven_summary.txt", d / "rom_snaps.npy", d / "qN.npy"
    if method == "POD-DL-ROM":
        d = RUNS / "ROM" / "PODDL_Best" / f"pod_dl_data_driven_{mt}_ntot151_nz10"
        return d / "pod_dl_data_driven_summary.txt", d / "rom_snaps.npy", d / "qN.npy"
    raise KeyError(method)


def is_current(method: str, kv: dict[str, str]) -> bool:
    expected = EXPECTED_MODELS.get(method)
    if expected is None:
        return True
    return Path(kv.get("model_path", "")) == expected


def numeric_from_summary(kv: dict[str, str], key: str) -> float | None:
    try:
        return float(kv[key])
    except Exception:
        return None


def hdm_path(p: Point) -> Path:
    candidates = [
        REPO / "Results" / "param_snaps" / p.hdm_file,
        PAPER.parent / "Results" / "param_snaps" / p.hdm_file,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing HDM snapshots for {p.key}: {candidates}")


def _cut_indices(state_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return $u_x$ midline indices in the tensor-product state ordering."""
    n = state_size // 2
    side = int(round(math.sqrt(n)))
    if 2 * n != state_size or side * side != n:
        raise ValueError(f"Cannot infer a square two-component grid from size {state_size}")
    return (side // 2) * side + np.arange(side), np.arange(side) * side + (side // 2)


def state_cut_lines_from_snaps(path: Path, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    idx_x, idx_y = _cut_indices(arr.shape[0])
    return np.asarray(arr[idx_x, tidx], dtype=np.float64), np.asarray(arr[idx_y, tidx], dtype=np.float64)


def state_cut_lines_from_q(q_path: Path, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.load(q_path, mmap_mode="r")
    if q.ndim != 2:
        raise ValueError(f"Expected a coefficient trajectory at {q_path}, found {q.shape}")
    if q.shape[0] != NTOT:
        if q.shape[1] == NTOT:
            q = q.T
        else:
            raise ValueError(f"Unexpected coefficient trajectory shape at {q_path}: {q.shape}")
    V = np.load(BASIS_PATH, mmap_mode="r")
    u_ref = np.load(U_REF_PATH, mmap_mode="r")
    idx_x, idx_y = _cut_indices(u_ref.size)
    q_t = np.asarray(q[:, tidx], dtype=np.float64)
    return (
        np.asarray(u_ref[idx_x] + V[idx_x, :] @ q_t, dtype=np.float64),
        np.asarray(u_ref[idx_y] + V[idx_y, :] @ q_t, dtype=np.float64),
    )


def point_role(p: Point) -> str:
    return {
        "verification": "verification",
        "offgrid1": "off-grid",
        "offgrid2": "off-grid",
        "extrapolation20pct": "extrapolation",
    }[p.key]


def generate_solution_overlay(rows: list[dict[str, object]]) -> Path:
    methods = ["HDM", "Linear PROM", "PROM-ANN C1", "PROM-ANN C2", "PROM-ANN C3", "PROM-POD-AE", "POD-NN-ROM", "POD-DL-ROM"]
    time_ids = (120, 300, 500)
    fig, axes = plt.subplots(len(POINTS), 2, figsize=(12.8, 13.0))
    for row, p in enumerate(POINTS):
        hdm = hdm_path(p)
        xline, yline = state_cut_lines_from_snaps(hdm, time_ids[-1])
        grids = (np.linspace(0.0, 100.0, xline.size), np.linspace(0.0, 100.0, yline.size))
        for column, (ax, grid, cut_label) in enumerate(
            zip(axes[row], grids, (r"$u_x(x,y_{\mathrm{mid}})$", r"$u_x(x_{\mathrm{mid}},y)$"))
        ):
            for tidx in time_ids[:-1]:
                hdm_lines = state_cut_lines_from_snaps(hdm, tidx)
                ax.plot(grid, hdm_lines[column], color=COLORS["HDM"], lw=0.9, alpha=0.22)
            hdm_final = state_cut_lines_from_snaps(hdm, time_ids[-1])[column]
            ax.plot(grid, hdm_final, color=COLORS["HDM"], lw=2.4, label="HDM" if row == 0 and column == 0 else None)
            for method in methods[1:]:
                summary, snaps, qpath = summary_and_snaps(method, p)
                kv = read_kv(summary)
                if not kv or not is_current(method, kv):
                    continue
                line_getter = None
                if snaps is not None and snaps.exists():
                    line_getter = lambda tidx, path=snaps: state_cut_lines_from_snaps(path, tidx)
                elif qpath is not None and qpath.exists():
                    line_getter = lambda tidx, path=qpath: state_cut_lines_from_q(path, tidx)
                if line_getter is None:
                    continue
                for tidx in time_ids[:-1]:
                    ax.plot(grid, line_getter(tidx)[column], color=COLORS[method], lw=0.85, alpha=0.20)
                ax.plot(
                    grid,
                    line_getter(time_ids[-1])[column],
                    color=COLORS[method],
                    lw=1.75,
                    alpha=0.96,
                    label=method if row == 0 and column == 0 else None,
                )
            ax.set_title(rf"{p.label}: $\mu=({p.mu1:.3f},{p.mu2:.4f})$: {point_role(p)}: {cut_label}")
            ax.set_xlabel(r"$x$" if column == 0 else r"$y$")
            ax.set_ylabel(r"$u_x$")
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(*STATE_CUTPLANE_YLIM)
            ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # Preserve full method order while dropping missing duplicates.
    by_label = dict(zip(labels, handles))
    ordered = [(m, by_label[m]) for m in methods if m in by_label]
    fig.legend([h for _, h in ordered], [m for m, _ in ordered], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("PROM campaign: solution cut-plane overlays", y=0.995)
    fig.text(0.5, 0.012, "Fainter solid curves: intermediate times; opaque solid curves: final time.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.93))
    out = FIG_DIR / "prom_only_solution_overlays.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def rel_q(q: np.ndarray, q_ref: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref))


_PROJECTOR_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_RECOVERED_Q_CACHE: dict[Path, np.ndarray] = {}


def projection_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return V, V^T V, and V^T u_ref for coefficient recovery.

    The MLSPG-sensitive basis is not Euclidean-orthonormal, so coefficients
    must be recovered by least squares rather than by V^T projection.
    """

    global _PROJECTOR_CACHE
    if _PROJECTOR_CACHE is None:
        V = np.load(BASIS_PATH, allow_pickle=False)
        u_ref = np.load(U_REF_PATH, allow_pickle=False)
        gram = V.T @ V
        vtu = V.T @ u_ref
        _PROJECTOR_CACHE = (V, gram, vtu)
    return _PROJECTOR_CACHE


def recover_q_from_snaps(snaps_path: Path) -> np.ndarray:
    """Recover linear-basis coefficients from saved PROM state snapshots."""

    snaps_path = snaps_path.resolve()
    if snaps_path in _RECOVERED_Q_CACHE:
        return _RECOVERED_Q_CACHE[snaps_path]
    if not snaps_path.exists():
        raise FileNotFoundError(snaps_path)
    V, gram, vtu = projection_cache()
    snaps = np.load(snaps_path, mmap_mode="r")
    rhs = V.T @ np.asarray(snaps, dtype=np.float64) - vtu[:, None]
    q = np.linalg.solve(gram, rhs)
    _RECOVERED_Q_CACHE[snaps_path] = q
    return q


def online_q_for_method(method: str, p: Point) -> np.ndarray | None:
    summary, snaps, qpath = summary_and_snaps(method, p)
    kv = read_kv(summary)
    if not kv or not is_current(method, kv):
        return None
    if qpath is not None and qpath.exists():
        return np.load(qpath, allow_pickle=False)
    if snaps is not None and snaps.exists():
        return recover_q_from_snaps(snaps)
    return None


def coefficient_errors(methods: list[str]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Compute all coefficient diagnostics from the saved online trajectories."""
    errors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for p in POINTS:
        qref = online_q_for_method("Linear PROM", p)
        if qref is None:
            raise FileNotFoundError(f"Missing linear PROM qN for {p.key}")
        ref_norm = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
        for method in methods:
            q = online_q_for_method(method, p)
            if q is None:
                continue
            error = q - qref
            errors[(p.key, method)] = {
                "abs_curve": np.linalg.norm(error, axis=1),
                "rel_curve": 100.0 * np.linalg.norm(error, axis=1) / ref_norm,
                "abs_heat": np.abs(error),
                "rel_heat": 100.0 * np.abs(error) / ref_norm[:, None],
            }
    return errors


def generate_coeff_error_plot() -> Path:
    methods = [
        "PROM-ANN C1",
        "PROM-ANN C2",
        "PROM-ANN C3",
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-DL-ROM",
    ]
    errors = coefficient_errors(methods)
    fig, axes = plt.subplots(2, len(POINTS), figsize=(16.2, 7.1), sharex=True)
    for column, p in enumerate(POINTS):
        ax_abs, ax_rel = axes[0, column], axes[1, column]
        for method in methods:
            error = errors.get((p.key, method))
            if error is None:
                continue
            for ax, value in ((ax_abs, error["abs_curve"]), (ax_rel, error["rel_curve"])):
                ax.semilogy(
                    np.arange(1, NTOT + 1),
                    np.maximum(value, 1.0e-14),
                    color=COLORS[method],
                    lw=1.8,
                    alpha=0.96,
                    label=method if ax is ax_abs else None,
                )
        for ax in (ax_abs, ax_rel):
            ax.axvline(10, color="#333333", lw=1.0, ls=":", alpha=0.72)
            ax.grid(True, which="both", alpha=0.22)
            ax.set_xlim(1, NTOT)
        ax_abs.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax_abs.set_ylim(*COEFF_ABS_YLIM)
        ax_rel.set_ylim(*COEFF_REL_PERCENT_YLIM)
        ax_rel.set_xlabel("coefficient index")
    axes[0, 0].set_ylabel(r"$\\|q_i-q_i^{\\mathrm{ref}}\\|_2$")
    axes[1, 0].set_ylabel(r"relative coefficient error (\%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = [(name, by_label[name]) for name in methods if name in by_label]
    fig.legend([h for _, h in ordered], [m for m, _ in ordered], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.05, h_pad=0.8)
    out = FIG_DIR / "prom_only_coeff_abs_rel_errors.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def generate_coefficient_heatmaps() -> list[Path]:
    methods = [
        "PROM-ANN C1",
        "PROM-ANN C2",
        "PROM-ANN C3",
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-DL-ROM",
    ]
    errors = coefficient_errors(methods)
    outputs: list[Path] = []
    for kind, vmax, label, stem in (
        ("abs_heat", COEFF_ABS_HEAT_VMAX, r"$|q_i-q_i^{\\mathrm{ref}}|$", "abs"),
        ("rel_heat", COEFF_REL_PERCENT_HEAT_VMAX, r"relative coefficient error (\%)", "rel"),
    ):
        fig, axes = plt.subplots(len(methods), len(POINTS), figsize=(15.6, 10.9), sharex=True, sharey=True)
        image = None
        for row, method in enumerate(methods):
            for column, p in enumerate(POINTS):
                ax = axes[row, column]
                error = errors.get((p.key, method))
                if error is None:
                    ax.set_axis_off()
                    continue
                image = ax.imshow(
                    error[kind], origin="lower", aspect="auto", interpolation="nearest",
                    extent=(0.0, 25.0, 1.0, float(NTOT)), cmap="viridis", vmin=0.0, vmax=vmax,
                )
                ax.axhline(10.5, color="white", linestyle=":", linewidth=0.75, alpha=0.82)
                if row == 0:
                    ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$", pad=5)
                if column == 0:
                    ax.set_ylabel(method)
                if row == len(methods) - 1:
                    ax.set_xlabel("time")
                ax.grid(False)
        fig.subplots_adjust(left=0.17, right=0.88, bottom=0.07, top=0.93, wspace=0.15, hspace=0.22)
        fig.supylabel("coefficient index", x=0.045)
        cax = fig.add_axes([0.905, 0.15, 0.017, 0.68])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(label)
        out = FIG_DIR / f"prom_only_coeff_{stem}_heatmaps.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def generate_case2_n10_n20_coeff_plot() -> Path:
    methods = [
        ("POD-NN-ROM", "POD-NN-ROM ($n=0$)", METHOD_COLORS["podnn"], "-"),
        ("PROM-ANN C2", "Case 2 ($n=10$)", METHOD_COLORS["case2_n10"], "-"),
        ("PROM-ANN C2 n20", "Case 2 ($n=20$)", METHOD_COLORS["case2_n20"], "-"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), POINTS):
        qref = online_q_for_method("Linear PROM", p)
        if qref is None:
            raise FileNotFoundError(f"Missing linear PROM qN for {p.key}")
        for method, label, color, ls in methods:
            q = online_q_for_method(method, p)
            if q is None:
                continue
            denom = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
            rel = 100.0 * np.linalg.norm(q - qref, axis=1) / denom
            ax.semilogy(np.arange(1, NTOT + 1), rel, color=color, lw=1.9, ls=ls, alpha=0.98, label=label)
        ax.axvline(10, color="#444444", lw=1.0, ls=":", alpha=0.65)
        ax.axvline(20, color="#444444", lw=1.0, ls=":", alpha=0.75)
        ax.text(10.5, 1.5e-3, r"$n=10$", rotation=90, va="bottom", ha="left", fontsize=7, color="#444444")
        ax.text(20.5, 1.5e-3, r"$n=20$", rotation=90, va="bottom", ha="left", fontsize=7, color="#444444")
        ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("coefficient index")
        ax.set_ylabel(r"relative coefficient error (\%)")
        ax.set_ylim(1e-3, 2e2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [label for _, label, _, _ in methods if label in by_label]
    fig.legend([by_label[x] for x in ordered_labels], ordered_labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "prom_only_case2_n10_n20_vs_podnn_coeff_rel_errors.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def copy_existing_figures() -> dict[str, Path]:
    copied: dict[str, Path] = {}
    sources = {
        "case2_n_sweep_state": DIAG / "figures" / "prom_case2_n_sweep_state_errors.png",
        "case2_n_sweep_coeff": DIAG / "figures" / "prom_case2_n_sweep_coeff_abs_rel_all_points.png",
        "case2_secondary_sensitivity": DIAG / "figures" / "case2_secondary_sensitivity_state_and_primary_error.png",
    }
    for key, src in sources.items():
        if src.exists():
            dst = FIG_DIR / src.name
            shutil.copy2(src, dst)
            copied[key] = dst
    # Four-panel image of the coefficient reconstruction overview diagnostics.
    overview_paths = [DIAG / "prom151_case1_dd_case3_podae_poddl_coeff_traces_4pts" / p.key / "overview_coeff_errors.png" for p in POINTS]
    if all(x.exists() for x in overview_paths):
        imgs = [plt.imread(x) for x in overview_paths]
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
        for ax, img, p in zip(axes.ravel(), imgs, POINTS):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$", pad=3)
        fig.tight_layout()
        dst = FIG_DIR / "prom_only_offline_coeff_reconstruction_overviews.png"
        fig.savefig(dst, dpi=180)
        plt.close(fig)
        copied["offline_coeff_overview"] = dst
    return copied


def collect_online_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = [
        "Linear PROM",
        "PROM-ANN C1",
        "PROM-ANN C2",
        "PROM-ANN C2 n20",
        "PROM-ANN C3",
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-DL-ROM",
    ]
    for method in methods:
        for p in POINTS:
            summary, _, _ = summary_and_snaps(method, p)
            kv = read_kv(summary)
            ok = bool(kv) and is_current(method, kv)
            err = numeric_from_summary(kv, "relative_error_percent") if ok else None
            rows.append({"method": method, "point": p.key, "label": p.label, "err": err, "ok": ok})
    return rows


def method_summary(rows: list[dict[str, object]], method: str) -> tuple[list[float | None], bool]:
    vals = [next(r for r in rows if r["method"] == method and r["point"] == p.key)["err"] for p in POINTS]
    ok = all(next(r for r in rows if r["method"] == method and r["point"] == p.key)["ok"] for p in POINTS)
    return vals, ok


def write_online_table(rows: list[dict[str, object]]) -> Path:
    methods = [
        "Linear PROM",
        "PROM-ANN C1",
        "PROM-ANN C2",
        "PROM-ANN C2 n20",
        "PROM-ANN C3",
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-DL-ROM",
    ]
    labels = {
        "Linear PROM": "Linear PROM",
        "PROM-ANN C1": "PROM--ANN Case 1",
        "PROM-ANN C2": "PROM--ANN Case 2 ($n=10$)",
        "PROM-ANN C2 n20": "PROM--ANN Case 2 ($n=20$)",
        "PROM-ANN C3": "PROM--ANN Case 3",
        "PROM-POD-AE": "PROM--POD--AE ($n_z=10$)",
        "POD-NN-ROM": "POD--NN--ROM",
        "POD-DL-ROM": "POD--DL--ROM ($n_z=10$)",
    }
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for method in methods:
        if method == "POD-NN-ROM":
            # Direct maps share the coefficient teacher, but evaluate no
            # residual. Keep the same visual/table separation as HPROM.
            lines.append(r"\midrule")
        vals, ok = method_summary(rows, method)
        mean = None if any(v is None for v in vals[:3]) else sum(float(v) for v in vals[:3]) / 3.0
        row = [labels[method], *(fmt(v, 3) for v in vals[:3]), fmt(mean, 3), fmt(vals[3], 3)]
        if not ok:
            row[0] += r"$^{\dagger}$"
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "prom_only_online_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_online_coeff_table() -> Path:
    methods = ["Linear PROM", "PROM-ANN C1", "PROM-ANN C2", "PROM-ANN C2 n20", "PROM-ANN C3", "PROM-POD-AE", "POD-NN-ROM", "POD-DL-ROM"]
    labels = {
        "Linear PROM": "Linear PROM",
        "PROM-ANN C1": "PROM--ANN Case 1",
        "PROM-ANN C2": "PROM--ANN Case 2 ($n=10$)",
        "PROM-ANN C2 n20": "PROM--ANN Case 2 ($n=20$)",
        "PROM-ANN C3": "PROM--ANN Case 3",
        "PROM-POD-AE": "PROM--POD--AE",
        "POD-NN-ROM": "POD--NN--ROM",
        "POD-DL-ROM": "POD--DL--ROM",
    }
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for method in methods:
        if method == "POD-NN-ROM":
            lines.append(r"\midrule")
        values: list[float | None] = []
        for p in POINTS:
            qref = online_q_for_method("Linear PROM", p)
            if qref is None:
                raise FileNotFoundError(f"Missing linear PROM qN for {p.key}")
            q = online_q_for_method(method, p)
            values.append(0.0 if method == "Linear PROM" else (rel_q(q, qref) if q is not None else None))
        in_domain = [value for value in values[:3] if value is not None]
        mean = float(np.mean(in_domain)) if len(in_domain) == 3 else None
        lines.append(
            f"{labels[method]} & {fmt(values[0], 3)} & {fmt(values[1], 3)} & "
            f"{fmt(values[2], 3)} & {fmt(mean, 3)} & {fmt(values[3], 3)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "prom_only_online_coeff_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def stage3_value(path: Path, key: str) -> str:
    return read_kv(path).get(key, "--")


def write_training_table() -> Path:
    specs = [
        ("PROM--ANN Case 1", "$\\q_p\\mapsto\\q_s$; $10\\to256\\to512\\to512\\to256\\to141$", STAGE3 / "case1_ann_ntot151_best_summary.txt"),
        ("Master POD--NN--ROM (Case 2 tail source)", "$(\\mu_1,\\mu_2,t)\\mapsto\\q_{151}$; $3\\to256\\to512\\to512\\to256\\to151$", STAGE3 / "master_ann_mu_t_to_qtot_ntot151_best_summary.txt"),
        ("PROM--ANN Case 3", "$(\\q_p,\\mu_1,\\mu_2,t)\\mapsto\\q_s$; $13\\to256\\to512\\to512\\to256\\to141$", STAGE3 / "case3_ann_ntot151_best_summary.txt"),
        ("PROM--POD--AE", "z-score AE; $151\\to512\\to256\\to128\\to10\\to128\\to256\\to512\\to151$", STAGE3 / "prom_pod_ae_ntot151_best_summary.txt"),
        ("POD--DL--ROM", "z-score latent dynamics; $3\\to256\\to512\\to512\\to256\\to10\\to151$", STAGE3 / "pod_dl_data_driven_ntot151_best_summary.txt"),
    ]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Map / network & Act. & Train $e_q$ (\%) & Val. $e_q$ (\%) & Params \\",
        r"\midrule",
    ]
    for model, mapping, path in specs:
        kv = read_kv(path)
        params = kv.get("trainable_parameters", "--")
        activation = {
            "silu": "SiLU",
            "elu": "ELU",
            "gelu": "GELU",
        }.get(kv.get("activation", "--").lower(), kv.get("activation", "--"))
        train = kv.get("train_rel_frob_percent", "--")
        val = kv.get("val_rel_frob_percent", "--")
        try:
            params_txt = f"{int(float(params)):,}"
        except Exception:
            params_txt = "--"
        lines.append(
            f"{model} & {mapping} & {activation} & "
            f"{fmt(float(train), 3) if train != '--' else '--'} & "
            f"{fmt(float(val), 3) if val != '--' else '--'} & {params_txt}"
            r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "prom_only_training_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_offline_coeff_table() -> Path | None:
    src = DIAG / "prom151_case1_dd_case3_podae_poddl_coeff_traces_4pts" / "all_points_global_summary.csv"
    if not src.exists():
        return None
    rows = list(csv.DictReader(src.open()))
    # One row per evaluation point; columns store the method-wise global errors.
    method_cols = [
        ("case1_global_rel_q_percent", "Case 1"),
        ("dd_case2_global_rel_q_percent", "POD--NN--ROM"),
        ("case3_global_rel_q_percent", "Case 3"),
        ("pod_ae_global_rel_q_percent", "POD--AE"),
        ("pod_dl_global_rel_q_percent", "POD--DL"),
    ]
    point_label = {p.key: p.label for p in POINTS}
    by = {r["label"]: r for r in rows}
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Point & " + " & ".join(name for _, name in method_cols) + r" \\",
        r"\midrule",
    ]
    for p in POINTS:
        row = by.get(p.key, {})
        vals = [fmt(float(row[col]), 3) if col in row else "--" for col, _ in method_cols]
        lines.append(f"{point_label[p.key]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "prom_only_offline_coeff_reconstruction.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_online_rows()
    written = []
    written.append(write_training_table())
    written.append(write_online_table(rows))
    written.append(write_online_coeff_table())
    coeff_tab = write_offline_coeff_table()
    if coeff_tab is not None:
        written.append(coeff_tab)
    figs = []
    figs.append(generate_solution_overlay(rows))
    figs.append(generate_coeff_error_plot())
    figs.extend(generate_coefficient_heatmaps())
    figs.append(generate_case2_n10_n20_coeff_plot())
    figs.extend(copy_existing_figures().values())
    print("[prom-only-assets] tables:")
    for p in written:
        print(f"  {p}")
    print("[prom-only-assets] figures:")
    for p in figs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
