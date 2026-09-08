#!/usr/bin/env python3
"""Build figures and tables for the clean Euclidean-POD PROM manuscript.

The input is the reproducible ``euclidean_prom_main`` campaign only.  This
script never runs a solver and deliberately writes into dedicated Euclidean
folders, so historic LSPG-sensitive and HPROM material cannot leak into the
new manuscript.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manuscript_plot_style import (
    CASE2_SWEEP_COLORS,
    COEFF_ABS_HEAT_VMAX,
    COEFF_ABS_YLIM,
    COEFF_REL_PERCENT_HEAT_VMAX,
    COEFF_REL_PERCENT_YLIM,
    HDM_COLOR,
    METHOD_COLORS,
    STATE_CUTPLANE_YLIM,
)


SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
REPO = PAPER.parents[1]
CAMPAIGN = PAPER / "euclidean_prom_main"
RUNS = CAMPAIGN / "Runs"
STAGE3 = CAMPAIGN / "Stage3"
DIAGNOSTIC = PAPER / "tmp_euclidean_case2_secondary_sensitivity"
BASIS_PATH = PAPER / "MetricStudy" / "euclidean" / "Stage1" / "basis.npy"
U_REF_PATH = PAPER / "MetricStudy" / "euclidean" / "Stage1" / "u_ref.npy"
FIG_DIR = PAPER / "Figures" / "euclidean_prom_only"
TAB_DIR = PAPER / "tables" / "euclidean_prom_only"
NTOT = 151
NPRIMARY = 10

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    }
)


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

METHODS = (
    "Linear PROM",
    "PROM-ANN C1",
    "PROM-ANN C2",
    "PROM-ANN C3",
    "PROM-POD-AE",
    "POD-NN-ROM",
    "POD-DL-ROM",
)

METHOD_LABELS = {
    "Linear PROM": "Linear PROM",
    "PROM-ANN C1": "PROM--ANN Case 1",
    "PROM-ANN C2": "PROM--ANN Case 2 ($n=10$)",
    "PROM-ANN C3": "PROM--ANN Case 3",
    "PROM-POD-AE": "PROM--POD--AE ($n_z=10$)",
    "POD-NN-ROM": "POD--NN--ROM",
    "POD-DL-ROM": "POD--DL--ROM ($n_z=10$)",
}

PLOT_LABELS = {
    "Linear PROM": "Linear PROM",
    "PROM-ANN C1": "PROM-ANN C1",
    "PROM-ANN C2": "PROM-ANN C2 ($n=10$)",
    "PROM-ANN C3": "PROM-ANN C3",
    "PROM-POD-AE": "PROM-POD-AE",
    "POD-NN-ROM": "POD-NN-ROM",
    "POD-DL-ROM": "POD-DL-ROM",
}

COLORS = {
    "HDM": HDM_COLOR,
    "Linear PROM": METHOD_COLORS["linear"],
    "PROM-ANN C1": METHOD_COLORS["case1"],
    "PROM-ANN C2": METHOD_COLORS["case2_n10"],
    "PROM-ANN C3": METHOD_COLORS["case3"],
    "PROM-POD-AE": METHOD_COLORS["podae"],
    "POD-NN-ROM": METHOD_COLORS["podnn"],
    "POD-DL-ROM": METHOD_COLORS["poddl"],
}


def mu_tag(p: Point) -> str:
    return f"mu1_{p.mu1:.3f}_mu2_{p.mu2:.4f}"


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def summary_value(path: Path, key: str) -> float:
    try:
        return float(read_kv(path)[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing numeric key {key!r} in {path}") from exc


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "--"


def method_paths(method: str, p: Point) -> tuple[Path, Path | None, Path | None]:
    """Return summary, snapshots, and reduced coordinates for one run."""

    tag = mu_tag(p)
    if method == "Linear PROM":
        root = RUNS / "Linear" / f"linear_prom_{tag}_ntot151"
        return root / "summary.txt", root / "rom_snaps.npy", root / "qN.npy"
    if method == "PROM-ANN C1":
        root = RUNS / "PROM" / "Case1_Best"
        stem = f"case1_prom_ann_{tag}_n10_ntot151"
        return root / f"{stem}_summary.txt", root / f"{stem}_snaps.npy", None
    if method == "PROM-ANN C2":
        root = RUNS / "PROM" / "Case2_MasterANN" / "np10"
        stem = f"case2_prom_ann_master_qtot_{tag}_n10_ntot151"
        return root / f"{stem}_summary.txt", root / f"{stem}_snaps.npy", root / f"{stem}_qN.npy"
    if method == "PROM-ANN C3":
        root = RUNS / "PROM" / "Case3_Best"
        stem = f"case3_prom_ann_{tag}_n10_ntot151"
        return root / f"{stem}_summary.txt", root / f"{stem}_snaps.npy", None
    if method == "PROM-POD-AE":
        root = RUNS / "PROM" / "PODAE_Best"
        stem = f"podae_prom_{tag}_ntot151_nz10"
        return root / f"{stem}_summary.txt", root / f"{stem}_snaps.npy", root / f"{stem}_qN.npy"
    if method == "POD-NN-ROM":
        root = RUNS / "ROM" / "DataDriven_MasterANN" / f"rom_data_driven_{tag}_ntot151"
        return root / "rom_data_driven_summary.txt", root / "rom_snaps.npy", root / "qN.npy"
    if method == "POD-DL-ROM":
        root = RUNS / "ROM" / "PODDL_Best" / f"pod_dl_data_driven_{tag}_ntot151_nz10"
        return root / "pod_dl_data_driven_summary.txt", root / "rom_snaps.npy", root / "qN.npy"
    raise KeyError(method)


def sweep_paths(n_primary: int, p: Point) -> tuple[Path, Path]:
    """Return summary and q trajectory for the shared-master Case-2 sweep."""

    if n_primary == 0:
        summary, _, qpath = method_paths("POD-NN-ROM", p)
    elif n_primary == 10:
        summary, _, qpath = method_paths("PROM-ANN C2", p)
    elif n_primary == NTOT:
        summary, _, qpath = method_paths("Linear PROM", p)
    else:
        root = RUNS / "PROM" / "Case2_MasterANN_NSweep" / f"np{n_primary}"
        stem = f"case2_prom_ann_master_qtot_{mu_tag(p)}_n{n_primary}_ntot151"
        summary, qpath = root / f"{stem}_summary.txt", root / f"{stem}_qN.npy"
    if qpath is None:
        raise FileNotFoundError(f"No coordinate path configured for n={n_primary}, {p.key}")
    return summary, qpath


def hdm_path(p: Point) -> Path:
    candidate = REPO / "Results" / "param_snaps" / p.hdm_file
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def normalize_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"Expected two-dimensional q trajectory, found {q.shape}")
    if q.shape[0] == NTOT:
        return q
    if q.shape[1] == NTOT:
        return q.T
    raise ValueError(f"Expected one q dimension to equal {NTOT}, found {q.shape}")


_V: np.ndarray | None = None
_U_REF: np.ndarray | None = None
_Q_FROM_SNAPS: dict[Path, np.ndarray] = {}


def basis_and_reference() -> tuple[np.ndarray, np.ndarray]:
    global _V, _U_REF
    if _V is None:
        _V = np.load(BASIS_PATH, allow_pickle=False)
        _U_REF = np.load(U_REF_PATH, allow_pickle=False)
        orth_error = np.linalg.norm(_V.T @ _V - np.eye(NTOT), ord=np.inf)
        if orth_error > 1.0e-8:
            raise ValueError(f"Euclidean POD basis is not orthonormal: {orth_error:.3e}")
    return _V, _U_REF


def q_from_snaps(path: Path) -> np.ndarray:
    path = path.resolve()
    if path not in _Q_FROM_SNAPS:
        basis, u_ref = basis_and_reference()
        snaps = np.load(path, mmap_mode="r")
        _Q_FROM_SNAPS[path] = basis.T @ (np.asarray(snaps, dtype=np.float64) - u_ref[:, None])
    return _Q_FROM_SNAPS[path]


def q_for_method(method: str, p: Point) -> np.ndarray:
    _, snaps, qpath = method_paths(method, p)
    if qpath is not None and qpath.exists():
        return normalize_q(np.load(qpath, allow_pickle=False))
    if snaps is not None and snaps.exists():
        return q_from_snaps(snaps)
    raise FileNotFoundError(f"Missing result arrays for {method} at {p.key}")


def q_for_sweep(n_primary: int, p: Point) -> np.ndarray:
    _, qpath = sweep_paths(n_primary, p)
    return normalize_q(np.load(qpath, allow_pickle=False))


def cut_indices(state_size: int) -> tuple[np.ndarray, np.ndarray]:
    n_scalar = state_size // 2
    side = int(round(math.sqrt(n_scalar)))
    if 2 * n_scalar != state_size or side * side != n_scalar:
        raise ValueError(f"Cannot infer a square two-field grid from state size {state_size}")
    return (side // 2) * side + np.arange(side), np.arange(side) * side + (side // 2)


def cut_from_snaps(path: Path, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    snaps = np.load(path, mmap_mode="r")
    ix, iy = cut_indices(snaps.shape[0])
    return np.asarray(snaps[ix, time_index]), np.asarray(snaps[iy, time_index])


def cut_from_q(q: np.ndarray, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    basis, u_ref = basis_and_reference()
    ix, iy = cut_indices(u_ref.size)
    return u_ref[ix] + basis[ix] @ q[:, time_index], u_ref[iy] + basis[iy] @ q[:, time_index]


def cuts_for_method(method: str, p: Point, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    _, snaps, qpath = method_paths(method, p)
    if snaps is not None and snaps.exists():
        return cut_from_snaps(snaps, time_index)
    if qpath is not None and qpath.exists():
        return cut_from_q(q_for_method(method, p), time_index)
    raise FileNotFoundError(f"Missing state representation for {method} at {p.key}")


def state_errors() -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for method in METHODS:
        for point in POINTS:
            summary, _, _ = method_paths(method, point)
            values[(method, point.key)] = summary_value(summary, "relative_error_percent")
    return values


def coefficient_errors(methods: tuple[str, ...] | list[str]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for point in POINTS:
        qref = q_for_method("Linear PROM", point)
        denom = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
        for method in methods:
            q = q_for_method(method, point)
            diff = q - qref
            data[(method, point.key)] = {
                "abs_curve": np.linalg.norm(diff, axis=1),
                "rel_curve": 100.0 * np.linalg.norm(diff, axis=1) / denom,
                "abs_heat": np.abs(diff),
                "rel_heat": 100.0 * np.abs(diff) / denom[:, None],
            }
    return data


def make_parameter_space_plot() -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    mu1_grid = (4.25, 4.875, 5.50)
    mu2_grid = (0.015, 0.0225, 0.030)
    train_x, train_y = np.meshgrid(mu1_grid, mu2_grid)
    ax.scatter(train_x.ravel(), train_y.ravel(), s=72, color="black", label="baseline PROM training (9)", zorder=3)
    validation = ((4.5625, 0.02625), (5.1875, 0.01875))
    ax.scatter(*zip(*validation), s=92, marker="D", facecolor="#9467BD", edgecolor="white", linewidth=0.7, label="held-out PROM validation (2)", zorder=4)
    ax.scatter(
        [point.mu1 for point in POINTS],
        [point.mu2 for point in POINTS],
        s=132,
        marker="*",
        color="#D62728",
        edgecolor="white",
        linewidth=0.65,
        label="online evaluation (4)",
        zorder=5,
    )
    for point in POINTS:
        offset = (8, -15) if point.key != "extrapolation20pct" else (8, 2)
        ax.annotate(point.label, (point.mu1, point.mu2), textcoords="offset points", xytext=offset, color="#A31B1B", fontsize=9)
    ax.set_title("Euclidean-POD PROM campaign in parameter space")
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_xlim(3.72, 5.78)
    ax.set_ylim(0.012, 0.0348)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "euclidean_prom_parameter_space.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_solution_overlay() -> Path:
    methods = ("HDM",) + METHODS
    time_indices = (120, 300, 500)
    fig, axes = plt.subplots(len(POINTS), 2, figsize=(10.6, 13.2))
    for row, point in enumerate(POINTS):
        hdm = hdm_path(point)
        initial = cut_from_snaps(hdm, time_indices[-1])
        grids = (np.linspace(0.0, 100.0, initial[0].size), np.linspace(0.0, 100.0, initial[1].size))
        for col, ax in enumerate(axes[row]):
            for tidx in time_indices[:-1]:
                cut = cut_from_snaps(hdm, tidx)[col]
                ax.plot(grids[col], cut, color=HDM_COLOR, lw=0.8, alpha=0.20)
            ax.plot(grids[col], cut_from_snaps(hdm, time_indices[-1])[col], color=HDM_COLOR, lw=2.3, label="HDM" if row == 0 and col == 0 else None)
            for method in METHODS:
                for tidx in time_indices[:-1]:
                    ax.plot(grids[col], cuts_for_method(method, point, tidx)[col], color=COLORS[method], lw=0.75, alpha=0.18)
                ax.plot(
                    grids[col],
                    cuts_for_method(method, point, time_indices[-1])[col],
                    color=COLORS[method],
                    lw=1.55,
                    label=PLOT_LABELS[method] if row == 0 and col == 0 else None,
                )
            direction = r"$u_x(x,y_{\mathrm{mid}})$" if col == 0 else r"$u_x(x_{\mathrm{mid}},y)$"
            role = "verification" if point.key == "verification" else ("extrapolation" if point.key == "extrapolation20pct" else "off-grid")
            ax.set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$; {role}: {direction}")
            ax.set_xlabel(r"$x$" if col == 0 else r"$y$")
            ax.set_ylabel(r"$u_x$")
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(*STATE_CUTPLANE_YLIM)
            ax.grid(True, alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Euclidean-POD PROM campaign: solution cut-plane overlays", y=0.998)
    fig.text(0.5, 0.012, "Fainter solid curves: intermediate times; opaque solid curves: final time.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))
    out = FIG_DIR / "euclidean_prom_solution_overlays.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_coefficient_curves() -> Path:
    methods = METHODS[1:]
    errors = coefficient_errors(methods)
    fig, axes = plt.subplots(2, len(POINTS), figsize=(10.6, 5.5), sharex=True)
    for col, point in enumerate(POINTS):
        for method in methods:
            err = errors[(method, point.key)]
            axes[0, col].semilogy(np.arange(1, NTOT + 1), np.maximum(err["abs_curve"], 1.0e-14), color=COLORS[method], lw=1.65, label=PLOT_LABELS[method])
            axes[1, col].semilogy(np.arange(1, NTOT + 1), np.maximum(err["rel_curve"], 1.0e-14), color=COLORS[method], lw=1.65)
        for ax in axes[:, col]:
            ax.axvline(NPRIMARY, color="#333333", lw=1.0, ls=":", alpha=0.75)
            ax.set_xlim(1, NTOT)
            ax.grid(True, which="both", alpha=0.22)
        axes[0, col].set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$")
        axes[0, col].set_ylim(*COEFF_ABS_YLIM)
        axes[1, col].set_ylim(*COEFF_REL_PERCENT_YLIM)
        axes[1, col].set_xlabel("coefficient index $i$")
    axes[0, 0].set_ylabel(r"$\|q_i-q_i^{\mathrm{lin}}\|_2$")
    axes[1, 0].set_ylabel(r"relative coefficient error (\%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=0.75, h_pad=0.7)
    out = FIG_DIR / "euclidean_prom_coefficient_abs_rel_errors.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_coefficient_heatmaps() -> list[Path]:
    methods = METHODS[1:]
    errors = coefficient_errors(methods)
    results: list[Path] = []
    for key, vmax, colorbar_label, name in (
        ("abs_heat", COEFF_ABS_HEAT_VMAX, r"$|q_i-q_i^{\mathrm{lin}}|$", "abs"),
        ("rel_heat", COEFF_REL_PERCENT_HEAT_VMAX, r"relative coefficient error (\%)", "rel"),
    ):
        fig, axes = plt.subplots(len(methods), len(POINTS), figsize=(10.6, 10.9), sharex=True, sharey=True)
        image = None
        for row, method in enumerate(methods):
            for col, point in enumerate(POINTS):
                ax = axes[row, col]
                image = ax.imshow(
                    errors[(method, point.key)][key],
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    extent=(0.0, 25.0, 1.0, float(NTOT)),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=vmax,
                )
                ax.axhline(NPRIMARY + 0.5, color="white", lw=0.75, ls=":", alpha=0.85)
                if row == 0:
                    ax.set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$", pad=5)
                if col == 0:
                    ax.set_ylabel(PLOT_LABELS[method])
                if row == len(methods) - 1:
                    ax.set_xlabel("time")
                ax.grid(False)
        fig.subplots_adjust(left=0.19, right=0.88, bottom=0.07, top=0.93, wspace=0.15, hspace=0.22)
        fig.supylabel("coefficient index", x=0.045)
        cbar = fig.colorbar(image, cax=fig.add_axes([0.905, 0.15, 0.018, 0.68]))
        cbar.set_label(colorbar_label)
        out = FIG_DIR / f"euclidean_prom_coefficient_{name}_heatmaps.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        results.append(out)
    return results


def make_sweep_state_plot() -> Path:
    sweep = (0, 10, 20, 30, 50, 100, NTOT)
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    values: dict[str, list[float]] = {point.key: [] for point in POINTS}
    for n_primary in sweep:
        for point in POINTS:
            summary, _ = sweep_paths(n_primary, point)
            values[point.key].append(summary_value(summary, "relative_error_percent"))
    for point in POINTS:
        ax.plot(sweep, values[point.key], marker="o", lw=1.9, ms=5.2, label=rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$")
    in_domain = np.mean(np.array([values[p.key] for p in POINTS[:3]]), axis=0)
    ax.plot(sweep, in_domain, color="black", marker="s", lw=2.4, ms=5.4, label="in-domain mean")
    ax.axvline(NPRIMARY, color="#555555", lw=1.0, ls=":", alpha=0.7)
    ax.set_xticks(sweep)
    ax.set_xlabel(r"residual-solved primary dimension $n$")
    ax.set_ylabel("state relative error against HDM (%)")
    ax.set_title("Case-2 PROM state error as the solved dimension is varied")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "euclidean_prom_case2_n_sweep_state_errors.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_sweep_coefficient_plot() -> Path:
    sweep = (0, 10, 20, 30, 50, 100)
    fig, axes = plt.subplots(2, len(POINTS), figsize=(10.6, 5.5), sharex=True)
    for col, point in enumerate(POINTS):
        qref = q_for_method("Linear PROM", point)
        denom = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
        for n_primary in sweep:
            q = q_for_sweep(n_primary, point)
            difference = q - qref
            label = "POD-NN-ROM ($n=0$)" if n_primary == 0 else rf"PROM-ANN C2 ($n={n_primary}$)"
            color = CASE2_SWEEP_COLORS[n_primary]
            axes[0, col].semilogy(np.arange(1, NTOT + 1), np.maximum(np.linalg.norm(difference, axis=1), 1.0e-14), color=color, lw=1.55, label=label)
            axes[1, col].semilogy(np.arange(1, NTOT + 1), np.maximum(100.0 * np.linalg.norm(difference, axis=1) / denom, 1.0e-14), color=color, lw=1.55)
        for ax in axes[:, col]:
            ax.axvline(NPRIMARY, color="#333333", lw=1.0, ls=":", alpha=0.75)
            ax.set_xlim(1, NTOT)
            ax.grid(True, which="both", alpha=0.22)
        axes[0, col].set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$")
        axes[0, col].set_ylim(*COEFF_ABS_YLIM)
        axes[1, col].set_ylim(*COEFF_REL_PERCENT_YLIM)
        axes[1, col].set_xlabel("coefficient index $i$")
    axes[0, 0].set_ylabel(r"$\|q_i-q_i^{\mathrm{lin}}\|_2$")
    axes[1, 0].set_ylabel(r"relative coefficient error (\%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=0.75, h_pad=0.7)
    out = FIG_DIR / "euclidean_prom_case2_n_sweep_coefficient_errors.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_tail_sensitivity_plot() -> Path:
    levels = {0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0}
    rows = [
        row
        for row in csv.DictReader((DIAGNOSTIC / "case2_secondary_sensitivity_summary.csv").open())
        if float(row["requested_secondary_error_percent"]) in levels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharex=True)
    for point in POINTS:
        point_rows = sorted((row for row in rows if row["point"] == point.key), key=lambda row: float(row["requested_secondary_error_percent"]))
        x = [float(row["actual_secondary_error_percent"]) for row in point_rows]
        state = [float(row["state_error_percent_vs_hdm"]) for row in point_rows]
        primary = [float(row["primary_q_error_percent_vs_linear_prom"]) for row in point_rows]
        label = rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$"
        axes[0].plot(x, state, marker="o", ms=4.2, lw=1.8, label=label)
        axes[1].plot(x, primary, marker="o", ms=4.2, lw=1.8, label=label)
    axes[0].set_title(r"effect on state error $\|u_{\mathrm{HDM}}-u\|_F/\|u_{\mathrm{HDM}}\|_F$")
    axes[1].set_title(r"effect on solved coordinates $q_1,\ldots,q_{10}$")
    for ax in axes:
        ax.set_xlabel(r"imposed relative error in Case-2 secondary coordinates (\%)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("relative error (%)")
    axes[1].set_ylabel("primary-coordinate error vs linear PROM (%)")
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Euclidean-POD Case-2 tail-perturbation diagnostic ($n=10$)", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "euclidean_prom_case2_tail_sensitivity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_training_table() -> Path:
    specifications = (
        ("PROM--ANN Case 1", r"$\q_p\mapsto\q_s$", STAGE3 / "case1_ann_ntot151_best_summary.txt"),
        ("Master map (Case 2 / POD--NN--ROM)", r"$(\bm\mu,t)\mapsto\q_{151}$", STAGE3 / "master_ann_mu_t_to_qtot_ntot151_best_summary.txt"),
        ("PROM--ANN Case 3", r"$(\q_p,\bm\mu,t)\mapsto\q_s$", STAGE3 / "case3_ann_ntot151_best_summary.txt"),
        ("PROM--POD--AE", r"$\q_{151}\mapsto\z\mapsto\widehat{\q}_{151}$", STAGE3 / "prom_pod_ae_ntot151_best_summary.txt"),
        ("POD--DL--ROM", r"$(\bm\mu,t)\mapsto\z\mapsto\widehat{\q}_{151}$", STAGE3 / "pod_dl_data_driven_ntot151_best_summary.txt"),
    )
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Learned map & Train $e_q$ (\%) & Validation $e_q$ (\%) & Parameters \\",
        r"\midrule",
    ]
    for model, mapping, path in specifications:
        data = read_kv(path)
        count = f"{int(float(data['trainable_parameters'])):,}"
        lines.append(
            f"{model} & {mapping} & {fmt(float(data['train_rel_frob_percent']))} & "
            f"{fmt(float(data['val_rel_frob_percent']))} & {count} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    out = TAB_DIR / "euclidean_prom_training_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_state_table(errors: dict[tuple[str, str], float]) -> Path:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        if method == "POD-NN-ROM":
            lines.append(r"\midrule")
        values = [errors[(method, point.key)] for point in POINTS]
        mean = float(np.mean(values[:3]))
        lines.append(" & ".join((METHOD_LABELS[method], *(fmt(value) for value in values[:3]), fmt(mean), fmt(values[3]))) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    out = TAB_DIR / "euclidean_prom_online_state_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_coefficient_table() -> Path:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        if method == "POD-NN-ROM":
            lines.append(r"\midrule")
        values = []
        for point in POINTS:
            qref = q_for_method("Linear PROM", point)
            q = q_for_method(method, point)
            values.append(100.0 * np.linalg.norm(q - qref) / np.maximum(np.linalg.norm(qref), 1.0e-14))
        lines.append(" & ".join((METHOD_LABELS[method], *(fmt(value) for value in values[:3]), fmt(float(np.mean(values[:3]))), fmt(values[3]))) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    out = TAB_DIR / "euclidean_prom_online_coefficient_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_sweep_table() -> Path:
    sweep = (0, 10, 20, 30, 50, 100, NTOT)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Solved $n$ & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for n_primary in sweep:
        values = [summary_value(sweep_paths(n_primary, point)[0], "relative_error_percent") for point in POINTS]
        name = "0 (POD--NN--ROM)" if n_primary == 0 else ("151 (linear PROM)" if n_primary == NTOT else str(n_primary))
        lines.append(" & ".join((name, *(fmt(value) for value in values[:3]), fmt(float(np.mean(values[:3]))), fmt(values[3]))) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    out = TAB_DIR / "euclidean_prom_case2_n_sweep_state_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_tail_table() -> Path:
    levels = (0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)
    rows = list(csv.DictReader((DIAGNOSTIC / "case2_secondary_sensitivity_summary.csv").open()))
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Imposed tail error (\%) & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for level in levels:
        values = []
        for point in POINTS:
            row = next(row for row in rows if row["point"] == point.key and abs(float(row["requested_secondary_error_percent"]) - level) < 1.0e-9)
            values.append(float(row["primary_q_error_percent_vs_linear_prom"]))
        lines.append(" & ".join((fmt(level, 0), *(fmt(value) for value in values))) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    out = TAB_DIR / "euclidean_prom_case2_tail_primary_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    for path in (BASIS_PATH, U_REF_PATH, DIAGNOSTIC / "case2_secondary_sensitivity_summary.csv"):
        if not path.exists():
            raise FileNotFoundError(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    errors = state_errors()
    tables = (
        write_training_table(),
        write_state_table(errors),
        write_coefficient_table(),
        write_sweep_table(),
        write_tail_table(),
    )
    figures = [
        make_parameter_space_plot(),
        make_solution_overlay(),
        make_coefficient_curves(),
        *make_coefficient_heatmaps(),
        make_sweep_state_plot(),
        make_sweep_coefficient_plot(),
        make_tail_sensitivity_plot(),
    ]
    print("[euclidean-prom-assets] tables:")
    for path in tables:
        print(f"  {path}")
    print("[euclidean-prom-assets] figures:")
    for path in figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
