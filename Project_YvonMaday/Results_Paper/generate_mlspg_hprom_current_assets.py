#!/usr/bin/env python3
"""Generate current MLSPG-sensitive HPROM manuscript assets.

This script is intentionally narrow: it uses only the current
Results_Paper/mlspg_hprom_main campaign and leaves not-yet-run model families
blank in the generated LaTeX tables.
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
MLSPG = ROOT / "mlspg_hprom_main"
METRIC = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1"
FIG_DIR = ROOT / "Figures" / "mlspg_hprom_current"
COEFF_DIR = FIG_DIR / "coeff_errors"
CACHE_DIR = FIG_DIR / "_coeff_cache"
DIAG_DIR = FIG_DIR / "case2_trimmed_diagnostic"
EXTRAP_DIR = FIG_DIR / "extrapolation20"
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
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.9,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.7,
    }
)


POINTS = [
    ("$\\bm\\mu^{(v)}$ \\textbf{(verification)}", 4.875, 0.0225, "mu1_4.875_mu2_0.0225", "mu1_4.875+mu2_0.0225.npy"),
    ("$\\bm\\mu^{(1)}$ (off-grid)", 4.560, 0.0190, "mu1_4.560_mu2_0.0190", "mu1_4.56+mu2_0.019.npy"),
    ("$\\bm\\mu^{(2)}$ (off-grid)", 5.190, 0.0260, "mu1_5.190_mu2_0.0260", "mu1_5.19+mu2_0.026.npy"),
    ("$\\bm\\mu^{(3)}$ \\textbf{(20\\% extrapolation)}", 4.000, 0.0330, "mu1_4.000_mu2_0.0330", "mu1_4.0+mu2_0.033.npy"),
]
PRIMARY_POINTS = POINTS[:3]

EXTRAP_POINT = (
    "$\\bm\\mu^{(3)}$ \\textbf{(20\\% extrapolation)}",
    4.000,
    0.0330,
    "mu1_4.000_mu2_0.0330",
)
EXTRAP_ROOT = MLSPG / "Runs" / "Extrapolation20pct"


def is_extrap_point(mu1: float, mu2: float) -> bool:
    return abs(mu1 - EXTRAP_POINT[1]) < 1.0e-12 and abs(mu2 - EXTRAP_POINT[2]) < 1.0e-12


def point_plot_title(tag: str, mu1: float, mu2: float) -> str:
    if tag == "mu1_4.875_mu2_0.0225":
        return rf"$\mu^{{(v)}}=({mu1:.3f},{mu2:.4f})$\quad\textbf{{verification}}"
    if tag == "mu1_4.000_mu2_0.0330":
        return rf"$\mu^{{(3)}}=({mu1:.3f},{mu2:.4f})$\quad\textbf{{extrapolation}}"
    index = "1" if tag == "mu1_4.560_mu2_0.0190" else "2"
    return rf"$\mu^{{({index})}}=({mu1:.3f},{mu2:.4f})$\quad\textit{{off-grid}}"


def point_plot_title_compact(tag: str, mu1: float, mu2: float) -> str:
    if tag == "mu1_4.875_mu2_0.0225":
        index = "v"
    elif tag == "mu1_4.000_mu2_0.0330":
        index = "3"
    else:
        index = "1" if tag == "mu1_4.560_mu2_0.0190" else "2"
    return rf"$\mu^{{({index})}}$" + "\n" + rf"$({mu1:.3f},{mu2:.4f})$"


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
    marker: str | None = None
    n_primary: int | None = None
    n_secondary: int | None = None
    coeff_split: int | None = None
    is_linear: bool = False
    is_data_driven: bool = False
    is_pod_ae: bool = False
    is_pod_dl: bool = False


MAIN_ECSW_TAG = "ECSW2pct"


MODELS = [
    ModelSpec(
        key="linear",
        label="Linear HPROM",
        table_label="Linear HPROM",
        color=METHOD_COLORS["linear"],
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        is_linear=True,
    ),
    ModelSpec(
        key="case1",
        label="PROM-ANN Case 1",
        table_label="PROM-ANN Case 1",
        color=METHOD_COLORS["case1"],
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
        color=METHOD_COLORS["case2_n10"],
        linestyle="-",
        alpha=0.92,
        linewidth=2.0,
        marker=None,
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
        color=METHOD_COLORS["case2_n20"],
        linestyle="-",
        alpha=0.92,
        linewidth=2.0,
        marker=None,
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
        color=METHOD_COLORS["case3"],
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
        color=METHOD_COLORS["podae"],
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        n_primary=10,
        is_pod_ae=True,
    ),
    ModelSpec(
        key="pod_nn_best",
        label="Data-driven POD-NN",
        table_label="POD-NN-ROM",
        color=METHOD_COLORS["podnn"],
        linestyle="-",
        alpha=0.92,
        linewidth=2.0,
        marker=None,
        is_data_driven=True,
    ),
    ModelSpec(
        key="pod_dl_best",
        label="POD-DL-ROM ($n_z=10$)",
        table_label="POD-DL-ROM",
        color=METHOD_COLORS["poddl"],
        linestyle="-",
        alpha=0.90,
        linewidth=2.0,
        n_primary=10,
        is_pod_dl=True,
    ),
]

CASE2_TRIMMED_FROM_NP10_SPEC = ModelSpec(
    key="case2_n20_trimmed_from_np10",
    label="PROM-ANN Case 2 ($n=20$, trimmed from $n=10$ map)",
    table_label="PROM-ANN Case 2 ($n=20$, trimmed $n=10$ map)",
    color="tab:olive",
    linestyle="--",
    alpha=0.92,
    linewidth=2.0,
    family_path="Case2_TrimmedFromNp10/np20",
    file_prefix="case2_hprom_ann_trimmed_from_np10",
    n_primary_for_file=20,
    n_primary=20,
    n_secondary=131,
    coeff_split=20,
)


def ensure_dirs() -> None:
    for d in (FIG_DIR, COEFF_DIR, CACHE_DIR, DIAG_DIR, EXTRAP_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def parameter_plot_limits(*point_sets: np.ndarray, pad_fraction: float = 0.18) -> tuple[tuple[float, float], tuple[float, float]]:
    arrays = [np.asarray(points, dtype=np.float64).reshape(-1, 2) for points in point_sets if np.asarray(points).size]
    if not arrays:
        raise ValueError("At least one non-empty point set is required.")
    pts = np.vstack(arrays)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    xspan = max(xmax - xmin, 1.0e-12)
    yspan = max(ymax - ymin, 1.0e-12)
    return (xmin - pad_fraction * xspan, xmax + pad_fraction * xspan), (
        ymin - pad_fraction * yspan,
        ymax + pad_fraction * yspan,
    )


def style_parameter_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_box_aspect(1)


def load_array(path: Path, mmap: bool = True) -> np.ndarray | None:
    try:
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - report and skip corrupted/incomplete assets
        print(f"[warn] cannot load {path}: {type(exc).__name__}: {exc}")
        return None


def read_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(errors="ignore").splitlines():
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def ffloat(d: dict[str, str], key: str) -> float | None:
    try:
        return float(d[key])
    except Exception:
        return None


def linear_dir(mu1: float, mu2: float) -> Path:
    if is_extrap_point(mu1, mu2):
        return extrap_linear_dir(mu1, mu2)
    return MLSPG / "Runs" / "Linear" / f"linear_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"


def run_stem(spec: ModelSpec, mu1: float, mu2: float) -> str:
    if spec.file_prefix is None or spec.n_primary_for_file is None:
        raise ValueError(f"ModelSpec {spec.key} has no run-file prefix.")
    return f"{spec.file_prefix}_mu1_{mu1:.3f}_mu2_{mu2:.4f}_n{spec.n_primary_for_file}_ntot151"


def model_ecsw_tag(spec: ModelSpec) -> str:
    return MAIN_ECSW_TAG


def model_run_dir(spec: ModelSpec, ecsw_tag: str | None = None, mu1: float | None = None, mu2: float | None = None) -> Path:
    if spec.family_path is None:
        raise ValueError(f"ModelSpec {spec.key} has no run family path.")
    tag = model_ecsw_tag(spec) if ecsw_tag is None else ecsw_tag
    if mu1 is not None and mu2 is not None and is_extrap_point(float(mu1), float(mu2)):
        return EXTRAP_ROOT / tag / spec.family_path
    return MLSPG / "Runs" / tag / spec.family_path


def model_summary_path(spec: ModelSpec, mu1: float, mu2: float, ecsw_tag: str | None = None) -> Path:
    return model_run_dir(spec, ecsw_tag, mu1, mu2) / f"{run_stem(spec, mu1, mu2)}_summary.txt"


def model_snaps_path(spec: ModelSpec, mu1: float, mu2: float, ecsw_tag: str | None = None) -> Path:
    return model_run_dir(spec, ecsw_tag, mu1, mu2) / f"{run_stem(spec, mu1, mu2)}_snaps.npy"


def model_qn_path(spec: ModelSpec, mu1: float, mu2: float, ecsw_tag: str | None = None) -> Path:
    return model_run_dir(spec, ecsw_tag, mu1, mu2) / f"{run_stem(spec, mu1, mu2)}_qN.npy"


def data_driven_dir(mu1: float, mu2: float) -> Path:
    if is_extrap_point(mu1, mu2):
        return extrap_data_driven_dir(mu1, mu2)
    return MLSPG / "Runs" / "DataDriven_Best" / f"rom_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"


def pod_ae_stem(mu1: float, mu2: float) -> str:
    return f"podae_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151_nz10"


def pod_ae_dir() -> Path:
    return MLSPG / "Runs" / MAIN_ECSW_TAG / "PODAE_Best"


def pod_dl_dir(mu1: float, mu2: float) -> Path:
    if is_extrap_point(mu1, mu2):
        return extrap_pod_dl_dir(mu1, mu2)
    expected = (
        MLSPG
        / "Runs"
        / "PODDL_Best"
        / f"pod_dl_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151_nz10"
    )
    if expected.is_dir():
        return expected
    candidates = sorted(
        (MLSPG / "Runs" / "PODDL_Best").glob(
            f"pod_dl_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151_nz*"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one POD-DL run for ({mu1:.3f},{mu2:.4f}), found {candidates}"
        )
    return candidates[0]


def extrap_linear_dir(mu1: float, mu2: float) -> Path:
    return EXTRAP_ROOT / "Linear" / f"linear_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"


def extrap_model_run_dir(spec: ModelSpec) -> Path:
    if spec.family_path is None:
        raise ValueError(f"ModelSpec {spec.key} has no run family path.")
    return EXTRAP_ROOT / model_ecsw_tag(spec) / spec.family_path


def extrap_model_summary_path(spec: ModelSpec, mu1: float, mu2: float) -> Path:
    return extrap_model_run_dir(spec) / f"{run_stem(spec, mu1, mu2)}_summary.txt"


def extrap_data_driven_dir(mu1: float, mu2: float) -> Path:
    return EXTRAP_ROOT / "DataDriven_Best" / f"rom_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"


def extrap_pod_ae_dir() -> Path:
    return EXTRAP_ROOT / MAIN_ECSW_TAG / "PODAE_Best"


def extrap_pod_dl_dir(mu1: float, mu2: float) -> Path:
    return (
        EXTRAP_ROOT
        / "PODDL_Best"
        / f"pod_dl_data_driven_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151_nz10"
    )


def extrap_summary_path(spec: ModelSpec, mu1: float, mu2: float) -> Path:
    if spec.is_linear:
        return extrap_linear_dir(mu1, mu2) / "summary.txt"
    if spec.is_data_driven:
        return extrap_data_driven_dir(mu1, mu2) / "rom_data_driven_summary.txt"
    if spec.is_pod_ae:
        return extrap_pod_ae_dir() / f"{pod_ae_stem(mu1, mu2)}_summary.txt"
    if spec.is_pod_dl:
        return extrap_pod_dl_dir(mu1, mu2) / "pod_dl_data_driven_summary.txt"
    return extrap_model_summary_path(spec, mu1, mu2)


def extrap_hdm_vs_rom_path(spec: ModelSpec, mu1: float, mu2: float) -> Path:
    if spec.is_linear:
        return extrap_linear_dir(mu1, mu2) / "hdm_vs_rom.png"
    if spec.is_data_driven:
        return extrap_data_driven_dir(mu1, mu2) / "hdm_vs_rom.png"
    if spec.is_pod_ae:
        return extrap_pod_ae_dir() / f"{pod_ae_stem(mu1, mu2)}_hdm_vs_rom.png"
    if spec.is_pod_dl:
        return extrap_pod_dl_dir(mu1, mu2) / "hdm_vs_rom.png"
    return extrap_model_run_dir(spec) / f"{run_stem(spec, mu1, mu2)}_hdm_vs_rom.png"


def extrap_spec_error(spec: ModelSpec, mu1: float, mu2: float) -> float | None:
    return ffloat(read_summary(extrap_summary_path(spec, mu1, mu2)), "relative_error_percent")


def extrap_spec_time(spec: ModelSpec, mu1: float, mu2: float) -> float | None:
    summary = read_summary(extrap_summary_path(spec, mu1, mu2))
    return ffloat(summary, "inference_time_s" if (spec.is_data_driven or spec.is_pod_dl) else "online_solve_elapsed_s")


def extrap_spec_ne(spec: ModelSpec, mu1: float, mu2: float) -> int | None:
    if spec.is_data_driven or spec.is_pod_dl:
        return None
    summary = read_summary(extrap_summary_path(spec, mu1, mu2))
    try:
        return int(summary["n_ecsw_elements"])
    except Exception:
        return spec_ne(spec, MAIN_ECSW_TAG)


def hdm_path(hdm_file: str) -> Path:
    for base in (PROJECT / "Results" / "param_snaps", PROJECT / "250x250" / "param_snaps", PROJECT.parent / "Results" / "param_snaps"):
        p = base / hdm_file
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find HDM snapshot file {hdm_file}")


def q_linear(mu1: float, mu2: float) -> np.ndarray:
    qpath = linear_dir(mu1, mu2) / "qN.npy"
    q = load_array(qpath, mmap=False)
    if q is None:
        raise FileNotFoundError(qpath)
    return np.asarray(q, dtype=np.float64)


def model_snaps(spec: ModelSpec, mu1: float, mu2: float) -> np.ndarray | None:
    if spec.is_linear:
        return None
    if spec.is_data_driven:
        return load_array(data_driven_dir(mu1, mu2) / "rom_snaps.npy")
    if spec.is_pod_ae:
        base = extrap_pod_ae_dir() if is_extrap_point(mu1, mu2) else pod_ae_dir()
        return load_array(base / f"{pod_ae_stem(mu1, mu2)}_snaps.npy")
    if spec.is_pod_dl:
        return load_array(pod_dl_dir(mu1, mu2) / "rom_snaps.npy")
    return load_array(model_snaps_path(spec, mu1, mu2))


def model_q(spec: ModelSpec, mu1: float, mu2: float, V: np.ndarray, u_ref: np.ndarray) -> np.ndarray | None:
    if spec.is_linear:
        return q_linear(mu1, mu2)
    if spec.is_data_driven:
        q = load_array(data_driven_dir(mu1, mu2) / "qN.npy", mmap=False)
        return None if q is None else np.asarray(q, dtype=np.float64)
    if spec.is_pod_ae:
        base = extrap_pod_ae_dir() if is_extrap_point(mu1, mu2) else pod_ae_dir()
        q = load_array(base / f"{pod_ae_stem(mu1, mu2)}_qN.npy", mmap=False)
        return None if q is None else np.asarray(q, dtype=np.float64)
    if spec.is_pod_dl:
        q = load_array(pod_dl_dir(mu1, mu2) / "qN.npy", mmap=False)
        return None if q is None else np.asarray(q, dtype=np.float64)

    qpath = model_qn_path(spec, mu1, mu2)
    q_saved = load_array(qpath, mmap=False)
    if q_saved is not None:
        return np.asarray(q_saved, dtype=np.float64)

    cache = CACHE_DIR / f"{spec.key}_mu1_{mu1:.3f}_mu2_{mu2:.4f}_q_projected_from_snaps.npy"
    if cache.exists():
        q = load_array(cache, mmap=False)
        if q is not None:
            return np.asarray(q, dtype=np.float64)
    snaps = model_snaps(spec, mu1, mu2)
    if snaps is None:
        return None
    print(
        f"[coeff] WARNING: missing solver-side qN for {spec.label} at "
        f"({mu1:.3f},{mu2:.4f}); using least-squares projection from snaps"
    )
    q = least_squares_coefficients(snaps, V, u_ref)
    np.save(cache, q)
    return q


def least_squares_coefficients(snaps: np.ndarray, V: np.ndarray, u_ref: np.ndarray, block: int = 32) -> np.ndarray:
    """Return q = argmin_q ||V q - (u-u_ref)||_2 for each snapshot."""
    G = V.T @ V
    nt = snaps.shape[1]
    q = np.empty((V.shape[1], nt), dtype=np.float64)
    for start in range(0, nt, block):
        end = min(start + block, nt)
        centered = np.asarray(snaps[:, start:end], dtype=np.float64) - u_ref[:, None]
        rhs = V.T @ centered
        q[:, start:end] = np.linalg.solve(G, rhs)
    return q


def state_lines_from_q(q: np.ndarray, V: np.ndarray, u_ref: np.ndarray, idx_xline: np.ndarray, idx_yline: np.ndarray, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    qt = q[:, tidx]
    ux = u_ref[:FULL_ELEMENTS]
    xline = ux[idx_xline] + V[idx_xline, :] @ qt
    yline = ux[idx_yline] + V[idx_yline, :] @ qt
    return xline, yline


def state_lines_from_snaps(snaps: np.ndarray, idx_xline: np.ndarray, idx_yline: np.ndarray, tidx: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(snaps[idx_xline, tidx]), np.asarray(snaps[idx_yline, tidx])


def make_solution_overlay(V: np.ndarray, u_ref: np.ndarray) -> Path:
    idx_xline = (NY // 2) * NX + np.arange(NX)
    idx_yline = np.arange(NY) * NX + (NX // 2)
    xgrid = np.linspace(0.0, 100.0, NX)
    ygrid = np.linspace(0.0, 100.0, NY)
    time_ids = [120, 300, 500]

    fig, axes = plt.subplots(len(POINTS), 2, figsize=(16.0, 3.55 * len(POINTS)), sharex=False)
    for r, (_, mu1, mu2, tag, hfile) in enumerate(POINTS):
        hdm = load_array(hdm_path(hfile))
        if hdm is None:
            continue
        qref = q_linear(mu1, mu2)
        for c, (ax, grid, idx, cut_label) in enumerate(
            [
                (axes[r, 0], xgrid, idx_xline, r"$u_x(x,y_{\mathrm{mid}})$"),
                (axes[r, 1], ygrid, idx_yline, r"$u_x(x_{\mathrm{mid}},y)$"),
            ]
        ):
            for tidx in time_ids[:-1]:
                hline = np.asarray(hdm[idx, tidx])
                ax.plot(grid, hline, color="black", linestyle="--", linewidth=1.25, alpha=0.45)
            hfinal = np.asarray(hdm[idx, time_ids[-1]])
            ax.plot(grid, hfinal, color="black", linestyle="-", linewidth=2.9, alpha=0.96, label="HDM" if r == 0 and c == 0 else None)

            for spec in MODELS:
                if spec.is_linear:
                    line_mid, line_final = None, None
                    for tidx in time_ids[:-1]:
                        xline, yline = state_lines_from_q(qref, V, u_ref, idx_xline, idx_yline, tidx)
                        line = xline if c == 0 else yline
                        ax.plot(grid, line, color=spec.color, linestyle="--", linewidth=1.0, alpha=0.38)
                    xline, yline = state_lines_from_q(qref, V, u_ref, idx_xline, idx_yline, time_ids[-1])
                    line_final = xline if c == 0 else yline
                else:
                    snaps = model_snaps(spec, mu1, mu2)
                    if snaps is None:
                        continue
                    for tidx in time_ids[:-1]:
                        xline, yline = state_lines_from_snaps(snaps, idx_xline, idx_yline, tidx)
                        line = xline if c == 0 else yline
                        ax.plot(grid, line, color=spec.color, linestyle="--", linewidth=1.0, alpha=0.36)
                    xline, yline = state_lines_from_snaps(snaps, idx_xline, idx_yline, time_ids[-1])
                    line_final = xline if c == 0 else yline

                ax.plot(
                    grid,
                    line_final,
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
    fig.suptitle("MLSPG-sensitive HPROM campaign: solution cut-plane overlays", y=1.055)
    fig.text(0.5, 0.012, "Dashed: intermediate times; solid: final time.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    out = FIG_DIR / "mlspg_hprom_solution_overlays.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def compute_coeff_errors(V: np.ndarray, u_ref: np.ndarray) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    errors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for _, mu1, mu2, tag, _ in POINTS:
        qref = q_linear(mu1, mu2)
        ref_norm = np.linalg.norm(qref, axis=1)
        ref_norm = np.maximum(ref_norm, 1e-14)
        for spec in MODELS:
            if spec.is_linear:
                continue
            q = model_q(spec, mu1, mu2, V, u_ref)
            if q is None:
                continue
            if q.shape != qref.shape:
                print(f"[warn] skip {spec.label} at {tag}: q shape {q.shape} != ref {qref.shape}")
                continue
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
    fig, axes = plt.subplots(2, len(POINTS), figsize=(18.0, 8.2), sharex=True)
    for c, (_, mu1, mu2, tag, _) in enumerate(POINTS):
        ax_abs, ax_rel = axes[0, c], axes[1, c]
        for spec in MODELS:
            if spec.is_linear:
                continue
            d = errors.get((tag, spec.key))
            if d is None:
                continue
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
    fig.suptitle("MLSPG-sensitive coefficient errors vs linear HPROM reference", y=1.075)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.6, h_pad=1.0)
    out = COEFF_DIR / "mlspg_hprom_coeff_abs_rel_all_points.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_heatmap_grid(errors: dict[tuple[str, str], dict[str, np.ndarray]], kind: str) -> Path:
    assert kind in {"abs_heat", "rel_heat"}
    plot_specs = [s for s in MODELS if not s.is_linear]
    fig, axes = plt.subplots(
        len(plot_specs),
        len(POINTS),
        figsize=(20.0, 2.35 * len(plot_specs) + 1.4),
        sharex=True,
        sharey=True,
    )
    all_values = []
    for _, _, _, tag, _ in POINTS:
        for spec in plot_specs:
            d = errors.get((tag, spec.key))
            if d is not None:
                values = d[kind] / 100.0 if kind == "rel_heat" else d[kind]
                all_values.append(values)
    if all_values:
        flat = np.concatenate([v.ravel() for v in all_values])
        vmin = 0.0
        vmax = float(np.nanpercentile(flat, 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im = None
    for r, spec in enumerate(plot_specs):
        for c, (_, mu1, mu2, tag, _) in enumerate(POINTS):
            ax = axes[r, c]
            d = errors.get((tag, spec.key))
            if d is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            image = d[kind] / 100.0 if kind == "rel_heat" else d[kind]
            im = ax.imshow(
                image,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[0.0, 25.0, 1, NTOT],
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            if spec.coeff_split:
                ax.axhline(spec.coeff_split + 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            if r == 0:
                ax.set_title(point_plot_title_compact(tag, mu1, mu2), fontsize=11, pad=5)
            if c == 0:
                ax.annotate(
                    spec.label,
                    xy=(-0.10, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=12,
                    annotation_clip=False,
                )
            if r == len(plot_specs) - 1:
                ax.set_xlabel(r"Time $t$")
            ax.grid(False)
    if im is not None:
        fig.subplots_adjust(left=0.22, right=0.89, bottom=0.055, top=0.895, wspace=0.16, hspace=0.26)
        fig.supylabel(r"Coefficient index $i$", x=0.035, fontsize=14)
        cax = fig.add_axes([0.91, 0.14, 0.022, 0.72])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(
            r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|$"
            if kind == "abs_heat"
            else r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|/\|q_i^{\mathrm{ref}}\|_2$"
        )
    fig.suptitle(
        "MLSPG-sensitive absolute coefficient error heatmaps"
        if kind == "abs_heat"
        else "MLSPG-sensitive relative coefficient error heatmaps",
        y=0.975,
    )
    out = COEFF_DIR / ("mlspg_hprom_coeff_abs_heatmaps.png" if kind == "abs_heat" else "mlspg_hprom_coeff_rel_heatmaps.png")
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


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def fmt_speedup_tex(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    if x >= 1.0e4:
        return f"${x / 1.0e4:.2f}\\times10^4$"
    return f"{x:.1f}"


def intrusive_specs() -> list[ModelSpec]:
    return [s for s in MODELS if s.key in {"case1", "case2_n10", "case2_n20", "case3"}]


def spec_error(spec: ModelSpec, mu1: float, mu2: float, ecsw_tag: str = MAIN_ECSW_TAG) -> float | None:
    if ecsw_tag == MAIN_ECSW_TAG and is_extrap_point(mu1, mu2):
        return extrap_spec_error(spec, mu1, mu2)
    if spec.is_linear:
        return ffloat(read_summary(linear_dir(mu1, mu2) / "summary.txt"), "relative_error_percent")
    if spec.is_data_driven:
        return ffloat(read_summary(data_driven_dir(mu1, mu2) / "rom_data_driven_summary.txt"), "relative_error_percent")
    if spec.is_pod_ae:
        return ffloat(read_summary(pod_ae_dir() / f"{pod_ae_stem(mu1, mu2)}_summary.txt"), "relative_error_percent")
    if spec.is_pod_dl:
        return ffloat(read_summary(pod_dl_dir(mu1, mu2) / "pod_dl_data_driven_summary.txt"), "relative_error_percent")
    return ffloat(read_summary(model_summary_path(spec, mu1, mu2, ecsw_tag)), "relative_error_percent")


def spec_time(spec: ModelSpec, mu1: float, mu2: float, ecsw_tag: str = MAIN_ECSW_TAG) -> float | None:
    if ecsw_tag == MAIN_ECSW_TAG and is_extrap_point(mu1, mu2):
        return extrap_spec_time(spec, mu1, mu2)
    if spec.is_linear:
        return ffloat(read_summary(linear_dir(mu1, mu2) / "summary.txt"), "online_solve_elapsed_s")
    if spec.is_data_driven:
        return ffloat(read_summary(data_driven_dir(mu1, mu2) / "rom_data_driven_summary.txt"), "inference_time_s")
    if spec.is_pod_ae:
        return ffloat(read_summary(pod_ae_dir() / f"{pod_ae_stem(mu1, mu2)}_summary.txt"), "online_solve_elapsed_s")
    if spec.is_pod_dl:
        return ffloat(read_summary(pod_dl_dir(mu1, mu2) / "pod_dl_data_driven_summary.txt"), "inference_time_s")
    return ffloat(read_summary(model_summary_path(spec, mu1, mu2, ecsw_tag)), "online_solve_elapsed_s")


def spec_ne(spec: ModelSpec, ecsw_tag: str = MAIN_ECSW_TAG) -> int | None:
    if spec.is_linear:
        path = MLSPG / "Stage2" / "ecsw" / "ecsw_weights_lspg_ntot151.npy"
    elif spec.is_data_driven or spec.is_pod_dl:
        return None
    elif spec.is_pod_ae:
        summary = read_summary(
            pod_ae_dir()
            / f"{pod_ae_stem(POINTS[0][1], POINTS[0][2])}_summary.txt"
        )
        try:
            return int(summary["n_ecsw_elements"])
        except Exception:
            return None
    else:
        if spec.family_path is None or spec.n_primary_for_file is None or spec.file_prefix is None:
            return None
        case_name = spec.file_prefix.split("_", 1)[0]
        # file_prefix is e.g. case2_hprom_ann; the ECSW weight stem uses case2.
        case_name = spec.file_prefix.split("_")[0]
        model_name = {
            "case1": "case1_ann_ntot151_best",
            "case2": f"case2_ann_ntot151_np{spec.n_primary_for_file}_best",
            "case3": "case3_ann_ntot151_best",
        }[case_name]
        path = (
            MLSPG
            / "ECSW"
            / ecsw_tag.replace("ECSW", "")
            / spec.family_path
            / f"ecsw_weights_ann_{case_name}_{model_name}_n{spec.n_primary_for_file}_ntot151.npy"
        )
    arr = load_array(path, mmap=False)
    return None if arr is None else int(np.count_nonzero(arr))


def mean_existing(vals: list[float | None]) -> float | None:
    xs = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.nanmean(xs)) if xs else None


def extrapolation_rows() -> list[dict[str, object]]:
    _, mu1, mu2, _ = EXTRAP_POINT
    rows: list[dict[str, object]] = []
    for spec in MODELS:
        time_s = extrap_spec_time(spec, mu1, mu2)
        if spec.is_linear or spec.is_data_driven:
            online_dim = 151
        elif spec.is_pod_ae or spec.is_pod_dl:
            online_dim = 10
        else:
            online_dim = spec.n_primary
        rows.append(
            {
                "method": spec.table_label,
                "key": spec.key,
                "online_dim": online_dim,
                "n_s": None if (spec.is_linear or spec.is_data_driven or spec.is_pod_ae or spec.is_pod_dl) else spec.n_secondary,
                "n_e": extrap_spec_ne(spec, mu1, mu2),
                "error_percent": extrap_spec_error(spec, mu1, mu2),
                "time_s": time_s,
                "speedup": None if time_s is None else HDM_REFERENCE_TIME_S / time_s,
            }
        )
    return rows


def make_extrapolation_table() -> tuple[Path, Path]:
    rows = extrapolation_rows()
    fields = ["method", "key", "online_dim", "n_s", "n_e", "error_percent", "time_s", "speedup"]
    csv_path = TABLE_DIR / "mlspg_hprom_extrapolation20_mu3.csv"
    write_csv(csv_path, rows, fields)

    tex_path = TABLE_DIR / "mlspg_hprom_extrapolation20_mu3.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline MLSPG-sensitive extrapolation diagnostic at $\\bm\\mu^{(3)}=(4.000,0.0330)$, located 20\\% beyond the upper-left corner of the $3\\times3$ training box. All rows use the non-enriched trained models and the already-selected ECSW rules; no ECSW rule is rebuilt for this diagnostic. Errors are relative trajectory errors (\\%) with respect to HDM, and speedups use $t_{\\mathrm{HDM}}=737.44$ s.}\n")
        f.write("\\label{tab:mlspg-hprom-extrapolation20-mu3}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Online/latent dim. & $n_s$ & $n_e$ & $E_{\\mu^{(3)}}$ (\\%) & Online/inference time (s) & Speedup vs HDM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            n_s = "--" if row["n_s"] is None else str(row["n_s"])
            n_e = "--" if row["n_e"] is None else str(row["n_e"])
            f.write(
                f"{row['method']} & {row['online_dim']} & {n_s} & {n_e} & "
                f"{fmt(row['error_percent'])} & {fmt(row['time_s'], 4)} & {fmt_speedup_tex(row['speedup'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")
    return csv_path, tex_path


def make_extrapolation_parameter_figure() -> Path:
    train_mu1 = np.array([4.25, 4.875, 5.50], dtype=float)
    train_mu2 = np.array([0.015, 0.0225, 0.030], dtype=float)
    baseline = np.array([(x, y) for x in train_mu1 for y in train_mu2], dtype=float)
    eval_pts = np.array([(mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS], dtype=float)
    _, mu3_1, mu3_2, _ = EXTRAP_POINT
    mu3 = np.array([[mu3_1, mu3_2]], dtype=float)
    xlim, ylim = parameter_plot_limits(baseline, eval_pts, mu3, pad_fraction=0.12)

    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.set_facecolor("#fbfbf7")
    ax.add_patch(
        plt.Rectangle(
            (train_mu1.min(), train_mu2.min()),
            train_mu1.max() - train_mu1.min(),
            train_mu2.max() - train_mu2.min(),
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="0.25",
            alpha=0.85,
            label="Training box",
        )
    )
    ax.scatter(baseline[:, 0], baseline[:, 1], s=78, facecolors="black", edgecolors="black", linewidths=1.3, label="Baseline $3\\times3$ grid")
    for _, mu1, mu2, tag, _ in PRIMARY_POINTS:
        if tag == "mu1_4.875_mu2_0.0225":
            suffix = "(v)"
            offset = (34, -20)
            va = "top"
        elif tag == "mu1_4.560_mu2_0.0190":
            suffix = "(1)"
            offset = (32, 18)
            va = "bottom"
        else:
            suffix = "(2)"
            offset = (32, 18)
            va = "bottom"
        ax.scatter(mu1, mu2, s=155, marker="*", color="#c62828", edgecolors="white", linewidths=0.7, zorder=5, label="In-domain evaluation points" if suffix == "(v)" else None)
        ax.annotate(
            rf"$\mu^{{{suffix}}}$",
            (mu1, mu2),
            xytext=offset,
            textcoords="offset points",
            fontsize=12,
            color="#7f1111",
            ha="left",
            va=va,
            arrowprops={"arrowstyle": "-", "color": "#7f1111", "lw": 0.8, "shrinkA": 2, "shrinkB": 5},
            bbox={"boxstyle": "round,pad=0.12", "fc": "#fbfbf7", "ec": "none", "alpha": 0.88},
            zorder=6,
        )
    ax.scatter(mu3_1, mu3_2, s=210, marker="*", color="#6a1b9a", edgecolors="white", linewidths=0.8, zorder=6, label="Extrapolation point")
    ax.annotate(
        r"$\mu^{(3)}$",
        (mu3_1, mu3_2),
        xytext=(-54, -28),
        textcoords="offset points",
        fontsize=12,
        color="#4a116b",
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "color": "#4a116b", "lw": 0.8, "shrinkA": 2, "shrinkB": 5},
        bbox={"boxstyle": "round,pad=0.12", "fc": "#fbfbf7", "ec": "none", "alpha": 0.88},
        zorder=7,
    )
    style_parameter_axis(ax, xlim, ylim)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title(r"Baseline training box and upper-left 20\% extrapolation point")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, borderaxespad=0.0)
    fig.tight_layout()
    out = EXTRAP_DIR / "parameter_domain_extrapolation20_mu3.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_extrapolation_error_time_figure() -> Path:
    rows = extrapolation_rows()
    labels = [str(row["method"]) for row in rows]
    errors = np.array([np.nan if row["error_percent"] is None else float(row["error_percent"]) for row in rows])
    times = np.array([np.nan if row["time_s"] is None else float(row["time_s"]) for row in rows])
    colors = [next(spec.color for spec in MODELS if spec.key == row["key"]) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.8))
    ax = axes[0]
    ax.bar(x, errors, color=colors, alpha=0.88)
    ax.set_ylabel(r"Trajectory error at $\mu^{(3)}$ (\%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=36, ha="right")
    ax.set_title(r"Extrapolation accuracy")
    ax.grid(True, axis="y")

    ax = axes[1]
    ax.bar(x, times, color=colors, alpha=0.88)
    ax.set_yscale("log")
    ax.set_ylabel(r"Online/inference time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=36, ha="right")
    ax.set_title(r"Extrapolation online cost")
    ax.grid(True, axis="y", which="both")

    fig.suptitle(r"Baseline models at $\mu^{(3)}=(4.000,0.0330)$", y=1.03)
    fig.tight_layout()
    out = EXTRAP_DIR / "mlspg_hprom_extrapolation20_error_time_bars.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_extrapolation_hdm_vs_rom_montage() -> Path:
    _, mu1, mu2, _ = EXTRAP_POINT
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 17.0))
    for ax, spec in zip(axes.ravel(), MODELS):
        image_path = extrap_hdm_vs_rom_path(spec, mu1, mu2)
        if not image_path.exists():
            ax.text(0.5, 0.5, f"Missing\n{spec.table_label}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(spec.table_label, pad=6)
        ax.set_axis_off()
    fig.suptitle(r"Saved HDM--ROM line-cut diagnostics at $\mu^{(3)}=(4.000,0.0330)$", y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    out = EXTRAP_DIR / "mlspg_hprom_extrapolation20_hdm_vs_rom_montage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_error_tables() -> tuple[Path, Path]:
    fields = [
        "point",
        "mu1",
        "mu2",
        "linear_hprom",
        "prom_ann_case1",
        "prom_ann_case2_n10",
        "prom_ann_case2_n20",
        "prom_ann_case3",
        "prom_pod_ae",
        "pod_nn_rom",
        "pod_dl_rom",
    ]
    rows: list[dict[str, object]] = []
    spec_by_key = {s.key: s for s in MODELS}
    for point_label, mu1, mu2, _, _ in POINTS:
        rows.append(
            {
                "point": point_label,
                "mu1": mu1,
                "mu2": mu2,
                "linear_hprom": spec_error(spec_by_key["linear"], mu1, mu2),
                "prom_ann_case1": spec_error(spec_by_key["case1"], mu1, mu2),
                "prom_ann_case2_n10": spec_error(spec_by_key["case2_n10"], mu1, mu2),
                "prom_ann_case2_n20": spec_error(spec_by_key["case2_n20"], mu1, mu2),
                "prom_ann_case3": spec_error(spec_by_key["case3"], mu1, mu2),
                "prom_pod_ae": spec_error(spec_by_key["pod_ae_best"], mu1, mu2),
                "pod_nn_rom": spec_error(spec_by_key["pod_nn_best"], mu1, mu2),
                "pod_dl_rom": spec_error(spec_by_key["pod_dl_best"], mu1, mu2),
            }
        )
    csv_path = TABLE_DIR / "mlspg_hprom_current_errors.csv"
    write_csv(csv_path, rows, fields)

    tex_path = TABLE_DIR / "mlspg_hprom_current_errors.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Current MLSPG-sensitive campaign: relative trajectory errors (\\%) with respect to HDM. The first three rows are the original verification/off-grid evaluation points; the final row is the extrapolatory stress-test point. Intrusive learned models use 2\\% ECSW rules; POD-NN-ROM and POD-DL-ROM are non-intrusive.}\n")
        f.write("\\label{tab:mlspg-hprom-current-errors}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("Point & Linear HPROM & PROM-ANN Case 1 & PROM-ANN Case 2 ($n=10$) & PROM-ANN Case 2 ($n=20$) & PROM-ANN Case 3 & PROM-POD-AE & POD-NN-ROM & POD-DL-ROM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['point']} & {fmt(row['linear_hprom'])} & {fmt(row['prom_ann_case1'])} & {fmt(row['prom_ann_case2_n10'])} & "
                f"{fmt(row['prom_ann_case2_n20'])} & {fmt(row['prom_ann_case3'])} & {fmt(row['prom_pod_ae'])} & "
                f"{fmt(row['pod_nn_rom'])} & {fmt(row['pod_dl_rom'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")
    return csv_path, tex_path


def case2_trimmed_diagnostic_rows() -> list[dict[str, object]]:
    spec_by_key = {s.key: s for s in MODELS}
    native_n10 = spec_by_key["case2_n10"]
    native_n20 = spec_by_key["case2_n20"]
    trimmed = CASE2_TRIMMED_FROM_NP10_SPEC
    rows: list[dict[str, object]] = []
    for point_label, mu1, mu2, _, _ in POINTS:
        summary = read_summary(trimmed_case2_summary_path(mu1, mu2))
        try:
            n_e = int(summary["n_ecsw_elements"])
        except Exception:
            n_e = None
        rows.append(
            {
                "point": point_label,
                "mu1": mu1,
                "mu2": mu2,
                "native_n10_error": spec_error(native_n10, mu1, mu2),
                "native_n20_error": spec_error(native_n20, mu1, mu2),
                "trimmed_n20_from_n10_error": ffloat(summary, "relative_error_percent"),
                "trimmed_online_time_s": ffloat(summary, "online_solve_elapsed_s"),
                "trimmed_n_e": n_e,
            }
        )
    rows.append(
        {
            "point": "Mean",
            "mu1": None,
            "mu2": None,
            "native_n10_error": mean_existing([row["native_n10_error"] for row in rows]),
            "native_n20_error": mean_existing([row["native_n20_error"] for row in rows]),
            "trimmed_n20_from_n10_error": mean_existing(
                [row["trimmed_n20_from_n10_error"] for row in rows]
            ),
            "trimmed_online_time_s": mean_existing([row["trimmed_online_time_s"] for row in rows]),
            "trimmed_n_e": rows[0]["trimmed_n_e"] if rows else None,
        }
    )
    return rows


def trimmed_case2_summary_path(mu1: float, mu2: float) -> Path:
    return (
        model_run_dir(CASE2_TRIMMED_FROM_NP10_SPEC, MAIN_ECSW_TAG)
        / f"{run_stem(CASE2_TRIMMED_FROM_NP10_SPEC, mu1, mu2)}_summary.txt"
    )


def trimmed_case2_error(mu1: float, mu2: float) -> float | None:
    return ffloat(read_summary(trimmed_case2_summary_path(mu1, mu2)), "relative_error_percent")


def trimmed_case2_time(mu1: float, mu2: float) -> float | None:
    return ffloat(read_summary(trimmed_case2_summary_path(mu1, mu2)), "online_solve_elapsed_s")


def diagnostic_spec_error(spec: ModelSpec, mu1: float, mu2: float) -> float | None:
    if spec.key == CASE2_TRIMMED_FROM_NP10_SPEC.key:
        return trimmed_case2_error(mu1, mu2)
    return spec_error(spec, mu1, mu2)


def diagnostic_spec_time(spec: ModelSpec, mu1: float, mu2: float) -> float | None:
    if spec.key == CASE2_TRIMMED_FROM_NP10_SPEC.key:
        return trimmed_case2_time(mu1, mu2)
    return spec_time(spec, mu1, mu2)


def make_case2_trimmed_diagnostic_table() -> tuple[Path, Path]:
    fields = [
        "method",
        "muv_error",
        "mu1_error",
        "mu2_error",
        "in_domain_mean_error_percent",
        "mu3_error",
        "mean_online_time_s",
        "n_e",
    ]
    spec_by_key = {s.key: s for s in MODELS}
    variants = [
        ("Native Case 2 ($n=10$)", spec_by_key["case2_n10"]),
        ("Native Case 2 ($n=20$)", spec_by_key["case2_n20"]),
        ("Trimmed $n=20$ from $n=10$ map", CASE2_TRIMMED_FROM_NP10_SPEC),
    ]
    rows: list[dict[str, object]] = []
    for method, spec in variants:
        errors = [diagnostic_spec_error(spec, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]
        mu3_error = diagnostic_spec_error(spec, EXTRAP_POINT[1], EXTRAP_POINT[2])
        times = [diagnostic_spec_time(spec, mu1, mu2) for _, mu1, mu2, _, _ in POINTS]
        if spec.key == CASE2_TRIMMED_FROM_NP10_SPEC.key:
            summary = read_summary(trimmed_case2_summary_path(PRIMARY_POINTS[0][1], PRIMARY_POINTS[0][2]))
            try:
                n_e = int(summary["n_ecsw_elements"])
            except Exception:
                n_e = None
        else:
            n_e = spec_ne(spec, MAIN_ECSW_TAG)
        rows.append(
            {
                "method": method,
                "muv_error": errors[0],
                "mu1_error": errors[1],
                "mu2_error": errors[2],
                "in_domain_mean_error_percent": mean_existing(errors),
                "mu3_error": mu3_error,
                "mean_online_time_s": mean_existing(times),
                "n_e": n_e,
            }
        )

    csv_path = TABLE_DIR / "mlspg_hprom_case2_trimmed_from_np10_diagnostic.csv"
    write_csv(csv_path, rows, fields)

    tex_path = TABLE_DIR / "mlspg_hprom_case2_trimmed_from_np10_diagnostic.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Case~2 diagnostic in the non-enriched MLSPG-sensitive campaign. The trimmed variant uses the trained $n=10$ Case~2 map $(\\mu_1,\\mu_2,t)\\mapsto(q_{11},\\ldots,q_{151})$ and discards its first ten secondary outputs, so that the online solve uses $n=20$ primary coordinates and injects only $(q_{21},\\ldots,q_{151})$. Errors are relative trajectory errors (\\%) with respect to HDM. The in-domain mean averages $\\bm\\mu^{(v)}$, $\\bm\\mu^{(1)}$, and $\\bm\\mu^{(2)}$; the extrapolatory point $\\bm\\mu^{(3)}$ is reported separately. Timings are averages over all four displayed points.}\n")
        f.write("\\label{tab:case2-trimmed-from-np10-diagnostic}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & $E_{\\mu^{(v)}}$ & $E_{\\mu^{(1)}}$ & $E_{\\mu^{(2)}}$ & In-domain mean $E$ & $E_{\\mu^{(3)}}$ & Mean online time (s) & $n_e$ \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            n_e = "--" if row["n_e"] is None else str(row["n_e"])
            f.write(
                f"{row['method']} & {fmt(row['muv_error'])} & {fmt(row['mu1_error'])} & "
                f"{fmt(row['mu2_error'])} & {fmt(row['in_domain_mean_error_percent'])} & "
                f"{fmt(row['mu3_error'])} & "
                f"{fmt(row['mean_online_time_s'], 4)} & {n_e} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")
    return csv_path, tex_path


def make_case2_trimmed_diagnostic_figure() -> Path:
    spec_by_key = {s.key: s for s in MODELS}
    native_n10 = spec_by_key["case2_n10"]
    native_n20 = spec_by_key["case2_n20"]
    trimmed = CASE2_TRIMMED_FROM_NP10_SPEC

    point_labels = [r"$\mu^{(v)}$", r"$\mu^{(1)}$", r"$\mu^{(2)}$", "In-domain\nmean", r"$\mu^{(3)}$"]
    errors_n10 = [diagnostic_spec_error(native_n10, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]
    errors_n20 = [diagnostic_spec_error(native_n20, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]
    errors_trimmed = [diagnostic_spec_error(trimmed, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]
    times_n10 = [diagnostic_spec_time(native_n10, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]
    times_n20 = [diagnostic_spec_time(native_n20, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]
    times_trimmed = [diagnostic_spec_time(trimmed, mu1, mu2) for _, mu1, mu2, _, _ in PRIMARY_POINTS]

    errors_n10.append(mean_existing(errors_n10))
    errors_n20.append(mean_existing(errors_n20))
    errors_trimmed.append(mean_existing(errors_trimmed))
    times_n10.append(mean_existing(times_n10))
    times_n20.append(mean_existing(times_n20))
    times_trimmed.append(mean_existing(times_trimmed))
    errors_n10.append(diagnostic_spec_error(native_n10, EXTRAP_POINT[1], EXTRAP_POINT[2]))
    errors_n20.append(diagnostic_spec_error(native_n20, EXTRAP_POINT[1], EXTRAP_POINT[2]))
    errors_trimmed.append(diagnostic_spec_error(trimmed, EXTRAP_POINT[1], EXTRAP_POINT[2]))
    times_n10.append(diagnostic_spec_time(native_n10, EXTRAP_POINT[1], EXTRAP_POINT[2]))
    times_n20.append(diagnostic_spec_time(native_n20, EXTRAP_POINT[1], EXTRAP_POINT[2]))
    times_trimmed.append(diagnostic_spec_time(trimmed, EXTRAP_POINT[1], EXTRAP_POINT[2]))

    def arr(values: list[float | None]) -> np.ndarray:
        return np.array([np.nan if v is None else float(v) for v in values], dtype=float)

    x = np.arange(len(point_labels))
    width = 0.24
    colors = {
        "n10": "#2f78b7",
        "n20": "#9a5a22",
        "trimmed": "#748b22",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    ax = axes[0]
    ax.bar(x - width, arr(errors_n10), width, label=r"Native Case 2 ($n=10$)", color=colors["n10"])
    ax.bar(x, arr(errors_n20), width, label=r"Native Case 2 ($n=20$)", color=colors["n20"])
    ax.bar(
        x + width,
        arr(errors_trimmed),
        width,
        label=r"Trimmed $n=20$ from $n=10$ map",
        color=colors["trimmed"],
    )
    ax.set_ylabel(r"Trajectory error (\%)")
    ax.set_xticks(x)
    ax.set_xticklabels(point_labels)
    ax.set_title(r"Case 2 state accuracy")
    ax.grid(True, axis="y")

    ax = axes[1]
    ax.bar(x - width, arr(times_n10), width, label=r"Native Case 2 ($n=10$)", color=colors["n10"])
    ax.bar(x, arr(times_n20), width, label=r"Native Case 2 ($n=20$)", color=colors["n20"])
    ax.bar(
        x + width,
        arr(times_trimmed),
        width,
        label=r"Trimmed $n=20$ from $n=10$ map",
        color=colors["trimmed"],
    )
    ax.set_ylabel(r"Online solve time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(point_labels)
    ax.set_title(r"Case 2 online cost")
    ax.grid(True, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = DIAG_DIR / "case2_trimmed_from_np10_error_time_bars.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


def make_hyperreduction_table() -> tuple[Path, Path]:
    rows = []
    for spec in MODELS:
        times = [spec_time(spec, mu1, mu2) for _, mu1, mu2, _, _ in POINTS]
        tmean = mean_existing(times)
        if spec.is_linear or spec.is_data_driven:
            online_dim = 151
        elif spec.is_pod_ae or spec.is_pod_dl:
            online_dim = 10
        else:
            online_dim = spec.n_primary
        is_nonintrusive = spec.is_data_driven or spec.is_pod_dl
        rows.append(
            {
                "method": spec.table_label,
                "n": online_dim,
                "n_s": None if (spec.is_linear or is_nonintrusive or spec.is_pod_ae) else spec.n_secondary,
                "ecsw_percent": None if is_nonintrusive else 2.0,
                "n_e": spec_ne(spec, MAIN_ECSW_TAG),
                "time_s": tmean,
                "speedup": None if tmean is None else HDM_REFERENCE_TIME_S / tmean,
                "notes": (
                    "Stage-2 shared ECSW"
                    if spec.is_linear
                    else (
                        "Non-intrusive inference; no ECSW"
                        if is_nonintrusive
                        else ("Latent-manifold ECSW" if spec.is_pod_ae else "Case-specific ECSW")
                    )
                ),
            }
        )
    fields = ["method", "n", "n_s", "ecsw_percent", "n_e", "time_s", "speedup", "notes"]
    csv_path = TABLE_DIR / "mlspg_hprom_current_hyperreduction.csv"
    write_csv(csv_path, rows, fields)

    tex_path = TABLE_DIR / "mlspg_hprom_current_hyperreduction.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Current MLSPG-sensitive campaign: online dimensions, mesh sizes, average online timings over the four reported evaluation points, and speedups with respect to the HDM mean time $t_{\\mathrm{HDM}}=737.44$ s. Learned intrusive models use selected 2\\% ECSW rules, while the linear HPROM uses the shared Stage--2 rule. POD-NN-ROM and POD-DL-ROM are non-intrusive. $N_e=62\\,500$ is the full number of finite elements.}\n")
        f.write("\\label{tab:mlspg-hprom-current-hyperreduction}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Online/latent dim. & $n_s$ & $n_e$ & Mean online time (s) & Mean speedup vs HDM \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            n_s = "--" if row["n_s"] is None else str(row["n_s"])
            n_e = "--" if row["n_e"] is None else str(row["n_e"])
            f.write(
                f"{row['method']} & {row['n']} & {n_s} & {n_e} & "
                f"{fmt(row['time_s'], 4)} & {fmt(row['speedup'], 1)} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    print(f"[tex] {tex_path}")
    return csv_path, tex_path


def current_stage3_summary_rows() -> list[dict[str, object]]:
    files = [
        (
            "PROM-ANN Case 1",
            MLSPG / "Stage3" / "case1_ann_ntot151_best_summary.txt",
            r"$\mathbf q\in\mathbb R^{10}\mapsto\bar{\mathbf q}\in\mathbb R^{141}$",
            "z-score; SiLU",
        ),
        (
            "PROM-ANN Case 2 ($n=10$)",
            MLSPG / "Stage3" / "case2_ann_ntot151_np10_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\bar{\mathbf q}\in\mathbb R^{141}$",
            "z-score; SiLU",
        ),
        (
            "PROM-ANN Case 2 ($n=20$)",
            MLSPG / "Stage3" / "case2_ann_ntot151_np20_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\bar{\mathbf q}\in\mathbb R^{131}$",
            "z-score; SiLU",
        ),
        (
            "PROM-ANN Case 3",
            MLSPG / "Stage3" / "case3_ann_ntot151_best_summary.txt",
            r"$(\mathbf q,\mu_1,\mu_2,t)\in\mathbb R^{13}\mapsto\bar{\mathbf q}\in\mathbb R^{141}$",
            "z-score; SiLU",
        ),
        (
            "PROM-POD-AE",
            MLSPG / "Stage3" / "prom_pod_ae_ntot151_best_summary.txt",
            r"$\mathbf q_N\in\mathbb R^{151}\mapsto\mathbf z\in\mathbb R^{10}\mapsto\widehat{\mathbf q}_N$",
            "z-score; GELU",
        ),
        (
            "POD-NN-ROM",
            MLSPG / "Stage3" / "data_driven_ann_ntot151_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\mathbf q_N\in\mathbb R^{151}$",
            "z-score; SiLU",
        ),
        (
            "POD-DL-ROM",
            MLSPG / "Stage3" / "pod_dl_data_driven_ntot151_best_summary.txt",
            r"$(\mu_1,\mu_2,t)\in\mathbb R^3\mapsto\mathbf z\in\mathbb R^{10}\mapsto\widehat{\mathbf q}_N$",
            "z-score; SiLU",
        ),
    ]
    rows = []
    for method, path, learned_map, norm_activation in files:
        d = read_summary(path)
        if method == "PROM-POD-AE":
            arch = rf"Encoder {d['hidden_dims']}; decoder reverse; $n_z={d['latent_dim']}$"
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


def make_training_table() -> tuple[Path, Path]:
    rows = current_stage3_summary_rows()
    fields = ["method", "learned_map", "architecture", "normalization_activation", "val_rel", "trainable_parameters"]
    csv_path = TABLE_DIR / "mlspg_hprom_current_training_winners.csv"
    write_csv(csv_path, rows, fields)

    tex_path = TABLE_DIR / "mlspg_hprom_current_training_winners.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Stage--3 network architectures used in the non-enriched MLSPG-sensitive campaign. The table reports the learned map, hidden widths or latent dimensions, normalization/activation choices, validation relative Frobenius error on the HPROM coefficient dataset, and the number of trainable parameters.}\n")
        f.write("\\label{tab:mlspg-hprom-current-training}\n")
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
    return csv_path, tex_path


def main() -> None:
    ensure_dirs()
    V = np.load(METRIC / "basis.npy", mmap_mode="r", allow_pickle=False)
    u_ref = np.load(METRIC / "u_ref.npy", mmap_mode="r", allow_pickle=False)
    make_training_table()
    make_error_tables()
    make_extrapolation_table()
    make_extrapolation_parameter_figure()
    make_extrapolation_error_time_figure()
    make_extrapolation_hdm_vs_rom_montage()
    make_case2_trimmed_diagnostic_table()
    make_case2_trimmed_diagnostic_figure()
    make_hyperreduction_table()
    make_solution_overlay(V, u_ref)
    errors = compute_coeff_errors(V, u_ref)
    make_coeff_curve_figure(errors)
    make_heatmap_grid(errors, "abs_heat")
    make_heatmap_grid(errors, "rel_heat")


if __name__ == "__main__":
    main()
