#!/usr/bin/env python3
"""Generate HPROM baseline tables and figures for manuscript_prom.tex."""

from __future__ import annotations

import argparse
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
    METHOD_COLORS,
    METHOD_LINE_STYLES,
    STATE_CUTPLANE_YLIM,
)

SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parent
REPO = PAPER.parents[1]
HPROM = PAPER / "mlspg_hprom_main"
RUNS = HPROM / "Runs"
STAGE3 = HPROM / "Stage3"
CASE2_DIAGNOSTICS = PAPER / "tmp_case2_hprom_diagnostics"
HPROM_ENRICHED = PAPER / "mlspg_hprom_enrichment_ext25_lhs36"
ENRICHED_RUNS = HPROM_ENRICHED / "Runs"
ENRICHED_STAGE3 = HPROM_ENRICHED / "Stage3"
FIG_DIR = PAPER / "Figures" / "hprom_only"
TAB_DIR = PAPER / "tables" / "hprom_only"
BASIS_PATH = PAPER / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy"
U_REF_PATH = PAPER / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy"
NTOT = 151
# The raw n=3 and n=5 diagnostics remain available, but the reported sweep
# starts from the production n=10 split.
CASE2_N_SWEEP = (0, 10, 20, 30, 50, 151)
ROW_END = r"\\"
HDM_REFERENCE_TIME_S = 7.37437560e02
DIRECT_TIMING_SUMMARY = "direct_inference_repeat10_summary.txt"
BASELINE_FILL = "#9ecae9"
BASELINE_EDGE = "#376795"
ENRICHED_FILL = "#a1d99b"
ENRICHED_EDGE = "#2b7a2b"

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


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    table_label: str
    color: str
    primary: int | None = None
    is_linear: bool = False
    is_pod_ae: bool = False


METHODS = (
    Method("linear", "Linear HPROM", "Linear HPROM", METHOD_COLORS["linear"], is_linear=True),
    Method("case1", "HPROM--ANN C1", "HPROM--ANN Case 1", METHOD_COLORS["case1"], primary=10),
    Method("case2_n10", "HPROM--ANN C2 n=10", "HPROM--ANN Case 2 ($n=10$)", METHOD_COLORS["case2_n10"], primary=10),
    Method("case2_n20", "HPROM--ANN C2 n=20", "HPROM--ANN Case 2 ($n=20$)", METHOD_COLORS["case2_n20"], primary=20),
    Method("case3", "HPROM--ANN C3", "HPROM--ANN Case 3", METHOD_COLORS["case3"], primary=10),
    Method("podae", "HPROM-POD-AE", "HPROM-POD-AE ($n_z=10$)", METHOD_COLORS["podae"], primary=10, is_pod_ae=True),
)

# Direct maps use the same linear-HPROM coefficient data as Case 2, but do
# not evaluate an empirical-cubature residual online.  Keep them separate
# from METHODS so the intrusive run-path logic remains unambiguous, while
# including their actual saved trajectories in every coefficient diagnostic.
DIRECT_COEFF_METHODS = (
    ("podnn", "POD-NN-ROM", METHOD_COLORS["podnn"]),
    ("poddl", "POD-DL-ROM", METHOD_COLORS["poddl"]),
)

# These are the production rows common to the PROM and HPROM enrichment
# comparisons.  The n=20 Case-2 study remains reported in the dedicated
# matched-rule diagnostic, rather than creating an unmatched PROM-only row.
COMPARISON_METHODS = tuple(method for method in METHODS[1:] if method.key != "case2_n20")

# The first four entries are residual-evaluating HPROMs.  The direct maps at
# right are non-intrusive and are separated explicitly in aggregate plots.
ENRICHMENT_BAR_LABELS = (
    r"HPROM--ANN C1",
    r"HPROM--ANN C2 ($n=10$)",
    r"HPROM--ANN C3",
    r"HPROM--POD--AE ($n_z=10$)",
    r"POD--NN--ROM",
    r"POD--DL--ROM ($n_z=10$)",
)
INTRUSIVE_DIRECT_SPLIT = 3.5


TRAINING_SUMMARIES = (
    ("HPROM-ANN Case 1", STAGE3 / "case1_ann_ntot151_best_summary.txt"),
    ("Master POD-NN-ROM (Case 2 tail source)", STAGE3 / "data_driven_ann_ntot151_best_summary.txt"),
    ("HPROM-ANN Case 3", STAGE3 / "case3_ann_ntot151_best_summary.txt"),
    ("HPROM-POD-AE", STAGE3 / "prom_pod_ae_ntot151_best_summary.txt"),
    ("POD-DL-ROM", STAGE3 / "pod_dl_data_driven_ntot151_best_summary.txt"),
)

ENRICHMENT_TRAINING_SUMMARIES = (
    ("HPROM-ANN Case 1", "case1_ann_ntot151_best_summary.txt"),
    ("Master POD-NN-ROM (Case 2 tail source)", "data_driven_ann_ntot151_best_summary.txt"),
    ("HPROM-ANN Case 3", "case3_ann_ntot151_best_summary.txt"),
    ("HPROM-POD-AE", "prom_pod_ae_ntot151_best_summary.txt"),
    ("POD-DL-ROM", "pod_dl_data_driven_ntot151_best_summary.txt"),
)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def mu_tag(p: Point) -> str:
    return f"mu1_{p.mu1:.3f}_mu2_{p.mu2:.4f}"


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def tex_escape(s: object) -> str:
    txt = str(s)
    for a, b in {
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
    }.items():
        txt = txt.replace(a, b)
    return txt


def num(kv: dict[str, str], key: str) -> float | None:
    try:
        return float(kv[key])
    except Exception:
        return None


def root_for_point(p: Point) -> Path:
    if p.key == "extrapolation20pct":
        return RUNS / "Extrapolation20pct" / "ECSW2pct"
    return RUNS / "ECSW2pct"


def method_paths(method: Method, p: Point) -> tuple[Path, Path | None, Path]:
    mt = mu_tag(p)
    if method.key == "linear":
        d = RUNS / "Linear" / f"linear_hprom_{mt}_ntot151"
        return d / "summary.txt", None, d / "qN.npy"
    base = root_for_point(p)
    if method.key == "case1":
        d = base / "Case1_Best"
        stem = f"case1_hprom_ann_{mt}_n10_ntot151"
    elif method.key == "case2_n10":
        d = base / "Case2_Master" / "np10"
        stem = f"case2_hprom_ann_{mt}_n10_ntot151"
    elif method.key == "case2_n20":
        d = base / "Case2_Master" / "np20"
        stem = f"case2_hprom_ann_{mt}_n20_ntot151"
    elif method.key == "case3":
        d = base / "Case3_Best"
        stem = f"case3_hprom_ann_{mt}_n10_ntot151"
    elif method.key == "podae":
        d = base / "PODAE_Best"
        stem = f"podae_hprom_{mt}_ntot151_nz10"
    else:
        raise KeyError(method.key)
    return d / f"{stem}_summary.txt", d / f"{stem}_snaps.npy", d / f"{stem}_qN.npy"


def enriched_method_paths(method: Method, p: Point) -> tuple[Path, Path]:
    """Return the online summary and coefficient output for the 9+36 campaign."""
    mt = mu_tag(p)
    if method.key == "linear":
        d = ENRICHED_RUNS / "LinearHPROM" / f"linear_hprom_{mt}_ntot151"
        return d / "summary.txt", d / "qN.npy"
    base = ENRICHED_RUNS / "ECSW2pct"
    if method.key == "case1":
        d = base / "Case1_Best"
        stem = f"case1_hprom_ann_{mt}_n10_ntot151"
    elif method.key == "case2_n10":
        d = base / "Case2_Master" / "np10"
        stem = f"case2_hprom_ann_{mt}_n10_ntot151"
    elif method.key == "case2_n20":
        d = base / "Case2_Master" / "np20"
        stem = f"case2_hprom_ann_{mt}_n20_ntot151"
    elif method.key == "case3":
        d = base / "Case3_Best"
        stem = f"case3_hprom_ann_{mt}_n10_ntot151"
    elif method.key == "podae":
        d = base / "PODAE_Best"
        stem = f"podae_hprom_{mt}_ntot151_nz10"
    else:
        raise KeyError(method.key)
    return d / f"{stem}_summary.txt", d / f"{stem}_qN.npy"


def hdm_path(p: Point) -> Path:
    candidates = [
        REPO / "Results" / "param_snaps" / p.hdm_file,
        PAPER.parent / "Results" / "param_snaps" / p.hdm_file,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Missing HDM snapshots for {p.key}: {candidates}")


def final_x_cut_vec(vec: np.ndarray) -> np.ndarray:
    n = vec.size // 2
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Cannot infer square grid from vector size {vec.size}")
    u = vec[:n].reshape(side, side)
    return u[side // 2, :]


def final_x_cut_snap(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return final_x_cut_vec(np.asarray(arr[:, -1], dtype=np.float64))


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


def load_final_state_from_q(q_path: Path) -> np.ndarray:
    V = np.load(BASIS_PATH, mmap_mode="r")
    u_ref = np.load(U_REF_PATH, mmap_mode="r")
    q = np.load(q_path, mmap_mode="r")
    return np.asarray(u_ref, dtype=np.float64) + np.asarray(V @ np.asarray(q[:, -1], dtype=np.float64), dtype=np.float64)


def q_rel(q: np.ndarray, q_ref: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref))


def q_per_coeff_rel(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(q_ref, axis=1)
    num = np.linalg.norm(q - q_ref, axis=1)
    return 100.0 * num / np.maximum(den, 1.0e-14)


def case2_diagnostic_paths(n: int, p: Point) -> tuple[Path, Path]:
    """Return the independent baseline Case--2 HPROM diagnostic output for n."""
    if n == 0:
        d = RUNS / "DataDriven_Best" / f"rom_data_driven_{mu_tag(p)}_ntot151"
        return d / "rom_data_driven_summary.txt", d / "qN.npy"
    if n == NTOT:
        d = RUNS / "Linear" / f"linear_hprom_{mu_tag(p)}_ntot151"
        return d / "summary.txt", d / "qN.npy"
    d = CASE2_DIAGNOSTICS / "n_sweep" / "Runs" / f"np{n}"
    stem = f"case2_hprom_ann_hprom_nsweep_{mu_tag(p)}_n{n}_ntot151"
    return d / f"{stem}_summary.txt", d / f"{stem}_qN.npy"


def case2_diagnostic_metrics(n: int) -> tuple[list[float], list[float], int | None]:
    """Return state errors, q errors, and the matched ECM sample count."""
    state_errors: list[float] = []
    coefficient_errors: list[float] = []
    n_ecsw: list[int] = []
    for point in POINTS:
        summary_path, qpath = case2_diagnostic_paths(n, point)
        linear_summary, qref_path = case2_diagnostic_paths(NTOT, point)
        if not summary_path.exists() or not qpath.exists():
            raise FileNotFoundError(f"Missing Case--2 HPROM diagnostic: {summary_path} or {qpath}")
        summary = read_kv(summary_path)
        state_errors.append(float(summary["relative_error_percent"]))
        coefficient_errors.append(q_rel(np.load(qpath), np.load(qref_path)))
        ne = num(summary, "n_ecsw_elements")
        if ne is not None:
            n_ecsw.append(int(round(ne)))
    return state_errors, coefficient_errors, (int(round(float(np.median(n_ecsw)))) if n_ecsw else None)


def case2_sensitivity_rows(point: Point) -> list[dict[str, float]]:
    """Load one point of the fixed-rule n=10 secondary-tail diagnostic."""
    root = CASE2_DIAGNOSTICS / "secondary_sensitivity_n10" / point.key
    rows: list[dict[str, float]] = []
    for path in sorted(root.glob("*_summary.txt")):
        raw = read_kv(path)
        try:
            rows.append({
                "tail": float(raw["actual_secondary_error_percent"]),
                "ann_tail": float(raw["ann_secondary_error_percent"]),
                "state": float(raw["state_error_percent_vs_hdm"]),
                "primary": float(raw["primary_q_error_percent_vs_linear_hprom"]),
                "total": float(raw["total_q_error_percent_vs_linear_hprom"]),
                "n_ecsw": float(raw["n_ecsw_elements"]),
            })
        except KeyError as exc:
            raise KeyError(f"Malformed sensitivity summary: {path}") from exc
    if not rows:
        raise FileNotFoundError(f"Missing HPROM Case--2 sensitivity summaries under {root}")
    return sorted(rows, key=lambda row: row["tail"])


def write_case2_n_sweep_table() -> None:
    state_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Solved $n$ & $N_{\rm ECM}$ & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for n in CASE2_N_SWEEP:
        states, _, n_ecsw = case2_diagnostic_metrics(n)
        n_text = "--" if n_ecsw is None else str(n_ecsw)
        state_lines.append(
            f"{n} & {n_text} & "
            + " & ".join(fmt(value) for value in [*states[:3], float(np.mean(states[:3])), states[3]])
            + f" {ROW_END}"
        )
    state_lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_case2_n_sweep_state_errors.tex").write_text("\n".join(state_lines) + "\n")


def write_case2_tail_sensitivity_table() -> None:
    tail_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Point & ANN tail error & $e_u$ at exact tail & $e_u$ at ANN tail & $e_{q_p}$ at ANN tail & $e_{q_{151}}$ at ANN tail \\",
        r"\midrule",
    ]
    for point in POINTS:
        rows = case2_sensitivity_rows(point)
        zero = min(rows, key=lambda row: abs(row["tail"]))
        ann_level = rows[0]["ann_tail"]
        actual = min(rows, key=lambda row: abs(row["tail"] - ann_level))
        tail_lines.append(
            f"{point.label} & {fmt(ann_level)} & {fmt(zero['state'])} & {fmt(actual['state'])} & "
            f"{fmt(actual['primary'])} & {fmt(actual['total'])} {ROW_END}"
        )
    tail_lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_case2_n10_tail_sensitivity.tex").write_text("\n".join(tail_lines) + "\n")


def write_case2_diagnostic_tables() -> None:
    write_case2_n_sweep_table()
    write_case2_tail_sensitivity_table()


def plot_case2_n_sweep_state() -> None:
    colors = (
        METHOD_COLORS["case1"],
        METHOD_COLORS["podnn"],
        METHOD_COLORS["case3"],
        METHOD_COLORS["podae"],
    )
    n_values = np.asarray(CASE2_N_SWEEP, dtype=int)
    states = np.asarray([case2_diagnostic_metrics(int(n))[0] for n in n_values], dtype=float)
    figure, ax = plt.subplots(figsize=(12.2, 6.2))
    for i, point in enumerate(POINTS):
        ax.plot(
            n_values, states[:, i], marker="o", markersize=6.2, lw=2.35,
            color=colors[i], label=rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$",
        )
    ax.plot(n_values, np.mean(states[:, :3], axis=1), marker="s", markersize=5.8, lw=2.5,
            color="#111111", label="in-domain mean")
    ax.axvline(0, color="0.35", ls=":", lw=1.0)
    ax.axvline(NTOT, color="0.35", ls=":", lw=1.0)
    ax.set_xticks(n_values)
    ax.set_xlabel(r"solved HPROM dimension $n$")
    ax.set_ylabel(r"state relative error against HDM (\%)")
    ax.set_title(r"Case--2 HPROM master-map sweep: state error versus solved dimension")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", frameon=True, ncol=1)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "hprom_case2_n_sweep_state_errors.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_case2_n_sweep_coefficients() -> None:
    n_values = tuple(n for n in CASE2_N_SWEEP if n < NTOT)
    labels = {0: r"$n=0$ direct", **{n: rf"$n={n}$" for n in n_values if n > 0}}
    figure, axes = plt.subplots(2, 4, figsize=(14.3, 7.5), sharex=True)
    for col, point in enumerate(POINTS):
        _, qref_path = case2_diagnostic_paths(NTOT, point)
        q_ref = np.load(qref_path)
        x = np.arange(1, NTOT + 1)
        ax_abs = axes[0, col]
        ax_rel = axes[1, col]
        for n in n_values:
            _, qpath = case2_diagnostic_paths(n, point)
            delta_norm = np.linalg.norm(np.load(qpath) - q_ref, axis=1)
            rel = 100.0 * delta_norm / np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
            ax_abs.semilogy(x, np.maximum(delta_norm, 1.0e-12), color=CASE2_SWEEP_COLORS[n], lw=1.45, alpha=0.94,
                            label=labels[n])
            ax_rel.semilogy(x, np.maximum(rel, 1.0e-8), color=CASE2_SWEEP_COLORS[n], lw=1.45, alpha=0.94)
        for ax in (ax_abs, ax_rel):
            ax.axvline(10, color="0.25", ls=":", lw=0.95, alpha=0.65)
            ax.grid(True, which="both", alpha=0.21)
            ax.set_xlabel(r"coefficient index $i$")
        ax_abs.set_title(rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$")
        if col == 0:
            ax_abs.set_ylabel(r"$\lVert q_i-q_i^{\rm lin,HPROM}\rVert_2$")
            ax_rel.set_ylabel(r"$\lVert q_i-q_i^{\rm lin,HPROM}\rVert_2/\lVert q_i^{\rm lin,HPROM}\rVert_2$ (\%)")
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(r"Case--2 HPROM master-map sweep: coefficient errors versus linear HPROM", y=0.995)
    figure.legend(handles, labels_out, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=True)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(FIG_DIR / "hprom_case2_n_sweep_coeff_errors.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_case2_tail_sensitivity() -> None:
    colors = (
        METHOD_COLORS["case1"],
        METHOD_COLORS["podnn"],
        METHOD_COLORS["case3"],
        METHOD_COLORS["podae"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    for point, color in zip(POINTS, colors):
        rows = case2_sensitivity_rows(point)
        tail = np.asarray([row["tail"] for row in rows])
        state = np.asarray([row["state"] for row in rows])
        primary = np.asarray([row["primary"] for row in rows])
        ann_level = rows[0]["ann_tail"]
        index = int(np.argmin(np.abs(tail - ann_level)))
        label = rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$"
        axes[0].plot(tail, state, marker="o", markersize=5.2, lw=2.15, color=color, label=label)
        axes[1].plot(tail, primary, marker="o", markersize=5.2, lw=2.15, color=color, label=label)
        for ax, y in ((axes[0], state), (axes[1], primary)):
            ax.scatter(tail[index], y[index], marker="*", s=150, color=color, edgecolor="#222222", linewidth=0.7,
                       zorder=4)
    axes[0].set_title(r"effect on state error $\lVert u_{\rm HDM}-u\rVert_F/\lVert u_{\rm HDM}\rVert_F$")
    axes[1].set_title(r"effect on solved coordinates $q_1,\ldots,q_{10}$")
    for ax in axes:
        ax.set_xlim(left=-1.0, right=51.0)
        ax.set_xlabel(r"imposed relative error in $q_{11:151}$ only (\%)")
        ax.grid(True, alpha=0.30)
    axes[0].set_ylabel(r"state relative error against HDM (\%)")
    axes[1].set_ylabel(r"primary coefficient error against linear HPROM (\%)")
    axes[0].legend(frameon=True, loc="upper left")
    figure.suptitle(r"Case--2 HPROM, $n=10$: sensitivity to prescribed secondary-coordinate error", y=1.01)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "hprom_case2_secondary_sensitivity_state_and_primary_error.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def training_architecture(label: str) -> tuple[str, str]:
    """Return publication-ready TeX for the selected baseline networks."""
    if label == "HPROM-ANN Case 1":
        return (
            r"$\q_p\mapsto\q_s$; $10\to256\to512\to512\to256\to141$",
            "SiLU",
        )
    if label == "Master POD-NN-ROM (Case 2 tail source)":
        return (
            r"$(\mu_1,\mu_2,t)\mapsto\q_{151}$; $3\to256\to512\to512\to256\to151$",
            "SiLU",
        )
    if label == "HPROM-ANN Case 3":
        return (
            r"$(\q_p,\mu_1,\mu_2,t)\mapsto\q_s$; $13\to256\to512\to512\to256\to141$",
            "SiLU",
        )
    if label == "HPROM-POD-AE":
        return (
            r"z-score AE; $151\to512\to256\to128\to10\to128\to256\to512\to151$",
            "GELU",
        )
    if label == "POD-DL-ROM":
        return (
            r"z-score latent dynamics; $3\to256\to512\to512\to256\to10\to151$",
            "SiLU",
        )
    raise KeyError(label)


def write_training_table() -> None:
    rows = []
    for label, path in TRAINING_SUMMARIES:
        kv = read_kv(path)
        architecture, activation = training_architecture(label)
        display_label = (label.replace("HPROM-ANN", "HPROM--ANN")
                          .replace("POD-NN-ROM", "POD--NN--ROM")
                          .replace("POD-DL-ROM", "POD--DL--ROM"))
        rows.append([
            display_label,
            architecture,
            activation,
            fmt(num(kv, "train_rel_frob_percent"), 3),
            fmt(num(kv, "val_rel_frob_percent"), 3),
            f"{int(float(kv.get('trainable_parameters', 'nan'))):,}" if kv.get("trainable_parameters", "").replace(".", "", 1).isdigit() else "--",
        ])
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Map / network & Act. & Train $e_q$ (\%) & Val. $e_q$ (\%) & Params \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} {ROW_END}"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_baseline_training_errors.tex").write_text("\n".join(lines) + "\n")


def write_online_tables() -> None:
    state_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ & Mean time (s) \\",
        r"\midrule",
    ]
    coeff_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    q_refs = {p.key: np.load(method_paths(METHODS[0], p)[2]) for p in POINTS}
    for method in METHODS:
        errs: list[float | None] = []
        times: list[float] = []
        qerrs: list[float | None] = []
        for p in POINTS:
            summary, _, qpath = method_paths(method, p)
            kv = read_kv(summary)
            errs.append(num(kv, "relative_error_percent"))
            t = num(kv, "online_solve_elapsed_s")
            if t is not None:
                times.append(t)
            if qpath.exists():
                qerrs.append(q_rel(np.load(qpath), q_refs[p.key]))
            else:
                qerrs.append(None)
        mean = np.nanmean([x for x in errs[:3] if x is not None])
        qmean = np.nanmean([x for x in qerrs[:3] if x is not None])
        time_txt = fmt(float(np.mean(times)) if times else None, 2)
        state_lines.append(
            f"{method.table_label} & {fmt(errs[0])} & {fmt(errs[1])} & "
            f"{fmt(errs[2])} & {fmt(mean)} & {fmt(errs[3])} & {time_txt} {ROW_END}"
        )
        coeff_lines.append(
            f"{method.table_label} & {fmt(qerrs[0])} & {fmt(qerrs[1])} & "
            f"{fmt(qerrs[2])} & {fmt(qmean)} & {fmt(qerrs[3])} {ROW_END}"
        )
    # Direct maps share the coefficient teacher but evaluate no ECM residual.
    # Keep them in the baseline tables below a rule, just as in the paired
    # baseline/enriched tables, so the plotting and tabular evidence agree.
    state_lines.append(r"\midrule")
    coeff_lines.append(r"\midrule")
    for label, kind in (("POD--NN--ROM", "podnn"), ("POD--DL--ROM ($n_z=10$)", "poddl")):
        states, coeffs = direct_rom_metrics(False, kind)
        times: list[float] = []
        for p in POINTS:
            summary, _ = direct_rom_paths(False, kind, p)
            kv = read_kv(summary)
            # Direct maps do not execute an ECM residual solve.  Their launcher
            # records the end-to-end map evaluation under inference_time_s.
            value = num(kv, "inference_time_s")
            if value is None:
                value = num(kv, "online_solve_elapsed_s")
            if value is None:
                value = num(kv, "elapsed_s")
            if value is not None:
                times.append(value)
        state_lines.append(
            f"{label} & {fmt(states[0])} & {fmt(states[1])} & {fmt(states[2])} & "
            f"{fmt(float(np.mean(states[:3])))} & {fmt(states[3])} & {fmt(float(np.mean(times)) if times else None, 2)} {ROW_END}"
        )
        coeff_lines.append(
            f"{label} & {fmt(coeffs[0])} & {fmt(coeffs[1])} & {fmt(coeffs[2])} & "
            f"{fmt(float(np.mean(coeffs[:3])))} & {fmt(coeffs[3])} {ROW_END}"
        )
    state_lines += [r"\bottomrule", r"\end{tabular}"]
    coeff_lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_baseline_online_errors.tex").write_text("\n".join(state_lines) + "\n")
    (TAB_DIR / "hprom_baseline_online_coeff_errors.tex").write_text("\n".join(coeff_lines) + "\n")


def generate_solution_overlay(enriched: bool = False) -> None:
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
                ax.plot(grid, hdm_lines[column], color="#111111", lw=0.9, alpha=0.22)
            ax.plot(
                grid,
                state_cut_lines_from_snaps(hdm, time_ids[-1])[column],
                color="#111111",
                lw=2.4,
                label="HDM" if row == 0 and column == 0 else None,
            )
            for method in METHODS:
                if enriched:
                    summary, qpath = campaign_paths(True, method, p)
                    snaps = None
                else:
                    summary, snaps, qpath = method_paths(method, p)
                if not summary.exists() or not qpath.exists():
                    continue
                if snaps is not None and snaps.exists():
                    line_getter = lambda tidx, path=snaps: state_cut_lines_from_snaps(path, tidx)
                else:
                    line_getter = lambda tidx, path=qpath: state_cut_lines_from_q(path, tidx)
                for tidx in time_ids[:-1]:
                    ax.plot(grid, line_getter(tidx)[column], color=method.color, lw=0.85, alpha=0.20)
                ax.plot(
                    grid,
                    line_getter(time_ids[-1])[column],
                    color=method.color,
                    lw=1.75,
                    alpha=0.96,
                    label=method.label if row == 0 and column == 0 else None,
                )
            for label, kind, color_key in (
                ("POD-NN-ROM", "podnn", "podnn"),
                ("POD-DL-ROM", "poddl", "poddl"),
            ):
                summary, qpath = direct_rom_paths(enriched, kind, p)
                if not summary.exists() or not qpath.exists():
                    continue
                for tidx in time_ids[:-1]:
                    ax.plot(grid, state_cut_lines_from_q(qpath, tidx)[column], color=METHOD_COLORS[color_key], lw=0.85, alpha=0.20)
                ax.plot(
                    grid,
                    state_cut_lines_from_q(qpath, time_ids[-1])[column],
                    color=METHOD_COLORS[color_key],
                    lw=1.75,
                    alpha=0.96,
                    label=label if row == 0 and column == 0 else None,
                )
            ax.set_title(rf"{p.label}: $\mu=({p.mu1:.3f},{p.mu2:.4f})$: {point_role(p)}: {cut_label}")
            ax.set_xlabel(r"$x$" if column == 0 else r"$y$")
            ax.set_ylabel(r"$u_x$")
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(*STATE_CUTPLANE_YLIM)
            ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    order = ["HDM"] + [m.label for m in METHODS] + ["POD-NN-ROM", "POD-DL-ROM"]
    by_label = dict(zip(labels, handles))
    ordered = [(lab, by_label[lab]) for lab in order if lab in by_label]
    fig.legend([h for _, h in ordered], [lab for lab, _ in ordered], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.975))
    campaign = "enriched HPROM campaign" if enriched else "baseline HPROM campaign"
    fig.suptitle(f"{campaign}: solution cut-plane overlays", y=0.995)
    fig.text(0.5, 0.012, "Fainter solid curves: intermediate times; opaque solid curves: final time.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.93))
    stem = "hprom_enriched_solution_overlays.png" if enriched else "hprom_baseline_solution_overlays.png"
    fig.savefig(FIG_DIR / stem, dpi=220, bbox_inches="tight")
    plt.close(fig)


def coefficient_method_specs() -> tuple[tuple[str, str, str], ...]:
    """Ordered plotting set: intrusive closures followed by the direct maps."""
    # The detailed baseline/enriched figures mirror the PROM presentation:
    # they show the production n=10 Case-2 closure and the two direct maps.
    # The n=20 variant is retained in the state/coefficient tables and in the
    # dedicated Case-2 sweep, where its role is unambiguous.
    return tuple(
        (method.key, method.label, method.color)
        for method in METHODS[1:]
        if method.key != "case2_n20"
    ) + DIRECT_COEFF_METHODS


def coefficient_errors(enriched: bool = False) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Return trajectory, relative, and time-resolved coefficient errors."""
    errors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for p in POINTS:
        _, qref_path = campaign_paths(enriched, METHODS[0], p)
        q_ref = np.load(qref_path)
        ref_norm = np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
        for method in METHODS[1:]:
            _, qpath = campaign_paths(enriched, method, p)
            if not qpath.exists():
                continue
            error = np.load(qpath) - q_ref
            errors[(p.key, method.key)] = {
                "abs_curve": np.linalg.norm(error, axis=1),
                "rel_curve": 100.0 * np.linalg.norm(error, axis=1) / ref_norm,
                "abs_heat": np.abs(error),
                "rel_heat": 100.0 * np.abs(error) / ref_norm[:, None],
            }
        for kind, _label, _color in DIRECT_COEFF_METHODS:
            _, qpath = direct_rom_paths(enriched, kind, p)
            if not qpath.exists():
                continue
            error = np.load(qpath) - q_ref
            errors[(p.key, kind)] = {
                "abs_curve": np.linalg.norm(error, axis=1),
                "rel_curve": 100.0 * np.linalg.norm(error, axis=1) / ref_norm,
                "abs_heat": np.abs(error),
                "rel_heat": 100.0 * np.abs(error) / ref_norm[:, None],
            }
    return errors


def generate_coeff_plot(enriched: bool = False) -> None:
    errors = coefficient_errors(enriched)
    methods = coefficient_method_specs()
    fig, axes = plt.subplots(2, len(POINTS), figsize=(16.2, 7.1), sharex=True)
    x = np.arange(1, NTOT + 1)
    for column, p in enumerate(POINTS):
        ax_abs, ax_rel = axes[0, column], axes[1, column]
        for key, label, color in methods:
            error = errors.get((p.key, key))
            if error is None:
                continue
            for ax, value in ((ax_abs, error["abs_curve"]), (ax_rel, error["rel_curve"])):
                ax.semilogy(
                    x,
                    np.maximum(value, 1.0e-14),
                    color=color,
                    lw=1.75,
                    alpha=0.95,
                    label=label if ax is ax_abs else None,
                )
        for ax in (ax_abs, ax_rel):
            ax.axvline(10, color="0.25", ls=":", lw=1.0, alpha=0.7)
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
    ordered = [(label, by_label[label]) for _, label, _ in methods if label in by_label]
    fig.legend([h for _, h in ordered], [label for label, _ in ordered], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.05, h_pad=0.8)
    stem = "hprom_enriched_coeff_abs_rel_errors.png" if enriched else "hprom_baseline_coeff_abs_rel_errors.png"
    fig.savefig(FIG_DIR / stem, dpi=220)
    plt.close(fig)


def generate_coefficient_heatmaps(enriched: bool = False) -> None:
    errors = coefficient_errors(enriched)
    methods = coefficient_method_specs()
    for kind, vmax, label, stem in (
        ("abs_heat", COEFF_ABS_HEAT_VMAX, r"$|q_i-q_i^{\\mathrm{ref}}|$", "abs"),
        ("rel_heat", COEFF_REL_PERCENT_HEAT_VMAX, r"relative coefficient error (\%)", "rel"),
    ):
        fig, axes = plt.subplots(len(methods), len(POINTS), figsize=(15.6, 11.0), sharex=True, sharey=True)
        image = None
        for row, (key, method_label, _color) in enumerate(methods):
            for column, p in enumerate(POINTS):
                ax = axes[row, column]
                error = errors.get((p.key, key))
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
                    ax.set_ylabel(method_label)
                if row == len(methods) - 1:
                    ax.set_xlabel("time")
                ax.grid(False)
        fig.subplots_adjust(left=0.17, right=0.88, bottom=0.075, top=0.92, wspace=0.15, hspace=0.24)
        fig.supylabel("coefficient index", x=0.045)
        cax = fig.add_axes([0.905, 0.16, 0.017, 0.65])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(label)
        prefix = "hprom_enriched" if enriched else "hprom_baseline"
        fig.savefig(FIG_DIR / f"{prefix}_coeff_{stem}_heatmaps.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_ecm_table() -> None:
    linear = read_kv(method_paths(METHODS[0], POINTS[0])[0])
    case1 = read_kv(method_paths(METHODS[1], POINTS[0])[0])
    case2n20 = read_kv(method_paths(METHODS[3], POINTS[0])[0])
    lines = [
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Family & Snapshot pct. & $N_{\rm ECM}$ & SVD / sampling rule \\",
        r"\midrule",
        f"Linear HPROM & {fmt(num(linear, 'ecsw_snapshot_percent'), 1)} & "
        f"{int(num(linear, 'n_ecsw_elements') or 0)} & "
        f"direct dense SVD, global parameter--time stratified {ROW_END}",
        f"Learned HPROM, $n=10$ & {fmt(num(case1, 'ecsw_snapshot_percent'), 1)} & "
        f"{int(num(case1, 'n_ecsw_elements') or 0)} & "
        f"direct dense SVD, global parameter--time stratified {ROW_END}",
        f"Learned HPROM, $n=20$ & {fmt(num(case2n20, 'ecsw_snapshot_percent'), 1)} & "
        f"{int(num(case2n20, 'n_ecsw_elements') or 0)} & "
        f"direct dense SVD, global parameter--time stratified {ROW_END}",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TAB_DIR / "hprom_ecm_setup.tex").write_text("\n".join(lines) + "\n")


def campaign_paths(enriched: bool, method: Method, p: Point) -> tuple[Path, Path]:
    if enriched:
        return enriched_method_paths(method, p)
    summary, _, qpath = method_paths(method, p)
    return summary, qpath


def campaign_metrics(enriched: bool, method: Method) -> tuple[list[float], list[float], list[float]]:
    """Return state errors, coefficient errors, and online times at the four points."""
    state_errors: list[float] = []
    coefficient_errors: list[float] = []
    online_times: list[float] = []
    for p in POINTS:
        summary, qpath = campaign_paths(enriched, method, p)
        reference_summary, reference_qpath = campaign_paths(enriched, METHODS[0], p)
        if not summary.exists() or not qpath.exists():
            raise FileNotFoundError(f"Missing HPROM output: {summary} or {qpath}")
        if not reference_summary.exists() or not reference_qpath.exists():
            raise FileNotFoundError(f"Missing linear HPROM reference: {reference_summary} or {reference_qpath}")
        kv = read_kv(summary)
        state_errors.append(float(kv["relative_error_percent"]))
        coefficient_errors.append(q_rel(np.load(qpath), np.load(reference_qpath)))
        online_times.append(float(kv.get("online_solve_elapsed_s", "nan")))
    return state_errors, coefficient_errors, online_times


def direct_rom_paths(enriched: bool, kind: str, p: Point) -> tuple[Path, Path]:
    """Return a direct-map output trained on baseline or enriched HPROM data."""
    runs = ENRICHED_RUNS if enriched else RUNS
    mt = mu_tag(p)
    if kind == "podnn":
        d = runs / "DataDriven_Best" / f"rom_data_driven_{mt}_ntot151"
        return d / "rom_data_driven_summary.txt", d / "qN.npy"
    if kind == "poddl":
        d = runs / "PODDL_Best" / f"pod_dl_data_driven_{mt}_ntot151_nz10"
        return d / "pod_dl_data_driven_summary.txt", d / "qN.npy"
    raise KeyError(kind)


def direct_rom_metrics(enriched: bool, kind: str) -> tuple[list[float], list[float]]:
    """Return direct-map state and coefficient errors at the reported points."""
    states: list[float] = []
    coeffs: list[float] = []
    for p in POINTS:
        summary, qpath = direct_rom_paths(enriched, kind, p)
        _, qref_path = campaign_paths(enriched, METHODS[0], p)
        if not summary.exists() or not qpath.exists():
            regime = "enriched" if enriched else "baseline"
            raise FileNotFoundError(
                f"Missing {regime} direct-ROM output: {summary} or {qpath}"
            )
        states.append(float(read_kv(summary)["relative_error_percent"]))
        coeffs.append(q_rel(np.load(qpath), np.load(qref_path)))
    return states, coeffs


def repeated_direct_rom_time(enriched: bool, kind: str) -> float | None:
    """Read the all-points repeated forward-pass timing summary, if available.

    The one-shot timestamps saved with the accuracy launchers include unrelated
    process and allocation effects.  They are deliberately not used for the
    manuscript timing comparison.
    """
    root = HPROM_ENRICHED if enriched else HPROM
    summary = root / "timing" / DIRECT_TIMING_SUMMARY
    return num(read_kv(summary), f"{kind}_all_points_mean_inference_time_s")


def fmt_duration(seconds: float | None) -> str:
    """Use enough digits to distinguish the millisecond-scale direct maps."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "--"
    if seconds < 0.1:
        return f"{seconds:.4f}"
    return f"{seconds:.3f}" if seconds < 1.0 else f"{seconds:.2f}"


def fmt_speedup(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{value:,.1f}"


def plot_direct_rom_enrichment_comparison() -> None:
    """Show the direct-map gain without conflating it with an ECM approximation."""
    methods = (("POD--NN--ROM", "podnn"), ("POD--DL--ROM", "poddl"))
    x = np.arange(len(methods))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex="col")
    panels = (
        (axes[0, 0], "state", 0, r"in-domain mean state error"),
        (axes[0, 1], "state", 3, r"extrapolatory state error"),
        (axes[1, 0], "coeff", 0, r"in-domain mean coefficient error"),
        (axes[1, 1], "coeff", 3, r"extrapolatory coefficient error"),
    )
    for ax, metric, point_index, title in panels:
        base_values = []
        enriched_values = []
        for _, kind in methods:
            base_state, base_coeff = direct_rom_metrics(False, kind)
            enriched_state, enriched_coeff = direct_rom_metrics(True, kind)
            base = base_state if metric == "state" else base_coeff
            enriched = enriched_state if metric == "state" else enriched_coeff
            base_values.append(float(np.mean(base[:3])) if point_index == 0 else base[point_index])
            enriched_values.append(float(np.mean(enriched[:3])) if point_index == 0 else enriched[point_index])
        ax.bar(x - width / 2, base_values, width, color=BASELINE_FILL, edgecolor=BASELINE_EDGE,
               alpha=0.96, label="baseline 9")
        ax.bar(x + width / 2, enriched_values, width, color=ENRICHED_FILL, edgecolor=ENRICHED_EDGE,
               alpha=0.96, label="enriched 9+36")
        if metric == "state":
            linear_state, _, _ = campaign_metrics(False, METHODS[0])
            ref = float(np.mean(linear_state[:3])) if point_index == 0 else linear_state[point_index]
        ax.axhline(ref, color="#222222", ls="-", lw=1.05,
                       label="linear HPROM" if ax is axes[0, 0] else None)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in methods])
        ax.grid(axis="y", which="both", alpha=0.25)
    axes[0, 0].set_ylabel(r"state error vs HDM (\%)")
    axes[1, 0].set_ylabel(r"coefficient error vs linear HPROM (\%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(FIG_DIR / "hprom_enrichment_direct_rom_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_enrichment_training_table() -> None:
    lines = [
        r"\begin{tabular}{lrr|rrr}",
        r"\toprule",
        r"Model & Base train $e_q$ (\%) & Base val. $e_q$ (\%) & Enriched train $e_q$ (\%) & Enriched val. $e_q$ (\%) & Val. reduction (\%) \\",
        r"\midrule",
    ]
    for label, filename in ENRICHMENT_TRAINING_SUMMARIES:
        base = read_kv(STAGE3 / filename)
        enriched = read_kv(ENRICHED_STAGE3 / filename)
        base_val = num(base, "val_rel_frob_percent")
        enriched_val = num(enriched, "val_rel_frob_percent")
        reduction = 100.0 * (base_val - enriched_val) / base_val if base_val and enriched_val is not None else None
        display_label = (label.replace("HPROM-ANN", "HPROM--ANN")
                          .replace("POD-NN-ROM", "POD--NN--ROM")
                          .replace("POD-DL-ROM", "POD--DL--ROM ($n_z=10$)")
                          .replace("HPROM-POD-AE", "HPROM-POD-AE ($n_z=10$)"))
        lines.append(
            f"{display_label} & {fmt(num(base, 'train_rel_frob_percent'))} & "
            f"{fmt(base_val)} & {fmt(num(enriched, 'train_rel_frob_percent'))} & "
            f"{fmt(enriched_val)} & {fmt(reduction, 1)} {ROW_END}"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_enrichment_training_comparison.tex").write_text("\n".join(lines) + "\n")


def write_enrichment_online_tables() -> None:
    state_lines = [
        r"\begin{tabular}{lrrrrr|rrrrr}",
        r"\toprule",
        r"& \multicolumn{5}{c|}{Baseline (9 trajectories)} & \multicolumn{5}{c}{Enriched (9+36 trajectories)} \\",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    coeff_lines = [
        r"\begin{tabular}{lrrrrr|rrrrr}",
        r"\toprule",
        r"& \multicolumn{5}{c|}{Baseline (9 trajectories)} & \multicolumn{5}{c}{Enriched (9+36 trajectories)} \\",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\",
        r"\midrule",
    ]
    for method in (METHODS[0], *COMPARISON_METHODS):
        base_state, base_coeff, _ = campaign_metrics(False, method)
        enriched_state, enriched_coeff, _ = campaign_metrics(True, method)
        base_state_fmt = [*base_state[:3], float(np.mean(base_state[:3])), base_state[3]]
        enriched_state_fmt = [*enriched_state[:3], float(np.mean(enriched_state[:3])), enriched_state[3]]
        base_coeff_fmt = [*base_coeff[:3], float(np.mean(base_coeff[:3])), base_coeff[3]]
        enriched_coeff_fmt = [*enriched_coeff[:3], float(np.mean(enriched_coeff[:3])), enriched_coeff[3]]
        state_lines.append(
            f"{method.table_label} & " + " & ".join(fmt(v) for v in base_state_fmt + enriched_state_fmt) + f" {ROW_END}"
        )
        coeff_lines.append(
            f"{method.table_label} & " + " & ".join(fmt(v) for v in base_coeff_fmt + enriched_coeff_fmt) + f" {ROW_END}"
        )
    # Separate intrusive sampled-residual models from direct coefficient maps.
    state_lines.append(r"\midrule")
    coeff_lines.append(r"\midrule")
    for label, kind in (("POD--NN--ROM", "podnn"), ("POD--DL--ROM ($n_z=10$)", "poddl")):
        base_state, base_coeff = direct_rom_metrics(False, kind)
        enriched_state, enriched_coeff = direct_rom_metrics(True, kind)
        base_state_fmt = [*base_state[:3], float(np.mean(base_state[:3])), base_state[3]]
        enriched_state_fmt = [*enriched_state[:3], float(np.mean(enriched_state[:3])), enriched_state[3]]
        base_coeff_fmt = [*base_coeff[:3], float(np.mean(base_coeff[:3])), base_coeff[3]]
        enriched_coeff_fmt = [*enriched_coeff[:3], float(np.mean(enriched_coeff[:3])), enriched_coeff[3]]
        state_lines.append(
            f"{label} & " + " & ".join(fmt(v) for v in base_state_fmt + enriched_state_fmt) + f" {ROW_END}"
        )
        coeff_lines.append(
            f"{label} & " + " & ".join(fmt(v) for v in base_coeff_fmt + enriched_coeff_fmt) + f" {ROW_END}"
        )
    state_lines += [r"\bottomrule", r"\end{tabular}"]
    coeff_lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_enrichment_online_state_comparison.tex").write_text("\n".join(state_lines) + "\n")
    (TAB_DIR / "hprom_enrichment_online_coeff_comparison.tex").write_text("\n".join(coeff_lines) + "\n")


def write_enrichment_online_timing_table() -> None:
    """Write wall times and speed-ups against one fixed HDM reference.

    The linear HPROM is not retrained during coefficient enrichment: its basis
    and ECM rule remain fixed.  Its measured baseline time is consequently
    reproduced in both columns rather than treating run-to-run wall-clock
    variation as an algorithmic change.
    """
    fixed_linear_time = float(np.mean(campaign_metrics(False, METHODS[0])[2]))
    lines = [
        r"\begin{tabular}{lrr|rr}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{Baseline (9 trajectories)} & \multicolumn{2}{c}{Enriched (9+36 trajectories)} \\",
        r"Model & Mean time (s) & $T_{\mathrm{HDM}}/T$ & Mean time (s) & $T_{\mathrm{HDM}}/T$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        baseline_time = float(np.mean(campaign_metrics(False, method)[2]))
        enriched_time = fixed_linear_time if method.is_linear else float(np.mean(campaign_metrics(True, method)[2]))
        lines.append(
            f"{method.table_label} & {fmt_duration(baseline_time)} & "
            f"{fmt_speedup(HDM_REFERENCE_TIME_S / baseline_time)} & "
            f"{fmt_duration(enriched_time)} & {fmt_speedup(HDM_REFERENCE_TIME_S / enriched_time)} {ROW_END}"
        )
    lines.append(r"\midrule")
    for label, kind in (("POD--NN--ROM", "podnn"), ("POD--DL--ROM ($n_z=10$)", "poddl")):
        baseline_time = repeated_direct_rom_time(False, kind)
        enriched_time = repeated_direct_rom_time(True, kind)
        lines.append(
            f"{label} & {fmt_duration(baseline_time)} & "
            f"{fmt_speedup(HDM_REFERENCE_TIME_S / baseline_time if baseline_time else None)} & "
            f"{fmt_duration(enriched_time)} & {fmt_speedup(HDM_REFERENCE_TIME_S / enriched_time if enriched_time else None)} {ROW_END}"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "hprom_enrichment_online_timing_comparison.tex").write_text("\n".join(lines) + "\n")


def plot_enrichment_state_comparison() -> None:
    methods: tuple[tuple[str, Method | str], ...] = tuple((m.label, m) for m in COMPARISON_METHODS) + (
        ("POD-NN-ROM", "podnn"), ("POD-DL-ROM", "poddl"),
    )
    x = np.arange(len(methods))
    width = 0.36
    linear_state, _, _ = campaign_metrics(False, METHODS[0])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    panels = (
        (axes[0], 0, r"in-domain mean: $\mu^{(v)},\mu^{(1)},\mu^{(2)}$", float(np.mean(linear_state[:3]))),
        (axes[1], 3, r"extrapolatory point $\mu^{(3)}$", linear_state[3]),
    )
    for ax, point_index, title, linear_value in panels:
        base_values = []
        enriched_values = []
        for _label, method in methods:
            if isinstance(method, Method):
                base, _, _ = campaign_metrics(False, method)
                enriched, _, _ = campaign_metrics(True, method)
            else:
                base, _ = direct_rom_metrics(False, method)
                enriched, _ = direct_rom_metrics(True, method)
            base_values.append(float(np.mean(base[:3])) if point_index == 0 else base[point_index])
            enriched_values.append(float(np.mean(enriched[:3])) if point_index == 0 else enriched[point_index])
        ax.bar(x - width / 2, base_values, width, color=BASELINE_FILL, edgecolor=BASELINE_EDGE,
               alpha=0.92, label="baseline 9")
        ax.bar(x + width / 2, enriched_values, width, color=ENRICHED_FILL, edgecolor=ENRICHED_EDGE,
               alpha=0.92, label="enriched 9+36")
        ax.axhline(linear_value, color="#222222", ls="-", lw=1.1, label="linear HPROM" if ax is axes[0] else None)
        ax.axvline(INTRUSIVE_DIRECT_SPLIT, color="#5a5a5a", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ENRICHMENT_BAR_LABELS, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", which="both", alpha=0.27)
    axes[0].set_ylabel(r"state relative error vs HDM (\%)")
    axes[0].legend(frameon=True, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "hprom_enrichment_state_error_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_enrichment_coeff_comparison() -> None:
    methods: tuple[tuple[str, Method | str], ...] = tuple((m.label, m) for m in COMPARISON_METHODS) + (
        ("POD-NN-ROM", "podnn"), ("POD-DL-ROM", "poddl"),
    )
    x = np.arange(len(methods))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    panels = (
        (axes[0], 0, r"in-domain mean coefficient error"),
        (axes[1], 3, r"extrapolatory coefficient error"),
    )
    for ax, point_index, title in panels:
        base_values = []
        enriched_values = []
        for _label, method in methods:
            if isinstance(method, Method):
                _, base, _ = campaign_metrics(False, method)
                _, enriched, _ = campaign_metrics(True, method)
            else:
                _, base = direct_rom_metrics(False, method)
                _, enriched = direct_rom_metrics(True, method)
            base_values.append(float(np.mean(base[:3])) if point_index == 0 else base[point_index])
            enriched_values.append(float(np.mean(enriched[:3])) if point_index == 0 else enriched[point_index])
        ax.bar(x - width / 2, base_values, width, color=BASELINE_FILL, edgecolor=BASELINE_EDGE,
               alpha=0.92, label="baseline 9")
        ax.bar(x + width / 2, enriched_values, width, color=ENRICHED_FILL, edgecolor=ENRICHED_EDGE,
               alpha=0.92, label="enriched 9+36")
        ax.axvline(INTRUSIVE_DIRECT_SPLIT, color="#5a5a5a", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ENRICHMENT_BAR_LABELS, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", which="both", alpha=0.27)
    axes[0].set_ylabel(r"relative coefficient error vs linear HPROM (\%)")
    axes[0].legend(frameon=True, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "hprom_enrichment_coeff_error_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_enrichment_case2_coefficients() -> None:
    comparison = (
        (False, METHODS[2], r"baseline, $n=10$", BASELINE_EDGE, "o"),
        (True, METHODS[2], r"enriched, $n=10$", ENRICHED_EDGE, "o"),
    )
    x = np.arange(1, NTOT + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), POINTS):
        for enriched, method, label, color, marker in comparison:
            _, qpath = campaign_paths(enriched, method, p)
            _, qref_path = campaign_paths(enriched, METHODS[0], p)
            y = np.maximum(q_per_coeff_rel(np.load(qpath), np.load(qref_path)), 1.0e-8)
            ax.semilogy(x, y, color=color, lw=1.55, alpha=0.88, marker=marker,
                        markevery=16, markersize=3.3, label=label)
        ax.axvline(10, color="0.25", ls=":", lw=1.0, alpha=0.75)
        ax.axvline(20, color="0.45", ls=":", lw=0.9, alpha=0.55)
        ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.set_xlabel("coefficient index")
        ax.set_ylabel(r"relative coefficient error (\%)")
        ax.grid(True, which="both", alpha=0.23)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(FIG_DIR / "hprom_enrichment_case2_coeff_rel_errors.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HPROM manuscript assets from saved campaign outputs."
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--skip-case2-diagnostics",
        action="store_true",
        help=(
            "Preserve the Case-2 HPROM n-sweep and tail-sensitivity assets. "
            "Use while the diagnostic rerun is pending."
        ),
    )
    diagnostics.add_argument(
        "--only-case2-n-sweep",
        action="store_true",
        help=(
            "Refresh only the Case-2 n-sweep table and figures; preserve the "
            "fixed-rule tail-sensitivity diagnostic."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.only_case2_n_sweep:
        write_case2_n_sweep_table()
        plot_case2_n_sweep_state()
        plot_case2_n_sweep_coefficients()
        print("[hprom-assets] refreshed the Case-2 n-sweep only")
        return

    write_training_table()
    write_online_tables()
    write_ecm_table()
    write_enrichment_training_table()
    write_enrichment_online_tables()
    write_enrichment_online_timing_table()
    generate_solution_overlay()
    generate_coeff_plot()
    generate_coefficient_heatmaps()
    if args.skip_case2_diagnostics:
        print("[hprom-assets] preserved existing Case-2 diagnostic assets")
    else:
        write_case2_diagnostic_tables()
        plot_case2_n_sweep_state()
        plot_case2_n_sweep_coefficients()
        plot_case2_tail_sensitivity()
    plot_enrichment_state_comparison()
    plot_enrichment_coeff_comparison()
    plot_enrichment_case2_coefficients()
    generate_solution_overlay(enriched=True)
    generate_coeff_plot(enriched=True)
    generate_coefficient_heatmaps(enriched=True)
    print(f"[hprom-assets] wrote {TAB_DIR}")
    print(f"[hprom-assets] wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
