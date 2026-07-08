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
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "text.usetex": False,
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
    "HDM": "#111111",
    "Linear PROM": "#4C78A8",
    "PROM-ANN C1": "#F58518",
    "PROM-ANN C2": "#54A24B",
    "PROM-ANN C2 n20": "#1B7F3A",
    "PROM-ANN C3": "#B279A2",
    "PROM-POD-AE": "#E45756",
    "POD-NN-ROM": "#72B7B2",
    "POD-DL-ROM": "#9D755D",
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


def final_x_cut(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    vec = np.asarray(arr[:, -1], dtype=np.float64)
    n = vec.size // 2
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Cannot infer square grid from {path}: {arr.shape}")
    u = vec[:n].reshape(side, side)
    return u[side // 2, :]


def generate_solution_overlay(rows: list[dict[str, object]]) -> Path:
    methods = ["HDM", "Linear PROM", "PROM-ANN C1", "PROM-ANN C2", "PROM-ANN C3", "PROM-POD-AE", "POD-NN-ROM", "POD-DL-ROM"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True)
    for ax, p in zip(axes.ravel(), POINTS):
        x = np.linspace(0.0, 100.0, final_x_cut(hdm_path(p)).size)
        ax.plot(x, final_x_cut(hdm_path(p)), color=COLORS["HDM"], lw=2.2, label="HDM")
        for method in methods[1:]:
            summary, snaps, _ = summary_and_snaps(method, p)
            kv = read_kv(summary)
            if not kv or snaps is None or not snaps.exists() or not is_current(method, kv):
                continue
            ax.plot(x, final_x_cut(snaps), color=COLORS[method], lw=1.4, alpha=0.78, label=method)
        ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x at final time, midline")
        ax.set_ylabel("first state component")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # Preserve full method order while dropping missing duplicates.
    by_label = dict(zip(labels, handles))
    ordered = [(m, by_label[m]) for m in methods if m in by_label]
    fig.legend([h for _, h in ordered], [m for m, _ in ordered], loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "prom_only_solution_overlays.png"
    fig.savefig(out, dpi=220)
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


def generate_coeff_error_plot() -> Path:
    methods = [
        "PROM-ANN C1",
        "PROM-ANN C2",
        "PROM-ANN C3",
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-DL-ROM",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), POINTS):
        qref = online_q_for_method("Linear PROM", p)
        if qref is None:
            raise FileNotFoundError(f"Missing linear PROM qN for {p.key}")
        for method in methods:
            q = online_q_for_method(method, p)
            if q is None:
                continue
            denom = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
            rel = 100.0 * np.linalg.norm(q - qref, axis=1) / denom
            ax.semilogy(
                np.arange(1, NTOT + 1),
                rel,
                color=COLORS[method],
                lw=1.45,
                alpha=0.80,
                label=method,
            )
        ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.axvline(10, color="#333333", lw=1.0, ls="--", alpha=0.65)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("coefficient index")
        ax.set_ylabel("relative coefficient error (%)")
        ax.set_ylim(1e-3, 2e2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = [(name, by_label[name]) for name in methods if name in by_label]
    fig.legend([h for _, h in ordered], [m for m, _ in ordered], loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "prom_only_coeff_rel_errors.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def generate_case2_n10_n20_coeff_plot() -> Path:
    methods = [
        ("POD-NN-ROM", "POD-NN-ROM ($n=0$)", "#72B7B2", "-"),
        ("PROM-ANN C2", "Case 2 ($n=10$)", "#54A24B", "-"),
        ("PROM-ANN C2 n20", "Case 2 ($n=20$)", "#1B7F3A", "-"),
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
            ax.semilogy(np.arange(1, NTOT + 1), rel, color=color, lw=1.7, ls=ls, alpha=0.86, label=label)
        ax.axvline(10, color="#444444", lw=1.0, ls="--", alpha=0.65)
        ax.axvline(20, color="#444444", lw=1.0, ls=":", alpha=0.75)
        ax.text(10.5, 1.5e-3, "n=10", rotation=90, va="bottom", ha="left", fontsize=7, color="#444444")
        ax.text(20.5, 1.5e-3, "n=20", rotation=90, va="bottom", ha="left", fontsize=7, color="#444444")
        ax.set_title(f"{p.label}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("coefficient index")
        ax.set_ylabel("relative coefficient error (%)")
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
            time_s = numeric_from_summary(kv, "online_solve_elapsed_s")
            if time_s is None:
                time_s = numeric_from_summary(kv, "inference_time_s")
            rows.append({"method": method, "point": p.key, "label": p.label, "err": err, "time_s": time_s if ok else None, "ok": ok})
    return rows


def method_summary(rows: list[dict[str, object]], method: str) -> tuple[list[float | None], float | None, bool]:
    vals = [next(r for r in rows if r["method"] == method and r["point"] == p.key)["err"] for p in POINTS]
    times = [next(r for r in rows if r["method"] == method and r["point"] == p.key)["time_s"] for p in POINTS]
    ok = all(next(r for r in rows if r["method"] == method and r["point"] == p.key)["ok"] for p in POINTS)
    valid_times = [float(t) for t in times if t is not None]
    return vals, (sum(valid_times) / len(valid_times) if valid_times else None), ok


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
        "PROM-ANN C2": "PROM--ANN Case 2, $n=10$",
        "PROM-ANN C2 n20": "PROM--ANN Case 2, $n=20$",
        "PROM-ANN C3": "PROM--ANN Case 3",
        "PROM-POD-AE": "PROM--POD--AE, $n_z=10$",
        "POD-NN-ROM": "POD--NN--ROM",
        "POD-DL-ROM": "POD--DL--ROM, $n_z=10$",
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & mean & $\mu^{(3)}$ & time (s) \\",
        r"\midrule",
    ]
    for method in methods:
        vals, time_s, ok = method_summary(rows, method)
        mean = None if any(v is None for v in vals[:3]) else sum(float(v) for v in vals[:3]) / 3.0
        row = [labels[method], *(fmt(v, 3) for v in vals[:3]), fmt(mean, 3), fmt(vals[3], 3), fmt(time_s, 3 if (time_s is not None and time_s < 1.0) else 1)]
        if not ok:
            row[0] += r"$^{\dagger}$"
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "prom_only_online_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_online_coeff_table() -> Path:
    methods = ["PROM-ANN C1", "PROM-ANN C2", "PROM-ANN C2 n20", "PROM-ANN C3", "PROM-POD-AE", "POD-NN-ROM", "POD-DL-ROM"]
    labels = {
        "PROM-ANN C1": "Case 1",
        "PROM-ANN C2": "Case 2 $n=10$",
        "PROM-ANN C2 n20": "Case 2 $n=20$",
        "PROM-ANN C3": "Case 3",
        "PROM-POD-AE": "POD--AE",
        "POD-NN-ROM": "POD--NN--ROM",
        "POD-DL-ROM": "POD--DL--ROM",
    }
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Point & " + " & ".join(labels[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    for p in POINTS:
        qref = online_q_for_method("Linear PROM", p)
        if qref is None:
            raise FileNotFoundError(f"Missing linear PROM qN for {p.key}")
        vals = []
        for method in methods:
            q = online_q_for_method(method, p)
            vals.append(fmt(rel_q(q, qref), 3) if q is not None else "--")
        lines.append(f"{p.label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "prom_only_online_coeff_errors.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def stage3_value(path: Path, key: str) -> str:
    return read_kv(path).get(key, "--")


def write_training_table() -> Path:
    specs = [
        ("PROM--ANN Case 1", "$(q_1,\\ldots,q_{10})\\mapsto(q_{11},\\ldots,q_{151})$", STAGE3 / "case1_ann_ntot151_best_summary.txt"),
        ("PROM--ANN Case 3", "$(q_1,\\ldots,q_{10},\\mu_1,\\mu_2,t)\\mapsto(q_{11},\\ldots,q_{151})$", STAGE3 / "case3_ann_ntot151_best_summary.txt"),
        ("Master POD--NN--ROM (Case 2 tail source)", "$(\\mu_1,\\mu_2,t)\\mapsto(q_1,\\ldots,q_{151})$", STAGE3 / "master_ann_mu_t_to_qtot_ntot151_best_summary.txt"),
        ("PROM--POD--AE", "$q_{1:151}\\mapsto z_{1:10}\\mapsto q_{1:151}$", STAGE3 / "prom_pod_ae_ntot151_best_summary.txt"),
        ("POD--DL--ROM", "$(\\mu_1,\\mu_2,t)\\mapsto z_{1:10}\\mapsto q_{1:151}$", STAGE3 / "pod_dl_data_driven_ntot151_best_summary.txt"),
    ]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & learned map & parameters & train (\%) & validation (\%) \\",
        r"\midrule",
    ]
    for model, mapping, path in specs:
        kv = read_kv(path)
        params = kv.get("trainable_parameters", "--")
        train = kv.get("train_rel_frob_percent", "--")
        val = kv.get("val_rel_frob_percent", "--")
        try:
            params_txt = f"{int(float(params)):,}"
        except Exception:
            params_txt = "--"
        lines.append(
            f"{model} & {mapping} & {params_txt} & "
            f"{fmt(float(train), 3) if train != '--' else '--'} & "
            f"{fmt(float(val), 3) if val != '--' else '--'}"
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
