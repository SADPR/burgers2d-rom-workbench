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

HERE = Path(__file__).resolve().parent
BASE = HERE / "mlspg_prom_main"
ENR = HERE / "mlspg_prom_enrichment_ext25_lhs36"
FIG_DIR = HERE / "Figures" / "prom_only"
TAB_DIR = HERE / "tables" / "prom_only"
BASIS_PATH = HERE / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy"
UREF_PATH = HERE / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy"
NTOT = 151

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "text.usetex": False})

@dataclass(frozen=True)
class Point:
    key: str
    tex: str
    short: str
    mu1: float
    mu2: float

POINTS = (
    Point("verification", r"$\mu^{(v)}$", "validation", 4.875, 0.0225),
    Point("offgrid1", r"$\mu^{(1)}$", "off-grid 1", 4.560, 0.0190),
    Point("offgrid2", r"$\mu^{(2)}$", "off-grid 2", 5.190, 0.0260),
    Point("extrapolation20pct", r"$\mu^{(3)}$", "extrapolation", 4.000, 0.0330),
)

METHODS = (
    ("Case 1", "PROM-ANN C1", "case1"),
    ("Case 2", "PROM-ANN C2", "case2"),
    ("Case 3", "PROM-ANN C3", "case3"),
    ("POD-AE", "PROM-POD-AE", "podae"),
    ("POD-NN-ROM", "POD-NN-ROM", "podnn"),
    ("POD-DL-ROM", "POD-DL-ROM", "poddl"),
)

COLORS = {
    "Case 1": "#4C78A8",
    "Case 2": "#54A24B",
    "Case 3": "#F58518",
    "POD-AE": "#B279A2",
    "POD-NN-ROM": "#72B7B2",
    "POD-DL-ROM": "#E45756",
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
        rows.append((label, train_b, val_b, train_e, val_e, float(100.0 * (val_b - val_e) / val_b) if val_b else float("nan")))
    with out.open("w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Base train & Base val. & Enriched train & Enriched val. & Val. reduction \\\\ \n")
        f.write("\\midrule\n")
        for label, tb, vb, te, ve, red in rows:
            f.write(f"{label} & {fmt(tb)} & {fmt(vb)} & {fmt(te)} & {fmt(ve)} & {fmt(red,1)}\\% \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def write_online_state_comparison() -> Path:
    out = TAB_DIR / "prom_enrichment_online_state_comparison.tex"
    rows = []
    # Linear row from baseline only.
    lin_vals = [state_error(BASE, "linear", p) for p in POINTS]
    rows.append(("Linear PROM", "reference", *lin_vals[:3], float(np.mean(lin_vals[:3])), lin_vals[3], float(np.mean([elapsed(BASE, "linear", p) for p in POINTS]))))
    for label, _, kind in METHODS:
        for root_label, root in (("base", BASE), ("enriched", ENR)):
            vals = [state_error(root, kind, p) for p in POINTS]
            times = [elapsed(root, kind, p) for p in POINTS]
            rows.append((label, root_label, *vals[:3], float(np.mean(vals[:3])), vals[3], float(np.nanmean(times))))
    with out.open("w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(r"Model & Data & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ & Time (s) \\" + "\n")
        f.write("\\midrule\n")
        for model, data, v, m1, m2, mean, m3, t in rows:
            f.write(f"{model} & {data} & {fmt(v)} & {fmt(m1)} & {fmt(m2)} & {fmt(mean)} & {fmt(m3)} & {fmt_time(t)} \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def write_coeff_comparison() -> Path:
    out = TAB_DIR / "prom_enrichment_online_coeff_comparison.tex"
    rows = []
    for label, _, kind in METHODS:
        for root_label, root in (("base", BASE), ("enriched", ENR)):
            vals = [rel_q_error(root, kind, p) for p in POINTS]
            rows.append((label, root_label, *vals[:3], float(np.mean(vals[:3])), vals[3]))
    with out.open("w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\toprule\n")
        f.write(r"Model & Data & $\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$ \\" + "\n")
        f.write("\\midrule\n")
        for model, data, v, m1, m2, mean, m3 in rows:
            f.write(f"{model} & {data} & {fmt(v)} & {fmt(m1)} & {fmt(m2)} & {fmt(mean)} & {fmt(m3)} \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return out


def copy_sampling_figure() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    src = ENR / "Stage2" / "prom_coeff_dataset_ntot151_enriched_lhs36" / "stage2_sampling_points.png"
    dst = FIG_DIR / "prom_enrichment_sampling_points.png"
    shutil.copy2(src, dst)
    return dst


def plot_state_bar() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [m[0] for m in METHODS]
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
        ax.bar(x - width/2, bvals, width, label="base 9", color="#9ecae9", edgecolor="#376795")
        ax.bar(x + width/2, evals, width, label="enriched 9+36", color="#a1d99b", edgecolor="#2b7a2b")
        ax.axhline(lin_val, color="black", linestyle="--", linewidth=1.1, label="linear PROM" if ax is axes[0] else None)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("state relative error vs HDM (%)")
    axes[0].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "prom_enrichment_state_error_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_coeff_bar() -> Path:
    labels = [m[0] for m in METHODS]
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
        ax.bar(x - width/2, bvals, width, label="base 9", color="#fdae6b", edgecolor="#a04a1f")
        ax.bar(x + width/2, evals, width, label="enriched 9+36", color="#bcbddc", edgecolor="#5e4fa2")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("relative coefficient error vs linear PROM (%)")
    axes[0].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "prom_enrichment_coeff_error_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_master_coeff_curves() -> Path:
    # Focus on the model most affected by enrichment: direct POD-NN/master Case-2 map.
    x = np.arange(1, NTOT + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.3), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), POINTS):
        qref = online_q(BASE, "linear", p)
        for root, label, color in ((BASE, "base 9", "#4C78A8"), (ENR, "enriched 9+36", "#E45756")):
            q = online_q(root, "podnn", p)
            denom = np.maximum(np.linalg.norm(qref, axis=1), 1.0e-14)
            rel = 100.0 * np.linalg.norm(q - qref, axis=1) / denom
            ax.semilogy(x, rel, color=color, lw=1.6, alpha=0.85, label=label)
        ax.axvline(10, color="#333333", linewidth=1.0, linestyle="--", alpha=0.65)
        ax.set_title(f"{p.tex}: $\\mu=({p.mu1:.3f},{p.mu2:.4f})$")
        ax.set_xlabel("coefficient index")
        ax.set_ylabel("relative coefficient error (%)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(1e-4, 2e2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "prom_enrichment_podnn_coeff_rel_errors.png"
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
        copy_sampling_figure(),
        plot_state_bar(),
        plot_coeff_bar(),
        plot_master_coeff_curves(),
    ]
    print("[prom-enrichment-assets] tables:")
    for t in tables:
        print(f"  {t}")
    print("[prom-enrichment-assets] figures:")
    for f in figures:
        print(f"  {f}")


if __name__ == "__main__":
    main()
