#!/usr/bin/env python3
"""Case 2 n=10 coefficient diagnostic: reconstruction vs PROM vs HPROM.

The reference is the saved linear-HPROM coefficient trajectory.  This script
only reads existing qN arrays and writes diagnostic figures; it does not run
any online solver.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
PAPER = PROJECT / "Results_Paper"
OUT_DIR = PAPER / "Figures" / "mlspg_hprom_current" / "case2_n10_recon_prom_hprom"
TABLE_DIR = PAPER / "tables"
NTOT = 151
TIME_END = 25.0


@dataclass(frozen=True)
class Point:
    key: str
    label: str
    mu1: float
    mu2: float


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str
    linestyle: str = "-"


POINTS = (
    Point("verification", r"$\mu^{(v)}$", 4.875, 0.0225),
    Point("offgrid1", r"$\mu^{(1)}$", 4.560, 0.0190),
    Point("offgrid2", r"$\mu^{(2)}$", 5.190, 0.0260),
    Point("extrapolation20pct", r"$\mu^{(3)}$", 4.000, 0.0330),
)

METHODS = (
    Method("recon", "reconstruction only", "#6f42c1", "-"),
    Method("prom", "full PROM", "#008c95", "--"),
    Method("hprom", "2% HPROM", "#d95f02", "-."),
)


def mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def linear_q_path(point: Point) -> Path:
    tag = mu_tag(point.mu1, point.mu2)
    if point.key == "extrapolation20pct":
        return PAPER / "mlspg_hprom_main" / "Runs" / "Extrapolation20pct" / "Linear" / f"linear_hprom_{tag}_ntot151" / "qN.npy"
    return PAPER / "mlspg_hprom_main" / "Runs" / "Linear" / f"linear_hprom_{tag}_ntot151" / "qN.npy"


def method_q_path(method: Method, point: Point) -> Path:
    tag = mu_tag(point.mu1, point.mu2)
    if method.key == "recon":
        return (
            PAPER
            / "tmp_reconstruction_only_mlspg_main"
            / "Runs"
            / "Case2_Best_np10"
            / f"case2_np10_reconstruction_only_{tag}_qN.npy"
        )
    if method.key == "prom":
        return (
            PAPER
            / "mlspg_prom_main"
            / "Runs"
            / "PROM"
            / "Case2_Best"
            / "np10"
            / f"case2_prom_ann_{tag}_n10_ntot151_qN.npy"
        )
    if method.key == "hprom":
        base = PAPER / "mlspg_hprom_main" / "Runs"
        if point.key == "extrapolation20pct":
            base = base / "Extrapolation20pct"
        return (
            base
            / "ECSW2pct"
            / "Case2_Best"
            / "np10"
            / f"case2_hprom_ann_{tag}_n10_ntot151_qN.npy"
        )
    raise KeyError(method.key)


def load_q(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    q = np.load(path, allow_pickle=False)
    if q.shape != (NTOT, 501):
        raise ValueError(f"Unexpected q shape {q.shape}: {path}")
    return np.asarray(q, dtype=np.float64)


def coeff_diagnostics(q_ref: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    diff = q - q_ref
    ref_norm = np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
    abs_curve = np.linalg.norm(diff, axis=1)
    rel_curve_pct = 100.0 * abs_curve / ref_norm
    abs_heat = np.abs(diff)
    rel_heat_pct = 100.0 * abs_heat / ref_norm[:, None]
    return {
        "abs_curve": abs_curve,
        "rel_curve_pct": rel_curve_pct,
        "abs_heat": abs_heat,
        "rel_heat_pct": rel_heat_pct,
    }


def display_data(method: Method, data: np.ndarray) -> np.ndarray:
    """Hide trivial injected primary coordinates for reconstruction-only plots."""
    arr = np.array(data, dtype=np.float64, copy=True)
    if method.key == "recon":
        if arr.ndim == 1:
            arr[:10] = np.nan
        elif arr.ndim == 2:
            arr[:10, :] = np.nan
    return arr


def point_title(point: Point) -> str:
    return rf"{point.label}: $\mu=({point.mu1:.3f},{point.mu2:.4f})$"


def build_errors() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    errors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for point in POINTS:
        q_ref = load_q(linear_q_path(point))
        for method in METHODS:
            q = load_q(method_q_path(method, point))
            errors[(point.key, method.key)] = coeff_diagnostics(q_ref, q)
    return errors


def plot_curves(errors: dict[tuple[str, str], dict[str, np.ndarray]]) -> Path:
    x = np.arange(1, NTOT + 1)
    fig, axes = plt.subplots(2, len(POINTS), figsize=(18.0, 7.4), sharex=True)
    for j, point in enumerate(POINTS):
        ax_abs = axes[0, j]
        ax_rel = axes[1, j]
        for method in METHODS:
            data = errors[(point.key, method.key)]
            label = method.label if j == 0 else None
            abs_curve = display_data(method, data["abs_curve"])
            rel_curve = display_data(method, data["rel_curve_pct"])
            ax_abs.semilogy(
                x,
                abs_curve + 1.0e-14,
                color=method.color,
                linestyle=method.linestyle,
                linewidth=2.0,
                label=label,
            )
            ax_rel.semilogy(
                x,
                rel_curve + 1.0e-14,
                color=method.color,
                linestyle=method.linestyle,
                linewidth=2.0,
            )
        for ax in (ax_abs, ax_rel):
            ax.axvline(10.5, color="0.25", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.grid(True, which="major", alpha=0.3)
            ax.set_xlim(1, NTOT)
        ax_abs.set_title(point_title(point), fontsize=12)
        ax_rel.set_xlabel(r"coefficient index $i$")
    axes[0, 0].set_ylabel(r"$\|q_i-q_i^{\mathrm{lin}}\|_2$")
    axes[1, 0].set_ylabel(r"$100\,\|q_i-q_i^{\mathrm{lin}}\|_2/\|q_i^{\mathrm{lin}}\|_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(r"PROM-ANN Case 2 ($n=10$): coefficient errors vs linear HPROM reference", y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.5, h_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "case2_n10_recon_prom_hprom_coeff_abs_rel_curves.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(errors: dict[tuple[str, str], dict[str, np.ndarray]], *, relative: bool) -> Path:
    field = "rel_heat_pct" if relative else "abs_heat"
    all_values = np.concatenate([display_data(m, errors[(p.key, m.key)][field]).ravel() for p in POINTS for m in METHODS])
    all_values = all_values[np.isfinite(all_values)]
    vmax = float(np.nanpercentile(all_values, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f5f5f5")

    fig, axes = plt.subplots(
        len(METHODS),
        len(POINTS),
        figsize=(18.0, 6.8),
        sharex=True,
        sharey=True,
    )
    im = None
    for i, method in enumerate(METHODS):
        for j, point in enumerate(POINTS):
            ax = axes[i, j]
            data = display_data(method, errors[(point.key, method.key)][field])
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[0.0, TIME_END, 1, NTOT],
                vmin=0.0,
                vmax=vmax,
                cmap=cmap,
            )
            ax.axhline(10.5, color="white", linestyle="--", linewidth=0.9, alpha=0.85)
            if i == 0:
                ax.set_title(point_title(point), fontsize=11)
            if j == 0:
                ax.set_ylabel(method.label + "\ncoefficient index")
            if i == len(METHODS) - 1:
                ax.set_xlabel(r"time $t$")

    fig.subplots_adjust(left=0.12, right=0.89, bottom=0.085, top=0.875, wspace=0.12, hspace=0.23)
    cax = fig.add_axes([0.91, 0.15, 0.022, 0.67])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("relative coefficient error (%)" if relative else r"$|q_i-q_i^{\mathrm{lin}}|$")
    title = "relative" if relative else "absolute"
    fig.suptitle(rf"PROM-ANN Case 2 ($n=10$): {title} coefficient-error heatmaps", y=0.975)
    suffix = "rel" if relative else "abs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case2_n10_recon_prom_hprom_coeff_{suffix}_heatmaps.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def write_metrics(errors: dict[tuple[str, str], dict[str, np.ndarray]]) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "case2_n10_recon_prom_hprom_coeff_metrics.csv"
    fields = [
        "point",
        "mu1",
        "mu2",
        "method",
        "mean_abs_coeff_error",
        "max_abs_coeff_error",
        "mean_rel_coeff_error_pct",
        "max_rel_coeff_error_pct",
        "mean_primary_rel_coeff_error_pct",
        "mean_secondary_rel_coeff_error_pct",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in POINTS:
            for method in METHODS:
                data = errors[(point.key, method.key)]
                abs_curve = display_data(method, data["abs_curve"])
                rel = display_data(method, data["rel_curve_pct"])
                finite_abs = abs_curve[np.isfinite(abs_curve)]
                finite_rel = rel[np.isfinite(rel)]
                writer.writerow(
                    {
                        "point": point.key,
                        "mu1": f"{point.mu1:.3f}",
                        "mu2": f"{point.mu2:.4f}",
                        "method": method.key,
                        "mean_abs_coeff_error": f"{float(np.mean(finite_abs)):.10e}",
                        "max_abs_coeff_error": f"{float(np.max(finite_abs)):.10e}",
                        "mean_rel_coeff_error_pct": f"{float(np.mean(finite_rel)):.10e}",
                        "max_rel_coeff_error_pct": f"{float(np.max(finite_rel)):.10e}",
                        "mean_primary_rel_coeff_error_pct": "masked" if method.key == "recon" else f"{float(np.mean(rel[:10])):.10e}",
                        "mean_secondary_rel_coeff_error_pct": f"{float(np.mean(rel[10:])):.10e}",
                    }
                )
    return out


def main() -> None:
    errors = build_errors()
    outputs = [
        plot_curves(errors),
        plot_heatmap(errors, relative=False),
        plot_heatmap(errors, relative=True),
        write_metrics(errors),
    ]
    for out in outputs:
        print(f"[written] {out}")


if __name__ == "__main__":
    main()
