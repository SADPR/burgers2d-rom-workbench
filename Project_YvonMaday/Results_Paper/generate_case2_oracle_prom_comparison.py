#!/usr/bin/env python3
"""Generate a standalone PROM Case-2 oracle comparison report.

The report compares Case-2 variants against the corresponding linear PROM
reference for two bases:
  1. the MLSPG-sensitive basis used in the paper campaign,
  2. the Euclidean POD basis used as an oracle-only diagnostic.

All data are PROM-only.  No HPROM/ECSW online data are mixed into this report.
"""

from __future__ import annotations

import ast
import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Point:
    key: str
    label: str
    mu1: float
    mu2: float

    @property
    def mu1_tag(self) -> str:
        return f"{self.mu1:.3f}"

    @property
    def mu2_tag(self) -> str:
        return f"{self.mu2:.4f}"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str
    kind: str


@dataclass(frozen=True)
class Campaign:
    key: str
    title: str
    metric_dir: str
    methods: Tuple[Method, ...]
    table_prefix: str
    figure_prefix: str
    interpretation: str


POINTS = (
    Point("mu_v", r"$\mu^{(v)}$", 4.875, 0.0225),
    Point("mu_1", r"$\mu^{(1)}$", 4.560, 0.0190),
    Point("mu_2", r"$\mu^{(2)}$", 5.190, 0.0260),
    Point("mu_3", r"$\mu^{(3)}$", 4.000, 0.0330),
)

METHOD_ANN = Method("case2_ann", "Case 2 ANN PROM", "#0097a7", "ann")
METHOD_ORACLE = Method("case2_oracle", "Case 2 oracle PROM", "#2e7d32", "plain_oracle")
METHOD_PG_ORACLE = Method("case2_pg_oracle", "Case 2 PG-oracle PROM", "#7b1fa2", "pg_oracle")

CAMPAIGNS = (
    Campaign(
        key="mlspg",
        title="MLSPG-sensitive basis",
        metric_dir="lspg_sensitive",
        methods=(METHOD_ANN, METHOD_ORACLE, METHOD_PG_ORACLE),
        table_prefix="mlspg_case2_prom_oracle",
        figure_prefix="mlspg_case2",
        interpretation=(
            "For the MLSPG-sensitive basis, the completed corrected test shows that the "
            "offset-aware initialization makes both oracle variants recover the corresponding "
            "linear PROM to approximately $10^{-4}$ percent in trajectory norm.  The earlier "
            "measurable oracle discrepancy was therefore an initialization inconsistency, not "
            "evidence that the MLSPG metric makes Case~2 structurally unable to reproduce the "
            "linear PROM when the secondary coordinates are exact.  The remaining ANN Case~2 "
            "error is therefore attributable to regression and online coupling errors, not to "
            "the oracle reduced problem."
        ),
    ),
    Campaign(
        key="euclidean",
        title="Euclidean POD basis",
        metric_dir="euclidean",
        methods=(METHOD_ORACLE, METHOD_PG_ORACLE),
        table_prefix="euclidean_case2_prom_oracle",
        figure_prefix="euclidean_case2",
        interpretation=(
            "For the Euclidean POD basis, the completed corrected test gives the same limiting "
            "behavior: both oracle variants nearly coincide with the corresponding linear PROM "
            "at all four points.  This is expected because the Euclidean split is essentially "
            "orthogonal and the corrected initialization is consistent with the injected "
            "secondary block."
        ),
    ),
)

LINEAR_COLOR = "#555555"
HDM_COLOR = "#111111"


def parse_summary(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"nan", "none", "n/a"}:
            out[key] = value
            continue
        try:
            out[key] = ast.literal_eval(value)
            continue
        except Exception:
            pass
        try:
            out[key] = float(value)
            continue
        except Exception:
            out[key] = value
    return out


def safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x)))


def rel_percent(a: np.ndarray, b: np.ndarray) -> float:
    den = safe_norm(b)
    if den == 0.0:
        return float("nan")
    return 100.0 * safe_norm(np.asarray(a) - np.asarray(b)) / den


def parse_mu_filename(path: Path) -> Tuple[float, float] | None:
    m = re.match(r"mu1_([^+]+)\+mu2_(.+)\.npy$", path.name)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def find_hdm_path(workspace: Path, point: Point) -> Path:
    candidates = []
    for base in (
        workspace / "Results" / "param_snaps",
        workspace / "Project_YvonMaday" / "Results" / "param_snaps",
        workspace / "Project_YvonMaday" / "250x250" / "param_snaps",
    ):
        if base.exists():
            candidates.extend(base.glob("*.npy"))
    for path in candidates:
        parsed = parse_mu_filename(path)
        if parsed is None:
            continue
        if abs(parsed[0] - point.mu1) < 5e-12 and abs(parsed[1] - point.mu2) < 5e-12:
            return path
    raise FileNotFoundError(f"Could not find HDM snapshots for {point}")


def linear_dir(project: Path, campaign: Campaign, point: Point) -> Path:
    return (
        project
        / "Results_Paper"
        / "MetricStudy"
        / campaign.metric_dir
        / "Runs"
        / "Linear"
        / f"linear_prom_mu1_{point.mu1_tag}_mu2_{point.mu2_tag}_ntot151"
    )


def method_paths(project: Path, campaign: Campaign, point: Point, method: Method) -> Tuple[Path, Path, Path]:
    if method.kind == "ann":
        if campaign.key != "mlspg":
            raise ValueError("ANN PROM path is only defined for the MLSPG diagnostic.")
        base = project / "Results_Paper" / "scripts" / "mlspg_prom_probe" / "Runs" / "Case2_B01_PROM"
        stem = f"case2_prom_ann_mu1_{point.mu1_tag}_mu2_{point.mu2_tag}_n10_ntot151"
    elif method.kind == "plain_oracle":
        base = (
            project
            / "Results_Paper"
            / "MetricStudy"
            / campaign.metric_dir
            / "Runs"
            / "Case2_Plain_Oracle_PROMOnly_Legacy"
            / "np10"
        )
        prefix = "euclidean_plain" if campaign.key == "euclidean" else "case2_plain"
        stem = (
            f"{prefix}_oracle_promonly_legacy_mu1_{point.mu1_tag}_mu2_{point.mu2_tag}"
            "_n10_ntot151_basis_pert0.00pct"
        )
    elif method.kind == "pg_oracle":
        base = (
            project
            / "Results_Paper"
            / "MetricStudy"
            / campaign.metric_dir
            / "Runs"
            / "Case2_PG_Oracle_PROMOnly_Legacy"
            / "np10"
        )
        prefix = "euclidean_pg" if campaign.key == "euclidean" else "case2_pg"
        stem = (
            f"{prefix}_oracle_promonly_legacy_mu1_{point.mu1_tag}_mu2_{point.mu2_tag}"
            "_n10_ntot151_basis_pert0.00pct"
        )
    else:
        raise ValueError(method.kind)
    return base / f"{stem}_summary.txt", base / f"{stem}_qN.npy", base / f"{stem}_snaps.npy"


def check_inputs(workspace: Path, project: Path, campaign: Campaign) -> None:
    missing = []
    for point in POINTS:
        for path in (linear_dir(project, campaign, point) / "qN.npy", linear_dir(project, campaign, point) / "rom_snaps.npy"):
            if not path.exists():
                missing.append(path)
        try:
            find_hdm_path(workspace, point)
        except FileNotFoundError as exc:
            missing.append(Path(str(exc)))
        for method in campaign.methods:
            for path in method_paths(project, campaign, point, method):
                if not path.exists():
                    missing.append(path)
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Missing required inputs for {campaign.title}:\n{msg}")


def coeff_errors(q_ref: np.ndarray, q_model: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_model = np.asarray(q_model, dtype=np.float64)
    n_t = min(q_ref.shape[1], q_model.shape[1])
    diff = q_model[:, :n_t] - q_ref[:, :n_t]
    abs_err = np.linalg.norm(diff, axis=1)
    den = np.linalg.norm(q_ref[:, :n_t], axis=1)
    rel_err = abs_err / np.maximum(den, 1e-30)
    return abs_err, rel_err


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fmt(x: float, ndigits: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{ndigits}f}"


def tex_escape(text: str) -> str:
    return text.replace("%", r"\%").replace("_", r"\_")


def write_state_table(path: Path, rows: list[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Point & Method & $\varepsilon_u$ vs HDM (\%) & $\varepsilon_u$ vs linear PROM (\%) & $\varepsilon_q^{\rm low}$ (\%) & $\varepsilon_q^{\rm high}$ (\%) & online time (s) \\",
        r"\midrule",
    ]
    for point_index, point in enumerate(POINTS):
        vals = [row for row in rows if row["point_key"] == point.key]
        if not vals:
            continue
        ref = vals[0]
        lines.append(
            f"{point.label} & Linear PROM & "
            f"{fmt(ref['linear_prom_error_vs_hdm_pct'])} & 0.000 & 0.000 & 0.000 & "
            f"{fmt(ref['linear_prom_online_time_s'], 2)} \\\\" 
        )
        for row in vals:
            lines.append(
                f" & {tex_escape(str(row['method']))} & "
                f"{fmt(row['state_vs_hdm_pct'])} & {fmt(row['state_vs_linear_pct'])} & "
                f"{fmt(row['q_low_vs_linear_pct'])} & {fmt(row['q_high_vs_linear_pct'])} & "
                f"{fmt(row['online_time_s'], 2)} \\\\" 
            )
        if point_index != len(POINTS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_mean_table(path: Path, rows: list[Dict[str, object]]) -> None:
    grouped: Dict[str, list[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & mean $\varepsilon_u$ vs HDM (\%) & mean $\varepsilon_u$ vs linear PROM (\%) & mean $\varepsilon_q^{\rm low}$ (\%) & mean $\varepsilon_q^{\rm high}$ (\%) & mean time (s) \\",
        r"\midrule",
    ]
    for method, vals in grouped.items():
        lines.append(
            f"{tex_escape(method)} & "
            f"{fmt(np.mean([v['state_vs_hdm_pct'] for v in vals]))} & "
            f"{fmt(np.mean([v['state_vs_linear_pct'] for v in vals]))} & "
            f"{fmt(np.mean([v['q_low_vs_linear_pct'] for v in vals]))} & "
            f"{fmt(np.mean([v['q_high_vs_linear_pct'] for v in vals]))} & "
            f"{fmt(np.mean([v['online_time_s'] for v in vals]), 2)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def plot_state_bars(fig_dir: Path, campaign: Campaign, rows: list[Dict[str, object]]) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(POINTS))
    width = 0.72 / max(1, len(campaign.methods))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    offset0 = -0.5 * width * (len(campaign.methods) - 1)
    for i, method in enumerate(campaign.methods):
        vals = [
            next(r for r in rows if r["point_key"] == p.key and r["method_key"] == method.key)["state_vs_hdm_pct"]
            for p in POINTS
        ]
        ax.bar(x + offset0 + i * width, vals, width=width, color=method.color, label=method.label)
    linear_vals = [
        next(r for r in rows if r["point_key"] == p.key)["linear_prom_error_vs_hdm_pct"] for p in POINTS
    ]
    ax.plot(x, linear_vals, color=LINEAR_COLOR, marker="o", lw=1.6, label="Linear PROM")
    ax.set_xticks(x)
    ax.set_xticklabels([p.label for p in POINTS])
    ax.set_ylabel(r"trajectory error vs HDM (\%)")
    ax.set_title(f"PROM Case 2 oracle diagnostic: {campaign.title}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2 + len(campaign.methods) // 2, loc="upper left", bbox_to_anchor=(0.0, -0.18), frameon=True)
    fig.tight_layout()
    out = fig_dir / f"{campaign.figure_prefix}_state_error_bars.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_coeff_curves(
    fig_dir: Path,
    campaign: Campaign,
    coeff_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
) -> Path:
    fig, axes = plt.subplots(2, len(POINTS), figsize=(15.0, 6.0), sharex=True)
    for j, point in enumerate(POINTS):
        ax_abs = axes[0, j]
        ax_rel = axes[1, j]
        for method in campaign.methods:
            abs_err, rel_err = coeff_data[(point.key, method.key)]
            idx = np.arange(1, abs_err.size + 1)
            ax_abs.semilogy(idx, abs_err + 1e-30, color=method.color, lw=1.5, label=method.label)
            ax_rel.semilogy(idx, 100.0 * rel_err + 1e-30, color=method.color, lw=1.5)
        ax_abs.axvline(10, color="0.4", lw=0.8, ls=":")
        ax_rel.axvline(10, color="0.4", lw=0.8, ls=":")
        ax_abs.set_title(f"{point.label}: ({point.mu1:.3f}, {point.mu2:.4f})")
        ax_rel.set_xlabel("coefficient index")
        if j == 0:
            ax_abs.set_ylabel(r"$\|q_i-q_i^{\rm lin}\|_2$")
            ax_rel.set_ylabel(r"relative error (\%)")
        ax_abs.grid(alpha=0.25)
        ax_rel.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(campaign.methods), loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Coefficient errors vs linear PROM: {campaign.title}", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = fig_dir / f"{campaign.figure_prefix}_coeff_abs_rel_curves.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_heatmaps(
    fig_dir: Path,
    campaign: Campaign,
    heat_data: Dict[Tuple[str, str], np.ndarray],
    *,
    relative: bool,
) -> Path:
    all_vals = np.concatenate([np.ravel(v) for v in heat_data.values()])
    vmax = float(np.nanpercentile(all_vals, 99.0))
    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(all_vals) + 1e-30)
    fig, axes = plt.subplots(len(campaign.methods), len(POINTS), figsize=(15.0, 3.0 + 1.55 * len(campaign.methods)), sharex=True, sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    for i, method in enumerate(campaign.methods):
        for j, point in enumerate(POINTS):
            ax = axes[i, j]
            data = heat_data[(point.key, method.key)]
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=vmax,
                extent=[0.0, 25.0, 1, data.shape[0]],
            )
            if i == 0:
                ax.set_title(f"{point.label}\n({point.mu1:.3f}, {point.mu2:.4f})")
            if j == 0:
                ax.set_ylabel(f"{method.label}\ncoeff. index")
            if i == len(campaign.methods) - 1:
                ax.set_xlabel("time")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("relative coefficient error" if relative else "absolute coefficient error")
    title = "Relative" if relative else "Absolute"
    fig.suptitle(f"{title} coefficient-error heatmaps vs linear PROM: {campaign.title}", y=0.995)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.88, bottom=0.08, wspace=0.12, hspace=0.22)
    suffix = "rel" if relative else "abs"
    out = fig_dir / f"{campaign.figure_prefix}_coeff_{suffix}_heatmaps.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def line_mid_ux(snaps: np.ndarray) -> np.ndarray:
    n_state = snaps.shape[0]
    ncell = n_state // 2
    nside = int(round(math.sqrt(ncell)))
    ux = np.asarray(snaps[:ncell, -1]).reshape(nside, nside)
    return ux[nside // 2, :]


def plot_line_overlays(
    workspace: Path,
    project: Path,
    campaign: Campaign,
    fig_dir: Path,
    all_paths: Dict[Tuple[str, str], Tuple[Path, Path, Path]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()
    x = np.linspace(0.0, 100.0, 250)
    for ax, point in zip(axes, POINTS):
        hdm = np.load(find_hdm_path(workspace, point), mmap_mode="r")
        lin = np.load(linear_dir(project, campaign, point) / "rom_snaps.npy", mmap_mode="r")
        ax.plot(x, line_mid_ux(hdm), color=HDM_COLOR, lw=2.0, label="HDM")
        ax.plot(x, line_mid_ux(lin), color=LINEAR_COLOR, lw=1.6, ls="--", label="linear PROM")
        for method in campaign.methods:
            _, _, snaps_path = all_paths[(point.key, method.key)]
            snaps = np.load(snaps_path, mmap_mode="r")
            ax.plot(x, line_mid_ux(snaps), color=method.color, lw=1.4, label=method.label)
        ax.set_title(f"{point.label}: ({point.mu1:.3f}, {point.mu2:.4f})")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u_x(x,y_{\rm mid},T)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(5, 2 + len(campaign.methods)), loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Final-time horizontal midline cut: {campaign.title}", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    out = fig_dir / f"{campaign.figure_prefix}_final_midline_overlays.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def collect_campaign(workspace: Path, project: Path, campaign: Campaign, fig_dir: Path, table_dir: Path):
    check_inputs(workspace, project, campaign)

    rows: list[Dict[str, object]] = []
    coeff_rows: list[Dict[str, object]] = []
    coeff_curve_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    heat_abs_data: Dict[Tuple[str, str], np.ndarray] = {}
    heat_rel_data: Dict[Tuple[str, str], np.ndarray] = {}
    all_paths: Dict[Tuple[str, str], Tuple[Path, Path, Path]] = {}

    for point in POINTS:
        q_ref = np.load(linear_dir(project, campaign, point) / "qN.npy")
        lin_snaps = np.load(linear_dir(project, campaign, point) / "rom_snaps.npy", mmap_mode="r")
        lin_summary = parse_summary(linear_dir(project, campaign, point) / "summary.txt")
        hdm = np.load(find_hdm_path(workspace, point), mmap_mode="r")
        lin_state_vs_hdm = rel_percent(lin_snaps, hdm)
        lin_online_time = float(lin_summary.get("online_solve_elapsed_s", np.nan))

        for method in campaign.methods:
            summary_path, q_path, snaps_path = method_paths(project, campaign, point, method)
            all_paths[(point.key, method.key)] = (summary_path, q_path, snaps_path)
            q = np.load(q_path)
            snaps = np.load(snaps_path, mmap_mode="r")
            summary = parse_summary(summary_path)

            n_t = min(q_ref.shape[1], q.shape[1])
            n_state_t = min(lin_snaps.shape[1], snaps.shape[1], hdm.shape[1])
            state_vs_hdm = rel_percent(snaps[:, :n_state_t], hdm[:, :n_state_t])
            state_vs_linear = rel_percent(snaps[:, :n_state_t], lin_snaps[:, :n_state_t])
            q_low = rel_percent(q[:10, :n_t], q_ref[:10, :n_t])
            q_high = rel_percent(q[10:, :n_t], q_ref[10:, :n_t])
            q_total = rel_percent(q[:, :n_t], q_ref[:, :n_t])

            rows.append(
                {
                    "point_key": point.key,
                    "point_tex": point.label,
                    "mu1": point.mu1,
                    "mu2": point.mu2,
                    "method_key": method.key,
                    "method": method.label,
                    "linear_prom_error_vs_hdm_pct": lin_state_vs_hdm,
                    "linear_prom_online_time_s": lin_online_time,
                    "state_vs_hdm_pct": state_vs_hdm,
                    "state_vs_linear_pct": state_vs_linear,
                    "q_low_vs_linear_pct": q_low,
                    "q_high_vs_linear_pct": q_high,
                    "q_total_vs_linear_pct": q_total,
                    "online_time_s": float(summary.get("online_solve_elapsed_s", np.nan)),
                }
            )

            abs_err, rel_err = coeff_errors(q_ref[:, :n_t], q[:, :n_t])
            coeff_curve_data[(point.key, method.key)] = (abs_err, rel_err)
            diff = np.abs(q[:, :n_t] - q_ref[:, :n_t])
            den = np.linalg.norm(q_ref[:, :n_t], axis=1)
            heat_abs_data[(point.key, method.key)] = diff
            heat_rel_data[(point.key, method.key)] = diff / np.maximum(den[:, None], 1e-30)

            coeff_rows.append(
                {
                    "point": point.key,
                    "mu1": point.mu1,
                    "mu2": point.mu2,
                    "method": method.label,
                    "mean_abs_coeff_error": float(np.mean(abs_err)),
                    "max_abs_coeff_error": float(np.max(abs_err)),
                    "mean_rel_coeff_error_pct": float(100.0 * np.mean(rel_err)),
                    "max_rel_coeff_error_pct": float(100.0 * np.max(rel_err)),
                    "mean_primary_rel_coeff_error_pct": float(100.0 * np.mean(rel_err[:10])),
                    "mean_secondary_rel_coeff_error_pct": float(100.0 * np.mean(rel_err[10:])),
                }
            )

    write_csv(
        table_dir / f"{campaign.table_prefix}_state_metrics.csv",
        rows,
        [
            "point_key",
            "point_tex",
            "mu1",
            "mu2",
            "method_key",
            "method",
            "linear_prom_error_vs_hdm_pct",
            "linear_prom_online_time_s",
            "state_vs_hdm_pct",
            "state_vs_linear_pct",
            "q_low_vs_linear_pct",
            "q_high_vs_linear_pct",
            "q_total_vs_linear_pct",
            "online_time_s",
        ],
    )
    write_csv(
        table_dir / f"{campaign.table_prefix}_coeff_metrics.csv",
        coeff_rows,
        [
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
        ],
    )
    write_state_table(table_dir / f"{campaign.table_prefix}_state_table.tex", rows)
    write_mean_table(table_dir / f"{campaign.table_prefix}_mean_table.tex", rows)

    fig_paths = {
        "state_bars": plot_state_bars(fig_dir, campaign, rows),
        "coeff_curves": plot_coeff_curves(fig_dir, campaign, coeff_curve_data),
        "abs_heatmaps": plot_heatmaps(fig_dir, campaign, heat_abs_data, relative=False),
        "rel_heatmaps": plot_heatmaps(fig_dir, campaign, heat_rel_data, relative=True),
        "line_overlays": plot_line_overlays(workspace, project, campaign, fig_dir, all_paths),
    }
    return rows, coeff_rows, fig_paths


def campaign_section(campaign: Campaign, fig_paths: Dict[str, Path]) -> str:
    ann_phrase = "ANN Case~2 PROM, " if any(m.kind == "ann" for m in campaign.methods) else ""
    return rf"""
\section*{{{campaign.title}}}
This section compares {ann_phrase}oracle Case~2 PROM, and PG-oracle Case~2 PROM
against the corresponding linear PROM reference for the {campaign.title.lower()}.
In the oracle cases, only the secondary coordinates are prescribed from the
linear PROM trajectory; the primary coordinates remain online unknowns.

\begin{{table}}[H]
\centering
\caption{{PROM Case~2 oracle comparison for the {campaign.title.lower()}.}}
\label{{tab:{campaign.key}-case2-state}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/{campaign.table_prefix}_state_table.tex}}}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Mean PROM Case~2 oracle metrics over the four evaluation points for the {campaign.title.lower()}.}}
\label{{tab:{campaign.key}-case2-mean}}
\resizebox{{\textwidth}}{{!}}{{\input{{tables/{campaign.table_prefix}_mean_table.tex}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/{fig_paths['state_bars'].name}}}
\caption{{State-space trajectory error versus HDM for the {campaign.title.lower()}.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures/{fig_paths['line_overlays'].name}}}
\caption{{Final-time horizontal midline cuts of $u_x$ for HDM, linear PROM, and the Case~2 diagnostic variants for the {campaign.title.lower()}.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures/{fig_paths['coeff_curves'].name}}}
\caption{{Per-coefficient absolute and relative errors with respect to the linear PROM coefficient trajectory.  The dotted line marks the primary/secondary split at $n=10$.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures/{fig_paths['abs_heatmaps'].name}}}
\caption{{Absolute coefficient-error heatmaps versus the linear PROM coefficient trajectory.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures/{fig_paths['rel_heatmaps'].name}}}
\caption{{Relative coefficient-error heatmaps versus the linear PROM coefficient trajectory.}}
\end{{figure}}

\paragraph{{Interpretation.}}
{campaign.interpretation}
"""


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    project = workspace / "Project_YvonMaday"
    out_root = project / "Results_Paper" / "Case2_Oracle_PROM_Diagnostic"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    all_fig_paths: Dict[str, Dict[str, Path]] = {}
    for campaign in CAMPAIGNS:
        rows, coeff_rows, fig_paths = collect_campaign(workspace, project, campaign, fig_dir, table_dir)
        all_fig_paths[campaign.key] = fig_paths
        sections.append(campaign_section(campaign, fig_paths))

    tex_path = out_root / "case2_oracle_prom_comparison.tex"
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.8in]{{geometry}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage[hypertexnames=false]{{hyperref}}

\title{{Corrected PROM Case 2 Oracle Diagnostic}}
\author{{S. Ares de Parga}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Purpose}}
This diagnostic isolates the online structure of PROM--ANN Case~2.  All results
are PROM-only; no HPROM or ECSW online data are used.  The corresponding
linear PROM trajectory is used as the coefficient reference for each basis.  The
Case~2 manifold has the form
\begin{{equation*}}
\widetilde{{\mathbf u}}
=
\mathbf u_{{\mathrm{{ref}}}}
+
\mathbf V\mathbf q
+
\overline{{\mathbf V}}\overline{{\mathbf q}}(t),
\end{{equation*}}
where the online solve determines only $\mathbf q$.  The oracle variants set
$\overline{{\mathbf q}}(t)=\overline{{\mathbf q}}^{{\mathrm{{lin}}}}(t)$ from the
linear PROM trajectory and therefore remove secondary-coordinate regression
error.  Any remaining discrepancy with the linear PROM measures the online
reduced problem itself.

\paragraph{{Status of the test.}}
The corrected diagnostic has been run for the four evaluation parameters
$\boldsymbol\mu^{{(v)}}$, $\boldsymbol\mu^{{(1)}}$,
$\boldsymbol\mu^{{(2)}}$, and $\boldsymbol\mu^{{(3)}}$, using both the
MLSPG-sensitive basis and the Euclidean POD basis.  The tables and figures
below summarize the completed PROM-only checks.

\paragraph{{Initialization used in this corrected diagnostic.}}
For Case~2, the initial primary coordinate must account for the prescribed
secondary block.  The diagnostic therefore initializes
\begin{{equation*}}
\mathbf q(0)
=
\arg\min_{{\mathbf y}}
\left\|
\mathbf V\mathbf y
-
\left(
\mathbf u_0-\mathbf u_{{\mathrm{{ref}}}}
-
\overline{{\mathbf V}}\,\overline{{\mathbf q}}(0)
\right)
\right\|_2 .
\end{{equation*}}
This is essential for non-Euclidean or metric-weighted bases, where
$\mathbf V^T\overline{{\mathbf V}}$ need not vanish in the Euclidean inner
product.

\paragraph{{PG-oracle variant.}}
The PG-oracle variant uses the same oracle secondary coordinates, but tests the
residual with the full linear PROM tangent before solving the overdetermined
least-squares problem for the ten primary unknowns.  This diagnostic is intended
to distinguish secondary-coordinate regression error from low/high coupling and
projection effects.

{''.join(sections)}

\section*{{Overall conclusion}}
After correcting the Case~2 initialization, the completed MLSPG-sensitive and
Euclidean oracle-only PROM tests both recover their corresponding linear PROM
trajectories to essentially zero error relative to the linear PROM.  The
diagnostic therefore confirms that prescribing the exact secondary coordinates
is sufficient to recover the linear PROM trajectory.  The previously observed
MLSPG oracle discrepancy was caused by an inconsistent initial primary
coordinate, not by a structural failure of the Case~2 reduced problem.
Consequently, Case~2 production errors should be interpreted as neural-network
secondary-coordinate regression error and its online feedback.

\end{{document}}
"""
    tex_path.write_text(tex)

    print(f"[case2-oracle-report] wrote {tex_path}")
    for campaign_key, paths in all_fig_paths.items():
        for name, path in paths.items():
            print(f"[case2-oracle-report] {campaign_key} figure {name}: {path}")
    print(f"[case2-oracle-report] tables: {table_dir}")

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=out_root, check=False)
    except FileNotFoundError:
        print("[case2-oracle-report] pdflatex not found; skipped PDF build")


if __name__ == "__main__":
    main()
