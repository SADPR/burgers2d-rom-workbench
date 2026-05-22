#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate figures and table-ready CSV metrics for Results_Paper (Burgers, HPROM backend).

Inputs:
- Results/
- Results_Enrichment/
- ../Results/param_snaps (HDM snapshots for the three evaluation points)

Outputs:
- Figures/
  - baseline_hprom_hdm_vs_all_models.png
  - enriched_hprom_hdm_vs_all_models.png
  - coeff_errors/* (curves + heatmaps)
- tables/
  - burgers_metrics_baseline.csv
  - burgers_metrics_enriched.csv
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "Figures"
COEFF_DIR = FIG_DIR / "coeff_errors"
TABLE_DIR = THIS_DIR / "tables"

RESULTS_BASE = THIS_DIR / "Results"
RESULTS_ENR = THIS_DIR / "Results_Enrichment"

# HDM snapshots are stored in Project_YvonMaday/Results/param_snaps
HDM_DIR = (THIS_DIR.parent / "Results" / "param_snaps").resolve()

POINTS = [
    ("v", 4.875, 0.0225),
    ("1", 4.560, 0.0190),
    ("2", 5.190, 0.0260),
]

METHOD_ORDER = [
    "Linear PROM",
    "PROM-ANN Case 1",
    "PROM-ANN Case 2",
    "PROM-ANN Case 3",
    "PROM-POD-AE",
    "POD-NN-ROM",
    "POD-DL-ROM",
]
MODEL_ORDER_NO_LINEAR = [
    "PROM-ANN Case 1",
    "PROM-ANN Case 2",
    "PROM-ANN Case 3",
    "PROM-POD-AE",
    "POD-NN-ROM",
    "POD-DL-ROM",
]
COLORS = {
    "HDM": "black",
    "Linear PROM": "dimgray",
    "PROM-ANN Case 1": "tab:red",
    "PROM-ANN Case 2": "tab:blue",
    "PROM-ANN Case 3": "tab:green",
    "POD-NN-ROM": "tab:orange",
    "PROM-POD-AE": "tab:purple",
    "POD-DL-ROM": "tab:brown",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.8,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    COEFF_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def parse_summary(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def as_float(d: Dict[str, str], key: str, default: float = np.nan) -> float:
    if key not in d:
        return default
    try:
        return float(d[key])
    except Exception:
        return default


def mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def find_hdm_snap(mu1: float, mu2: float) -> Path:
    if not HDM_DIR.exists():
        raise FileNotFoundError(f"HDM folder not found: {HDM_DIR}")
    patterns = [
        f"mu1_{mu1:g}+mu2_{mu2:g}.npy",
        f"mu1_{mu1:.2f}+mu2_{mu2:.3f}.npy",
        f"mu1_{mu1:.3f}+mu2_{mu2:.4f}.npy",
    ]
    for p in patterns:
        c = HDM_DIR / p
        if c.exists():
            return c
    # fallback by prefix scan
    pref = f"mu1_{mu1:.2f}"
    for c in sorted(HDM_DIR.glob(f"{pref}*mu2*.npy")):
        return c
    raise FileNotFoundError(f"HDM snapshot not found for mu=({mu1},{mu2}) in {HDM_DIR}")


def load_baseline_mu_points() -> np.ndarray:
    per_mu = RESULTS_BASE / "Stage2" / "prom_coeff_dataset_ntot151" / "per_mu"
    if not per_mu.exists():
        raise FileNotFoundError(f"Missing baseline per-mu dataset: {per_mu}")
    mus = []
    for d in sorted(per_mu.iterdir()):
        if not d.is_dir():
            continue
        f = d / "mu.npy"
        if not f.exists():
            continue
        mu = np.asarray(np.load(f, allow_pickle=False), dtype=np.float64).reshape(-1)
        if mu.size == 2:
            mus.append([float(mu[0]), float(mu[1])])
    if not mus:
        raise RuntimeError(f"No baseline mu points found in {per_mu}")
    return np.asarray(mus, dtype=np.float64)


def load_lhs_mu_points() -> np.ndarray:
    lhs = (
        RESULTS_ENR
        / "Stage2"
        / "prom_coeff_dataset_ntot151_enriched_lhs20"
        / "lhs_mu.npy"
    )
    if not lhs.exists():
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(np.load(lhs, allow_pickle=False), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Unexpected lhs_mu shape in {lhs}: {arr.shape}")
    return arr


def plot_parameter_domain_sampling(out_path: Path) -> None:
    base = load_baseline_mu_points()
    lhs = load_lhs_mu_points()
    eval_pts = np.asarray([[4.875, 0.0225], [4.56, 0.0190], [5.19, 0.0260]], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.scatter(
        base[:, 0],
        base[:, 1],
        s=95,
        c="black",
        marker="o",
        alpha=0.90,
        label=r"Baseline training points ($3\times3$)",
        zorder=3,
    )
    if lhs.shape[0] > 0:
        ax.scatter(
            lhs[:, 0],
            lhs[:, 1],
            s=82,
            c="tab:blue",
            marker="x",
            alpha=0.90,
            label="Enrichment LHS points",
            zorder=4,
        )

    eval_labels = [r"Verification $\mu^{(v)}$", r"Test $\mu^{(1)}$", r"Test $\mu^{(2)}$"]
    eval_colors = ["tab:red", "tab:orange", "tab:green"]
    for (mu1, mu2), lbl, c in zip(eval_pts, eval_labels, eval_colors):
        ax.scatter(mu1, mu2, s=170, c=c, marker="*", edgecolors="black", linewidths=0.8, label=lbl, zorder=6)

    mu1_lo, mu1_hi = 4.25, 5.50
    mu2_lo, mu2_hi = 0.015, 0.03
    pad_x = 0.06 * (mu1_hi - mu1_lo)
    pad_y = 0.08 * (mu2_hi - mu2_lo)
    ax.set_xlim(mu1_lo - pad_x, mu1_hi + pad_x)
    ax.set_ylim(mu2_lo - pad_y, mu2_hi + pad_y)
    ax.plot(
        [mu1_lo, mu1_hi, mu1_hi, mu1_lo, mu1_lo],
        [mu2_lo, mu2_lo, mu2_hi, mu2_hi, mu2_lo],
        color="0.25",
        linewidth=1.4,
        linestyle="-",
        alpha=0.85,
        zorder=1,
    )

    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Parameter domain, training points, and evaluation points")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_verification_3d_cutplanes(out_path: Path, t_idx: int) -> None:
    # Verification point and selected time index.
    mu1, mu2 = 4.875, 0.0225
    t_val = 0.05 * t_idx
    snaps = np.load(find_hdm_snap(mu1, mu2), allow_pickle=False)

    nx = ny = 250
    x = np.linspace(0.2, 99.8, nx)
    y = np.linspace(0.2, 99.8, ny)
    X, Y = np.meshgrid(x, y)
    ux = snaps[: nx * ny, t_idx].reshape(ny, nx)
    mid_x = nx // 2
    mid_y = ny // 2

    # Wider, shorter bottom panels (requested layout).
    fig = plt.figure(figsize=(13.4, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 0.85], hspace=0.28, wspace=0.20)

    ax3 = fig.add_subplot(gs[0, :], projection="3d")
    ax3.plot_surface(X, Y, ux, cmap="viridis", linewidth=0, antialiased=True, alpha=0.98)

    # translucent cut planes
    x_plane = np.full((ny, 2), x[mid_x])
    y_plane = np.column_stack([y, y])
    zmin, zmax = float(np.min(ux)), float(np.max(ux))
    z_plane = np.column_stack([np.full(ny, zmin), np.full(ny, zmax)])
    ax3.plot_surface(x_plane, y_plane, z_plane, color="gray", alpha=0.25, linewidth=0)

    y_plane2 = np.full((nx, 2), y[mid_y])
    x_plane2 = np.column_stack([x, x])
    z_plane2 = np.column_stack([np.full(nx, zmin), np.full(nx, zmax)])
    ax3.plot_surface(x_plane2, y_plane2, z_plane2, color="gray", alpha=0.25, linewidth=0)

    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$y$")
    ax3.set_zlabel(r"$u_x(x,y)$")
    ax3.set_title(rf"$\mu_1={mu1:.3f},\ \mu_2={mu2:.4f},\ t={t_val:.2f}\,\mathrm{{s}}$")
    ax3.view_init(elev=24, azim=-46)

    axx = fig.add_subplot(gs[1, 0])
    axx.plot(x, ux[mid_y, :], color="tab:red", label=rf"$u_x(x,\ y={y[mid_y]:.2f})$")
    axx.set_xlabel(r"$x$")
    axx.set_ylabel(r"$u_x$")
    # Match vertical scale to the 3D snapshot range for direct visual consistency.
    zpad = 0.04 * max(zmax - zmin, 1.0e-12)
    axx.set_ylim(zmin - zpad, zmax + zpad)
    axx.grid(True)
    axx.legend(loc="best")

    axy = fig.add_subplot(gs[1, 1])
    axy.plot(y, ux[:, mid_x], color="tab:blue", label=rf"$u_x(x={x[mid_x]:.2f},\ y)$")
    axy.set_xlabel(r"$y$")
    axy.set_ylabel(r"$u_x$")
    axy.set_ylim(zmin - zpad, zmax + zpad)
    axy.grid(True)
    axy.legend(loc="best")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.09, wspace=0.20, hspace=0.30)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def load_stage_snaps(stage: str, mu1: float, mu2: float) -> Dict[str, np.ndarray]:
    tag = mu_tag(mu1, mu2)
    # POD-DL-ROM folders can coexist for several latent sizes. We resolve against the current checkpoint latent size.
    def _select_poddl_dir(root_dir: Path, enriched: bool, tag_str: str) -> Path:
        if enriched:
            model_ckpt = (
                RESULTS_ENR
                / "Stage3"
                / "prom_coeff_dataset_ntot151_enriched_lhs20"
                / "models"
                / "pod_dl_data_driven_model_enriched_hprom.pt"
            )
            pref = f"pod_dl_data_driven_enriched_{tag_str}_ntot151_nz"
        else:
            model_ckpt = RESULTS_BASE / "Stage3" / "models" / "pod_dl_data_driven_model_hprom.pt"
            pref = f"pod_dl_data_driven_{tag_str}_ntot151_nz"

        target_nz = None
        if model_ckpt.exists():
            try:
                import torch

                ckpt = torch.load(model_ckpt, map_location="cpu")
                target_nz = int(ckpt.get("latent_dim", -1))
            except Exception:
                target_nz = None

        candidates = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith(pref)]
        if not candidates:
            raise FileNotFoundError(f"No POD-DL-ROM run folder found for pattern '{pref}*' in {root_dir}")

        if target_nz is not None:
            exact = [d for d in candidates if d.name.endswith(f"_nz{target_nz}")]
            if exact:
                return sorted(exact)[-1]
        return sorted(candidates)[-1]

    if stage == "baseline":
        root = RESULTS_BASE / "Runs"
        poddl_dir = _select_poddl_dir(root / "PODDL", enriched=False, tag_str=tag)
        snaps = {
            "HDM": np.load(find_hdm_snap(mu1, mu2)),
            "Linear PROM": np.load(root / "Linear" / f"linear_hprom_{tag}_ntot151" / "rom_snaps.npy"),
            "PROM-ANN Case 1": np.load(root / "Case1" / f"case1_hprom_ann_{tag}_n10_ntot151_snaps.npy"),
            "PROM-ANN Case 2": np.load(root / "Case2" / f"case2_hprom_ann_{tag}_n10_ntot151_snaps.npy"),
            "PROM-ANN Case 3": np.load(root / "Case3" / f"case3_hprom_ann_{tag}_n10_ntot151_snaps.npy"),
            "POD-NN-ROM": np.load(
                root
                / "DataDriven"
                / f"rom_data_driven_{tag}_ntot151"
                / "rom_snaps.npy"
            ),
            "PROM-POD-AE": np.load(
                root / "PODAE" / f"podae_hprom_{tag}_ntot151_nz5_snaps.npy"
            ),
            "POD-DL-ROM": np.load(poddl_dir / "rom_snaps.npy"),
        }
    elif stage == "enriched":
        root = RESULTS_ENR / "Runs"
        poddl_dir = _select_poddl_dir(root / "PODDL", enriched=True, tag_str=tag)
        snaps = {
            "HDM": np.load(find_hdm_snap(mu1, mu2)),
            # linear reference stays the same
            "Linear PROM": np.load(
                RESULTS_BASE / "Runs" / "Linear" / f"linear_hprom_{tag}_ntot151" / "rom_snaps.npy"
            ),
            "PROM-ANN Case 1": np.load(
                root / "Case1" / f"case1_hprom_ann_enriched_{tag}_n10_ntot151_snaps.npy"
            ),
            "PROM-ANN Case 2": np.load(
                root / "Case2" / f"case2_hprom_ann_enriched_{tag}_n10_ntot151_snaps.npy"
            ),
            "PROM-ANN Case 3": np.load(
                root / "Case3" / f"case3_hprom_ann_enriched_{tag}_n10_ntot151_snaps.npy"
            ),
            "POD-NN-ROM": np.load(
                root
                / "DataDriven"
                / f"rom_data_driven_enriched_{tag}_ntot151"
                / "rom_snaps.npy"
            ),
            "PROM-POD-AE": np.load(
                root / "PODAE" / f"podae_enriched_hprom_{tag}_ntot151_nz5_snaps.npy"
            ),
            "POD-DL-ROM": np.load(poddl_dir / "rom_snaps.npy"),
        }
    else:
        raise ValueError(stage)

    ref_shape = snaps["HDM"].shape
    for name, arr in snaps.items():
        if arr.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for {stage}/{name} at mu=({mu1},{mu2}): {arr.shape} vs {ref_shape}"
            )
    return snaps


def project_to_qn(snaps: np.ndarray, basis: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    # snaps: (N, nt)
    return basis.T @ (snaps - u_ref[:, None])


def plot_hdm_vs_models(stage: str, out_path: Path) -> None:
    # 250x250 mesh for Burgers campaign
    nx = ny = 250
    x = np.linspace(0.2, 99.8, nx)
    y = np.linspace(0.2, 99.8, ny)
    mid_x = nx // 2
    mid_y = ny // 2

    steps = [0, 125, 250, 375, 500]
    nrows = len(POINTS)
    fig, axs = plt.subplots(nrows, 2, figsize=(14, 4.0 * nrows))
    if nrows == 1:
        axs = np.array([axs])

    for row, (lab, mu1, mu2) in enumerate(POINTS):
        snaps = load_stage_snaps(stage, mu1, mu2)
        draw_order = [
            "HDM",
            "Linear PROM",
            "PROM-ANN Case 1",
            "PROM-ANN Case 2",
            "PROM-ANN Case 3",
            "PROM-POD-AE",
            "POD-NN-ROM",
            "POD-DL-ROM",
        ]
        for m in draw_order:
            arr = snaps[m]
            for s in steps:
                u = arr[: nx * ny, s].reshape(ny, nx)
                is_final = s == steps[-1]
                axs[row, 0].plot(
                    x,
                    u[mid_y, :],
                    color=COLORS[m],
                    alpha=0.85,
                    linestyle="-" if is_final else "--",
                    linewidth=2.1 if is_final else 0.9,
                    label=m if is_final else None,
                )
                axs[row, 1].plot(
                    y,
                    u[:, mid_x],
                    color=COLORS[m],
                    alpha=0.85,
                    linestyle="-" if is_final else "--",
                    linewidth=2.1 if is_final else 0.9,
                    label=m if is_final else None,
                )

        axs[row, 0].set_title(rf"$\mu=({mu1:.3f},{mu2:.4f})$: $u_x(x,y_{{mid}})$")
        axs[row, 1].set_title(rf"$\mu=({mu1:.3f},{mu2:.4f})$: $u_x(x_{{mid}},y)$")
        axs[row, 0].set_xlabel(r"$x$")
        axs[row, 1].set_xlabel(r"$y$")
        axs[row, 0].set_ylabel(r"$u_x$")
        axs[row, 1].set_ylabel(r"$u_x$")
        axs[row, 0].grid(True)
        axs[row, 1].grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995), frameon=True)
    title = "Baseline models (HPROM online)" if stage == "baseline" else "Enriched models (HPROM online)"
    fig.suptitle(title, y=1.02, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_coeff_errors_and_heatmaps() -> None:
    basis = np.asarray(np.load(RESULTS_BASE / "Stage1" / "basis.npy"), dtype=np.float64)[:, :151]
    u_ref = np.asarray(np.load(RESULTS_BASE / "Stage1" / "u_ref.npy"), dtype=np.float64).reshape(-1)
    n_p = 10
    n_tot = 151

    for stage in ("baseline", "enriched"):
        for lab, mu1, mu2 in POINTS:
            tag = mu_tag(mu1, mu2)
            snaps = load_stage_snaps(stage, mu1, mu2)
            q_ref = np.asarray(
                np.load(
                    RESULTS_BASE / "Runs" / "Linear" / f"linear_hprom_{tag}_ntot151" / "qN.npy",
                    allow_pickle=False,
                ),
                dtype=np.float64,
            )
            t_ref = np.asarray(
                np.load(
                    RESULTS_BASE / "Runs" / "Linear" / f"linear_hprom_{tag}_ntot151" / "t.npy",
                    allow_pickle=False,
                ),
                dtype=np.float64,
            ).reshape(-1)

            model_q: Dict[str, np.ndarray] = {}
            for m in MODEL_ORDER_NO_LINEAR:
                if m == "POD-NN-ROM":
                    if stage == "baseline":
                        q = np.load(
                            RESULTS_BASE
                            / "Runs"
                            / "DataDriven"
                            / f"rom_data_driven_{tag}_ntot151"
                            / "qN.npy",
                            allow_pickle=False,
                        )
                    else:
                        q = np.load(
                            RESULTS_ENR
                            / "Runs"
                            / "DataDriven"
                            / f"rom_data_driven_enriched_{tag}_ntot151"
                            / "qN.npy",
                            allow_pickle=False,
                        )
                    model_q[m] = np.asarray(q, dtype=np.float64)
                elif m == "POD-DL-ROM":
                    tag = mu_tag(mu1, mu2)
                    if stage == "baseline":
                        poddl_root = RESULTS_BASE / "Runs" / "PODDL"
                        pref = f"pod_dl_data_driven_{tag}_ntot151_nz"
                    else:
                        poddl_root = RESULTS_ENR / "Runs" / "PODDL"
                        pref = f"pod_dl_data_driven_enriched_{tag}_ntot151_nz"
                    poddl_dirs = sorted([d for d in poddl_root.iterdir() if d.is_dir() and d.name.startswith(pref)])
                    if len(poddl_dirs) == 0:
                        raise FileNotFoundError(f"Missing POD-DL-ROM qN folder for '{pref}*' in {poddl_root}")
                    # Prefer the folder aligned with current checkpoint latent size when available.
                    poddl_dir = poddl_dirs[-1]
                    try:
                        import torch

                        if stage == "baseline":
                            ckpt = torch.load(RESULTS_BASE / "Stage3" / "models" / "pod_dl_data_driven_model_hprom.pt", map_location="cpu")
                        else:
                            ckpt = torch.load(
                                RESULTS_ENR
                                / "Stage3"
                                / "prom_coeff_dataset_ntot151_enriched_lhs20"
                                / "models"
                                / "pod_dl_data_driven_model_enriched_hprom.pt",
                                map_location="cpu",
                            )
                        target_nz = int(ckpt.get("latent_dim", -1))
                        exact = [d for d in poddl_dirs if d.name.endswith(f"_nz{target_nz}")]
                        if exact:
                            poddl_dir = exact[-1]
                    except Exception:
                        pass
                    q = np.load(poddl_dir / "qN.npy", allow_pickle=False)
                    model_q[m] = np.asarray(q, dtype=np.float64)
                else:
                    model_q[m] = project_to_qn(snaps[m], basis, u_ref)

            abs_curves: Dict[str, np.ndarray] = {}
            rel_curves: Dict[str, np.ndarray] = {}
            abs_heatmaps: Dict[str, np.ndarray] = {}
            rel_heatmaps: Dict[str, np.ndarray] = {}
            ref_mode_norm = np.linalg.norm(q_ref, axis=1, keepdims=True)
            ref_mode_norm = np.maximum(ref_mode_norm, 1.0e-14)

            for m, q in model_q.items():
                diff = q_ref - q
                abs_curves[m] = np.linalg.norm(diff, axis=1)
                rel_curves[m] = abs_curves[m] / np.maximum(np.linalg.norm(q_ref, axis=1), 1.0e-14)
                abs_heatmaps[m] = np.abs(diff)
                rel_heatmaps[m] = np.abs(diff) / ref_mode_norm

            # Curves
            x = np.arange(1, n_tot + 1)
            fig, axs = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
            for m in MODEL_ORDER_NO_LINEAR:
                axs[0].semilogy(x, abs_curves[m], label=m, color=COLORS[m])
                axs[1].semilogy(x, rel_curves[m], label=m, color=COLORS[m])
            for ax in axs:
                ax.axvline(n_p + 0.5, color="0.35", linestyle="--", linewidth=1.0)
                ax.grid(True)
            axs[0].set_ylabel(r"$\|e_i\|_2$")
            axs[1].set_ylabel(r"$\|e_i\|_2/\|q_i^{\mathrm{ref}}\|_2$")
            axs[1].set_xlabel("Coefficient index $i$")
            axs[0].set_title(
                rf"{stage.capitalize()} -- $\mu=({mu1:.3f},{mu2:.4f})$ coefficient errors vs linear HPROM"
            )
            axs[0].legend(loc="best")
            fig.tight_layout()
            fig.savefig(
                COEFF_DIR / f"{stage}_{tag}_coeff_abs_rel_vs_index.png",
                dpi=220,
            )
            plt.close(fig)

            # Heatmap grid helper
            def _plot_heatmap_grid(hmaps: Dict[str, np.ndarray], out_path: Path, title: str, cbar_label: str) -> None:
                fig, axs_loc = plt.subplots(3, 2, figsize=(12.0, 9.2), sharex=True, sharey=True)
                axs_flat = axs_loc.ravel()
                vals = np.concatenate([hmaps[m].ravel() for m in MODEL_ORDER_NO_LINEAR])
                vmax = float(np.percentile(vals, 99.0))
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                im = None
                extent = [float(t_ref[0]), float(t_ref[-1]), 1, n_tot]
                for k, m in enumerate(MODEL_ORDER_NO_LINEAR):
                    im = axs_flat[k].imshow(
                        hmaps[m],
                        origin="lower",
                        aspect="auto",
                        extent=extent,
                        cmap="viridis",
                        vmin=0.0,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    axs_flat[k].set_title(m)
                axs_flat[0].set_ylabel("Coefficient index $i$")
                axs_flat[2].set_ylabel("Coefficient index $i$")
                axs_flat[4].set_ylabel("Coefficient index $i$")
                axs_flat[4].set_xlabel("Time $t$")
                axs_flat[5].set_xlabel("Time $t$")
                fig.suptitle(title)
                fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.92, wspace=0.08, hspace=0.16)
                cax = fig.add_axes([0.90, 0.14, 0.022, 0.72])
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(cbar_label)
                fig.savefig(out_path, dpi=220)
                plt.close(fig)

            _plot_heatmap_grid(
                abs_heatmaps,
                COEFF_DIR / f"{stage}_{tag}_coeff_abs_heatmap_grid.png",
                rf"{stage.capitalize()} -- $\mu=({mu1:.3f},{mu2:.4f})$ absolute coefficient error",
                r"$|q_i^{\mathrm{ref}}(t)-q_i(t)|$",
            )
            _plot_heatmap_grid(
                rel_heatmaps,
                COEFF_DIR / f"{stage}_{tag}_coeff_rel_heatmap_grid.png",
                rf"{stage.capitalize()} -- $\mu=({mu1:.3f},{mu2:.4f})$ relative coefficient error",
                r"$|q_i^{\mathrm{ref}}(t)-q_i(t)|/\|q_i^{\mathrm{ref}}\|_2$",
            )


def collect_metrics(stage: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if stage == "baseline":
        root = RESULTS_BASE / "Runs"
    elif stage == "enriched":
        root = RESULTS_ENR / "Runs"
    else:
        raise ValueError(stage)

    for lab, mu1, mu2 in POINTS:
        tag = mu_tag(mu1, mu2)
        # linear summary always comes from baseline linear runs
        lin_path = RESULTS_BASE / "Runs" / "Linear" / f"linear_hprom_{tag}_ntot151" / "summary.txt"
        lin = parse_summary(lin_path)

        if stage == "baseline":
            case1 = parse_summary(root / "Case1" / f"case1_hprom_ann_{tag}_n10_ntot151_summary.txt")
            case2 = parse_summary(root / "Case2" / f"case2_hprom_ann_{tag}_n10_ntot151_summary.txt")
            case3 = parse_summary(root / "Case3" / f"case3_hprom_ann_{tag}_n10_ntot151_summary.txt")
            dd = parse_summary(
                root / "DataDriven" / f"rom_data_driven_{tag}_ntot151" / "rom_data_driven_summary.txt"
            )
            podae = parse_summary(
                root / "PODAE" / f"podae_hprom_{tag}_ntot151_nz5_summary.txt"
            )
            poddl_root = root / "PODDL"
            poddl_pref = f"pod_dl_data_driven_{tag}_ntot151_nz"
            poddl_candidates = sorted([d for d in poddl_root.iterdir() if d.is_dir() and d.name.startswith(poddl_pref)])
            if len(poddl_candidates) == 0:
                raise FileNotFoundError(f"Missing POD-DL-ROM summary folder for '{poddl_pref}*' in {poddl_root}")
            poddl_dir = poddl_candidates[-1]
            try:
                import torch

                ckpt = torch.load(RESULTS_BASE / "Stage3" / "models" / "pod_dl_data_driven_model_hprom.pt", map_location="cpu")
                target_nz = int(ckpt.get("latent_dim", -1))
                exact = [d for d in poddl_candidates if d.name.endswith(f"_nz{target_nz}")]
                if exact:
                    poddl_dir = exact[-1]
            except Exception:
                pass
            poddl = parse_summary(poddl_dir / "pod_dl_data_driven_summary.txt")
        else:
            case1 = parse_summary(
                root / "Case1" / f"case1_hprom_ann_enriched_{tag}_n10_ntot151_summary.txt"
            )
            case2 = parse_summary(
                root / "Case2" / f"case2_hprom_ann_enriched_{tag}_n10_ntot151_summary.txt"
            )
            case3 = parse_summary(
                root / "Case3" / f"case3_hprom_ann_enriched_{tag}_n10_ntot151_summary.txt"
            )
            dd = parse_summary(
                root
                / "DataDriven"
                / f"rom_data_driven_enriched_{tag}_ntot151"
                / "rom_data_driven_enriched_summary.txt"
            )
            podae = parse_summary(
                root / "PODAE" / f"podae_enriched_hprom_{tag}_ntot151_nz5_summary.txt"
            )
            poddl_root = root / "PODDL"
            poddl_pref = f"pod_dl_data_driven_enriched_{tag}_ntot151_nz"
            poddl_candidates = sorted([d for d in poddl_root.iterdir() if d.is_dir() and d.name.startswith(poddl_pref)])
            if len(poddl_candidates) == 0:
                raise FileNotFoundError(f"Missing POD-DL-ROM summary folder for '{poddl_pref}*' in {poddl_root}")
            poddl_dir = poddl_candidates[-1]
            try:
                import torch

                ckpt = torch.load(
                    RESULTS_ENR
                    / "Stage3"
                    / "prom_coeff_dataset_ntot151_enriched_lhs20"
                    / "models"
                    / "pod_dl_data_driven_model_enriched_hprom.pt",
                    map_location="cpu",
                )
                target_nz = int(ckpt.get("latent_dim", -1))
                exact = [d for d in poddl_candidates if d.name.endswith(f"_nz{target_nz}")]
                if exact:
                    poddl_dir = exact[-1]
            except Exception:
                pass
            poddl = parse_summary(poddl_dir / "pod_dl_data_driven_enriched_summary.txt")

        lin_t = as_float(lin, "online_solve_elapsed_s")
        methods = [
            ("Linear PROM", lin, as_float(lin, "online_solve_elapsed_s")),
            ("PROM-ANN Case 1", case1, as_float(case1, "online_solve_elapsed_s")),
            ("PROM-ANN Case 2", case2, as_float(case2, "online_solve_elapsed_s")),
            ("PROM-ANN Case 3", case3, as_float(case3, "online_solve_elapsed_s")),
            ("PROM-POD-AE", podae, as_float(podae, "online_solve_elapsed_s")),
            ("POD-NN-ROM", dd, as_float(dd, "inference_time_s")),
            ("POD-DL-ROM", poddl, as_float(poddl, "inference_time_s")),
        ]
        for mname, summ, mtime in methods:
            rows.append(
                {
                    "stage": stage,
                    "point": lab,
                    "mu1": mu1,
                    "mu2": mu2,
                    "method": mname,
                    "rel_error_percent": as_float(summ, "relative_error_percent"),
                    "online_time_s": mtime,
                    "speedup_vs_linear": (lin_t / mtime) if (np.isfinite(lin_t) and np.isfinite(mtime) and mtime > 0) else np.nan,
                }
            )
    return rows


def write_metrics_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    fields = [
        "stage",
        "point",
        "mu1",
        "mu2",
        "method",
        "rel_error_percent",
        "online_time_s",
        "speedup_vs_linear",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def main() -> None:
    set_style()
    ensure_dirs()

    plot_hdm_vs_models("baseline", FIG_DIR / "baseline_hprom_hdm_vs_all_models.png")
    plot_hdm_vs_models("enriched", FIG_DIR / "enriched_hprom_hdm_vs_all_models.png")
    plot_parameter_domain_sampling(FIG_DIR / "parameter_domain_sampling_points.png")
    plot_verification_3d_cutplanes(FIG_DIR / "verification_hdm_3d_cutplanes_t10.png", t_idx=200)
    plot_verification_3d_cutplanes(FIG_DIR / "verification_hdm_3d_cutplanes_t20.png", t_idx=400)
    compute_coeff_errors_and_heatmaps()

    base_rows = collect_metrics("baseline")
    enr_rows = collect_metrics("enriched")
    write_metrics_csv(base_rows, TABLE_DIR / "burgers_metrics_baseline.csv")
    write_metrics_csv(enr_rows, TABLE_DIR / "burgers_metrics_enriched.csv")

    print("Generated figures in:", FIG_DIR)
    print("Generated tables in:", TABLE_DIR)


if __name__ == "__main__":
    main()
