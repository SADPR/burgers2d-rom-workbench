#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate sampling-layout figures for the 250x250 report.

Outputs:
- Figures/stage1_sampling_points.png   (baseline 3x3 + evaluation points)
- Figures/stage2_sampling_points.png   (baseline 3x3 + LHS enrichment + evaluation points)
"""

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU
from burgers.core import get_snapshot_params

MU_VERIFY = (4.875, 0.0225)
MU_TEST_1 = (4.56, 0.019)
MU_TEST_2 = (5.19, 0.026)


def _set_style():
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
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
        }
    )


def _load_lhs_points():
    patterns = [
        os.path.join(
            THIS_DIR,
            "Results_Enrichment",
            "Stage2",
            "prom_coeff_dataset_ntot*_enriched_lhs*",
            "lhs_mu.npy",
        ),
        os.path.join(
            PROJECT_ROOT,
            "Project_YvonMaday",
            "Results_Enrichment",
            "Stage2",
            "prom_coeff_dataset_ntot*_enriched_lhs*",
            "lhs_mu.npy",
        ),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return np.zeros((0, 2), dtype=np.float64), None
    latest = max(candidates, key=os.path.getmtime)
    lhs = np.asarray(np.load(latest, allow_pickle=False), dtype=np.float64)
    if lhs.ndim != 2 or lhs.shape[1] != 2:
        raise ValueError(f"Unexpected LHS shape in '{latest}': {lhs.shape}")
    return lhs, latest


def _apply_domain_layout(ax, mu1_range, mu2_range):
    mu1_lo, mu1_hi = float(mu1_range[0]), float(mu1_range[1])
    mu2_lo, mu2_hi = float(mu2_range[0]), float(mu2_range[1])
    pad_x = 0.06 * (mu1_hi - mu1_lo)
    pad_y = 0.08 * (mu2_hi - mu2_lo)

    ax.set_xlim(mu1_lo - pad_x, mu1_hi + pad_x)
    ax.set_ylim(mu2_lo - pad_y, mu2_hi + pad_y)
    # Visualize exact parameter domain boundaries.
    ax.plot(
        [mu1_lo, mu1_hi, mu1_hi, mu1_lo, mu1_lo],
        [mu2_lo, mu2_lo, mu2_hi, mu2_hi, mu2_lo],
        color="0.25",
        linewidth=1.4,
        linestyle="-",
        alpha=0.85,
        zorder=1,
    )
    ax.grid(True)


def _plot_eval_points(ax):
    eval_points = np.asarray([MU_VERIFY, MU_TEST_1, MU_TEST_2], dtype=np.float64)
    eval_labels = [r"Verification $\mu^{(v)}$", r"Test $\mu^{(1)}$", r"Test $\mu^{(2)}$"]
    eval_colors = ["tab:red", "tab:orange", "tab:green"]

    for (mu1, mu2), label, color in zip(eval_points, eval_labels, eval_colors):
        ax.scatter(
            mu1,
            mu2,
            s=170,
            c=color,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.95,
            label=label,
            zorder=6,
        )
    # Compact point tags for fast visual reference.
    ax.text(MU_VERIFY[0] + 0.02, MU_VERIFY[1] + 0.00035, r"$\mu^{(v)}$", color="tab:red", fontsize=11)
    ax.text(MU_TEST_1[0] + 0.02, MU_TEST_1[1] + 0.00035, r"$\mu^{(1)}$", color="tab:orange", fontsize=11)
    ax.text(MU_TEST_2[0] + 0.02, MU_TEST_2[1] + 0.00035, r"$\mu^{(2)}$", color="tab:green", fontsize=11)


def _plot_stage1(base_points, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.scatter(
        base_points[:, 0],
        base_points[:, 1],
        s=110,
        c="black",
        marker="o",
        alpha=0.9,
        label="Baseline training points (3x3)",
        zorder=3,
    )
    _plot_eval_points(ax)
    _apply_domain_layout(ax, MU1_RANGE, MU2_RANGE)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Baseline Sampling and Evaluation Points")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_stage2(base_points, lhs_points, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.scatter(
        base_points[:, 0],
        base_points[:, 1],
        s=100,
        c="black",
        marker="o",
        alpha=0.9,
        label="Baseline training (first 9)",
        zorder=3,
    )
    if lhs_points.shape[0] > 0:
        ax.scatter(
            lhs_points[:, 0],
            lhs_points[:, 1],
            s=90,
            c="tab:blue",
            marker="x",
            alpha=0.9,
            label="Enrichment LHS points",
            zorder=4,
        )
    _plot_eval_points(ax)
    _apply_domain_layout(ax, MU1_RANGE, MU2_RANGE)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Baseline, Enrichment, and Evaluation Points")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    _set_style()
    out_dir = os.path.join(THIS_DIR, "Figures")
    os.makedirs(out_dir, exist_ok=True)

    base_points = np.asarray(
        get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        ),
        dtype=np.float64,
    )
    if base_points.ndim != 2 or base_points.shape[1] != 2:
        raise ValueError(f"Unexpected baseline point shape: {base_points.shape}")
    # Keep only the initial 3x3 design.
    base_points = base_points[:9, :]

    lhs_points, lhs_path = _load_lhs_points()

    stage1_out = os.path.join(out_dir, "stage1_sampling_points.png")
    stage2_out = os.path.join(out_dir, "stage2_sampling_points.png")
    _plot_stage1(base_points, stage1_out)
    _plot_stage2(base_points, lhs_points, stage2_out)

    print(f"Saved: {stage1_out}")
    print(f"Saved: {stage2_out}")
    if lhs_path is not None:
        print(f"LHS source: {lhs_path}")
    else:
        print("LHS source: not found (stage2 plot generated with baseline points only).")


if __name__ == "__main__":
    main()
