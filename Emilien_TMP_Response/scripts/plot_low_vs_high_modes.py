#!/usr/bin/env python3
"""Plot representative low, mid, and high-index coefficient trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--q",
        type=Path,
        default=Path("Emilien_TMP_Response/data_ref/qN_mu1_5.500_mu2_0.0150.npy"),
        help="Path to qN array of shape (n_modes, n_samples)",
    )
    p.add_argument(
        "--t",
        type=Path,
        default=Path("Emilien_TMP_Response/data_ref/t_mu1_5.500_mu2_0.0150.npy"),
        help="Path to sample abscissa array of shape (n_samples,)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Emilien_TMP_Response/figures/low_vs_high_modes_mu1_5.500_mu2_0.0150.png"),
        help="Output figure path",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("Emilien_TMP_Response/figures/low_vs_high_modes_summary.txt"),
        help="Output text summary path",
    )
    return p.parse_args()


def _as_plottable(arr: np.ndarray) -> tuple[np.ndarray, str]:
    if np.iscomplexobj(arr):
        return np.abs(arr), "modulus"
    return arr, "value"


def _normalized_tv(x: np.ndarray) -> float:
    denom = np.sum(np.abs(x), axis=1) + 1.0e-14
    tv = np.sum(np.abs(np.diff(x, axis=1)), axis=1)
    return float(np.mean(tv / denom))


def main() -> None:
    args = parse_args()

    q = np.load(args.q)
    t = np.load(args.t)

    if q.ndim != 2:
        raise ValueError(f"Expected q with shape (n_modes, n_samples), got {q.shape}")
    if t.ndim != 1:
        raise ValueError(f"Expected t with shape (n_samples,), got {t.shape}")
    if q.shape[1] != t.shape[0]:
        raise ValueError(f"Mismatch: q has {q.shape[1]} samples, t has {t.shape[0]}")
    if q.shape[0] < 104:
        raise ValueError(f"Need at least 104 coefficients, got {q.shape[0]}")

    group_specs = [
        ("Modes $q_1$-$q_4$", [0, 1, 2, 3], 1.8),
        ("Modes $q_{21}$-$q_{24}$", [20, 21, 22, 23], 1.5),
        ("Modes $q_{101}$-$q_{104}$", [100, 101, 102, 103], 1.3),
    ]

    q_plot, y_label_kind = _as_plottable(q)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)

    for ax, (title, idxs, lw) in zip(axes, group_specs):
        for i in idxs:
            ax.plot(t, q_plot[i], linewidth=lw, label=fr"$q_{{{i+1}}}$")
        ax.set_title(title)
        ax.set_ylabel(f"Coefficient {y_label_kind}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=9, frameon=False)

    axes[-1].set_xlabel("Trajectory coordinate")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)

    roughness = {}
    for _, idxs, _ in group_specs:
        key = "_".join(str(i + 1) for i in idxs)
        roughness[key] = _normalized_tv(q_plot[idxs, :])

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "Coefficient trajectory roughness summary",
                f"q_path={args.q}",
                f"t_path={args.t}",
                f"n_modes={q.shape[0]}",
                f"n_samples={q.shape[1]}",
                f"group_q1_4={roughness['1_2_3_4']:.6e}",
                f"group_q21_24={roughness['21_22_23_24']:.6e}",
                f"group_q101_104={roughness['101_102_103_104']:.6e}",
                f"ratio_q21_24_over_q1_4={roughness['21_22_23_24']/max(roughness['1_2_3_4'],1.0e-14):.6f}",
                f"ratio_q101_104_over_q1_4={roughness['101_102_103_104']/max(roughness['1_2_3_4'],1.0e-14):.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
