#!/usr/bin/env python3
"""Generate a presentation preview for the PROM-consistent training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "previews" / "slide59_rom_consistent_pipeline.png"


def add_stage(
    ax: plt.Axes,
    x: float,
    width: float,
    color: str,
    title: str,
    body: str,
    footer: str,
    body_fontsize: float = 16,
) -> None:
    box = FancyBboxPatch(
        (x, 0.40),
        width,
        0.46,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor="#2d3748",
        linewidth=1.7,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        0.79,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        0.64,
        body,
        ha="center",
        va="center",
        fontsize=body_fontsize,
        linespacing=1.35,
    )
    ax.text(
        x + width / 2,
        0.46,
        footer,
        ha="center",
        va="center",
        fontsize=13,
        color="#30343b",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.2, 6.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        r"\textit{One common PROM-consistent coefficient dataset is used to train all learned models.}",
        ha="center",
        va="center",
        fontsize=16,
        color="#3f4650",
    )

    width = 0.255
    positions = [0.055, 0.3725, 0.690]
    add_stage(
        ax,
        positions[0],
        width,
        "#eaf2fb",
        r"\textbf{Stage 1}",
        (
            r"\textbf{HDM snapshots}"
            "\n"
            r"$4509\ \mathrm{states}$"
            "\n"
            r"$\Downarrow$"
            "\n"
            r"$\mathbf V_{\mathrm{tot}}:"
            r"\ 125000\times151$"
        ),
        r"LSPG-sensitive basis",
    )
    add_stage(
        ax,
        positions[1],
        width,
        "#edf7ef",
        r"\textbf{Stage 2}",
        (
            r"\textbf{Linear HPROM}"
            "\n"
            r"$n_{\mathrm{tot}}=151$"
            "\n"
            r"$\Downarrow$"
            "\n"
            r"$\mathbf q_{\mathrm{tot}}^{\mathrm{HPROM}}"
            r"(\boldsymbol\mu,t)$"
        ),
        r"PROM-consistent trajectories",
    )
    add_stage(
        ax,
        positions[2],
        width,
        "#f7eef8",
        r"\textbf{Stage 3}",
        (
            r"\textbf{Learned maps}"
            "\n"
            r"$\bar{\mathbf q}=\mathcal N(\mathbf q),\ "
            r"\mathcal M(\boldsymbol\mu,t),\ "
            r"\mathcal H(\mathbf q,\boldsymbol\mu,t)$"
            "\n"
            r"$\widehat{\mathbf q}_{\mathrm{tot}}="
            r"\mathcal D(\mathcal E(\mathbf q_{\mathrm{tot}}))$"
            "\n"
            r"$\mathbf q_{\mathrm{tot}}=\mathcal G_q(\boldsymbol\mu,t)$"
            "\n"
            r"$\mathbf z=\mathcal G_z(\boldsymbol\mu,t)$"
        ),
        r"PROM-consistent training targets",
        body_fontsize=12.0,
    )

    for start, end in ((positions[0] + width, positions[1]), (positions[1] + width, positions[2])):
        arrow = FancyArrowPatch(
            (start + 0.01, 0.64),
            (end - 0.01, 0.64),
            arrowstyle="-|>",
            mutation_scale=23,
            linewidth=2.2,
            color="#2d3748",
        )
        ax.add_patch(arrow)

    note = FancyBboxPatch(
        (0.105, 0.055),
        0.79,
        0.20,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#fff7df",
        edgecolor="#bb8a16",
        linewidth=1.6,
    )
    ax.add_patch(note)
    ax.text(
        0.5,
        0.205,
        r"\textbf{Why use linear-HPROM coordinates?}",
        ha="center",
        va="center",
        fontsize=16,
        color="#6f4b00",
    )
    ax.text(
        0.5,
        0.112,
        (
            r"$^{*}$The non-intrusive ROMs could also be trained from directly projected coefficients."
            "\n"
            r"However, the linear HPROM remains very close to the projection-only reconstruction,"
            "\n"
            r"so its $\mathbf q_{\mathrm{tot}}^{\mathrm{HPROM}}$ trajectories are used as PROM-consistent targets,"
            "\n"
            r"preserving consistency with the intrusive PROM online solves."
        ),
        ha="center",
        va="center",
        fontsize=11.8,
        linespacing=1.08,
    )

    fig.savefig(OUTPUT, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
