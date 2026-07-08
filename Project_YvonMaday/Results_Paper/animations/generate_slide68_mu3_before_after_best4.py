#!/usr/bin/env python3
"""Generate a before/after enrichment GIF for the four clearest mu^(3) gains."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D

import generate_burgers_presentation_assets as assets


HERE = Path(__file__).resolve().parent
BASE = assets.MAIN / "Runs" / "Extrapolation20pct"
EXT = assets.PAPER / "mlspg_hprom_enrichment_ext25_lhs36" / "Runs"
OUTPUT = HERE / "outputs" / "slide68_mu3_before_after_best4.gif"
PREVIEW = HERE / "previews" / "slide68_mu3_before_after_best4_t12p5.png"
HDM_PATH = assets.PROJECT.parent / "Results" / "param_snaps" / "mu1_4.0+mu2_0.033.npy"
PREVIEW_TIME_INDEX = 250


@dataclass(frozen=True)
class MethodSpec:
    label: str
    color: str
    base_snaps: Path
    base_summary: Path
    ext_snaps: Path
    ext_summary: Path


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "PROM-POD-AE",
        assets.COLORS["PROM-POD-AE"],
        BASE / "ECSW1pct" / "PODAE_Best" / "podae_hprom_mu1_4.000_mu2_0.0330_ntot151_nz10_snaps.npy",
        BASE / "ECSW1pct" / "PODAE_Best" / "podae_hprom_mu1_4.000_mu2_0.0330_ntot151_nz10_summary.txt",
        EXT / "ECSW1pct" / "PODAE_Best" / "podae_hprom_mu1_4.000_mu2_0.0330_ntot151_nz10_snaps.npy",
        EXT / "ECSW1pct" / "PODAE_Best" / "podae_hprom_mu1_4.000_mu2_0.0330_ntot151_nz10_summary.txt",
    ),
    MethodSpec(
        "PROM-ANN Case 2",
        assets.COLORS["PROM-ANN Case 2"],
        BASE / "ECSW1pct" / "Case2_Best" / "np10" / "case2_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_snaps.npy",
        BASE / "ECSW1pct" / "Case2_Best" / "np10" / "case2_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_summary.txt",
        EXT / "ECSW1pct" / "Case2_Best" / "np10" / "case2_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_snaps.npy",
        EXT / "ECSW1pct" / "Case2_Best" / "np10" / "case2_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_summary.txt",
    ),
    MethodSpec(
        "POD-NN-ROM",
        assets.COLORS["POD-NN-ROM"],
        BASE / "DataDriven_Best" / "rom_data_driven_mu1_4.000_mu2_0.0330_ntot151" / "rom_snaps.npy",
        BASE / "DataDriven_Best" / "rom_data_driven_mu1_4.000_mu2_0.0330_ntot151" / "rom_data_driven_summary.txt",
        EXT / "DataDriven_Best" / "rom_data_driven_mu1_4.000_mu2_0.0330_ntot151" / "rom_snaps.npy",
        EXT / "DataDriven_Best" / "rom_data_driven_mu1_4.000_mu2_0.0330_ntot151" / "rom_data_driven_summary.txt",
    ),
    MethodSpec(
        "POD-DL-ROM",
        assets.COLORS["POD-DL-ROM"],
        BASE / "PODDL_Best" / "pod_dl_data_driven_mu1_4.000_mu2_0.0330_ntot151_nz10" / "rom_snaps.npy",
        BASE / "PODDL_Best" / "pod_dl_data_driven_mu1_4.000_mu2_0.0330_ntot151_nz10" / "pod_dl_data_driven_summary.txt",
        EXT / "PODDL_Best" / "pod_dl_data_driven_mu1_4.000_mu2_0.0330_ntot151_nz10" / "rom_snaps.npy",
        EXT / "PODDL_Best" / "pod_dl_data_driven_mu1_4.000_mu2_0.0330_ntot151_nz10" / "pod_dl_data_driven_summary.txt",
    ),
)


def read_error(path: Path) -> float:
    text = path.read_text()
    match = re.search(r"^relative_error_percent:\s*([0-9.]+)", text, re.MULTILINE)
    if match is None:
        raise KeyError(f"relative_error_percent not found in {path}")
    return float(match.group(1))


def load_xcut(path: Path) -> np.ndarray:
    snaps = assets.load_npy(path)
    if snaps.shape != (assets.NXY * 2, assets.NT):
        raise ValueError(f"Unexpected snapshot shape for {path}: {snaps.shape}")
    return np.asarray(snaps[assets.IDX_X_CUT, :], dtype=float)


def load_data() -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    hdm = load_xcut(HDM_PATH)
    curves: dict[tuple[str, str], np.ndarray] = {}
    errors: dict[tuple[str, str], float] = {}
    for method in METHODS:
        curves[(method.label, "base")] = load_xcut(method.base_snaps)
        curves[(method.label, "ext")] = load_xcut(method.ext_snaps)
        errors[(method.label, "base")] = read_error(method.base_summary)
        errors[(method.label, "ext")] = read_error(method.ext_summary)
    return hdm, curves, errors


def reduction_percent(base: float, ext: float) -> float:
    return 100.0 * (base - ext) / base


def display_label(model: str) -> str:
    if model == "PROM-POD-AE":
        return "HPROM-POD-AE"
    if model.startswith("PROM-ANN Case "):
        return model.replace("PROM-ANN", "HPROM-ANN")
    return model


def formula_label(model: str) -> str:
    formulas = {
        "PROM-POD-AE": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathcal D(\mathbf z)$",
        "PROM-ANN Case 2": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V\mathbf q+\overline{\mathbf V}\mathcal M(\mu,t)$",
        "POD-NN-ROM": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathcal G_q(\mu,t)$",
        "POD-DL-ROM": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathcal D(\mathcal G_z(\mu,t))$",
    }
    return formulas[model]


def create_figure(
    hdm: np.ndarray,
    curves: dict[tuple[str, str], np.ndarray],
    errors: dict[tuple[str, str], float],
    time_index: int,
) -> tuple[plt.Figure, list[tuple[Line2D, Line2D, MethodSpec, str]], plt.Text]:
    fig, axes = plt.subplots(
        len(METHODS),
        3,
        figsize=(12.8, 7.2),
        gridspec_kw={"width_ratios": [0.66, 1.0, 1.0]},
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.125, top=0.835, wspace=0.165, hspace=0.360)

    fig.text(0.430, 0.875, r"\textbf{Before: 9 HPROM trajectories}", ha="center", va="center", fontsize=13)
    fig.text(0.760, 0.875, r"\textbf{After: 9 + 36 HPROM trajectories}", ha="center", va="center", fontsize=13)

    artists: list[tuple[Line2D, Line2D, MethodSpec, str]] = []
    for row, method in enumerate(METHODS):
        base_err = errors[(method.label, "base")]
        ext_err = errors[(method.label, "ext")]
        red = reduction_percent(base_err, ext_err)

        label_ax = axes[row, 0]
        label_ax.axis("off")
        label_ax.text(0.98, 0.73, rf"\textbf{{{display_label(method.label)}}}", ha="right", va="center", fontsize=13, color=method.color)
        label_ax.text(0.98, 0.51, formula_label(method.label), ha="right", va="center", fontsize=8.8, color=method.color)
        label_ax.text(0.98, 0.29, rf"$\downarrow\ {red:.0f}\%$", ha="right", va="center", fontsize=18, color="#1b7f3a", fontweight="bold")
        label_ax.text(0.98, 0.10, "trajectory error", ha="right", va="center", fontsize=8.5, color="#444444")

        for col, stage in enumerate(("base", "ext"), start=1):
            ax = axes[row, col]
            hdm_line, = ax.plot(assets.X, hdm[:, time_index], color=assets.COLORS["HDM"], linewidth=2.25, alpha=0.92)
            model_line, = ax.plot(assets.X, curves[(method.label, stage)][:, time_index], color=method.color, linewidth=2.1, alpha=0.92)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, 6.6)
            ax.grid(True)
            if row == len(METHODS) - 1:
                ax.set_xlabel(r"$x$")
            else:
                ax.tick_params(labelbottom=False)
            if col == 1:
                ax.set_ylabel(r"$u_x$")
            else:
                ax.tick_params(labelleft=False)
            err = errors[(method.label, stage)]
            title = rf"$\varepsilon_u={err:.2f}\%$"
            ax.set_title(title, color=method.color, pad=5)
            artists.append((hdm_line, model_line, method, stage))

    legend_handles = [
        Line2D([0], [0], color=assets.COLORS["HDM"], linewidth=2.5, label="HDM"),
        Line2D([0], [0], color="#555555", linewidth=2.3, label="model prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.59, 0.040), ncol=2, frameon=True, columnspacing=1.8, handlelength=3.0)
    header = fig.text(0.5, 0.944, "", ha="center", va="center", fontsize=15, fontweight="bold")
    header.set_text(rf"$\mu^{{(3)}}=(4.000,0.0330),\quad u_x(x,y={assets.Y[assets.MID_Y]:.1f},t),\quad t={time_index * assets.DT:.2f}$")
    return fig, artists, header


def save_preview(hdm: np.ndarray, curves: dict[tuple[str, str], np.ndarray], errors: dict[tuple[str, str], float]) -> None:
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig, _, _ = create_figure(hdm, curves, errors, PREVIEW_TIME_INDEX)
    fig.savefig(PREVIEW, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(PREVIEW)


def save_animation(hdm: np.ndarray, curves: dict[tuple[str, str], np.ndarray], errors: dict[tuple[str, str], float]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = np.arange(0, assets.NT, 10, dtype=int)
    fig, artists, header = create_figure(hdm, curves, errors, int(frame_ids[0]))

    def update(frame_index: int):
        time_index = int(frame_ids[frame_index])
        changed: list[Line2D | plt.Text] = []
        for hdm_line, model_line, method, stage in artists:
            hdm_line.set_ydata(hdm[:, time_index])
            model_line.set_ydata(curves[(method.label, stage)][:, time_index])
            changed.extend((hdm_line, model_line))
        header.set_text(rf"$\mu^{{(3)}}=(4.000,0.0330),\quad u_x(x,y={assets.Y[assets.MID_Y]:.1f},t),\quad t={time_index * assets.DT:.2f}$")
        changed.append(header)
        return changed

    movie = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=100, blit=False)
    movie.save(OUTPUT, writer=animation.PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    print(OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    assets.configure_style()
    hdm, curves, errors = load_data()
    save_preview(hdm, curves, errors)
    if not args.preview_only:
        save_animation(hdm, curves, errors)


if __name__ == "__main__":
    main()
