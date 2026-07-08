#!/usr/bin/env python3
"""Generate a mu^(3) horizontal-cut comparison for six baseline-trained models."""

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
OUTPUT = HERE / "outputs" / "slide65_mu3_baseline_horizontal_six_models.gif"
PREVIEW = HERE / "previews" / "slide65_mu3_baseline_horizontal_six_models_t12p5.png"
HDM_PATH = assets.PROJECT.parent / "Results" / "param_snaps" / "mu1_4.0+mu2_0.033.npy"
PREVIEW_TIME_INDEX = 250


@dataclass(frozen=True)
class PanelSpec:
    label: str
    short_label: str
    snaps_path: Path
    summary_path: Path
    color: str


PANELS: tuple[tuple[PanelSpec, PanelSpec], ...] = (
    (
        PanelSpec(
            "Linear HPROM",
            "Linear HPROM",
            BASE
            / "Linear"
            / "linear_hprom_mu1_4.000_mu2_0.0330_ntot151"
            / "rom_snaps.npy",
            BASE
            / "Linear"
            / "linear_hprom_mu1_4.000_mu2_0.0330_ntot151"
            / "summary.txt",
            assets.COLORS["Linear HPROM"],
        ),
        PanelSpec(
            "PROM-ANN Case 2",
            "HPROM-ANN Case 2",
            BASE
            / "ECSW1pct"
            / "Case2_Best"
            / "np10"
            / "case2_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_snaps.npy",
            BASE
            / "ECSW1pct"
            / "Case2_Best"
            / "np10"
            / "case2_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_summary.txt",
            assets.COLORS["PROM-ANN Case 2"],
        ),
    ),
    (
        PanelSpec(
            "PROM-ANN Case 3",
            "HPROM-ANN Case 3",
            BASE
            / "ECSW1pct"
            / "Case3_Best"
            / "case3_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_snaps.npy",
            BASE
            / "ECSW1pct"
            / "Case3_Best"
            / "case3_hprom_ann_mu1_4.000_mu2_0.0330_n10_ntot151_summary.txt",
            assets.COLORS["PROM-ANN Case 3"],
        ),
        PanelSpec(
            "POD-NN-ROM",
            "POD-NN-ROM",
            BASE
            / "DataDriven_Best"
            / "rom_data_driven_mu1_4.000_mu2_0.0330_ntot151"
            / "rom_snaps.npy",
            BASE
            / "DataDriven_Best"
            / "rom_data_driven_mu1_4.000_mu2_0.0330_ntot151"
            / "rom_data_driven_summary.txt",
            assets.COLORS["POD-NN-ROM"],
        ),
    ),
    (
        PanelSpec(
            "PROM-POD-AE",
            "HPROM-POD-AE",
            BASE
            / "ECSW1pct"
            / "PODAE_Best"
            / "podae_hprom_mu1_4.000_mu2_0.0330_ntot151_nz10_snaps.npy",
            BASE
            / "ECSW1pct"
            / "PODAE_Best"
            / "podae_hprom_mu1_4.000_mu2_0.0330_ntot151_nz10_summary.txt",
            assets.COLORS["PROM-POD-AE"],
        ),
        PanelSpec(
            "POD-DL-ROM",
            "POD-DL-ROM",
            BASE
            / "PODDL_Best"
            / "pod_dl_data_driven_mu1_4.000_mu2_0.0330_ntot151_nz10"
            / "rom_snaps.npy",
            BASE
            / "PODDL_Best"
            / "pod_dl_data_driven_mu1_4.000_mu2_0.0330_ntot151_nz10"
            / "pod_dl_data_driven_summary.txt",
            assets.COLORS["POD-DL-ROM"],
        ),
    ),
)


def read_summary_value(path: Path, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", path.read_text(), re.MULTILINE)
    if match is None:
        raise KeyError(f"Missing '{key}' in {path}")
    return float(match.group(1))


def load_xcut(path: Path) -> np.ndarray:
    snaps = assets.load_npy(path)
    if snaps.shape != (assets.NXY * 2, assets.NT):
        raise ValueError(f"Unexpected snapshot shape for {path}: {snaps.shape}")
    return np.asarray(snaps[assets.IDX_X_CUT, :], dtype=float)


def load_data() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    hdm = load_xcut(HDM_PATH)
    curves: dict[str, np.ndarray] = {}
    errors: dict[str, float] = {}
    for row in PANELS:
        for panel in row:
            curves[panel.label] = load_xcut(panel.snaps_path)
            errors[panel.label] = read_summary_value(panel.summary_path, "relative_error_percent")
    return hdm, curves, errors


def formula_label(model: str) -> str:
    formulas = {
        "Linear HPROM": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathbf q_{\mathrm{tot}}$",
        "PROM-ANN Case 2": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V\mathbf q+\overline{\mathbf V}\mathcal M(\mu,t)$",
        "PROM-ANN Case 3": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V\mathbf q+\overline{\mathbf V}\mathcal H(\mathbf q,\mu,t)$",
        "PROM-POD-AE": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathcal D(\mathbf z)$",
        "POD-NN-ROM": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathcal G_q(\mu,t)$",
        "POD-DL-ROM": r"$\tilde{\mathbf u}=\mathbf u_{\rm ref}+\mathbf V_{\rm tot}\mathcal D(\mathcal G_z(\mu,t))$",
    }
    return formulas[model]


def create_figure(
    hdm: np.ndarray,
    curves: dict[str, np.ndarray],
    errors: dict[str, float],
    time_index: int,
) -> tuple[plt.Figure, list[tuple[Line2D, Line2D, PanelSpec]], plt.Text]:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.8, 7.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.150,
        top=0.845,
        wspace=0.135,
        hspace=0.430,
    )

    artists: list[tuple[Line2D, Line2D, PanelSpec]] = []
    for row, panel_row in enumerate(PANELS):
        for col, panel in enumerate(panel_row):
            ax = axes[row, col]
            hdm_line, = ax.plot(
                assets.X,
                hdm[:, time_index],
                color=assets.COLORS["HDM"],
                linewidth=2.45,
                alpha=0.92,
                label="HDM",
                zorder=2,
            )
            model_line, = ax.plot(
                assets.X,
                curves[panel.label][:, time_index],
                color=panel.color,
                linewidth=2.25,
                alpha=0.92,
                label=panel.short_label,
                zorder=3,
            )
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, 6.6)
            ax.grid(True)
            if row == 2:
                ax.set_xlabel(r"$x$")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(r"$u_x$")
            ax.set_title(
                rf"\textbf{{{panel.short_label}}}"
                rf"\quad $\varepsilon_u={errors[panel.label]:.3f}\%$",
                color=panel.color,
                pad=8,
            )
            ax.text(
                0.50,
                0.88,
                formula_label(panel.label),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.0,
                color=panel.color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
            )
            artists.append((hdm_line, model_line, panel))

    legend_handles = [
        Line2D([0], [0], color=assets.COLORS["HDM"], linewidth=2.6, label="HDM"),
        Line2D([0], [0], color="#555555", linewidth=2.4, label="ROM/Surrogate"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=True,
        columnspacing=2.0,
        handlelength=3.0,
    )
    header = fig.text(
        0.5,
        0.944,
        "",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    header.set_text(
        rf"$\mu^{{(3)}}=(4.000,0.0330),\quad "
        rf"u_x(x,y={assets.Y[assets.MID_Y]:.1f},t),\quad "
        rf"t={time_index * assets.DT:.2f}$"
    )
    return fig, artists, header


def save_preview(hdm: np.ndarray, curves: dict[str, np.ndarray], errors: dict[str, float]) -> None:
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig, _, _ = create_figure(hdm, curves, errors, PREVIEW_TIME_INDEX)
    fig.savefig(PREVIEW, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(PREVIEW)


def save_animation(hdm: np.ndarray, curves: dict[str, np.ndarray], errors: dict[str, float]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = np.arange(0, assets.NT, 10, dtype=int)
    fig, artists, header = create_figure(hdm, curves, errors, int(frame_ids[0]))

    def update(frame_index: int):
        time_index = int(frame_ids[frame_index])
        changed: list[Line2D | plt.Text] = []
        for hdm_line, model_line, panel in artists:
            hdm_line.set_ydata(hdm[:, time_index])
            model_line.set_ydata(curves[panel.label][:, time_index])
            changed.extend((hdm_line, model_line))
        header.set_text(
            rf"$\mu^{{(3)}}=(4.000,0.0330),\quad "
            rf"u_x(x,y={assets.Y[assets.MID_Y]:.1f},t),\quad "
            rf"t={time_index * assets.DT:.2f}$"
        )
        changed.append(header)
        return changed

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=100,
        blit=False,
    )
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
