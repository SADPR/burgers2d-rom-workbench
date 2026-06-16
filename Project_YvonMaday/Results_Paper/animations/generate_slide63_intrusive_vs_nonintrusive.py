#!/usr/bin/env python3
"""Generate the Slide 63 intrusive vs non-intrusive comparison at mu=(5.19, 0.026)."""

from __future__ import annotations

import argparse
import os
import re
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
PREVIEW = HERE / "previews" / "slide63_intrusive_vs_nonintrusive_t12p5.png"
OUTPUT = HERE / "outputs" / "slide63_intrusive_vs_nonintrusive_mu2.gif"

POINT = assets.POINTS[2]
PREVIEW_TIME_INDEX = 250

MEAN_SPEEDUPS = {
    "PROM-POD-AE": r"$44.2$x",
    "POD-NN-ROM": r"$1.87\times10^4$",
    "POD-DL-ROM": r"$3.58\times10^4$",
}


def read_summary_value(path: Path, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", path.read_text(), re.MULTILINE)
    if match is None:
        raise KeyError(f"Missing '{key}' in {path}")
    return float(match.group(1))


def load_row_metrics() -> dict[str, tuple[float, str]]:
    summary_paths = {
        "PROM-POD-AE": (
            assets.MAIN
            / "Runs"
            / "ECSW1pct"
            / "PODAE_Best"
            / f"podae_hprom_{POINT.path_tag}_ntot151_nz10_summary.txt"
        ),
        "POD-NN-ROM": (
            assets.MAIN
            / "Runs"
            / "DataDriven_Best"
            / f"rom_data_driven_{POINT.path_tag}_ntot151"
            / "rom_data_driven_summary.txt"
        ),
        "POD-DL-ROM": (
            assets.MAIN
            / "Runs"
            / "PODDL_Best"
            / f"pod_dl_data_driven_{POINT.path_tag}_ntot151_nz10"
            / "pod_dl_data_driven_summary.txt"
        ),
    }
    metrics: dict[str, tuple[float, str]] = {}
    for label, path in summary_paths.items():
        error = read_summary_value(path, "relative_error_percent")
        metrics[label] = (error, MEAN_SPEEDUPS[label])
    return metrics


def load_comparison_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "HDM": assets.snapshot_cut_data(assets.hdm_path(POINT)),
        "PROM-POD-AE": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["podae"], POINT)
        ),
        "POD-NN-ROM": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["podnn"], POINT)
        ),
        "POD-DL-ROM": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["poddl"], POINT)
        ),
    }


def create_figure(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, tuple[float, str]],
    time_index: int,
) -> tuple[plt.Figure, list[tuple[Line2D, ...]], plt.Text]:
    comparisons = (
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-DL-ROM",
    )
    fig, axes = plt.subplots(
        len(comparisons),
        2,
        figsize=(12.8, 7.8),
        sharex="col",
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.175,
        right=0.985,
        bottom=0.16,
        top=0.86,
        wspace=0.18,
        hspace=0.42,
    )

    styles = {
        "HDM": {
            "color": assets.COLORS["HDM"],
            "linewidth": 2.4,
            "linestyle": "-",
            "alpha": 0.92,
            "zorder": 2,
        },
        "PROM-POD-AE": {
            "color": assets.COLORS["PROM-POD-AE"],
            "linewidth": 2.3,
            "linestyle": "-",
            "alpha": 0.90,
            "zorder": 3,
        },
        "POD-NN-ROM": {
            "color": assets.COLORS["POD-NN-ROM"],
            "linewidth": 2.3,
            "linestyle": "-",
            "alpha": 0.92,
            "zorder": 3,
        },
        "POD-DL-ROM": {
            "color": assets.COLORS["POD-DL-ROM"],
            "linewidth": 2.3,
            "linestyle": "-",
            "alpha": 0.92,
            "zorder": 3,
        },
    }

    grids = (assets.X, assets.Y)
    cut_titles = (
        rf"$u_x(x,y={assets.Y[assets.MID_Y]:.1f},t)$",
        rf"$u_x(x={assets.X[assets.MID_X]:.1f},y,t)$",
    )
    axis_labels = (r"$x$", r"$y$")
    y_limits = []
    for col in range(2):
        ymax = max(float(np.max(data[model][col])) for model in comparisons)
        y_limits.append(np.ceil((ymax + 0.15) * 10.0) / 10.0)
    artists: list[tuple[Line2D, ...]] = []

    for row, model in enumerate(comparisons):
        for col in range(2):
            ax = axes[row, col]
            hdm_line, = ax.plot(
                grids[col],
                data["HDM"][col][:, time_index],
                label="HDM",
                **styles["HDM"],
            )
            model_line, = ax.plot(
                grids[col],
                data[model][col][:, time_index],
                label=model,
                **styles[model],
            )

            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(0.0, y_limits[col])
            if row == len(comparisons) - 1:
                ax.set_xlabel(axis_labels[col])
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(r"$u_x$")
            if row == 0:
                ax.set_title(cut_titles[col], pad=10)
            ax.grid(True)
            artists.append((hdm_line, model_line))

        row_box = axes[row, 0].get_position()
        error, speedup = metrics[model]
        fig.text(
            0.018,
            row_box.y0 + 0.68 * row_box.height,
            model,
            ha="left",
            va="center",
            fontsize=14.8,
            fontweight="bold",
            linespacing=1.0,
            multialignment="left",
        )
        fig.text(
            0.018,
            row_box.y0 + 0.40 * row_box.height,
            f"error: {error:.3f}\\%\nspeedup: {speedup}",
            ha="left",
            va="center",
            fontsize=12.8,
            linespacing=1.32,
        )

    legend_handles = [
        Line2D([0], [0], label=label, **styles[label])
        for label in ("HDM", *comparisons)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=True,
        columnspacing=1.25,
        handlelength=2.8,
    )
    header = fig.text(
        0.5,
        0.945,
        "",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    header.set_text(
        rf"$\mu_1={POINT.mu1:.2f},\quad "
        rf"\mu_2={POINT.mu2:.3f}\quad"
        rf"\mathrm{{(off\mbox{{-}}grid\ test)}},\qquad "
        rf"t={time_index * assets.DT:.2f}$"
    )
    return fig, artists, header


def save_preview(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig, _, _ = create_figure(data, load_row_metrics(), PREVIEW_TIME_INDEX)
    fig.savefig(PREVIEW, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(PREVIEW)


def save_animation(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = np.arange(0, assets.NT, 10, dtype=int)
    fig, artists, header = create_figure(data, load_row_metrics(), int(frame_ids[0]))
    panel_models = (
        "PROM-POD-AE",
        "PROM-POD-AE",
        "POD-NN-ROM",
        "POD-NN-ROM",
        "POD-DL-ROM",
        "POD-DL-ROM",
    )

    def update(frame_index: int):
        time_index = int(frame_ids[frame_index])
        changed: list[Line2D | plt.Text] = []
        for panel_index, (hdm_line, model_line) in enumerate(artists):
            col = panel_index % 2
            model = panel_models[panel_index]
            hdm_line.set_ydata(data["HDM"][col][:, time_index])
            model_line.set_ydata(data[model][col][:, time_index])
            changed.extend((hdm_line, model_line))
        header.set_text(
            rf"$\mu_1={POINT.mu1:.2f},\quad "
            rf"\mu_2={POINT.mu2:.3f}\quad"
            rf"\mathrm{{(off\mbox{{-}}grid\ test)}},\qquad "
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
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Generate only the representative t=12.5 frame.",
    )
    args = parser.parse_args()

    assets.configure_style()
    data = load_comparison_data()
    save_preview(data)
    if not args.preview_only:
        save_animation(data)


if __name__ == "__main__":
    main()
