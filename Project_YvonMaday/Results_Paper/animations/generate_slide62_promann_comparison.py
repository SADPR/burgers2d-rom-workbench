#!/usr/bin/env python3
"""Generate the Slide 62 PROM-ANN comparison at mu=(4.56, 0.019)."""

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
PREVIEW = HERE / "previews" / "slide62_promann_comparison_t12p5.png"
OUTPUT = HERE / "outputs" / "slide62_promann_comparison_mu1.gif"

POINT = assets.POINTS[1]
PREVIEW_TIME_INDEX = 250
HDM_MEAN_RUNTIME_S = 737.44


def read_summary_value(path: Path, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", path.read_text(), re.MULTILINE)
    if match is None:
        raise KeyError(f"Missing '{key}' in {path}")
    return float(match.group(1))


def load_row_metrics() -> dict[str, tuple[float, float]]:
    summary_paths = {
        "PROM-ANN Case 1": (
            assets.MAIN
            / "Runs"
            / "ECSW1pct"
            / "Case1_Best"
            / f"case1_hprom_ann_{POINT.path_tag}_n10_ntot151_summary.txt"
        ),
        "PROM-ANN Case 2": (
            assets.MAIN
            / "Runs"
            / "ECSW1pct"
            / "Case2_Best"
            / "np10"
            / f"case2_hprom_ann_{POINT.path_tag}_n10_ntot151_summary.txt"
        ),
        "PROM-ANN Case 3": (
            assets.MAIN
            / "Runs"
            / "ECSW1pct"
            / "Case3_Best"
            / f"case3_hprom_ann_{POINT.path_tag}_n10_ntot151_summary.txt"
        ),
    }
    metrics: dict[str, tuple[float, float]] = {}
    for label, path in summary_paths.items():
        error = read_summary_value(path, "relative_error_percent")
        runtime = read_summary_value(path, "online_solve_elapsed_s")
        metrics[label] = (error, HDM_MEAN_RUNTIME_S / runtime)
    return metrics


def display_label(model: str) -> str:
    if model.startswith("PROM-ANN Case "):
        return model.replace("PROM-ANN ", "HPROM-ANN\n")
    return model


def legend_label(model: str) -> str:
    return display_label(model).replace("\n", " ")


def formula_label(model: str) -> str:
    formulas = {
        "PROM-ANN Case 1": (
            r"$\tilde{\mathbf u}=\mathbf u_{\mathrm{ref}}+\mathbf V\mathbf q+\overline{\mathbf V}\mathcal N(\mathbf q)$"
        ),
        "PROM-ANN Case 2": (
            r"$\tilde{\mathbf u}=\mathbf u_{\mathrm{ref}}+\mathbf V\mathbf q+\overline{\mathbf V}\mathcal M(\mu,t)$"
        ),
        "PROM-ANN Case 3": (
            r"$\tilde{\mathbf u}=\mathbf u_{\mathrm{ref}}+\mathbf V\mathbf q+\overline{\mathbf V}\mathcal H(\mathbf q,\mu,t)$"
        ),
    }
    return formulas[model]


def load_comparison_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "HDM": assets.snapshot_cut_data(assets.hdm_path(POINT)),
        "PROM-ANN Case 1": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["case1"], POINT)
        ),
        "PROM-ANN Case 2": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["case2"], POINT)
        ),
        "PROM-ANN Case 3": assets.snapshot_cut_data(
            assets.model_snaps_path(assets.MAIN, assets.MODELS["case3"], POINT)
        ),
    }


def create_figure(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, tuple[float, float]],
    time_index: int,
) -> tuple[plt.Figure, list[tuple[Line2D, ...]], plt.Text]:
    comparisons = (
        "PROM-ANN Case 1",
        "PROM-ANN Case 2",
        "PROM-ANN Case 3",
    )
    fig, axes = plt.subplots(
        len(comparisons),
        2,
        figsize=(12.8, 7.8),
        sharex="col",
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.225,
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
        "PROM-ANN Case 1": {
            "color": assets.COLORS["PROM-ANN Case 1"],
            "linewidth": 2.2,
            "linestyle": "-",
            "alpha": 0.86,
            "zorder": 3,
        },
        "PROM-ANN Case 2": {
            "color": assets.COLORS["PROM-ANN Case 2"],
            "linewidth": 2.2,
            "linestyle": "-",
            "alpha": 0.92,
            "zorder": 3,
        },
        "PROM-ANN Case 3": {
            "color": assets.COLORS["PROM-ANN Case 3"],
            "linewidth": 2.2,
            "linestyle": "-",
            "alpha": 0.90,
            "zorder": 3,
        },
    }

    grids = (assets.X, assets.Y)
    cut_titles = (
        rf"$u_x(x,y={assets.Y[assets.MID_Y]:.1f},t)$",
        rf"$u_x(x={assets.X[assets.MID_X]:.1f},y,t)$",
    )
    axis_labels = (r"$x$", r"$y$")
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
            ax.set_ylim(0.0, 6.2)
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
            display_label(model),
            ha="left",
            va="center",
            fontsize=14.8,
            fontweight="bold",
            linespacing=1.0,
            multialignment="left",
        )
        fig.text(
            0.018,
            row_box.y0 + 0.42 * row_box.height,
            f"error: {error:.3f}\\%\nspeedup: {speedup:.1f}x",
            ha="left",
            va="center",
            fontsize=13.2,
            linespacing=1.18,
        )
        fig.text(
            0.018,
            row_box.y0 + 0.15 * row_box.height,
            formula_label(model),
            ha="left",
            va="center",
            fontsize=11.6,
            linespacing=1.08,
        )

    legend_handles = [
        Line2D([0], [0], label=legend_label(label), **styles[label])
        for label in ("HDM", *comparisons)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=5,
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
        "PROM-ANN Case 1",
        "PROM-ANN Case 1",
        "PROM-ANN Case 2",
        "PROM-ANN Case 2",
        "PROM-ANN Case 3",
        "PROM-ANN Case 3",
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
