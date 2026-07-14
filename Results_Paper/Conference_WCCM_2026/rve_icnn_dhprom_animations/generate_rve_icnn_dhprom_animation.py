#!/usr/bin/env python3
"""Create a compact equivalent-stress GIF for the RVE conclusions slide."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


PROJECT_ROOT = Path("/home/kratos/ML_assisted_CLs_clean")
STAGE10_DIR = (
    PROJECT_ROOT
    / "RVE_homogenization_NeoHookean_using_Kratos"
    / "stage_10_hprom_ann_ls_results_mawecm_res_eps_sig_phase1to40_phase2to10_sum990_ann_hrom"
)
ICNN_PREDICTIONS = (
    PROJECT_ROOT
    / "RVE_homogenization_NeoHookean_using_Kratos"
    / "Sebastian_ICKAN_Tests"
    / "stage10_icnn_alltraj_allsamples_ortho_signed_128_128_64_seed11"
    / "stage10_prediction"
    / "predictions.npz"
)

ASSET_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ASSET_DIR / "outputs"
PREVIEW_DIR = ASSET_DIR / "previews"
GIF_PATH = OUTPUT_DIR / "rve_fom_dhprom_ann_icnn_stage10.gif"
PREVIEW_PATH = PREVIEW_DIR / "rve_fom_dhprom_ann_icnn_stage10_final.png"


def relative_l2_error_percent(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return the relative L2 error in percent."""
    return 100.0 * np.linalg.norm(prediction - reference) / np.linalg.norm(reference)


def equivalent_stress(stress: np.ndarray) -> np.ndarray:
    """Return the two-dimensional von Mises equivalent stress."""
    sigma_xx, sigma_yy, sigma_xy = np.asarray(stress, dtype=float).T[:3]
    radicand = (
        sigma_xx * sigma_xx
        - sigma_xx * sigma_yy
        + sigma_yy * sigma_yy
        + 3.0 * sigma_xy * sigma_xy
    )
    return np.sqrt(np.maximum(radicand, 0.0))


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load FOM, D-HPROM--ANN, and ICNN stress histories."""
    fom = np.load(STAGE10_DIR / "fom_stress.npy")
    dhprom_ann = np.load(STAGE10_DIR / "dhprom_ann_stress.npy")
    with np.load(ICNN_PREDICTIONS) as predictions:
        icnn = predictions["stress_predicted"][0]
        icnn_reference = predictions["stress_reference"][0]

    expected_shape = (1151, 3)
    for name, values in {
        "FOM": fom,
        "D-HPROM--ANN": dhprom_ann,
        "ICNN": icnn,
        "ICNN reference": icnn_reference,
    }.items():
        if values.shape != expected_shape:
            raise ValueError(
                "{} has shape {}; expected {}.".format(name, values.shape, expected_shape)
            )

    reference_difference = np.linalg.norm(fom - icnn_reference) / np.linalg.norm(fom)
    if reference_difference > 1.0e-6:
        raise ValueError("The ICNN archive does not use the matching FOM stress path.")

    return fom, dhprom_ann, icnn


def padded_limits(*signals: np.ndarray) -> tuple[float, float]:
    """Return a stable plotting interval with a small visual margin."""
    lower = min(float(np.min(signal)) for signal in signals)
    upper = max(float(np.max(signal)) for signal in signals)
    span = upper - lower
    padding = 0.09 * span if span > 0.0 else max(abs(upper), 1.0) * 0.10
    return lower - padding, upper + padding


def build_animation(
    fom: np.ndarray, dhprom_ann: np.ndarray, icnn: np.ndarray
) -> tuple[plt.Figure, FuncAnimation]:
    """Build a single-panel equivalent-stress animation."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.85,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.55,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "savefig.facecolor": "white",
        }
    )

    sigma_fom = equivalent_stress(fom) * 1.0e-9
    sigma_dhprom = equivalent_stress(dhprom_ann) * 1.0e-9
    sigma_icnn = equivalent_stress(icnn) * 1.0e-9
    dhprom_error = relative_l2_error_percent(sigma_fom, sigma_dhprom)
    icnn_error = relative_l2_error_percent(sigma_fom, sigma_icnn)
    n_steps = sigma_fom.size
    steps = np.arange(n_steps)

    colors = {"fom": "#202020", "dhprom": "#1769aa", "icnn": "#d95f02"}
    fig, axis = plt.subplots(figsize=(5.25, 4.90))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.66, bottom=0.17)
    fig.suptitle(
        r"Equivalent stress $\sigma_{\mathrm{eq}}$",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )

    (line_fom,) = axis.plot(
        [], [], color=colors["fom"], linewidth=2.4, label="FOM", zorder=3
    )
    (line_dhprom,) = axis.plot(
        [],
        [],
        color=colors["dhprom"],
        linewidth=2.0,
        linestyle=(0, (5, 2.4)),
        label="D-HPROM--ANN",
        zorder=4,
    )
    (line_icnn,) = axis.plot(
        [],
        [],
        color=colors["icnn"],
        linewidth=1.9,
        linestyle=(0, (4, 1.5, 1, 1.5)),
        label="ICNN",
        zorder=5,
    )
    marker_fom = axis.plot([], [], "o", color=colors["fom"], markersize=5.0, zorder=6)[0]
    marker_dhprom = axis.plot(
        [], [], "o", color=colors["dhprom"], markersize=4.0, zorder=7
    )[0]
    marker_icnn = axis.plot([], [], "o", color=colors["icnn"], markersize=4.0, zorder=8)[0]

    axis.set_xlim(0, n_steps - 1)
    axis.set_ylim(padded_limits(sigma_fom, sigma_dhprom, sigma_icnn))
    axis.set_xlabel("Loading step")
    axis.set_ylabel(r"$\sigma_{\mathrm{eq}}$ [GPa]")
    axis.set_axisbelow(True)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.legend(
        [line_fom, line_dhprom, line_icnn],
        ["FOM", "D-HPROM--ANN", "ICNN"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.865),
        frameon=True,
        borderpad=0.42,
        handlelength=2.5,
        columnspacing=1.3,
    )
    fig.text(
        0.50,
        0.755,
        r"rel. $\sigma_{\mathrm{eq}}$ error:  D-HPROM--ANN %.2f%%   |   ICNN %.2f%%"
        % (dhprom_error, icnn_error),
        ha="center",
        va="center",
        fontsize=9.4,
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "#f7f7f7",
            "edgecolor": "#b2b2b2",
            "linewidth": 0.8,
        },
    )
    progress = fig.text(
        0.50, 0.055, "", ha="center", va="center", fontsize=9.2, color="#3f3f3f"
    )

    artists: list[object] = [
        line_fom,
        line_dhprom,
        line_icnn,
        marker_fom,
        marker_dhprom,
        marker_icnn,
        progress,
    ]
    frame_steps = np.unique(np.rint(np.linspace(0, n_steps - 1, 42)).astype(int))

    def update(frame_number: int) -> list[object]:
        current = int(frame_steps[frame_number])
        current_steps = steps[: current + 1]
        line_fom.set_data(current_steps, sigma_fom[: current + 1])
        line_dhprom.set_data(current_steps, sigma_dhprom[: current + 1])
        line_icnn.set_data(current_steps, sigma_icnn[: current + 1])
        marker_fom.set_data([current], [sigma_fom[current]])
        marker_dhprom.set_data([current], [sigma_dhprom[current]])
        marker_icnn.set_data([current], [sigma_icnn[current]])
        progress.set_text("step {:d} / {:d}".format(current, n_steps - 1))
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_steps),
        interval=130,
        blit=False,
        repeat=True,
    )
    return fig, animation


def main() -> None:
    """Render the GIF and a static final-frame preview."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fom, dhprom_ann, icnn = load_data()
    fig, animation = build_animation(fom, dhprom_ann, icnn)
    animation.save(GIF_PATH, writer=PillowWriter(fps=8), dpi=130)

    animation._func(animation.save_count - 1)
    fig.savefig(PREVIEW_PATH, dpi=190, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    sigma_fom = equivalent_stress(fom)
    print("Wrote {}".format(GIF_PATH))
    print("Wrote {}".format(PREVIEW_PATH))
    print(
        "Equivalent-stress relative errors: D-HPROM--ANN {:.4f}%, ICNN {:.4f}%".format(
            relative_l2_error_percent(sigma_fom, equivalent_stress(dhprom_ann)),
            relative_l2_error_percent(sigma_fom, equivalent_stress(icnn)),
        )
    )


if __name__ == "__main__":
    main()
