#!/usr/bin/env python3
"""Generate PROM-vs-HDM comparison figures with consistent model colors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MU1 = 4.56
MU2 = 0.019
DOMAIN_MIN = 0.0
DOMAIN_MAX = 100.0
SNAP_INDICES = [0, 100, 200, 300, 400, 500]


def _infer_centers(full_state_size: int) -> tuple[np.ndarray, np.ndarray]:
    if full_state_size % 2 != 0:
        raise ValueError(f"Expected even full state size, got {full_state_size}.")

    nxy = full_state_size // 2
    n = int(round(np.sqrt(nxy)))
    if n * n != nxy:
        raise ValueError(
            f"Unable to infer square grid from state size {full_state_size} (nxy={nxy})."
        )

    grid = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n + 1)
    centers = 0.5 * (grid[1:] + grid[:-1])
    return centers, centers.copy()


def _plot_comparison(
    hdm_snaps: np.ndarray,
    rom_snaps: np.ndarray,
    model_label: str,
    model_color: str,
    out_path: Path,
) -> None:
    x, y = _infer_centers(hdm_snaps.shape[0])
    nx = x.size
    ny = y.size
    mid_x = nx // 2
    mid_y = ny // 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    first_hdm = True
    first_rom = True
    for ind in SNAP_INDICES:
        hdm_u = hdm_snaps[: nx * ny, ind].reshape(ny, nx)
        rom_u = rom_snaps[: nx * ny, ind].reshape(ny, nx)

        ax1.plot(
            x,
            hdm_u[mid_y, :],
            color="black",
            linewidth=2.8,
            linestyle="solid",
            label="HDM" if first_hdm else None,
            zorder=3,
        )
        ax2.plot(
            y,
            hdm_u[:, mid_x],
            color="black",
            linewidth=2.8,
            linestyle="solid",
            label="HDM" if first_hdm else None,
            zorder=3,
        )

        ax1.plot(
            x,
            rom_u[mid_y, :],
            color=model_color,
            linewidth=1.8,
            linestyle="solid",
            label=model_label if first_rom else None,
            zorder=4,
        )
        ax2.plot(
            y,
            rom_u[:, mid_x],
            color=model_color,
            linewidth=1.8,
            linestyle="solid",
            label=model_label if first_rom else None,
            zorder=4,
        )

        first_hdm = False
        first_rom = False

    ax1.set_xlabel(r"$x$")
    ax2.set_xlabel(r"$y$")
    ax1.set_ylabel(rf"$u_x(x, y={y[mid_y]:0.1f})$")
    ax2.set_ylabel(rf"$u_x(x={x[mid_x]:0.1f}, y)$")
    ax1.grid(True, alpha=0.35, linewidth=0.6)
    ax2.grid(True, alpha=0.35, linewidth=0.6)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    fig.suptitle(rf"$\mu_1 = {MU1:.2f},\, \mu_2 = {MU2:.3f}$", y=0.98)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent
    out_dir = base / "Figures"

    hdm_path = base / f"hdm_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy"
    hdm_snaps = np.asarray(np.load(hdm_path), dtype=np.float64)

    models = [
        {
            "label": "PROM",
            "color": "#B8860B",  # dark yellow
            "snap": base / f"prom_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "prom_vs_hdm.png",
        },
        {
            "label": "Local PROM",
            "color": "#B8860B",  # dark yellow
            "snap": base / f"local_prom_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "local_prom_vs_hdm.png",
        },
        {
            "label": "QPROM",
            "color": "#1f77b4",  # blue
            "snap": base / f"qprom_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "qprom_vs_hdm.png",
        },
        {
            "label": "Local QPROM",
            "color": "#1f77b4",  # blue
            "snap": base / f"local_qprom_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "local_qprom_vs_hdm.png",
        },
        {
            "label": "PROM-GPR",
            "color": "#228B22",  # green
            "snap": base / f"prom_gpr_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "prom_gpr_vs_hdm.png",
        },
        {
            "label": "Local PROM-GPR",
            "color": "#228B22",  # green
            "snap": base / f"local_prom_gpr_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "local_prom_gpr_vs_hdm.png",
        },
        {
            "label": "PROM-POD-DL",
            "color": "#d62728",  # red
            "snap": base / f"prom_dl_snaps_mu1_{MU1:.2f}_mu2_{MU2:.3f}.npy",
            "out": out_dir / "prom_dl_vs_hdm.png",
        },
    ]

    for item in models:
        rom_snaps = np.asarray(np.load(item["snap"]), dtype=np.float64)
        if rom_snaps.shape != hdm_snaps.shape:
            raise ValueError(
                f"Snapshot shape mismatch for {item['label']}: "
                f"{rom_snaps.shape} vs HDM {hdm_snaps.shape}."
            )

        _plot_comparison(
            hdm_snaps=hdm_snaps,
            rom_snaps=rom_snaps,
            model_label=item["label"],
            model_color=item["color"],
            out_path=item["out"],
        )
        print(f"Saved: {item['out']}")


if __name__ == "__main__":
    main()
