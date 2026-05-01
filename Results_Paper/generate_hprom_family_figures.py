#!/usr/bin/env python3
"""Generate HPROM-family vs HDM figures for all paper test points.

Outputs are written to Results_Paper/Figures with consistent colors.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POINTS = [(4.56, 0.019), (4.75, 0.020), (5.19, 0.026)]
SNAP_INDICES = [0, 100, 200, 300, 400, 500]
DOMAIN_MIN = 0.0
DOMAIN_MAX = 100.0


def infer_centers(full_state_size: int) -> tuple[np.ndarray, np.ndarray]:
    if full_state_size % 2 != 0:
        raise ValueError(f"Expected even state size, got {full_state_size}.")
    nxy = full_state_size // 2
    n = int(round(np.sqrt(nxy)))
    if n * n != nxy:
        raise ValueError(f"Cannot infer square grid from size {full_state_size}.")
    grid = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n + 1)
    centers = 0.5 * (grid[1:] + grid[:-1])
    return centers, centers.copy()


def plot_comparison(
    hdm_snaps: np.ndarray,
    rom_snaps: np.ndarray,
    model_label: str,
    model_color: str,
    mu1: float,
    mu2: float,
    out_path: Path,
) -> None:
    x, y = infer_centers(hdm_snaps.shape[0])
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
            x, hdm_u[mid_y, :],
            color="black", linewidth=2.8, linestyle="solid",
            label="HDM" if first_hdm else None, zorder=3,
        )
        ax2.plot(
            y, hdm_u[:, mid_x],
            color="black", linewidth=2.8, linestyle="solid",
            label="HDM" if first_hdm else None, zorder=3,
        )

        ax1.plot(
            x, rom_u[mid_y, :],
            color=model_color, linewidth=1.8, linestyle="solid",
            label=model_label if first_rom else None, zorder=4,
        )
        ax2.plot(
            y, rom_u[:, mid_x],
            color=model_color, linewidth=1.8, linestyle="solid",
            label=model_label if first_rom else None, zorder=4,
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
    fig.suptitle(rf"$\mu_1 = {mu1:.2f},\, \mu_2 = {mu2:.3f}$", y=0.98)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent
    out_dir = base / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("hprom", "HPROM", "#B8860B"),  # dark yellow
        ("local_hprom", "Local HPROM", "#B8860B"),
        ("hqprom", "HQPROM", "#1f77b4"),  # blue
        ("local_hqprom", "Local HQPROM", "#1f77b4"),
        ("hprom_gpr", "HPROM-GPR", "#228B22"),  # green
        ("local_hprom_gpr", "Local HPROM-GPR", "#228B22"),
    ]

    for mu1, mu2 in POINTS:
        hdm_path = base / f"hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
        hdm_snaps = np.asarray(np.load(hdm_path), dtype=np.float64)

        for key, label, color in models:
            rom_path = base / f"{key}_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
            rom_snaps = np.asarray(np.load(rom_path), dtype=np.float64)
            if rom_snaps.shape != hdm_snaps.shape:
                raise ValueError(
                    f"Shape mismatch for {key} @ ({mu1},{mu2}): "
                    f"{rom_snaps.shape} vs {hdm_snaps.shape}."
                )

            out_path = out_dir / f"{key}_vs_hdm_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png"
            plot_comparison(
                hdm_snaps=hdm_snaps,
                rom_snaps=rom_snaps,
                model_label=label,
                model_color=color,
                mu1=mu1,
                mu2=mu2,
                out_path=out_path,
            )
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

