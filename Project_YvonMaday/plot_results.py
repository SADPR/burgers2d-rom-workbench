
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_4_plot_prom_ann_online_cases_250x250.py

Load and plot HDM vs PROM-ANN online trajectories (Case 1/2/3),
compute relative errors, and save figures + a CSV summary.

Tailored for the 250x250 mesh and 4 test points.

Assumptions:
- You run this script from the directory containing the saved .npy outputs, e.g.
    case1_rnm_snaps_mu1_4.56_mu2_0.0190_n10_ntot150.npy
    case2_prom_ann_snaps_mu1_4.560_mu2_0.0190_n10_ntot150.npy
    case3_rnm_snaps_mu1_4.56_mu2_0.0190_n10_ntot150.npy
  and similarly for other test points.
  (Naming differences are handled by robust token search.)

- The project root contains hypernet2D.py and config.py
  and you can import: make_2D_grid, load_or_compute_snaps, plot_snaps

Outputs:
- results_online_250x250/
    case1_hdm_vs_case1_mu1_...png
    case2_hdm_vs_case2_mu1_...png
    case3_hdm_vs_case3_mu1_...png
    summary_errors.csv
"""

import os
import sys
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Import project utilities
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hypernet2D import make_2D_grid, load_or_compute_snaps, plot_snaps  # noqa: E402
from config import DT, NUM_STEPS  # noqa: E402


# -----------------------------
# Plot style (minimal)
# -----------------------------
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"],
})
plt.rc("font", size=13)


# -----------------------------
# User settings: 250x250 mesh, 4 test points
# -----------------------------
MESH = dict(num_cells_x=250, num_cells_y=250, xl=0, xu=100, yl=0, yu=100)

# EDIT THESE 4 TEST POINTS (placeholders based on what you showed you have saved)
# Your screenshot includes: (3.75,0.0350), (4.56,0.0190), (5.19,0.0260), (5.60,0.0315)
TEST_POINTS = [
    (3.75, 0.0350),
    (4.56, 0.0190),
    (5.19, 0.0260),
    (5.60, 0.0315),
]

# ROM configuration used in filenames
PRIMARY_MODES = 10
TOTAL_MODES = 150

# HDM time settings (must match how snapshots were generated)
DT_HDM = DT
NUM_STEPS_HDM = NUM_STEPS

# Where HDM cached snapshots live
SNAP_FOLDER = os.path.join(PROJECT_ROOT, "param_snaps")
os.makedirs(SNAP_FOLDER, exist_ok=True)

# Where to save plots + summary
OUTDIR = os.path.join(THIS_DIR, "results_online_250x250")
os.makedirs(OUTDIR, exist_ok=True)

# Which time indices to plot (like your launcher)
PLOT_EVERY = 100

# Colors per case (match your writing)
CASE_STYLE = {
    "case1": {"color": "red",   "label": r"Case 1 PROM--ANN"},
    "case2": {"color": "blue",  "label": r"Case 2 PROM--ANN"},
    "case3": {"color": "green", "label": r"Case 3 PROM--ANN"},
}

HDM_STYLE = {"color": "black", "label": r"HDM", "lw": 2.5}
ROM_LW = 1.5


# -----------------------------
# Helpers
# -----------------------------
def fro_rel_err_percent(U_true: np.ndarray, U_pred: np.ndarray, eps: float = 1e-30) -> float:
    """100 * ||U_true-U_pred||_F / ||U_true||_F"""
    num = np.linalg.norm(U_true - U_pred)
    den = np.linalg.norm(U_true) + eps
    return float(100.0 * num / den)


def safe_tag(mu1: float, mu2: float) -> str:
    # Keep consistent with your case2 formatter (3 decimals for mu1, 4 for mu2)
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def find_snapshot_file(case: str, mu1: float, mu2: float, n: int, ntot: int, search_dir: str) -> str:
    """
    Find a .npy file for a given case+mu+n+ntot.
    Handles your naming differences:
      case1: case1_rnm_snaps_mu1_4.56_mu2_0.0190_n10_ntot150.npy
      case2: case2_prom_ann_snaps_mu1_4.560_mu2_0.0190_n10_ntot150.npy
      case3: case3_rnm_snaps_mu1_4.56_mu2_0.0190_n10_ntot150.npy
    """
    # 1) glob by case + n + ntot
    base_glob = os.path.join(search_dir, f"{case}*n{n}_ntot{ntot}*.npy")
    candidates = sorted(glob.glob(base_glob))

    # 2) filter by mu tokens in likely precisions
    mu1_tokens = [f"mu1_{mu1:.2f}", f"mu1_{mu1:.3f}"]
    mu2_tokens = [f"mu2_{mu2:.4f}", f"mu2_{mu2:.3f}"]

    good = []
    for c in candidates:
        bn = os.path.basename(c)
        if any(t in bn for t in mu1_tokens) and any(t in bn for t in mu2_tokens):
            good.append(c)

    if len(good) == 0:
        raise FileNotFoundError(
            f"[{case}] Could not find snapshot .npy for mu=({mu1},{mu2}) "
            f"with n={n}, ntot={ntot} in {search_dir}.\n"
            f"Tried glob: {base_glob}\n"
            f"Candidates found: {len(candidates)}\n"
            f"Tip: run `ls {search_dir}` and verify filenames contain mu1/mu2 tokens."
        )

    # Deterministic pick
    return good[0]


def build_grid_and_ic(mesh: dict):
    nx, ny = mesh["num_cells_x"], mesh["num_cells_y"]
    grid_x, grid_y = make_2D_grid(mesh["xl"], mesh["xu"], mesh["yl"], mesh["yu"], nx, ny)

    u0 = np.ones((ny, nx), dtype=np.float64)
    v0 = np.ones((ny, nx), dtype=np.float64)
    w0 = np.concatenate([u0.ravel(), v0.ravel()])  # length 2*nx*ny

    return grid_x, grid_y, w0


def plot_hdm_vs_rom(grid_x, grid_y, U_hdm, U_rom, case: str, mu1: float, mu2: float, rel_err: float, outdir: str):
    steps = list(range(0, U_hdm.shape[1], PLOT_EVERY))
    if (U_hdm.shape[1] - 1) not in steps:
        steps.append(U_hdm.shape[1] - 1)

    # plot_snaps returns fig, ax1, ax2 when called without fig_ax
    fig, ax1, ax2 = plot_snaps(grid_x, grid_y, U_hdm, steps, label=HDM_STYLE["label"])
    plot_snaps(
        grid_x, grid_y, U_rom, steps,
        label=CASE_STYLE[case]["label"],
        fig_ax=(fig, ax1, ax2),
        color=CASE_STYLE[case]["color"],
        linewidth=ROM_LW,
    )

    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    fname = f"{case}_hdm_vs_{case}_{safe_tag(mu1, mu2)}_n{PRIMARY_MODES}_ntot{TOTAL_MODES}.png"
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=200)
    plt.close(fig)
    return fpath


def main():
    print(f"[info] OUTDIR = {OUTDIR}")
    print(f"[info] SNAP_FOLDER = {SNAP_FOLDER}")
    print(f"[info] mesh = {MESH['num_cells_x']}x{MESH['num_cells_y']}, dt={DT_HDM}, steps={NUM_STEPS_HDM}")
    print(f"[info] test points = {TEST_POINTS}")

    grid_x, grid_y, w0 = build_grid_and_ic(MESH)

    # Summary rows
    rows = []
    header = ["mesh", "mu1", "mu2", "case", "n", "ntot", "rom_file", "rel_err_percent", "plot_file"]

    # Directory to search ROM snapshot files: default is THIS_DIR
    search_dir = THIS_DIR

    for (mu1, mu2) in TEST_POINTS:
        mu = [float(mu1), float(mu2)]

        # Load HDM (will cache under SNAP_FOLDER)
        U_hdm = load_or_compute_snaps(
            mu, grid_x, grid_y, w0,
            DT_HDM, NUM_STEPS_HDM,
            snap_folder=SNAP_FOLDER,
        )
        if U_hdm.ndim != 2:
            raise ValueError(f"HDM snaps must be 2D (N,T), got shape {U_hdm.shape}")

        for case in ["case1", "case2", "case3"]:
            rom_path = find_snapshot_file(case, mu1, mu2, PRIMARY_MODES, TOTAL_MODES, search_dir)
            U_rom = np.load(rom_path)

            if U_rom.shape != U_hdm.shape:
                raise ValueError(
                    f"[{case}] Shape mismatch for mu=({mu1},{mu2}). "
                    f"HDM {U_hdm.shape} vs ROM {U_rom.shape}. "
                    f"File: {rom_path}"
                )

            rel_err = fro_rel_err_percent(U_hdm, U_rom)
            plot_path = plot_hdm_vs_rom(grid_x, grid_y, U_hdm, U_rom, case, mu1, mu2, rel_err, OUTDIR)

            print(f"[{case}] mu=({mu1:.2f},{mu2:.4f}) rel_err={rel_err:.3f}% | ROM={os.path.basename(rom_path)}")

            rows.append([
                f"{MESH['num_cells_x']}x{MESH['num_cells_y']}",
                f"{mu1:.6g}",
                f"{mu2:.6g}",
                case,
                str(PRIMARY_MODES),
                str(TOTAL_MODES),
                os.path.basename(rom_path),
                f"{rel_err:.8f}",
                os.path.basename(plot_path),
            ])

    # Save CSV summary
    csv_path = os.path.join(OUTDIR, "summary_errors.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    print(f"\nSaved summary: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
