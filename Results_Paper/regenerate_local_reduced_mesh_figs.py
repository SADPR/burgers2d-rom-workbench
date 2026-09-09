#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Regenerate the local reduced-mesh (ECSW) figures for main.tex.

The reduced-mesh plots are just `plt.spy` of the saved ECSW weight vector,
so they can be rebuilt from the `ecsw_weights.npy` that each sweep candidate
already stores -- no online solve has to be repeated. This matters because
the canonical copies in Results/ are overwritten by every run, which is how
mismatched (figure, error) pairs crept into the previous version of the
table.

Each entry below names the exact candidate folder whose configuration is
reported in the main results table, so the figure and the numbers cannot
drift apart. Run from the repository root.
"""

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# label, weights file (relative to repo root), output png name
FIGURES = [
    (
        "Local HPROM",
        "LocalPODSweep/nc3_postfix/tol1.0e-03/local_hprom_ecsw_weights.npy",
        "local_hprom_reduced_mesh.png",
    ),
    (
        "Local HQPROM",
        "LocalQuadraticSweep/nc3_alphahigh/z1.2_a1e06/local_hqprom_ecsw_weights.npy",
        "local_hqprom_reduced_mesh.png",
    ),
    (
        "Local HPROM-GPR",
        "LocalPOD-GPRSweep/nc3_check/nprimary9/ecsw_weights.npy",
        "local_hprom_gpr_reduced_mesh.png",
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(REPO, "Results_Paper"),
        help="Where the PNGs are written (default: Results_Paper/).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--num-cells",
        type=int,
        default=None,
        help="Cells per direction. Default: inferred as sqrt of the weight count.",
    )
    args = ap.parse_args()

    for label, rel_weights, out_name in FIGURES:
        path = os.path.join(REPO, rel_weights)
        if not os.path.exists(path):
            print("  MISSING  %-18s %s" % (label, rel_weights))
            continue

        weights = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)

        # The ECSW weight vector has one entry per mesh cell, so a square mesh
        # is recovered from its length. This keeps the figure script free of
        # the solver's imports (scipy et al.).
        if args.num_cells:
            nx = ny = int(args.num_cells)
        else:
            side = int(round(math.sqrt(weights.size)))
            if side * side != weights.size:
                print("  NOT SQUARE %-15s %d weights; pass --num-cells"
                      % (label, weights.size))
                continue
            nx = ny = side
        if weights.size != nx * ny:
            print("  SIZE MISMATCH %-14s got %d, expected %d"
                  % (label, weights.size, nx * ny))
            continue

        n_e = int(np.count_nonzero(weights > 0.0))

        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((ny, nx)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("%s Reduced Mesh (ECSW), $N_c=3$" % label)
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, out_name)
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close()

        print("  OK       %-18s N_e=%-6d -> %s" % (label, n_e, out_name))
        print("           source: %s" % rel_weights)

    print("\nUse the printed N_e values for the captions in main.tex.")


if __name__ == "__main__":
    main()
