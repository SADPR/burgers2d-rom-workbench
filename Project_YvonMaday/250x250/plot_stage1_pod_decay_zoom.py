#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot a zoomed Stage-1 POD residual-energy decay curve (first N modes)."""

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import plot_singular_value_decay


def main():
    sigma_path = os.path.join(THIS_DIR, "Results", "Stage1", "sigma.npy")
    out_path = os.path.join(THIS_DIR, "Figures", "stage1_pod_singular_value_decay_first50.png")

    if not os.path.exists(sigma_path):
        raise FileNotFoundError(f"Missing sigma file: {sigma_path}")

    sigma = np.asarray(np.load(sigma_path, allow_pickle=False), dtype=np.float64).reshape(-1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_singular_value_decay(
        sigma,
        out_path=out_path,
        max_modes=50,
        label="POD Stage 1 (first 50 modes)",
        title="POD residual energy decay (first 50 modes)",
        use_latex=True,
    )

    print(f"[OK] Saved zoomed POD decay figure: {out_path}")


if __name__ == "__main__":
    main()
