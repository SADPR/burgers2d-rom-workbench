#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebuild the HGV2 validation-error bar chart for the slide deck.

The archived figure carried the ECSW bookkeeping in its legend ("30 ECSW
cases, row stride 1"), which is a detail the deck now keeps in the appendix,
and it listed the models in a different order from the tables. This script
redraws it from the published numbers, so the figure and the online-performance
table cannot drift apart.

Two constraints worth stating:

* Each model keeps the hue it has in the sibling 3-D and projection figures,
  because colour identifies the model across the whole deck, not its position
  in this one chart. Only the order changes, to match the tables.
* Under protanopia the green (HPROM-RBF) and orange (Local HPROM) of that
  fixed mapping separate by only ~1.5 in OKLab, so colour alone does not
  identify a bar. The sibling figures solve this with distinct markers; here
  the bars carry distinct hatch patterns for the same reason.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COEFFS = ["CL", "CD", "CM"]

# label, relative L2 error per coefficient (%), hue, hatch
MODELS = [
    ("HPROM",                 [0.554, 0.975, 6.620], "tab:blue",   ""),
    ("HPROM-RBF",             [1.335, 1.388, 7.004], "tab:green",  "//"),
    ("Local HPROM",           [0.444, 1.101, 6.242], "tab:orange", ".."),
    ("Local HPROM-RBF",       [2.183, 1.978, 7.117], "tab:red",    "xx"),
]


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Figures", "overall_relative_errors.png")

    x = np.arange(len(COEFFS))
    width = 0.8 / len(MODELS)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (label, errs, color, hatch) in enumerate(MODELS):
        offset = (k - (len(MODELS) - 1) / 2.0) * width
        ax.bar(x + offset, np.asarray(errs) / 100.0, width * 0.92,
               label=label, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(COEFFS)
    ax.set_ylabel("global relative $L^2$ error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("written: %s" % out)


if __name__ == "__main__":
    main()
