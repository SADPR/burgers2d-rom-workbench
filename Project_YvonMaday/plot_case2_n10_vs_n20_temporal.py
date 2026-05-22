#!/usr/bin/env python3
"""
Build a side-by-side temporal comparison figure for Case 2:
- n=10 vs n=20
- baseline and enriched
- verification and two test points

It reuses existing *_hdm_vs_rom.png outputs (no reruns).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
BASE_RUNS = THIS_DIR / "Results" / "Runs" / "Case2"
ENR_RUNS = THIS_DIR / "Results_Enrichment" / "Runs" / "Case2"
FIG_DIR = THIS_DIR / "Figures"
OUT_PATH = FIG_DIR / "case2_n10_vs_n20_temporal_comparison.png"


POINTS = [
    ("4.875", "0.0225", r"$\mu^{(v)}=(4.875,0.0225)$"),
    ("4.560", "0.0190", r"$\mu^{(1)}=(4.56,0.019)$"),
    ("5.190", "0.0260", r"$\mu^{(2)}=(5.19,0.026)$"),
]


def _case2_png(root: Path, mu1: str, mu2: str, n: int, enriched: bool) -> Path:
    if enriched:
        name = f"case2_prom_ann_enriched_mu1_{mu1}_mu2_{mu2}_n{n}_ntot151_hdm_vs_rom.png"
    else:
        name = f"case2_prom_ann_mu1_{mu1}_mu2_{mu2}_n{n}_ntot151_hdm_vs_rom.png"
    return root / name


def main() -> None:
    rows = [
        ("Baseline, n=10", BASE_RUNS, 10, False),
        ("Baseline, n=20", BASE_RUNS, 20, False),
        ("Enriched, n=10", ENR_RUNS, 10, True),
        ("Enriched, n=20", ENR_RUNS, 20, True),
    ]

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle("Case 2 Temporal Solution Comparison: n=10 vs n=20", fontsize=20, y=0.995)

    for j, (_, _, col_title) in enumerate(POINTS):
        axes[0, j].set_title(col_title, fontsize=14, pad=10)

    for i, (row_label, root, nprim, enriched) in enumerate(rows):
        for j, (mu1, mu2, _) in enumerate(POINTS):
            ax = axes[i, j]
            p = _case2_png(root, mu1, mu2, nprim, enriched)
            if not p.exists():
                ax.text(0.5, 0.5, f"Missing file:\n{p.name}", ha="center", va="center", fontsize=10)
                ax.set_axis_off()
                continue
            img = plt.imread(p)
            ax.imshow(img)
            ax.set_axis_off()

        axes[i, 0].text(
            -0.03,
            0.5,
            row_label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0.02, 0.01, 1.0, 0.985])
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

