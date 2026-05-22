#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Write standalone LaTeX report for offline full-151 ANN vs GPR comparison."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


def _fmt(x: float) -> str:
    ax = abs(float(x))
    if ax >= 1e-2:
        return f"{x:.3f}"
    return f"{x:.2e}"


def _tex_escape(text: str) -> str:
    out = str(text)
    out = out.replace("\\", r"\textbackslash{}")
    out = out.replace("_", r"\_")
    out = out.replace("%", r"\%")
    out = out.replace("&", r"\&")
    out = out.replace("#", r"\#")
    return out


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _sort_points(rows: List[Dict[str, str]]) -> List[Tuple[float, float]]:
    pts = sorted({(round(float(r["mu1"]), 3), round(float(r["mu2"]), 4)) for r in rows})
    return pts


def _sort_models(rows: List[Dict[str, str]]) -> List[str]:
    models = sorted({str(r["model_label"]) for r in rows})
    return models


def _build_tables(rows: List[Dict[str, str]]) -> Tuple[str, str]:
    points = _sort_points(rows)
    models = _sort_models(rows)

    by_key = {}
    for r in rows:
        key = (str(r["model_label"]), round(float(r["mu1"]), 3), round(float(r["mu2"]), 4))
        by_key[key] = r

    point_lines = []
    for mu1, mu2 in points:
        for model in models:
            r = by_key.get((model, mu1, mu2))
            if r is None:
                point_lines.append(f"{_tex_escape(model)} & ({mu1:.3f},{mu2:.4f}) & -- & -- & -- & -- \\\\")
            else:
                point_lines.append(
                    f"{_tex_escape(model)} & ({mu1:.3f},{mu2:.4f}) & "
                    f"{_fmt(float(r['rel_frob_percent']))} & "
                    f"{_fmt(float(r['mean_coeff_rel_percent']))} & "
                    f"{_fmt(float(r['p95_coeff_rel_percent']))} & "
                    f"{_fmt(float(r['max_coeff_rel_percent']))} \\\\"
                )
        point_lines.append(r"\hline")
    point_table = "\n".join(point_lines)

    by_model = defaultdict(list)
    for r in rows:
        by_model[str(r["model_label"])].append(r)

    avg_lines = []
    for model in models:
        rs = by_model[model]
        n = float(len(rs))
        ef = sum(float(x["rel_frob_percent"]) for x in rs) / n
        emean = sum(float(x["mean_coeff_rel_percent"]) for x in rs) / n
        ep95 = sum(float(x["p95_coeff_rel_percent"]) for x in rs) / n
        emax = sum(float(x["max_coeff_rel_percent"]) for x in rs) / n
        avg_lines.append(
            f"{_tex_escape(model)} & {_fmt(ef)} & {_fmt(emean)} & {_fmt(ep95)} & {_fmt(emax)} \\\\"
        )
    avg_table = "\n".join(avg_lines)
    return point_table, avg_table


def _mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def _build_figure_blocks(points: List[Tuple[float, float]], figure_subdir: str, figure_suffix: str) -> str:
    blocks: List[str] = []
    for i, (mu1, mu2) in enumerate(points, start=1):
        rel = f"Figures/{figure_subdir}/{_mu_tag(mu1, mu2)}_coeff_abs_rel_vs_global_index_{figure_suffix}.png"
        blocks.append(
            rf"""\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\textwidth]{{{rel}}}
\caption{{Absolute and relative coefficient errors vs global coefficient index at $\mu^{{({i})}}=({mu1:.3f},{mu2:.4f})$.}}
\end{{figure}}"""
        )
    return "\n\n".join(blocks)


def _write_tex(
    tex_path: Path,
    point_table: str,
    avg_table: str,
    figure_blocks: str,
) -> None:
    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}

\title{Offline Coefficient-Space Comparison\\Full Data-Driven Maps (ANN151 vs GPR151)}
\author{Sebastian Ares de Parga Regalado}
\date{\today}

\begin{document}
\maketitle

\section*{Setup}
The two models compared here are full data-driven maps
\[
(\mu_1,\mu_2,t)\mapsto q_N(t;\mu)\in\mathbb{R}^{151},
\]
trained on Stage-2 PROM coefficients.

Reference trajectories are linear PROM coefficients $q_N^{lin}(t;\mu)$ at three verification points.
No online nonlinear ROM solve is used in this diagnostic.

\section*{Error definitions}
For each global coefficient $i=1,\dots,151$:
\[
E_i^{abs}(\mu)=\left\|q_i^{lin}-q_i^{pred}\right\|_2,\qquad
E_i^{rel}(\mu)=100\frac{\left\|q_i^{lin}-q_i^{pred}\right\|_2}{\left\|q_i^{lin}\right\|_2+\varepsilon}.
\]
Global trajectory error:
\[
E_F(\mu)=100\frac{\left\|Q^{lin}-Q^{pred}\right\|_F}{\left\|Q^{lin}\right\|_F+\varepsilon}.
\]

\section*{Pointwise summary (\%)}
\begin{table}[H]
\centering
\caption{Offline coefficient errors per point.}
\begin{tabular}{lccccc}
\toprule
Model & $\mu$ & $E_F$ & mean$(E_i^{rel})$ & p95$(E_i^{rel})$ & max$(E_i^{rel})$ \\
\midrule
__POINT_TABLE__
\bottomrule
\end{tabular}
\end{table}

\section*{Average over the three points (\%)}
\begin{table}[H]
\centering
\caption{Average offline coefficient errors over the three points.}
\begin{tabular}{lcccc}
\toprule
Model & mean $E_F$ & mean(mean$(E_i^{rel})$) & mean(p95$(E_i^{rel})$) & mean(max$(E_i^{rel})$) \\
\midrule
__AVG_TABLE__
\bottomrule
\end{tabular}
\end{table}

\section*{Per-coefficient curves}
__FIGURE_BLOCKS__

\end{document}
"""
    tex = tex.replace("__POINT_TABLE__", point_table)
    tex = tex.replace("__AVG_TABLE__", avg_table)
    tex = tex.replace("__FIGURE_BLOCKS__", figure_blocks)
    tex_path.write_text(tex, encoding="utf-8")


def _compile(tex_path: Path) -> None:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    subprocess.run(cmd, cwd=str(tex_path.parent), check=True)
    subprocess.run(cmd, cwd=str(tex_path.parent), check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write standalone report for ANN151 vs GPR151 offline comparison.")
    parser.add_argument("--maday-tag", type=str, default="maday_clean_try04")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to summary CSV from analyze_case2_ann_gpr_coeff_offline_maday.py",
    )
    parser.add_argument("--figure-subdir", type=str, default="data_driven151_coeff_offline")
    parser.add_argument("--figure-suffix", type=str, default="dd_ann_gpr_151")
    parser.add_argument("--tex-name", type=str, default="report_data_driven151_offline_coeff.tex")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    exp_dir = Path(paths.exp_dir).resolve()
    if args.summary_csv:
        summary_csv = Path(args.summary_csv).expanduser().resolve()
    else:
        summary_csv = exp_dir / "Figures" / args.figure_subdir / "data_driven151_coeff_offline_summary.csv"

    rows = _load_rows(summary_csv)
    point_table, avg_table = _build_tables(rows)
    points = _sort_points(rows)
    figure_blocks = _build_figure_blocks(points, args.figure_subdir, args.figure_suffix)

    tex_path = exp_dir / args.tex_name
    _write_tex(tex_path, point_table=point_table, avg_table=avg_table, figure_blocks=figure_blocks)
    print(f"[dd151-report] wrote tex: {tex_path}")

    if args.compile:
        _compile(tex_path)
        print(f"[dd151-report] compiled pdf: {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

