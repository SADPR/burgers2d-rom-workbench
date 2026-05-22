#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Write standalone LaTeX report for offline Case-2 ANN/GPR coefficient errors."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


MODEL_ORDER = [
    "ANN (n_s=131)",
    "GPR (n_s=131)",
    "ANN (n_s=141)",
    "GPR (n_s=141)",
    "ANN ns151",
    "GPR ns151",
]

POINT_ORDER = [
    (4.875, 0.0225),
    (4.560, 0.0190),
    (5.190, 0.0260),
]


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


def _parse_model(label: str) -> Tuple[str, int]:
    txt = str(label).strip()
    low = txt.lower()
    family = "GPR" if "gpr" in low else ("ANN" if "ann" in low else txt)

    m = re.search(r"n_s\s*=\s*(\d+)", low)
    if m is None:
        m = re.search(r"ns\s*(\d+)", low)
    if m is None:
        m = re.search(r"(\d+)", low)
    if m is None:
        raise ValueError(f"Could not parse n_s from model label '{label}'.")
    n_s = int(m.group(1))
    return family, n_s


def _load_summary(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _build_tables(rows: List[Dict]) -> Tuple[str, str]:
    by_key = {}
    for r in rows:
        key = (r["model_label"], round(float(r["mu1"]), 3), round(float(r["mu2"]), 4))
        by_key[key] = r

    present_labels = sorted({str(rr.get("model_label")) for rr in rows})
    model_info = []
    for m in present_labels:
        fam, ns = _parse_model(m)
        fam_rank = 0 if fam == "ANN" else (1 if fam == "GPR" else 2)
        model_info.append((fam_rank, ns, fam, m))
    model_info.sort()

    table_point_lines = []
    for mu1, mu2 in POINT_ORDER:
        table_point_lines.append(rf"\multicolumn{{6}}{{c}}{{\(\mu=({mu1:.3f},{mu2:.4f})\)}} \\")
        table_point_lines.append(r"\midrule")
        for _, _, fam, model in model_info:
            _, ns = _parse_model(model)
            r = by_key.get((model, round(mu1, 3), round(mu2, 4)))
            if r is None:
                table_point_lines.append(rf"{fam} & {ns} & -- & -- & -- & -- \\")
                continue
            table_point_lines.append(
                f"{fam} & {ns} & "
                f"{_fmt(float(r['rel_frob_percent']))} & "
                f"{_fmt(float(r['mean_coeff_rel_percent']))} & "
                f"{_fmt(float(r['p95_coeff_rel_percent']))} & "
                f"{_fmt(float(r['max_coeff_rel_percent']))} \\\\"
            )
        table_point_lines.append(r"\midrule")
    table_point = "\n".join(table_point_lines)

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_label"]].append(r)

    table_avg_lines = []
    for _, _, fam, model in model_info:
        _, ns = _parse_model(model)
        rs = by_model.get(model, [])
        if not rs:
            table_avg_lines.append(f"{fam} & {ns} & -- & -- & -- & -- \\\\")
            continue
        n = float(len(rs))
        relf = sum(float(x["rel_frob_percent"]) for x in rs) / n
        mean_coeff = sum(float(x["mean_coeff_rel_percent"]) for x in rs) / n
        p95 = sum(float(x["p95_coeff_rel_percent"]) for x in rs) / n
        mx = sum(float(x["max_coeff_rel_percent"]) for x in rs) / n
        table_avg_lines.append(f"{fam} & {ns} & {_fmt(relf)} & {_fmt(mean_coeff)} & {_fmt(p95)} & {_fmt(mx)} \\\\")
    table_avg = "\n".join(table_avg_lines)
    return table_point, table_avg


def _write_tex(tex_path: Path, table_point: str, table_avg: str, figure_suffix: str) -> None:
    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}

\title{Offline Coefficient-Space Comparison for Case-2 Maps\\ANN vs GPR, $n_s\in\{131,141,151\}$}
\author{Sebastian Ares de Parga Regalado}
\date{\today}

\begin{document}
\maketitle

\section*{Objective}
This report compares six previously trained Case-2 maps in pure offline reconstruction mode:
\[
(\mu,t)\mapsto q_s(\mu,t),
\]
without running online nonlinear ROM solves.

The compared models include ANN and GPR maps with
\[
n_s\in\{131,141,151\},
\]
using the same linear PROM reference trajectories.

Reference trajectories are the linear PROM coefficients $q_N^{lin}(t;\mu)$ with $n_{tot}=151$ at
\[
\mu^{(v)}=(4.875,0.0225),\quad
\mu^{(1)}=(4.560,0.0190),\quad
\mu^{(2)}=(5.190,0.0260).
\]

\section*{Error definitions}
For a model with split $n_{tot}=n_p+n_s$, we compare predicted secondary coefficients against the corresponding tail of the linear reference:
\[
q_s^{ref}(t;\mu)=\big[q_N^{lin}(t;\mu)\big]_{n_p+1:n_{tot}},
\qquad
q_s^{pred}(t;\mu)=\mathcal M_{\theta}(\mu,t).
\]
For each predicted coefficient index $i=1,\dots,n_s$:
\[
E_i^{abs}(\mu)=\left\|q_{s,i}^{ref}-q_{s,i}^{pred}\right\|_2,
\qquad
E_i^{rel}(\mu)=100\,
\frac{\left\|q_{s,i}^{ref}-q_{s,i}^{pred}\right\|_2}
{\left\|q_{s,i}^{ref}\right\|_2+\varepsilon}.
\]
Global secondary-trajectory error:
\[
E_F(\mu)=100\,
\frac{\left\|Q_s^{ref}-Q_s^{pred}\right\|_F}
{\left\|Q_s^{ref}\right\|_F+\varepsilon}.
\]

Since models have different $n_s$, each curve starts at global index $n_p+1$ with $n_p=n_{tot}-n_s$.
Therefore, missing lower global indices for some curves are expected.

\section*{Pointwise summary}
\begin{table}[H]
\centering
\caption{Offline coefficient errors per point (\%).}
\begin{tabular}{lccccc}
\toprule
Model & $n_s$ & $E_F$ & mean$(E_i^{rel})$ & p95$(E_i^{rel})$ & max$(E_i^{rel})$ \\
\midrule
__TABLE_POINT__
\bottomrule
\end{tabular}
\end{table}

\section*{Average over the three points}
\begin{table}[H]
\centering
\caption{Averaged offline coefficient errors over the three points (\%).}
\begin{tabular}{lccccc}
\toprule
Model & $n_s$ & mean $E_F$ & mean(mean$(E_i^{rel})$) & mean(p95$(E_i^{rel})$) & mean(max$(E_i^{rel})$) \\
\midrule
__TABLE_AVG__
\bottomrule
\end{tabular}
\end{table}

\section*{Per-coefficient curves}
\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{Figures/case2_ann_gpr_coeff_offline/mu1_4.875_mu2_0.0225_coeff_abs_rel_vs_global_index___FIG_SUFFIX__.png}
\caption{Absolute and relative coefficient errors vs global coefficient index at $\mu^{(v)}=(4.875,0.0225)$.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{Figures/case2_ann_gpr_coeff_offline/mu1_4.560_mu2_0.0190_coeff_abs_rel_vs_global_index___FIG_SUFFIX__.png}
\caption{Absolute and relative coefficient errors vs global coefficient index at $\mu^{(1)}=(4.560,0.0190)$.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{Figures/case2_ann_gpr_coeff_offline/mu1_5.190_mu2_0.0260_coeff_abs_rel_vs_global_index___FIG_SUFFIX__.png}
\caption{Absolute and relative coefficient errors vs global coefficient index at $\mu^{(2)}=(5.190,0.0260)$.}
\end{figure}

\section*{GPR zoom}
\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{Figures/case2_ann_gpr_coeff_offline/mu1_4.875_mu2_0.0225_gpr_rel_zoom___FIG_SUFFIX__.png}
\caption{Zoomed GPR comparison at $\mu^{(v)}=(4.875,0.0225)$, including pairwise differences.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{Figures/case2_ann_gpr_coeff_offline/mu1_4.560_mu2_0.0190_gpr_rel_zoom___FIG_SUFFIX__.png}
\caption{Zoomed GPR comparison at $\mu^{(1)}=(4.560,0.0190)$, including pairwise differences.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{Figures/case2_ann_gpr_coeff_offline/mu1_5.190_mu2_0.0260_gpr_rel_zoom___FIG_SUFFIX__.png}
\caption{Zoomed GPR comparison at $\mu^{(2)}=(5.190,0.0260)$, including pairwise differences.}
\end{figure}

\section*{Remarks}
This report is strictly an offline map-comparison diagnostic in coefficient space.
It does not include online ROM reconstruction of state trajectories.

\end{document}
"""
    tex = tex.replace("__TABLE_POINT__", table_point).replace("__TABLE_AVG__", table_avg)
    tex = tex.replace("__FIG_SUFFIX__", figure_suffix)
    tex_path.write_text(tex, encoding="utf-8")


def _compile_tex(tex_path: Path) -> None:
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    subprocess.run(cmd, cwd=str(tex_path.parent), check=True)
    subprocess.run(cmd, cwd=str(tex_path.parent), check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write standalone report from offline ANN/GPR coefficient CSV.")
    parser.add_argument("--maday-tag", type=str, default="maday_clean_try04")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional explicit path to case2_ann_gpr_offline_coeff_summary.csv",
    )
    parser.add_argument(
        "--tex-name",
        type=str,
        default="report_case2_ann_gpr_offline_coeff.tex",
    )
    parser.add_argument(
        "--figure-suffix",
        type=str,
        default="ann_gpr_131_141",
    )
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    exp_dir = Path(paths.exp_dir).resolve()
    summary_csv = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else (
            exp_dir
            / "Figures"
            / "case2_ann_gpr_coeff_offline"
            / "case2_ann_gpr_offline_coeff_summary.csv"
        ).resolve()
    )

    rows = _load_summary(summary_csv)
    table_point, table_avg = _build_tables(rows)
    tex_path = (exp_dir / args.tex_name).resolve()
    _write_tex(tex_path, table_point=table_point, table_avg=table_avg, figure_suffix=args.figure_suffix)
    print(f"[case2-offline-report] wrote tex: {tex_path}")

    if args.compile:
        _compile_tex(tex_path)
        print(f"[case2-offline-report] compiled pdf: {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
