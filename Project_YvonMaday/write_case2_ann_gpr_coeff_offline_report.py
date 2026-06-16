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
    if "gpr" in low:
        family = "GPR"
    elif "ann" in low and ("old" in low or "baseline" in low or "data-driven" in low or "data driven" in low):
        family = "Old DD-ANN"
    elif "ann" in low:
        family = "ANN"
    else:
        family = txt

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


def _read_kv_summary(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _build_tables(rows: List[Dict]) -> Tuple[str, str]:
    by_key = {}
    for r in rows:
        key = (r["model_label"], round(float(r["mu1"]), 3), round(float(r["mu2"]), 4))
        by_key[key] = r

    present_labels = sorted({str(rr.get("model_label")) for rr in rows})
    model_info = []
    for m in present_labels:
        fam, ns = _parse_model(m)
        fam_rank = 0 if fam == "ANN" else (1 if fam == "Old DD-ANN" else (2 if fam == "GPR" else 3))
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


def _ann_row(ns: int, kv: Dict[str, str], fallback_arch: str = "--") -> str:
    hidden = kv.get("hidden_dims", fallback_arch)
    act = kv.get("activation", "elu" if fallback_arch != "--" else "--")
    drop = kv.get("dropout", "0.0" if fallback_arch != "--" else "--")
    bs = kv.get("batch_size", "128" if fallback_arch != "--" else "--")
    lr = kv.get("lr", "1e-3" if fallback_arch != "--" else "--")
    wd = kv.get("weight_decay", "1e-6" if fallback_arch != "--" else "--")
    ep = kv.get("epochs_ran", "--")
    best = kv.get("best_val_mse", "--")
    return (
        f"{ns} & "
        f"\\texttt{{{_tex_escape(hidden)}}} & "
        f"{_tex_escape(act)} & "
        f"{_tex_escape(drop)} & "
        f"{_tex_escape(bs)} & "
        f"\\texttt{{{_tex_escape(lr)}}} & "
        f"\\texttt{{{_tex_escape(wd)}}} & "
        f"{_tex_escape(ep)} & "
        f"\\texttt{{{_tex_escape(best)}}} \\\\"
    )


def _gpr_row(ns: int, kv: Dict[str, str]) -> str:
    k0 = kv.get("kernel_init", "--")
    kf = kv.get("kernel_learned", "--")
    alpha = kv.get("alpha", "--")
    use_wk = kv.get("use_white_kernel", "--")
    wk_bounds = kv.get("white_noise_bounds", "--")
    val_rel = kv.get("val_rel_frob_percent", "--")
    return (
        f"{ns} & "
        f"\\texttt{{{_tex_escape(k0)}}} & "
        f"\\texttt{{{_tex_escape(kf)}}} & "
        f"\\texttt{{{_tex_escape(alpha)}}} & "
        f"{_tex_escape(use_wk)} & "
        f"\\texttt{{{_tex_escape(wk_bounds)}}} & "
        f"\\texttt{{{_tex_escape(val_rel)}}} \\\\"
    )


def _build_model_specs(paths, rows: List[Dict]) -> str:
    stage3 = Path(paths.stage3)
    present = {_parse_model(str(r["model_label"])) for r in rows}

    ann131 = _read_kv_summary(stage3 / "case2_ann_mu_t_ns131_summary.txt")
    ann141 = _read_kv_summary(stage3 / "case2_ann_mu_t_ns141_summary.txt")
    ann151 = _read_kv_summary(stage3 / "rom_data_driven_training_summary.txt")
    gpr131 = _read_kv_summary(stage3 / "case2_gpr_mu_t_ns131_summary.txt")
    gpr141 = _read_kv_summary(stage3 / "case2_gpr_mu_t_ns141_summary.txt")
    gpr151 = _read_kv_summary(stage3 / "rom_data_driven_gpr_mu_t_ntot151_summary.txt")

    ann_kv = {131: ann131, 141: ann141, 151: ann151}
    gpr_kv = {131: gpr131, 141: gpr141, 151: gpr151}

    ann_rows = "\n".join(
        _ann_row(ns, ann_kv[ns], fallback_arch="(32, 64, 128, 256, 256)" if ns == 151 else "--")
        for fam, ns in sorted(present, key=lambda x: x[1])
        if fam == "ANN"
    )
    gpr_rows = "\n".join(
        _gpr_row(ns, gpr_kv[ns])
        for fam, ns in sorted(present, key=lambda x: x[1])
        if fam == "GPR"
    )

    ann_table = ""
    if ann_rows:
        ann_table = rf"""
\begin{{table}}[H]
\centering
\caption{{ANN training setup (from Stage-3 summaries).}}
\scriptsize
\begin{{tabular}}{{ccccccccc}}
\toprule
$n_s$ & hidden dims & act. & drop. & batch & lr & weight decay & epochs ran & best val MSE \\
\midrule
{ann_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    gpr_table = ""
    if gpr_rows:
        gpr_table = rf"""
\begin{{table}}[H]
\centering
\caption{{GPR kernels after optimization, including WhiteKernel terms.}}
\scriptsize
\begin{{tabular}}{{cp{{0.20\textwidth}}p{{0.25\textwidth}}cccc}}
\toprule
$n_s$ & kernel init & kernel learned & $\alpha$ & WhiteK & WhiteK bounds & val rel. Frobenius (\%) \\
\midrule
{gpr_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    return rf"""
\section*{{Model specifications used in this report}}
All maps use inputs $(\mu_1,\mu_2,t)$. Closure maps output $q_s$; full data-driven maps output the complete $q_N$.
For ANN full-coordinate maps, the hidden architecture is the same as in the training script core MLP:
\[
(32,64,128,256,256)\ \text{{with ELU activations}}.
\]
{ann_table}
{gpr_table}
"""


def _build_state_tables(state_rows: List[Dict]) -> Tuple[str, str]:
    if not state_rows:
        return "--", "--"

    by_key = {}
    for r in state_rows:
        key = (r["model_label"], round(float(r["mu1"]), 3), round(float(r["mu2"]), 4))
        by_key[key] = r

    present_labels = sorted({str(rr.get("model_label")) for rr in state_rows})
    model_info = []
    for m in present_labels:
        fam, ns = _parse_model(m)
        fam_rank = 0 if fam == "ANN" else (1 if fam == "Old DD-ANN" else (2 if fam == "GPR" else 3))
        model_info.append((fam_rank, ns, fam, m))
    model_info.sort()

    table_point_lines = []
    for mu1, mu2 in POINT_ORDER:
        table_point_lines.append(rf"\multicolumn{{4}}{{c}}{{\(\mu=({mu1:.3f},{mu2:.4f})\)}} \\")
        table_point_lines.append(r"\midrule")
        for _, _, fam, model in model_info:
            _, ns = _parse_model(model)
            r = by_key.get((model, round(mu1, 3), round(mu2, 4)))
            if r is None:
                table_point_lines.append(rf"{fam} & {ns} & -- & -- \\")
                continue
            table_point_lines.append(
                f"{fam} & {ns} & "
                f"{_fmt(float(r['rel_error_percent_vs_hdm_state']))} & "
                f"{_fmt(float(r['rel_error_percent_vs_linear_state']))} \\\\"
            )
        table_point_lines.append(r"\midrule")
    table_point = "\n".join(table_point_lines)

    by_model = defaultdict(list)
    for r in state_rows:
        by_model[r["model_label"]].append(r)

    table_avg_lines = []
    for _, _, fam, model in model_info:
        _, ns = _parse_model(model)
        rs = by_model.get(model, [])
        if not rs:
            table_avg_lines.append(f"{fam} & {ns} & -- & -- \\\\")
            continue
        n = float(len(rs))
        eh = sum(float(x["rel_error_percent_vs_hdm_state"]) for x in rs) / n
        el = sum(float(x["rel_error_percent_vs_linear_state"]) for x in rs) / n
        table_avg_lines.append(f"{fam} & {ns} & {_fmt(eh)} & {_fmt(el)} \\\\")
    table_avg = "\n".join(table_avg_lines)
    return table_point, table_avg


def _write_tex(
    tex_path: Path,
    table_point: str,
    table_avg: str,
    figure_suffix: str,
    model_specs_tex: str,
    state_table_point: str,
    state_table_avg: str,
    has_gpr: bool,
    imported_model_note: str = "",
) -> None:
    report_title = (
        r"Offline Coefficient-Space Comparison for Case-2 Maps\\ANN vs GPR, $n_s\in\{131,141,151\}$"
        if has_gpr
        else r"Offline Coefficient-Space Comparison for ANN Maps\\$n_s\in\{131,141,151\}$"
    )
    gpr_zoom_section = ""
    if has_gpr:
        gpr_zoom_section = r"""
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
"""

    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{array}
\usepackage{hyperref}

\title{__REPORT_TITLE__}
\author{Sebastian Ares de Parga Regalado}
\date{\today}

\begin{document}
\maketitle

\section*{Objective}
This report compares previously trained maps in pure offline reconstruction mode:
\[
(\mu,t)\mapsto q_s(\mu,t),
\]
without running online nonlinear ROM solves.

The compared models use
\[
n_s\in\{131,141,151\},
\]
using the same linear PROM reference trajectories.

__IMPORTED_MODEL_NOTE__

Reference trajectories are the linear PROM coefficients $q_N^{lin}(t;\mu)$ with $n_{tot}=151$ at
\[
\mu^{(v)}=(4.875,0.0225),\quad
\mu^{(1)}=(4.560,0.0190),\quad
\mu^{(2)}=(5.190,0.0260).
\]

__MODEL_SPECS__

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

__GPR_ZOOM_SECTION__

\section*{Hybrid state-space overlays (HDM comparison)}
Using the same predicted coefficients, hybrid full reduced trajectories were built as
\[
q_N^{hyb}=
\begin{bmatrix}
q_p^{lin}\\ q_s^{pred}
\end{bmatrix},
\]
with $(n_p,n_s)=(20,131)$, $(10,141)$, and $(0,151)$ depending on the model.
Then states were reconstructed by
\[
u^{hyb}(t)=u_{ref}+V_{tot}q_N^{hyb}(t),
\]
and compared directly against HDM snapshots.

\begin{table}[H]
\centering
\caption{Hybrid state errors per point (\%).}
\begin{tabular}{lccc}
\toprule
Model & $n_s$ & err.\ vs HDM (\%) & err.\ vs linear state (\%) \\
\midrule
__STATE_TABLE_POINT__
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Average hybrid state errors over the three points (\%).}
\begin{tabular}{lccc}
\toprule
Model & $n_s$ & mean err.\ vs HDM (\%) & mean err.\ vs linear state (\%) \\
\midrule
__STATE_TABLE_AVG__
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{Figures/case2_ann_gpr_coeff_offline/hybrid_prom_hdm_vs_ann_gpr_models___FIG_SUFFIX__.png}
\caption{Hybrid offline state overlays: HDM, linear PROM reference, and learned-map hybrid reconstructions for $n_s\in\{131,141,151\}$.}
\end{figure}

\section*{Remarks}
This report is strictly an offline map-comparison diagnostic in coefficient space.
It does not include online ROM reconstruction of state trajectories.

\end{document}
"""
    tex = tex.replace("__TABLE_POINT__", table_point).replace("__TABLE_AVG__", table_avg)
    tex = tex.replace("__REPORT_TITLE__", report_title)
    tex = tex.replace("__FIG_SUFFIX__", figure_suffix)
    tex = tex.replace("__MODEL_SPECS__", model_specs_tex)
    tex = tex.replace("__IMPORTED_MODEL_NOTE__", imported_model_note)
    tex = tex.replace("__GPR_ZOOM_SECTION__", gpr_zoom_section)
    tex = tex.replace("__STATE_TABLE_POINT__", state_table_point)
    tex = tex.replace("__STATE_TABLE_AVG__", state_table_avg)
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
        default="ann_gpr_131_141_151",
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
    state_csv = (
        exp_dir
        / "Figures"
        / "case2_ann_gpr_coeff_offline"
        / f"case2_ann_gpr_hybrid_state_summary_{args.figure_suffix}.csv"
    ).resolve()

    rows = _load_summary(summary_csv)
    table_point, table_avg = _build_tables(rows)
    model_specs_tex = _build_model_specs(paths, rows)
    has_gpr = any(_parse_model(str(r["model_label"]))[0] == "GPR" for r in rows)
    has_old_dd = any(_parse_model(str(r["model_label"]))[0] == "Old DD-ANN" for r in rows)
    imported_model_note = (
        "The row labeled Old DD-ANN corresponds to the previously trained data-driven ANN checkpoint "
        "from the original 250x250 study; it is imported here only as a fixed baseline for comparison."
        if has_old_dd
        else ""
    )
    state_rows = _load_summary(state_csv) if state_csv.exists() else []
    state_table_point, state_table_avg = _build_state_tables(state_rows)
    tex_path = (exp_dir / args.tex_name).resolve()
    _write_tex(
        tex_path,
        table_point=table_point,
        table_avg=table_avg,
        figure_suffix=args.figure_suffix,
        model_specs_tex=model_specs_tex,
        state_table_point=state_table_point,
        state_table_avg=state_table_avg,
        has_gpr=has_gpr,
        imported_model_note=imported_model_note,
    )
    print(f"[case2-offline-report] wrote tex: {tex_path}")

    if args.compile:
        _compile_tex(tex_path)
        print(f"[case2-offline-report] compiled pdf: {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
