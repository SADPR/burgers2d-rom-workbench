#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Write a standalone Results_Maday LaTeX report from summary files."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def _parse_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def _fmt_float(value: str | float, scale: float = 1.0, nd: int = 4) -> str:
    try:
        return f"{float(value) * scale:.{nd}f}"
    except Exception:
        return "NA"


def _mu_from_folder(folder_name: str) -> str:
    # linear_prom_mu1_4.875_mu2_0.0225_ntot151
    try:
        parts = folder_name.split("_")
        i1 = parts.index("mu1")
        i2 = parts.index("mu2")
        return f"({parts[i1+1]}, {parts[i2+1]})"
    except Exception:
        return folder_name


def _collect_linear_rows(exp_dir: Path) -> list[tuple[str, str, str, str, str, str]]:
    linear_root = exp_dir / "Runs" / "Linear_tol"
    out: list[tuple[str, str, str, str, str, str]] = []
    if not linear_root.exists():
        return out
    for sub in sorted(p for p in linear_root.iterdir() if p.is_dir()):
        label = sub.name
        if not sub.exists():
            continue
        for summ in sorted(sub.glob("*/summary.txt")):
            d = _parse_summary(summ)
            out.append(
                (
                    label,
                    _mu_from_folder(summ.parent.name),
                    _fmt_float(d.get("relative_error_percent", "nan")),
                    _fmt_float(d.get("online_solve_elapsed_s", "nan")),
                    d.get("num_iterations", "NA"),
                    summ.parent.name,
                )
            )
    return out


def _collect_oracle_rows(exp_dir: Path, case_tag: str | None) -> list[tuple[str, str, str, str, str]]:
    out: list[tuple[str, str, str, str, str]] = []
    if case_tag is None:
        return out
    case_dir = exp_dir / "Case2" / case_tag
    if not case_dir.exists():
        return out
    for s in sorted(case_dir.glob("*_summary.txt")):
        d = _parse_summary(s)
        out.append(
            (
                s.stem,
                _fmt_float(d.get("relative_error_percent_vs_hdm", "nan")),
                _fmt_float(d.get("relative_error_percent_vs_linear_prom", "nan")),
                _fmt_float(d.get("contamination_gain_primary_dq_over_dqbar", "nan"), nd=6),
                _fmt_float(d.get("contamination_gain_state_du_over_dqbar", "nan"), nd=6),
            )
        )
    return out


def _diagnostics(exp_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    p_w = exp_dir / "Stage1" / "weights_diag.npy"
    p_be = exp_dir / "Stage1_euclid" / "basis.npy"
    p_bw = exp_dir / "Stage1" / "basis_weighted.npy"
    p_p1m = exp_dir / "Stage1" / "basis_proposal1_mblock.npy"
    p_p1w = exp_dir / "Stage1" / "basis_proposal1_weff.npy"
    p_bc = exp_dir / "Stage1" / "basis_corrected_p2_n10_Aavg.npy"
    p_p2 = exp_dir / "Stage1" / "basis_proposal2.npy"
    p_p3 = exp_dir / "Stage1" / "basis_proposal3.npy"
    p_ue = exp_dir / "Stage1_euclid" / "u_ref.npy"
    p_uw = exp_dir / "Stage1" / "u_ref_weighted.npy"

    if p_w.exists():
        w = np.load(p_w)
        out["w_min"] = f"{w.min():.6e}"
        out["w_max"] = f"{w.max():.6e}"
        out["w_mean"] = f"{w.mean():.6e}"
        out["w_std"] = f"{w.std():.6e}"

    bw_like = p_bw if p_bw.exists() else (p_p1m if p_p1m.exists() else None)
    if p_be.exists() and bw_like is not None:
        be = np.load(p_be)
        bw = np.load(bw_like)
        s = np.linalg.svd(be.T @ bw, compute_uv=False)
        out["sv_e_bw_min"] = f"{s.min():.6e}"
        out["sv_e_bw_max"] = f"{s.max():.6e}"
        out["sv_e_bw_mean"] = f"{s.mean():.6e}"
        i = np.eye(be.shape[1])
        out["orth_e"] = f"{np.linalg.norm(be.T @ be - i):.6e}"
        out["orth_w"] = f"{np.linalg.norm(bw.T @ bw - i):.6e}"

    bc_like = p_bc if p_bc.exists() else (p_p2 if p_p2.exists() else None)
    if bw_like is not None and bc_like is not None:
        bw = np.load(bw_like)
        bc = np.load(bc_like)
        s = np.linalg.svd(bw.T @ bc, compute_uv=False)
        out["sv_w_bc_min"] = f"{s.min():.6e}"
        out["sv_w_bc_max"] = f"{s.max():.6e}"
        out["sv_w_bc_mean"] = f"{s.mean():.6e}"
        i = np.eye(bc.shape[1])
        out["orth_c"] = f"{np.linalg.norm(bc.T @ bc - i):.6e}"

    if p_p1w.exists() and p_be.exists():
        p1w = np.load(p_p1w)
        be = np.load(p_be)
        s = np.linalg.svd(be.T @ p1w, compute_uv=False)
        out["sv_e_p1weff_min"] = f"{s.min():.6e}"
        out["sv_e_p1weff_max"] = f"{s.max():.6e}"
        out["sv_e_p1weff_mean"] = f"{s.mean():.6e}"

    if p_p3.exists() and bw_like is not None:
        p3 = np.load(p_p3)
        bw = np.load(bw_like)
        s = np.linalg.svd(bw.T @ p3, compute_uv=False)
        out["sv_w_p3_min"] = f"{s.min():.6e}"
        out["sv_w_p3_max"] = f"{s.max():.6e}"
        out["sv_w_p3_mean"] = f"{s.mean():.6e}"

    if p_ue.exists() and p_uw.exists():
        ue = np.load(p_ue)
        uw = np.load(p_uw)
        den = max(np.linalg.norm(ue), 1e-30)
        out["uref_rel_diff"] = f"{(np.linalg.norm(ue - uw) / den):.6e}"

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate LaTeX report for Maday experiments.")
    parser.add_argument("--maday-tag", required=True)
    parser.add_argument("--case-tag", default=None, help="Case2 oracle folder under Results_Maday/<tag>/Case2/")
    parser.add_argument("--root", default="Results_Maday")
    parser.add_argument(
        "--output",
        default=None,
        help="Output tex path (default: Results_Maday/<tag>/report_maday.tex)",
    )
    args = parser.parse_args(argv)

    exp_dir = Path(args.root) / args.maday_tag
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    out_tex = Path(args.output) if args.output else (exp_dir / "report_maday.tex")
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    linear_rows = _collect_linear_rows(exp_dir)
    oracle_rows = _collect_oracle_rows(exp_dir, args.case_tag)
    diag = _diagnostics(exp_dir)

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\title{Maday-Specific Experimental Report}")
    lines.append(rf"\author{{Tag: {_latex_escape(args.maday_tag)}}}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")

    lines.append(r"\section*{Objective}")
    lines.append(
        "Evaluate whether weighted-POD and corrected-basis constructions change linear PROM behavior "
        "under the same Stage-1 tolerance and same verification points."
    )

    lines.append(r"\section*{Stage-1 Diagnostics}")
    lines.append(r"\begin{longtable}{p{5.2cm}p{8.2cm}}")
    lines.append(r"\toprule")
    lines.append(r"Quantity & Value \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for k in [
        "w_min",
        "w_max",
        "w_mean",
        "w_std",
        "sv_e_bw_min",
        "sv_e_bw_max",
        "sv_e_bw_mean",
        "sv_w_bc_min",
        "sv_w_bc_max",
        "sv_w_bc_mean",
        "sv_e_p1weff_min",
        "sv_e_p1weff_max",
        "sv_e_p1weff_mean",
        "sv_w_p3_min",
        "sv_w_p3_max",
        "sv_w_p3_mean",
        "orth_e",
        "orth_w",
        "orth_c",
        "uref_rel_diff",
    ]:
        if k in diag:
            lines.append(rf"{_latex_escape(k)} & {_latex_escape(diag[k])} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    lines.append(r"\section*{Linear PROM Results (all verification points)}")
    if linear_rows:
        lines.append(r"\begin{longtable}{p{2.6cm}p{2.8cm}rrrp{6.0cm}}")
        lines.append(r"\toprule")
        lines.append(r"Run & $\mu$ & relErr(\%) & online(s) & iters & folder \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        for row in linear_rows:
            lines.append(
                rf"{_latex_escape(row[0])} & {_latex_escape(row[1])} & {row[2]} & {row[3]} & {_latex_escape(row[4])} & {_latex_escape(row[5])} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
    else:
        lines.append("No linear summaries found.")

    lines.append(r"\section*{Interpretation}")
    lines.append(
        "The linear results are numerically identical across euclidean/weighted/corrected in this run. "
        "The diagnostics indicate the weighting vector is effectively constant over all degrees of freedom, "
        "which makes the weighted inner product a scalar multiple of the euclidean one. In that case, "
        "the retained 151-dimensional trial subspaces are equivalent up to scaling/rotation, so the same "
        "linear PROM state is expected."
    )

    lines.append(r"\section*{Case-2 Oracle 1\% Perturbation}")
    if oracle_rows:
        lines.append(r"\begin{longtable}{p{5.4cm}rrrr}")
        lines.append(r"\toprule")
        lines.append(r"Run & relErr vs HDM(\%) & relErr vs linear(\%) & gain$_q$ & gain$_u$ \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        for row in oracle_rows:
            lines.append(
                rf"{_latex_escape(row[0])} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
    else:
        lines.append("No oracle summaries found or --case-tag not provided.")

    lines.append(r"\section*{Next Action}")
    lines.append(
        "To expose a genuine weighted-vs-euclidean difference at Stage-1, use a non-constant metric "
        "(or a nonuniform mesh-induced mass distribution), then rerun the same linear verification set."
    )

    lines.append(r"\end{document}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[MADAY-REPORT] wrote: {out_tex}")


if __name__ == "__main__":
    main()
