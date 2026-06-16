#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Offline coefficient-space comparison for Case-2 ANN/GPR maps.

Purpose:
  Compare ANN and GPR maps trained for:
    - n=20  -> n_s=131 secondary coefficients
    - n=10  -> n_s=141 secondary coefficients
  against linear PROM qN references at selected parameter points, without
  running online Case-2 ROM solves.

Outputs:
  - Per-point figure with absolute and relative coefficient errors vs global
    coefficient index (1..n_tot), four curves (ANN131/GPR131/ANN141/GPR141).
  - CSV summary per model and point.
  - CSV with per-coefficient errors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from check_case2_offline_errors import _load_case2_model, _predict_qs
except ModuleNotFoundError:
    from .check_case2_offline_errors import _load_case2_model, _predict_qs

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


EPS = 1e-30


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path


def _parse_points(raw_points: Sequence[str]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for item in raw_points:
        parts = [s.strip() for s in str(item).split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid point '{item}'. Expected 'mu1,mu2'.")
        out.append((float(parts[0]), float(parts[1])))
    return out


def _default_model_specs(models_dir: Path) -> List[ModelSpec]:
    return [
        ModelSpec("ANN (n_s=131)", models_dir / "case2_ann_mu_t_ns131.pt"),
        ModelSpec("GPR (n_s=131)", models_dir / "case2_gpr_mu_t_ns131.pt"),
        ModelSpec("ANN (n_s=141)", models_dir / "case2_ann_mu_t_ns141.pt"),
        ModelSpec("GPR (n_s=141)", models_dir / "case2_gpr_mu_t_ns141.pt"),
    ]


def _find_linear_run(linear_runs_root: Path, ntot: int, mu1: float, mu2: float) -> Path:
    d = linear_runs_root / f"linear_prom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot{ntot}"
    if not d.is_dir():
        raise FileNotFoundError(f"Missing linear run folder: {d}")
    qn = d / "qN.npy"
    tt = d / "t.npy"
    if not qn.exists() or not tt.exists():
        raise FileNotFoundError(f"Missing qN.npy or t.npy in {d}")
    return d


def _mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def _save_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _infer_family(label: str) -> str:
    low = str(label).lower()
    if "gpr" in low:
        return "GPR"
    if "rbf" in low:
        return "RBF"
    if "ann" in low and ("old" in low or "baseline" in low or "data-driven" in low or "data driven" in low):
        return "Old data-driven ANN"
    if "ann" in low:
        return "ANN"
    return str(label).strip()


def _canonical_label(family: str, n_s: int) -> str:
    return f"{family} (n_s={int(n_s)})"


def _tex_label(family: str, n_s: int) -> str:
    return f"{family} ($n_s={int(n_s)}$)"


def _style_for_family_ns(family: str, n_s: int):
    fam_up = str(family).upper()
    if fam_up.startswith("LINEAR"):
        return ("red", dict(linestyle="-.", linewidth=2.8, alpha=0.78, marker="None", zorder=3))
    if "OLD" in fam_up and "ANN" in fam_up:
        return ("teal", dict(linestyle=(0, (5, 1.6)), linewidth=3.0, alpha=0.82, marker="P", markersize=2.8, zorder=7))

    # Requested emphasis for this comparison: GPR in green.
    if fam_up.startswith("GPR"):
        return ("green", dict(linestyle="--", linewidth=2.8, alpha=0.92, marker="^", markersize=2.6, zorder=8))
    if fam_up.startswith("RBF"):
        return ("green", dict(linestyle="--", linewidth=2.8, alpha=0.92, marker="^", markersize=2.6, zorder=8))

    # Explicit high-contrast style set (requested): red, blue, green, black, darkgoldenrod, orange.
    key = (fam_up, int(n_s))
    mapping = {
        ("ANN", 131): ("blue", dict(linestyle="--", linewidth=2.8, alpha=0.90, marker="o", markersize=2.6, zorder=6)),
        ("GPR", 131): ("black", dict(linestyle=":", linewidth=3.4, alpha=0.88, marker="s", markersize=2.4, zorder=9)),
        ("ANN", 141): ("green", dict(linestyle="-.", linewidth=2.8, alpha=0.90, marker="o", markersize=2.6, zorder=5)),
        ("GPR", 141): ("red", dict(linestyle="-", linewidth=2.8, alpha=0.90, marker="^", markersize=2.5, zorder=8)),
        ("ANN", 151): ("darkgoldenrod", dict(linestyle="-", linewidth=2.8, alpha=0.84, marker="D", markersize=2.5, zorder=4)),
        ("GPR", 151): ("orange", dict(linestyle="--", linewidth=2.8, alpha=0.85, marker="x", markersize=2.7, zorder=7)),
    }
    if key in mapping:
        return mapping[key]
    # Fallback if another model appears.
    return "#1f77b4", dict(linestyle="-", linewidth=2.4, alpha=0.85, marker="o", markersize=2.3, zorder=4)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline coefficient error comparison for ANN/GPR Case-2 (131 vs 141 outputs)."
    )
    parser.add_argument("--maday-tag", type=str, default="maday_clean_try04")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument(
        "--linear-runs-root",
        type=str,
        default=None,
        help=(
            "Folder containing linear_prom_mu1_... directories. "
            "Default: Project_YvonMaday/250x250/Results/Runs/Linear"
        ),
    )
    parser.add_argument(
        "--point",
        action="append",
        default=None,
        help="Point 'mu1,mu2'. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Custom model entry 'Label=/abs/path/model.pt'. "
            "If omitted, uses the 4 default models in Results_Maday/<tag>/Stage3/models."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output-subdir", type=str, default="case2_ann_gpr_coeff_offline")
    parser.add_argument(
        "--csv-stem",
        type=str,
        default="case2_ann_gpr_offline_coeff",
        help="Stem used for summary/per-index CSV files.",
    )
    parser.add_argument(
        "--figure-suffix",
        type=str,
        default="ann_gpr_131_141",
        help="Suffix used in per-point figure file names.",
    )
    parser.add_argument(
        "--include-linear-primary-reference",
        action="store_true",
        help=(
            "Add baseline reference where only first n_p coefficients are kept "
            "from linear PROM and the remaining tail is zero."
        ),
    )
    parser.add_argument(
        "--linear-primary-modes-ref",
        type=int,
        default=20,
        help="Primary-mode count n_p for the linear baseline reference.",
    )
    args = parser.parse_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    project_root = Path(__file__).resolve().parent
    linear_runs_root = (
        Path(args.linear_runs_root).expanduser().resolve()
        if args.linear_runs_root
        else (project_root / "250x250" / "Results" / "Runs" / "Linear").resolve()
    )
    out_dir = (Path(paths.figures) / args.output_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = args.point if args.point is not None else ["4.875,0.0225", "4.560,0.0190", "5.190,0.0260"]
    points = _parse_points(raw_points)

    if args.model:
        model_specs: List[ModelSpec] = []
        for raw in args.model:
            if "=" not in raw:
                raise ValueError(f"Invalid --model '{raw}'. Expected 'Label=/path/model.pt'.")
            label, path = raw.rsplit("=", 1)
            model_specs.append(ModelSpec(label.strip(), Path(path.strip()).expanduser().resolve()))
    else:
        model_specs = _default_model_specs(Path(paths.stage3_models))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[case2-offline-coeff] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    loaded = []
    for spec in model_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {spec.checkpoint}")
        model, ntot, n_s = _load_case2_model(spec.checkpoint, device=device)
        family = _infer_family(spec.label)
        canonical_label = _canonical_label(family, n_s)
        plot_label = _tex_label(family, n_s)
        n_p = int(ntot - n_s)
        loaded.append(
            {
                "label": canonical_label,
                "plot_label": plot_label,
                "family": family,
                "checkpoint": spec.checkpoint,
                "model": model,
                "ntot": int(ntot),
                "n_s": int(n_s),
                "n_p": int(n_p),
            }
        )

    ntot_set = {d["ntot"] for d in loaded}
    if len(ntot_set) != 1:
        raise RuntimeError(f"All models must share n_tot for a common global-index plot. Got: {sorted(ntot_set)}")
    ntot_common = int(next(iter(ntot_set)))
    n_p_linear_ref = int(args.linear_primary_modes_ref)
    if args.include_linear_primary_reference and not (0 <= n_p_linear_ref < ntot_common):
        raise ValueError(
            f"--linear-primary-modes-ref={n_p_linear_ref} must satisfy 0 <= n_p < n_tot={ntot_common}."
        )

    summary_rows: List[Dict] = []
    coeff_rows: List[Dict] = []

    x_global = np.arange(1, ntot_common + 1, dtype=int)

    for (mu1, mu2) in points:
        run_dir = _find_linear_run(linear_runs_root, ntot_common, mu1, mu2)
        qn_ref = np.load(run_dir / "qN.npy", allow_pickle=False)
        t_ref = np.load(run_dir / "t.npy", allow_pickle=False).reshape(-1)

        if qn_ref.shape != (ntot_common, t_ref.size):
            raise ValueError(
                f"Reference shape mismatch at {run_dir}: qN={qn_ref.shape}, expected ({ntot_common},{t_ref.size})"
            )

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        plot_cache = {}
        for entry in loaded:
            label = entry["label"]
            plot_label = entry["plot_label"]
            model = entry["model"]
            n_p = entry["n_p"]
            n_s = entry["n_s"]
            family = entry["family"]

            q_s_ref = qn_ref[n_p:, :]  # (n_s, nt)
            q_s_pred = _predict_qs(model, mu1, mu2, t_ref, device=device)  # (n_s, nt)
            if q_s_pred.shape != q_s_ref.shape:
                raise ValueError(
                    f"Prediction shape mismatch for {label} at mu=({mu1},{mu2}): "
                    f"pred={q_s_pred.shape}, ref={q_s_ref.shape}"
                )

            err = q_s_ref - q_s_pred
            abs_local = np.linalg.norm(err, axis=1)
            ref_local = np.linalg.norm(q_s_ref, axis=1)
            rel_local = 100.0 * abs_local / (ref_local + EPS)

            rel_frob = 100.0 * np.linalg.norm(err) / (np.linalg.norm(q_s_ref) + EPS)

            abs_global = np.full((ntot_common,), np.nan, dtype=np.float64)
            rel_global = np.full((ntot_common,), np.nan, dtype=np.float64)
            abs_global[n_p:] = abs_local
            rel_global[n_p:] = rel_local

            plot_cache[label] = {
                "plot_label": plot_label,
                "abs_global": abs_global,
                "rel_global": rel_global,
                "n_p": n_p,
                "n_s": n_s,
                "family": family,
            }

            summary_rows.append(
                {
                    "mu1": float(mu1),
                    "mu2": float(mu2),
                    "model_label": label,
                    "model_file": str(entry["checkpoint"]),
                    "n_tot": int(ntot_common),
                    "n_p": int(n_p),
                    "n_s": int(n_s),
                    "nt": int(t_ref.size),
                    "rel_frob_percent": float(rel_frob),
                    "mean_coeff_rel_percent": float(np.mean(rel_local)),
                    "median_coeff_rel_percent": float(np.median(rel_local)),
                    "p95_coeff_rel_percent": float(np.percentile(rel_local, 95.0)),
                    "max_coeff_rel_percent": float(np.max(rel_local)),
                    "linear_qN_path": str((run_dir / "qN.npy").resolve()),
                }
            )

            local_idx = np.arange(1, n_s + 1, dtype=int)
            global_idx = np.arange(n_p + 1, ntot_common + 1, dtype=int)
            for i in range(n_s):
                coeff_rows.append(
                    {
                        "mu1": float(mu1),
                        "mu2": float(mu2),
                        "model_label": label,
                        "n_tot": int(ntot_common),
                        "n_p": int(n_p),
                        "n_s": int(n_s),
                        "global_coeff_1based": int(global_idx[i]),
                        "local_secondary_coeff_1based": int(local_idx[i]),
                        "abs_error_l2_time": float(abs_local[i]),
                        "ref_l2_time": float(ref_local[i]),
                        "rel_error_percent": float(rel_local[i]),
                    }
                )

        if args.include_linear_primary_reference:
            n_p = int(n_p_linear_ref)
            n_s = int(ntot_common - n_p)
            q_s_ref = qn_ref[n_p:, :]  # (n_s, nt)
            # Baseline "linear n_p only": predicted tail is zero.
            q_s_pred = np.zeros_like(q_s_ref)
            err = q_s_ref - q_s_pred
            abs_local = np.linalg.norm(err, axis=1)
            ref_local = np.linalg.norm(q_s_ref, axis=1)
            rel_local = 100.0 * abs_local / (ref_local + EPS)
            rel_frob = 100.0 * np.linalg.norm(err) / (np.linalg.norm(q_s_ref) + EPS)

            abs_global = np.full((ntot_common,), np.nan, dtype=np.float64)
            rel_global = np.full((ntot_common,), np.nan, dtype=np.float64)
            abs_global[n_p:] = abs_local
            rel_global[n_p:] = rel_local

            label = f"Linear PROM (n={n_p})"
            plot_cache[label] = {
                "plot_label": label,
                "abs_global": abs_global,
                "rel_global": rel_global,
                "n_p": n_p,
                "n_s": n_s,
                "family": "LINEAR",
            }

            summary_rows.append(
                {
                    "mu1": float(mu1),
                    "mu2": float(mu2),
                    "model_label": label,
                    "model_file": "constructed_from_linear_qN_zero_tail",
                    "n_tot": int(ntot_common),
                    "n_p": int(n_p),
                    "n_s": int(n_s),
                    "nt": int(t_ref.size),
                    "rel_frob_percent": float(rel_frob),
                    "mean_coeff_rel_percent": float(np.mean(rel_local)),
                    "median_coeff_rel_percent": float(np.median(rel_local)),
                    "p95_coeff_rel_percent": float(np.percentile(rel_local, 95.0)),
                    "max_coeff_rel_percent": float(np.max(rel_local)),
                    "linear_qN_path": str((run_dir / "qN.npy").resolve()),
                }
            )

            local_idx = np.arange(1, n_s + 1, dtype=int)
            global_idx = np.arange(n_p + 1, ntot_common + 1, dtype=int)
            for i in range(n_s):
                coeff_rows.append(
                    {
                        "mu1": float(mu1),
                        "mu2": float(mu2),
                        "model_label": label,
                        "n_tot": int(ntot_common),
                        "n_p": int(n_p),
                        "n_s": int(n_s),
                        "global_coeff_1based": int(global_idx[i]),
                        "local_secondary_coeff_1based": int(local_idx[i]),
                        "abs_error_l2_time": float(abs_local[i]),
                        "ref_l2_time": float(ref_local[i]),
                        "rel_error_percent": float(rel_local[i]),
                    }
                )

        draw_order = [m["label"] for m in loaded]
        if args.include_linear_primary_reference:
            draw_order = [f"Linear PROM (n={n_p_linear_ref})"] + draw_order

        linear_ref_label = f"Linear PROM (n={n_p_linear_ref})" if args.include_linear_primary_reference else None
        linear_ref_abs = None
        if linear_ref_label is not None and linear_ref_label in plot_cache:
            linear_ref_abs = np.asarray(plot_cache[linear_ref_label]["abs_global"], dtype=np.float64)

        for idx, label in enumerate(draw_order):
            if label not in plot_cache:
                continue
            curve = plot_cache[label]
            color, st = _style_for_family_ns(curve["family"], curve["n_s"])
            axes[0].plot(x_global, curve["abs_global"], label=curve["plot_label"], color=color, **st)
            if linear_ref_abs is not None:
                rel_vs_linear = np.full((ntot_common,), np.nan, dtype=np.float64)
                valid = np.isfinite(curve["abs_global"]) & np.isfinite(linear_ref_abs)
                rel_vs_linear[valid] = 100.0 * curve["abs_global"][valid] / (linear_ref_abs[valid] + EPS)
                axes[1].plot(x_global, rel_vs_linear, label=curve["plot_label"], color=color, **st)
            else:
                axes[1].plot(x_global, curve["rel_global"], label=curve["plot_label"], color=color, **st)

        separators = sorted({int(plot_cache[lbl]["n_p"]) for lbl in draw_order if lbl in plot_cache and int(plot_cache[lbl]["n_p"]) > 0})
        for ax in axes:
            ax.grid(True, alpha=0.3)
            for s in separators:
                ax.axvline(float(s) + 0.5, linestyle="--", linewidth=0.9, color="gray", alpha=0.7)
            ax.set_xlim(1, ntot_common)

        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_ylabel(r"$\|q_i^{ref}-q_i^{pred}\|_2$")
        if linear_ref_abs is not None:
            axes[1].set_ylabel(
                rf"$100\|e_i\|_2 / (\|e_i^{{Linear\,PROM\,(n={n_p_linear_ref})}}\|_2+\epsilon)$ [\%]"
            )
        else:
            axes[1].set_ylabel(r"$100\|q_i^{ref}-q_i^{pred}\|_2 / (\|q_i^{ref}\|_2+\epsilon)$ [\%]")
        axes[1].set_xlabel("Global coefficient index")
        axes[0].legend(loc="best", fontsize=9)
        axes[0].set_title(
            f"Case-2 offline coefficient errors at mu=({mu1:.3f}, {mu2:.4f}) "
            f"(reference: linear PROM qN, no online solve)"
        )

        fig.tight_layout()
        fig_path = out_dir / f"{_mu_tag(mu1, mu2)}_coeff_abs_rel_vs_global_index_{args.figure_suffix}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        print(f"[case2-offline-coeff] saved figure: {fig_path}")

        # Dedicated GPR zoom plot to verify non-identical behavior.
        gpr_labels = [lbl for lbl in draw_order if plot_cache[lbl]["family"].upper() == "GPR"]
        if len(gpr_labels) >= 2:
            i0 = max(int(plot_cache[lbl]["n_p"]) for lbl in gpr_labels)
            xz = x_global[i0:]

            figz, axz = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
            for lbl in gpr_labels:
                c = plot_cache[lbl]
                color, st = _style_for_family_ns(c["family"], c["n_s"])
                st2 = dict(st)
                st2["markersize"] = max(2.2, float(st2.get("markersize", 2.4)))
                axz[0].plot(xz, c["rel_global"][i0:], label=c["plot_label"], color=color, **st2)

            base = plot_cache[gpr_labels[0]]["rel_global"][i0:]
            base_name = plot_cache[gpr_labels[0]]["plot_label"]
            max_diff = 0.0
            for lbl in gpr_labels[1:]:
                c = plot_cache[lbl]
                rel = c["rel_global"][i0:]
                diff = np.abs(rel - base)
                max_diff = max(max_diff, float(np.nanmax(diff)))
                color, _ = _style_for_family_ns(c["family"], c["n_s"])
                axz[1].plot(xz, diff, color=color, linewidth=2.2, alpha=0.95, label=f"|{c['plot_label']} - {base_name}|")

            axz[0].set_title(
                f"GPR relative-error zoom at mu=({mu1:.3f}, {mu2:.4f}) "
                f"(global indices {i0+1}..{ntot_common})"
            )
            axz[0].set_ylabel(r"$100\|q_i^{ref}-q_i^{pred}\|_2/(\|q_i^{ref}\|_2+\epsilon)$ [\%]")
            axz[1].set_ylabel("absolute difference [%]")
            axz[1].set_xlabel("Global coefficient index")
            axz[0].set_yscale("log")
            if max_diff > 0.0:
                axz[1].set_yscale("log")
            else:
                axz[1].set_yscale("linear")
                axz[1].text(
                    0.02,
                    0.88,
                    "pairwise differences are numerically zero",
                    transform=axz[1].transAxes,
                    fontsize=8.5,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )
            for ax in axz:
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)

            figz.tight_layout()
            zpath = out_dir / f"{_mu_tag(mu1, mu2)}_gpr_rel_zoom_{args.figure_suffix}.png"
            figz.savefig(zpath, dpi=190)
            plt.close(figz)
            print(f"[case2-offline-coeff] saved GPR zoom: {zpath}")

    summary_csv = out_dir / f"{args.csv_stem}_summary.csv"
    coeff_csv = out_dir / f"{args.csv_stem}_per_index.csv"
    _save_csv(
        summary_csv,
        summary_rows,
        [
            "mu1",
            "mu2",
            "model_label",
            "model_file",
            "n_tot",
            "n_p",
            "n_s",
            "nt",
            "rel_frob_percent",
            "mean_coeff_rel_percent",
            "median_coeff_rel_percent",
            "p95_coeff_rel_percent",
            "max_coeff_rel_percent",
            "linear_qN_path",
        ],
    )
    _save_csv(
        coeff_csv,
        coeff_rows,
        [
            "mu1",
            "mu2",
            "model_label",
            "n_tot",
            "n_p",
            "n_s",
            "global_coeff_1based",
            "local_secondary_coeff_1based",
            "abs_error_l2_time",
            "ref_l2_time",
            "rel_error_percent",
        ],
    )

    print(f"[case2-offline-coeff] summary csv: {summary_csv}")
    print(f"[case2-offline-coeff] per-index csv: {coeff_csv}")


if __name__ == "__main__":
    main()
