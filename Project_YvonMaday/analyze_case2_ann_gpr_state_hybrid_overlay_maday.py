#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""State-space overlays for hybrid qN trajectories built from ANN/GPR maps.

For each model:
  - n_s=131: qN_hybrid = [qN_ref[1:20]; q_s_pred(131)]
  - n_s=141: qN_hybrid = [qN_ref[1:10]; q_s_pred(141)]
  - n_s=151: qN_hybrid = q_s_pred(151)

where qN_ref is the linear PROM reduced trajectory at the same mu point.
Then snapshots are reconstructed as:
  u_hybrid(t) = u_ref + V_tot qN_hybrid(t).

No online nonlinear ROM solve is run here; this is a pure offline map diagnostic
in state space, plotted in the same style as baseline_prom_hdm_vs_all_models.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import make_2D_grid

try:
    from check_case2_offline_errors import _load_case2_model, _predict_qs
except ModuleNotFoundError:
    from .check_case2_offline_errors import _load_case2_model, _predict_qs

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path


def set_latex_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
        }
    )


def _parse_points(raw_points: Sequence[str]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for item in raw_points:
        parts = [s.strip() for s in str(item).split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid point '{item}'. Expected 'mu1,mu2'.")
        out.append((float(parts[0]), float(parts[1])))
    return out


def _parse_steps(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in str(raw).split(","):
        txt = tok.strip()
        if not txt:
            continue
        v = int(txt)
        if v < 0:
            raise ValueError(f"Invalid negative time index in --steps: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("--steps produced an empty list.")
    return vals


def _default_model_specs(models_dir: Path) -> List[ModelSpec]:
    return [
        ModelSpec("ANN (n_s=131)", models_dir / "case2_ann_mu_t_ns131.pt"),
        ModelSpec("GPR (n_s=131)", models_dir / "case2_gpr_mu_t_ns131.pt"),
        ModelSpec("ANN (n_s=141)", models_dir / "case2_ann_mu_t_ns141.pt"),
        ModelSpec("GPR (n_s=141)", models_dir / "case2_gpr_mu_t_ns141.pt"),
        ModelSpec("ANN (n_s=151)", models_dir / "rom_data_driven_ann_mu_t_ntot151.pt"),
        ModelSpec("GPR (n_s=151)", models_dir / "rom_data_driven_gpr_mu_t_ntot151.pt"),
    ]


def _find_hdm_snap(hdm_dirs: Sequence[Path], mu1: float, mu2: float) -> Path:
    for hdm_dir in hdm_dirs:
        if not hdm_dir.is_dir():
            continue
        candidates = [
            hdm_dir / f"mu1_{mu1:g}+mu2_{mu2:g}.npy",
            hdm_dir / f"mu1_{mu1:.2f}+mu2_{mu2:.3f}.npy",
            hdm_dir / f"mu1_{mu1:.3f}+mu2_{mu2:.4f}.npy",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        # tolerant fallback
        pats = sorted(hdm_dir.glob(f"mu1_{mu1:.2f}*mu2_{mu2:.3f}*.npy"))
        if pats:
            return pats[0].resolve()
    raise FileNotFoundError(f"HDM snapshot not found for mu=({mu1},{mu2}) in {[str(d) for d in hdm_dirs]}")


def _find_linear_run(linear_runs_root: Path, ntot: int, mu1: float, mu2: float) -> Path:
    d = linear_runs_root / f"linear_prom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot{ntot}"
    if not d.is_dir():
        raise FileNotFoundError(f"Missing linear run folder: {d}")
    if not (d / "qN.npy").exists():
        raise FileNotFoundError(f"Missing qN.npy in {d}")
    if not (d / "t.npy").exists():
        raise FileNotFoundError(f"Missing t.npy in {d}")
    return d.resolve()


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


def _style_for_label(label: str):
    # User-requested palette for current comparison:
    # HDM -> black, linear PROM -> red, ANN131 -> blue, ANN141 -> green, ANN151 -> darkgoldenrod.
    key = str(label).strip()
    low = key.lower()
    if key == "HDM":
        return ("black", dict(linestyle="-", linewidth=3.0, alpha=0.88))
    if key == "Linear PROM":
        return ("red", dict(linestyle="-", linewidth=2.6, alpha=0.78))
    if key.startswith("Linear PROM (n="):
        return ("red", dict(linestyle="-.", linewidth=2.3, alpha=0.62))
    if "old" in low and "ann" in low:
        return ("teal", dict(linestyle=(0, (5, 1.6)), linewidth=2.8, alpha=0.76))
    if "ann" in low and "131" in low:
        return ("blue", dict(linestyle="--", linewidth=2.6, alpha=0.76))
    if "ann" in low and "141" in low:
        return ("green", dict(linestyle="-.", linewidth=2.6, alpha=0.76))
    if "ann" in low and "151" in low:
        return ("darkgoldenrod", dict(linestyle=":", linewidth=3.0, alpha=0.80))
    if "GPR" in key.upper():
        return ("green", dict(linestyle="--", linewidth=2.6, alpha=0.76))
    if "RBF" in key.upper():
        return ("green", dict(linestyle="--", linewidth=2.6, alpha=0.76))
    if "ANN" in key.upper():
        return ("darkgoldenrod", dict(linestyle=":", linewidth=2.6, alpha=0.80))
    return ("#1f77b4", dict(linestyle="-", linewidth=2.2, alpha=0.85))


def _reconstruct_snaps(u_ref: np.ndarray, basis: np.ndarray, qn: np.ndarray) -> np.ndarray:
    # qn: (ntot, nt), basis: (N, ntot), u_ref: (N,)
    if qn.ndim != 2:
        raise ValueError(f"qn must be 2D (ntot,nt), got {qn.shape}")
    return u_ref[:, None] + basis @ qn


def _mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def _save_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot HDM vs hybrid ANN/GPR state overlays from offline qN map predictions."
    )
    parser.add_argument("--maday-tag", type=str, default="maday_clean_try04")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument(
        "--linear-runs-root",
        type=str,
        default=None,
        help="Folder containing linear_prom_mu1_... run directories.",
    )
    parser.add_argument(
        "--stage1-dir",
        type=str,
        default=None,
        help="Folder containing basis.npy and u_ref.npy. Default: 250x250/Results/Stage1",
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
        help="Custom model entry 'Label=/abs/path/model.pt'.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output-subdir", type=str, default="case2_ann_gpr_coeff_offline")
    parser.add_argument("--figure-suffix", type=str, default="ann_gpr_131_141_151")
    parser.add_argument("--steps", type=str, default="0,125,250,375,500")
    parser.add_argument("--no-linear-reference", action="store_true")
    parser.add_argument(
        "--include-linear-primary-reference",
        action="store_true",
        help=(
            "Add baseline reference that keeps only first n_p coefficients from "
            "linear PROM and sets the tail to zero."
        ),
    )
    parser.add_argument(
        "--linear-primary-modes-ref",
        type=int,
        default=20,
        help="Primary-mode count n_p for the linear baseline reference.",
    )
    args = parser.parse_args(argv)

    set_latex_plot_style()

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)
    project_root = Path(__file__).resolve().parent

    linear_runs_root = (
        Path(args.linear_runs_root).expanduser().resolve()
        if args.linear_runs_root
        else (project_root / "250x250" / "Results" / "Runs" / "Linear").resolve()
    )
    stage1_dir = (
        Path(args.stage1_dir).expanduser().resolve()
        if args.stage1_dir
        else (project_root / "250x250" / "Results" / "Stage1").resolve()
    )
    out_dir = (Path(paths.figures) / args.output_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    basis_path = stage1_dir / "basis.npy"
    uref_path = stage1_dir / "u_ref.npy"
    if not basis_path.exists() or not uref_path.exists():
        raise FileNotFoundError(f"Missing basis.npy or u_ref.npy in {stage1_dir}")
    basis = np.load(basis_path, allow_pickle=False)
    u_ref = np.load(uref_path, allow_pickle=False).reshape(-1)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got {basis.shape}")
    if basis.shape[0] != u_ref.size:
        raise ValueError(f"basis/u_ref mismatch: basis rows={basis.shape[0]} vs u_ref={u_ref.size}")

    raw_points = args.point if args.point is not None else ["4.875,0.0225", "4.560,0.0190", "5.190,0.0260"]
    points = _parse_points(raw_points)
    steps = _parse_steps(args.steps)

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
        print("[case2-hybrid-overlay] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    loaded_models = []
    for spec in model_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {spec.checkpoint}")
        model, ntot, n_s = _load_case2_model(spec.checkpoint, device=device)
        fam = _infer_family(spec.label)
        label = _canonical_label(fam, n_s)
        n_p = int(ntot - n_s)
        loaded_models.append(
            {
                "label": label,
                "checkpoint": spec.checkpoint,
                "model": model,
                "n_tot": int(ntot),
                "n_s": int(n_s),
                "n_p": int(n_p),
            }
        )

    ntot_set = {d["n_tot"] for d in loaded_models}
    if len(ntot_set) != 1:
        raise RuntimeError(f"All selected models must share n_tot. Got: {sorted(ntot_set)}")
    ntot_common = int(next(iter(ntot_set)))
    n_p_linear_ref = int(args.linear_primary_modes_ref)
    if args.include_linear_primary_reference and not (0 <= n_p_linear_ref < ntot_common):
        raise ValueError(
            f"--linear-primary-modes-ref={n_p_linear_ref} must satisfy 0 <= n_p < n_tot={ntot_common}."
        )
    if basis.shape[1] < ntot_common:
        raise RuntimeError(f"Basis has {basis.shape[1]} columns but models require n_tot={ntot_common}.")

    # 250x250 mesh, same section lines as baseline plot.
    grid_x, grid_y = make_2D_grid(0, 100, 0, 100, 250, 250)
    x = 0.5 * (grid_x[1:] + grid_x[:-1])
    y = 0.5 * (grid_y[1:] + grid_y[:-1])
    nx = x.size
    ny = y.size
    nxy = nx * ny
    mid_x = nx // 2
    mid_y = ny // 2

    hdm_dirs = [
        (project_root / "250x250" / "param_snaps").resolve(),
        (project_root / "Results" / "param_snaps").resolve(),
    ]

    fig, axs = plt.subplots(len(points), 2, figsize=(14, 4.3 * len(points)), constrained_layout=False)
    if len(points) == 1:
        axs = np.array([axs])

    summary_rows: List[Dict] = []

    for row, (mu1, mu2) in enumerate(points):
        run_dir = _find_linear_run(linear_runs_root, ntot_common, mu1, mu2)
        qn_ref = np.load(run_dir / "qN.npy", allow_pickle=False)
        t_ref = np.load(run_dir / "t.npy", allow_pickle=False).reshape(-1)
        if qn_ref.shape != (ntot_common, t_ref.size):
            raise ValueError(f"Reference shape mismatch in {run_dir}: qN={qn_ref.shape}, t={t_ref.shape}")

        hdm_path = _find_hdm_snap(hdm_dirs, mu1, mu2)
        hdm = np.load(hdm_path, allow_pickle=False)
        if hdm.ndim != 2:
            raise ValueError(f"HDM snapshot must be 2D (N,nt), got {hdm.shape} at {hdm_path}")
        if hdm.shape[0] != u_ref.size:
            raise ValueError(f"HDM state size mismatch: {hdm.shape[0]} vs {u_ref.size} at {hdm_path}")

        nt_common = min(int(t_ref.size), int(hdm.shape[1]))
        step_idx = sorted({s for s in steps if s < nt_common})
        if not step_idx:
            raise RuntimeError(f"No valid time indices for mu=({mu1},{mu2}) with nt={nt_common} and steps={steps}")
        final_step = step_idx[-1]

        qn_ref_mu = qn_ref[:, :nt_common]
        hdm_mu = hdm[:, :nt_common]
        linear_snap = _reconstruct_snaps(u_ref, basis[:, :ntot_common], qn_ref_mu)

        snaps: Dict[str, np.ndarray] = {"HDM": hdm_mu}
        if not args.no_linear_reference:
            snaps["Linear PROM"] = linear_snap
        if args.include_linear_primary_reference:
            qn_linear_np = np.array(qn_ref_mu, copy=True)
            qn_linear_np[n_p_linear_ref:, :] = 0.0
            linear_np_snap = _reconstruct_snaps(u_ref, basis[:, :ntot_common], qn_linear_np)
            label_np = f"Linear PROM (n={n_p_linear_ref})"
            snaps[label_np] = linear_np_snap
            rel_vs_hdm = 100.0 * np.linalg.norm(linear_np_snap - hdm_mu) / (np.linalg.norm(hdm_mu) + 1e-30)
            rel_vs_linear = 100.0 * np.linalg.norm(linear_np_snap - linear_snap) / (np.linalg.norm(linear_snap) + 1e-30)
            summary_rows.append(
                {
                    "mu1": float(mu1),
                    "mu2": float(mu2),
                    "model_label": label_np,
                    "n_tot": int(ntot_common),
                    "n_p": int(n_p_linear_ref),
                    "n_s": int(ntot_common - n_p_linear_ref),
                    "nt": int(nt_common),
                    "rel_error_percent_vs_hdm_state": float(rel_vs_hdm),
                    "rel_error_percent_vs_linear_state": float(rel_vs_linear),
                    "model_file": "constructed_from_linear_qN_zero_tail",
                    "linear_qN_path": str((run_dir / "qN.npy").resolve()),
                    "hdm_path": str(hdm_path),
                }
            )

        for entry in loaded_models:
            q_s_pred = _predict_qs(entry["model"], mu1, mu2, t_ref[:nt_common], device=device)
            n_p = int(entry["n_p"])
            n_s = int(entry["n_s"])
            if q_s_pred.shape != (n_s, nt_common):
                raise ValueError(
                    f"{entry['label']} prediction shape mismatch at mu=({mu1},{mu2}): "
                    f"{q_s_pred.shape} vs ({n_s},{nt_common})"
                )

            qn_hybrid = np.array(qn_ref_mu, copy=True)
            qn_hybrid[n_p:, :] = q_s_pred
            snap_hybrid = _reconstruct_snaps(u_ref, basis[:, :ntot_common], qn_hybrid)
            snaps[str(entry["label"])] = snap_hybrid

            rel_vs_hdm = 100.0 * np.linalg.norm(snap_hybrid - hdm_mu) / (np.linalg.norm(hdm_mu) + 1e-30)
            rel_vs_linear = 100.0 * np.linalg.norm(snap_hybrid - linear_snap) / (np.linalg.norm(linear_snap) + 1e-30)
            summary_rows.append(
                {
                    "mu1": float(mu1),
                    "mu2": float(mu2),
                    "model_label": str(entry["label"]),
                    "n_tot": int(ntot_common),
                    "n_p": int(n_p),
                    "n_s": int(n_s),
                    "nt": int(nt_common),
                    "rel_error_percent_vs_hdm_state": float(rel_vs_hdm),
                    "rel_error_percent_vs_linear_state": float(rel_vs_linear),
                    "model_file": str(entry["checkpoint"]),
                    "linear_qN_path": str((run_dir / "qN.npy").resolve()),
                    "hdm_path": str(hdm_path),
                }
            )

        # Draw each model; dashed intermediates, solid final time.
        model_order = ["HDM", "Linear PROM"]
        if args.include_linear_primary_reference:
            model_order.append(f"Linear PROM (n={n_p_linear_ref})")
        model_order += [m["label"] for m in loaded_models]
        for model_name in model_order:
            if model_name not in snaps:
                continue
            arr = snaps[model_name]
            color, st_base = _style_for_label(model_name)
            for ind in step_idx:
                is_final = ind == final_step
                label = model_name if is_final else None
                snap_u = arr[:nxy, ind].reshape(ny, nx)

                st = dict(st_base)
                if is_final:
                    st["linestyle"] = st_base.get("linestyle", "-")
                    st["linewidth"] = max(2.2, float(st_base.get("linewidth", 2.2)))
                    st["zorder"] = 7 if model_name == "HDM" else 5
                else:
                    st["linestyle"] = "--"
                    st["linewidth"] = 0.95
                    st["alpha"] = min(0.80, float(st_base.get("alpha", 0.8)))
                    st["zorder"] = 2

                axs[row, 0].plot(
                    x,
                    snap_u[mid_y, :],
                    color=color,
                    label=label,
                    **st,
                )
                axs[row, 1].plot(
                    y,
                    snap_u[:, mid_x],
                    color=color,
                    label=label,
                    **st,
                )

        axs[row, 0].set_title(rf"$\mu=({mu1:.3f},{mu2:.4f})$: $u_x(x,y_{{mid}})$")
        axs[row, 1].set_title(rf"$\mu=({mu1:.3f},{mu2:.4f})$: $u_x(x_{{mid}},y)$")
        axs[row, 0].set_xlabel(r"$x$")
        axs[row, 1].set_xlabel(r"$y$")
        axs[row, 0].set_ylabel(r"$u_x$")
        axs[row, 1].set_ylabel(r"$u_x$")
        axs[row, 0].grid(True)
        axs[row, 1].grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(8, max(4, len(labels))),
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Offline hybrid state overlays: HDM vs ANN/GPR hybrid qN reconstructions",
        y=1.035,
        fontsize=16,
    )
    fig.text(
        0.5,
        0.012,
        r"Dashed: intermediate times; solid: final time $t=25$. Hybrid rule: low block from linear PROM, high block from map.",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.94])

    fig_path = out_dir / f"hybrid_prom_hdm_vs_ann_gpr_models_{args.figure_suffix}.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[case2-hybrid-overlay] saved figure: {fig_path}")

    summary_csv = out_dir / f"case2_ann_gpr_hybrid_state_summary_{args.figure_suffix}.csv"
    _save_csv(
        summary_csv,
        summary_rows,
        [
            "mu1",
            "mu2",
            "model_label",
            "n_tot",
            "n_p",
            "n_s",
            "nt",
            "rel_error_percent_vs_hdm_state",
            "rel_error_percent_vs_linear_state",
            "model_file",
            "linear_qN_path",
            "hdm_path",
        ],
    )
    print(f"[case2-hybrid-overlay] summary csv: {summary_csv}")


if __name__ == "__main__":
    main()
