#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Offline coefficient diagnostics for PROM-151 learned maps.

For each evaluation point, compare linear-PROM coefficients against:
  - Case 1: q_1..q_10 from linear PROM, ANN reconstructs q_11..q_151.
  - DD/Case 2 source: ANN maps (mu1, mu2, t) -> q_1..q_151.
  - Case 3: q_1..q_10 plus (mu1, mu2, t), ANN reconstructs q_11..q_151.
  - PROM-POD-AE: autoencoder reconstruction of linear-PROM q_1..q_151.
  - POD-DL-ROM: non-intrusive map (mu1, mu2, t) -> q_1..q_151.

This is a pure inference/plotting diagnostic; it does not run a PROM solve.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from stage3_perform_training_case_1_ann import Case1Model  # noqa: E402
from stage3_perform_training_case_3_ann import Case3Model  # noqa: E402
from stage3_perform_training_rom_data_driven import ROMDataDrivenModel  # noqa: E402
from pod_ae_common import PROMPODAEAutoencoder, infer_scaling_from_state_dict  # noqa: E402
from pod_dl_data_driven_common import PODDLDataDrivenModel  # noqa: E402


POINTS = (
    ("verification", 4.875, 0.0225),
    ("offgrid1", 4.560, 0.0190),
    ("offgrid2", 5.190, 0.0260),
    ("extrapolation20pct", 4.000, 0.0330),
)


MODEL_STYLE = {
    "case1": {
        "label": "Case 1",
        "long": r"Case 1: $q_p\mapsto q_s$",
        "color": "tab:blue",
        "alpha": 0.76,
        "lw": 1.35,
    },
    "dd_case2": {
        "label": "DD / Case 2",
        "long": r"DD / Case 2: $(\mu,t)\mapsto q_{\rm tot}$",
        "color": "tab:orange",
        "alpha": 0.76,
        "lw": 1.35,
    },
    "case3": {
        "label": "Case 3",
        "long": r"Case 3: $(q_p,\mu,t)\mapsto q_s$",
        "color": "tab:green",
        "alpha": 0.76,
        "lw": 1.35,
    },
    "pod_ae": {
        "label": "POD-AE",
        "long": r"POD-AE: $D(E(q_{\rm lin}))$",
        "color": "tab:purple",
        "alpha": 0.70,
        "lw": 1.35,
    },
    "pod_dl": {
        "label": "POD-DL-ROM",
        "long": r"POD-DL-ROM: $(\mu,t)\mapsto q_{\rm tot}$",
        "color": "tab:pink",
        "alpha": 0.74,
        "lw": 1.35,
    },
}


def _fmt_mu_dir(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def _safe_name(txt: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", txt).strip("_")


def _tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _model_from_checkpoint(path: Path, model_cls):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"{path} does not contain a 'state_dict' checkpoint entry.")
    sd = ckpt["state_dict"]
    x_mean = _tensor_to_numpy(sd["scaler.mean"]).reshape(-1)
    x_std = _tensor_to_numpy(sd["scaler.std"]).reshape(-1)
    y_mean = _tensor_to_numpy(sd["unscaler.mean"]).reshape(-1)
    y_std = _tensor_to_numpy(sd["unscaler.std"]).reshape(-1)
    hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", (256, 512, 512, 256)))
    activation = str(ckpt.get("activation", "silu"))
    dropout = float(ckpt.get("dropout", 0.0))
    model = model_cls(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    )
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def _state_array(state_dict: dict, key: str) -> np.ndarray:
    if key not in state_dict:
        raise KeyError(f"Missing checkpoint state key: {key}")
    return _tensor_to_numpy(state_dict[key]).reshape(1, -1)


def _pod_ae_from_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"{path} does not contain a 'state_dict' checkpoint entry.")
    sd = ckpt["state_dict"]
    scaling = infer_scaling_from_state_dict(sd, fallback=ckpt.get("scaling", "zscore"))
    if scaling == "zscore":
        q_stats = {"mean": _state_array(sd, "scaler.mean"), "std": _state_array(sd, "scaler.std")}
    elif scaling == "minmax_-1_1":
        q_min = _state_array(sd, "scaler.center") - _state_array(sd, "scaler.half_range")
        q_max = _state_array(sd, "scaler.center") + _state_array(sd, "scaler.half_range")
        q_stats = {"min": q_min, "max": q_max}
    else:
        raise ValueError(f"Unsupported POD-AE scaling in {path}: {scaling}")

    model = PROMPODAEAutoencoder(
        q_dim=int(ckpt.get("q_dim", 151)),
        latent_dim=int(ckpt.get("latent_dim", 10)),
        hidden_dims=tuple(int(v) for v in ckpt.get("hidden_dims", (512, 256, 128))),
        scaling=scaling,
        activation=str(ckpt.get("activation", "gelu")),
        q_stats=q_stats,
    )
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def _pod_dl_from_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"{path} does not contain a 'state_dict' checkpoint entry.")
    sd = ckpt["state_dict"]
    x_scaling = str(ckpt.get("x_scaling", "zscore")).lower()
    q_scaling = str(ckpt.get("q_scaling", "zscore")).lower()
    if x_scaling == "zscore":
        x_stats = {"mean": _state_array(sd, "x_scaler.mean"), "std": _state_array(sd, "x_scaler.std")}
    elif x_scaling == "minmax_-1_1":
        x_min = _state_array(sd, "x_scaler.center") - _state_array(sd, "x_scaler.half_range")
        x_max = _state_array(sd, "x_scaler.center") + _state_array(sd, "x_scaler.half_range")
        x_stats = {"min": x_min, "max": x_max}
    else:
        raise ValueError(f"Unsupported POD-DL x_scaling in {path}: {x_scaling}")

    if q_scaling == "zscore":
        q_stats = {"mean": _state_array(sd, "q_scaler.mean"), "std": _state_array(sd, "q_scaler.std")}
    elif q_scaling == "minmax_-1_1":
        q_min = _state_array(sd, "q_scaler.center") - _state_array(sd, "q_scaler.half_range")
        q_max = _state_array(sd, "q_scaler.center") + _state_array(sd, "q_scaler.half_range")
        q_stats = {"min": q_min, "max": q_max}
    else:
        raise ValueError(f"Unsupported POD-DL q_scaling in {path}: {q_scaling}")

    model = PODDLDataDrivenModel(
        q_dim=int(ckpt.get("q_dim", 151)),
        latent_dim=int(ckpt.get("latent_dim", 10)),
        encoder_hidden_dims=tuple(int(v) for v in ckpt.get("encoder_hidden_dims", (512, 256))),
        decoder_hidden_dims=tuple(int(v) for v in ckpt.get("decoder_hidden_dims", (256, 512))),
        dynamics_hidden_dims=tuple(int(v) for v in ckpt.get("dynamics_hidden_dims", (256, 512, 512, 256))),
        activation=str(ckpt.get("activation", "silu")),
        x_scaling=x_scaling,
        q_scaling=q_scaling,
        x_stats=x_stats,
        q_stats=q_stats,
    )
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def _predict(model, x_raw: np.ndarray, device: str) -> np.ndarray:
    model = model.to(device)
    with torch.no_grad():
        x = torch.as_tensor(x_raw, dtype=torch.float32, device=device)
        y = model(x).detach().cpu().numpy()
    return y


def _predict_pod_ae(model, q_raw: np.ndarray, device: str) -> np.ndarray:
    model = model.to(device)
    with torch.no_grad():
        x = torch.as_tensor(q_raw.T, dtype=torch.float32, device=device)
        y = model(x).detach().cpu().numpy().T
    return y


def _predict_pod_dl(model, x_raw: np.ndarray, device: str) -> np.ndarray:
    model = model.to(device)
    with torch.no_grad():
        x = torch.as_tensor(x_raw, dtype=torch.float32, device=device)
        y = model.predict_q_from_x(x).detach().cpu().numpy().T
    return y


def _relative_percent(pred: np.ndarray, ref: np.ndarray, axis=None, eps: float = 1e-14):
    num = np.linalg.norm(pred - ref, axis=axis)
    den = np.linalg.norm(ref, axis=axis)
    return 100.0 * num / np.maximum(den, eps)


def _load_linear_qn(root: Path, mu1: float, mu2: float) -> np.ndarray:
    path = root / "Runs" / "Linear" / f"linear_prom_{_fmt_mu_dir(mu1, mu2)}_ntot151" / "qN.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing linear PROM qN: {path}")
    q = np.load(path)
    if q.shape[0] != 151:
        raise ValueError(f"Expected qN shape (151,T), got {q.shape} at {path}")
    return q.astype(np.float64, copy=False)


def _predict_all(q_ref: np.ndarray, mu1: float, mu2: float, t: np.ndarray, models: dict, device: str):
    n_p = 10
    q_p = q_ref[:n_p, :]
    x_mu_t = np.column_stack(
        [
            np.full_like(t, float(mu1), dtype=np.float64),
            np.full_like(t, float(mu2), dtype=np.float64),
            t.astype(np.float64),
        ]
    )

    # Case 1: only primary coordinates are supplied.
    q_s_case1 = _predict(models["case1"], q_p.T, device=device).T
    q_case1 = np.vstack([q_p, q_s_case1])

    # Data-driven / Case-2 source: full q_tot from (mu,t).
    q_dd = _predict(models["dd"], x_mu_t, device=device).T

    # Case 3: primary coordinates plus mu and time.
    x_case3 = np.hstack([q_p.T, x_mu_t])
    q_s_case3 = _predict(models["case3"], x_case3, device=device).T
    q_case3 = np.vstack([q_p, q_s_case3])

    # PROM-POD-AE: pure autoencoder reconstruction of the linear-PROM coefficients.
    q_pod_ae = _predict_pod_ae(models["pod_ae"], q_ref, device=device)

    # POD-DL-ROM: non-intrusive parameter/time map.
    q_pod_dl = _predict_pod_dl(models["pod_dl"], x_mu_t, device=device)

    out = {
        "case1": q_case1,
        "dd_case2": q_dd,
        "case3": q_case3,
        "pod_ae": q_pod_ae,
        "pod_dl": q_pod_dl,
    }
    for label, q in out.items():
        if q.shape != q_ref.shape:
            raise ValueError(f"{label}: expected {q_ref.shape}, got {q.shape}")
    return out


def _plot_overview(label: str, mu1: float, mu2: float, q_ref: np.ndarray, preds: dict, out_dir: Path):
    coeff_idx = np.arange(1, q_ref.shape[0] + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for key, pred in preds.items():
        style = MODEL_STYLE[key]
        coeff_rel = _relative_percent(pred, q_ref, axis=1)
        axes[0].plot(
            coeff_idx,
            coeff_rel,
            lw=1.9,
            color=style["color"],
            alpha=style["alpha"],
            label=style["long"],
        )

    q_l2 = np.linalg.norm(q_ref, axis=1)
    axes[1].plot(coeff_idx, q_l2, color="black", lw=1.8, label=r"$\|q_j^{lin}\|_2$")
    for key, pred in preds.items():
        style = MODEL_STYLE[key]
        dq_l2 = np.linalg.norm(pred - q_ref, axis=1)
        axes[1].plot(
            coeff_idx,
            dq_l2,
            lw=1.35,
            color=style["color"],
            alpha=style["alpha"],
            label=rf"$\|\Delta q_j\|_2$ {style['label']}",
        )

    energy = q_l2**2 / np.sum(q_l2**2) * 100.0
    axes[2].plot(coeff_idx, energy, color="tab:purple", lw=1.8)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_ylabel("relative coeff. error (%)")
    axes[1].set_ylabel("time L2 norm")
    axes[2].set_ylabel("energy share (%)")
    axes[2].set_xlabel("coefficient index")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[2].grid(True, alpha=0.25, which="both")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle(
        rf"{label}: $\mu=({mu1:.3f},{mu2:.4f})$ coefficient diagnostics vs linear PROM",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_dir / "overview_coeff_errors.png", dpi=220)
    plt.close(fig)


def _plot_pages(
    label: str,
    mu1: float,
    mu2: float,
    t: np.ndarray,
    q_ref: np.ndarray,
    preds: dict,
    out_dir: Path,
    coeffs_per_page: int,
):
    rel_by_model = {k: _relative_percent(v, q_ref, axis=1) for k, v in preds.items()}

    pages_dir = out_dir / "pages_png"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{label}_case1_dd_case3_podae_poddl_q_coeff_traces_{coeffs_per_page}_per_page.pdf"

    n_coeff = q_ref.shape[0]
    with PdfPages(pdf_path) as pdf:
        page = 0
        for start in range(0, n_coeff, coeffs_per_page):
            stop = min(start + coeffs_per_page, n_coeff)
            n_here = stop - start
            ncols = 2
            nrows = int(np.ceil(n_here / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows), sharex=True)
            axes = np.atleast_1d(axes).reshape(-1)
            for local, j in enumerate(range(start, stop)):
                ax = axes[local]
                ax.plot(t, q_ref[j], color="black", lw=1.8, label="linear PROM")
                for key, pred in preds.items():
                    style = MODEL_STYLE[key]
                    ax.plot(
                        t,
                        pred[j],
                        color=style["color"],
                        lw=style["lw"],
                        alpha=style["alpha"],
                        label=style["label"],
                    )
                ax.set_title(rf"$q_{{{j+1}}}$", fontsize=10)
                err_txt = (
                    f"C1 {rel_by_model['case1'][j]:.2g}% | "
                    f"DD {rel_by_model['dd_case2'][j]:.2g}% | "
                    f"C3 {rel_by_model['case3'][j]:.2g}%\n"
                    f"AE {rel_by_model['pod_ae'][j]:.2g}% | "
                    f"DL {rel_by_model['pod_dl'][j]:.2g}%"
                )
                ax.text(
                    0.015,
                    0.965,
                    err_txt,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
                )
                ax.grid(True, alpha=0.25)
                if local % ncols == 0:
                    ax.set_ylabel("coefficient")
                if local >= (nrows - 1) * ncols:
                    ax.set_xlabel("time")
            for ax in axes[n_here:]:
                ax.axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=3,
                frameon=False,
                fontsize=9,
                bbox_to_anchor=(0.5, 0.006),
            )
            fig.suptitle(
                rf"{label}: $\mu=({mu1:.3f},{mu2:.4f})$, coefficients {start+1}--{stop}",
                y=0.992,
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.955))
            pdf.savefig(fig)
            page += 1
            fig.savefig(
                pages_dir / f"{label}_coeff_page_{page:02d}_q{start+1:03d}_to_q{stop:03d}.png",
                dpi=180,
            )
            plt.close(fig)
    return pdf_path


def _write_point_tables(label: str, q_ref: np.ndarray, preds: dict, out_dir: Path):
    coeff_idx = np.arange(1, q_ref.shape[0] + 1)
    q_l2 = np.linalg.norm(q_ref, axis=1)
    energy = q_l2**2 / np.sum(q_l2**2) * 100.0
    rel = {k: _relative_percent(v, q_ref, axis=1) for k, v in preds.items()}
    abs_l2 = {k: np.linalg.norm(v - q_ref, axis=1) for k, v in preds.items()}

    coeff_csv = out_dir / "coeff_error_summary.csv"
    with coeff_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "coeff_index_1based",
                "linear_coeff_l2",
                "energy_share_percent",
                "case1_rel_percent",
                "dd_case2_rel_percent",
                "case3_rel_percent",
                "pod_ae_rel_percent",
                "pod_dl_rel_percent",
                "case1_abs_l2",
                "dd_case2_abs_l2",
                "case3_abs_l2",
                "pod_ae_abs_l2",
                "pod_dl_abs_l2",
            ]
        )
        for i, j in enumerate(coeff_idx):
            writer.writerow(
                [
                    label,
                    int(j),
                    q_l2[i],
                    energy[i],
                    rel["case1"][i],
                    rel["dd_case2"][i],
                    rel["case3"][i],
                    rel["pod_ae"][i],
                    rel["pod_dl"][i],
                    abs_l2["case1"][i],
                    abs_l2["dd_case2"][i],
                    abs_l2["case3"][i],
                    abs_l2["pod_ae"][i],
                    abs_l2["pod_dl"][i],
                ]
            )

    global_csv = out_dir / "summary.txt"
    with global_csv.open("w") as f:
        f.write(f"label: {label}\n")
        for key, pred in preds.items():
            f.write(f"{key}_global_rel_q_percent: {_relative_percent(pred, q_ref):.12g}\n")
            f.write(f"{key}_primary_rel_q_percent: {_relative_percent(pred[:10], q_ref[:10]):.12g}\n")
            f.write(f"{key}_secondary_rel_q_percent: {_relative_percent(pred[10:], q_ref[10:]):.12g}\n")
        f.write("reference: linear PROM qN.npy\n")
        f.write("case1_case3_inputs: q_1..q_10 from linear PROM\n")
        f.write("dd_case2_input: mu1, mu2, t only\n")
        f.write("pod_ae_input: linear PROM q_1..q_151 through autoencoder\n")
        f.write("pod_dl_input: mu1, mu2, t only\n")

    return coeff_csv


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_DIR / "Results_Paper" / "mlspg_prom_main",
        help="PROM campaign root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "Results_Paper"
        / "Prom_MasterANN_Diagnostic"
        / "prom151_case1_dd_case3_podae_poddl_coeff_traces_4pts",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--coeffs-per-page", type=int, default=10)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = root / "Stage3" / "models"
    case1_model, case1_ckpt = _model_from_checkpoint(model_dir / "case1_ann_ntot151_best.pt", Case1Model)
    dd_model, dd_ckpt = _model_from_checkpoint(
        model_dir / "master_ann_mu_t_to_qtot_ntot151_best.pt",
        ROMDataDrivenModel,
    )
    case3_model, case3_ckpt = _model_from_checkpoint(model_dir / "case3_ann_ntot151_best.pt", Case3Model)
    pod_ae_model, pod_ae_ckpt = _pod_ae_from_checkpoint(model_dir / "prom_pod_ae_ntot151_best.pt")
    pod_dl_model, pod_dl_ckpt = _pod_dl_from_checkpoint(model_dir / "pod_dl_data_driven_ntot151_best.pt")
    models = {
        "case1": case1_model,
        "dd": dd_model,
        "case3": case3_model,
        "pod_ae": pod_ae_model,
        "pod_dl": pod_dl_model,
    }

    print(f"[coeff-diagnostic] root:       {root}")
    print(f"[coeff-diagnostic] output:     {out_root}")
    print(f"[coeff-diagnostic] device:     {device}")
    print(f"[coeff-diagnostic] case1:      {case1_ckpt.get('model_name', 'case1_ann_ntot151_best.pt')}")
    print(f"[coeff-diagnostic] dd/case2:   {dd_ckpt.get('model_name', 'master_ann_mu_t_to_qtot_ntot151_best.pt')}")
    print(f"[coeff-diagnostic] case3:      {case3_ckpt.get('model_name', 'case3_ann_ntot151_best.pt')}")
    print(f"[coeff-diagnostic] pod-ae:     {pod_ae_ckpt.get('model_name', 'prom_pod_ae_ntot151_best.pt')}")
    print(f"[coeff-diagnostic] pod-dl:     {pod_dl_ckpt.get('model_name', 'pod_dl_data_driven_ntot151_best.pt')}")

    all_rows = []
    for label, mu1, mu2 in POINTS:
        q_ref = _load_linear_qn(root, mu1, mu2)
        t = np.arange(q_ref.shape[1], dtype=np.float64) * 0.05
        preds = _predict_all(q_ref=q_ref, mu1=mu1, mu2=mu2, t=t, models=models, device=device)

        point_dir = out_root / _safe_name(label)
        point_dir.mkdir(parents=True, exist_ok=True)
        _plot_overview(label, mu1, mu2, q_ref, preds, point_dir)
        pdf_path = _plot_pages(label, mu1, mu2, t, q_ref, preds, point_dir, args.coeffs_per_page)
        _write_point_tables(label, q_ref, preds, point_dir)

        row = {"label": label, "mu1": mu1, "mu2": mu2, "pdf": str(pdf_path)}
        for key, pred in preds.items():
            row[f"{key}_global_rel_q_percent"] = float(_relative_percent(pred, q_ref))
            row[f"{key}_primary_rel_q_percent"] = float(_relative_percent(pred[:10], q_ref[:10]))
            row[f"{key}_secondary_rel_q_percent"] = float(_relative_percent(pred[10:], q_ref[10:]))
        all_rows.append(row)
        print(
            f"[coeff-diagnostic] {label}: "
            f"C1={row['case1_global_rel_q_percent']:.4f}% | "
            f"DD/Case2={row['dd_case2_global_rel_q_percent']:.4f}% | "
            f"C3={row['case3_global_rel_q_percent']:.4f}% | "
            f"POD-AE={row['pod_ae_global_rel_q_percent']:.4f}% | "
            f"POD-DL={row['pod_dl_global_rel_q_percent']:.4f}%"
        )

    summary_csv = out_root / "all_points_global_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[coeff-diagnostic] summary: {summary_csv}")


if __name__ == "__main__":
    main()
