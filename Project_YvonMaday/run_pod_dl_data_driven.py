#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run non-intrusive POD-DL-ROM against HDM.

Mapping:
    z = phi(mu1, mu2, t)
    qN = D(z)
    u_hat = u_ref + V_tot qN
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0
from burgers.core import load_or_compute_snaps, plot_snaps
try:
    from pod_dl_data_driven_common import PODDLDataDrivenModel
except ModuleNotFoundError:
    from .pod_dl_data_driven_common import PODDLDataDrivenModel
try:
    from project_layout import (
        RUNS_POD_DL_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        resolve_stage3_model,
        write_kv_txt,
    )
except ModuleNotFoundError:
    from .project_layout import (
        RUNS_POD_DL_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        resolve_stage3_model,
        write_kv_txt,
    )


def set_latex_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "figure.figsize": (12, 8),
    })


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _localize_project_path(path_like):
    if path_like is None:
        return None
    path = os.path.abspath(os.path.expanduser(str(path_like)))
    if os.path.exists(path):
        return path
    marker = f"{os.sep}Project_YvonMaday{os.sep}"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        candidate = os.path.join(THIS_DIR, suffix)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


def _select_snap_folder(project_root):
    candidates = [
        os.path.join(project_root, "Results", "param_snaps"),
        os.path.join(project_root, "param_snaps"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _resolve_device(device):
    dev = str(device).strip().lower()
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        print("[POD-DL] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if dev not in ("cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return dev


def _load_basis_and_reference(ckpt=None):
    ckpt = {} if ckpt is None else ckpt
    basis_candidates = []
    if ckpt.get("basis_file", None):
        basis_candidates.append(_localize_project_path(ckpt["basis_file"]))
    basis_candidates.append(resolve_stage1_artifact("basis.npy"))

    basis_path = None
    for candidate in basis_candidates:
        if os.path.exists(candidate):
            basis_path = candidate
            break
    if basis_path is None:
        raise FileNotFoundError(f"Missing basis file. Checked: {basis_candidates}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got shape {basis.shape}")

    uref_candidates = []
    if ckpt.get("u_ref_file", None):
        uref_candidates.append(_localize_project_path(ckpt["u_ref_file"]))
    uref_candidates.append(resolve_stage1_artifact("u_ref.npy"))

    uref_path = None
    for candidate in uref_candidates:
        if os.path.exists(candidate):
            uref_path = candidate
            break

    if uref_path is not None:
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(basis.shape[0], dtype=np.float64)
        uref_path = "zeros"

    if u_ref.size != basis.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {basis.shape[0]} from basis rows."
        )

    return basis, u_ref, basis_path, uref_path


def _load_pod_dl_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}\n"
            "Run stage3_perform_training_pod_dl_data_driven.py first."
        )

    ckpt = torch.load(model_path, map_location=device)
    q_dim = int(ckpt["q_dim"])
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"POD-DL checkpoint in_dim={in_dim}, expected 3")

    model = PODDLDataDrivenModel(
        q_dim=q_dim,
        latent_dim=int(ckpt["latent_dim"]),
        encoder_hidden_dims=tuple(int(v) for v in ckpt["encoder_hidden_dims"]),
        decoder_hidden_dims=tuple(int(v) for v in ckpt["decoder_hidden_dims"]),
        dynamics_hidden_dims=tuple(int(v) for v in ckpt["dynamics_hidden_dims"]),
        activation=str(ckpt.get("activation", "elu")),
        x_scaling=str(ckpt.get("x_scaling", "zscore")),
        q_scaling=str(ckpt.get("q_scaling", "zscore")),
        x_stats=None,
        q_stats=None,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, q_dim, ckpt


def main(
    mu_test=(4.56, 0.019),
    total_modes=None,
    device="auto",
    make_plots=True,
    save_hdm_reference=False,
    model_name="pod_dl_data_driven_model.pt",
    model_path_override=None,
    output_root=None,
):
    mu_test = [float(mu_test[0]), float(mu_test[1])]

    ensure_layout_dirs()
    if output_root is None:
        output_root = RUNS_POD_DL_DIR
    output_root = os.path.abspath(os.path.expanduser(str(output_root)))
    os.makedirs(output_root, exist_ok=True)
    set_latex_plot_style()

    runtime_device = _resolve_device(device)
    if model_path_override is None:
        model_name = str(model_name).strip()
        if len(model_name) == 0:
            raise ValueError("--model-name cannot be empty.")
        if not model_name.endswith(".pt"):
            model_name = f"{model_name}.pt"
        model_path = resolve_stage3_model(model_name)
    else:
        model_path = os.path.abspath(model_path_override)
        model_name = os.path.basename(model_path)
    model, model_ntot, ckpt = _load_pod_dl_model(model_path, device=runtime_device)

    basis_all, u_ref, basis_path, uref_path = _load_basis_and_reference(ckpt)
    basis_available = int(basis_all.shape[1])

    if total_modes is None:
        total_modes = model_ntot
    else:
        total_modes = int(total_modes)

    if total_modes < 1:
        raise ValueError("total_modes must be >= 1.")
    if total_modes > basis_available:
        raise ValueError(
            f"Requested total_modes={total_modes}, but basis has only {basis_available} modes."
        )
    if total_modes > model_ntot:
        raise ValueError(
            f"Requested total_modes={total_modes}, but POD-DL model outputs only {model_ntot} modes."
        )

    vtot = basis_all[:, :total_modes]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != vtot.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {vtot.shape[0]} from basis rows."
        )

    snap_folder = _select_snap_folder(PROJECT_ROOT)
    os.makedirs(snap_folder, exist_ok=True)

    print(f"[POD-DL] device = {runtime_device}")
    print(f"[POD-DL] model = {model_path}")
    print(f"[POD-DL] basis = {basis_path} (available={basis_available}, using={total_modes})")
    print(f"[POD-DL] u_ref = {uref_path}")
    print(f"[POD-DL] model_ntot = {model_ntot}")
    print(f"[POD-DL] snap_folder = {snap_folder}")

    tsize = NUM_STEPS + 1
    t_vec = DT * np.arange(tsize, dtype=np.float64)
    x_raw = np.column_stack([
        np.full((tsize,), mu_test[0], dtype=np.float32),
        np.full((tsize,), mu_test[1], dtype=np.float32),
        t_vec.astype(np.float32),
    ])

    t0 = time.time()
    with torch.no_grad():
        x_t = torch.from_numpy(x_raw).to(runtime_device)
        z_pred = model.predict_z_from_x(x_t).detach().cpu().numpy().astype(np.float64).T
        qn_full = model.predict_q_from_x(x_t).detach().cpu().numpy().astype(np.float64).T
    infer_elapsed = time.time() - t0

    qn = qn_full[:total_modes, :]
    rom_snaps = u_ref[:, None] + vtot @ qn

    hdm_snaps = load_or_compute_snaps(
        mu=mu_test,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        w0=w0,
        dt=DT,
        num_steps=NUM_STEPS,
        snap_folder=snap_folder,
    )

    if hdm_snaps.shape != rom_snaps.shape:
        raise RuntimeError(
            f"Shape mismatch: HDM {hdm_snaps.shape} vs reconstructed {rom_snaps.shape}."
        )

    rel_err = 100.0 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)

    tag = _safe_mu_tag(mu_test)
    run_tag = f"pod_dl_data_driven_{tag}_ntot{total_modes}_nz{z_pred.shape[0]}"
    out_dir = os.path.join(output_root, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "mu.npy"), np.asarray(mu_test, dtype=np.float64))
    np.save(os.path.join(out_dir, "t.npy"), t_vec)
    np.save(os.path.join(out_dir, "z.npy"), z_pred)
    np.save(os.path.join(out_dir, "qN.npy"), qn)
    np.save(os.path.join(out_dir, "rom_snaps.npy"), rom_snaps)
    if save_hdm_reference:
        np.save(os.path.join(out_dir, "hdm_snaps.npy"), hdm_snaps)

    out_png = os.path.join(out_dir, "hdm_vs_rom.png")
    if make_plots:
        plot_steps = list(range(0, NUM_STEPS + 1, 100))
        if NUM_STEPS not in plot_steps:
            plot_steps.append(NUM_STEPS)

        fig, ax1, ax2 = plot_snaps(
            GRID_X,
            GRID_Y,
            hdm_snaps,
            plot_steps,
            label="HDM",
            color="black",
            linewidth=2.8,
            linestyle="solid",
        )
        plot_snaps(
            GRID_X,
            GRID_Y,
            rom_snaps,
            plot_steps,
            label="POD-DL (data-driven)",
            fig_ax=(fig, ax1, ax2),
            color="blue",
            linewidth=1.8,
            linestyle="solid",
        )
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    summary_path = os.path.join(out_dir, "pod_dl_data_driven_summary.txt")
    write_kv_txt(
        summary_path,
        [
            ("mu_test", mu_test),
            ("method", "pod_dl_data_driven_nonintrusive"),
            ("device", runtime_device),
            ("model_name", model_name),
            ("model_path", model_path),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path),
            ("dataset_backend", ckpt.get("dataset_backend", "unknown")),
            ("dataset_ntot", ckpt.get("dataset_ntot", "unknown")),
            ("model_ntot", model_ntot),
            ("total_modes_used", total_modes),
            ("latent_dim", int(z_pred.shape[0])),
            ("inference_time_s", infer_elapsed),
            ("relative_error_percent", rel_err),
            ("output_dir", out_dir),
            ("snap_folder", snap_folder),
        ],
    )

    print(f"[POD-DL] inference_time = {infer_elapsed:.3e} s")
    print(f"[POD-DL] relative error vs HDM: {rel_err:.2f}%")
    print(f"[POD-DL] saved run folder: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run POD-DL non-intrusive ROM.")
    parser.add_argument("--mu1", type=float, default=4.56)
    parser.add_argument("--mu2", type=float, default=0.019)
    parser.add_argument("--total-modes", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-hdm-reference", action="store_true")
    parser.add_argument(
        "--model-name",
        type=str,
        default="pod_dl_data_driven_model.pt",
        help="Checkpoint filename under Results/Stage3/models (used if --model-path is not set).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument("--output-root", type=str, default=None, help="Optional output root for run folders.")
    args = parser.parse_args()

    main(
        mu_test=(args.mu1, args.mu2),
        total_modes=args.total_modes,
        device=args.device,
        make_plots=(not args.no_plots),
        save_hdm_reference=bool(args.save_hdm_reference),
        model_name=args.model_name,
        model_path_override=args.model_path,
        output_root=args.output_root,
    )
