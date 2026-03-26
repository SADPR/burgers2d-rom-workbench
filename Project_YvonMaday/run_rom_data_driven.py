#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run non-intrusive ROM (data-driven qN model) against HDM.

Mapping:
    qN = G(mu1, mu2, t)
    u_hat(t) = u_ref + V_tot @ qN(t)

The model checkpoint is produced by:
    stage3_perform_training_rom_data_driven.py
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0
from burgers.core import load_or_compute_snaps, plot_snaps

try:
    from project_layout import (
        RUNS_DATA_DRIVEN_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        resolve_stage3_model,
        write_kv_txt,
    )
except ModuleNotFoundError:
    from .project_layout import (
        RUNS_DATA_DRIVEN_DIR,
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


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        return self.fc6(x)


class ROMDataDrivenModel(nn.Module):
    """Input: (mu1, mu2, t), output: qN."""

    def __init__(self, n_tot):
        super().__init__()
        in_dim = 3
        self.scaler = Scaler(np.zeros((1, in_dim)), np.ones((1, in_dim)))
        self.core = CoreMLP(in_dim, n_tot)
        self.unscaler = Unscaler(np.zeros((1, n_tot)), np.ones((1, n_tot)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


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
        print("[DataDriven] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if dev not in ("cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return dev


def _load_basis_and_reference():
    basis_path = resolve_stage1_artifact("basis.npy")
    uref_path = resolve_stage1_artifact("u_ref.npy")

    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Missing basis file: {basis_path}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got shape {basis.shape}")

    if os.path.exists(uref_path):
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(basis.shape[0], dtype=np.float64)

    if u_ref.size != basis.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {basis.shape[0]} from basis rows."
        )

    return basis, u_ref, basis_path, uref_path


def _load_rom_data_driven_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}\n"
            "Run stage3_perform_training_rom_data_driven.py first."
        )

    ckpt = torch.load(model_path, map_location=device)
    n_tot = int(ckpt["n_tot"])
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"rom_data_driven checkpoint in_dim={in_dim}, expected 3")

    model = ROMDataDrivenModel(n_tot=n_tot).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, n_tot, ckpt


def main(
    mu_test=(4.56, 0.019),
    total_modes=None,
    device="auto",
    make_plots=True,
    save_hdm_reference=False,
):
    mu_test = [float(mu_test[0]), float(mu_test[1])]

    ensure_layout_dirs()
    os.makedirs(RUNS_DATA_DRIVEN_DIR, exist_ok=True)
    set_latex_plot_style()

    runtime_device = _resolve_device(device)
    model_path = resolve_stage3_model("rom_data_driven_model.pt")
    model, model_ntot, ckpt = _load_rom_data_driven_model(model_path, device=runtime_device)

    basis_all, u_ref, basis_path, uref_path = _load_basis_and_reference()
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
            f"Requested total_modes={total_modes}, but rom_data_driven model outputs only {model_ntot} modes."
        )

    Vtot = basis_all[:, :total_modes]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != Vtot.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {Vtot.shape[0]} from basis rows."
        )

    snap_folder = _select_snap_folder(PROJECT_ROOT)
    os.makedirs(snap_folder, exist_ok=True)

    print(f"[DataDriven] device = {runtime_device}")
    print(f"[DataDriven] model = {model_path}")
    print(f"[DataDriven] basis = {basis_path} (available={basis_available}, using={total_modes})")
    print(f"[DataDriven] u_ref = {uref_path if os.path.exists(uref_path) else 'zeros'}")
    print(f"[DataDriven] model_ntot = {model_ntot}")
    print(f"[DataDriven] snap_folder = {snap_folder}")

    T = NUM_STEPS + 1
    t_vec = DT * np.arange(T, dtype=np.float64)
    x_raw = np.column_stack([
        np.full((T,), mu_test[0], dtype=np.float32),
        np.full((T,), mu_test[1], dtype=np.float32),
        t_vec.astype(np.float32),
    ])

    t0 = time.time()
    with torch.no_grad():
        x_t = torch.from_numpy(x_raw).to(runtime_device)
        qn_full = model(x_t).detach().cpu().numpy().astype(np.float64).T
    infer_elapsed = time.time() - t0

    qn = qn_full[:total_modes, :]
    rom_snaps = u_ref[:, None] + Vtot @ qn

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
    run_tag = f"rom_data_driven_{tag}_ntot{total_modes}"
    out_dir = os.path.join(RUNS_DATA_DRIVEN_DIR, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "mu.npy"), np.asarray(mu_test, dtype=np.float64))
    np.save(os.path.join(out_dir, "t.npy"), t_vec)
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
            label="Data-Driven ROM",
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

    summary_path = os.path.join(out_dir, "rom_data_driven_summary.txt")
    write_kv_txt(
        summary_path,
        [
            ("mu_test", mu_test),
            ("method", "nonintrusive_data_driven"),
            ("device", runtime_device),
            ("model_path", model_path),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path if os.path.exists(uref_path) else "zeros"),
            ("dataset_backend", ckpt.get("dataset_backend", "unknown")),
            ("dataset_ntot", ckpt.get("dataset_ntot", "unknown")),
            ("model_ntot", model_ntot),
            ("total_modes_used", total_modes),
            ("inference_time_s", infer_elapsed),
            ("relative_error_percent", rel_err),
            ("output_dir", out_dir),
            ("snap_folder", snap_folder),
            ("plot_output", out_png if make_plots else "not_requested"),
        ],
    )

    print(f"[DataDriven] inference_time = {infer_elapsed:.3e} s")
    print(f"[DataDriven] relative error vs HDM: {rel_err:.2f}%")
    print(f"[DataDriven] outputs: {out_dir}")
    print(f"[DataDriven] summary: {summary_path}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run non-intrusive data-driven ROM (qN = G(mu1, mu2, t)) against HDM."
    )
    parser.add_argument("--mu1", type=float, default=4.56, help="First parameter value")
    parser.add_argument("--mu2", type=float, default=0.019, help="Second parameter value")
    parser.add_argument(
        "--total-modes",
        type=int,
        default=None,
        help="Number of retained POD modes for reconstruction. Default: model n_tot.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch runtime device",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip HDM-vs-ROM plotting")
    parser.add_argument("--save-hdm-reference", action="store_true", help="Also save hdm_snaps.npy")
    return parser


def cli(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    main(
        mu_test=(args.mu1, args.mu2),
        total_modes=args.total_modes,
        device=args.device,
        make_plots=not args.no_plot,
        save_hdm_reference=args.save_hdm_reference,
    )


if __name__ == "__main__":
    cli()
