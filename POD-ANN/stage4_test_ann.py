#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TEST POD-ANN RECONSTRUCTION

Loads the stage3 ANN checkpoint, reconstructs snapshots for a target parameter,
and compares against HDM (and optionally POD baseline).
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.35,
            "figure.figsize": (12, 8),
        }
    )


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(32, 64, 128, 256, 256)):
        super().__init__()
        dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PODANNModel(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims=(32, 64, 128, 256, 256)):
        super().__init__()
        in_dim = int(np.asarray(x_mean).reshape(-1).size)
        out_dim = int(np.asarray(y_mean).reshape(-1).size)
        self.scaler = Scaler(np.asarray(x_mean)[None, :], np.asarray(x_std)[None, :])
        self.core = CoreMLP(in_dim, out_dim, hidden_dims=hidden_dims)
        self.unscaler = Unscaler(np.asarray(y_mean)[None, :], np.asarray(y_std)[None, :])

    def forward(self, x_raw):
        x_norm = self.scaler(x_raw)
        y_norm = self.core(x_norm)
        y_raw = self.unscaler(y_norm)
        return y_raw


def resolve_u_ref(uref_mode, uref_file, checkpoint_u_ref, checkpoint_use_u_ref, expected_size):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    checkpoint_u_ref = (
        None
        if checkpoint_u_ref is None
        else np.asarray(checkpoint_u_ref, dtype=np.float64).reshape(-1)
    )

    if checkpoint_u_ref is not None and checkpoint_u_ref.size != expected_size:
        raise ValueError(
            f"Checkpoint u_ref size mismatch: got {checkpoint_u_ref.size}, expected {expected_size}."
        )

    if mode == "off":
        return False, np.zeros(expected_size, dtype=np.float64), "zeros(off)"

    if uref_file is not None and os.path.exists(uref_file):
        u_ref = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
        if u_ref.size != expected_size:
            raise ValueError(
                f"u_ref size mismatch in '{uref_file}': got {u_ref.size}, expected {expected_size}."
            )
        return True, u_ref, uref_file

    if mode == "on":
        if checkpoint_u_ref is not None:
            return True, checkpoint_u_ref, "checkpoint"
        raise FileNotFoundError(
            "uref_mode='on' but no explicit u_ref file and checkpoint has no valid u_ref."
        )

    if bool(checkpoint_use_u_ref) and checkpoint_u_ref is not None:
        return True, checkpoint_u_ref, "checkpoint"

    return False, np.zeros(expected_size, dtype=np.float64), "zeros(auto)"


def reconstruct_snapshot_with_ann(snapshot, u_ref, u_p, u_s, ann_model, device):
    snapshot = np.asarray(snapshot, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)

    centered = snapshot - u_ref[:, None]
    q_p = u_p.T @ centered

    with torch.no_grad():
        q_p_t = torch.from_numpy(q_p.T.astype(np.float32)).to(device)
        q_s_pred = ann_model(q_p_t).detach().cpu().numpy().T

    return u_ref[:, None] + u_p @ q_p + u_s @ q_s_pred


def reconstruct_snapshot_with_pod(snapshot, u_ref, basis):
    snapshot = np.asarray(snapshot, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    centered = snapshot - u_ref[:, None]
    q = basis.T @ centered
    return u_ref[:, None] + basis @ q


def main(
    target_mu=(4.56, 0.019),
    basis_file=os.path.join(script_dir, "basis.npy"),
    model_file=os.path.join(script_dir, "pod_ann_model", "case1_model.pt"),
    uref_file=None,
    output_dir=os.path.join(script_dir, "stage4_results"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    compare_pod=True,
    device=None,
):
    set_latex_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Missing ANN checkpoint: {model_file}. Run stage3 first.")
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Missing basis file: {basis_file}. Run stage1 first.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mu1 = float(target_mu[0])
    mu2 = float(target_mu[1])
    mu = [mu1, mu2]

    print("\n====================================================")
    print("          STAGE 4: TEST POD-ANN MODEL")
    print("====================================================")
    print(f"[STAGE4] target mu = [{mu1:.3f}, {mu2:.4f}]")
    print(f"[STAGE4] device = {device}")

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    n_dofs, _ = basis.shape

    checkpoint = torch.load(model_file, map_location="cpu")

    n_p = int(checkpoint["n_p"])
    n_s = int(checkpoint["n_s"])
    hidden_dims = tuple(checkpoint.get("hidden_dims", (32, 64, 128, 256, 256)))
    checkpoint_use_u_ref = bool(checkpoint.get("use_u_ref", False))
    checkpoint_u_ref = checkpoint.get("u_ref", None)

    if basis.shape[1] < n_p + n_s:
        raise ValueError(
            "Basis has insufficient columns for checkpoint dimensions: "
            f"basis={basis.shape[1]}, required={n_p + n_s}."
        )

    u_p = basis[:, :n_p]
    u_s = basis[:, n_p:n_p + n_s]

    model = PODANNModel(
        x_mean=np.zeros(n_p, dtype=np.float32),
        x_std=np.ones(n_p, dtype=np.float32),
        y_mean=np.zeros(n_s, dtype=np.float32),
        y_std=np.ones(n_s, dtype=np.float32),
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    use_u_ref, u_ref, u_ref_source = resolve_u_ref(
        uref_mode=uref_mode,
        uref_file=uref_file,
        checkpoint_u_ref=checkpoint_u_ref,
        checkpoint_use_u_ref=checkpoint_use_u_ref,
        expected_size=n_dofs,
    )

    w0 = np.asarray(W0, dtype=np.float64).copy()
    t0 = time.time()
    hdm_snap = np.asarray(
        load_or_compute_snaps(mu, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )
    elapsed_hdm = time.time() - t0

    if hdm_snap.shape[0] != n_dofs:
        raise RuntimeError(f"State size mismatch: hdm={hdm_snap.shape[0]}, basis={n_dofs}")

    print(
        f"[STAGE4] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )

    t0 = time.time()
    pod_ann_reconstructed = reconstruct_snapshot_with_ann(
        snapshot=hdm_snap,
        u_ref=u_ref,
        u_p=u_p,
        u_s=u_s,
        ann_model=model,
        device=device,
    )
    elapsed_ann = time.time() - t0

    pod_reconstructed = None
    elapsed_pod = None
    if compare_pod:
        t0 = time.time()
        u_full = np.hstack((u_p, u_s))
        pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, u_ref, u_full)
        elapsed_pod = time.time() - t0

    hdm_norm = np.linalg.norm(hdm_snap)
    if hdm_norm > 0.0:
        pod_ann_error = np.linalg.norm(hdm_snap - pod_ann_reconstructed) / hdm_norm
        pod_error = (
            np.linalg.norm(hdm_snap - pod_reconstructed) / hdm_norm
            if pod_reconstructed is not None
            else None
        )
    else:
        pod_ann_error = np.nan
        pod_error = np.nan if pod_reconstructed is not None else None

    print(f"[STAGE4] POD-ANN relative error: {100.0 * pod_ann_error:.4f}%")
    if pod_error is not None:
        print(f"[STAGE4] POD relative error: {100.0 * pod_error:.4f}%")

    pod_ann_file_path = os.path.join(
        output_dir,
        f"pod_ann_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(pod_ann_file_path, pod_ann_reconstructed)

    pod_file_path = None
    if pod_reconstructed is not None:
        pod_file_path = os.path.join(
            output_dir,
            f"pod_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
        )
        np.save(pod_file_path, pod_reconstructed)

    inds_to_plot = range(0, num_steps + 1, 100)
    fig, ax1, ax2 = plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snap,
        inds_to_plot,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        GRID_X,
        GRID_Y,
        pod_ann_reconstructed,
        inds_to_plot,
        label="POD-ANN",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )
    if pod_reconstructed is not None:
        plot_snaps(
            GRID_X,
            GRID_Y,
            pod_reconstructed,
            inds_to_plot,
            label="POD",
            fig_ax=(fig, ax1, ax2),
            color="#0a8f5a",
            linewidth=1.8,
            linestyle="dashed",
        )

    fig.suptitle(rf"$\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$", y=0.98)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    plt.tight_layout()

    plot_file = os.path.join(
        output_dir,
        f"pod_ann_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_file = os.path.join(
        output_dir,
        f"stage4_test_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_test_ann.py"),
                    ("mu1", mu1),
                    ("mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("model_file", model_file),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("compare_pod", bool(compare_pod)),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("device", device),
                ],
            ),
            (
                "ann_model",
                [
                    ("n_p", n_p),
                    ("n_s", n_s),
                    ("hidden_dims", hidden_dims),
                    ("checkpoint_use_u_ref", checkpoint_use_u_ref),
                ],
            ),
            (
                "errors",
                [
                    ("pod_ann_relative_l2", pod_ann_error),
                    ("pod_ann_relative_percent", 100.0 * pod_ann_error),
                    ("pod_relative_l2", pod_error),
                    (
                        "pod_relative_percent",
                        (100.0 * pod_error) if pod_error is not None else None,
                    ),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("load_or_compute_hdm", elapsed_hdm),
                    ("ann_reconstruction", elapsed_ann),
                    ("pod_reconstruction", elapsed_pod),
                    (
                        "total",
                        elapsed_hdm + elapsed_ann + (elapsed_pod if elapsed_pod is not None else 0.0),
                    ),
                ],
            ),
            (
                "outputs",
                [
                    ("pod_ann_reconstruction_npy", pod_ann_file_path),
                    ("pod_reconstruction_npy", pod_file_path),
                    ("comparison_plot_png", plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE4] Summary saved: {summary_file}")


if __name__ == "__main__":
    main()

