#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TEST POD-DL RECONSTRUCTION

Loads the stage3 autoencoder checkpoint, reconstructs snapshots for a target
parameter, and compares against HDM (and optionally POD baseline).
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
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > eps, half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, x):
        return (x - self.center) / self.half_range


class Unscaler(nn.Module):
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > eps, half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, y):
        return y * self.half_range + self.center


def build_mlp(in_dim, hidden_dims, out_dim):
    dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PODDLAutoencoder(nn.Module):
    def __init__(self, q_min, q_max, latent_dim, hidden_dims=(192, 96, 48)):
        super().__init__()
        q_dim = int(np.asarray(q_min).reshape(-1).size)
        z_dim = int(latent_dim)

        self.scaler = Scaler(np.asarray(q_min)[None, :], np.asarray(q_max)[None, :])
        self.encoder = build_mlp(q_dim, hidden_dims, z_dim)
        self.decoder = build_mlp(z_dim, tuple(reversed(hidden_dims)), q_dim)
        self.unscaler = Unscaler(np.asarray(q_min)[None, :], np.asarray(q_max)[None, :])

    def forward(self, q_raw):
        q_norm = self.scaler(q_raw)
        z = self.encoder(q_norm)
        q_norm_hat = self.decoder(z)
        q_raw_hat = self.unscaler(q_norm_hat)
        return q_raw_hat


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


def load_total_modes(stage2_metadata_file, fallback):
    if not os.path.exists(stage2_metadata_file):
        return int(fallback), None

    try:
        meta = np.load(stage2_metadata_file, allow_pickle=True)
    except Exception:
        return int(fallback), None

    if "total_modes" in meta.files:
        return int(np.asarray(meta["total_modes"]).reshape(-1)[0]), stage2_metadata_file
    return int(fallback), stage2_metadata_file


def main(
    target_mu=(4.56, 0.019),
    basis_file=os.path.join(script_dir, "basis.npy"),
    model_file=os.path.join(script_dir, "pod_dl_model", "pod_dl_autoencoder.pt"),
    stage2_metadata_file=os.path.join(script_dir, "stage2_projection_metadata.npz"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
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
        raise FileNotFoundError(f"Missing POD-DL checkpoint: {model_file}. Run stage3 first.")
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Missing basis file: {basis_file}. Run stage1 first.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mu1 = float(target_mu[0])
    mu2 = float(target_mu[1])
    mu = [mu1, mu2]

    print("\n====================================================")
    print("          STAGE 4: TEST POD-DL MODEL")
    print("====================================================")
    print(f"[STAGE4] target mu = [{mu1:.3f}, {mu2:.4f}]")
    print(f"[STAGE4] device = {device}")

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    n_dofs, n_basis = basis.shape

    checkpoint = torch.load(model_file, map_location="cpu")

    q_dim = int(checkpoint["q_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    hidden_dims = tuple(checkpoint.get("hidden_dims", (32, 64, 128, 256, 256)))
    checkpoint_use_u_ref = bool(checkpoint.get("use_u_ref", False))
    checkpoint_u_ref = checkpoint.get("u_ref", None)

    total_modes_meta, meta_source = load_total_modes(stage2_metadata_file, fallback=q_dim)
    if total_modes_meta != q_dim:
        raise ValueError(
            "Inconsistent reduced size between stage2 and stage3: "
            f"stage2 total_modes={total_modes_meta}, stage3 q_dim={q_dim}."
        )

    if n_basis < q_dim:
        raise ValueError(
            "Basis has insufficient columns for checkpoint q_dim: "
            f"basis={n_basis}, q_dim={q_dim}."
        )

    basis_modes = basis[:, :q_dim]

    model = PODDLAutoencoder(
        q_min=np.zeros(q_dim, dtype=np.float32),
        q_max=np.ones(q_dim, dtype=np.float32),
        latent_dim=latent_dim,
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
        raise RuntimeError(
            f"State mismatch: HDM has {hdm_snap.shape[0]} dofs, basis has {n_dofs}."
        )

    t0 = time.time()
    q_true = basis_modes.T @ (hdm_snap - u_ref[:, None])
    with torch.no_grad():
        q_pred = model(torch.from_numpy(q_true.T.astype(np.float32)).to(device)).cpu().numpy().T
    pod_dl_recon = u_ref[:, None] + basis_modes @ q_pred
    elapsed_recon = time.time() - t0

    if compare_pod:
        pod_recon = u_ref[:, None] + basis_modes @ q_true
    else:
        pod_recon = None

    hdm_norm = np.linalg.norm(hdm_snap)
    if hdm_norm <= 0.0:
        rel_dl = np.nan
        rel_pod = np.nan
    else:
        rel_dl = float(np.linalg.norm(hdm_snap - pod_dl_recon) / hdm_norm)
        rel_pod = (
            float(np.linalg.norm(hdm_snap - pod_recon) / hdm_norm)
            if pod_recon is not None
            else None
        )

    q_rel_pct = None
    q_norm = np.linalg.norm(q_true)
    if q_norm > 0.0:
        q_rel_pct = float(100.0 * np.linalg.norm(q_true - q_pred) / q_norm)

    print(f"[STAGE4] HDM load/solve time: {elapsed_hdm:.3e}s")
    print(f"[STAGE4] POD-DL reconstruction time: {elapsed_recon:.3e}s")
    print(f"[STAGE4] POD-DL global relative error: {100.0 * rel_dl:.4f}%")
    if rel_pod is not None:
        print(f"[STAGE4] POD baseline global relative error: {100.0 * rel_pod:.4f}%")
    if q_rel_pct is not None:
        print(f"[STAGE4] q-space relative reconstruction error: {q_rel_pct:.4f}%")

    pod_dl_recon_file = os.path.join(
        output_dir,
        f"pod_dl_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(pod_dl_recon_file, pod_dl_recon)

    pod_recon_file = None
    if pod_recon is not None:
        pod_recon_file = os.path.join(
            output_dir,
            f"pod_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
        )
        np.save(pod_recon_file, pod_recon)

    snaps_to_plot = range(0, num_steps + 1, 100)
    fig, ax1, ax2 = plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snap,
        snaps_to_plot,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )

    plot_snaps(
        GRID_X,
        GRID_Y,
        pod_dl_recon,
        snaps_to_plot,
        label="POD-DL",
        fig_ax=(fig, ax1, ax2),
        color="tab:blue",
        linewidth=2.0,
        linestyle="solid",
    )

    if pod_recon is not None:
        plot_snaps(
            GRID_X,
            GRID_Y,
            pod_recon,
            snaps_to_plot,
            label="POD",
            fig_ax=(fig, ax1, ax2),
            color="gray",
            linewidth=1.8,
            linestyle="dashed",
        )

    fig.suptitle(rf"$\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$", y=0.98)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    plt.tight_layout()

    fig_file = os.path.join(
        output_dir,
        f"pod_dl_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_file = os.path.join(
        output_dir,
        f"stage4_test_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_test_autoencoder.py"),
                    ("target_mu1", mu1),
                    ("target_mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("model_file", model_file),
                    ("stage2_metadata_file", meta_source),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("q_dim", q_dim),
                    ("latent_dim", latent_dim),
                    ("hidden_dims", hidden_dims),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("checkpoint_use_u_ref", checkpoint_use_u_ref),
                    ("compare_pod", compare_pod),
                    ("device", device),
                ],
            ),
            (
                "metrics",
                [
                    ("pod_dl_global_rel_error", rel_dl),
                    ("pod_dl_global_rel_error_percent", None if np.isnan(rel_dl) else 100.0 * rel_dl),
                    ("pod_global_rel_error", rel_pod),
                    ("pod_global_rel_error_percent", None if rel_pod is None else 100.0 * rel_pod),
                    ("q_rel_error_percent", q_rel_pct),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("hdm_load_or_solve", elapsed_hdm),
                    ("pod_dl_reconstruct", elapsed_recon),
                ],
            ),
            (
                "outputs",
                [
                    ("pod_dl_reconstruction_npy", pod_dl_recon_file),
                    ("pod_reconstruction_npy", pod_recon_file),
                    ("comparison_plot_png", fig_file),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )

    print(f"[STAGE4] Saved POD-DL reconstruction: {pod_dl_recon_file}")
    if pod_recon_file is not None:
        print(f"[STAGE4] Saved POD reconstruction: {pod_recon_file}")
    print(f"[STAGE4] Saved comparison plot: {fig_file}")
    print(f"[STAGE4] Saved summary: {report_file}")


if __name__ == "__main__":
    main()
