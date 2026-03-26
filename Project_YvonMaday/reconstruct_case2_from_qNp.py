
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reconstruct_case2_from_qNp.py

Reconstruction-only evaluation for Case 2 on the 250x250 setup.

Given:
  - HDM snapshots U_HDM(mu)
  - PROM-consistent primary coefficients q(t) from an n_tot PROM run:
        qN_p.npy  (n, T)
        t.npy     (T,)
        mu.npy    (2,)
  - Case 2 model: qbar_hat = N(mu1, mu2, t)   (scaler embedded)

We reconstruct:
  U_hat(t) = u_ref + V q(t) + Vbar qbar_hat(t)

Then compute:
  RelErr = 100 * ||U_HDM - U_hat||_F / ||U_HDM||_F

Saves outputs and plots in a NEW directory:
  results_reconstruction_case2_250x250/<mu_tag>/
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make ../ importable
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Imports from project root
# ---------------------------------------------------------------------
from hypernet2D import load_or_compute_snaps, plot_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

# ============================================================
# Case 2 model definition (scaler embedded) matching your loader
# ============================================================

class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))
    def forward(self, x):
        return (x - self.mean) / self.std

class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))
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

class Case2Model(nn.Module):
    """
    Input:  (mu1, mu2, t) raw
    Output: q_s raw (nbar,)
    """
    def __init__(self, n_s: int):
        super().__init__()
        in_dim = 3
        # placeholders overwritten by load_state_dict
        self.scaler   = Scaler(np.zeros((1, in_dim)), np.ones((1, in_dim)))
        self.core     = CoreMLP(in_dim, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw

def load_case2_model(model_path: str, device: str):
    ckpt = torch.load(model_path, map_location=device)
    n_s = int(ckpt["n_s"])
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"case2 checkpoint in_dim={in_dim}, expected 3")

    model = Case2Model(n_s).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, n_s

# ============================================================
# Helpers
# ============================================================

def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"

def rel_err_fro(U_ref, U_approx):
    return 100.0 * np.linalg.norm(U_ref - U_approx) / np.linalg.norm(U_ref)

def load_uref(path, N_expected):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing u_ref: {path}")
    u_ref = np.asarray(np.load(path)).reshape(-1)
    if u_ref.size != N_expected:
        raise ValueError(f"u_ref size {u_ref.size} != expected {N_expected}")
    return u_ref.astype(np.float64)

def find_coeff_dir_for_mu(coeff_root, mu, n_tot):
    """
    Expected naming from your single-μ PROM script:
      prom_coeff_single_{tag}_ntot{n_tot}
    """
    tag = _safe_mu_tag(mu)
    d = os.path.join(coeff_root, f"prom_coeff_single_{tag}_ntot{n_tot}")
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"Could not find coeff directory for {tag}\n"
            f"Expected: {d}\n"
            "Adjust find_coeff_dir_for_mu() if your folder naming differs."
        )
    return d

# ============================================================
# Main
# ============================================================

def main():
    # -----------------------------
    # User settings
    # -----------------------------
    MU_LIST = [
        [4.56, 0.019],
        [5.19, 0.026],
    ]

    primary_modes = 10
    total_modes   = 150
    nbar_expected = total_modes - primary_modes

    pod_dir = os.path.join(PROJECT_ROOT, "POD-RBF_YvonMaday")
    basis_path = os.path.join(pod_dir, "basis.npy")
    uref_path  = os.path.join(pod_dir, "u_ref.npy")
    model_path = os.path.join(pod_dir, "case2_model.pt")

    # Where your single-μ PROM coefficient folders live
    # If you ran the single-μ script from THIS_DIR, this is correct.
    COEFF_ROOT = THIS_DIR

    # HDM snapshot cache folder
    snap_folder = os.path.join(PROJECT_ROOT, "param_snaps")
    os.makedirs(snap_folder, exist_ok=True)

    # Output root (NEW directory)
    OUT_ROOT = os.path.join(THIS_DIR, "results_reconstruction_case2_250x250")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Plot steps
    inds_to_plot = list(range(0, NUM_STEPS + 1, 100))
    if NUM_STEPS not in inds_to_plot:
        inds_to_plot.append(NUM_STEPS)

    # Device
    device = "cpu"
    print(f"[Recon Case2] device = {device}")
    print(f"[Recon Case2] OUT_ROOT = {OUT_ROOT}")

    # -----------------------------
    # Load basis + split
    # -----------------------------
    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Missing basis: {basis_path}")
    full_basis = np.load(basis_path, allow_pickle=True)
    if full_basis.shape[1] < total_modes:
        raise ValueError(f"basis has {full_basis.shape[1]} modes, need >= {total_modes}")

    V_np    = full_basis[:, :primary_modes].astype(np.float64, copy=False)
    Vbar_np = full_basis[:, primary_modes:total_modes].astype(np.float64, copy=False)
    N = V_np.shape[0]

    # u_ref as (N,1) for snapshot broadcasting
    u_ref = load_uref(uref_path, N_expected=N).reshape(-1, 1)

    # -----------------------------
    # Load Case 2 model
    # -----------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    rnm, nbar = load_case2_model(model_path, device=device)
    if nbar != nbar_expected:
        raise ValueError(f"case2 model outputs nbar={nbar}, expected {nbar_expected}")

    print(f"[Recon Case2] Loaded model: {model_path}")

    # -----------------------------
    # Loop over μ values
    # -----------------------------
    for mu in MU_LIST:
        tag = _safe_mu_tag(mu)
        mu1, mu2 = float(mu[0]), float(mu[1])

        out_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[Recon Case2] mu = {mu} ({tag})")
        t0 = time.time()

        # 1) Load coefficients produced by n_tot PROM run
        coeff_dir = find_coeff_dir_for_mu(COEFF_ROOT, mu, total_modes)
        t_vec = np.load(os.path.join(coeff_dir, "t.npy")).astype(np.float64).reshape(-1)    # (T,)
        q = np.load(os.path.join(coeff_dir, "qN_p.npy")).astype(np.float64)                 # (n, T)

        if q.shape[0] != primary_modes:
            raise ValueError(f"{tag}: qN_p has shape {q.shape}, expected ({primary_modes},T)")
        T = q.shape[1]
        if t_vec.shape[0] != T:
            raise ValueError(f"{tag}: t length {t_vec.shape[0]} != qN_p T={T}")

        # 2) Load HDM snapshots
        hdm_snaps = load_or_compute_snaps(
            mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS,
            snap_folder=snap_folder
        ).astype(np.float64)

        if hdm_snaps.shape[1] != T:
            raise ValueError(f"{tag}: HDM snaps T={hdm_snaps.shape[1]} != coeff T={T}")

        # 3) Build model inputs X_raw = [mu1, mu2, t] (T,3)
        mu1_col = np.full((T, 1), mu1, dtype=np.float32)
        mu2_col = np.full((T, 1), mu2, dtype=np.float32)
        t_col   = t_vec.astype(np.float32).reshape(T, 1)
        X_raw = np.hstack([mu1_col, mu2_col, t_col])  # (T,3)

        # 4) Infer qbar_hat in one batch: (T,nbar) -> transpose to (nbar,T)
        with torch.no_grad():
            X_t = torch.from_numpy(X_raw).to(device)
            Y_t = rnm(X_t)  # (T, nbar)
            qbar_hat = Y_t.detach().cpu().numpy().astype(np.float64).T  # (nbar, T)

        # 5) Reconstruct U_hat = u_ref + V q + Vbar qbar_hat
        U_hat = u_ref + (V_np @ q) + (Vbar_np @ qbar_hat)  # (N,T)

        # 6) Error vs HDM
        rel = rel_err_fro(hdm_snaps, U_hat)
        print(f"[Recon Case2] RelErr vs HDM = {rel:.6f}%")

        # -----------------------------
        # Save outputs
        # -----------------------------
        np.save(os.path.join(out_dir, "mu.npy"), np.array(mu, dtype=np.float64))
        np.save(os.path.join(out_dir, "t.npy"), t_vec)
        np.save(os.path.join(out_dir, "q_used.npy"), q)  # q from qN_p (PROM-consistent)
        np.save(os.path.join(out_dir, "qbar_hat_case2.npy"), qbar_hat)
        np.save(os.path.join(out_dir, "recon_snaps_case2.npy"), U_hat)

        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(f"mu = {mu}\n")
            f.write(f"primary_modes = {primary_modes}\n")
            f.write(f"total_modes = {total_modes}\n")
            f.write(f"RelErr_vs_HDM_percent = {rel:.12f}\n")
            f.write(f"coeff_dir = {coeff_dir}\n")
            f.write(f"model_path = {model_path}\n")
            f.write("case2 input = [mu1, mu2, t] where t is physical time\n")

        # -----------------------------
        # Plot HDM vs reconstruction
        # -----------------------------
        fig, ax1, ax2 = plot_snaps(
            GRID_X,
            GRID_Y,
            hdm_snaps,
            inds_to_plot,
            label="HDM",
            color="black",
            linewidth=2.8,
            linestyle="solid",
        )
        plot_snaps(
            GRID_X, GRID_Y, U_hat, inds_to_plot,
            label="Case 2 reconstruction",
            fig_ax=(fig, ax1, ax2),
            color="blue",
            linewidth=1.8,
            linestyle="solid",
        )
        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"case2_hdm_vs_reconstruction_{tag}_n{primary_modes}_ntot{total_modes}.png")
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        print(f"[Recon Case2] saved plot: {out_png}")
        print(f"[Recon Case2] saved data in: {out_dir}")
        print(f"[Recon Case2] elapsed: {time.time() - t0:.2f} s")

    print("\n[Recon Case2] DONE.")


if __name__ == "__main__":
    main()
