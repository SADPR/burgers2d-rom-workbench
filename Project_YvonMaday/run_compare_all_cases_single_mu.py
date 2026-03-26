
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_compare_all_cases_single_mu.py

One script that, for a single parameter mu=(mu1,mu2):
  1) runs an n_tot PROM (LSPG) to get PROM-consistent coefficients (q_prom, qbar_prom)
  2) computes oracle closure errors in coefficient space:
       ||qbar_prom - qbar_pred|| / ||qbar_prom||
     where qbar_pred is produced by the NN using exact PROM inputs (q_prom, mu, t)
  3) computes oracle errors in state space:
       ||u_HDM - u_oracle|| / ||u_HDM||
     where u_oracle = u_ref + V q_prom + Vbar qbar_pred
     This is directly comparable to the online state errors.
  4) runs online PROM-ANN for Case 1/2/3
  5) computes state trajectory errors vs HDM for:
       - n_tot PROM
       - Case1/2/3 PROM-ANN
  6) saves outputs and a summary dict

Assumed layout:
  PROJECT_ROOT/
    hypernet2D.py
    config.py
    param_snaps/
    POD-RBF_YvonMaday/
      basis.npy
      u_ref.npy
      case1_model.pt
      case2_model.pt
      case3_model.pt
"""

import os
import sys
import time
import json
import numpy as np

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

POD_DIR = os.path.join(PROJECT_ROOT, "POD-RBF_YvonMaday")
SNAP_FOLDER = os.path.join(PROJECT_ROOT, "param_snaps")


# -------------------------------------------------------------
# Imports from your codebase
# -------------------------------------------------------------
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0  # single source of truth
from hypernet2D import (
    inviscid_burgers_implicit2D_LSPG,
    inviscid_burgers_rnm2D,
    inviscid_burgers_rnm2D_case2,
    inviscid_burgers_rnm2D_case3,
    load_or_compute_snaps,
    plot_snaps,
)


# ============================================================
# Torch model building blocks (match your checkpoints)
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


class Case1Model(nn.Module):
    # input: q (n,), output: qbar (nbar,)
    def __init__(self, n_p, n_s):
        super().__init__()
        self.scaler   = Scaler(np.zeros((1, n_p)), np.ones((1, n_p)))
        self.core     = CoreMLP(n_p, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        return self.unscaler(y_n)


class Case2Model(nn.Module):
    # input: (mu1,mu2,t), output: qbar
    def __init__(self, n_s):
        super().__init__()
        in_dim = 3
        self.scaler   = Scaler(np.zeros((1, in_dim)), np.ones((1, in_dim)))
        self.core     = CoreMLP(in_dim, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        return self.unscaler(y_n)


class Case3Model(nn.Module):
    # input: (q..., mu1, mu2, t), output: qbar
    def __init__(self, in_dim, n_s):
        super().__init__()
        self.scaler   = Scaler(np.zeros((1, in_dim)), np.ones((1, in_dim)))
        self.core     = CoreMLP(in_dim, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        return self.unscaler(y_n)


class RNMWrapper(nn.Module):
    # makes model accept 1D tensors and return 1D outputs
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        if x.ndim == 1:
            return self.base(x.unsqueeze(0)).squeeze(0)
        return self.base(x)


def load_case1_model(path, device):
    ckpt = torch.load(path, map_location=device)
    n_p = int(ckpt["n_p"])
    n_s = int(ckpt["n_s"])
    m = Case1Model(n_p, n_s).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return RNMWrapper(m).to(device), n_p, n_s


def load_case2_model(path, device):
    ckpt = torch.load(path, map_location=device)
    n_s = int(ckpt["n_s"])
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"Case2 checkpoint in_dim={in_dim}, expected 3")
    m = Case2Model(n_s).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return RNMWrapper(m).to(device), n_s


def load_case3_model(path, device):
    ckpt = torch.load(path, map_location=device)
    in_dim = int(ckpt["in_dim"])
    n_p = int(ckpt["n_p"])
    n_s = int(ckpt["n_s"])
    m = Case3Model(in_dim=in_dim, n_s=n_s).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return RNMWrapper(m).to(device), in_dim, n_p, n_s


# ============================================================
# Utilities
# ============================================================

def time_grid(dt, num_steps):
    return dt * np.arange(num_steps + 1, dtype=np.float64)


def safe_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def rel_frob(A, B, eps=1e-16):
    num = np.linalg.norm(A - B)
    den = np.linalg.norm(B)
    return float(num / max(den, eps))


def rel_frob_F(A, B, eps=1e-16):
    num = np.linalg.norm(A - B, ord="fro")
    den = np.linalg.norm(B, ord="fro")
    return float(num / max(den, eps))


def ensure_col(u_ref, N):
    u_ref = np.asarray(u_ref)
    if u_ref.ndim == 1:
        u_ref = u_ref.reshape(-1, 1)
    if u_ref.shape[0] != N:
        raise ValueError(f"u_ref has N={u_ref.shape[0]}, expected {N}")
    return u_ref


# ============================================================
# Main driver
# ============================================================

def main():
    # -----------------------------
    # User settings
    # -----------------------------
    mu = [4.56, 0.019]      # set once here
    n_tot = 150
    n = 10
    assert n_tot > n
    nbar = n_tot - n

    device = "cpu"

    MAKE_PLOTS = True
    SAVE_SNAPSHOTS = True

    # output folder
    out_dir = os.path.join(THIS_DIR, f"compare_all_{safe_tag(mu)}_n{n}_ntot{n_tot}")
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Load basis + u_ref
    # -----------------------------
    basis_path = os.path.join(POD_DIR, "basis.npy")
    uref_path  = os.path.join(POD_DIR, "u_ref.npy")
    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Missing basis: {basis_path}")
    if not os.path.exists(uref_path):
        raise FileNotFoundError(f"Missing u_ref: {uref_path}")

    full_basis = np.load(basis_path, allow_pickle=True).astype(np.float64, copy=False)
    if full_basis.shape[1] < n_tot:
        raise ValueError(f"basis has {full_basis.shape[1]} modes, need >= {n_tot}")

    Vtot = full_basis[:, :n_tot]
    V = full_basis[:, :n]
    Vbar = full_basis[:, n:n_tot]
    N_dof = V.shape[0]

    u_ref = ensure_col(np.load(uref_path), N_dof)  # (N,1)

    # torch versions for PROM-ANN kernels
    V_t    = torch.tensor(V,    dtype=torch.float32, device=device)
    Vbar_t = torch.tensor(Vbar, dtype=torch.float32, device=device)

    # -----------------------------
    # HDM reference (for state errors/plots)
    # -----------------------------
    hdm_snaps = load_or_compute_snaps(
        mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=SNAP_FOLDER
    )

    # -----------------------------
    # 1) n_tot PROM run
    # -----------------------------
    print("\n=== Running n_tot PROM (LSPG) ===")
    t0 = time.time()
    prom_snaps, prom_times = inviscid_burgers_implicit2D_LSPG(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, mu, Vtot, u_ref=u_ref
    )
    prom_elapsed = time.time() - t0

    # time vector
    t_vec_ref = time_grid(DT, NUM_STEPS)
    if (prom_times is not None) and (len(prom_times) == prom_snaps.shape[1]):
        t_vec = np.asarray(prom_times, dtype=np.float64)
    elif len(t_vec_ref) == prom_snaps.shape[1]:
        t_vec = t_vec_ref
    else:
        t_vec = DT * np.arange(prom_snaps.shape[1], dtype=np.float64)

    # PROM-consistent coefficients
    snaps_c = prom_snaps - u_ref
    qN = Vtot.T @ snaps_c              # (n_tot, T)
    q_prom = qN[:n, :]                 # (n, T)
    qbar_prom = qN[n:n_tot, :]         # (nbar, T)

    prom_rel_vs_hdm = 100.0 * rel_frob_F(prom_snaps, hdm_snaps)

    print(f"[PROM n_tot] elapsed = {prom_elapsed:.3e} s")
    print(f"[PROM n_tot] rel err vs HDM = {prom_rel_vs_hdm:.3f}%")

    # -----------------------------
    # Load Case1/2/3 models
    # -----------------------------
    case1_path = os.path.join(POD_DIR, "case1_model.pt")
    case2_path = os.path.join(POD_DIR, "case2_model.pt")
    case3_path = os.path.join(POD_DIR, "case3_model.pt")
    for p in [case1_path, case2_path, case3_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    rnm1, n_p1, n_s1 = load_case1_model(case1_path, device=device)
    rnm2, n_s2 = load_case2_model(case2_path, device=device)
    rnm3, in_dim3, n_p3, n_s3 = load_case3_model(case3_path, device=device)

    if n_p1 != n or n_s1 != nbar:
        raise ValueError(
            f"Case1 dims mismatch: ckpt (n_p,n_s)=({n_p1},{n_s1}) vs expected ({n},{nbar})"
        )
    if n_s2 != nbar:
        raise ValueError(f"Case2 dims mismatch: ckpt n_s={n_s2} vs expected {nbar}")
    if n_p3 != n or n_s3 != nbar or in_dim3 != (n + 3):
        raise ValueError(
            f"Case3 dims mismatch: ckpt (in_dim,n_p,n_s)=({in_dim3},{n_p3},{n_s3}) "
            f"vs expected ({n+3},{n},{nbar})"
        )

    # -----------------------------
    # 2) Oracle errors
    # -----------------------------
    print("\n=== Oracle errors ===")
    T = q_prom.shape[1]

    # Case 1: qbar_hat = N(q_prom)
    with torch.no_grad():
        q_prom_t = torch.tensor(q_prom.T, dtype=torch.float32, device=device)  # (T, n)
        qbar1_hat = rnm1(q_prom_t).cpu().numpy().T  # (nbar, T)

    # Case 2: qbar_hat = M(mu,t)
    X2 = np.column_stack([np.full(T, mu[0]), np.full(T, mu[1]), t_vec[:T]])
    with torch.no_grad():
        X2_t = torch.tensor(X2, dtype=torch.float32, device=device)  # (T, 3)
        qbar2_hat = rnm2(X2_t).cpu().numpy().T  # (nbar, T)

    # Case 3: qbar_hat = H(q_prom,mu,t)
    X3 = np.column_stack([q_prom.T, np.full(T, mu[0]), np.full(T, mu[1]), t_vec[:T]])
    with torch.no_grad():
        X3_t = torch.tensor(X3, dtype=torch.float32, device=device)  # (T, n+3)
        qbar3_hat = rnm3(X3_t).cpu().numpy().T  # (nbar, T)

    # Oracle errors in coefficient space
    e1_barq = 100.0 * rel_frob_F(qbar1_hat, qbar_prom)
    e2_barq = 100.0 * rel_frob_F(qbar2_hat, qbar_prom)
    e3_barq = 100.0 * rel_frob_F(qbar3_hat, qbar_prom)

    print(f"[oracle-qbar] Case1  ||qbar - N(q)|| / ||qbar|| = {e1_barq:.3f}%")
    print(f"[oracle-qbar] Case2  ||qbar - M(mu,t)|| / ||qbar|| = {e2_barq:.3f}%")
    print(f"[oracle-qbar] Case3  ||qbar - H(q,mu,t)|| / ||qbar|| = {e3_barq:.3f}%")

    # Oracle states using exact PROM primary coordinates q_prom
    u_oracle1 = u_ref + V @ q_prom + Vbar @ qbar1_hat
    u_oracle2 = u_ref + V @ q_prom + Vbar @ qbar2_hat
    u_oracle3 = u_ref + V @ q_prom + Vbar @ qbar3_hat

    # Oracle errors in state space (directly comparable to online state errors)
    e1_oracle_u = 100.0 * rel_frob_F(u_oracle1, hdm_snaps)
    e2_oracle_u = 100.0 * rel_frob_F(u_oracle2, hdm_snaps)
    e3_oracle_u = 100.0 * rel_frob_F(u_oracle3, hdm_snaps)

    print(f"[oracle-u]    Case1  ||u_HDM - u_oracle|| / ||u_HDM|| = {e1_oracle_u:.3f}%")
    print(f"[oracle-u]    Case2  ||u_HDM - u_oracle|| / ||u_HDM|| = {e2_oracle_u:.3f}%")
    print(f"[oracle-u]    Case3  ||u_HDM - u_oracle|| / ||u_HDM|| = {e3_oracle_u:.3f}%")

    # -----------------------------
    # 3) Online PROM-ANN runs (state space)
    # -----------------------------
    print("\n=== Online PROM-ANN runs ===")

    # Case 1
    t0 = time.time()
    snaps1, times1 = inviscid_burgers_rnm2D(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, mu,
        rnm=rnm1, ref=None, basis=V_t, basis2=Vbar_t, u_ref=u_ref.reshape(-1)
    )
    elapsed1 = time.time() - t0
    its1, jac1, res1, ls1 = times1
    err1_hdm = 100.0 * rel_frob_F(snaps1, hdm_snaps)

    print(
        f"[Case1] elapsed={elapsed1:.3e}s | its={its1} jac={jac1:.3e} "
        f"res={res1:.3e} ls={ls1:.3e} | err vs HDM={err1_hdm:.3f}%"
    )

    # Case 2
    t0 = time.time()
    snaps2, times2 = inviscid_burgers_rnm2D_case2(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, mu,
        rnm=rnm2, ref=None, basis=V_t, basis2=Vbar_t, u_ref=u_ref.reshape(-1)
    )
    elapsed2 = time.time() - t0
    its2, jac2, res2, ls2 = times2
    err2_hdm = 100.0 * rel_frob_F(snaps2, hdm_snaps)

    print(
        f"[Case2] elapsed={elapsed2:.3e}s | its={its2} jac={jac2:.3e} "
        f"res={res2:.3e} ls={ls2:.3e} | err vs HDM={err2_hdm:.3f}%"
    )

    # Case 3
    t0 = time.time()
    snaps3, times3 = inviscid_burgers_rnm2D_case3(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, mu,
        rnm=rnm3, ref=None, basis=V_t, basis2=Vbar_t, u_ref=u_ref.reshape(-1)
    )
    elapsed3 = time.time() - t0
    its3, jac3, res3, ls3 = times3
    err3_hdm = 100.0 * rel_frob_F(snaps3, hdm_snaps)

    print(
        f"[Case3] elapsed={elapsed3:.3e}s | its={its3} jac={jac3:.3e} "
        f"res={res3:.3e} ls={ls3:.3e} | err vs HDM={err3_hdm:.3f}%"
    )

    # -----------------------------
    # Save outputs
    # -----------------------------
    summary = {
        "mu": [float(mu[0]), float(mu[1])],
        "n": int(n),
        "n_tot": int(n_tot),
        "nbar": int(nbar),
        "DT": float(DT),
        "NUM_STEPS": int(NUM_STEPS),

        "prom_elapsed_s": float(prom_elapsed),
        "prom_relerr_vs_hdm_percent": float(prom_rel_vs_hdm),

        # oracle errors in coefficient space
        "oracle_relerr_qbar_case1_percent": float(e1_barq),
        "oracle_relerr_qbar_case2_percent": float(e2_barq),
        "oracle_relerr_qbar_case3_percent": float(e3_barq),

        # oracle errors in state space (directly comparable to online errors)
        "oracle_relerr_u_case1_percent": float(e1_oracle_u),
        "oracle_relerr_u_case2_percent": float(e2_oracle_u),
        "oracle_relerr_u_case3_percent": float(e3_oracle_u),

        # online errors vs HDM
        "online_relerr_vs_hdm_case1_percent": float(err1_hdm),
        "online_relerr_vs_hdm_case2_percent": float(err2_hdm),
        "online_relerr_vs_hdm_case3_percent": float(err3_hdm),

        # timing breakdowns returned by hypernet2D
        "case1_times": {
            "its": int(its1),
            "jac_s": float(jac1),
            "res_s": float(res1),
            "ls_s": float(ls1),
            "elapsed_s": float(elapsed1),
        },
        "case2_times": {
            "its": int(its2),
            "jac_s": float(jac2),
            "res_s": float(res2),
            "ls_s": float(ls2),
            "elapsed_s": float(elapsed2),
        },
        "case3_times": {
            "its": int(its3),
            "jac_s": float(jac3),
            "res_s": float(res3),
            "ls_s": float(ls3),
            "elapsed_s": float(elapsed3),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(out_dir, "mu.npy"), np.array(mu, dtype=np.float64))
    np.save(os.path.join(out_dir, "t.npy"), t_vec)
    np.save(os.path.join(out_dir, "q_prom.npy"), q_prom)
    np.save(os.path.join(out_dir, "qbar_prom.npy"), qbar_prom)
    np.save(os.path.join(out_dir, "qbar_hat_case1.npy"), qbar1_hat)
    np.save(os.path.join(out_dir, "qbar_hat_case2.npy"), qbar2_hat)
    np.save(os.path.join(out_dir, "qbar_hat_case3.npy"), qbar3_hat)
    np.save(os.path.join(out_dir, "oracle_case1_snaps.npy"), u_oracle1)
    np.save(os.path.join(out_dir, "oracle_case2_snaps.npy"), u_oracle2)
    np.save(os.path.join(out_dir, "oracle_case3_snaps.npy"), u_oracle3)

    if SAVE_SNAPSHOTS:
        np.save(os.path.join(out_dir, "hdm_snaps.npy"), hdm_snaps)
        np.save(os.path.join(out_dir, "prom_snaps.npy"), prom_snaps)
        np.save(os.path.join(out_dir, "case1_snaps.npy"), snaps1)
        np.save(os.path.join(out_dir, "case2_snaps.npy"), snaps2)
        np.save(os.path.join(out_dir, "case3_snaps.npy"), snaps3)

    # -----------------------------
    # Plots (optional): HDM vs each model
    # -----------------------------
    if MAKE_PLOTS:
        plot_steps = list(range(0, NUM_STEPS + 1, 100))
        if NUM_STEPS not in plot_steps:
            plot_steps.append(NUM_STEPS)

        # HDM vs PROM
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
            GRID_X, GRID_Y, prom_snaps, plot_steps,
            label="PROM ntot",
            fig_ax=(fig, ax1, ax2),
            color="blue",
            linewidth=1.8,
            linestyle="solid",
        )
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hdm_vs_prom_ntot.png"), dpi=200)
        plt.close(fig)

        # HDM vs oracle Case 1/2/3
        for snaps, err, name in [
            (u_oracle1, e1_oracle_u, "oracle_case1"),
            (u_oracle2, e2_oracle_u, "oracle_case2"),
            (u_oracle3, e3_oracle_u, "oracle_case3"),
        ]:
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
                GRID_X, GRID_Y, snaps, plot_steps,
                label=name.upper(),
                fig_ax=(fig, ax1, ax2),
                color="blue",
                linewidth=1.8,
                linestyle="solid",
            )
            ax1.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"hdm_vs_{name}.png"), dpi=200)
            plt.close(fig)

        # HDM vs online Case 1/2/3
        for snaps, err, name in [
            (snaps1, err1_hdm, "case1"),
            (snaps2, err2_hdm, "case2"),
            (snaps3, err3_hdm, "case3"),
        ]:
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
                GRID_X, GRID_Y, snaps, plot_steps,
                label=name.upper(),
                fig_ax=(fig, ax1, ax2),
                color="blue",
                linewidth=1.8,
                linestyle="solid",
            )
            ax1.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"hdm_vs_{name}.png"), dpi=200)
            plt.close(fig)

    print(f"\nSaved everything to: {out_dir}")
    print("Wrote: summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
