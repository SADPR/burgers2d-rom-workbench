#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_3_train_nn_case_3_minimal.py

Case 3 (ANN):
    qN_s = N(qN_p, mu1, mu2, t)

- Loads PROM-solved coefficients from:
    prom_coeff_dataset_ntot*/per_mu/*/mu.npy     (2,)
    prom_coeff_dataset_ntot*/per_mu/*/t.npy      (T,)
    prom_coeff_dataset_ntot*/per_mu/*/qN.npy     (n_tot, T)

- Builds dataset:
    X_raw = [qN_p(t)^T, mu1, mu2, t]  -> shape (M, n_p + 3)
    Y_raw = qN_s(t)^T                 -> shape (M, n_s)

- Embeds scaling inside the model (so inference is just model(X_raw)).
- Saves ONLY:
    case3_model.pt
"""

import os
import time
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset
try:
    from stage3_qn_utils import load_qn_from_mu_dir, resolve_primary_modes, split_qn
except ModuleNotFoundError:
    from .stage3_qn_utils import load_qn_from_mu_dir, resolve_primary_modes, split_qn
try:
    from project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Repro
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_prom_dataset_case3(dataset_root: str, primary_modes: int):
    """
    Return X_raw (M, n_p+3), Y_raw (M, n_s) in float32 from per_mu dirs.

    Each per_mu dir must contain:
      - mu.npy   (2,)
      - t.npy    (T,)
      - qN.npy   (n_tot, T)
    """
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")

    subdirs = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )
    if len(subdirs) == 0:
        raise RuntimeError(f"No per_mu subdirectories found in: {dataset_root}")

    X_list, Y_list = [], []
    n_p_ref = None
    n_s_ref = None

    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)

        mu = np.load(os.path.join(mu_dir, "mu.npy")).astype(np.float64).reshape(-1)
        if mu.size != 2:
            raise ValueError(f"{sd}: mu.npy must have shape (2,), got {mu.shape}")

        t = np.load(os.path.join(mu_dir, "t.npy")).astype(np.float64).reshape(-1)  # (T,)

        qN = load_qn_from_mu_dir(mu_dir).astype(np.float64)                 # (n_tot, T)
        qNp, qNs = split_qn(qN, primary_modes)                              # (n_p, T), (n_s, T)

        n_p, T1 = qNp.shape
        n_s, T2 = qNs.shape
        if T1 != T2:
            raise ValueError(f"{sd}: qN_p has T={T1} but qN_s has T={T2}")
        if t.shape[0] != T1:
            raise ValueError(f"{sd}: t has length {t.shape[0]} but qN_p has T={T1}")

        if n_p_ref is None:
            n_p_ref = n_p
        elif n_p != n_p_ref:
            raise ValueError(f"{sd}: n_p mismatch, got {n_p}, expected {n_p_ref}")

        if n_s_ref is None:
            n_s_ref = n_s
        elif n_s != n_s_ref:
            raise ValueError(f"{sd}: n_s mismatch, got {n_s}, expected {n_s_ref}")

        # Build X(t) = [qN_p(t), mu1, mu2, t]
        mu1 = np.full((T1, 1), mu[0], dtype=np.float64)
        mu2 = np.full((T1, 1), mu[1], dtype=np.float64)
        tt = t.reshape(-1, 1).astype(np.float64)

        Xi = np.hstack([qNp.T, mu1, mu2, tt])  # (T, n_p+3)
        Yi = qNs.T                              # (T, n_s)

        X_list.append(Xi)
        Y_list.append(Yi)

    X_raw = np.vstack(X_list).astype(np.float32)  # (M, n_p+3)
    Y_raw = np.vstack(Y_list).astype(np.float32)  # (M, n_s)
    return X_raw, Y_raw


# -----------------------------
# Scaler modules stored as buffers
# -----------------------------
class Scaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


# -----------------------------
# Core MLP in normalized space
# -----------------------------
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


# -----------------------------
# Full model: scale -> MLP -> unscale
# -----------------------------
class Case3Model(nn.Module):
    """
    X_raw = [qN_p, mu1, mu2, t] -> qN_s_raw
    Scaling is embedded as buffers.
    """

    def __init__(self, x_mean, x_std, y_mean, y_std):
        super().__init__()
        in_dim = x_mean.shape[0]   # n_p + 3
        out_dim = y_mean.shape[0]  # n_s
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(in_dim, out_dim)
        self.unscaler = Unscaler(y_mean[None, :], y_std[None, :])

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw


def main(argv=None):
    # -----------------------------
    # User settings
    # -----------------------------
    ensure_layout_dirs()

    parser = argparse.ArgumentParser(
        description="Train Case-3 ANN map from Stage-2 dataset."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="hprom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--primary-modes", type=int, default=None)
    args = parser.parse_args(argv)

    dataset_ntot = args.dataset_ntot
    dataset_backend = str(args.dataset_backend).strip().lower()
    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=dataset_ntot,
        expected_backend=dataset_backend,
    )
    primary_modes = resolve_primary_modes(args.primary_modes, dataset_meta, dataset_ntot)
    model_name = str(args.model_name).strip() if args.model_name is not None else "case3_model.pt"
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)
    summary_path = os.path.join(STAGE3_DIR, "case3_training_summary.txt")

    VAL_FRAC = 0.1
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-6
    epochs = 2000
    patience = 120
    min_improve = 1e-12
    clip_grad = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Case3] device = {device}")
    print(f"[Case3] dataset_dir = {dataset_dir}")
    print(f"[Case3] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case3] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case3] primary_modes (training split) = {primary_modes}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_raw, Y_raw = load_prom_dataset_case3(dataset_root, primary_modes=primary_modes)
    M, in_dim = X_raw.shape
    _, n_s = Y_raw.shape

    if in_dim < 4:
        raise ValueError(f"[Case3] Expected in_dim = n_p+3 >= 4, got {in_dim}")

    n_p = in_dim - 3
    print(f"[Case3] Loaded: M={M}, n_p={n_p}, in_dim={in_dim}, n_s={n_s}")

    # -----------------------------
    # Split
    # -----------------------------
    idx = np.arange(M, dtype=np.int64)
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, shuffle=True)

    Xtr, Ytr = X_raw[tr_idx], Y_raw[tr_idx]
    Xva, Yva = X_raw[va_idx], Y_raw[va_idx]

    # -----------------------------
    # Compute scaling stats on TRAIN only
    # -----------------------------
    x_mean = Xtr.mean(axis=0)
    x_std = Xtr.std(axis=0)
    y_mean = Ytr.mean(axis=0)
    y_std = Ytr.std(axis=0)

    # -----------------------------
    # Model
    # -----------------------------
    model = Case3Model(x_mean, x_std, y_mean, y_std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    Xva_t = torch.from_numpy(Xva).to(device)
    Yva_t = torch.from_numpy(Yva).to(device)

    # -----------------------------
    # Train with early stopping on VAL
    # -----------------------------
    best_val = float("inf")
    best_state = None
    bad = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0

        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()
            tr_loss += float(loss.detach().cpu().item()) * xb.shape[0]

        tr_loss /= Xtr.shape[0]

        model.eval()
        with torch.no_grad():
            va_loss = float(loss_fn(model(Xva_t), Yva_t).detach().cpu().item())

        if ep == 1 or ep % 25 == 0:
            print(f"[Epoch {ep:4d}] train_mse={tr_loss:.6e} | val_mse={va_loss:.6e} | bad={bad}")

        if va_loss < best_val - min_improve:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] epoch={ep} best_val={best_val:.6e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[Case3] Training done in {time.time() - t0:.2f}s. best_val={best_val:.6e}")

    # -----------------------------
    # Save ONLY one file (weights + scaler buffers)
    # -----------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": int(in_dim),      # n_p + 3
        "n_p": int(n_p),
        "n_s": int(n_s),
        "seed": int(SEED),
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "primary_modes": int(primary_modes),
        "secondary_modes": int(dataset_ntot - primary_modes),
        "mapping": "qN_s = N(qN_p, mu1, mu2, t)",
        "x_layout": "X_raw = [qN_p..., mu1, mu2, t]",
    }
    torch.save(ckpt, model_path)
    print(f"[Case3] Saved model checkpoint: {model_path}")
    write_kv_txt(
        summary_path,
        [
            ("model_name", model_name),
            ("model_path", model_path),
            ("dataset_dir", dataset_dir),
            ("dataset_root", dataset_root),
            ("dataset_ntot", dataset_ntot),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("primary_modes", primary_modes),
            ("secondary_modes", int(dataset_ntot - primary_modes)),
            ("samples_M", M),
            ("n_p", n_p),
            ("in_dim", in_dim),
            ("n_s", n_s),
            ("epochs_ran", ep),
            ("best_val_mse", best_val),
            ("seed", SEED),
            ("device", device),
        ],
    )
    print(f"[Case3] Summary: {summary_path}")


if __name__ == "__main__":
    main()
