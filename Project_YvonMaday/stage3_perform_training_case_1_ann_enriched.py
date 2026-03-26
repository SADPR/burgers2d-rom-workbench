#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage3_perform_training_case_1_ann_enriched.py

Case 1 (ANN):
    qN_s = N(qN_p)

- Loads PROM-solved coefficients from:
    prom_coeff_dataset_ntot*/per_mu/*/qN_p.npy
    prom_coeff_dataset_ntot*/per_mu/*/qN_s.npy

- Embeds scaling inside the model (so inference is just model(qp_raw)).
- Saves ONLY:
    case1_model_enriched.pt
"""

import os
import time
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from project_layout import ensure_layout_dirs, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import ensure_layout_dirs, write_kv_txt
try:
    from enrichment_layout import ENRICHMENT_STAGE3_DIR, ensure_enrichment_dirs
except ModuleNotFoundError:
    from .enrichment_layout import ENRICHMENT_STAGE3_DIR, ensure_enrichment_dirs
try:
    from enrichment_dataset_utils import resolve_enrichment_dataset
except ModuleNotFoundError:
    from .enrichment_dataset_utils import resolve_enrichment_dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Repro
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_prom_dataset_case1(dataset_root: str):
    """Return X_raw (M,n_p), Y_raw (M,n_s) in float32."""
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")

    subdirs = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )
    if len(subdirs) == 0:
        raise RuntimeError(f"No per_mu subdirectories found in: {dataset_root}")

    qp_list, qs_list = [], []
    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)
        qp_i = np.load(os.path.join(mu_dir, "qN_p.npy"))  # (n_p, T)
        qs_i = np.load(os.path.join(mu_dir, "qN_s.npy"))  # (n_s, T)
        qp_list.append(qp_i)
        qs_list.append(qs_i)

    qp = np.hstack(qp_list)  # (n_p, M)
    qs = np.hstack(qs_list)  # (n_s, M)

    X_raw = qp.T.astype(np.float32)  # (M, n_p)
    Y_raw = qs.T.astype(np.float32)  # (M, n_s)
    return X_raw, Y_raw


# -----------------------------
# Scaler modules stored as buffers
# -----------------------------
class Scaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))

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
class Case1Model(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std):
        super().__init__()
        in_dim = x_mean.shape[0]
        out_dim = y_mean.shape[0]
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(in_dim, out_dim)
        self.unscaler = Unscaler(y_mean[None, :], y_std[None, :])

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw


def main():
    # -----------------------------
    # User settings
    # -----------------------------
    ensure_layout_dirs()
    ensure_enrichment_dirs()

    dataset_ntot = None  # set int to force a specific ntot dataset
    dataset_backend = "hprom"
    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_enrichment_dataset(
        requested_ntot=dataset_ntot,
        expected_backend=dataset_backend,
    )
    dataset_name = os.path.basename(dataset_dir.rstrip(os.sep))
    stage3_out_dir = os.path.join(ENRICHMENT_STAGE3_DIR, dataset_name)
    stage3_models_dir = os.path.join(stage3_out_dir, "models")
    os.makedirs(stage3_models_dir, exist_ok=True)

    model_path = os.path.join(stage3_models_dir, "case1_model_enriched.pt")
    summary_path = os.path.join(stage3_out_dir, "case1_training_summary_enriched.txt")

    VAL_FRAC = 0.1
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-6
    epochs = 2000
    patience = 120
    min_improve = 1e-12
    clip_grad = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Case1] device = {device}")
    print(f"[Case1] dataset_dir = {dataset_dir}")
    print(f"[Case1] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case1] solve_backend = {dataset_meta.get('solve_backend')}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_raw, Y_raw = load_prom_dataset_case1(dataset_root)
    M, n_p = X_raw.shape
    _, n_s = Y_raw.shape
    print(f"[Case1] Loaded: M={M}, n_p={n_p}, n_s={n_s}")

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
    x_std  = Xtr.std(axis=0)
    y_mean = Ytr.mean(axis=0)
    y_std  = Ytr.std(axis=0)

    # -----------------------------
    # Model
    # -----------------------------
    model = Case1Model(x_mean, x_std, y_mean, y_std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # DataLoaders (raw space; model scales internally)
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

    print(f"[Case1] Training done in {time.time() - t0:.2f}s. best_val={best_val:.6e}")

    # -----------------------------
    # Save ONLY one file (weights + scaler buffers)
    # -----------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "n_p": int(n_p),
        "n_s": int(n_s),
        "seed": int(SEED),
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
    }
    torch.save(ckpt, model_path)
    print(f"[Case1] Saved model checkpoint: {model_path}")
    write_kv_txt(
        summary_path,
        [
            ("model_path", model_path),
            ("dataset_dir", dataset_dir),
            ("dataset_root", dataset_root),
            ("dataset_ntot", dataset_ntot),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("samples_M", M),
            ("n_p", n_p),
            ("n_s", n_s),
            ("epochs_ran", ep),
            ("best_val_mse", best_val),
            ("seed", SEED),
            ("device", device),
        ],
    )
    print(f"[Case1] Summary: {summary_path}")


if __name__ == "__main__":
    main()
