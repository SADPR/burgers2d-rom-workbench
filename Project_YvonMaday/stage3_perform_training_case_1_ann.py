#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_3_train_nn_case_1_minimal.py

Case 1 (ANN):
    qN_s = N(qN_p)

- Loads PROM-solved coefficients from:
    prom_coeff_dataset_ntot*/per_mu/*/qN.npy

- Embeds scaling inside the model (so inference is just model(qp_raw)).
- Saves ONLY:
    case1_model.pt
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


def load_prom_dataset_case1(dataset_root: str, primary_modes: int):
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
        qn_i = load_qn_from_mu_dir(mu_dir)                # (n_tot, T)
        qp_i, qs_i = split_qn(qn_i, primary_modes)        # (n_p, T), (n_s, T)
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
def _parse_hidden_dims(txt: str):
    dims = tuple(int(v.strip()) for v in str(txt).split(",") if v.strip())
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"Invalid hidden dimensions: {txt!r}")
    return dims


def _make_activation(name: str):
    key = str(name).strip().lower()
    activations = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    }
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[key]()


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation="elu", dropout=0.0):
        super().__init__()
        dims = [int(in_dim), *[int(d) for d in hidden_dims], int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), _make_activation(activation)])
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Full model: scale -> MLP -> unscale
# -----------------------------
class Case1Model(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims, activation="elu", dropout=0.0):
        super().__init__()
        in_dim = x_mean.shape[0]
        out_dim = y_mean.shape[0]
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(in_dim, out_dim, hidden_dims, activation=activation, dropout=dropout)
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
        description="Train Case-1 ANN map from Stage-2 dataset."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="hprom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Explicit Stage-2 dataset directory. Use this to keep paper campaigns isolated.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--primary-modes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-improve", type=float, default=1e-12)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=40)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--hidden-dims", type=str, default="32,64,128,256,256")
    parser.add_argument(
        "--activation",
        choices=("elu", "gelu", "silu", "tanh", "relu", "leaky_relu"),
        default="elu",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--summary-name", type=str, default="case1_training_summary.txt")
    args = parser.parse_args(argv)

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_ntot = args.dataset_ntot
    dataset_backend = str(args.dataset_backend).strip().lower()
    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=dataset_ntot,
        expected_backend=dataset_backend,
        requested_dataset_dir=args.dataset_dir,
    )
    primary_modes = resolve_primary_modes(args.primary_modes, dataset_meta, dataset_ntot)
    model_name = str(args.model_name).strip() if args.model_name is not None else "case1_model.pt"
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)
    summary_path = os.path.join(STAGE3_DIR, str(args.summary_name).strip())

    val_frac = float(args.val_frac)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    epochs = int(args.epochs)
    patience = int(args.patience)
    min_improve = float(args.min_improve)
    clip_grad = float(args.clip_grad)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    lr_scheduler_min_lr = float(args.lr_scheduler_min_lr)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    activation = str(args.activation).strip().lower()
    dropout = float(args.dropout)

    if not (0.0 < val_frac < 0.5):
        raise ValueError(f"--val-frac must be in (0,0.5), got {val_frac}")
    if batch_size <= 0 or epochs <= 0 or patience <= 0:
        raise ValueError("batch-size, epochs and patience must be positive")
    if lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("lr must be positive and weight-decay nonnegative")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"--dropout must be in [0,1), got {dropout}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Case1] device = {device}")
    print(f"[Case1] dataset_dir = {dataset_dir}")
    print(f"[Case1] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case1] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case1] primary_modes (training split) = {primary_modes}")
    print(f"[Case1] hidden_dims = {hidden_dims}")
    print(f"[Case1] activation = {activation}")
    print(f"[Case1] dropout = {dropout}")
    print(
        "[Case1] lr_scheduler = ReduceLROnPlateau("
        f"factor={lr_scheduler_factor}, patience={lr_scheduler_patience}, "
        f"min_lr={lr_scheduler_min_lr:.3e})"
    )
    print(f"[Case1] seed = {seed}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_raw, Y_raw = load_prom_dataset_case1(dataset_root, primary_modes=primary_modes)
    M, n_p = X_raw.shape
    _, n_s = Y_raw.shape
    print(f"[Case1] Loaded: M={M}, n_p={n_p}, n_s={n_s}")

    # -----------------------------
    # Split
    # -----------------------------
    idx = np.arange(M, dtype=np.int64)
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, random_state=seed, shuffle=True)

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
    model = Case1Model(
        x_mean,
        x_std,
        y_mean,
        y_std,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min_lr,
    )
    loss_fn = nn.MSELoss()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Case1] architecture = q_p -> {list(hidden_dims)} {activation.upper()} -> q_s")
    print(f"[Case1] optimizer = AdamW(lr={lr:g}, weight_decay={weight_decay:g})")
    print(f"[Case1] batch_size = {batch_size}")
    print(f"[Case1] trainable_parameters = {trainable_params}")

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

        scheduler.step(va_loss)

        if ep == 1 or ep % 25 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(
                f"[Epoch {ep:4d}] train_mse={tr_loss:.6e} | "
                f"val_mse={va_loss:.6e} | lr={current_lr:.3e} | bad={bad}"
            )

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

    model.eval()
    with torch.no_grad():
        Xtr_t = torch.from_numpy(Xtr).to(device)
        Ytr_t = torch.from_numpy(Ytr).to(device)
        pred_tr = model(Xtr_t)
        pred_va = model(Xva_t)
        train_rel_frob_percent = 100.0 * (
            torch.linalg.norm(pred_tr - Ytr_t) / torch.linalg.norm(Ytr_t)
        ).detach().cpu().item()
        val_rel_frob_percent = 100.0 * (
            torch.linalg.norm(pred_va - Yva_t) / torch.linalg.norm(Yva_t)
        ).detach().cpu().item()

    print(f"[Case1] Training done in {time.time() - t0:.2f}s. best_val={best_val:.6e}")
    print(f"[Case1] train_rel_frob_percent = {train_rel_frob_percent:.4f}%")
    print(f"[Case1] val_rel_frob_percent = {val_rel_frob_percent:.4f}%")

    # -----------------------------
    # Save ONLY one file (weights + scaler buffers)
    # -----------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "n_p": int(n_p),
        "n_s": int(n_s),
        "seed": int(seed),
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "primary_modes": int(primary_modes),
        "secondary_modes": int(dataset_ntot - primary_modes),
        "hidden_dims": tuple(int(d) for d in hidden_dims),
        "activation": activation,
        "dropout": float(dropout),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "trainable_parameters": int(trainable_params),
        "best_val_mse": float(best_val),
        "train_rel_frob_percent": float(train_rel_frob_percent),
        "val_rel_frob_percent": float(val_rel_frob_percent),
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_factor": float(lr_scheduler_factor),
        "lr_scheduler_patience": int(lr_scheduler_patience),
        "lr_scheduler_min_lr": float(lr_scheduler_min_lr),
        "mapping": "qN_s = N(qN_p)",
    }
    torch.save(ckpt, model_path)
    print(f"[Case1] Saved model checkpoint: {model_path}")
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
            ("n_s", n_s),
            ("epochs_ran", ep),
            ("best_val_mse", best_val),
            ("seed", seed),
            ("device", device),
            ("hidden_dims", tuple(int(d) for d in hidden_dims)),
            ("activation", activation),
            ("dropout", dropout),
            ("architecture", f"q_p -> {list(hidden_dims)} {activation.upper()} -> q_s"),
            ("optimizer", "AdamW"),
            ("lr", lr),
            ("weight_decay", weight_decay),
            ("batch_size", batch_size),
            ("trainable_parameters", trainable_params),
            ("lr_scheduler", "ReduceLROnPlateau"),
            ("lr_scheduler_factor", lr_scheduler_factor),
            ("lr_scheduler_patience", lr_scheduler_patience),
            ("lr_scheduler_min_lr", lr_scheduler_min_lr),
            ("train_rel_frob_percent", train_rel_frob_percent),
            ("val_rel_frob_percent", val_rel_frob_percent),
        ],
    )
    print(f"[Case1] Summary: {summary_path}")


if __name__ == "__main__":
    main()
