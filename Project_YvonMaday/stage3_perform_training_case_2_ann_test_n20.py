#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental Case-2 Stage-3 trainer for n=20 studies.

This script is intentionally separated from the production trainer:
  - stage3_perform_training_case_2_ann.py

It adds flexible architecture and training hyperparameters while keeping
the same data pipeline and checkpoint structure.
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
    from stage3_split_utils import split_indices_ecsw_param_time
except ModuleNotFoundError:
    from .stage3_split_utils import split_indices_ecsw_param_time
try:
    from stage3_split_utils import split_indices_holdout_mu_group
except ModuleNotFoundError:
    from .stage3_split_utils import split_indices_holdout_mu_group
try:
    from project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prom_dataset_case2(dataset_root: str, primary_modes: int):
    """
    Return X_raw (M,3), Y_raw (M,n_s) in float32 from per_mu dirs.

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
    n_s_ref = None

    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)

        mu = np.load(os.path.join(mu_dir, "mu.npy")).astype(np.float64).reshape(-1)
        if mu.size != 2:
            raise ValueError(f"{sd}: mu.npy must have shape (2,), got {mu.shape}")

        t = np.load(os.path.join(mu_dir, "t.npy")).astype(np.float64).reshape(-1)   # (T,)
        qN = load_qn_from_mu_dir(mu_dir).astype(np.float64)                           # (n_tot, T)
        _, qNs = split_qn(qN, primary_modes)                                           # (n_s, T)

        n_s, T = qNs.shape
        if t.shape[0] != T:
            raise ValueError(f"{sd}: t has length {t.shape[0]} but qN_s has T={T}")

        if n_s_ref is None:
            n_s_ref = n_s
        elif n_s != n_s_ref:
            raise ValueError(f"{sd}: n_s mismatch, got {n_s}, expected {n_s_ref}")

        # Build X for this trajectory: repeat mu across time
        mu1 = np.full((T,), mu[0], dtype=np.float64)
        mu2 = np.full((T,), mu[1], dtype=np.float64)
        Xi = np.column_stack([mu1, mu2, t])     # (T,3)
        Yi = qNs.T                               # (T,n_s)

        X_list.append(Xi)
        Y_list.append(Yi)

    X_raw = np.vstack(X_list).astype(np.float32)  # (M,3)
    Y_raw = np.vstack(Y_list).astype(np.float32)  # (M,n_s)

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


def _parse_hidden_dims(txt: str):
    vals = [s.strip() for s in str(txt).split(",")]
    dims = []
    for v in vals:
        if not v:
            continue
        d = int(v)
        if d <= 0:
            raise ValueError(f"Hidden dimensions must be positive, got {d}.")
        dims.append(d)
    if not dims:
        raise ValueError("At least one hidden layer must be provided.")
    return tuple(dims)


def _make_activation(name: str):
    key = str(name).strip().lower()
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(
        "Unsupported activation. Use one of: elu, gelu, silu, tanh, relu, leaky_relu."
    )


class CoreMLP(nn.Module):
    """Core MLP in normalized space with configurable width/depth/activation."""

    def __init__(self, in_dim, out_dim, hidden_dims, activation="elu", dropout=0.0):
        super().__init__()
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}.")

        dims = [int(in_dim)] + [int(d) for d in hidden_dims] + [int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Full model: scale -> MLP -> unscale
# -----------------------------
class Case2Model(nn.Module):
    """
    X_raw = (mu1, mu2, t) -> q_s_raw
    Scaling is embedded as buffers.
    """
    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims, activation="elu", dropout=0.0):
        super().__init__()
        in_dim = x_mean.shape[0]   # should be 3
        out_dim = y_mean.shape[0]  # n_s
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
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
        description="Experimental trainer for Case-2 ANN map (n=20 studies)."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional explicit dataset directory containing per_mu/ and meta.npy.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--primary-modes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--val-split-mode",
        choices=("row", "ecsw_param_time_stratified", "mu_group_holdout"),
        default="row",
        help="Validation strategy used for ANN training/evaluation split.",
    )
    parser.add_argument(
        "--val-snap-time-offset",
        type=int,
        default=1,
        help="ECSW-like split: minimum time column considered for validation (>=1).",
    )
    parser.add_argument(
        "--val-holdout-mu",
        type=str,
        default="",
        help="For mu_group_holdout mode: holdout parameter pair as 'mu1,mu2'.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
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
        type=str,
        default="elu",
        choices=("elu", "gelu", "silu", "tanh", "relu", "leaky_relu"),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--summary-name", type=str, default="case2_training_summary_test_n20.txt")
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
    model_name = str(args.model_name).strip() if args.model_name is not None else "case2_model_test_n20.pt"
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)
    summary_name = str(args.summary_name).strip() or "case2_training_summary_test_n20.txt"
    summary_path = os.path.join(STAGE3_DIR, summary_name)

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
        raise ValueError(f"--val-frac must be in (0, 0.5), got {val_frac}.")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}.")
    if lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {lr}.")
    if weight_decay < 0.0:
        raise ValueError(f"--weight-decay must be >= 0, got {weight_decay}.")
    if epochs <= 0 or patience <= 0:
        raise ValueError("--epochs and --patience must be positive.")
    if min_improve < 0.0:
        raise ValueError("--min-improve must be >= 0.")
    if not (0.0 < lr_scheduler_factor < 1.0):
        raise ValueError(
            f"--lr-scheduler-factor must be in (0,1), got {lr_scheduler_factor}."
        )
    if lr_scheduler_patience <= 0:
        raise ValueError("--lr-scheduler-patience must be positive.")
    if lr_scheduler_min_lr <= 0.0:
        raise ValueError("--lr-scheduler-min-lr must be positive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Case2] device = {device}")
    print(f"[Case2] dataset_dir = {dataset_dir}")
    print(f"[Case2] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case2] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case2] primary_modes (training split) = {primary_modes}")
    print(f"[Case2] hidden_dims = {hidden_dims}")
    print(f"[Case2] activation = {activation}")
    print(f"[Case2] dropout = {dropout}")
    print(
        "[Case2] lr_scheduler = ReduceLROnPlateau("
        f"factor={lr_scheduler_factor}, patience={lr_scheduler_patience}, "
        f"min_lr={lr_scheduler_min_lr:.3e})"
    )
    print(f"[Case2] seed = {seed}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_raw, Y_raw = load_prom_dataset_case2(dataset_root, primary_modes=primary_modes)
    M, in_dim = X_raw.shape
    _, n_s = Y_raw.shape
    if in_dim != 3:
        raise ValueError(f"[Case2] Expected X dim=3 (mu1,mu2,t), got {in_dim}")
    print(f"[Case2] Loaded: M={M}, in_dim={in_dim}, n_s={n_s}")

    # -----------------------------
    # Split
    # -----------------------------
    split_mode = str(args.val_split_mode).strip().lower()
    if split_mode == "row":
        idx = np.arange(M, dtype=np.int64)
        tr_idx, va_idx = train_test_split(idx, test_size=val_frac, random_state=seed, shuffle=True)
        split_info = {
            "split_mode": "row",
            "num_mu_groups": int(np.unique(np.round(X_raw[:, :2], decimals=12), axis=0).shape[0]),
            "num_time_per_mu": int(0),
            "num_candidates_total": int(M),
            "num_selected_total": int(va_idx.size),
            "val_frac_requested": float(val_frac),
            "val_frac_actual": float(va_idx.size / max(1, M)),
            "snap_time_offset": int(0),
            "holdout_mu1": None,
            "holdout_mu2": None,
        }
        print(
            "[Case2] split = row (train_test_split shuffle) "
            f"(val_requested={split_info['val_frac_requested']:.4f}, "
            f"val_actual={split_info['val_frac_actual']:.4f})"
        )
    elif split_mode == "ecsw_param_time_stratified":
        tr_idx, va_idx, split_info = split_indices_ecsw_param_time(
            X_raw,
            val_frac=float(val_frac),
            seed=int(seed),
            snap_time_offset=int(args.val_snap_time_offset),
            ensure_mu_coverage=True,
        )
        print(
            "[Case2] split = ecsw_param_time_stratified "
            f"(mu_groups={split_info['num_mu_groups']}, "
            f"time_per_mu={split_info['num_time_per_mu']}, "
            f"val_requested={split_info['val_frac_requested']:.4f}, "
            f"val_actual={split_info['val_frac_actual']:.4f})"
        )
    else:
        holdout_mu = None
        if str(args.val_holdout_mu).strip():
            parts = [s.strip() for s in str(args.val_holdout_mu).split(",")]
            if len(parts) != 2:
                raise ValueError("--val-holdout-mu must be 'mu1,mu2'.")
            holdout_mu = (float(parts[0]), float(parts[1]))
        tr_idx, va_idx, split_info = split_indices_holdout_mu_group(
            X_raw,
            holdout_mu=holdout_mu,
            avoid_center_and_corners=True,
        )
        print(
            "[Case2] split = mu_group_holdout "
            f"(holdout_mu=({split_info['holdout_mu1']:.3f},{split_info['holdout_mu2']:.4f}), "
            f"mu_groups={split_info['num_mu_groups']}, "
            f"val_samples={split_info['num_selected_total']})"
        )

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
    model = Case2Model(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    ).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Case2] trainable_parameters = {trainable_parameters}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min_lr,
    )
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

        scheduler.step(va_loss)

        if ep == 1 or ep % 25 == 0:
            lr_current = opt.param_groups[0]["lr"]
            print(
                f"[Epoch {ep:4d}] train_mse={tr_loss:.6e} | "
                f"val_mse={va_loss:.6e} | lr={lr_current:.3e} | bad={bad}"
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
        Xtr_t_eval = torch.from_numpy(Xtr).to(device)
        Ytr_t_eval = torch.from_numpy(Ytr).to(device)
        pred_tr = model(Xtr_t_eval)
        pred_va = model(Xva_t)
        train_rel_frob_percent = 100.0 * (
            torch.linalg.norm(pred_tr - Ytr_t_eval) / torch.linalg.norm(Ytr_t_eval)
        ).detach().cpu().item()
        val_rel_frob_percent = 100.0 * (
            torch.linalg.norm(pred_va - Yva_t) / torch.linalg.norm(Yva_t)
        ).detach().cpu().item()

    print(f"[Case2] Training done in {time.time() - t0:.2f}s. best_val={best_val:.6e}")
    print(f"[Case2] train_rel_frob_percent = {train_rel_frob_percent:.4f}%")
    print(f"[Case2] val_rel_frob_percent = {val_rel_frob_percent:.4f}%")

    # -----------------------------
    # Save ONLY one file (weights + scaler buffers)
    # -----------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": int(in_dim),   # should be 3
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
        "trainable_parameters": int(trainable_parameters),
        "best_val_mse": float(best_val),
        "train_rel_frob_percent": float(train_rel_frob_percent),
        "val_rel_frob_percent": float(val_rel_frob_percent),
        "epochs": int(epochs),
        "patience": int(patience),
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_factor": float(lr_scheduler_factor),
        "lr_scheduler_patience": int(lr_scheduler_patience),
        "lr_scheduler_min_lr": float(lr_scheduler_min_lr),
        "val_split": str(split_info["split_mode"]),
        "val_split_snap_time_offset": int(split_info["snap_time_offset"]),
        "val_frac_actual": float(split_info["val_frac_actual"]),
        "val_holdout_mu1": split_info.get("holdout_mu1"),
        "val_holdout_mu2": split_info.get("holdout_mu2"),
        "mapping": "qN_s = N(mu1, mu2, t)",
    }
    torch.save(ckpt, model_path)
    print(f"[Case2] Saved model checkpoint: {model_path}")
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
            ("in_dim", in_dim),
            ("n_s", n_s),
            ("hidden_dims", tuple(int(d) for d in hidden_dims)),
            ("activation", activation),
            ("dropout", float(dropout)),
            ("batch_size", int(batch_size)),
            ("lr", float(lr)),
            ("weight_decay", float(weight_decay)),
            ("trainable_parameters", int(trainable_parameters)),
            ("epochs", int(epochs)),
            ("patience", int(patience)),
            ("lr_scheduler", "ReduceLROnPlateau"),
            ("lr_scheduler_factor", float(lr_scheduler_factor)),
            ("lr_scheduler_patience", int(lr_scheduler_patience)),
            ("lr_scheduler_min_lr", float(lr_scheduler_min_lr)),
            ("epochs_ran", ep),
            ("best_val_mse", best_val),
            ("train_rel_frob_percent", train_rel_frob_percent),
            ("val_rel_frob_percent", val_rel_frob_percent),
            ("val_split", str(split_info["split_mode"])),
            ("val_split_snap_time_offset", int(split_info["snap_time_offset"])),
            ("val_frac_requested", float(val_frac)),
            ("val_frac_actual", float(split_info["val_frac_actual"])),
            ("val_holdout_mu1", split_info.get("holdout_mu1")),
            ("val_holdout_mu2", split_info.get("holdout_mu2")),
            ("n_unique_mu_groups", int(split_info["num_mu_groups"])),
            ("n_time_per_mu", int(split_info["num_time_per_mu"])),
            ("n_candidates_total", int(split_info["num_candidates_total"])),
            ("n_selected_total", int(split_info["num_selected_total"])),
            ("seed", seed),
            ("device", device),
        ],
    )
    print(f"[Case2] Summary: {summary_path}")


if __name__ == "__main__":
    main()
