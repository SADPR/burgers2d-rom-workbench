
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage3_perform_training_rom_data_driven.py

Full trajectory surrogate (ANN):
    qN = G(mu1, mu2, t)

- Loads PROM-solved coefficients from:
    prom_coeff_dataset_ntot*/per_mu/*/mu.npy     (2,)
    prom_coeff_dataset_ntot*/per_mu/*/t.npy      (T,)
    prom_coeff_dataset_ntot*/per_mu/*/qN.npy     (n_tot, T)

- Builds dataset:
    X_raw = [mu1, mu2, t]  -> shape (M, 3)
    Y_raw = qN^T           -> shape (M, n_tot)   where n_tot = n + n_s

- Embeds scaling inside the model (so inference is just model(X_raw)).
- Saves ONLY:
    rom_data_driven_model.pt
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
    from stage3_qn_utils import load_qn_from_mu_dir
except ModuleNotFoundError:
    from .stage3_qn_utils import load_qn_from_mu_dir
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


def load_prom_dataset_rom_data_driven(dataset_root: str):
    """
    Return X_raw (M,3), Y_raw (M,n_tot) in float32 from per_mu dirs.

    Each per_mu dir must contain:
      - mu.npy     (2,)
      - t.npy      (T,)
      - qN.npy     (n_tot, T)
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
    n_tot_ref = None

    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)

        mu = np.load(os.path.join(mu_dir, "mu.npy")).astype(np.float64).reshape(-1)
        if mu.size != 2:
            raise ValueError(f"{sd}: mu.npy must have shape (2,), got {mu.shape}")

        t = np.load(os.path.join(mu_dir, "t.npy")).astype(np.float64).reshape(-1)    # (T,)

        qN = load_qn_from_mu_dir(mu_dir).astype(np.float64)   # (n_tot, T)
        if qN.ndim != 2:
            raise ValueError(f"{sd}: qN must be 2D (n_tot,T), got {qN.shape}")

        n_tot, T = qN.shape
        if t.shape[0] != T:
            raise ValueError(f"{sd}: t has length {t.shape[0]} but qN has T={T}")

        if n_tot_ref is None:
            n_tot_ref = n_tot
        elif n_tot != n_tot_ref:
            raise ValueError(f"{sd}: n_tot mismatch, got {n_tot}, expected {n_tot_ref}")

        # Build X for this trajectory: repeat mu across time
        mu1 = np.full((T,), mu[0], dtype=np.float64)
        mu2 = np.full((T,), mu[1], dtype=np.float64)
        Xi = np.column_stack([mu1, mu2, t])          # (T,3)

        # Build Y directly from qN
        Yi = qN.T                                    # (T, n_tot)

        X_list.append(Xi)
        Y_list.append(Yi)

    X_raw = np.vstack(X_list).astype(np.float32)     # (M,3)
    Y_raw = np.vstack(Y_list).astype(np.float32)     # (M,n_tot)

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
# Configurable core MLP in normalized space.
# -----------------------------
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


def _make_loss(name: str):
    key = str(name).strip().lower()
    if key == "mse":
        return nn.MSELoss()
    if key == "smooth_l1":
        return nn.SmoothL1Loss()
    if key == "l1":
        return nn.L1Loss()
    raise ValueError("Unsupported loss. Use one of: mse, smooth_l1, l1.")


class CoreMLP(nn.Module):
    hidden_dims = (32, 64, 128, 256, 256)
    activation_name = "elu"

    def __init__(self, in_dim, out_dim, hidden_dims=None, activation=None, dropout=0.0):
        super().__init__()
        hidden_dims = self.hidden_dims if hidden_dims is None else tuple(int(d) for d in hidden_dims)
        activation = self.activation_name if activation is None else str(activation).strip().lower()
        dropout = float(dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}.")

        dims = [int(in_dim)] + list(hidden_dims) + [int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Full model: scale -> MLP -> unscale
# -----------------------------
class ROMDataDrivenModel(nn.Module):
    """
    X_raw = (mu1, mu2, t) -> qN_raw (n_tot,)
    Scaling is embedded as buffers.
    """
    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims=None, activation=None, dropout=0.0):
        super().__init__()
        in_dim = x_mean.shape[0]   # should be 3
        out_dim = y_mean.shape[0]  # n_tot
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(
            in_dim,
            out_dim,
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
        description="Train data-driven ROM surrogate from Stage-2 dataset."
    )
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional explicit dataset directory containing per_mu/ and meta.npy.",
    )
    parser.add_argument(
        "--validation-dataset-dir",
        type=str,
        default=None,
        help=(
            "Optional explicit validation dataset directory containing per_mu/ and meta.npy. "
            "When provided, no random row split is used."
        ),
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--summary-name", type=str, default="rom_data_driven_training_summary.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=5000)
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
    parser.add_argument(
        "--loss-function",
        type=str,
        default="mse",
        choices=("mse", "smooth_l1", "l1"),
        help="Training loss applied either in raw or normalized output space.",
    )
    parser.add_argument(
        "--loss-space",
        type=str,
        default="raw",
        choices=("raw", "normalized"),
        help=(
            "raw: loss on physical qN coefficients; normalized: loss after "
            "z-score normalization of qN using training statistics."
        ),
    )
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
    model_name = str(args.model_name).strip() if args.model_name is not None else "rom_data_driven_model.pt"
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)
    summary_name = str(args.summary_name).strip() or "rom_data_driven_training_summary.txt"
    summary_path = os.path.join(STAGE3_DIR, summary_name)

    VAL_FRAC = float(args.val_frac)
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
    loss_function = str(args.loss_function).strip().lower()
    loss_space = str(args.loss_space).strip().lower()

    external_validation = args.validation_dataset_dir is not None
    if (not external_validation) and not (0.0 < VAL_FRAC < 0.5):
        raise ValueError(f"--val-frac must be in (0,0.5), got {VAL_FRAC}.")
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
        raise ValueError("--lr-scheduler-factor must be in (0,1).")
    if lr_scheduler_patience <= 0:
        raise ValueError("--lr-scheduler-patience must be positive.")
    if lr_scheduler_min_lr <= 0.0:
        raise ValueError("--lr-scheduler-min-lr must be positive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ROM-DataDriven] device = {device}")
    print(f"[ROM-DataDriven] dataset_dir = {dataset_dir}")
    print(f"[ROM-DataDriven] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[ROM-DataDriven] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[ROM-DataDriven] mapping = qN = G(mu1, mu2, t)")
    print(f"[ROM-DataDriven] hidden_dims = {hidden_dims}")
    print(f"[ROM-DataDriven] activation = {activation}")
    print(f"[ROM-DataDriven] dropout = {dropout}")
    print(f"[ROM-DataDriven] loss_function = {loss_function}")
    print(f"[ROM-DataDriven] loss_space = {loss_space}")
    print(f"[ROM-DataDriven] optimizer = AdamW")
    print(f"[ROM-DataDriven] lr = {lr:.3e}")
    print(f"[ROM-DataDriven] weight_decay = {weight_decay:.3e}")
    print(f"[ROM-DataDriven] batch_size = {batch_size}")
    print(f"[ROM-DataDriven] epochs = {epochs}")
    print(f"[ROM-DataDriven] patience = {patience}")
    print(f"[ROM-DataDriven] min_improve = {min_improve:.3e}")
    print(f"[ROM-DataDriven] clip_grad = {clip_grad}")
    print(
        "[ROM-DataDriven] lr_scheduler = ReduceLROnPlateau("
        f"factor={lr_scheduler_factor}, patience={lr_scheduler_patience}, "
        f"min_lr={lr_scheduler_min_lr:.3e})"
    )
    if external_validation:
        print("[ROM-DataDriven] val_split = external validation dataset")
        print(f"[ROM-DataDriven] validation_dataset_dir = {args.validation_dataset_dir}")
    else:
        print("[ROM-DataDriven] val_split = row (train_test_split shuffle)")
        print(f"[ROM-DataDriven] val_frac = {VAL_FRAC}")
    print(f"[ROM-DataDriven] scaling = z-score on train split for X and Y")
    print(f"[ROM-DataDriven] seed = {seed}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_raw, Y_raw = load_prom_dataset_rom_data_driven(dataset_root)
    M, in_dim = X_raw.shape
    _, n_tot = Y_raw.shape
    if in_dim != 3:
        raise ValueError(f"[ROM-DataDriven] Expected X dim=3 (mu1,mu2,t), got {in_dim}")
    print(f"[ROM-DataDriven] Loaded training candidate data: M={M}, in_dim={in_dim}, n_tot={n_tot}")

    # -----------------------------
    # Split
    # -----------------------------
    validation_dataset_dir = None
    validation_dataset_root = None
    validation_dataset_meta = None
    if external_validation:
        (
            validation_dataset_root,
            validation_dataset_ntot,
            validation_dataset_dir,
            validation_dataset_meta,
            _,
        ) = resolve_stage3_dataset(
            this_dir=THIS_DIR,
            requested_ntot=dataset_ntot,
            expected_backend=dataset_backend,
            requested_dataset_dir=args.validation_dataset_dir,
        )
        if int(validation_dataset_ntot) != int(dataset_ntot):
            raise ValueError(
                f"Validation dataset ntot={validation_dataset_ntot}, expected {dataset_ntot}."
            )
        Xva, Yva = load_prom_dataset_rom_data_driven(validation_dataset_root)
        if Xva.shape[1] != in_dim or Yva.shape[1] != n_tot:
            raise ValueError(
                "Validation data dimensions do not match training data: "
                f"Xva={Xva.shape}, Yva={Yva.shape}, expected (*,{in_dim}) and (*,{n_tot})."
            )
        Xtr, Ytr = X_raw, Y_raw
        tr_idx = np.arange(Xtr.shape[0], dtype=np.int64)
        va_idx = np.arange(Xva.shape[0], dtype=np.int64)
    else:
        idx = np.arange(M, dtype=np.int64)
        tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=seed, shuffle=True)

        Xtr, Ytr = X_raw[tr_idx], Y_raw[tr_idx]
        Xva, Yva = X_raw[va_idx], Y_raw[va_idx]
    print(f"[ROM-DataDriven] train_samples = {Xtr.shape[0]}")
    print(f"[ROM-DataDriven] val_samples = {Xva.shape[0]}")

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
    model = ROMDataDrivenModel(
        x_mean,
        x_std,
        y_mean,
        y_std,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ROM-DataDriven] trainable_parameters = {trainable_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min_lr,
    )
    loss_fn = _make_loss(loss_function)

    def loss_inputs(pred_raw, target_raw):
        if loss_space == "raw":
            return pred_raw, target_raw
        if loss_space == "normalized":
            y_mean_t = model.unscaler.mean
            y_std_t = model.unscaler.std
            return (pred_raw - y_mean_t) / y_std_t, (target_raw - y_mean_t) / y_std_t
        raise RuntimeError(f"Unhandled loss_space={loss_space!r}")

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
            pred_loss, yb_loss = loss_inputs(pred, yb)
            loss = loss_fn(pred_loss, yb_loss)
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()
            tr_loss += float(loss.detach().cpu().item()) * xb.shape[0]

        tr_loss /= Xtr.shape[0]

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva_t)
            va_pred_loss, Yva_loss = loss_inputs(va_pred, Yva_t)
            va_loss = float(loss_fn(va_pred_loss, Yva_loss).detach().cpu().item())

        scheduler.step(va_loss)

        if ep == 1 or ep % 25 == 0:
            lr_current = opt.param_groups[0]["lr"]
            print(
                f"[Epoch {ep:4d}] train_loss={tr_loss:.6e} | "
                f"val_loss={va_loss:.6e} | lr={lr_current:.3e} | bad={bad}"
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

    print(f"[ROM-DataDriven] Training done in {time.time() - t0:.2f}s. best_val={best_val:.6e}")

    model.eval()
    with torch.no_grad():
        Ytr_pred = model(torch.from_numpy(Xtr).to(device)).detach().cpu().numpy()
        Yva_pred = model(Xva_t).detach().cpu().numpy()
    train_rel_frob_percent = 100.0 * np.linalg.norm(Ytr_pred - Ytr) / max(np.linalg.norm(Ytr), 1e-30)
    val_rel_frob_percent = 100.0 * np.linalg.norm(Yva_pred - Yva) / max(np.linalg.norm(Yva), 1e-30)
    print(f"[ROM-DataDriven] train_rel_frob_percent = {train_rel_frob_percent:.6f}%")
    print(f"[ROM-DataDriven] val_rel_frob_percent = {val_rel_frob_percent:.6f}%")

    # -----------------------------
    # Save ONLY one file (weights + scaler buffers)
    # -----------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": int(in_dim),     # should be 3
        "n_tot": int(n_tot),
        "seed": int(seed),
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "validation_dataset_root": validation_dataset_root,
        "validation_dataset_dir": validation_dataset_dir,
        "validation_dataset_backend": (
            None if validation_dataset_meta is None else validation_dataset_meta.get("solve_backend")
        ),
        "hidden_dims": tuple(int(d) for d in hidden_dims),
        "activation": activation,
        "dropout": float(dropout),
        "loss_function": loss_function,
        "loss_space": loss_space,
        "optimizer": "AdamW",
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "patience": int(patience),
        "min_improve": float(min_improve),
        "clip_grad": float(clip_grad),
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_factor": float(lr_scheduler_factor),
        "lr_scheduler_patience": int(lr_scheduler_patience),
        "lr_scheduler_min_lr": float(lr_scheduler_min_lr),
        "best_val_loss": float(best_val),
        "val_split": "external_dataset" if external_validation else "row",
        "val_frac": None if external_validation else float(VAL_FRAC),
        "train_samples": int(Xtr.shape[0]),
        "val_samples": int(Xva.shape[0]),
        "scaling": "z-score train split for X and Y",
        "trainable_parameters": int(trainable_params),
        "train_rel_frob_percent": float(train_rel_frob_percent),
        "val_rel_frob_percent": float(val_rel_frob_percent),
        "mapping": "qN = G(mu1, mu2, t)",
    }
    torch.save(ckpt, model_path)
    print(f"[ROM-DataDriven] Saved model checkpoint: {model_path}")
    write_kv_txt(
        summary_path,
        [
            ("model_name", model_name),
            ("model_path", model_path),
            ("dataset_dir", dataset_dir),
            ("dataset_root", dataset_root),
            ("dataset_ntot", dataset_ntot),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("validation_dataset_dir", validation_dataset_dir),
            ("validation_dataset_root", validation_dataset_root),
            (
                "validation_dataset_backend",
                None if validation_dataset_meta is None else validation_dataset_meta.get("solve_backend"),
            ),
            ("samples_M", int(Xtr.shape[0] + Xva.shape[0])),
            ("train_samples", int(Xtr.shape[0])),
            ("val_samples", int(Xva.shape[0])),
            ("in_dim", in_dim),
            ("n_tot", n_tot),
            ("hidden_dims", tuple(int(d) for d in hidden_dims)),
            ("activation", activation),
            ("dropout", float(dropout)),
            ("loss_function", loss_function),
            ("loss_space", loss_space),
            ("optimizer", "AdamW"),
            ("batch_size", batch_size),
            ("lr", lr),
            ("weight_decay", weight_decay),
            ("epochs", epochs),
            ("patience", patience),
            ("epochs_ran", ep),
            ("best_val_loss", best_val),
            ("best_val_mse", best_val),
            ("train_rel_frob_percent", train_rel_frob_percent),
            ("val_rel_frob_percent", val_rel_frob_percent),
            ("min_improve", min_improve),
            ("clip_grad", clip_grad),
            ("lr_scheduler", "ReduceLROnPlateau"),
            ("lr_scheduler_factor", lr_scheduler_factor),
            ("lr_scheduler_patience", lr_scheduler_patience),
            ("lr_scheduler_min_lr", lr_scheduler_min_lr),
            ("val_split", "external_dataset" if external_validation else "row"),
            ("val_frac", None if external_validation else VAL_FRAC),
            ("scaling", "z-score train split for X and Y"),
            ("trainable_parameters", trainable_params),
            ("seed", seed),
            ("device", device),
        ],
    )
    print(f"[ROM-DataDriven] Summary: {summary_path}")


if __name__ == "__main__":
    main()
