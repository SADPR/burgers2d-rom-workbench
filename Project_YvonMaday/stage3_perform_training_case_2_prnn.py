#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case-2 PRNN-style Stage-3 trainer (proper residual-based version).

Case 2 map:
    qN_s = N(mu1, mu2, t)

Training objective:
    L = omega_data * L_data + omega_residual * L_res

with
    L_data = MSE(qN_s_pred, qN_s_ref)
    L_res  = mean || P^T r(u_hat; mu) ||_2^2

where
    u_hat = u_ref + V qN_p_ref + Vbar qN_s_pred
    P in {V_tot, V}

The residual gradient wrt predicted secondary coordinates is computed manually:
    d L_res / d qN_s_pred = (2/k) Vbar^T J(u_hat)^T P (P^T r(u_hat))
with k = dim(P). This requires only first derivatives of the FEM residual.
"""

import os
import sys
import time
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset
try:
    from stage3_qn_utils import load_qn_from_mu_dir, resolve_primary_modes, split_qn
except ModuleNotFoundError:
    from .stage3_qn_utils import load_qn_from_mu_dir, resolve_primary_modes, split_qn
try:
    from project_layout import (
        STAGE3_DIR,
        ensure_layout_dirs,
        stage3_model_path,
        write_kv_txt,
        resolve_stage1_artifact,
    )
except ModuleNotFoundError:
    from .project_layout import (
        STAGE3_DIR,
        ensure_layout_dirs,
        stage3_model_path,
        write_kv_txt,
        resolve_stage1_artifact,
    )

from burgers.config import DT, GRID_X, GRID_Y
from burgers.core import get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D


def load_prom_dataset_case2_with_primary(dataset_root: str, primary_modes: int):
    """
    Build training arrays from Stage-2 per_mu dataset.

    Returns:
      X_raw      (M,3)     [mu1, mu2, t]
      Y_raw      (M,n_s)   target secondary coefficients qN_s
      QP_raw     (M,n_p)   reference primary coefficients qN_p (for u_hat)
      QP_prev    (M,n_p)   previous-step qN_p (for residual state wp)
      QS_prev    (M,n_s)   previous-step qN_s (for residual state wp)
      active_res (M,)      1 if residual is active (time step > 0), else 0
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
    QP_list, QP_prev_list, QS_prev_list = [], [], []
    active_list = []

    n_p_ref = None
    n_s_ref = None

    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)

        mu = np.load(os.path.join(mu_dir, "mu.npy")).astype(np.float64).reshape(-1)
        if mu.size != 2:
            raise ValueError(f"{sd}: mu.npy must have shape (2,), got {mu.shape}")

        t = np.load(os.path.join(mu_dir, "t.npy")).astype(np.float64).reshape(-1)  # (T,)
        qN = load_qn_from_mu_dir(mu_dir).astype(np.float64)  # (n_tot, T)
        qNp, qNs = split_qn(qN, primary_modes)  # (n_p,T), (n_s,T)

        n_p, T = qNp.shape
        n_s, T2 = qNs.shape
        if T != T2:
            raise ValueError(f"{sd}: split mismatch T_primary={T}, T_secondary={T2}")
        if t.shape[0] != T:
            raise ValueError(f"{sd}: t has length {t.shape[0]} but qN has T={T}")

        if n_p_ref is None:
            n_p_ref = n_p
            n_s_ref = n_s
        elif n_p != n_p_ref or n_s != n_s_ref:
            raise ValueError(
                f"{sd}: mode split mismatch, got (n_p,n_s)=({n_p},{n_s}), "
                f"expected ({n_p_ref},{n_s_ref})"
            )

        mu1 = np.full((T,), mu[0], dtype=np.float64)
        mu2 = np.full((T,), mu[1], dtype=np.float64)
        Xi = np.column_stack([mu1, mu2, t])  # (T,3)
        Yi = qNs.T  # (T,n_s)
        QPi = qNp.T  # (T,n_p)

        QPprev_i = np.zeros_like(QPi)
        QSprev_i = np.zeros_like(Yi)
        active_i = np.zeros((T,), dtype=np.float64)

        if T > 1:
            QPprev_i[1:, :] = QPi[:-1, :]
            QSprev_i[1:, :] = Yi[:-1, :]
            # dummy values for first time row (unused when active=0)
            QPprev_i[0, :] = QPi[0, :]
            QSprev_i[0, :] = Yi[0, :]
            active_i[1:] = 1.0
        elif T == 1:
            QPprev_i[0, :] = QPi[0, :]
            QSprev_i[0, :] = Yi[0, :]
            active_i[0] = 0.0

        X_list.append(Xi)
        Y_list.append(Yi)
        QP_list.append(QPi)
        QP_prev_list.append(QPprev_i)
        QS_prev_list.append(QSprev_i)
        active_list.append(active_i)

    X_raw = np.vstack(X_list).astype(np.float32)
    Y_raw = np.vstack(Y_list).astype(np.float32)
    QP_raw = np.vstack(QP_list).astype(np.float32)
    QP_prev = np.vstack(QP_prev_list).astype(np.float32)
    QS_prev = np.vstack(QS_prev_list).astype(np.float32)
    active_res = np.concatenate(active_list).astype(np.float32)

    return X_raw, Y_raw, QP_raw, QP_prev, QS_prev, active_res


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
    raise ValueError("Unsupported activation. Use one of: elu, gelu, silu, tanh, relu, leaky_relu.")


class CoreMLP(nn.Module):
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


class Case2Model(nn.Module):
    """
    X_raw = (mu1, mu2, t) -> qN_s_raw
    Scaling is embedded as buffers.
    """

    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims, activation="elu", dropout=0.0):
        super().__init__()
        in_dim = x_mean.shape[0]
        out_dim = y_mean.shape[0]
        self.scaler = Scaler(x_mean[None, :], x_std[None, :])
        self.core = CoreMLP(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, activation=activation, dropout=dropout)
        self.unscaler = Unscaler(y_mean[None, :], y_std[None, :])

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw


def _build_basis_and_ops(dataset_ntot: int, primary_modes: int):
    basis_path = resolve_stage1_artifact("basis.npy")
    uref_path = resolve_stage1_artifact("u_ref.npy")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got {basis.shape}")
    if basis.shape[1] < dataset_ntot:
        raise ValueError(
            f"basis.npy has {basis.shape[1]} modes but dataset_ntot={dataset_ntot} required"
        )

    vtot = basis[:, :dataset_ntot]
    v = vtot[:, :primary_modes]
    vbar = vtot[:, primary_modes:dataset_ntot]

    if os.path.exists(uref_path):
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(vtot.shape[0], dtype=np.float64)

    if u_ref.size != vtot.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {vtot.shape[0]}"
        )

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(GRID_X, GRID_Y)

    return {
        "basis_path": basis_path,
        "uref_path": uref_path,
        "Vtot": np.ascontiguousarray(vtot, dtype=np.float64),
        "V": np.ascontiguousarray(v, dtype=np.float64),
        "Vbar": np.ascontiguousarray(vbar, dtype=np.float64),
        "u_ref": np.ascontiguousarray(u_ref, dtype=np.float64),
        "Dxec": Dxec,
        "Dyec": Dyec,
        "JDxec": JDxec,
        "JDyec": JDyec,
        "Eye": Eye,
    }


def _compute_batch_residual_loss_and_grad(
    pred_qs: torch.Tensor,
    xb: torch.Tensor,
    qpb: torch.Tensor,
    qppb: torch.Tensor,
    qspb: torch.Tensor,
    activeb: torch.Tensor,
    cache: dict,
    projection: str,
    physics_subsample: int,
    with_grad: bool,
):
    """
    Compute projected residual loss and manual gradient wrt pred_qs for a batch.

    Returns:
      loss_res_avg (float)
      grad_tensor  (B,n_s) torch.Tensor on pred device, or None when with_grad=False
      n_used       number of active residual samples actually used
    """
    device = pred_qs.device
    dtype = pred_qs.dtype

    pred_np = pred_qs.detach().cpu().numpy().astype(np.float64)
    x_np = xb.detach().cpu().numpy().astype(np.float64)
    qp_np = qpb.detach().cpu().numpy().astype(np.float64)
    qpp_np = qppb.detach().cpu().numpy().astype(np.float64)
    qsp_np = qspb.detach().cpu().numpy().astype(np.float64)
    act_np = activeb.detach().cpu().numpy().reshape(-1)

    B = pred_np.shape[0]
    active_idx = np.where(act_np > 0.5)[0]
    if active_idx.size == 0:
        grad_out = torch.zeros_like(pred_qs) if with_grad else None
        return 0.0, grad_out, 0

    if physics_subsample > 0 and active_idx.size > physics_subsample:
        used_idx = np.random.choice(active_idx, size=physics_subsample, replace=False)
    else:
        used_idx = active_idx

    V = cache["V"]
    Vtot = cache["Vtot"]
    Vbar = cache["Vbar"]
    u_ref = cache["u_ref"]
    Dxec = cache["Dxec"]
    Dyec = cache["Dyec"]
    JDxec = cache["JDxec"]
    JDyec = cache["JDyec"]
    Eye = cache["Eye"]

    if projection == "vtot":
        P = Vtot
    elif projection == "v":
        P = V
    else:
        raise ValueError(f"Unknown projection '{projection}', expected 'vtot' or 'v'.")

    k = float(P.shape[1])
    n_s = Vbar.shape[1]

    grad_np = np.zeros((B, n_s), dtype=np.float64) if with_grad else None
    loss_acc = 0.0

    for ii in used_idx:
        mu = x_np[ii, :2]

        qbar = pred_np[ii, :]
        qp = qp_np[ii, :]
        qpp = qpp_np[ii, :]
        qsp = qsp_np[ii, :]

        u_hat = u_ref + V @ qp + Vbar @ qbar
        u_prev = u_ref + V @ qpp + Vbar @ qsp

        r = inviscid_burgers_res2D(u_hat, GRID_X, GRID_Y, DT, u_prev, mu, Dxec, Dyec)
        g = P.T @ r

        loss_i = float(np.dot(g, g) / k)
        loss_acc += loss_i

        if with_grad:
            J = inviscid_burgers_exact_jac2D(u_hat, DT, JDxec, JDyec, Eye)
            Pg = P @ g
            tmp = J.T @ Pg
            tmp = np.asarray(tmp, dtype=np.float64).reshape(-1)
            grad_i = (2.0 / k) * (Vbar.T @ tmp)
            grad_np[ii, :] = grad_i

    n_used = int(used_idx.size)
    loss_avg = loss_acc / float(max(n_used, 1))

    if with_grad:
        grad_np /= float(max(n_used, 1))
        grad_t = torch.from_numpy(grad_np).to(device=device, dtype=dtype)
    else:
        grad_t = None

    return loss_avg, grad_t, n_used


def _evaluate_validation(
    model: nn.Module,
    dl_val: DataLoader,
    loss_fn: nn.Module,
    cache: dict,
    projection: str,
    val_physics_subsample: int,
    omega_data: float,
    omega_residual: float,
    device: str,
):
    model.eval()

    data_sum = 0.0
    data_count = 0
    res_sum = 0.0
    res_count = 0

    with torch.no_grad():
        for xb, yb, qpb, qppb, qspb, activeb in dl_val:
            xb = xb.to(device)
            yb = yb.to(device)
            qpb = qpb.to(device)
            qppb = qppb.to(device)
            qspb = qspb.to(device)
            activeb = activeb.to(device)

            pred = model(xb)
            ld = float(loss_fn(pred, yb).detach().cpu().item())
            bs = xb.shape[0]

            data_sum += ld * bs
            data_count += bs

            if omega_residual > 0.0:
                lres, _, n_used = _compute_batch_residual_loss_and_grad(
                    pred_qs=pred,
                    xb=xb,
                    qpb=qpb,
                    qppb=qppb,
                    qspb=qspb,
                    activeb=activeb,
                    cache=cache,
                    projection=projection,
                    physics_subsample=val_physics_subsample,
                    with_grad=False,
                )
                if n_used > 0:
                    res_sum += lres * n_used
                    res_count += n_used

    va_data = data_sum / float(max(data_count, 1))
    va_res = res_sum / float(max(res_count, 1)) if res_count > 0 else 0.0
    va_total = omega_data * va_data + omega_residual * va_res

    return va_total, va_data, va_res, res_count


def main(argv=None):
    ensure_layout_dirs()

    parser = argparse.ArgumentParser(description="Case-2 PRNN trainer with projected residual loss.")
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--primary-modes", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-improve", type=float, default=1e-12)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument(
        "--log-epoch-every",
        type=int,
        default=10,
        help="Print epoch-level metrics every N epochs (epoch 1 is always printed).",
    )
    parser.add_argument(
        "--log-batch-every",
        type=int,
        default=0,
        help="If >0, print batch-level progress every N batches.",
    )

    parser.add_argument("--hidden-dims", type=str, default="32,64,128,256,256")
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        choices=("elu", "gelu", "silu", "tanh", "relu", "leaky_relu"),
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--omega-data", type=float, default=1.0)
    parser.add_argument("--omega-residual", type=float, default=1e-2)
    parser.add_argument("--physics-projection", choices=("vtot", "v"), default="vtot")
    parser.add_argument(
        "--physics-subsample",
        type=int,
        default=8,
        help="Number of active residual samples per training batch (<=0 means all active).",
    )
    parser.add_argument(
        "--val-physics-subsample",
        type=int,
        default=16,
        help="Number of active residual samples per validation batch (<=0 means all active).",
    )
    parser.add_argument(
        "--physics-every",
        type=int,
        default=1,
        help="Apply residual-loss term every N training batches (1 means every batch).",
    )

    parser.add_argument("--summary-name", type=str, default="case2_prnn_training_summary.txt")
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
    )

    primary_modes = resolve_primary_modes(args.primary_modes, dataset_meta, dataset_ntot)

    model_name = str(args.model_name).strip() if args.model_name is not None else "case2_model_prnn.pt"
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)

    summary_name = str(args.summary_name).strip() or "case2_prnn_training_summary.txt"
    summary_path = os.path.join(STAGE3_DIR, summary_name)

    val_frac = float(args.val_frac)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    epochs = int(args.epochs)
    patience = int(args.patience)
    min_improve = float(args.min_improve)
    clip_grad = float(args.clip_grad)
    log_epoch_every = int(args.log_epoch_every)
    log_batch_every = int(args.log_batch_every)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    activation = str(args.activation).strip().lower()
    dropout = float(args.dropout)

    omega_data = float(args.omega_data)
    omega_residual = float(args.omega_residual)
    physics_projection = str(args.physics_projection).strip().lower()
    physics_subsample = int(args.physics_subsample)
    val_physics_subsample = int(args.val_physics_subsample)
    physics_every = int(args.physics_every)

    if not (0.0 < val_frac < 0.5):
        raise ValueError(f"--val-frac must be in (0,0.5), got {val_frac}.")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}.")
    if lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {lr}.")
    if weight_decay < 0.0:
        raise ValueError(f"--weight-decay must be >=0, got {weight_decay}.")
    if epochs <= 0 or patience <= 0:
        raise ValueError("--epochs and --patience must be positive.")
    if min_improve < 0.0:
        raise ValueError("--min-improve must be >= 0.")
    if log_epoch_every <= 0:
        raise ValueError("--log-epoch-every must be positive.")
    if log_batch_every < 0:
        raise ValueError("--log-batch-every must be >= 0.")
    if omega_data < 0.0 or omega_residual < 0.0:
        raise ValueError("--omega-data and --omega-residual must be >=0.")
    if omega_data == 0.0 and omega_residual == 0.0:
        raise ValueError("At least one of --omega-data or --omega-residual must be >0.")
    if physics_every <= 0:
        raise ValueError("--physics-every must be positive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Case2-PRNN] device = {device}")
    print(f"[Case2-PRNN] dataset_dir = {dataset_dir}")
    print(f"[Case2-PRNN] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[Case2-PRNN] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[Case2-PRNN] primary_modes = {primary_modes}")
    print(f"[Case2-PRNN] hidden_dims = {hidden_dims}")
    print(f"[Case2-PRNN] activation = {activation}")
    print(f"[Case2-PRNN] dropout = {dropout}")
    print(f"[Case2-PRNN] omega_data = {omega_data}")
    print(f"[Case2-PRNN] omega_residual = {omega_residual}")
    print(f"[Case2-PRNN] physics_projection = {physics_projection}")
    print(f"[Case2-PRNN] physics_subsample = {physics_subsample}")
    print(f"[Case2-PRNN] val_physics_subsample = {val_physics_subsample}")
    print(f"[Case2-PRNN] physics_every = {physics_every}")
    print(f"[Case2-PRNN] log_epoch_every = {log_epoch_every}")
    print(f"[Case2-PRNN] log_batch_every = {log_batch_every}")
    print(f"[Case2-PRNN] seed = {seed}")

    X_raw, Y_raw, QP_raw, QP_prev, QS_prev, active_res = load_prom_dataset_case2_with_primary(
        dataset_root,
        primary_modes=primary_modes,
    )

    M, in_dim = X_raw.shape
    _, n_s = Y_raw.shape
    _, n_p = QP_raw.shape
    if in_dim != 3:
        raise ValueError(f"Expected input dim 3, got {in_dim}")
    if n_s != int(dataset_ntot - primary_modes):
        raise ValueError(
            f"Secondary dim mismatch: from data n_s={n_s}, expected {dataset_ntot - primary_modes}."
        )

    print(f"[Case2-PRNN] Loaded: M={M}, in_dim={in_dim}, n_p={n_p}, n_s={n_s}")
    print(f"[Case2-PRNN] Active residual rows: {int(np.sum(active_res > 0.5))}/{M}")

    idx = np.arange(M, dtype=np.int64)
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, random_state=seed, shuffle=True)

    Xtr, Ytr = X_raw[tr_idx], Y_raw[tr_idx]
    Xva, Yva = X_raw[va_idx], Y_raw[va_idx]

    QPtr, QPva = QP_raw[tr_idx], QP_raw[va_idx]
    QPprev_tr, QPprev_va = QP_prev[tr_idx], QP_prev[va_idx]
    QSprev_tr, QSprev_va = QS_prev[tr_idx], QS_prev[va_idx]
    Act_tr, Act_va = active_res[tr_idx], active_res[va_idx]

    x_mean = Xtr.mean(axis=0)
    x_std = Xtr.std(axis=0)
    y_mean = Ytr.mean(axis=0)
    y_std = Ytr.std(axis=0)

    model = Case2Model(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(Ytr),
            torch.from_numpy(QPtr),
            torch.from_numpy(QPprev_tr),
            torch.from_numpy(QSprev_tr),
            torch.from_numpy(Act_tr),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    dl_va = DataLoader(
        TensorDataset(
            torch.from_numpy(Xva),
            torch.from_numpy(Yva),
            torch.from_numpy(QPva),
            torch.from_numpy(QPprev_va),
            torch.from_numpy(QSprev_va),
            torch.from_numpy(Act_va),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    cache = _build_basis_and_ops(dataset_ntot=dataset_ntot, primary_modes=primary_modes)

    best_val = float("inf")
    best_state = None
    bad = 0

    t0 = time.time()
    num_tr_batches = len(dl_tr)
    for ep in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()

        tr_total_sum = 0.0
        tr_data_sum = 0.0
        tr_data_count = 0
        tr_res_sum = 0.0
        tr_res_count = 0

        if ep == 1 or (ep % log_epoch_every == 0):
            print(f"[Epoch {ep:4d}] start")

        for batch_idx, (xb, yb, qpb, qppb, qspb, activeb) in enumerate(dl_tr, start=1):
            batch_t0 = time.time()
            xb = xb.to(device)
            yb = yb.to(device)
            qpb = qpb.to(device)
            qppb = qppb.to(device)
            qspb = qspb.to(device)
            activeb = activeb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)

            loss_data_t = loss_fn(pred, yb)
            loss_data_val = float(loss_data_t.detach().cpu().item())

            lres_val = 0.0
            n_used = 0
            grad_res = None
            use_physics_batch = (omega_residual > 0.0) and (((batch_idx - 1) % physics_every) == 0)
            if use_physics_batch:
                lres_val, grad_res, n_used = _compute_batch_residual_loss_and_grad(
                    pred_qs=pred,
                    xb=xb,
                    qpb=qpb,
                    qppb=qppb,
                    qspb=qspb,
                    activeb=activeb,
                    cache=cache,
                    projection=physics_projection,
                    physics_subsample=physics_subsample,
                    with_grad=True,
                )

            need_retain = bool(omega_data > 0.0 and omega_residual > 0.0 and n_used > 0)
            if omega_data > 0.0:
                (omega_data * loss_data_t).backward(retain_graph=need_retain)

            if omega_residual > 0.0 and n_used > 0:
                pred.backward(gradient=omega_residual * grad_res)

            if clip_grad is not None and clip_grad > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            bs = xb.shape[0]
            tr_data_sum += loss_data_val * bs
            tr_data_count += bs

            if n_used > 0:
                tr_res_sum += lres_val * n_used
                tr_res_count += n_used

            total_batch = omega_data * loss_data_val + omega_residual * lres_val
            tr_total_sum += total_batch * bs

            if log_batch_every > 0 and (
                batch_idx == 1
                or batch_idx % log_batch_every == 0
                or batch_idx == num_tr_batches
            ):
                batch_elapsed = time.time() - batch_t0
                print(
                    f"[Epoch {ep:4d} | batch {batch_idx:4d}/{num_tr_batches}] "
                    f"data={loss_data_val:.3e} res={lres_val:.3e} nres={n_used} "
                    f"use_phys={int(use_physics_batch)} batch_s={batch_elapsed:.2f}"
                )

        tr_data = tr_data_sum / float(max(tr_data_count, 1))
        tr_res = tr_res_sum / float(max(tr_res_count, 1)) if tr_res_count > 0 else 0.0
        tr_total = tr_total_sum / float(max(tr_data_count, 1))

        va_total, va_data, va_res, va_res_count = _evaluate_validation(
            model=model,
            dl_val=dl_va,
            loss_fn=loss_fn,
            cache=cache,
            projection=physics_projection,
            val_physics_subsample=val_physics_subsample,
            omega_data=omega_data,
            omega_residual=omega_residual,
            device=device,
        )

        epoch_elapsed = time.time() - epoch_t0
        if ep == 1 or ep % log_epoch_every == 0:
            print(
                f"[Epoch {ep:4d}] "
                f"train_total={tr_total:.6e} (data={tr_data:.6e}, res={tr_res:.6e}, nres={tr_res_count}) | "
                f"val_total={va_total:.6e} (data={va_data:.6e}, res={va_res:.6e}, nres={va_res_count}) | "
                f"bad={bad} | epoch_s={epoch_elapsed:.2f}"
            )

        if va_total < best_val - min_improve:
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] epoch={ep} best_val_total={best_val:.6e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"[Case2-PRNN] Training done in {elapsed:.2f}s. best_val_total={best_val:.6e}")

    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": int(in_dim),
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
        "epochs": int(epochs),
        "patience": int(patience),
        "omega_data": float(omega_data),
        "omega_residual": float(omega_residual),
        "physics_projection": physics_projection,
        "physics_subsample": int(physics_subsample),
        "val_physics_subsample": int(val_physics_subsample),
        "physics_every": int(physics_every),
        "log_epoch_every": int(log_epoch_every),
        "log_batch_every": int(log_batch_every),
        "mapping": "qN_s = N(mu1, mu2, t)",
        "loss_form": "omega_data*MSE(qN_s) + omega_residual*mean(||P^T r(u_hat)||^2)",
        "residual_type": "projected_hdm_residual",
        "residual_dt": float(DT),
    }

    torch.save(ckpt, model_path)
    print(f"[Case2-PRNN] Saved model checkpoint: {model_path}")

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
            ("n_p", n_p),
            ("n_s", n_s),
            ("hidden_dims", tuple(int(d) for d in hidden_dims)),
            ("activation", activation),
            ("dropout", float(dropout)),
            ("batch_size", int(batch_size)),
            ("lr", float(lr)),
            ("weight_decay", float(weight_decay)),
            ("epochs", int(epochs)),
            ("patience", int(patience)),
            ("epochs_ran", ep),
            ("best_val_total", best_val),
            ("omega_data", float(omega_data)),
            ("omega_residual", float(omega_residual)),
            ("physics_projection", physics_projection),
            ("physics_subsample", int(physics_subsample)),
            ("val_physics_subsample", int(val_physics_subsample)),
            ("physics_every", int(physics_every)),
            ("log_epoch_every", int(log_epoch_every)),
            ("log_batch_every", int(log_batch_every)),
            ("basis_path", cache["basis_path"]),
            ("u_ref_path", cache["uref_path"] if os.path.exists(cache["uref_path"]) else "zeros"),
            ("seed", seed),
            ("device", device),
            ("elapsed_s", elapsed),
        ],
    )
    print(f"[Case2-PRNN] Summary: {summary_path}")


if __name__ == "__main__":
    main()
