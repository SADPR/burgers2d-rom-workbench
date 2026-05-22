#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-3 trainer for non-intrusive POD-DL data-driven model.

Learns:
  qN_hat(mu,t) = D(phi(mu,t))
with compatibility/regularization term:
  z_enc(qN) ~= phi(mu,t)
"""

import argparse
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset
try:
    from stage3_qn_utils import load_qn_from_mu_dir
except ModuleNotFoundError:
    from .stage3_qn_utils import load_qn_from_mu_dir
try:
    from pod_dl_data_driven_common import (
        PODDLDataDrivenModel,
        parse_hidden_dims,
    )
except ModuleNotFoundError:
    from .pod_dl_data_driven_common import (
        PODDLDataDrivenModel,
        parse_hidden_dims,
    )
try:
    from project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import STAGE3_DIR, ensure_layout_dirs, stage3_model_path, write_kv_txt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def _load_dataset(dataset_root: str):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")

    subdirs = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )
    if len(subdirs) == 0:
        raise RuntimeError(f"No per_mu subdirectories found in: {dataset_root}")

    x_list = []
    y_list = []
    q_dim_ref = None

    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)
        mu = np.asarray(np.load(os.path.join(mu_dir, "mu.npy"), allow_pickle=False), dtype=np.float64).reshape(-1)
        if mu.size != 2:
            raise ValueError(f"{sd}: mu.npy must have shape (2,), got {mu.shape}")

        t = np.asarray(np.load(os.path.join(mu_dir, "t.npy"), allow_pickle=False), dtype=np.float64).reshape(-1)
        qn = np.asarray(load_qn_from_mu_dir(mu_dir), dtype=np.float64)
        if qn.ndim != 2:
            raise ValueError(f"{sd}: qN must be 2D (n_tot,T), got {qn.shape}")

        q_dim, nt = qn.shape
        if nt != t.size:
            raise ValueError(f"{sd}: t size mismatch: {t.size} vs qN columns {nt}")
        if q_dim_ref is None:
            q_dim_ref = q_dim
        elif q_dim != q_dim_ref:
            raise ValueError(f"{sd}: q_dim mismatch ({q_dim} vs {q_dim_ref})")

        xi = np.column_stack([
            np.full((nt,), mu[0], dtype=np.float64),
            np.full((nt,), mu[1], dtype=np.float64),
            t,
        ])  # (T,3)
        yi = qn.T  # (T,q_dim)

        x_list.append(xi)
        y_list.append(yi)

    x_raw = np.vstack(x_list).astype(np.float32)
    y_raw = np.vstack(y_list).astype(np.float32)
    return x_raw, y_raw


def _build_stats(x_train: np.ndarray, y_train: np.ndarray, x_scaling: str, q_scaling: str):
    x_scaling = str(x_scaling).strip().lower()
    q_scaling = str(q_scaling).strip().lower()

    if x_scaling == "zscore":
        x_stats = {
            "mean": x_train.mean(axis=0, keepdims=True).astype(np.float32),
            "std": x_train.std(axis=0, keepdims=True).astype(np.float32),
        }
    elif x_scaling == "minmax_-1_1":
        x_stats = {
            "min": x_train.min(axis=0, keepdims=True).astype(np.float32),
            "max": x_train.max(axis=0, keepdims=True).astype(np.float32),
        }
    else:
        raise ValueError(f"Unsupported x_scaling: {x_scaling}")

    if q_scaling == "zscore":
        q_stats = {
            "mean": y_train.mean(axis=0, keepdims=True).astype(np.float32),
            "std": y_train.std(axis=0, keepdims=True).astype(np.float32),
        }
    elif q_scaling == "minmax_-1_1":
        q_stats = {
            "min": y_train.min(axis=0, keepdims=True).astype(np.float32),
            "max": y_train.max(axis=0, keepdims=True).astype(np.float32),
        }
    else:
        raise ValueError(f"Unsupported q_scaling: {q_scaling}")

    return x_stats, q_stats


def main(argv=None):
    ensure_layout_dirs()

    parser = argparse.ArgumentParser(description="Train non-intrusive POD-DL model from Stage-2 qN dataset.")
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="pod_dl_data_driven_model.pt")
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--encoder-hidden-dims", type=str, default="256,128")
    parser.add_argument("--decoder-hidden-dims", type=str, default="128,256")
    parser.add_argument("--dynamics-hidden-dims", type=str, default="64,128,128")
    parser.add_argument("--activation", choices=("tanh", "silu", "elu", "gelu"), default="elu")
    parser.add_argument("--x-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")
    parser.add_argument("--q-scaling", choices=("zscore", "minmax_-1_1"), default="zscore")
    parser.add_argument("--omega-data", type=float, default=1.0)
    parser.add_argument("--omega-latent", type=float, default=0.1)
    parser.add_argument(
        "--omega-recon",
        type=float,
        default=0.0,
        help="Optional AE self-reconstruction term ||D(E(q))-q||^2 (paper-like default is 0).",
    )
    parser.add_argument("--detach-encoder-target", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min-improve", type=float, default=1e-12)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--pretrain-epochs", type=int, default=0, help="AE-only pretraining epochs before joint training.")
    args = parser.parse_args(argv)

    model_name = str(args.model_name).strip()
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = stage3_model_path(model_name)
    summary_path = os.path.join(STAGE3_DIR, "pod_dl_data_driven_training_summary.txt")

    latent_dim = int(args.latent_dim)
    if latent_dim < 1:
        raise ValueError("--latent-dim must be >= 1.")

    omega_data = float(args.omega_data)
    omega_latent = float(args.omega_latent)
    omega_recon = float(args.omega_recon)
    if omega_data < 0.0 or omega_latent < 0.0 or omega_recon < 0.0:
        raise ValueError("All omega weights must be non-negative.")

    encoder_hidden_dims = parse_hidden_dims(args.encoder_hidden_dims)
    decoder_hidden_dims = parse_hidden_dims(args.decoder_hidden_dims)
    dynamics_hidden_dims = parse_hidden_dims(args.dynamics_hidden_dims)

    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=args.dataset_ntot,
        expected_backend=str(args.dataset_backend).strip().lower(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[POD-DL] device = {device}")
    print(f"[POD-DL] dataset_dir = {dataset_dir}")
    print(f"[POD-DL] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[POD-DL] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[POD-DL] latent_dim = {latent_dim}")
    print(f"[POD-DL] encoder_hidden_dims = {encoder_hidden_dims}")
    print(f"[POD-DL] decoder_hidden_dims = {decoder_hidden_dims}")
    print(f"[POD-DL] dynamics_hidden_dims = {dynamics_hidden_dims}")

    x_raw, y_raw = _load_dataset(dataset_root)
    m, in_dim = x_raw.shape
    _, q_dim = y_raw.shape
    if in_dim != 3:
        raise ValueError(f"Expected input dim=3 (mu1,mu2,t), got {in_dim}")
    if int(dataset_ntot) != int(q_dim):
        raise ValueError(f"Dataset ntot={dataset_ntot} but loaded q_dim={q_dim}.")
    if latent_dim >= q_dim:
        raise ValueError(f"latent_dim={latent_dim} must be < q_dim={q_dim}.")
    print(f"[POD-DL] Loaded: M={m}, in_dim={in_dim}, q_dim={q_dim}")

    idx = np.arange(m, dtype=np.int64)
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(args.val_frac),
        random_state=SEED,
        shuffle=True,
    )
    xtr, ytr = x_raw[tr_idx], y_raw[tr_idx]
    xva, yva = x_raw[va_idx], y_raw[va_idx]

    x_stats, q_stats = _build_stats(xtr, ytr, x_scaling=args.x_scaling, q_scaling=args.q_scaling)
    model = PODDLDataDrivenModel(
        q_dim=q_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        dynamics_hidden_dims=dynamics_hidden_dims,
        activation=args.activation,
        x_scaling=args.x_scaling,
        q_scaling=args.q_scaling,
        x_stats=x_stats,
        q_stats=q_stats,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    mse = nn.MSELoss()

    dl_tr = DataLoader(
        TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    xva_t = torch.from_numpy(xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    def _eval_loss(x_t, y_t):
        with torch.no_grad():
            out = model(x_t, y_t, return_terms=True)
            q_pred = out["q_pred"]
            z_pred = out["z_pred"]
            z_enc = out["z_enc"]
            q_rec = out["q_rec"]

            l_data = mse(q_pred, y_t)
            l_latent = mse(z_pred, z_enc)
            l_recon = mse(q_rec, y_t)
            l_total = omega_data * l_data + omega_latent * l_latent + omega_recon * l_recon
        return float(l_total.item()), float(l_data.item()), float(l_latent.item()), float(l_recon.item())

    best_val = float("inf")
    best_state = None
    bad = 0
    ep_last = 0

    t0 = time.time()

    pretrain_epochs = max(0, int(args.pretrain_epochs))
    if pretrain_epochs > 0:
        print(f"[POD-DL] Starting AE pretraining for {pretrain_epochs} epochs")
        for ep in range(1, pretrain_epochs + 1):
            model.train()
            tr_recon = 0.0
            ns = 0
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                del xb  # AE pretraining uses only q.
                opt.zero_grad(set_to_none=True)
                q_rec = model.reconstruct_q(yb)
                loss = mse(q_rec, yb)
                loss.backward()
                if args.clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                opt.step()
                bs = int(yb.shape[0])
                tr_recon += float(loss.detach().cpu().item()) * bs
                ns += bs
            tr_recon /= max(1, ns)
            if ep == 1 or ep % 25 == 0:
                print(f"[Pretrain {ep:4d}] recon_mse={tr_recon:.6e}")

    for ep in range(1, int(args.epochs) + 1):
        ep_last = ep
        model.train()
        tr_total = 0.0
        tr_data = 0.0
        tr_latent = 0.0
        tr_recon = 0.0
        ns = 0

        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(xb, yb, return_terms=True)
            q_pred = out["q_pred"]
            z_pred = out["z_pred"]
            z_enc = out["z_enc"]
            q_rec = out["q_rec"]

            if bool(args.detach_encoder_target):
                z_enc = z_enc.detach()

            l_data = mse(q_pred, yb)
            l_latent = mse(z_pred, z_enc)
            l_recon = mse(q_rec, yb)
            loss = omega_data * l_data + omega_latent * l_latent + omega_recon * l_recon
            loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
            opt.step()

            bs = int(xb.shape[0])
            tr_total += float(loss.detach().cpu().item()) * bs
            tr_data += float(l_data.detach().cpu().item()) * bs
            tr_latent += float(l_latent.detach().cpu().item()) * bs
            tr_recon += float(l_recon.detach().cpu().item()) * bs
            ns += bs

        tr_total /= max(1, ns)
        tr_data /= max(1, ns)
        tr_latent /= max(1, ns)
        tr_recon /= max(1, ns)

        va_total, va_data, va_latent, va_recon = _eval_loss(xva_t, yva_t)
        if ep == 1 or ep % 25 == 0:
            print(
                f"[Epoch {ep:4d}] train_total={tr_total:.6e} (data={tr_data:.6e}, latent={tr_latent:.6e}, recon={tr_recon:.6e}) "
                f"| val_total={va_total:.6e} (data={va_data:.6e}, latent={va_latent:.6e}, recon={va_recon:.6e}) | bad={bad}"
            )

        if va_total < best_val - float(args.min_improve):
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"[EarlyStop] epoch={ep} best_val={best_val:.6e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"[POD-DL] Training done in {elapsed:.2f}s. best_val_total={best_val:.6e}")

    ckpt = {
        "state_dict": model.state_dict(),
        "q_dim": int(q_dim),
        "in_dim": int(in_dim),
        "latent_dim": int(latent_dim),
        "encoder_hidden_dims": tuple(int(v) for v in encoder_hidden_dims),
        "decoder_hidden_dims": tuple(int(v) for v in decoder_hidden_dims),
        "dynamics_hidden_dims": tuple(int(v) for v in dynamics_hidden_dims),
        "activation": str(args.activation),
        "x_scaling": str(args.x_scaling),
        "q_scaling": str(args.q_scaling),
        "omega_data": omega_data,
        "omega_latent": omega_latent,
        "omega_recon": omega_recon,
        "detach_encoder_target": bool(args.detach_encoder_target),
        "seed": int(SEED),
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "formula": "q_hat = D(phi(mu,t)); loss = w_data||q_hat-q||^2 + w_latent||E(q)-phi(mu,t)||^2 + w_recon||D(E(q))-q||^2",
    }
    torch.save(ckpt, model_path)
    print(f"[POD-DL] Saved model checkpoint: {model_path}")

    write_kv_txt(
        summary_path,
        [
            ("model_name", model_name),
            ("model_path", model_path),
            ("dataset_dir", dataset_dir),
            ("dataset_root", dataset_root),
            ("dataset_ntot", dataset_ntot),
            ("dataset_backend", dataset_meta.get("solve_backend")),
            ("samples_M", m),
            ("in_dim", in_dim),
            ("q_dim", q_dim),
            ("latent_dim", latent_dim),
            ("encoder_hidden_dims", encoder_hidden_dims),
            ("decoder_hidden_dims", decoder_hidden_dims),
            ("dynamics_hidden_dims", dynamics_hidden_dims),
            ("activation", args.activation),
            ("x_scaling", args.x_scaling),
            ("q_scaling", args.q_scaling),
            ("omega_data", omega_data),
            ("omega_latent", omega_latent),
            ("omega_recon", omega_recon),
            ("detach_encoder_target", bool(args.detach_encoder_target)),
            ("pretrain_epochs", pretrain_epochs),
            ("epochs_ran", ep_last),
            ("best_val_total", best_val),
            ("elapsed_s", elapsed),
            ("seed", SEED),
            ("device", device),
        ],
    )
    print(f"[POD-DL] Summary: {summary_path}")


if __name__ == "__main__":
    main()
