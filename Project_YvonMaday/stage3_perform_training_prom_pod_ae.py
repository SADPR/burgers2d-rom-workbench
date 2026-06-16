#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 trainer for intrusive PROM-POD-AE (autoencoder in qN space).

Learns:
    qN_hat = D(E(qN))

from Stage-2 ROM-consistent coefficients qN.
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
    from pod_ae_common import PROMPODAEAutoencoder, parse_hidden_dims
except ModuleNotFoundError:
    from .pod_ae_common import PROMPODAEAutoencoder, parse_hidden_dims
try:
    from project_layout import (
        STAGE3_DIR,
        ensure_layout_dirs,
        stage3_model_path,
        resolve_stage1_artifact,
        write_kv_txt,
    )
except ModuleNotFoundError:
    from .project_layout import (
        STAGE3_DIR,
        ensure_layout_dirs,
        stage3_model_path,
        resolve_stage1_artifact,
        write_kv_txt,
    )


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEED = 42


def _localize_project_path(path_like):
    """Map metadata paths copied from another machine to this checkout."""
    if path_like is None:
        return None
    path = os.path.abspath(os.path.expanduser(str(path_like)))
    if os.path.exists(path):
        return path
    marker = f"{os.sep}Project_YvonMaday{os.sep}"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        candidate = os.path.join(THIS_DIR, suffix)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


def _load_qn_samples(dataset_root: str):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")

    subdirs = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )
    if len(subdirs) == 0:
        raise RuntimeError(f"No per_mu subdirectories found in: {dataset_root}")

    y_list = []
    n_tot_ref = None
    t_ref = None

    for sd in subdirs:
        mu_dir = os.path.join(dataset_root, sd)
        qn = np.asarray(load_qn_from_mu_dir(mu_dir), dtype=np.float64)  # (n_tot, T)
        t = np.asarray(np.load(os.path.join(mu_dir, "t.npy"), allow_pickle=False), dtype=np.float64).reshape(-1)

        if qn.ndim != 2:
            raise ValueError(f"{sd}: qN must be 2D (n_tot,T), got {qn.shape}")
        n_tot, nt = qn.shape
        if nt != t.size:
            raise ValueError(f"{sd}: t size mismatch: {t.size} vs qN columns {nt}")
        if n_tot_ref is None:
            n_tot_ref = n_tot
            t_ref = nt
        else:
            if n_tot != n_tot_ref:
                raise ValueError(f"{sd}: n_tot mismatch ({n_tot} vs {n_tot_ref})")
            if nt != t_ref:
                raise ValueError(f"{sd}: time size mismatch ({nt} vs {t_ref})")

        y_list.append(qn.T)  # (T, n_tot)

    y_raw = np.vstack(y_list).astype(np.float32)  # (M, n_tot)
    return y_raw


def _build_q_stats(y_train: np.ndarray, scaling: str):
    sc = str(scaling).strip().lower()
    if sc == "minmax_-1_1":
        return {
            "min": y_train.min(axis=0, keepdims=True).astype(np.float32),
            "max": y_train.max(axis=0, keepdims=True).astype(np.float32),
        }
    if sc == "zscore":
        return {
            "mean": y_train.mean(axis=0, keepdims=True).astype(np.float32),
            "std": y_train.std(axis=0, keepdims=True).astype(np.float32),
        }
    raise ValueError(f"Unsupported scaling: {scaling}")


def main(argv=None):
    ensure_layout_dirs()

    parser = argparse.ArgumentParser(description="Train PROM-POD-AE autoencoder from Stage-2 qN dataset.")
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="prom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="prom_pod_ae_model.pt")
    parser.add_argument("--stage3-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--summary-name", type=str, default="prom_pod_ae_training_summary.txt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--hidden-dims", type=str, default="192,96,48")
    parser.add_argument("--activation", choices=("tanh", "silu", "elu", "gelu"), default="tanh")
    parser.add_argument("--scaling", choices=("minmax_-1_1", "zscore"), default="minmax_-1_1")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min-improve", type=float, default=1e-12)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=50)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    args = parser.parse_args(argv)

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = str(args.model_name).strip()
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"

    stage3_dir = os.path.abspath(os.path.expanduser(args.stage3_dir)) if args.stage3_dir else STAGE3_DIR
    models_dir = os.path.abspath(os.path.expanduser(args.models_dir)) if args.models_dir else os.path.join(stage3_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)

    if args.stage3_dir or args.models_dir:
        model_path = os.path.join(models_dir, model_name)
    else:
        model_path = stage3_model_path(model_name)
    summary_name = str(args.summary_name).strip() or "prom_pod_ae_training_summary.txt"
    summary_path = os.path.join(stage3_dir, summary_name)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    latent_dim = int(args.latent_dim)
    if latent_dim < 1:
        raise ValueError("--latent-dim must be >= 1.")

    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=args.dataset_ntot,
        expected_backend=str(args.dataset_backend).strip().lower(),
        requested_dataset_dir=args.dataset_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[POD-AE] device = {device}")
    print(f"[POD-AE] dataset_dir = {dataset_dir}")
    print(f"[POD-AE] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[POD-AE] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[POD-AE] hidden_dims = {hidden_dims}")
    print(f"[POD-AE] latent_dim = {latent_dim}")
    print(f"[POD-AE] scaling = {args.scaling}")
    print(f"[POD-AE] activation = {args.activation}")
    print(f"[POD-AE] batch_size = {args.batch_size}")
    print(f"[POD-AE] lr = {args.lr}")
    print(f"[POD-AE] weight_decay = {args.weight_decay}")
    print(
        "[POD-AE] lr_scheduler = ReduceLROnPlateau("
        f"factor={args.lr_scheduler_factor}, patience={args.lr_scheduler_patience}, "
        f"min_lr={args.lr_scheduler_min_lr})"
    )

    y_raw = _load_qn_samples(dataset_root)
    m, q_dim = y_raw.shape
    if int(dataset_ntot) != int(q_dim):
        raise ValueError(f"Dataset ntot={dataset_ntot} but loaded q_dim={q_dim}.")
    if latent_dim >= q_dim:
        raise ValueError(f"latent_dim={latent_dim} must be < q_dim={q_dim}.")

    print(f"[POD-AE] Loaded: M={m}, q_dim={q_dim}")

    idx = np.arange(m, dtype=np.int64)
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(args.val_frac),
        random_state=seed,
        shuffle=True,
    )
    ytr = y_raw[tr_idx]
    yva = y_raw[va_idx]

    q_stats = _build_q_stats(ytr, scaling=args.scaling)
    model = PROMPODAEAutoencoder(
        q_dim=q_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        scaling=args.scaling,
        activation=args.activation,
        q_stats=q_stats,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=float(args.lr_scheduler_factor),
        patience=int(args.lr_scheduler_patience),
        min_lr=float(args.lr_scheduler_min_lr),
    )
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(
        TensorDataset(torch.from_numpy(ytr)),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    yva_t = torch.from_numpy(yva).to(device)

    best_val = float("inf")
    best_state = None
    bad = 0
    ep_last = 0

    t0 = time.time()
    for ep in range(1, int(args.epochs) + 1):
        ep_last = ep
        model.train()
        tr_loss = 0.0
        for (yb,) in dl_tr:
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(yb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
            opt.step()
            tr_loss += float(loss.detach().cpu().item()) * yb.shape[0]
        tr_loss /= max(1, ytr.shape[0])

        model.eval()
        with torch.no_grad():
            va_loss = float(loss_fn(model(yva_t), yva_t).detach().cpu().item())
        scheduler.step(va_loss)
        current_lr = float(opt.param_groups[0]["lr"])

        if ep == 1 or ep % 25 == 0:
            print(f"[Epoch {ep:4d}] train_mse={tr_loss:.6e} | val_mse={va_loss:.6e} | lr={current_lr:.3e} | bad={bad}")

        if va_loss < best_val - float(args.min_improve):
            best_val = va_loss
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
    print(f"[POD-AE] Training done in {elapsed:.2f}s. best_val={best_val:.6e}")

    model.eval()
    with torch.no_grad():
        ytr_pred = model(torch.from_numpy(ytr).to(device)).detach().cpu().numpy()
        yva_pred = model(yva_t).detach().cpu().numpy()
    train_rel_frob = 100.0 * float(np.linalg.norm(ytr_pred - ytr) / max(np.linalg.norm(ytr), 1e-300))
    val_rel_frob = 100.0 * float(np.linalg.norm(yva_pred - yva) / max(np.linalg.norm(yva), 1e-300))
    trainable_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[POD-AE] train_rel_frob_percent = {train_rel_frob:.6f}%")
    print(f"[POD-AE] val_rel_frob_percent = {val_rel_frob:.6f}%")
    print(f"[POD-AE] trainable_parameters = {trainable_parameters}")

    basis_path = _localize_project_path(dataset_meta.get("basis_path")) or resolve_stage1_artifact("basis.npy")
    uref_path = _localize_project_path(dataset_meta.get("u_ref_path") or dataset_meta.get("uref_path")) or resolve_stage1_artifact("u_ref.npy")
    ckpt = {
        "state_dict": model.state_dict(),
        "q_dim": int(q_dim),
        "latent_dim": int(latent_dim),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "scaling": str(args.scaling),
        "activation": str(args.activation),
        "seed": seed,
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "basis_file": basis_path,
        "u_ref_file": uref_path,
        "mapping": "qN_hat = D(E(qN))",
    }
    torch.save(ckpt, model_path)
    print(f"[POD-AE] Saved model checkpoint: {model_path}")

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
            ("q_dim", q_dim),
            ("latent_dim", latent_dim),
            ("hidden_dims", hidden_dims),
            ("scaling", args.scaling),
            ("activation", args.activation),
            ("batch_size", int(args.batch_size)),
            ("lr", float(args.lr)),
            ("weight_decay", float(args.weight_decay)),
            ("lr_scheduler", "ReduceLROnPlateau"),
            ("lr_scheduler_factor", float(args.lr_scheduler_factor)),
            ("lr_scheduler_patience", int(args.lr_scheduler_patience)),
            ("lr_scheduler_min_lr", float(args.lr_scheduler_min_lr)),
            ("trainable_parameters", trainable_parameters),
            ("epochs_ran", ep_last),
            ("best_val_mse", best_val),
            ("train_rel_frob_percent", train_rel_frob),
            ("val_rel_frob_percent", val_rel_frob),
            ("elapsed_s", elapsed),
            ("seed", seed),
            ("device", device),
        ],
    )
    print(f"[POD-AE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
