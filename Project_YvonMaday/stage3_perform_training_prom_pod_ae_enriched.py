#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 trainer for intrusive PROM-POD-AE (autoencoder in qN space), enriched dataset.

Learns:
    qN_hat = D(E(qN))
from Stage-2 enriched ROM-consistent coefficients qN.
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
    from enrichment_dataset_utils import resolve_enrichment_dataset
except ModuleNotFoundError:
    from .enrichment_dataset_utils import resolve_enrichment_dataset
try:
    from stage3_qn_utils import load_qn_from_mu_dir
except ModuleNotFoundError:
    from .stage3_qn_utils import load_qn_from_mu_dir
try:
    from pod_ae_common import PROMPODAEAutoencoder, parse_hidden_dims
except ModuleNotFoundError:
    from .pod_ae_common import PROMPODAEAutoencoder, parse_hidden_dims
try:
    from project_layout import ensure_layout_dirs, resolve_stage1_artifact, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import ensure_layout_dirs, resolve_stage1_artifact, write_kv_txt
try:
    from enrichment_layout import ENRICHMENT_STAGE3_DIR, ensure_enrichment_dirs
except ModuleNotFoundError:
    from .enrichment_layout import ENRICHMENT_STAGE3_DIR, ensure_enrichment_dirs


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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

    y_raw = np.vstack(y_list).astype(np.float32)
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
    ensure_enrichment_dirs()

    parser = argparse.ArgumentParser(description="Train enriched PROM-POD-AE autoencoder from enrichment Stage-2 qN dataset.")
    parser.add_argument("--dataset-backend", choices=("prom", "hprom"), default="hprom")
    parser.add_argument("--dataset-ntot", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="prom_pod_ae_model_enriched.pt")
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
    args = parser.parse_args(argv)

    dataset_root, dataset_ntot, dataset_dir, dataset_meta, _ = resolve_enrichment_dataset(
        requested_ntot=args.dataset_ntot,
        expected_backend=str(args.dataset_backend).strip().lower(),
    )
    dataset_name = os.path.basename(dataset_dir.rstrip(os.sep))
    stage3_out_dir = os.path.join(ENRICHMENT_STAGE3_DIR, dataset_name)
    stage3_models_dir = os.path.join(stage3_out_dir, "models")
    os.makedirs(stage3_models_dir, exist_ok=True)

    model_name = str(args.model_name).strip()
    if len(model_name) == 0:
        raise ValueError("--model-name cannot be empty.")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    model_path = os.path.join(stage3_models_dir, model_name)
    summary_path = os.path.join(stage3_out_dir, "prom_pod_ae_training_summary_enriched.txt")

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    latent_dim = int(args.latent_dim)
    if latent_dim < 1:
        raise ValueError("--latent-dim must be >= 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[POD-AE-Enriched] device = {device}")
    print(f"[POD-AE-Enriched] dataset_dir = {dataset_dir}")
    print(f"[POD-AE-Enriched] dataset_root = {dataset_root} (ntot={dataset_ntot})")
    print(f"[POD-AE-Enriched] solve_backend = {dataset_meta.get('solve_backend')}")
    print(f"[POD-AE-Enriched] hidden_dims = {hidden_dims}")
    print(f"[POD-AE-Enriched] latent_dim = {latent_dim}")
    print(f"[POD-AE-Enriched] scaling = {args.scaling}")
    print(f"[POD-AE-Enriched] activation = {args.activation}")

    y_raw = _load_qn_samples(dataset_root)
    m, q_dim = y_raw.shape
    if int(dataset_ntot) != int(q_dim):
        raise ValueError(f"Dataset ntot={dataset_ntot} but loaded q_dim={q_dim}.")
    if latent_dim >= q_dim:
        raise ValueError(f"latent_dim={latent_dim} must be < q_dim={q_dim}.")

    idx = np.arange(m, dtype=np.int64)
    tr_idx, va_idx = train_test_split(idx, test_size=float(args.val_frac), random_state=SEED, shuffle=True)
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

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(ytr)), batch_size=int(args.batch_size), shuffle=True, drop_last=False)
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

        if ep == 1 or ep % 25 == 0:
            print(f"[Epoch {ep:4d}] train_mse={tr_loss:.6e} | val_mse={va_loss:.6e} | bad={bad}")

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
    print(f"[POD-AE-Enriched] Training done in {elapsed:.2f}s. best_val={best_val:.6e}")

    basis_path = resolve_stage1_artifact("basis.npy")
    uref_path = resolve_stage1_artifact("u_ref.npy")
    ckpt = {
        "state_dict": model.state_dict(),
        "q_dim": int(q_dim),
        "latent_dim": int(latent_dim),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "scaling": str(args.scaling),
        "activation": str(args.activation),
        "seed": int(SEED),
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dataset_ntot": int(dataset_ntot),
        "dataset_backend": dataset_meta.get("solve_backend"),
        "basis_file": basis_path,
        "u_ref_file": uref_path,
        "mapping": "qN_hat = D(E(qN))",
    }
    torch.save(ckpt, model_path)
    print(f"[POD-AE-Enriched] Saved model checkpoint: {model_path}")

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
            ("epochs_ran", ep_last),
            ("best_val_mse", best_val),
            ("elapsed_s", elapsed),
            ("seed", SEED),
            ("device", device),
        ],
    )
    print(f"[POD-AE-Enriched] Summary: {summary_path}")


if __name__ == "__main__":
    main()

