#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: TRAIN POD-ANN MAP

Train a neural map

    q_s = N(q_p)

using Stage 2 projections. Input/output scaling is embedded inside the model,
so inference uses raw q_p directly.

Model artifact:
  - POD-ANN/pod_ann_model/case1_model.pt

Diagnostics:
  - POD-ANN/stage3_train_ann_summary.txt
  - POD-ANN/stage3_validation_mse.png
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def safe_rel_error_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.linalg.norm(y_true, ord="fro")
    if denom <= 0.0:
        return None
    return float(100.0 * np.linalg.norm(y_true - y_pred, ord="fro") / denom)


def split_train_validation_indices(n_samples, validation_fraction, random_seed):
    if n_samples < 2:
        raise RuntimeError("Need at least 2 samples for train/validation split.")

    validation_fraction = float(validation_fraction)
    if validation_fraction <= 0.0:
        n_val = 1
    else:
        n_val = int(np.floor(validation_fraction * n_samples))
        n_val = max(1, n_val)
    n_val = min(n_val, n_samples - 1)

    rng = np.random.default_rng(int(random_seed))
    perm = rng.permutation(n_samples)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def resolve_u_ref(uref_mode, uref_file, stage2_metadata_file, expected_size):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    stage2_use_u_ref = None
    u_ref_vec = None
    u_ref_source = None

    if os.path.exists(stage2_metadata_file):
        meta = np.load(stage2_metadata_file, allow_pickle=True)
        if "use_u_ref" in meta.files:
            stage2_use_u_ref = bool(np.asarray(meta["use_u_ref"]).reshape(-1)[0])
        if "u_ref_used" in meta.files:
            candidate = np.asarray(meta["u_ref_used"], dtype=np.float64).reshape(-1)
            if candidate.size == expected_size:
                u_ref_vec = candidate
                u_ref_source = f"{stage2_metadata_file}:u_ref_used"

    if mode == "off":
        use_u_ref = False
    elif mode == "on":
        use_u_ref = True
    else:
        if stage2_use_u_ref is not None:
            use_u_ref = stage2_use_u_ref
        else:
            use_u_ref = (u_ref_vec is not None) or os.path.exists(uref_file)

    if use_u_ref:
        if u_ref_vec is None:
            if not os.path.exists(uref_file):
                raise FileNotFoundError(
                    "u_ref is required by current settings but file is missing: "
                    f"{uref_file}"
                )
            u_ref_vec = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
            u_ref_source = uref_file
        if u_ref_vec.size != expected_size:
            raise ValueError(f"u_ref size mismatch: got {u_ref_vec.size}, expected {expected_size}.")
    else:
        u_ref_vec = np.zeros(expected_size, dtype=np.float64)
        u_ref_source = "zeros(off)"

    return bool(use_u_ref), u_ref_vec, u_ref_source, stage2_use_u_ref


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(32, 64, 128, 256, 256)):
        super().__init__()
        dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PODANNModel(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, hidden_dims=(32, 64, 128, 256, 256)):
        super().__init__()
        in_dim = int(np.asarray(x_mean).reshape(-1).size)
        out_dim = int(np.asarray(y_mean).reshape(-1).size)
        self.scaler = Scaler(np.asarray(x_mean)[None, :], np.asarray(x_std)[None, :])
        self.core = CoreMLP(in_dim, out_dim, hidden_dims=hidden_dims)
        self.unscaler = Unscaler(np.asarray(y_mean)[None, :], np.asarray(y_std)[None, :])

    def forward(self, x_raw):
        x_norm = self.scaler(x_raw)
        y_norm = self.core(x_norm)
        y_raw = self.unscaler(y_norm)
        return y_raw


def plot_validation_mse(train_hist, val_hist, out_path):
    if len(train_hist) == 0:
        return False
    epochs = np.arange(1, len(train_hist) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, train_hist, label="train MSE", color="tab:blue", linewidth=1.6)
    if len(val_hist) == len(train_hist):
        ax.plot(epochs, val_hist, label="val MSE", color="tab:red", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    q_p_file=os.path.join(script_dir, "q_p.npy"),
    q_s_file=os.path.join(script_dir, "q_s.npy"),
    stage2_metadata_file=os.path.join(script_dir, "stage2_projection_metadata.npz"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    model_dir=os.path.join(script_dir, "pod_ann_model"),
    model_file=os.path.join(script_dir, "pod_ann_model", "case1_model.pt"),
    report_file=os.path.join(script_dir, "stage3_train_ann_summary.txt"),
    validation_plot_file=os.path.join(script_dir, "stage3_validation_mse.png"),
    validation_fraction=0.1,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-6,
    epochs=2000,
    patience=120,
    min_improve=1e-12,
    clip_grad=1.0,
    hidden_dims=(32, 64, 128, 256, 256),
    random_seed=42,
    uref_mode="auto",
    device=None,
):
    for path in (basis_file, q_p_file, q_s_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}. Run stage1/stage2 first.")

    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    q_p = np.asarray(np.load(q_p_file, allow_pickle=False), dtype=np.float64)
    q_s = np.asarray(np.load(q_s_file, allow_pickle=False), dtype=np.float64)

    if q_p.ndim != 2 or q_s.ndim != 2:
        raise ValueError("q_p and q_s must be 2D arrays.")
    if q_p.shape[1] != q_s.shape[1]:
        raise ValueError(f"q_p and q_s sample mismatch: {q_p.shape[1]} vs {q_s.shape[1]}.")

    n_primary = int(q_p.shape[0])
    n_secondary = int(q_s.shape[0])
    n_samples = int(q_p.shape[1])

    if n_secondary < 1:
        raise ValueError("q_s has zero rows. Increase total_modes in stage2.")
    if basis.shape[1] < n_primary + n_secondary:
        raise ValueError(
            "Basis has insufficient columns for (q_p, q_s): "
            f"basis columns={basis.shape[1]}, required={n_primary + n_secondary}."
        )

    use_u_ref, u_ref_vec, u_ref_source, stage2_use_u_ref = resolve_u_ref(
        uref_mode=uref_mode,
        uref_file=uref_file,
        stage2_metadata_file=stage2_metadata_file,
        expected_size=basis.shape[0],
    )

    x_raw = q_p.T.astype(np.float32)
    y_raw = q_s.T.astype(np.float32)

    train_idx, val_idx = split_train_validation_indices(
        n_samples=n_samples,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    x_train = x_raw[train_idx]
    y_train = y_raw[train_idx]
    x_val = x_raw[val_idx]
    y_val = y_raw[val_idx]

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    model = PODANNModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    print("\n====================================================")
    print("           STAGE 3: TRAIN POD-ANN")
    print("====================================================")
    print(f"[STAGE3] device={device}")
    print(f"[STAGE3] n_primary={n_primary}, n_secondary={n_secondary}, n_samples={n_samples}")
    print(f"[STAGE3] train={x_train.shape[0]}, val={x_val.shape[0]}")
    print(
        f"[STAGE3] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref_vec):.3e}"
    )

    best_val = float("inf")
    best_state = None
    epochs_without_improve = 0

    train_hist = []
    val_hist = []

    t0 = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_acc = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))

            optimizer.step()
            train_loss_acc += float(loss.detach().cpu().item()) * xb.shape[0]

        train_mse = train_loss_acc / x_train.shape[0]
        train_hist.append(train_mse)

        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(x_val_t), y_val_t).detach().cpu().item())
        val_hist.append(val_mse)

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[Epoch {epoch:4d}] train_mse={train_mse:.6e} "
                f"| val_mse={val_mse:.6e} | bad={epochs_without_improve}"
            )

        if val_mse < best_val - float(min_improve):
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(patience):
                print(f"[EarlyStop] epoch={epoch}, best_val={best_val:.6e}")
                break

    elapsed_train = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_train_pred = model(torch.from_numpy(x_train).to(device)).cpu().numpy()
        y_val_pred = model(torch.from_numpy(x_val).to(device)).cpu().numpy()

    train_rel_qs_pct = safe_rel_error_percent(y_train, y_train_pred)
    val_rel_qs_pct = safe_rel_error_percent(y_val, y_val_pred)

    val_plot_saved = plot_validation_mse(train_hist, val_hist, validation_plot_file)

    checkpoint = {
        "state_dict": model.state_dict(),
        "n_p": int(n_primary),
        "n_s": int(n_secondary),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "seed": int(random_seed),
        "basis_file": basis_file,
        "use_u_ref": bool(use_u_ref),
        "u_ref": u_ref_vec.astype(np.float32),
        "u_ref_source": u_ref_source,
        "stage2_use_u_ref": stage2_use_u_ref,
        "uref_mode": str(uref_mode),
        "train_indices": train_idx.astype(np.int64),
        "validation_indices": val_idx.astype(np.int64),
    }
    torch.save(checkpoint, model_file)

    print(f"[STAGE3] Training done in {elapsed_train:.2f}s")
    print(f"[STAGE3] Best validation MSE: {best_val:.6e}")
    if train_rel_qs_pct is not None:
        print(f"[STAGE3] Train relative error in q_s: {train_rel_qs_pct:.4f}%")
    if val_rel_qs_pct is not None:
        print(f"[STAGE3] Validation relative error in q_s: {val_rel_qs_pct:.4f}%")
    print(f"[STAGE3] Saved model checkpoint: {model_file}")

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage3_train_ann.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("q_p_file", q_p_file),
                    ("q_s_file", q_s_file),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref_vec))),
                    ("stage2_use_u_ref", stage2_use_u_ref),
                    ("learning_rate", learning_rate),
                    ("weight_decay", weight_decay),
                    ("batch_size", batch_size),
                    ("epochs", epochs),
                    ("patience", patience),
                    ("min_improve", min_improve),
                    ("clip_grad", clip_grad),
                    ("validation_fraction", validation_fraction),
                    ("hidden_dims", tuple(int(v) for v in hidden_dims)),
                    ("random_seed", random_seed),
                    ("device", device),
                ],
            ),
            (
                "dataset",
                [
                    ("n_primary", n_primary),
                    ("n_secondary", n_secondary),
                    ("n_samples_total", n_samples),
                    ("n_train", x_train.shape[0]),
                    ("n_val", x_val.shape[0]),
                ],
            ),
            (
                "metrics",
                [
                    ("best_val_mse", best_val),
                    ("train_rel_qs_pct", train_rel_qs_pct),
                    ("val_rel_qs_pct", val_rel_qs_pct),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("training", elapsed_train),
                ],
            ),
            (
                "outputs",
                [
                    ("model_pt", model_file),
                    ("validation_mse_plot_png", validation_plot_file if val_plot_saved else None),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )
    print(f"[STAGE3] Summary saved: {report_file}")


if __name__ == "__main__":
    main()

