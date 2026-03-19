#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: TRAIN POD-DL AUTOENCODER

Train an autoencoder in POD coefficient space

    q -> z -> q

using Stage 2 projections. Input scaling is embedded inside the model,
so inference uses raw q directly.

Model artifact:
  - POD-DL/pod_dl_model/pod_dl_autoencoder.pt

Diagnostics:
  - POD-DL/stage3_train_autoencoder_summary.txt
  - POD-DL/stage3_validation_mse.png
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
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > eps, half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, x):
        return (x - self.center) / self.half_range


class Unscaler(nn.Module):
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        center = 0.5 * (x_max + x_min)
        half_range = 0.5 * (x_max - x_min)
        half_range = np.where(half_range > eps, half_range, 1.0).astype(np.float32)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("half_range", torch.tensor(half_range, dtype=torch.float32))

    def forward(self, y):
        return y * self.half_range + self.center


def build_mlp(in_dim, hidden_dims, out_dim):
    dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PODDLAutoencoder(nn.Module):
    def __init__(self, q_min, q_max, latent_dim, hidden_dims=(192, 96, 48)):
        super().__init__()
        q_dim = int(np.asarray(q_min).reshape(-1).size)
        z_dim = int(latent_dim)

        self.scaler = Scaler(np.asarray(q_min)[None, :], np.asarray(q_max)[None, :])
        self.encoder = build_mlp(q_dim, hidden_dims, z_dim)
        self.decoder = build_mlp(z_dim, tuple(reversed(hidden_dims)), q_dim)
        self.unscaler = Unscaler(np.asarray(q_min)[None, :], np.asarray(q_max)[None, :])

    def forward(self, q_raw):
        q_norm = self.scaler(q_raw)
        z = self.encoder(q_norm)
        q_norm_hat = self.decoder(z)
        q_raw_hat = self.unscaler(q_norm_hat)
        return q_raw_hat

    def encode(self, q_raw):
        q_norm = self.scaler(q_raw)
        return self.encoder(q_norm)

    def decode_from_latent(self, z):
        q_norm_hat = self.decoder(z)
        return self.unscaler(q_norm_hat)


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
    q_file=os.path.join(script_dir, "q.npy"),
    q_test_file=os.path.join(script_dir, "q_test.npy"),
    stage2_metadata_file=os.path.join(script_dir, "stage2_projection_metadata.npz"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    model_dir=os.path.join(script_dir, "pod_dl_model"),
    model_file=os.path.join(script_dir, "pod_dl_model", "pod_dl_autoencoder.pt"),
    report_file=os.path.join(script_dir, "stage3_train_autoencoder_summary.txt"),
    validation_plot_file=os.path.join(script_dir, "stage3_validation_mse.png"),
    validation_fraction=0.1,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-5,
    epochs=5000,
    patience=120,
    min_improve=1e-12,
    clip_grad=1.0,
    latent_dim=10,
    latent_sweep=(24, 32, 40, 48),
    use_latent_sweep=False,
    hidden_dims=(192, 96, 48),
    use_reduce_lr_on_plateau=True,
    scheduler_factor=0.5,
    scheduler_patience=20,
    scheduler_threshold=1e-4,
    scheduler_min_lr=1e-6,
    scheduler_cooldown=5,
    random_seed=42,
    uref_mode="auto",
    device=None,
):
    for path in (basis_file, q_file):
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
    q = np.asarray(np.load(q_file, allow_pickle=False), dtype=np.float64)

    if q.ndim != 2:
        raise ValueError("q must be a 2D array.")

    q_dim = int(q.shape[0])
    n_samples = int(q.shape[1])

    if q_dim < 1:
        raise ValueError("q has zero rows. Check stage2 outputs.")

    if basis.shape[1] < q_dim:
        raise ValueError(
            "Basis has fewer columns than q dimension: "
            f"basis columns={basis.shape[1]}, q_dim={q_dim}."
        )

    use_u_ref, u_ref_vec, u_ref_source, stage2_use_u_ref = resolve_u_ref(
        uref_mode=uref_mode,
        uref_file=uref_file,
        stage2_metadata_file=stage2_metadata_file,
        expected_size=basis.shape[0],
    )

    x_raw = q.T.astype(np.float32)

    train_idx, val_idx = split_train_validation_indices(
        n_samples=n_samples,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    x_train = x_raw[train_idx]
    x_val = x_raw[val_idx]

    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)

    x_val_t = torch.from_numpy(x_val).to(device)

    if use_latent_sweep:
        latent_candidates = sorted({int(v) for v in tuple(latent_sweep) if int(v) > 0})
        if int(latent_dim) > 0:
            latent_candidates = sorted(set(latent_candidates + [int(latent_dim)]))
    else:
        latent_candidates = [int(latent_dim)]

    if len(latent_candidates) == 0:
        raise ValueError("No valid latent dimensions found. Check latent_dim/latent_sweep.")

    print("\n====================================================")
    print("      STAGE 3: TRAIN POD-DL AUTOENCODER")
    print("====================================================")
    print(f"[STAGE3] device={device}")
    print(f"[STAGE3] q_dim={q_dim}, n_samples={n_samples}")
    print(
        f"[STAGE3] latent candidates={latent_candidates} "
        f"(sweep={'on' if use_latent_sweep else 'off'})"
    )
    print(f"[STAGE3] train={x_train.shape[0]}, val={x_val.shape[0]}")
    print(
        f"[STAGE3] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref_vec):.3e}"
    )

    best_val = float("inf")
    best_model_state = None
    best_latent_dim = None
    best_train_hist = []
    best_val_hist = []
    best_train_rel_q_pct = None
    best_val_rel_q_pct = None
    best_final_lr = None
    candidate_results = []

    t0 = time.time()
    for candidate_idx, z_dim in enumerate(latent_candidates):
        candidate_seed = int(random_seed) + candidate_idx
        torch.manual_seed(candidate_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(candidate_seed)

        model = PODDLAutoencoder(
            q_min=x_min,
            q_max=x_max,
            latent_dim=z_dim,
            hidden_dims=hidden_dims,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = None
        if use_reduce_lr_on_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_factor),
                patience=int(scheduler_patience),
                threshold=float(scheduler_threshold),
                threshold_mode="rel",
                cooldown=int(scheduler_cooldown),
                min_lr=float(scheduler_min_lr),
            )
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train)),
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
        )

        print(f"[STAGE3] Training candidate latent_dim={z_dim} ...")
        candidate_best_val = float("inf")
        candidate_best_state = None
        epochs_without_improve = 0
        train_hist = []
        val_hist = []

        for epoch in range(1, int(epochs) + 1):
            model.train()
            train_loss_acc = 0.0

            for (xb,) in train_loader:
                xb = xb.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, xb)
                loss.backward()

                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))

                optimizer.step()
                train_loss_acc += float(loss.detach().cpu().item()) * xb.shape[0]

            train_mse = train_loss_acc / x_train.shape[0]
            train_hist.append(train_mse)

            model.eval()
            with torch.no_grad():
                val_mse = float(loss_fn(model(x_val_t), x_val_t).detach().cpu().item())
            val_hist.append(val_mse)
            if scheduler is not None:
                scheduler.step(val_mse)
            current_lr = float(optimizer.param_groups[0]["lr"])

            if epoch == 1 or epoch % 25 == 0:
                print(
                    f"[z={z_dim:3d} | Epoch {epoch:4d}] "
                    f"train_mse={train_mse:.6e} | val_mse={val_mse:.6e} "
                    f"| lr={current_lr:.3e} | bad={epochs_without_improve}"
                )

            if val_mse < candidate_best_val - float(min_improve):
                candidate_best_val = val_mse
                candidate_best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= int(patience):
                    print(
                        f"[EarlyStop z={z_dim}] epoch={epoch}, "
                        f"best_val={candidate_best_val:.6e}"
                    )
                    break

        if candidate_best_state is None:
            candidate_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(candidate_best_state)
        model.eval()
        with torch.no_grad():
            x_train_pred = model(torch.from_numpy(x_train).to(device)).cpu().numpy()
            x_val_pred = model(torch.from_numpy(x_val).to(device)).cpu().numpy()

        train_rel_q_pct = safe_rel_error_percent(x_train, x_train_pred)
        val_rel_q_pct = safe_rel_error_percent(x_val, x_val_pred)

        candidate_results.append(
            {
                "latent_dim": int(z_dim),
                "best_val_mse": float(candidate_best_val),
                "train_rel_q_pct": train_rel_q_pct,
                "val_rel_q_pct": val_rel_q_pct,
                "epochs_ran": len(train_hist),
                "final_lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[STAGE3] Candidate z={z_dim}: best_val_mse={candidate_best_val:.6e}, "
            f"val_rel_q={val_rel_q_pct:.4f}%"
            if val_rel_q_pct is not None
            else f"[STAGE3] Candidate z={z_dim}: best_val_mse={candidate_best_val:.6e}"
        )

        if candidate_best_val < best_val:
            best_val = float(candidate_best_val)
            best_model_state = candidate_best_state
            best_latent_dim = int(z_dim)
            best_train_hist = list(train_hist)
            best_val_hist = list(val_hist)
            best_train_rel_q_pct = train_rel_q_pct
            best_val_rel_q_pct = val_rel_q_pct
            best_final_lr = float(optimizer.param_groups[0]["lr"])

    elapsed_train = time.time() - t0

    if best_model_state is None or best_latent_dim is None:
        raise RuntimeError("No trained model candidate was produced.")

    model = PODDLAutoencoder(
        q_min=x_min,
        q_max=x_max,
        latent_dim=best_latent_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    q_test_rel_q_pct = None
    q_test_shape = None
    if os.path.exists(q_test_file):
        q_test = np.asarray(np.load(q_test_file, allow_pickle=False), dtype=np.float64)
        if q_test.ndim == 2 and q_test.shape[0] == q_dim:
            q_test_raw = q_test.T.astype(np.float32)
            q_test_shape = q_test.shape
            with torch.no_grad():
                q_test_pred = model(torch.from_numpy(q_test_raw).to(device)).cpu().numpy()
            q_test_rel_q_pct = safe_rel_error_percent(q_test_raw, q_test_pred)

    val_plot_saved = plot_validation_mse(best_train_hist, best_val_hist, validation_plot_file)

    candidate_results_str = "; ".join(
        (
            f"z={entry['latent_dim']}: val_mse={entry['best_val_mse']:.6e}, "
            f"val_rel_q_pct={entry['val_rel_q_pct']:.4f}, "
            f"final_lr={entry['final_lr']:.3e}"
            if entry["val_rel_q_pct"] is not None
            else (
                f"z={entry['latent_dim']}: val_mse={entry['best_val_mse']:.6e}, "
                f"val_rel_q_pct=N/A, final_lr={entry['final_lr']:.3e}"
            )
        )
        for entry in candidate_results
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "q_dim": int(q_dim),
        "latent_dim": int(best_latent_dim),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "scaling": "minmax_-1_1",
        "activation": "tanh",
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
    print(f"[STAGE3] Selected latent_dim: {best_latent_dim}")
    print(f"[STAGE3] Best validation MSE: {best_val:.6e}")
    if best_final_lr is not None:
        print(f"[STAGE3] Final learning rate (selected model): {best_final_lr:.3e}")
    if best_train_rel_q_pct is not None:
        print(f"[STAGE3] Train relative error in q: {best_train_rel_q_pct:.4f}%")
    if best_val_rel_q_pct is not None:
        print(f"[STAGE3] Validation relative error in q: {best_val_rel_q_pct:.4f}%")
    if q_test_rel_q_pct is not None:
        print(f"[STAGE3] Test relative error in q: {q_test_rel_q_pct:.4f}%")
    print(f"[STAGE3] Saved model checkpoint: {model_file}")

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage3_train_autoencoder.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("q_file", q_file),
                    ("q_test_file", q_test_file if os.path.exists(q_test_file) else None),
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
                    ("activation", "tanh"),
                    ("scaling", "minmax_-1_1"),
                    ("latent_dim_requested", int(latent_dim)),
                    ("use_latent_sweep", bool(use_latent_sweep)),
                    ("latent_sweep", tuple(int(v) for v in latent_candidates)),
                    ("selected_latent_dim", int(best_latent_dim)),
                    ("use_reduce_lr_on_plateau", bool(use_reduce_lr_on_plateau)),
                    ("scheduler_factor", scheduler_factor if use_reduce_lr_on_plateau else None),
                    ("scheduler_patience", scheduler_patience if use_reduce_lr_on_plateau else None),
                    ("scheduler_threshold", scheduler_threshold if use_reduce_lr_on_plateau else None),
                    ("scheduler_min_lr", scheduler_min_lr if use_reduce_lr_on_plateau else None),
                    ("scheduler_cooldown", scheduler_cooldown if use_reduce_lr_on_plateau else None),
                    ("random_seed", random_seed),
                    ("device", device),
                ],
            ),
            (
                "dataset",
                [
                    ("q_dim", q_dim),
                    ("n_samples_total", n_samples),
                    ("n_train", x_train.shape[0]),
                    ("n_val", x_val.shape[0]),
                    ("q_test_shape", q_test_shape),
                ],
            ),
            (
                "metrics",
                [
                    ("best_val_mse", best_val),
                    ("train_rel_q_pct", best_train_rel_q_pct),
                    ("val_rel_q_pct", best_val_rel_q_pct),
                    ("test_rel_q_pct", q_test_rel_q_pct),
                    ("selected_final_lr", best_final_lr),
                    ("candidate_results", candidate_results_str),
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
