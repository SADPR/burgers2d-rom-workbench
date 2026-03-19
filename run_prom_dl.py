#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global POD-DL PROM for the 2D inviscid Burgers problem and compare to HDM.

Online manifold:
    u(z) = u_ref + V N(z)
    du/dz = V dN(z)/dz
"""

import os
import time
from datetime import datetime
from contextlib import contextmanager, nullcontext

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.pod_dl_manifold import inviscid_burgers_implicit2D_LSPG_pod_dl_2D
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "figure.figsize": (12, 8),
    })


@contextmanager
def _solver_thread_limit(num_threads):
    if num_threads is None:
        yield
        return

    nthreads = int(num_threads)
    if nthreads < 1:
        yield
        return

    prev_torch_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(nthreads)
    except Exception:
        prev_torch_threads = None

    limit_ctx = (
        threadpool_limits(limits=nthreads)
        if threadpool_limits is not None
        else nullcontext()
    )

    try:
        with limit_ctx:
            yield
    finally:
        if prev_torch_threads is not None:
            try:
                torch.set_num_threads(prev_torch_threads)
            except Exception:
                pass


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


class ZScoreScaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class ZScoreUnscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class MinMaxScaler(nn.Module):
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


class MinMaxUnscaler(nn.Module):
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


def _activation_module(name):
    key = str(name).strip().lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(in_dim, hidden_dims, out_dim, activation="tanh"):
    dims = [int(in_dim)] + [int(v) for v in hidden_dims] + [int(out_dim)]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(_activation_module(activation))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PODDLAutoencoder(nn.Module):
    def __init__(
        self,
        q_dim,
        latent_dim,
        hidden_dims=(192, 96, 48),
        scaling="minmax_-1_1",
        activation="tanh",
    ):
        super().__init__()
        q_dim = int(q_dim)
        latent_dim = int(latent_dim)

        zeros = np.zeros((1, q_dim), dtype=np.float32)
        ones = np.ones((1, q_dim), dtype=np.float32)

        self.scaling = str(scaling)
        if self.scaling == "minmax_-1_1":
            self.scaler = MinMaxScaler(zeros, ones)
            self.unscaler = MinMaxUnscaler(zeros, ones)
        elif self.scaling == "zscore":
            self.scaler = ZScoreScaler(zeros, ones)
            self.unscaler = ZScoreUnscaler(zeros, ones)
        else:
            raise ValueError(f"Unsupported scaling: {scaling}")

        self.encoder = build_mlp(q_dim, hidden_dims, latent_dim, activation=activation)
        self.decoder = build_mlp(latent_dim, tuple(reversed(hidden_dims)), q_dim, activation=activation)

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


def _infer_scaling_from_state_dict(state_dict, fallback=None):
    keys = set(state_dict.keys())
    if any(k.startswith("scaler.center") for k in keys):
        return "minmax_-1_1"
    if any(k.startswith("scaler.mean") for k in keys):
        return "zscore"
    if fallback is not None:
        return str(fallback)
    return "zscore"


def _resolve_activation(checkpoint, scaling):
    if "activation" in checkpoint and checkpoint["activation"] is not None:
        return str(checkpoint["activation"])
    # Backward-compatible heuristic:
    # min-max [-1,1] configs in this project use tanh; z-score configs used SiLU before.
    return "tanh" if str(scaling) == "minmax_-1_1" else "silu"


def _load_stage2_use_u_ref(model_dir):
    stage2_metadata_path = os.path.join(os.path.dirname(model_dir), "stage2_projection_metadata.npz")
    if not os.path.exists(stage2_metadata_path):
        return None, stage2_metadata_path

    try:
        meta = np.load(stage2_metadata_path, allow_pickle=True)
    except Exception:
        return None, stage2_metadata_path

    if "use_u_ref" not in meta.files:
        return None, stage2_metadata_path

    value = bool(np.asarray(meta["use_u_ref"]).reshape(-1)[0])
    return value, stage2_metadata_path


def _resolve_u_ref(
    uref_mode,
    explicit_uref_file,
    model_use_u_ref,
    checkpoint_u_ref,
    model_dir,
    expected_size,
):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    checkpoint_u_ref = (
        None
        if checkpoint_u_ref is None
        else np.asarray(checkpoint_u_ref, dtype=np.float64).reshape(-1)
    )
    if checkpoint_u_ref is not None and checkpoint_u_ref.size != expected_size:
        raise ValueError(
            f"checkpoint u_ref size mismatch: got {checkpoint_u_ref.size}, expected {expected_size}."
        )

    candidate_files = []
    if explicit_uref_file is not None:
        candidate_files.append(explicit_uref_file)
    candidate_files.append(os.path.join(model_dir, "u_ref.npy"))
    candidate_files.append(os.path.join(os.path.dirname(model_dir), "u_ref.npy"))

    seen = set()
    filtered_candidates = []
    for path in candidate_files:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            filtered_candidates.append(path)

    if mode == "off":
        use_u_ref = False
    elif mode == "on":
        use_u_ref = True
    else:
        if model_use_u_ref is None:
            use_u_ref = (checkpoint_u_ref is not None) or any(
                os.path.exists(path) for path in filtered_candidates
            )
        else:
            use_u_ref = bool(model_use_u_ref)

    if not use_u_ref:
        return False, np.zeros(expected_size, dtype=np.float64), "zeros(off)"

    for path in filtered_candidates:
        if os.path.exists(path):
            u_ref = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
            if u_ref.size != expected_size:
                raise ValueError(
                    f"u_ref size mismatch in '{path}': got {u_ref.size}, expected {expected_size}."
                )
            return True, u_ref, path

    if checkpoint_u_ref is not None:
        return True, checkpoint_u_ref, "checkpoint"

    raise FileNotFoundError(
        "u_ref is required by current settings but no candidate file exists and checkpoint has no u_ref. "
        f"Checked: {filtered_candidates}"
    )


def _load_model_artifacts(model_dir, basis_file=None):
    model_path = os.path.join(model_dir, "pod_dl_autoencoder.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing POD-DL model checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", None)
    if state_dict is None:
        raise KeyError("Checkpoint missing key: 'state_dict'.")

    q_dim = int(checkpoint["q_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    hidden_dims = tuple(int(v) for v in checkpoint.get("hidden_dims", (192, 96, 48)))

    scaling_ckpt = checkpoint.get("scaling", None)
    scaling = _infer_scaling_from_state_dict(state_dict, fallback=scaling_ckpt)
    activation = _resolve_activation(checkpoint, scaling=scaling)

    model = PODDLAutoencoder(
        q_dim=q_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        scaling=scaling,
        activation=activation,
    )
    model.load_state_dict(state_dict)
    model.eval()

    candidate_basis_files = []
    if basis_file is not None:
        candidate_basis_files.append(basis_file)
    if checkpoint.get("basis_file", None) is not None:
        candidate_basis_files.append(str(checkpoint["basis_file"]))
    candidate_basis_files.append(os.path.join(os.path.dirname(model_dir), "basis.npy"))

    seen = set()
    filtered_candidates = []
    for path in candidate_basis_files:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            filtered_candidates.append(path)

    basis_path = None
    for path in filtered_candidates:
        if os.path.exists(path):
            basis_path = path
            break
    if basis_path is None:
        raise FileNotFoundError(
            "Could not find basis.npy for POD-DL model. "
            f"Checked: {filtered_candidates}"
        )

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}.")
    if basis.shape[1] < q_dim:
        raise ValueError(
            "basis has insufficient columns for checkpoint reduced dimension: "
            f"basis columns={basis.shape[1]}, required={q_dim}."
        )

    basis_q = basis[:, :q_dim]

    checkpoint_u_ref = checkpoint.get("u_ref", None)
    if checkpoint_u_ref is not None:
        checkpoint_u_ref = np.asarray(checkpoint_u_ref, dtype=np.float64).reshape(-1)

    return {
        "model_path": model_path,
        "basis_path": basis_path,
        "checkpoint": checkpoint,
        "checkpoint_u_ref": checkpoint_u_ref,
        "q_dim": q_dim,
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "scaling": scaling,
        "activation": activation,
        "basis_q": basis_q,
        "model": model,
    }


def main(
    mu1=4.75,
    mu2=0.02,
    model_dir=os.path.join("POD-DL", "pod_dl_model"),
    basis_file=None,
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    uref_file=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    solver_threads=1,
):
    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    set_latex_plot_style()

    grid_x = GRID_X
    grid_y = GRID_Y
    w0 = np.asarray(W0, dtype=np.float64).copy()
    mu_rom = [float(mu1), float(mu2)]

    num_cells_x = grid_x.size - 1
    num_cells_y = grid_y.size - 1

    model = _load_model_artifacts(model_dir=model_dir, basis_file=basis_file)
    model_use_u_ref_stage2, stage2_metadata_path = _load_stage2_use_u_ref(model_dir)
    checkpoint_use_u_ref = model["checkpoint"].get("use_u_ref", None)
    model_use_u_ref = (
        model_use_u_ref_stage2 if model_use_u_ref_stage2 is not None else checkpoint_use_u_ref
    )

    use_u_ref, u_ref, u_ref_source = _resolve_u_ref(
        uref_mode=uref_mode,
        explicit_uref_file=uref_file,
        model_use_u_ref=model_use_u_ref,
        checkpoint_u_ref=model["checkpoint_u_ref"],
        model_dir=model_dir,
        expected_size=model["basis_q"].shape[0],
    )

    if w0.size != model["basis_q"].shape[0]:
        raise ValueError(
            f"Initial condition size mismatch: W0 has {w0.size}, model has {model['basis_q'].shape[0]}."
        )

    print(f"[PROM-DL] Loaded model checkpoint: {model['model_path']}")
    print(f"[PROM-DL] Loaded basis: {model['basis_path']}")
    print(
        f"[PROM-DL] basis_q shape={model['basis_q'].shape}, latent_dim={model['latent_dim']}, "
        f"hidden_dims={model['hidden_dims']}"
    )
    print(
        f"[PROM-DL] activation={model['activation']}, scaling={model['scaling']}"
    )
    print(
        f"[PROM-DL] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )
    if solver_threads is not None and int(solver_threads) > 0:
        print(f"[PROM-DL] Limiting solver threads to {int(solver_threads)}.")

    t0 = time.time()
    with _solver_thread_limit(solver_threads):
        rom_snaps, rom_latent, rom_times = inviscid_burgers_implicit2D_LSPG_pod_dl_2D(
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu_rom,
            basis=model["basis_q"],
            pod_dl_model=model["model"],
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )
    elapsed_rom = time.time() - t0

    num_its, jac_time, res_time, ls_time = rom_times
    print(f"[PROM-DL] Elapsed PROM time: {elapsed_rom:.3e} seconds")
    print(f"[PROM-DL] Gauss-Newton iterations: {num_its}")
    print(
        "[PROM-DL] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )

    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(
        mu_rom,
        grid_x,
        grid_y,
        w0,
        dt,
        num_steps,
        snap_folder=snap_folder,
    )
    elapsed_hdm = time.time() - t0
    print(f"[PROM-DL] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"prom_dl_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[PROM-DL] ROM snapshots saved to: {rom_path}")

    latent_path = os.path.join(
        results_dir,
        f"prom_dl_latent_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(latent_path, rom_latent)
    print(f"[PROM-DL] Latent trajectory saved to: {latent_path}")

    snaps_to_plot = range(0, num_steps + 1, 100)
    fig, ax1, ax2 = plot_snaps(
        grid_x,
        grid_y,
        hdm_snaps,
        snaps_to_plot,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        grid_x,
        grid_y,
        rom_snaps,
        snaps_to_plot,
        label="PROM-DL",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )
    fig.suptitle(rf"$\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$", y=0.98)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    plt.tight_layout()

    fig_path = os.path.join(
        results_dir,
        f"prom_dl_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PROM-DL] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[PROM-DL] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"prom_dl_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("mu1", mu1),
                    ("mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("model_dir", model_dir),
                    ("basis_file", basis_file),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("solver_threads", solver_threads),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("provided_uref_file", uref_file),
                    ("checkpoint_use_u_ref", checkpoint_use_u_ref),
                    ("stage2_use_u_ref", model_use_u_ref_stage2),
                    (
                        "stage2_projection_metadata",
                        stage2_metadata_path if os.path.exists(stage2_metadata_path) else None,
                    ),
                ],
            ),
            (
                "discretization",
                [
                    ("num_cells_x", num_cells_x),
                    ("num_cells_y", num_cells_y),
                    ("full_state_size", w0.size),
                ],
            ),
            (
                "pod_dl_model",
                [
                    ("model_pt", model["model_path"]),
                    ("basis_path", model["basis_path"]),
                    ("basis_q_shape", model["basis_q"].shape),
                    ("q_dim", model["q_dim"]),
                    ("latent_dim", model["latent_dim"]),
                    ("hidden_dims", model["hidden_dims"]),
                    ("activation", model["activation"]),
                    ("scaling", model["scaling"]),
                    ("checkpoint_seed", model["checkpoint"].get("seed", None)),
                ],
            ),
            (
                "prom_timing",
                [
                    ("total_prom_time_seconds", elapsed_rom),
                    ("avg_prom_time_per_step_seconds", elapsed_rom / max(1, num_steps)),
                    ("gn_iterations_total", num_its),
                    ("avg_gn_iterations_per_step", num_its / max(1, num_steps)),
                    ("jacobian_time_seconds", jac_time),
                    ("residual_time_seconds", res_time),
                    ("linear_solve_time_seconds", ls_time),
                    ("hdm_load_or_solve_time_seconds", elapsed_hdm),
                ],
            ),
            (
                "error_metrics",
                [
                    ("relative_l2_error", rel_err_l2),
                    ("relative_error_percent", relative_error),
                ],
            ),
            (
                "outputs",
                [
                    ("rom_snapshots_npy", rom_path),
                    ("latent_trajectory_npy", latent_path),
                    ("comparison_plot_png", fig_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[PROM-DL] Text summary saved to: {report_path}")

    return elapsed_rom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019)
