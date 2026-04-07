#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Case 2 ANN closure ROM with selectable backend.

Case 2 mapping:
    qN_s = N(mu1, mu2, t)

Backends:
- solve_backend='prom': full LSPG solve
- solve_backend='hprom' and use_ecsw=True: ECSW hyper-reduced solve
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import get_snapshot_params, load_or_compute_snaps, plot_snaps
from burgers.core import inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
from burgers.pod_ann_manifold import (
    compute_ECSW_training_matrix_2D_pod_ann_case2,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_ecsw,
)
from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset
try:
    from project_layout import (
        RUNS_CASE2_DIR,
        RUNS_ECSW_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        resolve_stage3_model,
        write_kv_txt,
    )
except ModuleNotFoundError:
    from .project_layout import (
        RUNS_CASE2_DIR,
        RUNS_ECSW_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        resolve_stage3_model,
        write_kv_txt,
    )


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


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        return self.fc6(x)


class Case2Model(nn.Module):
    """Input: (mu1, mu2, t), output: qN_s."""

    def __init__(self, n_s):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, 3)), np.ones((1, 3)))
        self.core = CoreMLP(3, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        x_n = self.scaler(x_raw)
        y_n = self.core(x_n)
        y_raw = self.unscaler(y_n)
        return y_raw


class ANNVectorWrapper(nn.Module):
    """Ensure vector output for vector input."""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        in_device = x.device
        model_device = next(self.base.parameters()).device

        if x.ndim == 1:
            x_in = x.unsqueeze(0)
            if x_in.device != model_device:
                x_in = x_in.to(model_device)
            out = self.base(x_in).reshape(-1)
            if out.device != in_device:
                out = out.to(in_device)
            return out

        x_in = x if x.device == model_device else x.to(model_device)
        out = self.base(x_in)
        if out.ndim == 2 and out.shape[0] == 1:
            out = out.reshape(-1)
        if out.device != in_device:
            out = out.to(in_device)
        return out


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _reconstruct_case2_full_snaps(red_coords, basis, basis2, u_ref, ann_model, mu, dt, device="cpu"):
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    red_coords = np.asarray(red_coords, dtype=np.float64)

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    snaps = np.zeros((u_ref.size, red_coords.shape[1]), dtype=np.float64)
    for k in range(red_coords.shape[1]):
        yk = red_coords[:, k]
        t = float(k * dt)
        with torch.no_grad():
            x_t = torch.tensor([mu1, mu2, t], dtype=torch.float32, device=device)
            qbar = ann_model(x_t).detach().cpu().numpy().reshape(-1)
        snaps[:, k] = u_ref + basis @ yk + basis2 @ qbar

    return snaps


def _load_or_build_case2_ecsw_weights(
    total_modes,
    n_primary,
    v,
    vbar,
    u_ref,
    ann_model,
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu_samples,
    snap_folder,
    rebuild_weights=False,
    snap_sample_factor=50,
    snap_time_offset=3,
):
    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)
    os.makedirs(RUNS_ECSW_DIR, exist_ok=True)
    weights_path = os.path.join(
        RUNS_ECSW_DIR,
        f"ecsw_weights_ann_case2_n{n_primary}_ntot{total_modes}.npy",
    )

    if (not rebuild_weights) and os.path.exists(weights_path):
        weights = np.asarray(np.load(weights_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if weights.size != expected_num_cells:
            raise ValueError(
                f"ECSW weights size mismatch at '{weights_path}': got {weights.size}, expected {expected_num_cells}."
            )
        return weights, weights_path, "loaded_local", np.nan, int(np.sum(weights > 0.0))

    clist = []

    for mu in mu_samples:
        mu_snaps = load_or_compute_snaps(
            mu=mu,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
        )

        stop_col = num_steps
        snaps_now = mu_snaps[:, snap_time_offset:stop_col:snap_sample_factor]
        snaps_prev = mu_snaps[:, 0:stop_col - snap_time_offset:snap_sample_factor]

        if snaps_now.shape[1] != snaps_prev.shape[1]:
            raise RuntimeError(
                "ECSW snapshot alignment failed: "
                f"snaps_now has {snaps_now.shape[1]} columns, snaps_prev has {snaps_prev.shape[1]} columns."
            )
        if snaps_now.shape[1] == 0:
            raise RuntimeError(
                "ECSW training produced zero columns. Adjust snap_time_offset or snap_sample_factor."
            )

        t_samples = dt * np.arange(snap_time_offset, stop_col, snap_sample_factor, dtype=np.float64)
        if t_samples.size != snaps_now.shape[1]:
            raise RuntimeError(
                f"Time-grid mismatch for ECSW training: t_samples={t_samples.size}, snaps={snaps_now.shape[1]}."
            )

        ci = compute_ECSW_training_matrix_2D_pod_ann_case2(
            snaps=snaps_now,
            prev_snaps=snaps_prev,
            t_samples=t_samples,
            basis=np.asarray(v, dtype=np.float64),
            basis2=np.asarray(vbar, dtype=np.float64),
            ann_model=ann_model,
            res=inviscid_burgers_res2D,
            jac=inviscid_burgers_exact_jac2D,
            grid_x=grid_x,
            grid_y=grid_y,
            dt=dt,
            mu=mu,
            u_ref=u_ref,
        )
        clist.append(ci)

    C = np.vstack(clist)
    C_ecm = np.ascontiguousarray(C, dtype=np.float64)
    b = np.ascontiguousarray(C_ecm.sum(axis=1), dtype=np.float64)

    rsvd = RandomizedSingularValueDecomposition()
    u, _, _, _ = rsvd.Calculate(C_ecm.T, 1e-8)

    selector = EmpiricalCubatureMethod()
    selector.SetUp(
        u,
        InitialCandidatesSet=None,
        constrain_sum_of_weights=True,
        constrain_conditions=False,
    )
    selector.Run()

    weights = np.zeros(expected_num_cells, dtype=np.float64)
    weights[selector.z] = selector.w

    np.save(weights_path, weights)
    denom = np.linalg.norm(b)
    rel_res = float(np.linalg.norm(C_ecm @ weights - b) / denom) if denom > 0.0 else np.nan
    n_ecsw = int(np.sum(weights > 0.0))
    return weights, weights_path, "computed", rel_res, n_ecsw


def _load_case2_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    n_s = int(ckpt["n_s"])
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"Case2 checkpoint in_dim={in_dim}, expected 3")

    model = Case2Model(n_s).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return model, n_s, ckpt


def _resolve_total_modes_from_checkpoint_or_dataset(ckpt):
    ntot = ckpt.get("dataset_ntot", None)
    if ntot is not None:
        return int(ntot)

    _, ntot, _, _, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=None,
        expected_backend=None,
    )
    return int(ntot)


def _load_basis_and_reference(total_modes, n_primary):
    basis_path = resolve_stage1_artifact("basis.npy")
    uref_path = resolve_stage1_artifact("u_ref.npy")

    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Missing basis file: {basis_path}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got shape {basis.shape}")
    if basis.shape[1] < total_modes:
        raise ValueError(
            f"basis.npy has {basis.shape[1]} modes, but total_modes={total_modes} is required."
        )

    vtot = basis[:, :total_modes]
    v = vtot[:, :n_primary]
    vbar = vtot[:, n_primary:total_modes]

    if os.path.exists(uref_path):
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(vtot.shape[0], dtype=np.float64)

    if u_ref.size != vtot.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {vtot.shape[0]} from basis rows."
        )

    return vtot, v, vbar, u_ref, basis_path, uref_path


def main(argv=None):
    # -----------------------------
    # User settings
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Run Case 2 ANN closure with PROM/HPROM backend."
    )
    parser.add_argument("--backend", choices=("prom", "hprom"), default="hprom")
    parser.add_argument("--mu1", type=float, default=4.56)
    parser.add_argument("--mu2", type=float, default=0.019)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--no-ecsw", action="store_true", help="Disable ECSW (HPROM falls back to PROM).")
    parser.add_argument("--rebuild-ecsw", action="store_true", help="Recompute ECSW weights.")
    parser.add_argument("--ecsw-num-training-mu", type=int, default=9)
    parser.add_argument("--ecsw-snap-sample-factor", type=int, default=50)
    parser.add_argument("--ecsw-snap-time-offset", type=int, default=3)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1e-12)
    parser.add_argument(
        "--model-name",
        type=str,
        default="case2_model.pt",
        help="Checkpoint filename under Results/Stage3/models (used if --model-path is not set).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    args = parser.parse_args(argv)

    mu_test = [float(args.mu1), float(args.mu2)]
    solve_backend = str(args.backend).strip().lower()
    use_ecsw = not bool(args.no_ecsw)
    rebuild_ecsw_weights = bool(args.rebuild_ecsw)
    ecsw_snap_sample_factor = int(args.ecsw_snap_sample_factor)
    ecsw_snap_time_offset = int(args.ecsw_snap_time_offset)
    ecsw_num_training_mu = int(args.ecsw_num_training_mu)
    max_its = int(args.max_its)
    relnorm_cutoff = float(args.relnorm_cutoff)
    min_delta = float(args.min_delta)
    linear_solver = str(args.linear_solver).strip().lower()
    normal_eq_reg = float(args.normal_eq_reg)
    model_name = str(args.model_name).strip()
    model_path_override = args.model_path

    device = str(args.device).strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("[Case2] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    set_latex_plot_style()
    ensure_layout_dirs()
    os.makedirs(RUNS_CASE2_DIR, exist_ok=True)

    solve_backend = str(solve_backend).strip().lower()
    if solve_backend not in ("prom", "hprom"):
        raise ValueError("solve_backend must be 'prom' or 'hprom'.")

    effective_backend = solve_backend
    if solve_backend == "hprom" and not use_ecsw:
        print("[Case2] solve_backend='hprom' with use_ecsw=False -> falling back to PROM solve.")
        effective_backend = "prom"
    if solve_backend == "prom" and use_ecsw:
        print("[Case2] use_ecsw=True ignored because solve_backend='prom'.")

    if model_path_override is None:
        if len(model_name) == 0:
            raise ValueError("--model-name cannot be empty.")
        if not model_name.endswith(".pt"):
            model_name = f"{model_name}.pt"
        model_path = resolve_stage3_model(model_name)
    else:
        model_path = os.path.abspath(model_path_override)
        model_name = os.path.basename(model_path)
    base_model, n_s, ckpt = _load_case2_model(model_path, device=device)
    ann_model = ANNVectorWrapper(base_model).to(device)
    ann_model.eval()

    total_modes = _resolve_total_modes_from_checkpoint_or_dataset(ckpt)
    n_p = int(total_modes - n_s)
    if n_p < 1:
        raise ValueError(f"Invalid mode split: total_modes={total_modes}, n_s={n_s}")

    vtot, v, vbar, u_ref, basis_path, uref_path = _load_basis_and_reference(total_modes, n_p)

    w0 = np.asarray(W0, dtype=np.float64).reshape(-1).copy()
    if w0.size != vtot.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {vtot.shape[0]} from basis."
        )

    snap_folder = os.path.join(PROJECT_ROOT, "Results", "param_snaps")
    os.makedirs(snap_folder, exist_ok=True)

    print(f"[Case2] device = {device}")
    print(f"[Case2] checkpoint = {model_path}")
    print(f"[Case2] basis = {basis_path}")
    print(f"[Case2] u_ref = {uref_path if os.path.exists(uref_path) else 'zeros'}")
    print(f"[Case2] solve_backend(requested) = {solve_backend}")
    print(f"[Case2] solve_backend(effective) = {effective_backend}")
    print(f"[Case2] use_ecsw = {use_ecsw}")

    hdm_snaps = load_or_compute_snaps(
        mu=mu_test,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        w0=w0,
        dt=DT,
        num_steps=NUM_STEPS,
        snap_folder=snap_folder,
    )

    ecsw_residual = np.nan
    n_ecsw_elements = None
    ecsw_setup_elapsed = 0.0
    online_solve_elapsed = np.nan

    if effective_backend == "hprom":
        mu_train_candidates = get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        )
        ecsw_num_training_mu = max(1, min(int(ecsw_num_training_mu), len(mu_train_candidates)))
        mu_train_list = mu_train_candidates[:ecsw_num_training_mu]

        t_ecsw0 = time.time()
        weights, weights_path, weights_source, ecsw_residual, n_ecsw_elements = _load_or_build_case2_ecsw_weights(
            total_modes=total_modes,
            n_primary=n_p,
            v=v,
            vbar=vbar,
            u_ref=u_ref,
            ann_model=ann_model,
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu_samples=mu_train_list,
            snap_folder=snap_folder,
            rebuild_weights=rebuild_ecsw_weights,
            snap_sample_factor=ecsw_snap_sample_factor,
            snap_time_offset=ecsw_snap_time_offset,
        )
        ecsw_setup_elapsed = time.time() - t_ecsw0
        if not os.path.abspath(weights_path).startswith(os.path.abspath(RUNS_ECSW_DIR) + os.sep):
            raise RuntimeError(
                f"ECSW weights must be under '{RUNS_ECSW_DIR}', got: {weights_path}"
            )

        t_solve0 = time.time()
        red_coords, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_ecsw(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            weights=weights,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu_test,
            ann_model=ann_model,
            ref=None,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        online_solve_elapsed = time.time() - t_solve0

        rom_snaps = _reconstruct_case2_full_snaps(
            red_coords=red_coords,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            ann_model=ann_model,
            mu=mu_test,
            dt=DT,
            device=device,
        )

        print(f"[Case2] ECSW weights: {weights_path} ({weights_source})")
        print(f"[Case2] ECSW training trajectories used = {ecsw_num_training_mu}")
        print(f"[Case2] N_e = {n_ecsw_elements}")
        print(f"[Case2] ECSW residual = {ecsw_residual}")

    else:
        t_solve0 = time.time()
        rom_snaps, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu_test,
            ann_model=ann_model,
            ref=None,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )
        online_solve_elapsed = time.time() - t_solve0

    elapsed = online_solve_elapsed

    num_its, jac_time, res_time, ls_time = rom_times

    rel_err = 100.0 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)

    backend_tag = "hprom" if effective_backend == "hprom" else "prom"
    tag = _safe_mu_tag(mu_test)
    run_tag = f"case2_{backend_tag}_ann_{tag}_n{n_p}_ntot{total_modes}"

    out_npy = os.path.join(
        RUNS_CASE2_DIR,
        f"{run_tag}_snaps.npy",
    )
    np.save(out_npy, rom_snaps)

    plot_steps = list(range(0, NUM_STEPS + 1, 100))
    if NUM_STEPS not in plot_steps:
        plot_steps.append(NUM_STEPS)

    fig, ax1, ax2 = plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snaps,
        plot_steps,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        GRID_X,
        GRID_Y,
        rom_snaps,
        plot_steps,
        label="HPROM-ANN Case 2" if effective_backend == "hprom" else "PROM-ANN Case 2",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    out_png = os.path.join(
        RUNS_CASE2_DIR,
        f"{run_tag}_hdm_vs_rom.png",
    )
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    summary_txt = os.path.join(RUNS_CASE2_DIR, f"{run_tag}_summary.txt")
    write_kv_txt(
        summary_txt,
        [
            ("mu_test", mu_test),
            ("device", device),
            ("model_name", model_name),
            ("model_path", model_path),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path if os.path.exists(uref_path) else "zeros"),
            ("solve_backend_requested", solve_backend),
            ("solve_backend_effective", effective_backend),
            ("use_ecsw", use_ecsw),
            ("rebuild_ecsw_weights", rebuild_ecsw_weights),
            ("ecsw_num_training_mu", ecsw_num_training_mu),
            ("ecsw_snap_sample_factor", ecsw_snap_sample_factor),
            ("ecsw_snap_time_offset", ecsw_snap_time_offset),
            ("ecsw_weights_path", weights_path if effective_backend == "hprom" else "N/A"),
            ("ecsw_residual", ecsw_residual),
            ("n_ecsw_elements", n_ecsw_elements),
            ("ecsw_setup_elapsed_s", ecsw_setup_elapsed),
            ("online_solve_elapsed_s", online_solve_elapsed),
            ("elapsed_s", elapsed),
            ("num_iterations", num_its),
            ("jac_time_s", jac_time),
            ("res_time_s", res_time),
            ("ls_time_s", ls_time),
            ("relative_error_percent", rel_err),
            ("snaps_output", out_npy),
            ("plot_output", out_png),
        ],
    )

    print(f"[Case2] ecsw_setup_elapsed = {ecsw_setup_elapsed:.3e} s")
    print(f"[Case2] online_solve_elapsed = {online_solve_elapsed:.3e} s")
    print(f"[Case2] elapsed = {elapsed:.3e} s")
    print(f"[Case2] its={num_its} | jac={jac_time:.3e} | res={res_time:.3e} | ls={ls_time:.3e}")
    print(f"[Case2] relative error vs HDM: {rel_err:.2f}%")
    print(f"[Case2] saved snaps: {out_npy}")
    print(f"[Case2] saved plot:  {out_png}")
    print(f"[Case2] summary:     {summary_txt}")


if __name__ == "__main__":
    main()
