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
from burgers.ecsw_utils import build_ecsw_snapshot_plan, direct_left_singular_vectors
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

try:
    from gpr_map_common import build_torch_case2_gpr_from_ckpt
except ModuleNotFoundError:
    from .gpr_map_common import build_torch_case2_gpr_from_ckpt


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
    def __init__(self, in_dim, out_dim, hidden_dims=(32, 64, 128, 256, 256), activation="elu", dropout=0.0):
        super().__init__()
        hidden_dims = tuple(int(d) for d in hidden_dims)
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


class Case2Model(nn.Module):
    """Input: (mu1, mu2, t), output: qN_s."""

    def __init__(self, n_s, hidden_dims=(32, 64, 128, 256, 256), activation="elu", dropout=0.0):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, 3)), np.ones((1, 3)))
        self.core = CoreMLP(
            3,
            n_s,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
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


class ANNVectorSliceWrapper(nn.Module):
    """Ensure vector output and drop leading secondary coordinates."""

    def __init__(self, base_model, drop_first=0, keep_last=None):
        super().__init__()
        self.base = base_model
        self.drop_first = int(drop_first)
        self.keep_last = None if keep_last is None else int(keep_last)
        if self.drop_first < 0:
            raise ValueError(f"drop_first must be nonnegative, got {self.drop_first}.")
        if self.keep_last is not None and self.keep_last < 1:
            raise ValueError(f"keep_last must be positive, got {self.keep_last}.")

    def _slice(self, out):
        if self.drop_first == 0 and self.keep_last is None:
            return out
        if out.ndim == 1:
            y = out[self.drop_first :]
            if self.keep_last is not None:
                y = y[: self.keep_last]
            return y
        if out.ndim == 2:
            y = out[:, self.drop_first :]
            if self.keep_last is not None:
                y = y[:, : self.keep_last]
            return y
        raise ValueError(f"Unsupported ANN output ndim={out.ndim}.")

    def forward(self, x):
        in_device = x.device
        model_device = next(self.base.parameters()).device

        if x.ndim == 1:
            x_in = x.unsqueeze(0)
            if x_in.device != model_device:
                x_in = x_in.to(model_device)
            out = self.base(x_in).reshape(-1)
            out = self._slice(out)
            if out.device != in_device:
                out = out.to(in_device)
            return out

        x_in = x if x.device == model_device else x.to(model_device)
        out = self.base(x_in)
        out = self._slice(out)
        if out.ndim == 2 and out.shape[0] == 1:
            out = out.reshape(-1)
        if out.device != in_device:
            out = out.to(in_device)
        return out


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _safe_tag_extra(tag):
    tag = str(tag or "").strip()
    if not tag:
        return ""
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in tag)


def _predict_case2_secondary_coords(num_steps, ann_model, mu, dt, device="cpu"):
    mu1 = float(mu[0])
    mu2 = float(mu[1])
    qbar_cols = []
    for k in range(int(num_steps)):
        t = float(k * dt)
        with torch.no_grad():
            x_t = torch.tensor([mu1, mu2, t], dtype=torch.float32, device=device)
            qbar = ann_model(x_t).detach().cpu().numpy().reshape(-1)
        qbar_cols.append(np.asarray(qbar, dtype=np.float64))
    return np.column_stack(qbar_cols)


def _assemble_case2_full_qN(red_coords, ann_model, mu, dt, device="cpu"):
    red_coords = np.asarray(red_coords, dtype=np.float64)
    qbar = _predict_case2_secondary_coords(red_coords.shape[1], ann_model, mu, dt, device=device)
    if qbar.shape[1] != red_coords.shape[1]:
        raise RuntimeError(
            f"Case2 qbar time length mismatch: qbar={qbar.shape}, red_coords={red_coords.shape}"
        )
    return np.vstack([red_coords, qbar])


def _reconstruct_case2_full_snaps(red_coords, basis, basis2, u_ref, ann_model, mu, dt, device="cpu"):
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    red_coords = np.asarray(red_coords, dtype=np.float64)
    qbar = _predict_case2_secondary_coords(red_coords.shape[1], ann_model, mu, dt, device=device)
    snaps = u_ref[:, None] + basis @ red_coords + basis2 @ qbar
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
    snap_time_offset=3,
    snapshot_percent=2.0,
    snapshot_random_seed=42,
    snapshot_mode="global_param_time_stratified",
    ensure_mu_coverage=True,
    ecsw_weights_dir=None,
    ecsw_tag=None,
):
    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)
    if ecsw_weights_dir is None:
        ecsw_weights_dir = RUNS_ECSW_DIR
    ecsw_weights_dir = os.path.abspath(os.path.expanduser(str(ecsw_weights_dir)))
    os.makedirs(ecsw_weights_dir, exist_ok=True)
    tag_part = ""
    if ecsw_tag:
        safe_tag = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(ecsw_tag))
        tag_part = f"_{safe_tag}"
    weights_path = os.path.join(
        ecsw_weights_dir,
        f"ecsw_weights_ann_case2{tag_part}_n{n_primary}_ntot{total_modes}.npy",
    )

    if (not rebuild_weights) and os.path.exists(weights_path):
        weights = np.asarray(np.load(weights_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if weights.size != expected_num_cells:
            raise ValueError(
                f"ECSW weights size mismatch at '{weights_path}': got {weights.size}, expected {expected_num_cells}."
            )
        return weights, weights_path, "loaded_local", np.nan, int(np.sum(weights > 0.0))

    snapshot_percent = float(snapshot_percent)
    if not np.isfinite(snapshot_percent) or snapshot_percent <= 0.0:
        raise ValueError("snapshot_percent must be a finite value > 0.")
    ecsw_plan = build_ecsw_snapshot_plan(
        num_steps=num_steps,
        snap_time_offset=snap_time_offset,
        num_mu=len(mu_samples),
        mode=snapshot_mode,
        total_snapshots=None,
        total_snapshots_percent=snapshot_percent,
        mu_points=mu_samples,
        random_seed=int(snapshot_random_seed),
        ensure_mu_coverage=bool(ensure_mu_coverage),
    )

    clist = []

    for imu, mu in enumerate(mu_samples):
        mu_snaps = load_or_compute_snaps(
            mu=mu,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            snap_folder=snap_folder,
        )

        now_cols = np.asarray(ecsw_plan["selected_now_cols_by_mu"][imu], dtype=int)
        if now_cols.size == 0:
            continue
        # `snap_time_offset` only controls the earliest eligible current state.
        # The backward-Euler residual always uses the immediate predecessor.
        prev_cols = now_cols - 1
        snaps_now = mu_snaps[:, now_cols]
        snaps_prev = mu_snaps[:, prev_cols]

        if snaps_now.shape[1] != snaps_prev.shape[1]:
            raise RuntimeError(
                "ECSW snapshot alignment failed: "
                f"snaps_now has {snaps_now.shape[1]} columns, snaps_prev has {snaps_prev.shape[1]} columns."
            )
        if snaps_now.shape[1] == 0:
            continue

        t_samples = dt * now_cols.astype(np.float64)
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

    if not clist:
        raise RuntimeError(
            "ECSW training produced zero columns for all mu samples. "
            "Increase ecsw_snapshot_percent or adjust snap_time_offset."
        )

    C = np.vstack(clist)
    C_ecm = np.ascontiguousarray(C, dtype=np.float64)
    b = np.ascontiguousarray(C_ecm.sum(axis=1), dtype=np.float64)

    u = direct_left_singular_vectors(C_ecm.T, relative_tolerance=1e-8)

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
    in_dim = int(ckpt.get("in_dim", 3))
    if in_dim != 3:
        raise ValueError(f"Case2 checkpoint in_dim={in_dim}, expected 3")

    checkpoint_format = str(ckpt.get("format", "")).strip().lower()
    is_gpr_checkpoint = checkpoint_format in (
        "gpr_map",
        "gpr_map_full",
        "sparse_gpr_map",
        "sparse_gpr_map_full",
    ) or ("gpr_payload" in ckpt) or ("sparse_gp_payload" in ckpt)

    if "n_s" in ckpt:
        out_dim = int(ckpt["n_s"])
        output_kind = "secondary"
    elif "n_tot" in ckpt:
        out_dim = int(ckpt["n_tot"])
        output_kind = "q_tot"
    else:
        raise KeyError(
            "Checkpoint must contain either 'n_s' for a Case-2 secondary model "
            "or 'n_tot' for a full q_tot master model."
        )

    if is_gpr_checkpoint:
        model = build_torch_case2_gpr_from_ckpt(ckpt).to(device)
        model.eval()
        return model, out_dim, output_kind, ckpt

    hidden_dims = tuple(int(d) for d in ckpt.get("hidden_dims", (32, 64, 128, 256, 256)))
    activation = str(ckpt.get("activation", "elu")).strip().lower()
    dropout = float(ckpt.get("dropout", 0.0))
    model = Case2Model(
        out_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return model, out_dim, output_kind, ckpt


def _resolve_total_modes_from_checkpoint_or_dataset(ckpt):
    ntot = ckpt.get("n_tot", None)
    if ntot is not None:
        return int(ntot)

    ntot = ckpt.get("dataset_ntot", None)
    if ntot is not None:
        return int(ntot)

    _, ntot, _, _, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=None,
        expected_backend=None,
    )
    return int(ntot)


def _load_basis_and_reference(total_modes, n_primary, basis_path_override=None, uref_path_override=None):
    if basis_path_override:
        basis_path = os.path.abspath(os.path.expanduser(str(basis_path_override)))
    else:
        basis_path = resolve_stage1_artifact("basis.npy")

    if uref_path_override:
        uref_path = os.path.abspath(os.path.expanduser(str(uref_path_override)))
    else:
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
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--no-ecsw", action="store_true", help="Disable ECSW (HPROM falls back to PROM).")
    parser.add_argument("--rebuild-ecsw", action="store_true", help="Recompute ECSW weights.")
    parser.add_argument(
        "--ecsw-only",
        action="store_true",
        help="Build or load the ECSW rule and exit before the online solve.",
    )
    parser.add_argument("--ecsw-num-training-mu", type=int, default=9)
    parser.add_argument(
        "--ecsw-snap-time-offset",
        type=int,
        default=3,
        help="Earliest current snapshot index eligible for ECM training; the residual predecessor is always one step earlier.",
    )
    parser.add_argument("--ecsw-snapshot-percent", type=float, default=2.0)
    parser.add_argument(
        "--ecsw-snapshot-mode",
        choices=("strided_per_mu", "global_stratified_random", "global_param_time_stratified"),
        default="global_param_time_stratified",
        help="Snapshot-column selection mode used to build ANN ECSW training matrices.",
    )
    parser.add_argument("--ecsw-random-seed", type=int, default=42)
    parser.add_argument("--ecsw-ensure-mu-coverage", dest="ecsw_ensure_mu_coverage", action="store_true")
    parser.add_argument("--ecsw-no-ensure-mu-coverage", dest="ecsw_ensure_mu_coverage", action="store_false")
    parser.set_defaults(ecsw_ensure_mu_coverage=True)
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
    parser.add_argument(
        "--target-primary-modes",
        type=int,
        default=None,
        help=(
            "Optional online primary dimension. If larger than the checkpoint "
            "primary dimension, the leading secondary outputs are discarded."
        ),
    )
    parser.add_argument(
        "--drop-first-secondary",
        type=int,
        default=None,
        help=(
            "Number of leading ANN secondary outputs to discard. Default is "
            "target_primary_modes - checkpoint_primary_modes when "
            "--target-primary-modes is provided."
        ),
    )
    parser.add_argument(
        "--run-tag-extra",
        type=str,
        default="",
        help="Optional extra token inserted into output filenames after 'case2_<backend>_ann'.",
    )
    parser.add_argument(
        "--basis-path",
        type=str,
        default=None,
        help="Optional basis override.",
    )
    parser.add_argument(
        "--u-ref-path",
        type=str,
        default=None,
        help="Optional reference-state override.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional Case-2 output root override.",
    )
    parser.add_argument(
        "--ecsw-weights-dir",
        type=str,
        default=None,
        help="Optional directory for Case-2 ANN ECSW weights.",
    )
    parser.add_argument("--no-save-rom-snaps", action="store_true", help="Do not save full rom_snaps array.")
    parser.add_argument("--no-plot", action="store_true", help="Skip HDM-vs-ROM plotting.")
    args = parser.parse_args(argv)

    mu_test = [float(args.mu1), float(args.mu2)]
    solve_backend = str(args.backend).strip().lower()
    use_ecsw = not bool(args.no_ecsw)
    rebuild_ecsw_weights = bool(args.rebuild_ecsw)
    ecsw_snap_time_offset = int(args.ecsw_snap_time_offset)
    ecsw_snapshot_percent = float(args.ecsw_snapshot_percent)
    ecsw_snapshot_mode = str(args.ecsw_snapshot_mode).strip().lower()
    ecsw_snapshot_random_seed = int(args.ecsw_random_seed)
    ecsw_ensure_mu_coverage = bool(args.ecsw_ensure_mu_coverage)
    ecsw_num_training_mu = int(args.ecsw_num_training_mu)
    max_its = int(args.max_its)
    relnorm_cutoff = float(args.relnorm_cutoff)
    min_delta = float(args.min_delta)
    linear_solver = str(args.linear_solver).strip().lower()
    normal_eq_reg = float(args.normal_eq_reg)
    model_name = str(args.model_name).strip()
    model_path_override = args.model_path
    target_primary_modes = args.target_primary_modes
    drop_first_secondary_arg = args.drop_first_secondary
    run_tag_extra = _safe_tag_extra(args.run_tag_extra)
    basis_path_override = args.basis_path
    uref_path_override = args.u_ref_path
    output_root = RUNS_CASE2_DIR if args.output_root is None else os.path.abspath(os.path.expanduser(str(args.output_root)))
    ecsw_weights_dir = RUNS_ECSW_DIR if args.ecsw_weights_dir is None else os.path.abspath(os.path.expanduser(str(args.ecsw_weights_dir)))
    save_rom_snaps = not bool(args.no_save_rom_snaps)
    make_plot = not bool(args.no_plot)

    device = str(args.device).strip().lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[Case2] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    set_latex_plot_style()
    ensure_layout_dirs()
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(ecsw_weights_dir, exist_ok=True)

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
    base_model, checkpoint_output_dim, model_output_kind, ckpt = _load_case2_model(
        model_path,
        device=device,
    )
    total_modes = _resolve_total_modes_from_checkpoint_or_dataset(ckpt)

    if model_output_kind == "secondary":
        n_s_checkpoint = int(checkpoint_output_dim)
        n_p_checkpoint = int(total_modes - n_s_checkpoint)
        if n_p_checkpoint < 1:
            raise ValueError(
                f"Invalid checkpoint mode split: total_modes={total_modes}, n_s={n_s_checkpoint}"
            )
    elif model_output_kind == "q_tot":
        if int(checkpoint_output_dim) != int(total_modes):
            raise ValueError(
                "Full q_tot checkpoint dimension mismatch: "
                f"output_dim={checkpoint_output_dim}, total_modes={total_modes}."
            )
        n_s_checkpoint = int(total_modes)
        n_p_checkpoint = 0
    else:
        raise ValueError(f"Unsupported model_output_kind={model_output_kind!r}.")

    if target_primary_modes is None:
        if model_output_kind == "q_tot":
            raise ValueError(
                "--target-primary-modes is required when using a full q_tot master model."
            )
        n_p = n_p_checkpoint
    else:
        n_p = int(target_primary_modes)
    if n_p < 1 or n_p >= total_modes:
        raise ValueError(f"Invalid target primary modes n_p={n_p} for total_modes={total_modes}.")
    if model_output_kind == "secondary" and n_p < n_p_checkpoint:
        raise ValueError(
            "This runner only supports trimming secondary outputs by increasing "
            f"the primary dimension. checkpoint n_p={n_p_checkpoint}, target n_p={n_p}."
        )

    if drop_first_secondary_arg is None:
        if model_output_kind == "q_tot":
            drop_first_secondary = n_p
        else:
            drop_first_secondary = n_p - n_p_checkpoint
    else:
        drop_first_secondary = int(drop_first_secondary_arg)
    if drop_first_secondary < 0:
        raise ValueError(f"drop_first_secondary must be nonnegative, got {drop_first_secondary}.")

    n_s = int(n_s_checkpoint - drop_first_secondary)
    expected_n_s = int(total_modes - n_p)
    if n_s != expected_n_s:
        raise ValueError(
            "Inconsistent trimmed Case-2 dimensions: "
            f"checkpoint n_s={n_s_checkpoint}, drop_first_secondary={drop_first_secondary}, "
            f"target n_p={n_p}, total_modes={total_modes}; got n_s={n_s}, expected {expected_n_s}."
        )
    if n_p < 1:
        raise ValueError(f"Invalid mode split: total_modes={total_modes}, n_s={n_s}")
    if drop_first_secondary > 0:
        ann_model = ANNVectorSliceWrapper(
            base_model,
            drop_first=drop_first_secondary,
            keep_last=n_s,
        ).to(device)
    else:
        ann_model = ANNVectorWrapper(base_model).to(device)
    ann_model.eval()

    vtot, v, vbar, u_ref, basis_path, uref_path = _load_basis_and_reference(
        total_modes,
        n_p,
        basis_path_override=basis_path_override,
        uref_path_override=uref_path_override,
    )

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
    print(f"[Case2] checkpoint primary modes = {n_p_checkpoint}")
    print(f"[Case2] checkpoint secondary modes = {n_s_checkpoint}")
    print(f"[Case2] target primary modes = {n_p}")
    print(f"[Case2] secondary modes used = {n_s}")
    print(f"[Case2] drop_first_secondary = {drop_first_secondary}")
    print(f"[Case2] ecsw_snapshot_mode = {ecsw_snapshot_mode}")

    hdm_snaps = None
    if not args.ecsw_only:
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
            snap_time_offset=ecsw_snap_time_offset,
            snapshot_percent=ecsw_snapshot_percent,
            snapshot_random_seed=ecsw_snapshot_random_seed,
            snapshot_mode=ecsw_snapshot_mode,
            ensure_mu_coverage=ecsw_ensure_mu_coverage,
            ecsw_weights_dir=ecsw_weights_dir,
            ecsw_tag=os.path.splitext(model_name)[0],
        )
        ecsw_setup_elapsed = time.time() - t_ecsw0
        if not os.path.abspath(weights_path).startswith(os.path.abspath(ecsw_weights_dir) + os.sep):
            raise RuntimeError(
                f"ECSW weights must be under '{ecsw_weights_dir}', got: {weights_path}"
            )

        print(f"[Case2] ECSW weights: {weights_path} ({weights_source})")
        print(f"[Case2] ECSW training trajectories used = {ecsw_num_training_mu}")
        print(f"[Case2] N_e = {n_ecsw_elements}")
        print(f"[Case2] ECSW residual = {ecsw_residual}")
        print(f"[Case2] ecsw_setup_elapsed = {ecsw_setup_elapsed:.3e} s")
        if args.ecsw_only:
            print("[Case2] ECSW-only mode complete; online solve skipped.")
            return

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

    else:
        if args.ecsw_only:
            raise ValueError("--ecsw-only requires --backend hprom with ECSW enabled.")
        t_solve0 = time.time()
        rom_snaps, red_coords, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
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
            return_red_coords=True,
        )
        online_solve_elapsed = time.time() - t_solve0

    elapsed = online_solve_elapsed

    num_its, jac_time, res_time, ls_time = rom_times

    rel_err = 100.0 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)

    backend_tag = "hprom" if effective_backend == "hprom" else "prom"
    tag = _safe_mu_tag(mu_test)
    extra_part = f"_{run_tag_extra}" if run_tag_extra else ""
    run_tag = f"case2_{backend_tag}_ann{extra_part}_{tag}_n{n_p}_ntot{total_modes}"

    full_qN = _assemble_case2_full_qN(
        red_coords=red_coords,
        ann_model=ann_model,
        mu=mu_test,
        dt=DT,
        device=device,
    )
    out_qn = os.path.join(output_root, f"{run_tag}_qN.npy")
    np.save(out_qn, full_qN)

    out_npy = os.path.join(output_root, f"{run_tag}_snaps.npy")
    if save_rom_snaps:
        np.save(out_npy, rom_snaps)
    else:
        out_npy = "not_saved"

    plot_steps = list(range(0, NUM_STEPS + 1, 100))
    if NUM_STEPS not in plot_steps:
        plot_steps.append(NUM_STEPS)

    out_png = os.path.join(output_root, f"{run_tag}_hdm_vs_rom.png")
    if make_plot:
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
            label=(
                "HPROM-ANN Case 2 trimmed"
                if (effective_backend == "hprom" and drop_first_secondary > 0)
                else "HPROM-ANN Case 2"
                if effective_backend == "hprom"
                else "PROM-ANN Case 2 trimmed"
                if drop_first_secondary > 0
                else "PROM-ANN Case 2"
            ),
            fig_ax=(fig, ax1, ax2),
            color="blue",
            linewidth=1.8,
            linestyle="solid",
        )
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
    else:
        out_png = "not_requested"

    summary_txt = os.path.join(output_root, f"{run_tag}_summary.txt")
    write_kv_txt(
        summary_txt,
        [
            ("mu_test", mu_test),
            ("device", device),
            ("model_name", model_name),
            ("model_path", model_path),
            ("model_output_kind", model_output_kind),
            ("checkpoint_output_dim", checkpoint_output_dim),
            ("run_tag_extra", run_tag_extra if run_tag_extra else "none"),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path if os.path.exists(uref_path) else "zeros"),
            ("checkpoint_primary_modes", n_p_checkpoint),
            ("checkpoint_secondary_modes", n_s_checkpoint),
            ("target_primary_modes", n_p),
            ("target_secondary_modes", n_s),
            ("drop_first_secondary", drop_first_secondary),
            ("trimmed_from_checkpoint", bool(drop_first_secondary > 0)),
            ("solve_backend_requested", solve_backend),
            ("solve_backend_effective", effective_backend),
            ("use_ecsw", use_ecsw),
            ("rebuild_ecsw_weights", rebuild_ecsw_weights),
            ("ecsw_num_training_mu", ecsw_num_training_mu),
            ("ecsw_snap_time_offset", ecsw_snap_time_offset),
            ("ecsw_predecessor_lag_steps", 1),
            ("ecsw_snapshot_percent", ecsw_snapshot_percent),
            ("ecsw_snapshot_mode", ecsw_snapshot_mode),
            ("ecsw_snapshot_random_seed", ecsw_snapshot_random_seed),
            ("ecsw_ensure_mu_coverage", bool(ecsw_ensure_mu_coverage)),
            ("ecsw_weights_path", weights_path if effective_backend == "hprom" else "N/A"),
            ("ecsw_weights_dir", ecsw_weights_dir),
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
            (
                "qN_source",
                "solver_primary_plus_master_ann_secondary"
                if model_output_kind == "q_tot"
                else "solver_primary_plus_trimmed_ann_secondary"
                if drop_first_secondary > 0
                else "solver_primary_plus_ann_secondary",
            ),
            ("qN_output", out_qn),
            ("save_rom_snaps", bool(save_rom_snaps)),
            ("output_root", output_root),
            ("snaps_output", out_npy),
            ("plot_output", out_png),
        ],
    )

    print(f"[Case2] ecsw_setup_elapsed = {ecsw_setup_elapsed:.3e} s")
    print(f"[Case2] online_solve_elapsed = {online_solve_elapsed:.3e} s")
    print(f"[Case2] elapsed = {elapsed:.3e} s")
    print(f"[Case2] its={num_its} | jac={jac_time:.3e} | res={res_time:.3e} | ls={ls_time:.3e}")
    print(f"[Case2] relative error vs HDM: {rel_err:.2f}%")
    print(f"[Case2] saved qN:    {out_qn}")
    print(f"[Case2] saved snaps: {out_npy}")
    print(f"[Case2] saved plot:  {out_png}")
    print(f"[Case2] summary:     {summary_txt}")


if __name__ == "__main__":
    main()
