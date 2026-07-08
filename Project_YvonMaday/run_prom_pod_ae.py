#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run intrusive PROM-POD-AE ROM with selectable backend.

Manifold:
    u(t;mu) = u_ref + V_tot qN(z(t))
where qN(z) is produced by the POD-AE decoder.

Backends:
- solve_backend='prom': full LSPG solve in latent coordinates
- solve_backend='hprom' and use_ecsw=True: ECSW hyper-reduced solve
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, GRID_X, GRID_Y, MU1_RANGE, MU2_RANGE, NUM_STEPS, SAMPLES_PER_MU, W0
from burgers.core import (
    get_snapshot_params,
    inviscid_burgers_exact_jac2D,
    inviscid_burgers_res2D,
    load_or_compute_snaps,
    plot_snaps,
)
from burgers.ecsw_utils import build_ecsw_snapshot_plan
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.pod_dl_manifold import (
    compute_ECSW_training_matrix_2D_pod_dl,
    inviscid_burgers_implicit2D_LSPG_pod_dl_2D,
    inviscid_burgers_implicit2D_LSPG_pod_dl_2D_ecsw,
)
from burgers.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
try:
    from pod_ae_common import (
        PROMPODAEAutoencoder,
        infer_scaling_from_state_dict,
        resolve_activation_from_checkpoint,
    )
except ModuleNotFoundError:
    from .pod_ae_common import (
        PROMPODAEAutoencoder,
        infer_scaling_from_state_dict,
        resolve_activation_from_checkpoint,
    )
try:
    from project_layout import (
        RUNS_ECSW_DIR,
        RUNS_POD_AE_DIR,
        ensure_layout_dirs,
        resolve_stage1_artifact,
        resolve_stage3_model,
        write_kv_txt,
    )
except ModuleNotFoundError:
    from .project_layout import (
        RUNS_ECSW_DIR,
        RUNS_POD_AE_DIR,
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


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _localize_project_path(path_like):
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


def _decode_snapshot_from_latent(z, pod_ae_model, basis_q, u_ref, device):
    z_t = torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float32, device=device).reshape(-1)
    with torch.no_grad():
        qn = pod_ae_model.decode_from_latent(z_t).detach().cpu().numpy().reshape(-1)
    return u_ref + basis_q @ qn


def _reconstruct_full_snaps_from_latent(latent_coords, pod_ae_model, basis_q, u_ref, device):
    latent_coords = np.asarray(latent_coords, dtype=np.float64)
    n_state = int(u_ref.size)
    snaps = np.zeros((n_state, latent_coords.shape[1]), dtype=np.float64)
    for k in range(latent_coords.shape[1]):
        snaps[:, k] = _decode_snapshot_from_latent(
            z=latent_coords[:, k],
            pod_ae_model=pod_ae_model,
            basis_q=basis_q,
            u_ref=u_ref,
            device=device,
        )
    return snaps


def _decode_qn_from_latent_trajectory(latent_coords, pod_ae_model, device):
    latent_coords = np.asarray(latent_coords, dtype=np.float64)
    q_cols = []
    with torch.no_grad():
        for k in range(latent_coords.shape[1]):
            z_t = torch.tensor(
                latent_coords[:, k],
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
            q_cols.append(
                pod_ae_model.decode_from_latent(z_t)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float64)
            )
    return np.column_stack(q_cols)


def _load_pod_ae_checkpoint(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint '{model_path}' missing key 'state_dict'.")

    state_dict = ckpt["state_dict"]
    q_dim = int(ckpt["q_dim"])
    latent_dim = int(ckpt["latent_dim"])
    hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", (192, 96, 48)))
    scaling = infer_scaling_from_state_dict(state_dict, fallback=ckpt.get("scaling", None))
    activation = resolve_activation_from_checkpoint(ckpt, scaling=scaling)

    model = PROMPODAEAutoencoder(
        q_dim=q_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        scaling=scaling,
        activation=activation,
        q_stats=None,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, q_dim, latent_dim, hidden_dims, scaling, activation, ckpt


def _load_basis_and_reference(ckpt, q_dim):
    basis_candidates = []
    if ckpt.get("basis_file", None) is not None:
        basis_candidates.append(_localize_project_path(ckpt["basis_file"]))
    basis_candidates.append(resolve_stage1_artifact("basis.npy"))

    basis_path = None
    for p in basis_candidates:
        if os.path.exists(p):
            basis_path = p
            break
    if basis_path is None:
        raise FileNotFoundError(f"Could not find basis file. Checked: {basis_candidates}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got {basis.shape}")
    if basis.shape[1] < q_dim:
        raise ValueError(f"basis has {basis.shape[1]} columns, but q_dim={q_dim} is required.")
    basis_q = basis[:, :q_dim]

    uref_candidates = []
    if ckpt.get("u_ref_file", None) is not None:
        uref_candidates.append(_localize_project_path(ckpt["u_ref_file"]))
    uref_candidates.append(resolve_stage1_artifact("u_ref.npy"))

    u_ref = None
    uref_path = None
    for p in uref_candidates:
        if os.path.exists(p):
            u_ref = np.asarray(np.load(p, allow_pickle=False), dtype=np.float64).reshape(-1)
            uref_path = p
            break
    if u_ref is None:
        u_ref = np.zeros(basis_q.shape[0], dtype=np.float64)
        uref_path = "zeros"

    if u_ref.size != basis_q.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {basis_q.shape[0]} from basis rows."
        )

    return basis_q, u_ref, basis_path, uref_path


def _load_or_build_pod_ae_ecsw_weights(
    q_dim,
    basis_q,
    u_ref,
    pod_ae_model,
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
    ensure_mu_coverage=True,
    weights_dir=None,
):
    expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)
    if weights_dir is None:
        weights_dir = RUNS_ECSW_DIR
    weights_dir = os.path.abspath(os.path.expanduser(str(weights_dir)))
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(
        weights_dir,
        f"ecsw_weights_pod_ae_ntot{q_dim}.npy",
    )

    if (not rebuild_weights) and os.path.exists(weights_path):
        weights = np.asarray(np.load(weights_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if weights.size != expected_num_cells:
            raise ValueError(
                f"ECSW weights size mismatch at '{weights_path}': got {weights.size}, expected {expected_num_cells}."
            )
        return weights, weights_path, "loaded_local", np.nan, int(np.sum(weights > 0.0))

    snapshot_percent = float(snapshot_percent)
    if (not np.isfinite(snapshot_percent)) or (snapshot_percent <= 0.0):
        raise ValueError("snapshot_percent must be a finite value > 0.")

    ecsw_plan = build_ecsw_snapshot_plan(
        num_steps=num_steps,
        snap_time_offset=snap_time_offset,
        num_mu=len(mu_samples),
        mode="global_param_time_stratified",
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
        prev_cols = now_cols - snap_time_offset
        snaps_now = mu_snaps[:, now_cols]
        snaps_prev = mu_snaps[:, prev_cols]

        if snaps_now.shape[1] != snaps_prev.shape[1]:
            raise RuntimeError(
                "ECSW snapshot alignment failed: "
                f"snaps_now has {snaps_now.shape[1]} columns, snaps_prev has {snaps_prev.shape[1]} columns."
            )
        if snaps_now.shape[1] == 0:
            continue

        ci = compute_ECSW_training_matrix_2D_pod_dl(
            snaps=snaps_now,
            prev_snaps=snaps_prev,
            basis=basis_q,
            pod_dl_model=pod_ae_model,
            res=inviscid_burgers_res2D,
            jac=inviscid_burgers_exact_jac2D,
            grid_x=grid_x,
            grid_y=grid_y,
            dt=dt,
            mu=mu,
            u_ref=u_ref,
        )
        clist.append(ci)

    if len(clist) == 0:
        raise RuntimeError(
            "ECSW training produced zero columns for all mu samples. "
            "Increase ecsw_snapshot_percent or adjust snap_time_offset."
        )

    c = np.vstack(clist)
    c_ecm = np.ascontiguousarray(c, dtype=np.float64)
    b = np.ascontiguousarray(c_ecm.sum(axis=1), dtype=np.float64)

    rsvd = RandomizedSingularValueDecomposition(USE_RANDOMIZATION=False)
    u, _, _, _ = rsvd.Calculate(c_ecm.T, 1e-8)

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
    rel_res = float(np.linalg.norm(c_ecm @ weights - b) / denom) if denom > 0.0 else np.nan
    n_ecsw = int(np.sum(weights > 0.0))
    return weights, weights_path, "computed", rel_res, n_ecsw


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run PROM-POD-AE intrusive ROM with PROM or HPROM backend."
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
    parser.add_argument("--ecsw-only", action="store_true", help="Only build/load ECSW weights and exit before online solve.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional output folder for run files.")
    parser.add_argument("--ecsw-weights-dir", type=str, default=None, help="Optional folder for POD-AE ECSW weights.")
    parser.add_argument("--ecsw-num-training-mu", type=int, default=9)
    parser.add_argument("--ecsw-snap-time-offset", type=int, default=3)
    parser.add_argument("--ecsw-snapshot-percent", type=float, default=2.0)
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
        default="prom_pod_ae_model.pt",
        help="Checkpoint filename under Results/Stage3/models (used if --model-path is not set).",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit checkpoint path.")
    args = parser.parse_args(argv)

    mu_test = [float(args.mu1), float(args.mu2)]
    solve_backend = str(args.backend).strip().lower()
    use_ecsw = not bool(args.no_ecsw)
    rebuild_ecsw_weights = bool(args.rebuild_ecsw)
    ecsw_only = bool(args.ecsw_only)
    ecsw_snap_time_offset = int(args.ecsw_snap_time_offset)
    ecsw_snapshot_percent = float(args.ecsw_snapshot_percent)
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
    output_root = os.path.abspath(os.path.expanduser(args.output_root)) if args.output_root else RUNS_POD_AE_DIR
    ecsw_weights_dir = os.path.abspath(os.path.expanduser(args.ecsw_weights_dir)) if args.ecsw_weights_dir else RUNS_ECSW_DIR

    device = str(args.device).strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("[POD-AE] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    set_latex_plot_style()
    ensure_layout_dirs()
    os.makedirs(output_root, exist_ok=True)

    if solve_backend not in ("prom", "hprom"):
        raise ValueError("solve_backend must be 'prom' or 'hprom'.")

    effective_backend = solve_backend
    if solve_backend == "hprom" and not use_ecsw:
        print("[POD-AE] solve_backend='hprom' with use_ecsw=False -> falling back to PROM solve.")
        effective_backend = "prom"
    if solve_backend == "prom" and use_ecsw:
        print("[POD-AE] use_ecsw=True ignored because solve_backend='prom'.")

    if model_path_override is None:
        if len(model_name) == 0:
            raise ValueError("--model-name cannot be empty.")
        if not model_name.endswith(".pt"):
            model_name = f"{model_name}.pt"
        model_path = resolve_stage3_model(model_name)
    else:
        model_path = os.path.abspath(model_path_override)
        model_name = os.path.basename(model_path)

    pod_ae_model, q_dim, latent_dim, hidden_dims, scaling, activation, ckpt = _load_pod_ae_checkpoint(
        model_path=model_path,
        device=device,
    )
    basis_q, u_ref, basis_path, uref_path = _load_basis_and_reference(ckpt=ckpt, q_dim=q_dim)

    w0 = np.asarray(W0, dtype=np.float64).reshape(-1).copy()
    if w0.size != basis_q.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {basis_q.shape[0]} from basis."
        )

    snap_folder = os.path.join(PROJECT_ROOT, "Results", "param_snaps")
    os.makedirs(snap_folder, exist_ok=True)

    print(f"[POD-AE] device = {device}")
    print(f"[POD-AE] checkpoint = {model_path}")
    print(f"[POD-AE] q_dim = {q_dim} | latent_dim = {latent_dim}")
    print(f"[POD-AE] hidden_dims = {hidden_dims}")
    print(f"[POD-AE] scaling = {scaling} | activation = {activation}")
    print(f"[POD-AE] basis = {basis_path}")
    print(f"[POD-AE] u_ref = {uref_path}")
    print(f"[POD-AE] solve_backend(requested) = {solve_backend}")
    print(f"[POD-AE] solve_backend(effective) = {effective_backend}")
    print(f"[POD-AE] use_ecsw = {use_ecsw}")

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
    weights_path = "N/A"

    if effective_backend == "hprom":
        mu_train_candidates = get_snapshot_params(
            mu1_range=MU1_RANGE,
            mu2_range=MU2_RANGE,
            samples_per_mu=SAMPLES_PER_MU,
        )
        ecsw_num_training_mu = max(1, min(int(ecsw_num_training_mu), len(mu_train_candidates)))
        mu_train_list = mu_train_candidates[:ecsw_num_training_mu]

        t_ecsw0 = time.time()
        weights, weights_path, weights_source, ecsw_residual, n_ecsw_elements = _load_or_build_pod_ae_ecsw_weights(
            q_dim=q_dim,
            basis_q=basis_q,
            u_ref=u_ref,
            pod_ae_model=pod_ae_model,
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
            ensure_mu_coverage=ecsw_ensure_mu_coverage,
            weights_dir=ecsw_weights_dir,
        )
        ecsw_setup_elapsed = time.time() - t_ecsw0

        if ecsw_only:
            print(f"[POD-AE] ECSW weights: {weights_path} ({weights_source})")
            print(f"[POD-AE] ECSW training trajectories used = {ecsw_num_training_mu}")
            print(f"[POD-AE] N_e = {n_ecsw_elements}")
            print(f"[POD-AE] ECSW residual = {ecsw_residual}")
            print(f"[POD-AE] ecsw_setup_elapsed = {ecsw_setup_elapsed:.3e} s")
            return

        t_solve0 = time.time()
        latent_coords, rom_times = inviscid_burgers_implicit2D_LSPG_pod_dl_2D_ecsw(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu_test,
            basis=basis_q,
            pod_dl_model=pod_ae_model,
            weights=weights,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        online_solve_elapsed = time.time() - t_solve0

        rom_snaps = _reconstruct_full_snaps_from_latent(
            latent_coords=latent_coords,
            pod_ae_model=pod_ae_model,
            basis_q=basis_q,
            u_ref=u_ref,
            device=device,
        )

        print(f"[POD-AE] ECSW weights: {weights_path} ({weights_source})")
        print(f"[POD-AE] ECSW training trajectories used = {ecsw_num_training_mu}")
        print(f"[POD-AE] N_e = {n_ecsw_elements}")
        print(f"[POD-AE] ECSW residual = {ecsw_residual}")
    else:
        t_solve0 = time.time()
        rom_snaps, latent_coords, rom_times = inviscid_burgers_implicit2D_LSPG_pod_dl_2D(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu_test,
            basis=basis_q,
            pod_dl_model=pod_ae_model,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        online_solve_elapsed = time.time() - t_solve0

    num_its, jac_time, res_time, ls_time = rom_times
    rel_err = 100.0 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)

    backend_tag = "hprom" if effective_backend == "hprom" else "prom"
    tag = _safe_mu_tag(mu_test)
    run_tag = f"podae_{backend_tag}_{tag}_ntot{q_dim}_nz{latent_dim}"

    qn = _decode_qn_from_latent_trajectory(
        latent_coords=latent_coords,
        pod_ae_model=pod_ae_model,
        device=device,
    )

    out_snaps = os.path.join(output_root, f"{run_tag}_snaps.npy")
    out_latent = os.path.join(output_root, f"{run_tag}_latent.npy")
    out_qn = os.path.join(output_root, f"{run_tag}_qN.npy")
    np.save(out_snaps, rom_snaps)
    np.save(out_latent, latent_coords)
    np.save(out_qn, qn)

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
        label="HPROM-POD-AE" if effective_backend == "hprom" else "PROM-POD-AE",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    out_plot = os.path.join(output_root, f"{run_tag}_hdm_vs_rom.png")
    plt.savefig(out_plot, dpi=200)
    plt.close(fig)

    summary_txt = os.path.join(output_root, f"{run_tag}_summary.txt")
    write_kv_txt(
        summary_txt,
        [
            ("mu_test", mu_test),
            ("device", device),
            ("model_name", model_name),
            ("model_path", model_path),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path),
            ("q_dim", q_dim),
            ("latent_dim", latent_dim),
            ("hidden_dims", hidden_dims),
            ("scaling", scaling),
            ("activation", activation),
            ("solve_backend_requested", solve_backend),
            ("solve_backend_effective", effective_backend),
            ("use_ecsw", use_ecsw),
            ("rebuild_ecsw_weights", rebuild_ecsw_weights),
            ("ecsw_num_training_mu", ecsw_num_training_mu),
            ("ecsw_snap_time_offset", ecsw_snap_time_offset),
            ("ecsw_snapshot_percent", ecsw_snapshot_percent),
            ("ecsw_snapshot_random_seed", ecsw_snapshot_random_seed),
            ("ecsw_ensure_mu_coverage", bool(ecsw_ensure_mu_coverage)),
            ("ecsw_weights_path", weights_path),
            ("ecsw_residual", ecsw_residual),
            ("n_ecsw_elements", n_ecsw_elements),
            ("ecsw_setup_elapsed_s", ecsw_setup_elapsed),
            ("online_solve_elapsed_s", online_solve_elapsed),
            ("elapsed_s", online_solve_elapsed),
            ("num_iterations", num_its),
            ("jac_time_s", jac_time),
            ("res_time_s", res_time),
            ("ls_time_s", ls_time),
            ("relative_error_percent", rel_err),
            ("snaps_output", out_snaps),
            ("latent_output", out_latent),
            ("qN_output", out_qn),
            ("plot_output", out_plot),
        ],
    )

    print(f"[POD-AE] ecsw_setup_elapsed = {ecsw_setup_elapsed:.3e} s")
    print(f"[POD-AE] online_solve_elapsed = {online_solve_elapsed:.3e} s")
    print(f"[POD-AE] its={num_its} | jac={jac_time:.3e} | res={res_time:.3e} | ls={ls_time:.3e}")
    print(f"[POD-AE] relative error vs HDM: {rel_err:.2f}%")
    print(f"[POD-AE] saved snaps:  {out_snaps}")
    print(f"[POD-AE] saved latent: {out_latent}")
    print(f"[POD-AE] saved qN:     {out_qn}")
    print(f"[POD-AE] saved plot:   {out_plot}")
    print(f"[POD-AE] summary:      {summary_txt}")


if __name__ == "__main__":
    main()
