#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global POD-ANN HPROM (ECSW-LSPG) for the 2D inviscid Burgers problem
using the modern `burgers/` modules and save outputs consistently in `Results`.
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

from burgers.core import (
    load_or_compute_snaps,
    plot_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
)
from burgers.pod_ann_manifold import (
    compute_ECSW_training_matrix_2D_pod_ann,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_ecsw,
)
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import (
    RandomizedSingularValueDecomposition,
)
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS

try:
    from torch.func import jacfwd as torch_jacfwd
except Exception:
    import functorch

    torch_jacfwd = functorch.jacfwd


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


class ANNDecoderWrapper(nn.Module):
    """
    Make ANN output shape compatible with manifold code expecting vector output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, y):
        if y.dim() == 1:
            return self.model(y.unsqueeze(0)).reshape(-1)
        out = self.model(y)
        if out.dim() == 2 and out.shape[0] == 1:
            return out.reshape(-1)
        return out


class ANNMuCompatWrapper(nn.Module):
    """
    Accept either q_p or [q_p, mu] and always evaluate ANN on q_p.
    Needed because ECSW ANN manifold currently calls ann_model([q_p, mu]).
    """

    def __init__(self, ann_model, n_primary):
        super().__init__()
        self.ann_model = ann_model
        self.n_primary = int(n_primary)

    def forward(self, x):
        x = x.reshape(-1)
        if x.numel() < self.n_primary:
            raise ValueError(
                f"ANN input size {x.numel()} smaller than n_primary={self.n_primary}."
            )
        return self.ann_model(x[:self.n_primary])


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
    model_path = os.path.join(model_dir, "case1_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing POD-ANN model checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing key: 'state_dict'.")
    if "n_p" not in checkpoint or "n_s" not in checkpoint:
        raise KeyError("Checkpoint missing keys: 'n_p' and/or 'n_s'.")

    n_p = int(checkpoint["n_p"])
    n_s = int(checkpoint["n_s"])
    hidden_dims = tuple(int(v) for v in checkpoint.get("hidden_dims", (32, 64, 128, 256, 256)))

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
            "Could not find basis.npy for POD-ANN model. "
            f"Checked: {filtered_candidates}"
        )

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}.")
    if basis.shape[1] < n_p + n_s:
        raise ValueError(
            "basis has insufficient columns for checkpoint dimensions: "
            f"basis columns={basis.shape[1]}, required={n_p + n_s}."
        )

    u_p = basis[:, :n_p]
    u_s = basis[:, n_p:n_p + n_s]

    core_model = PODANNModel(
        x_mean=np.zeros(n_p, dtype=np.float32),
        x_std=np.ones(n_p, dtype=np.float32),
        y_mean=np.zeros(n_s, dtype=np.float32),
        y_std=np.ones(n_s, dtype=np.float32),
        hidden_dims=hidden_dims,
    )
    core_model.load_state_dict(checkpoint["state_dict"])
    core_model.eval()

    ann_model = ANNDecoderWrapper(core_model)
    ann_model.eval()

    checkpoint_u_ref = checkpoint.get("u_ref", None)
    if checkpoint_u_ref is not None:
        checkpoint_u_ref = np.asarray(checkpoint_u_ref, dtype=np.float64).reshape(-1)

    return {
        "model_path": model_path,
        "basis_path": basis_path,
        "checkpoint": checkpoint,
        "checkpoint_u_ref": checkpoint_u_ref,
        "n_p": n_p,
        "n_s": n_s,
        "hidden_dims": hidden_dims,
        "U_p": u_p,
        "U_s": u_s,
        "ann_model": ann_model,
    }


def _build_ann_decode_helpers(u_p, u_s, u_ref, ann_model):
    u_p_t = torch.tensor(np.asarray(u_p), dtype=torch.float32)
    u_s_t = torch.tensor(np.asarray(u_s), dtype=torch.float32)
    u_ref_t = torch.tensor(np.asarray(u_ref).reshape(-1), dtype=torch.float32)

    def ann_eval(y):
        return ann_model(y)

    ann_jac = torch_jacfwd(ann_eval)

    def decode(y):
        return u_ref_t + u_p_t @ y + u_s_t @ ann_eval(y)

    def jacfwdfunc(y):
        return u_p_t + u_s_t @ ann_jac(y)

    return decode, jacfwdfunc


def _decode_full_snapshot(q_p, u_p, u_s, u_ref, ann_model):
    qp = np.asarray(q_p, dtype=np.float64).reshape(-1)
    with torch.no_grad():
        qpt = torch.tensor(qp, dtype=torch.float32)
        qs = ann_model(qpt).detach().cpu().numpy().reshape(-1)
    return u_ref + u_p @ qp + u_s @ qs


def main(
    mu1=4.56,
    mu2=0.019,
    model_dir=os.path.join("POD-ANN", "pod_ann_model"),
    basis_file=None,
    compute_ecsw=True,
    weights_file=None,
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    uref_file=None,
    snap_sample_factor=10,
    snap_time_offset=3,
    mu_samples=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    solver_threads=1,
):
    if mu_samples is None:
        mu_samples = [[4.25, 0.0225]]
    mu_samples = [list(mu) for mu in mu_samples]

    if snap_sample_factor < 1:
        raise ValueError("snap_sample_factor must be >= 1.")
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")

    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")

    if weights_file is None:
        weights_file = os.path.join(model_dir, "ecsw_weights_ann.npy")
    legacy_weights_files = [
        os.path.join(model_dir, "ecm_weights_ann_global.npy"),
        os.path.join(model_dir, "ecm_weights_rnm_global.npy"),
    ]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

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
        expected_size=model["U_p"].shape[0],
    )

    if w0.size != model["U_p"].shape[0]:
        raise ValueError(
            f"Initial condition size mismatch: W0 has {w0.size}, model has {model['U_p'].shape[0]}."
        )

    ann_ecsw = ANNMuCompatWrapper(model["ann_model"], model["n_p"])

    print(f"[HPROM-ANN] Loaded model checkpoint: {model['model_path']}")
    print(f"[HPROM-ANN] Loaded basis: {model['basis_path']}")
    print(
        f"[HPROM-ANN] U_p shape={model['U_p'].shape}, U_s shape={model['U_s'].shape}, "
        f"hidden_dims={model['hidden_dims']}"
    )
    print(
        f"[HPROM-ANN] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )
    if solver_threads is not None and int(solver_threads) > 0:
        print(f"[HPROM-ANN] Limiting solver threads to {int(solver_threads)} for stability/performance.")

    c_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None
    weights_source = None

    with _solver_thread_limit(solver_threads):
        if compute_ecsw:
            decode_train, jacfwdfunc_train = _build_ann_decode_helpers(
                u_p=model["U_p"],
                u_s=model["U_s"],
                u_ref=u_ref,
                ann_model=model["ann_model"],
            )

            clist = []
            t0 = time.time()

            for mu_train in mu_samples:
                mu_snaps = load_or_compute_snaps(
                    mu_train,
                    grid_x,
                    grid_y,
                    w0,
                    dt,
                    num_steps,
                    snap_folder=snap_folder,
                )

                stop_col = num_steps
                snaps_now = mu_snaps[:, snap_time_offset:stop_col:snap_sample_factor]
                snaps_prev = mu_snaps[:, 0:stop_col - snap_time_offset:snap_sample_factor]

                if snaps_now.shape[1] != snaps_prev.shape[1]:
                    raise RuntimeError(
                        "ECSW snapshot alignment failed: "
                        f"snaps_now has {snaps_now.shape[1]} columns, "
                        f"snaps_prev has {snaps_prev.shape[1]} columns."
                    )
                if snaps_now.shape[1] == 0:
                    raise RuntimeError(
                        "ECSW training produced zero columns. "
                        "Adjust snap_time_offset or snap_sample_factor."
                    )

                print(f"[HPROM-ANN] Generating ECSW training block for mu={mu_train}")
                ci = compute_ECSW_training_matrix_2D_pod_ann(
                    snaps_now,
                    snaps_prev,
                    model["U_p"],
                    decode_train,
                    jacfwdfunc_train,
                    inviscid_burgers_res2D,
                    inviscid_burgers_exact_jac2D,
                    grid_x,
                    grid_y,
                    dt,
                    mu_train,
                    u_ref=u_ref,
                )
                clist.append(ci)

            c = np.vstack(clist)
            c_shape = c.shape
            print(f"[HPROM-ANN] Stacked ECSW training matrix C shape: {c_shape}")

            c_ecm = np.ascontiguousarray(c, dtype=np.float64)
            b = np.ascontiguousarray(c_ecm.sum(axis=1), dtype=np.float64)

            rsvd = RandomizedSingularValueDecomposition()
            u, _, _, _ = rsvd.Calculate(c_ecm.T, 1e-8)

            selector = EmpiricalCubatureMethod()
            selector.SetUp(
                u,
                InitialCandidatesSet=None,
                constrain_sum_of_weights=True,
                constrain_conditions=False,
            )
            selector.Run()

            num_cells = (grid_x.size - 1) * (grid_y.size - 1)
            weights = np.zeros(num_cells, dtype=np.float64)
            weights[selector.z] = selector.w

            elapsed_ecsw = time.time() - t0
            denom = np.linalg.norm(b)
            if denom > 0.0:
                ecsw_residual = float(np.linalg.norm(c_ecm @ weights - b) / denom)
            else:
                ecsw_residual = np.nan

            np.save(weights_file, weights)
            weights_source = "computed"

            print(f"[HPROM-ANN] ECSW weights saved to: {weights_file}")
            print(f"[HPROM-ANN] ECSW solve time: {elapsed_ecsw:.3e} seconds")
            print(f"[HPROM-ANN] ECSW residual: {ecsw_residual:.3e}")

            reduced_mesh_plot_path = os.path.join(results_dir, "hprom_ann_reduced_mesh.png")
            plt.figure(figsize=(7, 6))
            plt.spy(weights.reshape((num_cells_y, num_cells_x)))
            plt.xlabel(r"$x$ cell index")
            plt.ylabel(r"$y$ cell index")
            plt.title("HPROM-ANN Reduced Mesh (ECSW)")
            plt.tight_layout()
            plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[HPROM-ANN] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
        else:
            if os.path.exists(weights_file):
                weights = np.asarray(np.load(weights_file, allow_pickle=False), dtype=np.float64)
                weights_source = "loaded"
                print(f"[HPROM-ANN] Loaded ECSW weights from: {weights_file}")
            else:
                loaded = False
                for legacy_file in legacy_weights_files:
                    if os.path.exists(legacy_file):
                        weights = np.asarray(np.load(legacy_file, allow_pickle=False), dtype=np.float64)
                        weights_source = f"loaded_legacy:{os.path.basename(legacy_file)}"
                        print(f"[HPROM-ANN] Loaded legacy ECSW weights from: {legacy_file}")
                        loaded = True
                        break
                if not loaded:
                    raise FileNotFoundError(
                        f"ECSW weights file not found: {weights_file}. "
                        "Run with compute_ecsw=True first."
                    )

        expected_num_cells = (grid_x.size - 1) * (grid_y.size - 1)
        if weights.size != expected_num_cells:
            raise ValueError(
                f"ECSW weights size mismatch: got {weights.size}, expected {expected_num_cells}."
            )

        n_ecsw_elements = int(np.sum(weights > 0.0))
        print(f"[HPROM-ANN] N_e (nonzero ECSW weights): {n_ecsw_elements}")

        t0 = time.time()
        red_coords, hprom_stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_ecsw(
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu_rom,
            ann_model=ann_ecsw,
            ref=None,
            basis=model["U_p"],
            basis2=model["U_s"],
            weights=weights,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )
        elapsed_hprom = time.time() - t0

    num_its, jac_time, res_time, ls_time = hprom_stats

    print(f"[HPROM-ANN] Elapsed HPROM time: {elapsed_hprom:.3e} seconds")
    print(f"[HPROM-ANN] Gauss-Newton iterations: {num_its}")
    print(
        "[HPROM-ANN] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )

    rom_snaps = np.zeros((w0.size, red_coords.shape[1]), dtype=np.float64)
    for k in range(red_coords.shape[1]):
        rom_snaps[:, k] = _decode_full_snapshot(
            q_p=red_coords[:, k],
            u_p=model["U_p"],
            u_s=model["U_s"],
            u_ref=u_ref,
            ann_model=model["ann_model"],
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
    print(f"[HPROM-ANN] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"hprom_ann_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[HPROM-ANN] HPROM snapshots saved to: {rom_path}")

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
        label="HPROM-ANN",
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
        f"hprom_ann_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[HPROM-ANN] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[HPROM-ANN] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"hprom_ann_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("compute_ecsw", compute_ecsw),
                    ("weights_file", weights_file),
                    ("weights_source", weights_source),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("snap_sample_factor", snap_sample_factor),
                    ("snap_time_offset", snap_time_offset),
                    ("mu_samples", mu_samples),
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
                "ann_model",
                [
                    ("model_pt", model["model_path"]),
                    ("basis_path", model["basis_path"]),
                    ("U_p_shape", model["U_p"].shape),
                    ("U_s_shape", model["U_s"].shape),
                    ("n_p", model["n_p"]),
                    ("n_s", model["n_s"]),
                    ("hidden_dims", model["hidden_dims"]),
                    ("checkpoint_seed", model["checkpoint"].get("seed", None)),
                ],
            ),
            (
                "ecsw",
                [
                    ("num_nonzero_weights", n_ecsw_elements),
                    ("weights_sum", float(np.sum(weights))),
                    ("ecsw_time_seconds", elapsed_ecsw),
                    ("ecsw_residual", ecsw_residual),
                    ("training_matrix_shape", c_shape),
                ],
            ),
            (
                "hprom_timing",
                [
                    ("total_hprom_time_seconds", elapsed_hprom),
                    ("avg_hprom_time_per_step_seconds", elapsed_hprom / max(1, num_steps)),
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
                    ("hprom_snapshots_npy", rom_path),
                    ("comparison_plot_png", fig_path),
                    ("ecsw_weights_npy", weights_file),
                    ("reduced_mesh_plot_png", reduced_mesh_plot_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )
    print(f"[HPROM-ANN] Text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)
