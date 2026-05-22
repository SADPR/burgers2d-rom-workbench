#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global POD-GPLVM HPROM (ECSW-LSPG) for the 2D inviscid Burgers problem
using the modern `burgers/` modules and save outputs consistently in `Results`.
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from burgers.core import (
    load_or_compute_snaps,
    plot_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
    get_snapshot_params,
)
from burgers.pod_gplvm_manifold import (
    decode_gplvm,
    compute_ECSW_training_matrix_2D_gplvm,
    inviscid_burgers_implicit2D_LSPG_pod_gplvm_ecsw,
)
from burgers.ecsw_utils import build_ecsw_snapshot_plan
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import (
    RandomizedSingularValueDecomposition,
)
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    plt.rcParams.update(
        {
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
        }
    )


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
    model_dir,
    expected_size,
):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

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
            use_u_ref = any(os.path.exists(path) for path in filtered_candidates)
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

    raise FileNotFoundError(
        "u_ref is required by current settings but no candidate file exists. "
        f"Checked: {filtered_candidates}"
    )


def _load_model_artifacts(model_dir):
    model_npz_path = os.path.join(model_dir, "gplvm_model.npz")
    uq_path = os.path.join(model_dir, "U_q.npy")

    for path in (model_npz_path, uq_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing POD-GPLVM model artifact: {path}")

    model_npz = np.load(model_npz_path, allow_pickle=False)
    model = {k: model_npz[k] for k in model_npz.files}
    U_q = np.asarray(np.load(uq_path, allow_pickle=False), dtype=np.float64)

    if U_q.ndim != 2:
        raise ValueError(f"U_q must be 2D, got shape {U_q.shape}.")

    required = ("Z_train", "Q_train_raw", "y_mean", "y_std", "log_ell", "log_sf", "log_sn")
    for key in required:
        if key not in model:
            raise KeyError(f"Missing key '{key}' in {model_npz_path}")
    decoder_mode = (
        str(np.asarray(model["decoder_mode"]).reshape(()))
        if "decoder_mode" in model
        else "full"
    ).lower()
    if decoder_mode == "sparse_dtc":
        for key in ("Z_inducing", "beta_inducing"):
            if key not in model:
                raise KeyError(f"Missing sparse GPLVM key '{key}' in {model_npz_path}")
    elif "alpha" not in model:
        raise KeyError(f"Missing key 'alpha' in {model_npz_path}")

    return {
        "model_npz_path": model_npz_path,
        "uq_path": uq_path,
        "model": model,
        "U_q": U_q,
    }


def main(
    mu1=4.56,
    mu2=0.019,
    model_dir=os.path.join("POD-GPLVM", "pod_gplvm_model"),
    compute_ecsw=True,
    weights_file=None,
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    uref_file=None,
    snap_time_offset=3,
    mu_samples=None,
    ecsw_snapshot_percent=2.0,
    ecsw_random_seed=42,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-12,
    ic_inverse_method="bounded_trf",
    ic_inverse_n_starts=5,
    ic_inverse_bound_margin_rel=0.20,
    ic_inverse_bound_margin_abs=0.25,
    ic_inverse_prior_weight=1e-3,
    ic_inverse_loss="linear",
    ic_inverse_f_scale=1.0,
    ecsw_inverse_method="bounded_trf",
    ecsw_inverse_n_starts=3,
    ecsw_inverse_bound_margin_rel=0.20,
    ecsw_inverse_bound_margin_abs=0.25,
    ecsw_inverse_prior_weight=1e-3,
    ecsw_inverse_loss="linear",
    ecsw_inverse_f_scale=1.0,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    if mu_samples is None:
        mu_samples = get_snapshot_params()
    mu_samples = [list(mu) for mu in mu_samples]

    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")
    ecsw_snapshot_percent = float(ecsw_snapshot_percent)
    if not np.isfinite(ecsw_snapshot_percent) or ecsw_snapshot_percent <= 0.0:
        raise ValueError("ecsw_snapshot_percent must be a finite value > 0.")
    ecsw_snapshot_mode = "global_param_time_stratified"
    ecsw_total_snapshots = None
    ecsw_total_snapshots_percent = ecsw_snapshot_percent
    ecsw_ensure_mu_coverage = True

    results_dir = "Results"
    if snap_folder is None:
        snap_folder = os.path.join(results_dir, "param_snaps")

    if weights_file is None:
        weights_file = os.path.join(model_dir, "ecsw_weights_gplvm.npy")
    legacy_weights_file = os.path.join(model_dir, "ecm_weights_gplvm_global.npy")

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

    model_data = _load_model_artifacts(model_dir)
    model_use_u_ref, stage2_metadata_path = _load_stage2_use_u_ref(model_dir)
    use_u_ref, u_ref, u_ref_source = _resolve_u_ref(
        uref_mode=uref_mode,
        explicit_uref_file=uref_file,
        model_use_u_ref=model_use_u_ref,
        model_dir=model_dir,
        expected_size=model_data["U_q"].shape[0],
    )

    if w0.size != model_data["U_q"].shape[0]:
        raise ValueError(
            f"Initial condition size mismatch: W0 has {w0.size}, model has {model_data['U_q'].shape[0]}."
        )

    latent_dim = int(np.asarray(model_data["model"]["latent_dim"]).reshape(()))
    n_train_latent = int(np.asarray(model_data["model"]["n_train"]).reshape(()))
    decoder_mode = (
        str(np.asarray(model_data["model"]["decoder_mode"]).reshape(()))
        if "decoder_mode" in model_data["model"]
        else "full"
    )
    sparse_num_inducing = (
        int(np.asarray(model_data["model"]["sparse_num_inducing"]).reshape(()))
        if "sparse_num_inducing" in model_data["model"]
        else None
    )

    print(f"[HPROM-GPLVM] Loaded model from: {model_dir}")
    print(
        f"[HPROM-GPLVM] U_q shape={model_data['U_q'].shape}, "
        f"latent_dim={latent_dim}, n_train_latent={n_train_latent}"
    )
    print(
        f"[HPROM-GPLVM] decoder_mode={decoder_mode}, "
        f"sparse_num_inducing={sparse_num_inducing}"
    )
    print(
        f"[HPROM-GPLVM] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )
    print(f"[HPROM-GPLVM] Reduced linear solver: {linear_solver}")
    if str(linear_solver).strip().lower() == "normal_eq":
        print(f"[HPROM-GPLVM] normal_eq_reg: {float(normal_eq_reg):.3e}")

    c_shape = None
    elapsed_ecsw = None
    ecsw_residual = None
    reduced_mesh_plot_path = None
    weights_source = None
    ecsw_plan = None

    if compute_ecsw:
        clist = []
        t0 = time.time()
        ecsw_plan = build_ecsw_snapshot_plan(
            num_steps=num_steps,
            snap_time_offset=snap_time_offset,
            num_mu=len(mu_samples),
            mode=ecsw_snapshot_mode,
            total_snapshots=ecsw_total_snapshots,
            total_snapshots_percent=ecsw_total_snapshots_percent,
            random_seed=ecsw_random_seed,
            ensure_mu_coverage=ecsw_ensure_mu_coverage,
            mu_points=mu_samples,
        )
        print(
            "[HPROM-GPLVM] ECSW snapshot selection mode="
            f"{ecsw_plan['mode']}, selected {ecsw_plan['num_selected_total']} / "
            f"{ecsw_plan['num_candidates_total']} candidate pairs."
        )
        print(f"[HPROM-GPLVM] Selected snapshots per mu: {ecsw_plan['num_selected_per_mu']}")

        for imu, mu_train in enumerate(mu_samples):
            mu_snaps = load_or_compute_snaps(
                mu_train,
                grid_x,
                grid_y,
                w0,
                dt,
                num_steps,
                snap_folder=snap_folder,
            )

            now_cols = np.asarray(ecsw_plan["selected_now_cols_by_mu"][imu], dtype=int)
            prev_cols = now_cols - snap_time_offset
            snaps_now = mu_snaps[:, now_cols]
            snaps_prev = mu_snaps[:, prev_cols]

            if snaps_now.shape[1] == 0:
                continue

            print(f"[HPROM-GPLVM] Generating ECSW training block for mu={mu_train}")
            ci = compute_ECSW_training_matrix_2D_gplvm(
                snaps_now,
                snaps_prev,
                model_data["U_q"],
                model_data["model"],
                inviscid_burgers_res2D,
                inviscid_burgers_exact_jac2D,
                grid_x,
                grid_y,
                dt,
                mu_train,
                u_ref=u_ref,
                inverse_method=ecsw_inverse_method,
                inverse_n_starts=ecsw_inverse_n_starts,
                inverse_bound_margin_rel=ecsw_inverse_bound_margin_rel,
                inverse_bound_margin_abs=ecsw_inverse_bound_margin_abs,
                inverse_prior_weight=ecsw_inverse_prior_weight,
                inverse_loss=ecsw_inverse_loss,
                inverse_f_scale=ecsw_inverse_f_scale,
            )
            clist.append(ci)

        if not clist:
            raise RuntimeError(
                "ECSW training produced zero columns for all mu samples. "
                "Increase ecsw_snapshot_percent or adjust snap_time_offset."
            )

        c = np.vstack(clist)
        c_shape = c.shape
        print(f"[HPROM-GPLVM] Stacked ECSW training matrix C shape: {c_shape}")

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

        print(f"[HPROM-GPLVM] ECSW weights saved to: {weights_file}")
        print(f"[HPROM-GPLVM] ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"[HPROM-GPLVM] ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "hprom_gplvm_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("HPROM-GPLVM Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[HPROM-GPLVM] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if os.path.exists(weights_file):
            weights = np.asarray(np.load(weights_file, allow_pickle=False), dtype=np.float64)
            weights_source = "loaded"
            print(f"[HPROM-GPLVM] Loaded ECSW weights from: {weights_file}")
        elif os.path.exists(legacy_weights_file):
            weights = np.asarray(np.load(legacy_weights_file, allow_pickle=False), dtype=np.float64)
            weights_source = f"loaded_legacy:{os.path.basename(legacy_weights_file)}"
            print(f"[HPROM-GPLVM] Loaded legacy ECSW weights from: {legacy_weights_file}")
        else:
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
    print(f"[HPROM-GPLVM] N_e (nonzero ECSW weights): {n_ecsw_elements}")

    t0 = time.time()
    latent_hist, hprom_stats = inviscid_burgers_implicit2D_LSPG_pod_gplvm_ecsw(
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu=mu_rom,
        basis_q=model_data["U_q"],
        gplvm_model=model_data["model"],
        weights=weights,
        u_ref=u_ref,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        max_its_ic=max_its_ic,
        tol_ic=tol_ic,
        ic_inverse_method=ic_inverse_method,
        ic_inverse_n_starts=ic_inverse_n_starts,
        ic_inverse_bound_margin_rel=ic_inverse_bound_margin_rel,
        ic_inverse_bound_margin_abs=ic_inverse_bound_margin_abs,
        ic_inverse_prior_weight=ic_inverse_prior_weight,
        ic_inverse_loss=ic_inverse_loss,
        ic_inverse_f_scale=ic_inverse_f_scale,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )
    elapsed_hprom = time.time() - t0
    num_its, jac_time, res_time, ls_time = hprom_stats

    print(f"[HPROM-GPLVM] Elapsed HPROM time: {elapsed_hprom:.3e} seconds")
    print(f"[HPROM-GPLVM] Gauss-Newton iterations: {num_its}")
    print(
        "[HPROM-GPLVM] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )

    rom_snaps = np.zeros((w0.size, latent_hist.shape[1]), dtype=np.float64)
    for k in range(latent_hist.shape[1]):
        rom_snaps[:, k] = decode_gplvm(
            z=latent_hist[:, k],
            gplvm_model=model_data["model"],
            basis_q=model_data["U_q"],
            u_ref=u_ref,
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
    print(f"[HPROM-GPLVM] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"hprom_gplvm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    latent_path = os.path.join(
        results_dir,
        f"hprom_gplvm_latent_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    np.save(latent_path, latent_hist)
    print(f"[HPROM-GPLVM] HPROM snapshots saved to: {rom_path}")
    print(f"[HPROM-GPLVM] Latent trajectory saved to: {latent_path}")

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
        label="HPROM-GPLVM",
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
        f"hprom_gplvm_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[HPROM-GPLVM] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[HPROM-GPLVM] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"hprom_gplvm_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("compute_ecsw", compute_ecsw),
                    ("weights_file", weights_file),
                    ("weights_source", weights_source),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("snap_time_offset", snap_time_offset),
                    ("ecsw_sampling_policy", ecsw_snapshot_mode),
                    ("ecsw_snapshot_percent", ecsw_snapshot_percent),
                    ("ecsw_random_seed", ecsw_random_seed),
                    ("mu_samples", mu_samples),
                    ("relnorm_cutoff", relnorm_cutoff),
                    ("min_delta", min_delta),
                    ("max_its", max_its),
                    ("max_its_ic", max_its_ic),
                    ("tol_ic", tol_ic),
                    ("ic_inverse_method", ic_inverse_method),
                    ("ic_inverse_n_starts", ic_inverse_n_starts),
                    ("ic_inverse_bound_margin_rel", ic_inverse_bound_margin_rel),
                    ("ic_inverse_bound_margin_abs", ic_inverse_bound_margin_abs),
                    ("ic_inverse_prior_weight", ic_inverse_prior_weight),
                    ("ic_inverse_loss", ic_inverse_loss),
                    ("ic_inverse_f_scale", ic_inverse_f_scale),
                    ("ecsw_inverse_method", ecsw_inverse_method),
                    ("ecsw_inverse_n_starts", ecsw_inverse_n_starts),
                    ("ecsw_inverse_bound_margin_rel", ecsw_inverse_bound_margin_rel),
                    ("ecsw_inverse_bound_margin_abs", ecsw_inverse_bound_margin_abs),
                    ("ecsw_inverse_prior_weight", ecsw_inverse_prior_weight),
                    ("ecsw_inverse_loss", ecsw_inverse_loss),
                    ("ecsw_inverse_f_scale", ecsw_inverse_f_scale),
                    ("linear_solver", linear_solver),
                    (
                        "normal_eq_reg",
                        normal_eq_reg if str(linear_solver).strip().lower() == "normal_eq" else None,
                    ),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("decoder_mode", decoder_mode),
                    ("sparse_num_inducing", sparse_num_inducing),
                    ("provided_uref_file", uref_file),
                    ("stage2_use_u_ref", model_use_u_ref),
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
                "gplvm_model",
                [
                    ("U_q_shape", model_data["U_q"].shape),
                    ("latent_dim", latent_dim),
                    ("n_train_latent", n_train_latent),
                    ("decoder_mode", decoder_mode),
                    ("sparse_num_inducing", sparse_num_inducing),
                    ("ell", float(np.exp(float(np.asarray(model_data['model']['log_ell']).reshape(()))))),
                    ("sf", float(np.exp(float(np.asarray(model_data['model']['log_sf']).reshape(()))))),
                    ("sn", float(np.exp(float(np.asarray(model_data['model']['log_sn']).reshape(()))))),
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
                    (
                        "snapshot_candidates_total",
                        ecsw_plan["num_candidates_total"] if ecsw_plan is not None else None,
                    ),
                    (
                        "snapshot_selected_total",
                        ecsw_plan["num_selected_total"] if ecsw_plan is not None else None,
                    ),
                    (
                        "snapshot_selected_per_mu",
                        ecsw_plan["num_selected_per_mu"] if ecsw_plan is not None else None,
                    ),
                ],
            ),
            (
                "hprom_timing",
                [
                    ("total_hprom_time_seconds", elapsed_hprom),
                    ("avg_hprom_time_per_step_seconds", elapsed_hprom / num_steps),
                    ("gn_iterations_total", num_its),
                    ("avg_gn_iterations_per_step", num_its / num_steps),
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
                    ("latent_trajectory_npy", latent_path),
                    ("comparison_plot_png", fig_path),
                    ("ecsw_weights_npy", weights_file),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                    ("summary_txt", report_path),
                    ("model_npz", model_data["model_npz_path"]),
                    ("U_q_npy", model_data["uq_path"]),
                ],
            ),
        ],
    )
    print(f"[HPROM-GPLVM] Text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)
