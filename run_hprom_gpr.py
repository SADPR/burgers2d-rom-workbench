#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run global POD-GPR HPROM (ECSW-LSPG) for the 2D inviscid Burgers problem
using the modern `burgers/` modules and save outputs consistently in `Results`.
"""

import os
import time
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product

from burgers.core import (
    load_or_compute_snaps,
    plot_snaps,
    inviscid_burgers_res2D,
    inviscid_burgers_exact_jac2D,
    get_snapshot_params,
)
from burgers.pod_gpr_manifold import (
    decode_gp,
    compute_ECSW_training_matrix_2D_gpr,
    inviscid_burgers_implicit2D_LSPG_pod_gpr_ecsw,
)
from burgers.ecsw_utils import build_ecsw_snapshot_plan
from burgers.empirical_cubature_method import EmpiricalCubatureMethod
from burgers.randomized_singular_value_decomposition import (
    RandomizedSingularValueDecomposition,
)
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


def _is_analytic_jacobian_compatible(kernel):
    if not isinstance(kernel, Product):
        return False
    if not isinstance(kernel.k1, ConstantKernel):
        return False
    if not isinstance(kernel.k2, Matern):
        return False
    return float(kernel.k2.nu) == 1.5


def _load_model_artifacts(model_dir):
    gpr_path = os.path.join(model_dir, "gpr_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    u_p_path = os.path.join(model_dir, "U_p.npy")
    u_s_path = os.path.join(model_dir, "U_s.npy")
    q_p_norm_path = os.path.join(model_dir, "q_p_normalized.npy")
    q_s_path = os.path.join(model_dir, "q_s.npy")

    for path in (gpr_path, scaler_path, u_p_path, u_s_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing POD-GPR model artifact: {path}")

    with open(gpr_path, "rb") as file:
        gpr_model = pickle.load(file)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    u_p = np.asarray(np.load(u_p_path, allow_pickle=False), dtype=np.float64)
    u_s = np.asarray(np.load(u_s_path, allow_pickle=False), dtype=np.float64)

    if u_p.ndim != 2 or u_s.ndim != 2:
        raise ValueError("U_p and U_s must be 2D arrays.")
    if u_p.shape[0] != u_s.shape[0]:
        raise ValueError(
            f"U_p and U_s row mismatch: {u_p.shape[0]} vs {u_s.shape[0]}"
        )

    q_p_norm_shape = None
    q_s_shape = None
    if os.path.exists(q_p_norm_path):
        q_p_norm_shape = np.asarray(np.load(q_p_norm_path, allow_pickle=False)).shape
    if os.path.exists(q_s_path):
        q_s_shape = np.asarray(np.load(q_s_path, allow_pickle=False)).shape

    learned_kernel_obj = getattr(gpr_model, "kernel_", getattr(gpr_model, "kernel", None))
    learned_kernel_str = str(learned_kernel_obj)

    return {
        "gpr_path": gpr_path,
        "scaler_path": scaler_path,
        "u_p_path": u_p_path,
        "u_s_path": u_s_path,
        "q_p_norm_path": q_p_norm_path if os.path.exists(q_p_norm_path) else None,
        "q_s_path": q_s_path if os.path.exists(q_s_path) else None,
        "q_p_norm_shape": q_p_norm_shape,
        "q_s_shape": q_s_shape,
        "gpr_model": gpr_model,
        "scaler": scaler,
        "U_p": u_p,
        "U_s": u_s,
        "learned_kernel_obj": learned_kernel_obj,
        "learned_kernel_str": learned_kernel_str,
        "analytic_jacobian_compatible": _is_analytic_jacobian_compatible(learned_kernel_obj),
    }


def main(
    mu1=4.56,
    mu2=0.019,
    model_dir=os.path.join("POD-GPR", "pod_gpr_model"),
    compute_ecsw=True,
    weights_file=None,
    snap_folder=None,
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    uref_file=None,
    use_custom_predict=True,
    jacobian_mode="analytic",
    fd_eps=1e-6,
    snap_time_offset=3,
    mu_samples=None,
    ecsw_snapshot_percent=2.0,
    ecsw_random_seed=42,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-12,
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
        weights_file = os.path.join(model_dir, "ecsw_weights_gpr.npy")
    legacy_weights_files = [
        os.path.join(model_dir, "ecm_weights_gpr_global.npy"),
        os.path.join(model_dir, "ecm_weights_gp_global.npy"),
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

    model = _load_model_artifacts(model_dir)
    model_use_u_ref, stage2_metadata_path = _load_stage2_use_u_ref(model_dir)
    use_u_ref, u_ref, u_ref_source = _resolve_u_ref(
        uref_mode=uref_mode,
        explicit_uref_file=uref_file,
        model_use_u_ref=model_use_u_ref,
        model_dir=model_dir,
        expected_size=model["U_p"].shape[0],
    )

    if w0.size != model["U_p"].shape[0]:
        raise ValueError(
            f"Initial condition size mismatch: W0 has {w0.size}, model has {model['U_p'].shape[0]}."
        )

    if jacobian_mode == "analytic" and not model["analytic_jacobian_compatible"]:
        raise ValueError(
            "jacobian_mode='analytic' requires learned kernel ConstantKernel*Matern(nu=1.5). "
            f"Found kernel: {model['learned_kernel_str']}. "
            "Use jacobian_mode='forward_fd' or 'central_fd', or retrain stage3 with kernel_name='matern15'."
        )

    print(f"[HPROM-GPR] Loaded model from: {model_dir}")
    print(
        f"[HPROM-GPR] U_p shape={model['U_p'].shape}, U_s shape={model['U_s'].shape}, "
        f"kernel={model['learned_kernel_str']}"
    )
    print(
        f"[HPROM-GPR] jacobian_mode={jacobian_mode}, "
        f"analytic_compatible={model['analytic_jacobian_compatible']}"
    )
    print(
        f"[HPROM-GPR] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )
    print(f"[HPROM-GPR] Reduced linear solver: {linear_solver}")
    if str(linear_solver).strip().lower() == "normal_eq":
        print(f"[HPROM-GPR] normal_eq_reg: {float(normal_eq_reg):.3e}")

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
            "[HPROM-GPR] ECSW snapshot selection mode="
            f"{ecsw_plan['mode']}, selected {ecsw_plan['num_selected_total']} / "
            f"{ecsw_plan['num_candidates_total']} candidate pairs."
        )
        print(f"[HPROM-GPR] Selected snapshots per mu: {ecsw_plan['num_selected_per_mu']}")

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

            print(f"[HPROM-GPR] Generating ECSW training block for mu={mu_train}")
            ci = compute_ECSW_training_matrix_2D_gpr(
                snaps_now,
                snaps_prev,
                model["U_p"],
                model["U_s"],
                model["gpr_model"],
                inviscid_burgers_res2D,
                inviscid_burgers_exact_jac2D,
                grid_x,
                grid_y,
                dt,
                mu_train,
                model["scaler"],
                u_ref=u_ref,
                use_custom_predict=use_custom_predict,
                jacobian_mode=jacobian_mode,
                fd_eps=fd_eps,
            )
            clist.append(ci)

        if not clist:
            raise RuntimeError(
                "ECSW training produced zero columns for all mu samples. "
                "Increase ecsw_snapshot_percent or adjust snap_time_offset."
            )

        c = np.vstack(clist)
        c_shape = c.shape
        print(f"[HPROM-GPR] Stacked ECSW training matrix C shape: {c_shape}")

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

        print(f"[HPROM-GPR] ECSW weights saved to: {weights_file}")
        print(f"[HPROM-GPR] ECSW solve time: {elapsed_ecsw:.3e} seconds")
        print(f"[HPROM-GPR] ECSW residual: {ecsw_residual:.3e}")

        reduced_mesh_plot_path = os.path.join(results_dir, "hprom_gpr_reduced_mesh.png")
        plt.figure(figsize=(7, 6))
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel(r"$x$ cell index")
        plt.ylabel(r"$y$ cell index")
        plt.title("HPROM-GPR Reduced Mesh (ECSW)")
        plt.tight_layout()
        plt.savefig(reduced_mesh_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[HPROM-GPR] Reduced mesh plot saved to: {reduced_mesh_plot_path}")
    else:
        if os.path.exists(weights_file):
            weights = np.asarray(np.load(weights_file, allow_pickle=False), dtype=np.float64)
            weights_source = "loaded"
            print(f"[HPROM-GPR] Loaded ECSW weights from: {weights_file}")
        else:
            loaded = False
            for legacy_file in legacy_weights_files:
                if os.path.exists(legacy_file):
                    weights = np.asarray(np.load(legacy_file, allow_pickle=False), dtype=np.float64)
                    weights_source = f"loaded_legacy:{os.path.basename(legacy_file)}"
                    print(f"[HPROM-GPR] Loaded legacy ECSW weights from: {legacy_file}")
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
    print(f"[HPROM-GPR] N_e (nonzero ECSW weights): {n_ecsw_elements}")

    t0 = time.time()
    red_coords, hprom_stats = inviscid_burgers_implicit2D_LSPG_pod_gpr_ecsw(
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu=mu_rom,
        basis=model["U_p"],
        basis2=model["U_s"],
        gp_model=model["gpr_model"],
        weights=weights,
        scaler=model["scaler"],
        u_ref=u_ref,
        use_custom_predict=use_custom_predict,
        jacobian_mode=jacobian_mode,
        fd_eps=fd_eps,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        max_its_ic=max_its_ic,
        tol_ic=tol_ic,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )
    elapsed_hprom = time.time() - t0
    num_its, jac_time, res_time, ls_time = hprom_stats

    print(f"[HPROM-GPR] Elapsed HPROM time: {elapsed_hprom:.3e} seconds")
    print(f"[HPROM-GPR] Gauss-Newton iterations: {num_its}")
    print(
        "[HPROM-GPR] Timing breakdown (s): "
        f"jac={jac_time:.3e}, res={res_time:.3e}, ls={ls_time:.3e}"
    )

    rom_snaps = np.zeros((w0.size, red_coords.shape[1]), dtype=np.float64)
    for k in range(red_coords.shape[1]):
        rom_snaps[:, k] = decode_gp(
            q_p=red_coords[:, k],
            gp_model=model["gpr_model"],
            basis=model["U_p"],
            basis2=model["U_s"],
            scaler=model["scaler"],
            u_ref=u_ref,
            use_custom_predict=use_custom_predict,
            echo_level=0,
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
    print(f"[HPROM-GPR] Elapsed HDM load/solve time: {elapsed_hdm:.3e} seconds")

    rom_path = os.path.join(
        results_dir,
        f"hprom_gpr_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(rom_path, rom_snaps)
    print(f"[HPROM-GPR] HPROM snapshots saved to: {rom_path}")

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
        label="HPROM-GPR",
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
        f"hprom_gpr_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[HPROM-GPR] Comparison plot saved to: {fig_path}")

    hdm_norm = np.linalg.norm(hdm_snaps)
    if hdm_norm > 0.0:
        rel_err_l2 = np.linalg.norm(hdm_snaps - rom_snaps) / hdm_norm
    else:
        rel_err_l2 = np.nan
    relative_error = 100.0 * rel_err_l2
    print(f"[HPROM-GPR] Relative error: {relative_error:.2f}%")

    report_path = os.path.join(
        results_dir,
        f"hprom_gpr_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
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
                    ("linear_solver", linear_solver),
                    (
                        "normal_eq_reg",
                        normal_eq_reg if str(linear_solver).strip().lower() == "normal_eq" else None,
                    ),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("provided_uref_file", uref_file),
                    ("stage2_use_u_ref", model_use_u_ref),
                    ("stage2_projection_metadata", stage2_metadata_path if os.path.exists(stage2_metadata_path) else None),
                    ("use_custom_predict", use_custom_predict),
                    ("jacobian_mode", jacobian_mode),
                    ("fd_eps", fd_eps),
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
                "gpr_model",
                [
                    ("U_p_shape", model["U_p"].shape),
                    ("U_s_shape", model["U_s"].shape),
                    ("q_p_normalized_shape", model["q_p_norm_shape"]),
                    ("q_s_shape", model["q_s_shape"]),
                    ("learned_kernel", model["learned_kernel_str"]),
                    ("analytic_jacobian_compatible", model["analytic_jacobian_compatible"]),
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
                    ("comparison_plot_png", fig_path),
                    ("ecsw_weights_npy", weights_file),
                    ("ecsw_reduced_mesh_png", reduced_mesh_plot_path),
                    ("summary_txt", report_path),
                    ("gpr_model_pkl", model["gpr_path"]),
                    ("scaler_pkl", model["scaler_path"]),
                    ("U_p_npy", model["u_p_path"]),
                    ("U_s_npy", model["u_s_path"]),
                    ("q_p_normalized_npy", model["q_p_norm_path"]),
                    ("q_s_npy", model["q_s_path"]),
                ],
            ),
        ],
    )
    print(f"[HPROM-GPR] Text summary saved to: {report_path}")

    return elapsed_hprom, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=True)
