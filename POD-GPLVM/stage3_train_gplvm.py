#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 3: TRAIN POD-GPLVM MODEL

Inputs from previous stages (inside POD-GPLVM):
  - basis.npy
  - q.npy

Outputs (inside POD-GPLVM/pod_gplvm_model):
  - gplvm_model.npz
  - U_q.npy
  - u_ref.npy
  - q_train_used.npy
  - stage3_train_gplvm_summary.txt
  - stage3_gplvm_objective_history.png
  - stage3_validation_relative_error.png (if q_test.npy exists)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.gplvm_inverse import (
    latent_box_bounds,
    nearest_seed_indices,
    solve_bounded_nls,
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


def remove_near_duplicates(x, tol):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and np.linalg.norm(x[i] - x[j]) < tol:
                keep[j] = False
    return keep


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

    return bool(use_u_ref), u_ref_vec, u_ref_source


def _pairwise_sq_dists(X, Y=None):
    X = np.asarray(X, dtype=np.float64)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    xx = np.sum(X * X, axis=1)[:, None]
    yy = np.sum(Y * Y, axis=1)[None, :]
    d2 = xx + yy - 2.0 * (X @ Y.T)
    return np.maximum(d2, 0.0)


def _build_kernel(Z, log_ell, log_sf, log_sn, jitter):
    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(2.0 * log_sf))
    sn2 = float(np.exp(2.0 * log_sn))

    d2 = _pairwise_sq_dists(Z)
    log_k = np.clip(-0.5 * d2 / (ell * ell), -700.0, 50.0)
    K_no_noise = sf2 * np.exp(log_k)

    n = Z.shape[0]
    K = K_no_noise + (sn2 + float(jitter)) * np.eye(n, dtype=np.float64)
    return K, K_no_noise, d2, ell, sf2, sn2


def _kernel_cross(X, Y, ell, sf2):
    d2 = _pairwise_sq_dists(X, Y)
    log_k = np.clip(-0.5 * d2 / (ell * ell), -700.0, 50.0)
    return sf2 * np.exp(log_k)


def _select_inducing_indices(Z, m, method="fps", rng=None):
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    m = int(max(1, min(int(m), n)))
    if m == n:
        return np.arange(n, dtype=np.int64)

    mode = str(method).strip().lower()
    if mode in ("random", "rand"):
        if rng is None:
            rng = np.random.default_rng(0)
        return np.sort(rng.choice(n, size=m, replace=False)).astype(np.int64)

    # Farthest-point sampling in latent space for good coverage.
    if rng is None:
        rng = np.random.default_rng(0)
    first = int(rng.integers(0, n))
    sel = np.empty(m, dtype=np.int64)
    sel[0] = first

    min_d2 = np.sum((Z - Z[first][None, :]) ** 2, axis=1)
    min_d2[first] = -1.0
    for i in range(1, m):
        idx = int(np.argmax(min_d2))
        sel[i] = idx
        d2_new = np.sum((Z - Z[idx][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[sel[: i + 1]] = -1.0
    return np.sort(sel)


def _build_sparse_decoder_dtc(
    Z_train,
    Y_norm,
    log_ell,
    log_sf,
    log_sn,
    jitter,
    num_inducing,
    inducing_method="fps",
    rng=None,
):
    Z_train = np.asarray(Z_train, dtype=np.float64)
    Y_norm = np.asarray(Y_norm, dtype=np.float64)
    n_train = Z_train.shape[0]
    if Y_norm.shape[0] != n_train:
        raise ValueError(
            f"Y_norm row mismatch with Z_train: {Y_norm.shape[0]} vs {n_train}"
        )

    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(2.0 * log_sf))
    sn2 = float(np.exp(2.0 * log_sn))

    inducing_idx = _select_inducing_indices(
        Z_train,
        num_inducing,
        method=inducing_method,
        rng=rng,
    )
    Z_ind = Z_train[inducing_idx, :]

    K_mm = _kernel_cross(Z_ind, Z_ind, ell=ell, sf2=sf2)
    K_nm = _kernel_cross(Z_train, Z_ind, ell=ell, sf2=sf2)
    K_mn = K_nm.T

    K_mm = K_mm + float(jitter) * np.eye(K_mm.shape[0], dtype=np.float64)

    # DTC posterior mean:
    # q(z) = k(z, Zm) @ beta,
    # beta = (Kmm + (1/sn2) Kmn Knm)^(-1) * (Kmn Y / sn2)
    B = K_mm + (K_mn @ K_nm) / sn2
    rhs = (K_mn @ Y_norm) / sn2

    B_cho, _ = _chol_with_jitter(B, initial_jitter=max(1e-12, float(jitter)), max_tries=6)
    beta = cho_solve(B_cho, rhs, check_finite=False)

    return {
        "Z_inducing": Z_ind.astype(np.float64),
        "inducing_indices": inducing_idx.astype(np.int64),
        "beta_inducing": beta.astype(np.float64),
        "num_inducing": int(Z_ind.shape[0]),
        "inducing_method": str(inducing_method),
    }


def _model_uses_sparse_decoder(model):
    if "decoder_mode" not in model:
        return False
    mode = str(np.asarray(model["decoder_mode"]).reshape(()))
    return mode.lower() == "sparse_dtc"


def _chol_with_jitter(K, initial_jitter=1e-10, max_tries=8):
    jitter = float(initial_jitter)
    n = K.shape[0]
    eye = np.eye(n, dtype=K.dtype)

    for _ in range(max_tries):
        try:
            L, lower = cho_factor(K + jitter * eye, lower=True, check_finite=False)
            return (L, lower), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise np.linalg.LinAlgError("Cholesky failed even after jitter escalation.")


def _decode_q_from_latent(z, model):
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    use_sparse = _model_uses_sparse_decoder(model)
    if use_sparse:
        Z_centers = np.asarray(model["Z_inducing"], dtype=np.float64)
        weights = np.asarray(model["beta_inducing"], dtype=np.float64)
    else:
        Z_centers = np.asarray(model["Z_train"], dtype=np.float64)
        weights = np.asarray(model["alpha"], dtype=np.float64)
    y_mean = np.asarray(model["y_mean"], dtype=np.float64).reshape(-1)
    y_std = np.asarray(model["y_std"], dtype=np.float64).reshape(-1)

    log_ell = float(np.asarray(model["log_ell"]).reshape(()))
    log_sf = float(np.asarray(model["log_sf"]).reshape(()))

    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(2.0 * log_sf))

    diff = Z_centers - z[None, :]
    diff = np.clip(diff, -1e150, 1e150)
    d2 = np.sum(diff * diff, axis=1)
    log_k = np.clip(-0.5 * d2 / (ell * ell), -700.0, 50.0)
    k = sf2 * np.exp(log_k)

    y_norm = k @ weights
    return y_mean + y_std * y_norm


def _jac_q_from_latent(z, model):
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    use_sparse = _model_uses_sparse_decoder(model)
    if use_sparse:
        Z_centers = np.asarray(model["Z_inducing"], dtype=np.float64)
        weights = np.asarray(model["beta_inducing"], dtype=np.float64)
    else:
        Z_centers = np.asarray(model["Z_train"], dtype=np.float64)
        weights = np.asarray(model["alpha"], dtype=np.float64)
    y_std = np.asarray(model["y_std"], dtype=np.float64).reshape(-1)

    log_ell = float(np.asarray(model["log_ell"]).reshape(()))
    log_sf = float(np.asarray(model["log_sf"]).reshape(()))

    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(2.0 * log_sf))
    ell2 = ell * ell

    diff = Z_centers - z[None, :]
    diff = np.clip(diff, -1e150, 1e150)
    d2 = np.sum(diff * diff, axis=1)
    log_k = np.clip(-0.5 * d2 / ell2, -700.0, 50.0)
    k = sf2 * np.exp(log_k)

    dk_dz = k[:, None] * (diff / ell2)
    dy_norm_dz = weights.T @ dk_dz
    dy_dz = y_std[:, None] * dy_norm_dz
    return dy_dz


def _infer_latent_for_q(
    q_target,
    model,
    max_its=30,
    tol_rel=1e-6,
    inverse_method="bounded_trf",
    n_starts=5,
    bound_margin_rel=0.20,
    bound_margin_abs=0.25,
    prior_weight=1e-3,
    robust_loss="linear",
    robust_f_scale=1.0,
):
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)

    q_train = np.asarray(model["Q_train_raw"], dtype=np.float64)
    z_train = np.asarray(model["Z_train"], dtype=np.float64)

    mode = str(inverse_method).strip().lower()
    if mode == "gauss_newton":
        d2 = np.sum((q_train - q_target[None, :]) ** 2, axis=1)
        z = z_train[int(np.argmin(d2))].copy()

        q_pred = _decode_q_from_latent(z, model)
        r0 = np.linalg.norm(q_pred - q_target)
        if r0 == 0.0:
            return z, q_pred, 0

        it = 0
        rel = 1.0
        while rel > tol_rel and it < max_its:
            J = _jac_q_from_latent(z, model)
            r = q_pred - q_target

            dz, *_ = np.linalg.lstsq(J, r, rcond=None)
            z -= dz

            q_pred = _decode_q_from_latent(z, model)
            rel = np.linalg.norm(q_pred - q_target) / r0
            it += 1

        return z, q_pred, it

    if mode not in ("bounded_trf", "trf", "least_squares"):
        raise ValueError(
            "inverse_method must be one of: 'bounded_trf', 'least_squares', 'trf', 'gauss_newton'."
        )

    seed_ids, _ = nearest_seed_indices(q_target, q_train, n_starts=n_starts)
    lb, ub = latent_box_bounds(
        z_train,
        margin_rel=bound_margin_rel,
        margin_abs=bound_margin_abs,
    )

    tol = float(max(tol_rel, 1e-12))
    max_nfev = int(max(50, 10 * max_its))

    best_cost = np.inf
    best_z = None
    best_q = None
    best_nfev = 0

    for sid in seed_ids:
        z0 = z_train[int(sid)].copy()

        def residual(z):
            return _decode_q_from_latent(z, model) - q_target

        def jac(z):
            return _jac_q_from_latent(z, model)

        z_sol, lsq_res = solve_bounded_nls(
            x0=z0,
            residual_func=residual,
            jac_func=jac,
            lb=lb,
            ub=ub,
            prior_center=z0,
            prior_weight=prior_weight,
            max_nfev=max_nfev,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            loss=robust_loss,
            f_scale=robust_f_scale,
        )

        q_hat = _decode_q_from_latent(z_sol, model)
        r = q_hat - q_target
        cost = 0.5 * float(np.dot(r, r))

        if cost < best_cost:
            best_cost = cost
            best_z = z_sol
            best_q = q_hat
            best_nfev = int(lsq_res.nfev) if lsq_res.nfev is not None else 0

    if best_z is None:
        best_z = z_train[int(seed_ids[0])].copy()
        best_q = _decode_q_from_latent(best_z, model)
        best_nfev = 0

    return best_z, best_q, best_nfev


def _save_objective_plot(history, out_path):
    if len(history) == 0:
        return False

    vals = np.asarray(history, dtype=np.float64)
    epochs = np.arange(1, vals.size + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, vals, color="tab:blue", linewidth=1.6)
    ax.set_xlabel("L-BFGS function call")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_validation_plot(rel_err_per_sample_pct, out_path):
    rel_err_per_sample_pct = np.asarray(rel_err_per_sample_pct, dtype=np.float64).reshape(-1)
    if rel_err_per_sample_pct.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        np.arange(rel_err_per_sample_pct.size),
        rel_err_per_sample_pct,
        color="tab:red",
        linewidth=1.5,
        marker="o",
        markersize=3.0,
    )
    ax.set_xlabel("Validation sample index")
    ax.set_ylabel("Relative error [%]")
    ax.grid(True, alpha=0.35)
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
    model_dir=os.path.join(script_dir, "pod_gplvm_model"),
    model_file=os.path.join(script_dir, "pod_gplvm_model", "gplvm_model.npz"),
    uq_file=os.path.join(script_dir, "pod_gplvm_model", "U_q.npy"),
    model_uref_file=os.path.join(script_dir, "pod_gplvm_model", "u_ref.npy"),
    q_train_used_file=os.path.join(script_dir, "pod_gplvm_model", "q_train_used.npy"),
    report_file=os.path.join(script_dir, "stage3_train_gplvm_summary.txt"),
    objective_plot_file=os.path.join(script_dir, "stage3_gplvm_objective_history.png"),
    validation_plot_file=os.path.join(script_dir, "stage3_validation_relative_error.png"),
    latent_dim=10,
    max_train_samples=None,
    duplicate_tol=1e-12,
    latent_reg=1e-4,
    sparse_decoder="auto",
    sparse_num_inducing=300,
    sparse_inducing_method="fps",
    ell_init=1.0,
    sf_init=1.0,
    sn_init=1e-2,
    ell_bounds=(1e-2, 20.0),
    sf_bounds=(1e-3, 20.0),
    sn_bounds=(1e-4, 1.0),
    jitter=1e-6,
    maxiter=400,
    maxfun=5000,
    gtol=1e-6,
    optimizer_ftol=1e-15,
    optimizer_maxls=100,
    optimizer_maxcor=20,
    auto_restart=True,
    max_restarts=2,
    restart_perturb_std=5e-2,
    objective_scale_mode="auto",
    verbose_training=True,
    print_every_eval=10,
    print_every_iter=1,
    monitor_train_relerr=True,
    monitor_train_relerr_samples=200,
    validation_max_samples=200,
    val_infer_max_its=30,
    val_infer_tol=1e-6,
    val_inverse_method="bounded_trf",
    val_inverse_n_starts=5,
    val_inverse_bound_margin_rel=0.20,
    val_inverse_bound_margin_abs=0.25,
    val_inverse_prior_weight=1e-3,
    val_inverse_loss="linear",
    val_inverse_f_scale=1.0,
    random_seed=42,
    uref_mode="auto",
):
    for path in (basis_file, q_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}. Run stage1/stage2 first.")

    os.makedirs(model_dir, exist_ok=True)

    rng = np.random.default_rng(int(random_seed))

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    q = np.asarray(np.load(q_file, allow_pickle=False), dtype=np.float64)

    if q.ndim != 2:
        raise ValueError("q must be a 2D array.")

    q_dim = int(q.shape[0])
    n_samples_total = int(q.shape[1])

    if q_dim < 1:
        raise ValueError("q has zero rows. Check stage2 outputs.")
    if basis.shape[1] < q_dim:
        raise ValueError(
            "Basis has fewer columns than q dimension: "
            f"basis columns={basis.shape[1]}, q_dim={q_dim}."
        )

    use_u_ref, u_ref_vec, u_ref_source = resolve_u_ref(
        uref_mode=uref_mode,
        uref_file=uref_file,
        stage2_metadata_file=stage2_metadata_file,
        expected_size=basis.shape[0],
    )

    print("\n====================================================")
    print("            STAGE 3: TRAIN POD-GPLVM")
    print("====================================================")
    print(f"[STAGE3] q_dim={q_dim}, n_samples_total={n_samples_total}, latent_dim={latent_dim}")
    print(
        f"[STAGE3] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref_vec):.3e}"
    )
    if verbose_training:
        print(
            f"[STAGE3] training logs enabled: print_every_eval={print_every_eval}, "
            f"print_every_iter={print_every_iter}"
        )

    sample_idx = np.arange(n_samples_total, dtype=np.int64)
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
        if max_train_samples < 2:
            raise ValueError("max_train_samples must be >= 2 when provided.")
        if n_samples_total > max_train_samples:
            sample_idx = np.sort(rng.choice(n_samples_total, size=max_train_samples, replace=False))

    q_used = q[:, sample_idx].T  # (n_used, q_dim)

    if duplicate_tol is not None and float(duplicate_tol) > 0.0:
        keep_mask = remove_near_duplicates(q_used, float(duplicate_tol))
    else:
        keep_mask = np.ones(q_used.shape[0], dtype=bool)

    q_used = q_used[keep_mask]
    duplicates_removed = int(np.sum(~keep_mask))

    n_used = q_used.shape[0]
    if n_used < 2:
        raise RuntimeError(
            "Not enough samples available for GPLVM training after filtering. "
            "Decrease duplicate_tol or increase max_train_samples."
        )
    if int(latent_dim) < 1:
        raise ValueError("latent_dim must be >= 1.")

    latent_dim = int(latent_dim)
    sparse_mode = str(sparse_decoder).strip().lower()
    if sparse_mode not in ("auto", "on", "off"):
        raise ValueError("sparse_decoder must be one of: 'auto', 'on', 'off'.")
    sparse_num_inducing = int(sparse_num_inducing)
    if sparse_num_inducing < 1:
        raise ValueError("sparse_num_inducing must be >= 1.")
    monitor_train_relerr_samples = int(monitor_train_relerr_samples)
    if monitor_train_relerr_samples < 1:
        monitor_train_relerr = False
    if sparse_mode == "on":
        use_sparse_decoder = True
    elif sparse_mode == "off":
        use_sparse_decoder = False
    else:
        use_sparse_decoder = (n_used > sparse_num_inducing)
    sparse_num_inducing_eff = int(min(sparse_num_inducing, n_used))
    print(
        f"[STAGE3] sparse decoder mode={sparse_mode}, "
        f"use_sparse_decoder={use_sparse_decoder}, num_inducing={sparse_num_inducing_eff}"
    )
    if monitor_train_relerr:
        print(
            f"[STAGE3] monitor_train_relerr enabled on "
            f"{min(monitor_train_relerr_samples, n_used)} samples"
        )

    y_mean = q_used.mean(axis=0)
    y_std = q_used.std(axis=0)
    y_std = np.where(y_std > 1e-12, y_std, 1.0)
    Y = (q_used - y_mean[None, :]) / y_std[None, :]
    monitor_idx = None
    monitor_q_denom = None
    if monitor_train_relerr:
        m_mon = int(min(monitor_train_relerr_samples, n_used))
        if m_mon >= n_used:
            monitor_idx = np.arange(n_used, dtype=np.int64)
        else:
            monitor_idx = np.sort(rng.choice(n_used, size=m_mon, replace=False)).astype(np.int64)
        q_mon = q_used[monitor_idx, :]
        denom = float(np.linalg.norm(q_mon, ord="fro"))
        monitor_q_denom = denom if denom > 0.0 else 1.0

    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Yc, full_matrices=False)
    d_init = min(latent_dim, U.shape[1])
    Z0 = np.zeros((n_used, latent_dim), dtype=np.float64)
    if d_init > 0:
        Z0[:, :d_init] = U[:, :d_init] * S[:d_init][None, :]
    if d_init < latent_dim:
        Z0[:, d_init:] = 1e-2 * rng.normal(size=(n_used, latent_dim - d_init))

    z_std = Z0.std(axis=0)
    z_std = np.where(z_std > 1e-12, z_std, 1.0)
    Z0 = Z0 / z_std[None, :]

    log_ell0 = float(np.log(float(ell_init)))
    log_sf0 = float(np.log(float(sf_init)))
    log_sn0 = float(np.log(float(sn_init)))

    theta0 = np.concatenate([Z0.ravel(), [log_ell0, log_sf0, log_sn0]])

    if len(ell_bounds) != 2 or len(sf_bounds) != 2 or len(sn_bounds) != 2:
        raise ValueError("Bounds must have exactly two values each.")

    e0, e1 = float(ell_bounds[0]), float(ell_bounds[1])
    s0, s1 = float(sf_bounds[0]), float(sf_bounds[1])
    n0, n1 = float(sn_bounds[0]), float(sn_bounds[1])
    if not (0.0 < e0 < e1 and 0.0 < s0 < s1 and 0.0 < n0 < n1):
        raise ValueError("All bounds must be positive and strictly increasing.")

    n_lat_vars = n_used * latent_dim
    bounds = [(None, None)] * n_lat_vars
    bounds += [
        (np.log(e0), np.log(e1)),
        (np.log(s0), np.log(s1)),
        (np.log(n0), np.log(n1)),
    ]

    if isinstance(objective_scale_mode, str):
        mode = objective_scale_mode.strip().lower()
        if mode == "auto":
            objective_scale = float(max(1, n_used * q_dim))
        elif mode in ("off", "none", "1"):
            objective_scale = 1.0
        else:
            raise ValueError(
                "objective_scale_mode must be 'auto', 'off'/'none', or a positive float."
            )
    else:
        objective_scale = float(objective_scale_mode)
        if (not np.isfinite(objective_scale)) or (objective_scale <= 0.0):
            raise ValueError("objective_scale_mode numeric value must be a finite positive float.")

    objective_history = []
    eval_counter = {"n": 0}
    iter_counter = {"n": 0}
    baseline_state = {
        "nll0": None,
        "grad_inf0": None,
    }
    last_eval_state = {
        "nll": None,
        "grad_inf": None,
        "log_ell": None,
        "log_sf": None,
        "log_sn": None,
        "jitter": None,
        "nll_rel_impr": None,
        "grad_rel0": None,
        "train_relerr_knownz_pct": None,
    }

    def _compute_relative_metrics(nll, grad_inf):
        nll0 = baseline_state["nll0"]
        g0 = baseline_state["grad_inf0"]

        if nll0 is None or (not np.isfinite(nll0)):
            nll_rel_impr = None
        else:
            denom = max(abs(float(nll0)), 1e-15)
            nll_rel_impr = (float(nll0) - float(nll)) / denom

        if g0 is None or (not np.isfinite(g0)):
            grad_rel0 = None
        else:
            denom = max(abs(float(g0)), 1e-15)
            grad_rel0 = float(grad_inf) / denom

        return nll_rel_impr, grad_rel0

    def objective_and_grad_unscaled(theta, record_history=True, count_eval=True):
        if count_eval:
            eval_counter["n"] += 1

        Z = theta[:n_lat_vars].reshape(n_used, latent_dim)
        log_ell = float(theta[n_lat_vars + 0])
        log_sf = float(theta[n_lat_vars + 1])
        log_sn = float(theta[n_lat_vars + 2])

        try:
            K, K_no_noise, d2, ell, _, sn2 = _build_kernel(Z, log_ell, log_sf, log_sn, jitter=0.0)
            (L, lower), used_jitter = _chol_with_jitter(K, initial_jitter=jitter, max_tries=8)

            alpha = cho_solve((L, lower), Y, check_finite=False)
            logdetK = 2.0 * np.sum(np.log(np.diag(L)))

            nll = 0.5 * q_dim * logdetK
            nll += 0.5 * float(np.sum(Y * alpha))
            nll += 0.5 * n_used * q_dim * np.log(2.0 * np.pi)

            if latent_reg > 0.0:
                nll += 0.5 * float(latent_reg) * float(np.sum(Z * Z))

            Kinv = cho_solve((L, lower), np.eye(n_used, dtype=np.float64), check_finite=False)
            G = 0.5 * (q_dim * Kinv - alpha @ alpha.T)

            ell2 = ell * ell

            Ssym = (G + G.T) * (K_no_noise / ell2)
            grad_Z = Ssym @ Z - Ssym.sum(axis=1, keepdims=True) * Z
            if latent_reg > 0.0:
                grad_Z += float(latent_reg) * Z

            dK_dlog_ell = K_no_noise * (d2 / ell2)
            dK_dlog_sf = 2.0 * K_no_noise
            dK_dlog_sn = 2.0 * sn2 * np.eye(n_used, dtype=np.float64)

            grad_log_ell = float(np.sum(G * dK_dlog_ell))
            grad_log_sf = float(np.sum(G * dK_dlog_sf))
            grad_log_sn = float(np.sum(G * dK_dlog_sn))

            grad = np.concatenate([
                grad_Z.ravel(),
                [grad_log_ell, grad_log_sf, grad_log_sn],
            ])
            grad_inf = float(np.max(np.abs(grad)))

            if baseline_state["nll0"] is None and np.isfinite(float(nll)):
                baseline_state["nll0"] = float(nll)
            if baseline_state["grad_inf0"] is None and np.isfinite(grad_inf):
                baseline_state["grad_inf0"] = grad_inf
            nll_rel_impr, grad_rel0 = _compute_relative_metrics(float(nll), grad_inf)
            train_relerr_knownz_pct = None
            if monitor_idx is not None:
                yhat_mon = K_no_noise[monitor_idx, :] @ alpha
                qhat_mon = y_mean[None, :] + y_std[None, :] * yhat_mon
                q_mon = q_used[monitor_idx, :]
                err = float(np.linalg.norm(qhat_mon - q_mon, ord="fro"))
                train_relerr_knownz_pct = 100.0 * err / float(monitor_q_denom)

            last_eval_state["nll"] = float(nll)
            last_eval_state["grad_inf"] = grad_inf
            last_eval_state["log_ell"] = log_ell
            last_eval_state["log_sf"] = log_sf
            last_eval_state["log_sn"] = log_sn
            last_eval_state["jitter"] = float(used_jitter)
            last_eval_state["nll_rel_impr"] = nll_rel_impr
            last_eval_state["grad_rel0"] = grad_rel0
            last_eval_state["train_relerr_knownz_pct"] = train_relerr_knownz_pct

            if verbose_training and count_eval:
                pe = None if print_every_eval is None else int(print_every_eval)
                if pe is not None and pe > 0 and (eval_counter["n"] % pe == 0):
                    nll_rel_txt = (
                        f"{nll_rel_impr:+.3e}" if nll_rel_impr is not None else "N/A"
                    )
                    grad_rel_txt = (
                        f"{grad_rel0:.3e}" if grad_rel0 is not None else "N/A"
                    )
                    relerr_txt = (
                        f"{train_relerr_knownz_pct:.3e}%"
                        if train_relerr_knownz_pct is not None
                        else "N/A"
                    )
                    print(
                        f"[STAGE3][eval {eval_counter['n']:04d}] "
                        f"nll={float(nll):.6e} grad_inf={grad_inf:.3e} "
                        f"nll_rel_impr={nll_rel_txt} grad_rel0={grad_rel_txt} "
                        f"train_relerr_knownz={relerr_txt} "
                        f"ell={np.exp(log_ell):.3e} sf={np.exp(log_sf):.3e} "
                        f"sn={np.exp(log_sn):.3e} jitter={float(used_jitter):.1e}"
                    )

            if record_history:
                objective_history.append(float(nll))
            return float(nll), grad

        except np.linalg.LinAlgError:
            big = 1e30
            grad = np.zeros_like(theta)
            last_eval_state["nll"] = float(big)
            last_eval_state["grad_inf"] = 0.0
            last_eval_state["log_ell"] = log_ell
            last_eval_state["log_sf"] = log_sf
            last_eval_state["log_sn"] = log_sn
            last_eval_state["jitter"] = np.nan
            last_eval_state["nll_rel_impr"] = None
            last_eval_state["grad_rel0"] = None
            last_eval_state["train_relerr_knownz_pct"] = None
            if record_history:
                objective_history.append(big)
            return big, grad

    def objective_and_grad(theta):
        nll, grad = objective_and_grad_unscaled(theta, record_history=True)
        return float(nll / objective_scale), grad / objective_scale

    def _iteration_callback(_theta):
        iter_counter["n"] += 1
        if not verbose_training:
            return

        pe_iter = None if print_every_iter is None else int(print_every_iter)
        if pe_iter is None or pe_iter <= 0 or (iter_counter["n"] % pe_iter != 0):
            return

        if last_eval_state["nll"] is None:
            return

        nll_rel_txt = (
            f"{last_eval_state['nll_rel_impr']:+.3e}"
            if last_eval_state["nll_rel_impr"] is not None
            else "N/A"
        )
        grad_rel_txt = (
            f"{last_eval_state['grad_rel0']:.3e}"
            if last_eval_state["grad_rel0"] is not None
            else "N/A"
        )
        relerr_txt = (
            f"{last_eval_state['train_relerr_knownz_pct']:.3e}%"
            if last_eval_state["train_relerr_knownz_pct"] is not None
            else "N/A"
        )
        print(
            f"[STAGE3][iter {iter_counter['n']:04d}] "
            f"nll={last_eval_state['nll']:.6e} "
            f"grad_inf={last_eval_state['grad_inf']:.3e} "
            f"nll_rel_impr={nll_rel_txt} "
            f"grad_rel0={grad_rel_txt} "
            f"train_relerr_knownz={relerr_txt} "
            f"ell={np.exp(last_eval_state['log_ell']):.3e} "
            f"sf={np.exp(last_eval_state['log_sf']):.3e} "
            f"sn={np.exp(last_eval_state['log_sn']):.3e}"
        )

    def _run_lbfgsb(theta_init):
        return minimize(
            fun=lambda th: objective_and_grad(th),
            x0=theta_init,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=_iteration_callback,
            options={
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "gtol": float(gtol),
                "ftol": float(optimizer_ftol),
                "maxls": int(optimizer_maxls),
                "maxcor": int(optimizer_maxcor),
                "disp": False,
            },
        )

    def _eval_solution(theta):
        theta = np.asarray(theta, dtype=np.float64)
        obj, grad = objective_and_grad_unscaled(theta, record_history=False, count_eval=False)
        grad_l2 = float(np.linalg.norm(grad))
        grad_inf = float(np.max(np.abs(grad)))
        return obj, grad, grad_l2, grad_inf

    def _is_false_converged(res_obj, grad_inf_value):
        return (
            int(res_obj.nit or 0) <= 1
            and "REL_REDUCTION_OF_F" in str(res_obj.message)
            and grad_inf_value > 1e3 * float(gtol)
        )

    restart_count = 0
    t0 = time.time()
    res = _run_lbfgsb(theta0)
    theta_opt = np.asarray(res.x, dtype=np.float64)
    final_obj_unscaled, final_grad_unscaled, final_grad_l2, final_grad_inf = _eval_solution(theta_opt)

    if bool(auto_restart) and _is_false_converged(res, final_grad_inf):
        print(
            "[STAGE3][WARN] Early-stop pattern detected. "
            "Launching restart(s) with latent perturbation."
        )
        rb = np.asarray(
            [
                [np.log(e0), np.log(e1)],
                [np.log(s0), np.log(s1)],
                [np.log(n0), np.log(n1)],
            ],
            dtype=np.float64,
        )
        best_pack = (res, theta_opt, final_obj_unscaled, final_grad_unscaled, final_grad_l2, final_grad_inf)
        for ir in range(int(max_restarts)):
            restart_count += 1
            theta_try0 = best_pack[1].copy()
            theta_try0[:n_lat_vars] += float(restart_perturb_std) * rng.normal(size=n_lat_vars)
            theta_try0[n_lat_vars:] = np.minimum(np.maximum(theta_try0[n_lat_vars:], rb[:, 0]), rb[:, 1])

            res_try = _run_lbfgsb(theta_try0)
            theta_try = np.asarray(res_try.x, dtype=np.float64)
            obj_try, grad_try, grad_l2_try, grad_inf_try = _eval_solution(theta_try)
            print(
                f"[STAGE3][restart {restart_count:02d}] "
                f"obj={obj_try:.6e}, grad_inf={grad_inf_try:.3e}, "
                f"nit={int(res_try.nit) if res_try.nit is not None else -1}, success={bool(res_try.success)}"
            )

            if obj_try < best_pack[2]:
                best_pack = (res_try, theta_try, obj_try, grad_try, grad_l2_try, grad_inf_try)
            if not _is_false_converged(res_try, grad_inf_try):
                break

        res, theta_opt, final_obj_unscaled, final_grad_unscaled, final_grad_l2, final_grad_inf = best_pack

    elapsed_train = time.time() - t0

    Z_opt = theta_opt[:n_lat_vars].reshape(n_used, latent_dim)
    log_ell_opt = float(theta_opt[n_lat_vars + 0])
    log_sf_opt = float(theta_opt[n_lat_vars + 1])
    log_sn_opt = float(theta_opt[n_lat_vars + 2])

    K_opt, _, _, _, _, _ = _build_kernel(Z_opt, log_ell_opt, log_sf_opt, log_sn_opt, jitter=0.0)
    (L_opt, lower_opt), used_jitter_opt = _chol_with_jitter(K_opt, initial_jitter=jitter, max_tries=8)
    alpha_opt = cho_solve((L_opt, lower_opt), Y, check_finite=False)
    final_train_relerr_knownz_pct = None
    if monitor_idx is not None:
        yhat_mon = K_opt[monitor_idx, :] @ alpha_opt
        qhat_mon = y_mean[None, :] + y_std[None, :] * yhat_mon
        q_mon = q_used[monitor_idx, :]
        err = float(np.linalg.norm(qhat_mon - q_mon, ord="fro"))
        final_train_relerr_knownz_pct = 100.0 * err / float(monitor_q_denom)

    sparse_meta = {
        "decoder_mode": "full",
        "sparse_enabled": 0,
        "num_inducing": 0,
        "inducing_method": str(sparse_inducing_method),
    }
    sparse_decoder_data = None
    if use_sparse_decoder:
        sparse_decoder_data = _build_sparse_decoder_dtc(
            Z_train=Z_opt,
            Y_norm=Y,
            log_ell=log_ell_opt,
            log_sf=log_sf_opt,
            log_sn=log_sn_opt,
            jitter=max(1e-12, float(jitter)),
            num_inducing=sparse_num_inducing_eff,
            inducing_method=sparse_inducing_method,
            rng=rng,
        )
        sparse_meta["decoder_mode"] = "sparse_dtc"
        sparse_meta["sparse_enabled"] = 1
        sparse_meta["num_inducing"] = int(sparse_decoder_data["num_inducing"])
        print(
            f"[STAGE3] Sparse decoder built with m={sparse_meta['num_inducing']} "
            f"(method={sparse_meta['inducing_method']})."
        )

    model_dict = {
        "Z_train": Z_opt,
        "Q_train_raw": q_used,
        "y_mean": y_mean,
        "y_std": y_std,
        "alpha": alpha_opt,
        "log_ell": np.asarray(log_ell_opt, dtype=np.float64),
        "log_sf": np.asarray(log_sf_opt, dtype=np.float64),
        "log_sn": np.asarray(log_sn_opt, dtype=np.float64),
        "jitter_used": np.asarray(used_jitter_opt, dtype=np.float64),
        "latent_dim": np.asarray(latent_dim, dtype=np.int64),
        "q_dim": np.asarray(q_dim, dtype=np.int64),
        "n_train": np.asarray(n_used, dtype=np.int64),
        "train_sample_indices": sample_idx,
        "use_u_ref": np.asarray(int(use_u_ref), dtype=np.int64),
        "latent_reg": np.asarray(float(latent_reg), dtype=np.float64),
        "decoder_mode": np.asarray(sparse_meta["decoder_mode"]),
        "sparse_enabled": np.asarray(int(sparse_meta["sparse_enabled"]), dtype=np.int64),
        "sparse_num_inducing": np.asarray(int(sparse_meta["num_inducing"]), dtype=np.int64),
        "sparse_inducing_method": np.asarray(str(sparse_meta["inducing_method"])),
    }
    if sparse_decoder_data is not None:
        model_dict["Z_inducing"] = sparse_decoder_data["Z_inducing"]
        model_dict["beta_inducing"] = sparse_decoder_data["beta_inducing"]
        model_dict["inducing_indices"] = sparse_decoder_data["inducing_indices"]

    np.savez(model_file, **model_dict)
    np.save(uq_file, basis[:, :q_dim])
    np.save(model_uref_file, u_ref_vec)
    np.save(q_train_used_file, q_used)

    objective_plot_saved = _save_objective_plot(objective_history, objective_plot_file)

    val_rel_err = None
    val_plot_saved = False
    val_rel_per_sample = np.array([], dtype=np.float64)

    if os.path.exists(q_test_file):
        q_test = np.asarray(np.load(q_test_file, allow_pickle=False), dtype=np.float64)
        if q_test.ndim == 2 and q_test.shape[0] == q_dim and q_test.shape[1] > 0:
            n_test = q_test.shape[1]
            if validation_max_samples is not None and n_test > int(validation_max_samples):
                pick = np.linspace(0, n_test - 1, int(validation_max_samples), dtype=int)
                q_eval = q_test[:, pick]
            else:
                q_eval = q_test

            q_pred = np.zeros_like(q_eval)
            for j in range(q_eval.shape[1]):
                _, q_hat, _ = _infer_latent_for_q(
                    q_eval[:, j],
                    model=model_dict,
                    max_its=val_infer_max_its,
                    tol_rel=val_infer_tol,
                    inverse_method=val_inverse_method,
                    n_starts=val_inverse_n_starts,
                    bound_margin_rel=val_inverse_bound_margin_rel,
                    bound_margin_abs=val_inverse_bound_margin_abs,
                    prior_weight=val_inverse_prior_weight,
                    robust_loss=val_inverse_loss,
                    robust_f_scale=val_inverse_f_scale,
                )
                q_pred[:, j] = q_hat

            denom = np.linalg.norm(q_eval, ord="fro")
            if denom > 0.0:
                val_rel_err = 100.0 * np.linalg.norm(q_eval - q_pred, ord="fro") / denom

            denom_col = np.linalg.norm(q_eval, axis=0)
            err_col = np.linalg.norm(q_eval - q_pred, axis=0)
            safe_denom = np.where(denom_col > 0.0, denom_col, 1.0)
            val_rel_per_sample = 100.0 * err_col / safe_denom
            val_plot_saved = _save_validation_plot(val_rel_per_sample, validation_plot_file)

    final_nll_rel_impr, final_grad_rel0 = _compute_relative_metrics(
        final_obj_unscaled, final_grad_inf
    )

    print(f"[STAGE3] Training finished in {elapsed_train:.3e} s")
    print(f"[STAGE3] Optimizer success: {res.success}")
    print(f"[STAGE3] Final objective: {final_obj_unscaled:.6e}")
    print(f"[STAGE3] Final gradient inf-norm: {final_grad_inf:.6e}")
    if final_nll_rel_impr is not None:
        print(f"[STAGE3] Final nll_rel_impr vs first eval: {final_nll_rel_impr:+.6e}")
    if final_grad_rel0 is not None:
        print(f"[STAGE3] Final grad_rel0 vs first eval: {final_grad_rel0:.6e}")
    if final_train_relerr_knownz_pct is not None:
        print(
            "[STAGE3] Final train_relerr_knownz "
            f"(subset, no inverse) [%]: {final_train_relerr_knownz_pct:.6e}"
        )
    print(f"[STAGE3] Restart count used: {int(restart_count)}")
    if (
        int(res.nit) <= 1
        and "REL_REDUCTION_OF_F" in str(res.message)
        and final_grad_inf > 1e3 * float(gtol)
    ):
        print(
            "[STAGE3][WARN] Early stop detected: relative objective reduction criterion "
            "triggered while gradient is still large."
        )
    print(
        "[STAGE3] Hyperparameters: "
        f"ell={np.exp(log_ell_opt):.6e}, sf={np.exp(log_sf_opt):.6e}, sn={np.exp(log_sn_opt):.6e}"
    )

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage3_train_gplvm.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("q_file", q_file),
                    ("q_test_file", q_test_file if os.path.exists(q_test_file) else None),
                    ("stage2_metadata_file", stage2_metadata_file if os.path.exists(stage2_metadata_file) else None),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref_vec))),
                    ("latent_dim", latent_dim),
                    ("max_train_samples", max_train_samples),
                    ("duplicate_tol", duplicate_tol),
                    ("latent_reg", latent_reg),
                    ("sparse_decoder", sparse_decoder),
                    ("sparse_num_inducing", sparse_num_inducing),
                    ("sparse_inducing_method", sparse_inducing_method),
                    ("ell_init", ell_init),
                    ("sf_init", sf_init),
                    ("sn_init", sn_init),
                    ("ell_bounds", ell_bounds),
                    ("sf_bounds", sf_bounds),
                    ("sn_bounds", sn_bounds),
                    ("jitter", jitter),
                    ("maxiter", maxiter),
                    ("maxfun", maxfun),
                    ("gtol", gtol),
                    ("optimizer_ftol", optimizer_ftol),
                    ("optimizer_maxls", optimizer_maxls),
                    ("optimizer_maxcor", optimizer_maxcor),
                    ("auto_restart", bool(auto_restart)),
                    ("max_restarts", max_restarts),
                    ("restart_perturb_std", restart_perturb_std),
                    ("objective_scale_mode", objective_scale_mode),
                    ("objective_scale", objective_scale),
                    ("verbose_training", bool(verbose_training)),
                    ("print_every_eval", print_every_eval),
                    ("print_every_iter", print_every_iter),
                    ("monitor_train_relerr", bool(monitor_train_relerr)),
                    ("monitor_train_relerr_samples", monitor_train_relerr_samples),
                    ("validation_max_samples", validation_max_samples),
                    ("val_infer_max_its", val_infer_max_its),
                    ("val_infer_tol", val_infer_tol),
                    ("val_inverse_method", val_inverse_method),
                    ("val_inverse_n_starts", val_inverse_n_starts),
                    ("val_inverse_bound_margin_rel", val_inverse_bound_margin_rel),
                    ("val_inverse_bound_margin_abs", val_inverse_bound_margin_abs),
                    ("val_inverse_prior_weight", val_inverse_prior_weight),
                    ("val_inverse_loss", val_inverse_loss),
                    ("val_inverse_f_scale", val_inverse_f_scale),
                    ("random_seed", random_seed),
                ],
            ),
            (
                "dataset",
                [
                    ("basis_shape", basis.shape),
                    ("q_shape", q.shape),
                    ("q_dim", q_dim),
                    ("n_samples_total", n_samples_total),
                    ("n_samples_used", n_used),
                    ("duplicates_removed", duplicates_removed),
                    ("monitor_train_relerr_subset_size", int(monitor_idx.size) if monitor_idx is not None else 0),
                ],
            ),
            (
                "optimization",
                [
                    ("success", bool(res.success)),
                    ("status", int(res.status)),
                    ("message", str(res.message)),
                    ("nfev", int(res.nfev)),
                    ("njev", int(res.njev) if res.njev is not None else None),
                    ("nit", int(res.nit) if res.nit is not None else None),
                    ("final_objective", final_obj_unscaled),
                    ("final_objective_scaled", float(res.fun)),
                    ("final_grad_l2_norm", final_grad_l2),
                    ("final_grad_inf_norm", final_grad_inf),
                    ("baseline_nll_first_eval", baseline_state["nll0"]),
                    ("baseline_grad_inf_first_eval", baseline_state["grad_inf0"]),
                    ("final_nll_relative_improvement_vs_first_eval", final_nll_rel_impr),
                    ("final_grad_relative_to_first_eval", final_grad_rel0),
                    ("final_train_relerr_knownz_subset_percent", final_train_relerr_knownz_pct),
                    ("elapsed_train_seconds", elapsed_train),
                    ("objective_history_length", len(objective_history)),
                    ("callback_iteration_count", int(iter_counter["n"])),
                    ("restart_count", int(restart_count)),
                ],
            ),
            (
                "learned_parameters",
                [
                    ("ell", float(np.exp(log_ell_opt))),
                    ("sf", float(np.exp(log_sf_opt))),
                    ("sn", float(np.exp(log_sn_opt))),
                    ("log_ell", log_ell_opt),
                    ("log_sf", log_sf_opt),
                    ("log_sn", log_sn_opt),
                    ("jitter_used", float(used_jitter_opt)),
                    ("decoder_mode", sparse_meta["decoder_mode"]),
                    ("sparse_enabled", bool(sparse_meta["sparse_enabled"])),
                    ("sparse_num_inducing", int(sparse_meta["num_inducing"])),
                    ("sparse_inducing_method", sparse_meta["inducing_method"]),
                ],
            ),
            (
                "validation",
                [
                    ("validation_relative_error_percent", val_rel_err),
                    ("validation_plot_saved", val_plot_saved),
                    (
                        "validation_mean_per_sample_relative_error_percent",
                        float(np.mean(val_rel_per_sample)) if val_rel_per_sample.size > 0 else None,
                    ),
                    (
                        "validation_max_per_sample_relative_error_percent",
                        float(np.max(val_rel_per_sample)) if val_rel_per_sample.size > 0 else None,
                    ),
                ],
            ),
            (
                "outputs",
                [
                    ("model_npz", model_file),
                    ("U_q_npy", uq_file),
                    ("u_ref_npy", model_uref_file),
                    ("q_train_used_npy", q_train_used_file),
                    ("objective_history_png", objective_plot_file if objective_plot_saved else None),
                    ("validation_relative_error_png", validation_plot_file if val_plot_saved else None),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )

    print(f"[STAGE3] Saved model: {model_file}")
    print(f"[STAGE3] Saved reduced basis used online: {uq_file}")
    print(f"[STAGE3] Saved u_ref: {model_uref_file}")
    print(f"[STAGE3] Saved train subset: {q_train_used_file}")
    print(f"[STAGE3] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
