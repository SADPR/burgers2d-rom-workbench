# pod_gpr_manifold.py
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.sparse as sp

from .core import (
    get_ops,
    inviscid_burgers_res2D,
    inviscid_burgers_res2D_ecsw,
    inviscid_burgers_exact_jac2D,
    inviscid_burgers_exact_jac2D_ecsw,
)
from .cluster_utils import select_cluster_reduced
from .ecsw_utils import generate_augmented_mesh
from .gauss_newton import (
    gauss_newton_pod_gpr,
    gauss_newton_pod_gpr_ecsw,
)


# ============================================================================
# Helpers
# ============================================================================

def _prepare_reference(u_ref, size):
    """
    Return a 1D affine reference vector of length `size`.
    If u_ref is None, return zeros.
    """
    if u_ref is None:
        return np.zeros(size, dtype=np.float64)

    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if u_ref.size != size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {size}")
    return u_ref


def _gp_analytic_kernel_info(gp_model):
    """
    Return (kind, cval, length_scale) for analytic kernels supported by jac_gp.
    kind in {"matern15", "rbf"} or None if unsupported.
    """
    kernel_obj = getattr(gp_model, "kernel_", getattr(gp_model, "kernel", None))
    if kernel_obj is None:
        return None, None, None

    k1 = getattr(kernel_obj, "k1", None)
    k2 = getattr(kernel_obj, "k2", None)
    if k1 is None or k2 is None:
        return None, None, None

    if k1.__class__.__name__ != "ConstantKernel":
        return None, None, None

    cval = float(getattr(k1, "constant_value", 1.0))
    length_scale = np.asarray(getattr(k2, "length_scale", 1.0), dtype=np.float64)

    if k2.__class__.__name__ == "RBF":
        return "rbf", cval, length_scale

    if k2.__class__.__name__ == "Matern":
        try:
            if float(getattr(k2, "nu", np.nan)) == 1.5:
                return "matern15", cval, length_scale
        except Exception:
            pass

    return None, None, None


def _gp_is_analytic_compatible(gp_model):
    kind, _, _ = _gp_analytic_kernel_info(gp_model)
    return kind is not None


def _resolve_global_jacobian_mode(jacobian_mode, gp_model):
    """
    Resolve Jacobian mode for global POD-GPR.
    """
    mode = str(jacobian_mode).strip().lower()
    if mode not in ("auto", "analytic", "forward_fd", "central_fd"):
        raise ValueError(
            "jacobian_mode must be one of: 'auto', 'analytic', 'forward_fd', 'central_fd'."
        )

    analytic_ok = _gp_is_analytic_compatible(gp_model)
    if mode == "auto":
        return "analytic" if analytic_ok else "forward_fd"

    if mode == "analytic" and not analytic_ok:
        raise ValueError(
            "Requested jacobian_mode='analytic' but the learned GP kernel is not "
            "ConstantKernel*(Matern(nu=1.5) or RBF). Use 'auto', 'forward_fd', or 'central_fd'."
        )

    return mode


def _gp_target_stats(gp_model, n_targets):
    """
    Return target (mean, std) used by sklearn GPR normalization.
    """
    if bool(getattr(gp_model, "normalize_y", False)):
        y_mean = np.asarray(getattr(gp_model, "_y_train_mean", 0.0), dtype=np.float64).reshape(-1)
        y_std = np.asarray(getattr(gp_model, "_y_train_std", 1.0), dtype=np.float64).reshape(-1)
    else:
        y_mean = np.zeros(1, dtype=np.float64)
        y_std = np.ones(1, dtype=np.float64)

    if y_mean.size == 1 and n_targets > 1:
        y_mean = np.full(n_targets, float(y_mean[0]), dtype=np.float64)
    if y_std.size == 1 and n_targets > 1:
        y_std = np.full(n_targets, float(y_std[0]), dtype=np.float64)

    if y_mean.size != n_targets:
        y_mean = np.resize(y_mean, n_targets).astype(np.float64, copy=False)
    if y_std.size != n_targets:
        y_std = np.resize(y_std, n_targets).astype(np.float64, copy=False)

    return y_mean, y_std


def _predict_gp_custom(gp_model, x_scaled):
    """
    Custom GP prediction from (kernel vector @ alpha), with optional
    de-normalization when normalize_y=True.
    """
    X_train_ = gp_model.X_train_
    alpha_ = np.asarray(gp_model.alpha_, dtype=np.float64)
    kernel_ = gp_model.kernel_

    k_vec = kernel_(X_train_, x_scaled).ravel()
    y_pred = np.asarray(k_vec @ alpha_, dtype=np.float64).reshape(-1)

    y_mean, y_std = _gp_target_stats(gp_model, y_pred.size)
    y_pred = y_mean + y_std * y_pred
    return y_pred


def _cluster_total_dim(model_k, r_k):
    """
    Total number of retained reduced coordinates for a local cluster.
    """
    n_total_k = int(model_k.get("n_total", r_k))
    return min(n_total_k, r_k)


def _cluster_primary_dim(model_k, r_k, n_primary):
    """
    Number of active online coordinates in a local cluster.
    """
    n_total_k = _cluster_total_dim(model_k, r_k)
    has_gpr = bool(model_k.get("has_gpr", False))
    n_secondary = int(model_k.get("n_secondary", 0))

    if has_gpr and n_secondary > 0 and n_total_k > n_primary:
        return min(int(n_primary), n_total_k)

    return n_total_k


def _resolve_local_jacobian_mode(jacobian_mode, model_k):
    """
    Resolve Jacobian mode for a specific local GPR model.
    """
    mode = str(jacobian_mode).strip().lower()
    if mode not in ("auto", "analytic", "forward_fd", "central_fd"):
        raise ValueError(
            "jacobian_mode must be one of: 'auto', 'analytic', 'forward_fd', 'central_fd'."
        )

    analytic_ok = bool(model_k.get("analytic_jacobian_compatible", False))
    if mode == "auto":
        return "analytic" if analytic_ok else "forward_fd"

    if mode == "analytic" and not analytic_ok:
        raise ValueError(
            "Requested jacobian_mode='analytic' for a local GPR cluster whose learned "
            "kernel is not compatible with the analytical tangent. "
            "Use jacobian_mode='auto', 'forward_fd', or 'central_fd'."
        )

    return mode


def _resolve_local_selector_mode(selector_mode):
    mode = str(selector_mode).strip().lower()
    if mode not in ("linear", "nonlinear"):
        raise ValueError(
            "selector_mode must be one of: 'linear', 'nonlinear'."
        )
    return mode


def _predict_local_secondary_coords(model_k, q_p_eff, use_custom_predict=True):
    """
    Predict local GPR secondary coordinates q_s from primary q_p.
    """
    gp_model = model_k["gpr_model"]
    scaler = model_k["scaler"]
    x_scaled = scaler.transform(np.asarray(q_p_eff, dtype=np.float64).reshape(1, -1))
    if use_custom_predict:
        return _predict_gp_custom(gp_model, x_scaled)
    return np.asarray(gp_model.predict(x_scaled), dtype=np.float64).reshape(-1)


def _build_local_selector_coords_gpr(
    *,
    k,
    state,
    u0_k,
    V_k,
    model_k,
    n_primary,
    selector_mode,
    use_custom_predict=True,
    q_primary_hint=None,
):
    """
    Build reduced coordinates used by local cluster selection for POD-GPR.

    - linear mode: full linear projection V_k^T (state - u0_k)
    - nonlinear mode: [q_p, q_s(q_p), 0, ..., 0], where q_s is GPR-predicted
      and q_p comes from q_primary_hint when provided (otherwise from projection).
    """
    q_lin_full = (V_k.T @ (state - u0_k)).reshape(-1)
    if selector_mode == "linear":
        return q_lin_full

    r_k = V_k.shape[1]
    n_total_k = _cluster_total_dim(model_k, r_k)
    n_primary_k = min(int(n_primary), n_total_k)
    n_secondary_k = int(model_k.get("n_secondary", 0))
    has_gpr = bool(model_k.get("has_gpr", False))

    q_sel = np.zeros(r_k, dtype=np.float64)

    if q_primary_hint is None:
        q_p_eff = q_lin_full[:n_primary_k].copy()
    else:
        q_primary_hint = np.asarray(q_primary_hint, dtype=np.float64).reshape(-1)
        q_p_eff = np.zeros(n_primary_k, dtype=np.float64)
        ncopy = min(q_primary_hint.size, n_primary_k)
        q_p_eff[:ncopy] = q_primary_hint[:ncopy]

    q_sel[:n_primary_k] = q_p_eff

    if has_gpr and n_secondary_k > 0 and n_total_k > int(n_primary):
        n_secondary_eff = min(n_secondary_k, max(0, n_total_k - n_primary_k))
        if n_secondary_eff > 0:
            q_s_pred = _predict_local_secondary_coords(
                model_k=model_k,
                q_p_eff=q_p_eff,
                use_custom_predict=use_custom_predict,
            )
            q_sel[n_primary_k : n_primary_k + n_secondary_eff] = q_s_pred[:n_secondary_eff]

    return q_sel


def _gauss_newton_decoder_inverse(
    q_init,
    target_state,
    decode_func,
    jac_func,
    max_its=20,
    tol_rel=1e-12,
    verbose=False,
    tag="GN-invert",
):
    """
    Solve approximately

        min_q || decode_func(q) - target_state ||_2^2

    by Gauss-Newton on the decoder reconstruction error.
    """
    q = np.asarray(q_init, dtype=np.float64).copy()
    target_state = np.asarray(target_state, dtype=np.float64).reshape(-1)

    w_rec = decode_func(q)
    r0 = np.linalg.norm(w_rec - target_state)
    if r0 == 0.0:
        return q

    r = r0
    it_gn = 0
    while (r / r0 > tol_rel) and (it_gn < max_its):
        Jf = jac_func(q)
        res_rec = w_rec - target_state

        JTJ = Jf.T @ Jf
        JTr = Jf.T @ res_rec

        try:
            dq = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            dq, *_ = np.linalg.lstsq(JTJ, JTr, rcond=None)

        q -= dq
        w_rec = decode_func(q)
        r = np.linalg.norm(w_rec - target_state)
        it_gn += 1

    if verbose:
        print(f"[{tag}] it={it_gn}, rel={r/r0:.2e}")

    return q


# ============================================================================
# GP decoders / tangents
# FULL decoder convention: returns affine full state
# ============================================================================

def decode_gp(
    q_p,
    gp_model,
    basis,
    basis2,
    scaler,
    u_ref=None,
    use_custom_predict=True,
    echo_level=0,
):
    """
    Global POD-GPR affine decoder:

        w(q_p) = u_ref + U_p q_p + U_s q_s(q_p)

    where q_s(q_p) is predicted by a trained Gaussian Process regressor.

    Parameters
    ----------
    q_p : ndarray of shape (r_p,)
        Primary reduced coordinates.
    gp_model : GaussianProcessRegressor
        Multi-output GP mapping q_p -> q_s.
    basis : ndarray of shape (N, r_p)
        Primary POD basis U_p.
    basis2 : ndarray of shape (N, r_s)
        Secondary POD basis U_s.
    scaler : fitted scaler
        Scaler used for GP inputs.
    u_ref : ndarray or None
        Affine reference state. If None, zero reference is used.
    use_custom_predict : bool
        If True, evaluate GP prediction manually via kernel-vector times alpha.
    """
    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, basis.shape[0])

    x_in = q_p.reshape(1, -1)
    x_scaled = scaler.transform(x_in)

    t0 = time.time()

    if not use_custom_predict:
        q_s_pred = np.asarray(gp_model.predict(x_scaled), dtype=np.float64).ravel()
    else:
        q_s_pred = _predict_gp_custom(gp_model, x_scaled)

    if echo_level > 0:
        print(f"[decode_gp] Time to predict q_s: {time.time() - t0:.6f} s")

    return u_ref + basis @ q_p + basis2 @ q_s_pred


def _normalize_length_scale(length_scale, ndim):
    ls = np.asarray(length_scale, dtype=np.float64).reshape(-1)
    if ls.size == 1:
        ls = np.full(ndim, float(ls[0]), dtype=np.float64)
    if ls.size != ndim:
        raise ValueError(
            f"length_scale has size {ls.size}, expected 1 or ndim={ndim}."
        )
    return ls


def matern15_grad(x_scaled, X_train, length_scale, cval):
    """
    Vectorized gradient of the Matérn(ν=1.5) kernel with respect to x_scaled.
    """
    sqrt3 = np.sqrt(3.0)
    diff = x_scaled[None, :] - X_train
    ls = _normalize_length_scale(length_scale, diff.shape[1])
    inv_l2 = 1.0 / (ls * ls)
    ratio = np.sqrt(np.sum((diff * diff) * inv_l2[None, :], axis=1))
    exp_term = np.exp(-sqrt3 * ratio)
    grad_k = -3.0 * float(cval) * exp_term[:, None] * diff * inv_l2[None, :]
    return grad_k


def rbf_grad(x_scaled, X_train, length_scale, cval):
    """
    Vectorized gradient of the RBF kernel with respect to x_scaled.
    """
    diff = x_scaled[None, :] - X_train
    ls = _normalize_length_scale(length_scale, diff.shape[1])
    inv_l2 = 1.0 / (ls * ls)
    scaled_sq = np.sum((diff * diff) * inv_l2[None, :], axis=1)
    k_vec = float(cval) * np.exp(-0.5 * scaled_sq)
    grad_k = -k_vec[:, None] * diff * inv_l2[None, :]
    return grad_k


def jac_gp(
    q_p,
    gp_model,
    basis,
    basis2,
    scaler,
    echo_level=0,
):
    """
    Analytical tangent matrix for POD-GPR with supported kernels:
    ConstantKernel*Matern(ν=1.5) or ConstantKernel*RBF.

        dw/dq_p = U_p + U_s (dq_s/dq_p)

    Notes
    -----
    If your trained GP uses a different kernel, this analytical Jacobian is not valid.
    """
    t0 = time.time()

    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)

    x_scaled = scaler.transform(q_p.reshape(1, -1)).ravel()
    scale_factors = scaler.scale_

    X_train = gp_model.X_train_
    alpha = np.asarray(gp_model.alpha_, dtype=np.float64)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    kind, cval, length_scale = _gp_analytic_kernel_info(gp_model)
    if kind is None:
        kernel_obj = getattr(gp_model, "kernel_", getattr(gp_model, "kernel", None))
        raise ValueError(
            "jac_gp supports only ConstantKernel*(Matern(nu=1.5) or RBF). "
            f"Found kernel: {kernel_obj}"
        )

    if kind == "matern15":
        grad_k = matern15_grad(x_scaled, X_train, length_scale, cval)
    else:
        grad_k = rbf_grad(x_scaled, X_train, length_scale, cval)
    y_mean, y_std = _gp_target_stats(gp_model, alpha.shape[1])
    del y_mean

    dq_s_dx_scaled = (alpha.T @ grad_k) * y_std[:, None]
    dq_s_dx_real = dq_s_dx_scaled * scale_factors

    full_jac = basis + basis2 @ dq_s_dx_real

    if echo_level > 0:
        print(f"[jac_gp] Total analytical Jacobian time: {time.time() - t0:.6f} s")

    return full_jac


def jac_gp_forward_difference(
    q_p,
    gp_model,
    basis,
    basis2,
    scaler,
    fd_eps=1e-6,
    echo_level=0,
    use_custom_predict=True,
):
    """
    Forward-difference approximation of the POD-GPR tangent matrix.
    Useful as a fallback if the analytical kernel assumptions do not hold.
    """
    total_start_time = time.time()

    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)

    step1_start = time.time()
    x_in = q_p.reshape(1, -1)
    x_scaled = scaler.transform(x_in)
    scale_factors = scaler.scale_
    step1_time = time.time() - step1_start

    step2_start = time.time()
    if not use_custom_predict:
        q_s_base = np.asarray(gp_model.predict(x_scaled), dtype=np.float64).ravel()
    else:
        q_s_base = _predict_gp_custom(gp_model, x_scaled)
    step2_time = time.time() - step2_start

    step3_start = time.time()
    r_p = x_in.shape[1]
    r_s = q_s_base.size
    dq_s_dxp = np.zeros((r_s, r_p))

    iteration_times = []

    if use_custom_predict:
        X_train_ = gp_model.X_train_
        alpha_ = np.asarray(gp_model.alpha_, dtype=np.float64)
        kernel_ = gp_model.kernel_

    for j in range(r_p):
        iter_start = time.time()
        x_plus = x_scaled.copy()
        x_plus[0, j] += fd_eps

        if not use_custom_predict:
            q_s_plus = np.asarray(gp_model.predict(x_plus), dtype=np.float64).ravel()
        else:
            k_vec_plus = kernel_(X_train_, x_plus).ravel()
            q_s_plus = np.asarray(k_vec_plus @ alpha_, dtype=np.float64).reshape(-1)
            y_mean, y_std = _gp_target_stats(gp_model, q_s_plus.size)
            q_s_plus = y_mean + y_std * q_s_plus

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        dq_s_scaled = (q_s_plus - q_s_base) / fd_eps
        dq_s_dxp[:, j] = dq_s_scaled * scale_factors[j]

    step3_time = time.time() - step3_start

    step4_start = time.time()
    full_jac = basis + basis2 @ dq_s_dxp
    step4_time = time.time() - step4_start

    total_time = time.time() - total_start_time
    if echo_level > 0:
        print("[jac_gp_forward_difference] Timing breakdown:")
        print(f"  Step 1 (reshape/scale): {step1_time:.6f} s")
        print(f"  Step 2 (baseline predict): {step2_time:.6f} s")
        print(f"  Step 3 (loop + forward diffs): {step3_time:.6f} s")
        for i, t_iter in enumerate(iteration_times):
            print(f"    - iteration {i}: {t_iter:.6f} s")
        print(f"  Step 4 (combine w/ bases): {step4_time:.6f} s")
        print(f"  Total Jacobian time: {total_time:.6f} s")

    return full_jac


def jac_gp_central_difference(
    q_p,
    gp_model,
    basis,
    basis2,
    scaler,
    fd_eps=1e-6,
    echo_level=0,
    use_custom_predict=True,
):
    """
    Central-difference approximation of the POD-GPR tangent matrix.
    More expensive than forward difference, but usually more accurate.
    """
    t0 = time.time()

    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)

    x_in = q_p.reshape(1, -1)
    x_scaled = scaler.transform(x_in)
    scale_factors = scaler.scale_

    if use_custom_predict:
        X_train_ = gp_model.X_train_
        alpha_ = np.asarray(gp_model.alpha_, dtype=np.float64)
        kernel_ = gp_model.kernel_

    baseline_start = time.time()
    if not use_custom_predict:
        q_s_base = np.asarray(gp_model.predict(x_scaled), dtype=np.float64).ravel()
    else:
        q_s_base = _predict_gp_custom(gp_model, x_scaled)
    baseline_time = time.time() - baseline_start

    r_p = x_scaled.shape[1]
    dq_s_dxp = np.zeros((q_s_base.size, r_p))

    loop_start = time.time()
    half_eps = 0.5 * fd_eps
    for j in range(r_p):
        x_plus = x_scaled.copy()
        x_minus = x_scaled.copy()

        x_plus[0, j] += half_eps
        x_minus[0, j] -= half_eps

        if not use_custom_predict:
            q_s_plus = np.asarray(gp_model.predict(x_plus), dtype=np.float64).ravel()
            q_s_minus = np.asarray(gp_model.predict(x_minus), dtype=np.float64).ravel()
        else:
            k_vec_plus = kernel_(X_train_, x_plus).ravel()
            q_s_plus = np.asarray(k_vec_plus @ alpha_, dtype=np.float64).reshape(-1)
            k_vec_minus = kernel_(X_train_, x_minus).ravel()
            q_s_minus = np.asarray(k_vec_minus @ alpha_, dtype=np.float64).reshape(-1)
            y_mean, y_std = _gp_target_stats(gp_model, q_s_plus.size)
            q_s_plus = y_mean + y_std * q_s_plus
            q_s_minus = y_mean + y_std * q_s_minus

        dq_s_dxp_scaled = (q_s_plus - q_s_minus) / (2.0 * half_eps)
        dq_s_dxp[:, j] = dq_s_dxp_scaled * scale_factors[j]

    loop_time = time.time() - loop_start

    combine_start = time.time()
    full_jac = basis + basis2 @ dq_s_dxp
    combine_time = time.time() - combine_start

    total_time = time.time() - t0
    if echo_level > 0:
        print("[jac_gp_central_difference] Timing breakdown:")
        print(f"  Step 2 (baseline predict): {baseline_time:.6f} s")
        print(f"  Step 3 (central diff loop): {loop_time:.6f} s")
        print(f"  Step 4 (combine w/ bases): {combine_time:.6f} s")
        print(f"  Total time: {total_time:.6f} s")

    return full_jac


# ============================================================================
# ECSW training matrices
# ============================================================================

def compute_ECSW_training_matrix_2D_gpr(
    snaps,
    prev_snaps,
    basis,
    basis2,
    gp_model,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    scaler,
    u_ref=None,
    use_custom_predict=True,
    jacobian_mode="analytic",
    fd_eps=1e-6,
    max_local_its=10,
    local_tol=1e-2,
):
    """
    ECSW training matrix for the POD-GPR ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    n_red = basis.shape[1]
    u_ref = _prepare_reference(u_ref, n_tot)

    C = np.zeros((n_red * n_snaps, n_hdm))

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    def jac_gp_train(q):
        if jacobian_mode == "analytic":
            return jac_gp(
                q_p=q,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                echo_level=0,
            )
        if jacobian_mode == "forward_fd":
            return jac_gp_forward_difference(
                q_p=q,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                fd_eps=fd_eps,
                echo_level=0,
                use_custom_predict=use_custom_predict,
            )
        if jacobian_mode == "central_fd":
            return jac_gp_central_difference(
                q_p=q,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                fd_eps=fd_eps,
                echo_level=0,
                use_custom_predict=use_custom_predict,
            )
        raise ValueError(f"Unsupported jacobian_mode: {jacobian_mode}")

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]

        y = (basis.T @ (snap - u_ref)).copy()

        w_rec = decode_gp(
            q_p=y,
            gp_model=gp_model,
            basis=basis,
            basis2=basis2,
            scaler=scaler,
            u_ref=u_ref,
            use_custom_predict=use_custom_predict,
            echo_level=0,
        )
        init_res = np.linalg.norm(w_rec - snap)
        curr_res = init_res
        num_it = 0

        snap_norm = np.linalg.norm(snap)
        denom = snap_norm if snap_norm > 0.0 else 1.0
        print("Initial reconstruction residual: {:3.2e}".format(init_res / denom))

        while init_res > 0.0 and (curr_res / init_res > local_tol) and num_it < max_local_its:
            Jf = jac_gp_train(y)

            res_rec = w_rec - snap
            JJ = Jf.T @ Jf
            Jr = Jf.T @ res_rec

            dy, *_ = np.linalg.lstsq(JJ, Jr, rcond=None)
            y -= dy

            w_rec = decode_gp(
                q_p=y,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                u_ref=u_ref,
                use_custom_predict=use_custom_predict,
                echo_level=0,
            )
            curr_res = np.linalg.norm(w_rec - snap)
            num_it += 1

        final_res = np.linalg.norm(w_rec - snap)
        print("Final reconstruction residual: {:3.2e}".format(final_res / denom))

        ires = res(w_rec, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_rec, dt, JDxec, JDyec, Eye)

        V = jac_gp_train(y)
        Wi = Ji @ V

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_gpr_local(
    snaps,
    prev_snaps,
    u0_list,
    V_list,
    models,
    n_primary,
    d_const,
    g_list,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
    max_gn_its=20,
    tol_rel=1e-2,
    init_cluster=None,
    selector_mode="linear",
):
    """
    ECSW training matrix for the local POD-GPR ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    K = len(V_list)
    selector_mode = _resolve_local_selector_mode(selector_mode)

    r_dof = []
    for k in range(K):
        V_k = np.asarray(V_list[k], dtype=np.float64)
        r_dof.append(_cluster_primary_dim(models[k], V_k.shape[1], n_primary))

    r_max = max(r_dof)
    C = np.zeros((r_max * n_snaps, n_hdm), dtype=np.float64)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        u_i = snaps[:, isnap]
        u_prev = prev_snaps[:, isnap]

        if init_cluster is None:
            k0 = int(np.argmin([np.linalg.norm(u_i - u0_k) ** 2 for u0_k in u0_list]))
        else:
            k0 = int(init_cluster)
            if not (0 <= k0 < K):
                raise ValueError(f"init_cluster={k0} out of range [0, {K-1}]")

        V_k0 = np.asarray(V_list[k0], dtype=np.float64)
        u0_k0 = np.asarray(u0_list[k0], dtype=np.float64)
        y_k0 = _build_local_selector_coords_gpr(
            k=k0,
            state=u_i,
            u0_k=u0_k0,
            V_k=V_k0,
            model_k=models[k0],
            n_primary=n_primary,
            selector_mode=selector_mode,
            use_custom_predict=use_custom_predict,
            q_primary_hint=None,
        )
        k = select_cluster_reduced(k0, y_k0, d_const, g_list)

        V_k = np.asarray(V_list[k], dtype=np.float64)
        u0_k = np.asarray(u0_list[k], dtype=np.float64)
        n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

        q = (V_k.T @ (u_i - u0_k))[:n_dof_k].copy()

        w_rec = decode_gpr_local(
            k,
            q,
            u0_list,
            V_list,
            models,
            n_primary,
            use_custom_predict=use_custom_predict,
        )
        r_rec = w_rec - u_i

        init_norm = np.linalg.norm(r_rec)
        curr_norm = init_norm
        num_it = 0
        u_norm = np.linalg.norm(u_i)
        denom = u_norm if u_norm > 0.0 else 1.0
        print("Initial residual: {:3.2e}".format(init_norm / denom))

        if init_norm > 0.0:
            while curr_norm / init_norm > tol_rel and num_it < max_gn_its:
                Jf = jac_gpr_local(
                    k,
                    q,
                    V_list,
                    models,
                    n_primary,
                    use_custom_predict=use_custom_predict,
                    jacobian_mode=jacobian_mode,
                    fd_eps=fd_eps,
                )

                JTJ = Jf.T @ Jf
                JTr = Jf.T @ r_rec

                dq, *_ = np.linalg.lstsq(JTJ, -JTr, rcond=None)
                q += dq

                w_rec = decode_gpr_local(
                    k,
                    q,
                    u0_list,
                    V_list,
                    models,
                    n_primary,
                    use_custom_predict=use_custom_predict,
                )
                r_rec = w_rec - u_i
                curr_norm = np.linalg.norm(r_rec)
                num_it += 1

        print("Final residual: {:3.2e}".format(curr_norm / denom))

        u_tilde = w_rec

        ires = res(u_tilde, grid_x, grid_y, dt, u_prev, mu, Dxec, Dyec)
        Ji = jac(u_tilde, dt, JDxec, JDyec, Eye)

        V_q = jac_gpr_local(
            k,
            q,
            V_list,
            models,
            n_primary,
            use_custom_predict=use_custom_predict,
            jacobian_mode=jacobian_mode,
            fd_eps=fd_eps,
        )
        Wi = Ji @ V_q

        row0 = isnap * r_max
        row1 = row0 + n_dof_k

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_gp(*args, **kwargs):
    """
    Backward-compatible alias for older naming.
    """
    return compute_ECSW_training_matrix_2D_gpr(*args, **kwargs)


# ============================================================================
# Global ROM
# ============================================================================

def inviscid_burgers_implicit2D_LSPG_pod_gpr(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    basis2,
    gp_model,
    scaler,
    u_ref=None,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_ic=20,
    tol_ic=1e-12,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Global POD-GPR manifold ROM for the 2D inviscid Burgers equation:

        w(q_p) = u_ref + U_p q_p + U_s q_s(q_p)

    Parameters
    ----------
    jacobian_mode : {"auto", "analytic", "forward_fd", "central_fd"}
        Choice of tangent approximation for dq_s/dq_p.
        "analytic" assumes ConstantKernel * Matern(nu=1.5).
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, w0.size)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    mode = _resolve_global_jacobian_mode(jacobian_mode, gp_model)

    def decode_func(q):
        return decode_gp(
            q_p=q,
            gp_model=gp_model,
            basis=basis,
            basis2=basis2,
            scaler=scaler,
            u_ref=u_ref,
            use_custom_predict=use_custom_predict,
            echo_level=0,
        )

    def jac_gp_func(q):
        if mode == "analytic":
            return jac_gp(
                q_p=q,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                echo_level=0,
            )
        if mode == "forward_fd":
            return jac_gp_forward_difference(
                q_p=q,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                fd_eps=fd_eps,
                echo_level=0,
                use_custom_predict=use_custom_predict,
            )
        if mode == "central_fd":
            return jac_gp_central_difference(
                q_p=q,
                gp_model=gp_model,
                basis=basis,
                basis2=basis2,
                scaler=scaler,
                fd_eps=fd_eps,
                echo_level=0,
                use_custom_predict=use_custom_predict,
            )

        raise ValueError(f"Unsupported jacobian_mode: {mode}")

    q0_guess = basis.T @ (w0 - u_ref)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0,
        decode_func=decode_func,
        jac_func=jac_gp_func,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="POD-GPR-IC",
    )

    w_init = decode_func(q0)

    nred = q0.size
    snaps = np.zeros((w0.size, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    snaps[:, 0] = w_init
    red_coords[:, 0] = q0

    wp = w_init.copy()
    qp = q0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-GPR ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for it in range(num_steps):
        print(f" ... Working on timestep {it}")

        def res(w):
            return inviscid_burgers_res2D(
                w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec
            )

        def jac(w):
            return inviscid_burgers_exact_jac2D(
                w, dt, JDxec, JDyec, Eye
            )

        q, resnorms, times = gauss_newton_pod_gpr(
            func=res,
            jac=jac,
            y0=qp,
            decode_gp=decode_func,
            jac_gp=jac_gp_func,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w = decode_func(q)

        red_coords[:, it + 1] = q
        snaps[:, it + 1] = w

        wp = w.copy()
        qp = q.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)


# ============================================================================
# Local ROM
# ============================================================================

def decode_gpr_local(
    k,
    q_p,
    u0_list,
    V_list,
    models,
    n_primary,
    use_custom_predict=True,
):
    """
    Local POD-GPR affine decoder for cluster k.
    """
    u0_k = np.asarray(u0_list[k], dtype=np.float64).reshape(-1)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    model_k = models[k]

    r_k = V_k.shape[1]
    n_total_k = _cluster_total_dim(model_k, r_k)
    n_secondary_k = int(model_k.get("n_secondary", 0))
    has_gpr = bool(model_k.get("has_gpr", False))

    if (not has_gpr) or (n_secondary_k <= 0) or (n_total_k <= int(n_primary)):
        q_full = np.zeros(r_k, dtype=np.float64)
        q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
        ncopy = min(q_p.size, n_total_k)
        q_full[:ncopy] = q_p[:ncopy]
        return u0_k + V_k @ q_full

    n_primary_k = min(int(n_primary), n_total_k)
    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    q_p_eff = np.zeros(n_primary_k, dtype=np.float64)
    ncopy = min(q_p.size, n_primary_k)
    q_p_eff[:ncopy] = q_p[:ncopy]

    return decode_gp(
        q_p=q_p_eff,
        gp_model=model_k["gpr_model"],
        basis=V_k[:, :n_primary_k],
        basis2=V_k[:, n_primary_k:n_total_k],
        scaler=model_k["scaler"],
        u_ref=u0_k,
        use_custom_predict=use_custom_predict,
        echo_level=0,
    )


def jac_gpr_local(
    k,
    q_p,
    V_list,
    models,
    n_primary,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
):
    """
    Local POD-GPR tangent matrix for cluster k.
    """
    V_k = np.asarray(V_list[k], dtype=np.float64)
    model_k = models[k]

    r_k = V_k.shape[1]
    n_total_k = _cluster_total_dim(model_k, r_k)
    n_secondary_k = int(model_k.get("n_secondary", 0))
    has_gpr = bool(model_k.get("has_gpr", False))

    if (not has_gpr) or (n_secondary_k <= 0) or (n_total_k <= int(n_primary)):
        n_dof_k = _cluster_primary_dim(model_k, r_k, n_primary)
        return V_k[:, :n_dof_k]

    n_primary_k = min(int(n_primary), n_total_k)
    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    q_p_eff = np.zeros(n_primary_k, dtype=np.float64)
    ncopy = min(q_p.size, n_primary_k)
    q_p_eff[:ncopy] = q_p[:ncopy]

    basis = V_k[:, :n_primary_k]
    basis2 = V_k[:, n_primary_k:n_total_k]
    gp_model = model_k["gpr_model"]
    scaler = model_k["scaler"]

    mode = _resolve_local_jacobian_mode(jacobian_mode, model_k)
    if mode == "analytic":
        return jac_gp(
            q_p=q_p_eff,
            gp_model=gp_model,
            basis=basis,
            basis2=basis2,
            scaler=scaler,
            echo_level=0,
        )
    if mode == "forward_fd":
        return jac_gp_forward_difference(
            q_p=q_p_eff,
            gp_model=gp_model,
            basis=basis,
            basis2=basis2,
            scaler=scaler,
            fd_eps=fd_eps,
            echo_level=0,
            use_custom_predict=use_custom_predict,
        )
    if mode == "central_fd":
        return jac_gp_central_difference(
            q_p=q_p_eff,
            gp_model=gp_model,
            basis=basis,
            basis2=basis2,
            scaler=scaler,
            fd_eps=fd_eps,
            echo_level=0,
            use_custom_predict=use_custom_predict,
        )

    raise ValueError(f"Unsupported local jacobian mode: {mode}")


def inviscid_burgers_implicit2D_LSPG_local_pod_gpr(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    u0_list,
    V_list,
    models,
    n_primary,
    d_const,
    g_list,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    verbose=True,
    init_cluster=None,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-10,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
    selector_mode="linear",
):
    """
    Local POD-GPR manifold ROM for the 2D inviscid Burgers equation.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    N = w0.size
    K = len(V_list)
    selector_mode = _resolve_local_selector_mode(selector_mode)

    if d_const.shape != (K, K):
        raise ValueError(f"d_const has shape {d_const.shape}, expected ({K}, {K})")

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    def select_cluster_by_u0(w, u0_list_):
        d2 = [np.linalg.norm(w - np.asarray(u0_k, dtype=np.float64)) ** 2 for u0_k in u0_list_]
        return int(np.argmin(d2))

    n_max = 0
    for k in range(K):
        V_k = np.asarray(V_list[k], dtype=np.float64)
        n_max = max(n_max, _cluster_primary_dim(models[k], V_k.shape[1], n_primary))

    snaps = np.zeros((N, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((n_max, num_steps + 1), dtype=np.float64)
    cluster_history = []

    if init_cluster is None:
        k = select_cluster_by_u0(w0, u0_list)
    else:
        k = int(init_cluster)
        if not (0 <= k < K):
            raise ValueError(f"init_cluster={k} is out of range [0, {K-1}]")

    if verbose:
        print(f"[LOCAL-POD-GPR] Initial cluster k = {k} / {K-1}")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

    def decode_local(q):
        return decode_gpr_local(
            k=k,
            q_p=q,
            u0_list=u0_list,
            V_list=V_list,
            models=models,
            n_primary=n_primary,
            use_custom_predict=use_custom_predict,
        )

    def jac_local(q):
        return jac_gpr_local(
            k=k,
            q_p=q,
            V_list=V_list,
            models=models,
            n_primary=n_primary,
            use_custom_predict=use_custom_predict,
            jacobian_mode=jacobian_mode,
            fd_eps=fd_eps,
        )

    q0_guess = V_k[:, :n_dof_k].T @ (w0 - u0_k)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0,
        decode_func=decode_local,
        jac_func=jac_local,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="LOCAL-POD-GPR-IC",
    )

    w_init = decode_local(q0)

    snaps[:, 0] = w_init
    red_coords[:n_dof_k, 0] = q0

    wp = w_init.copy()
    qp = q0.copy()
    cluster_history.append(k)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    if verbose:
        print(f"[LOCAL-POD-GPR] Running local POD-GPR with {K} clusters, dt={dt}")

    for it in range(num_steps):
        if verbose:
            print(f"[LOCAL-POD-GPR] Timestep {it}/{num_steps}")

        q_full_k = _build_local_selector_coords_gpr(
            k=k,
            state=wp,
            u0_k=u0_k,
            V_k=V_k,
            model_k=models[k],
            n_primary=n_primary,
            selector_mode=selector_mode,
            use_custom_predict=use_custom_predict,
            q_primary_hint=qp if selector_mode == "nonlinear" else None,
        )
        k_new = select_cluster_reduced(k, q_full_k, d_const, g_list)

        if k_new != k:
            if verbose:
                print(f"  -> Cluster switch: {k} -> {k_new}")

            k = k_new
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

            qp_guess = V_k[:, :n_dof_k].T @ (wp - u0_k)
            qp = _gauss_newton_decoder_inverse(
                q_init=qp_guess,
                target_state=wp,
                decode_func=decode_local,
                jac_func=jac_local,
                max_its=max_its_ic,
                tol_rel=tol_ic,
                verbose=False,
                tag="LOCAL-POD-GPR-SWITCH",
            )

        cluster_history.append(k)

        def compute_residual(w):
            return inviscid_burgers_res2D(
                w,
                grid_x,
                grid_y,
                dt,
                wp,
                mu,
                Dxec,
                Dyec,
            )

        def compute_jacobian(w):
            return inviscid_burgers_exact_jac2D(
                w,
                dt,
                JDxec,
                JDyec,
                Eye,
            )

        q, resnorms, times = gauss_newton_pod_gpr(
            func=compute_residual,
            jac=compute_jacobian,
            y0=qp,
            decode_gp=decode_local,
            jac_gp=jac_local,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w = decode_local(q)

        snaps[:, it + 1] = w
        red_coords[:n_dof_k, it + 1] = q

        wp = w.copy()
        qp = q.copy()

    stats = {
        "num_its": num_its,
        "jac_time": jac_time,
        "res_time": res_time,
        "ls_time": ls_time,
        "cluster_history": cluster_history,
        "red_coords": red_coords,
    }

    return snaps, stats


# ============================================================================
# Local ECSW ROM
# ============================================================================

def decode_gpr_local_ecsw(
    k,
    q_p,
    u0_loc_list,
    V_loc_list,
    models,
    n_primary,
    use_custom_predict=True,
):
    """
    ECSW-restricted local POD-GPR affine decoder.
    """
    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    u0_loc_k = np.asarray(u0_loc_list[k], dtype=np.float64).reshape(-1)
    V_loc_k = np.asarray(V_loc_list[k], dtype=np.float64)
    model_k = models[k]

    _, r_k = V_loc_k.shape
    n_total_k = _cluster_total_dim(model_k, r_k)
    n_secondary_k = int(model_k.get("n_secondary", 0))
    has_gpr = bool(model_k.get("has_gpr", False))

    if (not has_gpr) or (n_secondary_k <= 0) or (n_total_k <= int(n_primary)):
        n_dof_k = _cluster_primary_dim(model_k, r_k, n_primary)
        if q_p.size != n_dof_k:
            raise ValueError(
                f"[decode_gpr_local_ecsw] linear cluster {k}: q_p.size={q_p.size}, expected {n_dof_k}"
            )
        return u0_loc_k + V_loc_k[:, :n_dof_k] @ q_p

    n_primary_k = min(int(n_primary), n_total_k)
    if q_p.size != n_primary_k:
        raise ValueError(
            f"[decode_gpr_local_ecsw] GPR cluster {k}: q_p.size={q_p.size}, expected {n_primary_k}"
        )

    return decode_gp(
        q_p=q_p,
        gp_model=model_k["gpr_model"],
        basis=V_loc_k[:, :n_primary_k],
        basis2=V_loc_k[:, n_primary_k:n_total_k],
        scaler=model_k["scaler"],
        u_ref=u0_loc_k,
        use_custom_predict=use_custom_predict,
        echo_level=0,
    )


def jac_gpr_local_ecsw(
    k,
    q_p,
    V_loc_list,
    models,
    n_primary,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
):
    """
    ECSW-restricted local POD-GPR tangent matrix.
    """
    q_p = np.asarray(q_p, dtype=np.float64).reshape(-1)
    V_loc_k = np.asarray(V_loc_list[k], dtype=np.float64)
    model_k = models[k]

    _, r_k = V_loc_k.shape
    n_total_k = _cluster_total_dim(model_k, r_k)
    n_secondary_k = int(model_k.get("n_secondary", 0))
    has_gpr = bool(model_k.get("has_gpr", False))

    if (not has_gpr) or (n_secondary_k <= 0) or (n_total_k <= int(n_primary)):
        n_dof_k = _cluster_primary_dim(model_k, r_k, n_primary)
        if q_p.size != n_dof_k:
            raise ValueError(
                f"[jac_gpr_local_ecsw] linear cluster {k}: q_p.size={q_p.size}, expected {n_dof_k}"
            )
        return V_loc_k[:, :n_dof_k]

    n_primary_k = min(int(n_primary), n_total_k)
    if q_p.size != n_primary_k:
        raise ValueError(
            f"[jac_gpr_local_ecsw] GPR cluster {k}: q_p.size={q_p.size}, expected {n_primary_k}"
        )

    basis_loc = V_loc_k[:, :n_primary_k]
    basis2_loc = V_loc_k[:, n_primary_k:n_total_k]
    gp_model = model_k["gpr_model"]
    scaler = model_k["scaler"]

    mode = _resolve_local_jacobian_mode(jacobian_mode, model_k)
    if mode == "analytic":
        return jac_gp(
            q_p=q_p,
            gp_model=gp_model,
            basis=basis_loc,
            basis2=basis2_loc,
            scaler=scaler,
            echo_level=0,
        )
    if mode == "forward_fd":
        return jac_gp_forward_difference(
            q_p=q_p,
            gp_model=gp_model,
            basis=basis_loc,
            basis2=basis2_loc,
            scaler=scaler,
            fd_eps=fd_eps,
            echo_level=0,
            use_custom_predict=use_custom_predict,
        )
    if mode == "central_fd":
        return jac_gp_central_difference(
            q_p=q_p,
            gp_model=gp_model,
            basis=basis_loc,
            basis2=basis2_loc,
            scaler=scaler,
            fd_eps=fd_eps,
            echo_level=0,
            use_custom_predict=use_custom_predict,
        )

    raise ValueError(f"Unsupported local ECSW jacobian mode: {mode}")


def inviscid_burgers_implicit2D_LSPG_local_pod_gpr_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    u0_list,
    V_list,
    models,
    n_primary,
    d_const,
    g_list,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    verbose=True,
    init_cluster=None,
    use_custom_predict=True,
    jacobian_mode="auto",
    fd_eps=1e-6,
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-10,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
    selector_mode="linear",
):
    """
    ECSW local POD-GPR manifold ROM for the 2D inviscid Burgers equation.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    selector_mode = _resolve_local_selector_mode(selector_mode)

    N_full = w0.size
    N_cells = N_full // 2
    K = len(V_list)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(N_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_loc = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_loc = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_loc = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]

    idx_cells = augmented_sample
    idx_dofs = np.concatenate((idx_cells, N_cells + idx_cells))

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])
    shp = (dy.size, dx.size)

    lbc = np.zeros(sample_inds.shape[0], dtype=np.float64)
    rows, cols = np.unravel_index(sample_inds, shp)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if c == 0:
            lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]

    src_line = dt * 0.02 * np.exp(mu[1] * xc)
    src_full = np.tile(src_line, dy.size)
    src = src_full[sample_inds]

    n_max = 0
    for k in range(K):
        V_k = np.asarray(V_list[k], dtype=np.float64)
        n_max = max(n_max, _cluster_primary_dim(models[k], V_k.shape[1], n_primary))

    snaps = np.zeros((N_full, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((n_max, num_steps + 1), dtype=np.float64)
    cluster_hist = []

    u0_loc_list = []
    V_loc_list = []
    for k in range(K):
        u0_k = np.asarray(u0_list[k], dtype=np.float64)
        V_k = np.asarray(V_list[k], dtype=np.float64)
        u0_loc_list.append(u0_k[idx_dofs])
        V_loc_list.append(V_k[idx_dofs, :])

    def select_cluster_by_u0(w, u0_list_):
        d2 = [np.linalg.norm(w - np.asarray(u0_k, dtype=np.float64)) ** 2 for u0_k in u0_list_]
        return int(np.argmin(d2))

    if init_cluster is None:
        k = select_cluster_by_u0(w0, u0_list)
    else:
        k = int(init_cluster)
        if not (0 <= k < K):
            raise ValueError(f"init_cluster={k} is out of range [0, {K-1}]")

    if verbose:
        print(f"[LOCAL-POD-GPR-ECSW] Initial cluster k = {k} / {K-1}")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

    def decode_loc(q):
        return decode_gpr_local_ecsw(
            k,
            q,
            u0_loc_list,
            V_loc_list,
            models,
            n_primary,
            use_custom_predict=use_custom_predict,
        )

    def jac_loc(q):
        return jac_gpr_local_ecsw(
            k,
            q,
            V_loc_list,
            models,
            n_primary,
            use_custom_predict=use_custom_predict,
            jacobian_mode=jacobian_mode,
            fd_eps=fd_eps,
        )

    q0_guess = V_k[:, :n_dof_k].T @ (w0 - u0_k)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0[idx_dofs],
        decode_func=decode_loc,
        jac_func=jac_loc,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="LOCAL-POD-GPR-ECSW-IC",
    )

    w_init = decode_gpr_local(
        k,
        q0,
        u0_list,
        V_list,
        models,
        n_primary,
        use_custom_predict=use_custom_predict,
    )

    snaps[:, 0] = w_init
    red_coords[:n_dof_k, 0] = q0

    wp_full = w_init.copy()
    qp = q0.copy()
    cluster_hist.append(k)

    num_its = 0.0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    if verbose:
        print(f"[LOCAL-POD-GPR-ECSW] Running local POD-GPR-ECSW with {K} clusters, dt={dt}")

    for it in range(num_steps):
        if verbose and (it % 10 == 0 or it == num_steps - 1):
            print(f"[LOCAL-POD-GPR-ECSW] Timestep {it}/{num_steps}")

        q_full_k = _build_local_selector_coords_gpr(
            k=k,
            state=wp_full,
            u0_k=u0_k,
            V_k=V_k,
            model_k=models[k],
            n_primary=n_primary,
            selector_mode=selector_mode,
            use_custom_predict=use_custom_predict,
            q_primary_hint=qp if selector_mode == "nonlinear" else None,
        )
        k_new = select_cluster_reduced(k, q_full_k, d_const, g_list)

        if k_new != k:
            if verbose:
                print(f"  [LOCAL-POD-GPR-ECSW] Cluster switch: {k} -> {k_new}")

            k = k_new
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

            qp_guess = V_k[:, :n_dof_k].T @ (wp_full - u0_k)
            qp = _gauss_newton_decoder_inverse(
                q_init=qp_guess,
                target_state=wp_full[idx_dofs],
                decode_func=decode_loc,
                jac_func=jac_loc,
                max_its=max_its_ic,
                tol_rel=tol_ic,
                verbose=False,
                tag="LOCAL-POD-GPR-ECSW-SWITCH",
            )

        cluster_hist.append(k)

        wp_loc = wp_full[idx_dofs]

        def res_loc(w_loc):
            return inviscid_burgers_res2D_ecsw(
                w_loc,
                grid_x,
                grid_y,
                dt,
                wp_loc,
                mu,
                JDxec_loc,
                JDyec_loc,
                sample_inds,
                augmented_sample,
                lbc,
                src,
            )

        def jac_loc_res(w_loc):
            return inviscid_burgers_exact_jac2D_ecsw(
                w_loc,
                dt,
                JDxec_loc,
                JDyec_loc,
                Eye_loc,
                sample_inds,
                augmented_sample,
            )

        q, resnorms, times = gauss_newton_pod_gpr_ecsw(
            func=res_loc,
            jac=jac_loc_res,
            y0=qp,
            decode_gp=decode_loc,
            jac_gp=jac_loc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weights=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_full = decode_gpr_local(
            k,
            q,
            u0_list,
            V_list,
            models,
            n_primary,
            use_custom_predict=use_custom_predict,
        )

        snaps[:, it + 1] = w_full
        red_coords[:n_dof_k, it + 1] = q

        wp_full = w_full.copy()
        qp = q.copy()

    stats = {
        "num_its": num_its,
        "jac_time": jac_time,
        "res_time": res_time,
        "ls_time": ls_time,
        "cluster_history": cluster_hist,
        "red_coords": red_coords,
    }

    return snaps, stats


# ============================================================================
# Global ECSW ROM
# ============================================================================

def inviscid_burgers_implicit2D_LSPG_pod_gpr_ecsw(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    basis2,
    gp_model,
    weights,
    scaler,
    u_ref=None,
    use_custom_predict=True,
    jacobian_mode="analytic",
    fd_eps=1e-6,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_ic=20,
    tol_ic=1e-12,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    ECSW global POD-GPR manifold ROM for the 2D inviscid Burgers equation.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref = _prepare_reference(u_ref, w0.size)

    N_full = w0.size
    N_cells = N_full // 2

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(N_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_loc = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_loc = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_loc = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]

    idx_cells = augmented_sample
    idx_dofs = np.concatenate((idx_cells, N_cells + idx_cells))

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])
    shp = (dy.size, dx.size)

    lbc = np.zeros(sample_inds.shape[0], dtype=np.float64)
    rows, cols = np.unravel_index(sample_inds, shp)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if c == 0:
            lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]

    src_line = dt * 0.02 * np.exp(mu[1] * xc)
    src_full = np.tile(src_line, dy.size)
    src = src_full[sample_inds]

    basis_loc = basis[idx_dofs, :]
    basis2_loc = basis2[idx_dofs, :]
    u_ref_loc = u_ref[idx_dofs]

    def decode_loc(q):
        return decode_gp(
            q_p=q,
            gp_model=gp_model,
            basis=basis_loc,
            basis2=basis2_loc,
            scaler=scaler,
            u_ref=u_ref_loc,
            use_custom_predict=use_custom_predict,
            echo_level=0,
        )

    def jac_gp_loc(q):
        if jacobian_mode == "analytic":
            return jac_gp(
                q_p=q,
                gp_model=gp_model,
                basis=basis_loc,
                basis2=basis2_loc,
                scaler=scaler,
                echo_level=0,
            )
        if jacobian_mode == "forward_fd":
            return jac_gp_forward_difference(
                q_p=q,
                gp_model=gp_model,
                basis=basis_loc,
                basis2=basis2_loc,
                scaler=scaler,
                fd_eps=fd_eps,
                echo_level=0,
                use_custom_predict=use_custom_predict,
            )
        if jacobian_mode == "central_fd":
            return jac_gp_central_difference(
                q_p=q,
                gp_model=gp_model,
                basis=basis_loc,
                basis2=basis2_loc,
                scaler=scaler,
                fd_eps=fd_eps,
                echo_level=0,
                use_custom_predict=use_custom_predict,
            )

        raise ValueError(f"Unsupported jacobian_mode: {jacobian_mode}")

    q0_guess = basis.T @ (w0 - u_ref)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0[idx_dofs],
        decode_func=decode_loc,
        jac_func=jac_gp_loc,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="POD-GPR-ECSW-IC",
    )

    nred = q0.size
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = q0

    w0_loc = decode_loc(q0)
    wp_loc = w0_loc.copy()
    qp = q0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-GPR ECSW ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for it in range(num_steps):
        print(f" ... Working on timestep {it}")

        def res_loc(w_loc):
            return inviscid_burgers_res2D_ecsw(
                w_loc,
                grid_x,
                grid_y,
                dt,
                wp_loc,
                mu,
                JDxec_loc,
                JDyec_loc,
                sample_inds,
                augmented_sample,
                lbc,
                src,
            )

        def jac_loc(w_loc):
            return inviscid_burgers_exact_jac2D_ecsw(
                w_loc,
                dt,
                JDxec_loc,
                JDyec_loc,
                Eye_loc,
                sample_inds,
                augmented_sample,
            )

        q, resnorms, times = gauss_newton_pod_gpr_ecsw(
            func=res_loc,
            jac=jac_loc,
            y0=qp,
            decode_gp=decode_loc,
            jac_gp=jac_gp_loc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weights=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_loc = decode_loc(q)

        red_coords[:, it + 1] = q
        wp_loc = w_loc.copy()
        qp = q.copy()

    return red_coords, (num_its, jac_time, res_time, ls_time)
