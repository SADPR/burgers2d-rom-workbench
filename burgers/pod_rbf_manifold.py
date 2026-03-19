# pod_rbf_manifold.py
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
from .cluster_utils import (
    select_cluster_reduced,
    select_cluster_reduced_trunc,
)
from .gauss_newton import (
    gauss_newton_pod_rbf,
    gauss_newton_pod_rbf_ecsw,
)
from .ecsw_utils import generate_augmented_mesh
from .rbf_utils import RBFUtils


# ============================================================================
# Small helpers
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


def _cluster_total_dim(model_k, r_k):
    """
    Total number of reduced coordinates retained in cluster k.
    """
    n_total_k = int(model_k.get("n_total", r_k))
    return min(n_total_k, r_k)


def _cluster_primary_dim(model_k, r_k, n_primary):
    """
    Number of active online coordinates in cluster k.
    If no nonlinear closure is active, this equals the active linear dimension.
    """
    n_total_k = _cluster_total_dim(model_k, r_k)
    has_rbf = bool(model_k.get("has_rbf", False))
    n_secondary = int(model_k.get("n_secondary", 0))

    if has_rbf and n_secondary > 0 and n_total_k > n_primary:
        return min(n_primary, n_total_k)

    return n_total_k


def _predict_rbf_secondary(
    q_p,
    q_p_train,
    W,
    epsilon,
    scaler,
    kernel_name,
    echo_level=0,
):
    """
    Predict secondary coordinates q_s(q_p).
    """
    q_p = np.asarray(q_p, dtype=float).reshape(-1)

    if kernel_name == "gaussian":
        return RBFUtils.interpolate_with_rbf_global_gaussian(
            q_p, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "imq":
        return RBFUtils.interpolate_with_rbf_global_imq(
            q_p, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "linear":
        return RBFUtils.interpolate_with_rbf_global_linear(
            q_p, q_p_train, W, scaler, echo_level=echo_level
        )
    if kernel_name == "multiquadric":
        return RBFUtils.interpolate_with_rbf_global_multiquadric(
            q_p, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "matern":
        return RBFUtils.interpolate_with_rbf_global_matern32(
            q_p, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )

    raise ValueError(f"Unsupported kernel type: {kernel_name}")


def _rbf_secondary_jacobian(
    q_p,
    q_p_train,
    W,
    epsilon,
    scaler,
    kernel_name,
    echo_level=0,
):
    """
    Compute dq_s/dq_p.
    """
    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    q_p_norm = scaler.transform(q_p.reshape(1, -1))

    if kernel_name == "gaussian":
        return RBFUtils.compute_rbf_jacobian_global_gaussian(
            q_p_norm, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "imq":
        return RBFUtils.compute_rbf_jacobian_global_imq(
            q_p_norm, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "linear":
        return RBFUtils.compute_rbf_jacobian_global_linear(
            q_p_norm, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "multiquadric":
        return RBFUtils.compute_rbf_jacobian_global_multiquadric(
            q_p_norm, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )
    if kernel_name == "matern":
        return RBFUtils.compute_rbf_jacobian_global_matern32(
            q_p_norm, q_p_train, W, epsilon, scaler, echo_level=echo_level
        )

    raise ValueError(f"Unsupported kernel type: {kernel_name}")


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
    Solve approximately:

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
# Global POD-RBF manifold helpers
# FULL decoder convention: returns the affine full state
# ============================================================================

def decode_rbf_global(
    q_p,
    W_global,
    q_p_train,
    basis,
    basis2,
    epsilon,
    scaler,
    kernel_type="gaussian",
    u_ref=None,
    echo_level=0,
):
    """
    Global POD-RBF affine decoder:

        w(q_p) = u_ref + U_p q_p + U_s q_s(q_p)

    If u_ref is None, zero reference is used.
    """
    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    basis = np.asarray(basis, dtype=float)
    basis2 = np.asarray(basis2, dtype=float)
    u_ref = _prepare_reference(u_ref, basis.shape[0])

    q_s_pred = _predict_rbf_secondary(
        q_p=q_p,
        q_p_train=np.asarray(q_p_train, dtype=float),
        W=np.asarray(W_global, dtype=float),
        epsilon=epsilon,
        scaler=scaler,
        kernel_name=kernel_type,
        echo_level=echo_level,
    )

    return u_ref + basis @ q_p + basis2 @ q_s_pred


def decode_rbf(
    q_p,
    W,
    q_p_train,
    basis,
    basis2,
    epsilon,
    scaler,
    kernel_type="gaussian",
    u_ref=None,
    echo_level=0,
):
    """
    Neutral-name alias of decode_rbf_global with identical behavior.
    """
    return decode_rbf_global(
        q_p=q_p,
        W_global=W,
        q_p_train=q_p_train,
        basis=basis,
        basis2=basis2,
        epsilon=epsilon,
        scaler=scaler,
        kernel_type=kernel_type,
        u_ref=u_ref,
        echo_level=echo_level,
    )


def jac_rbf_global(
    q_p,
    W_global,
    q_p_train,
    q_s_train,
    basis,
    basis2,
    epsilon,
    scaler,
    kernel_type="gaussian",
    echo_level=0,
):
    """
    Global tangent matrix:

        dw/dq_p = U_p + U_s (dq_s/dq_p)

    Notes
    -----
    q_s_train is kept for interface compatibility and is not used here.
    """
    del q_s_train

    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    basis = np.asarray(basis, dtype=float)
    basis2 = np.asarray(basis2, dtype=float)

    rbf_jac = _rbf_secondary_jacobian(
        q_p=q_p,
        q_p_train=np.asarray(q_p_train, dtype=float),
        W=np.asarray(W_global, dtype=float),
        epsilon=epsilon,
        scaler=scaler,
        kernel_name=kernel_type,
        echo_level=echo_level,
    )

    return basis + basis2 @ rbf_jac


# ============================================================================
# Local POD-RBF manifold helpers
# FULL decoder convention: returns the affine full state
# ============================================================================

def decode_rbf_local(k, q_p, u0_list, V_list, models, n_primary, echo_level=0):
    """
    Local POD-RBF affine decoder for cluster k:

        w(q_p) = u0_k + V_k [q_p ; q_s(q_p)]

    If the cluster has no active RBF closure, this reduces to a local linear manifold.
    """
    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    u0_k = np.asarray(u0_list[k], dtype=float)
    V_k = np.asarray(V_list[k], dtype=float)
    model_k = models[k]

    r_k = V_k.shape[1]
    n_total_k = _cluster_total_dim(model_k, r_k)

    has_rbf = bool(model_k.get("has_rbf", False))
    n_secondary = int(model_k.get("n_secondary", 0))
    kernel_name = model_k.get("kernel_name", "gaussian")

    if (not has_rbf) or (n_secondary <= 0) or (n_total_k <= n_primary):
        n_dof_k = n_total_k
        if q_p.size != n_dof_k:
            raise ValueError(
                f"[decode_rbf_local] linear cluster {k}: q_p.size={q_p.size}, expected {n_dof_k}"
            )
        return u0_k + V_k[:, :n_dof_k] @ q_p

    n_p_k = min(n_primary, n_total_k)
    if q_p.size != n_p_k:
        raise ValueError(
            f"[decode_rbf_local] RBF cluster {k}: q_p.size={q_p.size}, expected {n_p_k}"
        )

    scaler = model_k["scaler"]
    q_p_train = np.asarray(model_k["q_p_train"], dtype=float)
    W_local = np.asarray(model_k["W"], dtype=float)
    epsilon = float(model_k["epsilon"])

    q_s_pred = _predict_rbf_secondary(
        q_p=q_p,
        q_p_train=q_p_train,
        W=W_local,
        epsilon=epsilon,
        scaler=scaler,
        kernel_name=kernel_name,
        echo_level=echo_level,
    )

    q_full = np.zeros(n_total_k, dtype=float)
    q_full[:n_p_k] = q_p
    q_full[n_p_k:n_p_k + n_secondary] = q_s_pred

    return u0_k + V_k[:, :n_total_k] @ q_full


def jac_rbf_local(k, q_p, u0_list, V_list, models, n_primary, echo_level=0):
    """
    Local tangent matrix:

        dw/dq_p = A_k + B_k (dq_s/dq_p)

    where A_k and B_k are the local primary and secondary basis blocks.
    """
    del u0_list

    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    V_k = np.asarray(V_list[k], dtype=float)
    model_k = models[k]

    _, r_k = V_k.shape
    n_total_k = _cluster_total_dim(model_k, r_k)

    has_rbf = bool(model_k.get("has_rbf", False))
    n_secondary = int(model_k.get("n_secondary", 0))
    kernel_name = model_k.get("kernel_name", "gaussian")

    if (not has_rbf) or (n_secondary <= 0) or (n_total_k <= n_primary):
        n_dof_k = n_total_k
        if q_p.size != n_dof_k:
            raise ValueError(
                f"[jac_rbf_local] linear cluster {k}: q_p.size={q_p.size}, expected {n_dof_k}"
            )
        return V_k[:, :n_dof_k]

    n_p_k = min(n_primary, n_total_k)
    if q_p.size != n_p_k:
        raise ValueError(
            f"[jac_rbf_local] RBF cluster {k}: q_p.size={q_p.size}, expected {n_p_k}"
        )

    A = V_k[:, :n_p_k]
    B = V_k[:, n_p_k:n_p_k + n_secondary]

    scaler = model_k["scaler"]
    q_p_train = np.asarray(model_k["q_p_train"], dtype=float)
    W_local = np.asarray(model_k["W"], dtype=float)
    epsilon = float(model_k["epsilon"])

    rbf_jac = _rbf_secondary_jacobian(
        q_p=q_p,
        q_p_train=q_p_train,
        W=W_local,
        epsilon=epsilon,
        scaler=scaler,
        kernel_name=kernel_name,
        echo_level=echo_level,
    )

    return A + B @ rbf_jac


# ============================================================================
# Local POD-RBF manifold helpers restricted to ECSW DOFs
# FULL decoder convention on the restricted DOFs
# ============================================================================

def decode_rbf_local_ecsw(k, q_p, u0_loc_list, V_loc_list, models, n_primary, echo_level=0):
    """
    ECSW-restricted local POD-RBF affine decoder:

        w_loc(q_p) = u0_loc_k + V_loc_k [q_p ; q_s(q_p)]
    """
    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    u0_loc_k = np.asarray(u0_loc_list[k], dtype=float)
    V_loc_k = np.asarray(V_loc_list[k], dtype=float)
    model_k = models[k]

    _, r_k = V_loc_k.shape
    n_total_k = _cluster_total_dim(model_k, r_k)

    has_rbf = bool(model_k.get("has_rbf", False))
    n_secondary = int(model_k.get("n_secondary", 0))
    kernel_name = model_k.get("kernel_name", "gaussian")

    if (not has_rbf) or (n_secondary <= 0) or (n_total_k <= n_primary):
        n_dof_k = n_total_k
        if q_p.size != n_dof_k:
            raise ValueError(
                f"[decode_rbf_local_ecsw] linear cluster {k}: q_p.size={q_p.size}, expected {n_dof_k}"
            )
        return u0_loc_k + V_loc_k[:, :n_dof_k] @ q_p

    n_p_k = min(n_primary, n_total_k)
    if q_p.size != n_p_k:
        raise ValueError(
            f"[decode_rbf_local_ecsw] RBF cluster {k}: q_p.size={q_p.size}, expected {n_p_k}"
        )

    scaler = model_k["scaler"]
    q_p_train = np.asarray(model_k["q_p_train"], dtype=float)
    W_local = np.asarray(model_k["W"], dtype=float)
    epsilon = float(model_k["epsilon"])

    q_s_pred = _predict_rbf_secondary(
        q_p=q_p,
        q_p_train=q_p_train,
        W=W_local,
        epsilon=epsilon,
        scaler=scaler,
        kernel_name=kernel_name,
        echo_level=echo_level,
    )

    q_full = np.zeros(n_total_k, dtype=float)
    q_full[:n_p_k] = q_p
    q_full[n_p_k:n_p_k + n_secondary] = q_s_pred

    return u0_loc_k + V_loc_k[:, :n_total_k] @ q_full


def jac_rbf_local_ecsw(k, q_p, V_loc_list, models, n_primary, echo_level=0):
    """
    ECSW-restricted local tangent matrix:

        dw_loc/dq_p = A_loc + B_loc (dq_s/dq_p)
    """
    q_p = np.asarray(q_p, dtype=float).reshape(-1)
    V_loc_k = np.asarray(V_loc_list[k], dtype=float)
    model_k = models[k]

    _, r_k = V_loc_k.shape
    n_total_k = _cluster_total_dim(model_k, r_k)

    has_rbf = bool(model_k.get("has_rbf", False))
    n_secondary = int(model_k.get("n_secondary", 0))
    kernel_name = model_k.get("kernel_name", "gaussian")

    if (not has_rbf) or (n_secondary <= 0) or (n_total_k <= n_primary):
        n_dof_k = n_total_k
        if q_p.size != n_dof_k:
            raise ValueError(
                f"[jac_rbf_local_ecsw] linear cluster {k}: q_p.size={q_p.size}, expected {n_dof_k}"
            )
        return V_loc_k[:, :n_dof_k]

    n_p_k = min(n_primary, n_total_k)
    if q_p.size != n_p_k:
        raise ValueError(
            f"[jac_rbf_local_ecsw] RBF cluster {k}: q_p.size={q_p.size}, expected {n_p_k}"
        )

    A_loc = V_loc_k[:, :n_p_k]
    B_loc = V_loc_k[:, n_p_k:n_p_k + n_secondary]

    scaler = model_k["scaler"]
    q_p_train = np.asarray(model_k["q_p_train"], dtype=float)
    W_local = np.asarray(model_k["W"], dtype=float)
    epsilon = float(model_k["epsilon"])

    rbf_jac = _rbf_secondary_jacobian(
        q_p=q_p,
        q_p_train=q_p_train,
        W=W_local,
        epsilon=epsilon,
        scaler=scaler,
        kernel_name=kernel_name,
        echo_level=echo_level,
    )

    return A_loc + B_loc @ rbf_jac


# ============================================================================
# ECSW training matrices
# ============================================================================

def compute_ECSW_training_matrix_2D_rbf_global(
    snaps,
    prev_snaps,
    basis,
    basis2,
    W_global,
    q_p_train,
    q_s_train,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    scaler,
    epsilon,
    kernel_type="gaussian",
    u_ref=None,
):
    """
    ECSW training matrix for the global POD-RBF ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    n_red = basis.shape[1]
    u_ref = _prepare_reference(u_ref, n_tot)

    C = np.zeros((n_red * n_snaps, n_hdm))

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]

        y = (basis.T @ (snap - u_ref)).copy()

        w_rec = decode_rbf_global(
            y,
            W_global,
            q_p_train,
            basis,
            basis2,
            epsilon,
            scaler,
            kernel_type,
            u_ref=u_ref,
        )
        init_res = np.linalg.norm(w_rec - snap)
        curr_res = init_res
        num_it = 0

        print("Initial residual: {:3.2e}".format(init_res / np.linalg.norm(snap)))

        while curr_res / init_res > 1e-2 and num_it < 10:
            Jf = jac_rbf_global(
                y, W_global, q_p_train, q_s_train, basis, basis2, epsilon, scaler, kernel_type
            )

            res_rec = w_rec - snap
            JJ = Jf.T @ Jf
            Jr = Jf.T @ res_rec

            dy, *_ = np.linalg.lstsq(JJ, Jr, rcond=None)
            y -= dy

            w_rec = decode_rbf_global(
                y,
                W_global,
                q_p_train,
                basis,
                basis2,
                epsilon,
                scaler,
                kernel_type,
                u_ref=u_ref,
            )
            curr_res = np.linalg.norm(w_rec - snap)
            num_it += 1

        final_res = np.linalg.norm(w_rec - snap)
        print("Final residual: {:3.2e}".format(final_res / np.linalg.norm(snap)))

        ires = res(w_rec, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_rec, dt, JDxec, JDyec, Eye)

        V = jac_rbf_global(
            y, W_global, q_p_train, q_s_train, basis, basis2, epsilon, scaler, kernel_type
        )
        Wi = Ji @ V

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_rbf_local(
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
    max_gn_its=20,
    tol_rel=1e-2,
    init_cluster=None,
):
    """
    ECSW training matrix for the local POD-RBF ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    K = len(V_list)

    r_dof = []
    for k in range(K):
        V_k = np.asarray(V_list[k], dtype=float)
        r_k = V_k.shape[1]

        model_k = models[k]
        n_total_k = int(model_k.get("n_total", r_k))
        n_total_k = min(n_total_k, r_k)
        has_rbf = bool(model_k.get("has_rbf", False))
        n_secondary = int(model_k.get("n_secondary", 0))

        if has_rbf and n_secondary > 0 and n_total_k > n_primary:
            n_dof_k = min(n_primary, n_total_k)
        else:
            n_dof_k = n_total_k

        r_dof.append(n_dof_k)

    r_max = max(r_dof)
    C = np.zeros((r_max * n_snaps, n_hdm))

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

        V_k0 = np.asarray(V_list[k0], dtype=float)
        u0_k0 = np.asarray(u0_list[k0], dtype=float)
        y_k0 = V_k0.T @ (u_i - u0_k0)

        k = select_cluster_reduced(k0, y_k0, d_const, g_list)

        u0_k = np.asarray(u0_list[k], dtype=float)
        V_k = np.asarray(V_list[k], dtype=float)
        model_k = models[k]

        r_k = V_k.shape[1]
        n_total_k = int(model_k.get("n_total", r_k))
        n_total_k = min(n_total_k, r_k)
        has_rbf = bool(model_k.get("has_rbf", False))
        n_secondary = int(model_k.get("n_secondary", 0))

        if has_rbf and n_secondary > 0 and n_total_k > n_primary:
            n_dof_k = min(n_primary, n_total_k)
        else:
            n_dof_k = n_total_k

        q = (V_k.T @ (u_i - u0_k))[:n_dof_k].copy()

        w_rec = decode_rbf_local(k, q, u0_list, V_list, models, n_primary)
        r_rec = w_rec - u_i

        init_norm = np.linalg.norm(r_rec)
        curr_norm = init_norm
        num_it = 0

        if init_norm > 0.0:
            while curr_norm / init_norm > tol_rel and num_it < max_gn_its:
                Jf = jac_rbf_local(k, q, u0_list, V_list, models, n_primary)

                JTJ = Jf.T @ Jf
                JTr = Jf.T @ r_rec

                dq = np.linalg.solve(JTJ, -JTr)
                q += dq

                w_rec = decode_rbf_local(k, q, u0_list, V_list, models, n_primary)
                r_rec = w_rec - u_i
                curr_norm = np.linalg.norm(r_rec)
                num_it += 1

        u_tilde = w_rec

        ires = res(u_tilde, grid_x, grid_y, dt, u_prev, mu, Dxec, Dyec)
        Ji = jac(u_tilde, dt, JDxec, JDyec, Eye)

        V_q = jac_rbf_local(k, q, u0_list, V_list, models, n_primary)
        Wi = Ji @ V_q

        row0 = isnap * r_max
        row1 = row0 + n_dof_k

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


# ============================================================================
# Top-level ROMs
# Naming: global is implicit by default, local is explicit in the name
# ============================================================================

def inviscid_burgers_implicit2D_LSPG_pod_rbf(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    basis2,
    W_global,
    q_p_train,
    q_s_train,
    epsilon,
    scaler,
    kernel_type="gaussian",
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_ic=20,
    tol_ic=1e-12,
):
    """
    Global POD-RBF manifold ROM for the 2D inviscid Burgers equation:

        w(q) = u_ref + U_p q + U_s q_s(q)
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, w0.size)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    def decode_func(q):
        return decode_rbf_global(
            q_p=q,
            W_global=W_global,
            q_p_train=q_p_train,
            basis=basis,
            basis2=basis2,
            epsilon=epsilon,
            scaler=scaler,
            kernel_type=kernel_type,
            u_ref=u_ref,
        )

    def jac_rbf_func(q):
        return jac_rbf_global(
            q_p=q,
            W_global=W_global,
            q_p_train=q_p_train,
            q_s_train=q_s_train,
            basis=basis,
            basis2=basis2,
            epsilon=epsilon,
            scaler=scaler,
            kernel_type=kernel_type,
        )

    q0_guess = basis.T @ (w0 - u_ref)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0,
        decode_func=decode_func,
        jac_func=jac_rbf_func,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="GLOBAL-POD-RBF-IC",
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

    print(f"Running POD-RBF ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for it in range(num_steps):
        print(f" ... Working on timestep {it}")

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        q, resnorms, times = gauss_newton_pod_rbf(
            func=res,
            jac=jac,
            y0=qp,
            decode_rbf=decode_func,
            jac_rbf=jac_rbf_func,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
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


def inviscid_burgers_implicit2D_LSPG_local_pod_rbf(
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
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-10,
):
    """
    Local POD-RBF manifold ROM for the 2D inviscid Burgers equation.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    N = w0.size
    K = len(V_list)

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
        print(f"[LOCAL-POD-RBF] Initial cluster k = {k} / {K-1}")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

    def decode_local(q):
        return decode_rbf_local(k, q, u0_list, V_list, models, n_primary)

    def jac_local(q):
        return jac_rbf_local(k, q, u0_list, V_list, models, n_primary)

    q0_guess = V_k[:, :n_dof_k].T @ (w0 - u0_k)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0,
        decode_func=decode_local,
        jac_func=jac_local,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="LOCAL-POD-RBF-IC",
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
        print(f"[LOCAL-POD-RBF] Running local POD-RBF with {K} clusters, dt={dt}")

    for it in range(num_steps):
        if verbose:
            print(f"[LOCAL-POD-RBF] Timestep {it}/{num_steps}")

        q_full_k = V_k.T @ (wp - u0_k)
        k_new = select_cluster_reduced(k, q_full_k, d_const, g_list)

        if k_new != k:
            if verbose:
                print(f"  -> Cluster switch: {k} -> {k_new}")

            k = k_new
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

            def decode_local(q):
                return decode_rbf_local(k, q, u0_list, V_list, models, n_primary)

            def jac_local(q):
                return jac_rbf_local(k, q, u0_list, V_list, models, n_primary)

            qp_guess = V_k[:, :n_dof_k].T @ (wp - u0_k)
            qp = _gauss_newton_decoder_inverse(
                q_init=qp_guess,
                target_state=wp,
                decode_func=decode_local,
                jac_func=jac_local,
                max_its=max_its_ic,
                tol_rel=tol_ic,
                verbose=False,
                tag="LOCAL-POD-RBF-SWITCH",
            )

        cluster_history.append(k)

        def compute_residual(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def compute_jacobian(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        q, resnorms, times = gauss_newton_pod_rbf(
            func=compute_residual,
            jac=compute_jacobian,
            y0=qp,
            decode_rbf=decode_local,
            jac_rbf=jac_local,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
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


def inviscid_burgers_implicit2D_LSPG_pod_rbf_ecsw(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    basis2,
    W_global,
    q_p_train,
    q_s_train,
    weights,
    epsilon,
    scaler,
    kernel_type="gaussian",
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_ic=20,
    tol_ic=1e-12,
):
    """
    ECSW global POD-RBF manifold ROM for the 2D inviscid Burgers equation.
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

    U_p_loc = basis[idx_dofs, :]
    U_s_loc = basis2[idx_dofs, :]
    u_ref_loc = u_ref[idx_dofs]

    def decode_loc(q):
        return decode_rbf_global(
            q_p=q,
            W_global=W_global,
            q_p_train=q_p_train,
            basis=U_p_loc,
            basis2=U_s_loc,
            epsilon=epsilon,
            scaler=scaler,
            kernel_type=kernel_type,
            u_ref=u_ref_loc,
        )

    def jac_rbf_loc(q):
        return jac_rbf_global(
            q_p=q,
            W_global=W_global,
            q_p_train=q_p_train,
            q_s_train=q_s_train,
            basis=U_p_loc,
            basis2=U_s_loc,
            epsilon=epsilon,
            scaler=scaler,
            kernel_type=kernel_type,
        )

    q0_guess = basis.T @ (w0 - u_ref)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0[idx_dofs],
        decode_func=decode_loc,
        jac_func=jac_rbf_loc,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="GLOBAL-POD-RBF-ECSW-IC",
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

    print(f"Running POD-RBF ECSW ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

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

        q, resnorms, times = gauss_newton_pod_rbf_ecsw(
            func=res_loc,
            jac=jac_loc,
            y0=qp,
            decode_rbf=decode_loc,
            jac_rbf=jac_rbf_loc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weights=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
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


def inviscid_burgers_implicit2D_LSPG_local_pod_rbf_ecsw(
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
    max_its=20,
    max_its_ic=20,
    tol_ic=1e-10,
):
    """
    ECSW local POD-RBF manifold ROM for the 2D inviscid Burgers equation.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

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
        print(f"[LOCAL-POD-RBF-ECSW] Initial cluster k = {k} / {K-1}")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

    def decode_loc(q):
        return decode_rbf_local_ecsw(k, q, u0_loc_list, V_loc_list, models, n_primary)

    def jac_loc(q):
        return jac_rbf_local_ecsw(k, q, V_loc_list, models, n_primary)

    q0_guess = V_k[:, :n_dof_k].T @ (w0 - u0_k)
    q0 = _gauss_newton_decoder_inverse(
        q_init=q0_guess,
        target_state=w0[idx_dofs],
        decode_func=decode_loc,
        jac_func=jac_loc,
        max_its=max_its_ic,
        tol_rel=tol_ic,
        verbose=False,
        tag="LOCAL-POD-RBF-ECSW-IC",
    )

    w_init = decode_rbf_local(k, q0, u0_list, V_list, models, n_primary)

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
        print(f"[LOCAL-POD-RBF-ECSW] Running local POD-RBF-ECSW with {K} clusters, dt={dt}")

    for it in range(num_steps):
        if verbose and (it % 10 == 0 or it == num_steps - 1):
            print(f"[LOCAL-POD-RBF-ECSW] Timestep {it}/{num_steps}")

        k_new = select_cluster_reduced_trunc(k, qp, d_const, g_list, n_dof_k)

        if k_new != k:
            if verbose:
                print(f"  [LOCAL-POD-RBF-ECSW] Cluster switch: {k} -> {k_new}")

            k = k_new
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            n_dof_k = _cluster_primary_dim(models[k], V_k.shape[1], n_primary)

            def decode_loc(q):
                return decode_rbf_local_ecsw(k, q, u0_loc_list, V_loc_list, models, n_primary)

            def jac_loc(q):
                return jac_rbf_local_ecsw(k, q, V_loc_list, models, n_primary)

            qp_guess = V_k[:, :n_dof_k].T @ (wp_full - u0_k)
            qp = _gauss_newton_decoder_inverse(
                q_init=qp_guess,
                target_state=wp_full[idx_dofs],
                decode_func=decode_loc,
                jac_func=jac_loc,
                max_its=max_its_ic,
                tol_rel=tol_ic,
                verbose=False,
                tag="LOCAL-POD-RBF-ECSW-SWITCH",
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

        q, resnorms, times = gauss_newton_pod_rbf_ecsw(
            func=res_loc,
            jac=jac_loc_res,
            y0=qp,
            decode_rbf=decode_loc,
            jac_rbf=jac_loc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weights=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=None,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_full = decode_rbf_local(k, q, u0_list, V_list, models, n_primary)

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
