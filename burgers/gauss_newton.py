# gauss_newton.py
# -*- coding: utf-8 -*-

import time
from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .quadratic_manifold_utils import u_qm, J_qm


def _prepare_u_ref(u_ref, size):
    if u_ref is None:
        return np.zeros(size, dtype=np.float64)

    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if u_ref.size != size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {size}")
    return u_ref


def _relative_drop(prev, curr):
    if prev == 0.0:
        return 0.0
    return abs((prev - curr) / prev)


def _safe_init_norm(val):
    return 1.0 if val == 0.0 else val


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _call_decode(decode, y, with_grad=False):
    try:
        return decode(y, with_grad=with_grad)
    except TypeError:
        try:
            return decode(y, with_grad)
        except TypeError:
            return decode(y)


def _solve_reduced_update(JV, r, linear_solver="lstsq", normal_eq_reg=1e-12):
    """
    Solve the reduced Gauss-Newton update:

        JV * dy ~= -r

    Parameters
    ----------
    JV : ndarray, shape (n_res, n_red)
    r : ndarray, shape (n_res,)
    linear_solver : {"lstsq", "normal_eq"}
        - "lstsq": robust SVD-based least-squares.
        - "normal_eq": solve (JV^T JV) dy = -JV^T r with optional ridge regularization.
    normal_eq_reg : float
        Non-negative ridge term for normal equations.
    """
    mode = str(linear_solver).strip().lower()
    if mode == "lstsq":
        dy, *_ = np.linalg.lstsq(JV, -r, rcond=None)
        return dy

    if mode == "normal_eq":
        reg = float(normal_eq_reg)
        if reg < 0.0:
            raise ValueError(f"normal_eq_reg must be non-negative, got {reg}.")

        ata = JV.T @ JV
        atb = -(JV.T @ r)

        if reg > 0.0:
            ata = ata + reg * np.eye(ata.shape[0], dtype=ata.dtype)

        try:
            chol = np.linalg.cholesky(ata)
            y = np.linalg.solve(chol, atb)
            dy = np.linalg.solve(chol.T, y)
            return dy
        except np.linalg.LinAlgError:
            try:
                return np.linalg.solve(ata, atb)
            except np.linalg.LinAlgError:
                dy, *_ = np.linalg.lstsq(JV, -r, rcond=None)
                return dy

    raise ValueError(
        "linear_solver must be one of: 'lstsq', 'normal_eq'. "
        f"Got: {linear_solver}"
    )


def newton_raphson(func, jac, x0, max_its=20, relnorm_cutoff=1e-12, u_ref=None):
    """
    Newton-Raphson solver for full-order nonlinear systems.

    u_ref is accepted for interface consistency, but not used.
    """
    x = np.asarray(x0, dtype=np.float64).copy()

    r0 = func(x)
    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        r = func(x)
        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            print(f"{it}: {resnorm / init_norm:3.2e}")
            break

        J = jac(x)
        dx = sp.linalg.spsolve(J, r)
        x -= dx

    return x, resnorms


def gauss_newton_LSPG(
    func,
    jac,
    basis,
    y0,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    basis = np.asarray(basis, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64).copy()
    u_ref = _prepare_u_ref(u_ref, basis.shape[0])

    w = u_ref + basis @ y

    r0 = func(w)
    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ basis
        dy = _solve_reduced_update(
            JV,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        y += dy
        w = u_ref + basis @ y

    print(f"iteration {it}: relative norm {resnorm / init_norm:3.2e}")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_ECSW_2D(
    func,
    jac,
    basis,
    y0,
    sample_inds,
    augmented_sample,
    sample_weights,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    del sample_inds, augmented_sample

    basis = np.asarray(basis, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64).copy()
    u_ref = _prepare_u_ref(u_ref, basis.shape[0])

    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    sqrt_w = np.sqrt(weights)

    w = u_ref + basis @ y

    r0 = func(w)
    if r0.shape[0] != weights.shape[0]:
        raise ValueError(
            f"sample_weights and residual size mismatch: {weights.shape[0]} vs {r0.shape[0]}"
        )

    init_norm = _safe_init_norm(np.linalg.norm(sqrt_w * r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(sqrt_w * r)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ basis
        JVw = sqrt_w[:, None] * JV
        rw = sqrt_w * r
        dy = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        y += dy
        w = u_ref + basis @ y

    print(f"iteration {it}: relative norm {resnorm / init_norm:3.2e}")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_LSPG_local(
    func,
    jac,
    Vloc,
    u0loc,
    y0,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Local affine LSPG. If u_ref is given, it overrides u0loc.
    """
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    Vloc = np.asarray(Vloc, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64).copy()

    if u_ref is None:
        u_ref_eff = np.asarray(u0loc, dtype=np.float64).reshape(-1)
    else:
        u_ref_eff = _prepare_u_ref(u_ref, Vloc.shape[0])

    w = u_ref_eff + Vloc @ y

    r0 = func(w)
    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ Vloc
        dy = _solve_reduced_update(
            JV,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        y += dy
        w = u_ref_eff + Vloc @ y

    print(f"[local GN] iteration {it}: relnorm = {resnorm / init_norm:3.2e}")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_LSPG_local_ecsw(
    res_fun,
    jac_fun,
    V_loc,
    u0_loc,
    q0,
    sample_weights_cells,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    V_loc = np.asarray(V_loc, dtype=np.float64)
    q = np.asarray(q0, dtype=np.float64).copy()

    if u_ref is None:
        u_ref_eff = np.asarray(u0_loc, dtype=np.float64).reshape(-1)
    else:
        u_ref_eff = _prepare_u_ref(u_ref, V_loc.shape[0])

    weights = np.concatenate((sample_weights_cells, sample_weights_cells)).astype(np.float64)

    w_loc = u_ref_eff + V_loc @ q

    t0 = time.time()
    r0 = res_fun(w_loc)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(weights * r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = res_fun(w_loc)
        res_time += time.time() - t0

        rw = weights * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J_loc = jac_fun(w_loc)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J_loc @ V_loc
        JVw = weights[:, None] * JV
        dq = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        q += dq
        w_loc = u_ref_eff + V_loc @ q

    print(f"[local GN ECSW] iteration {it}: weighted relnorm = {resnorm / init_norm:3.2e}")
    return q, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_LSPG_qm(
    func_res,
    func_jac,
    V,
    H,
    u_ref,
    q0,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q0, dtype=np.float64).copy()
    u_ref = _prepare_u_ref(u_ref, V.shape[0])

    w = u_qm(q, V, H, u_ref)

    t0 = time.time()
    r0 = func_res(w)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func_res(w)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        print(f"[GN-QM] it={it:2d}, ||R|| = {resnorm:.3e}")

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        Jw = func_jac(w)
        jac_time += time.time() - t0

        t0 = time.time()
        Jman = J_qm(q, V, H)
        J_eff = Jw @ Jman
        dq = _solve_reduced_update(
            J_eff,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        q += dq
        w = u_qm(q, V, H, u_ref)

    print(f"[GN-QM] final it={it}, relative norm={resnorms[-1] / init_norm:.3e}")
    return q, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_quadratic_q(
    u_snap,
    V,
    H,
    u_ref=None,
    max_its=20,
    tol_rel=1e-6,
    min_delta=1e-8,
    verbose=False,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    V = np.asarray(V, dtype=np.float64)
    u_snap = np.asarray(u_snap, dtype=np.float64).reshape(-1)
    u_ref = _prepare_u_ref(u_ref, V.shape[0])

    q = V.T @ (u_snap - u_ref)

    u = u_qm(q, V, H, u_ref)
    delta = u - u_snap
    delta0_norm = np.linalg.norm(delta)

    if delta0_norm == 0.0:
        if verbose:
            print("[GN-q] Initial guess is exact.")
        return q

    prev_norm = delta0_norm

    for it in range(max_its):
        delta = u - u_snap
        res_norm = np.linalg.norm(delta)

        if verbose:
            print(f"[GN-q] it {it:2d}: ||δ||/||δ0|| = {res_norm / delta0_norm:.3e}")

        if res_norm / delta0_norm < tol_rel:
            break

        if it > 0:
            rel_drop = _relative_drop(prev_norm, res_norm)
            if rel_drop < min_delta:
                if verbose:
                    print(f"[GN-q] Stopping: rel_drop={rel_drop:.3e} < {min_delta:.3e}")
                break

        prev_norm = res_norm

        J_delta = J_qm(q, V, H)
        dq = _solve_reduced_update(
            J_delta,
            delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        q += dq
        u = u_qm(q, V, H, u_ref)

    if verbose:
        final_ratio = np.linalg.norm(u - u_snap) / delta0_norm
        print(f"[GN-q] Done: it={it}, final ||δ||/||δ0|| = {final_ratio:.3e}")

    return q


def gauss_newton_LSPG_qm_ecsw(
    res_fun,
    jac_fun,
    V_loc,
    H_loc,
    u_ref_loc,
    q0,
    sample_weights,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    V_loc = np.asarray(V_loc, dtype=np.float64)
    q = np.asarray(q0, dtype=np.float64).copy()
    u_ref_loc = _prepare_u_ref(u_ref_loc, V_loc.shape[0])
    sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)

    w_loc = u_qm(q, V_loc, H_loc, u_ref_loc)

    t0 = time.time()
    r0 = res_fun(w_loc)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(sample_weights * r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = res_fun(w_loc)
        res_time += time.time() - t0

        rw = sample_weights * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        print(f"[GN-QM-ECSW] it={it:2d}, ||w*R|| = {resnorm:.3e}")

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J_loc = jac_fun(w_loc)
        jac_time += time.time() - t0

        t0 = time.time()
        Du_loc = J_qm(q, V_loc, H_loc)
        JV = J_loc @ Du_loc
        JVw = sample_weights[:, None] * JV
        dq = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        q += dq
        w_loc = u_qm(q, V_loc, H_loc, u_ref_loc)

    print(f"[GN-QM-ECSW] final it={it}, rel={resnorms[-1] / init_norm:.3e}")
    return q, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_pod_ann(
    func,
    jac,
    y0,
    decode,
    jacfwdfunc,
    max_its=20,
    relnorm_cutoff=1e-5,
    lookback=10,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    del lookback, u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    y = y0.detach().clone()

    with torch.no_grad():
        w = _call_decode(decode, y, with_grad=False)

    w_np = _to_numpy(w).squeeze()
    r0 = func(w_np)
    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        w_np = _to_numpy(w).squeeze()

        t0 = time.time()
        r = func(w_np)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w_np)
        V = _to_numpy(jacfwdfunc(y).detach())
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ V
        dy = _solve_reduced_update(
            JV,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        with torch.no_grad():
            y += torch.tensor(dy, dtype=y.dtype, device=y.device)
            w = _call_decode(decode, y, with_grad=False)

    print(f"{it} iterations: {resnorm / init_norm:3.2e} relative norm")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_pod_ann_joshua(
    func,
    jac,
    y0,
    decode,
    jacfwdfunc,
    max_its=20,
    relnorm_cutoff=1e-5,
    lookback=10,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    return gauss_newton_pod_ann(
        func=func,
        jac=jac,
        y0=y0,
        decode=decode,
        jacfwdfunc=jacfwdfunc,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        lookback=lookback,
        min_delta=min_delta,
        u_ref=u_ref,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )


def gauss_newton_pod_ann_ecsw(
    func,
    jac,
    y0,
    decode,
    jacfwdfunc,
    sample_inds,
    augmented_sample,
    weight,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    del sample_inds, augmented_sample, u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    y = y0.detach().clone()
    weights = np.concatenate((weight, weight)).astype(np.float64)

    with torch.no_grad():
        w = _call_decode(decode, y, with_grad=False)

    w_np = _to_numpy(w).squeeze()
    r0 = func(w_np)
    init_norm = _safe_init_norm(np.linalg.norm(weights * r0))
    resnorms = []

    for it in range(max_its):
        w_np = _to_numpy(w).squeeze()

        t0 = time.time()
        r = func(w_np)
        res_time += time.time() - t0

        rw = weights * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w_np)
        V = _to_numpy(jacfwdfunc(y).detach())
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ V
        JVw = weights[:, None] * JV
        dy = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        with torch.no_grad():
            y += torch.tensor(dy, dtype=y.dtype, device=y.device)
            w = _call_decode(decode, y, with_grad=False)

    print(f"{it} iterations: {resnorm / init_norm:3.2e} relative norm")
    return y, resnorms, (jac_time, res_time, ls_time)


# Backward-compatible aliases for older naming
def gauss_newton_rnm(*args, **kwargs):
    return gauss_newton_pod_ann(*args, **kwargs)


def gauss_newton_rnm_joshua(*args, **kwargs):
    return gauss_newton_pod_ann_joshua(*args, **kwargs)


def gauss_newton_rnm_ecsw(*args, **kwargs):
    return gauss_newton_pod_ann_ecsw(*args, **kwargs)


def gauss_newton_pod_gp(
    func,
    jac,
    y0,
    decode_gp,
    jac_gp,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Backward-compatible GP/GPR Gauss-Newton wrapper.
    Uses the same algebraic solve as POD-RBF with GP/GPR decode/tangent.
    """
    return gauss_newton_pod_rbf(
        func=func,
        jac=jac,
        y0=y0,
        decode_rbf=decode_gp,
        jac_rbf=jac_gp,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        u_ref=u_ref,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )


def gauss_newton_pod_gpr(
    func,
    jac,
    y0,
    decode_gp,
    jac_gp,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Preferred naming for GP/GPR manifold Gauss-Newton.
    """
    return gauss_newton_pod_gp(
        func=func,
        jac=jac,
        y0=y0,
        decode_gp=decode_gp,
        jac_gp=jac_gp,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        u_ref=u_ref,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )


def gauss_newton_pod_gpr_ecsw(
    func,
    jac,
    y0,
    decode_gp,
    jac_gp,
    sample_inds,
    augmented_sample,
    weights,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Preferred naming for ECSW GP/GPR manifold Gauss-Newton.
    """
    return gauss_newton_pod_gp_ecsw(
        func=func,
        jac=jac,
        y0=y0,
        decode_gp=decode_gp,
        jac_gp=jac_gp,
        sample_inds=sample_inds,
        augmented_sample=augmented_sample,
        weights=weights,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        u_ref=u_ref,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
    )


def gauss_newton_pod_rbf(
    func,
    jac,
    y0,
    decode_rbf,
    jac_rbf,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    del u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    y = np.asarray(y0, dtype=float).copy()
    w = decode_rbf(y)

    t0 = time.time()
    r0 = func(w)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        print(f"[GN-POD-RBF] it={it:2d}, ||R|| = {resnorm:.3e}")

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        Jw = jac(w)
        V = jac_rbf(y)
        jac_time += time.time() - t0

        t0 = time.time()
        J_eff = Jw @ V
        dy = _solve_reduced_update(
            J_eff,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        y += dy
        w = decode_rbf(y)

    print(f"[GN-POD-RBF] final it={it}, relative norm={resnorms[-1] / init_norm:.3e}")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_pod_rbf_ecsw(
    func,
    jac,
    y0,
    decode_rbf,
    jac_rbf,
    sample_inds,
    augmented_sample,
    weights,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    freeze_hdm_jacobian=True,
    normal_eqn=True,
    linear_solver=None,
    normal_eq_reg=1e-12,
    verbose=False,
    u_ref=None,
):
    del sample_inds, augmented_sample, u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    y = np.asarray(y0, dtype=float).copy()
    weights_uv = np.concatenate((np.asarray(weights, dtype=float),
                                 np.asarray(weights, dtype=float)))

    w = decode_rbf(y)

    t0 = time.time()
    r0 = func(w)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(weights_uv * r0))
    resnorms = []

    J_frozen = None

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        rw = weights_uv * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        if verbose:
            print(f"[GN-POD-RBF-ECSW] it={it:02d}, rel={resnorm / init_norm:.3e}")

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        if freeze_hdm_jacobian:
            if J_frozen is None:
                J_frozen = jac(w)
            J = J_frozen
        else:
            J = jac(w)

        V = jac_rbf(y)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J.dot(V) if hasattr(J, "dot") else (J @ V)
        JVw = weights_uv[:, None] * JV
        solver_mode = linear_solver
        if solver_mode is None:
            solver_mode = "normal_eq" if bool(normal_eqn) else "lstsq"

        dy = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=solver_mode,
            normal_eq_reg=normal_eq_reg,
        )

        ls_time += time.time() - t0

        y += dy
        w = decode_rbf(y)

    if verbose:
        print(f"[GN-POD-RBF-ECSW] final rel={resnorms[-1] / init_norm:.3e}")

    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_pod_rbf_ecsw_old(
    func,
    jac,
    y0,
    decode_rbf,
    jac_rbf,
    sample_inds,
    augmented_sample,
    weights,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
):
    del sample_inds, augmented_sample, u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    y = np.asarray(y0, dtype=float).copy()
    w = decode_rbf(y)
    weights_uv = np.concatenate((weights, weights))

    t0 = time.time()
    r0 = func(w)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(weights_uv * r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        rw = weights_uv * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w)
        V = jac_rbf(y)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ V
        JVw = weights_uv[:, None] * JV
        dy, *_ = np.linalg.lstsq(JVw, -rw, rcond=None)
        ls_time += time.time() - t0

        y += dy
        w = decode_rbf(y)

    print(f"{it} iterations: {resnorm / init_norm:.2e} relative norm")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_pod_gp_ecsw(
    func,
    jac,
    y0,
    decode_gp,
    jac_gp,
    sample_inds,
    augmented_sample,
    weights,
    max_its=10,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    u_ref=None,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    del sample_inds, augmented_sample, u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    y = np.asarray(y0, dtype=float).copy()
    w = decode_gp(y)
    weights_uv = np.concatenate((weights, weights))

    t0 = time.time()
    r0 = func(w)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(weights_uv * r0))
    resnorms = []

    for it in range(max_its):
        t0 = time.time()
        r = func(w)
        res_time += time.time() - t0

        rw = weights_uv * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(w)
        V = jac_gp(y)
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ V
        JVw = weights_uv[:, None] * JV
        dy = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        y += dy
        w = decode_gp(y)

    print(f"{it} iterations: {resnorm / init_norm:.2e} relative norm")
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_poddl(
    func,
    jac,
    z0: torch.Tensor,
    decode,
    jac_u_z,
    max_its: int = 20,
    relnorm_cutoff: float = 1e-5,
    min_delta: float = 1e-2,
    u_ref=None,
    linear_solver: str = "lstsq",
    normal_eq_reg: float = 1e-12,
) -> Tuple[torch.Tensor, Sequence[float], Tuple[float, float, float]]:
    del u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    z = z0.detach().clone()

    with torch.no_grad():
        u = _call_decode(decode, z, with_grad=False)

    u_np = _to_numpy(u).squeeze()

    t0 = time.time()
    r0 = func(u_np)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(r0))
    resnorms = []

    for it in range(max_its):
        u_np = _to_numpy(u).squeeze()

        t0 = time.time()
        r = func(u_np)
        res_time += time.time() - t0

        resnorm = np.linalg.norm(r)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(u_np)
        Uz = _to_numpy(jac_u_z(z).detach())
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ Uz
        dz = _solve_reduced_update(
            JV,
            r,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        with torch.no_grad():
            z += torch.tensor(dz, dtype=z.dtype, device=z.device)
            u = _call_decode(decode, z, with_grad=False)

    print(f"{it} iterations: {resnorm / init_norm:.2e} relative norm")
    return z, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_poddl_ecsw(
    func,
    jac,
    z0: torch.Tensor,
    decode,
    jac_u_z,
    sample_inds,
    augmented_sample,
    weight,
    max_its: int = 20,
    relnorm_cutoff: float = 1e-5,
    min_delta: float = 1e-2,
    u_ref=None,
    linear_solver: str = "lstsq",
    normal_eq_reg: float = 1e-12,
) -> Tuple[torch.Tensor, Sequence[float], Tuple[float, float, float]]:
    del sample_inds, augmented_sample, u_ref

    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    z = z0.detach().clone()
    weights = np.concatenate(
        (np.asarray(weight, dtype=np.float64), np.asarray(weight, dtype=np.float64))
    )

    with torch.no_grad():
        u = _call_decode(decode, z, with_grad=False)

    u_np = _to_numpy(u).squeeze()

    t0 = time.time()
    r0 = func(u_np)
    res_time += time.time() - t0

    init_norm = _safe_init_norm(np.linalg.norm(weights * r0))
    resnorms = []

    for it in range(max_its):
        u_np = _to_numpy(u).squeeze()

        t0 = time.time()
        r = func(u_np)
        res_time += time.time() - t0

        rw = weights * r
        resnorm = np.linalg.norm(rw)
        resnorms.append(resnorm)

        if resnorm / init_norm < relnorm_cutoff:
            break

        if len(resnorms) > 1 and _relative_drop(resnorms[-2], resnorms[-1]) < min_delta:
            break

        t0 = time.time()
        J = jac(u_np)
        Uz = _to_numpy(jac_u_z(z).detach())
        jac_time += time.time() - t0

        t0 = time.time()
        JV = J @ Uz
        JVw = weights[:, None] * JV
        dz = _solve_reduced_update(
            JVw,
            rw,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        ls_time += time.time() - t0

        with torch.no_grad():
            z += torch.tensor(dz, dtype=z.dtype, device=z.device)
            u = _call_decode(decode, z, with_grad=False)

    print(f"{it} iterations: {resnorm / init_norm:.2e} weighted relative norm")
    return z, resnorms, (jac_time, res_time, ls_time)
