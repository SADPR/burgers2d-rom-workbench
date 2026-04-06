# pod_ann_manifold.py
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.sparse as sp
import torch
import functorch

from .core import (
    get_ops,
    inviscid_burgers_res2D,
    inviscid_burgers_res2D_ecsw,
    inviscid_burgers_exact_jac2D,
    inviscid_burgers_exact_jac2D_ecsw,
)

from .ecsw_utils import generate_augmented_mesh

from .gauss_newton import (
    gauss_newton_pod_ann,
    gauss_newton_pod_ann_ecsw,
    _solve_reduced_update,
)

try:
    from torch.func import jacfwd as torch_jacfwd
except Exception:
    torch_jacfwd = functorch.jacfwd


def _prepare_reference(u_ref, size):
    """
    Return a safe reference vector of length `size`.

    If u_ref is None, return the zero vector.
    """
    if u_ref is None:
        return np.zeros(size, dtype=np.float64)

    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if u_ref.size != size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {size}")
    return u_ref


def _to_torch_vector(x, dtype=torch.float32, device=None):
    """
    Convert x to a 1D torch tensor.
    """
    if torch.is_tensor(x):
        out = x.reshape(-1)
        if device is not None:
            out = out.to(device=device, dtype=dtype)
        else:
            out = out.to(dtype=dtype)
        return out
    return torch.tensor(np.asarray(x), dtype=dtype, device=device).reshape(-1)


def _to_torch_matrix(x, dtype=torch.float32, device=None):
    """
    Convert x to a 2D torch tensor.
    """
    if torch.is_tensor(x):
        out = x
        if device is not None:
            out = out.to(device=device, dtype=dtype)
        else:
            out = out.to(dtype=dtype)
        return out
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


def _build_ann_decoder_jacobian(basis, basis2, ann_eval):
    """
    Return a fast Jacobian callable for

        decode(y) = basis @ y + basis2 @ ann_eval(y)

    using

        d(decode)/dy = basis + basis2 @ d(ann_eval)/dy
    """
    ann_jac = torch_jacfwd(ann_eval)

    def jacfwdfunc(y):
        return basis + basis2 @ ann_jac(y)

    return jacfwdfunc


def _align_module_device(module, target_device):
    """
    Ensure `module` lives on `target_device`.

    This avoids expensive implicit host<->device transfers during Jacobian
    evaluations inside the implicit ROM loops.
    """
    try:
        cur_device = next(module.parameters()).device
    except StopIteration:
        cur_device = target_device
    if cur_device != target_device:
        module = module.to(target_device)
    return module


def compute_ECSW_training_matrix_2D_pod_ann(
    snaps,
    prev_snaps,
    basis,
    approx,
    jacfwdfunc,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    u_ref=None,
):
    """
    ECSW training matrix for the POD-ANN ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    n_red = basis.shape[1]

    if u_ref is None:
        u_ref_vec = np.zeros(n_tot, dtype=np.float64)
    else:
        u_ref_vec = np.asarray(u_ref, dtype=np.float64).reshape(-1)
        if u_ref_vec.size != n_tot:
            raise ValueError(
                f"u_ref size mismatch in ECSW training matrix: got {u_ref_vec.size}, expected {n_tot}."
            )

    C = np.zeros((n_red * n_snaps, n_hdm))

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]

        y0 = basis.T @ (snap - u_ref_vec)
        y = torch.tensor(y0, dtype=torch.float32)

        w_rec = approx(y).squeeze().detach().cpu().numpy()
        init_res = np.linalg.norm(w_rec - snap)
        curr_res = init_res
        num_it = 0

        print("Initial residual: {:3.2e}".format(init_res / np.linalg.norm(snap)))

        while curr_res / init_res > 1e-2 and num_it < 10:
            w_t = approx(y).squeeze()
            snap_t = torch.tensor(snap, dtype=w_t.dtype, device=w_t.device)

            Jf = jacfwdfunc(y)
            JJ = Jf.T @ Jf
            Jr = Jf.T @ (w_t - snap_t)

            dy, *_ = np.linalg.lstsq(
                JJ.squeeze().detach().cpu().numpy(),
                Jr.squeeze().detach().cpu().numpy(),
                rcond=None,
            )

            y -= torch.tensor(dy, dtype=y.dtype, device=y.device)
            w_rec = approx(y).squeeze().detach().cpu().numpy()
            curr_res = np.linalg.norm(w_rec - snap)
            num_it += 1

        final_res = np.linalg.norm(w_rec - snap)
        print("Final residual: {:3.2e}".format(final_res / np.linalg.norm(snap)))

        ires = res(w_rec, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_rec, dt, JDxec, JDyec, Eye)

        V = jacfwdfunc(y).detach().squeeze().cpu().numpy()
        Wi = Ji @ V

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_pod_ann_case2(
    snaps,
    prev_snaps,
    t_samples,
    basis,
    basis2,
    ann_model,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    u_ref=None,
):
    """
    ECSW training matrix for POD-ANN Case 2 manifold

        w(y, t; mu) = u_ref + basis @ y + basis2 @ ann_model([mu1, mu2, t]).

    The decoder Jacobian wrt y is basis.
    """
    snaps = np.asarray(snaps, dtype=np.float64)
    prev_snaps = np.asarray(prev_snaps, dtype=np.float64)
    t_samples = np.asarray(t_samples, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)

    n_tot, n_snaps = snaps.shape
    if prev_snaps.shape != snaps.shape:
        raise ValueError(f"prev_snaps shape {prev_snaps.shape} must match snaps shape {snaps.shape}.")
    if t_samples.size != n_snaps:
        raise ValueError(
            f"t_samples length {t_samples.size} must match number of snapshots {n_snaps}."
        )
    if basis.shape[0] != n_tot or basis2.shape[0] != n_tot:
        raise ValueError(
            f"Basis row mismatch: snaps have {n_tot} rows, basis={basis.shape}, basis2={basis2.shape}."
        )

    n_hdm = n_tot // 2
    n_red = basis.shape[1]

    u_ref_vec = _prepare_reference(u_ref, n_tot)

    C = np.zeros((n_red * n_snaps, n_hdm), dtype=np.float64)
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    try:
        device = next(ann_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]
        t_now = float(t_samples[isnap])

        x = torch.tensor([mu1, mu2, t_now], dtype=torch.float32, device=device)
        with torch.no_grad():
            qbar = ann_model(x).reshape(-1).detach().cpu().numpy()

        offset = u_ref_vec + basis2 @ qbar
        q0 = basis.T @ (snap - u_ref_vec)
        w_init = offset + basis @ q0
        snap_norm = np.linalg.norm(snap)
        denom = snap_norm if snap_norm > 0.0 else 1.0
        init_res = np.linalg.norm(w_init - snap)
        print("Initial residual: {:3.2e}".format(init_res / denom))

        q = np.linalg.lstsq(basis, snap - offset, rcond=None)[0]
        w_tilde = offset + basis @ q
        final_res = np.linalg.norm(w_tilde - snap)
        print("Final residual: {:3.2e}".format(final_res / denom))

        ires = res(w_tilde, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_tilde, dt, JDxec, JDyec, Eye)
        Wi = Ji @ basis

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_pod_ann_case2_petrov_galerkin(
    snaps,
    prev_snaps,
    t_samples,
    basis,
    basis2,
    ann_model,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    u_ref=None,
):
    """
    ECSW training matrix for POD-ANN Case 2 with enriched residual testing.

    Uses the Case-2 manifold

        w(y, t; mu) = u_ref + basis @ y + basis2 @ ann_model([mu1, mu2, t]),

    but builds cell contributions for projected residual testing through V_tot:

        V_tot = [basis, basis2],  s = (J V_tot)^T r.
    """
    snaps = np.asarray(snaps, dtype=np.float64)
    prev_snaps = np.asarray(prev_snaps, dtype=np.float64)
    t_samples = np.asarray(t_samples, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)

    n_tot, n_snaps = snaps.shape
    if prev_snaps.shape != snaps.shape:
        raise ValueError(f"prev_snaps shape {prev_snaps.shape} must match snaps shape {snaps.shape}.")
    if t_samples.size != n_snaps:
        raise ValueError(
            f"t_samples length {t_samples.size} must match number of snapshots {n_snaps}."
        )
    if basis.shape[0] != n_tot or basis2.shape[0] != n_tot:
        raise ValueError(
            f"Basis row mismatch: snaps have {n_tot} rows, basis={basis.shape}, basis2={basis2.shape}."
        )

    n_hdm = n_tot // 2
    n_test = basis.shape[1] + basis2.shape[1]
    Vtot = np.concatenate((basis, basis2), axis=1)

    u_ref_vec = _prepare_reference(u_ref, n_tot)

    C = np.zeros((n_test * n_snaps, n_hdm), dtype=np.float64)
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    try:
        device = next(ann_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]
        t_now = float(t_samples[isnap])

        x = torch.tensor([mu1, mu2, t_now], dtype=torch.float32, device=device)
        with torch.no_grad():
            qbar = ann_model(x).reshape(-1).detach().cpu().numpy()

        offset = u_ref_vec + basis2 @ qbar
        q0 = basis.T @ (snap - u_ref_vec)
        w_init = offset + basis @ q0
        snap_norm = np.linalg.norm(snap)
        denom = snap_norm if snap_norm > 0.0 else 1.0
        init_res = np.linalg.norm(w_init - snap)
        print("Initial residual: {:3.2e}".format(init_res / denom))

        q = np.linalg.lstsq(basis, snap - offset, rcond=None)[0]
        w_tilde = offset + basis @ q
        final_res = np.linalg.norm(w_tilde - snap)
        print("Final residual: {:3.2e}".format(final_res / denom))

        ires = res(w_tilde, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_tilde, dt, JDxec, JDyec, Eye)
        Wi = Ji @ Vtot

        row0 = isnap * n_test
        row1 = row0 + n_test

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_pod_ann_case3(
    snaps,
    prev_snaps,
    t_samples,
    basis,
    basis2,
    ann_model,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    u_ref=None,
    projection_max_its=10,
    projection_relnorm_cutoff=1e-2,
):
    """
    ECSW training matrix for POD-ANN Case 3 manifold

        w(y, t; mu) = u_ref + basis @ y + basis2 @ ann_model([y, mu1, mu2, t]).
    """
    snaps = np.asarray(snaps, dtype=np.float64)
    prev_snaps = np.asarray(prev_snaps, dtype=np.float64)
    t_samples = np.asarray(t_samples, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)

    n_tot, n_snaps = snaps.shape
    if prev_snaps.shape != snaps.shape:
        raise ValueError(f"prev_snaps shape {prev_snaps.shape} must match snaps shape {snaps.shape}.")
    if t_samples.size != n_snaps:
        raise ValueError(
            f"t_samples length {t_samples.size} must match number of snapshots {n_snaps}."
        )
    if basis.shape[0] != n_tot or basis2.shape[0] != n_tot:
        raise ValueError(
            f"Basis row mismatch: snaps have {n_tot} rows, basis={basis.shape}, basis2={basis2.shape}."
        )

    n_hdm = n_tot // 2
    n_red = basis.shape[1]

    u_ref_vec = _prepare_reference(u_ref, n_tot)

    C = np.zeros((n_red * n_snaps, n_hdm), dtype=np.float64)
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    try:
        device = next(ann_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
    basis2_t = torch.tensor(basis2, dtype=torch.float32, device=device)
    u_ref_t = torch.tensor(u_ref_vec, dtype=torch.float32, device=device)
    mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
    if mu_vec.size != 2:
        raise ValueError(f"mu must have length 2, got {mu_vec.shape}")
    tmu = torch.tensor(mu_vec, dtype=torch.float32, device=device)

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]
        t_now = torch.tensor(float(t_samples[isnap]), dtype=torch.float32, device=device)
        snap_t = torch.tensor(snap, dtype=torch.float32, device=device)

        y = torch.tensor(basis.T @ (snap - u_ref_vec), dtype=torch.float32, device=device)

        def _ann_eval(y_vec):
            x = torch.cat([y_vec.reshape(-1), tmu, t_now.reshape(1)], dim=0)
            return ann_model(x).reshape(-1)

        def decode(y_vec):
            return u_ref_t + basis_t @ y_vec + basis2_t @ _ann_eval(y_vec)

        jac_decode = _build_ann_decoder_jacobian(
            basis=basis_t,
            basis2=basis2_t,
            ann_eval=_ann_eval,
        )

        with torch.no_grad():
            w_rec_t = decode(y)
        init_res = np.linalg.norm((w_rec_t - snap_t).detach().cpu().numpy()) + 1e-30
        curr_res = init_res
        proj_it = 0
        snap_norm = np.linalg.norm(snap)
        denom = snap_norm if snap_norm > 0.0 else 1.0
        print("Initial residual: {:3.2e}".format(init_res / denom))

        while (curr_res / init_res > projection_relnorm_cutoff) and (proj_it < projection_max_its):
            Jf_np = jac_decode(y).detach().cpu().numpy()
            w_rec_np = w_rec_t.detach().cpu().numpy()

            rhs = Jf_np.T @ (w_rec_np - snap)
            lhs = Jf_np.T @ Jf_np
            dy = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            with torch.no_grad():
                y = y - torch.tensor(dy, dtype=y.dtype, device=y.device)
                w_rec_t = decode(y)

            curr_res = np.linalg.norm((w_rec_t - snap_t).detach().cpu().numpy())
            proj_it += 1

        Jf_np = jac_decode(y).detach().cpu().numpy()
        w_tilde = w_rec_t.detach().cpu().numpy()
        final_res = np.linalg.norm(w_tilde - snap)
        print("Final residual: {:3.2e}".format(final_res / denom))

        ires = res(w_tilde, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_tilde, dt, JDxec, JDyec, Eye)
        Wi = Ji @ Jf_np

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    POD-ANN manifold ROM with decoder

        w(y) = u_ref + basis @ y + basis2 @ ann_model(y)

    The reduced coordinates y are advanced with Gauss-Newton LSPG.

    Notes
    -----
    - `ref` is kept only for backward compatibility and is not used.
    - If `u_ref` is None, a zero reference is used.
    """

    del ref

    w0_np = np.asarray(w0, dtype=np.float64).reshape(-1)
    u_ref_np = _prepare_reference(u_ref, w0_np.size)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    basis = _to_torch_matrix(basis, dtype=torch.float32)
    basis2 = _to_torch_matrix(basis2, dtype=torch.float32)

    device = basis.device
    dtype_t = basis.dtype
    try:
        ann_device_before = next(ann_model.parameters()).device
    except StopIteration:
        ann_device_before = device
    ann_model = _align_module_device(ann_model, device)
    try:
        ann_device_after = next(ann_model.parameters()).device
    except StopIteration:
        ann_device_after = device

    u_ref_t = _to_torch_vector(u_ref_np, dtype=dtype_t, device=device)
    y0 = basis.T @ _to_torch_vector(w0_np - u_ref_np, dtype=dtype_t, device=device)

    with torch.no_grad():
        w0_t = u_ref_t + basis @ y0 + basis2 @ ann_model(y0)

    nred = int(y0.numel())

    snaps = np.zeros((int(w0_t.numel()), num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    snaps[:, 0] = w0_t.detach().cpu().numpy().reshape(-1)
    red_coords[:, 0] = y0.detach().cpu().numpy().reshape(-1)

    wp = w0_t.detach().clone()
    yp = y0.detach().clone()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    def decode(y, with_grad=True):
        if with_grad:
            return u_ref_t + basis @ y + basis2 @ ann_model(y)
        with torch.no_grad():
            return u_ref_t + basis @ y + basis2 @ ann_model(y)

    jacfwdfunc = _build_ann_decoder_jacobian(
        basis=basis,
        basis2=basis2,
        ann_eval=ann_model,
    )

    print(f"Running POD-ANN ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        def res(w_np):
            return inviscid_burgers_res2D(
                w_np,
                grid_x,
                grid_y,
                dt,
                wp.detach().cpu().numpy().reshape(-1),
                mu,
                Dxec,
                Dyec,
            )

        def jac(w_np):
            return inviscid_burgers_exact_jac2D(w_np, dt, JDxec, JDyec, Eye)

        y, resnorms, times = gauss_newton_pod_ann(
            func=res,
            jac=jac,
            y0=yp,
            decode=decode,
            jacfwdfunc=jacfwdfunc,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_np,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        with torch.no_grad():
            w_t = decode(y, with_grad=False)

        red_coords[:, istep + 1] = y.detach().cpu().numpy().reshape(-1)
        snaps[:, istep + 1] = w_t.detach().cpu().numpy().reshape(-1)

        wp = w_t.detach().clone()
        yp = y.detach().clone()

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_ecsw(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    weights,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    ECSW POD-ANN manifold ROM with decoder

        w(y; mu) = u_ref + V y + Vbar N([y, mu])

    If u_ref is None, a zero reference is used.
    """

    del ref

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref_np = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full = w0.size
    n_cells = n_full // 2

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]
    idx = np.concatenate((augmented_sample, n_cells + augmented_sample))

    basis = _to_torch_matrix(basis, dtype=torch.float32)
    basis2 = _to_torch_matrix(basis2, dtype=torch.float32)

    device = basis.device
    dtype_t = basis.dtype
    try:
        ann_device_before = next(ann_model.parameters()).device
    except StopIteration:
        ann_device_before = device
    ann_model = _align_module_device(ann_model, device)
    try:
        ann_device_after = next(ann_model.parameters()).device
    except StopIteration:
        ann_device_after = device

    u_ref_t = _to_torch_vector(u_ref_np, dtype=dtype_t, device=device)
    u_ref_loc_t = u_ref_t[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]
    tmu = _to_torch_vector(np.asarray(mu, dtype=np.float64), dtype=dtype_t, device=device)

    y0 = basis.T @ _to_torch_vector(w0 - u_ref_np, dtype=dtype_t, device=device)

    with torch.no_grad():
        w0_loc = u_ref_loc_t + V @ y0 + Vbar @ ann_model(torch.cat((y0, tmu)))

    nred = int(y0.numel())
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = y0.detach().cpu().numpy().reshape(-1)

    wp = w0_loc.detach().clone()
    yp = y0.detach().clone()

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])
    shp = (dy.size, dx.size)

    lbc = np.zeros(sample_inds.shape[0], dtype=np.float64)
    rr, cc = np.unravel_index(sample_inds, shp)
    for i, (r, c) in enumerate(zip(rr, cc)):
        if c == 0:
            lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]

    src = dt * 0.02 * np.exp(mu[1] * xc)
    src = np.tile(src, dy.size)
    src = src[sample_inds]

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    def decode(y, with_grad=True):
        if with_grad:
            return u_ref_loc_t + V @ y + Vbar @ ann_model(torch.cat((y, tmu)))
        with torch.no_grad():
            return u_ref_loc_t + V @ y + Vbar @ ann_model(torch.cat((y, tmu)))

    def _ann_eval(y):
        return ann_model(torch.cat((y, tmu)))

    jacfwdfunc = _build_ann_decoder_jacobian(
        basis=V,
        basis2=Vbar,
        ann_eval=_ann_eval,
    )

    print(f"Running POD-ANN ECSW ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        def res(w):
            return inviscid_burgers_res2D_ecsw(
                w,
                grid_x,
                grid_y,
                dt,
                wp,
                mu,
                JDxec_ecsw,
                JDyec_ecsw,
                sample_inds,
                augmented_sample,
                lbc,
                src,
            )

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(
                w,
                dt,
                JDxec_ecsw,
                JDyec_ecsw,
                Eye_ecsw,
                sample_inds,
                augmented_sample,
            )

        y, resnorms, times = gauss_newton_pod_ann_ecsw(
            func=res,
            jac=jac,
            y0=yp,
            decode=decode,
            jacfwdfunc=jacfwdfunc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weight=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_loc_t.detach().cpu().numpy(),
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        with torch.no_grad():
            w_loc = decode(y, with_grad=False)

        red_coords[:, istep + 1] = y.detach().cpu().numpy().reshape(-1)
        wp = w_loc.detach().clone()
        yp = y.detach().clone()

    return red_coords, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
):
    """
    POD-ANN manifold ROM with decoder

        w(y, t; mu) = u_ref + V y + Vbar N([mu1, mu2, t])

    The nonlinear correction is independent of y, so dw/dy = V.
    """

    del ref

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    N = w0.size
    u_ref_np = _prepare_reference(u_ref, N)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    V = np.asarray(basis.detach().cpu().numpy() if hasattr(basis, "detach") else basis, dtype=np.float64)
    Vbar = np.asarray(basis2.detach().cpu().numpy() if hasattr(basis2, "detach") else basis2, dtype=np.float64)
    ann_model = _align_module_device(ann_model, torch.device("cpu"))

    nred = V.shape[1]
    y = V.T @ (w0 - u_ref_np)
    y = y.astype(np.float64, copy=False)

    snaps = np.zeros((N, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    tgrid = dt * np.arange(num_steps + 1, dtype=np.float64)

    device = next(ann_model.parameters()).device if hasattr(ann_model, "parameters") else torch.device("cpu")
    mu1 = float(mu[0])
    mu2 = float(mu[1])

    def offset_np(tval):
        inp = torch.tensor([mu1, mu2, float(tval)], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qbar = ann_model(inp).squeeze(0)
        qbar_np = qbar.detach().cpu().numpy()
        return u_ref_np + Vbar @ qbar_np

    w = offset_np(tgrid[0]) + V @ y

    snaps[:, 0] = w
    red_coords[:, 0] = y

    wp = w.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-ANN Case 2 ROM of size {nred} for mu1={mu1}, mu2={mu2}")

    for k in range(num_steps):
        tk1 = tgrid[k + 1]
        off = offset_np(tk1)

        yk = y.copy()
        wk = off + V @ yk

        def compute_residual(w_state):
            return inviscid_burgers_res2D(w_state, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def compute_jacobian(w_state):
            return inviscid_burgers_exact_jac2D(w_state, dt, JDxec, JDyec, Eye)

        t0 = time.time()
        fk = compute_residual(wk)
        res_time += time.time() - t0
        init_norm = np.linalg.norm(fk) + 1e-30

        resnorms = []

        for it in range(max_its):
            t0 = time.time()
            fk = compute_residual(wk)
            res_time += time.time() - t0

            rnorm = np.linalg.norm(fk)
            resnorms.append(rnorm)

            if rnorm / init_norm < relnorm_cutoff:
                break

            if len(resnorms) > 1:
                rel_drop = abs((resnorms[-2] - resnorms[-1]) / (resnorms[-2] + 1e-30))
                if rel_drop < min_delta:
                    break

            t0 = time.time()
            J = compute_jacobian(wk)
            jac_time += time.time() - t0

            t0 = time.time()
            JV = J @ V
            dy, *_ = np.linalg.lstsq(JV, -fk, rcond=None)
            ls_time += time.time() - t0

            yk += dy
            wk = off + V @ yk

        num_its += len(resnorms)

        y = yk
        w = wk
        wp = w.copy()

        snaps[:, k + 1] = w
        red_coords[:, k + 1] = y

        print(f"  step {k:4d}: GN iters={len(resnorms):2d} rel={resnorms[-1] / init_norm:.2e}")

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    POD-ANN Case 2 with enriched residual testing (PROM only).

    Trial manifold:

        w(y, t; mu) = u_ref + V y + Vbar N([mu1, mu2, t])

    Unknowns are still only y in R^n_p, but Gauss-Newton minimizes

        0.5 * || s(y) ||_2^2,
        s(y) = (J(w(y)) V_tot)^T r(w(y)),

    where V_tot = [V, Vbar].
    """

    del ref

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    n_full = w0.size
    u_ref_np = _prepare_reference(u_ref, n_full)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    V = np.asarray(
        basis.detach().cpu().numpy() if hasattr(basis, "detach") else basis,
        dtype=np.float64,
    )
    Vbar = np.asarray(
        basis2.detach().cpu().numpy() if hasattr(basis2, "detach") else basis2,
        dtype=np.float64,
    )
    Vtot = np.concatenate((V, Vbar), axis=1)

    ann_model = _align_module_device(ann_model, torch.device("cpu"))
    try:
        device = next(ann_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    nred = V.shape[1]
    y = (V.T @ (w0 - u_ref_np)).astype(np.float64, copy=False)

    snaps = np.zeros((n_full, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    tgrid = dt * np.arange(num_steps + 1, dtype=np.float64)

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    def offset_np(tval):
        inp = torch.tensor([mu1, mu2, float(tval)], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qbar = ann_model(inp).squeeze(0)
        qbar_np = qbar.detach().cpu().numpy()
        return u_ref_np + Vbar @ qbar_np

    w = offset_np(tgrid[0]) + V @ y
    wp = w.copy()

    snaps[:, 0] = w
    red_coords[:, 0] = y

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(
        "Running POD-ANN Case 2 (enriched residual testing) ROM "
        f"of size {nred} for mu1={mu1}, mu2={mu2}"
    )

    for k in range(num_steps):
        tk1 = tgrid[k + 1]
        off = offset_np(tk1)

        yk = y.copy()
        wk = off + V @ yk
        resnorms = []
        init_norm = None

        for _ in range(max_its):
            t0 = time.time()
            r = inviscid_burgers_res2D(wk, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)
            res_time += time.time() - t0

            t0 = time.time()
            J = inviscid_burgers_exact_jac2D(wk, dt, JDxec, JDyec, Eye)
            jac_time += time.time() - t0

            t0 = time.time()
            JV = J @ V
            JVTOT = J @ Vtot
            s = JVTOT.T @ r
            snorm = np.linalg.norm(s)
            if init_norm is None:
                init_norm = snorm + 1e-30
            resnorms.append(snorm)

            if snorm / init_norm < relnorm_cutoff:
                ls_time += time.time() - t0
                break

            if len(resnorms) > 1:
                rel_drop = abs((resnorms[-2] - resnorms[-1]) / (resnorms[-2] + 1e-30))
                if rel_drop < min_delta:
                    ls_time += time.time() - t0
                    break

            Z = JVTOT.T @ JV
            dy = _solve_reduced_update(
                Z,
                s,
                linear_solver=linear_solver,
                normal_eq_reg=normal_eq_reg,
            )
            ls_time += time.time() - t0

            yk += dy
            wk = off + V @ yk

        num_its += len(resnorms)
        y = yk
        w = wk
        wp = w.copy()

        snaps[:, k + 1] = w
        red_coords[:, k + 1] = y

        rel = (resnorms[-1] / (init_norm if init_norm is not None else 1.0)) if resnorms else np.nan
        print(f"  step {k:4d}: GN iters={len(resnorms):2d} rel={rel:.2e}")

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case3(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    POD-ANN manifold ROM with decoder

        w(y, t; mu) = u_ref + V y + Vbar N([y, mu1, mu2, t])
    """

    del ref

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    basis = _to_torch_matrix(basis, dtype=torch.float32)
    basis2 = _to_torch_matrix(basis2, dtype=torch.float32)

    device = basis.device
    dtype_t = basis.dtype
    ann_model = _align_module_device(ann_model, device)

    w0_np = np.asarray(w0, dtype=np.float64).reshape(-1)
    N = w0_np.size
    u_ref_np = _prepare_reference(u_ref, N)
    u_ref_t = _to_torch_vector(u_ref_np, dtype=dtype_t, device=device)

    mu_np = np.asarray(mu, dtype=np.float64).reshape(-1)
    if mu_np.size != 2:
        raise ValueError(f"mu must have length 2, got {mu_np.shape}")

    tmu = _to_torch_vector(mu_np, dtype=dtype_t, device=device)
    y0 = basis.T @ _to_torch_vector(w0_np - u_ref_np, dtype=dtype_t, device=device)

    def _ann_eval(y_vec, t_scalar):
        x = torch.cat([y_vec.reshape(-1), tmu, t_scalar.reshape(1)], dim=0)
        out = ann_model(x)
        return out.reshape(-1)

    t0_scalar = torch.tensor(0.0, dtype=dtype_t, device=device)

    with torch.no_grad():
        w0_t = u_ref_t + basis @ y0 + basis2 @ _ann_eval(y0, t0_scalar)

    nred = int(y0.numel())

    snaps = np.zeros((int(w0_t.numel()), num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    snaps[:, 0] = w0_t.detach().cpu().numpy().reshape(-1)
    red_coords[:, 0] = y0.detach().cpu().numpy().reshape(-1)

    wp = w0_t.detach().clone()
    yp = y0.detach().clone()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-ANN Case 3 ROM of size {nred} for mu1={mu_np[0]}, mu2={mu_np[1]}")

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        t_now = torch.tensor((istep + 1) * float(dt), dtype=dtype_t, device=device)

        def decode(y, with_grad=True):
            if with_grad:
                return u_ref_t + basis @ y + basis2 @ _ann_eval(y, t_now)
            with torch.no_grad():
                return u_ref_t + basis @ y + basis2 @ _ann_eval(y, t_now)

        def _ann_eval_t(y_vec):
            return _ann_eval(y_vec, t_now)

        jacfwdfunc = _build_ann_decoder_jacobian(
            basis=basis,
            basis2=basis2,
            ann_eval=_ann_eval_t,
        )

        def res(w_np):
            return inviscid_burgers_res2D(
                w_np,
                grid_x,
                grid_y,
                dt,
                wp.detach().cpu().numpy().reshape(-1),
                mu,
                Dxec,
                Dyec,
            )

        def jac(w_np):
            return inviscid_burgers_exact_jac2D(w_np, dt, JDxec, JDyec, Eye)

        y, resnorms, times = gauss_newton_pod_ann(
            func=res,
            jac=jac,
            y0=yp,
            decode=decode,
            jacfwdfunc=jacfwdfunc,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_np,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        with torch.no_grad():
            w_t = decode(y, with_grad=False)

        red_coords[:, istep + 1] = y.detach().cpu().numpy().reshape(-1)
        snaps[:, istep + 1] = w_t.detach().cpu().numpy().reshape(-1)

        wp = w_t.detach().clone()
        yp = y.detach().clone()

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    ECSW POD-ANN Case 2 ROM with decoder

        w(y, t; mu) = u_ref + V y + Vbar N([mu1, mu2, t])

    The nonlinear correction is independent of y, so dw/dy = V on the sampled mesh.

    Returns
    -------
    red_coords : ndarray
        Reduced coordinates of shape (n_p, num_steps + 1).
    stats : tuple
        (num_its, jac_time, res_time, ls_time)
    """

    del ref

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref_np = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full = w0.size
    n_cells = n_full // 2

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]
    idx = np.concatenate((augmented_sample, n_cells + augmented_sample))

    basis = _to_torch_matrix(basis, dtype=torch.float32)
    basis2 = _to_torch_matrix(basis2, dtype=torch.float32)

    device = basis.device
    dtype_t = basis.dtype
    ann_model = _align_module_device(ann_model, device)

    u_ref_t = _to_torch_vector(u_ref_np, dtype=dtype_t, device=device)
    u_ref_loc_t = u_ref_t[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]

    y0 = basis.T @ _to_torch_vector(w0 - u_ref_np, dtype=dtype_t, device=device)
    nred = int(y0.numel())

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    def _offset_loc(t_value):
        x = torch.tensor([mu1, mu2, float(t_value)], dtype=dtype_t, device=device)
        with torch.no_grad():
            qbar = ann_model(x).reshape(-1)
        return u_ref_loc_t + Vbar @ qbar

    wp = (V @ y0 + _offset_loc(0.0)).detach().clone()
    yp = y0.detach().clone()

    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = y0.detach().cpu().numpy().reshape(-1)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-ANN Case 2 ECSW ROM of size {nred} for mu1={mu1}, mu2={mu2}")

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        off_loc_t = _offset_loc((istep + 1) * float(dt))

        def decode(y, with_grad=True):
            if with_grad:
                return off_loc_t + V @ y
            with torch.no_grad():
                return off_loc_t + V @ y

        def jacfwdfunc(y):
            del y
            return V

        def res(w_loc):
            return inviscid_burgers_res2D_ecsw(
                w_loc,
                grid_x,
                grid_y,
                dt,
                wp,
                mu,
                JDxec_ecsw,
                JDyec_ecsw,
                sample_inds,
                augmented_sample,
            )

        def jac(w_loc):
            return inviscid_burgers_exact_jac2D_ecsw(
                w_loc,
                dt,
                JDxec_ecsw,
                JDyec_ecsw,
                Eye_ecsw,
                sample_inds,
                augmented_sample,
            )

        y, resnorms, times = gauss_newton_pod_ann_ecsw(
            func=res,
            jac=jac,
            y0=yp,
            decode=decode,
            jacfwdfunc=jacfwdfunc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weight=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_loc_t.detach().cpu().numpy(),
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        with torch.no_grad():
            w_loc = decode(y, with_grad=False)

        red_coords[:, istep + 1] = y.detach().cpu().numpy().reshape(-1)
        wp = w_loc.detach().clone()
        yp = y.detach().clone()

    return red_coords, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    ECSW POD-ANN Case 2 with enriched residual testing.

    Trial manifold:

        w(y, t; mu) = u_ref + V y + Vbar N([mu1, mu2, t])

    Unknowns remain y in the primary space, but Gauss-Newton uses
    weighted projected residual testing through V_tot = [V, Vbar]:

        s = (W J V_tot)^T (W r),
        Z = (W J V_tot)^T (W J V).
    """

    del ref

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref_np = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full = w0.size
    n_cells = n_full // 2

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]
    weights_full = np.concatenate((sample_weights_cells, sample_weights_cells)).astype(np.float64)
    idx = np.concatenate((augmented_sample, n_cells + augmented_sample))

    V_global = np.asarray(basis.detach().cpu().numpy() if hasattr(basis, "detach") else basis, dtype=np.float64)
    Vbar_global = np.asarray(
        basis2.detach().cpu().numpy() if hasattr(basis2, "detach") else basis2,
        dtype=np.float64,
    )

    ann_model = _align_module_device(ann_model, torch.device("cpu"))
    try:
        device = next(ann_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    u_ref_loc = u_ref_np[idx]
    V_loc = V_global[idx, :]
    Vbar_loc = Vbar_global[idx, :]
    Vtot_loc = np.concatenate((V_loc, Vbar_loc), axis=1)

    y0 = V_global.T @ (w0 - u_ref_np)
    nred = int(y0.size)

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    def _offset_loc(t_value):
        x = torch.tensor([mu1, mu2, float(t_value)], dtype=torch.float32, device=device)
        with torch.no_grad():
            qbar = ann_model(x).reshape(-1).detach().cpu().numpy()
        return u_ref_loc + Vbar_loc @ qbar

    wp = V_loc @ y0 + _offset_loc(0.0)
    yp = y0.copy()

    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = y0.reshape(-1)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(
        "Running POD-ANN Case 2 ECSW ROM (enriched residual testing) "
        f"of size {nred} for mu1={mu1}, mu2={mu2}"
    )

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        off_loc_t = _offset_loc((istep + 1) * float(dt))
        y = yp.copy()
        w_loc = off_loc_t + V_loc @ y

        resnorms = []
        init_norm = None

        for _ in range(max_its):
            t0 = time.time()
            r = inviscid_burgers_res2D_ecsw(
                w_loc,
                grid_x,
                grid_y,
                dt,
                wp,
                mu,
                JDxec_ecsw,
                JDyec_ecsw,
                sample_inds,
                augmented_sample,
            )
            res_time += time.time() - t0

            t0 = time.time()
            J = inviscid_burgers_exact_jac2D_ecsw(
                w_loc,
                dt,
                JDxec_ecsw,
                JDyec_ecsw,
                Eye_ecsw,
                sample_inds,
                augmented_sample,
            )
            jac_time += time.time() - t0

            t0 = time.time()
            rw = weights_full * r
            JV = J @ V_loc
            JVTOT = J @ Vtot_loc
            JVw = weights_full[:, None] * JV
            JVTOTw = weights_full[:, None] * JVTOT

            s = JVTOTw.T @ rw
            snorm = np.linalg.norm(s)
            if init_norm is None:
                init_norm = snorm + 1e-30
            resnorms.append(snorm)

            if snorm / init_norm < relnorm_cutoff:
                ls_time += time.time() - t0
                break

            if len(resnorms) > 1:
                rel_drop = abs((resnorms[-2] - resnorms[-1]) / (resnorms[-2] + 1e-30))
                if rel_drop < min_delta:
                    ls_time += time.time() - t0
                    break

            Z = JVTOTw.T @ JVw
            dy = _solve_reduced_update(
                Z,
                s,
                linear_solver=linear_solver,
                normal_eq_reg=normal_eq_reg,
            )
            ls_time += time.time() - t0

            y += dy
            w_loc = off_loc_t + V_loc @ y

        num_its += len(resnorms)
        red_coords[:, istep + 1] = y.reshape(-1)
        wp = w_loc.copy()
        yp = y.copy()

        rel = (resnorms[-1] / (init_norm if init_norm is not None else 1.0)) if resnorms else np.nan
        print(f"  step {istep:4d}: GN iters={len(resnorms):2d} rel={rel:.2e}")

    return red_coords, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case3_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    ann_model,
    ref,
    basis,
    basis2,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=0.1,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    ECSW POD-ANN Case 3 ROM with decoder

        w(y, t; mu) = u_ref + V y + Vbar N([y, mu1, mu2, t])

    Returns
    -------
    red_coords : ndarray
        Reduced coordinates of shape (n_p, num_steps + 1).
    stats : tuple
        (num_its, jac_time, res_time, ls_time)
    """

    del ref

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref_np = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full = w0.size
    n_cells = n_full // 2

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]
    idx = np.concatenate((augmented_sample, n_cells + augmented_sample))

    basis = _to_torch_matrix(basis, dtype=torch.float32)
    basis2 = _to_torch_matrix(basis2, dtype=torch.float32)

    device = basis.device
    dtype_t = basis.dtype
    ann_model = _align_module_device(ann_model, device)

    u_ref_t = _to_torch_vector(u_ref_np, dtype=dtype_t, device=device)
    u_ref_loc_t = u_ref_t[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]

    y0 = basis.T @ _to_torch_vector(w0 - u_ref_np, dtype=dtype_t, device=device)
    nred = int(y0.numel())

    mu_np = np.asarray(mu, dtype=np.float64).reshape(-1)
    if mu_np.size != 2:
        raise ValueError(f"mu must have length 2, got {mu_np.shape}")

    tmu = _to_torch_vector(mu_np, dtype=dtype_t, device=device)

    t0_now = torch.tensor(0.0, dtype=dtype_t, device=device)

    def _ann_eval(y_vec, t_scalar):
        x = torch.cat([y_vec.reshape(-1), tmu, t_scalar.reshape(1)], dim=0)
        out = ann_model(x)
        return out.reshape(-1)

    with torch.no_grad():
        wp = (u_ref_loc_t + V @ y0 + Vbar @ _ann_eval(y0, t0_now)).detach().clone()

    yp = y0.detach().clone()

    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = y0.detach().cpu().numpy().reshape(-1)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-ANN Case 3 ECSW ROM of size {nred} for mu1={mu_np[0]}, mu2={mu_np[1]}")

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        t_now = torch.tensor((istep + 1) * float(dt), dtype=dtype_t, device=device)

        def decode(y, with_grad=True):
            if with_grad:
                return u_ref_loc_t + V @ y + Vbar @ _ann_eval(y, t_now)
            with torch.no_grad():
                return u_ref_loc_t + V @ y + Vbar @ _ann_eval(y, t_now)

        def _ann_eval_t(y_vec):
            return _ann_eval(y_vec, t_now)

        jacfwdfunc = _build_ann_decoder_jacobian(
            basis=V,
            basis2=Vbar,
            ann_eval=_ann_eval_t,
        )

        def res(w_loc):
            return inviscid_burgers_res2D_ecsw(
                w_loc,
                grid_x,
                grid_y,
                dt,
                wp,
                mu,
                JDxec_ecsw,
                JDyec_ecsw,
                sample_inds,
                augmented_sample,
            )

        def jac(w_loc):
            return inviscid_burgers_exact_jac2D_ecsw(
                w_loc,
                dt,
                JDxec_ecsw,
                JDyec_ecsw,
                Eye_ecsw,
                sample_inds,
                augmented_sample,
            )

        y, resnorms, times = gauss_newton_pod_ann_ecsw(
            func=res,
            jac=jac,
            y0=yp,
            decode=decode,
            jacfwdfunc=jacfwdfunc,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weight=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_loc_t.detach().cpu().numpy(),
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        with torch.no_grad():
            w_loc = decode(y, with_grad=False)

        red_coords[:, istep + 1] = y.detach().cpu().numpy().reshape(-1)
        wp = w_loc.detach().clone()
        yp = y.detach().clone()

    return red_coords, (num_its, jac_time, res_time, ls_time)
