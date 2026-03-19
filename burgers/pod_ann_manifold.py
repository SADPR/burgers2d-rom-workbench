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
            Jf = jacfwdfunc(y)
            JJ = Jf.T @ Jf
            Jr = Jf.T @ (approx(y) - torch.tensor(snap, dtype=torch.float32))

            dy, *_ = np.linalg.lstsq(
                JJ.squeeze().detach().cpu().numpy(),
                Jr.squeeze().detach().cpu().numpy(),
                rcond=None,
            )

            y -= torch.tensor(dy, dtype=y.dtype)
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
):
    """
    POD-ANN manifold ROM with decoder

        w(y, t; mu) = u_ref + V y + Vbar N([y, mu1, mu2, t])
    """

    del ref

    try:
        jacfwd = functorch.jacfwd
    except Exception:
        jacfwd = torch.func.jacfwd

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    basis = _to_torch_matrix(basis, dtype=torch.float32)
    basis2 = _to_torch_matrix(basis2, dtype=torch.float32)

    device = basis.device
    dtype_t = basis.dtype

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

        jacfwdfunc = jacfwd(lambda yy: decode(yy, with_grad=True))

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
