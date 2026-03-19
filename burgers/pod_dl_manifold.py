# pod_dl_manifold.py
# -*- coding: utf-8 -*-

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
from .gauss_newton import gauss_newton_poddl, gauss_newton_poddl_ecsw

try:
    from torch.func import jacfwd as torch_jacfwd
except Exception:
    torch_jacfwd = functorch.jacfwd


def _prepare_reference(u_ref, size):
    if u_ref is None:
        return np.zeros(size, dtype=np.float64)

    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if u_ref.size != size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {size}")
    return u_ref


def _model_device_dtype(model):
    for p in model.parameters():
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32


def _build_decode_helpers(basis, pod_dl_model, u_ref):
    basis_np = np.asarray(basis, dtype=np.float64)
    u_ref_np = np.asarray(u_ref, dtype=np.float64).reshape(-1)

    device, dtype_t = _model_device_dtype(pod_dl_model)
    basis_t = torch.tensor(basis_np, dtype=dtype_t, device=device)
    u_ref_t = torch.tensor(u_ref_np, dtype=dtype_t, device=device)

    jac_q_z = torch_jacfwd(lambda z: pod_dl_model.decode_from_latent(z).reshape(-1))

    def decode_u(z, with_grad=True):
        if with_grad:
            q = pod_dl_model.decode_from_latent(z).reshape(-1)
            return u_ref_t + basis_t @ q
        with torch.no_grad():
            q = pod_dl_model.decode_from_latent(z).reshape(-1)
            return u_ref_t + basis_t @ q

    def jac_u_z(z):
        dq_dz = jac_q_z(z)
        return basis_t @ dq_dz

    return decode_u, jac_u_z, basis_t, u_ref_t, device, dtype_t


def _fit_latent_to_snapshot(
    z_init,
    target_snapshot,
    decode_u,
    jac_u_z,
    max_its=10,
    rel_tol=1e-2,
):
    target_snapshot = np.asarray(target_snapshot, dtype=np.float64).reshape(-1)
    z = z_init.detach().clone()

    with torch.no_grad():
        w_rec = decode_u(z, with_grad=False).detach().cpu().numpy().reshape(-1)

    init_res = np.linalg.norm(w_rec - target_snapshot)
    if init_res <= 0.0:
        return z, w_rec

    curr_res = init_res
    it = 0
    while curr_res / init_res > rel_tol and it < max_its:
        Jf = jac_u_z(z).detach().cpu().numpy()
        r = w_rec - target_snapshot

        JTJ = Jf.T @ Jf
        JTr = Jf.T @ r
        try:
            dz = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            dz, *_ = np.linalg.lstsq(JTJ, JTr, rcond=None)

        with torch.no_grad():
            z -= torch.tensor(dz, dtype=z.dtype, device=z.device)
            w_rec = decode_u(z, with_grad=False).detach().cpu().numpy().reshape(-1)

        curr_res = np.linalg.norm(w_rec - target_snapshot)
        it += 1

    return z, w_rec


def compute_ECSW_training_matrix_2D_pod_dl(
    snaps,
    prev_snaps,
    basis,
    pod_dl_model,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    u_ref=None,
):
    """
    Build ECSW training matrix C for POD-DL manifold with latent coordinates.
    """
    snaps = np.asarray(snaps, dtype=np.float64)
    prev_snaps = np.asarray(prev_snaps, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)

    if snaps.ndim != 2 or prev_snaps.ndim != 2:
        raise ValueError("snaps and prev_snaps must be 2D arrays.")
    if snaps.shape != prev_snaps.shape:
        raise ValueError(
            f"snaps and prev_snaps shape mismatch: {snaps.shape} vs {prev_snaps.shape}"
        )
    if basis.ndim != 2 or basis.shape[0] != snaps.shape[0]:
        raise ValueError(
            f"basis shape mismatch: basis={basis.shape}, snapshots={snaps.shape}"
        )

    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    if n_tot % 2 != 0:
        raise ValueError(f"full state size must be even, got {n_tot}")

    u_ref_np = _prepare_reference(u_ref, n_tot)

    pod_dl_model.eval()
    decode_u, jac_u_z, basis_t, _, device, dtype_t = _build_decode_helpers(
        basis=basis,
        pod_dl_model=pod_dl_model,
        u_ref=u_ref_np,
    )

    q0 = basis.T @ (snaps[:, 0] - u_ref_np)
    q0_t = torch.tensor(q0, dtype=dtype_t, device=device)
    with torch.no_grad():
        z0 = pod_dl_model.encode(q0_t).reshape(-1)
    n_latent = int(z0.numel())

    C = np.zeros((n_latent * n_snaps, n_hdm), dtype=np.float64)
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]

        q_init = basis.T @ (snap - u_ref_np)
        q_init_t = torch.tensor(q_init, dtype=dtype_t, device=device)
        with torch.no_grad():
            z_init = pod_dl_model.encode(q_init_t).reshape(-1)

        z_fit, w_rec = _fit_latent_to_snapshot(
            z_init=z_init,
            target_snapshot=snap,
            decode_u=decode_u,
            jac_u_z=jac_u_z,
            max_its=10,
            rel_tol=1e-2,
        )

        ires = res(w_rec, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_rec, dt, JDxec, JDyec, Eye)
        V = jac_u_z(z_fit).detach().cpu().numpy()
        Wi = Ji @ V

        row0 = isnap * n_latent
        row1 = row0 + n_latent

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def inviscid_burgers_implicit2D_LSPG_pod_dl_2D(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    pod_dl_model,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
):
    """
    POD-DL manifold ROM:

        u(z) = u_ref + V N(z)
        du/dz = V dN/dz

    where N(z) is the decoder from latent coordinates to POD coefficients.
    """
    if not hasattr(pod_dl_model, "encode"):
        raise AttributeError("pod_dl_model must implement 'encode(q_raw)'.")
    if not hasattr(pod_dl_model, "decode_from_latent"):
        raise AttributeError("pod_dl_model must implement 'decode_from_latent(z)'.")

    w0_np = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis_np = np.asarray(basis, dtype=np.float64)
    if basis_np.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis_np.shape}")
    if basis_np.shape[0] != w0_np.size:
        raise ValueError(
            f"basis row size mismatch: basis has {basis_np.shape[0]}, w0 has {w0_np.size}"
        )

    u_ref_np = _prepare_reference(u_ref, w0_np.size)
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    pod_dl_model.eval()
    decode_u, jac_u_z, basis_t, _, device, dtype_t = _build_decode_helpers(
        basis=basis_np,
        pod_dl_model=pod_dl_model,
        u_ref=u_ref_np,
    )

    q0 = basis_np.T @ (w0_np - u_ref_np)
    q0_t = torch.tensor(q0, dtype=dtype_t, device=device)

    with torch.no_grad():
        z0 = pod_dl_model.encode(q0_t).reshape(-1)
        w0_t = decode_u(z0, with_grad=False)

    n_dofs = int(w0_t.numel())
    n_z = int(z0.numel())

    snaps = np.zeros((n_dofs, num_steps + 1), dtype=np.float64)
    latent = np.zeros((n_z, num_steps + 1), dtype=np.float64)
    snaps[:, 0] = w0_t.detach().cpu().numpy().reshape(-1)
    latent[:, 0] = z0.detach().cpu().numpy().reshape(-1)

    wp = w0_t.detach().clone()
    zp = z0.detach().clone()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-DL PROM with latent size {n_z} for mu1={mu[0]}, mu2={mu[1]}")

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        wp_np = wp.detach().cpu().numpy().reshape(-1)

        def res(w_np):
            return inviscid_burgers_res2D(
                w_np,
                grid_x,
                grid_y,
                dt,
                wp_np,
                mu,
                Dxec,
                Dyec,
            )

        def jac(w_np):
            return inviscid_burgers_exact_jac2D(w_np, dt, JDxec, JDyec, Eye)

        z, resnorms, times = gauss_newton_poddl(
            func=res,
            jac=jac,
            z0=zp,
            decode=decode_u,
            jac_u_z=jac_u_z,
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
            w_t = decode_u(z, with_grad=False)

        snaps[:, istep + 1] = w_t.detach().cpu().numpy().reshape(-1)
        latent[:, istep + 1] = z.detach().cpu().numpy().reshape(-1)

        wp = w_t.detach().clone()
        zp = z.detach().clone()

    return snaps, latent, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_dl_2D_ecsw(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    pod_dl_model,
    weights,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
):
    """
    ECSW POD-DL manifold ROM in latent coordinates z.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if basis.ndim != 2 or basis.shape[0] != w0.size:
        raise ValueError(
            f"basis shape mismatch for POD-DL ECSW: basis={basis.shape}, w0={w0.shape}"
        )

    u_ref_np = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full = w0.size
    n_cells = n_full // 2
    if n_full % 2 != 0:
        raise ValueError(f"full state size must be even, got {n_full}")

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_cells).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]
    idx = np.concatenate((augmented_sample, n_cells + augmented_sample))

    basis_loc = basis[idx, :]
    u_ref_loc = u_ref_np[idx]

    pod_dl_model.eval()
    decode_u, jac_u_z, _, _, device, dtype_t = _build_decode_helpers(
        basis=basis_loc,
        pod_dl_model=pod_dl_model,
        u_ref=u_ref_loc,
    )

    q0 = basis.T @ (w0 - u_ref_np)
    q0_t = torch.tensor(q0, dtype=dtype_t, device=device)
    with torch.no_grad():
        z0 = pod_dl_model.encode(q0_t).reshape(-1)
        w0_loc = decode_u(z0, with_grad=False)

    n_latent = int(z0.numel())
    latent = np.zeros((n_latent, num_steps + 1), dtype=np.float64)
    latent[:, 0] = z0.detach().cpu().numpy().reshape(-1)

    wp = w0_loc.detach().clone()
    zp = z0.detach().clone()

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])
    shp = (dy.size, dx.size)

    lbc = np.zeros(sample_inds.shape[0], dtype=np.float64)
    rr, cc = np.unravel_index(sample_inds, shp)
    for i, c in enumerate(cc):
        if c == 0:
            lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]

    src = dt * 0.02 * np.exp(mu[1] * xc)
    src = np.tile(src, dy.size)
    src = src[sample_inds]

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-DL ECSW ROM with latent size {n_latent} for mu1={mu[0]}, mu2={mu[1]}")

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

        z, resnorms, times = gauss_newton_poddl_ecsw(
            func=res,
            jac=jac,
            z0=zp,
            decode=decode_u,
            jac_u_z=jac_u_z,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            weight=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_loc,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        with torch.no_grad():
            w_loc = decode_u(z, with_grad=False)

        latent[:, istep + 1] = z.detach().cpu().numpy().reshape(-1)
        wp = w_loc.detach().clone()
        zp = z.detach().clone()

    return latent, (num_its, jac_time, res_time, ls_time)
