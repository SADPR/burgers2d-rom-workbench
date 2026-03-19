# linear_manifold.py
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

from .core import (
    get_ops,
    inviscid_burgers_res2D,
    inviscid_burgers_res2D_ecsw,
    inviscid_burgers_exact_jac2D,
    inviscid_burgers_exact_jac2D_ecsw,
)

from .ecsw_utils import generate_augmented_mesh
from .cluster_utils import select_cluster_reduced

from .gauss_newton import (
    gauss_newton_LSPG,
    gauss_newton_ECSW_2D,
    gauss_newton_LSPG_local,
    gauss_newton_LSPG_local_ecsw,
)


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


def _select_initial_cluster(w, uc_list):
    """
    Select the initial local cluster in full space by nearest centroid.
    """
    d2 = [np.linalg.norm(w - np.asarray(uc, dtype=np.float64)) ** 2 for uc in uc_list]
    return int(np.argmin(d2))


def compute_ECSW_training_matrix_2D(
    snaps,
    prev_snaps,
    basis,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
):
    """
    ECSW training matrix for the global affine LSPG ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    n_red = basis.shape[1]

    C = np.zeros((n_red * n_snaps, n_hdm))

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]

        ires = res(snap, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(snap, dt, JDxec, JDyec, Eye)
        Wi = Ji @ basis

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_local(
    snaps,
    prev_snaps,
    u0_list,
    V_list,
    d_const,
    g_list,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    use_projection=True,
):
    """
    ECSW training matrix for the local affine LSPG ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    r_max = max(V.shape[1] for V in V_list)

    C = np.zeros((r_max * n_snaps, n_hdm))

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        u_i = snaps[:, isnap]
        u_prev = prev_snaps[:, isnap]

        k0 = int(np.argmin([np.linalg.norm(u_i - u0_k) ** 2 for u0_k in u0_list]))

        V_k0 = V_list[k0]
        u0_k0 = u0_list[k0]
        y_k0 = V_k0.T @ (u_i - u0_k0)

        k = select_cluster_reduced(k0, y_k0, d_const, g_list)

        u0_k = u0_list[k]
        V_k = V_list[k]
        r_k = V_k.shape[1]

        if use_projection:
            q_i = V_k.T @ (u_i - u0_k)
            u_tilde = u0_k + V_k @ q_i
        else:
            u_tilde = u_i

        ires = res(u_tilde, grid_x, grid_y, dt, u_prev, mu, Dxec, Dyec)
        Ji = jac(u_tilde, dt, JDxec, JDyec, Eye)
        Wi = Ji @ V_k

        row0 = isnap * r_max
        row1 = row0 + r_k

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def inviscid_burgers_implicit2D_LSPG(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
):
    """
    Global affine LSPG ROM for the 2D inviscid Burgers equation:

        w ≈ u_ref + basis @ y

    If u_ref is None, a zero reference is used.
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, w0.size)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    nred = basis.shape[1]
    snaps = np.zeros((w0.size, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    y0 = basis.T @ (w0 - u_ref)
    w_init = u_ref + basis @ y0

    snaps[:, 0] = w_init
    red_coords[:, 0] = y0

    wp = w_init.copy()
    yp = y0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running global affine LSPG ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for it in range(num_steps):
        print(f" ... Working on timestep {it}")

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        y, resnorms, times = gauss_newton_LSPG(
            func=res,
            jac=jac,
            basis=basis,
            y0=yp,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w = u_ref + basis @ y

        red_coords[:, it + 1] = y
        snaps[:, it + 1] = w

        wp = w.copy()
        yp = y.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    basis,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
):
    """
    Global affine ECSW-LSPG HROM for the 2D inviscid Burgers equation:

        w ≈ u_ref + basis @ y

    If u_ref is None, a zero reference is used.

    Returns
    -------
    red_coords : ndarray
        Reduced coordinates of shape (r, num_steps + 1)
    stats : tuple
        (num_its, jac_time, res_time, ls_time)
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full_scalar = w0.size // 2
    nred = basis.shape[1]

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_full_scalar).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights = np.hstack((weights[sample_inds], weights[sample_inds])).astype(np.float64)

    idx = np.concatenate((augmented_sample, n_full_scalar + augmented_sample))

    basis_ecsw = basis[idx, :]
    u_ref_ecsw = u_ref[idx]

    y0 = basis.T @ (w0 - u_ref)
    w0_full = u_ref + basis @ y0

    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = y0

    wp = w0_full[idx].copy()
    yp = y0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running global affine ECSW-LSPG ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    for it in range(num_steps):
        print(f" ... Working on timestep {it}")

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

        y, resnorms, times = gauss_newton_ECSW_2D(
            func=res,
            jac=jac,
            basis=basis_ecsw,
            y0=yp,
            sample_inds=sample_inds,
            augmented_sample=augmented_sample,
            sample_weights=sample_weights,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u_ref_ecsw,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_ecsw = u_ref_ecsw + basis_ecsw @ y

        red_coords[:, it + 1] = y
        wp = w_ecsw.copy()
        yp = y.copy()

    return red_coords, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_local(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    u0_list,
    V_list,
    uc_list,
    cluster_select_fun=None,
    d_const=None,
    g_list=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
):
    """
    Local affine LSPG ROM for the 2D inviscid Burgers equation:

        w ≈ u0_k + V_k @ q

    Cluster selection modes:
      - Full-space selector: cluster_select_fun(w, uc_list)
      - Reduced-space selector: cluster_select_fun(k, q_k, d_const, g_list)
        when both d_const and g_list are provided.
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)

    N = w0.size
    K = len(V_list)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    n_max = max(np.asarray(V, dtype=np.float64).shape[1] for V in V_list)

    snaps = np.zeros((N, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((n_max, num_steps + 1), dtype=np.float64)
    cluster_history = []

    if cluster_select_fun is None:
        k = _select_initial_cluster(w0, uc_list)
    else:
        # If reduced-space data are provided, initialize in full space once,
        # then switch in reduced space inside the time loop.
        if d_const is not None and g_list is not None:
            k = _select_initial_cluster(w0, uc_list)
        else:
            k = cluster_select_fun(w0, uc_list)
    print(f"[LOCAL-AFFINE-LSPG] Initial cluster k = {k} / {K - 1}")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    n_k = V_k.shape[1]

    q0 = V_k.T @ (w0 - u0_k)
    w_init = u0_k + V_k @ q0

    snaps[:, 0] = w_init
    red_coords[:n_k, 0] = q0

    wp = w_init.copy()
    qp = q0.copy()
    cluster_history.append(k)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"[LOCAL-AFFINE-LSPG] Running local affine LSPG with {K} clusters, dt={dt}")

    for it in range(num_steps):
        print(f"[LOCAL-AFFINE-LSPG] Timestep {it}/{num_steps}")

        if cluster_select_fun is None:
            k_new = _select_initial_cluster(wp, uc_list)
        else:
            if d_const is not None and g_list is not None:
                k_new = cluster_select_fun(k, qp, d_const, g_list)
            else:
                k_new = cluster_select_fun(wp, uc_list)

        if k_new != k:
            print(f"  -> Cluster switch: {k} -> {k_new}")
            k = k_new
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            n_k = V_k.shape[1]
            qp = V_k.T @ (wp - u0_k)
        else:
            n_k = V_k.shape[1]

        cluster_history.append(k)

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        q, resnorms, times = gauss_newton_LSPG_local(
            func=res,
            jac=jac,
            Vloc=V_k,
            u0loc=u0_k,
            y0=qp,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u0_k,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w = u0_k + V_k @ q

        snaps[:, it + 1] = w
        red_coords[:n_k, it + 1] = q

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


def inviscid_burgers_implicit2D_LSPG_local_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    u0_list,
    V_list,
    uc_list,
    cluster_select_fun=None,
    d_const=None,
    g_list=None,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its=20,
):
    """
    Local affine ECSW-LSPG HROM for the 2D inviscid Burgers equation:

        w ≈ u0_k + V_k @ q
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

    u0_loc_list = []
    V_loc_list = []
    for k in range(K):
        u0_k = np.asarray(u0_list[k], dtype=np.float64)
        V_k = np.asarray(V_list[k], dtype=np.float64)
        u0_loc_list.append(u0_k[idx_dofs])
        V_loc_list.append(V_k[idx_dofs, :])

    n_max = max(np.asarray(V, dtype=np.float64).shape[1] for V in V_list)

    snaps = np.zeros((N_full, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((n_max, num_steps + 1), dtype=np.float64)
    cluster_history = []

    k = _select_initial_cluster(w0, uc_list)
    print(f"[LOCAL-AFFINE-LSPG-ECSW] Initial cluster k = {k} / {K - 1}")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    u0_loc_k = u0_loc_list[k]
    V_loc_k = V_loc_list[k]
    n_k = V_k.shape[1]

    q0 = V_k.T @ (w0 - u0_k)
    w0_full = u0_k + V_k @ q0
    w0_loc = u0_loc_k + V_loc_k @ q0

    snaps[:, 0] = w0_full
    red_coords[:n_k, 0] = q0

    wp_full = w0_full.copy()
    wp_loc = w0_loc.copy()
    qp = q0.copy()
    cluster_history.append(k)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"[LOCAL-AFFINE-LSPG-ECSW] Running local affine ECSW-LSPG with {K} clusters, dt={dt}")

    for it in range(num_steps):
        print(f"[LOCAL-AFFINE-LSPG-ECSW] Timestep {it}/{num_steps}")

        if cluster_select_fun is None:
            k_new = _select_initial_cluster(wp_full, uc_list)
        else:
            if d_const is None or g_list is None:
                raise ValueError("For reduced-space local ECSW selection, d_const and g_list must be provided.")
            k_new = cluster_select_fun(k, qp, d_const, g_list)

        if k_new != k:
            print(f"  -> Cluster switch: {k} -> {k_new}")
            k = k_new
            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            u0_loc_k = u0_loc_list[k]
            V_loc_k = V_loc_list[k]
            n_k = V_k.shape[1]
            qp = V_k.T @ (wp_full - u0_k)
        else:
            n_k = V_k.shape[1]

        cluster_history.append(k)

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

        q, resnorms, times = gauss_newton_LSPG_local_ecsw(
            res_fun=res_loc,
            jac_fun=jac_loc,
            V_loc=V_loc_k,
            u0_loc=u0_loc_k,
            q0=qp,
            sample_weights_cells=sample_weights_cells,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            u_ref=u0_loc_k,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_full = u0_k + V_k @ q
        w_loc = u0_loc_k + V_loc_k @ q

        snaps[:, it + 1] = w_full
        red_coords[:n_k, it + 1] = q

        wp_full = w_full.copy()
        wp_loc = w_loc.copy()
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
