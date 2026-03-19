# quadratic_manifold.py
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
    gauss_newton_quadratic_q,
    gauss_newton_LSPG_qm,
    gauss_newton_LSPG_qm_ecsw,
)
from .quadratic_manifold_utils import u_qm
from .quadratic_manifold_utils import J_qm


def _select_initial_cluster(w, uc_list):
    """
    Select the initial local cluster in full space by nearest centroid.
    """
    d2 = [np.linalg.norm(w - np.asarray(uc, dtype=np.float64)) ** 2 for uc in uc_list]
    return int(np.argmin(d2))


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


def inviscid_burgers_implicit2D_LSPG_qm(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    V,
    H,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_q0=20,
    tol_q0=1e-6,
):
    """
    Global quadratic-manifold LSPG ROM for the 2D inviscid Burgers equation:

        w ≈ u_ref + V q + H Q(q)
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, w0.size)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    nred = V.shape[1]
    snaps = np.zeros((w0.size, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)

    q0 = gauss_newton_quadratic_q(
        u_snap=w0,
        V=V,
        H=H,
        u_ref=u_ref,
        max_its=max_its_q0,
        tol_rel=tol_q0,
        verbose=False,
    )
    w_init = u_qm(q0, V, H, u_ref)

    snaps[:, 0] = w_init
    red_coords[:, 0] = q0

    wp = w_init.copy()
    qp = q0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(
        f"[QM-LSPG] Running global quadratic-manifold LSPG ROM of size {nred} "
        f"for mu1={mu[0]}, mu2={mu[1]}"
    )

    for it in range(num_steps):
        print(f"[QM-LSPG] Timestep {it}/{num_steps}")

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        q, resnorms, times = gauss_newton_LSPG_qm(
            func_res=res,
            func_jac=jac,
            V=V,
            H=H,
            u_ref=u_ref,
            q0=qp,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w = u_qm(q, V, H, u_ref)

        red_coords[:, it + 1] = q
        snaps[:, it + 1] = w

        wp = w.copy()
        qp = q.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_qm_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    V,
    H,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_q0=20,
    tol_q0=1e-6,
):
    """
    Global quadratic-manifold ECSW-LSPG HROM for the 2D inviscid Burgers equation:

        w ≈ u_ref + V q + H Q(q)

    Returns
    -------
    red_coords : ndarray
        Reduced coordinates of shape (n, num_steps + 1)
    stats : tuple
        (num_its, jac_time, res_time, ls_time)
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    u_ref = _prepare_reference(u_ref, w0.size)

    _, _, JDxec, JDyec, _ = get_ops(grid_x, grid_y)
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    n_full_scalar = w0.size // 2
    nred = V.shape[1]

    if weights.size != n_full_scalar:
        raise ValueError(
            f"ECSW weights size mismatch: got {weights.size}, expected {n_full_scalar}."
        )

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye_u = sp.identity(n_full_scalar).tocsr()
    Eye_u = Eye_u[sample_inds, :][:, augmented_sample]
    Eye_ecsw = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights = np.hstack((weights[sample_inds], weights[sample_inds])).astype(np.float64)

    idx = np.concatenate((augmented_sample, n_full_scalar + augmented_sample))
    V_ecsw = V[idx, :]
    H_ecsw = H[idx, :]
    u_ref_ecsw = u_ref[idx]

    q0 = gauss_newton_quadratic_q(
        u_snap=w0,
        V=V,
        H=H,
        u_ref=u_ref,
        max_its=max_its_q0,
        tol_rel=tol_q0,
        verbose=False,
    )
    w0_ecsw = u_qm(q0, V_ecsw, H_ecsw, u_ref_ecsw)

    red_coords = np.zeros((nred, num_steps + 1), dtype=np.float64)
    red_coords[:, 0] = q0

    wp = w0_ecsw.copy()
    qp = q0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(
        f"[QM-LSPG-ECSW] Running global quadratic-manifold ECSW-LSPG ROM of size {nred} "
        f"for mu1={mu[0]}, mu2={mu[1]}"
    )

    for it in range(num_steps):
        print(f"[QM-LSPG-ECSW] Timestep {it}/{num_steps}")

        def res_loc(w):
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

        def jac_loc(w):
            return inviscid_burgers_exact_jac2D_ecsw(
                w,
                dt,
                JDxec_ecsw,
                JDyec_ecsw,
                Eye_ecsw,
                sample_inds,
                augmented_sample,
            )

        q, resnorms, times = gauss_newton_LSPG_qm_ecsw(
            res_fun=res_loc,
            jac_fun=jac_loc,
            V_loc=V_ecsw,
            H_loc=H_ecsw,
            u_ref_loc=u_ref_ecsw,
            q0=qp,
            sample_weights=sample_weights,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_ecsw = u_qm(q, V_ecsw, H_ecsw, u_ref_ecsw)
        red_coords[:, it + 1] = q

        wp = w_ecsw.copy()
        qp = q.copy()

    return red_coords, (num_its, jac_time, res_time, ls_time)


def compute_ECSW_training_matrix_2D_qm(
    snaps,
    prev_snaps,
    V,
    H,
    u_ref,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    max_gn_its=20,
    tol_rel=1e-5,
):
    """
    ECSW training matrix for the global quadratic-manifold ROM.
    """
    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    n_red = V.shape[1]

    C = np.zeros((n_red * n_snaps, n_hdm))

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        u_i = snaps[:, isnap]
        u_prev = prev_snaps[:, isnap]

        q_i = gauss_newton_quadratic_q(
            u_snap=u_i,
            V=V,
            H=H,
            u_ref=u_ref,
            max_its=max_gn_its,
            tol_rel=tol_rel,
            verbose=False,
        )

        u_tilde = u_qm(q_i, V, H, u_ref)

        ires = res(u_tilde, grid_x, grid_y, dt, u_prev, mu, Dxec, Dyec)
        Ji = jac(u_tilde, dt, JDxec, JDyec, Eye)

        Wi = Ji @ J_qm(q_i, V, H)

        row0 = isnap * n_red
        row1 = row0 + n_red

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def compute_ECSW_training_matrix_2D_qm_local(
    snaps,
    prev_snaps,
    u0_list,
    V_list,
    H_list,
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
):
    """
    ECSW training matrix for the local quadratic-manifold ROM.
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
        H_k = H_list[k]
        r_k = V_k.shape[1]

        q_i = gauss_newton_quadratic_q(
            u_snap=u_i,
            V=V_k,
            H=H_k,
            u_ref=u0_k,
            max_its=max_gn_its,
            tol_rel=tol_rel,
            verbose=False,
        )

        u_tilde = u_qm(q_i, V_k, H_k, u0_k)

        ires = res(u_tilde, grid_x, grid_y, dt, u_prev, mu, Dxec, Dyec)
        Ji = jac(u_tilde, dt, JDxec, JDyec, Eye)

        Wi = Ji @ J_qm(q_i, V_k, H_k)

        row0 = isnap * r_max
        row1 = row0 + r_k

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


def inviscid_burgers_implicit2D_LSPG_local_qm(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    u0_list,
    V_list,
    H_list,
    uc_list,
    d_const,
    g_list,
    relnorm_cutoff=1e-5,
    min_delta=1e-3,
    init_cluster=None,
    max_its=20,
    max_its_q0=20,
    tol_q0=1e-6,
):
    """
    Local quadratic-manifold LSPG ROM for the 2D inviscid Burgers equation:

        w ≈ u0_k + V_k q + H_k Q(q)
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)

    N = w0.size
    K = len(V_list)

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    n_max = max(np.asarray(V, dtype=np.float64).shape[1] for V in V_list)

    snaps = np.zeros((N, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((n_max, num_steps + 1), dtype=np.float64)
    cluster_history = []

    if init_cluster is None:
        k = _select_initial_cluster(w0, uc_list)
        print(f"[LOCAL-QM-LSPG] Initial cluster k = {k} / {K - 1}")
    else:
        k = int(init_cluster)
        if not (0 <= k < K):
            raise ValueError(f"init_cluster={k} is out of range [0, {K - 1}]")
        print(f"[LOCAL-QM-LSPG] Initial cluster k = {k} / {K - 1} (forced)")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    H_k = np.asarray(H_list[k], dtype=np.float64)
    n_k = V_k.shape[1]

    q0 = V_k.T @ (w0 - u0_k)
    w_init = u_qm(q0, V_k, H_k, u0_k)

    snaps[:, 0] = w_init
    red_coords[:n_k, 0] = q0

    wp = w_init.copy()
    qp = q0.copy()
    cluster_history.append(k)

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"[LOCAL-QM-LSPG] Running local quadratic-manifold LSPG with {K} clusters, dt={dt}")

    for it in range(num_steps):
        print(f"[LOCAL-QM-LSPG] Timestep {it}/{num_steps}")

        k_new = select_cluster_reduced(k, qp, d_const, g_list)

        if k_new != k:
            print(f"  -> Cluster switch: {k} -> {k_new}")
            k = k_new

            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            H_k = np.asarray(H_list[k], dtype=np.float64)
            n_k = V_k.shape[1]

            qp = gauss_newton_quadratic_q(
                u_snap=wp,
                V=V_k,
                H=H_k,
                u_ref=u0_k,
                max_its=max_its_q0,
                tol_rel=tol_q0,
                verbose=False,
            )
        else:
            n_k = V_k.shape[1]

        cluster_history.append(k)

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        q, resnorms, times = gauss_newton_LSPG_qm(
            func_res=res,
            func_jac=jac,
            V=V_k,
            H=H_k,
            u_ref=u0_k,
            q0=qp,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w = u_qm(q, V_k, H_k, u0_k)

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


def inviscid_burgers_implicit2D_LSPG_local_qm_ecsw(
    grid_x,
    grid_y,
    weights,
    w0,
    dt,
    num_steps,
    mu,
    u0_list,
    V_list,
    H_list,
    uc_list,
    d_const,
    g_list,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    init_cluster=None,
    max_its=20,
    max_its_q0=20,
    tol_q0=1e-6,
):
    """
    Local quadratic-manifold ECSW-LSPG HROM for the 2D inviscid Burgers equation:

        w ≈ u0_k + V_k q + H_k Q(q)
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
    sample_weights_full = np.concatenate((sample_weights_cells, sample_weights_cells))

    idx_cells = augmented_sample
    idx_dofs = np.concatenate((idx_cells, N_cells + idx_cells))

    u0_loc_list = []
    V_loc_list = []
    H_loc_list = []

    for k in range(K):
        u0_k = np.asarray(u0_list[k], dtype=np.float64)
        V_k = np.asarray(V_list[k], dtype=np.float64)
        H_k = np.asarray(H_list[k], dtype=np.float64)

        u0_loc_list.append(u0_k[idx_dofs])
        V_loc_list.append(V_k[idx_dofs, :])
        H_loc_list.append(H_k[idx_dofs, :])

    n_max = max(np.asarray(V, dtype=np.float64).shape[1] for V in V_list)

    snaps = np.zeros((N_full, num_steps + 1), dtype=np.float64)
    red_coords = np.zeros((n_max, num_steps + 1), dtype=np.float64)
    cluster_history = []

    if init_cluster is None:
        k = _select_initial_cluster(w0, uc_list)
        print(f"[LOCAL-QM-LSPG-ECSW] Initial cluster k = {k} / {K - 1}")
    else:
        k = int(init_cluster)
        if not (0 <= k < K):
            raise ValueError(f"init_cluster={k} is out of range [0, {K - 1}]")
        print(f"[LOCAL-QM-LSPG-ECSW] Initial cluster k = {k} / {K - 1} (forced)")

    u0_k = np.asarray(u0_list[k], dtype=np.float64)
    V_k = np.asarray(V_list[k], dtype=np.float64)
    H_k = np.asarray(H_list[k], dtype=np.float64)

    u0_loc_k = u0_loc_list[k]
    V_loc_k = V_loc_list[k]
    H_loc_k = H_loc_list[k]
    n_k = V_k.shape[1]

    q0 = gauss_newton_quadratic_q(
        u_snap=w0,
        V=V_k,
        H=H_k,
        u_ref=u0_k,
        max_its=max_its_q0,
        tol_rel=tol_q0,
        verbose=False,
    )

    w0_full = u_qm(q0, V_k, H_k, u0_k)
    w0_loc = u_qm(q0, V_loc_k, H_loc_k, u0_loc_k)

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

    print(f"[LOCAL-QM-LSPG-ECSW] Running local quadratic-manifold ECSW-LSPG with {K} clusters, dt={dt}")

    for it in range(num_steps):
        print(f"[LOCAL-QM-LSPG-ECSW] Timestep {it}/{num_steps}")

        k_new = select_cluster_reduced(k, qp, d_const, g_list)

        if k_new != k:
            print(f"  -> Cluster switch: {k} -> {k_new}")
            k = k_new

            u0_k = np.asarray(u0_list[k], dtype=np.float64)
            V_k = np.asarray(V_list[k], dtype=np.float64)
            H_k = np.asarray(H_list[k], dtype=np.float64)

            u0_loc_k = u0_loc_list[k]
            V_loc_k = V_loc_list[k]
            H_loc_k = H_loc_list[k]
            n_k = V_k.shape[1]

            qp = gauss_newton_quadratic_q(
                u_snap=wp_full,
                V=V_k,
                H=H_k,
                u_ref=u0_k,
                max_its=max_its_q0,
                tol_rel=tol_q0,
                verbose=False,
            )
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

        q, resnorms, times = gauss_newton_LSPG_qm_ecsw(
            res_fun=res_loc,
            jac_fun=jac_loc,
            V_loc=V_loc_k,
            H_loc=H_loc_k,
            u_ref_loc=u0_loc_k,
            q0=qp,
            sample_weights=sample_weights_full,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
        )

        jac_t, res_t, ls_t = times
        num_its += len(resnorms)
        jac_time += jac_t
        res_time += res_t
        ls_time += ls_t

        w_full = u_qm(q, V_k, H_k, u0_k)
        w_loc = u_qm(q, V_loc_k, H_loc_k, u0_loc_k)

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
