# pod_gplvm_manifold.py
# -*- coding: utf-8 -*-

"""
POD-GPLVM manifolds for 2D Burgers PROM/HPROM.

Manifold form:

    w(z) = u_ref + U_q q(z)

where q(z) is the GP-LVM decoder mean from latent z to POD coordinates q.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

from .gplvm_inverse import (
    latent_box_bounds,
    nearest_seed_indices,
    solve_bounded_nls,
)
from .core import (
    get_ops,
    inviscid_burgers_res2D,
    inviscid_burgers_res2D_ecsw,
    inviscid_burgers_exact_jac2D,
    inviscid_burgers_exact_jac2D_ecsw,
)
from .ecsw_utils import generate_augmented_mesh
from .gauss_newton import gauss_newton_pod_rbf, gauss_newton_pod_rbf_ecsw


# ============================================================================
# Helpers
# ============================================================================


def _prepare_reference(u_ref, size):
    if u_ref is None:
        return np.zeros(size, dtype=np.float64)

    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if u_ref.size != size:
        raise ValueError(f"u_ref has size {u_ref.size}, expected {size}")
    return u_ref


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


def _model_uses_sparse_decoder(gplvm_model):
    if "decoder_mode" not in gplvm_model:
        return False
    mode = str(np.asarray(gplvm_model["decoder_mode"]).reshape(()))
    return mode.lower() == "sparse_dtc"


def _decode_q_gplvm(z, gplvm_model):
    """
    Decode latent coordinate z to POD coefficients q using GP-LVM predictive mean.

    Expected keys in gplvm_model:
      - Z_train: (n_train, n_latent)
      - alpha:   (n_train, q_dim_norm)
      - y_mean:  (q_dim,)
      - y_std:   (q_dim,)
      - log_ell, log_sf (scalars)
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    use_sparse = _model_uses_sparse_decoder(gplvm_model)
    if use_sparse:
        Z_centers = np.asarray(gplvm_model["Z_inducing"], dtype=np.float64)
        weights = np.asarray(gplvm_model["beta_inducing"], dtype=np.float64)
    else:
        Z_centers = np.asarray(gplvm_model["Z_train"], dtype=np.float64)
        weights = np.asarray(gplvm_model["alpha"], dtype=np.float64)
    y_mean = np.asarray(gplvm_model["y_mean"], dtype=np.float64).reshape(-1)
    y_std = np.asarray(gplvm_model["y_std"], dtype=np.float64).reshape(-1)

    if z.size != Z_centers.shape[1]:
        raise ValueError(
            f"latent size mismatch: z has {z.size}, expected {Z_centers.shape[1]}"
        )

    ell = float(np.exp(float(np.asarray(gplvm_model["log_ell"]).reshape(()))))
    sf2 = float(np.exp(2.0 * float(np.asarray(gplvm_model["log_sf"]).reshape(()))))

    diff = Z_centers - z[None, :]
    diff = np.clip(diff, -1e150, 1e150)
    d2 = np.sum(diff * diff, axis=1)
    log_k = np.clip(-0.5 * d2 / (ell * ell), -700.0, 50.0)
    k = sf2 * np.exp(log_k)

    q_norm = k @ weights
    q = y_mean + y_std * q_norm
    return q


def _jac_q_gplvm(z, gplvm_model):
    """
    Jacobian dq/dz for GP-LVM decoder mean.

    Returns
    -------
    dq_dz : ndarray, shape (q_dim, n_latent)
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    use_sparse = _model_uses_sparse_decoder(gplvm_model)
    if use_sparse:
        Z_centers = np.asarray(gplvm_model["Z_inducing"], dtype=np.float64)
        weights = np.asarray(gplvm_model["beta_inducing"], dtype=np.float64)
    else:
        Z_centers = np.asarray(gplvm_model["Z_train"], dtype=np.float64)
        weights = np.asarray(gplvm_model["alpha"], dtype=np.float64)
    y_std = np.asarray(gplvm_model["y_std"], dtype=np.float64).reshape(-1)

    ell = float(np.exp(float(np.asarray(gplvm_model["log_ell"]).reshape(()))))
    sf2 = float(np.exp(2.0 * float(np.asarray(gplvm_model["log_sf"]).reshape(()))))
    ell2 = ell * ell

    diff = Z_centers - z[None, :]
    diff = np.clip(diff, -1e150, 1e150)
    d2 = np.sum(diff * diff, axis=1)
    log_k = np.clip(-0.5 * d2 / ell2, -700.0, 50.0)
    k = sf2 * np.exp(log_k)

    # dk/dz for isotropic RBF kernel
    dk_dz = k[:, None] * (diff / ell2)

    dq_norm_dz = weights.T @ dk_dz
    dq_dz = y_std[:, None] * dq_norm_dz
    return dq_dz


def decode_gplvm(z, gplvm_model, basis_q, u_ref=None):
    """
    Decode latent z to full state:

        w(z) = u_ref + U_q q(z)
    """
    basis_q = np.asarray(basis_q, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, basis_q.shape[0])

    q = _decode_q_gplvm(z, gplvm_model)
    if q.size != basis_q.shape[1]:
        raise ValueError(
            f"q size mismatch: got {q.size}, expected basis_q cols={basis_q.shape[1]}"
        )

    return u_ref + basis_q @ q


def jac_gplvm(z, gplvm_model, basis_q):
    """
    Full tangent matrix:

        dw/dz = U_q (dq/dz)
    """
    basis_q = np.asarray(basis_q, dtype=np.float64)
    dq_dz = _jac_q_gplvm(z, gplvm_model)
    if dq_dz.shape[0] != basis_q.shape[1]:
        raise ValueError(
            "dq/dz and basis mismatch: "
            f"dq/dz rows={dq_dz.shape[0]}, basis_q cols={basis_q.shape[1]}"
        )

    return basis_q @ dq_dz


def _initial_latent_from_state(w_target, basis_q, u_ref, gplvm_model):
    """
    Build a robust initial latent guess by nearest neighbor in q-space.
    """
    w_target = np.asarray(w_target, dtype=np.float64).reshape(-1)
    basis_q = np.asarray(basis_q, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, basis_q.shape[0])

    q_target = basis_q.T @ (w_target - u_ref)

    q_train = np.asarray(gplvm_model["Q_train_raw"], dtype=np.float64)
    z_train = np.asarray(gplvm_model["Z_train"], dtype=np.float64)

    if q_train.shape[0] != z_train.shape[0]:
        raise ValueError(
            f"Q_train_raw/Z_train row mismatch: {q_train.shape[0]} vs {z_train.shape[0]}"
        )
    if q_train.shape[1] != q_target.size:
        raise ValueError(
            f"Q_train_raw columns {q_train.shape[1]} do not match q_target size {q_target.size}"
        )

    d2 = np.sum((q_train - q_target[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return z_train[idx].copy()


def _initial_latent_seeds_from_state(w_target, basis_q, u_ref, gplvm_model, n_starts=5):
    """
    Return multiple nearest-neighbor latent seeds from q-space.
    """
    w_target = np.asarray(w_target, dtype=np.float64).reshape(-1)
    basis_q = np.asarray(basis_q, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, basis_q.shape[0])

    q_target = basis_q.T @ (w_target - u_ref)
    q_train = np.asarray(gplvm_model["Q_train_raw"], dtype=np.float64)
    z_train = np.asarray(gplvm_model["Z_train"], dtype=np.float64)

    if q_train.shape[0] != z_train.shape[0]:
        raise ValueError(
            f"Q_train_raw/Z_train row mismatch: {q_train.shape[0]} vs {z_train.shape[0]}"
        )
    if q_train.shape[1] != q_target.size:
        raise ValueError(
            f"Q_train_raw columns {q_train.shape[1]} do not match q_target size {q_target.size}"
        )

    ids, _ = nearest_seed_indices(q_target, q_train, n_starts=n_starts)
    return np.asarray(z_train[ids], dtype=np.float64)


def _gauss_newton_decoder_inverse(
    z_init,
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

        min_z ||decode_func(z) - target_state||_2^2

    via Gauss-Newton on reconstruction error.
    """
    z = np.asarray(z_init, dtype=np.float64).copy()
    target_state = np.asarray(target_state, dtype=np.float64).reshape(-1)

    w_rec = decode_func(z)
    r0 = np.linalg.norm(w_rec - target_state)
    if r0 == 0.0:
        return z

    r = r0
    it_gn = 0
    while (r / r0 > tol_rel) and (it_gn < max_its):
        Jf = jac_func(z)
        res_rec = w_rec - target_state

        JTJ = Jf.T @ Jf
        JTr = Jf.T @ res_rec

        try:
            dz = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            dz, *_ = np.linalg.lstsq(JTJ, JTr, rcond=None)

        z -= dz
        w_rec = decode_func(z)
        r = np.linalg.norm(w_rec - target_state)
        it_gn += 1

    if verbose:
        print(f"[{tag}] it={it_gn}, rel={r/r0:.2e}")

    return z


def _bounded_decoder_inverse_multistart(
    z_seeds,
    target_state,
    decode_func,
    jac_func,
    lb,
    ub,
    prior_weight=1e-3,
    max_nfev=200,
    tol=1e-10,
    loss="linear",
    f_scale=1.0,
):
    """
    Bounded multi-start nonlinear least-squares inverse:

        min_z ||decode_func(z) - target_state||_2^2 + prior_weight ||z-z_seed||_2^2
    """
    z_seeds = np.asarray(z_seeds, dtype=np.float64)
    if z_seeds.ndim == 1:
        z_seeds = z_seeds[None, :]
    if z_seeds.ndim != 2 or z_seeds.shape[0] < 1:
        raise ValueError("z_seeds must be a 2D array with at least one seed.")

    target_state = np.asarray(target_state, dtype=np.float64).reshape(-1)

    best_z = None
    best_cost = np.inf
    best_success = False
    best_nfev = 0

    for z0 in z_seeds:
        def residual(z):
            return decode_func(z) - target_state

        def jac(z):
            return jac_func(z)

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
            loss=loss,
            f_scale=f_scale,
        )

        r = decode_func(z_sol) - target_state
        cost = 0.5 * float(np.dot(r, r))

        # Prefer successful solves; otherwise keep the lowest residual cost.
        success = bool(lsq_res.success)
        replace = False
        if best_z is None:
            replace = True
        elif success and (not best_success):
            replace = True
        elif success == best_success and cost < best_cost:
            replace = True

        if replace:
            best_z = z_sol
            best_cost = cost
            best_success = success
            best_nfev = int(lsq_res.nfev) if lsq_res.nfev is not None else 0

    return best_z, best_cost, best_success, best_nfev


# ============================================================================
# ECSW training matrix
# ============================================================================


def compute_ECSW_training_matrix_2D_gplvm(
    snaps,
    prev_snaps,
    basis_q,
    gplvm_model,
    res,
    jac,
    grid_x,
    grid_y,
    dt,
    mu,
    u_ref=None,
    max_local_its=10,
    local_tol=1e-2,
    inverse_method="bounded_trf",
    inverse_n_starts=3,
    inverse_bound_margin_rel=0.20,
    inverse_bound_margin_abs=0.25,
    inverse_prior_weight=1e-3,
    inverse_loss="linear",
    inverse_f_scale=1.0,
):
    """
    ECSW training matrix for global POD-GPLVM ROM.
    """
    snaps = np.asarray(snaps, dtype=np.float64)
    prev_snaps = np.asarray(prev_snaps, dtype=np.float64)
    basis_q = np.asarray(basis_q, dtype=np.float64)

    n_tot, n_snaps = snaps.shape
    n_hdm = n_tot // 2
    n_latent = int(np.asarray(gplvm_model["Z_train"]).shape[1])

    if prev_snaps.shape != snaps.shape:
        raise ValueError(
            f"snaps/prev_snaps shape mismatch: {snaps.shape} vs {prev_snaps.shape}"
        )
    if basis_q.shape[0] != n_tot:
        raise ValueError(
            f"basis_q rows mismatch: basis_q={basis_q.shape}, snapshots={snaps.shape}"
        )

    u_ref = _prepare_reference(u_ref, n_tot)
    C = np.zeros((n_latent * n_snaps, n_hdm), dtype=np.float64)
    z_lb, z_ub = latent_box_bounds(
        np.asarray(gplvm_model["Z_train"], dtype=np.float64),
        margin_rel=inverse_bound_margin_rel,
        margin_abs=inverse_bound_margin_abs,
    )

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        snap_prev = prev_snaps[:, isnap]

        def decode_loc(z):
            return decode_gplvm(z, gplvm_model, basis_q, u_ref=u_ref)

        def jac_loc(z):
            return jac_gplvm(z, gplvm_model, basis_q)

        mode = str(inverse_method).strip().lower()
        if mode == "gauss_newton":
            z0 = _initial_latent_from_state(snap, basis_q, u_ref, gplvm_model)
            z = _gauss_newton_decoder_inverse(
                z_init=z0,
                target_state=snap,
                decode_func=decode_loc,
                jac_func=jac_loc,
                max_its=max_local_its,
                tol_rel=local_tol,
                verbose=False,
                tag="GPLVM-ECSW-TRAIN",
            )
        else:
            z_seeds = _initial_latent_seeds_from_state(
                snap,
                basis_q,
                u_ref,
                gplvm_model,
                n_starts=inverse_n_starts,
            )
            z, _, _, _ = _bounded_decoder_inverse_multistart(
                z_seeds=z_seeds,
                target_state=snap,
                decode_func=decode_loc,
                jac_func=jac_loc,
                lb=z_lb,
                ub=z_ub,
                prior_weight=inverse_prior_weight,
                max_nfev=max(20, 10 * int(max_local_its)),
                tol=float(max(local_tol, 1e-10)),
                loss=inverse_loss,
                f_scale=inverse_f_scale,
            )

        w_rec = decode_loc(z)

        ires = res(w_rec, grid_x, grid_y, dt, snap_prev, mu, Dxec, Dyec)
        Ji = jac(w_rec, dt, JDxec, JDyec, Eye)
        V = jac_loc(z)
        Wi = Ji @ V

        row0 = isnap * n_latent
        row1 = row0 + n_latent

        for inode in range(n_hdm):
            C[row0:row1, inode] = (
                ires[inode] * Wi[inode, :]
                + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C


# ============================================================================
# PROM / HPROM
# ============================================================================


def inviscid_burgers_implicit2D_LSPG_pod_gplvm(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis_q,
    gplvm_model,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_ic=20,
    tol_ic=1e-12,
    ic_inverse_method="bounded_trf",
    ic_inverse_n_starts=5,
    ic_inverse_bound_margin_rel=0.20,
    ic_inverse_bound_margin_abs=0.25,
    ic_inverse_prior_weight=1e-3,
    ic_inverse_loss="linear",
    ic_inverse_f_scale=1.0,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Global POD-GPLVM PROM in latent coordinates z.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis_q = np.asarray(basis_q, dtype=np.float64)
    u_ref = _prepare_reference(u_ref, w0.size)

    if basis_q.ndim != 2 or basis_q.shape[0] != w0.size:
        raise ValueError(
            f"basis_q shape mismatch: basis_q={basis_q.shape}, w0={w0.shape}"
        )

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    def decode_func(z):
        return decode_gplvm(z, gplvm_model, basis_q, u_ref=u_ref)

    def jac_func(z):
        return jac_gplvm(z, gplvm_model, basis_q)

    mode_ic = str(ic_inverse_method).strip().lower()
    if mode_ic == "gauss_newton":
        z0_guess = _initial_latent_from_state(w0, basis_q, u_ref, gplvm_model)
        z0 = _gauss_newton_decoder_inverse(
            z_init=z0_guess,
            target_state=w0,
            decode_func=decode_func,
            jac_func=jac_func,
            max_its=max_its_ic,
            tol_rel=tol_ic,
            verbose=False,
            tag="POD-GPLVM-IC",
        )
    else:
        z_seeds = _initial_latent_seeds_from_state(
            w0,
            basis_q,
            u_ref,
            gplvm_model,
            n_starts=ic_inverse_n_starts,
        )
        z_lb, z_ub = latent_box_bounds(
            np.asarray(gplvm_model["Z_train"], dtype=np.float64),
            margin_rel=ic_inverse_bound_margin_rel,
            margin_abs=ic_inverse_bound_margin_abs,
        )
        z0, _, _, _ = _bounded_decoder_inverse_multistart(
            z_seeds=z_seeds,
            target_state=w0,
            decode_func=decode_func,
            jac_func=jac_func,
            lb=z_lb,
            ub=z_ub,
            prior_weight=ic_inverse_prior_weight,
            max_nfev=max(50, 10 * int(max_its_ic)),
            tol=float(max(tol_ic, 1e-12)),
            loss=ic_inverse_loss,
            f_scale=ic_inverse_f_scale,
        )

    w_init = decode_func(z0)

    n_dofs = w0.size
    n_latent = z0.size

    snaps = np.zeros((n_dofs, num_steps + 1), dtype=np.float64)
    latent = np.zeros((n_latent, num_steps + 1), dtype=np.float64)

    snaps[:, 0] = w_init
    latent[:, 0] = z0

    wp = w_init.copy()
    zp = z0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-GPLVM PROM with latent size {n_latent} for mu1={mu[0]}, mu2={mu[1]}")

    for it in range(num_steps):
        print(f" ... Working on timestep {it}")

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        z, resnorms, times = gauss_newton_pod_rbf(
            func=res,
            jac=jac,
            y0=zp,
            decode_rbf=decode_func,
            jac_rbf=jac_func,
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

        w = decode_func(z)

        snaps[:, it + 1] = w
        latent[:, it + 1] = z

        wp = w.copy()
        zp = z.copy()

    return snaps, latent, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_implicit2D_LSPG_pod_gplvm_ecsw(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    basis_q,
    gplvm_model,
    weights,
    u_ref=None,
    max_its=20,
    relnorm_cutoff=1e-5,
    min_delta=1e-2,
    max_its_ic=20,
    tol_ic=1e-12,
    ic_inverse_method="bounded_trf",
    ic_inverse_n_starts=5,
    ic_inverse_bound_margin_rel=0.20,
    ic_inverse_bound_margin_abs=0.25,
    ic_inverse_prior_weight=1e-3,
    ic_inverse_loss="linear",
    ic_inverse_f_scale=1.0,
    linear_solver="lstsq",
    normal_eq_reg=1e-12,
):
    """
    Global POD-GPLVM HPROM (ECSW-LSPG) in latent coordinates z.
    """
    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
    basis_q = np.asarray(basis_q, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if basis_q.ndim != 2 or basis_q.shape[0] != w0.size:
        raise ValueError(
            f"basis_q shape mismatch for ECSW: basis_q={basis_q.shape}, w0={w0.shape}"
        )

    u_ref = _prepare_reference(u_ref, w0.size)

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
    Eye_loc = sp.bmat([[Eye_u, None], [None, Eye_u]]).tocsr()

    JDxec_loc = JDxec[sample_inds, :][:, augmented_sample].tocsr()
    JDyec_loc = JDyec[sample_inds, :][:, augmented_sample].tocsr()

    sample_weights_cells = weights[sample_inds]

    idx_cells = augmented_sample
    idx_dofs = np.concatenate((idx_cells, n_cells + idx_cells))

    basis_loc = basis_q[idx_dofs, :]
    u_ref_loc = u_ref[idx_dofs]

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

    def decode_loc(z):
        return decode_gplvm(z, gplvm_model, basis_loc, u_ref=u_ref_loc)

    def jac_loc(z):
        return jac_gplvm(z, gplvm_model, basis_loc)

    mode_ic = str(ic_inverse_method).strip().lower()
    if mode_ic == "gauss_newton":
        z0_guess = _initial_latent_from_state(w0, basis_q, u_ref, gplvm_model)
        z0 = _gauss_newton_decoder_inverse(
            z_init=z0_guess,
            target_state=w0[idx_dofs],
            decode_func=decode_loc,
            jac_func=jac_loc,
            max_its=max_its_ic,
            tol_rel=tol_ic,
            verbose=False,
            tag="POD-GPLVM-ECSW-IC",
        )
    else:
        z_seeds = _initial_latent_seeds_from_state(
            w0,
            basis_q,
            u_ref,
            gplvm_model,
            n_starts=ic_inverse_n_starts,
        )
        z_lb, z_ub = latent_box_bounds(
            np.asarray(gplvm_model["Z_train"], dtype=np.float64),
            margin_rel=ic_inverse_bound_margin_rel,
            margin_abs=ic_inverse_bound_margin_abs,
        )
        z0, _, _, _ = _bounded_decoder_inverse_multistart(
            z_seeds=z_seeds,
            target_state=w0[idx_dofs],
            decode_func=decode_loc,
            jac_func=jac_loc,
            lb=z_lb,
            ub=z_ub,
            prior_weight=ic_inverse_prior_weight,
            max_nfev=max(50, 10 * int(max_its_ic)),
            tol=float(max(tol_ic, 1e-12)),
            loss=ic_inverse_loss,
            f_scale=ic_inverse_f_scale,
        )

    n_latent = z0.size
    latent = np.zeros((n_latent, num_steps + 1), dtype=np.float64)
    latent[:, 0] = z0

    w0_loc = decode_loc(z0)
    wp_loc = w0_loc.copy()
    zp = z0.copy()

    num_its = 0
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    print(f"Running POD-GPLVM ECSW ROM with latent size {n_latent} for mu1={mu[0]}, mu2={mu[1]}")

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

        def jac_loc_state(w_loc):
            return inviscid_burgers_exact_jac2D_ecsw(
                w_loc,
                dt,
                JDxec_loc,
                JDyec_loc,
                Eye_loc,
                sample_inds,
                augmented_sample,
            )

        z, resnorms, times = gauss_newton_pod_rbf_ecsw(
            func=res_loc,
            jac=jac_loc_state,
            y0=zp,
            decode_rbf=decode_loc,
            jac_rbf=jac_loc,
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

        w_loc = decode_loc(z)

        latent[:, it + 1] = z
        wp_loc = w_loc.copy()
        zp = z.copy()

    return latent, (num_its, jac_time, res_time, ls_time)
