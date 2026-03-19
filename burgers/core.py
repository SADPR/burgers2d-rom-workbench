# core.py
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

from .gauss_newton import newton_raphson

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=16)

def inviscid_burgers_explicit2D(
    grid_x,
    grid_y,
    u0,
    v0,
    dt,
    num_steps,
    mu,
    plot_every=None,
):
    """
    Explicit solver for the 2D inviscid Burgers equation.

    Parameters
    ----------
    grid_x, grid_y : ndarray
        Cell-edge grids in x and y.
    u0, v0 : ndarray
        Initial states for the two velocity components.
    dt : float
        Time step size.
    num_steps : int
        Number of time steps.
    mu : sequence
        Parameters [mu1, mu2].
    plot_every : int or None
        If not None, plot v every plot_every steps.

    Returns
    -------
    snaps : ndarray
        Full-state snapshots of shape (2*Ncells, num_steps+1).
    """

    u0 = np.asarray(u0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)

    snaps = np.zeros((u0.size + v0.size, num_steps + 1), dtype=np.float64)
    snaps[:, 0] = np.concatenate((u0.ravel(), v0.ravel()))

    up = u0.copy()
    vp = v0.copy()

    dx = grid_x[1:] - grid_x[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])

    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)

    left_bc_u = np.zeros_like(up)
    left_bc_u[:, 0] = 0.5 * mu[0] ** 2

    for istep in range(num_steps):
        flux_ux = (0.5 * up**2).T
        flux_vy = 0.5 * vp**2
        flux_uv = up * vp

        u = (
            up
            - dt * ((Dxec @ flux_ux).T - left_bc_u / dx)
            + dt * 0.02 * np.exp(mu[1] * xc[None, :])
            - dt * (Dyec @ flux_uv)
        )

        v = (
            vp
            - dt * (Dyec @ flux_vy)
            - dt * ((Dxec @ flux_uv.T).T)
        )

        if istep % 10 == 0:
            print(f"... Working on timestep {istep}")

        if plot_every is not None and istep % plot_every == 0:
            plt.imshow(v)
            plt.colorbar()
            plt.title(f"i = {istep}")
            plt.show()
            time.sleep(0.2)

        snaps[:, istep + 1] = np.concatenate((u.ravel(), v.ravel()))

        up = u
        vp = v

    return snaps


def inviscid_burgers_implicit2D(
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    mu,
    max_its=100,
    relnorm_cutoff=1e-12,
):
    """
    Implicit HDM solver for the 2D inviscid Burgers equation.

    Parameters
    ----------
    grid_x, grid_y : ndarray
        Cell-edge grids in x and y.
    w0 : ndarray
        Initial full state, stacked as [u; v].
    dt : float
        Time step size.
    num_steps : int
        Number of time steps.
    mu : sequence
        Parameters [mu1, mu2].
    max_its : int
        Maximum Newton iterations per time step.
    relnorm_cutoff : float
        Relative residual tolerance for Newton.

    Returns
    -------
    snaps : ndarray
        Full-order snapshots of shape (N, num_steps+1).
    """

    w0 = np.asarray(w0, dtype=np.float64).reshape(-1)

    print(f"Running HDM for mu1={mu[0]}, mu2={mu[1]}")

    snaps = np.zeros((w0.size, num_steps + 1), dtype=np.float64)
    snaps[:, 0] = w0.copy()

    wp = w0.copy()

    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)

    JDxec = sp.kron(sp.eye(grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1), Dyec).tocsr()

    idx = np.arange((grid_y.size - 1) * (grid_x.size - 1)).reshape(
        (grid_y.size - 1, grid_x.size - 1)
    ).T.ravel()

    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]

    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))

    for istep in range(num_steps):
        print(f" ... Working on timestep {istep}")

        def res(w):
            return inviscid_burgers_res2D_alt(
                w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec
            )

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        w, _ = newton_raphson(
            func=res,
            jac=jac,
            x0=wp,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
        )

        snaps[:, istep + 1] = w
        wp = w.copy()

    return snaps


def make_ddx(grid_x):
    """
    First-order backward difference operator on cell-centered values.
    """
    grid_x = np.asarray(grid_x, dtype=np.float64)
    dx = grid_x[1:] - grid_x[:-1]

    return sp.spdiags(
        [-np.ones(grid_x.size - 1) / dx, np.ones(grid_x.size - 1) / dx],
        [-1, 0],
        grid_x.size - 1,
        grid_x.size - 1,
        format="lil",
    )


def make_mid(grid_x):
    """
    Averaging operator between adjacent points.
    """
    grid_x = np.asarray(grid_x, dtype=np.float64)

    return sp.spdiags(
        [0.5 * np.ones(grid_x.size - 1), 0.5 * np.ones(grid_x.size - 1)],
        [-1, 0],
        grid_x.size - 1,
        grid_x.size - 1,
        format="lil",
    )


def make_2D_grid(x_low, x_up, y_low, y_up, num_cells_x, num_cells_y):
    """
    Uniform 2D Cartesian grid.
    """
    grid_x = np.linspace(x_low, x_up, num_cells_x + 1)
    grid_y = np.linspace(y_low, y_up, num_cells_y + 1)
    return grid_x, grid_y


def get_ops(grid_x, grid_y):
    """
    Build first-derivative operators and full-state identity for the 2D system.
    """
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)

    JDxec = sp.kron(sp.eye(grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1), Dyec).tocsr()

    idx = np.arange((grid_y.size - 1) * (grid_x.size - 1)).reshape(
        (grid_y.size - 1, grid_x.size - 1)
    ).T.ravel()

    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]

    Eye = sp.identity(2 * (grid_x.size - 1) * (grid_y.size - 1), format="csr")

    return Dxec, Dyec, JDxec, JDyec, Eye


def inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec):
    """
    Trapezoidal-rule residual for the 2D inviscid Burgers equation.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    wp = np.asarray(wp, dtype=np.float64).reshape(-1)

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])

    ncell = dx.size * dy.size

    u = w[:ncell].reshape(dy.size, dx.size)
    v = w[ncell:].reshape(dy.size, dx.size)

    up = wp[:ncell].reshape(dy.size, dx.size)
    vp = wp[ncell:].reshape(dy.size, dx.size)

    Fux = (0.5 * u**2).T
    Fpux = (0.5 * up**2).T

    Fvy = 0.5 * v**2
    Fpvy = 0.5 * vp**2

    Fuv = 0.5 * u * v
    Fpuv = 0.5 * up * vp

    src = dt * 0.02 * np.exp(mu[1] * xc[None, :])

    ru = (
        u
        - up
        + 0.5 * dt * (Dxec @ (Fux + Fpux)).T
        + 0.5 * dt * (Dyec @ (Fuv + Fpuv))
        - src
    )
    ru[:, 0] -= 0.5 * dt * mu[0] ** 2 / dx

    rv = (
        v
        - vp
        + 0.5 * dt * (Dyec @ (Fvy + Fpvy))
        + 0.5 * dt * (Dxec @ (Fuv.T + Fpuv.T)).T
    )

    return np.concatenate((ru.ravel(), rv.ravel()))


def inviscid_burgers_res2D_alt(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec):
    """
    Same residual as inviscid_burgers_res2D, but written in fully vectorized form.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    wp = np.asarray(wp, dtype=np.float64).reshape(-1)

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])

    u, v = np.split(w, 2)
    up, vp = np.split(wp, 2)

    Fux = 0.5 * u**2
    Fpux = 0.5 * up**2

    Fvy = 0.5 * v**2
    Fpvy = 0.5 * vp**2

    Fuv = 0.5 * u * v
    Fpuv = 0.5 * up * vp

    src = dt * 0.02 * np.exp(mu[1] * xc)
    src = np.tile(src, dy.size)

    lbc = np.zeros_like(u)
    lbc.reshape((dy.size, dx.size))[:, 0] = 0.5 * dt * mu[0] ** 2 / dx
    lbc = lbc.ravel()

    ru = (
        u
        - up
        + 0.5 * dt * (JDxec @ (Fux + Fpux))
        + 0.5 * dt * (JDyec @ (Fuv + Fpuv))
        - src
        - lbc
    )

    rv = (
        v
        - vp
        + 0.5 * dt * (JDyec @ (Fvy + Fpvy))
        + 0.5 * dt * (JDxec @ (Fuv + Fpuv))
    )

    return np.concatenate((ru, rv))


def inviscid_burgers_res2D_ecsw(
    w,
    grid_x,
    grid_y,
    dt,
    wp,
    mu,
    JDxec,
    JDyec,
    sample_inds,
    augmented_sample,
    lbc=None,
    src=None,
):
    """
    Residual on the ECSW sampled mesh.
    """
    try:
        import torch
        if torch.is_tensor(w):
            w = w.detach().squeeze().cpu().numpy()
        if torch.is_tensor(wp):
            wp = wp.detach().squeeze().cpu().numpy()
    except ImportError:
        pass

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    wp = np.asarray(wp, dtype=np.float64).reshape(-1)

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = 0.5 * (grid_x[1:] + grid_x[:-1])

    shp = (dy.size, dx.size)

    if lbc is None:
        lbc = np.zeros(sample_inds.size, dtype=np.float64)
        rows, cols = np.unravel_index(sample_inds, shp)
        for i, c in enumerate(cols):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]

    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    u, v = np.split(w, 2)
    up, vp = np.split(wp, 2)

    if u.size > augmented_sample.size:
        u_aug = u[augmented_sample]
        v_aug = v[augmented_sample]
        up_aug = up[augmented_sample]
        vp_aug = vp[augmented_sample]

        Fux = 0.5 * u_aug**2
        Fpux = 0.5 * up_aug**2
        Fvy = 0.5 * v_aug**2
        Fpvy = 0.5 * vp_aug**2
        Fuv = 0.5 * u_aug * v_aug
        Fpuv = 0.5 * up_aug * vp_aug

        u_s = u[sample_inds]
        v_s = v[sample_inds]
        up_s = up[sample_inds]
        vp_s = vp[sample_inds]
    else:
        Fux = 0.5 * u**2
        Fpux = 0.5 * up**2
        Fvy = 0.5 * v**2
        Fpvy = 0.5 * vp**2
        Fuv = 0.5 * u * v
        Fpuv = 0.5 * up * vp

        overlap = np.isin(augmented_sample, sample_inds)

        u_s = u[overlap]
        v_s = v[overlap]
        up_s = up[overlap]
        vp_s = vp[overlap]

    ru = (
        u_s
        - up_s
        + 0.5 * dt * (JDxec @ (Fux + Fpux))
        + 0.5 * dt * (JDyec @ (Fuv + Fpuv))
        - src
        - lbc
    )

    rv = (
        v_s
        - vp_s
        + 0.5 * dt * (JDyec @ (Fvy + Fpvy))
        + 0.5 * dt * (JDxec @ (Fuv + Fpuv))
    )

    return np.concatenate((ru, rv))


def inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye):
    """
    Exact Jacobian of the full residual.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    u, v = np.split(w, 2)

    ud = 0.5 * dt * sp.diags(u)
    vd = 0.5 * dt * sp.diags(v)

    ul = JDxec @ ud + 0.5 * JDyec @ vd
    ur = 0.5 * JDyec @ ud
    ll = 0.5 * JDxec @ vd
    lr = JDyec @ vd + 0.5 * JDxec @ ud

    return sp.bmat([[ul, ur], [ll, lr]]) + Eye


def inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_inds):
    """
    Fast exact Jacobian on the ECSW mesh.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    u, v = np.split(w, 2)

    if u.size > augmented_inds.size:
        ucol = u[augmented_inds]
        vcol = v[augmented_inds]
    else:
        ucol = u
        vcol = v

    if not sp.isspmatrix_csr(JDxec):
        JDxec = JDxec.tocsr()
    if not sp.isspmatrix_csr(JDyec):
        JDyec = JDyec.tocsr()
    if Eye is not None and sp.issparse(Eye) and not sp.isspmatrix_csr(Eye):
        Eye = Eye.tocsr()

    a = 0.5 * dt

    JDx_u = JDxec.multiply(ucol)
    JDy_v = JDyec.multiply(vcol)
    JDy_u = JDyec.multiply(ucol)
    JDx_v = JDxec.multiply(vcol)

    ul = a * JDx_u + 0.5 * a * JDy_v
    ur = 0.5 * a * JDy_u
    ll = 0.5 * a * JDx_v
    lr = a * JDy_v + 0.5 * a * JDx_u

    top = sp.hstack([ul, ur], format="csr")
    bot = sp.hstack([ll, lr], format="csr")
    J = sp.vstack([top, bot], format="csr")

    if Eye is not None:
        J = J + Eye

    return J


def POD(
    snaps,
    num_modes=None,
    method="svd",
    random_state=None,
    energy_capture=None,
    energy_loss=None,
    min_size=None,
    max_size=None,
    return_truncation_info=False,
    center=False,
    u_ref=None,
    return_reference=False,
):
    """
    Proper Orthogonal Decomposition by SVD or randomized SVD.

    Parameters
    ----------
    snaps : ndarray, shape (n_dofs, n_snapshots)
        Snapshot matrix.
    num_modes : int or None
        Explicit number of modes to retain.
    method : {"svd", "rsvd"}
        POD backend.
    random_state : int or None
        Random seed for randomized SVD.
    energy_capture : float or None
        Desired captured energy fraction in [0, 1], e.g. 0.9999.
    energy_loss : float or None
        Desired discarded energy tolerance in [0, 1], e.g. 1e-4.
        Equivalent to energy_capture = 1 - energy_loss.
    min_size : int or None
        Minimum number of modes after truncation.
    max_size : int or None
        Maximum number of modes after truncation.
    return_truncation_info : bool
        If True, also return a dictionary with truncation metadata.
    center : bool
        If True, center snapshots before POD. If `u_ref` is None,
        the snapshot mean is used as reference.
    u_ref : ndarray or None
        Optional user-provided affine reference. If provided,
        POD is computed on `snaps - u_ref[:, None]`.
    return_reference : bool
        If True, also return the reference vector used for centering
        (or None if no centering was applied).

    Returns
    -------
    basis : ndarray
        Truncated POD basis.
    svals : ndarray
        Full singular values returned by the SVD backend.
    info : dict, optional
        Truncation metadata if return_truncation_info=True.
    u_ref_vec : ndarray or None, optional
        Reference used for centering if return_reference=True.
    """
    snaps = np.asarray(snaps, dtype=np.float64)
    n_dofs = snaps.shape[0]

    u_ref_vec = None
    reference_source = "none"
    if center or u_ref is not None:
        if u_ref is None:
            u_ref_vec = np.mean(snaps, axis=1)
            reference_source = "mean"
        else:
            u_ref_vec = np.asarray(u_ref, dtype=np.float64).reshape(-1)
            if u_ref_vec.size != n_dofs:
                raise ValueError(f"u_ref has size {u_ref_vec.size}, expected {n_dofs}")
            reference_source = "provided"

    snaps_pod = snaps if u_ref_vec is None else (snaps - u_ref_vec[:, None])

    if energy_capture is not None and energy_loss is not None:
        raise ValueError("Specify only one of energy_capture or energy_loss, not both.")

    if energy_loss is not None:
        if not (0.0 <= energy_loss <= 1.0):
            raise ValueError("energy_loss must lie in [0, 1].")
        energy_capture = 1.0 - energy_loss

    if energy_capture is not None:
        if not (0.0 <= energy_capture <= 1.0):
            raise ValueError("energy_capture must lie in [0, 1].")

    if method == "svd":
        u, s, _ = np.linalg.svd(snaps_pod, full_matrices=False)

    elif method == "rsvd":
        # Critical point:
        # if truncation is energy-based, randomized SVD cannot know the
        # required rank unless you oversample or estimate first.
        # So for energy-based truncation, full SVD is the safer default.
        if num_modes is None and energy_capture is None:
            num_modes_eff = min(snaps_pod.shape)
        elif num_modes is not None:
            num_modes_eff = num_modes
        else:
            raise ValueError(
                "For method='rsvd', num_modes must be provided explicitly. "
                "Energy-based truncation is not reliable unless a rank is prescribed first."
            )

        u, s, _ = randomized_svd(
            snaps_pod,
            n_components=num_modes_eff,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown POD method '{method}'. Use 'svd' or 'rsvd'.")

    # ------------------------------------------------------------
    # Decide truncation size
    # ------------------------------------------------------------
    if num_modes is not None and energy_capture is not None:
        raise ValueError("Specify either num_modes or an energy criterion, not both.")

    if num_modes is not None:
        n_keep = int(num_modes)
    elif energy_capture is not None:
        n_keep = podsize(
            s,
            energy_thresh=energy_capture,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        n_keep = s.size
        if min_size is not None:
            n_keep = max(n_keep, min_size)
        if max_size is not None:
            n_keep = min(n_keep, max_size)

    if n_keep < 1:
        raise ValueError("Truncation produced fewer than 1 mode.")

    if n_keep > u.shape[1]:
        raise ValueError(f"Requested {n_keep} modes, but only {u.shape[1]} are available.")

    basis = u[:, :n_keep]

    if return_truncation_info:
        energy = np.cumsum(s**2) / np.sum(s**2)
        info = {
            "n_keep": n_keep,
            "energy_captured": energy[n_keep - 1],
            "energy_lost": 1.0 - energy[n_keep - 1],
            "method": method,
            "n_available": s.size,
            "centered": u_ref_vec is not None,
            "reference_source": reference_source,
        }
        if return_reference:
            return basis, s, info, u_ref_vec
        return basis, s, info

    if return_reference:
        return basis, s, u_ref_vec

    return basis, s


def podsize(svals, energy_thresh=None, min_size=None, max_size=None):
    """
    Number of POD modes satisfying the requested truncation criteria.

    Parameters
    ----------
    svals : ndarray
        Singular values.
    energy_thresh : float or None
        Target captured energy fraction in [0, 1].
    min_size : int or None
        Minimum basis size.
    max_size : int or None
        Maximum basis size.

    Returns
    -------
    numvecs : int
        Number of retained modes.
    """
    svals = np.asarray(svals, dtype=np.float64)

    if energy_thresh is None and min_size is None and max_size is None:
        raise RuntimeError("Must specify at least one truncation criterion in podsize().")

    if energy_thresh is not None:
        if not (0.0 <= energy_thresh <= 1.0):
            raise ValueError("energy_thresh must lie in [0, 1].")
        cumulative_energy = np.cumsum(svals**2)
        cumulative_energy /= np.sum(svals**2)
        numvecs = np.searchsorted(cumulative_energy, energy_thresh) + 1
    else:
        numvecs = 1 if min_size is None else int(min_size)

    if min_size is not None:
        numvecs = max(numvecs, int(min_size))

    if max_size is not None:
        numvecs = min(numvecs, int(max_size))

    return numvecs


def singular_value_decay_data(svals, max_modes=None):
    """
    Build the singular-value residual energy curve used in decay plots.

    Returns y_j = 1 - (sum_{i=1}^j sigma_i^2) / (sum_i sigma_i^2)

    Parameters
    ----------
    svals : ndarray
        Singular values.
    max_modes : int or None
        Number of leading indices to include in the returned curve.

    Returns
    -------
    inds : ndarray
        Mode indices starting at 1.
    residual_energy : ndarray
        Residual energy fraction after retaining the first j modes.
    """
    svals = np.asarray(svals, dtype=np.float64)

    total_energy = np.sum(svals**2)
    captured_energy = np.cumsum(svals**2) / total_energy
    residual_energy = 1.0 - captured_energy

    if max_modes is not None:
        max_modes = min(int(max_modes), svals.size)
        residual_energy = residual_energy[:max_modes]
        inds = np.arange(1, max_modes + 1)
    else:
        inds = np.arange(1, svals.size + 1)

    return inds, residual_energy


def plot_singular_value_decay(
    svals,
    out_path,
    max_modes=1000,
    label=None,
    title=None,
    use_latex=True,
):
    """
    Plot residual singular-value energy decay on a semilog-y scale.

    The plotted quantity is

        1 - sum_{i=1}^j sigma_i^2 / sum_i sigma_i^2

    which is the unresolved energy fraction after j modes.
    """
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.35,
        })

    inds, residual_energy = singular_value_decay_data(svals, max_modes=max_modes)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.semilogy(inds, residual_energy, label=label)
    ax.set_xlabel(r"Singular value index $j$")
    ax.set_ylabel(r"$1 - \sum_{i=1}^{j}\sigma_i^2 / \sum_i \sigma_i^2$")
    ax.grid(True)

    if title is not None:
        ax.set_title(title)

    if label is not None:
        ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def compute_error(rom_snaps, hdm_snaps):
    """
    Relative error at each time step, measured against the HDM snapshots.
    """
    rom_snaps = np.asarray(rom_snaps, dtype=np.float64)
    hdm_snaps = np.asarray(hdm_snaps, dtype=np.float64)

    hdm_norm = np.sqrt(np.sum(hdm_snaps**2, axis=0))
    err_norm = np.sqrt(np.sum((rom_snaps - hdm_snaps)**2, axis=0))

    hdm_norm = np.where(hdm_norm > 0.0, hdm_norm, 1.0)

    rel_err = err_norm / hdm_norm
    return rel_err, rel_err.mean()


def get_snapshot_params(
    mu1_range=(4.25, 5.5),
    mu2_range=(0.015, 0.03),
    samples_per_mu=3,
):
    """
    Cartesian training grid for snapshot generation.

    Parameters
    ----------
    mu1_range, mu2_range : tuple(float, float)
        Inclusive ranges for the two parameters.
    samples_per_mu : int
        Number of evenly spaced samples per parameter.

    Returns
    -------
    mu_samples : list[list[float]]
        Parameter list [[mu1, mu2], ...].
    """
    if int(samples_per_mu) < 1:
        raise ValueError("samples_per_mu must be >= 1.")

    mu1_low, mu1_high = float(mu1_range[0]), float(mu1_range[1])
    mu2_low, mu2_high = float(mu2_range[0]), float(mu2_range[1])

    if mu1_low > mu1_high:
        raise ValueError(f"mu1_range must be increasing, got {mu1_range}.")
    if mu2_low > mu2_high:
        raise ValueError(f"mu2_range must be increasing, got {mu2_range}.")

    mu1_samples = np.linspace(mu1_low, mu1_high, int(samples_per_mu))
    mu2_samples = np.linspace(mu2_low, mu2_high, int(samples_per_mu))

    mu_samples = []
    for mu1 in mu1_samples:
        for mu2 in mu2_samples:
            mu_samples.append([float(mu1), float(mu2)])
    return mu_samples


def param_to_snap_fn(mu, snap_folder="param_snaps", suffix=".npy"):
    """
    Snapshot filename associated with parameter vector mu.
    """
    mu = np.asarray(mu).ravel()

    parts = []
    for i, val in enumerate(mu):
        parts.append(f"mu{i+1}_{val}")

    filename = "+".join(parts) + suffix
    return os.path.join(snap_folder, filename)


def get_saved_params(snap_folder="param_snaps"):
    """
    Set of saved snapshot file paths in snap_folder.
    """
    return set(glob.glob(os.path.join(snap_folder, "*")))


def load_or_compute_snaps(
    mu,
    grid_x,
    grid_y,
    w0,
    dt,
    num_steps,
    snap_folder="param_snaps",
):
    """
    Load snapshots for mu if available; otherwise compute and save them.
    """
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)

    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)

    if os.path.exists(snap_fn):
        print(f"Loading saved snaps for mu1={mu[0]}, mu2={mu[1]}")
        snaps = np.load(snap_fn)[:, :num_steps + 1]
    else:
        print(f"Computing new snaps for mu1={mu[0]}, mu2={mu[1]}")
        t0 = time.time()
        snaps = inviscid_burgers_implicit2D(grid_x, grid_y, w0, dt, num_steps, mu)
        print("Elapsed time: {:3.3e}".format(time.time() - t0))
        np.save(snap_fn, snaps)

    return snaps


def plot_snaps(
    grid_x,
    grid_y,
    snaps,
    snaps_to_plot,
    linewidth=2,
    color="black",
    linestyle="solid",
    label=None,
    fig_ax=None,
):
    """
    Plot midpoint slices of the u-component snapshots.
    """
    snaps = np.asarray(snaps, dtype=np.float64)

    if fig_ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        fig, ax1, ax2 = fig_ax

    x = 0.5 * (grid_x[1:] + grid_x[:-1])
    y = 0.5 * (grid_y[1:] + grid_y[:-1])

    nx = x.size
    ny = y.size

    mid_x = nx // 2
    mid_y = ny // 2

    first_line = True

    for ind in snaps_to_plot:
        line_label = label if first_line else None
        first_line = False

        snap_u = snaps[:nx * ny, ind].reshape(ny, nx)

        ax1.plot(
            x,
            snap_u[mid_y, :],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=line_label,
        )
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$u_x(x,y={:0.1f})$".format(y[mid_y]))
        ax1.grid()

        ax2.plot(
            y,
            snap_u[:, mid_x],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=line_label,
        )
        ax2.set_xlabel(r"$y$")
        ax2.set_ylabel(r"$u_x(x={:0.1f},y)$".format(x[mid_x]))
        ax2.grid()

    return fig, ax1, ax2
