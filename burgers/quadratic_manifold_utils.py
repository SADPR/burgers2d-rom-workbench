#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for the symmetric quadratic manifold:

    u(q) = u_ref + V q + H Q(q),

where Q(q) collects all symmetric products q_i q_j with i <= j.
"""

import numpy as np


_TRIU_CACHE = {}


def get_triu_indices(n):
    """Return (i_triu, j_triu) indices for the upper triangle including diagonal.

    The indices depend only on n and are needed on every Gauss-Newton
    iteration, so they are built once per n and cached. The cached arrays are
    read-only to make it explicit that callers must not modify them in place.
    """
    idx = _TRIU_CACHE.get(n)
    if idx is None:
        i_triu, j_triu = np.triu_indices(n)
        i_triu.flags.writeable = False
        j_triu.flags.writeable = False
        idx = (i_triu, j_triu)
        _TRIU_CACHE[n] = idx
    return idx


def build_Q_symmetric(q):
    """
    Symmetric quadratic monomials from q:

        q ∈ R^n  →  Q(q) ∈ R^{m},  m = n(n+1)/2

    with entries Q_k = q_i q_j for (i,j) in upper triangle (i<=j).
    """
    q = np.asarray(q, dtype=float)
    n = q.size
    i_triu, j_triu = get_triu_indices(n)
    Q = q[i_triu] * q[j_triu]
    return Q, (i_triu, j_triu)


def build_D_symmetric(q):
    """
    Build D(q) = ∂Q/∂q ∈ R^{m×n}, with m = n(n+1)/2.

    Each row corresponds to a pair (i,j) with i<=j, Q_ij = q_i q_j:

        ∂Q_ij / ∂q_l = δ_{il} q_j + δ_{jl} q_i.

    Note that when i == j, this gives 2 q_i.
    """
    q = np.asarray(q, dtype=float)
    n = q.size
    i_triu, j_triu = get_triu_indices(n)
    m = i_triu.size

    # Row k holds one entry at column i_triu[k] and one at j_triu[k]. The row
    # indices are distinct, so the two scatters below cannot collide with each
    # other; on the diagonal pairs (i == j) the "+=" accumulates them into the
    # required 2*q_i.
    D = np.zeros((m, n), dtype=float)
    rows = np.arange(m)
    D[rows, i_triu] = q[j_triu]
    D[rows, j_triu] += q[i_triu]
    return D


def u_qm(q, V, H, u_ref):
    """
    Quadratic manifold mapping:

        u(q) = u_ref + V q + H Q(q)

    Parameters
    ----------
    q      : (n,) reduced coordinates
    V      : (N, n) POD basis
    H      : (N, m) quadratic coefficients, m = n(n+1)/2
    u_ref  : (N,) reference state
    """
    Qq, _ = build_Q_symmetric(q)            # (m,)
    return u_ref + V @ q + H @ Qq           # (N,)


def J_qm(q, V, H):
    """
    Analytic Jacobian du/dq for the quadratic manifold:

        u(q) = u_ref + V q + H Q(q)

    ⇒ du/dq = V + H D(q),

    where D(q) = ∂Q/∂q ∈ R^{m×n}.
    """
    Dq = build_D_symmetric(q)               # (m, n)
    return V + H @ Dq                       # (N, n)
