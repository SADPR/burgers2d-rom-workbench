#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for the symmetric quadratic manifold:

    u(q) = u_ref + V q + H Q(q),

where Q(q) collects all symmetric products q_i q_j with i <= j.
"""

import numpy as np


def get_triu_indices(n):
    """Return (i_triu, j_triu) indices for the upper triangle including diagonal."""
    return np.triu_indices(n)


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
    m = len(i_triu)

    D = np.zeros((m, n), dtype=float)
    for k, (i, j) in enumerate(zip(i_triu, j_triu)):
        # ∂(q_i q_j)/∂q_i = q_j
        D[k, i] += q[j]
        # ∂(q_i q_j)/∂q_j = q_i  (for i == j this effectively becomes 2*q_i)
        D[k, j] += q[i]
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
