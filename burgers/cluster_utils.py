# cluster_utils.py
# -*- coding: utf-8 -*-

import numpy as np


def select_cluster_reduced(k_current, y_k, d_const, g_list):
    """
    Reduced-space cluster selector:
        l* = argmin_l (2 g_{k,l}^T y_k + d_const[k,l])
    """
    K = d_const.shape[0]
    y_k = np.asarray(y_k, dtype=np.float64).reshape(-1)

    scores = np.empty(K, dtype=np.float64)
    for l in range(K):
        g_kl = g_list[k_current][l]
        scores[l] = 2.0 * (g_kl @ y_k) + d_const[k_current, l]

    return int(np.argmin(scores))


def select_cluster_reduced_trunc(k_current, y_k, d_const, g_list, ncoords):
    """
    Reduced-space cluster selector using only the first ncoords coordinates.
    """
    K = d_const.shape[0]
    y_k = np.asarray(y_k, dtype=np.float64).reshape(-1)[:ncoords]

    scores = np.empty(K, dtype=np.float64)
    for l in range(K):
        g_kl = np.asarray(g_list[k_current][l], dtype=np.float64).reshape(-1)[:ncoords]
        scores[l] = 2.0 * (g_kl @ y_k) + d_const[k_current, l]

    return int(np.argmin(scores))


def select_initial_cluster_full(w, uc_list):
    """
    Full-space nearest-centroid cluster selector.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    d2 = [np.linalg.norm(w - np.asarray(uc, dtype=np.float64).reshape(-1)) ** 2 for uc in uc_list]
    return int(np.argmin(d2))