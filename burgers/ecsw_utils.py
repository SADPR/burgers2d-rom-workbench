# ecsw_utils.py
# -*- coding: utf-8 -*-

import numpy as np


def generate_augmented_mesh(grid_x, grid_y, sample_inds):
    """
    Augment sampled cell indices with the immediate stencil neighbors
    required by the 2D upwind-like Burgers discretization.

    Parameters
    ----------
    grid_x, grid_y : ndarray
        1D grid coordinate arrays defining the structured mesh.
    sample_inds : array_like
        Indices of sampled cells in flattened cell ordering.

    Returns
    -------
    augmented_sample : ndarray
        Sorted array of sampled cell indices plus the additional
        upstream neighbors needed by the residual/Jacobian stencil.
    """
    sample_inds = np.asarray(sample_inds, dtype=int).reshape(-1)
    augmented_sample = set(sample_inds.tolist())

    shp = (grid_y.size - 1, grid_x.size - 1)

    for idx in sample_inds:
        r, c = np.unravel_index(idx, shp)

        if r - 1 >= 0:
            augmented_sample.add(np.ravel_multi_index((r - 1, c), shp))

        if c - 1 >= 0:
            augmented_sample.add(np.ravel_multi_index((r, c - 1), shp))

    return np.sort(np.fromiter(augmented_sample, dtype=int))