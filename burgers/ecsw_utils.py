# ecsw_utils.py
# -*- coding: utf-8 -*-

import os

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


def _stratified_random_time_indices(candidates, n_pick, rng):
    """
    Pick `n_pick` indices from `candidates` with random-but-spread coverage.
    """
    candidates = np.asarray(candidates, dtype=int).reshape(-1)
    if n_pick <= 0 or candidates.size == 0:
        return np.zeros((0,), dtype=int)
    if n_pick >= candidates.size:
        return np.sort(candidates.copy())

    picks = np.zeros((n_pick,), dtype=int)
    edges = np.linspace(0, candidates.size, n_pick + 1, dtype=int)
    for i in range(n_pick):
        i0 = int(edges[i])
        i1 = int(edges[i + 1])
        if i1 <= i0:
            j = i0
        else:
            j = int(rng.integers(i0, i1))
        picks[i] = candidates[j]
    return np.sort(np.unique(picks))


def _normalize_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected 2D points array, got shape {points.shape}.")
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    span = np.where((pmax - pmin) > 0.0, (pmax - pmin), 1.0)
    return (points - pmin) / span


def _prepare_mu_points(mu_points, num_mu):
    pts = np.asarray(mu_points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    if pts.ndim != 2:
        raise ValueError(f"mu_points must be 2D or 1D, got shape {pts.shape}.")
    if int(pts.shape[0]) != int(num_mu):
        raise ValueError(
            f"mu_points row mismatch: got {pts.shape[0]}, expected num_mu={num_mu}."
        )
    return _normalize_points(pts)


def _farthest_point_order(points, rng):
    points = np.asarray(points, dtype=np.float64)
    n_pts = int(points.shape[0])
    if n_pts <= 1:
        return np.arange(n_pts, dtype=int)

    centroid = np.mean(points, axis=0, keepdims=True)
    d0 = np.linalg.norm(points - centroid, axis=1)
    max0 = float(np.max(d0))
    first_candidates = np.flatnonzero(np.isclose(d0, max0))
    first = int(rng.choice(first_candidates))

    order = [first]
    min_dist = np.linalg.norm(points - points[first], axis=1)
    min_dist[first] = -np.inf

    while len(order) < n_pts:
        best = float(np.max(min_dist))
        cand = np.flatnonzero(np.isclose(min_dist, best))
        nxt = int(rng.choice(cand))
        order.append(nxt)
        dnew = np.linalg.norm(points - points[nxt], axis=1)
        min_dist = np.minimum(min_dist, dnew)
        min_dist[np.asarray(order, dtype=int)] = -np.inf

    return np.asarray(order, dtype=int)


def _nearest_neighbor_weights(points):
    points = np.asarray(points, dtype=np.float64)
    n_pts = int(points.shape[0])
    if n_pts <= 1:
        return np.ones((n_pts,), dtype=np.float64)

    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    nn = np.where(np.isfinite(nn), nn, 0.0)
    if float(np.max(nn)) <= 0.0:
        return np.ones((n_pts,), dtype=np.float64)
    return nn


def _allocate_counts_with_capacity(weights, total_count, capacity):
    """
    Allocate `total_count` integer picks using `weights`, capped by `capacity`.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    capacity = np.asarray(capacity, dtype=int).reshape(-1)
    if weights.size != capacity.size:
        raise ValueError("weights and capacity must have the same size.")

    counts = np.zeros_like(capacity, dtype=int)
    remaining = int(min(int(total_count), int(np.sum(capacity))))
    if remaining <= 0:
        return counts

    while remaining > 0:
        room = capacity - counts
        active = room > 0
        if not np.any(active):
            break

        w = np.where(active, np.maximum(weights, 0.0), 0.0)
        if float(np.sum(w)) <= 0.0:
            w = np.where(active, 1.0, 0.0)

        ideal = remaining * w / np.sum(w)
        base = np.floor(ideal).astype(int)
        base = np.minimum(base, room)

        used = int(np.sum(base))
        if used > 0:
            counts += base
            remaining -= used
            if remaining <= 0:
                break

        room = capacity - counts
        frac = np.where(room > 0, ideal - np.floor(ideal), -np.inf)
        order = np.argsort(-frac)
        progressed = False
        for idx in order:
            if remaining <= 0:
                break
            if room[idx] <= 0:
                continue
            counts[idx] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break

        if not progressed and remaining > 0:
            active_idx = np.flatnonzero(capacity - counts > 0)
            if active_idx.size == 0:
                break
            for idx in active_idx:
                counts[idx] += 1
                remaining -= 1
                if remaining <= 0:
                    break

    return counts


def _compute_parameter_aware_counts(
    *,
    n_select,
    num_mu,
    n_candidates_per_mu,
    mu_points,
    ensure_mu_coverage,
    random_seed,
):
    counts = np.zeros((int(num_mu),), dtype=int)
    n_select = int(n_select)
    n_candidates_per_mu = int(n_candidates_per_mu)
    if n_select <= 0:
        return counts

    rng = np.random.default_rng(int(random_seed))
    points = _prepare_mu_points(mu_points, num_mu)
    order = _farthest_point_order(points, rng)

    if bool(ensure_mu_coverage) and n_select >= int(num_mu):
        counts += 1
        remaining = n_select - int(num_mu)
    else:
        remaining = n_select

    if remaining <= 0:
        return counts

    if (not bool(ensure_mu_coverage)) and n_select <= int(num_mu):
        take = int(min(remaining, order.size))
        counts[order[:take]] += 1
        return counts

    weights = _nearest_neighbor_weights(points)
    capacity = np.maximum(0, n_candidates_per_mu - counts)
    add_counts = _allocate_counts_with_capacity(weights, remaining, capacity)
    counts += add_counts
    remaining = n_select - int(np.sum(counts))

    if remaining > 0:
        progressed = True
        while remaining > 0 and progressed:
            progressed = False
            for idx in order:
                if counts[idx] >= n_candidates_per_mu:
                    continue
                counts[idx] += 1
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break

    return counts


def _resolve_total_snapshot_count(
    *,
    n_candidates_total,
    total_snapshots,
    total_snapshots_percent,
    snap_time_offset,
    num_steps,
    snap_sample_factor,
    num_mu,
):
    """
    Resolve the global number of ECSW snapshots to select.

    Priority:
      1) `total_snapshots_percent` (if provided)
      2) `total_snapshots`
      3) legacy count inferred from stride
    """
    if total_snapshots_percent is not None:
        pct = float(total_snapshots_percent)
        if not np.isfinite(pct) or pct <= 0.0:
            raise ValueError("total_snapshots_percent must be a finite value > 0.")
        n_select = int(np.ceil((pct / 100.0) * float(n_candidates_total)))
        n_select = max(1, n_select)
        return min(n_select, int(n_candidates_total))

    if total_snapshots is None:
        legacy_per_mu = int(
            np.arange(snap_time_offset, num_steps, snap_sample_factor, dtype=int).size
        )
        total_snapshots = max(1, legacy_per_mu * num_mu)

    total_snapshots = int(total_snapshots)
    if total_snapshots < 1:
        raise ValueError("total_snapshots must be >= 1 for global_stratified_random mode.")
    return min(total_snapshots, int(n_candidates_total))


def _cluster_assignment_for_candidates(
    *,
    mu_snaps,
    candidate_now_cols,
    cluster_centers,
):
    snaps = np.asarray(mu_snaps, dtype=np.float64)
    if snaps.ndim != 2:
        raise ValueError(f"mu_snaps must be 2D, got shape {snaps.shape}.")

    candidate_now_cols = np.asarray(candidate_now_cols, dtype=int).reshape(-1)
    if candidate_now_cols.size == 0:
        return candidate_now_cols, np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    centers = [np.asarray(c, dtype=np.float64).reshape(-1) for c in cluster_centers]
    num_clusters = int(len(centers))
    if num_clusters < 1:
        raise ValueError("cluster_centers must contain at least one cluster center.")

    state_size = int(snaps.shape[0])
    for ic, center in enumerate(centers):
        if center.size != state_size:
            raise ValueError(
                f"cluster_centers[{ic}] has size {center.size}, expected {state_size}."
            )

    candidate_states = snaps[:, candidate_now_cols]
    centers_mat = np.column_stack(centers)
    state_norm_sq = np.einsum("ij,ij->j", candidate_states, candidate_states)
    center_norm_sq = np.einsum("ij,ij->j", centers_mat, centers_mat)
    cross = centers_mat.T @ candidate_states
    dist_sq = center_norm_sq[:, None] - 2.0 * cross + state_norm_sq[None, :]
    cluster_ids = np.argmin(dist_sq, axis=0).astype(int)
    candidate_counts = np.bincount(cluster_ids, minlength=num_clusters).astype(int)
    return candidate_now_cols, cluster_ids, candidate_counts


def select_local_cluster_percent_snapshot_cols(
    *,
    mu_snaps,
    candidate_now_cols,
    cluster_centers,
    cluster_sample_percent=10.0,
    random_seed=42,
    min_per_cluster=1,
):
    """
    Select ECSW training snapshots with cluster-balanced percentage sampling.

    Parameters
    ----------
    mu_snaps : ndarray, shape (N, T)
        Full-order snapshots for one training parameter.
    candidate_now_cols : array_like
        Candidate "current-time" columns (typically arange(offset, num_steps)).
    cluster_centers : sequence of array_like
        Full-state cluster reference vectors used to assign each candidate
        snapshot to its closest cluster in L2.
    cluster_sample_percent : float
        Percentage of candidates selected inside each cluster.
    random_seed : int
        RNG seed used inside per-cluster stratified random selection.
    min_per_cluster : int
        Minimum selected snapshots per non-empty cluster.

    Returns
    -------
    dict with keys:
      - selected_now_cols
      - candidate_counts_by_cluster
      - selected_counts_by_cluster
      - cluster_ids_by_candidate
    """
    candidate_now_cols, cluster_ids, candidate_counts = _cluster_assignment_for_candidates(
        mu_snaps=mu_snaps,
        candidate_now_cols=candidate_now_cols,
        cluster_centers=cluster_centers,
    )
    if candidate_now_cols.size == 0:
        return {
            "selected_now_cols": np.zeros((0,), dtype=int),
            "candidate_counts_by_cluster": np.zeros((0,), dtype=int),
            "selected_counts_by_cluster": np.zeros((0,), dtype=int),
            "cluster_ids_by_candidate": np.zeros((0,), dtype=int),
        }
    num_clusters = int(candidate_counts.size)

    pct = float(cluster_sample_percent)
    if not np.isfinite(pct) or pct <= 0.0:
        raise ValueError("cluster_sample_percent must be a finite value > 0.")

    min_per_cluster = int(min_per_cluster)
    if min_per_cluster < 0:
        raise ValueError("min_per_cluster must be >= 0.")

    rng = np.random.default_rng(int(random_seed))
    selected_counts = np.zeros((num_clusters,), dtype=int)
    selected_parts = []

    for k in range(num_clusters):
        cols_k = candidate_now_cols[cluster_ids == k]
        n_k = int(cols_k.size)
        if n_k == 0:
            continue

        n_pick = int(np.ceil((pct / 100.0) * float(n_k)))
        n_pick = max(min_per_cluster, n_pick)
        n_pick = min(n_pick, n_k)

        chosen_k = _stratified_random_time_indices(cols_k, n_pick, rng)
        selected_counts[k] = int(chosen_k.size)
        selected_parts.append(chosen_k)

    if selected_parts:
        selected_now_cols = np.sort(np.concatenate(selected_parts).astype(int, copy=False))
    else:
        selected_now_cols = np.zeros((0,), dtype=int)

    return {
        "selected_now_cols": selected_now_cols,
        "candidate_counts_by_cluster": candidate_counts,
        "selected_counts_by_cluster": selected_counts,
        "cluster_ids_by_candidate": cluster_ids,
    }


def select_local_cluster_count_snapshot_cols(
    *,
    mu_snaps,
    candidate_now_cols,
    cluster_centers,
    target_count,
    random_seed=42,
    min_per_cluster=1,
):
    """
    Select exactly `target_count` snapshots with cluster-balanced allocation.
    """
    candidate_now_cols, cluster_ids, candidate_counts = _cluster_assignment_for_candidates(
        mu_snaps=mu_snaps,
        candidate_now_cols=candidate_now_cols,
        cluster_centers=cluster_centers,
    )
    if candidate_now_cols.size == 0:
        return {
            "selected_now_cols": np.zeros((0,), dtype=int),
            "candidate_counts_by_cluster": np.zeros((0,), dtype=int),
            "selected_counts_by_cluster": np.zeros((0,), dtype=int),
            "cluster_ids_by_candidate": np.zeros((0,), dtype=int),
            "target_count": 0,
        }

    num_clusters = int(candidate_counts.size)
    target_count = int(target_count)
    if target_count < 0:
        raise ValueError("target_count must be >= 0.")
    target_count = min(target_count, int(candidate_now_cols.size))

    min_per_cluster = int(min_per_cluster)
    if min_per_cluster < 0:
        raise ValueError("min_per_cluster must be >= 0.")

    selected_counts = np.zeros((num_clusters,), dtype=int)
    nonempty = np.flatnonzero(candidate_counts > 0)
    n_nonempty = int(nonempty.size)

    if target_count > 0 and n_nonempty > 0 and min_per_cluster > 0:
        full_min_needed = int(min_per_cluster * n_nonempty)
        if target_count >= full_min_needed:
            selected_counts[nonempty] = min_per_cluster
        elif target_count >= n_nonempty:
            selected_counts[nonempty] = 1
        else:
            # Not enough budget for all non-empty clusters: favor larger clusters.
            order = nonempty[np.argsort(-candidate_counts[nonempty])]
            selected_counts[order[:target_count]] = 1

    selected_counts = np.minimum(selected_counts, candidate_counts)
    remaining = target_count - int(np.sum(selected_counts))

    if remaining > 0:
        capacity = candidate_counts - selected_counts
        weights = candidate_counts.astype(np.float64)
        add_counts = _allocate_counts_with_capacity(weights, remaining, capacity)
        selected_counts += add_counts

    rng = np.random.default_rng(int(random_seed))
    selected_parts = []
    for k in range(num_clusters):
        n_pick = int(selected_counts[k])
        if n_pick <= 0:
            continue
        cols_k = candidate_now_cols[cluster_ids == k]
        chosen_k = _stratified_random_time_indices(cols_k, n_pick, rng)
        selected_parts.append(chosen_k)

    if selected_parts:
        selected_now_cols = np.sort(np.concatenate(selected_parts).astype(int, copy=False))
    else:
        selected_now_cols = np.zeros((0,), dtype=int)

    return {
        "selected_now_cols": selected_now_cols,
        "candidate_counts_by_cluster": candidate_counts,
        "selected_counts_by_cluster": selected_counts,
        "cluster_ids_by_candidate": cluster_ids,
        "target_count": int(target_count),
    }


def build_ecsw_snapshot_plan(
    *,
    num_steps,
    snap_time_offset,
    num_mu,
    snap_sample_factor=10,
    mode="global_stratified_random",
    total_snapshots=50,
    total_snapshots_percent=None,
    mu_points=None,
    random_seed=42,
    ensure_mu_coverage=True,
):
    """
    Build snapshot-column selection plan for ECSW training.

    Returns
    -------
    dict with keys:
      - mode
      - candidate_now_cols
      - selected_now_cols_by_mu (list of ndarray)
      - num_candidates_per_mu
      - num_candidates_total
      - num_selected_per_mu
      - num_selected_total
    """
    num_steps = int(num_steps)
    snap_time_offset = int(snap_time_offset)
    num_mu = int(num_mu)
    snap_sample_factor = int(snap_sample_factor)

    if num_mu < 1:
        raise ValueError("num_mu must be >= 1.")
    if snap_time_offset < 1:
        raise ValueError("snap_time_offset must be >= 1.")
    if snap_sample_factor < 1:
        raise ValueError("snap_sample_factor must be >= 1.")

    mode = str(mode).strip().lower()
    if mode not in ("strided_per_mu", "global_stratified_random", "global_param_time_stratified"):
        raise ValueError(
            "Unsupported ECSW snapshot mode. Use 'strided_per_mu' or "
            "'global_stratified_random' or 'global_param_time_stratified'."
        )

    candidate_now_cols = np.arange(snap_time_offset, num_steps, dtype=int)
    n_candidates_per_mu = int(candidate_now_cols.size)
    n_candidates_total = int(num_mu * n_candidates_per_mu)

    if n_candidates_per_mu == 0:
        raise ValueError(
            "No valid ECSW snapshot pairs with current (num_steps, snap_time_offset)."
        )

    if mode == "strided_per_mu":
        selected = np.arange(snap_time_offset, num_steps, snap_sample_factor, dtype=int)
        selected_by_mu = [selected.copy() for _ in range(num_mu)]
    else:
        n_select = _resolve_total_snapshot_count(
            n_candidates_total=n_candidates_total,
            total_snapshots=total_snapshots,
            total_snapshots_percent=total_snapshots_percent,
            snap_time_offset=snap_time_offset,
            num_steps=num_steps,
            snap_sample_factor=snap_sample_factor,
            num_mu=num_mu,
        )
        rng = np.random.default_rng(int(random_seed))

        if mode == "global_stratified_random":
            counts = np.zeros((num_mu,), dtype=int)
            remaining = int(n_select)

            if bool(ensure_mu_coverage) and n_select >= num_mu:
                counts += 1
                remaining -= num_mu

            if remaining > 0:
                order = rng.permutation(num_mu)
                for i in range(remaining):
                    counts[order[i % num_mu]] += 1
        else:
            if mu_points is None:
                raise ValueError(
                    "mu_points must be provided for mode='global_param_time_stratified'."
                )
            counts = _compute_parameter_aware_counts(
                n_select=n_select,
                num_mu=num_mu,
                n_candidates_per_mu=n_candidates_per_mu,
                mu_points=mu_points,
                ensure_mu_coverage=ensure_mu_coverage,
                random_seed=random_seed,
            )

        selected_by_mu = []
        for imu in range(num_mu):
            k = int(counts[imu])
            selected_by_mu.append(
                _stratified_random_time_indices(candidate_now_cols, k, rng)
            )

    num_selected_per_mu = [int(arr.size) for arr in selected_by_mu]
    num_selected_total = int(np.sum(num_selected_per_mu))

    return {
        "mode": mode,
        "candidate_now_cols": candidate_now_cols,
        "selected_now_cols_by_mu": selected_by_mu,
        "num_candidates_per_mu": n_candidates_per_mu,
        "num_candidates_total": n_candidates_total,
        "num_selected_per_mu": num_selected_per_mu,
        "num_selected_total": num_selected_total,
        "total_snapshots": total_snapshots,
        "total_snapshots_percent": total_snapshots_percent,
        "mu_points": None if mu_points is None else np.asarray(mu_points, dtype=np.float64),
    }


def save_ecsw_sampling_3d_plot(
    *,
    mu_points,
    dt,
    ecsw_plan,
    out_path,
    title="ECSW Snapshot Selection in $(\\mu_1,\\mu_2,t)$",
):
    """
    Save a 3D scatter plot of all candidate ECSW points and selected points.

    Axes:
      x -> mu_1
      y -> mu_2
      z -> physical time t = dt * column_index
    """
    import matplotlib.pyplot as plt

    if ecsw_plan is None:
        raise ValueError("ecsw_plan cannot be None.")

    dt = float(dt)
    if not np.isfinite(dt):
        raise ValueError("dt must be finite.")

    candidate_now_cols = np.asarray(ecsw_plan.get("candidate_now_cols"), dtype=int).reshape(-1)
    selected_by_mu = ecsw_plan.get("selected_now_cols_by_mu")
    if selected_by_mu is None:
        raise ValueError("ecsw_plan missing 'selected_now_cols_by_mu'.")

    mu_arr = np.asarray(mu_points, dtype=np.float64)
    if mu_arr.ndim == 1:
        mu_arr = mu_arr.reshape(-1, 1)
    if mu_arr.ndim != 2:
        raise ValueError(f"mu_points must be 2D or 1D, got shape {mu_arr.shape}.")
    if int(mu_arr.shape[0]) != int(len(selected_by_mu)):
        raise ValueError(
            "mu_points row count must match len(selected_now_cols_by_mu): "
            f"{mu_arr.shape[0]} vs {len(selected_by_mu)}."
        )

    mu1 = mu_arr[:, 0]
    if mu_arr.shape[1] >= 2:
        mu2 = mu_arr[:, 1]
    else:
        mu2 = np.zeros_like(mu1)

    num_mu = int(mu1.size)
    t_all = dt * candidate_now_cols.astype(np.float64)

    x_all = np.repeat(mu1, candidate_now_cols.size)
    y_all = np.repeat(mu2, candidate_now_cols.size)
    z_all = np.tile(t_all, num_mu)

    x_sel_parts = []
    y_sel_parts = []
    z_sel_parts = []
    for imu, cols in enumerate(selected_by_mu):
        cols_arr = np.asarray(cols, dtype=int).reshape(-1)
        if cols_arr.size == 0:
            continue
        x_sel_parts.append(np.full(cols_arr.size, mu1[imu], dtype=np.float64))
        y_sel_parts.append(np.full(cols_arr.size, mu2[imu], dtype=np.float64))
        z_sel_parts.append(dt * cols_arr.astype(np.float64))

    if x_sel_parts:
        x_sel = np.concatenate(x_sel_parts)
        y_sel = np.concatenate(y_sel_parts)
        z_sel = np.concatenate(z_sel_parts)
    else:
        x_sel = np.zeros((0,), dtype=np.float64)
        y_sel = np.zeros((0,), dtype=np.float64)
        z_sel = np.zeros((0,), dtype=np.float64)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if x_all.size > 0:
        ax.scatter(
            x_all,
            y_all,
            z_all,
            s=10,
            c="#B3B3B3",
            alpha=0.35,
            marker="o",
            linewidths=0.0,
            label="All candidates",
        )

    if x_sel.size > 0:
        ax.scatter(
            x_sel,
            y_sel,
            z_sel,
            s=42,
            c="red",
            alpha=0.95,
            marker="x",
            linewidths=1.2,
            label="Selected for ECSW",
        )

    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_zlabel(r"$t$")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "num_candidates": int(x_all.size),
        "num_selected": int(x_sel.size),
        "out_path": out_path,
    }
