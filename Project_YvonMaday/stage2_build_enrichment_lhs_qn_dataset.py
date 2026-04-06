#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 enrichment: build a PROM/HPROM qN dataset using extra LHS samples.

Key design choices:
- Keeps existing Results/* artifacts untouched.
- Writes everything under Results_Enrichment/*.
- Supports both backends:
  - PROM: full LSPG solves for enrichment samples.
  - HPROM: reuses baseline ECSW weights.
- Stores only reduced outputs for new LHS points (qN, stats).
- Does not save reconstructed full snapshots for enrichment solves.
"""

import os
import shutil
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0, MU1_RANGE, MU2_RANGE
from burgers.linear_manifold import (
    inviscid_burgers_implicit2D_LSPG,
    inviscid_burgers_implicit2D_LSPG_ecsw,
)

try:
    from enrichment_layout import ensure_enrichment_dirs, enrichment_stage2_dataset_dir
except ModuleNotFoundError:
    from .enrichment_layout import ensure_enrichment_dirs, enrichment_stage2_dataset_dir
try:
    from project_layout import ensure_layout_dirs, resolve_stage1_artifact, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import ensure_layout_dirs, resolve_stage1_artifact, write_kv_txt
try:
    from stage3_dataset_utils import resolve_stage3_dataset
except ModuleNotFoundError:
    from .stage3_dataset_utils import resolve_stage3_dataset


def _safe_mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def _mu_key(mu, ndigits=10):
    return (round(float(mu[0]), ndigits), round(float(mu[1]), ndigits))


def _lhs_2d(n_samples, mu1_range, mu2_range, seed):
    """Simple 2D Latin Hypercube Sampling (no external dependencies)."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, 2), dtype=np.float64)

    for j, (lo, hi) in enumerate((mu1_range, mu2_range)):
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.random(n_samples)
        points = cut[:-1] + u * (1.0 / n_samples)
        rng.shuffle(points)
        samples[:, j] = lo + points * (hi - lo)

    return samples


def _collect_mu_keys(per_mu_root):
    keys = set()
    if not os.path.isdir(per_mu_root):
        return keys
    for name in sorted(os.listdir(per_mu_root)):
        mu_dir = os.path.join(per_mu_root, name)
        if not os.path.isdir(mu_dir):
            continue
        mu_path = os.path.join(mu_dir, "mu.npy")
        if not os.path.exists(mu_path):
            continue
        mu = np.asarray(np.load(mu_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if mu.size != 2:
            continue
        keys.add(_mu_key(mu))
    return keys


def _collect_mu_list(per_mu_root):
    mu_list = []
    if not os.path.isdir(per_mu_root):
        return mu_list
    for name in sorted(os.listdir(per_mu_root)):
        mu_dir = os.path.join(per_mu_root, name)
        if not os.path.isdir(mu_dir):
            continue
        mu_path = os.path.join(mu_dir, "mu.npy")
        if not os.path.exists(mu_path):
            continue
        mu = np.asarray(np.load(mu_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if mu.size != 2:
            continue
        mu_list.append([float(mu[0]), float(mu[1])])
    return mu_list


def _generate_unique_lhs_mu_list(
    n_required,
    mu1_range,
    mu2_range,
    seed,
    forbidden_keys,
):
    selected = []
    taken = set(forbidden_keys)

    attempt = 0
    while len(selected) < n_required and attempt < 20:
        remaining = n_required - len(selected)
        batch = _lhs_2d(max(2 * remaining, remaining), mu1_range, mu2_range, seed + attempt)
        for mu in batch:
            key = _mu_key(mu)
            if key in taken:
                continue
            taken.add(key)
            selected.append([float(mu[0]), float(mu[1])])
            if len(selected) >= n_required:
                break
        attempt += 1

    # Rare fallback if near-collisions happen repeatedly.
    if len(selected) < n_required:
        rng = np.random.default_rng(seed + 10_000)
        while len(selected) < n_required:
            mu = [
                float(rng.uniform(mu1_range[0], mu1_range[1])),
                float(rng.uniform(mu2_range[0], mu2_range[1])),
            ]
            key = _mu_key(mu)
            if key in taken:
                continue
            taken.add(key)
            selected.append(mu)

    return selected


def _plot_sampling_points(base_first9_mu_list, lhs_mu_list, mu1_range, mu2_range, out_path):
    base_pts = np.asarray(base_first9_mu_list, dtype=np.float64) if len(base_first9_mu_list) > 0 else np.zeros((0, 2))
    lhs_pts = np.asarray(lhs_mu_list, dtype=np.float64) if len(lhs_mu_list) > 0 else np.zeros((0, 2))

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    if base_pts.shape[0] > 0:
        ax.scatter(
            base_pts[:, 0],
            base_pts[:, 1],
            s=100,
            c="black",
            marker="o",
            alpha=0.9,
            label="Baseline training (first 9)",
        )
    if lhs_pts.shape[0] > 0:
        ax.scatter(
            lhs_pts[:, 0],
            lhs_pts[:, 1],
            s=90,
            c="tab:blue",
            marker="x",
            alpha=0.9,
            label="Enrichment LHS points",
        )

    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Baseline vs Enrichment Sampling Points")
    mu1_lo, mu1_hi = float(mu1_range[0]), float(mu1_range[1])
    mu2_lo, mu2_hi = float(mu2_range[0]), float(mu2_range[1])
    pad_x = 0.06 * (mu1_hi - mu1_lo)
    pad_y = 0.08 * (mu2_hi - mu2_lo)
    ax.set_xlim(mu1_lo - pad_x, mu1_hi + pad_x)
    ax.set_ylim(mu2_lo - pad_y, mu2_hi + pad_y)
    # Draw the actual parameter-domain boundary to avoid visual ambiguity near corners.
    ax.plot(
        [mu1_lo, mu1_hi, mu1_hi, mu1_lo, mu1_lo],
        [mu2_lo, mu2_lo, mu2_hi, mu2_hi, mu2_lo],
        color="0.25",
        linewidth=1.4,
        linestyle="-",
        alpha=0.85,
        zorder=1,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _load_stage1_basis(total_modes=None):
    basis_path = resolve_stage1_artifact("basis.npy")
    uref_path = resolve_stage1_artifact("u_ref.npy")

    if not os.path.exists(basis_path):
        raise FileNotFoundError(f"Missing basis file: {basis_path}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"basis.npy must be 2D, got shape {basis.shape}")

    n_available = int(basis.shape[1])
    if total_modes is None:
        total_modes = n_available
    else:
        total_modes = int(total_modes)
        if total_modes < 1 or total_modes > n_available:
            raise ValueError(
                f"Requested total_modes={total_modes}, but basis has {n_available} modes."
            )

    vtot = basis[:, :total_modes]

    if os.path.exists(uref_path):
        u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        u_ref = np.zeros(vtot.shape[0], dtype=np.float64)

    if u_ref.size != vtot.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {vtot.shape[0]} from basis rows."
        )

    return vtot, u_ref, basis_path, uref_path, total_modes, n_available


def _resolve_base_ecsw_weights_path(base_dataset_dir, base_meta, total_modes):
    meta_path = base_meta.get("ecsw_weights_path", None)
    candidates = []
    if isinstance(meta_path, str) and len(meta_path) > 0:
        candidates.append(meta_path)
        candidates.append(os.path.join(base_dataset_dir, meta_path))
    candidates.append(os.path.join(base_dataset_dir, f"ecsw_weights_lspg_ntot{int(total_modes)}.npy"))
    candidates.append(os.path.join(THIS_DIR, "Results", "Stage2", f"ecsw_weights_lspg_ntot{int(total_modes)}.npy"))
    candidates.append(os.path.join(THIS_DIR, f"ecsw_weights_lspg_ntot{int(total_modes)}.npy"))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def _copy_base_dataset(base_per_mu_root, out_per_mu_dir):
    os.makedirs(out_per_mu_dir, exist_ok=True)
    keys = set()
    copied = 0

    keep_files = [
        "mu.npy",
        "t.npy",
        "qN.npy",
        "rom_stats.npy",
        "prom_stats.npy",
        "hprom_stats.npy",
    ]

    for name in sorted(os.listdir(base_per_mu_root)):
        src_mu_dir = os.path.join(base_per_mu_root, name)
        if not os.path.isdir(src_mu_dir):
            continue
        mu_path = os.path.join(src_mu_dir, "mu.npy")
        if not os.path.exists(mu_path):
            continue

        mu = np.asarray(np.load(mu_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if mu.size != 2:
            continue
        keys.add(_mu_key(mu))

        dst_mu_dir = os.path.join(out_per_mu_dir, name)
        os.makedirs(dst_mu_dir, exist_ok=True)
        for fname in keep_files:
            src = os.path.join(src_mu_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_mu_dir, fname))

        # Backward compatibility: if legacy split files exist but qN.npy does not,
        # rebuild canonical qN.npy so downstream Stage 3 can consume one format.
        dst_qn = os.path.join(dst_mu_dir, "qN.npy")
        if not os.path.exists(dst_qn):
            src_qnp = os.path.join(src_mu_dir, "qN_p.npy")
            src_qns = os.path.join(src_mu_dir, "qN_s.npy")
            if os.path.exists(src_qnp) and os.path.exists(src_qns):
                qnp = np.asarray(np.load(src_qnp, allow_pickle=False), dtype=np.float64)
                qns = np.asarray(np.load(src_qns, allow_pickle=False), dtype=np.float64)
                if qnp.ndim == 2 and qns.ndim == 2 and qnp.shape[1] == qns.shape[1]:
                    np.save(dst_qn, np.vstack([qnp, qns]))
        copied += 1

    return keys, copied


def main(argv=None):
    # -----------------------------
    # User settings
    # -----------------------------
    lhs_samples = 20
    lhs_seed = 42
    total_modes = None
    solve_backend = "hprom"
    copy_base_dataset = True
    reuse_base_ecsw_weights = True
    max_its = 20
    relnorm_cutoff = 1e-5
    min_delta = 1e-2
    linear_solver = "lstsq"
    normal_eq_reg = 1e-12

    parser = argparse.ArgumentParser(
        description="Build enrichment Stage-2 qN dataset with PROM/HPROM backend."
    )
    parser.add_argument("--backend", choices=("prom", "hprom"), default=solve_backend)
    parser.add_argument("--lhs-samples", type=int, default=lhs_samples)
    parser.add_argument("--lhs-seed", type=int, default=lhs_seed)
    parser.add_argument("--total-modes", type=int, default=total_modes)
    parser.add_argument("--max-its", type=int, default=max_its)
    parser.add_argument("--relnorm-cutoff", type=float, default=relnorm_cutoff)
    parser.add_argument("--min-delta", type=float, default=min_delta)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default=linear_solver)
    parser.add_argument("--normal-eq-reg", type=float, default=normal_eq_reg)
    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument("--copy-base-dataset", dest="copy_base_dataset", action="store_true")
    copy_group.add_argument("--no-copy-base-dataset", dest="copy_base_dataset", action="store_false")
    parser.set_defaults(copy_base_dataset=copy_base_dataset)
    ecsw_group = parser.add_mutually_exclusive_group()
    ecsw_group.add_argument("--reuse-base-ecsw-weights", dest="reuse_base_ecsw_weights", action="store_true")
    ecsw_group.add_argument("--no-reuse-base-ecsw-weights", dest="reuse_base_ecsw_weights", action="store_false")
    parser.set_defaults(reuse_base_ecsw_weights=reuse_base_ecsw_weights)
    args = parser.parse_args(argv)

    solve_backend = str(args.backend).strip().lower()
    lhs_samples = int(args.lhs_samples)
    lhs_seed = int(args.lhs_seed)
    total_modes = args.total_modes
    copy_base_dataset = bool(args.copy_base_dataset)
    reuse_base_ecsw_weights = bool(args.reuse_base_ecsw_weights)
    max_its = int(args.max_its)
    relnorm_cutoff = float(args.relnorm_cutoff)
    min_delta = float(args.min_delta)
    linear_solver = str(args.linear_solver).strip().lower()
    normal_eq_reg = float(args.normal_eq_reg)

    # -----------------------------
    # Setup
    # -----------------------------
    ensure_layout_dirs()
    ensure_enrichment_dirs()

    base_per_mu_root, base_ntot, base_dataset_dir, base_meta, _ = resolve_stage3_dataset(
        this_dir=THIS_DIR,
        requested_ntot=total_modes,
        expected_backend=solve_backend,
    )
    if total_modes is None:
        total_modes = int(base_ntot)
    else:
        total_modes = int(total_modes)
        if total_modes != int(base_ntot):
            raise ValueError(
                f"Requested total_modes={total_modes}, but base dataset has ntot={base_ntot}."
            )

    vtot, u_ref, basis_path, uref_path, total_modes, n_available = _load_stage1_basis(total_modes)
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != vtot.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {vtot.shape[0]} from basis rows."
        )

    out_dir = enrichment_stage2_dataset_dir(total_modes=total_modes, lhs_samples=lhs_samples)
    per_mu_dir = os.path.join(out_dir, "per_mu")
    os.makedirs(per_mu_dir, exist_ok=True)

    # -----------------------------
    # Copy baseline dataset (optional)
    # -----------------------------
    base_keys = _collect_mu_keys(base_per_mu_root)
    copied_base = 0
    if copy_base_dataset:
        _, copied_base = _copy_base_dataset(base_per_mu_root, per_mu_dir)

    # -----------------------------
    # LHS enrichment set
    # -----------------------------
    lhs_mu_list = _generate_unique_lhs_mu_list(
        n_required=int(lhs_samples),
        mu1_range=MU1_RANGE,
        mu2_range=MU2_RANGE,
        seed=int(lhs_seed),
        forbidden_keys=base_keys,
    )

    print(f"[Stage2-Enrich] base dataset: {base_dataset_dir}")
    print(f"[Stage2-Enrich] solve_backend: {solve_backend}")
    print(f"[Stage2-Enrich] base trajectories copied: {copied_base}")
    print(f"[Stage2-Enrich] LHS samples requested: {lhs_samples}")
    print(f"[Stage2-Enrich] LHS samples generated: {len(lhs_mu_list)}")
    print(f"[Stage2-Enrich] output dataset: {out_dir}")

    base_mu_list = _collect_mu_list(base_per_mu_root)
    base_first9_mu_list = base_mu_list[:9]
    sampling_plot_path = os.path.join(out_dir, "stage2_sampling_points.png")
    np.save(os.path.join(out_dir, "base_first9_mu.npy"), np.asarray(base_first9_mu_list, dtype=np.float64))
    np.save(os.path.join(out_dir, "lhs_mu.npy"), np.asarray(lhs_mu_list, dtype=np.float64))
    _plot_sampling_points(
        base_first9_mu_list=base_first9_mu_list,
        lhs_mu_list=lhs_mu_list,
        mu1_range=MU1_RANGE,
        mu2_range=MU2_RANGE,
        out_path=sampling_plot_path,
    )
    print(f"[Stage2-Enrich] sampling plot: {sampling_plot_path}")

    # -----------------------------
    # Optional ECSW setup (HPROM only)
    # -----------------------------
    ecsw_weights = None
    ecsw_weights_path = None
    ecsw_source = None
    ecsw_residual = np.nan
    n_ecsw_elements = None

    if solve_backend == "hprom":
        if not reuse_base_ecsw_weights:
            raise ValueError(
                "Enrichment HPROM currently supports only --reuse-base-ecsw-weights. "
                "Set --backend prom for enrichment without ECSW."
            )

        expected_num_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
        ecsw_weights_path = os.path.join(out_dir, f"ecsw_weights_lspg_ntot{total_modes}_lhs{lhs_samples}.npy")

        base_ecsw_weights_path = _resolve_base_ecsw_weights_path(
            base_dataset_dir=base_dataset_dir,
            base_meta=base_meta,
            total_modes=total_modes,
        )
        if base_ecsw_weights_path is None:
            raise FileNotFoundError(
                "Could not resolve baseline Stage2 ECSW weights path. "
                "Generate baseline HPROM Stage2 dataset first."
            )

        ecsw_weights = np.asarray(np.load(base_ecsw_weights_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if ecsw_weights.size != expected_num_cells:
            raise ValueError(
                f"Baseline ECSW weights size mismatch: got {ecsw_weights.size}, expected {expected_num_cells}."
            )
        np.save(ecsw_weights_path, ecsw_weights)
        ecsw_source = f"copied_from_base:{base_ecsw_weights_path}"
        n_ecsw_elements = int(np.sum(ecsw_weights > 0.0))

        print(f"[Stage2-Enrich] ECSW path: {ecsw_weights_path}")
        print(f"[Stage2-Enrich] ECSW source: {ecsw_source}")
        print(f"[Stage2-Enrich] ECSW N_e: {n_ecsw_elements}")
        print(f"[Stage2-Enrich] ECSW residual: {ecsw_residual}")
        print("[Stage2-Enrich] ECSW mode: reused baseline weights (no HDM run in this stage).")
    else:
        print("[Stage2-Enrich] backend=PROM -> no ECSW weights are used.")

    # -----------------------------
    # ROM solves for enrichment points (store qN only)
    # -----------------------------
    t_vec = DT * np.arange(NUM_STEPS + 1, dtype=np.float64)
    for i, mu in enumerate(lhs_mu_list, start=1):
        mu_tag = _safe_mu_tag(mu)
        mu_dir = os.path.join(per_mu_dir, mu_tag)
        os.makedirs(mu_dir, exist_ok=True)

        print(f"[Stage2-Enrich] [{i}/{len(lhs_mu_list)}] {solve_backend.upper()} solve for {mu_tag}")
        if solve_backend == "prom":
            rom_snaps, rom_times = inviscid_burgers_implicit2D_LSPG(
                grid_x=GRID_X,
                grid_y=GRID_Y,
                w0=w0,
                dt=DT,
                num_steps=NUM_STEPS,
                mu=mu,
                basis=vtot,
                u_ref=u_ref,
                max_its=max_its,
                relnorm_cutoff=relnorm_cutoff,
                min_delta=min_delta,
                linear_solver=linear_solver,
                normal_eq_reg=normal_eq_reg,
            )
            qN = vtot.T @ (rom_snaps - u_ref[:, None])
        else:
            qN, rom_times = inviscid_burgers_implicit2D_LSPG_ecsw(
                grid_x=GRID_X,
                grid_y=GRID_Y,
                weights=ecsw_weights,
                w0=w0,
                dt=DT,
                num_steps=NUM_STEPS,
                mu=mu,
                basis=vtot,
                u_ref=u_ref,
                max_its=max_its,
                relnorm_cutoff=relnorm_cutoff,
                min_delta=min_delta,
                linear_solver=linear_solver,
                normal_eq_reg=normal_eq_reg,
            )

        np.save(os.path.join(mu_dir, "mu.npy"), np.asarray(mu, dtype=np.float64))
        np.save(os.path.join(mu_dir, "t.npy"), t_vec)
        np.save(os.path.join(mu_dir, "qN.npy"), qN)
        np.save(os.path.join(mu_dir, "rom_stats.npy"), np.asarray(rom_times, dtype=np.float64))
        if solve_backend == "prom":
            np.save(os.path.join(mu_dir, "prom_stats.npy"), np.asarray(rom_times, dtype=np.float64))
        else:
            np.save(os.path.join(mu_dir, "hprom_stats.npy"), np.asarray(rom_times, dtype=np.float64))

    total_traj = int(copied_base + len(lhs_mu_list))
    meta = {
        "solve_backend": solve_backend,
        "is_enrichment_dataset": True,
        "total_modes": int(total_modes),
        "n_available_modes": int(n_available),
        "coefficient_storage": "qN_only",
        "dt": float(DT),
        "num_steps": int(NUM_STEPS),
        "basis_path": basis_path,
        "u_ref_path": uref_path,
        "base_dataset_dir": base_dataset_dir,
        "base_dataset_root": base_per_mu_root,
        "copy_base_dataset": bool(copy_base_dataset),
        "num_base_traj_copied": int(copied_base),
        "num_lhs_traj": int(len(lhs_mu_list)),
        "num_traj": int(total_traj),
        "lhs_samples_requested": int(lhs_samples),
        "lhs_seed": int(lhs_seed),
        "mu1_range": list(map(float, MU1_RANGE)),
        "mu2_range": list(map(float, MU2_RANGE)),
        "save_rom_snaps": False,
        "make_plots": False,
        "linear_solver": linear_solver,
        "normal_eq_reg": float(normal_eq_reg),
        "max_its": int(max_its),
        "relnorm_cutoff": float(relnorm_cutoff),
        "min_delta": float(min_delta),
        "state_size": int(vtot.shape[0]),
        "reduced_size": int(vtot.shape[1]),
        "ecsw_weights_path": ecsw_weights_path,
        "ecsw_weights_source": ecsw_source,
        "ecsw_residual": float(ecsw_residual) if np.isfinite(ecsw_residual) else np.nan,
        "n_ecsw_elements": None if n_ecsw_elements is None else int(n_ecsw_elements),
        "reuse_base_ecsw_weights": bool(reuse_base_ecsw_weights),
        "sampling_plot_path": sampling_plot_path,
    }
    np.save(os.path.join(out_dir, "meta.npy"), meta, allow_pickle=True)

    summary_path = os.path.join(out_dir, "stage2_enrichment_summary.txt")
    write_kv_txt(
        summary_path,
        [
            ("dataset_dir", out_dir),
            ("per_mu_dir", per_mu_dir),
            ("solve_backend", solve_backend),
            ("total_modes", total_modes),
            ("coefficient_storage", "qN_only"),
            ("num_base_traj_copied", copied_base),
            ("num_lhs_traj", len(lhs_mu_list)),
            ("num_traj_total", total_traj),
            ("lhs_samples_requested", lhs_samples),
            ("lhs_seed", lhs_seed),
            ("base_dataset_dir", base_dataset_dir),
            ("base_dataset_root", base_per_mu_root),
            ("ecsw_weights_path", ecsw_weights_path),
            ("ecsw_weights_source", ecsw_source),
            ("ecsw_residual", ecsw_residual),
            ("n_ecsw_elements", n_ecsw_elements),
            ("reuse_base_ecsw_weights", reuse_base_ecsw_weights),
            ("save_rom_snaps", False),
            ("make_plots", False),
            ("sampling_plot_path", sampling_plot_path),
        ],
    )

    print("[Stage2-Enrich] done.")
    print(f"[Stage2-Enrich] summary: {summary_path}")


if __name__ == "__main__":
    main()
