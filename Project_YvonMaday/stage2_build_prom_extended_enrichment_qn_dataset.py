#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the PROM 9+18+18 extended LHS coefficient dataset.

This is the PROM analogue of the extended HPROM enrichment dataset:
  - copy the 9 baseline PROM qN trajectories,
  - generate or reuse the same 18 interior + 18 exterior LHS design,
  - solve the 36 new points with the full linear PROM,
  - write Stage-2 metadata compatible with Stage-3 training scripts.

If --design-source-dir is provided and contains lhs_mu.npy/interior_lhs_mu.npy/
exterior_lhs_mu.npy/exterior_region_labels.json, those files are reused
directly. Otherwise the design is generated with the same deterministic code
and seed as the HPROM extended-enrichment builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0  # noqa: E402
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG  # noqa: E402
from stage2_build_paper_extended_enrichment_qn_dataset import (  # noqa: E402
    ORIGINAL_MU1_RANGE,
    ORIGINAL_MU2_RANGE,
    _atomic_save,
    _atomic_write_text,
    _baseline_mu_from_dataset,
    _load_mu_dirs,
    _load_or_create_design,
    _mu_tag,
    _plot_sampling,
    _sha256,
    _validate_design,
    _validate_trajectory,
    _write_manifest,
)
from stage3_dataset_utils import read_dataset_meta  # noqa: E402


def _copy_design_from_source(source: Path, output_dir: Path) -> bool:
    required = [
        "interior_lhs_mu.npy",
        "exterior_lhs_mu.npy",
        "lhs_mu.npy",
        "exterior_region_labels.json",
    ]
    if not source:
        return False
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"--design-source-dir does not exist: {source}")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Design source {source} is missing: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(source / name, output_dir / name)
    return True


def _copy_baseline_dataset(base_per_mu: Path, output_per_mu: Path, total_modes: int) -> list[list[float]]:
    keep = ("mu.npy", "t.npy", "qN.npy", "rom_stats.npy", "prom_stats.npy")
    copied = []
    for source_dir, mu in _load_mu_dirs(base_per_mu):
        _validate_trajectory(source_dir, total_modes)
        target_dir = output_per_mu / source_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in keep:
            src = source_dir / name
            if src.is_file():
                shutil.copy2(src, target_dir / name)
        _validate_trajectory(target_dir, total_modes)
        copied.append(mu.tolist())
    return copied


def _load_basis_and_ref(basis_path: Path, u_ref_path: Path, total_modes: int):
    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    u_ref = np.asarray(np.load(u_ref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or basis.shape[1] < total_modes:
        raise ValueError(f"Invalid basis shape {basis.shape} for total_modes={total_modes}.")
    if u_ref.size != basis.shape[0]:
        raise ValueError(f"u_ref size {u_ref.size} does not match basis rows {basis.shape[0]}.")
    return basis, u_ref


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basis-path", required=True)
    parser.add_argument("--u-ref-path", required=True)
    parser.add_argument("--design-source-dir", default="")
    parser.add_argument("--total-modes", type=int, default=151)
    parser.add_argument("--interior-samples", type=int, default=18)
    parser.add_argument("--exterior-samples", type=int, default=18)
    parser.add_argument("--lhs-seed", type=int, default=42)
    parser.add_argument("--margin-fraction", type=float, default=0.25)
    parser.add_argument("--exclude-mu", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--regenerate-design", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1e-12)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    base_dataset = Path(args.base_dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    basis_path = Path(args.basis_path).expanduser().resolve()
    u_ref_path = Path(args.u_ref_path).expanduser().resolve()
    design_source = Path(args.design_source_dir).expanduser().resolve() if args.design_source_dir else None
    base_per_mu = base_dataset / "per_mu"
    output_per_mu = output_dir / "per_mu"

    for path in (base_dataset, base_per_mu, basis_path, u_ref_path):
        if not path.exists():
            raise FileNotFoundError(path)

    base_meta, base_meta_path = read_dataset_meta(str(base_dataset))
    if str(base_meta.get("solve_backend", "")).lower() != "prom":
        raise ValueError("The baseline dataset must have solve_backend=prom.")
    if int(base_meta.get("total_modes", -1)) != int(args.total_modes):
        raise ValueError("The baseline dataset total_modes does not match --total-modes.")

    basis, u_ref = _load_basis_and_ref(basis_path, u_ref_path, int(args.total_modes))
    vtot = basis[:, : int(args.total_modes)]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != vtot.shape[0]:
        raise ValueError(f"W0 size {w0.size} does not match basis rows {vtot.shape[0]}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_mu = _baseline_mu_from_dataset(base_per_mu, int(args.total_modes))
    evaluation_mu = np.asarray(args.exclude_mu, dtype=np.float64).reshape(-1, 2)

    reused_design_source = False
    if design_source is not None and not bool(args.regenerate_design):
        reused_design_source = _copy_design_from_source(design_source, output_dir)

    interior_mu, exterior_mu, exterior_regions, expanded_mu1, expanded_mu2 = _load_or_create_design(
        output_dir,
        interior_samples=int(args.interior_samples),
        exterior_samples=int(args.exterior_samples),
        seed=int(args.lhs_seed),
        margin_fraction=float(args.margin_fraction),
        baseline_mu=baseline_mu,
        evaluation_mu=evaluation_mu,
        regenerate=bool(args.regenerate_design),
    )
    lhs_mu = np.vstack((interior_mu, exterior_mu))

    _validate_design(
        interior_mu=interior_mu,
        exterior_mu=exterior_mu,
        baseline_mu=baseline_mu,
        evaluation_mu=evaluation_mu,
        expanded_mu1=expanded_mu1,
        expanded_mu2=expanded_mu2,
    )
    _atomic_save(output_dir / "baseline_mu.npy", baseline_mu)
    _atomic_save(output_dir / "evaluation_mu.npy", evaluation_mu)
    _write_manifest(output_dir / "parameter_manifest.csv", baseline_mu, interior_mu, exterior_mu, exterior_regions, evaluation_mu)
    _plot_sampling(
        output_dir / "stage2_sampling_points.png",
        baseline_mu,
        interior_mu,
        exterior_mu,
        evaluation_mu,
        expanded_mu1,
        expanded_mu2,
    )

    print(f"[prom-extended-enrichment-stage2] base_dataset={base_dataset}")
    print(f"[prom-extended-enrichment-stage2] output_dataset={output_dir}")
    print(f"[prom-extended-enrichment-stage2] basis={basis_path}")
    print(f"[prom-extended-enrichment-stage2] u_ref={u_ref_path}")
    print(f"[prom-extended-enrichment-stage2] design_source={design_source if design_source else 'generated'}")
    print(f"[prom-extended-enrichment-stage2] reused_design_source={reused_design_source}")
    print(f"[prom-extended-enrichment-stage2] original_mu1_range={ORIGINAL_MU1_RANGE}")
    print(f"[prom-extended-enrichment-stage2] original_mu2_range={ORIGINAL_MU2_RANGE}")
    print(f"[prom-extended-enrichment-stage2] expanded_mu1_range={expanded_mu1}")
    print(f"[prom-extended-enrichment-stage2] expanded_mu2_range={expanded_mu2}")
    print(f"[prom-extended-enrichment-stage2] interior_lhs={len(interior_mu)} exterior_lhs={len(exterior_mu)} total_lhs={len(lhs_mu)}")

    if args.plan_only:
        summary_lines = [
            f"dataset_dir: {output_dir}",
            "plan_only: True",
            "solve_backend: prom",
            f"total_modes: {args.total_modes}",
            f"num_base_traj_available: {len(baseline_mu)}",
            f"num_interior_lhs_traj: {len(interior_mu)}",
            f"num_exterior_lhs_traj: {len(exterior_mu)}",
            f"num_lhs_traj: {len(lhs_mu)}",
            f"lhs_seed: {args.lhs_seed}",
            f"margin_fraction: {args.margin_fraction}",
            f"design_source_dir: {design_source if design_source else ''}",
            f"reused_design_source: {reused_design_source}",
        ]
        _atomic_write_text(output_dir / "stage2_extended_enrichment_plan_summary.txt", "\n".join(summary_lines) + "\n")
        print("[prom-extended-enrichment-stage2] PLAN_ONLY complete; no qN solves were run.")
        return

    output_per_mu.mkdir(parents=True, exist_ok=True)
    copied_baseline_mu = np.asarray(_copy_baseline_dataset(base_per_mu, output_per_mu, int(args.total_modes)))
    if copied_baseline_mu.shape != baseline_mu.shape:
        raise ValueError("Copied baseline parameter count does not match source baseline set.")

    t_vec = DT * np.arange(NUM_STEPS + 1, dtype=np.float64)
    total_new = len(lhs_mu)
    for index, mu in enumerate(lhs_mu, start=1):
        mu_dir = output_per_mu / _mu_tag(mu)
        complete = all((mu_dir / name).is_file() for name in ("mu.npy", "t.npy", "qN.npy", "rom_stats.npy", "prom_stats.npy"))
        if complete:
            try:
                saved_mu = np.asarray(np.load(mu_dir / "mu.npy", allow_pickle=False)).reshape(-1)
                _validate_trajectory(mu_dir, int(args.total_modes))
                if np.allclose(saved_mu, mu, rtol=0.0, atol=1e-14):
                    print(f"[prom-extended-enrichment-stage2] [{index}/{total_new}] reuse {_mu_tag(mu)}")
                    continue
            except (OSError, ValueError):
                pass

        print(f"[prom-extended-enrichment-stage2] [{index}/{total_new}] solve {_mu_tag(mu)}")
        rom_snaps, qn, stats = inviscid_burgers_implicit2D_LSPG(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=np.asarray(mu, dtype=np.float64),
            basis=vtot,
            u_ref=u_ref,
            max_its=int(args.max_its),
            relnorm_cutoff=float(args.relnorm_cutoff),
            min_delta=float(args.min_delta),
            linear_solver=str(args.linear_solver),
            normal_eq_reg=float(args.normal_eq_reg),
            return_red_coords=True,
        )
        qn = np.asarray(qn, dtype=np.float64)
        stats = np.asarray(stats, dtype=np.float64)
        reconstructed = u_ref[:, None] + vtot @ qn
        rel_state_consistency = float(np.linalg.norm(reconstructed - rom_snaps) / max(np.linalg.norm(rom_snaps), np.finfo(float).eps))
        if rel_state_consistency > 1e-10:
            raise RuntimeError(f"Solver-coordinate state consistency failed for {_mu_tag(mu)}: {rel_state_consistency:.3e}")

        _atomic_save(mu_dir / "mu.npy", np.asarray(mu, dtype=np.float64))
        _atomic_save(mu_dir / "t.npy", t_vec)
        _atomic_save(mu_dir / "qN.npy", qn)
        _atomic_save(mu_dir / "rom_stats.npy", stats)
        _atomic_save(mu_dir / "prom_stats.npy", stats)
        _validate_trajectory(mu_dir, int(args.total_modes))

    records = _load_mu_dirs(output_per_mu)
    expected_total = 9 + int(args.interior_samples) + int(args.exterior_samples)
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} trajectories, found {len(records)} in {output_per_mu}.")
    for mu_dir, _ in records:
        _validate_trajectory(mu_dir, int(args.total_modes))

    metadata = {
        "solve_backend": "prom",
        "is_enrichment_dataset": True,
        "enrichment_protocol": "baseline_9_plus_lhs36_ext25_linear_prom",
        "total_modes": int(args.total_modes),
        "n_available_modes": int(basis.shape[1]),
        "coefficient_storage": "direct_solver_qN_only",
        "dt": float(DT),
        "num_steps": int(NUM_STEPS),
        "num_traj": int(len(records)),
        "num_base_traj_copied": 9,
        "num_lhs_traj": int(len(lhs_mu)),
        "num_interior_lhs_traj": int(len(interior_mu)),
        "num_exterior_lhs_traj": int(len(exterior_mu)),
        "lhs_seed": int(args.lhs_seed),
        "margin_fraction": float(args.margin_fraction),
        "design_source_dir": str(design_source) if design_source else "",
        "reused_design_source": bool(reused_design_source),
        "original_mu1_range": list(ORIGINAL_MU1_RANGE),
        "original_mu2_range": list(ORIGINAL_MU2_RANGE),
        "expanded_mu1_range": list(map(float, expanded_mu1)),
        "expanded_mu2_range": list(map(float, expanded_mu2)),
        "exterior_region_labels": exterior_regions,
        "basis_path": str(basis_path),
        "basis_sha256": _sha256(basis_path),
        "u_ref_path": str(u_ref_path),
        "u_ref_sha256": _sha256(u_ref_path),
        "base_dataset_dir": str(base_dataset),
        "base_dataset_meta_path": str(base_meta_path),
        "linear_solver": str(args.linear_solver),
        "normal_eq_reg": float(args.normal_eq_reg),
        "max_its": int(args.max_its),
        "relnorm_cutoff": float(args.relnorm_cutoff),
        "min_delta": float(args.min_delta),
        "state_size": int(vtot.shape[0]),
        "reduced_size": int(vtot.shape[1]),
        "save_rom_snaps": False,
        "evaluation_points_excluded_from_lhs": evaluation_mu.tolist(),
    }
    _atomic_save(output_dir / "meta.npy", metadata, allow_pickle=True)
    _atomic_write_text(output_dir / "meta.json", json.dumps(metadata, indent=2) + "\n")

    summary_lines = [
        f"dataset_dir: {output_dir}",
        "solve_backend: prom",
        f"total_modes: {args.total_modes}",
        "coefficient_storage: direct_solver_qN_only",
        "num_base_traj_copied: 9",
        f"num_interior_lhs_traj: {len(interior_mu)}",
        f"num_exterior_lhs_traj: {len(exterior_mu)}",
        f"num_lhs_traj: {len(lhs_mu)}",
        f"num_traj_total: {len(records)}",
        f"lhs_seed: {args.lhs_seed}",
        f"margin_fraction: {args.margin_fraction}",
        f"design_source_dir: {design_source if design_source else ''}",
        f"reused_design_source: {reused_design_source}",
        f"expanded_mu1_range: {expanded_mu1}",
        f"expanded_mu2_range: {expanded_mu2}",
        f"basis_path: {basis_path}",
        f"u_ref_path: {u_ref_path}",
        "save_rom_snaps: False",
    ]
    _atomic_write_text(output_dir / "stage2_enrichment_summary.txt", "\n".join(summary_lines) + "\n")
    print(f"[prom-extended-enrichment-stage2] complete: {len(records)} trajectories")
    print(f"[prom-extended-enrichment-stage2] summary={output_dir / 'stage2_enrichment_summary.txt'}")


if __name__ == "__main__":
    main()
