#!/usr/bin/env python3
"""Build the nested Euclidean-POD PROM enrichment datasets used in the paper.

The study has two levels only: nine baseline trajectories plus eight or
eighteen additional trajectories.  Both are selected from the same fixed
18-interior/18-margin LHS candidate design.  The eight-point level is a strict
subset of the eighteen-point level, so its PROM coordinates need not be
solved independently.  No 9+36 coefficient dataset is generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG
from stage2_build_paper_extended_enrichment_qn_dataset import (
    ORIGINAL_MU1_RANGE,
    ORIGINAL_MU2_RANGE,
    _atomic_save,
    _atomic_write_text,
    _baseline_mu_from_dataset,
    _generate_extended_design,
    _load_mu_dirs,
    _mu_tag,
    _sha256,
    _storage_key,
    _validate_design,
    _validate_trajectory,
    _write_manifest,
)
from stage2_build_prom_extended_enrichment_qn_dataset import (
    _copy_baseline_dataset,
    _load_basis_and_ref,
)
from stage3_dataset_utils import read_dataset_meta


EVALUATION_MU = np.asarray(
    (
        (4.875, 0.0225),
        (4.560, 0.0190),
        (5.190, 0.0260),
        (4.000, 0.0330),
    ),
    dtype=np.float64,
)
PARAMETER_VALIDATION_MU = np.asarray(
    (
        (4.5625, 0.02625),
        (5.1875, 0.01875),
    ),
    dtype=np.float64,
)

# The two prefixes define the reported nested levels.  Four margin points at
# the first level preserve representation of all four margin directions.
LEVELS = {
    "lhs8": (4, 4),
    "lhs18": (9, 9),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--basis-path", type=Path, required=True)
    parser.add_argument("--u-ref-path", type=Path, required=True)
    parser.add_argument("--total-modes", type=int, default=151)
    parser.add_argument("--lhs-seed", type=int, default=42)
    parser.add_argument("--margin-fraction", type=float, default=0.25)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1e-12)
    parser.add_argument("--force", action="store_true", help="Replace only the two named nested datasets.")
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs and print the selected points without writing files.")
    return parser.parse_args()


def _normalize(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (points - lower) / (upper - lower)


def _greedy_maximin_order(points: np.ndarray, initial: list[int]) -> list[int]:
    """Return a deterministic space-filling order while preserving ``initial``."""
    chosen = list(dict.fromkeys(initial))
    remaining = [index for index in range(len(points)) if index not in chosen]
    if not chosen:
        centroid = points.mean(axis=0)
        first = min(remaining, key=lambda index: (np.linalg.norm(points[index] - centroid), index))
        chosen.append(first)
        remaining.remove(first)
    while remaining:
        selected = points[np.asarray(chosen)]
        candidate = max(
            remaining,
            key=lambda index: (float(np.min(np.linalg.norm(selected - points[index], axis=1))), -index),
        )
        chosen.append(candidate)
        remaining.remove(candidate)
    return chosen


def _exterior_order(points: np.ndarray, labels: list[str], lower: np.ndarray, upper: np.ndarray) -> list[int]:
    normalized = _normalize(points, lower, upper)
    strata = ("left_margin", "right_margin", "top_margin", "bottom_margin")
    if set(labels) != set(strata):
        raise ValueError(f"Expected margin strata {strata}, got {sorted(set(labels))}.")

    # Every prefix containing at least four margin samples contains each side.
    seeds: list[int] = []
    for stratum in strata:
        members = [index for index, label in enumerate(labels) if label == stratum]
        centroid = normalized[members].mean(axis=0)
        seeds.append(min(members, key=lambda index: (np.linalg.norm(normalized[index] - centroid), index)))
    return _greedy_maximin_order(normalized, seeds)


def _candidate_design(
    baseline_mu: np.ndarray, *, seed: int, margin_fraction: float
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[float, float], tuple[float, float], list[int], list[int]]:
    # Neither online evaluation nor held-out regression-validation parameters
    # may enter an enrichment-training target set.
    forbidden = np.vstack((baseline_mu, EVALUATION_MU, PARAMETER_VALIDATION_MU))
    interior, exterior, labels, expanded_mu1, expanded_mu2 = _generate_extended_design(
        interior_samples=18,
        exterior_samples=18,
        seed=seed,
        margin_fraction=margin_fraction,
        forbidden=forbidden,
    )
    _validate_design(
        interior_mu=interior,
        exterior_mu=exterior,
        evaluation_mu=EVALUATION_MU,
        baseline_mu=baseline_mu,
        expanded_mu1=expanded_mu1,
        expanded_mu2=expanded_mu2,
    )
    lower = np.asarray((expanded_mu1[0], expanded_mu2[0]), dtype=np.float64)
    upper = np.asarray((expanded_mu1[1], expanded_mu2[1]), dtype=np.float64)
    interior_order = _greedy_maximin_order(_normalize(interior, lower, upper), [])
    exterior_order = _exterior_order(exterior, labels, lower, upper)
    return interior, exterior, labels, expanded_mu1, expanded_mu2, interior_order, exterior_order


def _level_selection(
    interior: np.ndarray,
    exterior: np.ndarray,
    labels: list[str],
    interior_order: list[int],
    exterior_order: list[int],
    level: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], list[int]]:
    n_interior, n_exterior = LEVELS[level]
    interior_indices = interior_order[:n_interior]
    exterior_indices = exterior_order[:n_exterior]
    return (
        interior[np.asarray(interior_indices)],
        exterior[np.asarray(exterior_indices)],
        [labels[index] for index in exterior_indices],
        interior_indices,
        exterior_indices,
    )


def _dataset_dir(results_root: Path, level: str, total_modes: int) -> Path:
    campaign_root = results_root / f"euclidean_prom_enrichment_{level}"
    return campaign_root / "Stage2" / f"prom_coeff_dataset_ntot{total_modes}_enriched_{level}"


def _expected_tags(baseline: np.ndarray, interior: np.ndarray, exterior: np.ndarray) -> set[str]:
    return {_mu_tag(mu) for mu in np.vstack((baseline, interior, exterior))}


def _is_complete(path: Path, *, total_modes: int, expected_tags: set[str], level: str) -> bool:
    meta_path = path / "meta.json"
    if not meta_path.is_file() or not (path / "nested_subset_manifest.json").is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("solve_backend") != "prom" or int(meta.get("total_modes", -1)) != total_modes:
            return False
        if meta.get("subset_level") != level or int(meta.get("num_traj", -1)) != len(expected_tags):
            return False
        directories = {child.name for child in (path / "per_mu").iterdir() if child.is_dir()}
        if directories != expected_tags:
            return False
        for tag in directories:
            _validate_trajectory(path / "per_mu" / tag, total_modes)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _validate_existing_tags(dataset: Path, expected_tags: set[str]) -> None:
    """Reject partial datasets containing trajectories outside this campaign."""
    per_mu = dataset / "per_mu"
    if not per_mu.exists():
        return
    if not per_mu.is_dir():
        raise NotADirectoryError(per_mu)
    actual = {child.name for child in per_mu.iterdir() if child.is_dir()}
    unexpected = actual - expected_tags
    if unexpected:
        raise RuntimeError(
            f"Dataset {dataset} contains unexpected parameter directories: {sorted(unexpected)}. "
            "Pass --force to replace this named dataset."
        )


def _write_subset_metadata(
    path: Path,
    *,
    base_meta: dict[str, Any],
    base_meta_path: Path,
    baseline: np.ndarray,
    interior: np.ndarray,
    exterior: np.ndarray,
    exterior_labels: list[str],
    candidate_interior: np.ndarray,
    candidate_exterior: np.ndarray,
    candidate_labels: list[str],
    interior_order: list[int],
    exterior_order: list[int],
    selected_interior: list[int],
    selected_exterior: list[int],
    level: str,
    basis_path: Path,
    u_ref_path: Path,
    expanded_mu1: tuple[float, float],
    expanded_mu2: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    selected_lhs = np.vstack((interior, exterior))
    _atomic_save(path / "baseline_mu.npy", baseline)
    _atomic_save(path / "interior_lhs_mu.npy", interior)
    _atomic_save(path / "exterior_lhs_mu.npy", exterior)
    _atomic_save(path / "lhs_mu.npy", selected_lhs)
    _atomic_save(path / "evaluation_mu.npy", EVALUATION_MU)
    _atomic_write_text(path / "exterior_region_labels.json", json.dumps(exterior_labels, indent=2) + "\n")
    _write_manifest(path / "parameter_manifest.csv", baseline, interior, exterior, exterior_labels, EVALUATION_MU)

    candidate = {
        "candidate_design": "fixed seed 18 interior + 18 margin LHS pool; no 9+36 trajectories are solved",
        "lhs_seed": args.lhs_seed,
        "margin_fraction": args.margin_fraction,
        "interior_mu": candidate_interior.tolist(),
        "exterior_mu": candidate_exterior.tolist(),
        "exterior_region_labels": candidate_labels,
        "interior_order_source_indices": interior_order,
        "exterior_order_source_indices": exterior_order,
        "selected_interior_source_indices": selected_interior,
        "selected_exterior_source_indices": selected_exterior,
        "selected_exterior_labels": exterior_labels,
    }
    _atomic_write_text(path / "nested_subset_manifest.json", json.dumps(candidate, indent=2) + "\n")

    metadata = dict(base_meta)
    metadata.update(
        {
            "solve_backend": "prom",
            "is_enrichment_dataset": True,
            "is_nested_subset_dataset": True,
            "enrichment_protocol": f"euclidean_prom_baseline_9_plus_nested_{len(selected_lhs)}_from_fixed_lhs36_candidate_pool",
            "subset_level": level,
            "source_enrichment_dataset": (
                "not applicable; the 9+18 trajectories are solved directly"
                if level == "lhs18"
                else str(
                    path.parents[2]
                    / "euclidean_prom_enrichment_lhs18"
                    / "Stage2"
                    / f"prom_coeff_dataset_ntot{args.total_modes}_enriched_lhs18"
                )
            ),
            "base_dataset_dir": str(args.base_dataset_dir.resolve()),
            "base_dataset_meta_path": str(base_meta_path),
            "num_traj": int(len(baseline) + len(selected_lhs)),
            "num_base_traj_copied": int(len(baseline)),
            "num_lhs_traj": int(len(selected_lhs)),
            "num_interior_lhs_traj": int(len(interior)),
            "num_exterior_lhs_traj": int(len(exterior)),
            "lhs_seed": int(args.lhs_seed),
            "margin_fraction": float(args.margin_fraction),
            "original_mu1_range": list(ORIGINAL_MU1_RANGE),
            "original_mu2_range": list(ORIGINAL_MU2_RANGE),
            "expanded_mu1_range": list(map(float, expanded_mu1)),
            "expanded_mu2_range": list(map(float, expanded_mu2)),
            "exterior_region_labels": exterior_labels,
            "selected_interior_source_indices": selected_interior,
            "selected_exterior_source_indices": selected_exterior,
            "basis_path": str(basis_path),
            "basis_sha256": _sha256(basis_path),
            "u_ref_path": str(u_ref_path),
            "u_ref_sha256": _sha256(u_ref_path),
            "linear_solver": args.linear_solver,
            "normal_eq_reg": args.normal_eq_reg,
            "max_its": args.max_its,
            "relnorm_cutoff": args.relnorm_cutoff,
            "min_delta": args.min_delta,
            "coefficient_storage": "direct_solver_qN_only",
            "evaluation_points_excluded_from_lhs": EVALUATION_MU.tolist(),
            "parameter_validation_points_excluded_from_lhs": PARAMETER_VALIDATION_MU.tolist(),
        }
    )
    _atomic_save(path / "meta.npy", metadata, allow_pickle=True)
    _atomic_write_text(path / "meta.json", json.dumps(metadata, indent=2) + "\n")

    summary = [
        f"dataset_dir: {path}",
        "solve_backend: prom",
        f"subset_level: {level}",
        "candidate_pool: fixed seed 18 interior + 18 margin LHS; no full 9+36 solve",
        f"num_base_traj_copied: {len(baseline)}",
        f"num_interior_lhs_traj: {len(interior)}",
        f"num_exterior_lhs_traj: {len(exterior)}",
        f"num_lhs_traj: {len(selected_lhs)}",
        f"num_traj_total: {len(baseline) + len(selected_lhs)}",
        f"selected_interior_source_indices: {selected_interior}",
        f"selected_exterior_source_indices: {selected_exterior}",
    ]
    _atomic_write_text(path / "stage2_nested_enrichment_summary.txt", "\n".join(summary) + "\n")


def _solve_lhs18(
    dataset: Path,
    *,
    base_dataset: Path,
    baseline: np.ndarray,
    interior: np.ndarray,
    exterior: np.ndarray,
    total_modes: int,
    expected_tags: set[str],
    basis: np.ndarray,
    u_ref: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if dataset.exists():
        if args.force:
            shutil.rmtree(dataset)
        else:
            _validate_existing_tags(dataset, expected_tags)
    output_per_mu = dataset / "per_mu"
    output_per_mu.mkdir(parents=True, exist_ok=True)
    copied = np.asarray(_copy_baseline_dataset(base_dataset / "per_mu", output_per_mu, total_modes))
    if copied.shape != baseline.shape or not np.allclose(copied, baseline, rtol=0.0, atol=1e-14):
        raise RuntimeError("Copied baseline parameters do not match the source dataset.")

    vtot = basis[:, :total_modes]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != vtot.shape[0]:
        raise RuntimeError("Initial state size does not match the Euclidean basis.")
    time = DT * np.arange(NUM_STEPS + 1, dtype=np.float64)
    lhs = np.vstack((interior, exterior))
    for index, mu in enumerate(lhs, start=1):
        target = output_per_mu / _mu_tag(mu)
        if target.is_dir():
            try:
                saved_mu = np.asarray(np.load(target / "mu.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
                _validate_trajectory(target, total_modes)
                if np.allclose(saved_mu, mu, rtol=0.0, atol=1e-14):
                    print(f"[euclidean-nested-stage2] [{index}/{len(lhs)}] reuse {_mu_tag(mu)}")
                    continue
            except (OSError, ValueError):
                pass
            shutil.rmtree(target)
        elif target.exists():
            raise RuntimeError(f"Expected trajectory directory, found non-directory: {target}")
        print(f"[euclidean-nested-stage2] [{index}/{len(lhs)}] PROM solve {_mu_tag(mu)}")
        rom_snaps, qn, stats = inviscid_burgers_implicit2D_LSPG(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=np.asarray(mu, dtype=np.float64),
            basis=vtot,
            u_ref=u_ref,
            max_its=args.max_its,
            relnorm_cutoff=args.relnorm_cutoff,
            min_delta=args.min_delta,
            linear_solver=args.linear_solver,
            normal_eq_reg=args.normal_eq_reg,
            return_red_coords=True,
        )
        qn = np.asarray(qn, dtype=np.float64)
        stats = np.asarray(stats, dtype=np.float64)
        if np.asarray(rom_snaps).shape != (vtot.shape[0], NUM_STEPS + 1):
            raise RuntimeError(f"Unexpected PROM state shape for {_mu_tag(mu)}.")
        if qn.shape != (total_modes, NUM_STEPS + 1) or not np.all(np.isfinite(qn)):
            raise RuntimeError(f"Invalid PROM coordinates for {_mu_tag(mu)}.")
        # The solver already generated this state from qN; retain only qN so
        # the enrichment data remain compact between solves.
        del rom_snaps
        _atomic_save(target / "mu.npy", np.asarray(mu, dtype=np.float64))
        _atomic_save(target / "t.npy", time)
        _atomic_save(target / "qN.npy", qn)
        _atomic_save(target / "rom_stats.npy", stats)
        _atomic_save(target / "prom_stats.npy", stats)
        _validate_trajectory(target, total_modes)


def _copy_lhs8_from_lhs18(
    dataset: Path,
    source: Path,
    *,
    baseline: np.ndarray,
    interior: np.ndarray,
    exterior: np.ndarray,
    total_modes: int,
    expected_tags: set[str],
    force: bool,
) -> None:
    if dataset.exists():
        if force:
            shutil.rmtree(dataset)
        else:
            _validate_existing_tags(dataset, expected_tags)
    destination = dataset / "per_mu"
    destination.mkdir(parents=True, exist_ok=True)
    for mu in np.vstack((baseline, interior, exterior)):
        tag = _mu_tag(mu)
        source_dir = source / "per_mu" / tag
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing nested LHS18 trajectory: {source_dir}")
        target_dir = destination / tag
        if target_dir.is_dir():
            try:
                _validate_trajectory(target_dir, total_modes)
                print(f"[euclidean-nested-stage2] reuse copied LHS8 trajectory: {tag}")
                continue
            except (OSError, ValueError):
                shutil.rmtree(target_dir)
        elif target_dir.exists():
            raise RuntimeError(f"Expected trajectory directory, found non-directory: {target_dir}")
        shutil.copytree(source_dir, target_dir)
        _validate_trajectory(target_dir, total_modes)


def main() -> None:
    args = _parse_args()
    base_dataset = args.base_dataset_dir.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    basis_path = args.basis_path.expanduser().resolve()
    u_ref_path = args.u_ref_path.expanduser().resolve()
    for required in (base_dataset / "meta.json", base_dataset / "per_mu", basis_path, u_ref_path):
        if not required.exists():
            raise FileNotFoundError(required)
    base_meta, base_meta_path = read_dataset_meta(base_dataset)
    if str(base_meta.get("solve_backend", "")).lower() != "prom":
        raise ValueError("The baseline dataset must use full-residual PROM targets.")
    if int(base_meta.get("total_modes", -1)) != args.total_modes:
        raise ValueError("The baseline dataset total_modes does not match --total-modes.")
    baseline = _baseline_mu_from_dataset(base_dataset / "per_mu", args.total_modes)
    basis, u_ref = _load_basis_and_ref(basis_path, u_ref_path, args.total_modes)
    interior_pool, exterior_pool, labels_pool, expanded_mu1, expanded_mu2, interior_order, exterior_order = _candidate_design(
        baseline, seed=args.lhs_seed, margin_fraction=args.margin_fraction
    )
    level_data = {
        level: _level_selection(interior_pool, exterior_pool, labels_pool, interior_order, exterior_order, level)
        for level in LEVELS
    }

    print(f"[euclidean-nested-stage2] base dataset: {base_dataset}")
    print(f"[euclidean-nested-stage2] results root: {results_root}")
    print("[euclidean-nested-stage2] candidate pool: fixed 18 interior + 18 margin LHS (design only)")
    for level, (interior, exterior, labels, interior_indices, exterior_indices) in level_data.items():
        print(
            f"[euclidean-nested-stage2] {level}: 9 + {len(interior)} interior + {len(exterior)} margin "
            f"= {9 + len(interior) + len(exterior)} trajectories"
        )
        print(f"  interior candidate indices: {interior_indices}")
        print(f"  margin candidate indices:   {exterior_indices} ({labels})")
    if args.plan_only:
        print("[euclidean-nested-stage2] PLAN_ONLY=1; no files were written.")
        return

    lhs18 = _dataset_dir(results_root, "lhs18", args.total_modes)
    lhs8 = _dataset_dir(results_root, "lhs8", args.total_modes)
    interior18, exterior18, labels18, indices_i18, indices_e18 = level_data["lhs18"]
    expected18 = _expected_tags(baseline, interior18, exterior18)
    if not _is_complete(lhs18, total_modes=args.total_modes, expected_tags=expected18, level="lhs18"):
        _solve_lhs18(
            lhs18,
            base_dataset=base_dataset,
            baseline=baseline,
            interior=interior18,
            exterior=exterior18,
            total_modes=args.total_modes,
            expected_tags=expected18,
            basis=basis,
            u_ref=u_ref,
            args=args,
        )
        _write_subset_metadata(
            lhs18,
            base_meta=base_meta,
            base_meta_path=Path(base_meta_path),
            baseline=baseline,
            interior=interior18,
            exterior=exterior18,
            exterior_labels=labels18,
            candidate_interior=interior_pool,
            candidate_exterior=exterior_pool,
            candidate_labels=labels_pool,
            interior_order=interior_order,
            exterior_order=exterior_order,
            selected_interior=indices_i18,
            selected_exterior=indices_e18,
            level="lhs18",
            basis_path=basis_path,
            u_ref_path=u_ref_path,
            expanded_mu1=expanded_mu1,
            expanded_mu2=expanded_mu2,
            args=args,
        )
    else:
        print(f"[euclidean-nested-stage2] reusing complete LHS18 dataset: {lhs18}")

    interior8, exterior8, labels8, indices_i8, indices_e8 = level_data["lhs8"]
    expected8 = _expected_tags(baseline, interior8, exterior8)
    if not _is_complete(lhs8, total_modes=args.total_modes, expected_tags=expected8, level="lhs8"):
        _copy_lhs8_from_lhs18(
            lhs8,
            lhs18,
            baseline=baseline,
            interior=interior8,
            exterior=exterior8,
            total_modes=args.total_modes,
            expected_tags=expected8,
            force=args.force,
        )
        _write_subset_metadata(
            lhs8,
            base_meta=base_meta,
            base_meta_path=Path(base_meta_path),
            baseline=baseline,
            interior=interior8,
            exterior=exterior8,
            exterior_labels=labels8,
            candidate_interior=interior_pool,
            candidate_exterior=exterior_pool,
            candidate_labels=labels_pool,
            interior_order=interior_order,
            exterior_order=exterior_order,
            selected_interior=indices_i8,
            selected_exterior=indices_e8,
            level="lhs8",
            basis_path=basis_path,
            u_ref_path=u_ref_path,
            expanded_mu1=expanded_mu1,
            expanded_mu2=expanded_mu2,
            args=args,
        )
    else:
        print(f"[euclidean-nested-stage2] reusing complete LHS8 dataset: {lhs8}")

    if not _is_complete(lhs18, total_modes=args.total_modes, expected_tags=expected18, level="lhs18"):
        raise RuntimeError("LHS18 dataset failed validation after construction.")
    if not _is_complete(lhs8, total_modes=args.total_modes, expected_tags=expected8, level="lhs8"):
        raise RuntimeError("LHS8 dataset failed validation after construction.")
    print("[euclidean-nested-stage2] done. New full PROM solves: 18; LHS8 copied from the LHS18 prefix.")


if __name__ == "__main__":
    main()
