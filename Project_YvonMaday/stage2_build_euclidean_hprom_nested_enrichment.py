#!/usr/bin/env python3
"""Build nested 9+8, 9+12, and 9+18 linear-HPROM target datasets.

Only the eighteen points in the largest level are solved.  The smaller levels
copy strict prefixes of the same frozen design.  Every new trajectory uses the
baseline linear HPROM and its unchanged baseline ECSW rule.
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
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
from stage2_build_paper_extended_enrichment_qn_dataset import (
    _atomic_save,
    _atomic_write_text,
    _baseline_mu_from_dataset,
    _mu_tag,
    _sha256,
    _storage_key,
    _validate_trajectory,
    _write_manifest,
)
from stage2_build_prom_extended_enrichment_qn_dataset import (
    _copy_baseline_dataset,
    _load_basis_and_ref,
)
from stage3_dataset_utils import read_dataset_meta


LEVELS = ("lhs8", "lhs12", "lhs18")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--basis-path", type=Path, required=True)
    parser.add_argument("--u-ref-path", type=Path, required=True)
    parser.add_argument("--ecsw-weights-path", type=Path, required=True)
    parser.add_argument(
        "--design-path",
        type=Path,
        default=THIS_DIR / "euclidean_nested_enrichment_design.json",
    )
    parser.add_argument("--total-modes", type=int, default=151)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1e-12)
    parser.add_argument("--force", action="store_true", help="Replace the three named HPROM datasets.")
    parser.add_argument("--plan-only", action="store_true", help="Validate and print the design without writing files.")
    return parser.parse_args()


def _dataset_dir(results_root: Path, level: str, total_modes: int) -> Path:
    return (
        results_root
        / f"euclidean_hprom_enrichment_{level}"
        / "Stage2"
        / f"prom_coeff_dataset_ntot{total_modes}_enriched_{level}"
    )


def _load_design(path: Path) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, list[str]]]]:
    design = json.loads(path.read_text(encoding="utf-8"))
    interior_all = np.asarray(design["ordered_interior_mu"], dtype=np.float64)
    margin_all = np.asarray(design["ordered_margin_mu"], dtype=np.float64)
    margin_labels = list(design["ordered_margin_labels"])
    if interior_all.shape != (9, 2) or margin_all.shape != (9, 2) or len(margin_labels) != 9:
        raise ValueError("The frozen design must contain nine ordered interior and nine ordered margin points.")
    if not np.all(np.isfinite(interior_all)) or not np.all(np.isfinite(margin_all)):
        raise ValueError("The frozen design contains non-finite parameters.")

    mu1 = tuple(map(float, design["original_mu1_range"]))
    mu2 = tuple(map(float, design["original_mu2_range"]))
    inside = (
        (interior_all[:, 0] >= mu1[0])
        & (interior_all[:, 0] <= mu1[1])
        & (interior_all[:, 1] >= mu2[0])
        & (interior_all[:, 1] <= mu2[1])
    )
    if not np.all(inside):
        raise ValueError("A point labelled interior lies outside the original parameter rectangle.")
    margin_inside = (
        (margin_all[:, 0] >= mu1[0])
        & (margin_all[:, 0] <= mu1[1])
        & (margin_all[:, 1] >= mu2[0])
        & (margin_all[:, 1] <= mu2[1])
    )
    if np.any(margin_inside):
        raise ValueError("A point labelled margin lies inside the original parameter rectangle.")

    levels: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    previous_tags: set[tuple[float, float]] = set()
    for level in LEVELS:
        counts = design["levels"][level]
        n_interior = int(counts["interior_count"])
        n_margin = int(counts["margin_count"])
        interior = interior_all[:n_interior].copy()
        margin = margin_all[:n_margin].copy()
        labels = margin_labels[:n_margin]
        tags = {_storage_key(mu) for mu in np.vstack((interior, margin))}
        if len(tags) != n_interior + n_margin or not previous_tags.issubset(tags):
            raise ValueError(f"The {level} design is not a strict collision-free nested prefix.")
        previous_tags = tags
        levels[level] = interior, margin, labels
    if [sum(map(len, levels[level][:2])) for level in LEVELS] != [8, 12, 18]:
        raise ValueError("The frozen levels must contain 8, 12, and 18 enrichment points.")
    return design, levels


def _expected_tags(baseline: np.ndarray, interior: np.ndarray, margin: np.ndarray) -> set[str]:
    return {_mu_tag(mu) for mu in np.vstack((baseline, interior, margin))}


def _validate_no_leakage(
    baseline: np.ndarray,
    levels: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    design: dict[str, Any],
) -> None:
    forbidden = np.vstack(
        (
            baseline,
            np.asarray(design["parameter_validation_mu"], dtype=np.float64),
            np.asarray(design["reporting_mu"], dtype=np.float64),
        )
    )
    forbidden_tags = {_storage_key(mu) for mu in forbidden}
    enrichment_tags = {
        _storage_key(mu) for mu in np.vstack((levels["lhs18"][0], levels["lhs18"][1]))
    }
    overlap = sorted(forbidden_tags.intersection(enrichment_tags))
    if overlap:
        raise ValueError(f"Training enrichment overlaps baseline, validation, or reporting points: {overlap}")


def _is_complete(
    path: Path,
    *,
    total_modes: int,
    expected_tags: set[str],
    level: str,
    weights_sha256: str,
) -> bool:
    meta_path = path / "meta.json"
    manifest_path = path / "nested_subset_manifest.json"
    if (
        not meta_path.is_file()
        or not manifest_path.is_file()
        or not (path / "construction_config.json").is_file()
        or not (path / "per_mu").is_dir()
    ):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("solve_backend", "")).lower() != "hprom":
            return False
        if int(meta.get("total_modes", -1)) != total_modes or meta.get("subset_level") != level:
            return False
        if int(meta.get("num_traj", -1)) != len(expected_tags):
            return False
        if meta.get("ecsw_weights_sha256") != weights_sha256:
            return False
        actual = {child.name for child in (path / "per_mu").iterdir() if child.is_dir()}
        if actual != expected_tags:
            return False
        for tag in actual:
            _validate_trajectory(path / "per_mu" / tag, total_modes)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _validate_existing_tags(dataset: Path, expected_tags: set[str]) -> None:
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
            "Use --force only when replacing this named campaign intentionally."
        )


def _solve_lhs18(
    dataset: Path,
    *,
    base_dataset: Path,
    baseline: np.ndarray,
    interior: np.ndarray,
    margin: np.ndarray,
    expected_tags: set[str],
    basis: np.ndarray,
    u_ref: np.ndarray,
    weights: np.ndarray,
    construction_config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if dataset.exists():
        if args.force:
            shutil.rmtree(dataset)
        else:
            _validate_existing_tags(dataset, expected_tags)
    config_path = dataset / "construction_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != construction_config:
            raise RuntimeError(
                f"The partial LHS18 campaign has different provenance: {config_path}. "
                "Use --force only if replacing it is intentional."
            )
    elif (dataset / "per_mu").is_dir():
        baseline_tags = {_mu_tag(mu) for mu in baseline}
        existing = {path.name for path in (dataset / "per_mu").iterdir() if path.is_dir()}
        if existing - baseline_tags:
            raise RuntimeError(
                f"Cannot establish the provenance of partial enrichment trajectories in {dataset}. "
                "Use --force to replace them."
            )
    _atomic_write_text(config_path, json.dumps(construction_config, indent=2) + "\n")
    output_per_mu = dataset / "per_mu"
    output_per_mu.mkdir(parents=True, exist_ok=True)
    copied = np.asarray(_copy_baseline_dataset(base_dataset / "per_mu", output_per_mu, args.total_modes))
    if copied.shape != baseline.shape or not np.allclose(copied, baseline, rtol=0.0, atol=1e-14):
        raise RuntimeError("Copied baseline parameters do not match the source HPROM dataset.")

    vtot = basis[:, : args.total_modes]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    if w0.size != vtot.shape[0]:
        raise RuntimeError("Initial state size does not match the Euclidean basis.")
    time = DT * np.arange(NUM_STEPS + 1, dtype=np.float64)
    enrichment = np.vstack((interior, margin))
    for index, mu in enumerate(enrichment, start=1):
        target = output_per_mu / _mu_tag(mu)
        if target.is_dir():
            try:
                saved_mu = np.asarray(np.load(target / "mu.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
                _validate_trajectory(target, args.total_modes)
                if np.allclose(saved_mu, mu, rtol=0.0, atol=1e-14):
                    print(f"[euclidean-hprom-nested-stage2] [{index}/18] reuse {_mu_tag(mu)}")
                    continue
            except (OSError, ValueError):
                pass
            shutil.rmtree(target)
        elif target.exists():
            raise RuntimeError(f"Expected trajectory directory, found non-directory: {target}")

        print(f"[euclidean-hprom-nested-stage2] [{index}/18] linear HPROM solve {_mu_tag(mu)}")
        qn, stats = inviscid_burgers_implicit2D_LSPG_ecsw(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            weights=weights,
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
        )
        qn = np.asarray(qn, dtype=np.float64)
        stats = np.asarray(stats, dtype=np.float64)
        if qn.shape != (args.total_modes, NUM_STEPS + 1) or not np.all(np.isfinite(qn)):
            raise RuntimeError(f"Invalid linear-HPROM coordinates for {_mu_tag(mu)}: {qn.shape}")
        _atomic_save(target / "mu.npy", np.asarray(mu, dtype=np.float64))
        _atomic_save(target / "t.npy", time)
        _atomic_save(target / "qN.npy", qn)
        _atomic_save(target / "rom_stats.npy", stats)
        _atomic_save(target / "hprom_stats.npy", stats)
        _validate_trajectory(target, args.total_modes)


def _copy_nested(
    dataset: Path,
    source: Path,
    *,
    points: np.ndarray,
    expected_tags: set[str],
    total_modes: int,
    force: bool,
) -> None:
    if dataset.exists():
        if force:
            shutil.rmtree(dataset)
        else:
            _validate_existing_tags(dataset, expected_tags)
    destination = dataset / "per_mu"
    destination.mkdir(parents=True, exist_ok=True)
    source_config = source / "construction_config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing source construction provenance: {source_config}")
    shutil.copy2(source_config, dataset / "construction_config.json")
    for mu in points:
        tag = _mu_tag(mu)
        source_dir = source / "per_mu" / tag
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing LHS18 source trajectory: {source_dir}")
        target_dir = destination / tag
        if target_dir.is_dir():
            try:
                _validate_trajectory(target_dir, total_modes)
                continue
            except (OSError, ValueError):
                shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        _validate_trajectory(target_dir, total_modes)


def _write_metadata(
    path: Path,
    *,
    base_meta: dict[str, Any],
    base_meta_path: Path,
    baseline: np.ndarray,
    interior: np.ndarray,
    margin: np.ndarray,
    labels: list[str],
    design: dict[str, Any],
    design_path: Path,
    level: str,
    basis_path: Path,
    u_ref_path: Path,
    weights_path: Path,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> None:
    enrichment = np.vstack((interior, margin))
    all_mu = np.vstack((baseline, enrichment))
    reporting = np.asarray(design["reporting_mu"], dtype=np.float64)
    _atomic_save(path / "baseline_mu.npy", baseline)
    _atomic_save(path / "interior_lhs_mu.npy", interior)
    _atomic_save(path / "exterior_lhs_mu.npy", margin)
    _atomic_save(path / "lhs_mu.npy", enrichment)
    _atomic_save(path / "evaluation_mu.npy", reporting)
    _atomic_write_text(path / "exterior_region_labels.json", json.dumps(labels, indent=2) + "\n")
    _write_manifest(path / "parameter_manifest.csv", baseline, interior, margin, labels, reporting)

    manifest = {
        "design_id": design["design_id"],
        "design_path": str(design_path),
        "design_sha256": _sha256(design_path),
        "subset_level": level,
        "selection": "ordered prefixes of the frozen design",
        "selected_interior_mu": interior.tolist(),
        "selected_margin_mu": margin.tolist(),
        "selected_margin_labels": labels,
    }
    _atomic_write_text(path / "nested_subset_manifest.json", json.dumps(manifest, indent=2) + "\n")

    metadata = dict(base_meta)
    metadata.update(
        {
            "solve_backend": "hprom",
            "is_enrichment_dataset": True,
            "is_nested_subset_dataset": True,
            "enrichment_protocol": f"euclidean_linear_hprom_baseline_9_plus_nested_{len(enrichment)}",
            "subset_level": level,
            "design_id": design["design_id"],
            "design_path": str(design_path),
            "design_sha256": _sha256(design_path),
            "source_enrichment_dataset": (
                "not applicable; the 9+18 linear-HPROM trajectories are solved directly"
                if level == "lhs18"
                else str(_dataset_dir(args.results_root.resolve(), "lhs18", args.total_modes))
            ),
            "base_dataset_dir": str(args.base_dataset_dir.resolve()),
            "base_dataset_meta_path": str(base_meta_path),
            "num_traj": int(len(all_mu)),
            "num_base_traj_copied": int(len(baseline)),
            "num_lhs_traj": int(len(enrichment)),
            "num_interior_lhs_traj": int(len(interior)),
            "num_exterior_lhs_traj": int(len(margin)),
            "mu_list": all_mu.tolist(),
            "lhs_seed": int(design["lhs_seed"]),
            "margin_fraction": float(design["margin_fraction"]),
            "original_mu1_range": design["original_mu1_range"],
            "original_mu2_range": design["original_mu2_range"],
            "expanded_mu1_range": design["expanded_mu1_range"],
            "expanded_mu2_range": design["expanded_mu2_range"],
            "exterior_region_labels": labels,
            "basis_path": str(basis_path),
            "basis_sha256": _sha256(basis_path),
            "u_ref_path": str(u_ref_path),
            "u_ref_sha256": _sha256(u_ref_path),
            "ecsw_weights_path": str(weights_path),
            "ecsw_weights_sha256": _sha256(weights_path),
            "ecsw_weights_source": "strict_reference_to_baseline_linear_hprom",
            "ecsw_weights_copied": False,
            "ecsw_weights_rebuilt": False,
            "n_ecsw_elements": int(np.count_nonzero(weights > 0.0)),
            "linear_solver": args.linear_solver,
            "normal_eq_reg": float(args.normal_eq_reg),
            "max_its": int(args.max_its),
            "relnorm_cutoff": float(args.relnorm_cutoff),
            "min_delta": float(args.min_delta),
            "coefficient_storage": "direct_solver_qN_only",
            "evaluation_points_excluded_from_lhs": design["reporting_mu"],
            "parameter_validation_points_excluded_from_lhs": design["parameter_validation_mu"],
        }
    )
    _atomic_save(path / "meta.npy", metadata, allow_pickle=True)
    _atomic_write_text(path / "meta.json", json.dumps(metadata, indent=2) + "\n")
    summary = [
        f"dataset_dir: {path}",
        "solve_backend: hprom",
        f"subset_level: {level}",
        f"design_id: {design['design_id']}",
        f"num_base_traj_copied: {len(baseline)}",
        f"num_interior_lhs_traj: {len(interior)}",
        f"num_margin_lhs_traj: {len(margin)}",
        f"num_traj_total: {len(all_mu)}",
        f"ecsw_weights_path: {weights_path}",
        f"ecsw_weights_sha256: {_sha256(weights_path)}",
        "ecsw_policy: frozen baseline linear-HPROM rule; never rebuilt for target generation",
    ]
    _atomic_write_text(path / "stage2_nested_enrichment_summary.txt", "\n".join(summary) + "\n")


def main() -> None:
    args = _parse_args()
    args.base_dataset_dir = args.base_dataset_dir.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    basis_path = args.basis_path.expanduser().resolve()
    u_ref_path = args.u_ref_path.expanduser().resolve()
    weights_path = args.ecsw_weights_path.expanduser().resolve()
    design_path = args.design_path.expanduser().resolve()
    for required in (
        args.base_dataset_dir / "meta.json",
        args.base_dataset_dir / "per_mu",
        basis_path,
        u_ref_path,
        weights_path,
        design_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    base_meta, base_meta_path = read_dataset_meta(args.base_dataset_dir)
    if str(base_meta.get("solve_backend", "")).lower() != "hprom":
        raise ValueError("The baseline dataset must contain linear-HPROM targets.")
    if int(base_meta.get("total_modes", -1)) != args.total_modes:
        raise ValueError("The baseline dataset total_modes does not match --total-modes.")
    baseline = _baseline_mu_from_dataset(args.base_dataset_dir / "per_mu", args.total_modes)
    if baseline.shape != (9, 2):
        raise ValueError(f"Expected exactly nine baseline trajectories, found {baseline.shape[0]}.")
    basis, u_ref = _load_basis_and_ref(basis_path, u_ref_path, args.total_modes)
    weights = np.asarray(np.load(weights_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    expected_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    if weights.size != expected_cells or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"Invalid frozen ECSW rule: shape={weights.shape}, expected={expected_cells} nonnegative cells.")

    design, levels = _load_design(design_path)
    _validate_no_leakage(baseline, levels, design)
    weights_sha256 = _sha256(weights_path)
    construction_config = {
        "solve_backend": "hprom",
        "design_id": design["design_id"],
        "design_sha256": _sha256(design_path),
        "base_dataset_meta_sha256": _sha256(args.base_dataset_dir / "meta.json"),
        "basis_sha256": _sha256(basis_path),
        "u_ref_sha256": _sha256(u_ref_path),
        "ecsw_weights_sha256": weights_sha256,
        "ecsw_policy": "frozen baseline linear-HPROM rule",
        "total_modes": args.total_modes,
        "max_its": args.max_its,
        "relnorm_cutoff": args.relnorm_cutoff,
        "min_delta": args.min_delta,
        "linear_solver": args.linear_solver,
        "normal_eq_reg": args.normal_eq_reg,
        "dt": float(DT),
        "num_steps": int(NUM_STEPS),
    }
    print(f"[euclidean-hprom-nested-stage2] baseline: {args.base_dataset_dir}")
    print(f"[euclidean-hprom-nested-stage2] frozen design: {design_path} ({design['design_id']})")
    print(f"[euclidean-hprom-nested-stage2] frozen linear ECSW: {weights_path}")
    print(f"[euclidean-hprom-nested-stage2] frozen linear ECSW sha256: {weights_sha256}")
    for level in LEVELS:
        interior, margin, labels = levels[level]
        print(
            f"[euclidean-hprom-nested-stage2] {level}: 9 + {len(interior)} interior + "
            f"{len(margin)} margin = {9 + len(interior) + len(margin)} trajectories"
        )
        for mu, role in [(mu, "interior") for mu in interior] + list(zip(margin, labels)):
            print(f"  {role:13s} {_mu_tag(mu)}")
    if args.plan_only:
        print("[euclidean-hprom-nested-stage2] PLAN_ONLY=1; no files were written.")
        return

    lhs18 = _dataset_dir(args.results_root, "lhs18", args.total_modes)
    interior18, margin18, labels18 = levels["lhs18"]
    expected18 = _expected_tags(baseline, interior18, margin18)
    if not _is_complete(
        lhs18,
        total_modes=args.total_modes,
        expected_tags=expected18,
        level="lhs18",
        weights_sha256=weights_sha256,
    ):
        _solve_lhs18(
            lhs18,
            base_dataset=args.base_dataset_dir,
            baseline=baseline,
            interior=interior18,
            margin=margin18,
            expected_tags=expected18,
            basis=basis,
            u_ref=u_ref,
            weights=weights,
            construction_config=construction_config,
            args=args,
        )
        _write_metadata(
            lhs18,
            base_meta=base_meta,
            base_meta_path=Path(base_meta_path),
            baseline=baseline,
            interior=interior18,
            margin=margin18,
            labels=labels18,
            design=design,
            design_path=design_path,
            level="lhs18",
            basis_path=basis_path,
            u_ref_path=u_ref_path,
            weights_path=weights_path,
            weights=weights,
            args=args,
        )
    else:
        print(f"[euclidean-hprom-nested-stage2] reusing complete LHS18 dataset: {lhs18}")

    for level in ("lhs8", "lhs12"):
        interior, margin, labels = levels[level]
        dataset = _dataset_dir(args.results_root, level, args.total_modes)
        expected = _expected_tags(baseline, interior, margin)
        if not _is_complete(
            dataset,
            total_modes=args.total_modes,
            expected_tags=expected,
            level=level,
            weights_sha256=weights_sha256,
        ):
            _copy_nested(
                dataset,
                lhs18,
                points=np.vstack((baseline, interior, margin)),
                expected_tags=expected,
                total_modes=args.total_modes,
                force=args.force,
            )
            _write_metadata(
                dataset,
                base_meta=base_meta,
                base_meta_path=Path(base_meta_path),
                baseline=baseline,
                interior=interior,
                margin=margin,
                labels=labels,
                design=design,
                design_path=design_path,
                level=level,
                basis_path=basis_path,
                u_ref_path=u_ref_path,
                weights_path=weights_path,
                weights=weights,
                args=args,
            )
        else:
            print(f"[euclidean-hprom-nested-stage2] reusing complete {level.upper()} dataset: {dataset}")

    for level in LEVELS:
        interior, margin, _ = levels[level]
        dataset = _dataset_dir(args.results_root, level, args.total_modes)
        expected = _expected_tags(baseline, interior, margin)
        if not _is_complete(
            dataset,
            total_modes=args.total_modes,
            expected_tags=expected,
            level=level,
            weights_sha256=weights_sha256,
        ):
            raise RuntimeError(f"{level.upper()} failed final validation: {dataset}")
    print("[euclidean-hprom-nested-stage2] complete; exactly 18 new linear-HPROM solves supply all levels.")


if __name__ == "__main__":
    main()
