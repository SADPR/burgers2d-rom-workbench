#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the paper enrichment dataset with the fixed linear HPROM.

The existing baseline qN trajectories are copied into a separate dataset and
20 new LHS trajectories are solved with the exact baseline linear ECSW rule.
This script never computes or copies ECSW weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from burgers.config import DT, GRID_X, GRID_Y, MU1_RANGE, MU2_RANGE, NUM_STEPS, W0
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
from stage3_dataset_utils import read_dataset_meta


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mu_tag(mu) -> str:
    return f"mu1_{float(mu[0]):.3f}_mu2_{float(mu[1]):.4f}"


def _storage_key(mu) -> tuple[float, float]:
    return round(float(mu[0]), 3), round(float(mu[1]), 4)


def _atomic_save(path: Path, value, *, allow_pickle: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=allow_pickle)
    os.replace(temp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _load_mu_dirs(per_mu_dir: Path) -> list[tuple[Path, np.ndarray]]:
    records = []
    if not per_mu_dir.is_dir():
        return records
    for mu_dir in sorted(path for path in per_mu_dir.iterdir() if path.is_dir()):
        mu_path = mu_dir / "mu.npy"
        if not mu_path.is_file():
            continue
        mu = np.asarray(np.load(mu_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if mu.size != 2:
            raise ValueError(f"Invalid parameter vector in {mu_path}: shape={mu.shape}")
        records.append((mu_dir, mu))
    return records


def _validate_trajectory(mu_dir: Path, total_modes: int) -> None:
    mu = np.asarray(np.load(mu_dir / "mu.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
    time = np.asarray(np.load(mu_dir / "t.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
    qn = np.asarray(np.load(mu_dir / "qN.npy", allow_pickle=False), dtype=np.float64)
    expected_shape = (total_modes, NUM_STEPS + 1)
    if mu.size != 2:
        raise ValueError(f"{mu_dir}: expected two parameters, got {mu.shape}")
    if time.size != NUM_STEPS + 1:
        raise ValueError(f"{mu_dir}: expected {NUM_STEPS + 1} times, got {time.size}")
    if qn.shape != expected_shape:
        raise ValueError(f"{mu_dir}: expected qN shape {expected_shape}, got {qn.shape}")
    if not np.all(np.isfinite(qn)):
        raise ValueError(f"{mu_dir}: qN contains non-finite values")


def _copy_baseline_dataset(base_per_mu: Path, output_per_mu: Path, total_modes: int) -> list[list[float]]:
    keep = ("mu.npy", "t.npy", "qN.npy", "rom_stats.npy", "hprom_stats.npy")
    baseline_mu = []
    for source_dir, mu in _load_mu_dirs(base_per_mu):
        _validate_trajectory(source_dir, total_modes)
        target_dir = output_per_mu / source_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in keep:
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, target_dir / name)
        _validate_trajectory(target_dir, total_modes)
        baseline_mu.append([float(mu[0]), float(mu[1])])
    return baseline_mu


def _lhs_2d(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((n_samples, 2), dtype=np.float64)
    for column, bounds in enumerate((MU1_RANGE, MU2_RANGE)):
        points = (np.arange(n_samples, dtype=np.float64) + rng.random(n_samples)) / n_samples
        rng.shuffle(points)
        result[:, column] = float(bounds[0]) + points * (float(bounds[1]) - float(bounds[0]))
    return result


def _generate_lhs(n_samples: int, seed: int, forbidden) -> np.ndarray:
    selected = []
    used = {_storage_key(mu) for mu in forbidden}
    attempt = 0
    while len(selected) < n_samples:
        if attempt >= 100:
            raise RuntimeError("Could not generate a collision-free LHS set.")
        candidates = _lhs_2d(max(2 * (n_samples - len(selected)), n_samples), seed + attempt)
        for mu in candidates:
            key = _storage_key(mu)
            if key in used:
                continue
            used.add(key)
            selected.append([float(mu[0]), float(mu[1])])
            if len(selected) == n_samples:
                break
        attempt += 1
    return np.asarray(selected, dtype=np.float64)


def _load_or_create_lhs(
    lhs_path: Path,
    n_samples: int,
    seed: int,
    forbidden,
    regenerate: bool,
) -> np.ndarray:
    if lhs_path.is_file() and not regenerate:
        lhs = np.asarray(np.load(lhs_path, allow_pickle=False), dtype=np.float64)
        if lhs.shape != (n_samples, 2):
            raise ValueError(
                f"Existing {lhs_path} has shape {lhs.shape}; expected {(n_samples, 2)}. "
                "Use --regenerate-lhs only if a new campaign is intended."
            )
    else:
        lhs = _generate_lhs(n_samples, seed, forbidden)
        _atomic_save(lhs_path, lhs)

    forbidden_keys = {_storage_key(mu) for mu in forbidden}
    lhs_keys = [_storage_key(mu) for mu in lhs]
    if len(set(lhs_keys)) != len(lhs_keys):
        raise ValueError("The LHS set contains duplicate output parameter tags.")
    overlap = sorted(set(lhs_keys).intersection(forbidden_keys))
    if overlap:
        raise ValueError(f"The LHS set overlaps baseline/evaluation points: {overlap}")
    return lhs


def _plot_sampling(
    path: Path,
    baseline_mu: np.ndarray,
    lhs_mu: np.ndarray,
    evaluation_mu: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.scatter(
        baseline_mu[:, 0],
        baseline_mu[:, 1],
        s=72,
        c="black",
        marker="o",
        label="Baseline training (9)",
        zorder=3,
    )
    ax.scatter(
        lhs_mu[:, 0],
        lhs_mu[:, 1],
        s=76,
        c="#1f77b4",
        marker="x",
        linewidths=1.8,
        label="HPROM enrichment LHS (20)",
        zorder=4,
    )
    if evaluation_mu.size:
        ax.scatter(
            evaluation_mu[:, 0],
            evaluation_mu[:, 1],
            s=130,
            facecolors="none",
            edgecolors="#d62728",
            marker="*",
            linewidths=1.4,
            label="Evaluation points",
            zorder=5,
        )
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Baseline and HPROM enrichment parameter samples")
    ax.set_xlim(float(MU1_RANGE[0]), float(MU1_RANGE[1]))
    ax.set_ylim(float(MU2_RANGE[0]), float(MU2_RANGE[1]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_manifest(
    path: Path,
    baseline_mu: np.ndarray,
    lhs_mu: np.ndarray,
    evaluation_mu: np.ndarray,
) -> None:
    evaluation_keys = {_storage_key(mu) for mu in evaluation_mu}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("role", "mu1", "mu2", "directory_tag"))
        for mu in baseline_mu:
            role = "baseline_verification" if _storage_key(mu) in evaluation_keys else "baseline_training"
            writer.writerow((role, f"{mu[0]:.16g}", f"{mu[1]:.16g}", _mu_tag(mu)))
        for mu in lhs_mu:
            writer.writerow(("lhs_enrichment", f"{mu[0]:.16g}", f"{mu[1]:.16g}", _mu_tag(mu)))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basis-path", required=True)
    parser.add_argument("--u-ref-path", required=True)
    parser.add_argument("--ecsw-weights-path", required=True)
    parser.add_argument("--total-modes", type=int, default=151)
    parser.add_argument("--lhs-samples", type=int, default=20)
    parser.add_argument("--lhs-seed", type=int, default=42)
    parser.add_argument("--exclude-mu", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--regenerate-lhs", action="store_true")
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
    ecsw_path = Path(args.ecsw_weights_path).expanduser().resolve()
    base_per_mu = base_dataset / "per_mu"
    output_per_mu = output_dir / "per_mu"

    for path in (base_dataset, base_per_mu, basis_path, u_ref_path, ecsw_path):
        if not path.exists():
            raise FileNotFoundError(path)

    base_meta, base_meta_path = read_dataset_meta(str(base_dataset))
    if str(base_meta.get("solve_backend", "")).lower() != "hprom":
        raise ValueError("The baseline dataset must have solve_backend=hprom.")
    if int(base_meta.get("total_modes", -1)) != args.total_modes:
        raise ValueError("The baseline dataset total_modes does not match --total-modes.")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    u_ref = np.asarray(np.load(u_ref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    weights = np.asarray(np.load(ecsw_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    expected_cells = (GRID_X.size - 1) * (GRID_Y.size - 1)
    if basis.ndim != 2 or basis.shape[1] < args.total_modes:
        raise ValueError(f"Invalid basis shape {basis.shape} for n_tot={args.total_modes}.")
    if u_ref.size != basis.shape[0]:
        raise ValueError(f"u_ref size {u_ref.size} does not match basis rows {basis.shape[0]}.")
    if weights.size != expected_cells:
        raise ValueError(f"ECSW size {weights.size} does not match cell count {expected_cells}.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("The fixed ECSW weights must be finite and nonnegative.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_per_mu.mkdir(parents=True, exist_ok=True)

    baseline_mu = np.asarray(
        _copy_baseline_dataset(base_per_mu, output_per_mu, args.total_modes),
        dtype=np.float64,
    )
    if baseline_mu.shape != (9, 2):
        raise ValueError(f"Expected exactly 9 baseline trajectories, got {baseline_mu.shape[0]}.")

    evaluation_mu = np.asarray(args.exclude_mu, dtype=np.float64).reshape(-1, 2)
    forbidden = np.vstack((baseline_mu, evaluation_mu)) if evaluation_mu.size else baseline_mu
    lhs_path = output_dir / "lhs_mu.npy"
    lhs_mu = _load_or_create_lhs(
        lhs_path,
        args.lhs_samples,
        args.lhs_seed,
        forbidden,
        args.regenerate_lhs,
    )

    _atomic_save(output_dir / "baseline_mu.npy", baseline_mu)
    _write_manifest(output_dir / "parameter_manifest.csv", baseline_mu, lhs_mu, evaluation_mu)
    _plot_sampling(
        output_dir / "stage2_sampling_points.png",
        baseline_mu,
        lhs_mu,
        evaluation_mu,
    )

    vtot = basis[:, : args.total_modes]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    t_vec = DT * np.arange(NUM_STEPS + 1, dtype=np.float64)
    if w0.size != vtot.shape[0]:
        raise ValueError(f"W0 size {w0.size} does not match basis rows {vtot.shape[0]}.")

    print(f"[enrichment-stage2] base_dataset={base_dataset}")
    print(f"[enrichment-stage2] output_dataset={output_dir}")
    print(f"[enrichment-stage2] basis={basis_path}")
    print(f"[enrichment-stage2] u_ref={u_ref_path}")
    print(f"[enrichment-stage2] fixed_linear_ecsw={ecsw_path}")
    print(f"[enrichment-stage2] fixed_linear_ecsw_sha256={_sha256(ecsw_path)}")
    print("[enrichment-stage2] ECSW policy=strict reuse; no ECSW build or copy")

    for index, mu in enumerate(lhs_mu, start=1):
        mu_dir = output_per_mu / _mu_tag(mu)
        complete = all(
            (mu_dir / name).is_file()
            for name in ("mu.npy", "t.npy", "qN.npy", "rom_stats.npy", "hprom_stats.npy")
        )
        if complete:
            try:
                saved_mu = np.asarray(np.load(mu_dir / "mu.npy", allow_pickle=False)).reshape(-1)
                _validate_trajectory(mu_dir, args.total_modes)
                if np.allclose(saved_mu, mu, rtol=0.0, atol=1e-14):
                    print(f"[enrichment-stage2] [{index}/{args.lhs_samples}] reuse {_mu_tag(mu)}")
                    continue
            except (OSError, ValueError):
                pass

        print(f"[enrichment-stage2] [{index}/{args.lhs_samples}] solve {_mu_tag(mu)}")
        qn, stats = inviscid_burgers_implicit2D_LSPG_ecsw(
            grid_x=GRID_X,
            grid_y=GRID_Y,
            weights=weights,
            w0=w0,
            dt=DT,
            num_steps=NUM_STEPS,
            mu=mu,
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
            raise ValueError(f"Invalid solver qN for {_mu_tag(mu)}: shape={qn.shape}")
        _atomic_save(mu_dir / "mu.npy", np.asarray(mu, dtype=np.float64))
        _atomic_save(mu_dir / "t.npy", t_vec)
        _atomic_save(mu_dir / "qN.npy", qn)
        _atomic_save(mu_dir / "rom_stats.npy", stats)
        _atomic_save(mu_dir / "hprom_stats.npy", stats)
        _validate_trajectory(mu_dir, args.total_modes)

    records = _load_mu_dirs(output_per_mu)
    if len(records) != 9 + args.lhs_samples:
        raise ValueError(
            f"Expected {9 + args.lhs_samples} trajectories, found {len(records)} in {output_per_mu}."
        )
    for mu_dir, _ in records:
        _validate_trajectory(mu_dir, args.total_modes)

    ecsw_hash = _sha256(ecsw_path)
    metadata = {
        "solve_backend": "hprom",
        "is_enrichment_dataset": True,
        "enrichment_protocol": "baseline_9_plus_lhs_20_fixed_linear_hprom",
        "total_modes": int(args.total_modes),
        "n_available_modes": int(basis.shape[1]),
        "coefficient_storage": "direct_solver_qN_only",
        "dt": float(DT),
        "num_steps": int(NUM_STEPS),
        "num_traj": int(len(records)),
        "num_base_traj_copied": 9,
        "num_lhs_traj": int(args.lhs_samples),
        "lhs_seed": int(args.lhs_seed),
        "basis_path": str(basis_path),
        "basis_sha256": _sha256(basis_path),
        "u_ref_path": str(u_ref_path),
        "u_ref_sha256": _sha256(u_ref_path),
        "base_dataset_dir": str(base_dataset),
        "base_dataset_meta_path": str(base_meta_path),
        "ecsw_weights_path": str(ecsw_path),
        "ecsw_weights_sha256": ecsw_hash,
        "ecsw_weights_source": "strict_reference_to_baseline_linear_hprom",
        "ecsw_weights_copied": False,
        "ecsw_weights_rebuilt": False,
        "n_ecsw_elements": int(np.count_nonzero(weights > 0.0)),
        "linear_solver": args.linear_solver,
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
        "solve_backend: hprom",
        f"total_modes: {args.total_modes}",
        "coefficient_storage: direct_solver_qN_only",
        "num_base_traj_copied: 9",
        f"num_lhs_traj: {args.lhs_samples}",
        f"num_traj_total: {len(records)}",
        f"lhs_seed: {args.lhs_seed}",
        f"basis_path: {basis_path}",
        f"u_ref_path: {u_ref_path}",
        f"ecsw_weights_path: {ecsw_path}",
        f"ecsw_weights_sha256: {ecsw_hash}",
        "ecsw_weights_source: strict_reference_to_baseline_linear_hprom",
        "ecsw_weights_copied: False",
        "ecsw_weights_rebuilt: False",
        f"n_ecsw_elements: {np.count_nonzero(weights > 0.0)}",
        "save_rom_snaps: False",
    ]
    _atomic_write_text(output_dir / "stage2_enrichment_summary.txt", "\n".join(summary_lines) + "\n")
    print(f"[enrichment-stage2] complete: {len(records)} trajectories")
    print(f"[enrichment-stage2] summary={output_dir / 'stage2_enrichment_summary.txt'}")


if __name__ == "__main__":
    main()
