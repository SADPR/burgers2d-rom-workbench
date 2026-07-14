#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build an extended paper enrichment dataset with the fixed linear HPROM.

The baseline qN trajectories are copied into a separate dataset and additional
LHS trajectories are solved with the exact baseline linear ECSW rule.  The
sampling design intentionally separates points inside the original parameter
box from points in an expanded margin around it.  This script never computes or
copies ECSW weights.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from burgers.config import DT, GRID_X, GRID_Y, MU1_RANGE, MU2_RANGE, NUM_STEPS, W0
from burgers.linear_manifold import inviscid_burgers_implicit2D_LSPG_ecsw
from stage3_dataset_utils import read_dataset_meta

ORIGINAL_MU1_RANGE = (float(MU1_RANGE[0]), float(MU1_RANGE[1]))
ORIGINAL_MU2_RANGE = (float(MU2_RANGE[0]), float(MU2_RANGE[1]))

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


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


def _expanded_range(bounds: tuple[float, float], margin_fraction: float) -> tuple[float, float]:
    lo, hi = map(float, bounds)
    if hi <= lo:
        raise ValueError(f"Invalid range {bounds}")
    margin = float(margin_fraction) * (hi - lo)
    return lo - margin, hi + margin


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


def _baseline_mu_from_dataset(base_per_mu: Path, total_modes: int) -> np.ndarray:
    values = []
    for source_dir, mu in _load_mu_dirs(base_per_mu):
        _validate_trajectory(source_dir, total_modes)
        values.append([float(mu[0]), float(mu[1])])
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (9, 2):
        raise ValueError(f"Expected exactly 9 baseline trajectories, got {result.shape[0]}.")
    return result


def _lhs_2d(n_samples: int, seed: int, mu1_range, mu2_range) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((n_samples, 2), dtype=np.float64)
    for column, bounds in enumerate((mu1_range, mu2_range)):
        points = (np.arange(n_samples, dtype=np.float64) + rng.random(n_samples)) / n_samples
        rng.shuffle(points)
        result[:, column] = float(bounds[0]) + points * (float(bounds[1]) - float(bounds[0]))
    return result


def _generate_unique_lhs(
    *,
    n_samples: int,
    seed: int,
    mu1_range: tuple[float, float],
    mu2_range: tuple[float, float],
    used: set[tuple[float, float]],
    label: str,
) -> np.ndarray:
    selected = []
    attempt = 0
    while len(selected) < n_samples:
        if attempt >= 100:
            raise RuntimeError(f"Could not generate a collision-free LHS set for {label}.")
        n_batch = max(2 * (n_samples - len(selected)), n_samples)
        candidates = _lhs_2d(n_batch, seed + 7919 * attempt, mu1_range, mu2_range)
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


def _exterior_split(n_samples: int) -> list[tuple[str, int]]:
    if n_samples == 18:
        return [("left_margin", 5), ("right_margin", 5), ("top_margin", 4), ("bottom_margin", 4)]
    names = ["left_margin", "right_margin", "top_margin", "bottom_margin"]
    base = n_samples // 4
    rem = n_samples % 4
    return [(name, base + (1 if i < rem else 0)) for i, name in enumerate(names)]


def _generate_extended_design(
    *,
    interior_samples: int,
    exterior_samples: int,
    seed: int,
    margin_fraction: float,
    forbidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[float, float], tuple[float, float]]:
    if margin_fraction <= 0.0:
        raise ValueError("margin_fraction must be positive for an extended enrichment campaign.")
    expanded_mu1 = _expanded_range(ORIGINAL_MU1_RANGE, margin_fraction)
    expanded_mu2 = _expanded_range(ORIGINAL_MU2_RANGE, margin_fraction)
    used = {_storage_key(mu) for mu in np.asarray(forbidden, dtype=np.float64).reshape(-1, 2)}

    interior = _generate_unique_lhs(
        n_samples=interior_samples,
        seed=seed,
        mu1_range=ORIGINAL_MU1_RANGE,
        mu2_range=ORIGINAL_MU2_RANGE,
        used=used,
        label="interior",
    )

    exterior_parts = []
    exterior_regions = []
    for offset, (region, count) in enumerate(_exterior_split(exterior_samples)):
        if count <= 0:
            continue
        if region == "left_margin":
            mu1_range = (expanded_mu1[0], ORIGINAL_MU1_RANGE[0])
            mu2_range = expanded_mu2
        elif region == "right_margin":
            mu1_range = (ORIGINAL_MU1_RANGE[1], expanded_mu1[1])
            mu2_range = expanded_mu2
        elif region == "top_margin":
            mu1_range = ORIGINAL_MU1_RANGE
            mu2_range = (ORIGINAL_MU2_RANGE[1], expanded_mu2[1])
        elif region == "bottom_margin":
            mu1_range = ORIGINAL_MU1_RANGE
            mu2_range = (expanded_mu2[0], ORIGINAL_MU2_RANGE[0])
        else:
            raise ValueError(region)
        part = _generate_unique_lhs(
            n_samples=count,
            seed=seed + 1000 + 137 * offset,
            mu1_range=mu1_range,
            mu2_range=mu2_range,
            used=used,
            label=region,
        )
        exterior_parts.append(part)
        exterior_regions.extend([region] * count)

    exterior = np.vstack(exterior_parts) if exterior_parts else np.zeros((0, 2), dtype=np.float64)
    return interior, exterior, exterior_regions, expanded_mu1, expanded_mu2


def _is_inside_box(mu: np.ndarray, mu1_range, mu2_range, *, atol: float = 1e-14) -> bool:
    return (
        float(mu1_range[0]) - atol <= float(mu[0]) <= float(mu1_range[1]) + atol
        and float(mu2_range[0]) - atol <= float(mu[1]) <= float(mu2_range[1]) + atol
    )


def _validate_design(
    *,
    interior_mu: np.ndarray,
    exterior_mu: np.ndarray,
    evaluation_mu: np.ndarray,
    baseline_mu: np.ndarray,
    expanded_mu1: tuple[float, float],
    expanded_mu2: tuple[float, float],
) -> None:
    all_mu = np.vstack((interior_mu, exterior_mu)) if exterior_mu.size else interior_mu.copy()
    keys = [_storage_key(mu) for mu in all_mu]
    if len(set(keys)) != len(keys):
        raise ValueError("The enrichment design contains duplicate output parameter tags.")
    forbidden_keys = {_storage_key(mu) for mu in np.vstack((baseline_mu, evaluation_mu))}
    overlap = sorted(set(keys).intersection(forbidden_keys))
    if overlap:
        raise ValueError(f"The enrichment design overlaps baseline/evaluation points: {overlap}")
    for mu in interior_mu:
        if not _is_inside_box(mu, ORIGINAL_MU1_RANGE, ORIGINAL_MU2_RANGE):
            raise ValueError(f"Interior point outside original domain: {mu}")
    for mu in exterior_mu:
        if _is_inside_box(mu, ORIGINAL_MU1_RANGE, ORIGINAL_MU2_RANGE):
            raise ValueError(f"Exterior point inside original domain: {mu}")
        if not _is_inside_box(mu, expanded_mu1, expanded_mu2):
            raise ValueError(f"Exterior point outside expanded domain: {mu}")


def _load_or_create_design(
    output_dir: Path,
    *,
    interior_samples: int,
    exterior_samples: int,
    seed: int,
    margin_fraction: float,
    baseline_mu: np.ndarray,
    evaluation_mu: np.ndarray,
    regenerate: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[float, float], tuple[float, float]]:
    interior_path = output_dir / "interior_lhs_mu.npy"
    exterior_path = output_dir / "exterior_lhs_mu.npy"
    all_path = output_dir / "lhs_mu.npy"
    labels_path = output_dir / "exterior_region_labels.json"
    expanded_mu1 = _expanded_range(ORIGINAL_MU1_RANGE, margin_fraction)
    expanded_mu2 = _expanded_range(ORIGINAL_MU2_RANGE, margin_fraction)

    if all_path.is_file() and interior_path.is_file() and exterior_path.is_file() and labels_path.is_file() and not regenerate:
        interior_mu = np.asarray(np.load(interior_path, allow_pickle=False), dtype=np.float64)
        exterior_mu = np.asarray(np.load(exterior_path, allow_pickle=False), dtype=np.float64)
        all_mu = np.asarray(np.load(all_path, allow_pickle=False), dtype=np.float64)
        exterior_regions = json.loads(labels_path.read_text(encoding="utf-8"))
        if interior_mu.shape != (interior_samples, 2):
            raise ValueError(f"Existing {interior_path} has shape {interior_mu.shape}; expected {(interior_samples, 2)}.")
        if exterior_mu.shape != (exterior_samples, 2):
            raise ValueError(f"Existing {exterior_path} has shape {exterior_mu.shape}; expected {(exterior_samples, 2)}.")
        if all_mu.shape != (interior_samples + exterior_samples, 2):
            raise ValueError(f"Existing {all_path} has shape {all_mu.shape}; expected {(interior_samples + exterior_samples, 2)}.")
        if not np.allclose(all_mu, np.vstack((interior_mu, exterior_mu)), rtol=0.0, atol=0.0):
            raise ValueError("Existing lhs_mu.npy is inconsistent with interior/exterior split files.")
        if len(exterior_regions) != exterior_samples:
            raise ValueError("Existing exterior region label count does not match exterior_samples.")
    else:
        forbidden = np.vstack((baseline_mu, evaluation_mu)) if evaluation_mu.size else baseline_mu
        interior_mu, exterior_mu, exterior_regions, expanded_mu1, expanded_mu2 = _generate_extended_design(
            interior_samples=interior_samples,
            exterior_samples=exterior_samples,
            seed=seed,
            margin_fraction=margin_fraction,
            forbidden=forbidden,
        )
        _atomic_save(interior_path, interior_mu)
        _atomic_save(exterior_path, exterior_mu)
        _atomic_save(all_path, np.vstack((interior_mu, exterior_mu)))
        _atomic_write_text(labels_path, json.dumps(exterior_regions, indent=2) + "\n")

    _validate_design(
        interior_mu=interior_mu,
        exterior_mu=exterior_mu,
        evaluation_mu=evaluation_mu,
        baseline_mu=baseline_mu,
        expanded_mu1=expanded_mu1,
        expanded_mu2=expanded_mu2,
    )
    return interior_mu, exterior_mu, exterior_regions, expanded_mu1, expanded_mu2


def _plot_parameter_design(
    path: Path,
    *,
    title: str,
    baseline_mu: np.ndarray,
    interior_mu: np.ndarray,
    exterior_mu: np.ndarray,
    evaluation_mu: np.ndarray,
    expanded_mu1: tuple[float, float],
    expanded_mu2: tuple[float, float],
    include_enrichment: bool,
) -> None:
    # The two panels deliberately use identical axes so that the extra
    # coverage is visible without a change of visual scale.
    fig, ax = plt.subplots(figsize=(6.55, 7.15))
    ax.scatter(
        baseline_mu[:, 0],
        baseline_mu[:, 1],
        s=92,
        color="black",
        marker="o",
        zorder=5,
    )
    if include_enrichment:
        ax.scatter(
            interior_mu[:, 0],
            interior_mu[:, 1],
            s=65,
            color="#0072B2",
            edgecolors="white",
            linewidths=0.40,
            zorder=4,
        )
        ax.scatter(
            exterior_mu[:, 0],
            exterior_mu[:, 1],
            s=65,
            color="#009E73",
            edgecolors="white",
            linewidths=0.40,
            zorder=4,
        )
    ax.scatter(
        evaluation_mu[:, 0],
        evaluation_mu[:, 1],
        s=140,
        color="#D62728",
        marker="*",
        edgecolors="white",
        linewidths=0.80,
        zorder=7,
    )
    for label, mu, offset in zip(
        (r"$\mu^{(v)}$", r"$\mu^{(1)}$", r"$\mu^{(2)}$", r"$\mu^{(3)}$"),
        evaluation_mu,
        ((10, -18), (8, 7), (8, 7), (8, -5)),
    ):
        ax.annotate(label, xy=mu, xytext=offset, textcoords="offset points", color="#B22222", fontsize=11)

    ax.set_xlim(3.70, 6.00)
    ax.set_ylim(0.0088, 0.0372)
    # Equal data scaling would make the grid almost flat because the two
    # parameter ranges differ by two orders of magnitude.  A square axes box
    # instead keeps both coordinates readable.
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title(title, pad=8)
    ax.grid(True, color="#B8B8B8", alpha=0.34, linewidth=0.55)

    handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label=r"Baseline $3\times3$ grid"),
    ]
    if include_enrichment:
        handles.extend(
            [
                Line2D([], [], color="#0072B2", marker="o", linestyle="None", markersize=6.8, label=r"18 interior LHS HPROM points"),
                Line2D([], [], color="#009E73", marker="o", linestyle="None", markersize=6.8, label=r"18 margin LHS HPROM points"),
            ]
        )
    handles.append(Line2D([], [], color="#D62728", marker="*", linestyle="None", markersize=10, label=r"Evaluation points"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        columnspacing=1.8,
        handletextpad=0.65,
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.20, top=0.93)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def _plot_sampling(
    path: Path,
    baseline_mu: np.ndarray,
    interior_mu: np.ndarray,
    exterior_mu: np.ndarray,
    evaluation_mu: np.ndarray,
    expanded_mu1: tuple[float, float],
    expanded_mu2: tuple[float, float],
) -> None:
    """Write separate baseline/enrichment figures and refresh the legacy path."""
    common = {
        "baseline_mu": baseline_mu,
        "interior_mu": interior_mu,
        "exterior_mu": exterior_mu,
        "evaluation_mu": evaluation_mu,
        "expanded_mu1": expanded_mu1,
        "expanded_mu2": expanded_mu2,
    }
    _plot_parameter_design(
        path.with_name("stage2_sampling_points_baseline.png"),
        title=r"Baseline training set in parameter space",
        include_enrichment=False,
        **common,
    )
    _plot_parameter_design(
        path,
        title=r"Expanded enriched training set in parameter space",
        include_enrichment=True,
        **common,
    )


def _write_manifest(
    path: Path,
    baseline_mu: np.ndarray,
    interior_mu: np.ndarray,
    exterior_mu: np.ndarray,
    exterior_regions: list[str],
    evaluation_mu: np.ndarray,
) -> None:
    evaluation_keys = {_storage_key(mu) for mu in evaluation_mu}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("role", "region", "mu1", "mu2", "directory_tag"))
        for mu in baseline_mu:
            region = "verification" if _storage_key(mu) in evaluation_keys else "baseline_grid"
            writer.writerow(("baseline_training", region, f"{mu[0]:.16g}", f"{mu[1]:.16g}", _mu_tag(mu)))
        for mu in interior_mu:
            writer.writerow(("lhs_enrichment", "interior_original_box", f"{mu[0]:.16g}", f"{mu[1]:.16g}", _mu_tag(mu)))
        for mu, region in zip(exterior_mu, exterior_regions):
            writer.writerow(("lhs_enrichment", region, f"{mu[0]:.16g}", f"{mu[1]:.16g}", _mu_tag(mu)))
        for index, mu in enumerate(evaluation_mu):
            label = "verification" if index == 0 else f"evaluation_{index}"
            writer.writerow(("evaluation_excluded", label, f"{mu[0]:.16g}", f"{mu[1]:.16g}", _mu_tag(mu)))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basis-path", required=True)
    parser.add_argument("--u-ref-path", required=True)
    parser.add_argument("--ecsw-weights-path", required=True)
    parser.add_argument("--total-modes", type=int, default=151)
    parser.add_argument("--interior-samples", type=int, default=18)
    parser.add_argument("--exterior-samples", type=int, default=18)
    parser.add_argument("--lhs-seed", type=int, default=42)
    parser.add_argument("--margin-fraction", type=float, default=0.25)
    parser.add_argument("--exclude-mu", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--regenerate-design", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Create/check the sampling design and exit before solving qN.")
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
    baseline_mu = _baseline_mu_from_dataset(base_per_mu, args.total_modes)
    evaluation_mu = np.asarray(args.exclude_mu, dtype=np.float64).reshape(-1, 2)

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

    print(f"[extended-enrichment-stage2] base_dataset={base_dataset}")
    print(f"[extended-enrichment-stage2] output_dataset={output_dir}")
    print(f"[extended-enrichment-stage2] basis={basis_path}")
    print(f"[extended-enrichment-stage2] u_ref={u_ref_path}")
    print(f"[extended-enrichment-stage2] fixed_linear_ecsw={ecsw_path}")
    print(f"[extended-enrichment-stage2] fixed_linear_ecsw_sha256={_sha256(ecsw_path)}")
    print("[extended-enrichment-stage2] ECSW policy=strict reuse; no ECSW build or copy")
    print(f"[extended-enrichment-stage2] original_mu1_range={ORIGINAL_MU1_RANGE}")
    print(f"[extended-enrichment-stage2] original_mu2_range={ORIGINAL_MU2_RANGE}")
    print(f"[extended-enrichment-stage2] expanded_mu1_range={expanded_mu1}")
    print(f"[extended-enrichment-stage2] expanded_mu2_range={expanded_mu2}")
    print(f"[extended-enrichment-stage2] interior_lhs={len(interior_mu)} exterior_lhs={len(exterior_mu)} total_lhs={len(lhs_mu)}")

    if args.plan_only:
        summary_lines = [
            f"dataset_dir: {output_dir}",
            "plan_only: True",
            f"total_modes: {args.total_modes}",
            f"num_base_traj_available: {len(baseline_mu)}",
            f"num_interior_lhs_traj: {len(interior_mu)}",
            f"num_exterior_lhs_traj: {len(exterior_mu)}",
            f"num_lhs_traj: {len(lhs_mu)}",
            f"lhs_seed: {args.lhs_seed}",
            f"margin_fraction: {args.margin_fraction}",
            f"original_mu1_range: {ORIGINAL_MU1_RANGE}",
            f"original_mu2_range: {ORIGINAL_MU2_RANGE}",
            f"expanded_mu1_range: {expanded_mu1}",
            f"expanded_mu2_range: {expanded_mu2}",
            f"ecsw_weights_path: {ecsw_path}",
            f"ecsw_weights_sha256: {_sha256(ecsw_path)}",
        ]
        _atomic_write_text(output_dir / "stage2_extended_enrichment_plan_summary.txt", "\n".join(summary_lines) + "\n")
        print("[extended-enrichment-stage2] PLAN_ONLY complete; no qN solves were run.")
        return

    output_per_mu.mkdir(parents=True, exist_ok=True)
    copied_baseline_mu = np.asarray(
        _copy_baseline_dataset(base_per_mu, output_per_mu, args.total_modes),
        dtype=np.float64,
    )
    if not np.allclose(np.sort(copied_baseline_mu, axis=0), np.sort(baseline_mu, axis=0), rtol=0.0, atol=1e-14):
        raise ValueError("Copied baseline parameter set does not match source baseline set.")

    vtot = basis[:, : args.total_modes]
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    t_vec = DT * np.arange(NUM_STEPS + 1, dtype=np.float64)
    if w0.size != vtot.shape[0]:
        raise ValueError(f"W0 size {w0.size} does not match basis rows {vtot.shape[0]}.")

    total_new = len(lhs_mu)
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
                    print(f"[extended-enrichment-stage2] [{index}/{total_new}] reuse {_mu_tag(mu)}")
                    continue
            except (OSError, ValueError):
                pass

        print(f"[extended-enrichment-stage2] [{index}/{total_new}] solve {_mu_tag(mu)}")
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
    expected_total = 9 + int(args.interior_samples) + int(args.exterior_samples)
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} trajectories, found {len(records)} in {output_per_mu}.")
    for mu_dir, _ in records:
        _validate_trajectory(mu_dir, args.total_modes)

    ecsw_hash = _sha256(ecsw_path)
    metadata = {
        "solve_backend": "hprom",
        "is_enrichment_dataset": True,
        "enrichment_protocol": "baseline_9_plus_lhs36_ext25_fixed_linear_hprom",
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
        f"num_interior_lhs_traj: {len(interior_mu)}",
        f"num_exterior_lhs_traj: {len(exterior_mu)}",
        f"num_lhs_traj: {len(lhs_mu)}",
        f"num_traj_total: {len(records)}",
        f"lhs_seed: {args.lhs_seed}",
        f"margin_fraction: {args.margin_fraction}",
        f"original_mu1_range: {ORIGINAL_MU1_RANGE}",
        f"original_mu2_range: {ORIGINAL_MU2_RANGE}",
        f"expanded_mu1_range: {expanded_mu1}",
        f"expanded_mu2_range: {expanded_mu2}",
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
    print(f"[extended-enrichment-stage2] complete: {len(records)} trajectories")
    print(f"[extended-enrichment-stage2] summary={output_dir / 'stage2_enrichment_summary.txt'}")


if __name__ == "__main__":
    main()
