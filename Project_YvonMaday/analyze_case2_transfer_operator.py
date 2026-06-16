#!/usr/bin/env python3
r"""Compare the Case-2 low/high LSPG transfer operator across trial bases.

For each selected state and basis split B=[V, Vbar], this script computes

    T_LH = -(V^T J^T P J V)^\dagger V^T J^T P J Vbar.

The raw coordinate norm of T_LH is reported, but it is not invariant under a
change of basis scaling. The primary comparison is therefore the physical
mass-norm gain

    sup_z ||V T_LH z||_M / ||Vbar z||_M.

Both bases are evaluated at the same HDM states, isolating the effect of the
trial basis from differences between reduced trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from burgers.config import DT, GRID_X, GRID_Y
from burgers.core import get_ops, inviscid_burgers_exact_jac2D


DEFAULT_POINTS = (
    (4.560, 0.0190),
    (4.875, 0.0225),
    (5.190, 0.0260),
)


@dataclass(frozen=True)
class BasisSpec:
    label: str
    path: Path


@dataclass
class BasisData:
    spec: BasisSpec
    low: np.ndarray
    high: np.ndarray
    mass_low_sqrt: np.ndarray
    mass_high_inv_sqrt: np.ndarray
    low_mass_gram_cond: float
    high_mass_gram_cond: float
    low_high_mass_cross_fro: float


def _parse_basis(value: str) -> BasisSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --basis '{value}'. Expected LABEL=/path/to/basis.npy."
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Basis label cannot be empty.")
    return BasisSpec(label=label, path=Path(raw_path).expanduser().resolve())


def _parse_point(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid --point '{value}'. Expected MU1,MU2."
        )
    return float(parts[0]), float(parts[1])


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _mass_diagonal() -> np.ndarray:
    dx = np.asarray(GRID_X[1:] - GRID_X[:-1], dtype=np.float64)
    dy = np.asarray(GRID_Y[1:] - GRID_Y[:-1], dtype=np.float64)
    cell_area = np.outer(dy, dx).reshape(-1)
    return np.concatenate((cell_area, cell_area))


def _symmetric_sqrt(matrix: np.ndarray, *, inverse: bool) -> tuple[np.ndarray, float]:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = float(np.max(eigenvalues))
    cutoff = max(largest * 1.0e-13, np.finfo(np.float64).eps)
    if largest <= 0.0 or np.any(eigenvalues <= cutoff):
        raise np.linalg.LinAlgError(
            "The physical Gram matrix is numerically singular; "
            f"min={np.min(eigenvalues):.3e}, max={largest:.3e}."
        )
    factors = 1.0 / np.sqrt(eigenvalues) if inverse else np.sqrt(eigenvalues)
    root = (eigenvectors * factors[None, :]) @ eigenvectors.T
    return root, largest / float(np.min(eigenvalues))


def _load_basis(
    spec: BasisSpec,
    *,
    n_primary: int,
    n_tot: int,
    mass_diag: np.ndarray,
) -> BasisData:
    if not spec.path.is_file():
        raise FileNotFoundError(f"Missing basis: {spec.path}")
    basis = np.asarray(np.load(spec.path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] < n_tot:
        raise ValueError(
            f"Basis '{spec.label}' has shape {basis.shape}; "
            f"at least (*,{n_tot}) is required."
        )
    if basis.shape[0] != mass_diag.size:
        raise ValueError(
            f"Basis '{spec.label}' has {basis.shape[0]} rows, "
            f"but the mass vector has {mass_diag.size} entries."
        )

    low = np.ascontiguousarray(basis[:, :n_primary])
    high = np.ascontiguousarray(basis[:, n_primary:n_tot])
    low_gram = low.T @ (mass_diag[:, None] * low)
    high_gram = high.T @ (mass_diag[:, None] * high)
    low_sqrt, low_cond = _symmetric_sqrt(low_gram, inverse=False)
    high_inv_sqrt, high_cond = _symmetric_sqrt(high_gram, inverse=True)
    cross = low.T @ (mass_diag[:, None] * high)

    return BasisData(
        spec=spec,
        low=low,
        high=high,
        mass_low_sqrt=low_sqrt,
        mass_high_inv_sqrt=high_inv_sqrt,
        low_mass_gram_cond=low_cond,
        high_mass_gram_cond=high_cond,
        low_high_mass_cross_fro=float(np.linalg.norm(cross, ord="fro")),
    )


def _resolve_snapshot(snap_dir: Path, mu1: float, mu2: float) -> Path:
    candidates = (
        snap_dir / f"mu1_{mu1:g}+mu2_{mu2:g}.npy",
        snap_dir / f"mu1_{mu1:.3f}+mu2_{mu2:.4f}.npy",
        snap_dir / f"mu1_{mu1}+mu2_{mu2}.npy",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No HDM snapshot file found for mu=({mu1:g},{mu2:g}) in {snap_dir}. "
        f"Tried: {', '.join(str(path.name) for path in candidates)}"
    )


def _spectral_pseudoinverse_solve(
    h_ll: np.ndarray,
    h_lh: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, int, float, float, float]:
    h_ll = 0.5 * (h_ll + h_ll.T)
    eigenvalues, eigenvectors = np.linalg.eigh(h_ll)
    max_eigenvalue = float(np.max(eigenvalues))
    cutoff = max(float(rcond) * max_eigenvalue, np.finfo(np.float64).eps)
    keep = eigenvalues > cutoff
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise np.linalg.LinAlgError("H_LL has zero numerical rank.")

    projected = eigenvectors[:, keep].T @ h_lh
    transfer = -(
        eigenvectors[:, keep]
        @ (projected / eigenvalues[keep, None])
    )
    min_kept = float(np.min(eigenvalues[keep]))
    condition = max_eigenvalue / min_kept
    return transfer, rank, condition, min_kept, max_eigenvalue


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latex_table(path: Path, aggregate_rows: list[dict]) -> None:
    all_rows = [row for row in aggregate_rows if row["scope"] == "all_points"]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Basis & $\operatorname{mean}\sigma_{\max}(T_{LH})$"
            r" & $\operatorname{p95}\sigma_{\max}(T_{LH})$"
            r" & $\max\sigma_{\max}(T_{LH})$"
            r" & $\operatorname{mean}g_M$"
            r" & $\operatorname{p95}g_M$"
            r" & $\max g_M$ \\"
        ),
        r"\midrule",
    ]
    for row in all_rows:
        label = str(row["basis"]).replace("_", r"\_")
        lines.append(
            f"{label} & "
            f"{row['coordinate_sigma_max_mean']:.3e} & "
            f"{row['coordinate_sigma_max_p95']:.3e} & "
            f"{row['coordinate_sigma_max_max']:.3e} & "
            f"{row['mass_state_gain_mean']:.3e} & "
            f"{row['mass_state_gain_p95']:.3e} & "
            f"{row['mass_state_gain_max']:.3e} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_time_histories(
    path: Path,
    rows: list[dict],
    points: list[tuple[float, float]],
    labels: list[str],
) -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    fig, axes = plt.subplots(
        2,
        len(points),
        figsize=(5.0 * len(points), 7.0),
        sharex="col",
        constrained_layout=True,
    )
    if len(points) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, (mu1, mu2) in enumerate(points):
        for idx, label in enumerate(labels):
            selected = [
                row
                for row in rows
                if row["basis"] == label
                and math.isclose(float(row["mu1"]), mu1)
                and math.isclose(float(row["mu2"]), mu2)
            ]
            selected.sort(key=lambda row: int(row["step"]))
            time_values = [float(row["time"]) for row in selected]
            axes[0, col].plot(
                time_values,
                [float(row["coordinate_sigma_max"]) for row in selected],
                color=colors[idx % len(colors)],
                linewidth=1.4,
                alpha=0.85,
                label=label,
            )
            axes[1, col].plot(
                time_values,
                [float(row["mass_state_gain"]) for row in selected],
                color=colors[idx % len(colors)],
                linewidth=1.4,
                alpha=0.85,
                label=label,
            )

        axes[0, col].set_title(
            rf"$\mathbf{{\mu}}=({mu1:.3f},{mu2:.4f})$"
        )
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
        axes[0, col].grid(True, which="both", alpha=0.25)
        axes[1, col].grid(True, which="both", alpha=0.25)
        axes[1, col].set_xlabel(r"Time $t$")

    axes[0, 0].set_ylabel(r"Coordinate gain $\sigma_{\max}(T_{LH})$")
    axes[1, 0].set_ylabel(r"Physical mass-norm gain $g_M$")
    axes[0, -1].legend(frameon=True)
    fig.suptitle("Case-2 low/high LSPG transfer diagnostic")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_distributions(path: Path, rows: list[dict], labels: list[str]) -> None:
    plt.rcParams["text.usetex"] = False
    coordinate = [
        np.asarray(
            [float(row["coordinate_sigma_max"]) for row in rows if row["basis"] == label]
        )
        for label in labels
    ]
    physical = [
        np.asarray(
            [float(row["mass_state_gain"]) for row in rows if row["basis"] == label]
        )
        for label in labels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].boxplot(coordinate, labels=labels, showfliers=False)
    axes[1].boxplot(physical, labels=labels, showfliers=False)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$\sigma_{\max}(T_{LH})$")
    axes[1].set_ylabel(r"$g_M$")
    axes[0].set_title("Coordinate-space transfer")
    axes[1].set_title("Physical mass-norm transfer")
    for axis in axes:
        axis.grid(True, which="both", axis="y", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Case-2 low/high transfer operator across bases."
    )
    parser.add_argument(
        "--basis",
        action="append",
        type=_parse_basis,
        default=[],
        metavar="LABEL=PATH",
        help="Basis to evaluate. Repeat for multiple bases.",
    )
    parser.add_argument(
        "--point",
        action="append",
        type=_parse_point,
        default=[],
        metavar="MU1,MU2",
        help="HDM parameter point. Repeat for multiple points.",
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=REPO_DIR / "Results" / "param_snaps",
    )
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--n-tot", type=int, default=151)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--time-start-index", type=int, default=1)
    parser.add_argument("--time-stop-index", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--normal-rcond",
        type=float,
        default=1.0e-12,
        help="Relative spectral cutoff for H_LL pseudoinversion.",
    )
    parser.add_argument(
        "--p-diag-path",
        type=Path,
        default=None,
        help="Optional diagonal residual metric P. The default is P=I.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "Results_Paper" / "MetricStudy" / "low_high_transfer",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if not (1 <= args.n_primary < args.n_tot):
        raise ValueError("Require 1 <= n_primary < n_tot.")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.normal_rcond <= 0.0:
        raise ValueError("--normal-rcond must be positive.")

    basis_specs = args.basis or [
        BasisSpec(
            "Euclidean POD",
            PROJECT_DIR / "Results_Paper" / "MetricStudy" / "euclidean" / "Stage1" / "basis.npy",
        ),
        BasisSpec(
            "LSPG-sensitive POD",
            PROJECT_DIR
            / "Results_Paper"
            / "MetricStudy"
            / "lspg_sensitive"
            / "Stage1"
            / "basis.npy",
        ),
    ]
    points = args.point or list(DEFAULT_POINTS)
    snap_dir = args.snap_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mass_diag = _mass_diagonal()
    bases = [
        _load_basis(
            spec,
            n_primary=args.n_primary,
            n_tot=args.n_tot,
            mass_diag=mass_diag,
        )
        for spec in basis_specs
    ]

    p_diag = None
    if args.p_diag_path is not None:
        p_diag = np.asarray(
            np.load(args.p_diag_path.expanduser().resolve(), allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
        if p_diag.size != mass_diag.size:
            raise ValueError(
                f"P diagonal has {p_diag.size} entries; expected {mass_diag.size}."
            )
        if np.any(p_diag < 0.0):
            raise ValueError("P diagonal must be nonnegative.")

    _, _, jdx, jdy, identity = get_ops(GRID_X, GRID_Y)
    total_states = 0
    snapshots: list[tuple[float, float, Path, np.ndarray, list[int]]] = []
    for mu1, mu2 in points:
        path = _resolve_snapshot(snap_dir, mu1, mu2)
        states = np.load(path, mmap_mode="r", allow_pickle=False)
        if states.ndim != 2 or states.shape[0] != mass_diag.size:
            raise ValueError(f"Unexpected snapshot shape in {path}: {states.shape}")
        stop = min(int(args.time_stop_index), states.shape[1] - 1)
        indices = list(range(max(0, args.time_start_index), stop + 1, args.stride))
        if not indices:
            raise ValueError(f"No time indices selected for {path}.")
        snapshots.append((mu1, mu2, path, states, indices))
        total_states += len(indices)

    total_evaluations = total_states * len(bases)
    print(
        "[TRANSFER] "
        f"bases={len(bases)}, points={len(points)}, states={total_states}, "
        f"basis-state evaluations={total_evaluations}",
        flush=True,
    )
    print(
        f"[TRANSFER] split: n_primary={args.n_primary}, "
        f"n_secondary={args.n_tot - args.n_primary}, n_tot={args.n_tot}",
        flush=True,
    )
    print(
        f"[TRANSFER] residual metric P={'diagonal' if p_diag is not None else 'identity'}",
        flush=True,
    )

    rows: list[dict] = []
    worst: dict[str, dict] = {}
    completed = 0
    start = time.time()

    for mu1, mu2, snapshot_path, states, indices in snapshots:
        print(
            f"[TRANSFER] point mu=({mu1:.3f},{mu2:.4f}) | "
            f"snapshots={snapshot_path} | selected={len(indices)}",
            flush=True,
        )
        for step in indices:
            state = np.asarray(states[:, step], dtype=np.float64)
            jacobian = inviscid_burgers_exact_jac2D(
                state, args.dt, jdx, jdy, identity
            )

            for basis in bases:
                j_low = np.asarray(jacobian @ basis.low, dtype=np.float64)
                j_high = np.asarray(jacobian @ basis.high, dtype=np.float64)
                if p_diag is None:
                    h_ll = j_low.T @ j_low
                    h_lh = j_low.T @ j_high
                    high_jacobian_norm = np.linalg.norm(j_high, ord="fro")
                else:
                    weighted_low = p_diag[:, None] * j_low
                    weighted_high = p_diag[:, None] * j_high
                    h_ll = j_low.T @ weighted_low
                    h_lh = j_low.T @ weighted_high
                    high_jacobian_norm = math.sqrt(
                        float(np.sum(j_high * weighted_high))
                    )

                transfer, rank, condition, eig_min, eig_max = (
                    _spectral_pseudoinverse_solve(
                        h_ll, h_lh, rcond=args.normal_rcond
                    )
                )
                singular_values = np.linalg.svd(
                    transfer, compute_uv=False, full_matrices=False
                )
                coordinate_sigma_max = float(singular_values[0])
                coordinate_fro = float(np.linalg.norm(singular_values))

                physical_operator = (
                    basis.mass_low_sqrt
                    @ transfer
                    @ basis.mass_high_inv_sqrt
                )
                physical_singular_values = np.linalg.svd(
                    physical_operator, compute_uv=False, full_matrices=False
                )
                mass_state_gain = float(physical_singular_values[0])

                cancellation = j_low @ transfer + j_high
                if p_diag is None:
                    cancellation_norm = np.linalg.norm(cancellation, ord="fro")
                else:
                    cancellation_norm = math.sqrt(
                        float(np.sum(cancellation * (p_diag[:, None] * cancellation)))
                    )
                cancellation_rel = float(
                    cancellation_norm / (high_jacobian_norm + 1.0e-30)
                )

                row = {
                    "basis": basis.spec.label,
                    "basis_path": str(basis.spec.path),
                    "mu1": mu1,
                    "mu2": mu2,
                    "step": step,
                    "time": step * args.dt,
                    "coordinate_sigma_max": coordinate_sigma_max,
                    "coordinate_fro": coordinate_fro,
                    "mass_state_gain": mass_state_gain,
                    "hll_rank": rank,
                    "hll_condition": condition,
                    "hll_min_kept_eigenvalue": eig_min,
                    "hll_max_eigenvalue": eig_max,
                    "cancellation_relative_residual": cancellation_rel,
                }
                for index, value in enumerate(singular_values, start=1):
                    row[f"coordinate_sigma_{index}"] = float(value)
                rows.append(row)

                key = basis.spec.label
                if key not in worst or mass_state_gain > worst[key]["mass_state_gain"]:
                    worst[key] = {
                        "mass_state_gain": mass_state_gain,
                        "mu1": mu1,
                        "mu2": mu2,
                        "step": step,
                        "time": step * args.dt,
                        "transfer": transfer.copy(),
                        "coordinate_singular_values": singular_values.copy(),
                        "physical_singular_values": physical_singular_values.copy(),
                    }

                completed += 1
                if (
                    completed % max(1, args.progress_every) == 0
                    or completed == total_evaluations
                ):
                    elapsed = max(time.time() - start, 1.0e-12)
                    eta = (total_evaluations - completed) * elapsed / completed
                    print(
                        "[TRANSFER] "
                        f"{completed}/{total_evaluations} | "
                        f"mu=({mu1:.3f},{mu2:.4f}) step={step} "
                        f"basis={basis.spec.label} | "
                        f"sigma={coordinate_sigma_max:.3e} "
                        f"g_M={mass_state_gain:.3e} | "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )

                del j_low, j_high
            del jacobian

    singular_count = min(args.n_primary, args.n_tot - args.n_primary)
    per_sample_fields = [
        "basis",
        "basis_path",
        "mu1",
        "mu2",
        "step",
        "time",
        "coordinate_sigma_max",
        "coordinate_fro",
        "mass_state_gain",
        "hll_rank",
        "hll_condition",
        "hll_min_kept_eigenvalue",
        "hll_max_eigenvalue",
        "cancellation_relative_residual",
    ] + [f"coordinate_sigma_{index}" for index in range(1, singular_count + 1)]
    per_sample_path = output_dir / "low_high_transfer_per_sample.csv"
    _write_csv(per_sample_path, rows, per_sample_fields)

    aggregate_rows: list[dict] = []
    metric_names = (
        "coordinate_sigma_max",
        "coordinate_fro",
        "mass_state_gain",
        "hll_condition",
        "cancellation_relative_residual",
    )
    for basis in bases:
        scopes = [("all_points", None)] + [
            (f"mu=({mu1:.3f},{mu2:.4f})", (mu1, mu2)) for mu1, mu2 in points
        ]
        for scope_name, point in scopes:
            selected = [
                row
                for row in rows
                if row["basis"] == basis.spec.label
                and (
                    point is None
                    or (
                        math.isclose(float(row["mu1"]), point[0])
                        and math.isclose(float(row["mu2"]), point[1])
                    )
                )
            ]
            aggregate = {
                "basis": basis.spec.label,
                "basis_path": str(basis.spec.path),
                "scope": scope_name,
                "n_samples": len(selected),
                "low_mass_gram_condition": basis.low_mass_gram_cond,
                "high_mass_gram_condition": basis.high_mass_gram_cond,
                "low_high_mass_cross_fro": basis.low_high_mass_cross_fro,
            }
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in selected])
                for statistic, value in _stats(values).items():
                    aggregate[f"{metric}_{statistic}"] = value
            aggregate_rows.append(aggregate)

    aggregate_fields = list(aggregate_rows[0].keys())
    aggregate_path = output_dir / "low_high_transfer_summary.csv"
    _write_csv(aggregate_path, aggregate_rows, aggregate_fields)

    comparison_rows: list[dict] = []
    if len(bases) >= 2:
        first_label = bases[0].spec.label
        second_label = bases[1].spec.label
        first_rows = {
            row["scope"]: row
            for row in aggregate_rows
            if row["basis"] == first_label
        }
        second_rows = {
            row["scope"]: row
            for row in aggregate_rows
            if row["basis"] == second_label
        }
        for scope in first_rows.keys() & second_rows.keys():
            for metric in ("coordinate_sigma_max", "mass_state_gain"):
                for statistic in ("mean", "p95", "max"):
                    key = f"{metric}_{statistic}"
                    baseline = float(first_rows[scope][key])
                    candidate = float(second_rows[scope][key])
                    comparison_rows.append(
                        {
                            "scope": scope,
                            "metric": metric,
                            "statistic": statistic,
                            "baseline_basis": first_label,
                            "candidate_basis": second_label,
                            "baseline_value": baseline,
                            "candidate_value": candidate,
                            "candidate_over_baseline": candidate
                            / (baseline + 1.0e-30),
                            "reduction_percent": 100.0
                            * (1.0 - candidate / (baseline + 1.0e-30)),
                        }
                    )
        comparison_path = output_dir / "low_high_transfer_comparison.csv"
        _write_csv(
            comparison_path,
            comparison_rows,
            list(comparison_rows[0].keys()),
        )

    table_path = output_dir / "low_high_transfer_summary.tex"
    _write_latex_table(table_path, aggregate_rows)

    worst_payload = {}
    worst_metadata = {}
    for label, record in worst.items():
        key = _slug(label)
        worst_payload[f"{key}_transfer"] = record.pop("transfer")
        worst_payload[f"{key}_coordinate_singular_values"] = record.pop(
            "coordinate_singular_values"
        )
        worst_payload[f"{key}_physical_singular_values"] = record.pop(
            "physical_singular_values"
        )
        worst_metadata[label] = record
    np.savez(output_dir / "low_high_transfer_worst_cases.npz", **worst_payload)

    metadata = {
        "n_primary": args.n_primary,
        "n_secondary": args.n_tot - args.n_primary,
        "n_tot": args.n_tot,
        "dt": args.dt,
        "time_start_index": args.time_start_index,
        "time_stop_index": args.time_stop_index,
        "stride": args.stride,
        "normal_rcond": args.normal_rcond,
        "residual_metric": "diagonal" if p_diag is not None else "identity",
        "basis_order": [basis.spec.label for basis in bases],
        "points": [{"mu1": mu1, "mu2": mu2} for mu1, mu2 in points],
        "worst_mass_state_gain": worst_metadata,
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "low_high_transfer_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    _plot_time_histories(
        output_dir / "low_high_transfer_vs_time.png",
        rows,
        points,
        [basis.spec.label for basis in bases],
    )
    _plot_distributions(
        output_dir / "low_high_transfer_distributions.png",
        rows,
        [basis.spec.label for basis in bases],
    )

    print(f"[TRANSFER] per-sample CSV: {per_sample_path}")
    print(f"[TRANSFER] summary CSV:    {aggregate_path}")
    if comparison_rows:
        print(
            "[TRANSFER] comparison CSV: "
            f"{output_dir / 'low_high_transfer_comparison.csv'}"
        )
    print(f"[TRANSFER] LaTeX table:    {table_path}")
    print(
        "[TRANSFER] figures:        "
        f"{output_dir / 'low_high_transfer_vs_time.png'}"
    )
    print(
        "[TRANSFER]                 "
        f"{output_dir / 'low_high_transfer_distributions.png'}"
    )


if __name__ == "__main__":
    main()
