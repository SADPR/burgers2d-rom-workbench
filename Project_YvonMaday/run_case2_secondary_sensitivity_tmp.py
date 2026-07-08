#!/usr/bin/env python3
"""Temporary Case-2 secondary-oracle sensitivity diagnostic.

For a fixed primary dimension n=10, prescribe the secondary coefficients
q_{n+1:151}(t) from the linear PROM and contaminate them in the direction of
the actual master-ANN secondary error.  The PROM solve then recomputes only the
first n coordinates.  This isolates how secondary-coordinate errors propagate
into the recovered primary coordinates and the state error.

This script is intentionally diagnostic-only and writes under a temporary
Results_Paper folder by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from burgers.config import DT, NUM_STEPS  # noqa: E402
from burgers.core import load_or_compute_snaps  # noqa: E402
from burgers.pod_ann_manifold import (  # noqa: E402
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin,
)


NTOT_DEFAULT = 151
N_PRIMARY_DEFAULT = 10
POINTS = {
    "verification": (4.875, 0.0225),
    "offgrid1": (4.560, 0.0190),
    "offgrid2": (5.190, 0.0260),
    "extrapolation20pct": (4.000, 0.0330),
}


@dataclass(frozen=True)
class Point:
    key: str
    mu1: float
    mu2: float


class TimeTableSecondary(nn.Module):
    """Map (mu1, mu2, t) -> prescribed q_secondary(t)."""

    def __init__(self, qbar_table: np.ndarray, dt: float):
        super().__init__()
        qbar = np.asarray(qbar_table, dtype=np.float64)
        if qbar.ndim != 2:
            raise ValueError(f"qbar_table must be 2D, got {qbar.shape}")
        self.register_buffer("qbar_table", torch.from_numpy(qbar))
        self.dt = float(dt)
        self.n_t = int(qbar.shape[1])

    def _time_index(self, t: torch.Tensor) -> torch.Tensor:
        idx = torch.round(t / self.dt).long()
        return torch.clamp(idx, min=0, max=self.n_t - 1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = x_raw
        if x.ndim == 1:
            idx = self._time_index(x[2])
            return self.qbar_table[:, idx]
        if x.ndim == 2:
            idx = self._time_index(x[:, 2])
            return self.qbar_table.index_select(1, idx).T
        raise ValueError(f"Unsupported input shape for qbar table: {tuple(x.shape)}")


def mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def infer_square_grid(n_state: int):
    n_cells = int(n_state // 2)
    n_side = int(round(np.sqrt(n_cells)))
    if n_side * n_side != n_cells:
        raise ValueError(f"Cannot infer square grid from state size {n_state}.")
    grid_x = np.linspace(0.0, 100.0, n_side + 1, dtype=np.float64)
    grid_y = np.linspace(0.0, 100.0, n_side + 1, dtype=np.float64)
    return grid_x, grid_y


def parse_levels(values: list[str]) -> list[float]:
    levels: list[float] = []
    for v in values:
        for item in str(v).replace(",", " ").split():
            if item:
                levels.append(float(item))
    if not levels:
        raise ValueError("At least one perturbation level is required.")
    if any(v < 0.0 for v in levels):
        raise ValueError(f"Perturbation levels must be nonnegative: {levels}")
    return sorted(dict.fromkeys(levels))


def resolve_points(point_keys: list[str]) -> list[Point]:
    keys: list[str] = []
    for raw in point_keys:
        for item in str(raw).replace(",", " ").split():
            if item:
                keys.append(item)
    if not keys or keys == ["all"]:
        keys = list(POINTS)
    out = []
    for key in keys:
        if key not in POINTS:
            valid = ", ".join(["all", *POINTS.keys()])
            raise ValueError(f"Unknown point '{key}'. Valid choices: {valid}")
        mu1, mu2 = POINTS[key]
        out.append(Point(key, mu1, mu2))
    return out


def load_required_array(path: Path, *, ndim: int, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {arr.shape}: {path}")
    return arr


def align_time(*arrays: np.ndarray) -> tuple[list[np.ndarray], int]:
    n_t = min(int(a.shape[1]) for a in arrays)
    return [np.asarray(a)[:, :n_t] for a in arrays], n_t


def secondary_with_error(
    qbar_ref: np.ndarray,
    qbar_ann: np.ndarray,
    requested_percent: float,
) -> tuple[np.ndarray, float, float]:
    qbar_ref = np.asarray(qbar_ref, dtype=np.float64)
    qbar_ann = np.asarray(qbar_ann, dtype=np.float64)
    direction = qbar_ann - qbar_ref
    ref_norm = float(np.linalg.norm(qbar_ref))
    dir_norm = float(np.linalg.norm(direction))
    if requested_percent <= 0.0:
        return qbar_ref.copy(), 0.0, 100.0 * dir_norm / (ref_norm + 1.0e-30)
    if dir_norm <= 0.0:
        raise RuntimeError("ANN secondary direction has zero norm; cannot scale perturbation.")
    target_norm = (float(requested_percent) / 100.0) * ref_norm
    delta = direction * (target_norm / dir_norm)
    actual_percent = 100.0 * float(np.linalg.norm(delta)) / (ref_norm + 1.0e-30)
    ann_percent = 100.0 * dir_norm / (ref_norm + 1.0e-30)
    return qbar_ref + delta, actual_percent, ann_percent


def solve_one_level(
    *,
    solver_variant: str,
    qbar: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    w0: np.ndarray,
    dt: float,
    num_steps: int,
    mu: list[float],
    v: np.ndarray,
    vbar: np.ndarray,
    u_ref: np.ndarray,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    linear_solver: str,
    normal_eq_reg: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]:
    model = TimeTableSecondary(qbar, dt=dt)
    t0 = time.time()
    if solver_variant == "plain":
        snaps, red_coords, stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu,
            ann_model=model,
            ref=None,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            return_red_coords=True,
        )
    elif solver_variant == "pg":
        snaps, stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2_petrov_galerkin(
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps,
            mu=mu,
            ann_model=model,
            ref=None,
            basis=v,
            basis2=vbar,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        # PG helper does not return the primary variables, so recover them from
        # u - u_ref - Vbar qbar using least squares on V.
        red_coords = np.linalg.lstsq(v, snaps - u_ref[:, None] - vbar @ qbar[:, : snaps.shape[1]], rcond=None)[0]
    else:
        raise ValueError(f"Unsupported solver_variant={solver_variant!r}")
    return snaps, red_coords, stats, time.time() - t0


def run_point(
    *,
    point: Point,
    levels: list[float],
    prom_root: Path,
    basis_path: Path,
    u_ref_path: Path,
    output_root: Path,
    n_primary: int,
    n_tot: int,
    dt: float,
    num_steps: int,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    solver_variant: str,
    linear_solver: str,
    normal_eq_reg: float,
    save_arrays: bool,
    force: bool,
    include_ann_level: bool,
) -> list[dict[str, object]]:
    tag = mu_tag(point.mu1, point.mu2)
    point_out = output_root / point.key
    point_out.mkdir(parents=True, exist_ok=True)

    linear_dir = prom_root / "Runs" / "Linear" / f"linear_prom_{tag}_ntot{n_tot}"
    ann_dir = prom_root / "Runs" / "ROM" / "DataDriven_MasterANN" / f"rom_data_driven_{tag}_ntot{n_tot}"

    qn_linear = load_required_array(linear_dir / "qN.npy", ndim=2, name="linear qN")[:n_tot]
    lin_snaps = load_required_array(linear_dir / "rom_snaps.npy", ndim=2, name="linear rom_snaps")
    qn_ann = load_required_array(ann_dir / "qN.npy", ndim=2, name="data-driven ANN qN")[:n_tot]
    basis = load_required_array(basis_path, ndim=2, name="basis")[:, :n_tot]
    u_ref = load_required_array(u_ref_path, ndim=1, name="u_ref").reshape(-1)

    if basis.shape[0] != u_ref.size:
        raise ValueError(f"basis/u_ref size mismatch: {basis.shape[0]} vs {u_ref.size}")
    if n_primary <= 0 or n_primary >= n_tot:
        raise ValueError(f"Invalid n_primary={n_primary} for n_tot={n_tot}")

    (qn_linear, qn_ann, lin_snaps), n_t = align_time(qn_linear, qn_ann, lin_snaps)
    num_steps_eff = min(num_steps, n_t - 1)
    qn_linear = qn_linear[:, : num_steps_eff + 1]
    qn_ann = qn_ann[:, : num_steps_eff + 1]
    lin_snaps = lin_snaps[:, : num_steps_eff + 1]

    v = basis[:, :n_primary]
    vbar = basis[:, n_primary:n_tot]
    qbar_ref = qn_linear[n_primary:n_tot, :]
    qbar_ann = qn_ann[n_primary:n_tot, :]
    qp_ref = qn_linear[:n_primary, :]
    ann_secondary_percent = 100.0 * float(
        np.linalg.norm(qbar_ann - qbar_ref) / (np.linalg.norm(qbar_ref) + 1.0e-30)
    )

    w0 = lin_snaps[:, 0].copy()
    grid_x, grid_y = infer_square_grid(w0.size)
    snap_folder = REPO_DIR / "Results" / "param_snaps"
    hdm_snaps = load_or_compute_snaps(
        mu=[point.mu1, point.mu2],
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps_eff,
        snap_folder=snap_folder,
    )
    (hdm_snaps, lin_snaps), _ = align_time(hdm_snaps, lin_snaps)

    rows: list[dict[str, object]] = []
    point_levels = list(levels)
    if include_ann_level:
        point_levels.append(ann_secondary_percent)
    point_levels = sorted(dict.fromkeys(round(float(v), 12) for v in point_levels))

    for pct in point_levels:
        level_tag = f"eps{pct:.3f}".replace(".", "p")
        summary_path = point_out / f"{point.key}_{tag}_n{n_primary}_{level_tag}_summary.txt"
        if summary_path.exists() and not force:
            print(f"[skip] {point.key} eps={pct:g}% already exists: {summary_path}")
            continue

        qbar_used, actual_pct, ann_secondary_pct = secondary_with_error(qbar_ref, qbar_ann, pct)
        print(
            f"[run] {point.key} mu=({point.mu1:.3f},{point.mu2:.4f}) "
            f"n={n_primary} secondary_error={actual_pct:.3f}%"
        )
        snaps, qp, stats, elapsed = solve_one_level(
            solver_variant=solver_variant,
            qbar=qbar_used,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps_eff,
            mu=[point.mu1, point.mu2],
            v=v,
            vbar=vbar,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        (hdm_cmp, lin_cmp, snaps_cmp, qp_cmp, qp_ref_cmp, qbar_cmp, qbar_ref_cmp), n_t_used = align_time(
            hdm_snaps,
            lin_snaps,
            snaps,
            qp,
            qp_ref,
            qbar_used,
            qbar_ref,
        )

        q_full = np.vstack([qp_cmp, qbar_cmp])
        q_ref = np.vstack([qp_ref_cmp, qbar_ref_cmp])
        state_err = 100.0 * float(np.linalg.norm(hdm_cmp - snaps_cmp) / np.linalg.norm(hdm_cmp))
        state_vs_linear = 100.0 * float(np.linalg.norm(lin_cmp - snaps_cmp) / np.linalg.norm(lin_cmp))
        qp_err = 100.0 * float(np.linalg.norm(qp_cmp - qp_ref_cmp) / np.linalg.norm(qp_ref_cmp))
        qbar_err = 100.0 * float(np.linalg.norm(qbar_cmp - qbar_ref_cmp) / np.linalg.norm(qbar_ref_cmp))
        qtot_err = 100.0 * float(np.linalg.norm(q_full - q_ref) / np.linalg.norm(q_ref))

        row = {
            "point": point.key,
            "mu1": point.mu1,
            "mu2": point.mu2,
            "n_primary": n_primary,
            "n_tot": n_tot,
            "requested_secondary_error_percent": pct,
            "actual_secondary_error_percent": qbar_err,
            "ann_secondary_error_percent": ann_secondary_pct,
            "state_error_percent_vs_hdm": state_err,
            "state_error_percent_vs_linear_prom": state_vs_linear,
            "primary_q_error_percent_vs_linear_prom": qp_err,
            "total_q_error_percent_vs_linear_prom": qtot_err,
            "n_time_used": n_t_used,
            "online_solve_elapsed_s": elapsed,
            "num_iterations": stats[0],
            "jac_time_s": stats[1],
            "res_time_s": stats[2],
            "ls_time_s": stats[3],
        }
        rows.append(row)

        with summary_path.open("w", encoding="utf-8") as f:
            for k, vout in row.items():
                f.write(f"{k}: {vout}\n")
            f.write(f"linear_qn_path: {linear_dir / 'qN.npy'}\n")
            f.write(f"ann_qn_path: {ann_dir / 'qN.npy'}\n")
            f.write(f"basis_path: {basis_path}\n")
            f.write(f"u_ref_path: {u_ref_path}\n")
            f.write(f"solver_variant: {solver_variant}\n")

        if save_arrays:
            np.save(point_out / f"{point.key}_{tag}_n{n_primary}_{level_tag}_qN.npy", q_full)
            np.save(point_out / f"{point.key}_{tag}_n{n_primary}_{level_tag}_snaps.npy", snaps_cmp)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "point",
        "mu1",
        "mu2",
        "n_primary",
        "n_tot",
        "requested_secondary_error_percent",
        "actual_secondary_error_percent",
        "ann_secondary_error_percent",
        "state_error_percent_vs_hdm",
        "state_error_percent_vs_linear_prom",
        "primary_q_error_percent_vs_linear_prom",
        "total_q_error_percent_vs_linear_prom",
        "n_time_used",
        "online_solve_elapsed_s",
        "num_iterations",
        "jac_time_s",
        "res_time_s",
        "ls_time_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def plot_summary(rows: list[dict[str, object]], output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    colors = {
        "verification": "#4c78a8",
        "offgrid1": "#f58518",
        "offgrid2": "#54a24b",
        "extrapolation20pct": "#b279a2",
    }
    labels = {
        "verification": r"$\mu^{(v)}$",
        "offgrid1": r"$\mu^{(1)}$",
        "offgrid2": r"$\mu^{(2)}$",
        "extrapolation20pct": r"$\mu^{(3)}$",
    }
    rows_by_point = {key: [r for r in rows if r["point"] == key] for key in POINTS}

    def _plot(ykey: str, ylabel: str, out_name: str) -> Path:
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for key, subset in rows_by_point.items():
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: float(r["actual_secondary_error_percent"]))
            x = [float(r["actual_secondary_error_percent"]) for r in subset]
            y = [float(r[ykey]) for r in subset]
            ax.plot(x, y, marker="o", linewidth=2.0, color=colors[key], label=labels[key])
        ax.set_xlabel(r"imposed secondary error $\|e_s\|_F/\|q_s^{PROM}\|_F$ (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.32)
        ax.legend(frameon=True)
        fig.tight_layout()
        out = output_root / out_name
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out

    state_fig = _plot(
        "state_error_percent_vs_hdm",
        r"state relative error against HDM (%)",
        "case2_secondary_sensitivity_state_error.png",
    )
    primary_fig = _plot(
        "primary_q_error_percent_vs_linear_prom",
        r"primary coefficient error against linear PROM (%)",
        "case2_secondary_sensitivity_primary_q_error.png",
    )
    return state_fig, primary_fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporary Case-2 sensitivity to controlled secondary-coordinate errors."
    )
    parser.add_argument("--points", nargs="+", default=["all"], help="all or any of: verification offgrid1 offgrid2 extrapolation20pct")
    parser.add_argument("--levels", nargs="+", default=["0", "1", "3", "5", "10", "15", "20"])
    parser.add_argument("--n-primary", type=int, default=N_PRIMARY_DEFAULT)
    parser.add_argument("--n-tot", type=int, default=NTOT_DEFAULT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument("--solver-variant", choices=("plain", "pg"), default="plain")
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1e-12)
    parser.add_argument("--prom-root", type=Path, default=THIS_DIR / "Results_Paper" / "mlspg_prom_main")
    parser.add_argument("--basis-path", type=Path, default=THIS_DIR / "Results_Paper" / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy")
    parser.add_argument("--u-ref-path", type=Path, default=THIS_DIR / "Results_Paper" / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy")
    parser.add_argument("--output-root", type=Path, default=THIS_DIR / "Results_Paper" / "tmp_case2_secondary_sensitivity")
    parser.add_argument("--save-arrays", action="store_true", help="Save qN/snaps for every perturbation level. Off by default to save space.")
    parser.add_argument(
        "--include-ann-level",
        action="store_true",
        help="Also run each point at its actual master-ANN secondary error level.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing summaries.")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    points = resolve_points(args.points)
    levels = parse_levels(args.levels)
    output_root = args.output_root.expanduser().resolve()
    print("[case2-secondary-sensitivity] points:", " ".join(p.key for p in points))
    print("[case2-secondary-sensitivity] levels:", " ".join(f"{v:g}" for v in levels))
    print("[case2-secondary-sensitivity] output:", output_root)
    print("[case2-secondary-sensitivity] solver:", args.solver_variant)
    if args.plan_only:
        return

    all_rows: list[dict[str, object]] = []
    for point in points:
        rows = run_point(
            point=point,
            levels=levels,
            prom_root=args.prom_root.expanduser().resolve(),
            basis_path=args.basis_path.expanduser().resolve(),
            u_ref_path=args.u_ref_path.expanduser().resolve(),
            output_root=output_root,
            n_primary=int(args.n_primary),
            n_tot=int(args.n_tot),
            dt=float(args.dt),
            num_steps=int(args.num_steps),
            max_its=int(args.max_its),
            relnorm_cutoff=float(args.relnorm_cutoff),
            min_delta=float(args.min_delta),
            solver_variant=str(args.solver_variant),
            linear_solver=str(args.linear_solver),
            normal_eq_reg=float(args.normal_eq_reg),
            save_arrays=bool(args.save_arrays),
            force=bool(args.force),
            include_ann_level=bool(args.include_ann_level),
        )
        all_rows.extend(rows)

    # Include previously completed rows if this was a resumed run.
    completed_rows: list[dict[str, object]] = []
    for summary in sorted(output_root.glob("*/*_summary.txt")):
        row: dict[str, object] = {}
        for line in summary.read_text().splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k in {
                "mu1",
                "mu2",
                "requested_secondary_error_percent",
                "actual_secondary_error_percent",
                "ann_secondary_error_percent",
                "state_error_percent_vs_hdm",
                "state_error_percent_vs_linear_prom",
                "primary_q_error_percent_vs_linear_prom",
                "total_q_error_percent_vs_linear_prom",
                "online_solve_elapsed_s",
                "jac_time_s",
                "res_time_s",
                "ls_time_s",
            }:
                row[k] = float(v)
            elif k in {"n_primary", "n_tot", "n_time_used", "num_iterations"}:
                row[k] = int(float(v))
            elif k == "point":
                row[k] = v
        if "point" in row:
            completed_rows.append(row)

    csv_path = output_root / "case2_secondary_sensitivity_summary.csv"
    write_csv(csv_path, completed_rows)
    state_fig, primary_fig = plot_summary(completed_rows, output_root)
    print(f"[case2-secondary-sensitivity] csv: {csv_path}")
    print(f"[case2-secondary-sensitivity] state plot: {state_fig}")
    print(f"[case2-secondary-sensitivity] primary-q plot: {primary_fig}")


if __name__ == "__main__":
    main()
