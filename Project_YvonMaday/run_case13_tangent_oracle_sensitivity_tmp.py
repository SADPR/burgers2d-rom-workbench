#!/usr/bin/env python3
"""Temporary tangent-preserving oracle diagnostics for PROM--ANN Cases 1 and 3.

The Case-2 diagnostic prescribes a parameter--time secondary-coordinate table,
which is exactly the Case-2 closure.  Cases 1 and 3 are different: their ANN
tails depend on the online primary iterate, and the resulting ANN Jacobian is
part of the LSPG tangent.  This script therefore perturbs only the closure
value along the linear-PROM reference path while retaining the original ANN
derivative with respect to the online coordinate.

For a reference trajectory q_ref=[q_p_ref; q_s_ref], write d(t) for the
actual closure error evaluated at q_p_ref(t).  At a requested relative tail
error rho, the diagnostic closure is

    F_rho(q, mu, t) = F(q, mu, t) + (rho/rho_ann - 1) d(t),

where rho_ann = ||d||_F / ||q_s_ref||_F.  Thus rho=0 gives the oracle tail on
the reference trajectory, rho=rho_ann recovers the unmodified ANN exactly,
and the q-derivative is always dF/dq.  The diagnostic is intentionally not a
new production closure, especially for the state-only Case 1 map.

All output is written under a temporary Results_Paper directory.  The script
does not overwrite production Case-1 or Case-3 runs.
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
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D,
    inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case3,
)
from run_prom_ann_case_1 import (  # noqa: E402
    ANNVectorWrapper as Case1ANNVectorWrapper,
    _load_case1_model,
)
from run_prom_ann_case_3 import (  # noqa: E402
    ANNVectorWrapper as Case3ANNVectorWrapper,
    _load_case3_model,
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


class Case1TangentOracle(nn.Module):
    """Case-1 closure with a fixed-in-time additive oracle correction.

    ``set_time`` is called by the generic Case-1 PROM solver immediately
    before each time step.  The table term is independent of q, so its
    derivative is zero and the original ANN tangent is preserved.
    """

    def __init__(self, base: nn.Module, correction_table: np.ndarray, dt: float, scale: float):
        super().__init__()
        table = np.asarray(correction_table, dtype=np.float32)
        if table.ndim != 2:
            raise ValueError(f"correction_table must be 2D, got {table.shape}")
        self.base = base
        self.register_buffer("correction_table", torch.from_numpy(table))
        self.dt = float(dt)
        self.n_t = int(table.shape[1])
        self.scale = float(scale)
        self._time_index = 0

    def set_time(self, time_value: float) -> None:
        self._time_index = int(np.clip(round(float(time_value) / self.dt), 0, self.n_t - 1))

    def forward(self, q_primary: torch.Tensor) -> torch.Tensor:
        out = self.base(q_primary).reshape(-1)
        correction = self.correction_table[:, self._time_index].to(dtype=out.dtype, device=out.device)
        return out + self.scale * correction


class Case3TangentOracle(nn.Module):
    """Case-3 closure with an additive correction selected by its time input."""

    def __init__(self, base: nn.Module, correction_table: np.ndarray, dt: float, scale: float):
        super().__init__()
        table = np.asarray(correction_table, dtype=np.float32)
        if table.ndim != 2:
            raise ValueError(f"correction_table must be 2D, got {table.shape}")
        self.base = base
        self.register_buffer("correction_table", torch.from_numpy(table))
        self.dt = float(dt)
        self.n_t = int(table.shape[1])
        self.scale = float(scale)

    def _time_index(self, t: torch.Tensor) -> torch.Tensor:
        idx = torch.round(t / self.dt).long()
        return torch.clamp(idx, min=0, max=self.n_t - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if x.ndim == 1:
            idx = self._time_index(x[-1])
            correction = self.correction_table[:, idx].to(dtype=out.dtype, device=out.device)
            return out.reshape(-1) + self.scale * correction
        if x.ndim == 2:
            idx = self._time_index(x[:, -1])
            correction = self.correction_table.index_select(1, idx).T.to(dtype=out.dtype, device=out.device)
            return out + self.scale * correction
        raise ValueError(f"Unsupported Case-3 ANN input shape: {tuple(x.shape)}")


def mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def infer_square_grid(n_state: int) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(n_state // 2)
    n_side = int(round(np.sqrt(n_cells)))
    if n_side * n_side != n_cells:
        raise ValueError(f"Cannot infer a square spatial grid from state size {n_state}.")
    grid = np.linspace(0.0, 100.0, n_side + 1, dtype=np.float64)
    return grid, grid.copy()


def parse_levels(values: list[str]) -> list[float]:
    levels: list[float] = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            if item:
                levels.append(float(item))
    if not levels or any(value < 0.0 for value in levels):
        raise ValueError(f"Perturbation levels must be nonnegative and nonempty: {levels}")
    return sorted(dict.fromkeys(levels))


def resolve_points(point_keys: list[str]) -> list[Point]:
    keys: list[str] = []
    for raw in point_keys:
        keys.extend(item for item in str(raw).replace(",", " ").split() if item)
    if not keys or keys == ["all"]:
        keys = list(POINTS)
    result: list[Point] = []
    for key in keys:
        if key not in POINTS:
            raise ValueError(f"Unknown point '{key}'. Valid: all, {', '.join(POINTS)}")
        result.append(Point(key, *POINTS[key]))
    return result


def load_array(path: Path, *, ndim: int, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}D, got {value.shape}: {path}")
    return value


def align_time(*arrays: np.ndarray) -> tuple[list[np.ndarray], int]:
    n_t = min(int(array.shape[1]) for array in arrays)
    return [np.asarray(array)[:, :n_t] for array in arrays], n_t


def relative_percent(error: np.ndarray, reference: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(error) / (np.linalg.norm(reference) + 1.0e-30))


def load_base_model(case: str, model_path: Path, device: torch.device) -> tuple[nn.Module, int, int]:
    if case == "case1":
        model, n_primary, n_secondary = _load_case1_model(str(model_path), device=device)
        return Case1ANNVectorWrapper(model).to(device).eval(), int(n_primary), int(n_secondary)
    if case == "case3":
        model, _in_dim, n_primary, n_secondary, _ckpt = _load_case3_model(str(model_path), device=device)
        return Case3ANNVectorWrapper(model).to(device).eval(), int(n_primary), int(n_secondary)
    raise ValueError(f"Unsupported case '{case}'")


def evaluate_base_tail(
    *,
    case: str,
    base: nn.Module,
    q_primary: np.ndarray,
    mu: tuple[float, float],
    dt: float,
) -> np.ndarray:
    q_primary = np.asarray(q_primary, dtype=np.float64)
    n_t = int(q_primary.shape[1])
    device = next(base.parameters()).device
    tails: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(n_t):
            qk = torch.as_tensor(q_primary[:, k], dtype=torch.float32, device=device)
            if case == "case1":
                out = base(qk)
            else:
                inp = torch.cat(
                    (
                        qk,
                        torch.tensor([mu[0], mu[1], k * float(dt)], dtype=torch.float32, device=device),
                    )
                )
                out = base(inp)
            tails.append(out.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False))
    return np.column_stack(tails)


def build_oracle_closure(
    *,
    case: str,
    base: nn.Module,
    correction_table: np.ndarray,
    dt: float,
    requested_percent: float,
    ann_percent: float,
) -> tuple[nn.Module, float]:
    if ann_percent <= 1.0e-13:
        if requested_percent > 1.0e-13:
            raise RuntimeError("The actual ANN tail error is zero; a nonzero perturbation direction is undefined.")
        scale = -1.0
    else:
        scale = float(requested_percent) / float(ann_percent) - 1.0
    # Avoid roundoff in the native-ANN reproduction check.
    if abs(scale) < 1.0e-14:
        scale = 0.0
    if case == "case1":
        return Case1TangentOracle(base, correction_table, dt=dt, scale=scale), scale
    return Case3TangentOracle(base, correction_table, dt=dt, scale=scale), scale


def reconstruct_full_coordinates(
    *,
    case: str,
    closure: nn.Module,
    q_primary: np.ndarray,
    mu: tuple[float, float],
    dt: float,
) -> np.ndarray:
    q_primary = np.asarray(q_primary, dtype=np.float64)
    device = next(closure.parameters()).device
    tails: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(q_primary.shape[1]):
            qk = torch.as_tensor(q_primary[:, k], dtype=torch.float32, device=device)
            if case == "case1":
                closure.set_time(k * float(dt))
                out = closure(qk)
            else:
                inp = torch.cat(
                    (
                        qk,
                        torch.tensor([mu[0], mu[1], k * float(dt)], dtype=torch.float32, device=device),
                    )
                )
                out = closure(inp)
            tails.append(out.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False))
    return np.vstack((q_primary, np.column_stack(tails)))


def solve_one_level(
    *,
    case: str,
    closure: nn.Module,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    w0: np.ndarray,
    dt: float,
    num_steps: int,
    mu: tuple[float, float],
    v: np.ndarray,
    vbar: np.ndarray,
    u_ref: np.ndarray,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    linear_solver: str,
    normal_eq_reg: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]:
    t0 = time.perf_counter()
    common = dict(
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps,
        mu=list(mu),
        ann_model=closure,
        ref=None,
        basis=v,
        basis2=vbar,
        u_ref=u_ref,
        max_its=max_its,
        relnorm_cutoff=relnorm_cutoff,
        min_delta=min_delta,
        linear_solver=linear_solver,
        normal_eq_reg=normal_eq_reg,
        return_red_coords=True,
    )
    if case == "case1":
        snaps, q_primary, stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D(**common)
    else:
        snaps, q_primary, stats = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case3(**common)
    return snaps, q_primary, stats, time.perf_counter() - t0


def run_point(
    *,
    case: str,
    model_path: Path,
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
    linear_solver: str,
    normal_eq_reg: float,
    include_ann_level: bool,
    save_arrays: bool,
    force: bool,
    device: torch.device,
) -> list[dict[str, object]]:
    tag = mu_tag(point.mu1, point.mu2)
    point_out = output_root / point.key
    point_out.mkdir(parents=True, exist_ok=True)

    linear_dir = prom_root / "Runs" / "Linear" / f"linear_prom_{tag}_ntot{n_tot}"
    q_linear = load_array(linear_dir / "qN.npy", ndim=2, label="linear PROM qN")[:n_tot]
    linear_snaps = load_array(linear_dir / "rom_snaps.npy", ndim=2, label="linear PROM snapshots")
    basis = load_array(basis_path, ndim=2, label="basis")[:, :n_tot]
    u_ref = load_array(u_ref_path, ndim=1, label="u_ref").reshape(-1)
    if basis.shape[0] != u_ref.size:
        raise ValueError(f"basis/u_ref mismatch: {basis.shape[0]} vs {u_ref.size}")

    base, n_primary_model, n_secondary_model = load_base_model(case, model_path, device)
    if n_primary_model != n_primary or n_primary + n_secondary_model != n_tot:
        raise ValueError(
            f"{case} checkpoint dimensions are n_p={n_primary_model}, n_s={n_secondary_model}; "
            f"requested n_primary={n_primary}, n_tot={n_tot}."
        )

    (q_linear, linear_snaps), n_t = align_time(q_linear, linear_snaps)
    num_steps_eff = min(int(num_steps), n_t - 1)
    q_linear = q_linear[:, : num_steps_eff + 1]
    linear_snaps = linear_snaps[:, : num_steps_eff + 1]
    q_primary_ref = q_linear[:n_primary]
    q_secondary_ref = q_linear[n_primary:n_tot]
    base_secondary = evaluate_base_tail(
        case=case,
        base=base,
        q_primary=q_primary_ref,
        mu=(point.mu1, point.mu2),
        dt=dt,
    )
    correction_table = base_secondary - q_secondary_ref
    ann_percent = relative_percent(correction_table, q_secondary_ref)

    # At the measured ANN error the additive correction is identically zero,
    # hence the diagnostic closure must be the production ANN for every input.
    native_closure, native_scale = build_oracle_closure(
        case=case,
        base=base,
        correction_table=correction_table,
        dt=dt,
        requested_percent=ann_percent,
        ann_percent=ann_percent,
    )
    probe = torch.zeros(n_primary, dtype=torch.float32, device=device)
    with torch.no_grad():
        if case == "case1":
            native_closure.set_time(0.0)
            reproduction_max_abs = float(torch.max(torch.abs(native_closure(probe) - base(probe))).cpu())
        else:
            xprobe = torch.cat((probe, torch.tensor([point.mu1, point.mu2, 0.0], dtype=torch.float32, device=device)))
            reproduction_max_abs = float(torch.max(torch.abs(native_closure(xprobe) - base(xprobe))).cpu())
    if native_scale != 0.0 or reproduction_max_abs > 1.0e-10:
        raise RuntimeError(
            f"Native-ANN reproduction check failed for {case}/{point.key}: "
            f"scale={native_scale}, max_abs={reproduction_max_abs:.3e}"
        )

    v = basis[:, :n_primary]
    vbar = basis[:, n_primary:n_tot]
    w0 = linear_snaps[:, 0].copy()
    grid_x, grid_y = infer_square_grid(w0.size)
    hdm_snaps = load_or_compute_snaps(
        mu=[point.mu1, point.mu2],
        grid_x=grid_x,
        grid_y=grid_y,
        w0=w0,
        dt=dt,
        num_steps=num_steps_eff,
        snap_folder=REPO_DIR / "Results" / "param_snaps",
    )
    (hdm_snaps, linear_snaps), _ = align_time(hdm_snaps, linear_snaps)

    requested_levels = list(levels)
    if include_ann_level:
        requested_levels.append(ann_percent)
    requested_levels = sorted(dict.fromkeys(round(float(value), 12) for value in requested_levels))
    rows: list[dict[str, object]] = []
    for requested_percent in requested_levels:
        level_tag = f"rho{requested_percent:.6f}".replace(".", "p")
        summary_path = point_out / f"{case}_{tag}_n{n_primary}_{level_tag}_summary.txt"
        if summary_path.exists() and not force:
            print(f"[skip] {case} {point.key} rho={requested_percent:g}%: {summary_path}")
            continue

        closure, scale = build_oracle_closure(
            case=case,
            base=base,
            correction_table=correction_table,
            dt=dt,
            requested_percent=requested_percent,
            ann_percent=ann_percent,
        )
        closure = closure.to(device).eval()
        reference_secondary = q_secondary_ref + (1.0 + scale) * correction_table
        realized_percent = relative_percent(reference_secondary - q_secondary_ref, q_secondary_ref)
        print(
            f"[run] {case} {point.key} mu=({point.mu1:.3f},{point.mu2:.4f}) "
            f"tail={realized_percent:.3f}% (requested={requested_percent:.3f}%)"
        )
        snaps, q_primary, stats, elapsed = solve_one_level(
            case=case,
            closure=closure,
            grid_x=grid_x,
            grid_y=grid_y,
            w0=w0,
            dt=dt,
            num_steps=num_steps_eff,
            mu=(point.mu1, point.mu2),
            v=v,
            vbar=vbar,
            u_ref=u_ref,
            max_its=max_its,
            relnorm_cutoff=relnorm_cutoff,
            min_delta=min_delta,
            linear_solver=linear_solver,
            normal_eq_reg=normal_eq_reg,
        )
        q_full = reconstruct_full_coordinates(
            case=case,
            closure=closure,
            q_primary=q_primary,
            mu=(point.mu1, point.mu2),
            dt=dt,
        )
        (hdm_cmp, linear_cmp, snaps_cmp, q_ref_cmp, q_cmp), n_t_used = align_time(
            hdm_snaps,
            linear_snaps,
            snaps,
            q_linear,
            q_full,
        )
        q_primary_cmp = q_cmp[:n_primary]
        q_primary_ref_cmp = q_ref_cmp[:n_primary]
        q_secondary_cmp = q_cmp[n_primary:n_tot]
        q_secondary_ref_cmp = q_ref_cmp[n_primary:n_tot]
        row = {
            "case": case,
            "point": point.key,
            "mu1": point.mu1,
            "mu2": point.mu2,
            "n_primary": n_primary,
            "n_tot": n_tot,
            "requested_secondary_error_percent": requested_percent,
            "actual_secondary_error_percent": relative_percent(q_secondary_cmp - q_secondary_ref_cmp, q_secondary_ref_cmp),
            "reference_secondary_error_percent": realized_percent,
            "ann_secondary_error_percent": ann_percent,
            "state_error_percent_vs_hdm": relative_percent(hdm_cmp - snaps_cmp, hdm_cmp),
            "state_error_percent_vs_linear_prom": relative_percent(linear_cmp - snaps_cmp, linear_cmp),
            "primary_q_error_percent_vs_linear_prom": relative_percent(q_primary_cmp - q_primary_ref_cmp, q_primary_ref_cmp),
            "total_q_error_percent_vs_linear_prom": relative_percent(q_cmp - q_ref_cmp, q_ref_cmp),
            "n_time_used": n_t_used,
            "online_solve_elapsed_s": elapsed,
            "num_iterations": stats[0],
            "jac_time_s": stats[1],
            "res_time_s": stats[2],
            "ls_time_s": stats[3],
            "native_ann_reproduction_max_abs": reproduction_max_abs,
        }
        rows.append(row)
        with summary_path.open("w", encoding="utf-8") as handle:
            for key, value in row.items():
                handle.write(f"{key}: {value}\n")
            handle.write(f"model_path: {model_path}\n")
            handle.write(f"linear_qn_path: {linear_dir / 'qN.npy'}\n")
            handle.write(f"linear_snaps_path: {linear_dir / 'rom_snaps.npy'}\n")
            handle.write(f"basis_path: {basis_path}\n")
            handle.write(f"u_ref_path: {u_ref_path}\n")
            handle.write("diagnostic: tangent_preserving_oracle\n")
            handle.write(
                "interpretation: rho=0 makes the closure tail exact only along the linear-PROM "
                "reference path; it does not replace the nonlinear tangent by the linear tangent.\n"
            )
        if save_arrays:
            np.save(point_out / f"{case}_{tag}_n{n_primary}_{level_tag}_qN.npy", q_cmp)
            np.save(point_out / f"{case}_{tag}_n{n_primary}_{level_tag}_snaps.npy", snaps_cmp)
    return rows


def read_completed_rows(case_root: Path) -> list[dict[str, object]]:
    numeric_float = {
        "mu1", "mu2", "requested_secondary_error_percent", "actual_secondary_error_percent",
        "reference_secondary_error_percent", "ann_secondary_error_percent", "state_error_percent_vs_hdm",
        "state_error_percent_vs_linear_prom", "primary_q_error_percent_vs_linear_prom",
        "total_q_error_percent_vs_linear_prom", "online_solve_elapsed_s", "jac_time_s", "res_time_s",
        "ls_time_s", "native_ann_reproduction_max_abs",
    }
    numeric_int = {"n_primary", "n_tot", "n_time_used", "num_iterations"}
    rows: list[dict[str, object]] = []
    for summary in sorted(case_root.glob("*/*_summary.txt")):
        row: dict[str, object] = {}
        for line in summary.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in numeric_float:
                row[key] = float(value)
            elif key in numeric_int:
                row[key] = int(float(value))
            elif key in {"case", "point"}:
                row[key] = value
        if "case" in row and "point" in row:
            rows.append(row)
    return rows


CSV_FIELDS = [
    "case", "point", "mu1", "mu2", "n_primary", "n_tot",
    "requested_secondary_error_percent", "actual_secondary_error_percent",
    "reference_secondary_error_percent", "ann_secondary_error_percent",
    "state_error_percent_vs_hdm", "state_error_percent_vs_linear_prom",
    "primary_q_error_percent_vs_linear_prom", "total_q_error_percent_vs_linear_prom",
    "n_time_used", "online_solve_elapsed_s", "num_iterations", "jac_time_s", "res_time_s",
    "ls_time_s", "native_ann_reproduction_max_abs",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def plot_summary(case: str, rows: list[dict[str, object]], output_root: Path) -> Path:
    colors = {
        "verification": "#1f77b4",
        "offgrid1": "#ff7f0e",
        "offgrid2": "#2ca02c",
        "extrapolation20pct": "#9467bd",
    }
    labels = {
        "verification": r"$\mu^{(v)}$",
        "offgrid1": r"$\mu^{(1)}$",
        "offgrid2": r"$\mu^{(2)}$",
        "extrapolation20pct": r"$\mu^{(3)}$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True)
    targets = (
        ("state_error_percent_vs_hdm", r"state relative error against HDM (\%)"),
        ("primary_q_error_percent_vs_linear_prom", r"primary-coordinate error against linear PROM (\%)"),
    )
    for axis, (field, ylabel) in zip(axes, targets):
        for key in POINTS:
            subset = sorted((row for row in rows if row["point"] == key), key=lambda row: float(row["reference_secondary_error_percent"]))
            if not subset:
                continue
            x = np.asarray([float(row["reference_secondary_error_percent"]) for row in subset])
            y = np.asarray([float(row[field]) for row in subset])
            axis.plot(x, y, marker="o", markersize=3.5, linewidth=1.8, color=colors[key], label=labels[key])
            ann_row = min(subset, key=lambda row: abs(float(row["requested_secondary_error_percent"]) - float(row["ann_secondary_error_percent"])))
            axis.scatter(
                [float(ann_row["reference_secondary_error_percent"])], [float(ann_row[field])],
                marker="*", s=100, color=colors[key], edgecolor="black", linewidth=0.55, zorder=5,
            )
        axis.set_xlabel(r"reference-path closure-tail error (\%)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.28)
    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle(f"{case.replace('case', 'Case ')} tangent-preserving oracle tail diagnostic", y=1.03)
    out = output_root / f"{case}_tangent_oracle_sensitivity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary tangent-preserving oracle diagnostics for PROM--ANN Cases 1 and 3.")
    parser.add_argument("--case", choices=("case1", "case3", "all"), default="all")
    parser.add_argument("--points", nargs="+", default=["all"])
    parser.add_argument("--levels", nargs="+", default=["0", "1", "3", "5", "10", "15", "20", "30", "50"])
    parser.add_argument("--n-primary", type=int, default=N_PRIMARY_DEFAULT)
    parser.add_argument("--n-tot", type=int, default=NTOT_DEFAULT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1.0e-5)
    parser.add_argument("--min-delta", type=float, default=1.0e-2)
    parser.add_argument("--linear-solver", choices=("lstsq", "normal_eq"), default="lstsq")
    parser.add_argument("--normal-eq-reg", type=float, default=1.0e-12)
    parser.add_argument("--prom-root", type=Path, default=THIS_DIR / "Results_Paper" / "mlspg_prom_main")
    parser.add_argument("--basis-path", type=Path, default=THIS_DIR / "Results_Paper" / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy")
    parser.add_argument("--u-ref-path", type=Path, default=THIS_DIR / "Results_Paper" / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy")
    parser.add_argument("--case1-model", type=Path, default=None)
    parser.add_argument("--case3-model", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=THIS_DIR / "Results_Paper" / "tmp_case13_tangent_oracle_sensitivity")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--include-ann-level", action="store_true")
    parser.add_argument("--save-arrays", action="store_true", help="Save every diagnostic qN/state trajectory; disabled by default to save space.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    points = resolve_points(args.points)
    levels = parse_levels(args.levels)
    prom_root = args.prom_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_root = prom_root / "Stage3" / "models"
    model_paths = {
        "case1": (args.case1_model or model_root / "case1_ann_ntot151_best.pt").expanduser().resolve(),
        "case3": (args.case3_model or model_root / "case3_ann_ntot151_best.pt").expanduser().resolve(),
    }
    cases = ("case1", "case3") if args.case == "all" else (args.case,)

    print("[case13-tangent-oracle] cases:", " ".join(cases))
    print("[case13-tangent-oracle] points:", " ".join(point.key for point in points))
    print("[case13-tangent-oracle] levels (%):", " ".join(f"{value:g}" for value in levels))
    print("[case13-tangent-oracle] PROM root:", prom_root)
    print("[case13-tangent-oracle] output root:", output_root)
    print("[case13-tangent-oracle] device:", device)
    for case in cases:
        print(f"[case13-tangent-oracle] {case} model: {model_paths[case]}")
        if not model_paths[case].exists():
            raise FileNotFoundError(f"Missing {case} checkpoint: {model_paths[case]}")
    if args.plan_only:
        return

    for case in cases:
        case_root = output_root / case
        for point in points:
            run_point(
                case=case,
                model_path=model_paths[case],
                point=point,
                levels=levels,
                prom_root=prom_root,
                basis_path=args.basis_path.expanduser().resolve(),
                u_ref_path=args.u_ref_path.expanduser().resolve(),
                output_root=case_root,
                n_primary=int(args.n_primary),
                n_tot=int(args.n_tot),
                dt=float(args.dt),
                num_steps=int(args.num_steps),
                max_its=int(args.max_its),
                relnorm_cutoff=float(args.relnorm_cutoff),
                min_delta=float(args.min_delta),
                linear_solver=str(args.linear_solver),
                normal_eq_reg=float(args.normal_eq_reg),
                include_ann_level=bool(args.include_ann_level),
                save_arrays=bool(args.save_arrays),
                force=bool(args.force),
                device=device,
            )
        completed = read_completed_rows(case_root)
        csv_path = case_root / f"{case}_tangent_oracle_sensitivity_summary.csv"
        write_csv(csv_path, completed)
        figure_path = plot_summary(case, completed, case_root)
        print(f"[case13-tangent-oracle] {case} csv: {csv_path}")
        print(f"[case13-tangent-oracle] {case} figure: {figure_path}")


if __name__ == "__main__":
    main()
