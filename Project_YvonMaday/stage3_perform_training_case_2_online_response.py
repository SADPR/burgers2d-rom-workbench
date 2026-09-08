#!/usr/bin/env python3
"""Train the Case--2 master map with the exact online primary-coordinate loss.

For a parameter trajectory, the network supplies the secondary coordinates
``q_s = M_theta(mu, t)``.  The existing full-residual Case--2 LSPG solve then
defines the primary coordinates ``q_p = S(q_s; mu)``.  This script minimizes

    ||q_s - q_s^ref||^2 / ||q_s^ref||^2
  + alpha ||S(q_s; mu) - q_p^ref||^2 / ||q_p^ref||^2.

The derivative of the second term is obtained by discrete implicit
differentiation of the converged LSPG normal equations, not by unrolling
Gauss--Newton iterations.  The online runner is intentionally unchanged.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0
from burgers.core import get_ops, inviscid_burgers_exact_jac2D, inviscid_burgers_res2D
from run_prom_ann_case_2 import _load_case2_model


@dataclasses.dataclass(frozen=True)
class Trajectory:
    mu: np.ndarray
    t: np.ndarray
    q: np.ndarray  # Shape (n_tot, n_steps + 1).


def _truncate_trajectory(trajectory: Trajectory, num_steps: int) -> Trajectory:
    """Return an initial time window, used only for the cheap gradient audit."""
    if not (1 <= num_steps < trajectory.t.size):
        raise ValueError(f"Invalid audit window {num_steps} for {trajectory.t.size - 1} steps.")
    return Trajectory(
        mu=trajectory.mu,
        t=trajectory.t[: num_steps + 1].copy(),
        q=trajectory.q[:, : num_steps + 1].copy(),
    )


def _load_trajectories(dataset_dir: Path, ntot: int) -> list[Trajectory]:
    per_mu = dataset_dir / "per_mu"
    if not per_mu.is_dir():
        raise FileNotFoundError(f"Missing per-parameter data directory: {per_mu}")
    trajectories: list[Trajectory] = []
    for path in sorted(per_mu.glob("mu1_*")):
        mu = np.asarray(np.load(path / "mu.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
        t = np.asarray(np.load(path / "t.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
        q = np.asarray(np.load(path / "qN.npy", allow_pickle=False), dtype=np.float64)
        if mu.size != 2 or q.ndim != 2 or q.shape != (ntot, t.size):
            raise ValueError(
                f"Invalid trajectory in {path}: mu={mu.shape}, t={t.shape}, q={q.shape}; "
                f"expected ({ntot}, {t.size})."
            )
        trajectories.append(Trajectory(mu=mu, t=t, q=q))
    if not trajectories:
        raise RuntimeError(f"No trajectories found under {per_mu}")
    return trajectories


def _trajectory_inputs(trajectory: Trajectory) -> np.ndarray:
    return np.column_stack(
        (
            np.full(trajectory.t.size, trajectory.mu[0], dtype=np.float64),
            np.full(trajectory.t.size, trajectory.mu[1], dtype=np.float64),
            trajectory.t,
        )
    )


def _normal_residual_measure(
    q_primary: np.ndarray,
    tail: np.ndarray,
    trajectory: Trajectory,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
) -> tuple[float, float]:
    """Return median and maximum dimensionless LSPG normal-residual measures."""
    dxec, dyec, jdx, jdy, eye = operators
    values: list[float] = []
    previous = u_ref + primary_basis @ q_primary[:, 0] + secondary_basis @ tail[0]
    for step in range(1, q_primary.shape[1]):
        state = u_ref + primary_basis @ q_primary[:, step] + secondary_basis @ tail[step]
        residual = inviscid_burgers_res2D(state, GRID_X, GRID_Y, DT, previous, trajectory.mu, dxec, dyec)
        tangent = inviscid_burgers_exact_jac2D(state, DT, jdx, jdy, eye) @ primary_basis
        value = np.linalg.norm(tangent.T @ residual) / (
            np.linalg.norm(tangent, ord="fro") * np.linalg.norm(residual) + 1.0e-30
        )
        values.append(float(value))
        previous = state
    return float(np.median(values)), float(np.max(values))


def _solve_primary(
    tail: np.ndarray,
    trajectory: Trajectory,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
    primary_projector: np.ndarray,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    quiet: bool,
) -> tuple[np.ndarray, tuple[int, float, float, float]]:
    """Solve the same Case--2 full-residual LSPG problem as the online runner.

    The production implementation obtains each Gauss--Newton step by a dense
    least-squares SVD of ``J V_p``.  Here the primary dimension is ten, so the
    algebraically identical normal system is only 10 by 10.  Solving that
    system avoids repeated tall-SVD work while leaving the residual, tangent,
    stopping criteria, and reduced state recursion unchanged.
    """
    del quiet  # This implementation is silent by construction.
    dxec, dyec, jdx, jdy, eye = operators
    w0 = np.asarray(W0, dtype=np.float64).reshape(-1)
    nsteps = tail.shape[0]
    nprimary = primary_basis.shape[1]
    q_primary = np.zeros((nprimary, nsteps), dtype=np.float64)
    y = primary_projector @ (w0 - u_ref - secondary_basis @ tail[0])
    current = u_ref + primary_basis @ y + secondary_basis @ tail[0]
    previous = current.copy()
    q_primary[:, 0] = y
    num_its = 0
    jacobian_seconds = 0.0
    residual_seconds = 0.0
    linear_seconds = 0.0

    for step in range(1, nsteps):
        yk = y.copy()
        state = u_ref + primary_basis @ yk + secondary_basis @ tail[step]
        t0 = time.perf_counter()
        residual = inviscid_burgers_res2D(state, GRID_X, GRID_Y, DT, previous, trajectory.mu, dxec, dyec)
        residual_seconds += time.perf_counter() - t0
        initial_norm = np.linalg.norm(residual) + 1.0e-30
        residual_norms: list[float] = []

        for iteration in range(max_its):
            if iteration > 0:
                t0 = time.perf_counter()
                residual = inviscid_burgers_res2D(
                    state, GRID_X, GRID_Y, DT, previous, trajectory.mu, dxec, dyec
                )
                residual_seconds += time.perf_counter() - t0
            residual_norm = np.linalg.norm(residual)
            residual_norms.append(float(residual_norm))
            if residual_norm / initial_norm < relnorm_cutoff:
                break
            if len(residual_norms) > 1:
                relative_drop = abs((residual_norms[-2] - residual_norms[-1]) / (residual_norms[-2] + 1.0e-30))
                if relative_drop < min_delta:
                    break

            t0 = time.perf_counter()
            jacobian = inviscid_burgers_exact_jac2D(state, DT, jdx, jdy, eye)
            tangent = jacobian @ primary_basis
            jacobian_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            normal_matrix = tangent.T @ tangent
            normal_rhs = -(tangent.T @ residual)
            try:
                delta = np.linalg.solve(normal_matrix, normal_rhs)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(normal_matrix, normal_rhs, rcond=1.0e-12)
            linear_seconds += time.perf_counter() - t0
            yk += delta
            state = u_ref + primary_basis @ yk + secondary_basis @ tail[step]

        y = yk
        current = state
        previous = current.copy()
        q_primary[:, step] = y
        num_its += len(residual_norms)
    return q_primary, (num_its, jacobian_seconds, residual_seconds, linear_seconds)


def _residual_hessian_contract(
    primary_basis: np.ndarray,
    residual: np.ndarray,
    jdx,
    jdy,
    dt: float,
) -> np.ndarray:
    """Return K so K.T @ dw = (dJ[dw] V_p).T r.

    The Burgers residual is quadratic, hence its Jacobian is affine.  This is
    the exact residual-Hessian contraction needed by the LSPG normal equations.
    """
    ncell = residual.size // 2
    ru, rv = residual[:ncell], residual[ncell:]
    vu, vv = primary_basis[:ncell], primary_basis[ncell:]
    dx_ru = np.asarray(jdx.T @ ru).reshape(-1, 1)
    dy_ru = np.asarray(jdy.T @ ru).reshape(-1, 1)
    dx_rv = np.asarray(jdx.T @ rv).reshape(-1, 1)
    dy_rv = np.asarray(jdy.T @ rv).reshape(-1, 1)

    ku = 0.5 * dt * vu * dx_ru + 0.25 * dt * vv * (dy_ru + dx_rv)
    kv = 0.25 * dt * vu * (dy_ru + dx_rv) + 0.5 * dt * vv * dy_rv
    return np.vstack((ku, kv))


def _normal_equation_blocks(
    q_primary: np.ndarray,
    tail: np.ndarray,
    trajectory: Trajectory,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the primary-coordinate normal-equation blocks H and F."""
    dxec, dyec, jdx, jdy, eye = operators
    nsteps = q_primary.shape[1]
    nprimary = q_primary.shape[0]
    h = np.zeros((nsteps, nprimary, nprimary), dtype=np.float64)
    f = np.zeros_like(h)

    previous = u_ref + primary_basis @ q_primary[:, 0] + secondary_basis @ tail[0]
    for step in range(1, nsteps):
        state = u_ref + primary_basis @ q_primary[:, step] + secondary_basis @ tail[step]
        residual = inviscid_burgers_res2D(state, GRID_X, GRID_Y, DT, previous, trajectory.mu, dxec, dyec)
        j_current = inviscid_burgers_exact_jac2D(state, DT, jdx, jdy, eye)
        j_previous = inviscid_burgers_exact_jac2D(previous, DT, jdx, jdy, eye)
        a = j_current @ primary_basis
        k = _residual_hessian_contract(primary_basis, residual, jdx, jdy, DT)
        h_step = a.T @ a + k.T @ primary_basis
        h[step] = 0.5 * (h_step + h_step.T)
        previous_jacobian = j_previous - 2.0 * eye
        f[step] = a.T @ (previous_jacobian @ primary_basis)
        previous = state
    return h, f


def _primary_adjoint_gradient(
    q_primary: np.ndarray,
    q_reference: np.ndarray,
    h: np.ndarray,
    f: np.ndarray,
    initial: np.ndarray,
    tail: np.ndarray,
    trajectory: Trajectory,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
) -> tuple[float, np.ndarray]:
    """Return exact primary loss and its gradient with respect to the tail path."""
    error = q_primary.T - q_reference.T
    denominator = float(np.sum(np.square(q_reference))) + 1.0e-30
    loss = float(np.sum(np.square(error)) / denominator)
    nsteps, nprimary = error.shape
    nsecondary = tail.shape[1]
    dloss_dq = 2.0 * error / denominator
    adjoint = np.zeros((nsteps, nprimary), dtype=np.float64)

    # q_0 is imposed by the initial-condition projection, not by a normal equation.
    for step in range(nsteps - 1, 0, -1):
        rhs = dloss_dq[step].copy()
        if step + 1 < nsteps:
            rhs -= f[step + 1].T @ adjoint[step + 1]
        adjoint[step] = np.linalg.solve(h[step].T, rhs)

    gradient = np.zeros((nsteps, nsecondary), dtype=np.float64)
    if nsteps > 1:
        # q_p^0 depends on q_s^0 through the initial-condition projection and
        # also enters the first normal equation through F_1.  The latter has
        # the same minus sign as every implicit sensitivity contribution.
        gradient[0] = initial.T @ (dloss_dq[0] - f[1].T @ adjoint[1])
    else:
        gradient[0] = initial.T @ dloss_dq[0]

    # Apply G_k^T lambda_k and E_k^T lambda_k directly in the full state
    # space.  This is exactly equivalent to forming J V_s, but avoids storing
    # or multiplying a 125000-by-141 dense matrix at every time step.
    dxec, dyec, jdx, jdy, eye = operators
    previous = u_ref + primary_basis @ q_primary[:, 0] + secondary_basis @ tail[0]
    for step in range(1, nsteps):
        state = u_ref + primary_basis @ q_primary[:, step] + secondary_basis @ tail[step]
        residual = inviscid_burgers_res2D(state, GRID_X, GRID_Y, DT, previous, trajectory.mu, dxec, dyec)
        j_current = inviscid_burgers_exact_jac2D(state, DT, jdx, jdy, eye)
        j_previous = inviscid_burgers_exact_jac2D(previous, DT, jdx, jdy, eye)
        a = j_current @ primary_basis
        k = _residual_hessian_contract(primary_basis, residual, jdx, jdy, DT)
        lambda_step = adjoint[step]
        g_action = j_current.T @ (a @ lambda_step) + k @ lambda_step
        e_action = (j_previous - 2.0 * eye).T @ (a @ lambda_step)
        gradient[step] -= secondary_basis.T @ g_action
        gradient[step - 1] -= secondary_basis.T @ e_action
        previous = state
    return loss, gradient


def _predict_tail(model: nn.Module, trajectory: Trajectory, nprimary: int, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(_trajectory_inputs(trajectory), dtype=torch.float32, device=device)
    return model(x)[:, nprimary:]


def _solve_and_differentiate(
    model: nn.Module,
    trajectory: Trajectory,
    nprimary: int,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
    device: torch.device,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    quiet: bool,
    primary_projector: np.ndarray,
    initial: np.ndarray,
    check_normal: bool = False,
) -> tuple[torch.Tensor, float, np.ndarray, float, float, tuple[int, float, float, float]]:
    tail_prediction = _predict_tail(model, trajectory, nprimary, device)
    tail_np = tail_prediction.detach().cpu().double().numpy()
    q_primary, stats = _solve_primary(
        tail_np, trajectory, primary_basis, secondary_basis, u_ref, operators, primary_projector,
        max_its, relnorm_cutoff, min_delta, quiet,
    )
    h, f = _normal_equation_blocks(
        q_primary, tail_np, trajectory, primary_basis, secondary_basis, u_ref, operators
    )
    primary_loss, tail_gradient = _primary_adjoint_gradient(
        q_primary, trajectory.q[:nprimary], h, f, initial, tail_np, trajectory,
        primary_basis, secondary_basis, u_ref, operators,
    )
    if check_normal:
        normal_median, normal_max = _normal_residual_measure(
            q_primary, tail_np, trajectory, primary_basis, secondary_basis, u_ref, operators
        )
    else:
        normal_median, normal_max = float("nan"), float("nan")
    return tail_prediction, primary_loss, tail_gradient, normal_median, normal_max, stats


def _loss_components(
    tail_prediction: torch.Tensor,
    trajectory: Trajectory,
    nprimary: int,
) -> torch.Tensor:
    target = torch.as_tensor(trajectory.q[nprimary:].T, dtype=torch.float32, device=tail_prediction.device)
    return torch.sum((tail_prediction - target).square()) / (torch.sum(target.square()) + 1.0e-30)


def _evaluate(
    model: nn.Module,
    trajectories: list[Trajectory],
    nprimary: int,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
    device: torch.device,
    primary_projector: np.ndarray,
    initial: np.ndarray,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    quiet: bool,
) -> dict[str, float]:
    tail_losses: list[float] = []
    primary_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for trajectory in trajectories:
            # Evaluation needs the exact value of S(tail; mu), but not its
            # derivative.  Avoid assembling the adjoint blocks here; doing so
            # would not change either loss and would dominate validation time.
            tail_prediction = _predict_tail(model, trajectory, nprimary, device)
            tail_np = tail_prediction.cpu().double().numpy()
            q_primary, _ = _solve_primary(
                tail_np, trajectory, primary_basis, secondary_basis, u_ref, operators,
                primary_projector, max_its, relnorm_cutoff, min_delta, quiet,
            )
            primary_error = q_primary - trajectory.q[:nprimary]
            primary_loss = float(
                np.sum(np.square(primary_error)) /
                (np.sum(np.square(trajectory.q[:nprimary])) + 1.0e-30)
            )
            tail_losses.append(float(_loss_components(tail_prediction, trajectory, nprimary).item()))
            primary_losses.append(float(primary_loss))
    return {
        "tail_loss": float(np.mean(tail_losses)),
        "primary_loss": float(np.mean(primary_losses)),
    }


def _directional_audit(
    model: nn.Module,
    trajectory: Trajectory,
    nprimary: int,
    primary_basis: np.ndarray,
    secondary_basis: np.ndarray,
    u_ref: np.ndarray,
    operators: tuple,
    device: torch.device,
    primary_projector: np.ndarray,
    initial: np.ndarray,
    max_its: int,
    relnorm_cutoff: float,
    min_delta: float,
    epsilon_relative: float,
    seed: int,
) -> dict[str, float]:
    """Check the adjoint primary gradient against a centered finite difference."""
    model.eval()
    with torch.no_grad():
        tail_prediction, primary_loss, gradient, normal_median, normal_max, _ = _solve_and_differentiate(
            model, trajectory, nprimary, primary_basis, secondary_basis, u_ref, operators, device,
            max_its, relnorm_cutoff, min_delta, quiet=True, primary_projector=primary_projector,
            initial=initial, check_normal=True,
        )
    tail = tail_prediction.cpu().double().numpy()
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=tail.shape)
    direction /= np.linalg.norm(direction) + 1.0e-30
    epsilon = float(epsilon_relative) * (np.linalg.norm(tail) + 1.0e-30)

    def loss_at(candidate_tail: np.ndarray) -> float:
        q_primary, _ = _solve_primary(
            candidate_tail, trajectory, primary_basis, secondary_basis, u_ref, operators,
            primary_projector, max_its, relnorm_cutoff, min_delta, quiet=True,
        )
        error = q_primary - trajectory.q[:nprimary]
        return float(np.sum(np.square(error)) / (np.sum(np.square(trajectory.q[:nprimary])) + 1.0e-30))

    plus = loss_at(tail + epsilon * direction)
    minus = loss_at(tail - epsilon * direction)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    adjoint_directional = float(np.sum(gradient * direction))
    relative = abs(finite_difference - adjoint_directional) / max(
        abs(finite_difference), abs(adjoint_directional), 1.0e-14
    )
    return {
        "base_primary_loss": float(primary_loss),
        "finite_difference": float(finite_difference),
        "adjoint_directional": float(adjoint_directional),
        "relative_error": float(relative),
        "epsilon": float(epsilon),
        "normal_median": float(normal_median),
        "normal_max": float(normal_max),
    }


def _write_summary(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}: {value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--validation-dataset-dir", required=True)
    parser.add_argument("--stage3-dir", required=True)
    parser.add_argument("--basis-path", required=True)
    parser.add_argument("--u-ref-path", required=True)
    parser.add_argument("--initialize-from-master", required=True)
    parser.add_argument("--model-name", default="case2_online_response_master_qtot_ntot151_np10_best.pt")
    parser.add_argument("--summary-name", default="case2_online_response_master_qtot_ntot151_np10_summary.txt")
    parser.add_argument("--audit-summary-name", default="case2_online_response_directional_audit_ntot151_np10.txt")
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--alpha-primary", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-its", type=int, default=20)
    # Match the production Case--2 runner.  The offline response map must use
    # the same discrete solve S(theta; mu), including its stopping rule.
    parser.add_argument("--relnorm-cutoff", type=float, default=1.0e-5)
    parser.add_argument("--min-delta", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-epsilon-relative", type=float, default=1.0e-5)
    parser.add_argument("--audit-tolerance", type=float, default=2.0e-3)
    parser.add_argument(
        "--audit-steps", type=int, default=0,
        help="Use the first N steps only for a fast directional-derivative smoke audit; 0 uses the full path.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the epoch-level optimizer/model checkpoint written beside --model-name.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.n_primary < 1 or args.epochs < 1 or args.validation_every < 1 or args.patience < 1:
        raise ValueError("Invalid primary dimension or training schedule.")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset_dir = Path(args.dataset_dir).resolve()
    validation_dir = Path(args.validation_dataset_dir).resolve()
    stage3_dir = Path(args.stage3_dir).resolve()
    model_path = stage3_dir / "models" / args.model_name
    resume_path = model_path.with_name(f"{model_path.stem}_resume.pt")
    summary_path = stage3_dir / args.summary_name
    audit_path = stage3_dir / args.audit_summary_name
    basis = np.asarray(np.load(Path(args.basis_path).resolve(), allow_pickle=False), dtype=np.float64)
    u_ref = np.asarray(np.load(Path(args.u_ref_path).resolve(), allow_pickle=False), dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or basis.shape[0] != u_ref.size:
        raise ValueError("Incompatible basis and reference state.")
    ntot = basis.shape[1]
    if not (0 < args.n_primary < ntot):
        raise ValueError(f"n_primary={args.n_primary} incompatible with n_tot={ntot}.")
    primary_basis = basis[:, :args.n_primary]
    secondary_basis = basis[:, args.n_primary:ntot]
    primary_projector = np.linalg.pinv(primary_basis, rcond=1.0e-12)
    initial = -primary_projector @ secondary_basis
    train = _load_trajectories(dataset_dir, ntot)
    validation = _load_trajectories(validation_dir, ntot)
    operators = get_ops(GRID_X, GRID_Y)

    model, output_dim, output_kind, checkpoint = _load_case2_model(str(Path(args.initialize_from_master).resolve()), device)
    if output_kind != "q_tot" or output_dim != ntot:
        raise ValueError("The exact-response trainer requires a full q_tot master ANN checkpoint.")
    model = model.to(device)
    if model_path.exists() and not args.force and not args.audit_only:
        raise FileExistsError(f"Refusing to overwrite existing model: {model_path}; pass --force.")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"Cannot resume: missing epoch checkpoint {resume_path}.")

    print(f"[Case2-OnlineResponse] train/validation trajectories = {len(train)}/{len(validation)}")
    print(f"[Case2-OnlineResponse] n_tot/n_primary/n_secondary = {ntot}/{args.n_primary}/{ntot - args.n_primary}")
    print("[Case2-OnlineResponse] objective = exact tail error + alpha * exact online primary-coordinate error")
    print("[Case2-OnlineResponse] derivative = discrete adjoint of the converged LSPG normal equations")

    audit_trajectory = train[0] if args.audit_steps == 0 else _truncate_trajectory(train[0], args.audit_steps)
    audit = _directional_audit(
        model, audit_trajectory, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
        primary_projector, initial, args.max_its, args.relnorm_cutoff, args.min_delta,
        args.audit_epsilon_relative, args.seed,
    )
    _write_summary(audit_path, audit)
    print(
        "[Case2-OnlineResponse] directional audit: "
        f"fd={audit['finite_difference']:.6e} adjoint={audit['adjoint_directional']:.6e} "
        f"relative={audit['relative_error']:.3e}; normal max={audit['normal_max']:.3e}; "
        f"steps={audit_trajectory.t.size - 1}"
    )
    if audit["relative_error"] > args.audit_tolerance:
        raise RuntimeError(
            f"Directional derivative audit failed: {audit['relative_error']:.3e} > {args.audit_tolerance:.3e}."
        )
    if args.audit_only:
        print(f"[Case2-OnlineResponse] audit passed: {audit_path}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    resumed_from_epoch = 0

    if args.resume:
        resume_checkpoint = torch.load(resume_path, map_location=device)
        if int(resume_checkpoint.get("n_tot", -1)) != ntot or int(resume_checkpoint.get("n_primary", -1)) != args.n_primary:
            raise ValueError(f"Incompatible resume checkpoint: {resume_path}.")
        model.load_state_dict(resume_checkpoint["state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        initial_train = resume_checkpoint["initial_train"]
        initial_validation = resume_checkpoint["initial_validation"]
        alpha = float(resume_checkpoint["alpha_primary"])
        alpha_source = str(resume_checkpoint["alpha_primary_source"])
        best_state = copy.deepcopy(resume_checkpoint["best_state"])
        best_value = float(resume_checkpoint["best_value"])
        best_validation = resume_checkpoint["best_validation"]
        bad = int(resume_checkpoint["bad_epochs"])
        resumed_from_epoch = int(resume_checkpoint["epoch"])
        rng.bit_generator.state = resume_checkpoint["rng_state"]
        print(
            f"[Case2-OnlineResponse] resuming after epoch {resumed_from_epoch}: "
            f"alpha={alpha:.6e}, best_val={best_value:.6e}, bad={bad}"
        )
    else:
        initial_train = _evaluate(
            model, train, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
            primary_projector, initial, args.max_its, args.relnorm_cutoff, args.min_delta, quiet=True,
        )
        initial_validation = _evaluate(
            model, validation, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
            primary_projector, initial, args.max_its, args.relnorm_cutoff, args.min_delta, quiet=True,
        )
        alpha_arg = str(args.alpha_primary).strip().lower()
        if alpha_arg == "auto":
            alpha = initial_train["tail_loss"] / max(initial_train["primary_loss"], 1.0e-30)
            alpha_source = "auto_balanced_from_exact_online_training_losses"
        else:
            alpha = float(args.alpha_primary)
            if alpha < 0.0:
                raise ValueError("--alpha-primary must be nonnegative or 'auto'.")
            alpha_source = "user_specified"
        best_state = copy.deepcopy(model.state_dict())
        best_value = initial_validation["tail_loss"] + alpha * initial_validation["primary_loss"]
        best_validation = initial_validation
        bad = 0
        print(
            f"[Case2-OnlineResponse] initial train tail={initial_train['tail_loss']:.6e} "
            f"primary={initial_train['primary_loss']:.6e}; alpha={alpha:.6e} ({alpha_source})"
        )

    started = time.perf_counter()

    for epoch in range(resumed_from_epoch + 1, args.epochs + 1):
        model.train()
        tail_values: list[float] = []
        primary_values: list[float] = []
        for index in rng.permutation(len(train)):
            trajectory = train[int(index)]
            optimizer.zero_grad(set_to_none=True)
            tail_prediction, primary_loss, tail_gradient, _, normal_max, _ = _solve_and_differentiate(
                model, trajectory, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
                args.max_its, args.relnorm_cutoff, args.min_delta, quiet=True,
                primary_projector=primary_projector, initial=initial,
            )
            tail_loss = _loss_components(tail_prediction, trajectory, args.n_primary)
            torch.autograd.backward(
                tensors=(tail_loss, tail_prediction),
                grad_tensors=(torch.ones((), dtype=tail_loss.dtype, device=device),
                              torch.as_tensor(alpha * tail_gradient, dtype=tail_prediction.dtype, device=device)),
            )
            optimizer.step()
            tail_values.append(float(tail_loss.item()))
            primary_values.append(float(primary_loss))

        if epoch % args.validation_every == 0 or epoch == args.epochs:
            validation_metrics = _evaluate(
                model, validation, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
                primary_projector, initial, args.max_its, args.relnorm_cutoff, args.min_delta, quiet=True,
            )
            value = validation_metrics["tail_loss"] + alpha * validation_metrics["primary_loss"]
            if value < best_value - 1.0e-12:
                best_value = value
                best_state = copy.deepcopy(model.state_dict())
                best_validation = validation_metrics
                bad = 0
            else:
                bad += 1
            print(
                f"[Epoch {epoch:03d}] train_tail={np.mean(tail_values):.6e} "
                f"train_primary={np.mean(primary_values):.6e} "
                f"val_tail={validation_metrics['tail_loss']:.6e} val_primary={validation_metrics['primary_loss']:.6e} "
                f"val={value:.6e} bad={bad}",
                flush=True,
            )
        else:
            validation_metrics = None
            print(
                f"[Epoch {epoch:03d}] train_tail={np.mean(tail_values):.6e} "
                f"train_primary={np.mean(primary_values):.6e}",
                flush=True,
            )

        # Persist the complete optimizer state at every epoch.  A terminal or
        # IDE restart can then resume exactly from the next epoch.
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "case2_exact_online_response_resume_v1",
                "epoch": int(epoch),
                "n_tot": int(ntot),
                "n_primary": int(args.n_primary),
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state": best_state,
                "best_value": float(best_value),
                "best_validation": best_validation,
                "bad_epochs": int(bad),
                "initial_train": initial_train,
                "initial_validation": initial_validation,
                "alpha_primary": float(alpha),
                "alpha_primary_source": alpha_source,
                "rng_state": rng.bit_generator.state,
                "directional_audit": audit,
                "last_validation": validation_metrics,
            },
            resume_path,
        )
        print(f"[Case2-OnlineResponse] saved resume checkpoint after epoch {epoch}: {resume_path}", flush=True)
        if bad >= args.patience:
            print(f"[EarlyStop] epoch={epoch} best_val={best_value:.6e}")
            break

    model.load_state_dict(best_state)
    final_train = _evaluate(
        model, train, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
        primary_projector, initial, args.max_its, args.relnorm_cutoff, args.min_delta, quiet=True,
    )
    final_validation = _evaluate(
        model, validation, args.n_primary, primary_basis, secondary_basis, u_ref, operators, device,
        primary_projector, initial, args.max_its, args.relnorm_cutoff, args.min_delta, quiet=True,
    )
    output_checkpoint = dict(checkpoint)
    output_checkpoint.update(
        {
            "state_dict": model.state_dict(),
            "format": "case2_exact_online_response_qtot_master",
            "training_objective": "relative_tail_mse + alpha * exact_case2_online_primary_mse",
            "response_definition": "exact Case-2 LSPG solve with discrete adjoint of normal equations",
            "primary_modes": int(args.n_primary),
            "alpha_primary": float(alpha),
            "alpha_primary_source": alpha_source,
            "online_max_its": int(args.max_its),
            "online_relnorm_cutoff": float(args.relnorm_cutoff),
            "online_min_delta": float(args.min_delta),
            "directional_audit": audit,
        }
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, model_path)
    resume_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    summary = {
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "validation_dataset_dir": str(validation_dir),
        "basis_path": str(Path(args.basis_path).resolve()),
        "u_ref_path": str(Path(args.u_ref_path).resolve()),
        "n_tot": int(ntot),
        "primary_modes": int(args.n_primary),
        "secondary_modes": int(ntot - args.n_primary),
        "train_trajectories": len(train),
        "validation_trajectories": len(validation),
        "epochs_requested": int(args.epochs),
        "resumed_from_epoch": int(resumed_from_epoch),
        "alpha_primary": float(alpha),
        "alpha_primary_source": alpha_source,
        "best_validation_objective": float(best_value),
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "directional_audit": audit,
        "elapsed_s": float(elapsed),
    }
    _write_summary(summary_path, summary)
    print(f"[Case2-OnlineResponse] saved model: {model_path}")
    print(f"[Case2-OnlineResponse] summary: {summary_path}")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
