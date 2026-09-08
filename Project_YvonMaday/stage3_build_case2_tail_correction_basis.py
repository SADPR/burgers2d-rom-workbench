#!/usr/bin/env python3
"""Construct a low-rank Case--2 tail-correction basis from OOF ANN errors.

For each held-out parameter group, a full ``(mu1, mu2, t) -> q_tot`` ANN is
trained only on the remaining parameter trajectories.  The resulting tail
errors are out-of-fold (OOF): no target used to form an error column was seen
by its predictor during fitting.  A time-weighted SVD in the state norm
induced by the tail basis solves the optimal rank-r least-squares
approximation problem used by the tail-corrected Case--2 PROM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_prom_ann_case_2 import Case2Model

from burgers.config import DT, GRID_X, GRID_Y
from burgers.core import get_ops, inviscid_burgers_exact_jac2D


@dataclass(frozen=True)
class Trajectory:
    path: Path
    mu: np.ndarray
    time: np.ndarray
    q: np.ndarray


def _load_trajectories(dataset_dir: Path, ntot: int) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for path in sorted((dataset_dir / "per_mu").glob("mu1_*")):
        mu = np.asarray(np.load(path / "mu.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
        time_grid = np.asarray(np.load(path / "t.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
        q = np.asarray(np.load(path / "qN.npy", allow_pickle=False), dtype=np.float64)
        if mu.shape != (2,) or q.shape != (ntot, time_grid.size):
            raise ValueError(
                f"Invalid trajectory {path}: mu={mu.shape}, t={time_grid.shape}, q={q.shape}."
            )
        trajectories.append(Trajectory(path=path, mu=mu, time=time_grid, q=q))
    if not trajectories:
        raise RuntimeError(f"No trajectories found under {dataset_dir / 'per_mu'}.")
    return trajectories


def _trajectory_xy(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack(
        (
            np.full(trajectory.time.size, trajectory.mu[0], dtype=np.float32),
            np.full(trajectory.time.size, trajectory.mu[1], dtype=np.float32),
            trajectory.time.astype(np.float32),
        )
    )
    return x, trajectory.q.T.astype(np.float32, copy=False)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_scalers(model: Case2Model, x: np.ndarray, y: np.ndarray) -> None:
    x_mean = x.mean(axis=0)
    x_std = np.maximum(x.std(axis=0), 1.0e-12)
    y_mean = y.mean(axis=0)
    y_std = np.maximum(y.std(axis=0), 1.0e-12)
    with torch.no_grad():
        model.scaler.mean.copy_(torch.tensor(x_mean[None, :], dtype=torch.float32))
        model.scaler.std.copy_(torch.tensor(x_std[None, :], dtype=torch.float32))
        model.unscaler.mean.copy_(torch.tensor(y_mean[None, :], dtype=torch.float32))
        model.unscaler.std.copy_(torch.tensor(y_std[None, :], dtype=torch.float32))


def _train_fold(
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    activation: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
) -> tuple[Case2Model, dict[str, float]]:
    """Train on OOF parameter groups and stop on external parameter data."""
    _set_seed(seed)
    if xtrain.shape[0] < 1 or xval.shape[0] < 1:
        raise ValueError("OOF training and external validation sets must both contain samples.")
    model = Case2Model(ytrain.shape[1], hidden_dims=hidden_dims, activation=activation).to(device)
    _configure_scalers(model, xtrain, ytrain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=60, min_lr=1.0e-6
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    xval_t = torch.from_numpy(xval).to(device)
    yval_t = torch.from_numpy(yval).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(xval_t) - yval_t) ** 2).item())
        scheduler.step(val_loss)
        if val_loss < best_loss - 1.0e-12:
            best_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("OOF fold training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_external_val_mse": best_loss, "epochs_ran": float(epoch)}


def _folds(num_trajectories: int, folds: int) -> list[np.ndarray]:
    """Use a spatially spread three-fold partition on the 3-by-3 baseline grid."""
    if folds == 3 and num_trajectories == 9:
        return [
            np.array([0, 4, 8], dtype=np.int64),
            np.array([1, 5, 6], dtype=np.int64),
            np.array([2, 3, 7], dtype=np.int64),
        ]
    return [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(np.arange(num_trajectories), folds)]


def _time_weights(time_grid: np.ndarray) -> np.ndarray:
    if time_grid.size < 2:
        return np.ones_like(time_grid)
    weights = np.ones(time_grid.size, dtype=np.float64)
    weights[[0, -1]] = 0.5
    return weights / np.sum(weights)


def _write_summary(path: Path, values: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}: {value}\n")


def _state_metric_factor(basis_path: Path, ntot: int, n_primary: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``T`` with ``T.T @ T = Vbar.T @ Vbar`` and the Gram matrix.

    The LSPG-sensitive basis is not Euclidean orthonormal.  Thus the
    correction subspace is selected in the physical state norm
    ``||Vbar e||_2`` rather than through an unqualified coordinate-norm SVD.
    For ``G = Vbar.T Vbar = T.T T``, the weighted SVD of ``T E`` supplies the
    exact state-norm best rank-r correction space.
    """
    full_basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if full_basis.ndim != 2 or full_basis.shape[1] < ntot:
        raise ValueError(f"Invalid basis at {basis_path}: shape={full_basis.shape}, ntot={ntot}.")
    vbar = full_basis[:, n_primary:ntot]
    gram = vbar.T @ vbar
    gram = 0.5 * (gram + gram.T)
    eig_min = float(np.linalg.eigvalsh(gram)[0])
    if eig_min <= 0.0:
        raise RuntimeError(f"Tail state Gram matrix is not SPD (smallest eigenvalue={eig_min:.3e}).")
    # np.linalg.cholesky returns L with G = L L^T; hence T = L^T.
    factor = np.linalg.cholesky(gram).T
    return factor, gram


def _mass_diagonal() -> np.ndarray:
    """Physical cell-area mass diagonal, shared with the low/high transfer diagnostic."""
    dx = np.asarray(GRID_X[1:] - GRID_X[:-1], dtype=np.float64)
    dy = np.asarray(GRID_Y[1:] - GRID_Y[:-1], dtype=np.float64)
    cell_area = np.outer(dy, dx).reshape(-1)
    return np.concatenate((cell_area, cell_area))


def _mass_norm(vec: np.ndarray, mass_diag: np.ndarray) -> float:
    return float(np.sqrt(np.sum(mass_diag * vec**2)))


def _transfer_observability_gains(
    trajectory: Trajectory,
    full_basis: np.ndarray,
    u_ref: np.ndarray,
    n_primary: int,
    error_columns: np.ndarray,
    mass_diag: np.ndarray,
    operators: tuple,
) -> np.ndarray:
    """Per-column low/high transfer gain ``g_M = ||V T_LH e||_M / ||Vbar e||_M``.

    ``T_LH = -(V^T J^T J V)^+ (V^T J^T J Vbar)`` is the same low/high LSPG
    transfer operator used for basis selection (manuscript
    Section~2.3.2/Eq.~(low-high-transfer)): it measures how strongly a
    secondary-coordinate direction is seen by the residual-solved primary
    coordinates at a given state. Weighting each OOF tail-error column by
    its own gain -- rather than using the raw error magnitude alone --
    biases the correction-basis SVD toward directions that are both
    frequently wrong (large OOF error) and correctable online (large
    transfer gain), instead of directions that are merely large.
    """
    dxec, dyec, jdxec, jdyec, eye = operators
    v = full_basis[:, :n_primary]
    vbar = full_basis[:, n_primary:]
    num_steps = error_columns.shape[1]
    gains = np.empty(num_steps, dtype=np.float64)
    for step in range(num_steps):
        state = u_ref + full_basis @ trajectory.q[:, step]
        jac = inviscid_burgers_exact_jac2D(state, DT, jdxec, jdyec, eye)
        jv = jac @ v
        jvbar = jac @ vbar
        h_ll = jv.T @ jv
        h_lh = jv.T @ jvbar
        transfer = -np.linalg.pinv(h_ll, rcond=1.0e-12) @ h_lh
        e = error_columns[:, step]
        denominator = _mass_norm(vbar @ e, mass_diag) + 1.0e-30
        gains[step] = _mass_norm(v @ (transfer @ e), mass_diag) / denominator
    return gains


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--validation-dataset-dir", required=True, type=Path)
    parser.add_argument("--stage3-dir", required=True, type=Path)
    parser.add_argument("--basis-path", required=True, type=Path)
    parser.add_argument(
        "--u-ref-path",
        type=Path,
        default=None,
        help="Required when --weighting observability is used.",
    )
    parser.add_argument(
        "--weighting",
        choices=("plain", "observability"),
        default="plain",
        help=(
            "'plain' reproduces the original OOF-error-magnitude SVD. "
            "'observability' rescales each OOF error column by its low/high "
            "LSPG transfer gain before the SVD, so the basis favors "
            "directions the residual can actually correct online."
        ),
    )
    parser.add_argument("--ntot", type=int, default=151)
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--hidden-dims", default="256,512,512,256")
    parser.add_argument("--activation", choices=("elu", "gelu", "silu", "tanh", "relu"), default="elu")
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.n_primary < args.ntot:
        raise ValueError("--n-primary must lie in [1, ntot).")
    if args.folds < 2:
        raise ValueError("--folds must be at least two.")
    if args.weighting == "observability" and args.u_ref_path is None:
        raise ValueError("--u-ref-path is required when --weighting observability is used.")

    torch.set_num_threads(max(1, args.num_threads))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value.strip())
    output_dir = args.output_dir or args.stage3_dir / f"case2_tail_correction_oof_ntot{args.ntot}_np{args.n_primary}"
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = output_dir / "tail_correction_basis.npy"
    singular_values_path = output_dir / "tail_correction_singular_values.npy"
    errors_path = output_dir / "oof_tail_errors.npy"
    sample_mu_path = output_dir / "oof_sample_mu.npy"
    sample_time_path = output_dir / "oof_sample_time.npy"
    sample_fold_path = output_dir / "oof_sample_fold.npy"
    metric_gram_path = output_dir / "tail_correction_state_gram.npy"
    summary_path = output_dir / "summary.txt"
    if basis_path.exists() and not args.force:
        raise FileExistsError(f"Existing OOF correction basis: {basis_path}; use --force to replace it.")

    trajectories = _load_trajectories(args.dataset_dir, args.ntot)
    validation_trajectories = _load_trajectories(args.validation_dataset_dir, args.ntot)
    if any(
        np.allclose(validation.mu, train.mu, rtol=0.0, atol=1.0e-12)
        for validation in validation_trajectories
        for train in trajectories
    ):
        raise ValueError("External validation parameters must not overlap the OOF training grid.")
    xvalidation, yvalidation = zip(*(_trajectory_xy(trajectory) for trajectory in validation_trajectories))
    xvalidation_all = np.vstack(xvalidation)
    yvalidation_all = np.vstack(yvalidation)
    state_factor, state_gram = _state_metric_factor(args.basis_path, args.ntot, args.n_primary)
    fold_sets = _folds(len(trajectories), args.folds)
    if any(fold.size == 0 for fold in fold_sets):
        raise ValueError("At least one fold is empty.")
    print(f"[Case2-TailBasis] dataset = {args.dataset_dir}")
    print(f"[Case2-TailBasis] validation dataset = {args.validation_dataset_dir}")
    print(f"[Case2-TailBasis] state metric = ||Vbar e||_2 from {args.basis_path}")
    print(f"[Case2-TailBasis] trajectories/folds = {len(trajectories)}/{len(fold_sets)}")
    print(f"[Case2-TailBasis] architecture = {hidden_dims}, {args.activation}, raw MSE")
    print(f"[Case2-TailBasis] device/threads = {device}/{torch.get_num_threads()}")
    print(f"[Case2-TailBasis] weighting = {args.weighting}")
    print("[Case2-TailBasis] construction = parameter-group OOF errors + state-metric weighted direct SVD")

    full_basis = None
    u_ref_vec = None
    mass_diag = None
    operators = None
    if args.weighting == "observability":
        full_basis = np.asarray(np.load(args.basis_path, allow_pickle=False), dtype=np.float64)[:, : args.ntot]
        u_ref_vec = np.asarray(np.load(args.u_ref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if u_ref_vec.size != full_basis.shape[0]:
            raise ValueError(
                f"u_ref size {u_ref_vec.size} does not match basis row count {full_basis.shape[0]}."
            )
        mass_diag = _mass_diagonal()
        operators = get_ops(GRID_X, GRID_Y)
        print(f"[Case2-TailBasis] observability weighting uses basis={args.basis_path}, u_ref={args.u_ref_path}")

    errors: list[np.ndarray] = []
    sample_mu: list[np.ndarray] = []
    sample_time: list[np.ndarray] = []
    sample_fold: list[np.ndarray] = []
    sample_weight: list[np.ndarray] = []
    sample_gain: list[np.ndarray] = []
    fold_summaries: list[dict] = []
    started = time.time()

    for fold_id, held_out in enumerate(fold_sets):
        train_indices = np.setdiff1d(np.arange(len(trajectories)), held_out, assume_unique=True)
        xtrain, ytrain = zip(*(_trajectory_xy(trajectories[index]) for index in train_indices))
        xtrain_all = np.vstack(xtrain)
        ytrain_all = np.vstack(ytrain)
        heldout_mu = [trajectories[index].mu.tolist() for index in held_out]
        print(f"[Case2-TailBasis] fold {fold_id + 1}/{len(fold_sets)}: held out {heldout_mu}")
        model, train_info = _train_fold(
            xtrain_all,
            ytrain_all,
            xvalidation_all,
            yvalidation_all,
            hidden_dims=hidden_dims,
            activation=args.activation,
            seed=args.seed + fold_id,
            device=device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
        )
        fold_error_norm_sq = 0.0
        fold_target_norm_sq = 0.0
        with torch.no_grad():
            for index in held_out:
                xhold, yhold = _trajectory_xy(trajectories[index])
                prediction = model(torch.from_numpy(xhold).to(device)).detach().cpu().numpy()
                error = (yhold[:, args.n_primary:] - prediction[:, args.n_primary:]).T.astype(np.float64)
                errors.append(error)
                sample_mu.append(np.repeat(trajectories[index].mu[None, :], xhold.shape[0], axis=0))
                sample_time.append(trajectories[index].time.copy())
                sample_fold.append(np.full(xhold.shape[0], fold_id, dtype=np.int64))
                sample_weight.append(_time_weights(trajectories[index].time))
                if args.weighting == "observability":
                    gains = _transfer_observability_gains(
                        trajectories[index],
                        full_basis,
                        u_ref_vec,
                        args.n_primary,
                        error,
                        mass_diag,
                        operators,
                    )
                else:
                    gains = np.ones(error.shape[1], dtype=np.float64)
                sample_gain.append(gains)
                fold_error_norm_sq += float(np.sum(error**2))
                fold_target_norm_sq += float(np.sum(yhold[:, args.n_primary:] ** 2))
        fold_summary = {
            "fold": fold_id,
            "held_out_mu": heldout_mu,
            **train_info,
            "held_out_tail_rel_frob_percent": 100.0 * np.sqrt(fold_error_norm_sq / max(fold_target_norm_sq, 1.0e-30)),
        }
        fold_summaries.append(fold_summary)
        print(
            f"[Case2-TailBasis] fold {fold_id + 1}: "
            f"OOF tail relative error={fold_summary['held_out_tail_rel_frob_percent']:.3f}%"
        )

    error_matrix = np.hstack(errors)
    all_mu = np.vstack(sample_mu)
    all_time = np.concatenate(sample_time)
    all_fold = np.concatenate(sample_fold)
    weights = np.concatenate(sample_weight)
    gain_vector = np.concatenate(sample_gain)
    if weights.size != error_matrix.shape[1]:
        raise RuntimeError("OOF error columns and time weights do not have the same length.")
    if gain_vector.size != error_matrix.shape[1]:
        raise RuntimeError("OOF error columns and transfer gains do not have the same length.")
    # Each column is scaled by its own low/high transfer gain before the SVD
    # (identically 1 for --weighting plain); this reweights the correction
    # subspace toward directions that are both frequently wrong and
    # observable by the residual, without changing the state-norm
    # orthonormalization that follows.
    svd_input = error_matrix * gain_vector[None, :]
    weighted_errors = (state_factor @ svd_input) * np.sqrt(weights)[None, :]
    left_vectors, singular_values, _ = np.linalg.svd(weighted_errors, full_matrices=False)
    # B is state-orthonormal: B.T (Vbar.T Vbar) B = I.
    basis = np.linalg.solve(state_factor, left_vectors)
    rank_limit = min(basis.shape)
    tail_rel_frob_percent = 100.0 * np.linalg.norm(error_matrix) / max(
        np.linalg.norm(np.hstack([trajectory.q[args.n_primary:] for trajectory in trajectories])), 1.0e-30
    )

    np.save(basis_path, basis)
    np.save(singular_values_path, singular_values)
    np.save(errors_path, error_matrix)
    np.save(sample_mu_path, all_mu)
    np.save(sample_time_path, all_time)
    np.save(sample_fold_path, all_fold)
    np.save(metric_gram_path, state_gram)
    np.save(output_dir / "oof_transfer_gains.npy", gain_vector)
    state_orthonorm_fro_err = float(
        np.linalg.norm(basis.T @ state_gram @ basis - np.eye(rank_limit), ord="fro")
    )
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "validation_dataset_dir": str(args.validation_dataset_dir),
        "output_dir": str(output_dir),
        "ntot": args.ntot,
        "n_primary": args.n_primary,
        "n_secondary": args.ntot - args.n_primary,
        "folds": len(fold_sets),
        "fold_summaries": fold_summaries,
        "architecture": hidden_dims,
        "activation": args.activation,
        "loss": "raw_mse",
        "error_construction": "parameter_group_out_of_fold",
        "early_stopping_validation": "external_parameter_trajectories",
        "basis_construction": "time_weighted_direct_svd_in_tail_state_norm_without_error_centering",
        "weighting": args.weighting,
        "basis_path": str(args.basis_path),
        "u_ref_path": str(args.u_ref_path) if args.u_ref_path is not None else None,
        "state_metric": "Vbar^T Vbar",
        "state_metric_min_eigenvalue": float(np.linalg.eigvalsh(state_gram)[0]),
        "state_metric_max_eigenvalue": float(np.linalg.eigvalsh(state_gram)[-1]),
        "state_orthonorm_fro_err": state_orthonorm_fro_err,
        "oof_tail_rel_frob_percent": tail_rel_frob_percent,
        "transfer_gain_mean": float(np.mean(gain_vector)),
        "transfer_gain_median": float(np.median(gain_vector)),
        "transfer_gain_p95": float(np.percentile(gain_vector, 95.0)),
        "transfer_gain_max": float(np.max(gain_vector)),
        "transfer_gain_min": float(np.min(gain_vector)),
        "basis_rank": rank_limit,
        "singular_values": singular_values.tolist(),
        "elapsed_s": time.time() - started,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _write_summary(summary_path, summary)
    print(f"[Case2-TailBasis] OOF tail relative error = {tail_rel_frob_percent:.3f}%")
    print(f"[Case2-TailBasis] saved basis: {basis_path}")
    print(f"[Case2-TailBasis] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
