#!/usr/bin/env python3
"""
Analyze the Case-2 low/high contamination transfer operator

    T_k = (J_L^T P J_L)^{-1} (J_L^T P J_H),

with
    J_L = J(w_k) V,   J_H = J(w_k) Vbar.

By default P = I (Euclidean residual norm). Optionally, a diagonal SPD
weight P = diag(p_diag) can be provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from burgers.core import get_ops, inviscid_burgers_exact_jac2D


def _infer_square_grid_from_state_size(n_state: int):
    if n_state % 2 != 0:
        raise ValueError(f"State size must be even (ux, uy stacked), got {n_state}.")
    n_cells = n_state // 2
    n_side = int(round(np.sqrt(n_cells)))
    if n_side * n_side != n_cells:
        raise ValueError(
            f"State size {n_state} does not match a square grid with 2 components."
        )
    grid_x = np.linspace(0.0, 100.0, n_side + 1)
    grid_y = np.linspace(0.0, 100.0, n_side + 1)
    return grid_x, grid_y, n_side, n_side


def _fmt_sci(x: float) -> str:
    return f"{float(x):.6e}"


def _write_kv_txt(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in pairs:
            f.write(f"{k}: {v}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute ||T_k|| and SVD diagnostics for Case-2 contamination."
    )
    parser.add_argument(
        "--linear-run-dir",
        required=True,
        help="Directory containing rom_snaps.npy, qN.npy, t.npy from linear PROM.",
    )
    parser.add_argument(
        "--basis-path",
        required=True,
        help="Path to basis file used in the corresponding linear run.",
    )
    parser.add_argument(
        "--u-ref-path",
        required=True,
        help="Path to reference state used with basis.",
    )
    parser.add_argument("--n-primary", type=int, default=10)
    parser.add_argument("--n-tot", type=int, default=151)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional time-step override. If omitted, inferred from t.npy when possible.",
    )
    parser.add_argument(
        "--p-diag-path",
        type=str,
        default=None,
        help="Optional diagonal entries of residual metric P (identity if omitted).",
    )
    parser.add_argument(
        "--time-start-index",
        type=int,
        default=1,
        help="First time index used for diagnostics (default: 1).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride over time indices (default: 1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, keep only the first max-samples indices after stride.",
    )
    parser.add_argument(
        "--normal-eq-reg",
        type=float,
        default=1e-12,
        help="Tikhonov regularization for (J_L^T P J_L).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <linear-run-dir>/transfer_T).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="transfer_T",
        help="Tag prefix for output files.",
    )
    args = parser.parse_args()

    linear_run_dir = Path(args.linear_run_dir).expanduser().resolve()
    basis_path = Path(args.basis_path).expanduser().resolve()
    u_ref_path = Path(args.u_ref_path).expanduser().resolve()

    rom_snaps_path = linear_run_dir / "rom_snaps.npy"
    qn_path = linear_run_dir / "qN.npy"
    t_path = linear_run_dir / "t.npy"

    for p in (rom_snaps_path, qn_path, basis_path, u_ref_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    u_ref = np.asarray(np.load(u_ref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    rom_snaps = np.asarray(np.load(rom_snaps_path, allow_pickle=False), dtype=np.float64)
    qn = np.asarray(np.load(qn_path, allow_pickle=False), dtype=np.float64)

    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D, got shape {basis.shape}")
    if rom_snaps.ndim != 2:
        raise ValueError(f"rom_snaps must be 2D, got shape {rom_snaps.shape}")
    if qn.ndim != 2:
        raise ValueError(f"qN must be 2D, got shape {qn.shape}")
    if basis.shape[0] != u_ref.size:
        raise ValueError(
            f"basis rows ({basis.shape[0]}) and u_ref size ({u_ref.size}) mismatch."
        )
    if rom_snaps.shape[0] != basis.shape[0]:
        raise ValueError(
            f"rom_snaps rows ({rom_snaps.shape[0]}) and basis rows ({basis.shape[0]}) mismatch."
        )
    if args.n_tot > basis.shape[1]:
        raise ValueError(
            f"Requested n_tot={args.n_tot}, but basis has only {basis.shape[1]} columns."
        )
    if qn.shape[0] < args.n_tot:
        raise ValueError(
            f"qN has {qn.shape[0]} rows, requested n_tot={args.n_tot}."
        )
    if not (1 <= args.n_primary < args.n_tot):
        raise ValueError(
            f"Invalid split n_primary={args.n_primary}, n_tot={args.n_tot}."
        )

    if t_path.exists():
        t_vec = np.asarray(np.load(t_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    else:
        t_vec = None

    n_t = min(rom_snaps.shape[1], qn.shape[1], (t_vec.size if t_vec is not None else rom_snaps.shape[1]))
    rom_snaps = rom_snaps[:, :n_t]
    qn = qn[: args.n_tot, :n_t]
    if t_vec is not None:
        t_vec = t_vec[:n_t]

    if args.dt is not None:
        dt = float(args.dt)
    elif t_vec is not None and t_vec.size >= 2:
        dt = float(np.median(np.diff(t_vec)))
    else:
        raise ValueError("Could not infer dt. Provide --dt.")

    V = basis[:, : args.n_primary]
    Vbar = basis[:, args.n_primary : args.n_tot]

    p_diag = None
    if args.p_diag_path is not None:
        p_diag = np.asarray(
            np.load(Path(args.p_diag_path).expanduser().resolve(), allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
        if p_diag.size != basis.shape[0]:
            raise ValueError(
                f"p_diag size mismatch: got {p_diag.size}, expected {basis.shape[0]}."
            )

    # Sanity check: qN reconstructs rom_snaps for this basis/ref.
    qn_recon_rel = np.linalg.norm((u_ref[:, None] + basis[:, : args.n_tot] @ qn) - rom_snaps) / (
        np.linalg.norm(rom_snaps) + 1e-30
    )

    grid_x, grid_y, nx, ny = _infer_square_grid_from_state_size(basis.shape[0])
    _, _, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    idx0 = max(int(args.time_start_index), 0)
    stride = max(int(args.stride), 1)
    sample_idx = list(range(idx0, n_t, stride))
    if args.max_samples > 0:
        sample_idx = sample_idx[: int(args.max_samples)]
    if len(sample_idx) == 0:
        raise ValueError("No time indices selected for diagnostics.")

    n = V.shape[1]
    nbar = Vbar.shape[1]
    r = min(n, nbar)
    ns = len(sample_idx)

    sigma = np.zeros((ns, r), dtype=np.float64)
    norm2 = np.zeros(ns, dtype=np.float64)
    norm_fro = np.zeros(ns, dtype=np.float64)
    cond_g = np.zeros(ns, dtype=np.float64)
    rel_fit = np.zeros(ns, dtype=np.float64)
    left_u1 = np.zeros((ns, n), dtype=np.float64)
    right_v1 = np.zeros((ns, nbar), dtype=np.float64)
    times = np.zeros(ns, dtype=np.float64)
    steps = np.zeros(ns, dtype=np.int64)

    reg = float(args.normal_eq_reg)

    for j, k in enumerate(sample_idx):
        w = rom_snaps[:, k]
        J = inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)
        JL = J @ V
        JH = J @ Vbar

        if p_diag is None:
            G = JL.T @ JL
            C = JL.T @ JH
        else:
            P_JL = p_diag[:, None] * JL
            P_JH = p_diag[:, None] * JH
            G = JL.T @ P_JL
            C = JL.T @ P_JH

        if reg > 0.0:
            G_eff = G + reg * np.eye(n, dtype=np.float64)
        else:
            G_eff = G

        try:
            T = np.linalg.solve(G_eff, C)
        except np.linalg.LinAlgError:
            # Fallback: least-squares solve for each RHS.
            T, *_ = np.linalg.lstsq(G_eff, C, rcond=None)

        U, S, VT = np.linalg.svd(T, full_matrices=False)

        sigma[j, : S.size] = S
        norm2[j] = S[0] if S.size > 0 else 0.0
        norm_fro[j] = np.linalg.norm(S)
        cond_g[j] = np.linalg.cond(G_eff)

        fit_num = np.linalg.norm(JL @ T - JH)
        fit_den = np.linalg.norm(JH) + 1e-30
        rel_fit[j] = fit_num / fit_den

        if S.size > 0:
            left_u1[j, :] = U[:, 0]
            right_v1[j, :] = VT[0, :]

        times[j] = (t_vec[k] if t_vec is not None else k * dt)
        steps[j] = k

        if (j + 1) % max(1, ns // 10) == 0 or (j + 1) == ns:
            print(
                f"[T-DIAG] {j+1:4d}/{ns}: step={k:4d}, "
                f"sigma_max={norm2[j]:.3e}, fit={rel_fit[j]:.3e}"
            )

    i_max = int(np.argmax(norm2))
    i_mean = float(np.mean(norm2))
    i_p95 = float(np.percentile(norm2, 95.0))
    i_max_step = int(steps[i_max])
    i_max_time = float(times[i_max])
    v1_max = right_v1[i_max, :]
    top_idx = np.argsort(np.abs(v1_max))[::-1][: min(8, v1_max.size)]

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else linear_run_dir / "transfer_T"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = str(args.tag).strip()
    out_npz = out_dir / f"{tag}_diagnostics.npz"
    out_txt = out_dir / f"{tag}_summary.txt"

    np.savez(
        out_npz,
        steps=steps,
        times=times,
        sigma=sigma,
        norm2=norm2,
        norm_fro=norm_fro,
        cond_g=cond_g,
        rel_fit=rel_fit,
        left_u1=left_u1,
        right_v1=right_v1,
        top_right_indices_at_sigma_max=top_idx,
        top_right_values_at_sigma_max=v1_max[top_idx],
        qn_reconstruction_rel_error=qn_recon_rel,
        n_primary=np.int64(args.n_primary),
        n_tot=np.int64(args.n_tot),
        nx=np.int64(nx),
        ny=np.int64(ny),
        dt=np.float64(dt),
        normal_eq_reg=np.float64(reg),
        p_diag_used=np.array(p_diag is not None, dtype=bool),
    )

    kv = [
        ("linear_run_dir", str(linear_run_dir)),
        ("basis_path", str(basis_path)),
        ("u_ref_path", str(u_ref_path)),
        ("n_primary", int(args.n_primary)),
        ("n_tot", int(args.n_tot)),
        ("nbar", int(nbar)),
        ("n_samples", int(ns)),
        ("time_start_index", int(idx0)),
        ("stride", int(stride)),
        ("dt", _fmt_sci(dt)),
        ("normal_eq_reg", _fmt_sci(reg)),
        ("p_diag_used", bool(p_diag is not None)),
        ("qn_reconstruction_rel_error", _fmt_sci(qn_recon_rel)),
        ("sigma_max_max_over_time", _fmt_sci(norm2[i_max])),
        ("sigma_max_mean_over_time", _fmt_sci(i_mean)),
        ("sigma_max_p95_over_time", _fmt_sci(i_p95)),
        ("sigma_max_argmax_step", int(i_max_step)),
        ("sigma_max_argmax_time", _fmt_sci(i_max_time)),
        ("T_fro_at_sigma_max_step", _fmt_sci(norm_fro[i_max])),
        ("normal_matrix_cond_at_sigma_max_step", _fmt_sci(cond_g[i_max])),
        ("relative_fit_error_at_sigma_max_step", _fmt_sci(rel_fit[i_max])),
        ("relative_fit_error_mean", _fmt_sci(float(np.mean(rel_fit)))),
        ("relative_fit_error_p95", _fmt_sci(float(np.percentile(rel_fit, 95.0)))),
        ("top_right_mode_indices_at_sigma_max", ",".join(str(int(x)) for x in top_idx.tolist())),
        (
            "top_right_mode_values_at_sigma_max",
            ",".join(_fmt_sci(v1_max[ii]) for ii in top_idx.tolist()),
        ),
        ("output_npz", str(out_npz)),
    ]
    _write_kv_txt(out_txt, kv)

    print(f"[T-DIAG] saved summary: {out_txt}")
    print(f"[T-DIAG] saved npz:     {out_npz}")


if __name__ == "__main__":
    main()
