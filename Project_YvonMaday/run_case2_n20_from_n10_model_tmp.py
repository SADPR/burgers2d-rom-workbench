#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TMP experiment:
Run Case 2 with n=20 using a checkpoint trained with n=10 (141 secondary outputs),
by slicing ANN outputs to keep only modes 21..151 (i.e., drop first 10 outputs).

This is a temporary diagnostic script and writes outputs with `tmp_` prefixes.
"""

import argparse
import os
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.pod_ann_manifold import inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2
from burgers.config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

from run_prom_ann_case_2_test_n20 import (
    _load_case2_model as _load_case2_model_flex,
    _resolve_total_modes_from_checkpoint_or_dataset,
    _load_basis_and_reference,
    _safe_mu_tag,
    set_latex_plot_style,
)
from run_prom_ann_case_2 import _load_case2_model as _load_case2_model_legacy
try:
    from project_layout import RUNS_CASE2_DIR, ensure_layout_dirs, resolve_stage3_model, write_kv_txt
except ModuleNotFoundError:
    from .project_layout import RUNS_CASE2_DIR, ensure_layout_dirs, resolve_stage3_model, write_kv_txt


class Case2SliceVectorWrapper(nn.Module):
    """
    Wrap a Case-2 model and slice the output dimensions.

    Expected base output ordering for n=10 checkpoint:
      [mode 11, mode 12, ..., mode 151]  (141 entries)

    For n=20 experiment, use drop_first=10 -> keep [mode 21..151] (131 entries).
    """

    def __init__(self, base_model: nn.Module, drop_first: int = 10, keep_last: Optional[int] = None):
        super().__init__()
        self.base = base_model
        self.drop_first = int(drop_first)
        self.keep_last = None if keep_last is None else int(keep_last)

    def _slice(self, out):
        if out.ndim == 1:
            y = out[self.drop_first:]
            if self.keep_last is not None:
                y = y[: self.keep_last]
            return y
        if out.ndim == 2:
            y = out[:, self.drop_first:]
            if self.keep_last is not None:
                y = y[:, : self.keep_last]
            return y
        raise ValueError(f"Unsupported ANN output ndim={out.ndim}")

    def forward(self, x):
        in_device = x.device
        model_device = next(self.base.parameters()).device

        if x.ndim == 1:
            xin = x.unsqueeze(0)
            if xin.device != model_device:
                xin = xin.to(model_device)
            out = self.base(xin)
            if out.ndim == 2 and out.shape[0] == 1:
                out = out.reshape(-1)
            out = self._slice(out)
            if out.device != in_device:
                out = out.to(in_device)
            return out

        xin = x if x.device == model_device else x.to(model_device)
        out = self.base(xin)
        out = self._slice(out)
        if out.device != in_device:
            out = out.to(in_device)
        return out


def _load_case2_model_any(model_path, device):
    """
    Load Case-2 checkpoint with compatibility across:
      - flexible/sequential CoreMLP checkpoints (new)
      - legacy fc1..fc6 checkpoints (old)
    """
    try:
        model, n_s, ckpt, hidden_dims, activation, dropout = _load_case2_model_flex(
            model_path=model_path,
            device=device,
        )
        return model, n_s, ckpt, hidden_dims, activation, dropout, "flex"
    except RuntimeError as exc:
        msg = str(exc)
        if ("Missing key(s) in state_dict" not in msg) and ("Unexpected key(s) in state_dict" not in msg):
            raise

    # Fallback to legacy architecture loader.
    model, n_s, ckpt = _load_case2_model_legacy(model_path=model_path, device=device)
    return model, n_s, ckpt, None, None, None, "legacy"


def _reconstruct_case2_full_snaps(red_coords, basis, basis2, u_ref, ann_model, mu, dt, device="cpu"):
    basis = np.asarray(basis, dtype=np.float64)
    basis2 = np.asarray(basis2, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    red_coords = np.asarray(red_coords, dtype=np.float64)

    mu1 = float(mu[0])
    mu2 = float(mu[1])

    snaps = np.zeros((u_ref.size, red_coords.shape[1]), dtype=np.float64)
    for k in range(red_coords.shape[1]):
        yk = red_coords[:, k]
        t = float(k * dt)
        with torch.no_grad():
            x_t = torch.tensor([mu1, mu2, t], dtype=torch.float32, device=device)
            qbar = ann_model(x_t).detach().cpu().numpy().reshape(-1)
        snaps[:, k] = u_ref + basis @ yk + basis2 @ qbar

    return snaps


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TMP: use n=10 Case-2 model (141 outputs) as sliced map for n=20 run."
    )
    parser.add_argument("--mu1", type=float, default=4.875)
    parser.add_argument("--mu2", type=float, default=0.0225)
    parser.add_argument("--backend", choices=("prom",), default="prom")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--relnorm-cutoff", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-2)
    parser.add_argument(
        "--model-name",
        type=str,
        default="case2_model.pt",
        help="n=10 model checkpoint name under Results/Stage3/models (unless --model-path is set).",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit model checkpoint path.")
    parser.add_argument(
        "--target-primary-modes",
        type=int,
        default=20,
        help="Primary modes n for the run (expected 20 for this check).",
    )
    parser.add_argument(
        "--drop-first-secondary",
        type=int,
        default=10,
        help="How many first ANN secondary outputs to discard (10 => drop modes 11..20).",
    )
    parser.add_argument("--run-tag-prefix", type=str, default="tmp_case2_transfer_n10_to_n20")
    args = parser.parse_args(argv)

    ensure_layout_dirs()
    set_latex_plot_style()
    os.makedirs(RUNS_CASE2_DIR, exist_ok=True)

    mu_test = [float(args.mu1), float(args.mu2)]
    device = str(args.device).strip().lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("[TMP-Case2] CUDA requested but not available; using CPU.")
        device = "cpu"

    if args.model_path:
        model_path = os.path.abspath(args.model_path)
        model_name = os.path.basename(model_path)
    else:
        model_name = str(args.model_name).strip()
        if not model_name.endswith(".pt"):
            model_name = f"{model_name}.pt"
        model_path = resolve_stage3_model(model_name)

    base_model, n_s_full, ckpt, hidden_dims, activation, dropout, loader_variant = _load_case2_model_any(
        model_path=model_path,
        device=device,
    )
    total_modes = _resolve_total_modes_from_checkpoint_or_dataset(ckpt)

    n_p_target = int(args.target_primary_modes)
    drop_first = int(args.drop_first_secondary)
    n_s_target = int(n_s_full - drop_first)

    if n_s_target <= 0:
        raise ValueError(
            f"Invalid n_s_target={n_s_target}. n_s_full={n_s_full}, drop_first={drop_first}"
        )
    if n_p_target + n_s_target != total_modes:
        raise ValueError(
            "Inconsistent split after slicing: "
            f"n_p_target({n_p_target}) + n_s_target({n_s_target}) != total_modes({total_modes})."
        )

    # Build sliced ANN map.
    ann_model = Case2SliceVectorWrapper(base_model, drop_first=drop_first, keep_last=n_s_target).to(device)
    ann_model.eval()

    vtot, _, _, u_ref, basis_path, uref_path = _load_basis_and_reference(total_modes, n_p_target)
    v = vtot[:, :n_p_target]
    vbar = vtot[:, n_p_target:total_modes]
    if vbar.shape[1] != n_s_target:
        raise ValueError(
            f"Basis split mismatch: vbar has {vbar.shape[1]} cols, expected n_s_target={n_s_target}"
        )

    w0 = np.asarray(W0, dtype=np.float64).reshape(-1).copy()
    if w0.size != vtot.shape[0]:
        raise ValueError(
            f"W0 size mismatch: got {w0.size}, expected {vtot.shape[0]} from basis."
        )

    snap_folder = os.path.join(PROJECT_ROOT, "Results", "param_snaps")
    os.makedirs(snap_folder, exist_ok=True)

    print(f"[TMP-Case2] device = {device}")
    print(f"[TMP-Case2] checkpoint = {model_path}")
    print(f"[TMP-Case2] loader_variant = {loader_variant}")
    print(f"[TMP-Case2] model arch = hidden_dims={hidden_dims}, activation={activation}, dropout={dropout}")
    print(f"[TMP-Case2] total_modes = {total_modes}")
    print(f"[TMP-Case2] original n_s = {n_s_full}")
    print(f"[TMP-Case2] drop_first_secondary = {drop_first}")
    print(f"[TMP-Case2] target n_p = {n_p_target}")
    print(f"[TMP-Case2] target n_s = {n_s_target}")

    hdm_snaps = load_or_compute_snaps(
        mu=mu_test,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        w0=w0,
        dt=DT,
        num_steps=NUM_STEPS,
        snap_folder=snap_folder,
    )

    t0 = time.time()
    rom_snaps, rom_times = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
        grid_x=GRID_X,
        grid_y=GRID_Y,
        w0=w0,
        dt=DT,
        num_steps=NUM_STEPS,
        mu=mu_test,
        ann_model=ann_model,
        ref=None,
        basis=v,
        basis2=vbar,
        u_ref=u_ref,
        max_its=int(args.max_its),
        relnorm_cutoff=float(args.relnorm_cutoff),
        min_delta=float(args.min_delta),
    )
    online_solve_elapsed = time.time() - t0

    num_its, jac_time, res_time, ls_time = rom_times
    rel_err = 100.0 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)

    tag = _safe_mu_tag(mu_test)
    run_tag = f"{args.run_tag_prefix}_prom_{tag}_n{n_p_target}_ntot{total_modes}"
    out_npy = os.path.join(RUNS_CASE2_DIR, f"{run_tag}_snaps.npy")
    out_png = os.path.join(RUNS_CASE2_DIR, f"{run_tag}_hdm_vs_rom.png")
    summary_txt = os.path.join(RUNS_CASE2_DIR, f"{run_tag}_summary.txt")

    np.save(out_npy, rom_snaps)

    plot_steps = list(range(0, NUM_STEPS + 1, 100))
    if NUM_STEPS not in plot_steps:
        plot_steps.append(NUM_STEPS)

    fig, ax1, ax2 = plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snaps,
        plot_steps,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        GRID_X,
        GRID_Y,
        rom_snaps,
        plot_steps,
        label="TMP Case2 (n10 model -> n20 slice)",
        fig_ax=(fig, ax1, ax2),
        color="tab:blue",
        linewidth=1.8,
        linestyle="solid",
    )
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    write_kv_txt(
        summary_txt,
        [
            ("solver_variant", "tmp_case2_transfer_n10_model_to_n20"),
            ("mu_test", mu_test),
            ("device", device),
            ("model_name", model_name),
            ("model_path", model_path),
            ("loader_variant", loader_variant),
            ("basis_path", basis_path),
            ("u_ref_path", uref_path if os.path.exists(uref_path) else "zeros"),
            ("total_modes", total_modes),
            ("n_primary_target", n_p_target),
            ("n_secondary_full_checkpoint", n_s_full),
            ("drop_first_secondary", drop_first),
            ("n_secondary_used", n_s_target),
            ("online_solve_elapsed_s", online_solve_elapsed),
            ("num_iterations", num_its),
            ("jac_time_s", jac_time),
            ("res_time_s", res_time),
            ("ls_time_s", ls_time),
            ("relative_error_percent", rel_err),
            ("snaps_output", out_npy),
            ("plot_output", out_png),
        ],
    )

    print(f"[TMP-Case2] relative error vs HDM: {rel_err:.3f}%")
    print(f"[TMP-Case2] saved snaps: {out_npy}")
    print(f"[TMP-Case2] saved plot:  {out_png}")
    print(f"[TMP-Case2] summary:     {summary_txt}")


if __name__ == "__main__":
    main()
