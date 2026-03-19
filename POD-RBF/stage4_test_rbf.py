#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TEST POD-RBF RECONSTRUCTION

Loads the trained POD-RBF model from stage3, reconstructs snapshots
for a target parameter, compares against HDM (and optionally POD baseline),
and saves diagnostics.
"""

import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.pod_rbf_manifold import decode_rbf
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def set_latex_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.35,
            "figure.figsize": (12, 8),
        }
    )


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def reconstruct_snapshot_with_rbf(
    snapshot,
    u_ref,
    u_p,
    u_s,
    q_p_train,
    w_rbf,
    scaler,
    epsilon,
    kernel_name,
):
    snapshot = np.asarray(snapshot, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)

    centered = snapshot - u_ref[:, None]
    q_p = u_p.T @ centered

    recon = np.zeros_like(snapshot)
    for i in range(q_p.shape[1]):
        recon[:, i] = decode_rbf(
            q_p[:, i],
            W=w_rbf,
            q_p_train=q_p_train,
            basis=u_p,
            basis2=u_s,
            epsilon=epsilon,
            scaler=scaler,
            kernel_type=kernel_name,
            u_ref=u_ref,
        )
    return recon


def reconstruct_snapshot_with_pod(snapshot, u_ref, basis):
    snapshot = np.asarray(snapshot, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    centered = snapshot - u_ref[:, None]
    q = basis.T @ centered
    return u_ref[:, None] + basis @ q


def main(
    target_mu=(4.56, 0.019),
    model_dir=os.path.join(script_dir, "pod_rbf_model"),
    uref_file=None,
    output_dir=os.path.join(script_dir, "stage4_results"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    compare_pod=True,
):
    set_latex_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    mu1 = float(target_mu[0])
    mu2 = float(target_mu[1])
    mu = [mu1, mu2]

    print("\n====================================================")
    print("          STAGE 4: TEST POD-RBF MODEL")
    print("====================================================")
    print(f"[STAGE4] target mu = [{mu1:.3f}, {mu2:.4f}]")

    # ------------------------------------------------------------------
    # Load HDM snapshot trajectory
    # ------------------------------------------------------------------
    w0 = np.asarray(W0, dtype=np.float64).copy()

    t0 = time.time()
    hdm_snap = np.asarray(
        load_or_compute_snaps(mu, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )
    elapsed_hdm = time.time() - t0

    # ------------------------------------------------------------------
    # Load trained model artifacts
    # ------------------------------------------------------------------
    weights_file = os.path.join(model_dir, "rbf_weights.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    u_p_file = os.path.join(model_dir, "U_p.npy")
    u_s_file = os.path.join(model_dir, "U_s.npy")

    for path in (weights_file, scaler_file, u_p_file, u_s_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model artifact: {path}. Run stage3 first.")

    with open(weights_file, "rb") as file:
        data = pickle.load(file)
    with open(scaler_file, "rb") as file:
        scaler = pickle.load(file)

    w_rbf = np.asarray(data["W"], dtype=np.float64)
    q_p_train = np.asarray(data["q_p_train"], dtype=np.float64)
    epsilon = float(data["epsilon"])
    kernel_name = data.get("kernel_name", "imq")
    model_use_u_ref = data.get("use_u_ref", None)

    u_p = np.asarray(np.load(u_p_file, allow_pickle=False), dtype=np.float64)
    u_s = np.asarray(np.load(u_s_file, allow_pickle=False), dtype=np.float64)

    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    candidate_uref_files = []
    if uref_file is not None:
        candidate_uref_files.append(uref_file)
    else:
        candidate_uref_files.append(os.path.join(model_dir, "u_ref.npy"))
        candidate_uref_files.append(os.path.join(script_dir, "u_ref.npy"))

    if mode == "off":
        use_u_ref = False
    elif mode == "on":
        use_u_ref = True
    else:
        if model_use_u_ref is not None:
            use_u_ref = bool(model_use_u_ref)
        else:
            use_u_ref = any(os.path.exists(path) for path in candidate_uref_files)

    u_ref_source = None
    if use_u_ref:
        loaded = None
        for path in candidate_uref_files:
            if os.path.exists(path):
                loaded = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
                u_ref_source = path
                break
        if loaded is None:
            raise FileNotFoundError(
                "u_ref is required by current settings but no candidate file exists. "
                f"Checked: {candidate_uref_files}"
            )
        u_ref = loaded
    else:
        u_ref = np.zeros(u_p.shape[0], dtype=np.float64)
        u_ref_source = "zeros(off)"

    if hdm_snap.shape[0] != u_p.shape[0]:
        raise RuntimeError(
            f"State size mismatch: hdm={hdm_snap.shape[0]}, basis={u_p.shape[0]}"
        )
    if u_ref.size != u_p.shape[0]:
        raise RuntimeError(
            f"u_ref size mismatch: got {u_ref.size}, expected {u_p.shape[0]}."
        )

    print(
        f"[STAGE4] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    t0 = time.time()
    pod_rbf_reconstructed = reconstruct_snapshot_with_rbf(
        hdm_snap,
        u_ref,
        u_p,
        u_s,
        q_p_train,
        w_rbf,
        scaler,
        epsilon,
        kernel_name,
    )
    elapsed_rbf = time.time() - t0

    pod_reconstructed = None
    elapsed_pod = None
    if compare_pod:
        t0 = time.time()
        u_full = np.hstack((u_p, u_s))
        pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, u_ref, u_full)
        elapsed_pod = time.time() - t0

    hdm_norm = np.linalg.norm(hdm_snap)
    if hdm_norm > 0.0:
        pod_rbf_error = np.linalg.norm(hdm_snap - pod_rbf_reconstructed) / hdm_norm
        pod_error = (
            np.linalg.norm(hdm_snap - pod_reconstructed) / hdm_norm
            if pod_reconstructed is not None
            else None
        )
    else:
        pod_rbf_error = np.nan
        pod_error = np.nan if pod_reconstructed is not None else None

    print(f"[STAGE4] POD-RBF relative error: {100.0 * pod_rbf_error:.4f}%")
    if pod_error is not None:
        print(f"[STAGE4] POD relative error: {100.0 * pod_error:.4f}%")

    # ------------------------------------------------------------------
    # Save arrays
    # ------------------------------------------------------------------
    pod_rbf_file_path = os.path.join(
        output_dir,
        f"pod_rbf_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(pod_rbf_file_path, pod_rbf_reconstructed)

    pod_file_path = None
    if pod_reconstructed is not None:
        pod_file_path = os.path.join(
            output_dir,
            f"pod_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
        )
        np.save(pod_file_path, pod_reconstructed)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    inds_to_plot = range(0, num_steps + 1, 100)

    fig, ax1, ax2 = plot_snaps(
        GRID_X,
        GRID_Y,
        hdm_snap,
        inds_to_plot,
        label="HDM",
        color="black",
        linewidth=2.8,
        linestyle="solid",
    )
    plot_snaps(
        GRID_X,
        GRID_Y,
        pod_rbf_reconstructed,
        inds_to_plot,
        label="POD-RBF",
        fig_ax=(fig, ax1, ax2),
        color="blue",
        linewidth=1.8,
        linestyle="solid",
    )
    if pod_reconstructed is not None:
        plot_snaps(
            GRID_X,
            GRID_Y,
            pod_reconstructed,
            inds_to_plot,
            label="POD",
            fig_ax=(fig, ax1, ax2),
            color="#0a8f5a",
            linewidth=1.8,
            linestyle="dashed",
        )

    fig.suptitle(rf"$\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$", y=0.98)
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)
    plt.tight_layout()

    plot_file = os.path.join(
        output_dir,
        f"pod_rbf_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_file = os.path.join(
        output_dir,
        f"stage4_test_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt",
    )
    write_txt_report(
        summary_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_test_rbf.py"),
                    ("mu1", mu1),
                    ("mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("model_dir", model_dir),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_source", u_ref_source),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("kernel_name", kernel_name),
                    ("epsilon", epsilon),
                    ("compare_pod", compare_pod),
                ],
            ),
            (
                "model_shapes",
                [
                    ("U_p_shape", u_p.shape),
                    ("U_s_shape", u_s.shape),
                    ("W_shape", w_rbf.shape),
                    ("q_p_train_shape", q_p_train.shape),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("hdm_load_or_solve", elapsed_hdm),
                    ("pod_rbf_reconstruction", elapsed_rbf),
                    ("pod_reconstruction", elapsed_pod),
                ],
            ),
            (
                "errors",
                [
                    ("pod_rbf_relative_l2_error", pod_rbf_error),
                    ("pod_rbf_relative_error_percent", 100.0 * pod_rbf_error),
                    ("pod_relative_l2_error", pod_error),
                    (
                        "pod_relative_error_percent",
                        None if pod_error is None else 100.0 * pod_error,
                    ),
                ],
            ),
            (
                "outputs",
                [
                    ("pod_rbf_reconstruction_npy", pod_rbf_file_path),
                    ("pod_reconstruction_npy", pod_file_path),
                    ("comparison_plot_png", plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )
    print(f"[STAGE4] Saved POD-RBF reconstruction: {pod_rbf_file_path}")
    if pod_file_path is not None:
        print(f"[STAGE4] Saved POD reconstruction: {pod_file_path}")
    print(f"[STAGE4] Saved comparison plot: {plot_file}")
    print(f"[STAGE4] Summary saved: {summary_file}")


if __name__ == "__main__":
    main(target_mu=(4.75, 0.020), compare_pod=False)
