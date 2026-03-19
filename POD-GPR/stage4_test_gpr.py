#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TEST POD-GPR RECONSTRUCTION

Loads the trained POD-GPR model from stage3, reconstructs snapshots
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


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, plot_snaps
from burgers.pod_gpr_manifold import decode_gp
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


def _load_stage2_use_u_ref(model_dir):
    stage2_metadata_path = os.path.join(os.path.dirname(model_dir), "stage2_projection_metadata.npz")
    if not os.path.exists(stage2_metadata_path):
        return None, stage2_metadata_path

    try:
        meta = np.load(stage2_metadata_path, allow_pickle=True)
    except Exception:
        return None, stage2_metadata_path

    if "use_u_ref" not in meta.files:
        return None, stage2_metadata_path

    value = bool(np.asarray(meta["use_u_ref"]).reshape(-1)[0])
    return value, stage2_metadata_path


def _resolve_u_ref(uref_mode, uref_file, model_use_u_ref, model_dir, expected_size):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    candidate_files = []
    if uref_file is not None:
        candidate_files.append(uref_file)
    candidate_files.append(os.path.join(model_dir, "u_ref.npy"))
    candidate_files.append(os.path.join(os.path.dirname(model_dir), "u_ref.npy"))

    seen = set()
    filtered_candidates = []
    for path in candidate_files:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            filtered_candidates.append(path)

    if mode == "off":
        use_u_ref = False
    elif mode == "on":
        use_u_ref = True
    else:
        if model_use_u_ref is None:
            use_u_ref = any(os.path.exists(path) for path in filtered_candidates)
        else:
            use_u_ref = bool(model_use_u_ref)

    if not use_u_ref:
        return False, np.zeros(expected_size, dtype=np.float64), "zeros(off)"

    for path in filtered_candidates:
        if os.path.exists(path):
            u_ref = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
            if u_ref.size != expected_size:
                raise ValueError(
                    f"u_ref size mismatch in '{path}': got {u_ref.size}, expected {expected_size}."
                )
            return True, u_ref, path

    raise FileNotFoundError(
        "u_ref is required by current settings but no candidate file exists. "
        f"Checked: {filtered_candidates}"
    )


def reconstruct_snapshot_with_gpr(snapshot, u_ref, u_p, u_s, scaler, gpr_model, use_custom_predict):
    snapshot = np.asarray(snapshot, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)

    centered = snapshot - u_ref[:, None]
    q_p = u_p.T @ centered

    recon = np.zeros_like(snapshot)
    for i in range(q_p.shape[1]):
        recon[:, i] = decode_gp(
            q_p=q_p[:, i],
            gp_model=gpr_model,
            basis=u_p,
            basis2=u_s,
            scaler=scaler,
            u_ref=u_ref,
            use_custom_predict=use_custom_predict,
            echo_level=0,
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
    model_dir=os.path.join(script_dir, "pod_gpr_model"),
    uref_file=None,
    output_dir=os.path.join(script_dir, "stage4_results"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    uref_mode="auto",
    compare_pod=True,
    use_custom_predict=True,
):
    set_latex_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    mu1 = float(target_mu[0])
    mu2 = float(target_mu[1])
    mu = [mu1, mu2]

    print("\n====================================================")
    print("          STAGE 4: TEST POD-GPR MODEL")
    print("====================================================")
    print(f"[STAGE4] target mu = [{mu1:.3f}, {mu2:.4f}]")

    w0 = np.asarray(W0, dtype=np.float64).copy()
    t0 = time.time()
    hdm_snap = np.asarray(
        load_or_compute_snaps(mu, GRID_X, GRID_Y, w0, dt, num_steps, snap_folder=snap_folder),
        dtype=np.float64,
    )
    elapsed_hdm = time.time() - t0

    gpr_file = os.path.join(model_dir, "gpr_model.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    u_p_file = os.path.join(model_dir, "U_p.npy")
    u_s_file = os.path.join(model_dir, "U_s.npy")

    for path in (gpr_file, scaler_file, u_p_file, u_s_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model artifact: {path}. Run stage3 first.")

    with open(gpr_file, "rb") as file:
        gpr_model = pickle.load(file)
    with open(scaler_file, "rb") as file:
        scaler = pickle.load(file)

    u_p = np.asarray(np.load(u_p_file, allow_pickle=False), dtype=np.float64)
    u_s = np.asarray(np.load(u_s_file, allow_pickle=False), dtype=np.float64)

    if u_p.ndim != 2 or u_s.ndim != 2:
        raise ValueError("U_p and U_s must be 2D arrays.")
    if u_p.shape[0] != u_s.shape[0]:
        raise ValueError(f"U_p and U_s row mismatch: {u_p.shape[0]} vs {u_s.shape[0]}.")

    model_use_u_ref, stage2_metadata_path = _load_stage2_use_u_ref(model_dir)
    use_u_ref, u_ref, u_ref_source = _resolve_u_ref(
        uref_mode=uref_mode,
        uref_file=uref_file,
        model_use_u_ref=model_use_u_ref,
        model_dir=model_dir,
        expected_size=u_p.shape[0],
    )

    if hdm_snap.shape[0] != u_p.shape[0]:
        raise RuntimeError(f"State size mismatch: hdm={hdm_snap.shape[0]}, basis={u_p.shape[0]}")

    print(
        f"[STAGE4] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )

    t0 = time.time()
    pod_gpr_reconstructed = reconstruct_snapshot_with_gpr(
        snapshot=hdm_snap,
        u_ref=u_ref,
        u_p=u_p,
        u_s=u_s,
        scaler=scaler,
        gpr_model=gpr_model,
        use_custom_predict=use_custom_predict,
    )
    elapsed_gpr = time.time() - t0

    pod_reconstructed = None
    elapsed_pod = None
    if compare_pod:
        t0 = time.time()
        u_full = np.hstack((u_p, u_s))
        pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, u_ref, u_full)
        elapsed_pod = time.time() - t0

    hdm_norm = np.linalg.norm(hdm_snap)
    if hdm_norm > 0.0:
        pod_gpr_error = np.linalg.norm(hdm_snap - pod_gpr_reconstructed) / hdm_norm
        pod_error = (
            np.linalg.norm(hdm_snap - pod_reconstructed) / hdm_norm
            if pod_reconstructed is not None
            else None
        )
    else:
        pod_gpr_error = np.nan
        pod_error = np.nan if pod_reconstructed is not None else None

    print(f"[STAGE4] POD-GPR relative error: {100.0 * pod_gpr_error:.4f}%")
    if pod_error is not None:
        print(f"[STAGE4] POD relative error: {100.0 * pod_error:.4f}%")

    gpr_file_path = os.path.join(
        output_dir,
        f"pod_gpr_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
    )
    np.save(gpr_file_path, pod_gpr_reconstructed)

    pod_file_path = None
    if pod_reconstructed is not None:
        pod_file_path = os.path.join(
            output_dir,
            f"pod_reconstruction_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy",
        )
        np.save(pod_file_path, pod_reconstructed)

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
        pod_gpr_reconstructed,
        inds_to_plot,
        label="POD-GPR",
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
        f"pod_gpr_projection_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png",
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
                    ("script", "stage4_test_gpr.py"),
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
                    ("stage2_use_u_ref", model_use_u_ref),
                    ("stage2_projection_metadata", stage2_metadata_path if os.path.exists(stage2_metadata_path) else None),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("compare_pod", compare_pod),
                    ("use_custom_predict", use_custom_predict),
                    ("learned_kernel", str(getattr(gpr_model, "kernel_", getattr(gpr_model, "kernel", None)))),
                ],
            ),
            (
                "model_shapes",
                [
                    ("U_p_shape", u_p.shape),
                    ("U_s_shape", u_s.shape),
                    ("hdm_shape", hdm_snap.shape),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("hdm_load_or_solve", elapsed_hdm),
                    ("pod_gpr_reconstruction", elapsed_gpr),
                    ("pod_reconstruction", elapsed_pod),
                ],
            ),
            (
                "errors",
                [
                    ("pod_gpr_relative_l2_error", pod_gpr_error),
                    ("pod_gpr_relative_error_percent", 100.0 * pod_gpr_error),
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
                    ("pod_gpr_reconstruction_npy", gpr_file_path),
                    ("pod_reconstruction_npy", pod_file_path),
                    ("comparison_plot_png", plot_file),
                    ("summary_txt", summary_file),
                ],
            ),
        ],
    )

    print(f"[STAGE4] Saved POD-GPR reconstruction: {gpr_file_path}")
    if pod_file_path is not None:
        print(f"[STAGE4] Saved POD reconstruction: {pod_file_path}")
    print(f"[STAGE4] Saved comparison plot: {plot_file}")
    print(f"[STAGE4] Summary saved: {summary_file}")


if __name__ == "__main__":
    main(target_mu=(4.56, 0.019), compare_pod=True)
