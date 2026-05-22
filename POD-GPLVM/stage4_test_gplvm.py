#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 4: TEST POD-GPLVM RECONSTRUCTION

Loads the trained POD-GPLVM model from stage3, reconstructs q_test snapshots,
and saves reconstruction diagnostics.
"""

import os
import sys
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.gplvm_inverse import (
    latent_box_bounds,
    nearest_seed_indices,
    solve_bounded_nls,
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


def _model_uses_sparse_decoder(model):
    if "decoder_mode" not in model:
        return False
    mode = str(np.asarray(model["decoder_mode"]).reshape(()))
    return mode.lower() == "sparse_dtc"


def _decode_q_from_latent(z, model):
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    use_sparse = _model_uses_sparse_decoder(model)
    if use_sparse:
        Z_centers = np.asarray(model["Z_inducing"], dtype=np.float64)
        weights = np.asarray(model["beta_inducing"], dtype=np.float64)
    else:
        Z_centers = np.asarray(model["Z_train"], dtype=np.float64)
        weights = np.asarray(model["alpha"], dtype=np.float64)
    y_mean = np.asarray(model["y_mean"], dtype=np.float64).reshape(-1)
    y_std = np.asarray(model["y_std"], dtype=np.float64).reshape(-1)

    log_ell = float(np.asarray(model["log_ell"]).reshape(()))
    log_sf = float(np.asarray(model["log_sf"]).reshape(()))

    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(2.0 * log_sf))

    diff = Z_centers - z[None, :]
    diff = np.clip(diff, -1e150, 1e150)
    d2 = np.sum(diff * diff, axis=1)
    log_k = np.clip(-0.5 * d2 / (ell * ell), -700.0, 50.0)
    k = sf2 * np.exp(log_k)

    y_norm = k @ weights
    return y_mean + y_std * y_norm


def _jac_q_from_latent(z, model):
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    use_sparse = _model_uses_sparse_decoder(model)
    if use_sparse:
        Z_centers = np.asarray(model["Z_inducing"], dtype=np.float64)
        weights = np.asarray(model["beta_inducing"], dtype=np.float64)
    else:
        Z_centers = np.asarray(model["Z_train"], dtype=np.float64)
        weights = np.asarray(model["alpha"], dtype=np.float64)
    y_std = np.asarray(model["y_std"], dtype=np.float64).reshape(-1)

    log_ell = float(np.asarray(model["log_ell"]).reshape(()))
    log_sf = float(np.asarray(model["log_sf"]).reshape(()))

    ell = float(np.exp(log_ell))
    sf2 = float(np.exp(2.0 * log_sf))
    ell2 = ell * ell

    diff = Z_centers - z[None, :]
    diff = np.clip(diff, -1e150, 1e150)
    d2 = np.sum(diff * diff, axis=1)
    log_k = np.clip(-0.5 * d2 / ell2, -700.0, 50.0)
    k = sf2 * np.exp(log_k)

    dk_dz = k[:, None] * (diff / ell2)
    dy_norm_dz = weights.T @ dk_dz
    dy_dz = y_std[:, None] * dy_norm_dz
    return dy_dz


def _infer_latent_for_q(
    q_target,
    model,
    max_its=30,
    tol_rel=1e-6,
    inverse_method="bounded_trf",
    n_starts=5,
    bound_margin_rel=0.20,
    bound_margin_abs=0.25,
    prior_weight=1e-3,
    robust_loss="linear",
    robust_f_scale=1.0,
):
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)

    q_train = np.asarray(model["Q_train_raw"], dtype=np.float64)
    z_train = np.asarray(model["Z_train"], dtype=np.float64)

    mode = str(inverse_method).strip().lower()
    if mode == "gauss_newton":
        d2 = np.sum((q_train - q_target[None, :]) ** 2, axis=1)
        z = z_train[int(np.argmin(d2))].copy()

        q_pred = _decode_q_from_latent(z, model)
        r0 = np.linalg.norm(q_pred - q_target)
        if r0 == 0.0:
            return z, q_pred, 0

        it = 0
        rel = 1.0
        while rel > tol_rel and it < max_its:
            J = _jac_q_from_latent(z, model)
            r = q_pred - q_target

            dz, *_ = np.linalg.lstsq(J, r, rcond=None)
            z -= dz

            q_pred = _decode_q_from_latent(z, model)
            rel = np.linalg.norm(q_pred - q_target) / r0
            it += 1

        return z, q_pred, it

    if mode not in ("bounded_trf", "trf", "least_squares"):
        raise ValueError(
            "inverse_method must be one of: 'bounded_trf', 'least_squares', 'trf', 'gauss_newton'."
        )

    seed_ids, _ = nearest_seed_indices(q_target, q_train, n_starts=n_starts)
    lb, ub = latent_box_bounds(
        z_train,
        margin_rel=bound_margin_rel,
        margin_abs=bound_margin_abs,
    )

    tol = float(max(tol_rel, 1e-12))
    max_nfev = int(max(50, 10 * max_its))

    best_cost = np.inf
    best_z = None
    best_q = None
    best_nfev = 0
    n_success = 0

    for sid in seed_ids:
        z0 = z_train[int(sid)].copy()

        def residual(z):
            return _decode_q_from_latent(z, model) - q_target

        def jac(z):
            return _jac_q_from_latent(z, model)

        z_sol, lsq_res = solve_bounded_nls(
            x0=z0,
            residual_func=residual,
            jac_func=jac,
            lb=lb,
            ub=ub,
            prior_center=z0,
            prior_weight=prior_weight,
            max_nfev=max_nfev,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            loss=robust_loss,
            f_scale=robust_f_scale,
        )

        q_hat = _decode_q_from_latent(z_sol, model)
        r = q_hat - q_target
        cost = 0.5 * float(np.dot(r, r))

        if bool(lsq_res.success):
            n_success += 1

        if cost < best_cost:
            best_cost = cost
            best_z = z_sol
            best_q = q_hat
            best_nfev = int(lsq_res.nfev) if lsq_res.nfev is not None else 0

    if best_z is None:
        # Defensive fallback (should never happen with valid seeds).
        best_z = z_train[int(seed_ids[0])].copy()
        best_q = _decode_q_from_latent(best_z, model)
        best_nfev = 0

    # Encode convergence health in signed nfev:
    #  - positive: at least one successful TRF run
    #  - negative: all starts failed according to solver status
    it_out = best_nfev if n_success > 0 else -best_nfev
    return best_z, best_q, it_out


def _save_error_plot(rel_err_per_sample_pct, out_path):
    rel_err_per_sample_pct = np.asarray(rel_err_per_sample_pct, dtype=np.float64).reshape(-1)
    if rel_err_per_sample_pct.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        np.arange(rel_err_per_sample_pct.size),
        rel_err_per_sample_pct,
        color="tab:blue",
        linewidth=1.6,
    )
    ax.set_xlabel("q_test sample index")
    ax.set_ylabel("Relative error [%]")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _sanitize_overlay_indices(indices, n_available):
    if n_available < 1:
        return np.array([], dtype=np.int64)
    if indices is None:
        return None
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    valid = idx[(idx >= 0) & (idx < int(n_available))]
    if valid.size == 0:
        return np.array([], dtype=np.int64)
    return np.unique(valid)


def _default_overlay_indices(n_available, n_pick):
    n_available = int(n_available)
    if n_available < 1:
        return np.array([], dtype=np.int64)
    n_pick = int(max(1, min(int(n_pick), n_available)))
    return np.unique(np.linspace(0, n_available - 1, n_pick, dtype=int)).astype(np.int64)


def _save_q_overlay_plots(
    q_true,
    q_pred,
    rel_per_sample_pct,
    out_dir,
    plot_indices,
    max_modes=None,
    prefix="gplvm_q_overlay",
):
    q_true = np.asarray(q_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    rel_per_sample_pct = np.asarray(rel_per_sample_pct, dtype=np.float64).reshape(-1)

    if q_true.shape != q_pred.shape:
        raise ValueError(f"q_true/q_pred shape mismatch: {q_true.shape} vs {q_pred.shape}")
    if q_true.ndim != 2:
        raise ValueError(f"q_true must be 2D, got shape {q_true.shape}")
    if q_true.shape[1] != rel_per_sample_pct.size:
        raise ValueError(
            "rel_per_sample_pct size mismatch with q snapshots: "
            f"{rel_per_sample_pct.size} vs {q_true.shape[1]}"
        )

    os.makedirs(out_dir, exist_ok=True)

    q_dim = int(q_true.shape[0])
    if max_modes is None:
        m = q_dim
    else:
        m = int(max_modes)
        if m < 1:
            m = q_dim
        m = min(m, q_dim)

    x = np.arange(m, dtype=int)
    out_files = []

    for j in plot_indices:
        j = int(j)
        qj = q_true[:m, j]
        qhat = q_pred[:m, j]
        dj = qhat - qj

        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(9.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax0.plot(x, qj, color="black", linewidth=2.0, label="q_true")
        ax0.plot(x, qhat, color="tab:blue", linewidth=1.6, linestyle="--", label="q_reconstructed")
        ax0.set_ylabel("q")
        ax0.grid(True, alpha=0.35)
        ax0.legend(loc="best", frameon=True)
        ax0.set_title(
            f"Snapshot {j} | relative error = {float(rel_per_sample_pct[j]):.3e}%"
        )

        ax1.plot(x, dj, color="tab:red", linewidth=1.3)
        ax1.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
        ax1.set_xlabel("POD mode index")
        ax1.set_ylabel("dq")
        ax1.grid(True, alpha=0.35)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_sample_{j:04d}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_files.append(out_path)

    return out_files


def _save_overlay_manifest(file_path, overlay_files):
    lines = [str(path) for path in overlay_files]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + ("\n" if len(lines) > 0 else ""))


def main(
    model_file=os.path.join(script_dir, "pod_gplvm_model", "gplvm_model.npz"),
    q_test_file=os.path.join(script_dir, "q_test.npy"),
    output_dir=os.path.join(script_dir, "stage4_results"),
    report_file=None,
    plot_file=None,
    max_test_samples=None,
    infer_max_its=30,
    infer_tol=1e-6,
    inverse_method="bounded_trf",
    inverse_n_starts=5,
    inverse_bound_margin_rel=0.20,
    inverse_bound_margin_abs=0.25,
    inverse_prior_weight=1e-3,
    inverse_loss="linear",
    inverse_f_scale=1.0,
    save_overlay_plots=True,
    overlay_num_samples=4,
    overlay_indices=None,
    overlay_max_modes=None,
    overlay_dir=None,
):
    if report_file is None:
        report_file = os.path.join(output_dir, "stage4_test_summary.txt")
    if plot_file is None:
        plot_file = os.path.join(output_dir, "gplvm_q_test_relative_error.png")
    if overlay_dir is None:
        overlay_dir = os.path.join(output_dir, "q_overlays")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Missing model file: {model_file}. Run stage3 first.")
    if not os.path.exists(q_test_file):
        raise FileNotFoundError(f"Missing q_test file: {q_test_file}. Run stage2 first.")

    os.makedirs(output_dir, exist_ok=True)

    model_npz = np.load(model_file, allow_pickle=False)
    model = {k: model_npz[k] for k in model_npz.files}
    decoder_mode = (
        str(np.asarray(model["decoder_mode"]).reshape(()))
        if "decoder_mode" in model
        else "full"
    )
    sparse_num_inducing = (
        int(np.asarray(model["sparse_num_inducing"]).reshape(()))
        if "sparse_num_inducing" in model
        else None
    )
    if decoder_mode.lower() == "sparse_dtc":
        for key in ("Z_inducing", "beta_inducing"):
            if key not in model:
                raise KeyError(f"Missing sparse decoder key '{key}' in model file: {model_file}")
    elif "alpha" not in model:
        raise KeyError(f"Missing key 'alpha' in model file: {model_file}")

    q_test = np.asarray(np.load(q_test_file, allow_pickle=False), dtype=np.float64)
    if q_test.ndim != 2:
        raise ValueError("q_test must be a 2D array.")

    q_dim = q_test.shape[0]
    n_test_total = q_test.shape[1]
    q_model_dim = int(np.asarray(model["q_dim"]).reshape(()))
    if q_dim != q_model_dim:
        raise ValueError(f"q_test dimension mismatch: q_test={q_dim}, model={q_model_dim}")

    if max_test_samples is not None and n_test_total > int(max_test_samples):
        pick = np.linspace(0, n_test_total - 1, int(max_test_samples), dtype=int)
        q_eval = q_test[:, pick]
    else:
        q_eval = q_test

    q_pred = np.zeros_like(q_eval)
    latent = np.zeros((int(np.asarray(model["latent_dim"]).reshape(())), q_eval.shape[1]), dtype=np.float64)
    inverse_success_count = 0
    inverse_fail_count = 0

    for j in range(q_eval.shape[1]):
        z, q_hat, inv_stat = _infer_latent_for_q(
            q_eval[:, j],
            model=model,
            max_its=infer_max_its,
            tol_rel=infer_tol,
            inverse_method=inverse_method,
            n_starts=inverse_n_starts,
            bound_margin_rel=inverse_bound_margin_rel,
            bound_margin_abs=inverse_bound_margin_abs,
            prior_weight=inverse_prior_weight,
            robust_loss=inverse_loss,
            robust_f_scale=inverse_f_scale,
        )
        if inv_stat >= 0:
            inverse_success_count += 1
        else:
            inverse_fail_count += 1
        latent[:, j] = z
        q_pred[:, j] = q_hat

    denom = np.linalg.norm(q_eval, ord="fro")
    rel_err_pct = None
    if denom > 0.0:
        rel_err_pct = 100.0 * np.linalg.norm(q_eval - q_pred, ord="fro") / denom

    denom_col = np.linalg.norm(q_eval, axis=0)
    err_col = np.linalg.norm(q_eval - q_pred, axis=0)
    safe_denom = np.where(denom_col > 0.0, denom_col, 1.0)
    rel_per_sample_pct = 100.0 * err_col / safe_denom

    plot_saved = _save_error_plot(rel_per_sample_pct, plot_file)
    overlay_files = []
    overlay_manifest = None
    if bool(save_overlay_plots):
        idx_user = _sanitize_overlay_indices(overlay_indices, q_eval.shape[1])
        if idx_user is None:
            idx_plot = _default_overlay_indices(q_eval.shape[1], overlay_num_samples)
        else:
            idx_plot = idx_user

        overlay_files = _save_q_overlay_plots(
            q_true=q_eval,
            q_pred=q_pred,
            rel_per_sample_pct=rel_per_sample_pct,
            out_dir=overlay_dir,
            plot_indices=idx_plot,
            max_modes=overlay_max_modes,
            prefix="gplvm_q_overlay",
        )
        overlay_manifest = os.path.join(output_dir, "q_overlay_files.txt")
        _save_overlay_manifest(overlay_manifest, overlay_files)

    latent_file = os.path.join(output_dir, "gplvm_latent_test.npy")
    q_pred_file = os.path.join(output_dir, "gplvm_q_test_reconstructed.npy")
    np.save(latent_file, latent)
    np.save(q_pred_file, q_pred)

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage4_test_gplvm.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("model_file", model_file),
                    ("model_decoder_mode", decoder_mode),
                    ("model_sparse_num_inducing", sparse_num_inducing),
                    ("q_test_file", q_test_file),
                    ("max_test_samples", max_test_samples),
                    ("infer_max_its", infer_max_its),
                    ("infer_tol", infer_tol),
                    ("inverse_method", inverse_method),
                    ("inverse_n_starts", inverse_n_starts),
                    ("inverse_bound_margin_rel", inverse_bound_margin_rel),
                    ("inverse_bound_margin_abs", inverse_bound_margin_abs),
                    ("inverse_prior_weight", inverse_prior_weight),
                    ("inverse_loss", inverse_loss),
                    ("inverse_f_scale", inverse_f_scale),
                    ("save_overlay_plots", bool(save_overlay_plots)),
                    ("overlay_num_samples", overlay_num_samples),
                    ("overlay_indices", overlay_indices),
                    ("overlay_max_modes", overlay_max_modes),
                    ("overlay_dir", overlay_dir if bool(save_overlay_plots) else None),
                ],
            ),
            (
                "dataset",
                [
                    ("q_test_shape", q_test.shape),
                    ("q_eval_shape", q_eval.shape),
                ],
            ),
            (
                "error_metrics",
                [
                    ("relative_error_percent", rel_err_pct),
                    ("mean_relative_error_percent_per_sample", float(np.mean(rel_per_sample_pct))),
                    ("max_relative_error_percent_per_sample", float(np.max(rel_per_sample_pct))),
                    ("inverse_success_count", int(inverse_success_count)),
                    ("inverse_fail_count", int(inverse_fail_count)),
                    ("overlay_plots_count", int(len(overlay_files))),
                ],
            ),
            (
                "outputs",
                [
                    ("latent_test_npy", latent_file),
                    ("q_test_reconstructed_npy", q_pred_file),
                    ("error_plot_png", plot_file if plot_saved else None),
                    ("overlay_manifest_txt", overlay_manifest),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )

    print("\n====================================================")
    print("         STAGE 4: TEST POD-GPLVM MODEL")
    print("====================================================")
    print(f"[STAGE4] q_test shape: {q_test.shape}")
    print(f"[STAGE4] q_eval shape: {q_eval.shape}")
    print(
        f"[STAGE4] model decoder_mode={decoder_mode}, "
        f"sparse_num_inducing={sparse_num_inducing}"
    )
    print(f"[STAGE4] Relative error: {rel_err_pct:.6f}%" if rel_err_pct is not None else "[STAGE4] Relative error: N/A")
    if bool(save_overlay_plots):
        print(f"[STAGE4] Overlay plots saved: {len(overlay_files)}")
        if overlay_manifest is not None:
            print(f"[STAGE4] Overlay manifest: {overlay_manifest}")
    print(f"[STAGE4] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
