#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE CHECK: QUADRATIC MANIFOLD TANGENT / TAYLOR TEST

Checks that
    ||u(q0 + eps d) - [u(q0) + J(q0)(eps d)]|| = O(eps^2)
for the manifold produced by stage1.
"""

import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
quadratic_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(quadratic_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.quadratic_manifold_utils import u_qm, J_qm


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


def taylor_test(V, H, u_ref, q0=None, n_eps=8, eps_min=1e-6, eps_max=1e-0, seed=1):
    np.random.seed(seed)
    _, n = V.shape

    if q0 is None:
        q0 = 0.1 * np.random.randn(n)
    else:
        q0 = np.asarray(q0, dtype=float).reshape(-1)

    d = np.random.randn(n)
    d /= np.linalg.norm(d)

    u0 = u_qm(q0, V, H, u_ref)
    J0 = J_qm(q0, V, H)

    eps_vals = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
    errors = np.zeros_like(eps_vals)

    for i, eps in enumerate(eps_vals):
        dq = eps * d
        u_eps = u_qm(q0 + dq, V, H, u_ref)
        u_lin = u0 + J0 @ dq
        errors[i] = np.linalg.norm(u_eps - u_lin)

    slope = float(np.polyfit(np.log10(eps_vals), np.log10(errors), 1)[0])
    return eps_vals, errors, slope, q0, d


def main(
    n_eps=8,
    eps_min=1e-6,
    eps_max=1e-0,
    seed=1,
):
    V_path = os.path.join(quadratic_dir, "qm_V.npy")
    H_path = os.path.join(quadratic_dir, "qm_H.npy")
    uref_path = os.path.join(quadratic_dir, "qm_uref.npy")

    for path in (V_path, H_path, uref_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file '{path}'. Run stage1_quadratic_offline.py first."
            )

    V = np.load(V_path, allow_pickle=False)
    H = np.load(H_path, allow_pickle=False)
    u_ref = np.load(uref_path, allow_pickle=False).reshape(-1)

    eps_vals, errors, slope, q0, d = taylor_test(
        V,
        H,
        u_ref,
        q0=None,
        n_eps=n_eps,
        eps_min=eps_min,
        eps_max=eps_max,
        seed=seed,
    )

    plot_path = os.path.join(quadratic_dir, "qm_taylor_test.png")
    plt.figure(figsize=(6, 5))
    plt.loglog(eps_vals, errors, "o-", label="QM Taylor error")
    ref = (errors[-1] / (eps_vals[-1] ** 2)) * (eps_vals ** 2)
    plt.loglog(eps_vals, ref, "--", label=r"$C \varepsilon^2$ (reference)")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$\|u(q_0+\varepsilon d)-[u(q_0)+J(q_0)\varepsilon d]\|_2$")
    plt.title("Taylor test for quadratic manifold")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report_path = os.path.join(quadratic_dir, "stage_check_qm_tangent_summary.txt")
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage_check_qm_tangent.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("n_eps", n_eps),
                    ("eps_min", eps_min),
                    ("eps_max", eps_max),
                    ("seed", seed),
                ],
            ),
            (
                "manifold",
                [
                    ("V_shape", V.shape),
                    ("H_shape", H.shape),
                    ("u_ref_shape", u_ref.shape),
                    ("q0_l2_norm", np.linalg.norm(q0)),
                    ("d_l2_norm", np.linalg.norm(d)),
                ],
            ),
            (
                "results",
                [
                    ("fitted_loglog_slope", slope),
                    ("expected_slope", 2.0),
                    ("first_error", errors[0]),
                    ("last_error", errors[-1]),
                ],
            ),
            (
                "outputs",
                [
                    ("taylor_plot_png", plot_path),
                    ("summary_txt", report_path),
                ],
            ),
        ],
    )

    print(f"[CHECK] Fitted slope: {slope:.3f} (target ~ 2.0)")
    print(f"[CHECK] Saved plot: {plot_path}")
    print(f"[CHECK] Saved summary: {report_path}")
    return slope


if __name__ == "__main__":
    main()
