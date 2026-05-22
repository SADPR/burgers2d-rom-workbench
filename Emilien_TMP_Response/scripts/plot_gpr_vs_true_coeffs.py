#!/usr/bin/env python3
"""Plot true vs GPR-predicted coefficients using saved POD-GPR closure model.

Requested blocks:
- q_1..q_4 (context coefficients)
- q_21..q_24 (secondary coefficients, GPR-predicted)
- q_101..q_104 (secondary coefficients, GPR-predicted)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-gpr-dir", type=Path, default=Path("POD-GPR"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "Emilien_TMP_Response/figures/"
            "gpr_vs_true_modes_1_4_and_101_104_mu1_4.56_mu2_0.019.png"
        ),
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "Emilien_TMP_Response/figures/"
            "gpr_vs_true_modes_1_4_and_101_104_summary.txt"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    modes_plot = [1, 2, 3, 4, 21, 22, 23, 24, 101, 102, 103, 104]
    primary_modes = 20

    pod_gpr_dir = args.pod_gpr_dir
    q_p_test = np.asarray(np.load(pod_gpr_dir / "q_p_test.npy", allow_pickle=False), dtype=np.float64)
    q_s_test = np.asarray(np.load(pod_gpr_dir / "q_s_test.npy", allow_pickle=False), dtype=np.float64)

    with open(pod_gpr_dir / "pod_gpr_model" / "gpr_model.pkl", "rb") as f:
        gpr_model = pickle.load(f)
    with open(pod_gpr_dir / "pod_gpr_model" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    x_test = scaler.transform(q_p_test.T)
    q_s_pred = np.asarray(gpr_model.predict(x_test), dtype=np.float64).T  # (131, 501)

    t_idx = np.arange(q_p_test.shape[1], dtype=np.int64)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharex=True)
    axes = np.asarray(axes).ravel()

    lines = [
        "Saved closure-GPR comparison",
        "model: X=q_1..q_20, Y=q_21..q_151",
        "test_mu=(4.56, 0.019)",
        f"primary_modes={primary_modes}",
        "",
        "per_mode_notes:",
    ]

    for j, mode in enumerate(modes_plot):
        ax = axes[j]

        if mode <= primary_modes:
            y = q_p_test[mode - 1, :]
            ax.plot(t_idx, y, color="tab:blue", linewidth=1.8, label="Context trajectory")
            ax.set_title(rf"$q_{{{mode}}}$ (context)")
            lines.append(f"q_{mode}: predicted=False (context only)")
        else:
            sec_idx = mode - (primary_modes + 1)
            y_true = q_s_test[sec_idx, :]
            y_pred = q_s_pred[sec_idx, :]

            denom = np.linalg.norm(y_true)
            rel = np.linalg.norm(y_true - y_pred) / denom if denom > 0 else np.nan
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            ax.plot(t_idx, y_true, color="black", linewidth=1.8, label="True")
            ax.plot(t_idx, y_pred, color="tab:red", linewidth=1.3, linestyle="--", label="GPR pred")
            ax.set_title(rf"$q_{{{mode}}}$ (rel err = {100.0*rel:.2f}%)")
            lines.append(
                f"q_{mode}: predicted=True, rel_error_percent={100.0*rel:.6f}, "
                f"mae={mae:.6e}, rmse={rmse:.6e}"
            )

        ax.set_ylabel("Coefficient value")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        if j // 4 == 2:
            ax.set_xlabel("Time step index")

    fig.suptitle(
        "Requested coefficient blocks at test mu=(4.56, 0.019)\n"
        "q_1-q_4 (context), q_21-q_24 (true vs GPR), q_101-q_104 (true vs GPR)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)
    plt.close(fig)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
