#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Spatial x/y cuts for the four test points, m=80 pure reconstructions."""

from __future__ import annotations

import os
import shutil
import csv
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_mu456_cuts_m80")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

M = 80

BASIS = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy"
U_REF = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy"
OUT_DIR = ROOT / "Prom_MasterANN_Diagnostic" / "test_points_reconstruction_cuts_m80"

POINTS = [
    {
        "label": "verification",
        "title": r"$\mu=(4.875,0.0225)$ verification",
        "mu_label": "mu1_4.875_mu2_0.0225",
        "hdm": Path("Results/param_snaps/mu1_4.875+mu2_0.0225.npy"),
    },
    {
        "label": "offgrid1",
        "title": r"$\mu=(4.560,0.0190)$ off-grid 1",
        "mu_label": "mu1_4.560_mu2_0.0190",
        "hdm": Path("Results/param_snaps/mu1_4.56+mu2_0.019.npy"),
    },
    {
        "label": "offgrid2",
        "title": r"$\mu=(5.190,0.0260)$ off-grid 2",
        "mu_label": "mu1_5.190_mu2_0.0260",
        "hdm": Path("Results/param_snaps/mu1_5.19+mu2_0.026.npy"),
    },
    {
        "label": "extrapolation20pct",
        "title": r"$\mu=(4.000,0.0330)$ 20\% extrapolation",
        "mu_label": "mu1_4.000_mu2_0.0330",
        "hdm": Path("Results/param_snaps/mu1_4.0+mu2_0.033.npy"),
    },
]


def _load_q(path: Path) -> np.ndarray:
    q = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if q.shape == (501, 151):
        q = q.T
    if q.shape[0] != 151:
        raise ValueError(f"Expected q shape (151, nt), got {q.shape} from {path}")
    return q


def _load_hdm(path: Path, ndof: int, nt: int) -> np.ndarray:
    if not path.exists():
        alt = PROJECT / path
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Missing HDM snapshots: {path}")
    hdm = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if hdm.shape == (nt, ndof):
        hdm = hdm.T
    if hdm.shape != (ndof, nt):
        raise ValueError(f"HDM shape {hdm.shape} incompatible with ndof={ndof}, nt={nt}")
    return hdm


def _reconstruct(v: np.ndarray, u_ref: np.ndarray, q: np.ndarray, m: int) -> np.ndarray:
    return u_ref[:, None] + v[:, :m] @ q[:m, :]


def _split_components(snaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = snaps.shape[0] // 2
    return snaps[:n, :], snaps[n:, :]


def _grid_size(n_scalar: int) -> int:
    n = int(round(np.sqrt(n_scalar)))
    if n * n != n_scalar:
        raise ValueError(f"Cannot infer square grid from scalar dofs={n_scalar}")
    return n


def _rel(reference: np.ndarray, approx: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(reference - approx) / np.linalg.norm(reference))


def _path_for_q(point: dict[str, object], family: str) -> Path:
    mu_label = str(point["mu_label"])
    if family == "linear":
        return ROOT / "mlspg_prom_main" / "Runs" / "Linear" / f"linear_prom_{mu_label}_ntot151" / "qN.npy"
    if family == "ann":
        return (
            ROOT
            / "mlspg_prom_main"
            / "Runs"
            / "DataDriven_MasterANN"
            / f"rom_data_driven_{mu_label}_ntot151"
            / "qN.npy"
        )
    raise ValueError(f"Unknown family={family}")


def _plot_point(point: dict[str, object], v: np.ndarray, u_ref: np.ndarray) -> dict[str, object]:
    label = str(point["label"])
    out_dir = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    q_lin = _load_q(_path_for_q(point, "linear"))
    q_ann = _load_q(_path_for_q(point, "ann"))
    hdm = _load_hdm(PROJECT.parent / Path(point["hdm"]), v.shape[0], q_lin.shape[1])

    u_lin80 = _reconstruct(v, u_ref, q_lin, M)
    u_ann80 = _reconstruct(v, u_ref, q_ann, M)
    u_lin151 = _reconstruct(v, u_ref, q_lin, 151)

    hdm_x, hdm_y = _split_components(hdm)
    lin_x, lin_y = _split_components(u_lin80)
    ann_x, ann_y = _split_components(u_ann80)
    ngrid = _grid_size(hdm_x.shape[0])
    grid = np.linspace(0.0, 1.0, ngrid)
    xcut = ngrid // 2
    ycut = ngrid // 2

    time_ids = [0, 100, 200, 300, 400, 500]
    time_ids = [i for i in time_ids if i < q_lin.shape[1]]

    fig, axes = plt.subplots(len(time_ids), 2, figsize=(11.0, 2.15 * len(time_ids)), sharex=True)
    if len(time_ids) == 1:
        axes = np.asarray([axes])

    for r, tidx in enumerate(time_ids):
        fields = {
            "HDM": (hdm_x[:, tidx].reshape(ngrid, ngrid), hdm_y[:, tidx].reshape(ngrid, ngrid)),
            "linear PROM m=80": (lin_x[:, tidx].reshape(ngrid, ngrid), lin_y[:, tidx].reshape(ngrid, ngrid)),
            "ANN m=80": (ann_x[:, tidx].reshape(ngrid, ngrid), ann_y[:, tidx].reshape(ngrid, ngrid)),
        }

        ax = axes[r, 0]
        ax.plot(grid, fields["HDM"][0][ycut, :], color="black", linewidth=2.0, label="HDM")
        ax.plot(grid, fields["linear PROM m=80"][0][ycut, :], color="#d62728", linewidth=1.45, linestyle="--", label="linear PROM m=80")
        ax.plot(grid, fields["ANN m=80"][0][ycut, :], color="#1f77b4", linewidth=1.45, linestyle="-.", label="ANN m=80")
        ax.set_ylabel(f"t step {tidx}")
        ax.set_title(r"$u_x$ cut at mid-$y$" if r == 0 else "")
        ax.grid(True, alpha=0.25)

        ax = axes[r, 1]
        ax.plot(grid, fields["HDM"][1][:, xcut], color="black", linewidth=2.0, label="HDM")
        ax.plot(grid, fields["linear PROM m=80"][1][:, xcut], color="#d62728", linewidth=1.45, linestyle="--", label="linear PROM m=80")
        ax.plot(grid, fields["ANN m=80"][1][:, xcut], color="#1f77b4", linewidth=1.45, linestyle="-.", label="ANN m=80")
        ax.set_title(r"$u_y$ cut at mid-$x$" if r == 0 else "")
        ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("coordinate")
    axes[0, 0].legend(loc="best", fontsize=8.5)
    axes[0, 1].legend(loc="best", fontsize=8.5)

    fig.suptitle(
        str(point["title"])
        + rf", pure reconstruction cuts with $m={M}$ "
        rf"(linear80={_rel(hdm, u_lin80):.2f}\%, ANN80={_rel(hdm, u_ann80):.2f}\%, linear151={_rel(hdm, u_lin151):.2f}\%)",
        y=0.997,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out_png = out_dir / f"{label}_reconstruction_cuts_m80.png"
    out_pdf = out_dir / f"{label}_reconstruction_cuts_m80.pdf"
    fig.savefig(out_png, dpi=230)
    fig.savefig(out_pdf)
    plt.close(fig)

    point_summary = out_dir / "summary.txt"
    point_summary.write_text(
        "\n".join(
            [
                f"point: {label}",
                f"title: {point['title']}",
                f"modes: {M}",
                f"linear_m80_vs_hdm_percent: {_rel(hdm, u_lin80):.12g}",
                f"ann_m80_vs_hdm_percent: {_rel(hdm, u_ann80):.12g}",
                f"linear_m151_vs_hdm_percent: {_rel(hdm, u_lin151):.12g}",
                f"figure_png: {out_png}",
                f"figure_pdf: {out_pdf}",
            ]
        )
        + "\n"
    )
    print(f"[cuts-m80:{label}] png = {out_png}")
    print(f"[cuts-m80:{label}] pdf = {out_pdf}")

    return {
        "label": label,
        "title": str(point["title"]),
        "linear_m80_vs_hdm_percent": _rel(hdm, u_lin80),
        "ann_m80_vs_hdm_percent": _rel(hdm, u_ann80),
        "linear_m151_vs_hdm_percent": _rel(hdm, u_lin151),
        "figure_png": str(out_png),
        "figure_pdf": str(out_pdf),
    }


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v = np.asarray(np.load(BASIS, allow_pickle=False), dtype=np.float64)
    u_ref = np.asarray(np.load(U_REF, allow_pickle=False), dtype=np.float64).reshape(-1)

    rows = []
    for point in POINTS:
        rows.append(_plot_point(point, v, u_ref))

    summary = OUT_DIR / "summary.csv"
    with summary.open("w", newline="") as f:
        keys = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[cuts-m80] summary = {summary}")


if __name__ == "__main__":
    main()
